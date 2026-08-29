//! The M5 training receipt (ADR-045), split out of
//! `bin/train_m5_receiver_lora.rs` for file-size discipline. The trainer keeps
//! the pipeline; this keeps the one large auditable object it writes.
//!
//! The receipt is written BEFORE the transfer check and before the draw — that
//! ordering is the freeze point, and the transfer criterion quoted here is the
//! one the check reads back rather than re-deciding.

use crate::dataset::{PairedDataset, VerifiedFile};
use crate::deploy::BACKWARD_GUARD;
use crate::receiver_lora::LoraAdapter;
use crate::split::{rows_sha256, LeakageSafeSplit, FIT_SPLIT_SEED};
use crate::taskdata::{SkippedItem, STREAMS_SHA256};
use latentmesh_runtime::inject::InjectionMode;
use latentmesh_runtime::lora::{ARTIFACT_LAYOUT, LORA_ALPHA};

/// Environment block for the receipt — evidence grade recorded with every
/// number, per ADR-014/018.
pub fn env_info(nvcc: &str) -> serde_json::Value {
    let gpu = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|e| format!("nvidia-smi unavailable: {e}"));
    let git = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    serde_json::json!({
        "evidence_label": "seeded deterministic GPU training (candle 0.9.2 AdamW) of a RECEIVER-side LoRA driving a LIVE receiver forward (composed differentiable BF16 qwen2_c); sender states are live-model capture output (run2-pertoken-dump-receipt.json)",
        "gpu": gpu,
        "nvcc": nvcc,
        "git_commit": git,
        "crate": "latentmesh-train 0.1.0 (candle 0.9.2, lockfile copied from latentmesh-runtime)",
        "unix_time": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0),
    })
}

/// Everything the receipt records that is not already in the split or the
/// dataset. Kept flat and explicit: every field is a registered protocol
/// element, and a reader auditing against ADR-045 should see them named.
pub struct TrainingReceipt<'a> {
    pub rank: usize,
    pub cell: &'a str,
    pub env: serde_json::Value,
    pub m3_hash: &'a str,
    pub m3_training_receipt: &'a str,
    pub inject_mode: InjectionMode,
    pub n_slots: usize,
    pub inject_after_block: usize,
    pub hidden: usize,
    pub seq_cap: usize,
    pub lr: f64,
    pub epochs: usize,
    pub train_seed: u64,
    pub golden_seed: u64,
    pub golden_pairs: usize,
    pub run_dir: String,
    pub verified: &'a [VerifiedFile],
    pub fit_items: usize,
    pub fit_skipped: &'a [SkippedItem],
    pub holdout_items: usize,
    pub holdout_skipped: &'a [SkippedItem],
    pub natural_median_stats: serde_json::Value,
    pub step0_loss: f32,
    pub step0_grad_a: f32,
    pub step0_grad_b: f32,
    pub peak_vram_mib: u64,
    pub curve: Vec<serde_json::Value>,
    pub init_holdout_ce: f64,
    pub best_epoch: usize,
    pub best_holdout_ce: f64,
    pub artifact_file: String,
    pub content_hash: String,
    pub golden_file: String,
    pub golden_file_sha256: String,
    pub init_file: String,
    pub init_hash: String,
    pub init_golden_file: String,
    pub init_golden_file_sha256: String,
    pub smoke: Option<usize>,
    pub wall_clock_s: f64,
}

fn what_is_trained(c: &TrainingReceipt<'_>, param_count: usize) -> serde_json::Value {
    serde_json::json!({
        "trained": format!("the receiver-side LoRA ONLY — A ({} x {}) and B ({} x {}), {param_count} parameters", c.hidden, c.rank, c.rank, c.hidden),
        "frozen_sender": "Qwen2.5-3B-Instruct; its L18 states are read from the committed M2 dump, never recomputed",
        "frozen_translator": format!("M3's already-trained reconstruction MLP 2048->512->1536 ReLU, byte-identical, sha256 {}, hash-asserted against {}", c.m3_hash, c.m3_training_receipt),
        "frozen_receiver_weights": "every Qwen2.5-1.5B-Instruct weight; candle 0.9.2 materialises frozen-weight grads internally (visible in the VRAM figure), but no frozen tensor is a Var and none is passed to the optimiser",
        "why_this_translator": "it makes M4i the comparator. M4i ran this exact artifact, this exact derivation (apply_last_row), this exact operator (fuse), this exact site (question tail), this exact stream and this exact statistic, on an UNADAPTED receiver. The single changed factor in M5 is therefore 'the receiver carries a trained LoRA'. Using an M4c/M4d/M4g task-loss adapter instead would reintroduce the OFF-MANIFOLD payload ADR-045 names as the inherited hazard, confounding the rung from the start.",
    })
}

fn objective() -> serde_json::Value {
    serde_json::json!({
        "loss": "teacher-forced next-token CE (nats/token, F32 upcast, detached-max log-softmax) on the GOLD-ANSWER CONTINUATION '#### {gold}', conditioned on the question-tail prompt with the frozen aligned payload fused at the 8 tail positions",
        "target_is_the_probes_own": "the CE target tokens are literally encode(\"#### {gold}\") — the same string examples/common/m3.rs builds for its teacher-forced NLL arm. Training optimises the probe's likelihood endpoint directly.",
        "not_the_sender_span": "docs/research/034 §5.2: task-loss training on the sender's generated span steered M4c toward reproducing the sender's tokens rather than the answer format the probe scores. ADR-045 registers the gold continuation; this trainer never sees the sender's generated tokens as a target.",
        "delta_v_never_used": "ΔV is not computed here and is not a training signal. docs/research/034 §3 prices ONE properly powered verify_edge draw at ~3 GPU-h — more than this entire run — and ADR-028 forbids frozen-probe fitness for any adapter search. ADR-045 registers ΔV as a single post-hoc characterisation.",
    })
}

fn delivery(c: &TrainingReceipt<'_>) -> serde_json::Value {
    serde_json::json!({
        "site": "question_tail_ordinary_tokens — the last 8 tokens whose byte span lies wholly inside the item's own question, read off the canonical tokenisation's offset map (the same algorithm as examples/common/m3.rs::build_site_prompt under Site::QuestionTail, gate for gate)",
        "prompt": "chat_prompt(SYSTEM, '{question}\\n\\n{ANSWER_FORMAT}') — no slot sentence, no placeholder token. BYTE-IDENTICAL to the S2c capture prompt, so the existing prompt-parity gate (re-encode == the stream's stored prompt_tokens) pins M5's injected prompt bit-for-bit; it passed on every built item.",
        "payload_derivation": "apply_last_row: M3's MLP on the LAST generated-span token state only (de-pooled), identical to M4h S1 / M4i",
        "payload_source_rows": "the committed M2 dump's sender_L18 block for the item; only the final row is read",
        "rescale": format!("to the per-item natural block-14 median, via the probe's own operator order (latentmesh_train::deploy::deploy_slot_vectors, pinned to InjectionSpec::effective_vector by unit test at <=1e-6; +{BACKWARD_GUARD:e} backward guard)"),
        "natural_norm_source": "per-item median of per-position L2 after block 14 over the question-tail prompt, computed on the BASE receiver with the adapter OFF — the capture tap runs before the adapter by construction, so this target is the same one every frozen-receiver rung used",
        "natural_median_stats_across_items": c.natural_median_stats,
        "operator": {"mode": c.inject_mode.tag(), "equation": c.inject_mode.equation(),
            "why_fuse": "overwriting real question tokens would destroy content the receiver needs, confounding an inert site with a deleted question (docs/research/043 §4). Fuse is also what M4i ran."},
        "adapter_order": "block 14 -> injection edit -> LoRA. The adapter sees the injected content; that is its job. Identical in the composed training forward (qwen2_c) and the deployed vendored forward (models::Model).",
        "n_slots": c.n_slots, "inject_after_block": c.inject_after_block,
    })
}

fn architecture(c: &TrainingReceipt<'_>, lora: &LoraAdapter) -> serde_json::Value {
    serde_json::json!({
        "form": "h' = h + ((h @ A) @ B) * (alpha / rank), F32 matmuls, delta cast to BF16 before the add",
        "rank": c.rank, "alpha": LORA_ALPHA, "scaling": lora.scaling(),
        "hidden": c.hidden, "param_count": lora.param_count(),
        "init": "A ~ U(-1/sqrt(hidden), +1/sqrt(hidden)) from one seeded ChaCha8 stream; B = 0 (standard LoRA init), so the init adapter is the EXACT identity",
        "prior_art_note": "docs/research/034 §2: ruvector's in-stack micro_lora.rs supplied the forward-pass architecture only. Its accumulate_gradient is a Hebbian/REINFORCE delta rule keyed on a scalar quality score with no connection to any receiver forward or loss, and is not used.",
        "artifact_layout": ARTIFACT_LAYOUT,
    })
}

fn split_block(c: &TrainingReceipt<'_>, split: &LeakageSafeSplit) -> serde_json::Value {
    serde_json::json!({
        "rule": "fit_holdout_split(2560, FIT_SPLIT_SEED) BY ITEM first, THEN the 13 probe-overlap rows dropped from whichever side they landed in (ADR-024 frozen leakage rule) — identical to every prior rung",
        "fit_split_seed": FIT_SPLIT_SEED,
        "n_fit": split.fit.len(), "n_holdout": split.holdout.len(),
        "excluded_probe_overlap_rows": split.excluded,
        "fit_rows_sha256_comma_joined": rows_sha256(&split.fit),
        "holdout_rows_sha256_comma_joined": rows_sha256(&split.holdout),
        "holdout_rows": split.holdout,
        "seq_cap_rule": format!("skip if prompt_len + gold_continuation_len > {} (SEQ_CAP = M4c's largest MEASURED envelope on this card)", c.seq_cap),
        "fit_items_trained": c.fit_items, "fit_skipped": c.fit_skipped,
        "holdout_items_evaluated": c.holdout_items, "holdout_skipped": c.holdout_skipped,
        "item_stream_disjointness": {
            "measured": true, "overlap_with_adaptation_512": 0,
            "note": "the draw consumes adaptation-512; this asserts the trained items and the drawn items are disjoint sets, rather than inheriting the claim",
        },
    })
}

fn training_block(c: &TrainingReceipt<'_>) -> serde_json::Value {
    serde_json::json!({
        "optimizer": {"name": "AdamW (candle-nn 0.9.2)", "lr": c.lr,
            "beta1": 0.9, "beta2": 0.999, "eps": 1e-8, "weight_decay": 0.01,
            "params": "the two LoRA Vars ONLY"},
        "batch": 1, "epochs": c.epochs, "train_seed_chacha8": c.train_seed,
        "receiver_forward": "qwen2_c composed differentiable BF16 forward (3 substitutions: rms_norm_slow, composed softmax w/ detached max, composed rotate-half rope)",
        "stopping_rule": format!("fixed {}-epoch budget; artifact = the epoch checkpoint with the LOWEST holdout gold-continuation CE, frozen by this source before any transfer-check or draw invocation", c.epochs),
        "step0_grad_gate": {
            "pass": true, "loss": c.step0_loss,
            "grad_l2_a": c.step0_grad_a, "grad_l2_b": c.step0_grad_b,
            "registered_expectation": "with B = 0 the gradient on A is EXACTLY zero — d(delta)/dA carries a factor of B — so the gate requires finite grads on BOTH Vars and a NONZERO grad on B. A's zero is the standard LoRA init's arithmetic, not a cut graph; B's nonzero grad is what proves the graph reaches the adapter through inject -> composed forward -> CE.",
        },
        "runs_performed": 1,
        "note_no_discarded_runs": "single training run; no restarts, no hyperparameter retries",
        "measured_process_peak_vram_mib": c.peak_vram_mib,
    })
}

fn caveat_and_plan() -> (serde_json::Value, serde_json::Value) {
    (
        serde_json::json!({
            "statement": "training runs through the composed BF16 forward but the draw runs the vendored FUSED BF16 forward; the measured gap at L=128 is 116/128 argmax agreement, max|dlogit| 8.19 (pure rounding amplification — F32 parity 128/128 at max|dlogit| 0.119 proves same function). Inherited unchanged from M4c.",
            "mitigation_frozen_here": "BEFORE any draw, run2_m5_transfer_check (separate process, inference-only, no draw items, no generation) evaluates the trained adapter's teacher-forced gold-continuation NLL through the VENDORED fused forward on the holdout items, against the SAME receiver with the adapter OFF, under the identical aligned-payload delivery path",
            "why_off_and_not_the_init_artifact": "B is zero-initialised, so the init adapter IS the identity: 'adapter off' and 'adapter at init' are the same function, and the check uses the cheaper of the two. The init artifact and its goldens are committed anyway so the claim is auditable.",
            "transfer_pass_criterion_frozen": "mean vendored-fused gold-continuation NLL(trained adapter) < mean vendored-fused NLL(adapter off) over the evaluable holdout items; per-item wins/losses + sign test reported as secondary, not gating",
            "on_transfer_fail": "the draw is NOT invoked; the transfer receipt plus diagnosis is the honest M5 outcome for that branch (a null would be confounded by the numeric gap)",
        }),
        serde_json::json!({
            "order": "1) transfer check (must pass) -> 2) the M5 draw, ONCE",
            "draw": "ADR-036 e-process on the adaptation-512 stream in fixed index order, question-tail site, N_max=300, lambda=0.30, PASS at W >= 20; conditions aligned / baseline / zerovec / random (norm-matched), ALL FOUR measured on the SAME adapted receiver",
            "primary": "aligned vs random. Both arms run on the adapted receiver, so a general fine-tuning gain raises both and cancels — ADR-045's reason for keeping the primary here rather than making it baseline-relative.",
            "baseline_must_be_re_measured": "ADR-045: the baseline arm is re-measured on THIS adapted receiver. No frozen-receiver rung's baseline may be reused — a task-loss-adapted receiver could simply be a better GSM8K solver, and reusing an old baseline would credit that to the channel.",
            "registered_bar": "ADR-045's power calculation: >= 45 of an expected ~65 discordant wins. If the realised n_disc falls below 30 the rung is reported UNINFORMATIVE and the power model is recorded as wrong.",
            "mandatory_co_reports": "the likelihood arm against baseline, zerovec AND norm-matched random with per-item sign tests; control-vs-control comparisons computed IN the probe; the full wealth trajectory's shape, not just its endpoint.",
            "cell_scope": "S2-winner cell L18->L14 only",
        }),
    )
}

/// Assemble the receipt. The outer object is built through a `Map` rather than
/// one `json!` literal so the macro expansion stays shallow.
pub fn build(
    c: &TrainingReceipt<'_>,
    lora: &LoraAdapter,
    split: &LeakageSafeSplit,
    ds: &PairedDataset,
) -> serde_json::Value {
    let (caveat, plan) = caveat_and_plan();
    let mut m = serde_json::Map::new();
    let mut put = |k: &str, v: serde_json::Value| {
        m.insert(k.to_string(), v);
    };
    put("stage", serde_json::json!("run2-m5-receiver-lora-training"));
    put("design", serde_json::json!("docs/adr/045-m5-receiver-side-adaptation-pre-registration.md (ACCEPTED — EXECUTING); scout docs/research/034. The FIRST rung that trains the receiver rather than the payload."));
    put("cell", serde_json::json!(c.cell));
    put("rank", serde_json::json!(c.rank));
    put("env", c.env.clone());
    put(
        "what_is_trained_and_what_is_frozen",
        what_is_trained(c, lora.param_count()),
    );
    put("training_objective", objective());
    put("delivery_path", delivery(c));
    put("architecture", architecture(c, lora));
    put(
        "dataset",
        serde_json::json!({
            "dump_receipt": "run2-pertoken-dump-receipt.json",
            "run_dir": c.run_dir,
            "verified_before_training": c.verified.iter().map(|v| serde_json::json!({
                "file": v.file, "sha256": v.sha256, "bytes": v.bytes, "pass": v.pass})).collect::<Vec<_>>(),
            "index_sha256": ds.index_sha256,
            "n_items": ds.index.n_items, "total_tokens": ds.index.total_tokens,
            "streams_sha256": STREAMS_SHA256,
        }),
    );
    put("split", split_block(c, split));
    put("training", training_block(c));
    put("curves", serde_json::json!({ "per_epoch": c.curve }));
    put(
        "results",
        serde_json::json!({
            "init_holdout_gold_ce": c.init_holdout_ce,
            "best_epoch": c.best_epoch,
            "best_holdout_gold_ce": c.best_holdout_ce,
            "composed_forward_improvement_nats": c.init_holdout_ce - c.best_holdout_ce,
        }),
    );
    put(
        "artifact",
        serde_json::json!({
            "file": c.artifact_file,
            "layout": ARTIFACT_LAYOUT,
            "content_hash_sha256": c.content_hash,
            "golden_file": c.golden_file,
            "golden_file_sha256": c.golden_file_sha256,
            "init_file": c.init_file,
            "init_content_hash_sha256": c.init_hash,
            "init_golden_file": c.init_golden_file,
            "init_golden_file_sha256": c.init_golden_file_sha256,
            "golden_input_seed_chacha8": c.golden_seed, "golden_pairs": c.golden_pairs,
        }),
    );
    put("registered_caveat_bf16_composed_vs_fused", caveat);
    put("eval_plan_frozen", plan);
    put(
        "smoke_run",
        match c.smoke {
            None => serde_json::Value::Null,
            Some(n) => serde_json::json!({
                "capped_items_per_side": n, "epochs": c.epochs,
                "warning": "SMOKE RUN — a pipeline proof, NOT the registered rung. Item counts and the epoch budget are cut, so no number here is evidence about anything. Written into the gitignored target/ tree; the draw reads receipts/ and can never load this artifact.",
            }),
        },
    );
    put("wall_clock_s", serde_json::json!(c.wall_clock_s));
    serde_json::Value::Object(m)
}
