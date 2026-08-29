//! M4c trainer (ADR-024 § Registered contingency — task-loss ablation rung,
//! MANDATORY after M4's null): the M3 MLP projector architecture
//! (2048→512→1536 ReLU — best-so-far by holdout fit: M3 rel residual 0.461
//! vs FastGRNN's 0.633/0.663/0.704; ADR-024's "same or best-so-far
//! architecture" rule, choice recorded in the receipt), trained ONCE, from a
//! FRESH seeded init (not warm-started from M3's reconstruction-trained
//! weights — keeping the ablation to exactly one changed factor: the loss),
//! through the FROZEN receiver's next-token cross-entropy on each item's own
//! sender-generated span (C2C-style task loss), under the same frozen
//! leakage rule as M3/M4 (`fit_holdout_split(2560, 0x24C0_DE03)` by item,
//! then the 13 probe-overlap rows dropped).
//!
//! Per training step (batch=1, the M4c-feasibility-measured configuration):
//! sender per-token L18 rows (FULL generated span, from the verified M2
//! dump) → MLP per token (F32 Vars) → mean-pool the TRANSLATED rows →
//! rescale to the receiver's natural inject-block median norm → broadcast to
//! the 8 placeholder slots of the probe's own injection prompt → injected
//! after receiver block 14 inside the differentiable composed BF16 forward
//! (`qwen2_c`, feasibility-validated) → teacher-forced CE on the generated
//! span (capped at SEQ_CAP total tokens, the measured VRAM envelope) →
//! AdamW step on the four MLP Vars only (receiver frozen; candle 0.9.2
//! materializes frozen-weight grads internally — measured in the VRAM
//! numbers, not trainable state).
//!
//! Everything is frozen by this source + the training receipt, written
//! BEFORE the registered transfer check and frozen probe run (the probe
//! invocation order is the freeze point).
//!
//! Run: PATH=/usr/local/cuda-12.8/bin:$PATH \
//!      cargo run --release --features cuda --bin train_m4c_taskloss

use latentmesh_train::dataset::{expected_from_receipt, open_verified, sha256_file};
use latentmesh_train::mlp::{golden_pairs, Mlp, ARTIFACT_LAYOUT, D_IN, D_OUT, PARAM_COUNT};
use latentmesh_train::qwen2_c::{load_config, span_ce, TrainReceiver};
use latentmesh_train::split::{
    leakage_safe_split, rows_sha256, FIT_SPLIT_SEED, PROBE_OVERLAP_ROW_ITEMS,
};
use latentmesh_train::taskdata::{self, TaskItem};

use candle_core::{DType, Device, Tensor};
use candle_nn::Optimizer;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::path::{Path, PathBuf};

const RECEIVER: &str = "Qwen/Qwen2.5-1.5B-Instruct";
const SENDER_FILE: &str = "sender_L18.tok.f32bin";
const RECEIVER_FILE: &str = "receiver_L14.tok.f32bin";
const CELL: &str = "L18->L14";
const INJECT_AFTER_BLOCK: usize = 14;
const N_SLOTS: usize = 8;
/// Training RNG seed (MLP init + per-epoch item shuffle), frozen here.
const TRAIN_SEED: u64 = 0x4D34_C001;
/// Golden-pair input seed (probe-side forward verification), frozen here.
const GOLDEN_SEED: u64 = 0x4D34_C61D;
const GOLDEN_PAIRS: usize = 8;
const LR: f64 = 1e-3;
const EPOCHS: usize = 10;
/// Max receiver sequence (injection prompt + CE target span) — the M4c
/// feasibility probe's largest MEASURED seq len (81 ms/step, 9,906 MiB
/// process peak on this 16 GB card). Longer is unmeasured; not used.
const SEQ_CAP: usize = 256;
/// Minimum CE-target tokens for an item to train/evaluate (frozen).
const MIN_TARGET: usize = 8;

fn crate_rel(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

fn env_info(nvcc: &str) -> serde_json::Value {
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
        "evidence_label": "seeded deterministic GPU training (candle 0.9.2 AdamW) driving a LIVE frozen receiver forward (composed differentiable BF16 qwen2_c); sender states are live-model capture output (run2-pertoken-dump-receipt.json)",
        "gpu": gpu,
        "nvcc": nvcc,
        "git_commit": git,
        "crate": "latentmesh-train 0.1.0 (candle 0.9.2, lockfile copied from latentmesh-runtime)",
        "unix_time": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0),
    })
}

/// One item's task loss through the composed forward: rows → MLP → pool →
/// rescale-to-natural-median → 8 slots → inject → teacher-forced span CE.
fn item_loss(
    model: &TrainReceiver,
    mlp: &Mlp,
    it: &TaskItem,
    rows: &[f32],
    natural_median: f32,
    dev: &Device,
) -> anyhow::Result<Tensor> {
    let x = Tensor::from_vec(rows.to_vec(), (it.n_rows, D_IN), dev)?;
    let y = mlp.forward(&x)?; // (n_rows, D_OUT), F32
    let pooled = y.mean(0)?; // (D_OUT)
                             // scale = natural_median / ||pooled|| — the probe's rescale-to-natural-
                             // median semantics (+1e-12 backward guard; negligible vs measured norms).
    let norm = (pooled.sqr()?.sum_all()?.sqrt()? + 1e-12)?;
    let scaled = ((pooled.broadcast_div(&norm))? * natural_median as f64)?;
    let row = scaled.reshape((1, D_OUT))?;
    let vectors = Tensor::cat(&[&row; N_SLOTS], 0)?; // (N_SLOTS, D_OUT)
    let tokens = Tensor::new(&it.full_tokens[..], dev)?.unsqueeze(0)?;
    let logits = model.forward_span_logits(
        &tokens,
        Some((&vectors, &it.slot_positions, INJECT_AFTER_BLOCK)),
        it.span_start,
        it.target_tokens.len(),
    )?;
    Ok(span_ce(&logits, &it.target_tokens, dev)?)
}

/// Mean task CE over a set of items (no optimizer step).
fn eval_ce(
    model: &TrainReceiver,
    mlp: &Mlp,
    items: &[TaskItem],
    sender: &latentmesh_train::dataset::LayerMap,
    medians: &[f32],
    dev: &Device,
) -> anyhow::Result<f64> {
    let mut sum = 0f64;
    for (i, it) in items.iter().enumerate() {
        let loss = item_loss(
            model,
            mlp,
            it,
            sender.span(it.tok0, it.n_rows),
            medians[i],
            dev,
        )?;
        sum += loss.to_scalar::<f32>()? as f64;
    }
    Ok(sum / items.len() as f64)
}

fn main() -> anyhow::Result<()> {
    let t0 = std::time::Instant::now();
    let nvcc = latentmesh_runtime::assert_cuda_build_env().map_err(anyhow::Error::msg)?;
    println!("build-env guard: {nvcc}");
    let receipts_dir = crate_rel("../latentmesh-runtime/receipts");

    // ---- Dataset: verify ALL 4 bin sha256s + index sha256 vs the M2 receipt
    let dump_receipt = receipts_dir.join("run2-pertoken-dump-receipt.json");
    println!(
        "verifying dump integrity against {}...",
        dump_receipt.display()
    );
    let (expected, run_dir) = expected_from_receipt(&dump_receipt)?;
    let (ds, verified) = open_verified(&run_dir, SENDER_FILE, RECEIVER_FILE, &expected)?;
    for v in &verified {
        println!(
            "  {}: sha256 {} == receipt ({} bytes) OK",
            v.file, v.sha256, v.bytes
        );
    }
    anyhow::ensure!(ds.sender.dim == D_IN && ds.receiver.dim == D_OUT);

    // ---- Leakage-safe split (frozen rule) + row→item hard assert ----------
    for (row, item) in PROBE_OVERLAP_ROW_ITEMS {
        anyhow::ensure!(
            ds.index.item_indices[row] == item,
            "probe-overlap row {row} maps to item {} in the index, expected {item}",
            ds.index.item_indices[row]
        );
    }
    let split = leakage_safe_split(ds.index.n_items);
    anyhow::ensure!((2035..=2048).contains(&split.fit.len()));

    // ---- Streams + questions + tokenizer; task items under the cap rule --
    let streams = taskdata::load_streams(&crate_rel(taskdata::STREAMS_REL))?;
    anyhow::ensure!(streams.len() == ds.index.n_items);
    for (row, sr) in streams.iter().enumerate() {
        anyhow::ensure!(
            sr.item == ds.index.item_indices[row] && sr.gen_tokens.len() == ds.index.gen_len[row],
            "row {row}: stream/index mismatch"
        );
    }
    let questions = taskdata::load_gsm8k_questions(&crate_rel(taskdata::GSM8K_TRAIN_REL))?;
    let tok = taskdata::load_tokenizer(RECEIVER)?;
    let pad_id = tok
        .token_to_id("<|fim_pad|>")
        .ok_or_else(|| anyhow::anyhow!("<|fim_pad|> not in tokenizer"))?;
    let build = |rows: &[usize]| {
        taskdata::build_task_items(
            rows,
            &streams,
            &questions,
            &tok,
            &ds.index.token_offsets,
            pad_id,
            N_SLOTS,
            SEQ_CAP,
            MIN_TARGET,
        )
    };
    let (fit_items, fit_skipped) = build(&split.fit)?;
    let (holdout_items, holdout_skipped) = build(&split.holdout)?;
    println!(
        "split: fit {} items ({} skipped by cap rule), holdout {} items ({} skipped); \
         prompt-parity gate passed on all {} built items",
        fit_items.len(),
        fit_skipped.len(),
        holdout_items.len(),
        holdout_skipped.len(),
        fit_items.len() + holdout_items.len()
    );

    // ---- Frozen receiver (composed differentiable BF16 forward) -----------
    let dev = Device::new_cuda(0).map_err(anyhow::Error::msg)?;
    let cfg = load_config(RECEIVER)?;
    println!("loading {RECEIVER} (BF16, composed qwen2_c forward)...");
    let model = TrainReceiver::load(RECEIVER, &cfg, DType::BF16, &dev)?;
    anyhow::ensure!(model.hidden_size() == D_OUT);

    // ---- Natural inject-block medians (composed forward; disclosed
    // deviation: the frozen probe recomputes its own through the vendored
    // fused forward at eval time) — constant per item, computed once. ------
    println!("computing per-item natural inject-block median norms...");
    let medians_for = |items: &[TaskItem]| -> anyhow::Result<Vec<f32>> {
        let mut out = Vec::with_capacity(items.len());
        for (n, it) in items.iter().enumerate() {
            let inj = Tensor::new(&it.full_tokens[..it.inj_len], &dev)?.unsqueeze(0)?;
            let l2 = model.natural_per_position_l2(&inj, INJECT_AFTER_BLOCK)?;
            out.push(latentmesh_runtime::norms::stats(l2).median);
            if (n + 1) % 512 == 0 {
                println!(
                    "  {}/{} ({:.0}s)",
                    n + 1,
                    items.len(),
                    t0.elapsed().as_secs_f32()
                );
            }
        }
        Ok(out)
    };
    let fit_medians = medians_for(&fit_items)?;
    let holdout_medians = medians_for(&holdout_items)?;
    let med_stats = latentmesh_runtime::norms::stats(
        fit_medians
            .iter()
            .chain(&holdout_medians)
            .copied()
            .collect(),
    );
    println!("natural medians across items: {med_stats:?}");

    // ---- MLP: fresh seeded init; INIT artifact + golden saved first -------
    let mlp = Mlp::new_seeded(TRAIN_SEED, &dev)?;
    anyhow::ensure!(mlp.param_count() == PARAM_COUNT);
    let init_artifact = receipts_dir.join("run2-m4c-mlp-taskloss-init-cellL18toL14.f32bin");
    let init_hash = mlp.save_artifact(&init_artifact)?;
    let (init_golden_sha, init_in, init_out) =
        golden_pairs(&init_artifact, GOLDEN_SEED, GOLDEN_PAIRS)?;
    anyhow::ensure!(init_golden_sha == init_hash);
    let init_golden = receipts_dir.join("run2-m4c-golden-mlp-taskloss-init-cellL18toL14.json");
    std::fs::write(
        &init_golden,
        serde_json::to_string(&serde_json::json!({
            "artifact_file_sha256": init_hash, "input_seed_chacha8": GOLDEN_SEED,
            "n_pairs": GOLDEN_PAIRS, "inputs": init_in, "outputs": init_out,
            "note": "SEEDED-INIT adapter (transfer-check baseline), outputs from the network itself (candle CPU forward)",
        }))?,
    )?;
    println!("init artifact: {} ({init_hash})", init_artifact.display());

    // ---- Step-0 gradient gate: grads must reach ALL FOUR MLP Vars through
    // pool → rescale → broadcast → inject → composed forward → CE. ----------
    {
        let it = &fit_items[0];
        let loss = item_loss(
            &model,
            &mlp,
            it,
            ds.sender.span(it.tok0, it.n_rows),
            fit_medians[0],
            &dev,
        )?;
        let lv = loss.to_scalar::<f32>()?;
        let grads = loss.backward()?;
        for (name, var) in [
            ("w1", &mlp.w1),
            ("b1", &mlp.b1),
            ("w2", &mlp.w2),
            ("b2", &mlp.b2),
        ] {
            let g = grads.get(var.as_tensor()).ok_or_else(|| {
                anyhow::anyhow!("step-0 gate: NO gradient on {name} — graph silently cut")
            })?;
            let gn = g
                .to_dtype(DType::F32)?
                .sqr()?
                .sum_all()?
                .to_scalar::<f32>()?
                .sqrt();
            anyhow::ensure!(
                gn.is_finite() && gn > 0.0,
                "step-0 gate: gradient on {name} not finite/nonzero ({gn})"
            );
            println!("step-0 grad gate: {name} grad_l2 {gn:.6} OK (loss {lv:.4})");
        }
        // Inspection only — no update applied; training below starts from
        // the saved init exactly.
    }

    // ---- Initial (composed-forward) holdout task CE -----------------------
    let init_holdout_ce = eval_ce(
        &model,
        &mlp,
        &holdout_items,
        &ds.sender,
        &holdout_medians,
        &dev,
    )?;
    println!("holdout task CE at init: {init_holdout_ce:.6} nats/token");

    // ---- Training loop ----------------------------------------------------
    let mut opt = candle_nn::AdamW::new_lr(mlp.vars(), LR)?;
    let mut curve_epochs: Vec<serde_json::Value> = Vec::new();
    let mut best: Option<(usize, f64, Vec<f32>)> = None;
    let mut order: Vec<usize> = (0..fit_items.len()).collect();
    let mut peak_vram = 0u64;
    for epoch in 0..EPOCHS {
        let te = std::time::Instant::now();
        let mut rng = ChaCha8Rng::seed_from_u64(TRAIN_SEED ^ (0x5EED_0000 + epoch as u64));
        order.shuffle(&mut rng);
        let mut train_sum = 0f64;
        for (step, &i) in order.iter().enumerate() {
            let it = &fit_items[i];
            let loss = item_loss(
                &model,
                &mlp,
                it,
                ds.sender.span(it.tok0, it.n_rows),
                fit_medians[i],
                &dev,
            )?;
            train_sum += loss.to_scalar::<f32>()? as f64;
            opt.backward_step(&loss)?;
            if step % 64 == 0 {
                if let Some(v) = taskdata::process_vram_mib() {
                    peak_vram = peak_vram.max(v);
                }
            }
            if step % 500 == 0 {
                println!(
                    "epoch {epoch} step {step}/{}: mean train CE so far {:.6} ({:.0}s)",
                    order.len(),
                    train_sum / (step + 1) as f64,
                    t0.elapsed().as_secs_f32()
                );
            }
        }
        let train_ce = train_sum / order.len() as f64;
        let holdout_ce = eval_ce(
            &model,
            &mlp,
            &holdout_items,
            &ds.sender,
            &holdout_medians,
            &dev,
        )?;
        println!(
            "epoch {epoch}: train CE {train_ce:.6}, holdout CE {holdout_ce:.6} ({:.1}s, peak vram {peak_vram} MiB)",
            te.elapsed().as_secs_f32()
        );
        curve_epochs.push(serde_json::json!({
            "epoch": epoch, "train_ce_mean": train_ce, "holdout_ce": holdout_ce,
            "wall_s": te.elapsed().as_secs_f32(),
        }));
        if best
            .as_ref()
            .map(|(_, b, _)| holdout_ce < *b)
            .unwrap_or(true)
        {
            best = Some((epoch, holdout_ce, mlp.to_flat()?));
        }
    }
    let (best_epoch, best_holdout_ce, best_flat) = best.unwrap();
    println!(
        "best epoch {best_epoch}: holdout task CE {best_holdout_ce:.6} (init {init_holdout_ce:.6})"
    );

    // ---- Trained artifact + golden pairs ----------------------------------
    mlp.set_flat(&best_flat, &dev)?;
    let artifact = receipts_dir.join("run2-m4c-mlp-taskloss-cellL18toL14.f32bin");
    let content_hash = mlp.save_artifact(&artifact)?;
    let (golden_sha, inputs, outputs) = golden_pairs(&artifact, GOLDEN_SEED, GOLDEN_PAIRS)?;
    anyhow::ensure!(
        golden_sha == content_hash,
        "artifact hash drift during golden generation"
    );
    let golden_path = receipts_dir.join("run2-m4c-golden-mlp-taskloss-cellL18toL14.json");
    std::fs::write(
        &golden_path,
        serde_json::to_string(&serde_json::json!({
            "artifact_file_sha256": content_hash, "input_seed_chacha8": GOLDEN_SEED,
            "n_pairs": GOLDEN_PAIRS, "inputs": inputs, "outputs": outputs,
            "note": "outputs produced by the TRAINED network itself (artifact reloaded from disk, candle CPU forward); probe-side hand-rolled apply must match each output to relative L2 error <= 1e-5 before any model runs (S2b golden discipline)",
        }))?,
    )?;
    println!("trained artifact: {} ({content_hash})", artifact.display());

    // ---- Training receipt (BEFORE transfer check and probe) ---------------
    let receipt = serde_json::json!({
        "stage": "run2-m4c-taskloss-training",
        "design": "docs/adr/024-run2-trained-thought-adapter-ladder.md § 'Registered contingency — task-loss training rung' (M4c, mandatory after M4's null); config from the M4c feasibility probe (viable=true, BF16 composed forward)",
        "cell": CELL,
        "env": env_info(&nvcc),
        "architecture_choice": {
            "rule": "ADR-024 M4c: 'same or best-so-far architecture'",
            "chosen": "M3 MLP 2048->512->1536 ReLU (param_count 1,837,056)",
            "why": "best-so-far by holdout reconstruction fit: M3 MLP rel residual 0.461 vs M4 FastGRNN 0.633 (r256) / 0.663 (r128) / 0.704 (r64)",
            "init": "FRESH seeded init (ChaCha8, train_seed), NOT warm-started from M3's reconstruction-trained weights — the ablation isolates exactly one factor, the loss function",
        },
        "dataset": {
            "dump_receipt": "run2-pertoken-dump-receipt.json",
            "run_dir": run_dir.display().to_string(),
            "verified_before_training": verified.iter().map(|v| serde_json::json!({
                "file": v.file, "sha256": v.sha256, "bytes": v.bytes, "pass": v.pass})).collect::<Vec<_>>(),
            "index_sha256": ds.index_sha256,
            "sender_file": SENDER_FILE, "receiver_file": RECEIVER_FILE,
            "n_items": ds.index.n_items, "total_tokens": ds.index.total_tokens,
            "streams_file": "harness/latentmesh-live/data/s2c-token-streams.jsonl",
            "streams_sha256": taskdata::STREAMS_SHA256,
            "gsm8k_train_sha256": taskdata::GSM8K_TRAIN_SHA256,
        },
        "split": {
            "rule": "fit_holdout_split(2560, FIT_SPLIT_SEED) BY ITEM first (replica==harness ground truth pinned by unit test), THEN the 13 probe-overlap rows dropped from whichever side they landed in (ADR-024 frozen leakage rule) — identical to M3/M4",
            "fit_split_seed": FIT_SPLIT_SEED,
            "n_fit": split.fit.len(), "n_holdout": split.holdout.len(),
            "excluded_probe_overlap_rows": split.excluded,
            "fit_rows_sha256_comma_joined": rows_sha256(&split.fit),
            "holdout_rows_sha256_comma_joined": rows_sha256(&split.holdout),
            "holdout_rows": split.holdout,
            "seq_cap_rule": format!("target_len = min(gen_len, {SEQ_CAP} - inj_prompt_len); items with target_len < {MIN_TARGET} skipped (SEQ_CAP {SEQ_CAP} = the feasibility probe's largest MEASURED seq len; MIN_TARGET frozen here)"),
            "fit_items_trained": fit_items.len(), "fit_skipped": fit_skipped,
            "holdout_items_evaluated": holdout_items.len(), "holdout_skipped": holdout_skipped,
        },
        "prompt_parity_gate": {
            "pass": true,
            "rule": "per item, the reconstructed sender capture prompt (chat_prompt(SYSTEM, question + ANSWER_FORMAT)) re-encoded token-identical to the stream's stored prompt_tokens — pins SYSTEM, ANSWER_FORMAT, chat template and tokenizer parity against the probe side in one measured gate; injection prompt additionally asserted to carry exactly 8 placeholder slots",
            "items_gated": fit_items.len() + holdout_items.len(),
        },
        "training": {
            "loss": "C2C-style task loss: frozen receiver's teacher-forced next-token CE (nats/token, F32 upcast, detached-max log-softmax) on the item's own sender-generated span, conditioned on the probe's slotted injection prompt with the adapter's vectors injected after block 14",
            "pipeline": "sender L18 per-token rows (FULL generated span) -> MLP per token (F32 Vars) -> mean-pool TRANSLATED rows -> rescale to natural inject-block median (scale = median/||pooled||, probe semantics; +1e-12 backward guard) -> broadcast to 8 slots -> slice_assign injection (apply_edit op-for-op, F32->BF16 cast inside) -> composed BF16 forward -> CE on gen span capped per seq_cap_rule",
            "receiver": {"model": RECEIVER, "dtype": "BF16", "frozen": true,
                "forward": "qwen2_c composed differentiable forward (3 substitutions: rms_norm_slow, composed softmax w/ detached max, composed rotate-half rope) — feasibility-validated: F32 parity 128/128 argmax vs vendored (max|dlogit| 0.119)"},
            "inject_after_block": INJECT_AFTER_BLOCK, "n_slots": N_SLOTS,
            "natural_norm_source": "per-item median of per-position L2 after block 14 over the injection prompt, computed ONCE through the composed forward (disclosed deviation: the frozen probe recomputes its own through the vendored fused forward at eval)",
            "natural_median_stats_across_items": med_stats,
            "optimizer": {"name": "AdamW (candle-nn 0.9.2)", "lr": LR,
                "beta1": 0.9, "beta2": 0.999, "eps": 1e-8, "weight_decay": 0.01},
            "batch": 1, "epochs": EPOCHS, "train_seed_chacha8": TRAIN_SEED,
            "stopping_rule": format!("fixed {EPOCHS}-epoch budget; artifact = epoch checkpoint with LOWEST holdout task CE (frozen by this source before any transfer-check/probe invocation). NOTE: the feasibility notes' '2000 items x 3 epochs = ~8 min' was a budget EXTRAPOLATION EXAMPLE, not a spec; the 4 GPU-h budget supports ~88 epochs at L=256 — {EPOCHS} is the M3/M4 ladder discipline applied to it"),
            "step0_grad_gate": "pass — grads present, finite, nonzero on all four MLP Vars through the full pipeline (inspection-only backward before training; no update applied)",
            "runs_performed": 1,
            "note_no_discarded_runs": "single training run; no restarts, no hyperparameter retries",
            "measured_process_peak_vram_mib": peak_vram,
            "feasibility_basis": "m4c feasibility probe (scratchpad, this session): grad-through-receiver OK at L in {64,128,256} (grad_l2 0.897/0.426/0.482, epsilon-descent OK), 81 ms/AdamW step and 9,906 MiB peak at L=256, BF16 only (F32 receiver training OOMs: candle materializes frozen-weight grads)",
        },
        "curves": {"per_epoch": curve_epochs},
        "results": {
            "init_holdout_task_ce": init_holdout_ce,
            "best_epoch": best_epoch,
            "best_holdout_task_ce": best_holdout_ce,
            "composed_forward_improvement_nats": init_holdout_ce - best_holdout_ce,
        },
        "artifact": {
            "file": "run2-m4c-mlp-taskloss-cellL18toL14.f32bin",
            "layout": ARTIFACT_LAYOUT,
            "content_hash_sha256": content_hash,
            "golden_file": "run2-m4c-golden-mlp-taskloss-cellL18toL14.json",
            "golden_file_sha256": sha256_file(&golden_path)?,
            "init_file": "run2-m4c-mlp-taskloss-init-cellL18toL14.f32bin",
            "init_content_hash_sha256": init_hash,
            "init_golden_file": "run2-m4c-golden-mlp-taskloss-init-cellL18toL14.json",
            "init_golden_file_sha256": sha256_file(&init_golden)?,
            "golden_input_seed_chacha8": GOLDEN_SEED, "golden_pairs": GOLDEN_PAIRS,
        },
        "registered_caveat_bf16_composed_vs_fused": {
            "statement": "training runs through the composed BF16 forward but the frozen probe runs the vendored FUSED BF16 forward; measured gap at L=128: 116/128 argmax agreement, max|dlogit| 8.19 (pure rounding amplification — F32 parity 128/128 at max|dlogit| 0.119 proves same function)",
            "mitigation_frozen_here": "BEFORE any probe draw, a transfer check (run2_m4c_transfer_check, separate process, inference-only, no probe items, no generation) evaluates the trained adapter's teacher-forced NLL through the VENDORED fused forward on the holdout items, against the SEEDED-INIT adapter under the identical delivery path",
            "transfer_pass_criterion_frozen": "mean vendored-fused NLL(trained) < mean vendored-fused NLL(init) over the evaluable holdout items (same seq-cap/min-target spans as training); per-item wins/losses + sign test reported as secondary, not gating",
            "on_transfer_fail": "the frozen probe is NOT invoked; the transfer receipt plus diagnosis is the honest M4c outcome for that branch (a probe null would be confounded by the numeric gap)",
        },
        "eval_plan_frozen": {
            "order": "1) transfer check (must pass) -> 2) frozen probe ONCE",
            "probe": "ADR-023 frozen 40-item S1a/S2b protocol, inherited unchanged (item_seed_chacha8 20897 == 0x51A1, one-sided exact sign test alpha=0.05, 8 slots, rescale-to-natural-median, greedy/batch=1/max 400 tokens), via run2_m4c_probe (run2_m3_probe lineage, common::m3 shared per-item mechanics)",
            "variant": "per-token pathway ONLY — MLP per generated-span token state, then mean-pooling the TRANSLATED stream, then the frozen 8-slot injection (the pathway this training optimized); ONE probe invocation, no pooled-variant second draw",
            "gate": "aligned_real (task-trained MLP output) > random, one-sided exact sign test, p < 0.05",
            "cell_scope": "S2-winner cell L18->L14 only",
        },
        "wall_clock_s": t0.elapsed().as_secs_f64(),
    });
    let receipt_path = receipts_dir.join("run2-m4c-training-receipt-cellL18toL14.json");
    std::fs::write(&receipt_path, serde_json::to_string_pretty(&receipt)?)?;
    println!("training receipt: {}", receipt_path.display());
    println!("total wall clock: {:.1}s", t0.elapsed().as_secs_f64());
    Ok(())
}
