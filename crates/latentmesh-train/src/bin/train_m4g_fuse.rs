//! M4g trainer (ADR-024 § "M4g PRE-REGISTRATION (2026-08-29, before any run)
//! — fuse instead of overwrite", registered BEFORE any M4g run).
//!
//! **THE ONE CHANGED FACTOR: the injection operator.** Every rung to date
//! (M3, M4, M4c, M4d) delivered the adapter's payload by OVERWRITING the
//! receiver's residual rows at the 8 placeholder positions
//! (`latentmesh_runtime::inject` -> `LayerEdit::Inject` -> `slice_assign`,
//! mirrored in `qwen2_c::forward_span_logits`). M4g delivers it as a
//! **residual ADD** — `h[slot] += c*v` — preserving the receiver's own state
//! at those positions. This is Cache-to-Cache's own fuser equation
//! `C_F = C_n(X) + F_n(...)` (arXiv:2510.03215 Eq. 3), verified from the
//! paper's method section in `docs/research/038` §4; LatentMesh's overwrite
//! was verified from source in the same place. The injection operator is an
//! ADR-028 **evolvable** surface; the frozen probe is **protected**.
//!
//! Everything else is byte-for-byte `train_m4d_deploymatch.rs`: the M3-shaped
//! MLP (2048->512->1536 ReLU), the SAME `TRAIN_SEED` (so the seeded init is
//! bit-identical to M4c's and M4d's — the artifact hash is asserted equal
//! below, which is what makes M4c/M4d/M4g a clean three-way ablation), the
//! same split rule and seed, the same 13 probe-overlap exclusions, the same
//! C2C-style task loss on the sender's own generated span, the same
//! AdamW/lr/epochs/batch/SEQ_CAP/MIN_TARGET and best-holdout stopping rule,
//! and the same deployment transform in the loop (M4d's: rescale target from
//! the probe's own vendored-fused `forward_capture` + `norms::stats`, in the
//! probe's operator order, gate-verified to <= 1e-6).
//!
//! **The control-semantics question the fuse operator forces, resolved and
//! pre-registered here** (see `control_semantics_under_fuse` in the receipt,
//! written BEFORE the probe is invoked): under overwrite, the registered
//! zero-vector control ZEROED the eight rows — a destructive intervention.
//! Under fuse the same registered payload (`vector = 0`, `scale = None`) is
//! `h += 0`, an exact no-op, so the zero condition becomes mathematically
//! identical to the uninjected baseline. The control's DEFINITION is
//! unchanged (the same zero payload through the same delivery path); its
//! SEMANTICS change as an unavoidable consequence of the one changed factor.
//! It is NOT silently redefined, NOT replaced, and NOT dropped — it is run,
//! and the resulting baseline-identity is measured and reported as an
//! operator-correctness diagnostic. The random control (per-item seeded
//! Gaussian, norm-matched to the effective aligned vector) keeps both its
//! definition and its meaning, so the PRIMARY statistic
//! `aligned_real > random` is unaffected.
//!
//! Training receipt is written BEFORE the registered transfer check and the
//! single frozen-probe draw (the probe invocation is the freeze point).
//! Honest-fail path unchanged (ADR-024/ADR-032): one run, no retries, no
//! protocol iteration, full numbers either way.
//!
//! Run: PATH=/usr/local/cuda-12.8/bin:$PATH \
//!      cargo run --release --features cuda --bin train_m4g_fuse

use latentmesh_train::dataset::{expected_from_receipt, open_verified, sha256_file};
use latentmesh_train::deploy::{deploy_slot_vectors, verify_deploy_matches_probe};
use latentmesh_train::mlp::{golden_pairs, Mlp, ARTIFACT_LAYOUT, D_IN, D_OUT, PARAM_COUNT};
use latentmesh_train::qwen2_c::{load_config, span_ce, TrainReceiver};
use latentmesh_train::split::{
    leakage_safe_split, rows_sha256, FIT_SPLIT_SEED, PROBE_OVERLAP_ROW_ITEMS,
};
use latentmesh_train::taskdata::{self, TaskItem};

use candle_core::{DType, Device, Tensor};
use candle_nn::Optimizer;
use latentmesh_runtime::inject::InjectionMode;
use latentmesh_runtime::{capture::forward_capture, norms, QwenRuntime};
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
/// Training RNG seed — IDENTICAL to M4c's and M4d's, so M4g starts from the
/// SAME seeded init and the three-way ablation isolates the injection
/// operator and nothing else. The init artifact hash is asserted equal to
/// M4c's/M4d's below rather than assumed.
const TRAIN_SEED: u64 = 0x4D34_C001;
/// Golden-pair input seed (probe-side forward verification) — M4c's/M4d's.
const GOLDEN_SEED: u64 = 0x4D34_C61D;
const GOLDEN_PAIRS: usize = 8;
/// Deployment-equivalence input seed — M4d's, unchanged.
const DEPLOY_EQUIV_SEED: u64 = 0x4D34_D317;
const DEPLOY_EQUIV_VECTORS: usize = 8;
const DEPLOY_EQUIV_TOL: f32 = 1e-6;
const LR: f64 = 1e-3;
const EPOCHS: usize = 10;
/// Max receiver sequence — M4c's measured VRAM envelope, unchanged.
const SEQ_CAP: usize = 256;
/// Minimum CE-target tokens for an item to train/evaluate — M4c's.
const MIN_TARGET: usize = 8;
/// **THE ONE CHANGED FACTOR** — residual ADD instead of overwrite at the
/// 8 placeholder rows. Used in the training loop here and pinned to the same
/// value in `run2_m4g_transfer_check` and `run2_m4g_probe`.
const INJECT_MODE: InjectionMode = InjectionMode::Fuse;

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
        "evidence_label": "seeded deterministic GPU training (candle 0.9.2 AdamW) driving a LIVE frozen receiver forward (composed differentiable BF16 qwen2_c); rescale targets from the LIVE vendored fused forward (the probe's own capture path); sender states are live-model capture output (run2-pertoken-dump-receipt.json)",
        "gpu": gpu,
        "nvcc": nvcc,
        "git_commit": git,
        "crate": "latentmesh-train 0.1.0 (candle 0.9.2, lockfile copied from latentmesh-runtime)",
        "unix_time": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0),
    })
}

/// One item's task loss: rows → MLP → pool → **the probe's deployment
/// transform** (`deploy_slot_vectors`) → **FUSE** (`h[slot] += c*v`) →
/// teacher-forced span CE. Identical to M4d's except for the operator.
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
    let vectors = deploy_slot_vectors(&pooled, natural_median, N_SLOTS)?; // (N_SLOTS, D_OUT)
    let tokens = Tensor::new(&it.full_tokens[..], dev)?.unsqueeze(0)?;
    let logits = model.forward_span_logits(
        &tokens,
        Some((
            &vectors,
            &it.slot_positions,
            INJECT_AFTER_BLOCK,
            INJECT_MODE,
        )),
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

/// Per-item natural inject-block medians through the PROBE'S OWN path:
/// `QwenRuntime` (vendored fused BF16) + `capture::forward_capture` +
/// `norms::stats`. Also re-derives the slot positions with the runtime's
/// own `placeholder_positions` and asserts they equal the trainer's.
fn fused_natural_medians(
    items: &[TaskItem],
    dev: &Device,
    t0: &std::time::Instant,
) -> anyhow::Result<(Vec<f32>, String)> {
    let e = anyhow::Error::msg;
    println!("loading {RECEIVER} (BF16, VENDORED FUSED forward — the probe's own capture path)...");
    let mut rt = QwenRuntime::load(RECEIVER, dev, DType::BF16).map_err(e)?;
    let pad_id = rt
        .tokenizer
        .token_to_id("<|fim_pad|>")
        .ok_or_else(|| anyhow::anyhow!("<|fim_pad|> not in tokenizer"))?;
    let mut out = Vec::with_capacity(items.len());
    for (n, it) in items.iter().enumerate() {
        let inj = &it.full_tokens[..it.inj_len];
        let positions = QwenRuntime::placeholder_positions(inj, pad_id);
        anyhow::ensure!(
            positions == it.slot_positions,
            "row {}: runtime placeholder_positions {:?} != trainer slot_positions {:?}",
            it.row,
            positions,
            it.slot_positions
        );
        let (_, cap) = forward_capture(&mut rt.model, inj, INJECT_AFTER_BLOCK, 0..inj.len(), dev)
            .map_err(e)?;
        out.push(norms::stats(cap.per_position_l2).median);
        if (n + 1) % 512 == 0 {
            println!(
                "  fused medians {}/{} ({:.0}s)",
                n + 1,
                items.len(),
                t0.elapsed().as_secs_f32()
            );
        }
    }
    let model_id = rt.model_id.clone();
    drop(rt); // free the fused receiver before the composed one loads
    Ok((out, model_id))
}

/// Relative-difference stats of two median vectors, for the receipt.
fn median_gap(fused: &[f32], composed: &[f32]) -> serde_json::Value {
    let rel: Vec<f32> = fused
        .iter()
        .zip(composed)
        .map(|(f, c)| (c - f).abs() / f.abs().max(1e-12))
        .collect();
    let signed: Vec<f32> = fused
        .iter()
        .zip(composed)
        .map(|(f, c)| (c - f) / f.abs().max(1e-12))
        .collect();
    serde_json::json!({
        "abs_relative": norms::stats(rel),
        "signed_relative_composed_minus_fused": norms::stats(signed),
    })
}

/// Assert M4g's seeded init is byte-identical to M4c's and M4d's, reading
/// both committed training receipts. This is what licenses the three-way
/// "only the injection operator changed" comparison.
fn shared_init_gate(receipts_dir: &Path, init_hash: &str) -> anyhow::Result<serde_json::Value> {
    let read = |name: &str| -> anyhow::Result<Option<String>> {
        let p = receipts_dir.join(name);
        if !p.exists() {
            return Ok(None);
        }
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(&p)?)?;
        Ok(v["artifact"]["init_content_hash_sha256"]
            .as_str()
            .map(str::to_string))
    };
    let m4c = read("run2-m4c-training-receipt-cellL18toL14.json")?;
    let m4d = read("run2-m4d-training-receipt-cellL18toL14.json")?;
    for (rung, h) in [("M4c", &m4c), ("M4d", &m4d)] {
        if let Some(h) = h {
            anyhow::ensure!(
                h == init_hash,
                "{rung} init hash {h} != M4g init hash {init_hash} — the rungs no longer share \
                 one seeded init, so the operator ablation would not be isolated"
            );
        }
    }
    Ok(serde_json::json!({
        "pass": true,
        "m4g_init_sha256": init_hash,
        "m4c_init_sha256": m4c,
        "m4d_init_sha256": m4d,
        "rule": "M4c, M4d and M4g must start from ONE byte-identical seeded init so the only difference between the three rungs is the factor each registers (loss config / deployment config / injection operator)",
    }))
}

/// Operator-correctness gate: under FUSE, a zero payload must reproduce the
/// un-injected forward exactly (`h += 0`). Measured through the same composed
/// differentiable forward the training loop uses, on one fit item, before any
/// optimizer step. This is also the mechanical statement of what happens to
/// the registered zero-vector control once the operator changes.
fn fuse_zero_noop_gate(
    model: &TrainReceiver,
    it: &TaskItem,
    dev: &Device,
) -> anyhow::Result<serde_json::Value> {
    let tokens = Tensor::new(&it.full_tokens[..], dev)?.unsqueeze(0)?;
    let zeros = Tensor::zeros((N_SLOTS, D_OUT), DType::F32, dev)?;
    let ce = |inject: Option<(&Tensor, &[usize], usize, InjectionMode)>| -> anyhow::Result<f32> {
        let logits =
            model.forward_span_logits(&tokens, inject, it.span_start, it.target_tokens.len())?;
        Ok(span_ce(&logits, &it.target_tokens, dev)?.to_scalar::<f32>()?)
    };
    let ce_none = ce(None)?;
    let ce_fuse_zero = ce(Some((
        &zeros,
        &it.slot_positions,
        INJECT_AFTER_BLOCK,
        InjectionMode::Fuse,
    )))?;
    let ce_over_zero = ce(Some((
        &zeros,
        &it.slot_positions,
        INJECT_AFTER_BLOCK,
        InjectionMode::Overwrite,
    )))?;
    let pass = ce_fuse_zero == ce_none;
    anyhow::ensure!(
        pass,
        "fuse with a zero payload changed the loss ({ce_fuse_zero} vs {ce_none}) — the residual \
         add is not a no-op, so the operator implementation is wrong"
    );
    Ok(serde_json::json!({
        "pass": pass,
        "row": it.row,
        "span_ce_uninjected": ce_none,
        "span_ce_fuse_zero_payload": ce_fuse_zero,
        "span_ce_overwrite_zero_payload": ce_over_zero,
        "overwrite_zero_delta_nats": ce_over_zero - ce_none,
        "rule": "under FUSE, h += 0 must equal the un-injected forward EXACTLY; the overwrite number is reported alongside to show, on this repo's own model, how far from a no-op the SAME registered zero payload was under the operator every prior rung used",
    }))
}

fn main() -> anyhow::Result<()> {
    let t0 = std::time::Instant::now();
    let nvcc = latentmesh_runtime::assert_cuda_build_env().map_err(anyhow::Error::msg)?;
    println!("build-env guard: {nvcc}");
    let receipts_dir = crate_rel("../latentmesh-runtime/receipts");
    let dev = Device::new_cuda(0).map_err(anyhow::Error::msg)?;

    // ---- Gate 0 (M4d): the deployment transform IS the probe's -----------
    let equiv = verify_deploy_matches_probe(
        &dev,
        D_OUT,
        DEPLOY_EQUIV_VECTORS,
        N_SLOTS,
        DEPLOY_EQUIV_SEED,
        DEPLOY_EQUIV_TOL,
    )?;
    println!(
        "deploy-equivalence gate: {} vectors vs InjectionSpec::effective_vector, max rel L2 {:.3e}, \
         max rel norm err {:.3e} <= {:.0e} OK",
        equiv.n_vectors, equiv.max_relative_l2_error, equiv.max_relative_norm_error, equiv.tolerance
    );

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

    // ---- M4d CHANGE 1: rescale targets from the PROBE'S OWN fused path ----
    println!("computing per-item natural inject-block median norms (FUSED / probe path)...");
    let (fit_medians, fused_model_id) = fused_natural_medians(&fit_items, &dev, &t0)?;
    let (holdout_medians, _) = fused_natural_medians(&holdout_items, &dev, &t0)?;
    anyhow::ensure!(fused_model_id == RECEIVER);
    let med_stats = norms::stats(
        fit_medians
            .iter()
            .chain(&holdout_medians)
            .copied()
            .collect(),
    );
    println!("natural medians (fused) across items: {med_stats:?}");

    // ---- Frozen receiver (composed differentiable BF16 forward) -----------
    let cfg = load_config(RECEIVER)?;
    println!("loading {RECEIVER} (BF16, composed qwen2_c forward)...");
    let model = TrainReceiver::load(RECEIVER, &cfg, DType::BF16, &dev)?;
    anyhow::ensure!(model.hidden_size() == D_OUT);

    // Diagnostic ONLY (never used for training): the composed-forward medians
    // M4c trained against, so the receipt measures the gap M4d removed.
    println!("measuring the composed<->fused median gap M4c carried (diagnostic only)...");
    let composed_medians_for = |items: &[TaskItem]| -> anyhow::Result<Vec<f32>> {
        let mut out = Vec::with_capacity(items.len());
        for it in items {
            let inj = Tensor::new(&it.full_tokens[..it.inj_len], &dev)?.unsqueeze(0)?;
            let l2 = model.natural_per_position_l2(&inj, INJECT_AFTER_BLOCK)?;
            out.push(norms::stats(l2).median);
        }
        Ok(out)
    };
    let composed_fit = composed_medians_for(&fit_items)?;
    let composed_holdout = composed_medians_for(&holdout_items)?;
    let all_fused: Vec<f32> = fit_medians
        .iter()
        .chain(&holdout_medians)
        .copied()
        .collect();
    let all_composed: Vec<f32> = composed_fit
        .iter()
        .chain(&composed_holdout)
        .copied()
        .collect();
    let gap = median_gap(&all_fused, &all_composed);
    println!("composed-vs-fused natural-median gap (M4c's residual mismatch): {gap}");

    // ---- MLP: fresh seeded init (M4c's seed); INIT artifact + golden ------
    let mlp = Mlp::new_seeded(TRAIN_SEED, &dev)?;
    anyhow::ensure!(mlp.param_count() == PARAM_COUNT);
    let init_artifact = receipts_dir.join("run2-m4g-mlp-fuse-init-cellL18toL14.f32bin");
    let init_hash = mlp.save_artifact(&init_artifact)?;
    let (init_golden_sha, init_in, init_out) =
        golden_pairs(&init_artifact, GOLDEN_SEED, GOLDEN_PAIRS)?;
    anyhow::ensure!(init_golden_sha == init_hash);
    let init_golden = receipts_dir.join("run2-m4g-golden-mlp-fuse-init-cellL18toL14.json");
    std::fs::write(
        &init_golden,
        serde_json::to_string(&serde_json::json!({
            "artifact_file_sha256": init_hash, "input_seed_chacha8": GOLDEN_SEED,
            "n_pairs": GOLDEN_PAIRS, "inputs": init_in, "outputs": init_out,
            "note": "SEEDED-INIT adapter (transfer-check baseline), outputs from the network itself (candle CPU forward); same TRAIN_SEED as M4c/M4d, so these bytes are asserted identical to M4c's and M4d's init artifacts",
        }))?,
    )?;
    println!("init artifact: {} ({init_hash})", init_artifact.display());

    // ---- Gate: the seeded init is BYTE-IDENTICAL to M4c's and M4d's -------
    // What makes M4c -> M4d -> M4g a clean ablation is that all three start
    // from the same weights; asserted against both committed receipts rather
    // than assumed from the shared seed constant.
    let shared_init = shared_init_gate(&receipts_dir, &init_hash)?;
    println!("shared-init gate: {shared_init}");

    // ---- Gate: the FUSE operator is a no-op for a zero payload ------------
    // The operator-correctness check that also pins the M4g control
    // semantics: under fuse, `h += 0` must reproduce the un-injected forward
    // exactly. Measured on the first fit item through the SAME composed
    // forward the training loop uses, before a single optimizer step.
    let fuse_zero = fuse_zero_noop_gate(&model, &fit_items[0], &dev)?;
    println!("fuse zero-payload no-op gate: {fuse_zero}");

    // ---- Step-0 gradient gate through the PROBE'S deployment transform ----
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
        // Inspection only — no update applied.
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

    // ---- Training loop (M4c's, unchanged) ---------------------------------
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
    let artifact = receipts_dir.join("run2-m4g-mlp-fuse-cellL18toL14.f32bin");
    let content_hash = mlp.save_artifact(&artifact)?;
    let (golden_sha, inputs, outputs) = golden_pairs(&artifact, GOLDEN_SEED, GOLDEN_PAIRS)?;
    anyhow::ensure!(
        golden_sha == content_hash,
        "artifact hash drift during golden generation"
    );
    let golden_path = receipts_dir.join("run2-m4g-golden-mlp-fuse-cellL18toL14.json");
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
        "stage": "run2-m4g-fuse-training",
        "design": "docs/adr/024-run2-trained-thought-adapter-ladder.md § 'M4g PRE-REGISTRATION (2026-08-29, before any run) — fuse instead of overwrite' and § 'M4g REGISTERED (2026-08-29): overwrite vs fuse'; the injection OPERATOR is the single changed factor. ADR-028: the injection operator is an EVOLVABLE surface, the frozen probe is PROTECTED. Probe protocol, items, controls and statistics untouched.",
        "cell": CELL,
        "env": env_info(&nvcc),
        "rung_delta_vs_m4d": {
            "the_one_changed_factor": "injection operator: OVERWRITE -> FUSE (residual add).",
            "before_m3_m4_m4c_m4d": "h[slot] = c*v — LayerEdit::Inject / slice_assign of the payload row over the receiver's own residual row at each of the 8 placeholder positions (latentmesh-runtime src/models/qwen2_b.rs, Inject arm), mirrored in qwen2_c::forward_span_logits for training.",
            "now_m4g": "h[slot] += c*v — LayerEdit::Fuse, the receiver's own row is READ and ADDED to, so its state at those positions survives. Mirrors Cache-to-Cache Eq.3, C_F = C_n(X) + F_n(...), verified from the paper's method section in docs/research/038 §4.",
            "implementation_discipline": "Fuse is a NEW LayerEdit variant, not a flag on Inject: the overwrite arm's executed op sequence is byte-for-byte unchanged, so every prior receipt (run 1 S1a/S2b, run 2 M3/M4/M4c/M4d) stays reproducible. Every InjectionSpec construction site in the repo now names its mode explicitly.",
            "what_did_NOT_change": "architecture (M3 MLP 2048->512->1536 ReLU), TRAIN_SEED (byte-identical seeded init to M4c and M4d, asserted below), split rule and seed, the 13 probe-overlap exclusions, loss (C2C task CE on the sender's own generated span), optimizer/lr/epochs/batch, SEQ_CAP, MIN_TARGET, stopping rule, golden discipline, the deployment transform in the loop (M4d's: fused-sourced natural-median rescale in the probe's operator order), the frozen probe protocol.",
            "rescale_under_fuse_disclosed": "rescale-to-natural-median is a PROTECTED element of the frozen protocol and is therefore kept exactly as registered. Its meaning shifts as a consequence of the operator: under overwrite the slot row BECAME a vector of L2 = the natural per-position median; under fuse a delta of that same L2 is ADDED to a row whose own L2 is about that median. This is disclosed, not adjusted — adjusting it would be a second changed factor.",
            "residual_mismatch_still_present": "the training forward is the COMPOSED differentiable BF16 forward and the probe's is the VENDORED FUSED one — unfixable while gradients are required (the fused kernels are apply_op*_no_bwd). This is the M4c caveat, carried unchanged, and mitigated by the same registered transfer check before any probe draw.",
        },
        "control_semantics_under_fuse": {
            "registered_before_the_probe": true,
            "why_this_section_exists": "ADR-028 forbids a search or a rung from redefining a control. Changing the injection operator changes what two of the four registered conditions MEAN without changing what they ARE, and that must be recorded before the draw rather than explained after it.",
            "aligned_real": {
                "definition_unchanged": "sender per-token L18 capture -> trained MLP per token -> mean-pool the translated rows -> scale = natural inject-block median / ||pooled|| -> broadcast to 8 slots.",
                "delivery_under_fuse": "h[slot] += c*v instead of h[slot] = c*v.",
                "meaning": "unchanged in kind — it is still 'the adapter's content, delivered at the registered site with the registered rescale'.",
            },
            "random": {
                "definition_unchanged": "per-item seeded ChaCha8 Gaussian (RANDVEC_SEED_BASE + item index), norm-matched to the EFFECTIVE aligned vector, delivered through the same path.",
                "meaning": "unchanged and still the right comparator: under fuse it is a norm-matched random PERTURBATION of the receiver's own state at those rows, which is exactly the 'same magnitude, no information' control the primary statistic needs. THE PRIMARY STATISTIC aligned_real > random IS THEREFORE UNAFFECTED IN MEANING.",
            },
            "zerovec_injected": {
                "definition_unchanged": "the true zero vector (vector = 0, scale = None) through the real 8-slot path.",
                "meaning_CHANGED": "under overwrite this ZEROED the eight residual rows — a destructive intervention, and the reason the registered zerovec gate ('2 x zerovec accuracy >= baseline accuracy') was a catastrophe check. Under fuse the identical payload is h += 0, an EXACT no-op, so the zero condition becomes mathematically identical to the uninjected baseline.",
                "deviation_declared": "This is a genuine semantic deviation forced by the one changed factor. It is declared here, not concealed. The control is NOT redefined, NOT replaced by a substitute (e.g. an overwrite-with-zero condition retained under a fuse rung), and NOT dropped: it is run exactly as registered, all 40 items, and its now-expected identity to the baseline is MEASURED and reported as an operator-correctness diagnostic (fuse_zero_payload_is_noop, below and in the probe receipt).",
                "why_no_substitute_control_was_added": "Retaining a destructive overwrite-with-zero condition inside an M4g draw would introduce a second injection operator into a rung whose entire content is that exactly one operator changed, and would be an unregistered addition to a protected control set. The registered four-condition quad already contains the correct 'no information delivered' reference for a fuse rung: baseline_uninjected. It is unchanged and is reported.",
                "consequence_for_the_registered_zerovec_gate": "the gate '2 x zerovec_correct >= baseline_correct' becomes trivially satisfied under fuse (the two conditions are the same computation). It is still computed and reported, and it is labelled DEGENERATE-UNDER-FUSE in the probe receipt so no reader mistakes it for evidence.",
            },
            "baseline_uninjected": {
                "definition_unchanged": "no injection at all (spec = None), same slotted prompt.",
                "meaning": "unchanged, and PROMOTED in importance: under fuse it is the reference the zero condition collapses onto, so aligned_vs_baseline is the informative 'did adding anything help' contrast.",
            },
        },
        "shared_seeded_init_gate": shared_init,
        "fuse_zero_payload_is_noop_gate": fuse_zero,
        "deploy_equivalence_gate": equiv,
        "architecture_choice": {
            "rule": "ADR-024 M4g: retrain the M3-shaped MLP under task loss with the FUSE operator in the loop, one factor changed",
            "chosen": "M3 MLP 2048->512->1536 ReLU (param_count 1,837,056) — M4c's/M4d's architecture, unchanged",
            "init": "FRESH seeded init at M4c's/M4d's TRAIN_SEED; the init artifact hash is ASSERTED byte-identical to both (shared_seeded_init_gate), so the M4c<->M4d<->M4g comparison isolates loss config / deployment config / injection operator respectively",
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
            "rule": "fit_holdout_split(2560, FIT_SPLIT_SEED) BY ITEM first, THEN the 13 probe-overlap rows dropped from whichever side they landed in (ADR-024 frozen leakage rule) — identical to M3/M4/M4c/M4d",
            "fit_split_seed": FIT_SPLIT_SEED,
            "n_fit": split.fit.len(), "n_holdout": split.holdout.len(),
            "excluded_probe_overlap_rows": split.excluded,
            "fit_rows_sha256_comma_joined": rows_sha256(&split.fit),
            "holdout_rows_sha256_comma_joined": rows_sha256(&split.holdout),
            "holdout_rows": split.holdout,
            "seq_cap_rule": format!("target_len = min(gen_len, {SEQ_CAP} - inj_prompt_len); items with target_len < {MIN_TARGET} skipped"),
            "fit_items_trained": fit_items.len(), "fit_skipped": fit_skipped,
            "holdout_items_evaluated": holdout_items.len(), "holdout_skipped": holdout_skipped,
        },
        "prompt_parity_gate": {
            "pass": true,
            "rule": "per item, the reconstructed sender capture prompt re-encoded token-identical to the stream's stored prompt_tokens; injection prompt asserted to carry exactly 8 placeholder slots AND to yield the same slot positions under the runtime's own QwenRuntime::placeholder_positions (inherited from M4d)",
            "items_gated": fit_items.len() + holdout_items.len(),
        },
        "training": {
            "loss": "C2C-style task loss (M4c's/M4d's, unchanged): frozen receiver's teacher-forced next-token CE on the item's own sender-generated span, conditioned on the probe's slotted injection prompt with the adapter's vectors FUSED after block 14",
            "pipeline": "sender L18 per-token rows (FULL generated span) -> MLP per token (F32 Vars) -> mean-pool TRANSLATED rows -> deploy::deploy_slot_vectors (scale = fused natural median / ||pooled||, then pooled * scale; probe operator order, +1e-12 backward guard) -> 8 slots -> RESIDUAL ADD (h[slot] += c*v; LayerEdit::Fuse semantics, mirrored op-for-op in qwen2_c) -> composed BF16 forward -> CE on gen span capped per seq_cap_rule",
            "injection_operator": {"mode": INJECT_MODE.tag(), "equation": INJECT_MODE.equation(),
                "prior_rungs": "overwrite (M3, M4, M4c, M4d and all of run 1)",
                "in_training_loop": true, "in_transfer_check": true, "in_probe": true},
            "receiver": {"model": RECEIVER, "dtype": "BF16", "frozen": true,
                "training_forward": "qwen2_c composed differentiable forward",
                "rescale_target_forward": "VENDORED FUSED (QwenRuntime + capture::forward_capture) — the probe's own"},
            "inject_after_block": INJECT_AFTER_BLOCK, "n_slots": N_SLOTS,
            "natural_norm_source": "per-item median of per-position L2 after block 14 over the injection prompt, computed through the PROBE'S OWN vendored fused forward (M4d change 1; M4c used the composed forward here)",
            "natural_median_stats_across_items_fused": med_stats,
            "composed_vs_fused_median_gap_diagnostic": gap,
            "optimizer": {"name": "AdamW (candle-nn 0.9.2)", "lr": LR,
                "beta1": 0.9, "beta2": 0.999, "eps": 1e-8, "weight_decay": 0.01},
            "batch": 1, "epochs": EPOCHS, "train_seed_chacha8": TRAIN_SEED,
            "stopping_rule": format!("fixed {EPOCHS}-epoch budget; artifact = epoch checkpoint with LOWEST holdout task CE (M4c's rule, frozen by this source before any transfer-check/probe invocation)"),
            "step0_grad_gate": "pass — grads present, finite, nonzero on all four MLP Vars through the probe's deployment transform, the FUSE operator and the composed forward (inspection-only backward; no update applied)",
            "runs_performed": 1,
            "note_no_discarded_runs": "single training run; no restarts, no hyperparameter retries",
            "measured_process_peak_vram_mib": peak_vram,
        },
        "curves": {"per_epoch": curve_epochs},
        "results": {
            "init_holdout_task_ce": init_holdout_ce,
            "best_epoch": best_epoch,
            "best_holdout_task_ce": best_holdout_ce,
            "composed_forward_improvement_nats": init_holdout_ce - best_holdout_ce,
        },
        "artifact": {
            "file": "run2-m4g-mlp-fuse-cellL18toL14.f32bin",
            "layout": ARTIFACT_LAYOUT,
            "content_hash_sha256": content_hash,
            "golden_file": "run2-m4g-golden-mlp-fuse-cellL18toL14.json",
            "golden_file_sha256": sha256_file(&golden_path)?,
            "init_file": "run2-m4g-mlp-fuse-init-cellL18toL14.f32bin",
            "init_content_hash_sha256": init_hash,
            "init_golden_file": "run2-m4g-golden-mlp-fuse-init-cellL18toL14.json",
            "init_golden_file_sha256": sha256_file(&init_golden)?,
            "golden_input_seed_chacha8": GOLDEN_SEED, "golden_pairs": GOLDEN_PAIRS,
        },
        "registered_caveat_bf16_composed_vs_fused": {
            "statement": "training runs through the composed BF16 forward but the frozen probe runs the vendored FUSED BF16 forward; carried unchanged from M4c (the rescale CONSTANT is now fused-sourced, but the training forward still cannot be)",
            "mitigation_frozen_here": "BEFORE any probe draw, a transfer check (run2_m4g_transfer_check, separate process, inference-only, no probe items, no generation) evaluates the trained adapter's teacher-forced NLL through the VENDORED fused forward on the holdout items, against the SEEDED-INIT adapter under the identical delivery path — byte-for-byte M4c's and M4d's check with the FUSE operator substituted, so all three rungs' transfer numbers are directly comparable",
            "transfer_pass_criterion_frozen": "mean vendored-fused NLL(trained) < mean vendored-fused NLL(init) over the evaluable holdout items (same seq-cap/min-target spans as training); per-item wins/losses + sign test reported as secondary, not gating",
            "on_transfer_fail": "the frozen probe is NOT invoked; the transfer receipt plus diagnosis is the honest M4g outcome for that branch",
        },
        "eval_plan_frozen": {
            "order": "1) transfer check (must pass) -> 2) manifold pre-check (diagnostic only, gates nothing) -> 3) frozen probe ONCE",
            "manifold_precheck": "run2_manifold_precheck is re-run over this artifact and reports cosine-to-natural and item-invariance alongside every prior rung. Per ADR-024's M4f pre-check framing it is DIAGNOSTIC ONLY and is NOT a gate on whether the probe is drawn.",
            "probe": "ADR-023 frozen 40-item S1a/S2b protocol, inherited UNCHANGED (item_seed_chacha8 20897 == 0x51A1, one-sided exact sign test alpha=0.05, 8 slots, rescale-to-natural-median, greedy/batch=1/max 400 tokens, four conditions), via run2_m4g_probe (run2_m4d_probe lineage, common::m3 shared per-item mechanics with the injection MODE threaded through as the only new parameter)",
            "variant": "per-token pathway ONLY (the pathway this training optimized); ONE probe invocation, no second draw",
            "gate": "aligned_real > random, one-sided exact sign test, p < 0.05 — the recorded verdict statistic, unchanged",
            "primary_statistic_for_this_rung": "ADR-024's M4g pre-registration names mid-p McNemar as primary with exact-sign reported alongside. BOTH are computed on the same collected pairs and BOTH are reported; the receipt's machine gate_pass field stays on the ADR-028-protected exact sign test so it remains comparable with every prior rung's gate_pass, and the M4g verdict prose reports the mid-p value the pre-registration named. Neither number is chosen after seeing the other.",
            "secondary_statistic_reported": "whichever of the two is not being quoted as primary, plus the NLL sign tests, all on the same collected pairs",
            "cell_scope": "S2-winner cell L18->L14 only",
            "honest_fail_path": "ADR-024/ADR-032: ONE probe draw, no retry, no protocol iteration, full numbers reported either way. A null leaves M4f (structural on-manifold constraint) and M4b (receiver scale) as the live hypotheses and strengthens the joint negative across loss function, deployment configuration and injection operator.",
        },
        "wall_clock_s": t0.elapsed().as_secs_f64(),
    });
    let receipt_path = receipts_dir.join("run2-m4g-training-receipt-cellL18toL14.json");
    std::fs::write(&receipt_path, serde_json::to_string_pretty(&receipt)?)?;
    println!("training receipt: {}", receipt_path.display());
    println!("total wall clock: {:.1}s", t0.elapsed().as_secs_f64());
    Ok(())
}
