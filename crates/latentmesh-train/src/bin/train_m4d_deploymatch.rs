//! M4d trainer (ADR-024 § "Registered contingency — M4d, train/deploy
//! configuration match", registered 2026-08-29 BEFORE any M4d run).
//!
//! M4d repeats M4c's task-loss training with the probe's exact deployment
//! transform in the loop, so the trained object is the object deployed.
//! **Exactly one thing changes from `train_m4c_taskloss.rs`** — the
//! configuration of the deployment path. Architecture, init seed, split,
//! exclusions, loss, optimizer, epoch budget, seq cap and stopping rule are
//! byte-for-byte M4c's, so the M4c↔M4d comparison isolates configuration.
//!
//! HONEST SCOPE NOTE, recorded here and in the receipt because it corrects
//! the ADR's own framing: ADR-024's M4d registration says "the probe applies
//! `rescale_to_natural_median` to the injected vector, and no training rung
//! has ever had that operator in its loop". That is **factually wrong about
//! M4c**: `train_m4c_taskloss.rs::item_loss` already pooled → rescaled to a
//! natural inject-block median → broadcast to 8 slots → injected before the
//! receiver forward. The named candidate was therefore already largely
//! discharged when M4d was registered. What M4c actually left mismatched,
//! and what this rung removes, is narrower and is stated exactly:
//!
//!   1. **Rescale target source.** M4c computed each item's natural median
//!      through the COMPOSED forward (`qwen2_c::natural_per_position_l2`) —
//!      a disclosed deviation in its own receipt. The probe computes it
//!      through the VENDORED FUSED forward. M4d takes the constant from the
//!      probe's OWN code path (`QwenRuntime` + `capture::forward_capture` +
//!      `norms::stats`), and measures the per-item composed↔fused gap it
//!      removes.
//!   2. **Rescale operator identity.** M4c computed `(v/‖v‖)·median`; the
//!      probe computes `scale = median/‖v‖` then `v·scale`
//!      (`InjectionSpec::effective_vector`). M4d uses the probe's operator
//!      order and pins the duplication with a measured equivalence gate
//!      against the probe's own function (`deploy::verify_deploy_matches_probe`,
//!      ≥8 vectors, relative L2 ≤ 1e-6) before any model loads.
//!
//! The probe protocol, controls, items and statistics are UNTOUCHED — the
//! deployment transform is an ADR-028 *evolvable* surface, the probe is
//! *protected*. Training receipt is written BEFORE the registered transfer
//! check and the single frozen-probe draw (the probe invocation is the
//! freeze point). Honest-fail path unchanged (ADR-024/ADR-032): one run, no
//! retries, full numbers either way.
//!
//! Run: PATH=/usr/local/cuda-12.8/bin:$PATH \
//!      cargo run --release --features cuda --bin train_m4d_deploymatch

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
/// Training RNG seed — IDENTICAL to M4c's, so M4d starts from the same
/// seeded init and the ablation isolates the deployment configuration.
const TRAIN_SEED: u64 = 0x4D34_C001;
/// Golden-pair input seed (probe-side forward verification) — M4c's.
const GOLDEN_SEED: u64 = 0x4D34_C61D;
const GOLDEN_PAIRS: usize = 8;
/// Deployment-equivalence input seed, frozen here (new to M4d).
const DEPLOY_EQUIV_SEED: u64 = 0x4D34_D317;
const DEPLOY_EQUIV_VECTORS: usize = 8;
const DEPLOY_EQUIV_TOL: f32 = 1e-6;
const LR: f64 = 1e-3;
const EPOCHS: usize = 10;
/// Max receiver sequence — M4c's measured VRAM envelope, unchanged.
const SEQ_CAP: usize = 256;
/// Minimum CE-target tokens for an item to train/evaluate — M4c's.
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
/// transform** (`deploy_slot_vectors`) → inject → teacher-forced span CE.
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
    let init_artifact = receipts_dir.join("run2-m4d-mlp-deploymatch-init-cellL18toL14.f32bin");
    let init_hash = mlp.save_artifact(&init_artifact)?;
    let (init_golden_sha, init_in, init_out) =
        golden_pairs(&init_artifact, GOLDEN_SEED, GOLDEN_PAIRS)?;
    anyhow::ensure!(init_golden_sha == init_hash);
    let init_golden = receipts_dir.join("run2-m4d-golden-mlp-deploymatch-init-cellL18toL14.json");
    std::fs::write(
        &init_golden,
        serde_json::to_string(&serde_json::json!({
            "artifact_file_sha256": init_hash, "input_seed_chacha8": GOLDEN_SEED,
            "n_pairs": GOLDEN_PAIRS, "inputs": init_in, "outputs": init_out,
            "note": "SEEDED-INIT adapter (transfer-check baseline), outputs from the network itself (candle CPU forward); same TRAIN_SEED as M4c, so these bytes are expected identical to M4c's init artifact",
        }))?,
    )?;
    println!("init artifact: {} ({init_hash})", init_artifact.display());

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
    let artifact = receipts_dir.join("run2-m4d-mlp-deploymatch-cellL18toL14.f32bin");
    let content_hash = mlp.save_artifact(&artifact)?;
    let (golden_sha, inputs, outputs) = golden_pairs(&artifact, GOLDEN_SEED, GOLDEN_PAIRS)?;
    anyhow::ensure!(
        golden_sha == content_hash,
        "artifact hash drift during golden generation"
    );
    let golden_path = receipts_dir.join("run2-m4d-golden-mlp-deploymatch-cellL18toL14.json");
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
        "stage": "run2-m4d-deploymatch-training",
        "design": "docs/adr/024-run2-trained-thought-adapter-ladder.md § 'Registered contingency — M4d, train/deploy configuration match' (registered 2026-08-29 BEFORE any M4d run); training-only change, probe protocol untouched (ADR-028: deployment transform evolvable, probe protected)",
        "cell": CELL,
        "env": env_info(&nvcc),
        "rung_delta_vs_m4c": {
            "adr_premise_correction": "ADR-024's M4d registration states 'no training rung has ever had [rescale_to_natural_median] in its loop'. That is factually WRONG about M4c: train_m4c_taskloss.rs::item_loss already applied pool -> rescale-to-natural-median -> 8-slot broadcast -> inject before the receiver forward. The named candidate was therefore already largely discharged at registration time. This receipt records the correction rather than restating the premise; the interpretation rule registered with M4d still applies to whatever this rung measures.",
            "what_actually_changed_1_rescale_target_source": "M4c took each item's natural inject-block median from the COMPOSED forward (qwen2_c::natural_per_position_l2 — a deviation disclosed in M4c's own receipt). M4d takes it from the PROBE'S OWN code path: QwenRuntime (vendored FUSED BF16) + capture::forward_capture + norms::stats, the literal functions examples/common/m3.rs::four_conditions step 4 calls.",
            "what_actually_changed_2_rescale_operator_identity": "M4c computed (v/||v||)*median; the probe computes scale = median/||v|| then v*scale (InjectionSpec::effective_vector). M4d uses the probe's operator order via latentmesh_train::deploy::deploy_slot_vectors and pins it with the measured equivalence gate below.",
            "what_did_NOT_change": "architecture, TRAIN_SEED (same seeded init as M4c), split rule and seed, the 13 probe-overlap exclusions, loss (C2C task CE on the sender's own generated span), optimizer/lr/epochs/batch, SEQ_CAP, MIN_TARGET, stopping rule, golden discipline, probe protocol.",
            "residual_mismatch_still_present": "the training forward is the COMPOSED differentiable BF16 forward and the probe's is the VENDORED FUSED one — unfixable while gradients are required (the fused kernels are apply_op*_no_bwd). This is the M4c caveat, carried unchanged, and mitigated by the same registered transfer check before any probe draw.",
        },
        "deploy_equivalence_gate": equiv,
        "architecture_choice": {
            "rule": "ADR-024 M4d: repeat M4c's training, one factor changed",
            "chosen": "M3 MLP 2048->512->1536 ReLU (param_count 1,837,056) — M4c's architecture, unchanged",
            "init": "FRESH seeded init at M4c's TRAIN_SEED, so the M4c<->M4d comparison isolates the deployment configuration and nothing else",
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
            "rule": "fit_holdout_split(2560, FIT_SPLIT_SEED) BY ITEM first, THEN the 13 probe-overlap rows dropped from whichever side they landed in (ADR-024 frozen leakage rule) — identical to M3/M4/M4c",
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
            "rule": "per item, the reconstructed sender capture prompt re-encoded token-identical to the stream's stored prompt_tokens; injection prompt asserted to carry exactly 8 placeholder slots AND to yield the same slot positions under the runtime's own QwenRuntime::placeholder_positions (new in M4d)",
            "items_gated": fit_items.len() + holdout_items.len(),
        },
        "training": {
            "loss": "C2C-style task loss (M4c's, unchanged): frozen receiver's teacher-forced next-token CE on the item's own sender-generated span, conditioned on the probe's slotted injection prompt with the adapter's vectors injected after block 14",
            "pipeline": "sender L18 per-token rows (FULL generated span) -> MLP per token (F32 Vars) -> mean-pool TRANSLATED rows -> deploy::deploy_slot_vectors (scale = fused natural median / ||pooled||, then pooled * scale; probe operator order, +1e-12 backward guard) -> 8 slots -> slice_assign injection -> composed BF16 forward -> CE on gen span capped per seq_cap_rule",
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
            "step0_grad_gate": "pass — grads present, finite, nonzero on all four MLP Vars through the probe's deployment transform and the composed forward (inspection-only backward; no update applied)",
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
            "file": "run2-m4d-mlp-deploymatch-cellL18toL14.f32bin",
            "layout": ARTIFACT_LAYOUT,
            "content_hash_sha256": content_hash,
            "golden_file": "run2-m4d-golden-mlp-deploymatch-cellL18toL14.json",
            "golden_file_sha256": sha256_file(&golden_path)?,
            "init_file": "run2-m4d-mlp-deploymatch-init-cellL18toL14.f32bin",
            "init_content_hash_sha256": init_hash,
            "init_golden_file": "run2-m4d-golden-mlp-deploymatch-init-cellL18toL14.json",
            "init_golden_file_sha256": sha256_file(&init_golden)?,
            "golden_input_seed_chacha8": GOLDEN_SEED, "golden_pairs": GOLDEN_PAIRS,
        },
        "registered_caveat_bf16_composed_vs_fused": {
            "statement": "training runs through the composed BF16 forward but the frozen probe runs the vendored FUSED BF16 forward; carried unchanged from M4c (the rescale CONSTANT is now fused-sourced, but the training forward still cannot be)",
            "mitigation_frozen_here": "BEFORE any probe draw, a transfer check (run2_m4d_transfer_check, separate process, inference-only, no probe items, no generation) evaluates the trained adapter's teacher-forced NLL through the VENDORED fused forward on the holdout items, against the SEEDED-INIT adapter under the identical delivery path — the same check M4c ran, so the two rungs' transfer numbers are directly comparable",
            "transfer_pass_criterion_frozen": "mean vendored-fused NLL(trained) < mean vendored-fused NLL(init) over the evaluable holdout items (same seq-cap/min-target spans as training); per-item wins/losses + sign test reported as secondary, not gating",
            "on_transfer_fail": "the frozen probe is NOT invoked; the transfer receipt plus diagnosis is the honest M4d outcome for that branch",
        },
        "eval_plan_frozen": {
            "order": "1) transfer check (must pass) -> 2) frozen probe ONCE",
            "probe": "ADR-023 frozen 40-item S1a/S2b protocol, inherited UNCHANGED (item_seed_chacha8 20897 == 0x51A1, one-sided exact sign test alpha=0.05, 8 slots, rescale-to-natural-median, greedy/batch=1/max 400 tokens), via run2_m4d_probe (run2_m4c_probe lineage, common::m3 shared per-item mechanics)",
            "variant": "per-token pathway ONLY (the pathway this training optimized); ONE probe invocation, no second draw",
            "gate": "aligned_real > random, one-sided exact sign test, p < 0.05 — the recorded verdict statistic, unchanged",
            "secondary_statistic_reported": "mid-p McNemar (ADR-030 / docs/research/031 §2.4) reported ALONGSIDE the exact sign p on the same collected pairs; it gates nothing and changes no recorded verdict",
            "cell_scope": "S2-winner cell L18->L14 only",
        },
        "wall_clock_s": t0.elapsed().as_secs_f64(),
    });
    let receipt_path = receipts_dir.join("run2-m4d-training-receipt-cellL18toL14.json");
    std::fs::write(&receipt_path, serde_json::to_string_pretty(&receipt)?)?;
    println!("training receipt: {}", receipt_path.display());
    println!("total wall clock: {:.1}s", t0.elapsed().as_secs_f64());
    Ok(())
}
