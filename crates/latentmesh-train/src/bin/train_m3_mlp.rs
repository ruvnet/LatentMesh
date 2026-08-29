//! M3 trainer (ADR-024): the frozen MLP projector 2048→512→1536 (ReLU,
//! 1,837,056 params), trained ONCE with AdamW on per-token pairs from the M2
//! dump for the S2-winner cell L18→L14, under the frozen leakage rule
//! (`fit_holdout_split(2560, 0x24C0_DE03)` by item first, then the 13
//! probe-overlap rows dropped from whichever side they landed in).
//!
//! Frozen-by-this-source training configuration (ADR-024 defers exactly
//! these to the training receipt, which this binary writes BEFORE any probe
//! is invoked — the probe invocation order is the freeze point):
//!   - init/shuffle RNG seed `TRAIN_SEED`, ChaCha8 throughout
//!   - AdamW lr=1e-3 (candle 0.9.2 defaults otherwise: β=(0.9,0.999),
//!     eps=1e-8, weight_decay=0.01), batch 256, MSE loss
//!   - stopping rule: fixed 10-epoch budget; the artifact is the epoch
//!     checkpoint with the LOWEST holdout MSE (holdout = the 20% side after
//!     probe-row exclusion; item-level split is the leakage boundary —
//!     token shuffling for SGD batches happens within the fit side only)
//!
//! Run: PATH=/usr/local/cuda-12.8/bin:$PATH \
//!      cargo run --release --features cuda --bin train_m3_mlp

use latentmesh_train::dataset::{expected_from_receipt, open_verified, sha256_file};
use latentmesh_train::mlp::{golden_pairs, Mlp, ARTIFACT_LAYOUT, D_IN, D_OUT, PARAM_COUNT};
use latentmesh_train::split::{
    leakage_safe_split, rows_sha256, FIT_SPLIT_SEED, PROBE_OVERLAP_ROW_ITEMS,
};

use candle_core::{Device, Tensor};
use candle_nn::{loss, Optimizer};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::path::{Path, PathBuf};

const SENDER_FILE: &str = "sender_L18.tok.f32bin";
const RECEIVER_FILE: &str = "receiver_L14.tok.f32bin";
const CELL: &str = "L18->L14";
/// Training RNG seed (init + per-epoch batch shuffle), frozen here.
const TRAIN_SEED: u64 = 0x4D33_0001;
/// Golden-pair input seed (probe-side forward verification), frozen here.
const GOLDEN_SEED: u64 = 0x4D33_601D;
const GOLDEN_PAIRS: usize = 8;
const LR: f64 = 1e-3;
const BATCH: usize = 256;
const EPOCHS: usize = 10;
const VAL_BATCH: usize = 1024;

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
        "evidence_label": "seeded deterministic GPU training (candle 0.9.2 AdamW) over captured per-token pairs; the pairs are live-model capture output (run2-pertoken-dump-receipt.json, evidence label live-model, single-host, simulation-free)",
        "gpu": gpu,
        "nvcc": nvcc,
        "git_commit": git,
        "crate": "latentmesh-train 0.1.0 (candle 0.9.2, lockfile copied from latentmesh-runtime)",
        "unix_time": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0),
    })
}

/// Gather `[n × dim]` host rows for a batch of global token indices.
fn gather(map: &latentmesh_train::dataset::LayerMap, idx: &[u32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(idx.len() * map.dim);
    for &t in idx {
        out.extend_from_slice(map.row(t as usize));
    }
    out
}

fn main() -> anyhow::Result<()> {
    let t0 = std::time::Instant::now();
    let nvcc = latentmesh_runtime::assert_cuda_build_env().map_err(anyhow::Error::msg)?;
    println!("build-env guard: {nvcc}");
    let receipts_dir = crate_rel("../latentmesh-runtime/receipts");
    let dump_receipt = receipts_dir.join("run2-pertoken-dump-receipt.json");

    // ---- Dataset: verify ALL 4 bin sha256s + index sha256 vs the M2 receipt
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

    // ---- Leakage-safe split (frozen rule), with row→item hard assert ------
    for (row, item) in PROBE_OVERLAP_ROW_ITEMS {
        anyhow::ensure!(
            ds.index.item_indices[row] == item,
            "probe-overlap row {row} maps to item {} in the index, expected {item}",
            ds.index.item_indices[row]
        );
    }
    let split = leakage_safe_split(ds.index.n_items);
    let fit_tokens = ds.token_indices_for_rows(&split.fit);
    let holdout_tokens = ds.token_indices_for_rows(&split.holdout);
    println!(
        "split: n_fit={} n_holdout={} (post-exclusion; {} probe rows dropped), fit_tokens={} holdout_tokens={}",
        split.fit.len(), split.holdout.len(), split.excluded.len(),
        fit_tokens.len(), holdout_tokens.len()
    );
    anyhow::ensure!((2035..=2048).contains(&split.fit.len()));

    // ---- μ_r (fit-side target mean) + holdout denominator (f64) ----------
    println!("computing fit-side target mean and holdout variance...");
    let mut mu_r = vec![0f64; D_OUT];
    for &t in &fit_tokens {
        for (m, v) in mu_r.iter_mut().zip(ds.receiver.row(t as usize)) {
            *m += *v as f64;
        }
    }
    for m in mu_r.iter_mut() {
        *m /= fit_tokens.len() as f64;
    }
    let mut holdout_den = 0f64; // Σ ||y − μ_r||²  over holdout tokens
    for &t in &holdout_tokens {
        for (m, v) in mu_r.iter().zip(ds.receiver.row(t as usize)) {
            let d = *v as f64 - m;
            holdout_den += d * d;
        }
    }
    let holdout_var_mse = holdout_den / (holdout_tokens.len() * D_OUT) as f64;
    println!("holdout mean-predictor MSE (variance baseline): {holdout_var_mse:.6}");

    // ---- Model + optimizer ------------------------------------------------
    let dev = Device::new_cuda(0).map_err(anyhow::Error::msg)?;
    let mlp = Mlp::new_seeded(TRAIN_SEED, &dev)?;
    anyhow::ensure!(mlp.param_count() == PARAM_COUNT);
    let mut opt = candle_nn::AdamW::new_lr(mlp.vars(), LR)?;
    println!("mlp params: {} (frozen ADR-024 count)", mlp.param_count());

    // ---- Training loop ----------------------------------------------------
    let mut curve_steps: Vec<serde_json::Value> = Vec::new();
    let mut curve_epochs: Vec<serde_json::Value> = Vec::new();
    let mut best: Option<(usize, f64, Vec<f32>)> = None; // (epoch, val_mse, weights)
    let mut shuffled = fit_tokens.clone();
    let mut global_step = 0usize;
    for epoch in 0..EPOCHS {
        let te = std::time::Instant::now();
        let mut rng = ChaCha8Rng::seed_from_u64(TRAIN_SEED ^ (0x5EED_0000 + epoch as u64));
        shuffled.shuffle(&mut rng);
        let mut train_sse = 0f64;
        let mut train_n = 0usize;
        for chunk in shuffled.chunks(BATCH) {
            let x = Tensor::from_vec(gather(&ds.sender, chunk), (chunk.len(), D_IN), &dev)?;
            let y = Tensor::from_vec(gather(&ds.receiver, chunk), (chunk.len(), D_OUT), &dev)?;
            let pred = mlp.forward(&x)?;
            let l = loss::mse(&pred, &y)?;
            opt.backward_step(&l)?;
            let lv = l.to_scalar::<f32>()? as f64;
            train_sse += lv * (chunk.len() * D_OUT) as f64;
            train_n += chunk.len() * D_OUT;
            if global_step % 500 == 0 {
                curve_steps.push(
                    serde_json::json!({"step": global_step, "epoch": epoch, "train_mse": lv}),
                );
                println!("epoch {epoch} step {global_step}: train mse {lv:.6}");
            }
            global_step += 1;
        }
        // Holdout pass (no optimizer step).
        let mut val_sse = 0f64;
        for chunk in holdout_tokens.chunks(VAL_BATCH) {
            let x = Tensor::from_vec(gather(&ds.sender, chunk), (chunk.len(), D_IN), &dev)?;
            let y = Tensor::from_vec(gather(&ds.receiver, chunk), (chunk.len(), D_OUT), &dev)?;
            let pred = mlp.forward(&x)?;
            let sse = pred.sub(&y)?.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
            val_sse += sse;
        }
        let val_mse = val_sse / (holdout_tokens.len() * D_OUT) as f64;
        let val_rel_residual = (val_sse / holdout_den).sqrt();
        let train_mse = train_sse / train_n as f64;
        println!(
            "epoch {epoch}: train mse {train_mse:.6}, holdout mse {val_mse:.6}, holdout rel residual {val_rel_residual:.6} ({:.1}s)",
            te.elapsed().as_secs_f32()
        );
        curve_epochs.push(serde_json::json!({
            "epoch": epoch, "train_mse_mean": train_mse, "holdout_mse": val_mse,
            "holdout_rel_residual_vs_fit_mean": val_rel_residual,
            "wall_s": te.elapsed().as_secs_f32(),
        }));
        if best.as_ref().map(|(_, b, _)| val_mse < *b).unwrap_or(true) {
            best = Some((epoch, val_mse, mlp.to_flat()?));
        }
    }
    let (best_epoch, best_val_mse, best_flat) = best.unwrap();
    let best_rel_residual =
        (best_val_mse * (holdout_tokens.len() * D_OUT) as f64 / holdout_den).sqrt();
    println!("best epoch {best_epoch}: holdout mse {best_val_mse:.6} (rel residual {best_rel_residual:.6})");

    // ---- Artifact + golden pairs (trained network itself, CPU forward) ----
    mlp.set_flat(&best_flat, &dev)?;
    let artifact = receipts_dir.join("run2-m3-mlp-cellL18toL14.f32bin");
    let content_hash = mlp.save_artifact(&artifact)?;
    println!(
        "artifact: {} content_hash {content_hash}",
        artifact.display()
    );
    let (golden_sha, inputs, outputs) = golden_pairs(&artifact, GOLDEN_SEED, GOLDEN_PAIRS)?;
    anyhow::ensure!(
        golden_sha == content_hash,
        "artifact hash drift during golden generation"
    );
    let golden_path = receipts_dir.join("run2-m3-golden-mlp-cellL18toL14.json");
    std::fs::write(
        &golden_path,
        serde_json::to_string(&serde_json::json!({
            "artifact_file_sha256": content_hash,
            "input_seed_chacha8": GOLDEN_SEED,
            "n_pairs": GOLDEN_PAIRS,
            "note": "outputs produced by the TRAINED network itself (artifact reloaded from disk, candle CPU forward); probe-side hand-rolled apply must match each output to relative L2 error <= 1e-5 before any model runs (S2b golden discipline)",
            "inputs": inputs,
            "outputs": outputs,
        }))?,
    )?;
    println!("golden pairs: {}", golden_path.display());

    // ---- Training receipt (written BEFORE any probe invocation) -----------
    let receipt = serde_json::json!({
        "stage": "run2-m3-mlp-training",
        "design": "docs/adr/024-run2-trained-thought-adapter-ladder.md M3 (frozen architecture 2048->512->1536 ReLU, AdamW; leakage rule per section 'Leakage discipline')",
        "cell": CELL,
        "env": env_info(&nvcc),
        "dataset": {
            "dump_receipt": "run2-pertoken-dump-receipt.json",
            "run_dir": run_dir.display().to_string(),
            "verified_before_training": verified.iter().map(|v| serde_json::json!({
                "file": v.file, "sha256": v.sha256, "bytes": v.bytes, "pass": v.pass})).collect::<Vec<_>>(),
            "index_sha256": ds.index_sha256,
            "sender_file": SENDER_FILE, "receiver_file": RECEIVER_FILE,
            "n_items": ds.index.n_items, "total_tokens": ds.index.total_tokens,
        },
        "split": {
            "rule": "fit_holdout_split(2560, FIT_SPLIT_SEED) BY ITEM first (replica of harness/latentmesh-live/src/calibrate.rs:115, replica==harness ground truth pinned by unit test), THEN the 13 probe-overlap rows dropped from whichever side they landed in (ADR-024 frozen leakage rule); token-level shuffling happens only WITHIN the fit side for SGD batches",
            "fit_split_seed": FIT_SPLIT_SEED,
            "n_fit_pre_exclusion": 2048, "n_holdout_pre_exclusion": 512,
            "n_fit": split.fit.len(), "n_holdout": split.holdout.len(),
            "excluded_probe_overlap_rows": split.excluded,
            "fit_rows_sha256_comma_joined": rows_sha256(&split.fit),
            "holdout_rows_sha256_comma_joined": rows_sha256(&split.holdout),
            "fit_token_pairs": fit_tokens.len(), "holdout_token_pairs": holdout_tokens.len(),
        },
        "training": {
            "architecture": "MLP 2048->512->1536, ReLU, x@W convention",
            "param_count": PARAM_COUNT,
            "optimizer": {"name": "AdamW (candle-nn 0.9.2, the only Adam variant shipped — ADR-024 frozen optimizer constraint)",
                           "lr": LR, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8, "weight_decay": 0.01},
            "loss": "MSE",
            "batch": BATCH, "epochs": EPOCHS,
            "train_seed_chacha8": TRAIN_SEED,
            "stopping_rule": "fixed 10-epoch budget; artifact = epoch checkpoint with lowest holdout MSE (frozen in this receipt before any frozen-probe invocation — ADR-024: the probe-invocation order is the freeze point)",
            "runs_performed": 1,
            "note_no_discarded_runs": "single training run; no restarts, no hyperparameter retries",
        },
        "curves": {"per_epoch": curve_epochs, "sampled_steps_every_500": curve_steps},
        "results": {
            "best_epoch": best_epoch,
            "best_holdout_mse": best_val_mse,
            "best_holdout_rel_residual_vs_fit_mean": best_rel_residual,
            "holdout_mean_predictor_mse_variance_baseline": holdout_var_mse,
            "rel_residual_formula": "sqrt(sum ||pred-y||^2 / sum ||y-mu_r||^2) over holdout tokens, mu_r = fit-side per-dim mean (S2 A6 formula shape, for run-1 comparability)",
        },
        "artifact": {
            "file": "run2-m3-mlp-cellL18toL14.f32bin",
            "layout": ARTIFACT_LAYOUT,
            "content_hash_sha256": content_hash,
            "golden_file": "run2-m3-golden-mlp-cellL18toL14.json",
            "golden_file_sha256": sha256_file(&golden_path)?,
            "golden_input_seed_chacha8": GOLDEN_SEED,
            "golden_pairs": GOLDEN_PAIRS,
        },
        "eval_plan_frozen": {
            "probe": "ADR-023 frozen 40-item S1a/S2b protocol, inherited unchanged (item_seed_chacha8 20897 == 0x51A1, one-sided exact sign test alpha=0.05, 8 slots, rescale-to-natural-median, greedy/batch=1/max 400 tokens)",
            "variants": [
                "(i) per-token translator: MLP per generated-span token state, then the existing mean-pooling + 8-slot injection on the TRANSLATED stream",
                "(ii) pooled-in/pooled-out: pool sender per-token states first (run-1 pipeline shape), then the SAME per-token-trained MLP on the pooled vector (ADR-024 authorial choice; confound disclosed there)"
            ],
            "gate": "aligned_real (MLP output) > random, one-sided exact sign test, p < 0.05, per variant separately",
            "cell_scope": "S2-winner cell L18->L14 only; ADR-024's M3 gate names variants, not cells — an anchor-cell (L24->L19) M3 run has no ADR-024 registration and would be an extra unregistered draw against the frozen probe; escalation on failure is M4 (next architecture), not another cell (coordinator decision if ever wanted)",
        },
        "wall_clock_s": t0.elapsed().as_secs_f64(),
    });
    let receipt_path = receipts_dir.join("run2-m3-training-receipt-cellL18toL14.json");
    std::fs::write(&receipt_path, serde_json::to_string_pretty(&receipt)?)?;
    println!("training receipt: {}", receipt_path.display());
    println!("total wall clock: {:.1}s", t0.elapsed().as_secs_f64());
    Ok(())
}
