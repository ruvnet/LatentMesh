//! M4 trainer (ADR-024): the frozen low-rank FastGRNN sequence translator
//! (D_in 2048 → D_h 1536; sub-rung ladder r ∈ {64, 128, 256}, ascending,
//! each rank trained ONCE), sequence-to-sequence regression from the
//! sender's per-token L18 sequence to the receiver's per-token L14 sequence
//! over the M2 dump, under the same frozen leakage rule as M3
//! (`fit_holdout_split(2560, 0x24C0_DE03)` by item first, then the 13
//! probe-overlap rows dropped from whichever side they landed in).
//!
//! Frozen-by-this-source training configuration (ADR-024 defers exactly
//! these to the training receipt, written BEFORE that rank's probe is
//! invoked — the probe-invocation order is the freeze point):
//!   - init/shuffle RNG seed `TRAIN_SEED_BASE + rank`, ChaCha8 throughout;
//!     init scheme per `fastgrnn.rs` (factors U(±1/√fan_in), b=0,
//!     zeta_raw=1.0, nu_raw=-4.0 — EdgeML reference defaults)
//!   - AdamW lr=1e-3 (candle 0.9.2 defaults otherwise), masked MSE loss
//!   - sequences: each fit item is ONE full sequence from h_0 = 0 (state
//!     resets only at item boundaries — no cross-item sequences, per
//!     ADR-024), consumed as truncated-BPTT windows of 16 steps with the
//!     hidden state carried (detached) across windows; items batched 64 at
//!     a time, padded + masked. The scout's proven 16-step-BPTT pattern,
//!     with the training-time h distribution equal to the probe's
//!     full-sequence deployment by construction.
//!   - stopping rule: fixed 10-epoch budget; the artifact is the epoch
//!     checkpoint with the LOWEST FULL-SEQUENCE holdout MSE (each holdout
//!     item's whole generated span run from h_0 = 0 — the exact forward the
//!     probe uses, so checkpoint selection matches deployment)
//!
//! For r=64 ONLY there is one PRE-PROBE superseded training run, preserved
//! at `*-superseded-windowzeroinit.*`: the first scheme (independent
//! zero-init 16-token windows) diverged on the full-sequence holdout
//! (1.48 → 7.87 across epochs, best rel residual 1.33 > 1.0) — discovered
//! entirely via the calibration-derived holdout metric BEFORE any frozen
//! probe invocation (zero probe bits consumed; ADR-024's freeze point is
//! probe invocation). Disclosed in the receipt's `superseded_run` block.
//!
//! Run: PATH=/usr/local/cuda-12.8/bin:$PATH \
//!      cargo run --release --features cuda --bin train_m4_fastgrnn -- --rank 64

use latentmesh_train::dataset::{expected_from_receipt, open_verified, sha256_file};
use latentmesh_train::fastgrnn::{
    golden_pairs, param_count, FastGrnn, ARTIFACT_LAYOUT, D_H, D_IN, RANKS,
};
use latentmesh_train::split::{
    leakage_safe_split, rows_sha256, FIT_SPLIT_SEED, PROBE_OVERLAP_ROW_ITEMS,
};

use candle_core::{DType, Device, Tensor};
use candle_nn::Optimizer;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::path::{Path, PathBuf};

const SENDER_FILE: &str = "sender_L18.tok.f32bin";
const RECEIVER_FILE: &str = "receiver_L14.tok.f32bin";
const CELL: &str = "L18->L14";
/// Training RNG seed base (init + per-epoch window shuffle); the per-rank
/// seed is `TRAIN_SEED_BASE + rank`, frozen here.
const TRAIN_SEED_BASE: u64 = 0x4D34_0000;
/// Golden-sequence input seed (probe-side forward verification), frozen here.
const GOLDEN_SEED: u64 = 0x4D34_601D;
const GOLDEN_SEQS: usize = 8;
const GOLDEN_LEN: usize = 8;
const LR: f64 = 1e-3;
/// BPTT truncation length (the M0 scout's proven 16-step pattern).
const SEQ_LEN: usize = 16;
/// Items per padded/masked training batch (sequences stepped in lockstep).
const ITEM_BATCH: usize = 64;
const EPOCHS: usize = 10;
const EVAL_ITEMS_PER_BATCH: usize = 64;

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

fn main() -> anyhow::Result<()> {
    let t0 = std::time::Instant::now();
    let rank: usize = {
        let args: Vec<String> = std::env::args().collect();
        anyhow::ensure!(
            args.len() == 3 && args[1] == "--rank",
            "usage: train_m4_fastgrnn --rank {{64|128|256}}"
        );
        args[2].parse()?
    };
    anyhow::ensure!(RANKS.contains(&rank), "rank must be one of {RANKS:?}");
    let train_seed = TRAIN_SEED_BASE + rank as u64;
    let nvcc = latentmesh_runtime::assert_cuda_build_env().map_err(anyhow::Error::msg)?;
    println!(
        "build-env guard: {nvcc}; rank {rank} (params {})",
        param_count(rank)
    );
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
    anyhow::ensure!(ds.sender.dim == D_IN && ds.receiver.dim == D_H);

    // ---- Leakage-safe split (frozen rule), with row→item hard assert ------
    for (row, item) in PROBE_OVERLAP_ROW_ITEMS {
        anyhow::ensure!(ds.index.item_indices[row] == item);
    }
    let split = leakage_safe_split(ds.index.n_items);
    anyhow::ensure!((2035..=2048).contains(&split.fit.len()));

    // BPTT truncation windows per epoch (for the receipt): each fit item is
    // processed as one full sequence, gradient-truncated every SEQ_LEN steps.
    let bptt_windows_per_epoch: usize = split
        .fit
        .iter()
        .map(|&r| {
            let t_i = (ds.index.token_offsets[r + 1] - ds.index.token_offsets[r]) as usize;
            t_i.div_ceil(SEQ_LEN)
        })
        .sum();
    let fit_tokens = ds.token_indices_for_rows(&split.fit);
    let holdout_tokens = ds.token_indices_for_rows(&split.holdout);
    println!(
        "split: n_fit={} n_holdout={} ({} probe rows dropped); fit_tokens={} holdout_tokens={} bptt_windows/epoch={}",
        split.fit.len(), split.holdout.len(), split.excluded.len(),
        fit_tokens.len(), holdout_tokens.len(), bptt_windows_per_epoch
    );

    // ---- μ_r (fit-side target mean) + holdout denominator (f64) ----------
    println!("computing fit-side target mean and holdout variance...");
    let mut mu_r = vec![0f64; D_H];
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
    let holdout_var_mse = holdout_den / (holdout_tokens.len() * D_H) as f64;
    println!("holdout mean-predictor MSE (variance baseline): {holdout_var_mse:.6}");

    // ---- Model + optimizer ------------------------------------------------
    let dev = Device::new_cuda(0).map_err(anyhow::Error::msg)?;
    let cell = FastGrnn::new_seeded(rank, train_seed, &dev)?;
    anyhow::ensure!(cell.param_count() == param_count(rank), "param count drift");
    let mut opt = candle_nn::AdamW::new_lr(cell.vars(), LR)?;
    println!(
        "fastgrnn r={rank} params: {} (frozen ADR-024 formula)",
        cell.param_count()
    );

    // Holdout eval batches: items sorted by length desc, padded + masked.
    let mut holdout_sorted = split.holdout.clone();
    holdout_sorted.sort_by_key(|&r| {
        std::cmp::Reverse(ds.index.token_offsets[r + 1] - ds.index.token_offsets[r])
    });
    // Full-sequence holdout forward from h0=0 (the probe's exact forward);
    // returns Σ ||h_t − y_t||² over all real (unpadded) holdout positions.
    let eval_holdout = |cell: &FastGrnn| -> anyhow::Result<f64> {
        let mut sse = 0f64;
        for chunk in holdout_sorted.chunks(EVAL_ITEMS_PER_BATCH) {
            let lens: Vec<usize> = chunk
                .iter()
                .map(|&r| (ds.index.token_offsets[r + 1] - ds.index.token_offsets[r]) as usize)
                .collect();
            let t_max = *lens.iter().max().unwrap();
            let b = chunk.len();
            let mut xs = vec![0f32; b * t_max * D_IN];
            let mut ys = vec![0f32; b * t_max * D_H];
            let mut mask = vec![0f32; b * t_max];
            for (i, (&r, &l)) in chunk.iter().zip(&lens).enumerate() {
                let start = ds.index.token_offsets[r] as usize;
                xs[i * t_max * D_IN..i * t_max * D_IN + l * D_IN]
                    .copy_from_slice(ds.sender.span(start, l));
                ys[i * t_max * D_H..i * t_max * D_H + l * D_H]
                    .copy_from_slice(ds.receiver.span(start, l));
                mask[i * t_max..i * t_max + l].fill(1.0);
            }
            let xs = Tensor::from_vec(xs, (b, t_max, D_IN), &dev)?;
            let ys = Tensor::from_vec(ys, (b, t_max, D_H), &dev)?;
            let mask = Tensor::from_vec(mask, (b, t_max), &dev)?;
            let mut h = Tensor::zeros((b, D_H), DType::F32, &dev)?;
            let mut total = Tensor::zeros((), DType::F32, &dev)?;
            for t in 0..t_max {
                let x_t = xs.narrow(1, t, 1)?.squeeze(1)?.contiguous()?; // CUDA matmul needs contiguous
                h = cell.step(&x_t, &h)?.detach(); // graph truncated every step
                let y_t = ys.narrow(1, t, 1)?.squeeze(1)?;
                let m_t = mask.narrow(1, t, 1)?; // (b, 1)
                let r2 = h.sub(&y_t)?.sqr()?.broadcast_mul(&m_t)?.sum_all()?;
                total = total.add(&r2)?;
            }
            sse += total.to_scalar::<f32>()? as f64;
        }
        Ok(sse)
    };

    // ---- Training loop: full-sequence TBPTT with carried, detached state --
    // Each fit item is processed as ONE sequence from h_0 = 0 (item boundary
    // = state reset), stepping through SEQ_LEN-step windows: the autograd
    // graph spans one window (the scout's proven 16-step BPTT), the hidden
    // state carries ACROSS windows via detach — so the training-time h
    // distribution equals the probe's full-sequence deployment by
    // construction. Items are batched padded+masked; masked positions
    // contribute nothing to the loss.
    let mut curve_steps: Vec<serde_json::Value> = Vec::new();
    let mut curve_epochs: Vec<serde_json::Value> = Vec::new();
    let mut best: Option<(usize, f64, Vec<f32>)> = None; // (epoch, val_mse, weights)
    let mut shuffled = split.fit.clone();
    let mut global_step = 0usize;
    for epoch in 0..EPOCHS {
        let te = std::time::Instant::now();
        let mut rng = ChaCha8Rng::seed_from_u64(train_seed ^ (0x5EED_0000 + epoch as u64));
        shuffled.shuffle(&mut rng);
        let mut train_sse = 0f64;
        let mut train_n = 0usize;
        for chunk in shuffled.chunks(ITEM_BATCH) {
            let lens: Vec<usize> = chunk
                .iter()
                .map(|&r| (ds.index.token_offsets[r + 1] - ds.index.token_offsets[r]) as usize)
                .collect();
            let t_max = *lens.iter().max().unwrap();
            let b = chunk.len();
            let mut xs = vec![0f32; b * t_max * D_IN];
            let mut ys = vec![0f32; b * t_max * D_H];
            let mut mask = vec![0f32; b * t_max];
            for (i, (&r, &l)) in chunk.iter().zip(&lens).enumerate() {
                let start = ds.index.token_offsets[r] as usize;
                xs[i * t_max * D_IN..i * t_max * D_IN + l * D_IN]
                    .copy_from_slice(ds.sender.span(start, l));
                ys[i * t_max * D_H..i * t_max * D_H + l * D_H]
                    .copy_from_slice(ds.receiver.span(start, l));
                mask[i * t_max..i * t_max + l].fill(1.0);
            }
            let xs = Tensor::from_vec(xs, (b, t_max, D_IN), &dev)?;
            let ys = Tensor::from_vec(ys, (b, t_max, D_H), &dev)?;
            let mask = Tensor::from_vec(mask, (b, t_max), &dev)?;
            let mut h = Tensor::zeros((b, D_H), DType::F32, &dev)?;
            let mut w = 0usize;
            while w < t_max {
                let wlen = SEQ_LEN.min(t_max - w);
                // Valid (unmasked) positions in this window, from the lengths.
                let valid: usize = lens
                    .iter()
                    .map(|&l| l.min(w + wlen).saturating_sub(w))
                    .sum();
                let mut outs = Vec::with_capacity(wlen);
                for t in w..w + wlen {
                    let x_t = xs.narrow(1, t, 1)?.squeeze(1)?.contiguous()?;
                    h = cell.step(&x_t, &h)?;
                    outs.push(h.clone());
                }
                let pred = Tensor::stack(&outs, 1)?; // (b, wlen, D_H)
                let y_win = ys.narrow(1, w, wlen)?;
                let m_win = mask.narrow(1, w, wlen)?.reshape((b, wlen, 1))?;
                let sse_t = pred.sub(&y_win)?.sqr()?.broadcast_mul(&m_win)?.sum_all()?;
                let l = sse_t.affine(1.0 / (valid * D_H) as f64, 0.0)?;
                opt.backward_step(&l)?;
                h = h.detach(); // carry across windows, gradient truncated
                let lv = l.to_scalar::<f32>()? as f64;
                train_sse += lv * (valid * D_H) as f64;
                train_n += valid * D_H;
                if global_step % 200 == 0 {
                    curve_steps.push(
                        serde_json::json!({"step": global_step, "epoch": epoch, "train_mse": lv}),
                    );
                    println!("epoch {epoch} step {global_step}: train window mse {lv:.6}");
                }
                global_step += 1;
                w += wlen;
            }
        }
        let val_sse = eval_holdout(&cell)?;
        let val_mse = val_sse / (holdout_tokens.len() * D_H) as f64;
        let val_rel_residual = (val_sse / holdout_den).sqrt();
        let train_mse = train_sse / train_n as f64;
        println!(
            "epoch {epoch}: train mse {train_mse:.6}, full-seq holdout mse {val_mse:.6}, holdout rel residual {val_rel_residual:.6} ({:.1}s)",
            te.elapsed().as_secs_f32()
        );
        curve_epochs.push(serde_json::json!({
            "epoch": epoch, "train_mse_mean": train_mse, "holdout_mse": val_mse,
            "holdout_rel_residual_vs_fit_mean": val_rel_residual,
            "wall_s": te.elapsed().as_secs_f32(),
        }));
        if best.as_ref().map(|(_, b, _)| val_mse < *b).unwrap_or(true) {
            best = Some((epoch, val_mse, cell.to_flat()?));
        }
    }
    let (best_epoch, best_val_mse, best_flat) = best.unwrap();
    let best_rel_residual =
        (best_val_mse * (holdout_tokens.len() * D_H) as f64 / holdout_den).sqrt();
    println!("best epoch {best_epoch}: holdout mse {best_val_mse:.6} (rel residual {best_rel_residual:.6})");

    // ---- Artifact + golden sequences (trained network, CPU forward) -------
    cell.set_flat(&best_flat, &dev)?;
    let artifact = receipts_dir.join(format!("run2-m4-fastgrnn-r{rank}-cellL18toL14.f32bin"));
    let content_hash = cell.save_artifact(&artifact)?;
    println!(
        "artifact: {} content_hash {content_hash}",
        artifact.display()
    );
    let (golden_sha, inputs, outputs, pooled) =
        golden_pairs(&artifact, GOLDEN_SEED, GOLDEN_SEQS, GOLDEN_LEN)?;
    anyhow::ensure!(
        golden_sha == content_hash,
        "artifact hash drift during golden generation"
    );
    let golden_path =
        receipts_dir.join(format!("run2-m4-golden-fastgrnn-r{rank}-cellL18toL14.json"));
    std::fs::write(
        &golden_path,
        serde_json::to_string(&serde_json::json!({
            "artifact_file_sha256": content_hash,
            "input_seed_chacha8": GOLDEN_SEED,
            "n_seqs": GOLDEN_SEQS,
            "seq_len": GOLDEN_LEN,
            "note": "sequence outputs AND pooled payloads produced by the TRAINED network itself (artifact reloaded from disk, candle CPU forward from h0=0); probe-side hand-rolled sequential apply must match every per-step output and the mean-pooled payload to relative L2 error <= 1e-5 before any model runs (S2b/M3 golden discipline extended to sequences)",
            "inputs": inputs,
            "outputs": outputs,
            "pooled": pooled,
        }))?,
    )?;
    println!("golden sequences: {}", golden_path.display());

    // ---- Training receipt (written BEFORE this rank's probe invocation) ---
    let receipt = serde_json::json!({
        "stage": "run2-m4-fastgrnn-training",
        "design": "docs/adr/024-run2-trained-thought-adapter-ladder.md M4 (frozen FastGRNN cell, low-rank, sub-rung ladder r in {64,128,256} ascending; leakage rule per section 'Leakage discipline')",
        "cell": CELL,
        "rank": rank,
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
            "rule": "fit_holdout_split(2560, FIT_SPLIT_SEED) BY ITEM first (replica of harness/latentmesh-live/src/calibrate.rs:115, replica==harness ground truth pinned by unit test), THEN the 13 probe-overlap rows dropped from whichever side they landed in (ADR-024 frozen leakage rule); sequence windows are drawn WITHIN fit items only (no cross-item sequences)",
            "fit_split_seed": FIT_SPLIT_SEED,
            "n_fit_pre_exclusion": 2048, "n_holdout_pre_exclusion": 512,
            "n_fit": split.fit.len(), "n_holdout": split.holdout.len(),
            "excluded_probe_overlap_rows": split.excluded,
            "fit_rows_sha256_comma_joined": rows_sha256(&split.fit),
            "holdout_rows_sha256_comma_joined": rows_sha256(&split.holdout),
            "fit_token_pairs": fit_tokens.len(), "holdout_token_pairs": holdout_tokens.len(),
            "bptt_windows_per_epoch": bptt_windows_per_epoch,
        },
        "training": {
            "architecture": format!("low-rank FastGRNN cell, D_in 2048 -> D_h 1536, r={rank} (shared W,U between gate and candidate; W=W1@W2, U=U1@U2; h_t = z*h_prev + (sigmoid(zeta)(1-z)+sigmoid(nu))*h_tilde; Kusupati et al. arXiv:1901.02358, equations per M0 scout primary fetch)"),
            "param_count": param_count(rank),
            "param_count_formula": "r(D_in + D_h) + 2*r*D_h + 2*D_h + 2",
            "init": "factor matrices U(-1/sqrt(fan_in), +1/sqrt(fan_in)) from one seeded ChaCha8 stream in artifact-layout order; b_z=b_h=0; zeta_raw=1.0, nu_raw=-4.0 (EdgeML reference-implementation defaults, adopted because the scout pinned the equations but not the init)",
            "optimizer": {"name": "AdamW (candle-nn 0.9.2, the only Adam variant shipped — ADR-024 frozen optimizer constraint)",
                           "lr": LR, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8, "weight_decay": 0.01},
            "loss": "masked MSE over each BPTT window's valid (unpadded) positions (teacher-forced-capture states on both sides)",
            "sequences": {
                "bptt_steps": SEQ_LEN, "item_batch": ITEM_BATCH,
                "scheme": "full-sequence truncated BPTT with carried state: each fit item is ONE sequence from h_0 = 0 (state resets only at item boundaries — no cross-item sequences, ADR-024), stepped in 16-step gradient windows with the hidden state carried DETACHED across windows; items batched 64 at a time, padded + masked (the M0 scout's proven 16-step-BPTT compute pattern, with the training-time h distribution equal to full-sequence deployment by construction)",
            },
            "epochs": EPOCHS,
            "train_seed_chacha8": train_seed,
            "train_seed_rule": "TRAIN_SEED_BASE 0x4D340000 + rank",
            "stopping_rule": "fixed 10-epoch budget; artifact = epoch checkpoint with lowest FULL-SEQUENCE holdout MSE (each holdout item's whole generated span forward from h0=0 — the probe's exact forward, so selection matches deployment; frozen in this receipt before any frozen-probe invocation, ADR-024: the probe-invocation order is the freeze point)",
            "runs_performed": if rank == 64 { 2 } else { 1 },
            "superseded_run": if rank == 64 { serde_json::json!({
                "receipt": "run2-m4-training-receipt-cellL18toL14-r64-superseded-windowzeroinit.json",
                "artifact": "run2-m4-fastgrnn-r64-cellL18toL14-superseded-windowzeroinit.f32bin",
                "scheme": "independent zero-init 16-token windows (no state carry)",
                "why_superseded": "full-sequence holdout MSE DIVERGED across epochs (1.484 -> 7.868; best epoch 0, rel residual 1.327 > 1.0 constant-predictor) while window-train MSE improved — the exposure-bias signature of training only on 16-step zero-init horizons then deploying over 100-400 steps; superseded PRE-PROBE by the carried-state TBPTT scheme in this receipt",
                "discipline": "discovered entirely via the calibration-derived holdout metric; the frozen 40-item probe was NEVER invoked against the superseded artifact — zero probe bits consumed (ADR-024: the probe-invocation order is the freeze point, and it had not occurred); superseded trio preserved, not deleted, per the repo's kept-failure-receipt precedent",
            })} else { serde_json::json!(null) },
            "note_no_discarded_runs": if rank == 64 {
                "one preserved+disclosed pre-probe superseded run (see superseded_run); the run in this receipt is the single training run of the frozen carried-state scheme — no probe-informed retries ever"
            } else {
                "single training run at this rank; no restarts, no hyperparameter retries"
            },
            "disclosure_train_eval_match": "training, holdout metric, and probe all run the full per-item sequence from h_0 = 0 (gradient truncation at 16 steps is the only training-specific element) — the train/deployment distribution mismatch of the superseded window scheme is closed by construction",
        },
        "curves": {"per_epoch": curve_epochs, "sampled_steps_every_200": curve_steps},
        "results": {
            "best_epoch": best_epoch,
            "best_holdout_mse": best_val_mse,
            "best_holdout_rel_residual_vs_fit_mean": best_rel_residual,
            "holdout_mean_predictor_mse_variance_baseline": holdout_var_mse,
            "rel_residual_formula": "sqrt(sum ||pred-y||^2 / sum ||y-mu_r||^2) over holdout tokens, mu_r = fit-side per-dim mean (verbatim M3 formula for direct M3-vs-M4 comparability)",
        },
        "artifact": {
            "file": format!("run2-m4-fastgrnn-r{rank}-cellL18toL14.f32bin"),
            "layout": ARTIFACT_LAYOUT,
            "content_hash_sha256": content_hash,
            "golden_file": format!("run2-m4-golden-fastgrnn-r{rank}-cellL18toL14.json"),
            "golden_file_sha256": sha256_file(&golden_path)?,
            "golden_input_seed_chacha8": GOLDEN_SEED,
            "golden_seqs": GOLDEN_SEQS, "golden_seq_len": GOLDEN_LEN,
        },
        "eval_plan_frozen": {
            "probe": "ADR-023 frozen 40-item S1a/S2b protocol, inherited unchanged (item_seed_chacha8 20897 == 0x51A1, one-sided exact sign test alpha=0.05, 8 slots, rescale-to-natural-median, greedy/batch=1/max 400 tokens)",
            "payload_derivation": "AUTHORIAL CHOICE, recorded here before the probe (ADR-024's M4 text fixes the architecture and gate but not the pooling step): the FastGRNN consumes the sender's full generated-span per-token L18 sequence from h_0 = 0 and emits the translated sequence h_1..h_T; the 8-slot injection payload is the mean-pool (f64 accumulation) of that TRANSLATED sequence — sequence processing upstream of the pool, the direct sequence analog of M3's variant (i), which is exactly the pooling-hypothesis contrast M4 exists to test",
            "gate": "aligned_real (FastGRNN payload) > random, one-sided exact sign test, p < 0.05",
            "sub_rung_order": "r=64 first; on gate fail r=128 is trained+probed; on fail r=256; each rank one training run + one probe run, ALL receipts kept regardless of outcome (ADR-024's registered exception: three pre-declared architectures in a pre-declared order, not a sweep-until-pass)",
            "cell_scope": "S2-winner cell L18->L14 only, matching M3; anchor cell deliberately not probed (unregistered extra draw)",
        },
        "wall_clock_s": t0.elapsed().as_secs_f64(),
    });
    let receipt_path = receipts_dir.join(format!(
        "run2-m4-training-receipt-cellL18toL14-r{rank}.json"
    ));
    std::fs::write(&receipt_path, serde_json::to_string_pretty(&receipt)?)?;
    println!("training receipt: {}", receipt_path.display());
    println!("total wall clock: {:.1}s", t0.elapsed().as_secs_f64());
    Ok(())
}
