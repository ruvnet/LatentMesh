//! S2 calibration fit (design doc 024 §5, §7 S2): read the runtime's
//! teacher-forced hidden-state dump, fit the affine mean-centered Procrustes
//! transform for each of the 3×3 depth-sweep cells, evaluate the HELD-OUT
//! relative residual (the honest quality number — never the crate's on-train
//! confidence), select the winning cell, and write the calibration receipt.
//!
//! A6 acceptance gate (§6): held-out relative residual **< 0.9** for the
//! chosen depth pair.
//!
//! Held-out residual formula (recorded verbatim in the receipt):
//! `‖apply(X_h) − Y_h‖_F / ‖Y_h − μ_r‖_F`, where `apply` is the fitted
//! affine map `μ_r + α·(x − μ_s)·R`, `X_h`/`Y_h` are the held-out 20% of the
//! calibration pairs, and `μ_s`/`μ_r` are the FIT-SET means stored in the
//! hashed transform struct. The denominator is centered on μ_r so the large
//! mean offset of LLM hidden states cannot inflate the baseline and make the
//! gate trivially passable.
//!
//! Evidence label: deterministic CPU fit OVER live-model dumps — the vectors
//! were produced by live single-host GPU inference (see the dump receipt,
//! referenced by sha256); the fit itself involves no model and no simulation.

use latentmesh_align::AlignmentTransform;
use latentmesh_gate::Policy;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::Deserialize;
use std::path::Path;

use crate::gsm8k::sha256_hex;

/// 80/20 fit/held-out row split seed — an arbitrary constant committed
/// before any dump was read (design §5.3 / §6 seed discipline).
pub const FIT_SPLIT_SEED: u64 = 0x24C0_DE03;
/// A6: held-out relative residual threshold (design §6).
pub const A6_THRESHOLD: f32 = 0.9;
pub const HELD_OUT_RESIDUAL_FORMULA: &str =
    "||apply(X_h) - Y_h||_F / ||Y_h - mu_r||_F, apply(x) = mu_r + alpha*(x - mu_s)*R, \
     mu from the 80% fit set only";

/// One side of the dump manifest (sender or receiver).
#[derive(Debug, Deserialize)]
pub struct ManifestSide {
    pub model: String,
    pub n_layers: usize,
    pub hidden_size: usize,
    /// Swept 1-based block indices, e.g. [18, 24, 29].
    pub layers: Vec<usize>,
    /// Per-layer bin file name + sha256, keyed by layer index as a string.
    pub files: std::collections::BTreeMap<String, BinFile>,
}

#[derive(Debug, Deserialize)]
pub struct BinFile {
    pub file: String,
    pub sha256: String,
}

/// The runtime dump manifest (`manifest.json` in the dump directory).
#[derive(Debug, Deserialize)]
pub struct Manifest {
    pub n_rows: usize,
    pub item_indices: Vec<usize>,
    pub sender: ManifestSide,
    pub receiver: ManifestSide,
}

/// Rows of one dumped matrix, loaded and sha256-verified.
pub struct DepthMatrix {
    pub layer: usize,
    pub rows: Vec<Vec<f32>>,
}

/// Load one raw little-endian f32 bin file (`n_rows × dim`), verifying its
/// sha256 against the manifest.
pub fn load_bin(
    path: &Path,
    expect_sha: &str,
    n_rows: usize,
    dim: usize,
) -> anyhow::Result<Vec<Vec<f32>>> {
    let bytes = std::fs::read(path)?;
    let sha = sha256_hex(&bytes);
    anyhow::ensure!(
        sha == expect_sha,
        "{}: sha256 {sha} != manifest {expect_sha}",
        path.display()
    );
    anyhow::ensure!(
        bytes.len() == n_rows * dim * 4,
        "{}: {} bytes != {n_rows} rows x {dim} dims x 4",
        path.display(),
        bytes.len()
    );
    let mut rows = Vec::with_capacity(n_rows);
    for r in 0..n_rows {
        let mut row = Vec::with_capacity(dim);
        for c in 0..dim {
            let o = (r * dim + c) * 4;
            row.push(f32::from_le_bytes([
                bytes[o],
                bytes[o + 1],
                bytes[o + 2],
                bytes[o + 3],
            ]));
        }
        rows.push(row);
    }
    Ok(rows)
}

/// Deterministic 80/20 row split: ChaCha8-seeded shuffle of `0..n`, first
/// 80% fit, rest held out. The SAME split is used for all nine sweep cells,
/// so cell residuals are directly comparable.
pub fn fit_holdout_split(n: usize, seed: u64) -> (Vec<usize>, Vec<usize>) {
    let mut order: Vec<usize> = (0..n).collect();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    order.shuffle(&mut rng);
    let n_fit = n * 8 / 10;
    (order[..n_fit].to_vec(), order[n_fit..].to_vec())
}

/// Held-out relative residual per [`HELD_OUT_RESIDUAL_FORMULA`].
pub fn held_out_residual(t: &AlignmentTransform, x_h: &[&Vec<f32>], y_h: &[&Vec<f32>]) -> f32 {
    let (_, mu_r) = t.means();
    let mut num = 0f64;
    let mut den = 0f64;
    for (x, y) in x_h.iter().zip(y_h) {
        let pred = t.apply(x);
        for ((p, yy), m) in pred.iter().zip(y.iter()).zip(mu_r) {
            num += ((p - yy) as f64).powi(2);
            den += ((yy - m) as f64).powi(2);
        }
    }
    (num.sqrt() / den.sqrt().max(1e-12)) as f32
}

/// One sweep-cell result.
pub struct CellResult {
    pub sender_layer: usize,
    pub receiver_layer: usize,
    pub held_out_residual: f32,
    pub on_train_confidence: f32,
    pub content_hash: String,
    pub fit_seconds: f64,
    pub transform: AlignmentTransform,
}

/// Fit one cell on the fit rows and evaluate on the held-out rows.
pub fn fit_cell(
    sender: &DepthMatrix,
    receiver: &DepthMatrix,
    fit_idx: &[usize],
    holdout_idx: &[usize],
) -> CellResult {
    let t0 = std::time::Instant::now();
    let pairs: Vec<(Vec<f32>, Vec<f32>)> = fit_idx
        .iter()
        .map(|&i| (sender.rows[i].clone(), receiver.rows[i].clone()))
        .collect();
    let t = AlignmentTransform::fit_affine(&pairs);
    let fit_seconds = t0.elapsed().as_secs_f64();
    let x_h: Vec<&Vec<f32>> = holdout_idx.iter().map(|&i| &sender.rows[i]).collect();
    let y_h: Vec<&Vec<f32>> = holdout_idx.iter().map(|&i| &receiver.rows[i]).collect();
    let residual = held_out_residual(&t, &x_h, &y_h);
    CellResult {
        sender_layer: sender.layer,
        receiver_layer: receiver.layer,
        held_out_residual: residual,
        on_train_confidence: t.confidence,
        content_hash: t.content_hash(),
        fit_seconds,
        transform: t,
    }
}

/// Full S2 calibration: load dump, sweep 3×3, gate A6, write the winning
/// transform artifact + receipt JSON. Returns the receipt value.
pub fn run(dump_dir: &Path) -> anyhow::Result<serde_json::Value> {
    let t0 = std::time::Instant::now();
    let manifest_path = dump_dir.join("manifest.json");
    let manifest_bytes = std::fs::read(&manifest_path)
        .map_err(|e| anyhow::anyhow!("read {}: {e}", manifest_path.display()))?;
    let manifest_sha = sha256_hex(&manifest_bytes);
    let manifest: Manifest = serde_json::from_slice(&manifest_bytes)?;
    println!(
        "manifest: {} rows, sender {} ({} layers swept), receiver {} ({} swept)",
        manifest.n_rows,
        manifest.sender.model,
        manifest.sender.layers.len(),
        manifest.receiver.model,
        manifest.receiver.layers.len()
    );

    let load_side = |side: &ManifestSide| -> anyhow::Result<Vec<DepthMatrix>> {
        side.layers
            .iter()
            .map(|&layer| {
                let bf = side
                    .files
                    .get(&layer.to_string())
                    .ok_or_else(|| anyhow::anyhow!("layer {layer} missing from manifest files"))?;
                let rows = load_bin(
                    &dump_dir.join(&bf.file),
                    &bf.sha256,
                    manifest.n_rows,
                    side.hidden_size,
                )?;
                println!(
                    "loaded {} ({} rows x {})",
                    bf.file,
                    rows.len(),
                    side.hidden_size
                );
                Ok(DepthMatrix { layer, rows })
            })
            .collect()
    };
    let senders = load_side(&manifest.sender)?;
    let receivers = load_side(&manifest.receiver)?;

    let (fit_idx, holdout_idx) = fit_holdout_split(manifest.n_rows, FIT_SPLIT_SEED);
    // n>d gate (design §1 fix / §5.1): fit rows must exceed sender dim.
    let n_over_d = fit_idx.len() >= manifest.sender.hidden_size;
    anyhow::ensure!(
        n_over_d,
        "n<d: {} fit rows < sender dim {} — arbitrary null-space rotation (design §1)",
        fit_idx.len(),
        manifest.sender.hidden_size
    );

    let mut cells: Vec<CellResult> = Vec::new();
    for s in &senders {
        for r in &receivers {
            let cell = fit_cell(s, r, &fit_idx, &holdout_idx);
            println!(
                "cell L{}->L{}: held-out residual {:.4} (on-train confidence {:.4}, fit {:.1}s)",
                cell.sender_layer,
                cell.receiver_layer,
                cell.held_out_residual,
                cell.on_train_confidence,
                cell.fit_seconds
            );
            cells.push(cell);
        }
    }

    let winner = cells
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.held_out_residual.total_cmp(&b.1.held_out_residual))
        .map(|(i, _)| i)
        .expect("nine cells");
    let a6_pass = cells[winner].held_out_residual < A6_THRESHOLD;

    // §5.6: register the winning transform hash via Policy::trust_transform.
    let mut policy = Policy::new(0.8);
    policy.trust_transform(cells[winner].content_hash.clone());

    // Persist the winning transform artifact (S3 needs it; refitting costs
    // minutes). Stays in the run dir — never in the committed receipts dir.
    let w = &cells[winner];
    let artifact = dump_dir.join(format!(
        "transform-L{}-to-L{}.json",
        w.sender_layer, w.receiver_layer
    ));
    std::fs::write(&artifact, serde_json::to_vec(&w.transform)?)?;
    let artifact_sha = sha256_hex(&std::fs::read(&artifact)?);

    let cell_rows: Vec<serde_json::Value> = cells
        .iter()
        .map(|c| {
            serde_json::json!({
                "sender_layer": c.sender_layer,
                "receiver_layer": c.receiver_layer,
                "held_out_relative_residual": c.held_out_residual,
                "on_train_confidence_OPTIMISTIC_not_the_quality_number": c.on_train_confidence,
                "transform_content_hash": c.content_hash,
                "fit_seconds": c.fit_seconds,
            })
        })
        .collect();

    let receipt = serde_json::json!({
        "stage": "S2-calibration-fit",
        "design": "docs/research/024-live-latent-experiment-design.md sections 5, 6 (A6), 7 S2",
        "evidence_label": "deterministic CPU fit over live-model dumps (dump receipt referenced by sha256 below); the fit itself involves no model and no simulation",
        "dump": {
            "dir": dump_dir.display().to_string(),
            "manifest_sha256": manifest_sha,
            "n_rows": manifest.n_rows,
            "item_indices_min": manifest.item_indices.iter().min(),
            "item_indices_max": manifest.item_indices.iter().max(),
            "sender": {"model": manifest.sender.model, "hidden_size": manifest.sender.hidden_size,
                        "layers": manifest.sender.layers,
                        "files": manifest.sender.files.iter().map(|(k, v)| (k.clone(), serde_json::json!({"file": v.file, "sha256": v.sha256}))).collect::<serde_json::Map<_,_>>()},
            "receiver": {"model": manifest.receiver.model, "hidden_size": manifest.receiver.hidden_size,
                          "layers": manifest.receiver.layers,
                          "files": manifest.receiver.files.iter().map(|(k, v)| (k.clone(), serde_json::json!({"file": v.file, "sha256": v.sha256}))).collect::<serde_json::Map<_,_>>()},
        },
        "fit": {
            "method": "AlignmentTransform::fit_affine (affine mean-centered semi-orthogonal Procrustes, dense rectangular SVD path)",
            "split_seed_chacha8": FIT_SPLIT_SEED,
            "split": "seeded shuffle of row indices; first 80% fit, last 20% held out; SAME split for all nine cells",
            "n_fit_rows": fit_idx.len(),
            "n_holdout_rows": holdout_idx.len(),
            "n_over_d_gate": {"fit_rows": fit_idx.len(), "sender_dim": manifest.sender.hidden_size, "pass": n_over_d},
            "held_out_residual_formula": HELD_OUT_RESIDUAL_FORMULA,
        },
        "sweep_cells": cell_rows,
        "winner": {
            "sender_layer": w.sender_layer,
            "receiver_layer": w.receiver_layer,
            "held_out_relative_residual": w.held_out_residual,
            "transform_content_hash": w.content_hash,
            "transform_artifact": {"file": artifact.display().to_string(), "sha256": artifact_sha},
            "policy_trust_transform_registered": true,
        },
        "gate_A6": {"threshold": A6_THRESHOLD, "statistic": "held-out relative residual of the winning cell",
                     "measured": w.held_out_residual, "pass": a6_pass},
        "wall_clock_s": t0.elapsed().as_secs_f64(),
        "unix_time": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0),
    });
    Ok(receipt)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_is_deterministic_and_80_20() {
        let (f1, h1) = fit_holdout_split(4000, FIT_SPLIT_SEED);
        let (f2, h2) = fit_holdout_split(4000, FIT_SPLIT_SEED);
        assert_eq!(f1, f2);
        assert_eq!(h1, h2);
        assert_eq!(f1.len(), 3200);
        assert_eq!(h1.len(), 800);
        let mut all: Vec<usize> = f1.iter().chain(h1.iter()).copied().collect();
        all.sort_unstable();
        assert_eq!(all, (0..4000).collect::<Vec<_>>());
    }

    #[test]
    fn held_out_residual_zero_on_exact_affine_map() {
        // y = mu + x*R for an exact rotation: residual on held-out points
        // generated by the same map must be ~0.
        use rand::Rng;
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let n = 40;
        let d = 8;
        let angle = 0.7f32;
        let rot = |x: &[f32]| {
            let mut y = x.to_vec();
            y[0] = x[0] * angle.cos() - x[1] * angle.sin();
            y[1] = x[0] * angle.sin() + x[1] * angle.cos();
            y
        };
        let pairs: Vec<(Vec<f32>, Vec<f32>)> = (0..n)
            .map(|_| {
                let x: Vec<f32> = (0..d).map(|_| rng.gen_range(-1.0..1.0)).collect();
                let y: Vec<f32> = rot(&x).iter().map(|v| v + 3.0).collect();
                (x, y)
            })
            .collect();
        let t = AlignmentTransform::fit_affine(&pairs[..30]);
        let x_h: Vec<&Vec<f32>> = pairs[30..].iter().map(|(x, _)| x).collect();
        let y_h: Vec<&Vec<f32>> = pairs[30..].iter().map(|(_, y)| y).collect();
        let r = held_out_residual(&t, &x_h, &y_h);
        assert!(
            r < 0.05,
            "exact affine map should fit near-perfectly, got {r}"
        );
    }

    #[test]
    fn a6_threshold_matches_design() {
        assert_eq!(A6_THRESHOLD, 0.9);
    }
}
