//! A6 permutation-null baseline (ADR-024 "Registered analysis (protocol-safe,
//! no probe draw)"): what does the SAME affine fit score on the SAME held-out
//! residual metric when the sender↔receiver row pairing is destroyed?
//!
//! Motivation (registered verbatim in ADR-024): the PRH-critique literature
//! (arXiv:2602.14486) shows representation-similarity metrics can be inflated
//! by depth/width artifacts, and the A6 pass threshold (< 0.9) was never
//! calibrated against chance. This module computes the chance level directly.
//!
//! Protocol safety: no probe is drawn, no model is run, no recorded A6 number
//! is changed. It re-reads the committed S2 (gold) and S2c (generated) dumps,
//! re-fits `AlignmentTransform::fit_affine` under permuted pairings, and
//! annotates the recorded outcomes.
//!
//! **Shuffle choice — item-level, within-split.** Both dumps carry exactly one
//! row per GSM8K item (`manifest.item_indices` is a bijection onto rows: 4000
//! unique indices over 4000 rows for S2, 2560 over 2560 for S2c, each row a
//! mean-pool over that item's solution/generated token span). A row shuffle is
//! therefore *identically* an item shuffle — there is no within-item slot
//! structure left to break, and no need for a block permutation. The
//! permutation is applied **within each side of the 80/20 split** (fit rows
//! permute among fit rows, held-out rows among held-out rows), so every
//! permuted fit uses the exact same row memberships, the exact same marginal
//! distributions, and the exact same evaluation rows as the real fit — only
//! the pairing changes. Permuting globally instead would let a held-out
//! receiver row also appear in the fit set, which can only *help* the null and
//! would make the test anti-conservative.
//!
//! Permutations are uniform (not derangements): a uniform permutation is the
//! textbook null, and the ~1 expected fixed point in 3200 rows is recorded
//! per-permutation in the receipt rather than engineered away.
//!
//! Evidence label: deterministic CPU analysis over committed dumps.

use crate::calibrate::{
    fit_cell, fit_holdout_split, load_bin, DepthMatrix, Manifest, A6_THRESHOLD, FIT_SPLIT_SEED,
    HELD_OUT_RESIDUAL_FORMULA,
};
use crate::calibrate_gen::REGISTERED_CELLS;
use crate::gsm8k::sha256_hex;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

/// Per-permutation seed base — an arbitrary constant fixed here, before any
/// null residual was computed. Permutation `k` uses `PERM_SEED_BASE + k`.
pub const PERM_SEED_BASE: u64 = 0x00A6_2026_0828_0001;
/// Registered minimum permutation count (ADR-024 asks for ≥ 20).
pub const DEFAULT_PERMUTATIONS: usize = 20;
/// Worker threads for the (single-threaded, nalgebra) per-permutation fits.
pub const DEFAULT_THREADS: usize = 8;

/// Build the row permutation σ: `receiver_permuted.rows[i] = receiver.rows[σ[i]]`,
/// shuffled **within** the fit block and **within** the held-out block.
pub fn within_split_permutation(
    n: usize,
    fit_idx: &[usize],
    holdout_idx: &[usize],
    seed: u64,
) -> Vec<usize> {
    let mut sigma: Vec<usize> = (0..n).collect();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    for block in [fit_idx, holdout_idx] {
        let mut vals: Vec<usize> = block.to_vec();
        vals.shuffle(&mut rng);
        for (pos, val) in block.iter().zip(&vals) {
            sigma[*pos] = *val;
        }
    }
    sigma
}

/// Number of rows a permutation leaves paired with themselves.
pub fn fixed_points(sigma: &[usize]) -> usize {
    sigma.iter().enumerate().filter(|(i, s)| i == *s).count()
}

fn permute_rows(m: &DepthMatrix, sigma: &[usize]) -> DepthMatrix {
    DepthMatrix {
        layer: m.layer,
        rows: sigma.iter().map(|&s| m.rows[s].clone()).collect(),
    }
}

/// Summary statistics of one null distribution against the real value.
pub struct NullStats {
    pub mean: f64,
    pub sd: f64,
    pub min: f64,
    pub max: f64,
    pub z: f64,
    pub n_le_real: usize,
    pub percentile: f64,
    pub p_one_sided: f64,
}

/// Mean/sd/min/max of `nulls`, plus where `real` falls. Lower residual = better,
/// so a large NEGATIVE z means the real pairing beats chance.
pub fn null_stats(nulls: &[f64], real: f64) -> NullStats {
    let n = nulls.len().max(1) as f64;
    let mean = nulls.iter().sum::<f64>() / n;
    let var = if nulls.len() > 1 {
        nulls.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)
    } else {
        0.0
    };
    let sd = var.sqrt();
    let min = nulls.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = nulls.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let n_le_real = nulls.iter().filter(|v| **v <= real).count();
    NullStats {
        mean,
        sd,
        min,
        max,
        z: if sd > 0.0 { (real - mean) / sd } else { 0.0 },
        n_le_real,
        percentile: n_le_real as f64 / n,
        // Standard permutation p-value with the +1 correction (Phipson &
        // Smyth): the observed value is itself one of the achievable outcomes.
        p_one_sided: (n_le_real as f64 + 1.0) / (n + 1.0),
    }
}

/// One cell's real fit plus its permutation null.
pub struct CellNull {
    pub sender_layer: usize,
    pub receiver_layer: usize,
    pub real_residual: f64,
    pub real_content_hash: String,
    pub perms: Vec<(u64, f64, usize)>, // (seed, residual, fixed points)
}

impl CellNull {
    pub fn nulls(&self) -> Vec<f64> {
        self.perms.iter().map(|(_, r, _)| *r).collect()
    }
}

/// Fit one cell for real, then for `n_perm` permuted pairings.
fn run_cell(
    sender: &DepthMatrix,
    receiver: &DepthMatrix,
    fit_idx: &[usize],
    holdout_idx: &[usize],
    n_perm: usize,
    threads: usize,
) -> CellNull {
    let real = fit_cell(sender, receiver, fit_idx, holdout_idx);
    println!(
        "cell L{}->L{}: REAL held-out residual {:.6} (content hash {}…)",
        sender.layer,
        receiver.layer,
        real.held_out_residual,
        &real.content_hash[..12]
    );

    let n = sender.rows.len();
    let next = AtomicUsize::new(0);
    let out: Mutex<Vec<Option<(u64, f64, usize)>>> = Mutex::new(vec![None; n_perm]);
    std::thread::scope(|scope| {
        for _ in 0..threads.max(1) {
            scope.spawn(|| loop {
                let k = next.fetch_add(1, Ordering::SeqCst);
                if k >= n_perm {
                    break;
                }
                let seed = PERM_SEED_BASE + k as u64;
                let sigma = within_split_permutation(n, fit_idx, holdout_idx, seed);
                let fp = fixed_points(&sigma);
                let permuted = permute_rows(receiver, &sigma);
                let c = fit_cell(sender, &permuted, fit_idx, holdout_idx);
                println!(
                    "  perm {k:>3} (seed {seed:#x}): null residual {:.6} (fixed points {fp})",
                    c.held_out_residual
                );
                out.lock().expect("perm result lock")[k] =
                    Some((seed, c.held_out_residual as f64, fp));
            });
        }
    });
    let perms: Vec<(u64, f64, usize)> = out
        .into_inner()
        .expect("perm results")
        .into_iter()
        .map(|v| v.expect("every permutation ran"))
        .collect();

    CellNull {
        sender_layer: sender.layer,
        receiver_layer: receiver.layer,
        real_residual: real.held_out_residual as f64,
        real_content_hash: real.content_hash,
        perms,
    }
}

/// A loaded dataset: manifest, manifest sha256, and one (sender, receiver)
/// matrix pair per registered cell.
type LoadedDataset = (Manifest, String, Vec<(DepthMatrix, DepthMatrix)>);

/// Load one dump's manifest and the registered cells' layer matrices.
fn load_dataset(dump_dir: &Path) -> anyhow::Result<LoadedDataset> {
    let manifest_bytes = std::fs::read(dump_dir.join("manifest.json"))
        .map_err(|e| anyhow::anyhow!("read {}/manifest.json: {e}", dump_dir.display()))?;
    let manifest_sha = sha256_hex(&manifest_bytes);
    let manifest: Manifest = serde_json::from_slice(&manifest_bytes)?;
    anyhow::ensure!(
        manifest.item_indices.len() == manifest.n_rows,
        "manifest item_indices ({}) != n_rows ({}) — the one-row-per-item premise of the \
         item-level shuffle does not hold",
        manifest.item_indices.len(),
        manifest.n_rows
    );
    let uniq: std::collections::BTreeSet<usize> = manifest.item_indices.iter().copied().collect();
    anyhow::ensure!(
        uniq.len() == manifest.n_rows,
        "manifest item_indices are not unique ({} unique of {} rows) — rows are grouped by item, \
         so a row shuffle is NOT an item shuffle; use a block permutation instead",
        uniq.len(),
        manifest.n_rows
    );

    let load =
        |side: &crate::calibrate::ManifestSide, layer: usize| -> anyhow::Result<DepthMatrix> {
            let bf = side
                .files
                .get(&layer.to_string())
                .ok_or_else(|| anyhow::anyhow!("layer {layer} missing from manifest"))?;
            let rows = load_bin(
                &dump_dir.join(&bf.file),
                &bf.sha256,
                manifest.n_rows,
                side.hidden_size,
            )?;
            Ok(DepthMatrix { layer, rows })
        };
    let mut cells = Vec::new();
    for (sl, rl) in REGISTERED_CELLS {
        cells.push((load(&manifest.sender, sl)?, load(&manifest.receiver, rl)?));
    }
    Ok((manifest, manifest_sha, cells))
}

fn cell_json(c: &CellNull, s: &NullStats, n_perm: usize) -> serde_json::Value {
    serde_json::json!({
        "cell": format!("L{}->L{}", c.sender_layer, c.receiver_layer),
        "sender_layer": c.sender_layer,
        "receiver_layer": c.receiver_layer,
        "real_held_out_relative_residual": c.real_residual,
        "real_transform_content_hash": c.real_content_hash,
        "real_A6_pass": c.real_residual < A6_THRESHOLD as f64,
        "null": {
            "n_permutations": n_perm,
            "mean": s.mean, "sd": s.sd, "min": s.min, "max": s.max,
            "n_below_or_equal_real": s.n_le_real,
            "empirical_percentile_of_real": s.percentile,
            "p_one_sided_plus_one_corrected": s.p_one_sided,
            "z_of_real_vs_null": s.z,
            "A6_pass_count_under_null": c.nulls().iter().filter(|v| **v < A6_THRESHOLD as f64).count(),
        },
        "permutations": c.perms.iter().enumerate().map(|(k, (seed, r, fp))| serde_json::json!({
            "k": k, "seed_chacha8": seed, "held_out_relative_residual": r, "fixed_points": fp,
        })).collect::<Vec<_>>(),
    })
}

/// Full registered analysis over both calibration datasets. Returns the receipt.
pub fn run(
    gold_dir: &Path,
    gen_dir: &Path,
    n_perm: usize,
    threads: usize,
) -> anyhow::Result<serde_json::Value> {
    let t0 = std::time::Instant::now();
    let mut datasets = Vec::new();
    for (label, dir, stage) in [
        ("gold-s2", gold_dir, "S2-calibration-fit"),
        ("generated-s2c", gen_dir, "S2c-generated-calibration-fit"),
    ] {
        println!("=== dataset {label}: {}", dir.display());
        let (manifest, manifest_sha, cells) = load_dataset(dir)?;
        let (fit_idx, holdout_idx) = fit_holdout_split(manifest.n_rows, FIT_SPLIT_SEED);
        println!(
            "  {} rows ({} fit / {} holdout), sender {} d={}, receiver {} d={}",
            manifest.n_rows,
            fit_idx.len(),
            holdout_idx.len(),
            manifest.sender.model,
            manifest.sender.hidden_size,
            manifest.receiver.model,
            manifest.receiver.hidden_size
        );
        let mut cell_rows = Vec::new();
        for (s, r) in &cells {
            let cn = run_cell(s, r, &fit_idx, &holdout_idx, n_perm, threads);
            let stats = null_stats(&cn.nulls(), cn.real_residual);
            println!(
                "  => L{}->L{}: real {:.4} vs null {:.4} ± {:.4} (min {:.4}); z = {:.2}, {} of {n_perm} nulls ≤ real",
                cn.sender_layer, cn.receiver_layer, cn.real_residual,
                stats.mean, stats.sd, stats.min, stats.z, stats.n_le_real
            );
            cell_rows.push(cell_json(&cn, &stats, n_perm));
        }
        datasets.push(serde_json::json!({
            "label": label,
            "source_receipt_stage": stage,
            "dump": {"dir": dir.display().to_string(), "manifest_sha256": manifest_sha,
                      "n_rows": manifest.n_rows,
                      "sender": {"model": manifest.sender.model, "hidden_size": manifest.sender.hidden_size},
                      "receiver": {"model": manifest.receiver.model, "hidden_size": manifest.receiver.hidden_size}},
            "split": {"seed_chacha8": FIT_SPLIT_SEED, "n_fit_rows": fit_idx.len(),
                       "n_holdout_rows": holdout_idx.len()},
            "cells": cell_rows,
        }));
    }

    Ok(serde_json::json!({
        "stage": "run2-A6-permutation-null",
        "design": "docs/adr/024 'Registered analysis (protocol-safe, no probe draw)'; annotates (never changes) the recorded A6 outcomes in s2-calibration-receipt.json and s2c-calibration-receipt.json",
        "evidence_label": "deterministic CPU analysis over committed dumps",
        "analysis": "docs/research/029-a6-permutation-null.md",
        "method": {
            "fit": "AlignmentTransform::fit_affine — the SAME machinery as S2/S2c (calibrate::fit_cell), unchanged",
            "held_out_residual_formula": HELD_OUT_RESIDUAL_FORMULA,
            "split": "calibrate::fit_holdout_split with the recorded FIT_SPLIT_SEED — identical 80/20 rows as the real fits",
            "shuffle": "ITEM-LEVEL: each dump row is one GSM8K item (item_indices is a bijection onto rows — asserted at load), so a row permutation IS an item permutation; applied WITHIN the fit block and WITHIN the held-out block so row memberships, marginals and evaluation rows are identical to the real fit and only the pairing changes",
            "shuffle_rejected_alternative": "global (cross-split) permutation — it lets a held-out receiver row also appear in the fit set, which can only help the null and would make the test anti-conservative",
            "permutation_family": "uniform permutation (not a derangement); fixed-point count recorded per permutation",
            "seeds": {"base_chacha8": PERM_SEED_BASE, "rule": "permutation k uses PERM_SEED_BASE + k"},
            "statistics": "null mean/sd/min/max; z = (real - null_mean)/null_sd (lower residual is better, so a large NEGATIVE z means the real pairing beats chance); empirical percentile = #{null <= real}/N; p = (#{null <= real} + 1)/(N + 1)",
            "A6_threshold_for_reference": A6_THRESHOLD,
            "analytic_anchor": "the trivial predictor 'output mu_r for every input' scores EXACTLY 1.0 on this metric: the numerator becomes sum||mu_r - y||^2 and the denominator is sum||y - mu_r||^2 over the same held-out rows and the same fit-set mu_r. 1.0, not the 0.9 gate, is the do-nothing level — so a null (or real) residual above 1.0 is worse than predicting the fit-set mean for every input",
        },
        "n_permutations_per_cell": n_perm,
        "threads": threads,
        "datasets": datasets,
        "wall_clock_s": t0.elapsed().as_secs_f64(),
        "unix_time": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permutation_is_deterministic_and_within_split() {
        let (fit, hold) = fit_holdout_split(100, FIT_SPLIT_SEED);
        let a = within_split_permutation(100, &fit, &hold, PERM_SEED_BASE);
        let b = within_split_permutation(100, &fit, &hold, PERM_SEED_BASE);
        assert_eq!(a, b);
        // It is a permutation of 0..100.
        let mut sorted = a.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..100).collect::<Vec<_>>());
        // Fit positions map to fit rows; held-out positions to held-out rows.
        let fit_set: std::collections::BTreeSet<usize> = fit.iter().copied().collect();
        for i in &fit {
            assert!(fit_set.contains(&a[*i]), "fit row {i} left its block");
        }
        for i in &hold {
            assert!(!fit_set.contains(&a[*i]), "held-out row {i} left its block");
        }
    }

    #[test]
    fn different_seeds_give_different_permutations() {
        let (fit, hold) = fit_holdout_split(100, FIT_SPLIT_SEED);
        let a = within_split_permutation(100, &fit, &hold, PERM_SEED_BASE);
        let b = within_split_permutation(100, &fit, &hold, PERM_SEED_BASE + 1);
        assert_ne!(a, b);
        assert!(fixed_points(&a) < 100);
    }

    #[test]
    fn null_stats_place_the_real_value() {
        let nulls = vec![1.0, 1.02, 0.98, 1.01, 0.99];
        let s = null_stats(&nulls, 0.5);
        assert!((s.mean - 1.0).abs() < 1e-9);
        assert!(s.sd > 0.0);
        assert_eq!(s.n_le_real, 0);
        assert!(s.z < -10.0, "real far below the null, got z = {}", s.z);
        assert!((s.p_one_sided - 1.0 / 6.0).abs() < 1e-9);
    }

    #[test]
    fn null_stats_handle_a_real_value_inside_the_null() {
        let nulls = vec![0.4, 0.5, 0.6, 0.7];
        let s = null_stats(&nulls, 0.55);
        assert_eq!(s.n_le_real, 2);
        assert!((s.percentile - 0.5).abs() < 1e-9);
    }
}
