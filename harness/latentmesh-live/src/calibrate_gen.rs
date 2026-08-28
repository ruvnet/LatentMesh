//! S2c generated-pairs calibration fit (ADR-023 Deviation 7 contingency;
//! design doc 024 §5.4 / §8 risk 6). Consumes the runtime's
//! `s2c_generated_dump` (pairs pooled over sender-GENERATED reasoning, same
//! text through both models) and fits `AlignmentTransform::fit_affine` for
//! **exactly the two ADR-023-registered cells** — L18→L14 (S2 winner) and
//! L24→L19 (Deviation 6 anchor). No 3×3 sweep re-opening: the sweep was an
//! S2 selection device; the contingency re-tests the registered cells only.
//!
//! Everything statistical is reused verbatim from the S2 fit
//! ([`calibrate`](crate::calibrate)): the same `FIT_SPLIT_SEED` 80/20 split
//! machinery, the same [`held_out_residual`](crate::calibrate::held_out_residual)
//! A6 formula, the same `A6_THRESHOLD` (< 0.9).
//!
//! PRE-COMMITTED BRANCHES (decided here, before any generated-pairs residual
//! is computed):
//! - **Winner rule**: of the two cells, the one with the LOWER held-out
//!   relative residual **on generated pairs** carries the first re-probe
//!   (mirror of the registered §5.3 min-residual rule). If the probe fails
//!   there, the other cell is probed once — mirroring the ADR-023
//!   Deviation 6 fallback chain. No further iteration under any outcome.
//! - **A6-fail branch**: a cell failing A6 (residual ≥ 0.9) is NOT probed —
//!   A6's registered role is "transform is usable", and probing an unusable
//!   transform would burn the one-shot re-probe on a known-bad artifact. If
//!   BOTH cells fail A6 on generated pairs, the contingency outcome is
//!   CONTINGENCY-FAILED without a probe run (the aligned-real gate cannot
//!   pass through a transform its own gate rejects).
//! - **n>d handling**: the S2 fit ABORTS below n_fit = 2048 because the item
//!   count was controllable there. Here n is budget-laddered by the dump
//!   (its receipt carries the arithmetic), so the gate is RECORDED, not
//!   fatal: n_fit ≥ 2048 (= d_sender) is the registered design §1 gate;
//!   n_fit ≥ 1536 (= d_receiver) is the floor below which the Procrustes
//!   polar factor itself is non-unique (rank(AᵀB) < d_receiver ⇒ zero
//!   singular values ⇒ arbitrary completion; ADR-002's MᵀM-fallback
//!   equivalence holds only on full-rank problems). Both are reported.
//!
//! For each fitted cell this module also emits the golden input/output
//! pairs the S2b probe's hand-rolled affine apply is verified against
//! (produced by `AlignmentTransform::apply` itself, mirroring the committed
//! `s2b-golden-affine-*.json` format).

use crate::calibrate::{
    fit_cell, fit_holdout_split, load_bin, CellResult, DepthMatrix, Manifest, A6_THRESHOLD,
    FIT_SPLIT_SEED, HELD_OUT_RESIDUAL_FORMULA,
};
use crate::gsm8k::sha256_hex;
use latentmesh_gate::Policy;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::path::Path;

/// The two ADR-023-registered cells: (sender_layer, receiver_layer).
pub const REGISTERED_CELLS: [(usize, usize); 2] = [(18, 14), (24, 19)];
/// Golden-pair input seed (fresh constant, committed before any fit ran;
/// same 5.0×standard-normal Box-Muller generator as the S2b goldens).
pub const GOLDEN_INPUT_SEED: u64 = 0x2C08_2807;
pub const GOLDEN_PAIRS: usize = 8;

/// Standard-normal via Box-Muller over ChaCha8 (the S2b golden generator).
fn gaussian_vec(rng: &mut ChaCha8Rng, n: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(n);
    while v.len() < n {
        let u1: f64 = rng.gen_range(f64::MIN_POSITIVE..1.0);
        let u2: f64 = rng.gen::<f64>();
        let r = (-2.0 * u1.ln()).sqrt();
        let t = 2.0 * std::f64::consts::PI * u2;
        v.push((r * t.cos()) as f32);
        if v.len() < n {
            v.push((r * t.sin()) as f32);
        }
    }
    v
}

/// Write the transform artifact (exact `serde_json::to_vec` bytes, so file
/// sha256 == `content_hash`) plus its golden apply pairs. Returns
/// (artifact_file, artifact_sha, golden_file, golden_sha).
fn write_artifacts(
    dir: &Path,
    cell: &CellResult,
    git_commit: &str,
) -> anyhow::Result<(String, String, String, String)> {
    let t_name = format!(
        "transform-gen-L{}-to-L{}.json",
        cell.sender_layer, cell.receiver_layer
    );
    let bytes = serde_json::to_vec(&cell.transform)?;
    let sha = sha256_hex(&bytes);
    anyhow::ensure!(
        sha == cell.content_hash,
        "artifact bytes sha {sha} != content_hash {} — hash contract broken",
        cell.content_hash
    );
    std::fs::write(dir.join(&t_name), &bytes)?;

    let mut rng = ChaCha8Rng::seed_from_u64(GOLDEN_INPUT_SEED);
    let (d_s, _) = cell.transform.dims();
    let inputs: Vec<Vec<f32>> = (0..GOLDEN_PAIRS)
        .map(|_| {
            gaussian_vec(&mut rng, d_s)
                .iter()
                .map(|v| v * 5.0)
                .collect()
        })
        .collect();
    let outputs: Vec<Vec<f32>> = inputs.iter().map(|x| cell.transform.apply(x)).collect();
    let g_name = format!(
        "s2c-golden-affine-L{}-to-L{}.json",
        cell.sender_layer, cell.receiver_layer
    );
    let golden = serde_json::json!({
        "transform_content_hash": cell.content_hash,
        "transform_file_sha256": sha,
        "dim_sender": cell.transform.dims().0,
        "dim_receiver": cell.transform.dims().1,
        "input_seed_chacha8": GOLDEN_INPUT_SEED,
        "input_distribution": "5.0 x standard normal (Box-Muller over ChaCha8), deterministic",
        "n_pairs": GOLDEN_PAIRS,
        "inputs": inputs,
        "outputs": outputs,
        "provenance": "produced by latentmesh-align::AlignmentTransform::apply (the crate's own affine apply), golden reference for the S2b hand-rolled apply (ADR-023 Deviation 7 generated-pairs contingency)",
        "repo_git_commit": git_commit,
    });
    let g_bytes = serde_json::to_vec_pretty(&golden)?;
    std::fs::write(dir.join(&g_name), &g_bytes)?;
    Ok((t_name, sha, g_name, sha256_hex(&g_bytes)))
}

/// Full S2c fit: load the generated-pairs dump, fit the two registered
/// cells, gate A6 per cell, pick the winner among A6-passers, write
/// artifacts + goldens + receipt. Returns the receipt value.
pub fn run(dump_dir: &Path) -> anyhow::Result<serde_json::Value> {
    let t0 = std::time::Instant::now();
    let manifest_bytes = std::fs::read(dump_dir.join("manifest.json"))
        .map_err(|e| anyhow::anyhow!("read {}/manifest.json: {e}", dump_dir.display()))?;
    let manifest_sha = sha256_hex(&manifest_bytes);
    let manifest: Manifest = serde_json::from_slice(&manifest_bytes)?;
    println!(
        "manifest: {} rows, sender {} layers {:?}, receiver {} layers {:?}",
        manifest.n_rows,
        manifest.sender.model,
        manifest.sender.layers,
        manifest.receiver.model,
        manifest.receiver.layers
    );

    let load_layer = |side: &crate::calibrate::ManifestSide, layer: usize| {
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
        anyhow::Ok(DepthMatrix { layer, rows })
    };

    let (fit_idx, holdout_idx) = fit_holdout_split(manifest.n_rows, FIT_SPLIT_SEED);
    let n_fit = fit_idx.len();
    let d_sender = manifest.sender.hidden_size;
    let d_receiver = manifest.receiver.hidden_size;
    let n_over_d = n_fit >= d_sender;
    let polar_unique = n_fit >= d_receiver;
    println!(
        "split: {n_fit} fit / {} holdout; n>d(d_sender {d_sender}): {n_over_d}; \
         polar-uniqueness floor (d_receiver {d_receiver}): {polar_unique}",
        holdout_idx.len()
    );

    let git_commit = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();

    let mut cells: Vec<CellResult> = Vec::new();
    let mut artifacts: Vec<serde_json::Value> = Vec::new();
    for (sl, rl) in REGISTERED_CELLS {
        let s = load_layer(&manifest.sender, sl)?;
        let r = load_layer(&manifest.receiver, rl)?;
        let cell = fit_cell(&s, &r, &fit_idx, &holdout_idx);
        println!(
            "cell L{sl}->L{rl}: held-out residual {:.4} (on-train confidence {:.4}, fit {:.1}s) A6 {}",
            cell.held_out_residual,
            cell.on_train_confidence,
            cell.fit_seconds,
            if cell.held_out_residual < A6_THRESHOLD { "PASS" } else { "FAIL" }
        );
        let (t_file, t_sha, g_file, g_sha) = write_artifacts(dump_dir, &cell, &git_commit)?;
        artifacts.push(serde_json::json!({
            "cell": format!("L{sl}->L{rl}"),
            "transform_artifact": {"file": t_file, "sha256_equals_content_hash": t_sha},
            "golden_pairs": {"file": g_file, "sha256": g_sha,
                              "input_seed_chacha8": GOLDEN_INPUT_SEED, "n_pairs": GOLDEN_PAIRS},
        }));
        cells.push(cell);
    }

    let a6_pass: Vec<bool> = cells
        .iter()
        .map(|c| c.held_out_residual < A6_THRESHOLD)
        .collect();
    // Winner among A6-passers by min held-out residual (pre-committed rule).
    let winner = cells
        .iter()
        .enumerate()
        .filter(|(i, _)| a6_pass[*i])
        .min_by(|a, b| a.1.held_out_residual.total_cmp(&b.1.held_out_residual))
        .map(|(i, _)| i);
    let both_fail_a6 = winner.is_none();
    if let Some(w) = winner {
        let mut policy = Policy::new(0.8);
        policy.trust_transform(cells[w].content_hash.clone());
    }

    let cell_rows: Vec<serde_json::Value> = cells
        .iter()
        .zip(&a6_pass)
        .map(|(c, pass)| {
            serde_json::json!({
                "sender_layer": c.sender_layer,
                "receiver_layer": c.receiver_layer,
                "held_out_relative_residual": c.held_out_residual,
                "on_train_confidence_OPTIMISTIC_not_the_quality_number": c.on_train_confidence,
                "transform_content_hash": c.content_hash,
                "fit_seconds": c.fit_seconds,
                "gate_A6": {"threshold": A6_THRESHOLD, "pass": pass},
            })
        })
        .collect();

    let receipt = serde_json::json!({
        "stage": "S2c-generated-calibration-fit",
        "design": "docs/adr/023 Deviation 7 (generated-pairs contingency); docs/research/024 sections 5.4, 8 risk 6, 6 (A6)",
        "evidence_label": "deterministic CPU fit over live-model GENERATED-pairs dumps (dump receipt referenced by manifest sha256 below); the fit itself involves no model and no simulation",
        "dump": {"dir": dump_dir.display().to_string(), "manifest_sha256": manifest_sha,
                  "n_rows": manifest.n_rows},
        "fit": {
            "method": "AlignmentTransform::fit_affine (affine mean-centered semi-orthogonal Procrustes, dense rectangular SVD path) — identical to S2",
            "split_seed_chacha8": FIT_SPLIT_SEED,
            "split": "seeded shuffle of row indices; first 80% fit, last 20% held out; SAME split for both cells — identical machinery to S2",
            "n_fit_rows": n_fit,
            "n_holdout_rows": holdout_idx.len(),
            "n_over_d_gate": {"fit_rows": n_fit, "d_sender": d_sender, "pass": n_over_d,
                "recorded_not_fatal": "n is budget-laddered by the dump (see its receipt); the S2 abort exists where n was freely controllable",
                "polar_uniqueness_floor": {"d_receiver": d_receiver, "pass": polar_unique,
                    "note": "below n_fit = d_receiver the Procrustes polar factor is non-unique (rank(A'B) < d_receiver => zero singular values => arbitrary completion); ADR-002's MtM-fallback equivalence holds only on full-rank problems"}},
            "held_out_residual_formula": HELD_OUT_RESIDUAL_FORMULA,
        },
        "cells_registered_only": "L18->L14 (S2 winner) and L24->L19 (Deviation 6 anchor); no 3x3 sweep re-opening",
        "cells": cell_rows,
        "artifacts": artifacts,
        "winner_rule": "min held-out residual on generated pairs among A6-passing cells; probe order = winner first, other cell once on failure (ADR-023 Deviation 6 fallback mirror); A6-failing cells are not probed; both failing => CONTINGENCY-FAILED without a probe",
        "winner": winner.map(|w| serde_json::json!({
            "sender_layer": cells[w].sender_layer,
            "receiver_layer": cells[w].receiver_layer,
            "held_out_relative_residual": cells[w].held_out_residual,
            "transform_content_hash": cells[w].content_hash,
            "policy_trust_transform_registered": true,
        })),
        "both_cells_fail_A6": both_fail_a6,
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
    fn registered_cells_are_the_adr023_pair() {
        assert_eq!(REGISTERED_CELLS, [(18, 14), (24, 19)]);
    }

    #[test]
    fn golden_generator_is_deterministic() {
        let mut a = ChaCha8Rng::seed_from_u64(GOLDEN_INPUT_SEED);
        let mut b = ChaCha8Rng::seed_from_u64(GOLDEN_INPUT_SEED);
        assert_eq!(gaussian_vec(&mut a, 16), gaussian_vec(&mut b, 16));
    }
}
