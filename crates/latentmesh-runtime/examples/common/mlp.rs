//! Hand-rolled M3 MLP apply for the run-2 probes (ADR-024).
//!
//! latentmesh-train cannot be a path dependency of these examples (the
//! dependency runs the other way: train → runtime), so the trained
//! projector's forward is HAND-ROLLED here in plain Rust and verified at
//! startup against golden input/output pairs produced by the trained
//! network itself (candle CPU forward at training time) — the exact S2b
//! golden-pair discipline, per-vector relative L2 error ≤ 1e-5, asserted
//! before any model runs.
//!
//! Artifact layout (see latentmesh-train `mlp.rs` / the training receipt):
//! little-endian f32 `w1[2048×512 row-major, x·W] ‖ b1[512] ‖
//! w2[512×1536 row-major] ‖ b2[1536]`.

use super::sha256_hex;
use std::path::Path;

pub const D_IN: usize = 2048;
pub const D_HID: usize = 512;
pub const D_OUT: usize = 1536;
pub const PARAM_COUNT: usize = D_IN * D_HID + D_HID + D_HID * D_OUT + D_OUT;

/// The trained 2-layer MLP projector, loaded from the raw-f32 artifact.
pub struct MlpTransform {
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
    /// sha256 of the artifact file bytes (the content hash).
    pub content_hash: String,
}

impl MlpTransform {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let bytes = std::fs::read(path)?;
        anyhow::ensure!(
            bytes.len() == PARAM_COUNT * 4,
            "MLP artifact {}: {} bytes != {} params x 4",
            path.display(),
            bytes.len(),
            PARAM_COUNT
        );
        let content_hash = sha256_hex(&bytes);
        let flat: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let (w1, rest) = flat.split_at(D_IN * D_HID);
        let (b1, rest) = rest.split_at(D_HID);
        let (w2, b2) = rest.split_at(D_HID * D_OUT);
        Ok(Self {
            w1: w1.to_vec(),
            b1: b1.to_vec(),
            w2: w2.to_vec(),
            b2: b2.to_vec(),
            content_hash,
        })
    }

    /// `relu(x·W1 + b1)·W2 + b2` — hand-rolled, f32, cache-friendly row-major.
    pub fn apply(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), D_IN, "MLP input dimension mismatch");
        let mut h = self.b1.clone();
        for (i, &xi) in x.iter().enumerate() {
            if xi == 0.0 {
                continue;
            }
            let row = &self.w1[i * D_HID..(i + 1) * D_HID];
            for (hj, w) in h.iter_mut().zip(row) {
                *hj += xi * w;
            }
        }
        for hj in h.iter_mut() {
            *hj = hj.max(0.0);
        }
        let mut y = self.b2.clone();
        for (i, &hi) in h.iter().enumerate() {
            if hi == 0.0 {
                continue;
            }
            let row = &self.w2[i * D_OUT..(i + 1) * D_OUT];
            for (yj, w) in y.iter_mut().zip(row) {
                *yj += hi * w;
            }
        }
        y
    }

    /// Apply per token over row-major `[n_rows × D_IN]` states, then
    /// mean-pool the TRANSLATED rows (f64 accumulation) — ADR-024 M3
    /// variant (i): pooling happens after translation.
    pub fn apply_rows_then_pool(&self, rows: &[f32], n_rows: usize) -> Vec<f32> {
        assert_eq!(rows.len(), n_rows * D_IN, "rows buffer shape mismatch");
        assert!(n_rows > 0, "cannot pool zero translated rows");
        let mut acc = vec![0f64; D_OUT];
        for r in 0..n_rows {
            let y = self.apply(&rows[r * D_IN..(r + 1) * D_IN]);
            for (a, v) in acc.iter_mut().zip(&y) {
                *a += *v as f64;
            }
        }
        acc.iter().map(|a| (*a / n_rows as f64) as f32).collect()
    }
}

#[derive(serde::Deserialize)]
pub struct MlpGoldenFile {
    pub artifact_file_sha256: String,
    pub input_seed_chacha8: u64,
    pub n_pairs: usize,
    pub inputs: Vec<Vec<f32>>,
    pub outputs: Vec<Vec<f32>>,
}

/// Verify the hand-rolled apply against the trained network's own golden
/// pairs. Returns `(n_pairs, max relative L2 error, input seed)`.
pub fn verify_against_golden(
    t: &MlpTransform,
    golden_path: &Path,
    rel_tol: f32,
) -> anyhow::Result<(usize, f32, u64)> {
    let g: MlpGoldenFile = serde_json::from_slice(&std::fs::read(golden_path)?)?;
    anyhow::ensure!(
        g.artifact_file_sha256 == t.content_hash,
        "golden file was produced for artifact {} but this run loads {}",
        g.artifact_file_sha256,
        t.content_hash
    );
    anyhow::ensure!(
        g.n_pairs >= 8 && g.inputs.len() == g.n_pairs && g.outputs.len() == g.n_pairs,
        "golden file must carry >=8 input/output pairs"
    );
    let mut max_rel = 0f32;
    for (x, y_gold) in g.inputs.iter().zip(&g.outputs) {
        let y = t.apply(x);
        let diff: f32 = y
            .iter()
            .zip(y_gold)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            .sqrt();
        let norm: f32 = y_gold.iter().map(|v| v * v).sum::<f32>().sqrt();
        let rel = diff / norm.max(1e-12);
        max_rel = max_rel.max(rel);
        anyhow::ensure!(
            rel <= rel_tol,
            "hand-rolled MLP apply diverges from the trained network: relative L2 error {rel} > {rel_tol}"
        );
    }
    Ok((g.n_pairs, max_rel, g.input_seed_chacha8))
}
