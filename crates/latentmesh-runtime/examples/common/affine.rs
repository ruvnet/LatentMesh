//! Hand-rolled mirror of `latentmesh_align::AlignmentTransform`'s serialized
//! shape and affine apply `y = μ_r + α·(z − μ_s)·R`.
//!
//! latentmesh-align cannot be a path dependency of these examples (align →
//! latentmesh-core pins `half =2.4.1` vs candle's `half ^2.5`), so the apply
//! is hand-rolled here and verified at startup against golden input/output
//! pairs produced by the align crate's own apply — the same golden-pair
//! discipline `mlp.rs` / `fastgrnn.rs` use for the trained adapters.
//!
//! Extracted from `s2b_bridge_probe.rs` (run-1, S2b) so that the run-1
//! transform's apply exists in exactly one place and cannot silently diverge
//! between the frozen probe and later diagnostics that re-project the same
//! transforms.

use latentmesh_runtime::norms;
use std::path::Path;

/// The run-1 closed-form affine bridge, deserialized from the committed
/// `transform-*.json` (the align crate's own serialization).
#[derive(serde::Deserialize)]
pub struct AffineTransform {
    pub r: Vec<Vec<f32>>,
    pub alpha: f32,
    pub dim_sender: usize,
    pub dim_receiver: usize,
    #[serde(default)]
    pub mu_s: Vec<f32>,
    #[serde(default)]
    pub mu_r: Vec<f32>,
    pub confidence: f32,
}

impl AffineTransform {
    /// `y = μ_r + α·(z − μ_s)·R`.
    pub fn apply(&self, z: &[f32]) -> Vec<f32> {
        assert_eq!(z.len(), self.dim_sender, "input dimension mismatch");
        let mut out = vec![0f32; self.dim_receiver];
        for (i, row) in self.r.iter().enumerate() {
            let c = if self.mu_s.is_empty() {
                z[i]
            } else {
                z[i] - self.mu_s[i]
            };
            for (o, rij) in out.iter_mut().zip(row) {
                *o += c * rij;
            }
        }
        for (j, o) in out.iter_mut().enumerate() {
            *o *= self.alpha;
            if !self.mu_r.is_empty() {
                *o += self.mu_r[j];
            }
        }
        out
    }
}

#[derive(serde::Deserialize)]
pub struct GoldenFile {
    pub transform_file_sha256: String,
    pub input_seed_chacha8: u64,
    pub n_pairs: usize,
    pub inputs: Vec<Vec<f32>>,
    pub outputs: Vec<Vec<f32>>,
}

/// Verify the hand-rolled apply against the align-crate golden pairs.
/// Returns `(n_pairs, max relative L2 error, golden input seed)`.
pub fn verify_against_golden(
    t: &AffineTransform,
    golden_path: &Path,
    transform_file_sha: &str,
    rel_tol: f32,
) -> anyhow::Result<(usize, f32, u64)> {
    let g: GoldenFile = serde_json::from_slice(&std::fs::read(golden_path)?)?;
    anyhow::ensure!(
        g.transform_file_sha256 == transform_file_sha,
        "golden file was produced for transform {} but this run loads {transform_file_sha}",
        g.transform_file_sha256
    );
    anyhow::ensure!(
        g.n_pairs >= 8 && g.inputs.len() == g.n_pairs && g.outputs.len() == g.n_pairs,
        "golden file must carry >=8 input/output pairs"
    );
    let mut max_rel = 0f32;
    for (x, y_gold) in g.inputs.iter().zip(&g.outputs) {
        let y = t.apply(x);
        let diff_l2 = norms::l2(
            &y.iter()
                .zip(y_gold)
                .map(|(a, b)| a - b)
                .collect::<Vec<f32>>(),
        );
        let rel = diff_l2 / norms::l2(y_gold).max(1e-12);
        max_rel = max_rel.max(rel);
        anyhow::ensure!(
            rel <= rel_tol,
            "hand-rolled apply diverges from latentmesh-align apply: relative L2 error {rel} > {rel_tol}"
        );
    }
    Ok((g.n_pairs, max_rel, g.input_seed_chacha8))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The affine map commutes exactly with mean-pooling:
    /// `mean_i apply(x_i) == apply(mean_i x_i)`. This is why re-projecting
    /// the run-1 transforms per token and pooling afterwards is the SAME
    /// object the run-1 probe injected (which pooled first) — no
    /// pool-order confound exists for the linear baseline, unlike the
    /// ReLU MLP.
    #[test]
    fn affine_commutes_with_mean_pooling() {
        let t = AffineTransform {
            r: (0..4)
                .map(|i| (0..3).map(|j| ((i * 3 + j) as f32) / 7.0 - 0.5).collect())
                .collect(),
            alpha: 0.63,
            dim_sender: 4,
            dim_receiver: 3,
            mu_s: vec![0.1, -0.2, 0.3, 0.4],
            mu_r: vec![1.0, -1.0, 0.5],
            confidence: 0.5,
        };
        let xs: Vec<Vec<f32>> = (0..5)
            .map(|k| (0..4).map(|j| ((k * 4 + j) as f32) / 3.0 - 2.0).collect())
            .collect();
        let mut pooled_in = vec![0f32; 4];
        let mut pooled_out = vec![0f32; 3];
        for x in &xs {
            let y = t.apply(x);
            for (a, v) in pooled_in.iter_mut().zip(x) {
                *a += v / xs.len() as f32;
            }
            for (a, v) in pooled_out.iter_mut().zip(&y) {
                *a += v / xs.len() as f32;
            }
        }
        for (a, b) in t.apply(&pooled_in).iter().zip(&pooled_out) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }
}
