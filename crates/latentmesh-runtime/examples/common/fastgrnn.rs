//! Hand-rolled M4 FastGRNN sequence apply for the run-2 probes (ADR-024).
//!
//! latentmesh-train cannot be a path dependency of these examples (the
//! dependency runs the other way: train → runtime), so the trained sequence
//! translator's forward is HAND-ROLLED here in plain Rust and verified at
//! startup against golden SEQUENCE pairs produced by the trained network
//! itself (candle CPU forward at training time) — the S2b/M3 golden
//! discipline extended to sequences: every per-step output AND the
//! mean-pooled injection payload must match to relative L2 error ≤ 1e-5,
//! asserted before any model runs.
//!
//! Cell (Kusupati et al., arXiv:1901.02358; equations per the M0 scout's
//! primary-graded fetch):
//! `pre = (x@W1)@W2 + (h@U1)@U2`; `z = sigmoid(pre + b_z)`;
//! `h~ = tanh(pre + b_h)`; `h' = z⊙h + (sigmoid(zeta)(1−z) + sigmoid(nu))⊙h~`.
//!
//! Artifact layout (see latentmesh-train `fastgrnn.rs` / the M4 training
//! receipt): little-endian f32 `w1[2048×R] ‖ w2[R×1536] ‖ u1[1536×R] ‖
//! u2[R×1536] ‖ b_z[1536] ‖ b_h[1536] ‖ zeta_raw ‖ nu_raw`, row-major, x·W.

use super::sha256_hex;
use std::path::Path;

pub const D_IN: usize = 2048;
pub const D_H: usize = 1536;

/// Frozen parameter-count formula: `r(D_in + D_h) + 2·r·D_h + 2·D_h + 2`.
pub const fn param_count(rank: usize) -> usize {
    rank * (D_IN + D_H) + 2 * rank * D_H + 2 * D_H + 2
}

/// The trained low-rank FastGRNN cell, loaded from the raw-f32 artifact
/// (rank inferred from the byte count via the parameter formula).
pub struct FastGrnnTransform {
    pub rank: usize,
    w1: Vec<f32>,
    w2: Vec<f32>,
    u1: Vec<f32>,
    u2: Vec<f32>,
    b_z: Vec<f32>,
    b_h: Vec<f32>,
    /// sigmoid(zeta_raw), sigmoid(nu_raw) — precomputed.
    sz: f32,
    sn: f32,
    /// sha256 of the artifact file bytes (the content hash).
    pub content_hash: String,
}

fn sig(v: f32) -> f32 {
    1.0 / (1.0 + (-v).exp())
}

impl FastGrnnTransform {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let bytes = std::fs::read(path)?;
        anyhow::ensure!(bytes.len() % 4 == 0, "artifact not f32-aligned");
        let n = bytes.len() / 4;
        anyhow::ensure!(
            n >= 2 * D_H + 2 && (n - 2 * D_H - 2) % (D_IN + 3 * D_H) == 0,
            "FastGRNN artifact {}: {} floats does not match the param formula",
            path.display(),
            n
        );
        let rank = (n - 2 * D_H - 2) / (D_IN + 3 * D_H);
        anyhow::ensure!(param_count(rank) == n);
        let content_hash = sha256_hex(&bytes);
        let flat: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let (w1, rest) = flat.split_at(D_IN * rank);
        let (w2, rest) = rest.split_at(rank * D_H);
        let (u1, rest) = rest.split_at(D_H * rank);
        let (u2, rest) = rest.split_at(rank * D_H);
        let (b_z, rest) = rest.split_at(D_H);
        let (b_h, rest) = rest.split_at(D_H);
        Ok(Self {
            rank,
            w1: w1.to_vec(),
            w2: w2.to_vec(),
            u1: u1.to_vec(),
            u2: u2.to_vec(),
            b_z: b_z.to_vec(),
            b_h: b_h.to_vec(),
            sz: sig(rest[0]),
            sn: sig(rest[1]),
            content_hash,
        })
    }

    /// One cell step in place: `x` (len 2048) advances `h` (len 1536).
    pub fn step(&self, x: &[f32], h: &mut [f32]) {
        assert_eq!(x.len(), D_IN, "FastGRNN input dimension mismatch");
        assert_eq!(h.len(), D_H, "FastGRNN hidden dimension mismatch");
        let r = self.rank;
        // pre = (x@W1)@W2 + (h@U1)@U2
        let mut xr = vec![0f32; r];
        for (i, &xi) in x.iter().enumerate() {
            if xi == 0.0 {
                continue;
            }
            for (o, w) in xr.iter_mut().zip(&self.w1[i * r..(i + 1) * r]) {
                *o += xi * w;
            }
        }
        let mut hr = vec![0f32; r];
        for (i, &hi) in h.iter().enumerate() {
            if hi == 0.0 {
                continue;
            }
            for (o, w) in hr.iter_mut().zip(&self.u1[i * r..(i + 1) * r]) {
                *o += hi * w;
            }
        }
        let mut pre = vec![0f32; D_H];
        for (i, &v) in xr.iter().enumerate() {
            for (o, w) in pre.iter_mut().zip(&self.w2[i * D_H..(i + 1) * D_H]) {
                *o += v * w;
            }
        }
        for (i, &v) in hr.iter().enumerate() {
            for (o, w) in pre.iter_mut().zip(&self.u2[i * D_H..(i + 1) * D_H]) {
                *o += v * w;
            }
        }
        for j in 0..D_H {
            let z = sig(pre[j] + self.b_z[j]);
            let ht = (pre[j] + self.b_h[j]).tanh();
            h[j] = z * h[j] + (self.sz * (1.0 - z) + self.sn) * ht;
        }
    }

    /// The M4 payload derivation (frozen in the training receipt's eval
    /// plan): run the full sequence from `h_0 = 0` over row-major
    /// `[n_rows × D_IN]` states, mean-pool the TRANSLATED output sequence
    /// (f64 accumulation) — sequence processing upstream of the pool.
    pub fn translate_seq_then_pool(&self, rows: &[f32], n_rows: usize) -> Vec<f32> {
        assert_eq!(rows.len(), n_rows * D_IN, "rows buffer shape mismatch");
        assert!(n_rows > 0, "cannot pool zero translated steps");
        let mut h = vec![0f32; D_H];
        let mut acc = vec![0f64; D_H];
        for t in 0..n_rows {
            self.step(&rows[t * D_IN..(t + 1) * D_IN], &mut h);
            for (a, v) in acc.iter_mut().zip(&h) {
                *a += *v as f64;
            }
        }
        acc.iter().map(|a| (*a / n_rows as f64) as f32).collect()
    }
}

#[derive(serde::Deserialize)]
pub struct FastGrnnGoldenFile {
    pub artifact_file_sha256: String,
    pub input_seed_chacha8: u64,
    pub n_seqs: usize,
    pub seq_len: usize,
    pub inputs: Vec<Vec<Vec<f32>>>,
    pub outputs: Vec<Vec<Vec<f32>>>,
    pub pooled: Vec<Vec<f32>>,
}

/// Verify the hand-rolled sequential apply against the trained network's own
/// golden sequences: every per-step output AND every pooled payload.
/// Returns `(n_seqs, seq_len, max relative L2 error, input seed)`.
pub fn verify_against_golden(
    t: &FastGrnnTransform,
    golden_path: &Path,
    rel_tol: f32,
) -> anyhow::Result<(usize, usize, f32, u64)> {
    let g: FastGrnnGoldenFile = serde_json::from_slice(&std::fs::read(golden_path)?)?;
    anyhow::ensure!(
        g.artifact_file_sha256 == t.content_hash,
        "golden file was produced for artifact {} but this run loads {}",
        g.artifact_file_sha256,
        t.content_hash
    );
    anyhow::ensure!(
        g.n_seqs >= 4
            && g.seq_len >= 4
            && g.inputs.len() == g.n_seqs
            && g.outputs.len() == g.n_seqs
            && g.pooled.len() == g.n_seqs,
        "golden file must carry >=4 sequences of >=4 steps with pooled payloads"
    );
    let rel_l2 = |a: &[f32], b: &[f32]| -> f32 {
        let diff: f32 = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
            .sqrt();
        let norm: f32 = b.iter().map(|v| v * v).sum::<f32>().sqrt();
        diff / norm.max(1e-12)
    };
    let mut max_rel = 0f32;
    for (s, (xs, ys_gold)) in g.inputs.iter().zip(&g.outputs).enumerate() {
        anyhow::ensure!(xs.len() == g.seq_len && ys_gold.len() == g.seq_len);
        let mut h = vec![0f32; D_H];
        let mut acc = vec![0f64; D_H];
        for (step, (x, y_gold)) in xs.iter().zip(ys_gold).enumerate() {
            t.step(x, &mut h);
            for (a, v) in acc.iter_mut().zip(&h) {
                *a += *v as f64;
            }
            let rel = rel_l2(&h, y_gold);
            max_rel = max_rel.max(rel);
            anyhow::ensure!(
                rel <= rel_tol,
                "hand-rolled FastGRNN diverges from the trained network at seq {s} step {step}: relative L2 error {rel} > {rel_tol}"
            );
        }
        let pool: Vec<f32> = acc.iter().map(|a| (*a / g.seq_len as f64) as f32).collect();
        let rel = rel_l2(&pool, &g.pooled[s]);
        max_rel = max_rel.max(rel);
        anyhow::ensure!(
            rel <= rel_tol,
            "hand-rolled FastGRNN POOLED payload diverges at seq {s}: relative L2 error {rel} > {rel_tol}"
        );
    }
    Ok((g.n_seqs, g.seq_len, max_rel, g.input_seed_chacha8))
}
