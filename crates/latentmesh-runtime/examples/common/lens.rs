//! The logit-lens metric kit built for `docs/research/033`
//! (`run2_rescale_diagnostic`) §4 — the LAP `A_lin` output-alignment check
//! (`docs/research/032` §4/§5.1, arXiv:2604.15557).
//!
//! Extracted from `run2_rescale_diagnostic.rs` so that every later
//! re-projection diagnostic (`run2_manifold_precheck`, the M4f pre-check)
//! measures with the SAME code rather than a re-implementation. Pure
//! functions over f32 logit vectors; no model, no device, no I/O.
//!
//! Two lenses are used throughout:
//!   * `plain`   — `W_U · h`, the bare logit lens;
//!   * `rmsnorm` — `W_U · RMSNorm(h)`, the receiver's ACTUAL readout, whose
//!     leading RMSNorm is scale-invariant.

use candle_core::{Device, Tensor};
use std::collections::BTreeSet;

/// Indices of the `k` largest entries, descending by value.
pub fn top_k(z: &[f32], k: usize) -> Vec<u32> {
    let k = k.min(z.len());
    let mut idx: Vec<u32> = (0..z.len() as u32).collect();
    idx.select_nth_unstable_by(k.saturating_sub(1), |&a, &b| {
        z[b as usize]
            .partial_cmp(&z[a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx.truncate(k);
    idx.sort_unstable_by(|&a, &b| {
        z[b as usize]
            .partial_cmp(&z[a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx
}

/// Fraction of `a`'s top-k that also appears in `b`'s top-k.
pub fn overlap(a: &[u32], b: &[u32]) -> f64 {
    if a.is_empty() {
        return 1.0;
    }
    let set: BTreeSet<u32> = b.iter().copied().collect();
    a.iter().filter(|t| set.contains(t)).count() as f64 / a.len() as f64
}

/// log-sum-exp, numerically stable (f64 accumulation).
pub fn logsumexp(z: &[f32]) -> f64 {
    let m = z.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    m + z.iter().map(|&x| (x as f64 - m).exp()).sum::<f64>().ln()
}

/// Shannon entropy (nats) of `softmax(z)`.
pub fn entropy_nats(z: &[f32]) -> f64 {
    let lse = logsumexp(z);
    -z.iter()
        .map(|&x| {
            let lp = x as f64 - lse;
            lp.exp() * lp
        })
        .sum::<f64>()
}

/// 1-based rank of `t` by descending logit (ties broken pessimistically:
/// strictly-greater entries only, so equal logits share the best rank).
pub fn rank_of(z: &[f32], t: usize) -> usize {
    z.iter().filter(|&&x| x > z[t]).count() + 1
}

pub fn logprob_of(z: &[f32], t: usize, lse: f64) -> f64 {
    z[t] as f64 - lse
}

pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        d += x as f64 * y as f64;
        na += x as f64 * x as f64;
        nb += y as f64 * y as f64;
    }
    d / (na.sqrt() * nb.sqrt()).max(1e-300)
}

/// Qwen2 RMSNorm: `x / sqrt(mean(x^2) + eps) * gain`.
pub fn rms_norm(x: &[f32], gain: &[f32], eps: f64) -> Vec<f32> {
    let ms = x.iter().map(|&v| v as f64 * v as f64).sum::<f64>() / x.len() as f64;
    let inv = 1.0 / (ms + eps).sqrt();
    x.iter()
        .zip(gain)
        .map(|(&v, &g)| ((v as f64 * inv) as f32) * g)
        .collect()
}

/// Per-token-set statistics against one logit vector.
#[derive(Debug, Clone, Copy)]
pub struct TokenSetStats {
    pub mean_rank: f64,
    pub best_rank: usize,
    pub mean_logprob: f64,
}

pub fn token_set_stats(z: &[f32], toks: &[u32], lse: f64) -> TokenSetStats {
    let n = toks.len().max(1) as f64;
    let mut sum_rank = 0f64;
    let mut best = usize::MAX;
    let mut sum_lp = 0f64;
    for &t in toks {
        let r = rank_of(z, t as usize);
        sum_rank += r as f64;
        best = best.min(r);
        sum_lp += logprob_of(z, t as usize, lse);
    }
    TokenSetStats {
        mean_rank: sum_rank / n,
        best_rank: if best == usize::MAX { 0 } else { best },
        mean_logprob: sum_lp / n,
    }
}

/// `logits = unembed · h` on CPU (f32), one hidden vector.
pub fn project(unembed: &Tensor, h: &[f32], device: &Device) -> anyhow::Result<Vec<f32>> {
    let v = Tensor::from_slice(h, (h.len(), 1), device)?;
    Ok(unembed.matmul(&v)?.squeeze(1)?.to_vec1::<f32>()?)
}

/// `logits = unembed · [h_0 … h_{n-1}]` for a batch of hidden vectors,
/// returned row-per-vector. One matmul instead of `n`.
pub fn project_batch(
    unembed: &Tensor,
    hs: &[Vec<f32>],
    device: &Device,
) -> anyhow::Result<Vec<Vec<f32>>> {
    anyhow::ensure!(!hs.is_empty(), "project_batch needs at least one vector");
    let d = hs[0].len();
    let mut flat = Vec::with_capacity(d * hs.len());
    // Column-major fill: element (i, c) of a [d x n] tensor.
    for i in 0..d {
        for h in hs {
            anyhow::ensure!(h.len() == d, "ragged batch");
            flat.push(h[i]);
        }
    }
    let m = Tensor::from_vec(flat, (d, hs.len()), device)?;
    let logits = unembed.matmul(&m)?.t()?.contiguous()?;
    Ok(logits.to_vec2::<f32>()?)
}

/// Mean of a slice (0.0 when empty).
pub fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
}

/// `(min, max)` of a slice.
pub fn minmax(xs: &[f64]) -> (f64, f64) {
    (
        xs.iter().copied().fold(f64::INFINITY, f64::min),
        xs.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    )
}

/// Mean cosine over all unordered pairs `i < j` — the item-invariance
/// measure. 1.0 means every vector points the same way (total collapse).
pub fn mean_pairwise_cosine(vs: &[Vec<f32>]) -> (f64, f64, f64) {
    let mut acc = Vec::new();
    for i in 0..vs.len() {
        for j in (i + 1)..vs.len() {
            acc.push(cosine(&vs[i], &vs[j]));
        }
    }
    let (lo, hi) = minmax(&acc);
    (mean(&acc), lo, hi)
}

/// Length of the mean of the L2-normalised vectors ("mean resultant
/// length"): 1.0 under total directional collapse, ~0 when directions are
/// spread. Reported alongside `mean_pairwise_cosine` because it is a single
/// number rather than a pair average.
pub fn mean_resultant_length(vs: &[Vec<f32>]) -> f64 {
    if vs.is_empty() {
        return 0.0;
    }
    let d = vs[0].len();
    let mut acc = vec![0f64; d];
    for v in vs {
        let n = (v.iter().map(|&x| x as f64 * x as f64).sum::<f64>())
            .sqrt()
            .max(1e-300);
        for (a, &x) in acc.iter_mut().zip(v) {
            *a += x as f64 / n;
        }
    }
    (acc.iter().map(|a| a * a).sum::<f64>()).sqrt() / vs.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_k_is_descending_and_correct() {
        let z = [0.1f32, 5.0, -2.0, 3.0, 4.0];
        assert_eq!(top_k(&z, 3), vec![1, 4, 3]);
        assert_eq!(top_k(&z, 10).len(), 5);
        assert!((overlap(&top_k(&z, 3), &top_k(&z, 3)) - 1.0).abs() < 1e-12);
        assert!((overlap(&[1, 2, 3], &[3, 4, 5]) - 1.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn positive_scaling_preserves_every_rank_metric() {
        // `docs/research/033` in miniature: z -> c*z leaves ordering, top-k
        // and cosine invariant; only entropy (temperature) moves.
        let z: Vec<f32> = (0..64)
            .map(|i| ((i * 37 % 61) as f32) / 7.0 - 4.0)
            .collect();
        for c in [0.25f32, 0.5, 2.0, 9.0] {
            let s: Vec<f32> = z.iter().map(|x| x * c).collect();
            assert_eq!(top_k(&z, 16), top_k(&s, 16));
            for t in [0usize, 5, 33, 63] {
                assert_eq!(rank_of(&z, t), rank_of(&s, t));
            }
            assert!((cosine(&z, &s) - 1.0).abs() < 1e-12, "c={c}");
        }
        assert!(entropy_nats(&z.iter().map(|x| x * 9.0).collect::<Vec<_>>()) < entropy_nats(&z));
    }

    #[test]
    fn rms_norm_is_scale_invariant() {
        let gain = vec![1.0f32; 32];
        let x: Vec<f32> = (0..32).map(|i| (i as f32) - 15.5).collect();
        let a = rms_norm(&x, &gain, 1e-6);
        for c in [0.1f32, 3.0, 40.0] {
            let b = rms_norm(&x.iter().map(|v| v * c).collect::<Vec<_>>(), &gain, 1e-6);
            for (p, q) in a.iter().zip(&b) {
                assert!((p - q).abs() < 2e-4, "c={c}: {p} vs {q}");
            }
        }
    }

    #[test]
    fn logsumexp_and_entropy_are_sane() {
        let z = [0.0f32, 0.0, 0.0, 0.0];
        assert!((logsumexp(&z) - 4f64.ln()).abs() < 1e-12);
        assert!((entropy_nats(&z) - 4f64.ln()).abs() < 1e-12);
        assert_eq!(rank_of(&[1.0f32, 3.0, 2.0], 1), 1);
        assert_eq!(rank_of(&[1.0f32, 3.0, 2.0], 0), 3);
    }

    #[test]
    fn invariance_measures_bracket_the_two_extremes() {
        // Identical directions (different magnitudes) => total collapse.
        let same: Vec<Vec<f32>> = (1..=5)
            .map(|k| vec![1.0 * k as f32, -2.0 * k as f32, 0.5 * k as f32])
            .collect();
        let (m, lo, hi) = mean_pairwise_cosine(&same);
        assert!((m - 1.0).abs() < 1e-9 && (lo - 1.0).abs() < 1e-9 && (hi - 1.0).abs() < 1e-9);
        assert!((mean_resultant_length(&same) - 1.0).abs() < 1e-9);
        // Orthonormal basis => pairwise cosine 0, resultant 1/sqrt(n).
        let orth: Vec<Vec<f32>> = (0..3)
            .map(|i| (0..3).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
            .collect();
        assert!(mean_pairwise_cosine(&orth).0.abs() < 1e-9);
        assert!((mean_resultant_length(&orth) - (1.0 / 3f64).sqrt()).abs() < 1e-9);
    }

    #[test]
    fn project_batch_matches_per_vector_project() {
        let dev = Device::Cpu;
        let w = Tensor::from_vec(
            (0..(7 * 4))
                .map(|i| (i as f32) / 3.0 - 2.0)
                .collect::<Vec<f32>>(),
            (7, 4),
            &dev,
        )
        .unwrap();
        let hs: Vec<Vec<f32>> = (0..3)
            .map(|k| (0..4).map(|j| ((k * 4 + j) as f32) / 5.0 - 1.0).collect())
            .collect();
        let batch = project_batch(&w, &hs, &dev).unwrap();
        for (i, h) in hs.iter().enumerate() {
            let single = project(&w, h, &dev).unwrap();
            for (a, b) in batch[i].iter().zip(&single) {
                assert!((a - b).abs() < 1e-5, "{a} vs {b}");
            }
        }
    }
}
