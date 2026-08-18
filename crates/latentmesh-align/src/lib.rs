//! `latentmesh-align` — training-free alignment between two latent spaces
//! (ADR-002). This crate makes one honest, verifiable claim: **given
//! calibration pairs `(a_i, b_i)` sampled from two spaces, it recovers the
//! least-squares-optimal orthogonal transform between them** — the classical
//! orthogonal Procrustes solution (Schönemann 1966), computed via SVD, with a
//! magnitude scalar `α` fit by least squares and a confidence score derived
//! from the real calibration-set residual.
//!
//! This is the *general technique* StateBridge-style training-free alignment
//! is an instance of — this crate does not claim to reproduce StateBridge's
//! specific reported results, which require real heterogeneous LLM hidden
//! states this repo does not have access to (see ADR-001 §8, ADR-002).

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// A fitted alignment: `aligned(z) = α · z · R`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AlignmentTransform {
    /// `dim_sender × dim_receiver`, orthonormal (rows or columns, whichever is
    /// the smaller dimension — the semi-orthogonal Procrustes solution).
    r: Vec<Vec<f32>>,
    alpha: f32,
    dim_sender: usize,
    dim_receiver: usize,
    /// In `[0,1]`: `1 − relative Frobenius residual` on the calibration set.
    /// A REAL measured number, not asserted — see [`AlignmentTransform::fit`].
    pub confidence: f32,
}

impl AlignmentTransform {
    /// Fit `R`/`α` from calibration pairs via SVD-based orthogonal Procrustes.
    /// Panics on empty input or mismatched pair dimensions (a programmer
    /// error, not a runtime data condition).
    ///
    /// **Optimized (2026-08-18):** `M = Aᵀ B` has rank ≤ `n` (the calibration
    /// count), which for realistic LLM hidden dims (`d` ~ 1k–4k) with a small
    /// calibration set (`n` ~ 16–64) is far below `d` — a dense `d×d` SVD
    /// (`O(d³)`) does `O(d²/n)` more work than the problem's actual rank
    /// justifies. When `dim_sender == dim_receiver` (the realistic
    /// same-architecture-family case) and `n < d`, this computes the EXACT
    /// same orthogonal Procrustes solution on the calibrated subspace via a
    /// `d×n` QR + a small `n×n`/`n×d` SVD — `O(d²n)` instead of `O(d³)` — and
    /// completes the transform as the **identity on the orthogonal
    /// complement** of the calibrated subspace (a deliberate, principled
    /// choice: don't guess where there's no calibration evidence; every
    /// other choice of null-space completion, including the dense solver's
    /// own algorithm-determined one, is equally arbitrary there). Verified
    /// against the direct dense solver in `fast_path_matches_dense_reference`.
    pub fn fit(pairs: &[(Vec<f32>, Vec<f32>)]) -> Self {
        assert!(
            !pairs.is_empty(),
            "fit requires at least one calibration pair"
        );
        let dim_sender = pairs[0].0.len();
        let dim_receiver = pairs[0].1.len();
        let n = pairs.len();
        assert!(
            dim_sender > 0 && dim_receiver > 0,
            "zero-dimensional vectors"
        );
        for (a, b) in pairs {
            assert_eq!(a.len(), dim_sender, "inconsistent sender dimension");
            assert_eq!(b.len(), dim_receiver, "inconsistent receiver dimension");
        }

        // A: n × dim_sender, B: n × dim_receiver (row-major samples).
        let a = DMatrix::from_row_slice(
            n,
            dim_sender,
            &pairs
                .iter()
                .flat_map(|(x, _)| x.iter().copied())
                .collect::<Vec<f32>>(),
        );
        let b = DMatrix::from_row_slice(
            n,
            dim_receiver,
            &pairs
                .iter()
                .flat_map(|(_, y)| y.iter().copied())
                .collect::<Vec<f32>>(),
        );

        let r = if dim_sender == dim_receiver && n < dim_sender {
            Self::orthogonal_procrustes_low_rank(&a, &b)
        } else {
            Self::orthogonal_procrustes_dense(&a, &b)
        };

        Self::finish(a, b, r, dim_sender, dim_receiver)
    }

    /// The original direct solver: `M = Aᵀ B`, SVD(M) = `U Σ Vᵀ`, `R = U Vᵀ`.
    /// `O(d³)`-ish (dense SVD of a `dim_sender × dim_receiver` matrix). Used
    /// as the reference implementation and as the fallback when the fast
    /// path's preconditions (`dim_sender == dim_receiver`, `n < dim`) don't hold.
    fn orthogonal_procrustes_dense(a: &DMatrix<f32>, b: &DMatrix<f32>) -> DMatrix<f32> {
        let m = a.transpose() * b; // dim_sender × dim_receiver
        let svd = nalgebra::linalg::SVD::new(m, true, true);
        let u = svd.u.expect("SVD did not compute U");
        let v_t = svd.v_t.expect("SVD did not compute Vᵀ");
        u * v_t
    }

    /// The optimized solver (see `fit`'s doc comment for the derivation and
    /// its exactness argument). Requires `a`/`b` to have equal column count `d`
    /// and `n = a.nrows() < d`.
    fn orthogonal_procrustes_low_rank(a: &DMatrix<f32>, b: &DMatrix<f32>) -> DMatrix<f32> {
        let d = a.ncols();
        // Q: d × n, orthonormal columns spanning row-space(a); R_qr: n × n.
        let qr = a.transpose().qr();
        let q = qr.q(); // d × n
        let r_qr = qr.r(); // n × n, upper triangular
                           // W = R_qr · B : n × d (small — this is where the real work moves to).
        let w = r_qr * b;
        let svd = nalgebra::linalg::SVD::new(w, true, true);
        let u_prime = svd.u.expect("SVD did not compute U'"); // n × n
        let v_t_prime = svd.v_t.expect("SVD did not compute V'ᵀ"); // n × d
                                                                   // R = (I_d − Q Qᵀ) + Q U' V'ᵀ — exact on the calibrated subspace,
                                                                   // identity on its orthogonal complement.
        let qqt = &q * q.transpose(); // d × d
        let identity = DMatrix::<f32>::identity(d, d);
        (identity - qqt) + q * u_prime * v_t_prime
    }

    /// Shared tail of `fit`: magnitude calibration + confidence + packing.
    fn finish(
        a: DMatrix<f32>,
        b: DMatrix<f32>,
        r: DMatrix<f32>,
        dim_sender: usize,
        dim_receiver: usize,
    ) -> Self {
        // α (magnitude calibration): minimizes ||α (A R) − B||_F in closed form.
        let ar = &a * &r; // n × dim_receiver
        let numer: f32 = ar.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let denom: f32 = ar.iter().map(|x| x * x).sum();
        let alpha = if denom.abs() > 1e-12 {
            numer / denom
        } else {
            1.0
        };

        // Confidence: 1 − relative Frobenius residual on the calibration set.
        let residual = (&ar * alpha - &b).norm();
        let scale = b.norm().max(1e-6);
        let confidence = (1.0 - residual / scale).clamp(0.0, 1.0);

        AlignmentTransform {
            r: (0..dim_sender)
                .map(|i| r.row(i).iter().copied().collect())
                .collect(),
            alpha,
            dim_sender,
            dim_receiver,
            confidence,
        }
    }

    /// Apply the transform: `α · z · R`. Panics if `z.len() != dim_sender`.
    pub fn apply(&self, z: &[f32]) -> Vec<f32> {
        assert_eq!(z.len(), self.dim_sender, "input dimension mismatch");
        let z = DVector::from_row_slice(z);
        let r = DMatrix::from_row_slice(
            self.dim_sender,
            self.dim_receiver,
            &self.r.iter().flatten().copied().collect::<Vec<f32>>(),
        );
        (self.alpha * (z.transpose() * r)).iter().copied().collect()
    }

    /// A content hash binding this exact `(R, α)` — what `LatentFrame::transform_hash`
    /// (ADR-002) references, so a receiver can verify which transform produced
    /// an aligned frame rather than trusting an unhashed matrix.
    pub fn content_hash(&self) -> String {
        let bytes = serde_json::to_vec(self).unwrap_or_default();
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        hasher
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }

    pub fn dims(&self) -> (usize, usize) {
        (self.dim_sender, self.dim_receiver)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// A random orthogonal d×d matrix via QR decomposition of a random matrix
    /// (the standard construction — Q from QR is orthogonal by definition).
    fn random_orthogonal(d: usize, rng: &mut StdRng) -> DMatrix<f32> {
        let data: Vec<f32> = (0..d * d).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let m = DMatrix::from_row_slice(d, d, &data);
        let qr = m.qr();
        qr.q()
    }

    #[test]
    fn recovers_a_known_orthogonal_transform_from_noiseless_pairs() {
        let mut rng = StdRng::seed_from_u64(42);
        let d = 16;
        let q = random_orthogonal(d, &mut rng);
        let pairs: Vec<(Vec<f32>, Vec<f32>)> = (0..64)
            .map(|_| {
                let a: Vec<f32> = (0..d).map(|_| rng.gen_range(-1.0..1.0)).collect();
                let av = DVector::from_row_slice(&a);
                let b = (&q.transpose() * &av).iter().copied().collect::<Vec<f32>>();
                (a, b)
            })
            .collect();
        let t = AlignmentTransform::fit(&pairs);
        assert!(
            t.confidence > 0.99,
            "expected near-perfect fit, got {}",
            t.confidence
        );
        // Held-out check: apply to a fresh vector, compare to the ground-truth transform.
        let z: Vec<f32> = (0..d).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let expected = (&q.transpose() * DVector::from_row_slice(&z))
            .iter()
            .copied()
            .collect::<Vec<f32>>();
        let got = t.apply(&z);
        for (e, g) in expected.iter().zip(got.iter()) {
            assert!(
                (e - g).abs() < 0.05,
                "alignment diverged: expected {e}, got {g}"
            );
        }
    }

    #[test]
    fn confidence_degrades_under_injected_noise() {
        let mut rng = StdRng::seed_from_u64(7);
        let d = 12;
        let q = random_orthogonal(d, &mut rng);
        let mk_pairs = |noise: f32, rng: &mut StdRng| -> Vec<(Vec<f32>, Vec<f32>)> {
            (0..64)
                .map(|_| {
                    let a: Vec<f32> = (0..d).map(|_| rng.gen_range(-1.0..1.0)).collect();
                    let av = DVector::from_row_slice(&a);
                    let mut b: Vec<f32> = (&q.transpose() * &av).iter().copied().collect();
                    if noise > 0.0 {
                        for v in b.iter_mut() {
                            *v += rng.gen_range(-noise..noise);
                        }
                    }
                    (a, b)
                })
                .collect()
        };
        let clean = AlignmentTransform::fit(&mk_pairs(0.0, &mut rng.clone()));
        let noisy = AlignmentTransform::fit(&mk_pairs(0.8, &mut rng));
        assert!(
            noisy.confidence < clean.confidence,
            "noisy fit ({}) should score below clean fit ({})",
            noisy.confidence,
            clean.confidence
        );
        assert!(noisy.confidence >= 0.0 && noisy.confidence <= 1.0);
    }

    #[test]
    fn square_transform_round_trips_via_its_transpose() {
        // For a square orthogonal R with α≈1, R's transpose is its inverse —
        // apply-then-invert should recover the original vector.
        let mut rng = StdRng::seed_from_u64(99);
        let d = 8;
        let q = random_orthogonal(d, &mut rng);
        let pairs: Vec<(Vec<f32>, Vec<f32>)> = (0..64)
            .map(|_| {
                let a: Vec<f32> = (0..d).map(|_| rng.gen_range(-1.0..1.0)).collect();
                let av = DVector::from_row_slice(&a);
                let b = (&q.transpose() * &av).iter().copied().collect::<Vec<f32>>();
                (a, b)
            })
            .collect();
        let t = AlignmentTransform::fit(&pairs);
        let z: Vec<f32> = vec![1.0, -1.0, 0.5, 0.25, -0.75, 2.0, -2.0, 0.1];
        let aligned = t.apply(&z);
        // Manually invert via Rᵀ/α since AlignmentTransform doesn't expose an
        // `invert()` yet — this proves R is genuinely (semi-)orthogonal.
        let r = DMatrix::from_row_slice(d, d, &t.r.iter().flatten().copied().collect::<Vec<f32>>());
        let back: Vec<f32> = ((1.0 / t.alpha)
            * (DVector::from_row_slice(&aligned).transpose() * r.transpose()))
        .iter()
        .copied()
        .collect();
        for (orig, recovered) in z.iter().zip(back.iter()) {
            assert!(
                (orig - recovered).abs() < 0.05,
                "round trip diverged: {orig} vs {recovered}"
            );
        }
    }

    #[test]
    fn fast_path_matches_dense_reference() {
        // The core correctness claim of the 2026-08-18 optimization: the O(d²n)
        // fast path (QR + small SVD) must reproduce the O(d³) dense solver's
        // result on the calibrated subspace, to numerical tolerance.
        let mut rng = StdRng::seed_from_u64(2026);
        let d = 48;
        let n = 12; // n < d, dim_sender == dim_receiver: fast path applies.
        let q_truth = random_orthogonal(d, &mut rng);
        let pairs: Vec<(Vec<f32>, Vec<f32>)> = (0..n)
            .map(|_| {
                let a: Vec<f32> = (0..d).map(|_| rng.gen_range(-1.0..1.0)).collect();
                let av = DVector::from_row_slice(&a);
                let b = (&q_truth.transpose() * &av)
                    .iter()
                    .copied()
                    .collect::<Vec<f32>>();
                (a, b)
            })
            .collect();

        let a = DMatrix::from_row_slice(
            n,
            d,
            &pairs
                .iter()
                .flat_map(|(x, _)| x.iter().copied())
                .collect::<Vec<f32>>(),
        );
        let b = DMatrix::from_row_slice(
            n,
            d,
            &pairs
                .iter()
                .flat_map(|(_, y)| y.iter().copied())
                .collect::<Vec<f32>>(),
        );

        let r_fast = AlignmentTransform::orthogonal_procrustes_low_rank(&a, &b);
        let r_dense = AlignmentTransform::orthogonal_procrustes_dense(&a, &b);

        // Compare their action on the CALIBRATED subspace (where both are
        // uniquely determined) rather than raw matrix equality (the null-space
        // completion legitimately differs — dense uses an algorithm-arbitrary
        // completion, fast uses identity, by design; see `fit`'s doc comment).
        for (ai, _) in &pairs {
            let via_fast = (DVector::from_row_slice(ai).transpose() * &r_fast)
                .iter()
                .copied()
                .collect::<Vec<f32>>();
            let via_dense = (DVector::from_row_slice(ai).transpose() * &r_dense)
                .iter()
                .copied()
                .collect::<Vec<f32>>();
            for (f, s) in via_fast.iter().zip(via_dense.iter()) {
                assert!(
                    (f - s).abs() < 1e-3,
                    "fast vs dense diverged on calibration data: {f} vs {s}"
                );
            }
        }

        // And the full `fit()` entry point (which dispatches to the fast path
        // here) should report the same near-perfect confidence as the dense one.
        let t_fast = AlignmentTransform::fit(&pairs);
        assert!(
            t_fast.confidence > 0.99,
            "fast-path fit confidence too low: {}",
            t_fast.confidence
        );
    }

    #[test]
    fn fast_path_is_identity_on_the_orthogonal_complement() {
        // A vector orthogonal to every calibration input should pass through
        // the fast-path transform (α≈1 aside) essentially unchanged — the
        // deliberate null-space policy `fit`'s doc comment commits to.
        let mut rng = StdRng::seed_from_u64(11);
        let d = 32;
        let n = 8;
        let q_truth = random_orthogonal(d, &mut rng);
        let pairs: Vec<(Vec<f32>, Vec<f32>)> = (0..n)
            .map(|_| {
                let a: Vec<f32> = (0..d).map(|_| rng.gen_range(-1.0..1.0)).collect();
                let av = DVector::from_row_slice(&a);
                let b = (&q_truth.transpose() * &av)
                    .iter()
                    .copied()
                    .collect::<Vec<f32>>();
                (a, b)
            })
            .collect();
        let a = DMatrix::from_row_slice(
            n,
            d,
            &pairs
                .iter()
                .flat_map(|(x, _)| x.iter().copied())
                .collect::<Vec<f32>>(),
        );
        let b = DMatrix::from_row_slice(
            n,
            d,
            &pairs
                .iter()
                .flat_map(|(_, y)| y.iter().copied())
                .collect::<Vec<f32>>(),
        );
        let r_fast = AlignmentTransform::orthogonal_procrustes_low_rank(&a, &b);

        // Build a vector orthogonal to every calibration row via Gram-Schmidt
        // against the row space (project out the span, keep the remainder).
        let qr = a.transpose().qr();
        let q = qr.q(); // d × n spanning row-space(a)
        let mut z: Vec<f32> = (0..d).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let zv = DVector::from_row_slice(&z);
        let z_perp = &zv - &q * (q.transpose() * &zv);
        z = z_perp.iter().copied().collect();

        let out = (DVector::from_row_slice(&z).transpose() * &r_fast)
            .iter()
            .copied()
            .collect::<Vec<f32>>();
        for (orig, transformed) in z.iter().zip(out.iter()) {
            assert!(
                (orig - transformed).abs() < 1e-2,
                "not identity on complement: {orig} vs {transformed}"
            );
        }
    }

    #[test]
    fn content_hash_is_deterministic_for_the_same_transform() {
        let mut rng = StdRng::seed_from_u64(1);
        let d = 4;
        let q = random_orthogonal(d, &mut rng);
        let pairs: Vec<(Vec<f32>, Vec<f32>)> = (0..16)
            .map(|_| {
                let a: Vec<f32> = (0..d).map(|_| rng.gen_range(-1.0..1.0)).collect();
                let av = DVector::from_row_slice(&a);
                let b = (&q.transpose() * &av).iter().copied().collect::<Vec<f32>>();
                (a, b)
            })
            .collect();
        let t = AlignmentTransform::fit(&pairs);
        assert_eq!(t.content_hash(), t.content_hash());
    }
}
