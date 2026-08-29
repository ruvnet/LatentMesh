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
//!
//! **Patched 2026-08-28 for the live latent experiment (design doc
//! `docs/research/024` §3, ADR-002 amendment):**
//! - [`AlignmentTransform::fit_affine`] adds affine mean-centering:
//!   `aligned(z) = μ_r + α · (z − μ_s) · R`, with `μ_s`/`μ_r` carried in the
//!   hashed transform struct so [`content_hash`](AlignmentTransform::content_hash)
//!   binds the FULL affine map, not just `(R, α)`.
//! - [`apply`](AlignmentTransform::apply) no longer rebuilds the projection
//!   `DMatrix` on every call (previously ~25 MB of allocation per apply at
//!   2048×1536) — the matrix is built once per fit and cached.
//! - `content_hash` is computed once per fit and cached, not re-serialized
//!   (~25 MB of JSON at 2048×1536) on every call.
//! - A named `MᵀM` eigendecomposition fallback for the dense rectangular SVD
//!   lives behind the same fit API (`procrustes_mtm.rs`) — see the measured
//!   timing evidence in that module before trusting either path's cost.

mod procrustes_mtm;

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

/// A fitted alignment: `aligned(z) = μ_r + α · (z − μ_s) · R`.
///
/// The plain [`fit`](AlignmentTransform::fit) path stores empty `μ` vectors
/// (no centering — byte-compatible behavior with the pre-2026-08-28 crate);
/// [`fit_affine`](AlignmentTransform::fit_affine) stores the real calibration
/// means. Both `μ` vectors are serialized, so they participate in
/// [`content_hash`](AlignmentTransform::content_hash) — two transforms that
/// differ only in centering hash differently, as they must.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AlignmentTransform {
    /// `dim_sender × dim_receiver`, orthonormal (rows or columns, whichever is
    /// the smaller dimension — the semi-orthogonal Procrustes solution).
    r: Vec<Vec<f32>>,
    alpha: f32,
    dim_sender: usize,
    dim_receiver: usize,
    /// Sender-side calibration mean `μ_s` (empty ⇒ no centering, plain `fit`).
    #[serde(default)]
    mu_s: Vec<f32>,
    /// Receiver-side calibration mean `μ_r` (empty ⇒ no centering, plain `fit`).
    #[serde(default)]
    mu_r: Vec<f32>,
    /// In `[0,1]`: `1 − relative Frobenius residual` on the calibration set
    /// (mean-centered residual for `fit_affine`). A REAL measured number, not
    /// asserted — see [`AlignmentTransform::fit`]. NOTE: this is an ON-TRAIN
    /// number; the live-experiment design (docs/research/024 §5) reports
    /// held-out residual instead, precisely because this one is optimistic.
    pub confidence: f32,
    /// Projection matrix cache — built once per fit (or lazily after
    /// deserialization), never serialized, never part of the hash.
    #[serde(skip)]
    r_cached: OnceLock<DMatrix<f32>>,
    /// Content-hash cache — computed once per fit (or lazily after
    /// deserialization), never serialized.
    #[serde(skip)]
    hash_cached: OnceLock<String>,
}

impl AlignmentTransform {
    /// Fit `R`/`α` from calibration pairs via SVD-based orthogonal Procrustes.
    /// Panics on empty input or mismatched pair dimensions (a programmer
    /// error, not a runtime data condition). No mean-centering — see
    /// [`fit_affine`](AlignmentTransform::fit_affine) for the affine variant.
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
        Self::fit_inner(pairs, Vec::new(), Vec::new())
    }

    /// Fit the AFFINE alignment `aligned(z) = μ_r + α · (z − μ_s) · R`:
    /// mean-center each side on the calibration means `μ_s`/`μ_r`, then run
    /// the same Procrustes solvers as [`fit`](AlignmentTransform::fit) on the
    /// centered pairs. This is the live-experiment calibration path
    /// (docs/research/024 §5.3, ADR-002 amendment 2026-08-28): LLM hidden
    /// states are far from zero-mean, and orthogonal maps fix the origin, so
    /// uncentered Procrustes wastes fit capacity reconciling the two offsets.
    ///
    /// `μ_s`/`μ_r` are stored in the (serialized, hashed) struct — the
    /// transform hash a `LatentFrame` carries binds the whole affine map.
    /// Panics if fewer than 2 pairs (the means consume the only sample) or on
    /// mismatched dimensions, matching `fit`'s panic-on-programmer-error style.
    pub fn fit_affine(pairs: &[(Vec<f32>, Vec<f32>)]) -> Self {
        assert!(
            pairs.len() >= 2,
            "fit_affine requires at least two calibration pairs (n=1 centers to all-zeros)"
        );
        let dim_sender = pairs[0].0.len();
        let dim_receiver = pairs[0].1.len();
        let n = pairs.len() as f32;
        let mut mu_s = vec![0.0f32; dim_sender];
        let mut mu_r = vec![0.0f32; dim_receiver];
        for (a, b) in pairs {
            assert_eq!(a.len(), dim_sender, "inconsistent sender dimension");
            assert_eq!(b.len(), dim_receiver, "inconsistent receiver dimension");
            for (m, v) in mu_s.iter_mut().zip(a) {
                *m += v / n;
            }
            for (m, v) in mu_r.iter_mut().zip(b) {
                *m += v / n;
            }
        }
        Self::fit_inner(pairs, mu_s, mu_r)
    }

    /// Shared fit core: validates dimensions, builds (optionally centered)
    /// `A`/`B`, dispatches to a Procrustes solver, and packs the result.
    /// Empty `mu` vectors mean "no centering".
    fn fit_inner(pairs: &[(Vec<f32>, Vec<f32>)], mu_s: Vec<f32>, mu_r: Vec<f32>) -> Self {
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

        let center = |row: &[f32], mu: &[f32]| -> Vec<f32> {
            if mu.is_empty() {
                row.to_vec()
            } else {
                row.iter().zip(mu).map(|(v, m)| v - m).collect()
            }
        };
        // A: n × dim_sender, B: n × dim_receiver (row-major samples).
        let a = DMatrix::from_row_slice(
            n,
            dim_sender,
            &pairs
                .iter()
                .flat_map(|(x, _)| center(x, &mu_s))
                .collect::<Vec<f32>>(),
        );
        let b = DMatrix::from_row_slice(
            n,
            dim_receiver,
            &pairs
                .iter()
                .flat_map(|(_, y)| center(y, &mu_r))
                .collect::<Vec<f32>>(),
        );

        let r = if dim_sender == dim_receiver && n < dim_sender {
            Self::orthogonal_procrustes_low_rank(&a, &b)
        } else {
            Self::orthogonal_procrustes_dense(&a, &b)
        };

        Self::finish(a, b, r, dim_sender, dim_receiver, mu_s, mu_r)
    }

    /// The original direct solver: `M = Aᵀ B`, SVD(M) = `U Σ Vᵀ`, `R = U Vᵀ`.
    /// `O(d³)`-ish (dense SVD of a `dim_sender × dim_receiver` matrix). Used
    /// as the reference implementation and as the fallback when the fast
    /// path's preconditions (`dim_sender == dim_receiver`, `n < dim`) don't hold.
    /// Measured at the live-experiment shape (2048×1536, release build):
    /// see `procrustes_mtm.rs`'s ignored timing test for the real number
    /// against the design's <5 min S1b gate.
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
    /// `a`/`b` arrive already centered when `mu_s`/`mu_r` are non-empty, so
    /// α and the confidence residual are computed in the centered frame.
    /// Also seeds the projection-matrix and content-hash caches — both are
    /// computed ONCE here, not per `apply()`/`content_hash()` call.
    fn finish(
        a: DMatrix<f32>,
        b: DMatrix<f32>,
        r: DMatrix<f32>,
        dim_sender: usize,
        dim_receiver: usize,
        mu_s: Vec<f32>,
        mu_r: Vec<f32>,
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

        let t = AlignmentTransform {
            r: (0..dim_sender)
                .map(|i| r.row(i).iter().copied().collect())
                .collect(),
            alpha,
            dim_sender,
            dim_receiver,
            mu_s,
            mu_r,
            confidence,
            r_cached: OnceLock::new(),
            hash_cached: OnceLock::new(),
        };
        // Seed the caches once per fit (`r` is moved in, not rebuilt).
        let _ = t.r_cached.set(r);
        let _ = t.hash_cached.set(t.compute_content_hash());
        t
    }

    /// Apply the transform: `μ_r + α · (z − μ_s) · R` (plain-`fit` transforms
    /// have empty `μ` vectors, reducing this to `α · z · R`). Panics if
    /// `z.len() != dim_sender`. Uses the cached projection matrix — no
    /// per-call `DMatrix` rebuild (previously ~25 MB of allocation per call
    /// at the 2048×1536 live-experiment shape).
    pub fn apply(&self, z: &[f32]) -> Vec<f32> {
        assert_eq!(z.len(), self.dim_sender, "input dimension mismatch");
        let r = self.projection();
        let zv = if self.mu_s.is_empty() {
            DVector::from_row_slice(z)
        } else {
            DVector::from_iterator(z.len(), z.iter().zip(&self.mu_s).map(|(v, m)| v - m))
        };
        let mut out: Vec<f32> = (self.alpha * (zv.transpose() * r))
            .iter()
            .copied()
            .collect();
        if !self.mu_r.is_empty() {
            for (o, m) in out.iter_mut().zip(&self.mu_r) {
                *o += m;
            }
        }
        out
    }

    /// The cached `dim_sender × dim_receiver` projection matrix. Built once
    /// per fit; after deserialization it is rebuilt lazily on first use.
    fn projection(&self) -> &DMatrix<f32> {
        self.r_cached.get_or_init(|| {
            DMatrix::from_row_slice(
                self.dim_sender,
                self.dim_receiver,
                &self.r.iter().flatten().copied().collect::<Vec<f32>>(),
            )
        })
    }

    /// A content hash binding this exact `(R, α, μ_s, μ_r)` — what
    /// `LatentFrame::transform_hash` (ADR-002) references, so a receiver can
    /// verify which transform produced an aligned frame rather than trusting
    /// an unhashed matrix. Computed once per fit and cached (previously
    /// re-serialized the full matrix on every call).
    pub fn content_hash(&self) -> String {
        self.hash_cached
            .get_or_init(|| self.compute_content_hash())
            .clone()
    }

    /// The actual hash computation — serialize (cache fields are
    /// `#[serde(skip)]`, so they never influence the hash) and SHA-256.
    fn compute_content_hash(&self) -> String {
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

    /// The stored calibration means `(μ_s, μ_r)` — both empty for plain
    /// [`fit`](AlignmentTransform::fit) transforms.
    pub fn means(&self) -> (&[f32], &[f32]) {
        (&self.mu_s, &self.mu_r)
    }
}

#[cfg(test)]
mod tests;
