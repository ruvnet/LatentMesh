//! Named `MᵀM` eigendecomposition fallback for the dense rectangular
//! Procrustes SVD (live-experiment design `docs/research/024` §5.5, risk #7).
//!
//! The dense solver computes `SVD(M)` of `M = AᵀB` (`d_s × d_r`, 2048×1536 at
//! the run-1 shape). This fallback instead eigendecomposes the receiver-side
//! Gram matrix `MᵀM` (`d_r × d_r` = 1536×1536, symmetric PSD): with
//! `M = U Σ Vᵀ`, `MᵀM = V Σ² Vᵀ`, so `V` and `Σ` come from a symmetric
//! eigenproblem and `R = U Vᵀ = M V Σ⁻¹ Vᵀ` needs no left singular vectors.
//! The Gram side is the RECEIVER side deliberately — `d_r < d_s` at the run-1
//! shape, so it is the smaller of the two possible Gram matrices.
//!
//! **Honesty caveats (evidence-labelled, per ADR-014):**
//! - Forming `MᵀM` squares the condition number, so the Gram/eigen step runs
//!   in `f64` (`M` itself is `f32`); singular values below `σ_max · 1e-6` are
//!   dropped (their polar-factor contribution is not identifiable), and
//!   eigenvalues are clamped at 0 before `sqrt` (symmetric eigensolvers
//!   return tiny negatives from rounding).
//! - Equivalence to the dense solver is verified on full-rank rectangular
//!   problems in `mtm_eig_matches_dense_reference` below; on rank-deficient
//!   input the dropped components make `R` only sub-orthogonal there.
//! - The <5 min S1b timing gate is measured by the `#[ignore]`d test below —
//!   run `cargo test -p latentmesh-align --release -- --ignored --nocapture`
//!   (release build; a debug-build timing would be meaningless).
//! - **Measured 2026-08-28** (this workstation, release build, seed
//!   20260828, two runs): dense SVD 2048×1536 = **16.48 s / 17.65 s**,
//!   MᵀM-eig = **2.17 s / 3.67 s**, max entrywise |ΔR| = 1e-6. The dense
//!   path passes the design's <5 min gate with ~17× margin, so it REMAINS
//!   the default rectangular solver; this fallback stays the named, tested
//!   alternative for larger shapes.

use crate::AlignmentTransform;
use nalgebra::DMatrix;

impl AlignmentTransform {
    /// `R = M V Σ⁻¹ Vᵀ` via symmetric eigendecomposition of `MᵀM`, where
    /// `M = AᵀB`. Same signature and contract as
    /// `orthogonal_procrustes_dense` — behind the same fit API, selectable if
    /// the dense SVD ever fails the timing gate at a larger shape.
    #[allow(dead_code)] // named fallback: wired behind the fit API on gate failure
    pub(crate) fn orthogonal_procrustes_mtm_eig(
        a: &DMatrix<f32>,
        b: &DMatrix<f32>,
    ) -> DMatrix<f32> {
        let m = a.transpose() * b; // d_s × d_r
        Self::polar_via_gram(&m)
    }

    /// The polar factor `U Vᵀ` of `m`, computed from the `mᵀm` Gram
    /// eigendecomposition (in `f64`) instead of a full SVD of `m`.
    pub(crate) fn polar_via_gram(m: &DMatrix<f32>) -> DMatrix<f32> {
        let md = m.map(|x| x as f64); // d_s × d_r
        let gram = md.transpose() * &md; // d_r × d_r, symmetric PSD
        let eig = gram.symmetric_eigen();
        // σ_i = sqrt(λ_i) with λ clamped at 0 (rounding can go slightly negative).
        let sigmas: Vec<f64> = eig.eigenvalues.iter().map(|l| l.max(0.0).sqrt()).collect();
        let sigma_max = sigmas.iter().cloned().fold(0.0f64, f64::max);
        let threshold = sigma_max * 1e-6;
        // V · diag(1/σ) with unidentifiable (σ ≤ threshold) components dropped.
        let mut v_inv_sigma = eig.eigenvectors.clone(); // d_r × d_r
        for (j, sigma) in sigmas.iter().enumerate() {
            let scale = if *sigma > threshold { 1.0 / sigma } else { 0.0 };
            v_inv_sigma.column_mut(j).scale_mut(scale);
        }
        // R = (M · V Σ⁻¹) · Vᵀ : d_s × d_r.
        let r64 = md * v_inv_sigma * eig.eigenvectors.transpose();
        r64.map(|x| x as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::time::Instant;

    fn random_matrix(rows: usize, cols: usize, rng: &mut StdRng) -> DMatrix<f32> {
        let data: Vec<f32> = (0..rows * cols).map(|_| rng.gen_range(-1.0..1.0)).collect();
        DMatrix::from_row_slice(rows, cols, &data)
    }

    #[test]
    fn mtm_eig_matches_dense_reference() {
        // Full-rank rectangular problem (d_s > d_r, n > d_s so M is full
        // rank): the polar factor is unique, so the two solvers must agree
        // entrywise, not just in action.
        let mut rng = StdRng::seed_from_u64(2048);
        let (n, d_s, d_r) = (48usize, 20usize, 12usize);
        let a = random_matrix(n, d_s, &mut rng);
        let b = random_matrix(n, d_r, &mut rng);
        let r_dense = AlignmentTransform::orthogonal_procrustes_dense(&a, &b);
        let r_mtm = AlignmentTransform::orthogonal_procrustes_mtm_eig(&a, &b);
        for (x, y) in r_dense.iter().zip(r_mtm.iter()) {
            assert!(
                (x - y).abs() < 1e-3,
                "MᵀM-eig fallback diverged from dense SVD: {x} vs {y}"
            );
        }
    }

    #[test]
    fn mtm_eig_result_is_semi_orthogonal() {
        // RᵀR should be the d_r×d_r identity when d_s ≥ d_r and M is full rank.
        let mut rng = StdRng::seed_from_u64(7);
        let (n, d_s, d_r) = (40usize, 16usize, 10usize);
        let a = random_matrix(n, d_s, &mut rng);
        let b = random_matrix(n, d_r, &mut rng);
        let r = AlignmentTransform::orthogonal_procrustes_mtm_eig(&a, &b);
        let rtr = r.transpose() * &r;
        for i in 0..d_r {
            for j in 0..d_r {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (rtr[(i, j)] - expected).abs() < 1e-3,
                    "RᵀR[{i},{j}] = {} (expected {expected})",
                    rtr[(i, j)]
                );
            }
        }
    }

    /// The S1b timing gate (design §7): dense SVD at the run-1 shape
    /// 2048×1536 must complete in <5 min (release build), else the fit API
    /// switches to the MᵀM fallback and re-gates. Both paths are timed on the
    /// SAME matrix so the numbers are directly comparable. `#[ignore]`d: this
    /// is a measurement, not a correctness test, and it is only meaningful
    /// under `--release`.
    #[test]
    #[ignore = "timing gate: run with `cargo test -p latentmesh-align --release -- --ignored --nocapture`"]
    fn svd_timing_gate_2048x1536() {
        let mut rng = StdRng::seed_from_u64(20260828);
        let m = random_matrix(2048, 1536, &mut rng);

        let t0 = Instant::now();
        let svd = nalgebra::linalg::SVD::new(m.clone(), true, true);
        let u = svd.u.expect("SVD did not compute U");
        let v_t = svd.v_t.expect("SVD did not compute Vᵀ");
        let r_dense = u * v_t;
        let dense_secs = t0.elapsed().as_secs_f64();

        let t1 = Instant::now();
        let r_mtm = AlignmentTransform::polar_via_gram(&m);
        let mtm_secs = t1.elapsed().as_secs_f64();

        let max_abs_diff = r_dense
            .iter()
            .zip(r_mtm.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);

        println!("dense SVD 2048x1536: {dense_secs:.2} s");
        println!("MᵀM eig  2048x1536: {mtm_secs:.2} s");
        println!("max |R_dense − R_mtm|: {max_abs_diff:.6}");
        assert!(
            dense_secs < 300.0 || mtm_secs < 300.0,
            "BOTH paths failed the 5-min gate: dense {dense_secs:.1} s, mtm {mtm_secs:.1} s"
        );
    }
}
