//! M6's manifold-conformity axis: displace a payload to a TARGET COSINE
//! (ADR-047 §4.1/§4.2), plus the norm-matched random far anchor.
//!
//! **Why a target cosine and not a bit width.** ADR-047 §4.1 retires bit-width
//! parameterisation: for symmetric absmax quantise-dequantise, cosine to the
//! unrounded activation is 0.99999 at 8-bit and 0.99610 at 4-bit, against
//! ≈0.026 for a norm-matched random vector in 1536 dims. A 4-bit-rounded
//! activation is 99.6% of the way to `aligned`; it is not a perturbation. So
//! the axis is a **dose ladder on target cosine** instead.
//!
//! **The construction is an exact rotation, and that is deliberate.** For unit
//! `u = v/‖v‖` and a unit `e` drawn orthogonal to `u`,
//!
//! ```text
//! w = ‖v‖ · ( c·u + √(1−c²)·e )
//! ```
//!
//! gives `cosine(v, w) = c` **exactly** (to float precision) and `‖w‖ = ‖v‖`.
//! ADR-047 §4.2(2) requires displace-then-rescale so that *direction* is the
//! only difference between `aligned` and `aligned_displaced`; preserving the
//! norm here means the later rescale to `natural.median` is a no-op on the
//! relationship this axis manipulates.
//!
//! **Consequence to state plainly**: because the cosine is exact by
//! construction, ADR-047 §5's ±0.02 admission arm verifies *arithmetic*, not an
//! empirical risk. The empirical content of the manipulation check is the
//! typicality arm and the width of the band it must fit inside.
//!
//! Displacement is drawn from a per-(item, dose) seeded ChaCha8 stream
//! (ADR-047 §4.2(3)) so every arm is reproducible from the receipt.

use super::gaussian_vec;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// ADR-047 §4.1's registered dose ladder on target cosine to the payload.
pub const M6_TARGET_COSINES: [f64; 4] = [0.99, 0.95, 0.90, 0.75];

/// ADR-047 §5: a dose is admitted only if its MEASURED cosine to the unrounded
/// payload is within this tolerance of its target.
pub const M6_COSINE_TOLERANCE: f64 = 0.02;

/// Seed base for the displacement RNG, distinct from the probe's
/// `RANDVEC_SEED_BASE` so displacement draws can never collide with the
/// `random` control's draws.
pub const M6_DISPLACE_SEED_BASE: u64 = 0x4D36_0D15;
/// Seed base for M6's own norm-matched random far anchor in the pre-check.
pub const M6_RANDOM_SEED_BASE: u64 = 0x4D36_5A4E;

/// The manifold band M6 admits doses into.
///
/// **This is NOT [`super::lens::OFF_MANIFOLD_COSINE`]** (0.20), which ADR-047
/// §5 explicitly forbids reusing: that constant is a *collapse* detector and
/// would classify every dose as on-manifold, telling us nothing.
///
/// The band is instead **anchored on the two measured endpoints of this very
/// run** — `random`'s typicality below, `aligned`'s above. A fixed constant
/// cannot serve, because the question is not "is this dose on the manifold"
/// but "does this dose sit strictly between the two arms it must separate".
#[derive(Debug, Clone, Copy)]
pub struct ManifoldBand {
    /// Measured typicality of the `random` far anchor.
    pub lo: f64,
    /// Measured typicality of the `aligned` payload.
    pub hi: f64,
}

impl ManifoldBand {
    /// ADR-047 §5's typicality arm: **strictly** between the two endpoints.
    /// No margin is applied — a margin would be stricter than the registered
    /// gate, which is a deviation, not a safety improvement.
    pub fn admits(&self, typicality: f64) -> bool {
        typicality > self.lo && typicality < self.hi
    }

    /// Width of the usable band. A narrow band is itself a finding: it bounds
    /// how much room the manifold axis has to move at all.
    pub fn width(&self) -> f64 {
        self.hi - self.lo
    }
}

/// ADR-047 §5's cosine arm.
pub fn cosine_in_tolerance(measured: f64, target: f64) -> bool {
    (measured - target).abs() <= M6_COSINE_TOLERANCE
}

fn l2(v: &[f32]) -> f64 {
    v.iter().map(|&x| x as f64 * x as f64).sum::<f64>().sqrt()
}

/// Displace `v` to a target cosine, preserving `‖v‖`.
///
/// Returns `v` unchanged when `target >= 1.0`. Panics on an empty vector or a
/// zero-norm input, both of which are programming errors here rather than data
/// conditions — every payload this is applied to is a live adapter output.
pub fn displace_to_cosine(v: &[f32], target: f64, seed: u64) -> Vec<f32> {
    assert!(!v.is_empty(), "cannot displace an empty vector");
    assert!(
        (0.0..=1.0).contains(&target),
        "target cosine {target} outside [0, 1]"
    );
    let norm = l2(v);
    assert!(norm > 0.0, "cannot displace a zero vector");
    if target >= 1.0 {
        return v.to_vec();
    }
    let u: Vec<f64> = v.iter().map(|&x| x as f64 / norm).collect();

    // A Gaussian draw, projected off `u` and renormalised: uniform on the unit
    // sphere of the orthogonal complement.
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut e: Vec<f64> = gaussian_vec(&mut rng, v.len())
        .into_iter()
        .map(|x| x as f64)
        .collect();
    let dot: f64 = e.iter().zip(&u).map(|(a, b)| a * b).sum();
    for (x, ui) in e.iter_mut().zip(&u) {
        *x -= dot * ui;
    }
    let en = e.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!(
        en > 1e-12,
        "orthogonal component collapsed; degenerate RNG draw"
    );

    let perp = (1.0 - target * target).sqrt();
    u.iter()
        .zip(&e)
        .map(|(&ui, &ei)| ((target * ui + perp * ei / en) * norm) as f32)
        .collect()
}

/// The far anchor: a Gaussian vector norm-matched to `v`. Mirrors the probe's
/// own `random` control (`common/m3.rs`), which draws a seeded Gaussian and
/// scales it to the effective aligned norm.
pub fn norm_matched_random(v: &[f32], seed: u64) -> Vec<f32> {
    assert!(!v.is_empty(), "cannot match an empty vector");
    let norm = l2(v);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let g = gaussian_vec(&mut rng, v.len());
    let gn = l2(&g).max(1e-300);
    g.iter().map(|&x| (x as f64 * norm / gn) as f32).collect()
}

#[cfg(test)]
mod tests {
    use super::super::lens::cosine;
    use super::*;

    fn payload(d: usize, seed: u64) -> Vec<f32> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        gaussian_vec(&mut rng, d).iter().map(|x| x * 3.5).collect()
    }

    /// The load-bearing property: the cosine is hit EXACTLY, at the real
    /// receiver width, for every registered dose — and the norm is preserved,
    /// so direction is the only thing that moved (ADR-047 §4.2(2)).
    #[test]
    fn hits_every_registered_dose_exactly_and_preserves_norm() {
        let v = payload(1536, 0xC0FFEE);
        let n0 = l2(&v);
        for (i, &c) in M6_TARGET_COSINES.iter().enumerate() {
            let w = displace_to_cosine(&v, c, M6_DISPLACE_SEED_BASE + i as u64);
            let measured = cosine(&v, &w);
            assert!((measured - c).abs() < 1e-6, "dose {c}: measured {measured}");
            assert!(cosine_in_tolerance(measured, c));
            assert!(
                (l2(&w) - n0).abs() / n0 < 1e-6,
                "dose {c}: norm {} vs {n0}",
                l2(&w)
            );
        }
    }

    #[test]
    fn is_seed_deterministic_and_seed_sensitive() {
        let v = payload(1536, 1);
        let a = displace_to_cosine(&v, 0.90, 7);
        let b = displace_to_cosine(&v, 0.90, 7);
        let c = displace_to_cosine(&v, 0.90, 8);
        assert_eq!(a, b);
        assert_ne!(a, c);
        // Different seeds still hit the same target.
        assert!((cosine(&v, &c) - 0.90).abs() < 1e-6);
    }

    #[test]
    fn target_one_is_the_identity() {
        let v = payload(64, 3);
        assert_eq!(displace_to_cosine(&v, 1.0, 5), v);
    }

    /// The far anchor is norm-matched and near-orthogonal — ADR-047 §4.1's
    /// ≈0.026 for 1536 dims is `1/√d`, so |cos| should sit within a few
    /// multiples of that.
    #[test]
    fn norm_matched_random_is_norm_matched_and_near_orthogonal() {
        let v = payload(1536, 11);
        let r = norm_matched_random(&v, M6_RANDOM_SEED_BASE);
        assert!((l2(&r) - l2(&v)).abs() / l2(&v) < 1e-6);
        let c = cosine(&v, &r).abs();
        assert!(c < 5.0 / (1536f64).sqrt(), "cos {c} implausibly large");
    }

    /// The band is strict at both ends, and is not the collapse detector.
    #[test]
    fn band_is_strict_and_distinct_from_the_collapse_threshold() {
        let b = ManifoldBand { lo: 0.02, hi: 0.66 };
        assert!(b.admits(0.30));
        assert!(!b.admits(0.02), "lower endpoint must be excluded");
        assert!(!b.admits(0.66), "upper endpoint must be excluded");
        assert!(!b.admits(0.70));
        assert!((b.width() - 0.64).abs() < 1e-12);
        // A dose at 0.30 sits far above the collapse detector, which is
        // exactly why ADR-047 §5 forbids reusing it here.
        assert!(0.30 > super::super::lens::OFF_MANIFOLD_COSINE);
    }
}
