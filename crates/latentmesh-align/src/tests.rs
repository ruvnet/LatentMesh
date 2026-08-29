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

// ---------------------------------------------------------------------------
// 2026-08-28 patch tests (live-experiment design docs/research/024 §3):
// affine mean-centering, projection-matrix cache, hash-once semantics.
// ---------------------------------------------------------------------------

/// Ground-truth AFFINE map: `b = (a − c) · Q + d` with a genuinely offset
/// sender distribution. Uncentered Procrustes cannot represent the offsets
/// (orthogonal maps fix the origin); `fit_affine` must recover the map.
#[test]
fn fit_affine_recovers_a_known_affine_map_that_plain_fit_cannot() {
    let mut rng = StdRng::seed_from_u64(240828);
    let d = 16;
    let q = random_orthogonal(d, &mut rng);
    let c: Vec<f32> = (0..d).map(|_| rng.gen_range(2.0..3.0)).collect();
    let offset: Vec<f32> = (0..d).map(|_| rng.gen_range(-3.0..-2.0)).collect();
    let ground_truth = |a: &[f32]| -> Vec<f32> {
        let centered = DVector::from_iterator(d, a.iter().zip(&c).map(|(v, m)| v - m));
        (&q.transpose() * centered)
            .iter()
            .zip(&offset)
            .map(|(v, o)| v + o)
            .collect()
    };
    let mk = |rng: &mut StdRng| -> Vec<f32> {
        // Sender vectors clustered AROUND c, so the data is far from zero-mean.
        c.iter().map(|m| m + rng.gen_range(-1.0..1.0)).collect()
    };
    let pairs: Vec<(Vec<f32>, Vec<f32>)> = (0..64)
        .map(|_| {
            let a = mk(&mut rng);
            let b = ground_truth(&a);
            (a, b)
        })
        .collect();

    let affine = AlignmentTransform::fit_affine(&pairs);
    let plain = AlignmentTransform::fit(&pairs);
    assert!(
        affine.confidence > 0.99,
        "affine fit should be near-perfect on an exactly-affine map, got {}",
        affine.confidence
    );
    assert!(
        affine.confidence > plain.confidence,
        "affine ({}) should beat uncentered ({}) on offset data",
        affine.confidence,
        plain.confidence
    );

    // Held-out application check against the ground-truth affine map.
    let z = mk(&mut rng);
    let expected = ground_truth(&z);
    let got = affine.apply(&z);
    for (e, g) in expected.iter().zip(got.iter()) {
        assert!(
            (e - g).abs() < 0.05,
            "affine apply diverged: expected {e}, got {g}"
        );
    }
}

#[test]
fn fit_affine_stores_the_calibration_means_and_hashes_them() {
    let mut rng = StdRng::seed_from_u64(31);
    let d = 6;
    let pairs: Vec<(Vec<f32>, Vec<f32>)> = (0..24)
        .map(|_| {
            let a: Vec<f32> = (0..d).map(|_| rng.gen_range(0.5..1.5)).collect();
            let b: Vec<f32> = a.iter().rev().map(|v| v + 2.0).collect();
            (a, b)
        })
        .collect();
    let affine = AlignmentTransform::fit_affine(&pairs);
    let (mu_s, mu_r) = affine.means();
    assert_eq!(mu_s.len(), d);
    assert_eq!(mu_r.len(), d);
    assert!(
        mu_s.iter().all(|m| *m > 0.4),
        "μ_s should reflect the offset"
    );
    assert!(
        mu_r.iter().all(|m| *m > 2.0),
        "μ_r should reflect the offset"
    );

    // The means are part of the hashed struct: the same pairs fit with and
    // without centering must hash differently (they are different maps).
    let plain = AlignmentTransform::fit(&pairs);
    let (ps, pr) = plain.means();
    assert!(ps.is_empty() && pr.is_empty(), "plain fit stores no means");
    assert_ne!(
        affine.content_hash(),
        plain.content_hash(),
        "μ_s/μ_r must participate in the content hash"
    );
    // And the hash is stable across calls (computed once, cached).
    assert_eq!(affine.content_hash(), affine.content_hash());
}

#[test]
#[should_panic(expected = "at least two calibration pairs")]
fn fit_affine_rejects_a_single_pair() {
    let _ = AlignmentTransform::fit_affine(&[(vec![1.0, 2.0], vec![3.0, 4.0])]);
}

#[test]
fn apply_after_serde_round_trip_matches_fresh_fit() {
    // The projection-matrix and hash caches are #[serde(skip)] — after a
    // serialize/deserialize round trip they rebuild lazily and must produce
    // identical outputs and an identical content hash.
    let mut rng = StdRng::seed_from_u64(88);
    let d = 10;
    let q = random_orthogonal(d, &mut rng);
    let pairs: Vec<(Vec<f32>, Vec<f32>)> = (0..32)
        .map(|_| {
            let a: Vec<f32> = (0..d).map(|_| rng.gen_range(0.0..2.0)).collect();
            let av = DVector::from_row_slice(&a);
            let b = (&q.transpose() * &av).iter().copied().collect::<Vec<f32>>();
            (a, b)
        })
        .collect();
    for t in [
        AlignmentTransform::fit(&pairs),
        AlignmentTransform::fit_affine(&pairs),
    ] {
        let json = serde_json::to_string(&t).unwrap();
        let back: AlignmentTransform = serde_json::from_str(&json).unwrap();
        let z: Vec<f32> = (0..d).map(|_| rng.gen_range(0.0..2.0)).collect();
        assert_eq!(t.apply(&z), back.apply(&z), "apply changed across serde");
        assert_eq!(
            t.content_hash(),
            back.content_hash(),
            "content hash changed across serde"
        );
    }
}

#[test]
fn repeated_apply_is_consistent_with_the_cached_projection() {
    // The cache must not change results call-over-call (OnceLock seeded at
    // fit time; this pins the "no rebuild per apply" patch behaviorally).
    let pairs: Vec<(Vec<f32>, Vec<f32>)> = (0..8)
        .map(|i| {
            let a: Vec<f32> = (0..4).map(|j| (i * 4 + j) as f32 * 0.1).collect();
            let b: Vec<f32> = a.iter().map(|v| v * 2.0 + 1.0).collect();
            (a, b)
        })
        .collect();
    let t = AlignmentTransform::fit_affine(&pairs);
    let z = vec![0.3, 0.1, 0.7, 0.2];
    let first = t.apply(&z);
    for _ in 0..10 {
        assert_eq!(t.apply(&z), first);
    }
}
