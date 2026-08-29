//! M4d deployment transform (ADR-024 § "Registered contingency — M4d,
//! train/deploy configuration match").
//!
//! The frozen probe delivers an adapter output to the receiver by exactly two
//! operations (`examples/common/m3.rs::four_conditions` steps 3–4 plus
//! `latentmesh_runtime::inject`):
//!
//! 1. `scale = natural.median / ‖v‖`, where `natural.median` is
//!    `norms::stats(forward_capture(receiver, inj_tokens, block, ..).per_position_l2).median`
//!    — the VENDORED FUSED forward's per-position L2 median over the slotted
//!    injection prompt;
//! 2. `InjectionSpec { vector: v, scale: Some(scale) }`, whose
//!    `effective_vector()` is `v * scale` and whose `vectors_tensor()`
//!    replicates that one effective vector across all `n_slots` positions.
//!
//! Training cannot literally call `InjectionSpec::effective_vector` — it
//! operates on `Vec<f32>` and is therefore off the autodiff tape, which is
//! the whole reason the adapter is optimized through a tensor path. So the
//! two ops are re-expressed here as candle tensor ops with the SAME operator
//! order (divide once to form the scalar `scale`, then multiply the vector by
//! it), and the duplication is pinned by [`verify_deploy_matches_probe`],
//! which calls the probe's OWN `InjectionSpec::effective_vector` on ≥8 seeded
//! vectors and asserts a relative L2 agreement ≤ 1e-6. The measured number is
//! recorded in the M4d training receipt.
//!
//! Deliberate, disclosed deviation of exactly one term: a `+1e-12` guard is
//! added to `‖v‖` before the division, because the probe's f32 host
//! arithmetic has no backward pass to protect and this one does. At the
//! measured pooled norms (O(1)–O(10)) the guard shifts `scale` by ~1e-13
//! relative — twelve orders below the 1e-6 tolerance the equivalence check
//! enforces, and the check is what proves that rather than the argument.

use candle_core::{Device, Tensor};
use latentmesh_runtime::inject::{InjectionMode, InjectionSpec};
use latentmesh_runtime::norms;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Backward-stability guard added to the norm before the reciprocal.
pub const BACKWARD_GUARD: f64 = 1e-12;

/// The probe's deployment transform, differentiable.
///
/// `pooled` is the rank-1 `(D_OUT)` F32 adapter output; the result is the
/// `(n_slots, D_OUT)` slot matrix the receiver forward writes at the
/// placeholder positions — the same object `InjectionSpec::vectors_tensor`
/// builds at probe time.
pub fn deploy_slot_vectors(
    pooled: &Tensor,
    natural_median: f32,
    n_slots: usize,
) -> candle_core::Result<Tensor> {
    assert!(n_slots > 0, "n_slots must be positive");
    let d_out = pooled.dim(0)?;
    let norm = (pooled.sqr()?.sum_all()?.sqrt()? + BACKWARD_GUARD)?;
    // Probe order: form the scalar `scale = median / ‖v‖` FIRST, then scale
    // the vector by it (`InjectionSpec::effective_vector`).
    let scale = Tensor::new(natural_median, pooled.device())?.broadcast_div(&norm)?;
    let row = pooled.broadcast_mul(&scale)?.reshape((1, d_out))?;
    Tensor::cat(&vec![&row; n_slots], 0)
}

/// Measured equivalence between [`deploy_slot_vectors`] and the probe's own
/// `InjectionSpec::effective_vector`, for the training receipt.
#[derive(Debug, Clone, serde::Serialize)]
pub struct DeployEquivalence {
    pub n_vectors: usize,
    pub n_slots: usize,
    pub input_seed_chacha8: u64,
    pub max_relative_l2_error: f32,
    pub max_relative_norm_error: f32,
    pub tolerance: f32,
    pub reference: &'static str,
    pub pass: bool,
}

/// Verify the training-side deployment transform against the PROBE'S OWN
/// function on `n_vectors` seeded inputs spanning several magnitudes.
///
/// For each draw this builds the exact `InjectionSpec` the probe builds
/// (`scale: Some(natural_median / norms::l2(&pooled))`), calls
/// `effective_vector()`, and compares it — per slot row — against
/// [`deploy_slot_vectors`]'s output. It additionally checks the semantic the
/// operator exists for: the effective vector's L2 norm equals
/// `natural_median`.
pub fn verify_deploy_matches_probe(
    dev: &Device,
    d_out: usize,
    n_vectors: usize,
    n_slots: usize,
    seed: u64,
    tolerance: f32,
) -> anyhow::Result<DeployEquivalence> {
    anyhow::ensure!(n_vectors >= 8, "equivalence check needs >= 8 vectors");
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut max_rel = 0f32;
    let mut max_norm_rel = 0f32;
    for k in 0..n_vectors {
        // Magnitudes 1e-2 .. 1e2 so the check is not a single-scale accident.
        let mag = 10f32.powi(k as i32 % 5 - 2);
        let pooled: Vec<f32> = (0..d_out)
            .map(|_| (rng.gen::<f32>() * 2.0 - 1.0) * mag)
            .collect();
        let natural_median = 1.0 + rng.gen::<f32>() * 40.0;

        // --- the probe's own code path -----------------------------------
        let spec = InjectionSpec {
            after_block: 14,
            positions: (0..n_slots).collect(),
            vector: pooled.clone(),
            scale: Some(natural_median / norms::l2(&pooled)),
            mode: InjectionMode::Overwrite,
        };
        let reference = spec.effective_vector();

        // --- the training-side tensor path -------------------------------
        let t = Tensor::from_vec(pooled.clone(), (d_out,), dev)?;
        let ours = deploy_slot_vectors(&t, natural_median, n_slots)?.to_vec2::<f32>()?;
        anyhow::ensure!(ours.len() == n_slots, "slot broadcast produced wrong rows");

        let ref_norm = norms::l2(&reference).max(1e-12);
        for row in &ours {
            let diff: f32 = row
                .iter()
                .zip(&reference)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>()
                .sqrt();
            max_rel = max_rel.max(diff / ref_norm);
            let nrel = (norms::l2(row) - natural_median).abs() / natural_median;
            max_norm_rel = max_norm_rel.max(nrel);
        }
    }
    let pass = max_rel <= tolerance && max_norm_rel <= tolerance;
    anyhow::ensure!(
        pass,
        "M4d deployment transform diverges from the probe's InjectionSpec: \
         max relative L2 {max_rel:.3e}, max relative norm error {max_norm_rel:.3e} > {tolerance:.0e}"
    );
    Ok(DeployEquivalence {
        n_vectors,
        n_slots,
        input_seed_chacha8: seed,
        max_relative_l2_error: max_rel,
        max_relative_norm_error: max_norm_rel,
        tolerance,
        reference: "latentmesh_runtime::inject::InjectionSpec::effective_vector (the frozen probe's own function), replicated across n_slots as InjectionSpec::vectors_tensor does",
        pass,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_the_probes_own_injection_spec() {
        let dev = Device::Cpu;
        let r = verify_deploy_matches_probe(&dev, 1536, 8, 8, 0x4D34_D317, 1e-6).unwrap();
        assert!(r.pass);
        assert!(r.max_relative_l2_error <= 1e-6, "{r:?}");
        assert!(r.max_relative_norm_error <= 1e-6, "{r:?}");
    }

    #[test]
    fn slot_rows_are_identical_and_norm_matched() {
        let dev = Device::Cpu;
        let v: Vec<f32> = (0..16).map(|i| (i as f32) - 7.5).collect();
        let t = Tensor::from_vec(v, (16,), &dev).unwrap();
        let out = deploy_slot_vectors(&t, 12.5, 8).unwrap().to_vec2().unwrap();
        assert_eq!(out.len(), 8);
        for row in &out {
            assert_eq!(row, &out[0]);
            assert!((norms::l2(row) - 12.5).abs() < 1e-4);
        }
    }
}
