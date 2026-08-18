//! Causal edge verification (ADR-003): an agent-to-agent latent edge earns
//! execution authority only by surviving a counterfactual test against five
//! controls (zero / random / mismatched / self-generated / text-equivalent
//! state), using a nonparametric sign-flip permutation test — no assumption
//! that task-value noise is Gaussian or that outcomes are continuous. Beating
//! `text_equivalent` specifically is what makes a surviving edge a claim
//! about LATENT communication, not just about communication vs. silence.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

/// Paired trial outcomes for one candidate edge `A → B`: `real[i]` is `B`'s
/// task value when it received `A`'s actual latent state on trial `i`; each
/// control is `B`'s value on the SAME trial `i` with `A`'s state replaced.
/// All slices must be the same length (paired design — this is what makes the
/// sign-flip test valid without a distributional assumption). Six controls,
/// not four (ADR-003 rev. 2026-08-18): `text_equivalent` is what makes this a
/// claim about LATENT communication specifically, not merely about
/// communication-vs-silence — an edge must beat the text-serialized version
/// of the same content, not just beat having no content at all.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EdgeTrial {
    pub real: Vec<f64>,
    pub zero: Vec<f64>,
    pub random: Vec<f64>,
    pub mismatched: Vec<f64>,
    pub self_generated: Vec<f64>,
    /// `B`'s value when given the same content as `real`, but serialized to
    /// text and re-tokenized instead of transferred as a latent frame.
    pub text_equivalent: Vec<f64>,
}

impl EdgeTrial {
    fn controls(&self) -> [(&'static str, &[f64]); 5] {
        [
            ("zero", &self.zero),
            ("random", &self.random),
            ("mismatched", &self.mismatched),
            ("self_generated", &self.self_generated),
            ("text_equivalent", &self.text_equivalent),
        ]
    }
}

/// The outcome of testing one candidate edge.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum EdgeVerdict {
    /// Survived all five controls: `real` beats every control with p < α.
    Admit {
        delta_v: f64,
        worst_p_value: f64,
    },
    Reject {
        reason: String,
    },
}

/// One-sided sign-flip (Fisher randomization) permutation test for paired
/// data: H0 is "the sign of each difference is equally likely +/-" (no
/// systematic effect); H1 is "the mean difference is positive." Returns the
/// p-value. Deterministic given `seed` — reproducible, independently
/// re-runnable, matching this repo's "no unverified claim" stance.
pub fn sign_flip_permutation_test(differences: &[f64], resamples: usize, seed: u64) -> f64 {
    if differences.is_empty() {
        return 1.0;
    }
    let observed: f64 = differences.iter().sum::<f64>() / differences.len() as f64;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut at_least_as_extreme = 0usize;
    for _ in 0..resamples {
        let permuted_mean: f64 = differences
            .iter()
            .map(|&d| if rng.gen_bool(0.5) { d } else { -d })
            .sum::<f64>()
            / differences.len() as f64;
        if permuted_mean >= observed {
            at_least_as_extreme += 1;
        }
    }
    // +1/+1 correction: the observed arrangement is itself one valid permutation.
    (at_least_as_extreme as f64 + 1.0) / (resamples as f64 + 1.0)
}

/// Verify a candidate edge against all five controls (zero, random,
/// mismatched, self-generated, text-equivalent). Admission requires the WORST
/// (largest, most-favorable-to-the-null) p-value across all five to still
/// clear `alpha` — a marginal edge cannot hide behind one weak control
/// (ADR-003's stricter-than-mean-of-controls bar). Beating `text_equivalent`
/// specifically is what makes a survived edge a claim about latent
/// communication, not merely about communication-vs-silence.
pub fn verify_edge(trial: &EdgeTrial, alpha: f64, resamples: usize, seed: u64) -> EdgeVerdict {
    let n = trial.real.len();
    for (name, c) in trial.controls() {
        if c.len() != n {
            return EdgeVerdict::Reject {
                reason: format!("control `{name}` has {} trials, expected {n}", c.len()),
            };
        }
    }
    if n == 0 {
        return EdgeVerdict::Reject {
            reason: "no trials".into(),
        };
    }

    let mean = |xs: &[f64]| xs.iter().sum::<f64>() / xs.len() as f64;
    let real_mean = mean(&trial.real);
    let controls_mean = mean(
        &[
            &trial.zero[..],
            &trial.random[..],
            &trial.mismatched[..],
            &trial.self_generated[..],
            &trial.text_equivalent[..],
        ]
        .concat(),
    );
    let delta_v = real_mean - controls_mean;

    let mut worst_p = 0.0f64;
    let mut worst_name = "";
    for (i, (name, c)) in trial.controls().into_iter().enumerate() {
        let diffs: Vec<f64> = trial
            .real
            .iter()
            .zip(c.iter())
            .map(|(r, cv)| r - cv)
            .collect();
        // Distinct seed per control so the five tests aren't accidentally correlated.
        let p = sign_flip_permutation_test(&diffs, resamples, seed.wrapping_add(i as u64 + 1));
        if p > worst_p {
            worst_p = p;
            worst_name = name;
        }
    }

    if delta_v > 0.0 && worst_p < alpha {
        EdgeVerdict::Admit {
            delta_v,
            worst_p_value: worst_p,
        }
    } else {
        EdgeVerdict::Reject {
            reason: format!(
                "delta_v={delta_v:.4}, worst control `{worst_name}` p={worst_p:.4} (need > 0 and p < {alpha})"
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trial_with_shift(shift: f64, n: usize, seed: u64) -> EdgeTrial {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut noise = |shift: f64| shift + rng.gen_range(-0.5..0.5);
        let mut vecs: [Vec<f64>; 6] = Default::default();
        for _ in 0..n {
            vecs[0].push(noise(shift)); // real (shifted = has signal)
            vecs[1].push(noise(0.0)); // zero
            vecs[2].push(noise(0.0)); // random
            vecs[3].push(noise(0.0)); // mismatched
            vecs[4].push(noise(0.0)); // self_generated
            vecs[5].push(noise(0.0)); // text_equivalent
        }
        let [real, zero, random, mismatched, self_generated, text_equivalent] = vecs;
        EdgeTrial {
            real,
            zero,
            random,
            mismatched,
            self_generated,
            text_equivalent,
        }
    }

    #[test]
    fn a_synthetic_edge_with_real_signal_is_admitted() {
        let trial = trial_with_shift(0.9, 40, 1);
        let verdict = verify_edge(&trial, 0.05, 2000, 123);
        assert!(
            matches!(verdict, EdgeVerdict::Admit { .. }),
            "expected Admit, got {verdict:?}"
        );
        if let EdgeVerdict::Admit {
            delta_v,
            worst_p_value,
        } = verdict
        {
            assert!(delta_v > 0.5, "delta_v too small: {delta_v}");
            assert!(worst_p_value < 0.05);
        }
    }

    #[test]
    fn a_synthetic_edge_with_no_signal_is_rejected_almost_always() {
        // All five conditions from the SAME distribution: no real effect.
        let mut false_admits = 0;
        for seed in 0..20u64 {
            let trial = trial_with_shift(0.0, 40, seed * 7 + 1);
            if matches!(
                verify_edge(&trial, 0.05, 2000, seed * 13 + 5),
                EdgeVerdict::Admit { .. }
            ) {
                false_admits += 1;
            }
        }
        // Requiring ALL four controls to individually clear p<0.05 makes this
        // an intersection test — the false-admit rate under the null should be
        // far below the per-control 5%, not just bounded by it.
        assert!(
            false_admits <= 2,
            "too many false admits under the null: {false_admits}/20"
        );
    }

    #[test]
    fn p_value_is_invariant_to_positive_rescaling_of_outcomes() {
        let trial = trial_with_shift(0.9, 24, 42);
        let scaled = EdgeTrial {
            real: trial.real.iter().map(|v| v * 10.0).collect(),
            zero: trial.zero.iter().map(|v| v * 10.0).collect(),
            random: trial.random.iter().map(|v| v * 10.0).collect(),
            mismatched: trial.mismatched.iter().map(|v| v * 10.0).collect(),
            self_generated: trial.self_generated.iter().map(|v| v * 10.0).collect(),
            text_equivalent: trial.text_equivalent.iter().map(|v| v * 10.0).collect(),
        };
        let a = verify_edge(&trial, 0.05, 2000, 999);
        let b = verify_edge(&scaled, 0.05, 2000, 999);
        let p = |v: &EdgeVerdict| match v {
            EdgeVerdict::Admit { worst_p_value, .. } => *worst_p_value,
            EdgeVerdict::Reject { .. } => f64::NAN,
        };
        assert!(
            (p(&a) - p(&b)).abs() < 1e-12,
            "p-value changed under positive rescaling: {a:?} vs {b:?}"
        );
    }

    #[test]
    fn permutation_test_p_value_is_bounded_in_zero_one() {
        let diffs = vec![0.1, -0.2, 0.3, 0.05, -0.1, 0.4];
        let p = sign_flip_permutation_test(&diffs, 1000, 1);
        assert!((0.0..=1.0).contains(&p));
    }
}
