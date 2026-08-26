//! Local-scope validation (ADR-017): "does this hold for me?" — ADR-003's
//! admission question applied to transition rules instead of communication
//! edges, reusing the gate's sign-flip permutation test rather than inventing
//! a second statistical mechanism.
//!
//! A candidate rule is admitted only if the paired improvement it brings to
//! held-out local prediction is positive and significant against each decoy
//! control:
//!
//! 1. **no-rule baseline** — the model without the candidate;
//! 2. **support-shuffled decoy** — the candidate with its post-state replaced
//!    by a shuffled draw from local post-states (same shape, wrong content);
//! 3. **scope-mismatch decoy** — the candidate applied to the wrong
//!    `(pre, action)` key (right content, wrong place).

use crate::model::WorldModel;
use crate::rule::{RuleScope, TransitionRule};
use latentmesh_gate::causal::sign_flip_permutation_test;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

/// Statistical knobs, mirroring `latentmesh-gate::causal`'s conventions.
#[derive(Clone, Copy, Debug)]
pub struct AdmissionConfig {
    /// Significance level for each control.
    pub alpha: f64,
    /// Permutation resamples per control.
    pub resamples: usize,
    /// Seed for the deterministic decoys and permutations.
    pub seed: u64,
    /// Minimum held-out transitions required to decide at all.
    pub min_holdout: usize,
}

impl Default for AdmissionConfig {
    fn default() -> Self {
        AdmissionConfig {
            alpha: 0.05,
            resamples: 1000,
            seed: 0x4c4d_4645_4431,
            min_holdout: 8,
        }
    }
}

/// The admission outcome, with the failing control named on rejection
/// (mirroring the gate's audit style).
#[derive(Clone, Debug, PartialEq)]
pub enum RuleVerdict {
    Admit { gain: f64, worst_p: f64 },
    Reject { control: String, reason: String },
}

/// Validate a candidate rule from a peer against the local held-out log.
/// Cluster scoping is checked first: a cluster-scoped rule from a different
/// cluster is rejected before any statistics run.
pub fn validate_candidate(
    model: &WorldModel,
    candidate: &TransitionRule,
    local_cluster: u32,
    config: &AdmissionConfig,
) -> RuleVerdict {
    if let RuleScope::Cluster(id) = candidate.scope {
        if id != local_cluster {
            return RuleVerdict::Reject {
                control: "scope".into(),
                reason: format!("cluster {id} rule offered to cluster {local_cluster}"),
            };
        }
    }
    if candidate.scope == RuleScope::Private {
        return RuleVerdict::Reject {
            control: "scope".into(),
            reason: "private rules are not federable".into(),
        };
    }
    let n = model.holdout().len();
    if n < config.min_holdout {
        return RuleVerdict::Reject {
            control: "holdout".into(),
            reason: format!("only {n} held-out transitions, need {}", config.min_holdout),
        };
    }

    let with_candidate = model.holdout_scores(Some(candidate));
    let baseline = model.holdout_scores(None);

    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut posts: Vec<&str> = model.holdout().iter().map(|t| t.post.as_str()).collect();
    posts.shuffle(&mut rng);
    let shuffled_post = posts
        .iter()
        .find(|p| **p != candidate.post)
        .copied()
        .unwrap_or("__shuffled__");
    let shuffled_decoy = TransitionRule {
        post: shuffled_post.to_string(),
        ..candidate.clone()
    };
    let mismatched_decoy = TransitionRule {
        pre: format!("__not_{}", candidate.pre),
        ..candidate.clone()
    };

    let controls: [(&str, Vec<f64>); 3] = [
        ("no_rule", baseline),
        (
            "support_shuffled",
            model.holdout_scores(Some(&shuffled_decoy)),
        ),
        (
            "scope_mismatched",
            model.holdout_scores(Some(&mismatched_decoy)),
        ),
    ];

    let mut worst_p = 0.0f64;
    let mut worst_control = "";
    let mut gain_vs_baseline = 0.0f64;
    for (i, (name, control_scores)) in controls.iter().enumerate() {
        let differences: Vec<f64> = with_candidate
            .iter()
            .zip(control_scores.iter())
            .map(|(a, b)| a - b)
            .collect();
        let mean_diff = differences.iter().sum::<f64>() / differences.len() as f64;
        if *name == "no_rule" {
            gain_vs_baseline = mean_diff;
        }
        let p = sign_flip_permutation_test(
            &differences,
            config.resamples,
            config.seed.wrapping_add(i as u64 + 1),
        );
        if p > worst_p {
            worst_p = p;
            worst_control = name;
        }
        if mean_diff < 0.0 {
            return RuleVerdict::Reject {
                control: (*name).into(),
                reason: format!("candidate underperforms control ({mean_diff:.4})"),
            };
        }
    }

    if gain_vs_baseline > 0.0 && worst_p < config.alpha {
        RuleVerdict::Admit {
            gain: gain_vs_baseline,
            worst_p,
        }
    } else {
        RuleVerdict::Reject {
            control: worst_control.into(),
            reason: format!(
                "gain={gain_vs_baseline:.4}, worst p={worst_p:.4} (need gain > 0 and p < {})",
                config.alpha
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Transition;

    fn rule(pre: &str, action: &str, post: &str, scope: RuleScope) -> TransitionRule {
        TransitionRule {
            pre: pre.into(),
            action: action.into(),
            post: post.into(),
            support: 20,
            confidence: 0.9,
            scope,
        }
    }

    /// A holdout log where `(hot, cool) → warm` recurs — a rule that captures
    /// it should be admitted; one that contradicts it should not.
    fn model_with_signal() -> WorldModel {
        let mut w = WorldModel::new();
        for i in 0..24 {
            w.record_holdout(Transition {
                pre: "hot".into(),
                action: "cool".into(),
                post: "warm".into(),
            });
            w.record_holdout(Transition {
                pre: format!("s{i}"),
                action: "noop".into(),
                post: format!("s{i}"),
            });
        }
        w
    }

    #[test]
    fn a_genuinely_predictive_rule_is_admitted() {
        let model = model_with_signal();
        let candidate = rule("hot", "cool", "warm", RuleScope::Global);
        let verdict = validate_candidate(&model, &candidate, 0, &AdmissionConfig::default());
        match verdict {
            RuleVerdict::Admit { gain, worst_p } => {
                assert!(gain > 0.0);
                assert!(worst_p < 0.05);
            }
            RuleVerdict::Reject { control, reason } => {
                panic!("expected admit, got reject on {control}: {reason}")
            }
        }
    }

    #[test]
    fn a_wrong_prediction_rule_is_rejected() {
        let model = model_with_signal();
        let candidate = rule("hot", "cool", "cold", RuleScope::Global);
        assert!(matches!(
            validate_candidate(&model, &candidate, 0, &AdmissionConfig::default()),
            RuleVerdict::Reject { .. }
        ));
    }

    #[test]
    fn cluster_mismatch_and_private_scope_reject_before_statistics() {
        let model = model_with_signal();
        let foreign = rule("hot", "cool", "warm", RuleScope::Cluster(9));
        assert!(matches!(
            validate_candidate(&model, &foreign, 1, &AdmissionConfig::default()),
            RuleVerdict::Reject { .. }
        ));
        let private = rule("hot", "cool", "warm", RuleScope::Private);
        assert!(matches!(
            validate_candidate(&model, &private, 0, &AdmissionConfig::default()),
            RuleVerdict::Reject { .. }
        ));
    }

    #[test]
    fn thin_holdout_evidence_refuses_to_decide() {
        let mut model = WorldModel::new();
        model.record_holdout(Transition {
            pre: "a".into(),
            action: "b".into(),
            post: "c".into(),
        });
        let candidate = rule("a", "b", "c", RuleScope::Global);
        assert!(matches!(
            validate_candidate(&model, &candidate, 0, &AdmissionConfig::default()),
            RuleVerdict::Reject { .. }
        ));
    }
}
