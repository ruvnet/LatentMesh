//! `latentmesh-gate` — capability-governed latent execution (ADR-008):
//! `execute(z) ⟺ signature(z) ∧ authority(z) ∧ provenance(z) ∧ risk(z) < τ`.
//! Ported in shape from `cognitum-one/slack`'s AGL admission module — a typed,
//! deterministic, first-violated-rule admission check — applied to opaque
//! latent payloads instead of code mutations. `causal` (ADR-003) supplies the
//! measurement that decides how high an edge's authority ceiling may go.

pub mod causal;

use latentmesh_core::{Authority, LatentFrame};
use std::collections::{HashMap, HashSet};

/// The `delta_v` thresholds that map an admitted edge's measured causal value
/// to an authority ceiling (see [`ceiling_from_verdict_with`]). Made
/// policy-configurable 2026-08-28 for the live latent experiment
/// (docs/research/024 §3, risk #2): on a 0/1 accuracy scale a per-item
/// `delta_v` above the stock 1.0/0.5 is arithmetically unreachable, so the
/// authority ladder would be dead — the run-1 values ([`CeilingThresholds::run1`])
/// are recalibrated to that scale. The stock values remain the [`Default`],
/// so all pre-existing behavior (and every existing test) is unchanged.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CeilingThresholds {
    /// `delta_v` strictly above this ⇒ `ActionInfluencing`.
    pub action_influencing_dv: f64,
    /// `delta_v` strictly above this (but not `action_influencing_dv`) ⇒
    /// `LatentPrefix`; at or below ⇒ `ContextInject`.
    pub latent_prefix_dv: f64,
}

impl Default for CeilingThresholds {
    /// The crate's original hard-coded ladder: `>1.0` ⇒ `ActionInfluencing`,
    /// `>0.5` ⇒ `LatentPrefix`.
    fn default() -> Self {
        CeilingThresholds {
            action_influencing_dv: 1.0,
            latent_prefix_dv: 0.5,
        }
    }
}

impl CeilingThresholds {
    /// Run-1 values, pre-declared in the live-experiment design
    /// (docs/research/024 §3/§8 risk #2) on the GSM8K accuracy scale:
    /// `delta_v > 0.05` ⇒ `LatentPrefix`, `> 0.15` ⇒ `ActionInfluencing`
    /// (paired with `Policy::new(0.8)` for the run-1 risk threshold — that
    /// knob was already configurable). Experimental parameters, not validated
    /// constants: the design's honesty section (§10.9) records that run 1
    /// demonstrates ceilings MOVING with verdicts, not that these values are
    /// the right ones.
    pub fn run1() -> Self {
        CeilingThresholds {
            action_influencing_dv: 0.15,
            latent_prefix_dv: 0.05,
        }
    }
}

/// The admission policy an edge/frame is checked against. `authority_ceiling_by_edge`
/// is keyed by `(sender_model, receiver_space)`; an edge absent from this map
/// defaults to `ObserveOnly` — **default-deny**, matching AGL's stance that
/// authority never silently expands.
#[derive(Clone, Debug, Default)]
pub struct Policy {
    pub authority_ceiling_by_edge: HashMap<(String, String), Authority>,
    pub risk_threshold: f32,
    pub known_transforms: HashSet<String>,
    /// The `delta_v` → authority-ceiling ladder used by
    /// [`Policy::set_ceiling_from_verdict`]. Defaults to the stock values.
    pub ceiling_thresholds: CeilingThresholds,
}

impl Policy {
    pub fn new(risk_threshold: f32) -> Self {
        Policy {
            risk_threshold,
            ..Default::default()
        }
    }

    /// Register a transform as known/trusted (ADR-002 `transform_hash`).
    pub fn trust_transform(&mut self, transform_hash: impl Into<String>) -> &mut Self {
        self.known_transforms.insert(transform_hash.into());
        self
    }

    /// Set an edge's authority ceiling from a causal verification verdict
    /// (ADR-003 → ADR-008's link): unverified/rejected edges are capped at
    /// `ObserveOnly`; the ceiling rises with measured, significant `delta_v`,
    /// judged against this policy's own [`CeilingThresholds`].
    pub fn set_ceiling_from_verdict(
        &mut self,
        sender_model: impl Into<String>,
        receiver_space: impl Into<String>,
        verdict: &causal::EdgeVerdict,
    ) -> &mut Self {
        let ceiling = ceiling_from_verdict_with(verdict, &self.ceiling_thresholds);
        self.authority_ceiling_by_edge
            .insert((sender_model.into(), receiver_space.into()), ceiling);
        self
    }
}

/// Map a causal-verification verdict to an authority ceiling using the stock
/// (default) thresholds. A rejected (or never-tested) edge can only ever
/// reach `ObserveOnly` — this is the enforcement point for ADR-003's
/// "causally unverified edges never reach `ActionInfluencing`."
pub fn ceiling_from_verdict(verdict: &causal::EdgeVerdict) -> Authority {
    ceiling_from_verdict_with(verdict, &CeilingThresholds::default())
}

/// [`ceiling_from_verdict`] with an explicit `delta_v` ladder. Regardless of
/// thresholds, `Reject` is always capped at `ObserveOnly` — configurability
/// tunes how far a VERIFIED edge may rise, never whether an unverified one
/// rises at all.
pub fn ceiling_from_verdict_with(
    verdict: &causal::EdgeVerdict,
    thresholds: &CeilingThresholds,
) -> Authority {
    match verdict {
        causal::EdgeVerdict::Reject { .. } => Authority::ObserveOnly,
        causal::EdgeVerdict::Admit { delta_v, .. }
            if *delta_v > thresholds.action_influencing_dv =>
        {
            Authority::ActionInfluencing
        }
        causal::EdgeVerdict::Admit { delta_v, .. } if *delta_v > thresholds.latent_prefix_dv => {
            Authority::LatentPrefix
        }
        causal::EdgeVerdict::Admit { .. } => Authority::ContextInject,
    }
}

/// Why a frame was refused execution — the first violated rule, in the same
/// style as AGL's `AdmissionError`.
#[derive(Clone, Debug, PartialEq)]
pub enum AdmissionError {
    UnknownTransform(String),
    AuthorityExceedsCeiling {
        requested: Authority,
        ceiling: Authority,
    },
    UnresolvedProvenance(String),
    RiskTooHigh {
        risk: f32,
        threshold: f32,
    },
}

impl AdmissionError {
    pub fn reason(&self) -> String {
        match self {
            AdmissionError::UnknownTransform(h) => {
                format!("transform `{h}` is not a known/trusted alignment")
            }
            AdmissionError::AuthorityExceedsCeiling { requested, ceiling } => {
                format!("requested authority {requested:?} exceeds this edge's ceiling {ceiling:?}")
            }
            AdmissionError::UnresolvedProvenance(why) => {
                format!("provenance does not resolve: {why}")
            }
            AdmissionError::RiskTooHigh { risk, threshold } => {
                format!("estimated risk {risk:.3} >= threshold {threshold:.3}")
            }
        }
    }
}

/// The admission gate. Deterministic, clock-free in the sense that `now` is a
/// parameter, not read from the system clock — independently re-runnable.
pub struct Gate;

impl Gate {
    /// Check whether `frame` may execute (take effect at its receiver) under
    /// `policy`. Returns the FIRST violated rule.
    pub fn admit(frame: &LatentFrame, policy: &Policy) -> Result<(), AdmissionError> {
        // Rule 1 — signature: the alignment transform must be known/trusted.
        if !policy.known_transforms.contains(&frame.transform_hash) {
            return Err(AdmissionError::UnknownTransform(
                frame.transform_hash.clone(),
            ));
        }
        // Rule 2 — authority: default-deny; unknown edges get the ObserveOnly floor.
        let key = (frame.sender_model.clone(), frame.receiver_space.clone());
        let ceiling = policy
            .authority_ceiling_by_edge
            .get(&key)
            .copied()
            .unwrap_or(Authority::ObserveOnly);
        if frame.authority > ceiling {
            return Err(AdmissionError::AuthorityExceedsCeiling {
                requested: frame.authority,
                ceiling,
            });
        }
        // Rule 3 — provenance: must carry a resolvable context binding.
        if frame.provenance.context_hash.is_empty() {
            return Err(AdmissionError::UnresolvedProvenance(
                "empty context_hash".into(),
            ));
        }
        // Rule 4 — risk: an EXPLICIT STAND-IN (ADR-008), not a trained
        // trajectory-risk model. Documented, not hidden, per ADR-001 §8.
        let risk = estimate_risk(frame);
        if risk >= policy.risk_threshold {
            return Err(AdmissionError::RiskTooHigh {
                risk,
                threshold: policy.risk_threshold,
            });
        }
        Ok(())
    }
}

/// Deterministic placeholder risk score in `[0,1]`: higher requested
/// authority, lower alignment confidence, and shallower provenance lineage
/// each raise risk. NOT a claim of DreamGuard-equivalent trajectory-risk
/// modeling (ADR-008) — a real risk model needs labeled trajectory data this
/// repo does not have.
fn estimate_risk(frame: &LatentFrame) -> f32 {
    let authority_weight = match frame.authority {
        Authority::ObserveOnly => 0.05,
        Authority::ContextInject => 0.2,
        Authority::LatentPrefix => 0.45,
        Authority::ActionInfluencing => 0.75,
    };
    let confidence_penalty = (1.0 - frame.confidence).clamp(0.0, 1.0);
    let provenance_depth_bonus = (frame.provenance.parents.len() as f32 / 8.0).min(0.2);
    (authority_weight + 0.5 * confidence_penalty - provenance_depth_bonus).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use latentmesh_core::{Encoding, Payload, Provenance};

    fn frame(
        authority: Authority,
        confidence: f32,
        transform_hash: &str,
        context_hash: &str,
    ) -> LatentFrame {
        LatentFrame {
            id: "f1".into(),
            sender_model: "sender-7b".into(),
            receiver_space: "receiver-13b".into(),
            transform_hash: transform_hash.into(),
            sequence: 0,
            payload: Payload::encode(&[0.1, 0.2, 0.3], Encoding::F16),
            confidence,
            provenance: Provenance {
                sender_model: "sender-7b".into(),
                context_hash: context_hash.into(),
                parents: vec![],
            },
            authority,
            timestamp: 1_800_000_000,
        }
    }

    fn policy_allowing(edge_ceiling: Authority) -> Policy {
        let mut p = Policy::new(0.9); // permissive risk threshold for these tests
        p.trust_transform("t1");
        p.authority_ceiling_by_edge
            .insert(("sender-7b".into(), "receiver-13b".into()), edge_ceiling);
        p
    }

    #[test]
    fn a_well_formed_frame_within_ceiling_is_admitted() {
        let f = frame(Authority::ContextInject, 0.95, "t1", "ctx-hash");
        let p = policy_allowing(Authority::LatentPrefix);
        assert!(Gate::admit(&f, &p).is_ok());
    }

    #[test]
    fn unknown_transform_is_rejected() {
        let f = frame(Authority::ObserveOnly, 0.95, "unregistered", "ctx-hash");
        let p = policy_allowing(Authority::ActionInfluencing);
        assert_eq!(
            Gate::admit(&f, &p),
            Err(AdmissionError::UnknownTransform("unregistered".into()))
        );
    }

    #[test]
    fn authority_above_the_edge_ceiling_is_rejected() {
        let f = frame(Authority::ActionInfluencing, 0.95, "t1", "ctx-hash");
        let p = policy_allowing(Authority::ContextInject);
        assert!(matches!(
            Gate::admit(&f, &p),
            Err(AdmissionError::AuthorityExceedsCeiling { .. })
        ));
    }

    #[test]
    fn empty_provenance_is_rejected() {
        let f = frame(Authority::ObserveOnly, 0.95, "t1", "");
        let p = policy_allowing(Authority::ActionInfluencing);
        assert!(matches!(
            Gate::admit(&f, &p),
            Err(AdmissionError::UnresolvedProvenance(_))
        ));
    }

    #[test]
    fn low_confidence_high_authority_trips_the_risk_threshold() {
        let f = frame(Authority::ActionInfluencing, 0.1, "t1", "ctx-hash");
        let mut p = policy_allowing(Authority::ActionInfluencing);
        p.risk_threshold = 0.5; // strict
        assert!(matches!(
            Gate::admit(&f, &p),
            Err(AdmissionError::RiskTooHigh { .. })
        ));
    }

    #[test]
    fn an_edge_with_no_verification_record_defaults_to_observe_only() {
        // No entry in authority_ceiling_by_edge for this (sender, receiver) —
        // requesting ActionInfluencing must be refused regardless of confidence.
        let f = frame(Authority::ActionInfluencing, 0.99, "t1", "ctx-hash");
        let mut p = Policy::new(0.99);
        p.trust_transform("t1");
        // deliberately do NOT set a ceiling for this edge
        assert!(matches!(
            Gate::admit(&f, &p),
            Err(AdmissionError::AuthorityExceedsCeiling {
                ceiling: Authority::ObserveOnly,
                ..
            })
        ));
    }

    #[test]
    fn ceiling_from_verdict_never_grants_action_influencing_to_a_rejected_edge() {
        let rejected = causal::EdgeVerdict::Reject {
            reason: "no signal".into(),
        };
        assert_eq!(ceiling_from_verdict(&rejected), Authority::ObserveOnly);
        let strong = causal::EdgeVerdict::Admit {
            delta_v: 1.5,
            worst_p_value: 0.001,
        };
        assert_eq!(ceiling_from_verdict(&strong), Authority::ActionInfluencing);
        let weak = causal::EdgeVerdict::Admit {
            delta_v: 0.1,
            worst_p_value: 0.04,
        };
        assert_eq!(ceiling_from_verdict(&weak), Authority::ContextInject);
    }

    // -----------------------------------------------------------------------
    // 2026-08-28 patch tests: policy-configurable ceiling thresholds
    // (docs/research/024 §3, risk #2 — the run-1 accuracy-scale ladder).
    // -----------------------------------------------------------------------

    fn admit(delta_v: f64) -> causal::EdgeVerdict {
        causal::EdgeVerdict::Admit {
            delta_v,
            worst_p_value: 0.01,
        }
    }

    #[test]
    fn default_thresholds_reproduce_the_stock_ladder_exactly() {
        // Guard against a silent behavior change: the default-thresholds path
        // must agree with the historical free function at every rung.
        let defaults = CeilingThresholds::default();
        for verdict in [
            admit(1.5),
            admit(0.7),
            admit(0.1),
            causal::EdgeVerdict::Reject { reason: "x".into() },
        ] {
            assert_eq!(
                ceiling_from_verdict(&verdict),
                ceiling_from_verdict_with(&verdict, &defaults),
                "default thresholds diverged from the stock ladder on {verdict:?}"
            );
        }
        assert_eq!(defaults.action_influencing_dv, 1.0);
        assert_eq!(defaults.latent_prefix_dv, 0.5);
    }

    #[test]
    fn run1_thresholds_make_the_ladder_reachable_on_the_accuracy_scale() {
        // On 0/1 accuracy outcomes, per-item delta_v > 1.0 is unreachable —
        // the run-1 values (0.05 → LatentPrefix, 0.15 → ActionInfluencing)
        // must let realistic accuracy deltas climb the ladder.
        let run1 = CeilingThresholds::run1();
        assert_eq!(
            ceiling_from_verdict_with(&admit(0.03), &run1),
            Authority::ContextInject
        );
        assert_eq!(
            ceiling_from_verdict_with(&admit(0.06), &run1),
            Authority::LatentPrefix
        );
        assert_eq!(
            ceiling_from_verdict_with(&admit(0.16), &run1),
            Authority::ActionInfluencing
        );
        // The same delta_v values are dead under the stock ladder — this IS
        // the bug the configurability exists to fix.
        assert_eq!(ceiling_from_verdict(&admit(0.16)), Authority::ContextInject);
    }

    #[test]
    fn rejected_edges_stay_observe_only_under_any_thresholds() {
        let rejected = causal::EdgeVerdict::Reject {
            reason: "no signal".into(),
        };
        for t in [
            CeilingThresholds::default(),
            CeilingThresholds::run1(),
            // Even absurdly permissive thresholds never lift a Reject.
            CeilingThresholds {
                action_influencing_dv: -1.0,
                latent_prefix_dv: -1.0,
            },
        ] {
            assert_eq!(
                ceiling_from_verdict_with(&rejected, &t),
                Authority::ObserveOnly
            );
        }
    }

    #[test]
    fn policy_with_run1_thresholds_raises_the_ceiling_above_context_inject() {
        // The S4 gate's live mechanism (design §7): with run-1 thresholds a
        // small-but-real accuracy delta must move an edge's ceiling above
        // ContextInject, so a LatentPrefix frame is then admitted.
        let mut p = Policy::new(0.8); // run-1 risk threshold
        p.ceiling_thresholds = CeilingThresholds::run1();
        p.trust_transform("t1");
        p.set_ceiling_from_verdict("sender-7b", "receiver-13b", &admit(0.07));
        assert_eq!(
            p.authority_ceiling_by_edge[&("sender-7b".into(), "receiver-13b".into())],
            Authority::LatentPrefix
        );
        let f = frame(Authority::LatentPrefix, 0.95, "t1", "ctx-hash");
        assert!(Gate::admit(&f, &p).is_ok());

        // A default-thresholds policy given the same verdict stays at
        // ContextInject (defaults preserved — no silent recalibration).
        let mut stock = Policy::new(0.8);
        stock.set_ceiling_from_verdict("sender-7b", "receiver-13b", &admit(0.07));
        assert_eq!(
            stock.authority_ceiling_by_edge[&("sender-7b".into(), "receiver-13b".into())],
            Authority::ContextInject
        );
    }
}
