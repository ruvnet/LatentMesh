//! `latentmesh-gate` — capability-governed latent execution (ADR-008):
//! `execute(z) ⟺ signature(z) ∧ authority(z) ∧ provenance(z) ∧ risk(z) < τ`.
//! Ported in shape from `cognitum-one/slack`'s AGL admission module — a typed,
//! deterministic, first-violated-rule admission check — applied to opaque
//! latent payloads instead of code mutations. `causal` (ADR-003) supplies the
//! measurement that decides how high an edge's authority ceiling may go.

pub mod causal;

use latentmesh_core::{Authority, LatentFrame};
use std::collections::{HashMap, HashSet};

/// The admission policy an edge/frame is checked against. `authority_ceiling_by_edge`
/// is keyed by `(sender_model, receiver_space)`; an edge absent from this map
/// defaults to `ObserveOnly` — **default-deny**, matching AGL's stance that
/// authority never silently expands.
#[derive(Clone, Debug, Default)]
pub struct Policy {
    pub authority_ceiling_by_edge: HashMap<(String, String), Authority>,
    pub risk_threshold: f32,
    pub known_transforms: HashSet<String>,
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
    /// `ObserveOnly`; the ceiling rises with measured, significant `delta_v`.
    pub fn set_ceiling_from_verdict(
        &mut self,
        sender_model: impl Into<String>,
        receiver_space: impl Into<String>,
        verdict: &causal::EdgeVerdict,
    ) -> &mut Self {
        let ceiling = ceiling_from_verdict(verdict);
        self.authority_ceiling_by_edge
            .insert((sender_model.into(), receiver_space.into()), ceiling);
        self
    }
}

/// Map a causal-verification verdict to an authority ceiling. A rejected (or
/// never-tested) edge can only ever reach `ObserveOnly` — this is the
/// enforcement point for ADR-003's "causally unverified edges never reach
/// `ActionInfluencing`."
pub fn ceiling_from_verdict(verdict: &causal::EdgeVerdict) -> Authority {
    match verdict {
        causal::EdgeVerdict::Reject { .. } => Authority::ObserveOnly,
        causal::EdgeVerdict::Admit { delta_v, .. } if *delta_v > 1.0 => {
            Authority::ActionInfluencing
        }
        causal::EdgeVerdict::Admit { delta_v, .. } if *delta_v > 0.5 => Authority::LatentPrefix,
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
}
