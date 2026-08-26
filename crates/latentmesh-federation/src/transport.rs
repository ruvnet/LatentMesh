//! Rules over Radio (ADR-017): a transition rule rides an LMS1
//! `SemanticEnvelope` with `SemanticClass::Control` — a bounded control
//! event, exactly what ADR-007 says Radio carries. CRC, fragmentation, and
//! the replay window come from the Air stack unchanged; this module only
//! binds rule bytes into the envelope and back, plus the selective
//! transmission decision.

use crate::model::WorldModel;
use crate::rule::{FederationError, TransitionRule};
use latentmesh_air_core::{SemanticClass, SemanticEnvelope};

/// Domain-separated 16-byte state hash over the rule bytes, carried in the
/// envelope's `state_hash` so receivers can bind frame `state_tag`s to
/// content.
fn rule_state_hash(rule_bytes: &[u8]) -> [u8; 16] {
    use sha2::{Digest, Sha256};
    // SHA-256 truncated to 128 bits with a domain tag — the same construction
    // the Air crate uses for its critical-state hash.
    let mut hasher = Sha256::new();
    hasher.update(b"LatentMesh-Federation-Rule-v1");
    hasher.update(rule_bytes);
    let digest = hasher.finalize();
    let mut out = [0u8; 16];
    out.copy_from_slice(&digest[..16]);
    out
}

/// Wrap a rule for transmission. `Private` rules are refused by the rule
/// encoder before any envelope is built.
pub fn encode_rule_envelope(
    rule: &TransitionRule,
    source_id: u32,
    epoch: u32,
    message_id: u32,
    logical_sequence: u64,
    priority: u8,
    signature: Option<[u8; 64]>,
) -> Result<SemanticEnvelope, FederationError> {
    let body = rule.encode_for_transmission()?;
    let envelope = SemanticEnvelope {
        class: SemanticClass::Control,
        priority,
        source_id,
        epoch,
        message_id,
        logical_sequence,
        state_hash: rule_state_hash(&body),
        body,
        signature,
    };
    envelope.validate()?;
    Ok(envelope)
}

/// Unwrap and validate a received rule envelope: class must be `Control`,
/// the state hash must match the body, and the rule must decode cleanly.
pub fn decode_rule_envelope(
    envelope: &SemanticEnvelope,
) -> Result<TransitionRule, FederationError> {
    if envelope.class != SemanticClass::Control {
        return Err(FederationError::Malformed("not a control envelope"));
    }
    if envelope.state_hash != rule_state_hash(&envelope.body) {
        return Err(FederationError::Malformed("state hash does not bind body"));
    }
    TransitionRule::decode(&envelope.body)
}

/// Sender-side scope filter: may this rule be *offered* to a peer in
/// `peer_cluster` at all? `Private` never leaves (also enforced by the
/// encoder); cluster-scoped rules are offered only to same-cluster peers.
pub fn offerable_to(rule: &TransitionRule, peer_cluster: u32) -> bool {
    match rule.scope {
        crate::rule::RuleScope::Private => false,
        crate::rule::RuleScope::Cluster(id) => id == peer_cluster,
        crate::rule::RuleScope::Global | crate::rule::RuleScope::Unresolved => true,
    }
}

/// Selective transmission (ADR-007 / the RuView case): transmit only when the
/// rule may be offered to this peer's cluster at all AND it adds information
/// beyond what the peer set already knows. Structurally the sending-side
/// version of the admission question: if the shared model already predicts
/// what this rule predicts, stay silent.
pub fn should_transmit_to(rule: &TransitionRule, shared: &WorldModel, peer_cluster: u32) -> bool {
    if !offerable_to(rule, peer_cluster) {
        return false;
    }
    should_transmit(rule, shared)
}

/// Information-gain half of the transmission decision (scope-agnostic).
pub fn should_transmit(rule: &TransitionRule, shared: &WorldModel) -> bool {
    match shared.predict(&rule.pre, &rule.action) {
        Some(post) => post != rule.post,
        None => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rule::RuleScope;

    fn rule() -> TransitionRule {
        TransitionRule {
            pre: "hot".into(),
            action: "cool".into(),
            post: "warm".into(),
            support: 12,
            confidence: 0.8,
            scope: RuleScope::Global,
        }
    }

    #[test]
    fn envelope_round_trips_a_rule() {
        let envelope = encode_rule_envelope(&rule(), 7, 1, 100, 1, 8, None).unwrap();
        let bytes = envelope.encode().unwrap();
        let back = SemanticEnvelope::decode(&bytes).unwrap();
        assert_eq!(decode_rule_envelope(&back).unwrap(), rule());
    }

    #[test]
    fn tampered_body_fails_the_state_hash_binding() {
        let mut envelope = encode_rule_envelope(&rule(), 7, 1, 100, 1, 8, None).unwrap();
        envelope.body[10] ^= 0xff;
        assert!(matches!(
            decode_rule_envelope(&envelope),
            Err(FederationError::Malformed(_))
        ));
    }

    #[test]
    fn wrong_class_envelopes_are_refused() {
        let mut envelope = encode_rule_envelope(&rule(), 7, 1, 100, 1, 8, None).unwrap();
        envelope.class = SemanticClass::Telemetry;
        assert!(matches!(
            decode_rule_envelope(&envelope),
            Err(FederationError::Malformed(_))
        ));
    }

    #[test]
    fn scope_filters_offers_on_the_sending_side() {
        let shared = WorldModel::new();
        let mut r = rule();
        r.scope = RuleScope::Cluster(3);
        assert!(should_transmit_to(&r, &shared, 3));
        assert!(!should_transmit_to(&r, &shared, 4));
        r.scope = RuleScope::Private;
        assert!(!should_transmit_to(&r, &shared, 3));
        r.scope = RuleScope::Global;
        assert!(should_transmit_to(&r, &shared, 9));
    }

    #[test]
    fn silent_when_the_shared_model_already_knows() {
        let mut shared = WorldModel::new();
        assert!(should_transmit(&rule(), &shared));
        shared.install(rule());
        assert!(!should_transmit(&rule(), &shared));
        // A *different* prediction for the same key is informative again.
        let mut updated = rule();
        updated.post = "cold".into();
        assert!(should_transmit(&updated, &shared));
    }
}
