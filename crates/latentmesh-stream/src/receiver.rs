//! The receive-side composition (ADR-015): gate admission first (ADR-008 —
//! an unadmittable frame never touches sequencing or escalation state), then
//! sequence tracking, then authority escalation. The output is the frame plus
//! the *effective* authority the consumer may act at.

use crate::error::StreamError;
use crate::escalation::{AuthorityEscalator, EscalationConfig};
use crate::sequence::{SequenceEvent, SequenceTracker};
use latentmesh_core::{Authority, LatentFrame};
use latentmesh_gate::{Gate, Policy};

/// A frame that has cleared the gate and the stream checks, with the
/// authority the stream has actually earned for it.
#[derive(Clone, Debug)]
pub struct AdmittedFrame {
    pub frame: LatentFrame,
    /// `min(frame.authority, earned-by-stream)`; the consumer must not act
    /// above this level.
    pub effective_authority: Authority,
    /// What sequencing observed (a reported gap also resets escalation).
    pub sequence_event: SequenceEvent,
}

/// One receiving endpoint of a latent stream.
pub struct LatentStreamReceiver {
    policy: Policy,
    tracker: SequenceTracker,
    escalator: AuthorityEscalator,
}

impl LatentStreamReceiver {
    pub fn new(policy: Policy, escalation: EscalationConfig) -> Self {
        LatentStreamReceiver {
            policy,
            tracker: SequenceTracker::new(),
            escalator: AuthorityEscalator::new(escalation),
        }
    }

    /// The stream's currently earned authority.
    pub fn earned_authority(&self) -> Authority {
        self.escalator.current()
    }

    /// Ingest one decoded frame.
    pub fn ingest(&mut self, frame: LatentFrame) -> Result<AdmittedFrame, StreamError> {
        Gate::admit(&frame, &self.policy).map_err(|e| StreamError::from_admission(&e))?;
        let sequence_event = self.tracker.observe(frame.sequence)?;
        if let SequenceEvent::Gap { .. } = sequence_event {
            self.escalator.reset();
        }
        let effective_authority = self.escalator.observe(frame.authority, frame.confidence);
        Ok(AdmittedFrame {
            frame,
            effective_authority,
            sequence_event,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use latentmesh_core::{Encoding, Payload, Provenance};
    use latentmesh_gate::causal::{verify_edge, EdgeTrial};

    fn frame(seq: u64, confidence: f32) -> LatentFrame {
        LatentFrame {
            id: format!("f{seq}"),
            sender_model: "sender".into(),
            receiver_space: "receiver".into(),
            transform_hash: "trusted".into(),
            sequence: seq,
            payload: Payload::encode(&[0.5; 16], Encoding::F16),
            confidence,
            provenance: Provenance {
                sender_model: "sender".into(),
                context_hash: "ctx".into(),
                parents: vec![],
            },
            authority: Authority::ActionInfluencing,
            timestamp: 1,
        }
    }

    fn verified_policy() -> Policy {
        // Establish the edge ceiling the honest way: run the causal
        // verification (ADR-003) and derive the ceiling from its verdict.
        let n = 24;
        let trial = EdgeTrial {
            real: (0..n).map(|i| 2.0 + (i as f64 % 3.0) * 0.01).collect(),
            zero: vec![0.1; n],
            random: vec![0.15; n],
            mismatched: vec![0.12; n],
            self_generated: vec![0.2; n],
            text_equivalent: vec![0.3; n],
        };
        let verdict = verify_edge(&trial, 0.05, 500, 42);
        let mut policy = Policy::new(0.99);
        policy
            .trust_transform("trusted")
            .set_ceiling_from_verdict("sender", "receiver", &verdict);
        policy
    }

    #[test]
    fn admitted_stream_escalates_but_gap_resets_to_observe_only() {
        let mut rx = LatentStreamReceiver::new(verified_policy(), EscalationConfig::default());
        for seq in 0..10 {
            rx.ingest(frame(seq, 0.95)).unwrap();
        }
        assert!(rx.earned_authority() > Authority::ObserveOnly);
        let admitted = rx.ingest(frame(15, 0.95)).unwrap();
        assert_eq!(admitted.sequence_event, SequenceEvent::Gap { missing: 5 });
        // The gap frame itself is back to the floor.
        assert_eq!(admitted.effective_authority, Authority::ObserveOnly);
    }

    #[test]
    fn unadmitted_frame_never_advances_stream_state() {
        let mut rx = LatentStreamReceiver::new(verified_policy(), EscalationConfig::default());
        let mut bad = frame(0, 0.95);
        bad.transform_hash = "unknown".into();
        assert!(matches!(rx.ingest(bad), Err(StreamError::Rejected(_))));
        // Sequence 0 was not consumed by the rejected frame.
        let ok = rx.ingest(frame(0, 0.95)).unwrap();
        assert_eq!(ok.sequence_event, SequenceEvent::InOrder);
    }

    #[test]
    fn default_deny_policy_keeps_everything_at_observe_only() {
        let mut policy = Policy::new(0.99);
        policy.trust_transform("trusted");
        let mut rx = LatentStreamReceiver::new(policy, EscalationConfig::default());
        // Edge has no verified ceiling → gate default-denies anything above
        // ObserveOnly.
        let mut f = frame(0, 0.99);
        f.authority = Authority::ContextInject;
        assert!(matches!(rx.ingest(f), Err(StreamError::Rejected(_))));
    }
}
