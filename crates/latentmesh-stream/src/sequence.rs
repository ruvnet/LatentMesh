//! Per-stream total order (ADR-004): the sender assigns a monotonic
//! `sequence`; the receiver detects gaps, duplicates, and regressions instead
//! of silently reordering. A gap is reported, not repaired — what to do about
//! missing partial state is the consumer's decision (and resets authority
//! escalation, see [`crate::escalation`]).

use crate::error::StreamError;
use latentmesh_core::LatentFrame;

/// Sender half: stamps consecutive sequence numbers onto outgoing frames.
#[derive(Debug, Default)]
pub struct LatentStreamSender {
    next_sequence: u64,
}

impl LatentStreamSender {
    pub fn new() -> Self {
        Self::default()
    }

    /// The sequence the next frame will carry.
    pub fn next_sequence(&self) -> u64 {
        self.next_sequence
    }

    /// Stamp `frame` with the next sequence number and advance.
    pub fn stamp(&mut self, frame: &mut LatentFrame) -> u64 {
        let seq = self.next_sequence;
        frame.sequence = seq;
        self.next_sequence = self.next_sequence.saturating_add(1);
        seq
    }
}

/// What the tracker observed about an accepted frame's position.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SequenceEvent {
    /// The expected next sequence arrived.
    InOrder,
    /// The stream jumped forward, skipping `missing` sequences.
    Gap { missing: u64 },
}

/// Receiver half: accepts strictly-forward sequences, rejects duplicates and
/// regressions. Bounded state (a single watermark), so a hostile sender
/// cannot grow receiver memory by scattering sequence numbers.
#[derive(Debug, Default)]
pub struct SequenceTracker {
    /// Highest accepted sequence + 1; `None` until the first frame.
    next_expected: Option<u64>,
}

impl SequenceTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Highest accepted sequence, if any frame has been accepted.
    pub fn watermark(&self) -> Option<u64> {
        self.next_expected.map(|n| n - 1)
    }

    /// Observe `sequence`. First frame establishes the baseline (any starting
    /// sequence is allowed, so a receiver can join mid-stream); after that,
    /// only forward progress is accepted. `u64::MAX` is rejected outright:
    /// its successor is unrepresentable, so accepting it would wedge the
    /// tracker and misclassify later frames (self-DoS a hostile peer could
    /// otherwise trigger with one frame).
    pub fn observe(&mut self, sequence: u64) -> Result<SequenceEvent, StreamError> {
        if sequence == u64::MAX {
            return Err(StreamError::Malformed(
                "sequence u64::MAX is reserved (successor unrepresentable)".into(),
            ));
        }
        match self.next_expected {
            None => {
                self.next_expected = Some(sequence + 1);
                Ok(SequenceEvent::InOrder)
            }
            Some(expected) => {
                if sequence + 1 == expected {
                    return Err(StreamError::DuplicateSequence(sequence));
                }
                if sequence < expected {
                    return Err(StreamError::RegressedSequence {
                        got: sequence,
                        watermark: expected - 1,
                    });
                }
                let missing = sequence - expected;
                self.next_expected = Some(sequence + 1);
                if missing == 0 {
                    Ok(SequenceEvent::InOrder)
                } else {
                    Ok(SequenceEvent::Gap { missing })
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blank_frame() -> LatentFrame {
        use latentmesh_core::{Authority, Encoding, Payload, Provenance};
        LatentFrame {
            id: "f".into(),
            sender_model: "m".into(),
            receiver_space: "r".into(),
            transform_hash: "t".into(),
            sequence: 0,
            payload: Payload::encode(&[1.0], Encoding::F32),
            confidence: 0.5,
            provenance: Provenance {
                sender_model: "m".into(),
                context_hash: "c".into(),
                parents: vec![],
            },
            authority: Authority::ObserveOnly,
            timestamp: 0,
        }
    }

    #[test]
    fn sender_stamps_consecutively_from_zero() {
        let mut sender = LatentStreamSender::new();
        assert_eq!(sender.next_sequence(), 0);
        let mut f = blank_frame();
        assert_eq!(sender.stamp(&mut f), 0);
        assert_eq!(sender.stamp(&mut f), 1);
        assert_eq!(f.sequence, 1);
    }

    #[test]
    fn tracker_accepts_in_order_and_reports_gaps() {
        let mut t = SequenceTracker::new();
        assert_eq!(t.observe(10).unwrap(), SequenceEvent::InOrder);
        assert_eq!(t.observe(11).unwrap(), SequenceEvent::InOrder);
        assert_eq!(t.observe(14).unwrap(), SequenceEvent::Gap { missing: 2 });
        assert_eq!(t.watermark(), Some(14));
    }

    #[test]
    fn tracker_rejects_the_reserved_max_sequence() {
        let mut t = SequenceTracker::new();
        assert!(matches!(
            t.observe(u64::MAX),
            Err(StreamError::Malformed(_))
        ));
        // The tracker is not wedged: normal sequences still work.
        assert_eq!(t.observe(0).unwrap(), SequenceEvent::InOrder);
        assert_eq!(
            t.observe(u64::MAX - 1).unwrap(),
            SequenceEvent::Gap {
                missing: u64::MAX - 2
            }
        );
        assert!(matches!(
            t.observe(u64::MAX),
            Err(StreamError::Malformed(_))
        ));
    }

    #[test]
    fn tracker_rejects_duplicates_and_regressions() {
        let mut t = SequenceTracker::new();
        t.observe(5).unwrap();
        assert_eq!(t.observe(5).unwrap_err(), StreamError::DuplicateSequence(5));
        assert_eq!(
            t.observe(3).unwrap_err(),
            StreamError::RegressedSequence {
                got: 3,
                watermark: 5
            }
        );
    }
}
