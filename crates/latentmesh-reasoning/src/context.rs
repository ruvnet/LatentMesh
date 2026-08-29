//! `ContextState` — the recurrent task context S (ADR-041 §4.2, §5).
//!
//! `S_t = U(S_{t-1}, D_t)`: each [`ContextState::update`] absorbs one piece
//! of evidence into a bounded state vector and returns a new, hash-chained
//! [`ContextCheckpoint`]. The state is data only — see the crate-level
//! Invariant 1 note for why nothing here can authorize an action.
//!
//! Bounded means enforced, not aspirational: [`ContextCapacity::max_dim`]
//! rejects an oversized update instead of silently growing, and
//! [`ContextCapacity::max_history`] evicts the oldest rollback checkpoint
//! once the ring is full, so memory never grows without limit no matter how
//! many updates a session performs.

use std::collections::VecDeque;
use std::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Hard, enforced limits on a [`ContextState`]. Not advisory: [`ContextState::update`]
/// returns [`ContextError::CapacityExceeded`] rather than growing past `max_dim`,
/// and the rollback ring never holds more than `max_history` checkpoints.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextCapacity {
    /// Maximum number of scalar elements the state vector may hold.
    pub max_dim: usize,
    /// Maximum number of prior checkpoints retained for [`ContextState::rollback`]
    /// / [`ContextState::rollback_to`]. Oldest is evicted first once exceeded.
    pub max_history: usize,
}

/// A pointer to the evidence a [`ContextState::update`] absorbed. Not itself
/// authenticated — provenance authentication is the gate crate's job; this
/// is only the record of what was claimed to have been used.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProvenanceRef {
    /// Content hash of the evidence item (`D_t`) this update absorbed.
    pub evidence_hash: [u8; 32],
    /// Free-form origin tag, e.g. `"tool:web_search"`, `"observation:sensor_7"`.
    pub source: String,
}

/// One immutable, hash-chained snapshot of the recurrent context S.
///
/// Every field the ADR requires of a conforming `ContextUpdate` (ADR-041
/// §4.2) is present: `state` is `new_state`, `result_hash`/`parent_hash`
/// chain the lineage, `confidence` and `provenance` carry the evidence
/// trail, and `model_fingerprint`/`adapter_fingerprint` bind the update to
/// the model that produced it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ContextCheckpoint {
    /// Monotonically increasing logical clock, not wall-clock time.
    pub sequence: u64,
    pub state: Vec<f32>,
    /// `result_hash` of the checkpoint this one was derived from. All-zero
    /// for the genesis checkpoint.
    pub parent_hash: [u8; 32],
    /// Content hash over every field below plus `parent_hash` — see
    /// [`checkpoint_hash`].
    pub result_hash: [u8; 32],
    pub confidence: f32,
    pub provenance: Vec<ProvenanceRef>,
    pub model_fingerprint: Option<[u8; 32]>,
    pub adapter_fingerprint: Option<[u8; 32]>,
}

/// Deterministic content hash over a checkpoint's fields. No wall-clock, no
/// randomness — two calls with identical inputs always produce the same
/// digest, which is what lets [`ContextState`] tests assert bit-exact
/// reproducibility.
fn checkpoint_hash(
    sequence: u64,
    state: &[f32],
    parent_hash: &[u8; 32],
    confidence: f32,
    provenance: &[ProvenanceRef],
    model_fingerprint: Option<[u8; 32]>,
    adapter_fingerprint: Option<[u8; 32]>,
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"latentmesh-reasoning.context.v1");
    hasher.update(sequence.to_le_bytes());
    hasher.update((state.len() as u64).to_le_bytes());
    for v in state {
        hasher.update(v.to_le_bytes());
    }
    hasher.update(parent_hash);
    hasher.update(confidence.to_le_bytes());
    hasher.update((provenance.len() as u64).to_le_bytes());
    for p in provenance {
        hasher.update(p.evidence_hash);
        hasher.update((p.source.len() as u64).to_le_bytes());
        hasher.update(p.source.as_bytes());
    }
    hasher.update([model_fingerprint.is_some() as u8]);
    if let Some(fp) = model_fingerprint {
        hasher.update(fp);
    }
    hasher.update([adapter_fingerprint.is_some() as u8]);
    if let Some(fp) = adapter_fingerprint {
        hasher.update(fp);
    }
    hasher.finalize().into()
}

/// Why an update or rollback was refused. Refusal, never silent coercion —
/// ADR-041 §17.14.
#[derive(Clone, Debug, PartialEq)]
pub enum ContextError {
    /// The candidate state exceeds [`ContextCapacity::max_dim`].
    CapacityExceeded { len: usize, max_dim: usize },
    /// The candidate state contained NaN or infinite values (ADR-041 §5.2).
    NonFinite,
    /// `confidence` was outside `[0.0, 1.0]` or non-finite.
    ConfidenceOutOfRange(f32),
    /// [`ContextState::rollback`] was called with no history to roll back to.
    EmptyHistory,
    /// [`ContextState::rollback_to`] was given a hash not present in history.
    UnknownCheckpoint([u8; 32]),
}

impl fmt::Display for ContextError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ContextError::CapacityExceeded { len, max_dim } => write!(
                f,
                "candidate state has {len} elements, exceeds max_dim {max_dim}"
            ),
            ContextError::NonFinite => write!(f, "candidate state contains NaN or Inf"),
            ContextError::ConfidenceOutOfRange(c) => {
                write!(f, "confidence {c} is not finite and in [0.0, 1.0]")
            }
            ContextError::EmptyHistory => write!(f, "no prior checkpoint to roll back to"),
            ContextError::UnknownCheckpoint(h) => {
                write!(f, "no checkpoint with result_hash {h:02x?} in history")
            }
        }
    }
}

impl std::error::Error for ContextError {}

/// The bounded recurrent task context S (ADR-041 §4.2).
///
/// Holds exactly one live checkpoint ([`ContextState::current`]) plus a
/// capped ring of prior checkpoints for rollback. See the crate-level
/// Invariant 1 note: this type has no method that authorizes anything —
/// it is a hash-chained log of state vectors and confidence scores.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ContextState {
    capacity: ContextCapacity,
    current: ContextCheckpoint,
    /// Oldest-first. Bounded to `capacity.max_history`.
    history: VecDeque<ContextCheckpoint>,
    next_sequence: u64,
}

impl ContextState {
    /// Start a fresh context: a zero vector of `dim` elements, sequence 0,
    /// all-zero `parent_hash`, confidence 1.0, no provenance.
    pub fn genesis(capacity: ContextCapacity, dim: usize) -> Result<Self, ContextError> {
        if dim > capacity.max_dim {
            return Err(ContextError::CapacityExceeded {
                len: dim,
                max_dim: capacity.max_dim,
            });
        }
        let state = vec![0.0f32; dim];
        let parent_hash = [0u8; 32];
        let confidence = 1.0f32;
        let provenance = Vec::new();
        let result_hash =
            checkpoint_hash(0, &state, &parent_hash, confidence, &provenance, None, None);
        let current = ContextCheckpoint {
            sequence: 0,
            state,
            parent_hash,
            result_hash,
            confidence,
            provenance,
            model_fingerprint: None,
            adapter_fingerprint: None,
        };
        Ok(Self {
            capacity,
            current,
            history: VecDeque::new(),
            next_sequence: 1,
        })
    }

    pub fn capacity(&self) -> ContextCapacity {
        self.capacity
    }

    pub fn current(&self) -> &ContextCheckpoint {
        &self.current
    }

    /// Prior checkpoints, oldest first. Never longer than `capacity.max_history`.
    pub fn history(&self) -> impl DoubleEndedIterator<Item = &ContextCheckpoint> {
        self.history.iter()
    }

    /// `S_t = U(S_{t-1}, D_t)`: absorb one evidence-derived state vector.
    ///
    /// Rejects (never truncates or coerces) a candidate that overflows
    /// `max_dim`, contains a non-finite value, or carries an out-of-range
    /// confidence — mirroring ADR-041 §5.2's `reject` branch. On success,
    /// the previous checkpoint is pushed onto the bounded rollback history
    /// and the new checkpoint becomes current.
    pub fn update(
        &mut self,
        new_state: Vec<f32>,
        confidence: f32,
        provenance: Vec<ProvenanceRef>,
        model_fingerprint: Option<[u8; 32]>,
        adapter_fingerprint: Option<[u8; 32]>,
    ) -> Result<&ContextCheckpoint, ContextError> {
        if new_state.len() > self.capacity.max_dim {
            return Err(ContextError::CapacityExceeded {
                len: new_state.len(),
                max_dim: self.capacity.max_dim,
            });
        }
        if !new_state.iter().all(|v| v.is_finite()) {
            return Err(ContextError::NonFinite);
        }
        if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
            return Err(ContextError::ConfidenceOutOfRange(confidence));
        }

        let sequence = self.next_sequence;
        let parent_hash = self.current.result_hash;
        let result_hash = checkpoint_hash(
            sequence,
            &new_state,
            &parent_hash,
            confidence,
            &provenance,
            model_fingerprint,
            adapter_fingerprint,
        );
        let next = ContextCheckpoint {
            sequence,
            state: new_state,
            parent_hash,
            result_hash,
            confidence,
            provenance,
            model_fingerprint,
            adapter_fingerprint,
        };

        let previous = std::mem::replace(&mut self.current, next);
        self.history.push_back(previous);
        if self.capacity.max_history == 0 {
            self.history.clear();
        } else {
            while self.history.len() > self.capacity.max_history {
                self.history.pop_front();
            }
        }
        self.next_sequence += 1;

        Ok(&self.current)
    }

    /// Undo the most recent update, restoring the immediately prior
    /// checkpoint as current. Errors if there is no history (e.g. still at
    /// genesis, or history was evicted by `max_history`).
    pub fn rollback(&mut self) -> Result<&ContextCheckpoint, ContextError> {
        let previous = self.history.pop_back().ok_or(ContextError::EmptyHistory)?;
        self.current = previous;
        Ok(&self.current)
    }

    /// Restore a specific earlier checkpoint by its `result_hash`, discarding
    /// every checkpoint newer than it (including whatever was current).
    /// A no-op (`Ok`) if `result_hash` already names the current checkpoint.
    pub fn rollback_to(
        &mut self,
        result_hash: [u8; 32],
    ) -> Result<&ContextCheckpoint, ContextError> {
        if self.current.result_hash == result_hash {
            return Ok(&self.current);
        }
        let idx = self
            .history
            .iter()
            .position(|c| c.result_hash == result_hash)
            .ok_or(ContextError::UnknownCheckpoint(result_hash))?;
        let mut at_and_after = self.history.split_off(idx);
        self.current = at_and_after
            .pop_front()
            .expect("idx returned by position() is always in bounds");
        // Everything still in `at_and_after` is newer than the restored
        // checkpoint and is discarded here.
        Ok(&self.current)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cap(max_dim: usize, max_history: usize) -> ContextCapacity {
        ContextCapacity {
            max_dim,
            max_history,
        }
    }

    fn evidence(byte: u8, source: &str) -> ProvenanceRef {
        ProvenanceRef {
            evidence_hash: [byte; 32],
            source: source.to_string(),
        }
    }

    #[test]
    fn genesis_is_zeroed_and_deterministic() {
        let a = ContextState::genesis(cap(4, 4), 3).unwrap();
        let b = ContextState::genesis(cap(4, 4), 3).unwrap();
        assert_eq!(a.current().state, vec![0.0, 0.0, 0.0]);
        assert_eq!(a.current().parent_hash, [0u8; 32]);
        assert_eq!(a.current().result_hash, b.current().result_hash);
    }

    #[test]
    fn genesis_rejects_oversized_dim() {
        let err = ContextState::genesis(cap(2, 4), 3).unwrap_err();
        assert_eq!(err, ContextError::CapacityExceeded { len: 3, max_dim: 2 });
    }

    #[test]
    fn update_chains_parent_hash_and_bumps_sequence() {
        let mut s = ContextState::genesis(cap(4, 4), 2).unwrap();
        let genesis_hash = s.current().result_hash;

        s.update(vec![1.0, 2.0], 0.9, vec![evidence(1, "obs")], None, None)
            .unwrap();
        assert_eq!(s.current().sequence, 1);
        assert_eq!(s.current().parent_hash, genesis_hash);
        assert_eq!(s.current().state, vec![1.0, 2.0]);

        let first_hash = s.current().result_hash;
        s.update(vec![3.0, 4.0], 0.8, vec![], None, None).unwrap();
        assert_eq!(s.current().sequence, 2);
        assert_eq!(s.current().parent_hash, first_hash);
    }

    #[test]
    fn update_rejects_capacity_overflow() {
        let mut s = ContextState::genesis(cap(2, 4), 2).unwrap();
        let err = s
            .update(vec![1.0, 2.0, 3.0], 0.5, vec![], None, None)
            .unwrap_err();
        assert_eq!(err, ContextError::CapacityExceeded { len: 3, max_dim: 2 });
        // Rejected update must not have mutated state.
        assert_eq!(s.current().sequence, 0);
    }

    #[test]
    fn update_rejects_non_finite() {
        let mut s = ContextState::genesis(cap(2, 4), 2).unwrap();
        let err = s
            .update(vec![f32::NAN, 1.0], 0.5, vec![], None, None)
            .unwrap_err();
        assert_eq!(err, ContextError::NonFinite);

        let err = s
            .update(vec![f32::INFINITY, 1.0], 0.5, vec![], None, None)
            .unwrap_err();
        assert_eq!(err, ContextError::NonFinite);
    }

    #[test]
    fn update_rejects_out_of_range_confidence() {
        let mut s = ContextState::genesis(cap(2, 4), 2).unwrap();
        assert_eq!(
            s.update(vec![1.0], 1.5, vec![], None, None).unwrap_err(),
            ContextError::ConfidenceOutOfRange(1.5)
        );
        assert_eq!(
            s.update(vec![1.0], -0.1, vec![], None, None).unwrap_err(),
            ContextError::ConfidenceOutOfRange(-0.1)
        );
        // NaN != NaN under PartialEq, so match the variant instead of the payload.
        assert!(matches!(
            s.update(vec![1.0], f32::NAN, vec![], None, None)
                .unwrap_err(),
            ContextError::ConfidenceOutOfRange(c) if c.is_nan()
        ));
    }

    #[test]
    fn history_is_bounded_and_evicts_oldest() {
        let mut s = ContextState::genesis(cap(4, 2), 1).unwrap();
        for i in 1..=5u32 {
            s.update(vec![i as f32], 1.0, vec![], None, None).unwrap();
        }
        assert_eq!(s.history().count(), 2);
        // The two retained are the two most recent prior checkpoints (seq 3, 4);
        // current is seq 5.
        let seqs: Vec<u64> = s.history().map(|c| c.sequence).collect();
        assert_eq!(seqs, vec![3, 4]);
        assert_eq!(s.current().sequence, 5);
    }

    #[test]
    fn zero_max_history_keeps_no_rollback_points() {
        let mut s = ContextState::genesis(cap(4, 0), 1).unwrap();
        s.update(vec![1.0], 1.0, vec![], None, None).unwrap();
        assert_eq!(s.history().count(), 0);
        assert_eq!(s.rollback().unwrap_err(), ContextError::EmptyHistory);
    }

    #[test]
    fn rollback_restores_previous_checkpoint() {
        let mut s = ContextState::genesis(cap(4, 4), 1).unwrap();
        let genesis_hash = s.current().result_hash;
        s.update(vec![1.0], 1.0, vec![], None, None).unwrap();
        s.update(vec![2.0], 1.0, vec![], None, None).unwrap();

        s.rollback().unwrap();
        assert_eq!(s.current().state, vec![1.0]);
        assert_eq!(s.current().sequence, 1);

        s.rollback().unwrap();
        assert_eq!(s.current().result_hash, genesis_hash);

        assert_eq!(s.rollback().unwrap_err(), ContextError::EmptyHistory);
    }

    #[test]
    fn rollback_to_discards_newer_checkpoints() {
        let mut s = ContextState::genesis(cap(4, 8), 1).unwrap();
        s.update(vec![1.0], 1.0, vec![], None, None).unwrap(); // seq 1
        let target_hash = s.current().result_hash;
        s.update(vec![2.0], 1.0, vec![], None, None).unwrap(); // seq 2
        s.update(vec![3.0], 1.0, vec![], None, None).unwrap(); // seq 3

        s.rollback_to(target_hash).unwrap();
        assert_eq!(s.current().sequence, 1);
        assert_eq!(s.current().state, vec![1.0]);
        // seq 2 and 3 are gone; only genesis remains in history.
        assert_eq!(s.history().count(), 1);
        assert_eq!(s.history().next().unwrap().sequence, 0);
    }

    #[test]
    fn rollback_to_unknown_hash_errors() {
        let mut s = ContextState::genesis(cap(4, 4), 1).unwrap();
        s.update(vec![1.0], 1.0, vec![], None, None).unwrap();
        assert_eq!(
            s.rollback_to([9u8; 32]).unwrap_err(),
            ContextError::UnknownCheckpoint([9u8; 32])
        );
    }

    #[test]
    fn rollback_to_current_is_a_noop() {
        let mut s = ContextState::genesis(cap(4, 4), 1).unwrap();
        s.update(vec![1.0], 1.0, vec![], None, None).unwrap();
        let hash = s.current().result_hash;
        s.rollback_to(hash).unwrap();
        assert_eq!(s.current().result_hash, hash);
    }

    #[test]
    fn same_update_sequence_is_bit_exact_reproducible() {
        let run = || {
            let mut s = ContextState::genesis(cap(8, 8), 3).unwrap();
            s.update(
                vec![1.0, 2.0, 3.0],
                0.9,
                vec![evidence(7, "x")],
                Some([1u8; 32]),
                None,
            )
            .unwrap();
            s.update(vec![4.0, 5.0, 6.0], 0.7, vec![], None, Some([2u8; 32]))
                .unwrap();
            s.current().result_hash
        };
        assert_eq!(run(), run());
    }
}
