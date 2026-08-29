//! `ReasoningWorkspace` — the iterative latent workspace H (ADR-041 §4.3, §6).
//!
//! `H_{r+1} = F(H_r, S)`: each [`ReasoningWorkspace::step`] accepts the next
//! iterate (computed elsewhere — `F` is the model/runtime, not this crate)
//! and reports [`ConvergenceSignal`], the `||H_{r+1} - H_r||` measurement
//! ADR-041 §6.2 defines as `delta`. A (separately owned) budget controller
//! reads that signal to decide whether to keep iterating; this type never
//! decides that itself and never stops on convergence alone (ADR-041 §6.3:
//! "must not use latent convergence alone").
//!
//! H is ephemeral and holds only the *hash* of the context checkpoint it was
//! seeded from ([`ReasoningWorkspace::context_ref`]), never a live reference
//! to a [`crate::context::ContextState`]. That is what keeps S and H from
//! sharing mutable state — see the crate-level Invariant 1 note.

use std::fmt;

use serde::{Deserialize, Serialize};

/// Numerical floor for the denominator of the relative convergence term, so
/// a zero-norm workspace doesn't divide by zero (ADR-041 §6.2: `max(norm(H), tiny)`).
const EPSILON: f32 = 1e-8;

/// The stopping signal a budget controller reads after each iteration.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConvergenceSignal {
    /// The iteration index this signal was produced at (`r` after the step
    /// that produced `H_{r}`; starts at 1 after the first `step`).
    pub iteration: u32,
    /// `||H_{r+1} - H_r||` — the raw L2 distance between iterates.
    pub absolute: f32,
    /// `absolute / max(||H_r||, EPSILON)` — ADR-041 §6.2's `delta` term,
    /// scale-invariant to the workspace's own magnitude.
    pub relative: f32,
}

/// Why a workspace operation was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WorkspaceError {
    /// The proposed next iterate has a different dimension than the current one.
    DimensionMismatch { expected: usize, found: usize },
    /// The proposed next iterate (or the seed) contained NaN or Inf.
    NonFinite,
}

impl fmt::Display for WorkspaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WorkspaceError::DimensionMismatch { expected, found } => write!(
                f,
                "next iterate has {found} elements, workspace dimension is {expected}"
            ),
            WorkspaceError::NonFinite => write!(f, "iterate contains NaN or Inf"),
        }
    }
}

impl std::error::Error for WorkspaceError {}

/// The ephemeral latent workspace H (ADR-041 §4.3).
///
/// Not canonical memory and not authority (see the crate-level Invariant 1
/// note): this type is exactly one `Vec<f32>` iterate, an iteration count,
/// and the last measured [`ConvergenceSignal`]. It has no method that reads
/// or writes a [`crate::context::ContextState`] — only [`Self::context_ref`],
/// the content hash of the checkpoint the caller seeded it from.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReasoningWorkspace {
    context_ref: [u8; 32],
    state: Vec<f32>,
    iteration: u32,
    last_signal: Option<ConvergenceSignal>,
}

impl ReasoningWorkspace {
    /// `H_0 = E(query, S)`: seed a workspace from an already-computed
    /// initial iterate, tagged with the content hash of the `S` checkpoint
    /// it was derived from. `E` itself lives outside this crate.
    pub fn seed(context_ref: [u8; 32], initial: Vec<f32>) -> Result<Self, WorkspaceError> {
        if !initial.iter().all(|v| v.is_finite()) {
            return Err(WorkspaceError::NonFinite);
        }
        Ok(Self {
            context_ref,
            state: initial,
            iteration: 0,
            last_signal: None,
        })
    }

    /// Content hash of the [`crate::context::ContextState`] checkpoint this
    /// workspace was seeded from. A hash only, never a live handle.
    pub fn context_ref(&self) -> [u8; 32] {
        self.context_ref
    }

    pub fn state(&self) -> &[f32] {
        &self.state
    }

    pub fn dim(&self) -> usize {
        self.state.len()
    }

    pub fn iteration(&self) -> u32 {
        self.iteration
    }

    pub fn last_signal(&self) -> Option<ConvergenceSignal> {
        self.last_signal
    }

    /// Accept `H_{r+1}`, the result of applying `F(H_r, S)` elsewhere, and
    /// report the convergence signal the budget controller uses to decide
    /// whether to keep iterating. Rejects a dimension change or a non-finite
    /// iterate rather than silently coercing it (ADR-041 §17.14).
    pub fn step(&mut self, next: Vec<f32>) -> Result<ConvergenceSignal, WorkspaceError> {
        if next.len() != self.state.len() {
            return Err(WorkspaceError::DimensionMismatch {
                expected: self.state.len(),
                found: next.len(),
            });
        }
        if !next.iter().all(|v| v.is_finite()) {
            return Err(WorkspaceError::NonFinite);
        }

        let prev_norm = l2_norm(&self.state);
        let absolute = l2_distance(&self.state, &next);
        let relative = absolute / prev_norm.max(EPSILON);

        self.state = next;
        self.iteration += 1;
        let signal = ConvergenceSignal {
            iteration: self.iteration,
            absolute,
            relative,
        };
        self.last_signal = Some(signal);
        Ok(signal)
    }
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_rejects_non_finite() {
        assert_eq!(
            ReasoningWorkspace::seed([0u8; 32], vec![1.0, f32::NAN]).unwrap_err(),
            WorkspaceError::NonFinite
        );
    }

    #[test]
    fn seed_records_context_ref_and_zero_iteration() {
        let ctx_hash = [7u8; 32];
        let h = ReasoningWorkspace::seed(ctx_hash, vec![0.0, 0.0]).unwrap();
        assert_eq!(h.context_ref(), ctx_hash);
        assert_eq!(h.iteration(), 0);
        assert_eq!(h.last_signal(), None);
        assert_eq!(h.dim(), 2);
    }

    #[test]
    fn step_computes_exact_l2_distance() {
        let mut h = ReasoningWorkspace::seed([0u8; 32], vec![0.0, 0.0]).unwrap();
        let signal = h.step(vec![3.0, 4.0]).unwrap();
        assert_eq!(signal.iteration, 1);
        assert!((signal.absolute - 5.0).abs() < 1e-6);
        assert_eq!(h.state(), &[3.0, 4.0]);
        assert_eq!(h.iteration(), 1);
        assert_eq!(h.last_signal(), Some(signal));
    }

    #[test]
    fn step_relative_uses_epsilon_floor_when_prev_is_zero() {
        let mut h = ReasoningWorkspace::seed([0u8; 32], vec![0.0, 0.0]).unwrap();
        let signal = h.step(vec![3.0, 4.0]).unwrap();
        // prev_norm = 0 -> denominator floors at EPSILON.
        let expected_relative = 5.0 / EPSILON;
        assert!((signal.relative - expected_relative).abs() / expected_relative < 1e-3);
    }

    #[test]
    fn step_relative_scales_by_prev_norm_when_nonzero() {
        let mut h = ReasoningWorkspace::seed([0u8; 32], vec![3.0, 4.0]).unwrap(); // norm 5
        let signal = h.step(vec![3.0, 4.0 + 5.0]).unwrap(); // moved by 5 along one axis
        assert!((signal.absolute - 5.0).abs() < 1e-6);
        assert!((signal.relative - 1.0).abs() < 1e-6);
    }

    #[test]
    fn step_signals_zero_on_exact_convergence() {
        let mut h = ReasoningWorkspace::seed([0u8; 32], vec![1.0, 2.0, 3.0]).unwrap();
        let signal = h.step(vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(signal.absolute, 0.0);
        assert_eq!(signal.relative, 0.0);
    }

    #[test]
    fn step_rejects_dimension_change() {
        let mut h = ReasoningWorkspace::seed([0u8; 32], vec![1.0, 2.0]).unwrap();
        assert_eq!(
            h.step(vec![1.0]).unwrap_err(),
            WorkspaceError::DimensionMismatch {
                expected: 2,
                found: 1
            }
        );
        // Rejected step must not have mutated state or iteration.
        assert_eq!(h.state(), &[1.0, 2.0]);
        assert_eq!(h.iteration(), 0);
    }

    #[test]
    fn step_rejects_non_finite() {
        let mut h = ReasoningWorkspace::seed([0u8; 32], vec![1.0, 2.0]).unwrap();
        assert_eq!(
            h.step(vec![1.0, f32::INFINITY]).unwrap_err(),
            WorkspaceError::NonFinite
        );
        assert_eq!(h.iteration(), 0);
    }

    #[test]
    fn iteration_advances_monotonically_across_steps() {
        let mut h = ReasoningWorkspace::seed([0u8; 32], vec![0.0]).unwrap();
        for expected in 1..=5u32 {
            let signal = h.step(vec![expected as f32]).unwrap();
            assert_eq!(signal.iteration, expected);
            assert_eq!(h.iteration(), expected);
        }
    }

    #[test]
    fn same_step_sequence_is_bit_exact_reproducible() {
        let run = || {
            let mut h = ReasoningWorkspace::seed([3u8; 32], vec![0.1, 0.2, 0.3]).unwrap();
            h.step(vec![0.4, 0.5, 0.6]).unwrap();
            h.step(vec![0.41, 0.51, 0.61]).unwrap().absolute
        };
        assert_eq!(run(), run());
    }
}
