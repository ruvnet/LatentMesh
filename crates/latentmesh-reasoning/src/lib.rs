//! `latentmesh-reasoning` — Phase 1 deterministic scaffold for ADR-041
//! ("Distributed recurrent latent reasoning over LatentMesh"). Per §20 Phase
//! 1, this crate exists to provide in-memory reference algorithms and
//! deterministic test fixtures for the pieces ADR-041 specifies — recurrent
//! context, latent workspace, the reasoning-delta envelope/codec, and the
//! adaptive reasoning-budget controller — before any of it is wired to a real
//! recurrent model (Phase 3) or a network transport (Phase 4+).
//!
//! Each module here is owned independently and is pure/zero-I/O by
//! construction: no clock, no thread, no model weights. That is not a style
//! preference — ADR-041 §17 requires remote/derived latent state to be
//! governed and replayable, which is only possible if the pieces that decide
//! *how much* reasoning to spend, and *whether* to admit a reasoning delta,
//! are themselves deterministic and independently testable.
//!
//! # Invariant 1 — latent state is non-authoritative
//!
//! Per ADR-041 §4 and §17: [`context::ContextState`] (S) and
//! [`workspace::ReasoningWorkspace`] (H) cannot authorize an action. Neither
//! type has a method that returns a capability, executes anything, or
//! converts into an action type, and this crate has no dependency on
//! `latentmesh-gate` (or any capability/authority/action type), so no such
//! method could exist here without first adding one. Both are inert data:
//! hashes, floats, and provenance tags. Turning either into a governed
//! action is entirely the job of the gate crate, which sits above this one,
//! never below.
//!
//! `ReasoningWorkspace` never holds a live reference to `ContextState` — it
//! only carries the content hash of the checkpoint it was seeded from
//! ([`workspace::ReasoningWorkspace::context_ref`]) — so S and H cannot
//! share mutable state (ADR-041 §4.2/§4.3: S absorbs evidence, H computes;
//! that split is enforced by the types, not just documented).
//!
//! Currently implemented:
//! - [`context`] — the bounded recurrent task context S (ADR-041 §4.2, §5
//!   Algorithm 1).
//! - [`workspace`] — the iterative latent workspace H (ADR-041 §4.3, §6
//!   Algorithm 2), exposing the `||H_{r+1} - H_r||` convergence signal.
//!
//! Landing next, each as its own module file: `budget` (adaptive
//! reasoning-depth controller, §6 Algorithm 2 / §8 Algorithm 4) and
//! `envelope`/`compat` (the reasoning-delta envelope and the compatibility
//! handshake, §11–§13). **Module declarations are added here only once the
//! corresponding file exists** — declaring a module ahead of its file breaks
//! the whole crate with E0583 for every sibling, which is exactly what
//! happened during concurrent assembly.

pub mod context;
pub mod workspace;

pub use context::{ContextCapacity, ContextCheckpoint, ContextError, ContextState, ProvenanceRef};
pub use workspace::{ConvergenceSignal, ReasoningWorkspace, WorkspaceError};
