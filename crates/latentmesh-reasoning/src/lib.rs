//! `latentmesh-reasoning` — Phase 1 deterministic scaffold for
//! [ADR-041](../../../docs/adr/041-distributed-recurrent-latent-reasoning.md),
//! "Distributed recurrent latent reasoning over LatentMesh".
//!
//! In-memory reference algorithms and deterministic fixtures for the pieces
//! ADR-041 specifies — recurrent context, latent workspace, the
//! reasoning-delta envelope/codec, the compatibility handshake, and the
//! adaptive reasoning-budget controller — before any of it is wired to a real
//! recurrent model or a network transport.
//!
//! Every module is pure and zero-I/O: no clock, no thread, no model weights,
//! no RNG without an explicit seed. That is not style. ADR-041 requires
//! remote/derived latent state to be governed and replayable, which is only
//! possible if the pieces deciding *how much* to reason and *whether* to admit
//! a delta are themselves deterministic and independently testable.
//!
//! # Invariant 1 — latent state is NON-AUTHORITATIVE
//!
//! [`context::ContextState`] (S) and [`workspace::ReasoningWorkspace`] (H)
//! **cannot authorize an action.** No type here has a method that returns a
//! capability, executes anything, or converts into an action type — and this
//! crate has **zero dependency on `latentmesh-gate`** or any authority type,
//! so no such method could be added without first adding that dependency and
//! being noticed in review. Everything here is inert data: hashes, floats,
//! and provenance tags. Turning any of it into a governed action is the job
//! of the gate crate, which sits *above* this one, never below.
//!
//! `ReasoningWorkspace` never holds a live reference to `ContextState` — only
//! the content hash of the checkpoint it was seeded from
//! ([`workspace::ReasoningWorkspace::context_ref`]) — so S and H cannot share
//! mutable state. ADR-041 §4.2/§4.3's split (S absorbs evidence, H computes)
//! is enforced by the types, not merely documented.
//!
//! # Measured finding: this envelope does not fit a mesh frame
//!
//! [`envelope::FIXED_HEADER_BYTES`] is **282 bytes** of fixed metadata before
//! any payload. Against the verified Air budget (~106 B unsigned / ~42 B
//! signed per fragment, from a 211-byte usable Meshtastic MTU minus LMS1/LMAD
//! tax), the header **alone** needs **3 fragments unsigned and 9 signed**.
//! Asserted, not assumed, by
//! `envelope::tests::fixed_header_overhead_exceeds_a_single_air_fragment`.
//! See `docs/research/049-adjacent-areas-survey.md` for the duty-cycle
//! consequence.
//!
//! # Module ownership
//!
//! Coordinate by type name; do not edit another module's file. `lib.rs` and
//! `Cargo.toml` are owned by the integrator — module declarations are added
//! only once the corresponding file exists on disk, because declaring a
//! module ahead of its file breaks the whole crate with E0583 for every
//! sibling.

pub mod budget;
pub mod compat;
pub mod context;
pub mod envelope;
pub mod workspace;

pub use context::{ContextCapacity, ContextCheckpoint, ContextError, ContextState, ProvenanceRef};
pub use workspace::{ConvergenceSignal, ReasoningWorkspace, WorkspaceError};
