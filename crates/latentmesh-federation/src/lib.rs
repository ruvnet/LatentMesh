//! `latentmesh-federation` — federated world models (ADR-017, implementing
//! ADR-007): each node maintains a local world model `W_i : (state, action) →
//! state'` and exchanges only the compatible portions of its learned
//! dynamics, never raw experience.
//!
//! ```text
//! node A learns rules ──┐
//! node B learns rules ──┼──► scoped TransitionRules over LMS1 Air envelopes
//! node C learns rules ──┘         │
//!                                 ▼
//!                     local-scope validation (decoy-controlled,
//!                     reusing ADR-003's permutation test)
//!                                 ▼
//!                     A + compatible portions of B and C
//! ```
//!
//! Three structural guarantees:
//! - `Private`-scoped rules cannot leave a node: the transmit encoder refuses
//!   them (multi-tenancy by construction, not caller discipline).
//! - A candidate rule is admitted only if it improves held-out local
//!   prediction versus its decoy controls (support-shuffled, scope-mismatch)
//!   with a significant sign-flip permutation p — the same statistical
//!   machinery as `latentmesh-gate::causal`.
//! - Transport rides the Air stack unchanged: CRC, replay window, bounded
//!   frames, and optional signatures come from ADR-010's envelope, not new
//!   code.

pub mod admission;
pub mod model;
pub mod rule;
pub mod transport;

pub use admission::{validate_candidate, AdmissionConfig, RuleVerdict};
pub use model::{Transition, WorldModel};
pub use rule::{FederationError, RuleScope, TransitionRule, MAX_RULE_BYTES};
pub use transport::{decode_rule_envelope, encode_rule_envelope, should_transmit};
