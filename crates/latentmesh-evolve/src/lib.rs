//! `latentmesh-evolve` — the MetaHarness Darwin loop (ADR-018, implementing
//! ADR-006): the communication topology is an evolved object whose fitness is
//! measured causal edge value, not token traffic.
//!
//! ```text
//! G_t = (A, E, Z, M, P)          (ADR-009 notation)
//! G*  = argmax_G [ Quality(G) − λ·Cost(G) − μ·Latency(G) − ρ·Risk(G) ]
//! ```
//!
//! Non-negotiables carried from the ADRs:
//! - **Quality is causal**: an edge contributes only if its ΔV survives the
//!   four-control test (`latentmesh-gate::causal::verify_edge`) in the
//!   evaluation environment; apparent correlation contributes zero.
//! - **Authority never expands**: a mutation that would raise any edge above
//!   its constitution cap is rejected by the applier before evaluation.
//! - **Governance-mandatory edges are never auto-removed**, even at ΔV ≈ 0
//!   (control vs. cognition, ADR-006).
//! - **Determinism**: seeded RNG, frozen synthetic environment, receipts that
//!   label the evidence "deterministic simulation".

pub mod darwin;
pub mod env;
pub mod fitness;
pub mod receipt;
pub mod topology;

pub use darwin::{evolve, DarwinConfig, EvolveOutcome, Mutation, MutationError};
pub use env::SyntheticEnv;
pub use fitness::{evaluate, FitnessBreakdown, FitnessWeights, VerificationCache};
pub use receipt::{acceptance_check, AcceptanceReport, Receipt, RECEIPT_SCHEMA};
pub use topology::{AgentId, Edge, Topology};
