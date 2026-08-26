//! `latentmesh-memory` — persistent latent memory (ADR-016, implementing
//! ADR-005): don't throw away successful latent trajectories. A trajectory is
//! `M = {z, r, c, a, o}` (latent state, reward, context, action, outcome),
//! stored at an explicit fidelity level on the continuum
//!
//! ```text
//! raw latent trajectory → compressed latent trajectory → semantic prototype → symbolic rule
//! ```
//!
//! with three commitments carried over from the ADRs:
//!
//! - **Admission is causal** (ADR-003): only trajectories whose measured
//!   causal value clears the configured floor may occupy the expensive Raw
//!   tier; everything else enters compressed.
//! - **Compression is measured, not silent** (CrystalMem's hysteresis
//!   finding): each fidelity step re-encodes through `latentmesh-core`'s real
//!   quantizers and records the reconstruction error in the record.
//! - **Recall returns candidates, not authority** (ADR-008): a recalled
//!   trajectory re-enters the mesh at `ObserveOnly` like any other inbound
//!   state; nothing in this crate grants execution rights.

pub mod compress;
pub mod record;
pub mod store;

#[cfg(feature = "ruvector")]
pub mod ruvector;

pub use compress::{compress_latent, CompressionOutcome};
pub use record::{Fidelity, SymbolicRule, TopologyRecord, TrajectoryRecord};
pub use store::{InMemoryStore, LatentMemory, MemoryConfig, MemoryError, Recalled};

#[cfg(feature = "ruvector")]
pub use ruvector::RuVectorStore;
