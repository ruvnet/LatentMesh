//! The trajectory schema and the fidelity continuum (ADR-005). Fidelity is a
//! property *of a record*, moves only downward (a lossy step is never
//! silently undone), and each downward step carries its measured
//! reconstruction error.

use serde::{Deserialize, Serialize};

/// Where a record sits on ADR-005's continuum. Ordered: `Raw` is the most
/// expensive, `Rule` the cheapest.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Fidelity {
    /// Full `f32` latent, bounded retention, reserved for causally-verified
    /// trajectories.
    Raw,
    /// Quantized latent (`F16` or `Int8` via `latentmesh-core`), with the
    /// measured reconstruction error recorded.
    Compressed,
    /// A centroid over a problem family; members' ids are retained as
    /// lineage.
    Prototype,
    /// A symbolic rule; retrievable without touching the vector index.
    Rule,
}

/// One stored trajectory: ADR-005's `M = {z_t, r_t, c_t, a_t, o_t}` plus the
/// bookkeeping the continuum needs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrajectoryRecord {
    /// Stable id (usually the originating frame's id).
    pub id: String,
    /// `z` — the latent state at the stored fidelity.
    pub latent: Vec<f32>,
    /// `r` — scalar reward/outcome value for the trajectory.
    pub reward: f32,
    /// `c` — content hash of the producing context (never raw text).
    pub context_hash: String,
    /// `a` — the action taken.
    pub action: String,
    /// `o` — the observed outcome.
    pub outcome: String,
    /// Measured causal value (ADR-003's ΔV) that admitted this record.
    pub causal_value: f64,
    pub fidelity: Fidelity,
    /// Mean absolute reconstruction error accumulated across lossy steps
    /// (0.0 at `Raw`).
    pub reconstruction_error: f32,
    /// Lineage: record ids this record was compressed/folded from.
    pub parents: Vec<String>,
}

/// Procedural memory (ADR-005 §skills): a problem family mapped to the agent
/// sequence that solved it — restorable as a topology warm start (ADR-006).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TopologyRecord {
    /// Problem-family label the embedding was derived from.
    pub family: String,
    /// The successful agent pipeline, in execution order.
    pub agent_sequence: Vec<String>,
    /// How many times this topology has been recalled and reused
    /// successfully.
    pub successful_uses: u32,
}

/// The cheapest tier: an explicit rule promoted from a reused prototype.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SymbolicRule {
    pub family: String,
    /// Human-readable rule body ("family X → pipeline Y").
    pub rule: String,
    /// Prototype record id this rule was promoted from.
    pub promoted_from: String,
    pub uses: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fidelity_orders_raw_cheapest_last() {
        assert!(Fidelity::Raw < Fidelity::Compressed);
        assert!(Fidelity::Compressed < Fidelity::Prototype);
        assert!(Fidelity::Prototype < Fidelity::Rule);
    }

    #[test]
    fn records_serialize_round_trip() {
        let r = TrajectoryRecord {
            id: "t1".into(),
            latent: vec![0.1, 0.2],
            reward: 1.0,
            context_hash: "c".into(),
            action: "plan".into(),
            outcome: "ok".into(),
            causal_value: 0.8,
            fidelity: Fidelity::Raw,
            reconstruction_error: 0.0,
            parents: vec![],
        };
        let json = serde_json::to_string(&r).unwrap();
        let back: TrajectoryRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(back, r);
    }
}
