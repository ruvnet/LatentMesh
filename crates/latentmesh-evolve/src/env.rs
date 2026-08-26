//! The frozen evaluation environment (ADR-018): ten agents with deliberately
//! redundant capabilities, a hidden true-contribution structure known to the
//! environment and hidden from the optimizer, and a deterministic generator
//! of per-edge [`EdgeTrial`]s so causal verification runs against honest
//! paired samples. Everything derives from the seed — same seed, same world.

use latentmesh_gate::causal::EdgeTrial;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::topology::AgentId;

/// The synthetic ten-agent world of ADR-006's acceptance test.
#[derive(Clone, Debug)]
pub struct SyntheticEnv {
    seed: u64,
    /// Hidden ground truth: `true_value[from][to]` is the real marginal task
    /// value of the edge; redundant agents have all-zero rows.
    true_value: [[f64; 10]; 10],
    /// Edges whose purpose is control, not cognition — near-zero value but
    /// governance-mandatory.
    mandatory_edges: Vec<(AgentId, AgentId)>,
}

impl SyntheticEnv {
    pub const AGENTS: u8 = 10;

    /// Build the frozen world. Agents 0–5 form a genuine pipeline with a few
    /// useful cross-links; agents 6–9 are deliberately redundant (their edges
    /// carry no true value). Edge `9 → 1` is the mandatory "security → coder"
    /// style control edge.
    pub fn new(seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed ^ 0x454e_5631);
        let mut true_value = [[0.0f64; 10]; 10];
        let useful: [(usize, usize, f64); 7] = [
            (0, 1, 1.2), // planner → coder
            (2, 1, 0.7), // researcher → coder
            (3, 0, 0.9), // memory → planner
            (4, 1, 0.6), // verifier → coder
            (1, 4, 0.5), // coder → verifier
            (0, 4, 0.4), // planner → verifier
            (2, 0, 0.3), // researcher → planner
        ];
        for (from, to, value) in useful {
            // Small seeded jitter so no two worlds are byte-identical.
            true_value[from][to] = value + rng.gen_range(-0.05..0.05);
        }
        SyntheticEnv {
            seed,
            true_value,
            mandatory_edges: vec![(9, 1)],
        }
    }

    pub fn agents(&self) -> Vec<AgentId> {
        (0..Self::AGENTS).collect()
    }

    pub fn mandatory_edges(&self) -> &[(AgentId, AgentId)] {
        &self.mandatory_edges
    }

    /// Hidden ground-truth value of an edge (test-only introspection).
    pub fn true_value(&self, from: AgentId, to: AgentId) -> f64 {
        self.true_value[from as usize][to as usize]
    }

    /// Deterministic paired trials for causal verification of one edge: the
    /// `real` condition reflects the hidden value plus noise; every control
    /// is noise around the baseline. A redundant edge's `real` is
    /// indistinguishable from its controls — exactly what the four-control
    /// test must catch.
    pub fn edge_trial(&self, from: AgentId, to: AgentId, trials: usize) -> EdgeTrial {
        let mut rng = StdRng::seed_from_u64(
            self.seed
                .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                .wrapping_add((u64::from(from) << 8) | u64::from(to)),
        );
        let value = self.true_value[from as usize][to as usize];
        let noise = 0.15;
        let mut sample = |offset: f64| -> Vec<f64> {
            (0..trials)
                .map(|_| offset + rng.gen_range(-noise..noise))
                .collect()
        };
        EdgeTrial {
            real: sample(0.5 + value),
            zero: sample(0.5),
            random: sample(0.5),
            mismatched: sample(0.5),
            self_generated: sample(0.5),
            text_equivalent: sample(0.5 + value * 0.15),
        }
    }

    /// Task success proxy for a topology: the sum of hidden values of live
    /// edges, saturating — adding redundant edges never raises it.
    pub fn task_success(&self, live_edges: impl Iterator<Item = (AgentId, AgentId)>) -> f64 {
        let total: f64 = live_edges
            .map(|(f, t)| self.true_value[f as usize][t as usize])
            .sum();
        total.min(5.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use latentmesh_gate::causal::{verify_edge, EdgeVerdict};

    #[test]
    fn same_seed_same_world() {
        let a = SyntheticEnv::new(7);
        let b = SyntheticEnv::new(7);
        let ta = a.edge_trial(0, 1, 16);
        let tb = b.edge_trial(0, 1, 16);
        assert_eq!(ta.real, tb.real);
        assert_eq!(ta.zero, tb.zero);
        assert_eq!(ta.text_equivalent, tb.text_equivalent);
    }

    #[test]
    fn useful_edges_pass_verification_redundant_edges_fail() {
        let env = SyntheticEnv::new(7);
        let useful = env.edge_trial(0, 1, 32);
        assert!(matches!(
            verify_edge(&useful, 0.05, 1000, 7),
            EdgeVerdict::Admit { .. }
        ));
        let redundant = env.edge_trial(6, 7, 32);
        assert!(matches!(
            verify_edge(&redundant, 0.05, 1000, 7),
            EdgeVerdict::Reject { .. }
        ));
    }
}
