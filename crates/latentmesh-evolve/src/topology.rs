//! The evolved object: agents, edges, per-edge encoding and authority, and
//! the constitution caps mutations must respect.

use latentmesh_core::{Authority, Encoding};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub type AgentId = u8;

/// One directed communication edge `from → to`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Edge {
    pub from: AgentId,
    pub to: AgentId,
    /// Wire encoding — Darwin may compress a low-value edge harder
    /// (ADR-006 §bandwidth).
    pub encoding: Encoding,
    /// The authority this edge currently operates at.
    pub authority: Authority,
    /// Governance-mandatory: measured near-zero cognitive value does NOT make
    /// this edge removable (control vs. cognition, ADR-006).
    pub mandatory: bool,
}

/// A candidate communication graph.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Topology {
    /// Agents currently active (an inactive agent costs nothing but can be
    /// reactivated by mutation).
    pub active_agents: Vec<AgentId>,
    /// Edges keyed by `(from, to)` for deterministic iteration.
    pub edges: BTreeMap<(AgentId, AgentId), Edge>,
    /// The constitution: per-edge authority caps mutations may never exceed.
    /// Missing entry means `ObserveOnly` (default-deny, matching the gate).
    pub authority_caps: BTreeMap<(AgentId, AgentId), Authority>,
}

impl Topology {
    /// A fully-connected starting topology over `agents`, everything at
    /// `ContextInject`/`F32` — the deliberately wasteful graph evolution is
    /// supposed to prune.
    pub fn fully_connected(agents: &[AgentId], cap: Authority) -> Self {
        let mut topology = Topology {
            active_agents: agents.to_vec(),
            ..Default::default()
        };
        for &from in agents {
            for &to in agents {
                if from == to {
                    continue;
                }
                topology.authority_caps.insert((from, to), cap);
                topology.edges.insert(
                    (from, to),
                    Edge {
                        from,
                        to,
                        encoding: Encoding::F32,
                        authority: Authority::ContextInject.min(cap),
                        mandatory: false,
                    },
                );
            }
        }
        topology
    }

    pub fn cap_for(&self, from: AgentId, to: AgentId) -> Authority {
        self.authority_caps
            .get(&(from, to))
            .copied()
            .unwrap_or(Authority::ObserveOnly)
    }

    /// Compute-cost proxy: one unit per active agent plus per-edge bytes at a
    /// nominal 512-dim payload. A deterministic stand-in for real compute —
    /// the receipt labels it as such.
    pub fn compute_proxy(&self) -> f64 {
        let agent_cost = self.active_agents.len() as f64 * 100.0;
        let edge_cost: f64 = self
            .edges
            .values()
            .map(|e| (512 * e.encoding.bytes_per_element()) as f64 / 100.0)
            .sum();
        agent_cost + edge_cost
    }

    /// Edges originating from or arriving at inactive agents carry nothing.
    pub fn live_edges(&self) -> impl Iterator<Item = &Edge> {
        self.edges
            .values()
            .filter(|e| self.active_agents.contains(&e.from) && self.active_agents.contains(&e.to))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fully_connected_topology_has_n_times_n_minus_one_edges() {
        let t = Topology::fully_connected(&[0, 1, 2, 3], Authority::LatentPrefix);
        assert_eq!(t.edges.len(), 12);
        assert!(t.compute_proxy() > 0.0);
    }

    #[test]
    fn missing_cap_defaults_to_observe_only() {
        let t = Topology::default();
        assert_eq!(t.cap_for(1, 2), Authority::ObserveOnly);
    }

    #[test]
    fn int8_edges_cost_less_than_f32_edges() {
        let mut a = Topology::fully_connected(&[0, 1], Authority::ContextInject);
        let base = a.compute_proxy();
        for edge in a.edges.values_mut() {
            edge.encoding = Encoding::Int8;
        }
        assert!(a.compute_proxy() < base);
    }
}
