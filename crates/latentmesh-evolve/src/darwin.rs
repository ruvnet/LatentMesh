//! The Darwin loop: seeded mutation, elitist selection, and the
//! authority-never-expands invariant enforced in the applier — a mutation
//! that would exceed a constitution cap is rejected before evaluation, so no
//! amount of fitness can buy authority (ADR-006 / ADR-008).

use crate::env::SyntheticEnv;
use crate::fitness::{evaluate, FitnessBreakdown, FitnessWeights};
use crate::topology::{AgentId, Edge, Topology};
use latentmesh_core::{Authority, Encoding};
use latentmesh_memory::{InMemoryStore, TopologyRecord};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

/// The mutation vocabulary. Darwin proposes; the applier enforces the
/// constitution.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Mutation {
    AddEdge {
        from: AgentId,
        to: AgentId,
    },
    RemoveEdge {
        from: AgentId,
        to: AgentId,
    },
    RaiseAuthority {
        from: AgentId,
        to: AgentId,
    },
    LowerAuthority {
        from: AgentId,
        to: AgentId,
    },
    ChangeEncoding {
        from: AgentId,
        to: AgentId,
        encoding: Encoding,
    },
    DeactivateAgent {
        agent: AgentId,
    },
    ReactivateAgent {
        agent: AgentId,
    },
}

/// Why the applier refused a proposal.
#[derive(Clone, Debug, PartialEq)]
pub enum MutationError {
    /// The proposal would raise an edge above its constitution cap —
    /// authority never expands, whatever the fitness gain.
    AuthorityExceedsCap {
        requested: Authority,
        cap: Authority,
    },
    /// Removing or deactivating would drop a governance-mandatory edge.
    MandatoryEdge { from: AgentId, to: AgentId },
    /// Self-edges and unknown agents are structurally invalid.
    Invalid(&'static str),
}

fn raise(a: Authority) -> Option<Authority> {
    match a {
        Authority::ObserveOnly => Some(Authority::ContextInject),
        Authority::ContextInject => Some(Authority::LatentPrefix),
        Authority::LatentPrefix => Some(Authority::ActionInfluencing),
        Authority::ActionInfluencing => None,
    }
}

fn lower(a: Authority) -> Option<Authority> {
    match a {
        Authority::ObserveOnly => None,
        Authority::ContextInject => Some(Authority::ObserveOnly),
        Authority::LatentPrefix => Some(Authority::ContextInject),
        Authority::ActionInfluencing => Some(Authority::LatentPrefix),
    }
}

/// Apply one mutation, or explain the refusal. Never panics on a hostile
/// proposal.
pub fn apply_mutation(topology: &mut Topology, mutation: Mutation) -> Result<(), MutationError> {
    match mutation {
        Mutation::AddEdge { from, to } => {
            if from == to {
                return Err(MutationError::Invalid("self edge"));
            }
            let cap = topology.cap_for(from, to);
            topology.edges.entry((from, to)).or_insert(Edge {
                from,
                to,
                encoding: Encoding::Int8,
                authority: Authority::ObserveOnly.min(cap),
                mandatory: false,
            });
            Ok(())
        }
        Mutation::RemoveEdge { from, to } => {
            if let Some(edge) = topology.edges.get(&(from, to)) {
                if edge.mandatory {
                    return Err(MutationError::MandatoryEdge { from, to });
                }
                topology.edges.remove(&(from, to));
            }
            Ok(())
        }
        Mutation::RaiseAuthority { from, to } => {
            let cap = topology.cap_for(from, to);
            let edge = topology
                .edges
                .get_mut(&(from, to))
                .ok_or(MutationError::Invalid("no such edge"))?;
            match raise(edge.authority) {
                Some(next) if next <= cap => {
                    edge.authority = next;
                    Ok(())
                }
                Some(next) => Err(MutationError::AuthorityExceedsCap {
                    requested: next,
                    cap,
                }),
                None => Err(MutationError::AuthorityExceedsCap {
                    requested: Authority::ActionInfluencing,
                    cap,
                }),
            }
        }
        Mutation::LowerAuthority { from, to } => {
            let edge = topology
                .edges
                .get_mut(&(from, to))
                .ok_or(MutationError::Invalid("no such edge"))?;
            if let Some(prev) = lower(edge.authority) {
                edge.authority = prev;
            }
            Ok(())
        }
        Mutation::ChangeEncoding { from, to, encoding } => {
            let edge = topology
                .edges
                .get_mut(&(from, to))
                .ok_or(MutationError::Invalid("no such edge"))?;
            edge.encoding = encoding;
            Ok(())
        }
        Mutation::DeactivateAgent { agent } => {
            let anchors_mandatory = topology
                .edges
                .values()
                .any(|e| e.mandatory && (e.from == agent || e.to == agent));
            if anchors_mandatory {
                return Err(MutationError::MandatoryEdge {
                    from: agent,
                    to: agent,
                });
            }
            topology.active_agents.retain(|a| *a != agent);
            Ok(())
        }
        Mutation::ReactivateAgent { agent } => {
            if !topology.active_agents.contains(&agent) {
                topology.active_agents.push(agent);
                topology.active_agents.sort_unstable();
            }
            Ok(())
        }
    }
}

/// Loop parameters. Everything defaults to values that converge in seconds.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct DarwinConfig {
    pub seed: u64,
    pub generations: usize,
    pub population: usize,
    pub mutations_per_child: usize,
    pub weights: FitnessWeights,
}

impl Default for DarwinConfig {
    fn default() -> Self {
        DarwinConfig {
            seed: 0x4441_5257,
            generations: 40,
            population: 12,
            mutations_per_child: 3,
            weights: FitnessWeights::default(),
        }
    }
}

/// The loop's result: the surviving topology, its evaluation, and the audit
/// counters the receipt reports.
#[derive(Clone, Debug)]
pub struct EvolveOutcome {
    pub initial: FitnessBreakdown,
    pub best: FitnessBreakdown,
    pub best_topology: Topology,
    pub generations_run: usize,
    pub mutations_proposed: usize,
    pub mutations_rejected_by_constitution: usize,
    pub total_verification_evaluations: usize,
    /// True when the starting population was seeded from procedural memory.
    pub warm_started: bool,
}

/// Sample one mutation. Edge-targeted operations pick from the topology's
/// *existing* edges (a proposal against a random absent pair is wasted
/// search); the distribution deliberately favors pruning and compression —
/// the direction ADR-006's objective rewards — while keeping additive moves
/// available so the search is not one-way.
fn random_mutation(rng: &mut StdRng, agents: &[AgentId], topology: &Topology) -> Mutation {
    let pick_agent = |rng: &mut StdRng| agents[rng.gen_range(0..agents.len())];
    let existing: Vec<(AgentId, AgentId)> = topology.edges.keys().copied().collect();
    let pick_edge = |rng: &mut StdRng, existing: &[(AgentId, AgentId)]| {
        existing[rng.gen_range(0..existing.len())]
    };
    if existing.is_empty() {
        return Mutation::AddEdge {
            from: pick_agent(rng),
            to: pick_agent(rng),
        };
    }
    match rng.gen_range(0..10u8) {
        0 => Mutation::AddEdge {
            from: pick_agent(rng),
            to: pick_agent(rng),
        },
        1..=3 => {
            let (from, to) = pick_edge(rng, &existing);
            Mutation::RemoveEdge { from, to }
        }
        4 => {
            let (from, to) = pick_edge(rng, &existing);
            Mutation::RaiseAuthority { from, to }
        }
        5 => {
            let (from, to) = pick_edge(rng, &existing);
            Mutation::LowerAuthority { from, to }
        }
        6..=7 => {
            let (from, to) = pick_edge(rng, &existing);
            let encoding = *[Encoding::F32, Encoding::F16, Encoding::Int8]
                .choose(rng)
                .unwrap_or(&Encoding::Int8);
            Mutation::ChangeEncoding { from, to, encoding }
        }
        8 => Mutation::DeactivateAgent {
            agent: pick_agent(rng),
        },
        _ => Mutation::ReactivateAgent {
            agent: pick_agent(rng),
        },
    }
}

/// Run the deterministic Darwin loop. When `memory` holds a topology record
/// for `problem_family`, the initial population is seeded from it (ADR-005/
/// ADR-006 warm start): remembered agents stay active, everything else starts
/// deactivated.
pub fn evolve(
    env: &SyntheticEnv,
    config: &DarwinConfig,
    memory: Option<(&InMemoryStore, &str)>,
) -> EvolveOutcome {
    let agents = env.agents();
    let mut base = Topology::fully_connected(&agents, Authority::LatentPrefix);
    for &(from, to) in env.mandatory_edges() {
        if let Some(edge) = base.edges.get_mut(&(from, to)) {
            edge.mandatory = true;
        }
    }

    let mut warm_started = false;
    if let Some((store, family)) = memory {
        if let Some(record) = store.recall_topology(family) {
            let keep: Vec<AgentId> = record
                .agent_sequence
                .iter()
                .filter_map(|name| name.strip_prefix("agent-"))
                .filter_map(|n| n.parse::<AgentId>().ok())
                .collect();
            if !keep.is_empty() {
                let mandatory_agents: Vec<AgentId> = env
                    .mandatory_edges()
                    .iter()
                    .flat_map(|&(f, t)| [f, t])
                    .collect();
                base.active_agents
                    .retain(|a| keep.contains(a) || mandatory_agents.contains(a));
                warm_started = true;
            }
        }
    }

    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut cache = crate::fitness::VerificationCache::new();
    let initial = evaluate(&base, env, &config.weights, config.seed, &mut cache);
    let mut best_topology = base.clone();
    let mut best = initial.clone();
    let mut mutations_proposed = 0usize;
    let mut mutations_rejected = 0usize;
    let mut total_verifications = initial.verification_evaluations;

    for _generation in 0..config.generations {
        let mut children: Vec<(Topology, FitnessBreakdown)> = Vec::new();
        for _child_index in 0..config.population {
            let mut child = best_topology.clone();
            for _ in 0..config.mutations_per_child {
                let mutation = random_mutation(&mut rng, &agents, &child);
                mutations_proposed += 1;
                if apply_mutation(&mut child, mutation).is_err() {
                    mutations_rejected += 1;
                }
            }
            let breakdown = evaluate(&child, env, &config.weights, config.seed, &mut cache);
            total_verifications += breakdown.verification_evaluations;
            children.push((child, breakdown));
        }
        // Elitist: the best child replaces the incumbent only if it wins.
        if let Some((child, breakdown)) = children.into_iter().max_by(|a, b| {
            a.1.fitness
                .partial_cmp(&b.1.fitness)
                .unwrap_or(core::cmp::Ordering::Equal)
        }) {
            if breakdown.fitness > best.fitness {
                best_topology = child;
                best = breakdown;
            }
        }
    }

    // Final audit pass (ADR-006): an edge that failed the five-control test
    // contributes zero quality by construction, so cutting every unverified
    // non-mandatory edge — and every edge stranded on an inactive agent —
    // can only lower cost and risk. This is a deterministic sweep, not a
    // search step; random mutation merely usually finds these first.
    let mut audited = best_topology.clone();
    let unverified: Vec<(AgentId, AgentId)> = best
        .edge_results
        .iter()
        .filter(|(_, _, _, verified)| !verified)
        .map(|(f, t, _, _)| (*f, *t))
        .collect();
    audited.edges.retain(|key, edge| {
        edge.mandatory
            || (!unverified.contains(key)
                && audited_agents_contains(&best_topology, key.0)
                && audited_agents_contains(&best_topology, key.1))
    });
    // Deactivate agents left with no edges at all (unless a mandatory edge
    // needs them).
    let needed: Vec<AgentId> = audited
        .edges
        .values()
        .flat_map(|e| [e.from, e.to])
        .collect();
    audited.active_agents.retain(|a| needed.contains(a));
    let audited_fit = evaluate(&audited, env, &config.weights, config.seed, &mut cache);
    if audited_fit.fitness >= best.fitness {
        best_topology = audited;
        best = audited_fit;
    }

    EvolveOutcome {
        initial,
        best,
        best_topology,
        generations_run: config.generations,
        mutations_proposed,
        mutations_rejected_by_constitution: mutations_rejected,
        total_verification_evaluations: total_verifications,
        warm_started,
    }
}

fn audited_agents_contains(topology: &Topology, agent: AgentId) -> bool {
    topology.active_agents.contains(&agent)
}

/// Persist the surviving topology as procedural memory for the next run
/// (ADR-005 §skills).
pub fn remember_topology(store: &mut InMemoryStore, family: &str, outcome: &EvolveOutcome) {
    store.store_topology(TopologyRecord {
        family: family.to_string(),
        agent_sequence: outcome
            .best_topology
            .active_agents
            .iter()
            .map(|a| format!("agent-{a}"))
            .collect(),
        successful_uses: 1,
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn authority_never_expands_past_the_cap() {
        let mut t = Topology::fully_connected(&[0, 1], Authority::ContextInject);
        assert!(apply_mutation(&mut t, Mutation::RaiseAuthority { from: 0, to: 1 }).is_err());
        // Lowering is always allowed; re-raising back to the cap is fine.
        apply_mutation(&mut t, Mutation::LowerAuthority { from: 0, to: 1 }).unwrap();
        apply_mutation(&mut t, Mutation::RaiseAuthority { from: 0, to: 1 }).unwrap();
        assert_eq!(t.edges[&(0, 1)].authority, Authority::ContextInject);
    }

    #[test]
    fn mandatory_edges_cannot_be_removed_or_orphaned() {
        let mut t = Topology::fully_connected(&[0, 1], Authority::ContextInject);
        t.edges.get_mut(&(0, 1)).unwrap().mandatory = true;
        assert_eq!(
            apply_mutation(&mut t, Mutation::RemoveEdge { from: 0, to: 1 }),
            Err(MutationError::MandatoryEdge { from: 0, to: 1 })
        );
        assert!(apply_mutation(&mut t, Mutation::DeactivateAgent { agent: 0 }).is_err());
    }

    #[test]
    fn evolution_is_deterministic_per_seed() {
        let env = SyntheticEnv::new(5);
        let config = DarwinConfig {
            generations: 4,
            population: 4,
            ..Default::default()
        };
        let a = evolve(&env, &config, None);
        let b = evolve(&env, &config, None);
        assert_eq!(a.best.fitness, b.best.fitness);
        assert_eq!(a.best_topology.edges.len(), b.best_topology.edges.len());
    }
}
