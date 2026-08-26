//! ADR-006's objective with causal Quality: only edges that survive the
//! four-control test contribute; verification cost is counted and reported.

use crate::env::SyntheticEnv;
use crate::topology::Topology;
use latentmesh_core::Authority;
use latentmesh_gate::causal::{verify_edge, EdgeVerdict};
use serde::{Deserialize, Serialize};

/// λ, μ, ρ of `G* = argmax[Quality − λ·Cost − μ·Latency − ρ·Risk]`, plus the
/// statistical knobs of the per-edge verification.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct FitnessWeights {
    pub lambda_cost: f64,
    pub mu_latency: f64,
    pub rho_risk: f64,
    pub alpha: f64,
    pub trials: usize,
    pub resamples: usize,
}

impl Default for FitnessWeights {
    fn default() -> Self {
        FitnessWeights {
            lambda_cost: 0.002,
            mu_latency: 0.01,
            rho_risk: 0.5,
            alpha: 0.05,
            trials: 24,
            resamples: 600,
        }
    }
}

/// The evaluated objective, kept decomposed so receipts can show the terms.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FitnessBreakdown {
    pub quality: f64,
    pub cost: f64,
    pub latency: f64,
    pub risk: f64,
    pub fitness: f64,
    pub task_success: f64,
    /// Per live edge: `(from, to, delta_v, verified)`.
    pub edge_results: Vec<(u8, u8, f64, bool)>,
    /// Control evaluations spent verifying this topology (ADR-006's
    /// "verification bounds the search", made visible).
    pub verification_evaluations: usize,
}

fn authority_risk(authority: Authority) -> f64 {
    match authority {
        Authority::ObserveOnly => 0.05,
        Authority::ContextInject => 0.2,
        Authority::LatentPrefix => 0.45,
        Authority::ActionInfluencing => 0.75,
    }
}

/// Cache of per-edge verification verdicts, valid for one (environment,
/// seed, weights) triple. The environment is frozen, so an edge's trial —
/// and therefore its verdict — is a property of the edge, not of which
/// candidate topology contains it; caching it is correctness-preserving and
/// is what makes the Darwin loop affordable (ADR-006's "verification bounds
/// the search" consequence, engineered rather than suffered).
pub type VerificationCache = std::collections::BTreeMap<(u8, u8), (f64, bool)>;

/// Evaluate one topology in the environment. `seed` drives the permutation
/// tests; per-edge sub-seeds are derived from it deterministically. Fresh
/// verifications populate `cache`; hits are free and not double-counted.
pub fn evaluate(
    topology: &Topology,
    env: &SyntheticEnv,
    weights: &FitnessWeights,
    seed: u64,
    cache: &mut VerificationCache,
) -> FitnessBreakdown {
    let mut quality = 0.0;
    let mut risk = 0.0;
    let mut edge_results = Vec::new();
    let mut verification_evaluations = 0usize;

    for edge in topology.live_edges() {
        let key = (edge.from, edge.to);
        let (delta_v, verified) = match cache.get(&key) {
            Some(&cached) => cached,
            None => {
                let trial = env.edge_trial(edge.from, edge.to, weights.trials);
                // Five controls, each `trials` samples, each permutation-tested.
                verification_evaluations += weights.trials * 5;
                let edge_seed = seed
                    .wrapping_mul(0x100)
                    .wrapping_add((u64::from(edge.from) << 8) | u64::from(edge.to));
                let verdict = verify_edge(&trial, weights.alpha, weights.resamples, edge_seed);
                let result = match verdict {
                    EdgeVerdict::Admit { delta_v, .. } => (delta_v, true),
                    EdgeVerdict::Reject { .. } => (0.0, false),
                };
                cache.insert(key, result);
                result
            }
        };
        // An unverified edge contributes zero regardless of correlation.
        if verified {
            quality += delta_v;
        }
        risk += authority_risk(edge.authority);
        edge_results.push((edge.from, edge.to, delta_v, verified));
    }

    let cost = topology.compute_proxy();
    // Latency proxy: the longest chain a frame can traverse scales with edge
    // count over active agents (a dense graph synchronizes more).
    let latency = if topology.active_agents.is_empty() {
        0.0
    } else {
        topology.live_edges().count() as f64 / topology.active_agents.len() as f64
    };
    let task_success = env.task_success(topology.live_edges().map(|e| (e.from, e.to)));

    let fitness = quality
        - weights.lambda_cost * cost
        - weights.mu_latency * latency
        - weights.rho_risk * risk;

    FitnessBreakdown {
        quality,
        cost,
        latency,
        risk,
        fitness,
        task_success,
        edge_results,
        verification_evaluations,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pruning_redundant_edges_improves_fitness() {
        let env = SyntheticEnv::new(11);
        let weights = FitnessWeights::default();
        let mut cache = VerificationCache::new();
        let dense = Topology::fully_connected(&env.agents(), Authority::ContextInject);
        let dense_fit = evaluate(&dense, &env, &weights, 11, &mut cache);

        // Keep only the edges that individually verified.
        let mut pruned = dense.clone();
        let verified: Vec<(u8, u8)> = dense_fit
            .edge_results
            .iter()
            .filter(|(_, _, _, ok)| *ok)
            .map(|(f, t, _, _)| (*f, *t))
            .collect();
        pruned.edges.retain(|key, _| verified.contains(key));
        pruned.active_agents = (0..10)
            .filter(|a| verified.iter().any(|(f, t)| f == a || t == a))
            .collect();
        let pruned_fit = evaluate(&pruned, &env, &weights, 11, &mut cache);

        assert!(pruned_fit.fitness > dense_fit.fitness);
        assert!(pruned_fit.cost < dense_fit.cost);
        assert!(pruned_fit.task_success >= dense_fit.task_success - 1e-9);
    }

    #[test]
    fn evaluation_is_deterministic_with_and_without_cache() {
        let env = SyntheticEnv::new(3);
        let weights = FitnessWeights::default();
        let t = Topology::fully_connected(&env.agents(), Authority::ContextInject);
        let mut cache_a = VerificationCache::new();
        let mut cache_b = VerificationCache::new();
        let a = evaluate(&t, &env, &weights, 3, &mut cache_a);
        let b = evaluate(&t, &env, &weights, 3, &mut cache_b);
        assert_eq!(a.fitness, b.fitness);
        assert_eq!(a.edge_results, b.edge_results);
        // A warm cache changes cost, never results.
        let c = evaluate(&t, &env, &weights, 3, &mut cache_a);
        assert_eq!(c.fitness, a.fitness);
        assert_eq!(c.verification_evaluations, 0);
    }
}
