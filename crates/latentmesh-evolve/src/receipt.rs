//! The MetaHarness receipt (ADR-018): every run emits a JSON document that
//! says exactly what it proves and what it does not — the same evidence-label
//! discipline as `harness/air`'s stage benchmark receipt (ADR-014). The
//! `harness/evolve` suite validates receipts against ADR-006's acceptance
//! bounds.

use crate::darwin::{DarwinConfig, EvolveOutcome};
use serde::{Deserialize, Serialize};

pub const RECEIPT_SCHEMA: &str = "latentmesh-evolve-receipt-v1";
pub const EVIDENCE_LABEL: &str = "simulated";

/// ADR-006's acceptance bounds, evaluated over an [`EvolveOutcome`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AcceptanceReport {
    /// Required: ≥ 30% compute-proxy reduction vs. the dense start.
    pub compute_reduction: f64,
    pub compute_reduction_target: f64,
    pub compute_reduction_met: bool,
    /// Required: task success does not decrease.
    pub task_success_before: f64,
    pub task_success_after: f64,
    pub task_success_maintained: bool,
    /// Required: every surviving live edge individually passed the
    /// four-control test, or is governance-mandatory.
    pub surviving_edges: usize,
    pub unverified_nonmandatory_edges: usize,
    pub all_surviving_edges_verified: bool,
    pub passed: bool,
}

/// Evaluate ADR-006's acceptance test over a finished run.
pub fn acceptance_check(outcome: &EvolveOutcome) -> AcceptanceReport {
    let before = outcome.initial.cost;
    let after = outcome.best.cost;
    let compute_reduction = if before > 0.0 {
        1.0 - after / before
    } else {
        0.0
    };
    let compute_reduction_target = 0.30;
    let compute_reduction_met = compute_reduction >= compute_reduction_target;

    let task_success_maintained = outcome.best.task_success >= outcome.initial.task_success - 1e-9;

    let mandatory: Vec<(u8, u8)> = outcome
        .best_topology
        .edges
        .values()
        .filter(|e| e.mandatory)
        .map(|e| (e.from, e.to))
        .collect();
    let unverified_nonmandatory = outcome
        .best
        .edge_results
        .iter()
        .filter(|(f, t, _, verified)| !verified && !mandatory.contains(&(*f, *t)))
        .count();
    let all_verified = unverified_nonmandatory == 0;

    AcceptanceReport {
        compute_reduction,
        compute_reduction_target,
        compute_reduction_met,
        task_success_before: outcome.initial.task_success,
        task_success_after: outcome.best.task_success,
        task_success_maintained,
        surviving_edges: outcome.best.edge_results.len(),
        unverified_nonmandatory_edges: unverified_nonmandatory,
        all_surviving_edges_verified: all_verified,
        passed: compute_reduction_met && task_success_maintained && all_verified,
    }
}

/// The full receipt document.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Receipt {
    pub schema: String,
    /// Always [`EVIDENCE_LABEL`]: a deterministic simulation proves the loop
    /// optimizes the causal objective; it proves nothing about live agents.
    pub evidence: String,
    pub seed: u64,
    pub generations: usize,
    pub population: usize,
    pub warm_started: bool,
    pub fitness_before: f64,
    pub fitness_after: f64,
    pub quality_after: f64,
    pub compute_before: f64,
    pub compute_after: f64,
    pub mutations_proposed: usize,
    pub mutations_rejected_by_constitution: usize,
    pub verification_evaluations: usize,
    pub acceptance: AcceptanceReport,
    pub not_claimed: Vec<String>,
}

impl Receipt {
    pub fn from_outcome(config: &DarwinConfig, outcome: &EvolveOutcome) -> Receipt {
        Receipt {
            schema: RECEIPT_SCHEMA.to_string(),
            evidence: EVIDENCE_LABEL.to_string(),
            seed: config.seed,
            generations: outcome.generations_run,
            population: config.population,
            warm_started: outcome.warm_started,
            fitness_before: outcome.initial.fitness,
            fitness_after: outcome.best.fitness,
            quality_after: outcome.best.quality,
            compute_before: outcome.initial.cost,
            compute_after: outcome.best.cost,
            mutations_proposed: outcome.mutations_proposed,
            mutations_rejected_by_constitution: outcome.mutations_rejected_by_constitution,
            verification_evaluations: outcome.total_verification_evaluations,
            acceptance: acceptance_check(outcome),
            not_claimed: vec![
                "live multi-agent workload evidence is absent".to_string(),
                "real model routing evidence is absent".to_string(),
                "production MidStream bandwidth control is absent".to_string(),
            ],
        }
    }

    pub fn to_json(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(self)
    }
}
