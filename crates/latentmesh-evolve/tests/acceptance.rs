//! ADR-006's acceptance test, executable (ADR-018): ten agents with
//! deliberately redundant capabilities, a self-mutating topology, and the
//! requirement that evolution converges to a smaller graph (≥30% lower
//! compute proxy) at non-decreasing task success, with every surviving
//! non-mandatory edge individually passing the four-control causal test.
//! Evidence label: deterministic simulation.

use latentmesh_evolve::{
    acceptance_check, evolve, DarwinConfig, Receipt, SyntheticEnv, RECEIPT_SCHEMA,
};
use latentmesh_memory::{InMemoryStore, MemoryConfig};

fn config() -> DarwinConfig {
    DarwinConfig::default()
}

#[test]
fn the_acceptance_bounds_hold_in_the_frozen_environment() {
    let config = config();
    let env = SyntheticEnv::new(config.seed);
    let outcome = evolve(&env, &config, None);
    let report = acceptance_check(&outcome);

    assert!(
        report.compute_reduction_met,
        "compute reduction {:.3} below the 30% bound (before {:.1}, after {:.1})",
        report.compute_reduction, outcome.initial.cost, outcome.best.cost
    );
    assert!(
        report.task_success_maintained,
        "task success regressed: {:.3} → {:.3}",
        report.task_success_before, report.task_success_after
    );
    assert!(
        report.all_surviving_edges_verified,
        "{} surviving non-mandatory edges failed causal verification",
        report.unverified_nonmandatory_edges
    );
    assert!(report.passed);
}

#[test]
fn the_receipt_is_labelled_and_structured() {
    let config = config();
    let env = SyntheticEnv::new(config.seed);
    let outcome = evolve(&env, &config, None);
    let receipt = Receipt::from_outcome(&config, &outcome);
    assert_eq!(receipt.schema, RECEIPT_SCHEMA);
    assert_eq!(receipt.evidence, "simulated");
    assert!(!receipt.not_claimed.is_empty());
    assert!(receipt.verification_evaluations > 0);
    // Round-trips as JSON for the harness suite.
    let json = receipt.to_json().unwrap();
    let back: Receipt = serde_json::from_str(&json).unwrap();
    assert_eq!(back.acceptance.passed, receipt.acceptance.passed);
}

#[test]
fn warm_start_from_procedural_memory_is_used_and_recorded() {
    let config = config();
    let env = SyntheticEnv::new(config.seed);

    // First run, cold; remember the surviving topology.
    let cold = evolve(&env, &config, None);
    let mut store = InMemoryStore::new(MemoryConfig::default());
    latentmesh_evolve::darwin::remember_topology(&mut store, "synthetic-ten-agent", &cold);

    // Second run warm-starts from the remembered agent set.
    let warm = evolve(&env, &config, Some((&store, "synthetic-ten-agent")));
    assert!(warm.warm_started);
    assert!(!cold.warm_started);
    // The warm start begins from a pruned agent set, so its dense-start cost
    // is already at or below the cold start's.
    assert!(warm.initial.cost <= cold.initial.cost);
}
