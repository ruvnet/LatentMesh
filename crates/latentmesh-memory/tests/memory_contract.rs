//! Backend-agnostic contract tests (ADR-016): the same behavioral assertions
//! run against `InMemoryStore` always and against `RuVectorStore` under the
//! `ruvector` feature — the trait is the contract, not the backend.

use latentmesh_memory::{
    Fidelity, InMemoryStore, LatentMemory, MemoryConfig, MemoryError, TopologyRecord,
    TrajectoryRecord,
};

const DIMS: usize = 16;

fn config() -> MemoryConfig {
    MemoryConfig {
        dimensions: DIMS,
        raw_tier_causal_floor: 0.5,
        rule_promotion_uses: 3,
    }
}

fn record(id: &str, direction: f32, causal_value: f64) -> TrajectoryRecord {
    // Distinct, well-separated directions so approximate recall is stable.
    let latent: Vec<f32> = (0..DIMS)
        .map(|i| {
            if i as f32 == direction {
                1.0
            } else {
                0.01 * direction
            }
        })
        .collect();
    TrajectoryRecord {
        id: id.into(),
        latent,
        reward: 1.0,
        context_hash: format!("ctx-{id}"),
        action: "solve".into(),
        outcome: "ok".into(),
        causal_value,
        fidelity: Fidelity::Raw,
        reconstruction_error: 0.0,
        parents: vec![],
    }
}

fn contract<M: LatentMemory>(store: &mut M) {
    // Raw tier enforces the causal floor.
    let weak = record("weak", 1.0, 0.1);
    assert!(matches!(
        store.store(weak.clone()),
        Err(MemoryError::CausalValueBelowFloor { .. })
    ));
    // The same trajectory is admissible compressed.
    store.store_compressed(weak).unwrap();

    for (i, id) in ["a", "b", "c"].iter().enumerate() {
        store.store(record(id, i as f32 + 2.0, 0.9)).unwrap();
    }
    assert_eq!(store.len(), 4);

    // Recall finds the matching direction.
    let query = record("q", 3.0, 0.9).latent;
    let hits = store.recall(&query, 2).unwrap();
    assert!(!hits.is_empty());
    assert_eq!(hits[0].record.id, "b");
    assert!(hits[0].similarity > 0.9);

    // Dimension mismatches are typed errors on both paths.
    assert!(matches!(
        store.recall(&[0.0; 3], 1),
        Err(MemoryError::DimensionMismatch { .. })
    ));
    let mut bad = record("bad", 2.0, 0.9);
    bad.latent = vec![0.0; 3];
    assert!(matches!(
        store.store(bad),
        Err(MemoryError::DimensionMismatch { .. })
    ));

    // Duplicate ids are refused.
    assert!(matches!(
        store.store(record("a", 2.0, 0.9)),
        Err(MemoryError::DuplicateId(_))
    ));

    // Compression moves down the continuum, accumulates measured error, and
    // refuses to move up.
    let err = store.compress_to("a", Fidelity::Compressed).unwrap();
    assert!(err >= 0.0);
    assert!(matches!(
        store.compress_to("a", Fidelity::Compressed),
        Err(MemoryError::FidelityRegression { .. })
    ));

    // Prototype promotion folds members into a centroid with lineage.
    store
        .promote_to_prototype("proto-1", "solve-family", &["b", "c"])
        .unwrap();
    let hits = store.recall(&query, 4).unwrap();
    let proto = hits
        .iter()
        .find(|h| h.record.id == "proto-1")
        .expect("prototype is recallable");
    assert_eq!(proto.record.fidelity, Fidelity::Prototype);
    assert_eq!(proto.record.parents, vec!["b".to_string(), "c".to_string()]);

    assert!(matches!(
        store.promote_to_prototype("proto-2", "f", &["b"]),
        Err(MemoryError::NotEnoughMembers(1))
    ));
}

#[test]
fn in_memory_store_satisfies_the_contract() {
    let mut store = InMemoryStore::new(config());
    contract(&mut store);
}

#[test]
fn in_memory_store_topology_and_rule_promotion() {
    let mut store = InMemoryStore::new(config());
    store.store_topology(TopologyRecord {
        family: "distributed-rust-debugging".into(),
        agent_sequence: vec![
            "planner".into(),
            "rust-specialist".into(),
            "verifier".into(),
        ],
        successful_uses: 1,
    });
    let warm = store.recall_topology("distributed-rust-debugging").unwrap();
    assert_eq!(warm.agent_sequence.len(), 3);
    store.record_topology_use("distributed-rust-debugging");
    assert_eq!(
        store
            .recall_topology("distributed-rust-debugging")
            .unwrap()
            .successful_uses,
        2
    );
    assert!(store.recall_topology("unknown-family").is_none());

    // Rule promotion requires the configured reuse count.
    for (i, id) in ["a", "b"].iter().enumerate() {
        store.store(record(id, i as f32 + 2.0, 0.9)).unwrap();
    }
    store
        .promote_to_prototype("proto", "solve-family", &["a", "b"])
        .unwrap();
    assert!(matches!(
        store.promote_to_rule("proto", 2, "family → pipeline"),
        Err(MemoryError::NotReusedEnough { .. })
    ));
    store
        .promote_to_rule("proto", 3, "family → pipeline")
        .unwrap();
    assert_eq!(store.rules().len(), 1);
    assert_eq!(store.get("proto").unwrap().fidelity, Fidelity::Rule);
}

#[cfg(feature = "ruvector")]
#[test]
fn ruvector_store_satisfies_the_contract() {
    let mut store = latentmesh_memory::RuVectorStore::new(config()).unwrap();
    contract(&mut store);
}
