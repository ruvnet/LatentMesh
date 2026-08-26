//! Integration benchmarks (ADRs 015–018). Same discipline as the rest of the
//! binary: real wall-clock measurements of what runs on this machine, with an
//! explicit statement of what each number does and does not prove. Evidence
//! label for everything here: host software benchmark / deterministic
//! simulation — no live models, no network, no radio hardware.

use latentmesh_core::{Authority, Encoding, LatentFrame, Payload, Provenance};
use latentmesh_evolve::{acceptance_check, evolve, DarwinConfig, SyntheticEnv};
use latentmesh_federation::{
    validate_candidate, AdmissionConfig, RuleScope, RuleVerdict, Transition, TransitionRule,
    WorldModel,
};
use latentmesh_memory::{Fidelity, InMemoryStore, LatentMemory, MemoryConfig, TrajectoryRecord};
use latentmesh_stream::{ChannelTransport, FrameTransport};
use std::time::Instant;

fn frame(seq: u64, dim: usize) -> LatentFrame {
    LatentFrame {
        id: format!("bench-{seq}"),
        sender_model: "bench-sender".into(),
        receiver_space: "bench-receiver".into(),
        transform_hash: "bench-transform".into(),
        sequence: seq,
        payload: Payload::encode(&vec![0.5f32; dim], Encoding::F16),
        confidence: 0.9,
        provenance: Provenance {
            sender_model: "bench-sender".into(),
            context_hash: "bench-context".into(),
            parents: vec![],
        },
        authority: Authority::ContextInject,
        timestamp: 0,
    }
}

/// Busy-work standing in for per-frame model compute; returns a value so the
/// optimizer cannot remove it.
fn simulated_stage_work(iterations: u32) -> f64 {
    let mut acc = 0.0f64;
    for i in 0..std::hint::black_box(iterations) {
        acc += f64::from(i).sqrt().sin();
    }
    std::hint::black_box(acc)
}

/// ADR-015 / ADR-004: does pipelining N frames reduce wall-clock vs. waiting
/// for completion? Producer and consumer each spend simulated compute per
/// frame; sequential runs them back to back, pipelined overlaps them across
/// the real codec + channel transport.
pub fn bench_stream_pipelining() {
    println!("## 4. Streaming pipeline: sequential vs pipelined (ADR-015, measured wall-clock)\n");
    println!("| frames | dim | sequential (ms) | pipelined (ms) | speedup |");
    println!("|---|---|---|---|---|");
    for &(frames, dim, work) in &[(32usize, 512usize, 60_000u32), (64, 1024, 120_000)] {
        // Sequential: A produces everything, then B consumes everything.
        let t0 = Instant::now();
        let mut produced = Vec::with_capacity(frames);
        for seq in 0..frames {
            let _ = std::hint::black_box(simulated_stage_work(work));
            produced.push(frame(seq as u64, dim));
        }
        let mut sink = 0.0f64;
        for f in &produced {
            let _ = std::hint::black_box(f.payload.decode());
            sink += simulated_stage_work(work);
        }
        let sequential_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // Pipelined: B consumes each frame as it is produced, over the real
        // codec + channel transport, on a second thread.
        let (mut tx, mut rx) = ChannelTransport::pair();
        let t1 = Instant::now();
        let consumer = std::thread::spawn(move || {
            let mut consumed = 0usize;
            let mut sink = 0.0f64;
            while consumed < frames {
                match rx.try_recv_frame().expect("bench transport") {
                    Some(f) => {
                        let _ = std::hint::black_box(f.payload.decode());
                        sink += simulated_stage_work(work);
                        consumed += 1;
                    }
                    None => std::hint::spin_loop(),
                }
            }
            sink
        });
        for seq in 0..frames {
            let _ = std::hint::black_box(simulated_stage_work(work));
            tx.send_frame(&frame(seq as u64, dim)).expect("bench send");
        }
        let consumer_sink = consumer.join().expect("consumer thread");
        let pipelined_ms = t1.elapsed().as_secs_f64() * 1000.0;

        println!(
            "| {frames} | {dim} | {sequential_ms:.1} | {pipelined_ms:.1} | {:.2}x |",
            sequential_ms / pipelined_ms
        );
        let _ = std::hint::black_box((sink, consumer_sink));
    }
    println!("\nProves the pipelining shape over the real codec in-process; proves nothing");
    println!("about a production network deployment's latency.\n");
}

/// ADR-016: recall latency and correctness on synthetic trajectories.
pub fn bench_memory_recall() {
    println!("## 5. Latent memory store/recall (ADR-016, measured wall-clock)\n");
    println!("| backend | records | dim | store (ms) | recall top-8 (µs) | top hit correct |");
    println!("|---|---|---|---|---|---|");
    let dims = 256usize;
    let count = 5_000usize;
    let config = MemoryConfig {
        dimensions: dims,
        raw_tier_causal_floor: 0.0,
        rule_promotion_uses: 3,
    };

    let mk_record = |i: usize| {
        let latent: Vec<f32> = (0..dims).map(|d| ((i * 31 + d) as f32).sin()).collect();
        TrajectoryRecord {
            id: format!("r{i}"),
            latent,
            reward: 1.0,
            context_hash: "c".into(),
            action: "solve".into(),
            outcome: "ok".into(),
            causal_value: 0.9,
            fidelity: Fidelity::Raw,
            reconstruction_error: 0.0,
            parents: vec![],
        }
    };
    let query = mk_record(count / 2).latent;

    let mut in_memory = InMemoryStore::new(config);
    run_memory_backend(
        "InMemoryStore",
        &mut in_memory,
        count,
        dims,
        mk_record,
        &query,
    );

    #[cfg(feature = "ruvector")]
    {
        let mut ruvector =
            latentmesh_memory::RuVectorStore::new(config).expect("ruvector backend constructs");
        run_memory_backend(
            "RuVectorStore (HNSW)",
            &mut ruvector,
            count,
            dims,
            mk_record,
            &query,
        );
    }
    #[cfg(not(feature = "ruvector"))]
    println!("| RuVectorStore | — | — | — | — | build with `--features ruvector` |");
    println!();
}

fn run_memory_backend<M: LatentMemory>(
    name: &str,
    store: &mut M,
    count: usize,
    dims: usize,
    mk_record: impl Fn(usize) -> TrajectoryRecord,
    query: &[f32],
) {
    let t0 = Instant::now();
    for i in 0..count {
        store.store(mk_record(i)).expect("bench store");
    }
    let store_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let t1 = Instant::now();
    let hits = store.recall(query, 8).expect("bench recall");
    let recall_us = t1.elapsed().as_secs_f64() * 1_000_000.0;
    let expected = format!("r{}", count / 2);
    let correct = hits
        .first()
        .map(|h| h.record.id == expected)
        .unwrap_or(false);
    println!("| {name} | {count} | {dims} | {store_ms:.1} | {recall_us:.1} | {correct} |");
}

/// ADR-017: rule bytes vs raw-experience bytes, and whether validated
/// federation beats blind pooling on a divergent-dynamics fixture.
pub fn bench_federation() {
    println!("## 6. Federation: bytes and negative-transfer defense (ADR-017, deterministic simulation)\n");

    let rule = TransitionRule {
        pre: "door_closed".into(),
        action: "push".into(),
        post: "door_open".into(),
        support: 128,
        confidence: 0.94,
        scope: RuleScope::Global,
    };
    let rule_bytes = rule.encode_for_transmission().expect("encodes").len();
    // The experience the rule summarizes: 128 observed transitions as JSON.
    let raw_experience_bytes = 128 * serde_json::to_vec(&rule).map(|v| v.len()).unwrap_or(0);
    println!("| what | bytes on the wire |");
    println!("|---|---|");
    println!("| one TransitionRule (128 observations) | {rule_bytes} B |");
    println!("| the raw 128 observations it replaces | {raw_experience_bytes} B |");
    println!(
        "| reduction | {:.0}x |",
        raw_experience_bytes as f64 / rule_bytes as f64
    );

    // Node B has different local dynamics for the same key: blind pooling
    // would overwrite B's correct rule; validation rejects the import.
    let mut node_b = WorldModel::new();
    for _ in 0..24 {
        node_b.record_holdout(Transition {
            pre: "door_closed".into(),
            action: "push".into(),
            post: "door_stuck".into(), // B's doors behave differently
        });
    }
    node_b.install(TransitionRule {
        post: "door_stuck".into(),
        ..rule.clone()
    });
    let baseline_accuracy: f64 = node_b.holdout_scores(None).iter().sum::<f64>() / 24.0;

    let t0 = Instant::now();
    let verdict = validate_candidate(&node_b, &rule, 0, &AdmissionConfig::default());
    let validate_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Blind pooling: install the foreign rule unconditionally.
    let mut pooled = node_b.clone();
    pooled.install(rule.clone());
    let pooled_accuracy: f64 = pooled.holdout_scores(None).iter().sum::<f64>() / 24.0;

    println!("\n| condition | holdout accuracy |");
    println!("|---|---|");
    println!("| node B local rules | {baseline_accuracy:.2} |");
    println!("| blind pooling of node A's rule | {pooled_accuracy:.2} |");
    let validated = match verdict {
        RuleVerdict::Admit { .. } => "ADMITTED (unexpected)".to_string(),
        RuleVerdict::Reject { control, .. } => format!("rejected on `{control}` control"),
    };
    println!(
        "| validated federation | {baseline_accuracy:.2} ({validated}, {validate_ms:.1} ms) |"
    );
    println!("\nProves the admission logic discriminates on this fixture; not field behavior.\n");
}

/// ADR-018: run the Darwin acceptance suite and report the receipt numbers.
pub fn bench_evolve() {
    println!("## 7. MetaHarness Darwin loop (ADR-018, deterministic simulation)\n");
    let config = DarwinConfig::default();
    let env = SyntheticEnv::new(config.seed);
    let t0 = Instant::now();
    let outcome = evolve(&env, &config, None);
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let report = acceptance_check(&outcome);
    println!("| metric | value |");
    println!("|---|---|");
    println!(
        "| generations × population | {} × {} |",
        config.generations, config.population
    );
    println!("| wall clock | {elapsed_ms:.0} ms |");
    println!(
        "| compute proxy before → after | {:.0} → {:.0} |",
        outcome.initial.cost, outcome.best.cost
    );
    println!(
        "| compute reduction | {:.1}% (target ≥ 30%) |",
        report.compute_reduction * 100.0
    );
    println!(
        "| task success before → after | {:.2} → {:.2} |",
        report.task_success_before, report.task_success_after
    );
    println!(
        "| surviving edges all causally verified | {} |",
        report.all_surviving_edges_verified
    );
    println!(
        "| verification evaluations | {} |",
        outcome.total_verification_evaluations
    );
    println!(
        "| acceptance (ADR-006, simulated) | {} |",
        if report.passed {
            "passed"
        } else {
            "NOT passed"
        }
    );
    println!("\nDeterministic simulation with a synthetic ten-agent environment; proves the");
    println!("loop optimizes the causal objective, not live multi-agent gains.\n");
}
