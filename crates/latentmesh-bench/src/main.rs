//! `latentmesh-bench` — real, measured numbers only.
//!
//! This environment has no open-weight model hidden-state access, so this
//! binary does NOT claim end-to-end cross-model task accuracy, latency, or
//! token savings — those require live models (ADR-001 §8) and are explicitly
//! out of scope here. What it DOES measure, for real, on this machine:
//!
//!   1. Wire bytes per `LatentFrame` payload at each `Encoding`, at realistic
//!      dimensions — validates (or corrects) the back-of-envelope scaling
//!      claim that 16×4096-dim FP16 vectors is ~128 KiB.
//!   2. Wall-clock cost of `AlignmentTransform::fit`/`apply` at increasing
//!      dimension — is the alignment step itself cheap enough to sit on a
//!      streaming hot path (ADR-004)?
//!   3. Wall-clock cost of the causal-verification permutation test
//!      (`latentmesh-gate::causal::verify_edge`) at realistic trial/resample
//!      counts — bounds how many candidate edges a topology search (ADR-006)
//!      can afford to test per generation.
//!
//! Plus the integration benchmarks (ADRs 015–018) in `integrations.rs`:
//!
//!   4. Streamed-vs-sequential latent pipelining over the real codec.
//!   5. Latent memory store/recall (`InMemoryStore` always; RuVector HNSW
//!      backend with `--features ruvector`).
//!   6. Federation rule bytes and negative-transfer defense.
//!   7. The Darwin acceptance suite's receipt numbers.
//!
//! Run: `cargo run --release -p latentmesh-bench`
//! Run with the RuVector backend: `cargo run --release -p latentmesh-bench --features ruvector`

mod integrations;

use latentmesh_align::AlignmentTransform;
use latentmesh_core::{Encoding, Payload};
use latentmesh_gate::causal::{verify_edge, EdgeTrial};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

fn main() {
    println!("# LatentMesh bench — measured, not asserted (ADR-001 §8, ADR-002, ADR-003)\n");
    bench_wire_bytes();
    bench_alignment();
    bench_causal_verification();
    integrations::bench_stream_pipelining();
    integrations::bench_memory_recall();
    integrations::bench_federation();
    integrations::bench_evolve();
    println!("\n## What this does NOT measure");
    println!("Cross-model semantic transfer quality, real task accuracy, real end-to-end");
    println!("latency vs. a text pipeline, and every specific figure attributed to external");
    println!("papers in the ADRs. Those require open-weight model access this run doesn't");
    println!("have. See ADR-001 §8 (Honest feasibility).");
}

fn bench_wire_bytes() {
    println!("## 1. Wire bytes per payload (measured, `Payload::wire_bytes()`)\n");
    println!("| dim | count | encoding | bytes/vec | total | vs claim |");
    println!("|---|---|---|---|---|---|");
    let cases: &[(usize, usize, &str)] = &[
        (4096, 16, "16×4096 FP16 ≈ 128 KiB"),
        (4096, 16, "16×4096 Int8 ≈ 64 KiB (+headers)"),
        (512, 1, "512-dim quantized ≈ 8 KiB per 16"),
    ];
    for &(dim, count, claim) in cases {
        for enc in [Encoding::F32, Encoding::F16, Encoding::Int8] {
            let v: Vec<f32> = (0..dim).map(|i| (i as f32).sin()).collect();
            let bytes = Payload::encode(&v, enc).wire_bytes();
            let total = bytes * count;
            println!(
                "| {dim} | {count} | {enc:?} | {bytes} B | {} | {claim} |",
                human_bytes(total)
            );
        }
    }
    println!();
}

fn bench_alignment() {
    println!("## 2. Alignment fit/apply latency (measured wall-clock, release build)\n");
    println!("| dim | calibration pairs | fit (ms) | apply (µs) | confidence |");
    println!("|---|---|---|---|---|");
    for &dim in &[64usize, 256, 1024, 4096] {
        for &n_pairs in &[16usize, 64] {
            let mut rng = StdRng::seed_from_u64(dim as u64 * 31 + n_pairs as u64);
            // Synthetic ground truth: B = A rotated by a random orthogonal Q,
            // so fit quality is measurable (confidence should be high) even
            // though this says nothing about real cross-model geometry.
            let pairs: Vec<(Vec<f32>, Vec<f32>)> = (0..n_pairs)
                .map(|_| {
                    let a: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
                    // Cheap synthetic "rotation": a fixed random permutation +
                    // sign flip, applied consistently — enough to give the
                    // fitter real structure to recover without a full QR.
                    let b: Vec<f32> = a.iter().rev().map(|v| -v).collect();
                    (a, b)
                })
                .collect();

            let t0 = Instant::now();
            let transform = AlignmentTransform::fit(&pairs);
            let fit_ms = t0.elapsed().as_secs_f64() * 1000.0;

            let z: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let t1 = Instant::now();
            let _ = transform.apply(&z);
            let apply_us = t1.elapsed().as_secs_f64() * 1_000_000.0;

            println!(
                "| {dim} | {n_pairs} | {fit_ms:.3} | {apply_us:.1} | {:.3} |",
                transform.confidence
            );
        }
    }
    println!();
}

fn bench_causal_verification() {
    println!("## 3. Causal edge verification latency (measured wall-clock)\n");
    println!("| trials | resamples | verify_edge (ms) |");
    println!("|---|---|---|");
    for &(n_trials, resamples) in &[(20usize, 1000usize), (40, 2000), (100, 5000)] {
        let mut rng = StdRng::seed_from_u64(7);
        let noise = |shift: f64, rng: &mut StdRng| shift + rng.gen_range(-0.5..0.5);
        let mut real = vec![];
        let mut zero = vec![];
        let mut random = vec![];
        let mut mismatched = vec![];
        let mut self_generated = vec![];
        let mut text_equivalent = vec![];
        for _ in 0..n_trials {
            real.push(noise(0.9, &mut rng));
            zero.push(noise(0.0, &mut rng));
            random.push(noise(0.0, &mut rng));
            mismatched.push(noise(0.0, &mut rng));
            self_generated.push(noise(0.0, &mut rng));
            text_equivalent.push(noise(0.3, &mut rng)); // text helps some, less than latent
        }
        let trial = EdgeTrial {
            real,
            zero,
            random,
            mismatched,
            self_generated,
            text_equivalent,
        };
        let t0 = Instant::now();
        let verdict = verify_edge(&trial, 0.05, resamples, 42);
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        println!("| {n_trials} | {resamples} | {ms:.3} ({verdict:?}) |");
    }
    println!();
}

fn human_bytes(n: usize) -> String {
    if n >= 1024 * 1024 {
        format!("{:.1} MiB", n as f64 / (1024.0 * 1024.0))
    } else if n >= 1024 {
        format!("{:.1} KiB", n as f64 / 1024.0)
    } else {
        format!("{n} B")
    }
}
