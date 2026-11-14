//! Receipt `latentmesh.aoi-bench.v1`: assembly, digests, and git binding.

use std::process::Command;
use std::thread;

use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use super::canonical::{canonical_json, hex, sha256_hex};
use super::grid::{Axes, Cell};
use super::sim::{simulate, Metrics, SimParams, LINK_BYTES_PER_S};
use super::utility::{
    HALF_LIFE_MS, STALE_AGE_MS, USEFUL_COVERAGE_PPM, WEIGHT_RELEVANT_RESIDUAL, WEIGHT_SYMBOL,
};
use super::workload::{
    mask_bytes, materialize, BASELINES, GEN_PERIOD_MS, MSG_COUNT, RESIDUAL_SLOTS, RESIDUAL_VALUES,
    SYMBOLS, TASK_ID,
};

pub const SCHEMA: &str = "latentmesh.aoi-bench.v1";

/// Git binding. `dirty_snapshot_sha256` is the SHA-256 of `git diff HEAD`
/// (tracked changes only; untracked files are not part of the snapshot).
#[derive(Clone, Debug)]
pub struct GitInfo {
    pub commit: String,
    pub dirty: bool,
    pub dirty_snapshot_sha256: Option<String>,
}

fn git(args: &[&str]) -> Result<Vec<u8>, String> {
    let out = Command::new("git")
        .args(args)
        .output()
        .map_err(|e| format!("git unavailable: {e}"))?;
    if !out.status.success() {
        return Err(format!("git {args:?} failed"));
    }
    Ok(out.stdout)
}

/// Read git state at runtime; fails rather than fabricating it.
pub fn git_info() -> Result<GitInfo, String> {
    let commit = String::from_utf8_lossy(&git(&["rev-parse", "HEAD"])?)
        .trim()
        .to_string();
    let status = git(&["status", "--porcelain", "--untracked-files=no"])?;
    let dirty = !status.is_empty();
    let dirty_snapshot_sha256 = if dirty {
        Some(hex(&Sha256::digest(git(&["diff", "HEAD"])?)))
    } else {
        None
    };
    Ok(GitInfo {
        commit,
        dirty,
        dirty_snapshot_sha256,
    })
}

type Row = [(&'static str, u64); 20];

fn pairs(m: &Metrics) -> Row {
    [
        ("messages_generated", m.messages_generated),
        ("messages_delivered", m.messages_delivered),
        ("bytes_on_air", m.bytes_on_air),
        ("payload_bytes_tx", m.payload_bytes_tx),
        ("utility_micro", m.utility_micro),
        ("utility_per_byte_micro", m.utility_per_byte_micro),
        ("stale_deliveries", m.stale_deliveries),
        ("stale_delivery_rate_ppm", m.stale_delivery_rate_ppm),
        ("useful_fresh_deliveries", m.useful_fresh_deliveries),
        ("queue_occupancy_mean_milli", m.queue_occupancy_mean_milli),
        ("queue_occupancy_max", m.queue_occupancy_max),
        ("queue_overflow_drops", m.queue_overflow_drops),
        ("stale_drops", m.stale_drops),
        ("unschedulable_drops", m.unschedulable_drops),
        ("evidence_age_p50_ms", m.evidence_age_p50_ms),
        ("evidence_age_p95_ms", m.evidence_age_p95_ms),
        ("mandatory_violations", m.mandatory_violations),
        ("replay_inconsistencies", m.replay_inconsistencies),
        ("queue_bound_violations", m.queue_bound_violations),
        ("airtime_window_violations", m.airtime_window_violations),
    ]
}

fn row_json(seed: Option<u64>, m: &Metrics) -> Value {
    let mut o = Map::new();
    if let Some(s) = seed {
        o.insert("seed".into(), json!(s));
    }
    for (k, v) in pairs(m) {
        o.insert(k.into(), json!(v));
    }
    Value::Object(o)
}

/// Integer (floor) mean of each metric across seeds.
fn mean_metrics(rows: &[Metrics]) -> Metrics {
    let n = rows.len() as u64;
    let mut sums = [0u64; 20];
    for r in rows {
        for (i, (_, v)) in pairs(r).iter().enumerate() {
            sums[i] += v;
        }
    }
    let f = |i: usize| sums[i] / n;
    Metrics {
        messages_generated: f(0),
        messages_delivered: f(1),
        bytes_on_air: f(2),
        payload_bytes_tx: f(3),
        utility_micro: f(4),
        utility_per_byte_micro: f(5),
        stale_deliveries: f(6),
        stale_delivery_rate_ppm: f(7),
        useful_fresh_deliveries: f(8),
        queue_occupancy_mean_milli: f(9),
        queue_occupancy_max: f(10),
        queue_overflow_drops: f(11),
        stale_drops: f(12),
        unschedulable_drops: f(13),
        evidence_age_p50_ms: f(14),
        evidence_age_p95_ms: f(15),
        mandatory_violations: f(16),
        replay_inconsistencies: f(17),
        queue_bound_violations: f(18),
        airtime_window_violations: f(19),
    }
}

pub fn params_for(c: &Cell, stale_drop: bool, seq_base: u16) -> SimParams {
    SimParams {
        age_ms: c.age_ms,
        budget_bytes: c.budget_bytes,
        queue_depth: c.queue_depth,
        loss_ppm: c.loss_ppm,
        reorder_ppm: c.reorder_ppm,
        outage_ms: c.outage_ms,
        stale_drop,
        seq_base,
    }
}

/// `[baseline][seed]` metrics for one cell.
fn run_cell(c: &Cell, seeds: &[u64]) -> Result<Vec<Vec<Metrics>>, String> {
    let p = params_for(c, true, 0);
    let mut per: Vec<Vec<Metrics>> = vec![Vec::new(); BASELINES.len()];
    for &seed in seeds {
        let m = materialize(c, seed, 0)?;
        for (b, rows) in per.iter_mut().enumerate() {
            rows.push(simulate(&m, b, &p).metrics);
        }
    }
    Ok(per)
}

fn cell_json(c: &Cell, seeds: &[u64], per: &[Vec<Metrics>]) -> Value {
    let mut baselines = Map::new();
    for (b, name) in BASELINES.iter().enumerate() {
        let rows: Vec<Value> = seeds
            .iter()
            .zip(&per[b])
            .map(|(&s, m)| row_json(Some(s), m))
            .collect();
        baselines.insert(
            (*name).into(),
            json!({"per_seed": rows, "mean": row_json(None, &mean_metrics(&per[b]))}),
        );
    }
    json!({
        "cell_id": c.id,
        "axes": {
            "age_of_information_ms": c.age_ms, "airtime_budget_bytes": c.budget_bytes,
            "queue_depth": c.queue_depth, "loss_rate_ppm": c.loss_ppm,
            "reorder_rate_ppm": c.reorder_ppm, "link_outage_ms": c.outage_ms,
            "task_relevance_ppm": c.relevance_ppm,
        },
        "baselines": Value::Object(baselines),
    })
}

/// Evaluate cells on all cores; results come back in cell order.
fn run_cells(cells: &[Cell], seeds: &[u64]) -> Result<Vec<Vec<Vec<Metrics>>>, String> {
    let n = thread::available_parallelism().map_or(1, |n| n.get());
    let mut slots: Vec<Option<Result<Vec<Vec<Metrics>>, String>>> = vec![None; cells.len()];
    thread::scope(|s| {
        let handles: Vec<_> = (0..n)
            .map(|w| {
                s.spawn(move || {
                    (w..cells.len())
                        .step_by(n)
                        .map(|i| (i, run_cell(&cells[i], seeds)))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        for h in handles {
            for (i, r) in h.join().expect("worker panicked") {
                slots[i] = Some(r);
            }
        }
    });
    slots
        .into_iter()
        .map(|r| r.expect("cell computed"))
        .collect()
}

fn axes_json(a: &Axes) -> Value {
    json!({
        "age_of_information_ms": a.age_ms, "airtime_budget_bytes": a.budget_bytes,
        "queue_depth": a.queue_depth, "loss_rate_ppm": a.loss_ppm,
        "reorder_rate_ppm": a.reorder_ppm, "link_outage_ms": a.outage_ms,
        "task_relevance_ppm": a.relevance_ppm,
    })
}

/// The mask below is the wire-form illustration for rho = 0.5 (slots 0..3
/// Required, 4..7 Discardable). Per-cell masks use a seeded relevant set.
fn config_json() -> Value {
    let mut illustrative = [false; RESIDUAL_SLOTS];
    illustrative[..RESIDUAL_SLOTS / 2].fill(true);
    json!({
        "rng": "ChaCha8Rng/rand_chacha-0.3", "msg_count": MSG_COUNT,
        "gen_period_ms": GEN_PERIOD_MS, "link_bytes_per_s": LINK_BYTES_PER_S,
        "stale_age_ms": STALE_AGE_MS, "half_life_ms": HALF_LIFE_MS,
        "useful_coverage_ppm": USEFUL_COVERAGE_PPM,
        "weights": {"symbol": WEIGHT_SYMBOL, "relevant_residual": WEIGHT_RELEVANT_RESIDUAL},
        "workload": {"symbols": SYMBOLS, "residual_slots": RESIDUAL_SLOTS,
                     "residual_values": RESIDUAL_VALUES},
        "baselines": BASELINES,
        "mask_hex": hex(&mask_bytes(TASK_ID, &illustrative)),
        "primary_cells": "task_relevance_ppm<=500000",
    })
}

pub struct ReceiptSpec<'a> {
    pub seeds: &'a [u64],
    pub axes: &'a Axes,
    pub frozen: bool,
    pub command: &'a str,
    pub git: &'a GitInfo,
}

fn integrity_totals(per: &[Vec<Vec<Metrics>>]) -> Value {
    let (mut m, mut r, mut q, mut a) = (0, 0, 0, 0);
    for cell in per {
        for x in cell.iter().flatten() {
            m += x.mandatory_violations;
            r += x.replay_inconsistencies;
            q += x.queue_bound_violations;
            a += x.airtime_window_violations;
        }
    }
    json!({"mandatory_violations_total": m, "replay_inconsistencies_total": r,
           "queue_bound_violations_total": q, "airtime_window_violations_total": a})
}

/// Build the complete receipt (with both digests) as a JSON value.
pub fn build_receipt(spec: &ReceiptSpec) -> Result<Value, String> {
    let cells = spec.axes.cells();
    let per = run_cells(&cells, spec.seeds)?;
    let cells_json: Vec<Value> = cells
        .iter()
        .zip(&per)
        .map(|(c, p)| cell_json(c, spec.seeds, p))
        .collect();
    let (config, axes) = (config_json(), axes_json(spec.axes));
    let frozen_src = json!({"config": config, "axes": axes, "seeds": spec.seeds});
    let frozen_digest = sha256_hex(&canonical_json(&frozen_src)?);
    let mut receipt = json!({
        "schema": SCHEMA, "schema_version": 1, "evidence_label": "simulated-deterministic",
        "frozen": spec.frozen,
        "git": {"commit": spec.git.commit, "dirty": spec.git.dirty,
                "dirty_snapshot_sha256": spec.git.dirty_snapshot_sha256},
        "bench": {"crate_version": env!("CARGO_PKG_VERSION"), "command": spec.command},
        "seeds": spec.seeds, "axes": axes, "config": config,
        "frozen_digest_sha256": frozen_digest,
        "cells": cells_json, "integrity": integrity_totals(&per),
    });
    let digest = sha256_hex(&canonical_json(&receipt)?);
    receipt["receipt_sha256"] = json!(digest);
    Ok(receipt)
}
