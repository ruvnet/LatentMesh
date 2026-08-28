//! latentmesh-train — run-2 trained thought-adapter training (ADR-024).
//!
//! M3 scope: the frozen 2-layer MLP projector (2048→512→1536, ReLU,
//! 1,837,056 parameters) trained with AdamW over the M2 per-token paired
//! dump, under ADR-024's leakage discipline (`fit_holdout_split` first, then
//! the 13 probe-overlap rows dropped from whichever side they land in).
//!
//! Evidence honesty: training itself is seeded and deterministic given the
//! captured dump; the dump was produced by live single-host GPU inference
//! (see `run2-pertoken-dump-receipt.json`). Training receipts are labelled
//! accordingly and written BEFORE any frozen-probe invocation — the probe
//! invocation order is the freeze point (ADR-024 § Frozen registration).

pub mod dataset;
pub mod mlp;
pub mod split;
