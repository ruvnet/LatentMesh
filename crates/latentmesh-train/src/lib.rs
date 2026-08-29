//! latentmesh-train — run-2 trained thought-adapter training (ADR-024).
//!
//! M3 scope: the frozen 2-layer MLP projector (2048→512→1536, ReLU,
//! 1,837,056 parameters) trained with AdamW over the M2 per-token paired
//! dump, under ADR-024's leakage discipline (`fit_holdout_split` first, then
//! the 13 probe-overlap rows dropped from whichever side they land in).
//!
//! M4 scope: the frozen low-rank FastGRNN sequence translator
//! (`fastgrnn.rs`, D_in 2048 → D_h 1536, sub-rung ladder r ∈ {64, 128, 256}
//! in ascending order), trained as sequence-to-sequence regression over the
//! same per-token dump consumed as within-item windows — same splits, same
//! 13 exclusions, same receipts discipline.
//!
//! M4c scope (ADR-024 § Registered contingency, mandatory after M4's null):
//! the task-loss ablation rung — the M3 MLP architecture trained through the
//! FROZEN receiver's next-token CE on each item's own generated span
//! (C2C-style), via the composed differentiable BF16 receiver forward in
//! `qwen2_c` (the vendored inference forward silently cuts the graph —
//! measured; see that module's docs) and the task-data assembly in
//! `taskdata` — same splits, same 13 exclusions, same receipts discipline.
//!
//! Evidence honesty: training itself is seeded and deterministic given the
//! captured dump; the dump was produced by live single-host GPU inference
//! (see `run2-pertoken-dump-receipt.json`). Training receipts are labelled
//! accordingly and written BEFORE any frozen-probe invocation — the probe
//! invocation order is the freeze point (ADR-024 § Frozen registration).

pub mod dataset;
pub mod fastgrnn;
pub mod mlp;
pub mod qwen2_c;
pub mod split;
pub mod taskdata;
