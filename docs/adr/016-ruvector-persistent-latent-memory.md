# 016. RuVector-backed persistent latent memory

- **Status**: Accepted and implemented (embedded `ruvector-core`; a remote RuVector service remains external).
- **Date**: 2026-08-26
- **Related**: [005](005-persistent-latent-memory.md) (the fidelity-continuum contract this ADR implements), [003](003-causal-edge-verification.md) (what is worth persisting), [006](006-self-evolving-topology.md) (procedural-memory warm starts)

## Context

ADR-005 fixed the trajectory schema `M = {z_t, r_t, c_t, a_t, o_t}` and the
four-tier fidelity continuum (raw → compressed → prototype → rule) but had no
live RuVector wiring. RuVector's embeddable core is now published as
`ruvector-core 2.3.0` (MIT, MSRV 1.77 — equal to this workspace's MSRV), with
`VectorDB` / HNSW indexing / optional redb persistence. Depending on the lean
feature set (`default-features = false`, `hnsw`) keeps the dependency graph
small and avoids pulling `reqwest`, SIMD, or storage backends into a research
prototype by default.

## Decision

New crate **`latentmesh-memory`**:

- **`LatentMemory` trait** — `store` / `store_compressed` / `recall` /
  `compress_to` / `promote_to_prototype` / `len` over
  `TrajectoryRecord { id, latent: Vec<f32>, reward, context_hash, action,
  outcome, causal_value, fidelity, reconstruction_error, parents }`. The trait
  is the contract; two backends ship:
  - `InMemoryStore` — deterministic brute-force cosine store, zero extra
    dependencies, always compiled (tests, WASM-adjacent contexts).
  - `RuVectorStore` — behind the `ruvector` feature (exercised in CI and
    benchmarks on stable; kept off the MSRV check path so the workspace's
    1.77 floor is not hostage to a third-party dependency graph), backed by
    `ruvector_core::VectorDB` with HNSW + cosine distance; trajectory metadata
    rides in the entry's metadata map, so recall returns full records, not
    bare vectors.
- **Fidelity is explicit and downward-only per record**:
  `Fidelity::{Raw, Compressed, Prototype, Rule}`. `compress_to` re-encodes the
  latent through `latentmesh-core`'s real quantizers (`F16`, then `Int8`) so
  the fidelity loss is the measured quantization error, not an unspecified
  transform; each step records the reconstruction error so ADR-005's
  "deliberate, measured tradeoff" is a number in the record.
- **Admission policy**: only records whose `causal_value` clears a configured
  floor are admitted to the Raw tier — a below-floor record is *rejected* at
  Raw (not silently downgraded) and must be stored via `store_compressed`.
  The value itself is caller-supplied: measuring ΔV is `latentmesh-gate`'s job
  upstream (ADR-003); this crate enforces the floor, it does not re-measure.
  Storage cost is bounded by demonstrated usefulness, per ADR-005.
- **Prototype + rule promotion**: `promote_to_prototype` folds k members of a
  problem family into a centroid prototype with lineage (both backends), and a
  prototype reused successfully ≥ N times can be promoted to a
  `SymbolicRule { family, rule, promoted_from, uses }` retrievable without
  touching the vector index (rule promotion and the rules list live on
  `InMemoryStore` today).
- **Procedural memory**: `TopologyRecord { family, agent_sequence,
  successful_uses }` is stored on `InMemoryStore` keyed by family label,
  giving ADR-006's topology search its warm start. Promoting topology records
  and rules onto the backend-agnostic trait (so the RuVector backend carries
  them too) is named follow-up work, not claimed here.

## Consequences

- The recall-quality question ADR-005 names ("does prototype-retrieval beat
  document-retrieval on repeat families?") is now testable in-repo; the
  benchmark added here measures recall precision and latency for both backends
  (evidence label: host software benchmark on synthetic trajectories).
- `ruvector-core`'s HNSW is approximate; tests that need determinism use
  `InMemoryStore`, and the RuVector-backed tests assert set-level recall, not
  exact ordering.
- Recalled latents re-enter the mesh as *candidate* state: recall returns
  records, never authority. Nothing in this crate grants execution rights —
  it is the caller's obligation (enforced by `latentmesh-gate` at the point
  of use, per ADR-008) to replay recalled state through admission at
  `ObserveOnly` like any other inbound frame.

## Implementation status

Implemented this session: the trait, both backends, quantized compression with
measured error, prototype/rule promotion, topology records, tests for both
backends, and the recall benchmark. Not claimed: a remote/multi-tenant
RuVector deployment, learned (non-quantization) compression, or real agent
trajectories.
