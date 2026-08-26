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

- **`LatentMemory` trait** — `store` / `recall` / `compress_to` / `promote` /
  `len` over `TrajectoryRecord { frame_id, latent: Vec<f32>, reward, context_hash,
  action, outcome, fidelity, causal_value }`. The trait is the contract; two
  backends ship:
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
- **Admission policy**: only records whose `causal_value` (ADR-003's ΔV, as
  measured by `latentmesh-gate`) clears a configured floor are eligible for the
  Raw tier; everything else enters at Compressed or below. Storage cost is
  bounded by demonstrated usefulness, per ADR-005.
- **Prototype + rule promotion**: `promote` folds k members of a problem
  family into a centroid prototype (nearest-prototype recall), and a prototype
  reused successfully ≥ N times can be promoted to a `SymbolicRule { family,
  topology, uses }` retrievable without touching the vector index.
- **Procedural memory**: `TopologyRecord { family, agent_sequence }` is stored
  through the same trait (family embedding as the key), giving ADR-006's
  topology search its warm start.

## Consequences

- The recall-quality question ADR-005 names ("does prototype-retrieval beat
  document-retrieval on repeat families?") is now testable in-repo; the
  benchmark added here measures recall precision and latency for both backends
  (evidence label: host software benchmark on synthetic trajectories).
- `ruvector-core`'s HNSW is approximate; tests that need determinism use
  `InMemoryStore`, and the RuVector-backed tests assert set-level recall, not
  exact ordering.
- Recalled latents re-enter the mesh as *candidate* state: recall never
  returns authority — a recalled trajectory is replayed through the gate at
  `ObserveOnly` like any other inbound frame (ADR-008's "authority never
  expands" applies to memory too).

## Implementation status

Implemented this session: the trait, both backends, quantized compression with
measured error, prototype/rule promotion, topology records, tests for both
backends, and the recall benchmark. Not claimed: a remote/multi-tenant
RuVector deployment, learned (non-quantization) compression, or real agent
trajectories.
