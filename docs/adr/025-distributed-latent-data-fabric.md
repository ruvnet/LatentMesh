# 025. Distributed latent-data fabric (ruvector-replication)

- **Status**: Proposed. Design contract only — no code in this wave.
- **Date**: 2026-08-28.
- **Numbering note**: `docs/research/025-run1-negative-result.md` is a different document (run 1's
  results writeup) that happens to share the number `025`. ADR numbering and
  `docs/research/` numbering are independent sequences in this repo (see e.g. ADR-023 citing
  `docs/research/024`, ADR-024 citing `docs/research/025`/`026`) — no cross-reference is implied
  by the shared digit, and none should be assumed.
- **Related**: [005](005-persistent-latent-memory.md) (the fidelity continuum this ADR proposes
  replicating), [016](016-ruvector-persistent-latent-memory.md) (the embedded, single-host RuVector
  backend this ADR extends toward multi-host), [007](007-federated-world-models.md) (the existing
  federation of *structured rules*, distinct from replicating the memory store itself),
  [024](024-run2-trained-thought-adapter-ladder.md) (the per-token latent shards and training
  receipts this ADR names as replication candidates), [026](026-verified-edge-federation-wire-contract.md)
  (the governed contract for gate-verified causal edges specifically — explicitly **not** covered
  by this ADR's generic replication layer)

## Context

ADR-016 implements `latentmesh-memory`'s `LatentMemory` trait with an in-memory store and an
optional `ruvector-core` (HNSW) backend — both single-host. ADR-005's fidelity continuum (raw
latent trajectory → compressed → semantic prototype → symbolic rule) and ADR-024's run-2 per-token
latent shards (~20.6 GB of captured hidden states, per `docs/research/026-run2-bootstrap-scouts.json`)
both produce artifacts that would benefit from distribution across more than one host as the
project's compute grows beyond a single workstation — this workstation's own tailnet already
spans several peers (`CLAUDE.local.md`: `ruv-mac-mini`, `zenbook`, the V0 cluster). Nothing in this
repo today addresses multi-host replication of the latent-memory store or its artifacts; ADR-007's
federation is a categorically different thing — structured, bounded `TransitionRule`s exchanged
over Air, not the underlying vector store.

**A ready-made candidate already exists and is MSRV-compatible with this workspace, unlike the
GPU-facing crates.** `ruvector-replication` (crates.io `0.1.1`, `README.md` fetched and read this
session from `~/ruvector-upstream/crates/ruvector-replication/README.md`) ships multi-master
replication, quorum writes, a choice of `ConsistencyLevel::{One, Quorum, All}`, conflict resolution
via `ConflictResolution::{LastWriteWins, VectorClock, Custom}`, incremental/compressed delta sync,
and — critically — pins **`rust-version = "1.77"`** (badge confirmed in the fetched README),
meaning it does **not** reintroduce the `half`/MSRV conflict that forces `latentmesh-runtime` and
`latentmesh-train` out of the root workspace (ADR-023 Deviation 1, ADR-024). This is the deciding
fact for adopting it as the *designated* layer rather than a hypothetical one: it is buildable
inside the existing workspace MSRV floor, not a third candle-adjacent standalone crate.

## Decision

**Adopt `ruvector-replication` 0.1.1 as the designated replication layer for LatentMesh's
distributed latent-data fabric, when and if distribution is needed.** This ADR names which
artifacts replicate, under what consistency guidance, and draws one hard boundary explicitly out
of its own scope.

### Which artifacts replicate

| Artifact class | Source | Replication shape |
|---|---|---|
| Latent memory entries | ADR-016 `TrajectoryRecord`/`SymbolicRule`, at `Prototype`/`Rule` fidelity | Mutable, content-versioned; these are the compact, high-value tiers worth replicating broadly |
| Latent memory entries, `Raw`/`Compressed` tiers | ADR-005/016 | Mutable but larger and lower marginal value; replicate under a configured budget, not unconditionally |
| Dataset shards | Run-2 per-token f32 dumps (ADR-024, ~20.6 GB), the S2/S2c pooled dumps | **Immutable once captured** — write-once, content-addressed by their own sha256 (matching every existing receipt's own hashing discipline) |
| Receipts | Every JSON evidence artifact this repo already produces (ADR-014/018/023 discipline) | Immutable, append-only, small; durability is the whole point of replicating these |

### Which artifacts never replicate through this layer

**Gate-verified causal edges (ADR-003's `EdgeVerdict`) are explicitly out of this ADR's scope.**
A verified edge is not inert data — publishing one carries an implicit claim about learned causal
structure that a receiving node could act on with elevated authority (ADR-008). Generic
quorum-write replication has no re-verification step and no signature requirement; it is the wrong
mechanism for something that feeds an authority decision. **[ADR-026](026-verified-edge-federation-wire-contract.md)
is the governed contract for that case** — this ADR draws the boundary and hands the harder problem
to a document built for it, rather than quietly extending a generic replication layer to cover a
governance-sensitive artifact.

### Consistency-level guidance per artifact class

| Artifact class | Consistency level | Conflict resolution | Rationale |
|---|---|---|---|
| `Prototype`/`Rule` memory entries | `Quorum` | `VectorClock` | Compact, high-value, actionable — worth the write-latency cost to keep nodes agreeing; genuine concurrent-promotion races are possible (two nodes promoting the same trajectory to `Rule` independently) |
| `Raw`/`Compressed` memory entries | `One` | `LastWriteWins` | Larger, lower marginal value per entry; eventual consistency avoids paying quorum latency on the bulk of the data |
| Dataset shards | `One` | **N/A — structurally conflict-free** | Content-addressed immutability means two writes of the same shard are byte-identical by construction; there is nothing for a conflict-resolution strategy to resolve, so the full vector-clock/CRDT machinery is unneeded overhead for this class specifically |
| Receipts | `All` | `LastWriteWins` (append-only, no real conflicts expected) | Small, durability-critical, and this repo's entire evidence culture depends on receipts surviving — worth the extra replica cost |

### Tailnet-fleet deployment sketch — illustrative, not normative

Not a commitment to deploy anything. To make the shape concrete: a `replication_factor: 3`
topology using hardware this workstation's own tailnet already has (`CLAUDE.local.md`) could place
the primary on `ruvultra` (this host, where captures and training happen) with replicas on
`ruv-mac-mini` and `zenbook` — both already tailnet-reachable, no new infrastructure implied. This
sketch exists only to show that the deployment shape is achievable with hardware already in
inventory, not to propose scheduling it.

## Consequences

Naming a specific, MSRV-compatible replication crate now — rather than leaving "distribution" as
an unscoped future concern — means a later ADR that actually wires this in has a starting API
surface (`Replicator`, `ReplicationConfig`, `ConsistencyLevel`, `ConflictResolution`) to build
against instead of a fresh crate survey. Drawing the verified-edge boundary here, explicitly, means
`latentmesh-gate`'s authority model is never accidentally weakened by treating a governance-bearing
artifact the same as an immutable dataset shard — a mistake that would be easy to make by extending
this layer's coverage incrementally without re-reading ADR-008.

## What is simulated / what is hardware-pending

| Claim | Status |
|---|---|
| `ruvector-replication`'s API surface and MSRV compatibility | Verified by direct read of the published `README.md` this session — not tested against this repo's build |
| Any actual replication of LatentMesh artifacts | **Not implemented** — no code in this wave |
| The tailnet-fleet deployment sketch | **Illustrative only** — no host has been provisioned, no `Replicator` has been configured anywhere |
| Consistency-level guidance per artifact class | **Design guidance, unvalidated** — no benchmark or failure-injection test has run against any of these choices |

## Implementation status

Not implemented. This ADR is a design contract naming the replication layer and drawing the
verified-edge boundary; no crate, module, or deployment exists yet.
