# 026. Verified-edge federation wire contract

- **Status**: Proposed. Design contract only — no code in this wave.
- **Date**: 2026-08-28.
- **Numbering note**: `docs/research/026-run2-bootstrap-scouts.json` is a different document (the
  M0 scout evidence for ADR-024) that happens to share the number `026`. ADR numbering and
  `docs/research/` numbering are independent sequences — no relationship is implied by the shared
  digit.
- **Related**: [003](003-causal-edge-verification.md) (the five-control test and `EdgeVerdict`
  this contract carries), [007](007-federated-world-models.md) (the existing rule-federation
  pattern this contract parallels, for a different payload type), [008](008-capability-governed-execution.md)
  (the authority model this contract must never bypass), [010](010-latentmesh-air-protocol.md)/
  [017](017-radio-federated-world-models.md) (one candidate bounded transport), [020](020-agentbbs-store-and-forward-bridge.md)
  (another candidate transport, and the payload-boundedness precedent it already established),
  [021](021-cognitum-one-api-integration.md) (the "servers discard self-reported verification
  claims" principle this contract adopts), [025](025-distributed-latent-data-fabric.md) (the
  generic replication layer this contract is explicitly *not* built on — verified edges need
  governance, not just replication)

## Context

ADR-003's `verify_edge` produces an `EdgeVerdict` — `Admit{delta_v, worst_p_value}` or
`Reject{reason}` — after an edge survives (or fails) the five-control causal test. Today that
verdict lives on one node. If LatentMesh nodes ever federate what they've learned about which
edges are causally valuable — the natural next step once more than one node runs the causal gate
— the wire contract for that exchange needs the same governance discipline ADR-025 explicitly
declined to provide for this exact artifact class. A ready-made, near-exact wire shape for this
already exists as prior art in a sibling project on this host.

**`agentdb`'s QUIC synchronization architecture defines `CausalEdgeSync`** (fetched and read this
session, `~/projects/agentdb/docs/quic/QUIC-ARCHITECTURE.md`):

```protobuf
message CausalEdgeSync {
  uint64 edge_id = 1;
  uint64 from_memory_id = 2;
  uint64 to_memory_id = 3;
  float uplift = 4;
  float confidence = 5;
  VectorClock version = 6;
  ConflictResolutionMetadata conflict_metadata = 7;
}
```

carried inside a `SyncMessage` envelope (`sequence_number`, `timestamp_ms`, `node_id`,
`vector_clock`, oneof payload) on its own dedicated QUIC stream (`Stream 3: Causal edge sync (low
priority)`), with a documented operational-transform conflict-resolution rule for the same-edge
case: weighted-average `uplift` by sample size, `max()` on `confidence`, union `evidenceIds`. This
is materially the same shape ADR-003's `ΔV`/`worst_p_value` need to travel in, and the OT merge
rule is a directly reusable pattern for reconciling two nodes' independently-measured `ΔV` for the
same nominal edge.

## Decision

**Adopt `CausalEdgeSync`'s shape as reference prior art, adapted to carry LatentMesh's own
verification artifacts, not agentdb's.** The adapted message:

| Field | Source | Purpose |
|---|---|---|
| `edge_id` | as `CausalEdgeSync` | Identity |
| `from_node`, `to_node` | adapted from `from_memory_id`/`to_memory_id` | LatentMesh nodes, not agentdb memory ids |
| `delta_v` | **new, replaces `uplift`** | ADR-003's measured causal value — the actual quantity this repo's gate reasons about |
| `worst_p_value` | **new** | The gate's own "significance against the most favorable control" statistic (`latentmesh-gate::causal::verify_edge`'s documented stricter-than-mean-of-controls bar) — carried explicitly so a receiver can judge evidence strength, not just a pass/fail bit |
| `decoy_control_provenance` | **new** | Which of the five controls (`zero`/`random`/`mismatched`/`self_generated`/`text_equivalent`) were run and each one's own p-value — enough for a receiver to *audit* the claim, not merely trust it |
| `authority_ceiling` | **new** | The `Authority` tier (`ContextInject`/`LatentPrefix`/`ActionInfluencing`) the sending node computed via `ceiling_from_verdict` — informational only; see the authority rule below |
| `version` | as `CausalEdgeSync`'s `VectorClock` | Causal ordering across nodes |
| `conflict_metadata` | as `CausalEdgeSync` | Reuses agentdb's OT merge pattern: weighted-average `delta_v` by trial count, `max()` on confidence-equivalent (here, minimum-of-worst-p-values, since *lower* p is stronger evidence — the merge direction is inverted from `uplift`'s "bigger is better", a deliberate, documented adaptation, not a copy error) |
| `signature` | **new** | Ed25519, consistent with LMS1's `SIGNED_ENVELOPE` model and this repo's existing device-key precedent (ADR-021's `/v1/seed/register` canonical-request signing) |

**Only edges that PASSED the ADR-003 five-control test may be published.** `Reject` verdicts never
leave the originating node — this mirrors ADR-017's `Private`-scope rule exactly (`TransitionRule`
federation: "encoder-refused, decoder-rejected"), applied here to the admission verdict itself
rather than to a rule's declared scope.

**Federated claims are hints, never authority.** A receiving node re-verifies the edge locally
(re-running its own five-control test against its own local task distribution) before granting any
authority above `ObserveOnly` — a federated `Admit` verdict, however strongly evidenced, is not
itself sufficient grounds for a receiver to skip its own verification. This is the same principle
ADR-021 already cited from `cognitum-one/api`'s ADR-099: **"self-reported hardware verification
claims are discarded server-side; a device cannot mark its own hardware verified."** Applied here:
a remote node cannot mark its own edge verified in a way that binds a receiver's authority
decisions — `execute(z) ⟺ signature(z) ∧ authority(z) ∧ provenance(z) ∧ risk(z) < τ` (ADR-008) still
requires the *receiver's* own evaluation of `authority(z)`, and a federated `EdgeVerdict` is
provenance input to that evaluation, not a substitute for it.

**Transport-agnostic, payload bounded.** This wire contract does not name a required transport —
it could ride the agentbbs bridge (ADR-020, as a new `FederationPayload` variant or embedded inside
`ReplicateMessage`'s opaque payload), a native QUIC transport (agentdb's own precedent), or
LatentMesh Air's LMS1 envelope (ADR-010/017). **Sizing check against the Air constraint, since Air
is the tightest of the three**: `edge_id` (8B) + `from_node`/`to_node` (8B each) + `delta_v` (4B,
f32) + `worst_p_value` (4B) + `decoy_control_provenance` (5 controls × 4B p-value + 1B ran-flag ≈
25B) + `authority_ceiling` (1B) + a compact vector-clock representation (small map, bounded by node
count — budget ~16B for a handful of peers) + `signature` (64B, Ed25519) ≈ **~140 bytes** —
comfortably inside Air's native 240-byte payload ceiling (ADR-010) with no fragmentation required,
and well inside the Meshtastic adapter's tighter 217-byte usable budget (ADR-019) too. This is a
useful property to note explicitly: this contract, unlike LMAD deltas carrying residual data, never
needs Air's multi-fragment path.

## Consequences

This ADR gives a future edge-federation ADR a concrete wire shape to implement against instead of
inventing one from scratch, and — more importantly — it states the governance rule (re-verify
locally, never trust a remote self-report) before any federation code exists, the same sequencing
discipline ADR-021 already modeled for the cognitum-one integration. Choosing a transport-agnostic
contract keeps this decision decoupled from ADR-020's bridge, ADR-010's Air stack, or any future
QUIC work — whichever transport actually ships first, the message shape doesn't need to change.

## What is simulated / what is hardware-pending

| Claim | Status |
|---|---|
| `CausalEdgeSync`'s protobuf shape and agentdb's OT merge rule | Verified by direct read of `agentdb/docs/quic/QUIC-ARCHITECTURE.md` this session |
| The adapted LatentMesh message's field-by-field sizing (~140 bytes) | Arithmetic from field widths, not measured against a real serialization |
| Any actual federation of verified edges between LatentMesh nodes | **Not implemented** — no code in this wave, no live multi-node deployment exists |
| Local re-verification of a federated claim | **Not implemented** — this is a stated rule, not yet enforced by any code path |

## Implementation status

Not implemented. This ADR is a design contract for the wire shape and the governance rule; no
crate, module, or transport binding exists yet.
