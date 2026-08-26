# 017. Radio federation of world-model rules over LatentMesh Air

- **Status**: Accepted and implemented (deterministic simulation + Air envelope transport; over-the-air evidence not claimed).
- **Date**: 2026-08-26
- **Related**: [007](007-federated-world-models.md) (the federation contract this ADR implements), [010](010-latentmesh-air-protocol.md) (LMS1 envelope), [011](011-radio-adapters-and-legal-boundary.md) (legal boundary unchanged), [003](003-causal-edge-verification.md) (admission logic)

## Context

ADR-007 fixed the contract — federate only the compatible portions of each
node's learned dynamics, gated by local-scope validation, over Radio's bounded
control events — and left the wiring open. This repository already has the
bounded control-event transport ADR-007 assumes: the LMS1 envelope + LMAD
delta of LatentMesh Air (ADR-010), with replay defense, CRC, optional
signatures, and a 16–256 byte physical frame budget. A transition rule is
exactly the kind of small, structured, signed payload Air was built to carry.

## Decision

New crate **`latentmesh-federation`**:

- **`TransitionRule`** — `(pre_state_hash, action, post_state_hash, support,
  confidence, scope)` where `scope ∈ {Global, Cluster(id), Private,
  Unresolved}` per FedWorld's knowledge classes. Rules serialize to a compact
  bounded binary form (hard cap ≤ 200 bytes) so one rule fits one Air frame.
- **`WorldModel`** — a per-node table of rules with prediction
  (`predict(pre, action) → post?`) and a held-out transition log.
- **Local-scope validation is ADR-003's test, not a new mechanism**: a
  candidate rule from another node is admitted only if it improves held-out
  prediction accuracy on the *local* log versus (a) no rule, (b) a
  support-shuffled decoy of the same rule, and (c) a scope-mismatched decoy.
  Private-scoped rules are never transmitted; cluster-scoped rules are only
  offered to same-cluster peers. Rejected rules are recorded with the failing
  control, mirroring the gate's audit style.
- **Selective transmission** — `should_transmit(rule, shared_state) → bool`
  implements ADR-007's information-gain rule: a node stays silent when its
  rule's prediction is already implied by what the peer set has acknowledged.
- **Transport** — rules ride LMS1 envelopes via `latentmesh-air-core`
  (deterministic critical bytes = the rule encoding; replay window and CRC
  apply unchanged); the loopback test additionally runs envelopes through the
  `latentmesh-air-radio` modem path. The ADR-011 legal boundary is untouched:
  nothing here emits RF.

## Consequences

- Federation reuses two existing verified layers (gate-style admission, Air
  envelopes) rather than inventing parallel ones — the new code is the rule
  schema, the world model, and the admission harness.
- The negative-transfer claim becomes measurable: the benchmark seeds three
  nodes with overlapping-but-divergent dynamics and reports prediction
  accuracy with blind pooling vs. validated federation (evidence label:
  deterministic simulation — it proves the admission logic discriminates, not
  field behavior).
- Multi-tenancy is structural: `Private` rules cannot leave the node by
  construction (the encoder refuses them), not by caller discipline.

## Implementation status

Implemented this session: rule schema + bounded codec, world model, decoy-
controlled local-scope validation, selective transmission, Air envelope
transport with replay defense, modem loopback test, and the federation-vs-
pooling benchmark. Not claimed: over-the-air federation, RuView RF sensing,
or heterogeneous hardware placement.
