# 007. Federated world models

- **Status**: Proposed (design; not wired to a live Radio deployment).
- **Date**: 2026-08-18
- **Related**: [003](003-causal-edge-verification.md) (federation is edge-verification at machine scale), [006](006-self-evolving-topology.md) (placement is a topology decision)

## Context

FedWorld (reported, Aug 3 2026) exchanges structured transition rules between nodes while distinguishing globally-valid knowledge from cluster-specific, private, or unresolved knowledge, reportedly reducing negative transfer vs. blind experience pooling. That maps onto Radio (this stack's coordination layer — bounded control events, not bulk telemetry) plus RuVector (ADR-005) almost directly.

## Decision

Each node maintains a local world model `W_i : (state, action) → state'`. Nodes exchange only the **compatible** portions of each other's learned dynamics, not raw experience:

```
Toronto node learns A
Budapest node learns B
Edge node learns C
        ↓ federation
shared world model
        ↓
local-scope validation
        ↓
A + compatible portions of B and C
```

This is distributed *experiential* intelligence, not distributed inference — the thing being federated is "what did this node learn about how the world behaves," gated by the same admission logic the rest of this stack uses: a candidate rule from another node is only merged if it passes local-scope validation (structurally, the same "does this hold for me?" check ADR-003 runs for edges, applied to transition rules instead of communication edges).

### Selective transmission (the RuView case)

The same principle applies to edge sensing, not just reasoning nodes. Instead of every sensing device reporting raw observations (`CSI → server`, `CSI → server`, `CSI → server`), a node develops a local representation and transmits only when it is informative:

```
Node A detects movement            → transmits
Node B's observation adds nothing  → stays silent
Node C detects a trajectory change → transmits
```

Formally, the same causal question as ADR-003 applied to sensing: transmit only when `I(node's observation; world state | what's already known) > 0` — B's silence is the same decision as an edge failing the four-control test, just made by the sending node instead of the network. This is the mechanism that would let a RuView-style RF-sensing mesh reduce bandwidth without losing information, rather than a heuristic duty-cycle.

### Collective world models as a selective union

```
WorldModel = ⋃_i M_i
```

but the union is selective — only relationships that causally improve prediction survive, using the same admission logic as ADR-003/ADR-006 rather than a separate mechanism. This is deliberately not a new primitive; it's ADR-003's edge test applied to "does node j's transition rule improve node i's prediction" instead of "does agent A's state improve agent B's task value."

### Placement as a function, not a default

Once nodes are heterogeneous (a reasoning GPU node, a Pi 5 doing local sensing, an ESP32 emitting environmental features, a cloud node holding a large model, a node whose role is verification), *where* cognition happens becomes a decision:

```
placement = f(intelligence_gain, latency, bandwidth, privacy, cost, energy, authority)
```

This is a topology decision (ADR-006) at the scale of machines instead of in-process agents — Radio is the transport, ADR-006's `G* = argmax[...]` objective is the same objective with `Cost`/`Latency` terms that now include network hops, and ADR-008's authority model is what stops an unauthorized node from injecting state regardless of measured value.

## Consequences

- Federation reuses ADR-003's admission machinery rather than inventing a second validation mechanism — a transition rule and a communication edge are the same kind of claim ("does this measurably help") at different scope.
- The RuView framing turns "reduce RF-sensing bandwidth" into a specific, falsifiable instance of the general causal-edge question this repo already commits to answering, rather than a separate sensor-fusion heuristic.
- Multi-tenancy matters here specifically: federated transition rules and placement decisions must stay scoped per deployment/tenant (never pooled across tenant boundaries by default), the same constraint this stack already enforces for comms and Buzz topic scoping elsewhere.

## Implementation status

Not implemented this session — no live Radio wiring, no RuView integration. This ADR fixes the "federate only what passes local-scope validation" contract and the placement objective as the integration target for later work.
