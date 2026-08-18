# 006. Self-evolving topology

- **Status**: Proposed (design; not wired to a live MetaHarness Darwin loop).
- **Date**: 2026-08-18
- **Related**: [003](003-causal-edge-verification.md) (the fitness signal this ADR consumes), [005](005-persistent-latent-memory.md) (procedural-memory warm starts), [001](001-latentmesh-architecture-and-prior-art.md) §3 (DMoA prior art), [009](009-online-causal-control-loop.md) (supersedes this ADR's `G_t=(A,E,R,M,H)` notation with `G_t=(A,E,Z,M,P)` — `Z` subsumes latent transforms `R`, `P` subsumes harness policy + execution authority `H`; read `R`/`H` below as the engineering-level decomposition of `Z`/`P`)

## Context

DMoA (reported) already dynamically activates agents and alters communication topology during reasoning — dynamic topology alone is not new (ADR-001 §3). What this stack can add is a MetaHarness Darwin mutation loop whose fitness signal is **measured causal edge value** (ADR-003), not token traffic, agent count, or a raw benchmark delta a skeptical reader can attribute to extra compute.

The reframe: most multi-agent systems today optimize *activity* — more agents, more messages, more tool calls, more tokens. This ADR optimizes *marginal intelligence contribution*.

## Decision

**The topology is a Darwin-evolved object, and its fitness function is causal, not superficial.**

```
G_t = (A, E, R, M, H)
```

where `A` = agents, `E` = communication edges, `R` = latent transformations (ADR-002), `M` = memories (ADR-005), `H` = harness policies. Darwin mutations may change agent selection, model selection, latent dimensionality, communication topology, stream timing (ADR-004), memory retrieval/compression (ADR-005), verification (ADR-003), and routing — the system is not learning answers, it is learning its own cognitive architecture.

### The optimization objective

```
G* = argmax_G [ Quality(G) − λ·Cost(G) − μ·Latency(G) − ρ·Risk(G) ]
```

`Quality(G)` is not raw task success — it is built from the set of edges in `G` that individually pass ADR-003's four-control test; an edge that fails verification contributes nothing to `Quality` regardless of what it appears to correlate with. `Risk(G)` is bounded by ADR-008's authority model — Darwin can propose a topology change, it cannot grant itself execution authority the constitution doesn't allow (mirroring the AGL "authority never expands" invariant from `cognitum-one/slack`'s ADR-0008).

### Agents that disappear when useless

Traditional orchestration assumes every configured agent has value. This system should instead *measure* it, per candidate edge, e.g.:

```
planner → coder       +12.4%   (kept, strengthened)
researcher → coder     +7.1%   (kept)
critic → coder         +1.0%   (marginal — reweighted, not auto-removed if mandatory)
memory → planner       +9.8%   (kept, strengthened)
security → coder       +0.2%   (cognitively near-zero, but may be governance-mandatory)
verifier → coder       +6.4%   (kept)
```

The last two rows are the point of ADR-003's "mandatory control vs. useful cognition" distinction: `security → coder` can measure near-zero cognitive contribution and still be non-removable because its purpose is control, not intelligence. Conflating the two is the failure mode this ADR explicitly refuses.

### A real mixture of agents

A transformer MoE learns `router(x) → experts`. This system's router should learn:

```
router(state, history, cost, risk) → agents
```

Only agents whose activation is currently justified by measured value participate; effective compute scales with task complexity rather than with the number of agents configured — if 20 agents exist but only 4 currently contribute, the system pays roughly for those 4, not for coordinating all 20. A reasonable first target (stated as a target, not a measured result) is 30–60% lower agent compute vs. static orchestration at equal task quality — this is exactly what the acceptance test below checks the lower bound of.

### Bandwidth allocated by cognitive value

```
Bandwidth_{ij} ∝ ΔIntelligence_{ij} / Cost_{ij}
```

An edge with high, verified `ΔV` (ADR-003) can be given a larger share of MidStream's transport budget (ADR-004) and a richer encoding (`Encoding::F32`/`F16` — ADR-002); a rarely-useful edge is compressed harder (`Encoding::Int8`) or throttled. This gives MidStream's existing chunk/transport machinery a principled allocation input instead of a fixed per-stream budget.

### Learned model routing by downstream contribution

Instead of routing by price/latency/benchmark-class heuristics alone, the router can learn *empirically* which model contributes where — e.g. one model contributes most during architecture generation, another during code repair, a small local model contributes little to planning but is efficient at summarization — from the same `ΔV` measurements this ADR already requires for edges. This is a MetaHarness primitive candidate, not implemented here.

## Consequences

- The topology-search fitness function is defensible against the ADR-001 §5 objection by construction: it cannot reward an edge that only *appears* to help, because appearing-to-help without surviving the four controls contributes zero to `Quality(G)`.
- Verification cost (ADR-003) is now also a topology-search cost: every candidate edge Darwin proposes needs to clear the four-control test before its `ΔV` can be trusted as a fitness input, which bounds how many candidate edges can be explored per generation.
- Procedural memory (ADR-005) is not optional infrastructure for this ADR — a topology search with no warm start re-discovers the same edges every time a recognizable problem family recurs, which is strictly worse than recalling the last successful `G` for that family and mutating from there.

## Acceptance test

Shared with ADR-003: ten agents with deliberately redundant capabilities, 1,000 tasks, self-mutating topology. **Pass** if the topology converges toward a smaller graph reducing compute by ≥30% while maintaining or improving task success, and every surviving edge individually passes ADR-003's counterfactual replacement test. See ADR-003's Acceptance test section for why both halves are required together.

## Implementation status

Not implemented this session — no live Darwin loop. `latentmesh-gate`'s causal module (ADR-003) is the fitness primitive this ADR would wire into a MetaHarness Darwin mutation loop; that wiring is open follow-up work.
