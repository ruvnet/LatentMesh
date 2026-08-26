# 018. MetaHarness Darwin loop for topology evolution

- **Status**: Accepted and implemented (deterministic simulation with evidence-labelled JSON receipts; live multi-agent deployment not claimed).
- **Date**: 2026-08-26
- **Related**: [006](006-self-evolving-topology.md) (the objective and acceptance test this ADR implements), [003](003-causal-edge-verification.md) (fitness primitive), [008](008-capability-governed-execution.md) (authority invariant), [014](014-benchmark-and-acceptance-method.md) (evidence-label method)

## Context

ADR-006 fixed the objective `G* = argmax_G [Quality(G) − λ·Cost(G) −
μ·Latency(G) − ρ·Risk(G)]` with causal `Quality` and the ten-agent acceptance
test, and left the Darwin loop unwired. The MetaHarness side of this
repository (`harness/air`) already established the pattern the loop needs:
frozen policy domains, deterministic seeds, and signed benchmark receipts that
say exactly what a run does and does not prove. This ADR applies that same
harness discipline to topology evolution instead of the radio link.

## Decision

New crate **`latentmesh-evolve`** plus a MetaHarness suite:

- **`Topology`** — `(active_agents, edges, authority_caps)` with per-edge
  `Encoding` and `Authority` (measured ΔV lives in the evaluation's
  `FitnessBreakdown`, not on the edge — it is a property of an edge *in an
  environment*); `Mutation ∈ {AddEdge, RemoveEdge, RaiseAuthority,
  LowerAuthority, ChangeEncoding, DeactivateAgent, ReactivateAgent}`.
  Mutations are proposals; one that would raise any edge's authority above
  its constitution cap is rejected before evaluation (`authority never
  expands` is enforced in the mutation applier, not the fitness function).
- **Fitness is ADR-006's objective, with causal Quality**: an edge contributes
  to `Quality(G)` only if its ΔV survives `latentmesh-gate`'s five-control
  causal test (zero / random / mismatched / self-generated / text-equivalent)
  in the evaluation environment; an unverified edge contributes zero
  regardless of apparent correlation. Because the environment is frozen, an
  edge's verdict is a property of the edge, so verdicts are cached per run
  (`VerificationCache`) — the "verification bounds the search" cost is
  engineered down and still reported in the receipt. Governance-mandatory
  edges are never auto-removed even at ΔV ≈ 0, per ADR-006's
  control-vs-cognition distinction.
- **The loop is deterministic**: seeded RNG, frozen task environment
  (a synthetic ten-agent environment with deliberately redundant agents whose
  true contribution structure is known to the environment, hidden from the
  optimizer), generation-by-generation evaluation, elitist selection — then a
  deterministic **audit sweep**: after the last generation, every unverified
  non-mandatory edge and every edge stranded on an inactive agent is cut and
  the result re-evaluated (kept only if fitness does not regress). The sweep
  is what *guarantees* the every-surviving-edge-verified acceptance property;
  random search merely usually finds those cuts first.
- **Warm starts from procedural memory**: when `latentmesh-memory` holds a
  `TopologyRecord` for the recognized problem family, the initial population
  is seeded from it (ADR-005/006 coupling).
- **MetaHarness receipts**: each run emits a plain (unsigned) JSON receipt —
  seed, generations, fitness/compute before and after, mutation and
  constitution-rejection counts, verification evaluations, the acceptance
  report, the evidence label `"simulated"`, and an explicit `not_claimed`
  list — in the same spirit as `harness/air`'s stage-separated benchmark
  receipt (which is flywheel-signed; signing evolve receipts is follow-up). A new `harness/evolve`
  package (same MetaHarness discipline, zero external dependencies) runs the
  suite via the Rust binary and verifies the receipt against ADR-006's
  acceptance bounds.

## Consequences

- ADR-006's acceptance test is now executable: the shipped environment checks
  that evolution converges to a smaller graph with ≥30% lower compute proxy at
  non-decreasing task success, with every surviving edge individually passing
  the five-control test. Passing it proves the loop optimizes the causal
  objective in simulation — it does not prove real-agent gains, and the
  receipt says so.
- Verification cost is visible in the receipt (control evaluations per
  generation), making ADR-006's "verification bounds the search" consequence a
  reported number.
- The `harness/` directory becomes the home for cross-suite MetaHarness
  evidence (air + evolve), keeping evidence-label discipline in one place.

## Implementation status

Implemented this session: the crate (topology, mutations with the authority
invariant, causal fitness, deterministic Darwin loop, warm starts), the
receipt emitter, the harness suite script, tests, and the acceptance-bound
check. Not claimed: live model routing, real agent workloads, or bandwidth
control of a production MidStream deployment.
