# 005. Persistent latent memory

- **Status**: Proposed (design; not wired to a live RuVector instance).
- **Date**: 2026-08-18
- **Related**: [002](002-latent-packet-protocol.md), [003](003-causal-edge-verification.md) (only causally-verified trajectories are worth persisting), [006](006-self-evolving-topology.md) (procedural memory feeds topology recall)

## Context

Do not throw away successful latent trajectories. RuVector already exists in this stack as a persistent, adaptive memory substrate (local semantic embeddings, persistent vector retrieval, graph relationships, explicit feedback learning, memory lifecycle controls) — it is the natural store for latent trajectories, not just text memories. CrystalMem (reported) identifies *memory hysteresis* — capability that does not fully recover after memory compression — and introduces reversible levels of memory fidelity, reportedly a 4.6pp average advantage at equal memory budgets and matching the strongest full-provision baseline at a 50% memory budget in its tested environments.

## Decision

Store successful latent trajectories as:

```
M = { z_t, r_t, c_t, a_t, o_t }
```

— latent state, reward, context, action, outcome — and maintain a **continuum of fidelity levels** in RuVector rather than one fixed representation:

```
raw latent trajectory → compressed latent trajectory → semantic prototype → symbolic rule
```

- **Raw** — the full `z_t` sequence, kept only for trajectories with high measured causal value (ADR-003) and a bounded retention window; this is the expensive tier.
- **Compressed** — a lossy but *reversible-fidelity* reduction (per CrystalMem's finding, motivating an explicit, chosen compression level rather than one irreversible collapse), so capability lost to compression is a deliberate, measured tradeoff, not silent decay.
- **Semantic prototype** — a small number of representative latents per problem family (nearest-prototype retrieval, RuVector's existing graph/vector machinery), replacing "retrieve documents about how this was solved" with "retrieve something closer to the computational state that produced the solution":

  ```
  query → documents → reasoning        (today)
  query → cognitive state → continuation   (this ADR)
  ```

- **Symbolic rule** — once a prototype has been reused successfully enough times, promote it to an explicit rule (the neuro-symbolic continuum this stack has been circling), retrievable without touching the vector store at all.

**Persistent computational skills, not just memories.** RuVector should store *topologies*, not only latent vectors — a successful `problem family → agent sequence` mapping (e.g. "distributed Rust debugging → planner → rust-specialist → test-generator → security-reviewer → verifier") is procedural memory: `memory + execution structure + communication structure`, restorable as a starting topology the next time a similar problem arrives (feeds ADR-006's topology search a warm start instead of a cold one).

## Consequences

- Memory budget becomes a chosen fidelity level per trajectory, not a single global compression ratio — expensive raw storage is reserved for the trajectories ADR-003 has actually verified as causally valuable, which bounds cost by demonstrated usefulness rather than by volume.
- "Reusable thought" is a testable retrieval-quality question (does prototype-retrieval-then-continue beat document-retrieval-then-reason on repeat problem families?) independent of the compression-fidelity question — this ADR keeps them as two named, separately measurable claims rather than one bundled assertion.
- Storing topologies as procedural memory directly couples this ADR to ADR-006: a topology search that ignores prior successful structures is strictly more expensive than one that starts from a remembered warm start for a recognized problem family.

## Implementation status

Not implemented this session — no live RuVector wiring. This ADR fixes the four-tier fidelity contract and the `M = {z_t, r_t, c_t, a_t, o_t}` trajectory schema as the integration target; `LatentFrame`'s `provenance.parents` (ADR-002) already carries the lineage chain a compression step would need to preserve.
