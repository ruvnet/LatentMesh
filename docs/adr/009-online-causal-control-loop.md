# 009. The online causal control loop — and what each ruvnet component contributes

- **Status**: Proposed. **The statistical primitive (ADR-003) and admission gate (ADR-008) are implemented; the closed loop across live components is not wired.**
- **Date**: 2026-08-18
- **Related**: [001](001-latentmesh-architecture-and-prior-art.md), [003](003-causal-edge-verification.md), [005](005-persistent-latent-memory.md), [006](006-self-evolving-topology.md), [008](008-capability-governed-execution.md)
- **Provenance note**: as ADR-001. The prior-art table below reflects a second literature pass (2026-08-18) that materially narrowed the novelty claim from ADR-001's first draft — read as a correction, not an addendum.

## 1. The claim has moved twice in one day — say so

ADR-001 (first pass) proposed "a dynamically evolving graph of causally useful agents" as the whitespace. A second literature pass found that claim is now also substantially occupied:

| Capability | Prior work (reported) | Novelty (1–5) |
|---|---|---|
| Latent agent communication | StateBridge, LatentMAS, KV methods | 1 |
| Heterogeneous latent transfer | StateBridge; dense heterogeneous latent communication (reported Qwen↔Qwen, ~2–3× lower compute in context-aware settings) | 2 |
| Dynamic communication topology | MANTA (reported +5.8pp over its strongest baseline), DyTopo, SkillGraph | 1 |
| Agent marginal contribution | Removal attribution, Shapley-style methods | 1 |
| Causal communication attribution | **E2 Explainer** (reported 2026-08-13 — masks channels, measures outcome change, identifies critical subgraphs, can execute the reduced graph) | 2 |
| Value-aware communication pruning | **BANDMAS** (reported 53.2–77.3% application-traffic reduction at selected constraints) | 2 |
| Distributed latent state | AAFLOW+ | 2 |
| Online causal *control* of a distributed latent topology | none found | 4 |
| Persistent, self-evolving, causal cognitive mesh | none found | 5 |
| Same, plus runtime authority and provenance | none found | 5 |

**Consequence**: "identify which agent edges causally matter and remove useless ones" is no longer a defensible novelty claim — E2 Explainer reportedly published essentially that four days before this ADR. "Let the network restructure itself during execution" is also taken (MANTA, reported). What remains open, per this pass, is treating causal edge value as a **live control input** — not a post-hoc explanation, not a one-shot pruning pass — that continuously and persistently reconfigures a topology whose agents also hold typed execution authority. This ADR names that the actual, narrower target, and states self-assessed confidence honestly rather than implying certainty (§5).

## 2. The reframed object and loop

```
G_t = (A, E, Z, M, P)
```

- `A` — computational agents
- `E` — active cognitive connections (the topology)
- `Z` — transferable latent state (supersedes ADR-006's `R`: a transform is one instance of transferable state, not the whole of it)
- `M` — persistent experiential memory (ADR-005)
- `P` — execution authority and policy (supersedes ADR-006's `H`: harness policy and execution authority are the same governance surface — ADR-008)

Each connection carries an empirical value, not an assumed one:

```
C_ij = E[V | correct state_ij]  vs.  E[V | matched controls]
```

— the operational form ADR-003 already implements (`ΔV`, five controls including `text_equivalent`). What ADR-003 alone does not say, and this ADR commits to: **the measurement controls future execution, continuously, not once**:

```
execute
  ↓
transfer latent state
  ↓
counterfactual audit           (ADR-003)
  ↓
measure causal value
  ↓
update edge value / authority ceiling   (ADR-008 Policy)
  ↓
persist result                 (ADR-005)
  ↓
change topology                (ADR-006, Darwin)
  ↓
next execution
```

This closed loop — not any single stage of it — is the systems contribution this ADR claims. Every stage already has a named ADR; this one is the statement that they compose into one online control loop, not seven independent designs.

## 3. What each existing ruvnet component becomes in this loop

The premise this ADR pushes back on: the ruvnet stack does not need a new capability. It needs one falsifiable thesis connecting primitives that already exist. Read as `Sense → Represent → Route → Reason → Verify → Remember → Evolve`:

| Component | Today | In this loop |
|---|---|---|
| **RuFlo** | Orchestrator: `researcher → planner → coder → reviewer` | Manages `G_t = (agents, models, memory, tools, latent channels)`; edges are scored by `ROI_edge = ΔTaskValue / (Tokens + GPUTime + Latency + NetworkCost)`, so adding an agent is a decision RuFlo can justify or refuse, not a static pipeline definition. |
| **MetaHarness / Darwin** | Mutation → benchmark → score | The harness is the genotype (models, agents, tools, permissions, memory policy, topology, latent transforms, verification); Darwin's fitness signal becomes `mutation → execution → intervention → causal attribution (ADR-003) → persistence (ADR-005) → mutation`, so Darwin selects on demonstrated computational contribution, not benchmark correlation. |
| **RuVector** | Memory of successful *information* | Memory of successful *computation*: `Memory = Content + State + Topology + Outcome + Causality` — a stored record names which topology and which edges (with their measured `ΔV`) produced a given outcome, not just the text of the outcome (ADR-005's procedural-memory extension). |
| **MidStream** | `inspect(computation_finished)` — post-hoc token-stream analysis | `control(computation_t)` — the same chunk-level intervention machinery already used for token streams (ADR-004), extended to intervene on latent state, confidence, tool results, and causal probes while computation is still in flight. |
| **Radio** | Message bus; agents subscribe, senders broadcast | Cognitive networking: `priority_ij = f(causalValue, latency, confidence, bandwidth, cost, trust)` (ADR-007) — a node transmits because historical evidence predicts the recipient can use the information, not because a channel exists. |
| **RVF / RVM** | Portable artifact packaging / capability-isolated execution | The answer to a problem the cited literature largely leaves open: latent payloads aren't human-inspectable, so `Accept(z) ⟺ Identity ∧ Signature ∧ Authority ∧ Provenance ∧ Policy ∧ Risk<τ` (ADR-008) is enforced as a capability grant, and RVF carries model identity, transform, permitted recipients, provenance, and witness history as the artifact's own metadata — zero-trust latent cognition. |
| **RuView** | Raw RF/CSI observation reporting | A causal sensing participant: a node transmits only when `I(node's observation; world state | what's already known) > 0` (ADR-007's selective-transmission case); physical outcomes (did the predicted event actually happen) become an external truth signal closing `predict → observe → error → memory → adapt`. |
| **Autogenous** | "Agents that run forever" (an under-specified framing) | A technically testable definition: `Topology_{t+1} = f(Topology_t, Experience_t)` — accumulated computational knowledge (the causal graph, procedural memory, learned routing) survives replacement of any individual model, agent, or machine. Identity of the system ≠ identity of any model in it. |

## 4. Cognitum's positioning, restated

Not "another agent platform." A layered runtime where models are the interchangeable bottom layer, not the product:

```
MODEL LAYER          OpenAI / Anthropic / Google / open weights
        ↓
COGNITIVE RUNTIME     RuFlo · MetaHarness · MidStream
        ↓
INTELLIGENCE STATE    RuVector · causal graph · world model
        ↓
DISTRIBUTED FABRIC    Radio · edge · cloud · devices
        ↓
TRUST                 RVF · RVM · provenance · policy
        ↓
PHYSICAL WORLD        RuView · routers · sensors · appliances
```

"Cognitum governs where intelligence executes, what it can communicate, what it remembers, and how it is allowed to evolve" — a claim about the runtime, not about any one model or wire format, which is also why ADR-001 §3's finding (AVP/LatentMAS/AAFLOW+ already own the wire-format and raw-transfer ground) doesn't undercut this positioning: none of the cited prior art claims the governance/persistence/evolution layers together.

## 5. The one experiment, not a dozen repos

Explicit decision: **do not launch further speculative repositories.** Implement one vertical experiment across the *existing* stack:

```
3 heterogeneous models → MidStream state capture → Radio transport
  → RuVector persistence → causal intervention (ADR-003)
  → MetaHarness topology mutation → Darwin selection → RVF provenance
```

against one benchmark, comparing four conditions:

| Condition | Communication | Topology |
|---|---|---|
| `StaticText` | Text | Fixed, hand-designed |
| `DynamicText` | Text | Darwin-mutated |
| `DynamicLatent` | Latent (`LatentFrame`) | Darwin-mutated |
| `CausalDynamicLatent` | Latent | Darwin-mutated, fitness = ADR-003 causal value |

Measure: task quality, wall-clock latency, GPU-seconds, tokens, bytes transferred (ADR-002's real, measured wire costs — §"Honest bench numbers" below), marginal contribution per edge, and topology stability across repeated runs.

### Ablation is part of the acceptance test, not an afterthought

**Remove any named component from the end-to-end experiment. If you cannot quantitatively show what capability or measurable performance disappears, that component does not yet belong in the architecture.** This is a standing discipline, not a one-time check — a component that survives every ablation unfalsified is a component whose necessity hasn't actually been tested.

### Combined acceptance test (supersedes the narrower per-ADR tests where they overlap)

`CausalDynamicLatent` achieves at least 20% lower compute or latency than `StaticText` at equal quality, **and** more than 80% of the edges it retains individually survive the mismatched-state control (ADR-003) — not just the aggregate condition-level comparison. Either alone is insufficient: a cheaper-but-unverified topology could be cheap by accident; a verified-but-unshrunk topology hasn't demonstrated self-organization (this restates ADR-003/006's acceptance test at the level of the full closed loop, per §2).

## 6. Honest bench numbers (this repo, this session — see root README)

`latentmesh-bench` (ADR-002) measured, on this machine, that `AlignmentTransform::fit`'s dense SVD costs **~156 seconds at dim=4096** (a realistic LLM hidden size) — not milliseconds. This is a real, load-bearing finding, not a footnote: it means the *current* implementation of ADR-002's alignment step is **not** yet cheap enough to sit on ADR-004's streaming hot path at LLM-realistic dimensions, and the one-vertical-experiment plan above must either (a) fit calibration transforms offline/infrequently rather than per-frame, (b) use a cheaper alignment (randomized/truncated SVD, a lower-rank projection), or (c) restrict early experiments to smaller hidden dimensions. Reporting the current cost honestly, instead of assuming it away, is exactly the discipline ADR-001 §8 commits this repo to.

## 7. Consequences

- This ADR does not add a new mechanism; it is the statement that ADR-003/005/006/008 compose into one loop, and that every existing ruvnet component this repo depends on has a specific, testable role in that loop rather than an aspirational one.
- The novelty claim is now precise and narrower than ADR-001's first draft: an *online causal control plane*, not causal attribution (E2 Explainer has that) and not dynamic topology (MANTA has that) in isolation.
- Self-assessed probability of a defensible, publishable systems contribution: **roughly 75–85%**, conditional on causal topology adaptation actually beating reward-driven topology optimization in the four-condition experiment (§5) — not a claim of certainty, and explicitly not independently verified by anyone but the requester's own literature pass.
- Named risk: E2 Explainer and MANTA are independently published components that a third party could combine into approximately this loop; the window is estimated in months, not years.

## Implementation status

Not implemented this session beyond what ADR-003/008 already ship (the statistical test and the admission gate). This ADR is the integration contract for the one-vertical-experiment plan (§5) — the next concrete step is wiring `latentmesh-core`/`latentmesh-align`/`latentmesh-gate` into an actual MidStream capture + Radio transport + RuVector persistence pipeline, which is out of scope for this session (ADR-001 §8: no live open-weight model access here).
