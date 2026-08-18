# 001. LatentMesh architecture and prior art

- **Status**: Proposed (research prototype → controlled experiments). **Mostly not built.**
- **Date**: 2026-08-18
- **Owners**: ruvnet architecture and research maintainers
- **Scope**: MidStream, RuVector, MetaHarness/Darwin, Radio, Autogenous, RVF, RVM
- **Related**: [002](002-latent-packet-protocol.md)–[008](008-capability-governed-execution.md) (this ADR is the north star they make concrete)
- **Provenance note**: the external research claims cited below (LatentMAS, AVP, AAFLOW+, StateBridge, StreamMA, DMoA, Cisco Cognitive Fabric, LCGuard, CrystalMem, TTHE, FedWorld, DreamGuard, and the July/August 2026 causal-audit papers) are reported from the requester's own literature survey. They are **not independently verified by this ADR** — no paper was fetched, re-derived, or benchmarked as part of authoring this document. Treat every specific number below as an externally claimed figure, not a LatentMesh result, until this repo reproduces it.

## 1. Decision summary

Build **LatentMesh**: agents exchange aligned hidden-state slices — "latent frames" — instead of first converting everything to language, so that (a) communication stops being the serialization/detokenization bottleneck, (b) downstream agents can consume upstream cognition incrementally, and (c) the communication graph itself is discovered and pruned by **measured causal contribution**, not designed by hand or trusted by default.

The distinguishing bet, after a literature pass (§3), is **not** the wire format or the alignment trick — both are already published, more than once. It is: **an agent-to-agent latent edge only exists in the topology if it survives a counterfactual test proving it transferred real information** (ADR-003), and the whole stack — streaming (ADR-004), memory (ADR-005), topology evolution (ADR-006), federation (ADR-007), governance (ADR-008) — is built around that one measurement.

## 2. The core idea, and its immediate grounding

Instead of:

```
Agent A → generate tokens → serialize text → network → tokenize text → Agent B
```

build:

```
Agent A hidden state → latent alignment → latent packet → P2P fabric
                                                          → receiver alignment → Agent B latent prefix
```

**StateBridge** (reported, Aug 13 2026) demonstrates *training-free* alignment of one model's hidden states into another model's input space via a closed-form orthogonal transformation — reported best-or-tied-best on 22 of 26 evaluated model/task combinations. Conceptually:

```
Z_B = α · Z_A · R
```

where `Z_A` is the sender representation, `R` an orthogonal alignment (this repo implements the *general, verifiable* technique — orthogonal Procrustes via SVD — as `latentmesh-align`; see ADR-002 for what "training-free" honestly means when the sender/receiver spaces have no known correspondence), and `α` calibrates magnitude for the receiver.

## 3. Prior art — what already occupies this ground

A July 2026 survey reportedly catalogs 18 representative latent-communication methods. This is now a populated research category, not an empty one. Closest overlapping systems (reported by the requester's survey, scored 1–10 for closeness to this ADR's target):

| Project | What overlaps | Closeness |
|---|---|---|
| **LatentMAS** | Latent thoughts + shared latent working memory between agents; reported up to 14.6pp higher accuracy, 70.8–83.7% fewer output tokens, ~4× faster inference vs. text baselines; reported ICML 2026 Spotlight | 9/10 |
| **AVP** (VectorArc's Agent Vector Protocol) | A binary, transport-agnostic wire protocol for KV cache, hidden states, model metadata, cross-model projection, and fallback representations (HTTP/2, gRPC, WebSockets, A2A). Its own spec reportedly says it handles latent *communication*, not discovery or orchestration | 9/10 |
| **AAFLOW+** | Makes KV state a first-class distributed-systems object — materialize/transfer/fork/compose/evict/schedule — carrying model identity, config, tensor blocks, position info, lineage, placement, ownership. Reported up to 50.2× TTFT reduction, 7.63× lower multi-agent compute; bandwidth-dependent (wins broadly at ≥25 Gbps, not always at 10 Gbps) | 9/10 |
| **StateBridge** | Training-free heterogeneous hidden-state alignment (§2) | 8/10 |
| **Mixture of Thoughts** | Heterogeneous experts sharing hidden representations | 8/10 |
| **DMoA** | Self-evolving, dynamic agent topology during reasoning | 7/10 |
| **StreamMA** | Real-time pipelined agent reasoning; reported +7.3pp average across tested benchmarks from letting downstream agents consume upstream reasoning incrementally | 7/10 |
| **Cisco Cognitive Fabric (Nodes)** | Memory + topology selection + semantic grounding + security policy in one node abstraction — but reportedly over a conventional (non-latent-state) communication substrate | 7/10 |
| **LCGuard** | Security for latent/KV communication — attempts to strip reconstructable sensitive information before transmission | 6/10 |

There is also reportedly an August 2 proposal to treat network infrastructure itself as a KV-state distribution fabric — decoupling inference compute from KV storage and making bandwidth/latency/price part of state placement ("an Internet for the KV cache").

**Consequence for this repo**: `latentmesh-core`'s `LatentFrame` (ADR-002) is deliberately scoped as a **reference wire type for this workspace's own crates**, not a competing protocol — AVP already specifies the harder ground (transport-agnostic binary KV/hidden-state transfer) more thoroughly than a from-scratch design authored here could justify. A bridge/adapter to AVP is the likely long-run wire format; `LatentFrame` exists so MidStream/Radio/RVF/RVM integration (ADR-004–008) has a concrete, own-repo type to build and test against today.

## 4. Where the actual whitespace is

Searching specifically for the combination:

```
Streaming + Latent + Heterogeneous + P2P + Persistent + SelfEvolving + Governed
```

no public implementation combining all seven was found as of 2026-08-17 (per the requester's search — an inference from currently published components, not a claim that no one has built this privately). Nobody surveyed appears to connect:

```
        agent
          │
     latent stream
          │
          ▼
    ┌───────────┐
    │ cognitive │
    │   mesh    │
    └───────────┘
      ↙   ↓   ↘
 model   edge   model
   │      │      │
   └── persistent ──┐
       latent memory │
                    ▼
             topology evolution
                    │
                    ▼
             governed execution
```

That integration — not any single piece of it — is the target.

## 5. The harder, more important problem: causal contamination

Two more-recent audits reportedly complicate the premise directly. A July causal audit found that a benchmark improvement from "latent communication" can arise from message *presence*, extra *computation*, or generic state effects — **not** necessarily sender-specific information. An August 5 audit reportedly found a striking case: removing a KV relay cost 14.7 points, but replacing it with a *mismatched example's* cache cost only 0.4 points in one tested condition — i.e. the relay's specific content barely mattered, only its presence did. (LatentMAS reportedly did show genuine example-specific transfer in other conditions; several other methods reportedly did not.)

**This is the decision this ADR commits to**: LatentMesh does not implement "latent communication" and stop. It implements **causally verified** latent communication — every edge `A → B` is tested against zero-state, random-state, mismatched-task-state, and self-generated-state controls (ADR-003) before it is trusted, and MetaHarness Darwin evolves the topology against that measured signal (ADR-006), not against token traffic or superficial accuracy.

## 6. Architectural model — six layers

1. **Packet** (ADR-002) — `LatentFrame`: sender/receiver space, transform binding, confidence, provenance, authority, quantized payload. A reference type, not a wire-format claim.
2. **Verification** (ADR-003) — the causal-edge gate. An edge is admitted, strengthened, or removed based on `ΔV_{A→B} = V(B|A) − V(B|control)` across four controls, not on raw benchmark deltas.
3. **Streaming** (ADR-004) — MidStream carries `z_1, z_2, …, z_t` continuously instead of waiting for a complete generation.
4. **Memory** (ADR-005) — RuVector stores a continuum: raw latent trajectory → compressed trajectory → semantic prototype → symbolic rule. Includes persistent *computational skills* (successful topologies, not just text).
5. **Topology** (ADR-006) — MetaHarness Darwin mutates which agents exist, which edges exist, what representation crosses each edge, and how much bandwidth it gets — fitness-scored by measured causal contribution.
6. **Governance** (ADR-008) — RVF/RVM: `execute(z) ⟺ signature(z) ∧ authority(z) ∧ provenance(z) ∧ risk(z) < τ`. Latent execution is a capability grant, never an implicit trust, because latent payloads are not human-inspectable.

Federation across machines (ADR-007) and RuView-style edge sensing folding into the same fabric are treated as the same primitive at a different physical scale, not a separate system.

## 7. Decision drivers (hard rules)

1. No claim of novelty on the wire format or the alignment math alone — both are prior art (§3).
2. Every edge's right to exist is **earned by a counterfactual test**, re-checked continuously, not granted once.
3. Alignment confidence and causal edge value are first-class, measured fields — never asserted.
4. A latent payload crossing a trust boundary is a **governed capability grant** (ADR-008), symmetric with how `cognitum-one/slack`'s AGL module governs code mutations — same shape of problem (an opaque, consequential action) applied to a different substrate.
5. Nothing in this ADR set is claimed benchmarked against live heterogeneous LLMs unless a specific ADR's Implementation section says so with numbers from this repo's own `cargo bench`/tests.

## 8. Honest feasibility

**What is implementable and tested *today*, in this repo, without live model weights**: the packet type and codecs (ADR-002), the general orthogonal-alignment algorithm and its correctness on synthetic data (ADR-002), the governance gate's admission logic (ADR-008), and the *statistical machinery* for causal edge verification (ADR-003) — all of these are real math/software problems with ground truth independent of any specific LLM.

**What requires live, open-weight model access and is explicitly out of scope for this session**: whether real cross-model hidden states are semantically compatible after alignment, whether a real causal edge survives the four-control test on a real task, and every specific percentage figure quoted from external literature in §2–§5. This repo's own claims are limited to what its tests and benchmarks actually measure (see each ADR's Implementation section and the root README's Honest Status).

## 9. Consequences

**Positive**: a defensible, falsifiable research target instead of "build a message bus for vectors"; the causal-verification requirement is exactly the property that would make this hard to dismiss as "LatentMAS plus networking"; the existing ruvnet stack (MidStream, RuVector, MetaHarness/Darwin, Radio, RVF, RVM) already has a component for every layer in §6, so the work is mostly integration + one new measurement primitive (ADR-003), not seven new systems.

**Negative**: causal verification requires running controlled interventions (four variants per candidate edge) which multiplies compute cost during topology search; the whole thesis is falsifiable in the boring way — if edges never separate from controls, there is no paper, only a negative result; alignment quality is fundamentally bounded by whether two models' hidden states encode compatible concepts at all, which this repo cannot resolve without live model access.

## 10. References

Requester-supplied literature survey (external, unverified by this ADR): StateBridge (2026-08-13); LatentMAS (ICML 2026 Spotlight, reported); AVP / Agent Vector Protocol (VectorArc); AAFLOW+; Mixture of Thoughts; DMoA; StreamMA; Cisco Research Cognitive Fabric; LCGuard; CrystalMem; TTHE; FedWorld (2026-08-03); DreamGuard; a July 2026 causal-audit paper; an August 5 2026 causal-audit paper; an August 2 2026 "Internet for the KV cache" proposal. In-repo: [autogenous ADR-391/392](https://github.com/ruvnet/autogenous/blob/main/docs/adr/) (the sibling governed-evolution architecture this design borrows its honesty/evidence conventions from); `cognitum-one/slack` ADR-0008 (AGL admission, the model for ADR-008's execution gate).
