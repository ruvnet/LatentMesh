# 043. Cross-ecosystem architecture: where LatentMesh actually sits, and six corrections to a plausible-sounding narrative

- **Status**: Proposed. **Defers to [ruvnet/ruvector ADR-305](https://github.com/ruvnet/ruvector/blob/main/docs/adr/305) as the program of record for cross-repo scope; this ADR corrects and re-grounds this repo's own ADR-009 within that scope.**
- **Date**: 2026-08-29
- **Related**: [009](009-online-causal-control-loop.md), [001](001-latentmesh-architecture-and-prior-art.md), [008](008-capability-governed-execution.md), [016](016-ruvector-persistent-latent-memory.md), [018](018-metaharness-darwin-topology-loop.md), [023](023-live-four-condition-run1-pre-registration.md)
- **Provenance note**: this ADR's cross-repo claims (§§2–7) are sourced from an automated research pass across `ruvnet/ruvector`, `ruvnet/metaharness`, `ruvnet/rvm`, `ruvnet/midstream`, `ruvnet/RuView`, and `ruvnet/core-memory` on 2026-08-29 — README + ADR-index reads, with targeted full-text reads of the specific ADRs cited. Not a line-by-line audit of every cited document by the author of this ADR personally. Applies the same discipline ADR-305 itself established after finding stale citations in its own inputs (§1): every claim below is attributed to a specific numbered source, and every place that source could not be directly verified is marked as such rather than presented as settled.

## 1. Why this ADR exists

A plausible, well-structured narrative arrived proposing LatentMesh as "the connective tissue of the ruvnet ecosystem" — cognitive-state routing between RuFlo (coordination) and RuVector (memory), with RVM as a hard trust boundary, RVF as portable evidence, MidStream as a cross-agent streaming layer, and RuView/Air as the physical-world bridge. It read as an accurate description of an already-working system. The research pass behind this ADR found the underlying components are real and mostly actively developed — but the narrative describes an **intended** architecture, not a **current** one, and gets several specific mechanisms wrong by attributing one component's real capability to a different component's name. Both errors matter: understating maturity would waste effort re-deriving what's already decided (this repo's own ADR-009 already assigns a role to every component the narrative names, dated 2026-08-18); overstating it would build on load-bearing claims that aren't true yet.

> **Verification status of the ADR-305 claim (added during integration review):**
> the reviewer attempted to confirm `ruvector` ADR-305 and `autogenous` ADR-401
> against the RuvNet knowledge base (25 repos indexed, including `ruvector`).
> **Neither document was surfaced**, and `autogenous` is not in that corpus at
> all. This is **not** evidence the claim is false — the search is not a
> substitute for reading the repos directly, which §6.5 already requires — but
> the claim remains **unverified from this side**, and the test-plan item asking
> for direct confirmation stays open.

**The single most important correction**: this repo's own ADR-305 equivalent already exists one repo over. `ruvnet/ruvector`'s ADR-305 (dated within the last ~10 days of this ADR) performed almost exactly this reconciliation already — it read ADR-009, found the same "components have named roles but the loop isn't wired" gap, corrected stale citations in its own inputs, and made a binding scope decision (§2). Writing a competing top-level spec here instead of deferring to it would repeat the exact mistake ADR-305 itself was written to fix.

## 2. Scope: what this ADR is (and isn't) claiming

Per ADR-305's decision (as reported by the research pass — not independently re-verified line-by-line): the ruvnet-wide program of record is `autogenous`'s ADR-401 ten-capability map. ADR-009's causal-control loop (this repo) is the design contract for exactly one bounded context inside that map — **the Latent Communication Fabric** — not for the ecosystem as a whole. This ADR:

- **Does** restate and correct ADR-009's per-component table (§3) for that one bounded context, using what the 2026-08-29 research pass found about each named component's actual current documentation.
- **Does not** attempt to re-derive the ten-capability map, RVM's authority model, or Core Memory's governance model — those are owned elsewhere (§4) and this ADR cites them rather than restating them.
- **Does not** claim any of the corrections below have been implemented. Every "not wired" finding in ADR-009 (dated 2026-08-18) was independently re-confirmed as still true on 2026-08-29 for every component checked.

## 3. ADR-009's table, corrected against what each component actually documents today

ADR-009 §3 assigned each component a role "in this loop." The research pass checked each component's own documentation for whether it recognizes that role. Six real discrepancies, each with a citation:

| Component | ADR-009's framing | What its own docs (2026-08-29 pass) actually say | Correction |
|---|---|---|---|
| **MetaHarness/Darwin** | Fitness = `mutation → execution → intervention → causal attribution (ADR-003) → persistence` | MetaHarness ADR-130 (`darwin-swe-fitness-function`): fitness is `score(variant) = resolveRate(corpus)` with cost as a tie-break, on SWE-bench-style code-repair tasks. No ADR title or content in ~237 ADRs matches "causal utility" or "causal value" as an optimization signal. | Causal-value-as-fitness is a **proposed integration this repo wants**, not an existing MetaHarness mechanism. Say so as a gap, not a fact. Darwin's actual current scope is harness-configuration evolution for code repair — agent-topology/bandwidth/communication-representation optimization (as a broader narrative might imply) is not documented anywhere in MetaHarness's own repo. |
| **RVF / RVM** | "`Accept(z) ⟺ Identity ∧ Signature ∧ Authority ∧ Provenance ∧ Policy ∧ Risk<τ`, enforced as a capability grant" | ADR-008 (this repo) states directly: *"RVF packaging and RVM enforcement are not wired."* RVM's own docs (27 ADRs, checked via full-text grep) never mention LatentMesh, and frame RVM around graph-partitioned "coherence domains" for agent placement/isolation, not as a downstream gate for arbitrary external "candidate cognition." | The RVM-as-gate-for-LatentMesh flow is this repo's aspiration, stated nowhere in RVM's own design. Real, concrete, already-specified RVM integration exists but is narrower and points the other way: RVM ADR-156 anchors **RuFlo's** flywheel evaluation receipts into RVM's witness chain — a real RuFlo↔RVM contract that predates and doesn't mention LatentMesh. |
| **"RVF" naming** | Treated as one artifact format | Two things share the name. RVM's `rvm-rvf` crate loads "RVForge packages" (RuVector-ecosystem package format, RVM ADR-149/155). `ruvnet/ruflo` separately documents "RVF (Ruflo Vector Format)" as its own container format (`ruflo-rvf` plugin). Whether these are the same format reused across repos, or a genuine naming collision, was **not resolved** by this research pass. | Do not use "RVF" in future LatentMesh ADRs without stating which one is meant. This needs a direct spec comparison, not an assumption either way. |
| **MidStream** | Supplies bounded `LatentFrame` streaming with gap/duplicate/regression/confidence-change detection | MidStream's own repo (searched directly via `gh api search/code` for "LatentMesh" and "LatentFrame": zero hits, both terms) documents single-conversation LLM token-stream analysis — pattern-matching (`temporal-compare`), chaos/attractor detection, meta-learning, a QUIC transport (`midstreamer-quic`). No cross-agent latent-state protocol anywhere in its docs. | `LatentFrame` is this repo's own abstraction (ADR-004, ADR-015), built **on top of** MidStream's transport primitives (`midstreamer-quic`, already a real dependency per ADR-015). Correct framing: MidStream is infrastructure LatentMesh consumes, not a project that itself does what ADR-009 credits it with. |
| **RuView** | A thin sensor reporting raw observations, selectively transmitting per `I(observation; world state) > 0` | RuView (254 ADRs, its own crates.io/PyPI/HuggingFace releases) already ships ADR-273 "Unified RF Spatial World Model" — its **own** persistent, queryable, multi-modal world model, built on RuVector directly, independent of LatentMesh. It also has its own governed-evidence mechanism (privacy policy, uncertainty, provenance, Ed25519 witness chain on sensing events) architecturally adjacent to what §3's RVF/RVM row assigns this repo. RuView's own MidStream integration (ADR-099) is narrower than ADR-009 implies too: two specific signal-processing primitives (DTW, Lyapunov/regime classification) as an introspection tap on its own CSI stream — not inter-agent state transport (ADR-098 explicitly *rejected* MidStream as a transport-layer replacement). | RuView is a peer system with its own substantial, already-decided world-model and governance stack — not a thin upstream data source waiting for LatentMesh to route its output. Any future wiring needs to integrate with RuView's existing world-model, not assume LatentMesh originates one. |
| **"Core Memory"** | Not present in ADR-009 at all | A real, separate repo (`ruvnet/core-memory`, status: experimental 0.1.0) exists and is *not* a memory-timescale tier — its own README: *"a governed coordination and memory control plane... for human and agent development"* (capability checks, signed receipts, promotion policy, separation of duties). It already publishes a precise one-responsibility-per-component table (Core Memory / RVM Context / RuVector / RVF / Ruflo / MetaHarness / GitHub) more exact than this ADR's own §4. It has **zero mentions** of LatentMesh, MidStream, or RuView anywhere in its docs (confirmed by direct grep). | This is a real, current integration gap, not a resolved boundary — say so plainly (§4). |

## 4. The governance/federation boundary, resolved

ADR-009 §3 assigns RVF/RVM one combined "trust" row and doesn't mention Core Memory or Autogenous's relationship to it. Cross-referencing Core Memory's own boundary table (§3, row 6 above) against this repo's federation ADRs (007, 020, 025, 026) and RuFlo's own `ruflo-bbs-federation` plugin resolves the apparent overlap into three layers on one substrate, not competing ownership of "federation":

```
Ruflo / AgentBBS      transport substrate — peer identity, transport, store-and-forward
        │
        ├── Core Memory     governs AGENT-ACTION federation over that transport
        │                   (capability checks, receipts, promotion — never infers
        │                    authorization from retrieved content)
        │
        └── LatentMesh      governs LATENT-STATE federation over that transport
                            (ADR-007/020/025/026 — this repo's own scope,
                             per §2's bounded-context framing)
```

Core Memory's own stated invariant ("RuVector: physically scoped, rebuildable semantic retrieval projection — never trusted to establish truth, approve, or grant access") is the same invariant ADR-008's admission gate needs to inherit rather than reinvent: **this repo's gate should defer to whatever authority decision Core Memory/RVM eventually make, not implement a separate authority model of its own.** ADR-008 does not yet say this explicitly. It should.

Autogenous — a real, separate repo (`ruvnet/autogenous`, MIT, alpha/research-prototype) assigned "persistent identity across model swaps" in ADR-009's own table — is, per ADR-305 (§2), the actual program-of-record owner for the ten-capability ecosystem map. Its own README states it is "not wired to live MidStream/MetaHarness/RVF/RVM yet" (as reported by the research pass) — consistent with every other "not wired" finding here.

## 5. Honest status, restated

ADR-009 already says the loop isn't wired. The 2026-08-29 pass adds:

- **Size**: LatentMesh is ~1,407 LOC across 4 crates (`latentmesh-core`, `latentmesh-align`, `latentmesh-gate`, `latentmesh-bench`), 23 tests, no network transport crate in the workspace at all. This is a small research prototype, not a substantial system, and should be described as such in any document that names it alongside RuVector (published crates.io crates, 300+ ADRs) or RuView (254 ADRs, three package registries) as a peer.
- **CORRECTION (2026-08-29, later same day — added during integration review).**
  Three of §5's claims about *this repo* were stale or wrong when checked
  against `main` at integration time. They are corrected here rather than left,
  because an ADR whose thesis is "stale citations mislead" cannot itself carry
  stale citations about its own repository:
  - **"ADR-035 and ADR-037 are BLOCKED PERMANENTLY"** — half right. ADR-035 is
    blocked. **ADR-037 was UNBLOCKED earlier the same day** and its rung (M5X)
    was drawing at the time this PR was reviewed. The unblocking is recorded in
    ADR-037 itself: C2C's Table 10 ablation collapses to ~0.1pp at a single
    layer, reproducing this repo's own null signature, so layer count — the one
    variable every rung held fixed — became worth testing.
  - **"ADR-039/040 are still-pending pre-registrations"** — both have **run**.
    PC2 and PC3 draw receipts are committed. PC3 is the out-of-sample
    confirmation the whole negative result rests on.
  - **"~1,407 LOC across 4 crates … no network transport crate in the workspace
    at all"** — **`main` carries 15 crates**, including `latentmesh-meshtastic`
    (merged 2026-08-27, two days before this PR), `latentmesh-air-radio`,
    `latentmesh-stream` and `latentmesh-federation`. There is network transport,
    validated against real `meshtasticd` firmware over TCP. The 4-crate figure
    describes a subset, not the workspace.

  **This does not weaken §5's argument — it sharpens it.** Two of the three
  errors *understated* the repo, and the corrected picture still supports the
  section's actual point: LatentMesh is a research prototype whose core
  hypothesis is trending negative, and a cross-ecosystem spec should not assert
  "LatentMesh owns cognitive state movement" as settled.

- **The core hypothesis is trending negative, not confirmed.** Only ADRs 010–018 are marked Accepted-and-implemented, and every one is caveated (deterministic simulation, no over-the-air claim, no live-agent claim, remote deployment external). ADRs 019–041 are a separate research thread testing whether cross-model latent transfer beats text at all: ADR-023 is a clean negative result; ADR-035 and ADR-037 are **BLOCKED PERMANENTLY**; ADR-038 is a measurement-methodology fix, not a positive transfer result; ADR-039/040 are still-pending pre-registrations. A cross-ecosystem spec asserting "LatentMesh owns cognitive state movement" as settled fact would overstate what this repo's own most recent evidence supports. State the intended role (§2's bounded context) as intended, not as validated.
- **RuFlo is the least-integrated named component in ADR-009's own table** — it appears only as a table row (`ROI_edge` scoring), with no design ADR and no implementation ADR, unlike every other row. If future work gives RuFlo real teeth as the coordination plane, that's new work, not a restatement of an existing decision.

## 6. What changes as a result of this ADR

1. **ADR-008** should be amended (separate, future ADR — not this one) to state explicitly that its admission gate defers to Core Memory/RVM's eventual authority decision rather than implementing an independent one, per §4.
2. **Any future reference to "RVF" in this repo's ADRs must disambiguate** which artifact format is meant until §3's naming collision is independently resolved against both specs directly.
3. **ADR-009's per-component table** should be read alongside §3 of this ADR, not in isolation — the corrections here narrow several rows without changing the loop's overall shape (§2 of ADR-009 stands; the claims about what MetaHarness/RVM/MidStream/RuView *already do* do not).
4. **The acceptance test** stays ADR-009 §5's four-condition experiment (`StaticText` / `DynamicText` / `DynamicLatent` / `CausalDynamicLatent`), unchanged — this ADR found no basis to replace it with a broader end-to-end claim spanning components (Core Memory, RVM's real anchor mechanism) that ADR-009 didn't originally include and that aren't wired to this repo regardless. Extending the experiment to include a real RVF/RVM evidence leg is future work gated on §3's RVF naming resolution first, not on this ADR.
5. **Next concrete step**, per ADR-305's own precedent (§1): read `ruvnet/ruvector`'s ADR-305, ADR-309, and ADR-333 directly (not via this ADR's secondhand summary) before any implementation work on the causal-control loop, capability governance, or the RVM anchor mechanism begins. This ADR's cross-repo claims are one automated research pass, dated; ADR-305's own fix-history rule exists because a single pass — including this one — can still be wrong about a primary source it didn't read directly.

## 7. Consequences

**Positive**: this repo's own architecture claims now match what six other real, actively-developed repos actually say about themselves, with citations, instead of a plausible narrative that was right about the shape of the system and wrong about several specific mechanisms. The RVF naming collision and the Core Memory/LatentMesh federation-layer ambiguity are now named risks instead of implicit assumptions.

**Negative**: this ADR resolves nothing by itself — every correction here is a "say this honestly" change, not a "build this" change. The actual cross-repo wiring (RuFlo↔LatentMesh, LatentMesh↔Core Memory/RVM authority deference, RVF disambiguation) remains exactly as unimplemented after this ADR as before it. That is the intended outcome: get the map right before spending engineering effort on the territory.

## Implementation status

Not implemented — this is a documentation-only correction. No code, crate, or workspace member changes. The next concrete engineering step is named in §6.4/6.5 and is out of scope here (ADR-001 §8: no live open-weight model access in this session; this ADR additionally had no direct primary-source access to the six external repos beyond README/ADR-index/targeted-file reads via the GitHub API).
