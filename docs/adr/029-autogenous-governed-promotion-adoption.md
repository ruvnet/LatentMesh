# 029. Adopting autogenous's governed-promotion contract for LatentMesh's authority layer

- **Status**: Proposed. Design contract only — no code, no dependency added, no promotion in this
  wave.
- **Date**: 2026-08-29.
- **Related**: [003](003-causal-edge-verification.md) (the ΔV fitness vector this ADR maps onto
  autogenous's promotion gate), [008](008-capability-governed-execution.md) (the authority ceiling
  this ADR assesses for a type-level upgrade — currently "RVF packaging and RVM enforcement not
  wired," quoted verbatim from that ADR's own status line), [026](026-verified-edge-federation-wire-contract.md)
  (verified edges' missing TTL/containment/rollback — the gap this ADR names as real and unclosed),
  [027](027-latent-prefix-context-window-delivery.md) (the hierarchical-vouching design that
  §2/§3 sketched but could not ground — this ADR is a concrete instantiation of that sketch),
  [025](025-distributed-latent-data-fabric.md) (the precedent this ADR follows for handling an
  unverified-but-promising external Rust crate: designate + flag unverified-against-build, not
  silently adopt or silently reject)
- **Evidence base**: `github.com/ruvnet/autogenous` (`~/projects/autogenous`, `README.md` read in
  full this session — a real, cloned repo, not a search summary), `docs/research/027-global-ambient-intelligence-track.md`
  §2-3 (the hierarchical-trust and anti-gaming governance gap this ADR responds to),
  `crates/latentmesh-gate/src/lib.rs` (this repo's own gate implementation, read this session for
  the attribution discrepancy noted below), `docs/research/030-economics-and-gate-standalone.md`
  (in flight at this ADR's authoring — the causally-gated-text run this ADR's ΔV mapping would
  also govern if adopted)

## Context

`docs/research/027` §2-3 identified hierarchical trust, bounded-false-trust promotion, and
anti-gaming governance as **unsolved** for LatentMesh at scale — and separately, ADR-008 has always
stated plainly that "RVF packaging and RVM enforcement" — the production targets for its own
admission gate's output — are "not wired." `autogenous` (`github.com/ruvnet/autogenous`) is a
sibling Rust research prototype (MIT, Rust 1.74+, alpha, explicitly labeled "research prototype of
the control-plane contract — typed, tested, deterministic, and offline... not wired to live
MidStream/MetaHarness/RVF/RVM yet," quoted verbatim from its own README's Honest Status section)
whose entire thesis is exactly this missing layer: "a governed operating system for software that
can learn from production, redesign parts of itself, prove the redesign is better, deploy it
safely, and reverse it when wrong."

**A load-bearing attribution note, found this session, stated here rather than smoothed over**:
autogenous's own README states, in its "Related" section, that LatentMesh's causal edge-verification
and admission gate (ADR-003/008) "are directly ported in shape from this repo's AGL admission
model" — i.e., autogenous claims to be the design source. But `crates/latentmesh-gate/src/lib.rs`'s
own header, in this repo, states the gate was "Ported in shape from `cognitum-one/slack`'s AGL
admission module" — a different named source. Both statements cannot describe the same lineage
literally as written unless `cognitum-one/slack`'s AGL and autogenous's AGL (the Autogenous Genome
Language) are themselves the same underlying design shared across repos under one name, which this
ADR did not independently verify. **This is a genuine discrepancy, not resolved here** — flagged
so nobody reading both repos assumes a clean, single-sourced lineage. It does not change this ADR's
substantive analysis (the mapping below is evaluated on its own merits regardless of which repo the
pattern originated in), but the coordinator should know the provenance claim itself is currently
inconsistent across the two repos.

Autogenous's measured numbers (its own README, offline benchmarks, `cargo run --release`):
stream observation ≈0.9 µs/chunk with 1 armed antibody, ≈14 µs with 16; canary decision ≈2 ns;
100,000-labeled-stream replay ≈7 ms; 71 tests including an end-to-end acceptance lifecycle. Its
TypeScript companion `packages/radio-moe` reports a fusion benchmark this ADR does **not** cite
uncritically — see the corrected framing below, which supersedes an earlier miscitation caught and
fixed before this document was finalized.

## Decision — adopt / cite / reject, stated explicitly

### ADOPT (design-level contract, no implementation this wave)

- **Promotion as a hard AND-gate over a fitness vector**, mapped onto ADR-003's existing five-decoy
  ΔV measurement. Autogenous's `FitnessVector::passes_hard_gates` uses `min`-semantics — "exceptional
  quality can never compensate for a safety or governance miss," quoted from its README. **This is
  the promotion machinery LatentMesh currently lacks, not a new fitness concept**: ADR-003's `ΔV`
  (real vs. the worst of five controls, `latentmesh-gate::causal::verify_edge`'s "stricter-than-mean
  bar") already *is* a fitness vector in autogenous's sense — a scalar that must clear a threshold
  against the least-favorable control, not an average. What LatentMesh has never had is the
  **promotion machinery around that scalar**: a staged process that consumes a verdict and turns it
  into a governed, reversible, auditable state change. Adopting this contract means: an
  ADR-003-verified edge (or, per ADR-028, an evolutionary-search champion) becomes a candidate for
  autogenous-style promotion, not that ADR-003's statistics themselves change.
- **"Authority never silently expands" as the design pattern to assess for a type-level upgrade.**
  ADR-008's `Authority` enum (`ObserveOnly → ContextInject → LatentPrefix → ActionInfluencing`) and
  `latentmesh-gate::CeilingThresholds` currently enforce this as a **runtime check**
  (`ceiling_from_verdict` computes a ceiling from a measured `ΔV`, and nothing in the type system
  prevents a caller from constructing an `Authority::ActionInfluencing` value directly). Autogenous
  enforces the equivalent invariant "in the types, not in policy docs" — `Mutation::admissible` in
  its `agl-types` crate rejects any mutation requesting more authority than its parent's ceiling, as
  a compile-time-checkable structural property, not a runtime assertion. **This ADR adopts the goal
  (type-level authority-never-expands) as worth pursuing for `latentmesh-core::Authority`**, without
  committing to autogenous's specific type machinery — a genuine Rust type-system redesign of
  `Authority` is real engineering work this ADR does not scope or schedule.
- **Expiring, capability-constrained packages with rollback, as the missing shape for federated
  edges.** ADR-026's `CausalEdgeSync` contract has **no TTL, no containment, no rollback path** — a
  federated edge, once published and locally re-verified, has no expiration and no defined undo.
  Autogenous's Antibody Package (AAP — "signed, capability-constrained, expiring adaptation:
  trigger, evidence, detector, containment, regression corpus, fitness envelope, lineage, rollback")
  is a directly applicable shape for closing this gap. **This ADR names the gap as real and adopts
  AAP's shape as the reference to close it against** — it does not redesign ADR-026's wire contract
  in this document; that redesign is named as required follow-up work, not done here.
- **≥2 pinned judge signatures as a concrete instantiation of the hierarchical-vouching sketch.**
  `docs/research/027` §2-3 sketched hierarchical trust and bounded-false-trust promotion as unsolved
  design space without a concrete mechanism. Autogenous's ADR-394 (cryptographic closure of
  promotion) requires "ed25519-signed evaluation receipts from ≥2 distinct pinned judges measuring
  candidate-vs-parent on the same corpus" before any promotion — a candidate must beat its parent
  with a non-inferiority margin, and effects/rollback-target/invariant-proofs live inside a
  content-addressed manifest a verifier can independently re-check. This is a working answer to
  exactly the shape of problem §2-3 left open, and this ADR adopts it as the reference design for
  any future LatentMesh promotion gate (ADR-028's flywheel adoption, or any edge-federation
  promotion path).

### CITE (prior art / motivating data point, not a design commitment)

- **Autogenous's measured performance numbers** (0.9 µs/chunk observation, 2 ns canary decision,
  100k-stream replay in 7 ms) — cited as evidence that this class of governance machinery can be
  made cheap enough to sit in a hot path, not as a performance target LatentMesh commits to
  matching. These numbers are autogenous's own, on autogenous's own workload; nothing here claims
  they transfer to LatentMesh's workload without independent measurement.
- **The AGL (Autogenous Genome Language) typed-mutation schema itself** — cited as a worked example
  of what a typed mutation declaration looks like (what changes, why, where valid, what authority,
  which invariants, how tested, when expires, how reversed), useful as a checklist when any future
  ADR designs LatentMesh's own mutation/promotion schema, without adopting AGL's Rust types directly
  as a dependency.
- **`packages/radio-moe`'s fusion result — cited with the corrected, hedged framing, not the
  originally-circulated one.** An earlier framing of this result ("fusing independent experts beats
  the strongest single expert by 33.3%") **overstates what the benchmark measures and is corrected
  here.** Verified by direct read of `packages/radio-moe/README.md`'s own committed benchmark table
  (`npm run bench:fusion`, deterministic, synthetic corpus with controlled error-correlation
  structure): the 33.3-percentage-point gain (100% vs. 66.7% best-single) holds specifically in the
  **independent-errors regime**. In the **correlated-errors regime**, the gain is +25pp (100% vs.
  75%) — and the load-bearing detail is that **naive-vote and dedup-only fusion both lose to
  best-single once errors correlate** (66.7% < 75.0%); only **lineage-weighted independence
  weighting** (`effectiveSupport`, provider/architecture-graded, not just distinct-source-counted)
  wins in both regimes. **The honest claim this ADR cites: governed/weighted fusion beats both naive
  fusion and best-single on a synthetic benchmark with known error-correlation structure — evidence
  that *how* combination is governed matters, not a real-task multi-agent result.** This framing
  strengthens rather than weakens the autogenous-adoption argument specifically: the gain comes from
  the governance layer (lineage-weighted independence), not from the mere presence of multiple
  agents — a direct data point for why adopting autogenous's governance patterns (not just running
  more agents) is where the value would come from.
  - **Scope note, also verified by direct read, relevant to whether this threatens ADR-030's
    novelty claim**: `radio-moe` verifies source **provenance and independence** before release
    (`ActionGate`, `independentSupportSet`); it **never applies a decoy condition to any expert's
    message content** — no zero/random/mismatched/self-generated substitution at the content level,
    anywhere in the package. It answers "does combining N experts beat the best single expert,"
    never ADR-003's question, "does this specific sender's real content — versus content-matched
    noise — change this specific receiver's outcome." **This is complementary to ADR-003's gate, not
    a substitute for it, and not a priority threat to the run-3 causally-gated-text experiment**
    (`docs/research/030`, in flight) — the two verify different things at different layers.

### DO NOT ADOPT (assessed and rejected, stated honestly)

- **`autogenous` as a Cargo dependency of any LatentMesh crate, in this wave.** Per the ADR-025
  precedent for handling an unverified-but-promising external crate (designate + flag
  unverified-against-build, don't silently adopt): autogenous is explicitly self-labeled "alpha,"
  "research prototype," and "not wired to live MidStream/MetaHarness/RVF/RVM" by its own README.
  Taking a research-alpha dependency for a production authority layer is not something this repo
  should do lightly, and this ADR does not do it — nothing here adds `autogenous` or any of its
  crates to any `Cargo.toml`.
- **Wholesale replacement of `latentmesh-gate`'s runtime-checked `Authority` enum with autogenous's
  type machinery, as scoped work in this wave.** Assessing the *goal* (type-level enforcement) is
  adopted above; actually redesigning `Authority` to be unconstructible outside an
  authority-preserving path is real Rust type-system engineering (likely a sealed-trait or
  phantom-type pattern) that is named as future work, not designed or scoped here.
- **AAP's specific expiration/renewal/containment mechanics as LatentMesh's edge-federation
  contract, verbatim.** Adopted above is the *shape* (expiring, capability-constrained, with
  rollback); this ADR does not redesign ADR-026's wire message to literally carry AAP's Rust struct
  fields. That is a follow-up ADR's job, informed by this one.
- **`radio-moe`'s mixture-of-experts pattern as a substitute for latent transfer, or as this repo's
  answer to "what if latent transfer keeps failing."** `docs/research/030`'s Q1/Q2 analysis (in
  flight) already frames this correctly: the compute case for latent transfer is separately
  strong (≈92× decode/prefill ratio, receipted), and the open question is causal usefulness, not
  economics. `radio-moe`'s result is cited above as a *data point* about governed combination
  generally — it is explicitly not adopted as a pivot away from the latent-transfer research
  program, and this ADR does not recommend one.

## Consequences

Adopting the promotion-machinery goal and the authority-never-expands goal, while declining the
Cargo dependency, follows the same discipline ADR-025 already established for `ruvector-replication`
— name a concrete external reference, state what's verified (a README read, a repo cloned and
inspected) versus what's not (build compatibility against this workspace, live behavior under this
repo's own workload), and let a later ADR do the actual integration once the goal is scoped as real
engineering work. The attribution discrepancy (autogenous's README vs. `latentmesh-gate`'s own
header) is left unresolved deliberately — resolving it would require either repo's history to be
audited, which is out of scope for a design-contract ADR, but silently picking one attribution and
presenting it as settled would misrepresent the evidence this session actually gathered. The
corrected `radio-moe` framing (percentage points, regime-dependent, governance-attributable) is a
better argument for this ADR's thesis than the uncorrected one would have been — it locates the
value in the governance layer specifically, which is exactly the layer this ADR proposes adopting.

## What is simulated / what is hardware-pending

| Claim | Status |
|---|---|
| Autogenous's README claims (workspace layout, structural guarantees, measured performance numbers) | Verified by direct read of `~/projects/autogenous/README.md` this session — a real cloned repo, not a search summary; the *numbers themselves* are autogenous's own self-reported measurements, not independently re-run here |
| The attribution discrepancy (autogenous's README vs. `latentmesh-gate`'s header) | **Verified as a discrepancy this session** — both sources read directly; not resolved, flagged as open |
| `radio-moe`'s fusion benchmark table (independent-errors 33.3pp, correlated-errors 25pp, naive-fusion losing to best-single under correlation) | Verified by direct read of `packages/radio-moe/README.md`'s committed table this session, per the corrected framing above |
| `radio-moe`'s scope relative to ADR-003 (provenance/independence verification, never content-level decoy controls) | Verified by direct read this session |
| Any actual integration — Cargo dependency, type-level `Authority` redesign, AAP-shaped edge federation, ≥2-judge promotion gate | **Not implemented** — no code in this wave, explicitly declined for `autogenous`-as-dependency per the DO NOT ADOPT list above |
| Autogenous's own performance numbers reproduced on this repo's workload | **Not measured** — cited as autogenous's own result, not re-benchmarked here |

## Implementation status

Not implemented. This ADR is a design contract stating what LatentMesh adopts as a governance
pattern (promotion AND-gates, type-level authority-never-expands as a goal, expiring/rollback-able
federated edges, ≥2-judge signed promotion) versus what it merely cites versus what it explicitly
declines (the Cargo dependency itself, wholesale type-machinery replacement, AAP's mechanics
verbatim, and `radio-moe`'s pattern as a pivot away from latent transfer). No crate, module, or
integration exists yet.
