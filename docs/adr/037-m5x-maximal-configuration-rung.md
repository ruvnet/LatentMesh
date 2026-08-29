# 037. M5X: the multi-factor "maximal configuration" rung

- **Status**: **UNBLOCKED (2026-08-29, same day) — the permanent block below was
  my error and is withdrawn.** I blocked this rung on the ground that it
  *"varies payload content, which is demonstrably not what moves decisions."*
  **Factor 1 of this ADR is not content — it is multi-layer injection**, which
  this document itself calls *"the single largest evidence gap this ladder has
  left untested"* (the L24→L19 pair has never been used by any rung).
  PC3 held delivery architecture **fixed** at a single site and varied content,
  so its null is **conditional on single-layer delivery**; I generalised it
  beyond what it supports. **Cache-to-Cache's own Table 10 (arXiv:2510.03215)
  is direct external evidence against that generalisation**: reduced to a
  *single* layer, C2C yields ~0.1pp — our exact null signature — while its
  6.4–14.2pp effect requires gated fusion across ~top-5-of-28 layers.
  **Our null may be a fact about single-layer injection, not about injection.**
  Recorded as coordinator error #16 in
  [research/049](../research/049-adjacent-areas-survey.md). M4b (ADR-035)
  stays blocked on its own separate rationale.
  **Before drawing, this rung must state ADR-040's mandatory power calculation.**
- **Superseded status**: BLOCKED PERMANENTLY (2026-08-29) — withdrawn same day. PC3 confirmed out of sample that the injection pathway is
  *non-semantic at the decision level*: payload content changes which answer the
  receiver gives no more than a norm-matched Gaussian does (36W/32L, n_disc 68,
  p = 0.72 two-sided, min attainable p 3.4e-21). **M5X varies payload content
  and configuration — neither is the binding constraint**, so its result would
  be uninterpretable regardless of outcome. See
  [research/048](../research/048-run2-final-synthesis.md) and ADR-024
  §"PC3 — THE DISSOCIATION IS CONFIRMED OUT OF SAMPLE". The pre-registration
  below is preserved unaltered and remains valid should a *different* apparatus
  — one shown to steer decisions — become available.
- **Superseded status**: Proposed (pre-registration — frozen before any M5X item is consumed, per ADR-023's
  own discipline, inherited by every rung since). This ADR deliberately breaks the ladder's own
  one-factor-per-rung discipline (ADR-024's standing rule since M3) — which is exactly why it
  needs a pre-registration as careful as any single-factor rung's, arguably more so, since a
  conjunction result cannot be decomposed after the fact the way a single-factor result can.
- **Date**: 2026-08-29.
- **Naming note**: "M5X" is a placeholder name chosen specifically to avoid colliding with M5
  (ADR-024's redesigned receiver-side MicroLoRA rung, itself already superseded once and now
  scoped to run after M4d, per ADR-024 §"M5 SUPERSEDED AND REDESIGNED"). M5X is not part of the
  M5 lineage and does not depend on M5 running first or at all — it is a sibling rung testing a
  different hypothesis (conjunction of factors vs. receiver-side adaptation).
- **Related**: [024](024-run2-trained-thought-adapter-ladder.md) (the ladder this rung belongs to
  — read in full for this ADR, including the LAYER COVERAGE section, the MAJOR CORRECTION
  two-family taxonomy, and the M4g/M4h Stage 1/M4i outcome and pre-registration sections),
  [036](036-successor-rung-evaluation-protocol.md) (the e-process protocol this rung uses as
  primary, being a successor rung under that ADR's own default rule), [032](032-negative-result-publication-contract.md)
  (the publishability bar this rung's eventual outcome must clear — its four criteria are checked
  explicitly against this design below, not assumed satisfied by analogy), [028](028-evolutionary-adapter-search-anti-gaming.md)
  (the evolvable-vs-protected boundary this rung's every changed factor is checked against, and
  the unresolved slot-count contradiction this ADR must rule on for its own purposes without
  adjudicating generally)
- **Evidence base**: `docs/research/044-layer-coverage.md` (the decisive new evidence — C2C's own
  Table 10 single-layer ablation, read in full and cited by exact figure below), `docs/research/040-the-pooling-gap.md`
  (pooling, already refuted as sole cause at M4h Stage 1 but still a live factor to combine
  correctly here), `docs/research/043-placeholder-token-choice.md` (the FIM-token hypothesis M4i is
  currently testing), `docs/research/028-sota-continuous-sweep-1.md` (the task-loss finding, M4c's
  origin), ADR-024's own "LAYER COVERAGE," "M4h Stage 1 OUTCOME," and "MAJOR CORRECTION" sections
  (read immediately before this ADR's authoring, per the task's own instruction, cited by exact
  quote below)

## Context

Four single-factor root-cause candidates have now been tested against this ladder's on-manifold
family, and three are refuted with adversarially-verified receipts: task-loss training (M4f
pre-check — the collapse was already present at zero optimizer steps, not caused by the loss),
injection operator (M4g — fuse changed the NLL inversion by ~0.1 nats and zero win/loss counts),
and pooling (M4h Stage 1 — a geometrically real, un-pooled, item-varying payload still transferred
nothing, and cost slightly *more* NLL than the pooled version of the same weights). A fourth
(placeholder-token choice, injecting onto `<|fim_pad|>` rather than an ordinary token) is being
probed by M4i as this ADR is authored — GPU state checked live this session confirms the card is
currently idle (`ruv_gpu_status`: 0% utilization, 1,693 MiB used, only the standing
`ruvultra-embedder` process resident), consistent with M4i's own probe having completed or being
between phases; this ADR does not assume M4i's outcome either way and is written to stand
regardless of it.

`docs/research/044` supplies the decisive new piece: **the one method in the entire literature
survey that directly ablates single-layer content injection — C2C's own Table 10 — reports
single-layer enrichment moves accuracy from a 58.42% baseline to 58.45-58.52% for its two best
individual layers, roughly 0.1 percentage points, with most other individual layers net-negative.**
Quoted directly from ADR-024's own citation of this finding: "the method that works, reduced to a
single layer, produces exactly our signature — inert, not harmful, indistinguishable from
baseline." No content-transfer method in the survey uses one layer: C2C gates over roughly the
top-5 of 28 layers; LatentMAS transfers the full per-token cache at all layers; the Bicameral Model
couples at exactly 4 fixed layer indices, found by sweeping 890 configurations — a correction to
this repository's own earlier characterization of Bicameral as merely "continuous," which
understated that it is also multi-site.

**The methodological reckoning ADR-024 records, quoted in full because it is this ADR's entire
justification:**

> Every working method combines **multi-layer + continuous delivery + task-loss training
> simultaneously**. This ladder, by design, has only ever varied **one factor at a time** — that
> discipline is what makes each null attributable, and it may also be exactly why every rung
> nulls. If transfer requires a *conjunction* of these properties, no single-factor rung can ever
> succeed, and a one-factor-at-a-time ladder is structurally guaranteed to produce the result we
> have.

If this is right, twelve-plus clean nulls (per ADR-032's own accounting, and growing) would be a
fact about this ladder's own design discipline, not a fact about whether latent transfer is
possible on this stack at all. **This ADR tests the conjunction directly, once, with the same
freeze-before-probe rigor every single-factor rung has already used — and names, in advance, both
what a PASS licenses and what a NULL means, since neither can be worked out after the fact for a
result that deliberately confounds multiple factors.**

## Decision — which factors to combine, and why exactly those

**Included, with justification for each:**

1. **Multi-layer injection (sender L18/L24 → receiver L14/L19, both depth pairs simultaneously).**
   The single largest evidence gap this ladder has left untested, per `docs/research/044`'s own
   central finding. **Zero new capture required** — the per-token dumps already hold both depth
   pairs (`sender_L18.tok.f32bin`, `sender_L24.tok.f32bin`, `receiver_L14.tok.f32bin`,
   `receiver_L19.tok.f32bin`), and no rung to date has ever used the L24→L19 pair (confirmed by
   `docs/research/044` §4(b) via grep across ADR-024). Requires one new `LayerEdit` variant
   (`FuseMany`, not `InjectMany` — see the operator decision below), precedented directly by how
   M4g added `Fuse` beside `Inject` as a wholly separate variant so every existing receipt stays
   reproducible.
2. **De-pooled per-token payloads.** Already implemented and probe-verified at M4h Stage 1 (last-
   token-instead-of-mean, classified `on-manifold-item-varying` by the pre-check, the first
   candidate in the entire ladder to be both). Included because `docs/research/044` §5 names
   pooling and layer coverage as two genuinely independent, both-partial explanations for the
   on-manifold ceiling — combining them, rather than testing layer coverage alone against a still-
   pooled payload, is the more decisive test of the conjunction hypothesis. Reuses M4h Stage 1's
   already-trained, byte-identical weights at the L18→L14 site; the L24→L19 site needs one freshly
   trained MLP of the identical shape and recipe (M3's, not M4c/M4d/M4g's — see the loss decision
   below).
3. **Fuse delivery (residual add, not overwrite).** Implemented at M4g and refuted as a *sole*
   cause, but never yet combined with the two factors above. Included both because it is the
   already-built, receipt-precedented operator (no new engineering) and because it is what C2C
   itself does — Eq. 3 of the paper, verified from source in `docs/research/038` §4, is a residual
   add onto the receiver's own cache, never a replacement. Excluding it from a "maximal
   configuration" rung explicitly modeled on the working comparator method would be an
   unmotivated omission.
4. **Task-loss training.** M4c's finding (the field's own reported recipe — next-token CE through
   the receiver's own output, C2C-style) is included specifically because ADR-024's LAYER COVERAGE
   section frames the working methods as combining "multi-layer + continuous delivery + task-loss
   training simultaneously" — omitting task-loss from a rung explicitly designed to test that
   conjunction would leave the central claim untested. This is the one inclusion decision that
   reopens a factor the M4f pre-check already found to be off-manifold *by itself* (M4c/M4d/M4g's
   family, cosine ≈ −0.02 to +0.05); combining it with de-pooling and multi-layer coverage inside
   one rung is a deliberate wager that the conjunction's other factors (richer per-token payload,
   more injection surface) may pull a task-loss-trained adapter back toward the manifold in a way
   a single-layer, pooled task-loss adapter never had the surface area to achieve — this is stated
   as a hypothesis this rung tests, not a prediction this ADR makes.

**Excluded, with justification for each:**

5. **Ordinary-token injection sites (M4i's axis, `docs/research/043`).** **Excluded from M5X's
   registered configuration, decided now, before M4i's own outcome is known — this ADR does not
   wait for or inherit M4i's result.** Reasoning: M4i is a live, in-flight, single-factor rung
   testing a specific, narrower hypothesis (whether `<|fim_pad|>` is an experientially vacant
   embedding region for a non-Coder Instruct model); folding its untested axis into M5X now would
   mean M5X's own eventual result could not distinguish "the conjunction of layer/pooling/operator/
   loss works" from "M4i's specific fix was doing the real work," defeating M5X's own purpose as a
   test of the *other* four factors' conjunction. If M4i reports a PASS before M5X executes, that
   is grounds for a **separate**, later ADR registering an M5X-plus-ordinary-tokens variant — not
   a retroactive edit to this one.
6. **Continuous per-step injection (the M4e axis, `docs/research/032`/`039`).** **Explicitly
   deferred, decided now.** Two independent reasons, both stated in the source documents this ADR
   is grounded against: (a) it is, by ADR-024's own accounting, "the most expensive and least
   implemented" of the candidate factors — no engineering exists for it today, unlike the other
   four which either reuse existing weights or have direct precedent (`Fuse`) already in the
   codebase; (b) `docs/research/039` already corrected the premise that motivated treating this as
   load-bearing — only the Bicameral Model actually injects continuously among the surveyed
   methods, and it does so bidirectionally, at 4 fixed layers, never as a clean single-axis
   instance of "continuous injection" isolated from multi-layer coupling or bidirectionality. Since
   this ADR already includes multi-layer injection as factor 1, adding continuous delivery on top
   would introduce a fifth confound this rung's attribution trade (below) cannot afford without
   losing even the coarse-grained "did the conjunction work" signal M5X is built to produce. Named
   as the natural next escalation if M5X nulls — not built here.

**Slot-count ruling for this rung, explicit and scoped, not a general ADR-028 adjudication:**
`docs/research/044` §4 names the open tension directly: "8 slots across 2 sites is either 16 total
or a 4+4 split." **This ADR rules: 4+4 (4 slots at each of the two injection sites, 8 slots
total).** Reasoning: ADR-028's protected list names "the frozen S1a/S2b probe protocol itself — the
40 items, the exact sign test, α=0.05, **slot count**, rescale-to-median switch, greedy/batch=1
decoding" as never-evolvable, and separately names "slot count" on its evolvable list — the exact
self-contradiction ADR-024 already flagged and left unadjudicated. **This ADR does not resolve
that contradiction generally** — it is explicitly out of scope, and ADR-028's owner should still
adjudicate it, per ADR-024's own carried-forward recommendation. What this ADR decides, narrowly,
for M5X only: keeping the *total* slot count at 8 (the number every completed rung has used, on
either side of ADR-028's contradictory list) is the reading that stays inside the more
conservative interpretation of "slot count is protected" — an 8-total, 4+4 split changes *where*
the 8 slots go, not *how many* there are. A 16-total configuration would unambiguously violate the
protected-list reading regardless of which side of ADR-028's contradiction is correct, so this ADR
avoids that ambiguity entirely by construction rather than by adjudication.

## The explicit attribution trade — what a PASS licenses, and what it does not

**A PASS at M5X cannot attribute the result to any single included factor.** This is stated in
the clearest possible terms because it is the entire cost of abandoning single-factor discipline:

- **What a PASS licenses**: that transfer is achievable on this model pair, this stack, this
  injection mechanism family, under *some* combination of multi-layer coverage, de-pooled
  per-token payloads, fuse delivery, and task-loss training. This is a genuinely important
  positive result — it would mean the twelve-plus single-factor nulls are a fact about testing
  factors in isolation, not a fact about the underlying possibility of transfer, and it would
  reopen the entire ladder's interpretation.
- **What a PASS does NOT license**: any claim about which of the four included factors mattered,
  how much each contributed, or whether all four are jointly necessary versus some proper subset
  being sufficient. A PASS is a single data point in a four-dimensional configuration space; it
  says the corner tested works, nothing about the space's interior.
- **The decomposition ladder a PASS triggers, registered now**: drop exactly one factor at a time
  from the working configuration, in the reverse order of §"Included" above (drop task-loss first,
  reverting to reconstruction loss on the multi-layer/de-pooled/fuse configuration; if that still
  passes, drop fuse next, reverting to overwrite; then de-pooling; then multi-layer, reverting to
  the single L18→L14 site). Each dropped-factor rung is itself a new, separately pre-registered
  successor rung under ADR-036, evaluated once, honest-fail-path unchanged — this is the mirror
  image of how the ladder built *up* to M5X, now run in reverse to find the minimal sufficient
  configuration. **This decomposition ladder is named here as the required follow-up work a PASS
  triggers; it is not built or scheduled by this ADR.**

## What a NULL means, registered before the run, precise and unsparing

**A NULL under the maximal configuration would be the strongest negative result this project can
produce on this stack.** Stated without softening, because ADR-032's publishability contract
requires confounds and scope be named honestly, and this is the honest scope: a NULL here means
every single factor this ladder has identified as a candidate explanation — task loss, injection
operator, pooling, and now layer coverage — has been refuted **individually**, and their
**conjunction**, tested directly, has also been refuted. `docs/research/044` §5's own closing
argument, restated here as this ADR's registered interpretation of a NULL: two non-exclusive
readings would remain live, and neither is closable by any further single-factor or
already-scoped-multi-factor rung —

1. **A genuinely different conjunction is required** — one this ADR did not test, most plausibly
   continuous delivery (excluded above, §6) layered on top of everything M5X already combines,
   which is architecturally close to reimplementing a scoped Bicameral-style coupled system rather
   than any remaining LatentMesh-shaped rung.
2. **M4b (ADR-035, the receiver-scale arm) becomes the last named, not-yet-run candidate this
   ladder has not touched at all.** Every rung discussed in this ADR, on-manifold and off-manifold
   alike, and M5X itself, uses the same ≤1.5B-parameter receiver. `docs/research/035`'s external
   corroboration (a capability-gap threshold, and the Bicameral Model's own GSM8K degradation when
   the coupled models are close in scale) is untouched by anything M5X tests. **A NULL at M5X, if
   it lands, must be reported alongside this fact explicitly** — not "transfer doesn't work on this
   stack," but "transfer doesn't work on this stack at this receiver scale, under every combination
   of the four other factors tested" — the same distinction ADR-036 already requires between "an
   effect wasn't detected" and "no effect exists," extended here from statistical power to
   hypothesis coverage, per `docs/research/044`'s own closing framing.

**This ADR does not predict which outcome will occur.** Per ADR-036's own closing discipline for
its e-process adoption, restated here: a pre-registration buys attribution clarity, it does not
create or predict an effect.

## Protocol: ADR-036's e-process, with the item-supply and comparability rules inherited unchanged

**M5X is a successor rung under ADR-036's own default table** ("Any later rung (M4f re-scoped,
M4e continuous injection, a future M5) — Yes, by default, unless its own pre-registration
explicitly opts into a different protocol with its own justification"). This ADR does not opt out
— the e-process is M5X's **primary** statistic, drawn entirely from `adaptation-512` in fixed
index order (ADR-036's corrected item-supply rule, not the frozen-40-as-subset framing ADR-036
itself found and corrected as false), with the same 13-item leakage exclusion, the same
never-restarted-after-seeing-`W_t` discipline, and the same betting-rule parameters (`λ=0.30`,
`PASS` at `W_i≥1/α=20`, `N_max≈300`).

**Comparability, restated for this rung specifically**: M5X's e-process outcome is not
comparable to any completed single-factor rung's fixed-40-item sign-test result, for two
independent reasons — the statistical protocol differs (per ADR-036) and the configuration itself
differs by construction (four combined factors vs. one). A future write-up must state both facts
in the same sentence as any headline claim about M5X's result, mirroring ADR-036's own
comparability requirement.

**No secondary fixed-40-item draw is registered for M5X**, unlike ADR-035's amended treatment of
M4b (which retains the old 40-item draw as a secondary statistic). Reasoning: M4b's secondary draw
exists for continuity with the receiver-scale confound's original registration, made before
ADR-036 existed. M5X has no equivalent prior single-factor registration to stay continuous with —
it is a new rung, registered after ADR-036, and adding a fixed-40-item secondary here would
reintroduce exactly the power-floor problem ADR-036 exists to solve, for a configuration
specifically expected (per the conjunction hypothesis) to produce a small `n_disc` if it works at
all. The e-process alone is the correct instrument for this rung.

## Publishability check against ADR-032's own four criteria, before any run

Stated explicitly, per the task's own instruction that this pre-registration be as careful as any
single-factor rung's:

1. **Pre-registration before outcome known**: this document, frozen before any M5X item is drawn.
2. **Mechanics and integrity gates**: M5X inherits the same gate discipline every prior rung has
   used — artifact hash verification, hand-rolled-apply-matches-trained-network golden-pair checks,
   item-set/seed reproduction, transfer-check-before-probe (per M4c/M4d/M4g's own pattern, since
   M5X reuses their task-loss training recipe at the second site) — all required before the
   e-process's first draw, not assumed satisfied by analogy to completed rungs.
3. **Adversarial verification**: required for M5X's result before it is reported as publishable,
   per the standing bar every completed rung has already met (a fresh re-run producing identical
   numbers) — not waived for this rung despite its higher engineering cost.
4. **Confounds named and scoped**: this ADR's own §"What a PASS licenses" and §"What a NULL means"
   sections are exactly this requirement, applied to a conjunction result rather than a
   single-factor one — the attribution trade *is* the confound, named in advance rather than
   discovered at write-up time.

## Cost, from receipted rates

Every figure below is read directly from a committed receipt this session, or explicitly marked
as an extrapolation with the reason stated.

| Component | Receipted precedent | M5X estimate |
|---|---|---|
| `FuseMany` `LayerEdit` variant (engineering, not GPU time) | M4g's `Fuse` addition, described in its own outcome section as "a new enum variant beside `Inject`... every `InjectionSpec` construction site... names its mode explicitly" | One-time engineering cost, not a per-run GPU cost; no receipted time exists for engineering work, correctly excluded from the GPU-hour budget |
| L18→L14 site training | Already trained (M3's weights, reused byte-identical at M4h Stage 1) | **$0 additional** — no retraining needed for this site |
| L24→L19 site training (new, M3-shaped reconstruction-loss recipe, then a second task-loss pass per the included-factors decision) | M3's own training receipt: `run2-m3-training-receipt-cellL18toL14.json wall_clock_s = 47.11 s`; M4c's task-loss training: `run2-m4c-training-receipt-cellL18toL14.json wall_clock_s = 1603.48 s` (0.4454 GPU-h) | Two training passes at the new site — a reconstruction pass (≈47-60 s by M3's precedent) and a task-loss pass (≈1600-1700 s by M4c/M4d/M4g's precedent, all clustering at 1603-1656 s) — **≈0.46-0.48 GPU-h** for the second-site training alone |
| One fixed-40-item draw (for comparison scale only — not part of M5X's own registered protocol) | `run2-m4c-receipt-cellL18toL14-mlp-taskloss-slots8-poolfull-rescaletrue-n40.json wall_clock_s = 422.48 s` | Cited for scale only; M5X uses the e-process, not this draw shape |
| e-process, per-item rate | ADR-036's own derivation from M4c (422.48 s) and M4h Stage 1 (395 s) over 40 items each: ≈10-10.5 s/item | Reused unchanged for M5X — no reason to expect a different per-item probe cost from the statistic alone (the *training* cost differs, the *per-item evaluation* cost does not) |
| e-process, full `N_max≈300` budget | ADR-036's own arithmetic: ≈300 × 10.5 s ≈ 3,150 s ≈ **0.88 GPU-hours** | Reused unchanged, per ADR-036's own worked figure |
| **Total estimated** | | **≈1.3-1.4 GPU-hours** (0.46-0.48 for second-site training + 0.88 for the full e-process budget), an early wealth-crossing costing proportionally less |

### Honest-fail path

Unchanged from every rung in the ladder (ADR-024, ADR-032, ADR-036): the e-process runs once, its
full `W_t` trajectory is committed to the receipt regardless of outcome, and a FAIL (no crossing by
`N_max`) is reported with the complete trajectory and the §"What a NULL means" interpretation
above — never retried with an adjusted configuration against the same or a new e-process draw.
M5X's own configuration (which four factors, which slot split, which two sites) is frozen by this
ADR and does not change based on an intermediate result; if the decomposition ladder (triggered
only by a PASS) is later pursued, each of its steps is its own new pre-registration, per the
standing rule.

## Consequences

Naming the attribution trade explicitly, before any run, is the single most important thing this
ADR does — a conjunction rung that reported a bare "PASS" or "NULL" without the §"What a PASS
licenses" and §"What a NULL means" sections attached would be exactly the kind of overclaimed or
underscoped result ADR-032's publishability contract exists to prevent, applied here to a rung
whose very design (four combined factors) makes overclaiming easier than any single-factor rung
has been. Deliberately excluding M4i's axis and continuous injection, with reasons stated rather
than left implicit, keeps M5X's own eventual result interpretable as a test of exactly four named
factors — no more, no less — rather than an uncontrolled everything-at-once experiment whose
result would answer no question cleanly. The slot-count ruling (4+4, total 8) resolves what this
rung needs without resolving ADR-028's own standing contradiction, which remains a debt owed to a
future ADR, not discharged here.

## What is frozen / what is pending

| Item | Status |
|---|---|
| Included factors (multi-layer, de-pooling, fuse, task loss) and excluded factors (ordinary tokens, continuous injection) with reasoning for each, the slot-count ruling (4+4, total 8, scoped to this rung only), the attribution trade (PASS/NULL interpretation registered in advance, decomposition ladder named), protocol (e-process primary, no secondary fixed-draw), the publishability self-check, and the cost estimate | **Frozen by this ADR** |
| `FuseMany` `LayerEdit` variant | **Not implemented** — named engineering work, precedented by `Fuse`'s addition at M4g |
| L24→L19 site's reconstruction and task-loss training | **Not started** |
| M5X's own e-process draw | **Not started** |
| The decomposition ladder (contingent on a PASS) | **Named, not built, not scheduled** |
| ADR-028's slot-count contradiction (general case) | **Not adjudicated** — explicitly out of scope for this ADR, carried forward as an open item for ADR-028's owner |

## Implementation status

Not implemented. This ADR is a complete pre-registration for the maximal-configuration rung —
factor set, slot allocation, protocol, attribution trade, and cost are all specified in full,
sufficient to execute M5X directly from this document once the `FuseMany` engineering lands.
