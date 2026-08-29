# 036. Successor-rung evaluation protocol: replacing the frozen 40-item sign test for run-2's remaining rungs

- **Status**: Proposed (pre-registration — frozen before any successor-rung item is consumed,
  per ADR-023's own discipline, inherited by every rung since). This ADR unblocks the ladder: per
  ADR-024's own M4h Stage 1 outcome section, "further rungs drawn against the frozen 40-item
  protocol are not capable of detecting the effects we are now looking for" — any further rung run
  on the old protocol before this ADR lands would be a wasted GPU-hour.
- **Date**: 2026-08-29.
- **Related**: [023](023-live-four-condition-run1-pre-registration.md) (the frozen 40-item protocol
  this ADR replaces for successor rungs only — not retroactively, not for completed rungs),
  [030](030-run3-causally-gated-text-pre-registration.md) §3.2 (the e-process design this ADR
  adopts verbatim, with one dataset-provenance correction), [031](031-evidence-receipt-and-statistical-protocol-governance.md)
  (the receipt contract and frozen-probe-is-a-one-shot-resource principle this ADR's item-supply
  decision must respect), [032](032-negative-result-publication-contract.md) (the "never re-run
  under a changed protocol to convert a fail into a pass" rule — this ADR's comparability section
  exists specifically so a protocol change is never mistaken for exactly that), [035](035-m4b-scale-control-pre-registration.md)
  (M4b's statistics section, amended below to match this ADR rather than left silently
  inconsistent), [024](024-run2-trained-thought-adapter-ladder.md) (the ladder whose newest
  sections — the MAJOR CORRECTION, M4g/M4h/M4i outcomes and pre-registrations — this ADR was
  authored against, read in full)
- **Evidence base**: ADR-024's "M4h Stage 1 OUTCOME (2026-08-29) — mechanically successful,
  statistically unmeasurable" section (the problem statement this ADR exists to solve, quoted and
  cited by exact figure below) and its "MAJOR CORRECTION" section immediately following it (the
  two-family null taxonomy that explains *why* discordance collapsed as adapters improved);
  `docs/research/031-statistical-power-and-design.md` (the power-floor arithmetic and the
  Waudby-Smith/Ramdas e-process citations this ADR reuses); every receipt cited by exact field in
  § Cost below, re-verified against the committed JSON this session, not retyped from a prior
  summary; `harness/latentmesh-live/data/{adaptation-512,calibration-4000}.json` and the S1a probe
  receipt's `dataset.indices`, cross-checked by direct set intersection this session (see § Item
  supply — the discrepancy this uncovered)

## Context

The frozen 40-item S1a/S2b protocol (ADR-023, inherited unchanged through M3, M4, M4c, M4d, M4g,
and M4h Stage 1) has stopped being able to detect what this repository is now trying to measure —
not because the protocol is broken, but because the adapters it is evaluating got better. ADR-024's
own account of the mechanism, quoted directly:

> **THE BINDING CONSTRAINT IS NOW THE INSTRUMENT, NOT THE SCIENCE.** Discordant counts across the
> ladder's recent draws: M4d **7** (the only non-limited one), M4g **3** (floor 0.125), M4h-S1
> **2** (floor 0.25). As payloads become *better behaved* — on-manifold, non-destructive,
> item-varying — they perturb fewer items, so discordance **falls** and the frozen 40-item sign
> test loses what little power it had. The better our adapters get, the blinder the probe becomes.

M4h Stage 1 is the sharpest illustration: the ladder's best accuracy to date (24/40), 2 wins and
**zero losses** against baseline, the first payload classified `on-manifold-item-varying` (the
combination every prior rung lacked), and no NLL inversion (the destructive signature that
characterized the off-manifold family) — and the primary statistic cannot say anything about it,
because `n_disc=2` puts the minimum attainable one-sided p at 0.25, five times α=0.05. The receipt
flags this itself (`power_limited: true`). This is not a marginal case; it is the instrument
returning no information at all on the exact rung that most needed one.

`docs/research/035` (surveyed for this ADR, read in full) independently explains *why* discordance
is scarce on this task in the first place: this repository's own 90-92.5% cross-model item
concordance is consistent with GSM8K's item difficulty being correlated across model scale (hard
items are hard for both models, easy items are easy for both) — corroborated by an external
paper's ARC-Challenge arm showing the identical near-zero-effect pattern despite being a knowledge,
not reasoning, task. This does not change what this ADR must decide (a higher-powered protocol),
but it explains why raising N alone, without addressing the underlying low-discordance mechanism,
requires a real budget increase, not a marginal one — the power tables below are sized accordingly.

## Decision 1 — the successor protocol: the e-process, adopted with the dataset-provenance
correction ADR-030 needs anyway

**Adopt the anytime-valid Bernoulli e-process ADR-030 §3.2 already registered for run 3, verbatim
in its betting-rule mechanics, for every run-2 successor rung this ADR governs.** Not a different
route, and not the fixed-N-scaling alternative, for reasons argued below — but the adoption is not
a blind copy-paste; it inherits one correction this ADR's own item-supply investigation surfaced
(§ Item supply) that ADR-030's text does not yet reflect.

**Why the e-process over the two named alternatives:**

- **A larger fixed N with mid-p McNemar** was considered and rejected as the primary route. It
  requires committing to one N in advance, and `docs/research/031`'s own power tables show why that
  commitment is expensive to get wrong: at the ladder's observed ~9.5% discordance rate, roughly
  25-30 discordant pairs are needed for 80% power at a plausible effect size (θ=0.75) — five to six
  times what any 40-item draw has produced. Reaching that many discordant pairs at ~9.5%
  discordance needs on the order of 260-320 items fixed in advance, and a fixed-N design pays the
  full cost of that N even when a genuine effect would have been obvious from the first 50 items
  (as M4h Stage 1's own 2W/0L sweep suggests it might have been, had the instrument been able to
  keep counting). Fixed-N mid-p McNemar remains a legitimate fallback if the e-process proves
  operationally awkward to implement, but is not this ADR's primary route.
- **A paired continuous outcome** (per-item NLL as a Wilcoxon-style statistic) was checked directly
  against this ladder's own receipts by `docs/research/031` §2.4 and found to be **blind to the one
  real effect the ladder has produced** (S1a: exact-sign p=0.0312 vs. Wilcoxon-on-NLL p=0.6834 on
  the identical data). Adopting a continuous statistic that has already been shown not to track this
  ladder's own accuracy effect would be choosing more apparent power over real sensitivity. Rejected
  for the same reason ADR-030 already rejected it.
- **Stratified sampling toward high-discordance-probability items** (items where sender is likely
  right and receiver is likely wrong) is a real, distinct idea from raising N, and this ADR does
  **not** adopt it as primary, for a concrete reason: identifying such items requires scoring both
  models' baseline (no-injection) accuracy on a candidate pool first, which is itself new
  measurement work with its own leakage and pre-registration requirements, and `docs/research/035`
  §5's own Tier-0/Tier-1 validation ladder treats exactly this kind of stratification-by-difficulty
  question as a separate, not-yet-validated research question (whether GSM8K's item-difficulty
  correlation is even the dominant discordance-suppressing mechanism, versus other candidates,
  remains a `docs/research/035` open item). Building a stratified sampler on an unvalidated
  mechanism risks solving the wrong problem. **Named as a live secondary option**: if the e-process
  itself runs out of `adaptation-512` budget without resolving (§ Item supply), stratified
  resampling of the *next* item tranche toward the difficulty band `docs/research/035` §5's Tier-0
  re-analysis identifies is registered as the escalation path, not built here.

**Power arithmetic, stated explicitly, not asserted:** the e-process trades a small amount of
fixed-N power for the anytime-validity guarantee (Ville's inequality on nonnegative martingales — a
real guarantee, not a heuristic, per Ramdas, Grünwald, Vovk & Shafer 2023). Concretely, using the
identical parameters ADR-030 already registered (`λ=0.30`, tuned to θ=0.65; `PASS` at `W_i ≥
1/α=20`; `N_max≈300`, chosen so the ladder's own ~10% discordance rate yields ≈30 discordant
pairs, the count `docs/research/031` §2.2 identifies as necessary for real power at a plausible
effect size): a rung with a strong, unambiguous effect (θ near 1, the shape S1a's own passing draw
showed) would very likely cross the boundary well before the full `N_max`, exactly the property
M4h Stage 1's own 2W/0L (100% win rate on its 2 discordant pairs) suggests might apply here — the
e-process lets that signal keep accumulating past item 40 instead of being declared uninformative
at a fixed, arbitrary stopping point. A rung with a real-but-modest effect (θ≈0.65-0.75) gets to
use the full budget rather than being cut off at 40 items, which is precisely the failure mode
`docs/research/031` §2.2's table already quantifies (even a θ=0.75 effect needs ~25-30 discordant
pairs for 80% power — 40 items yields at most 4-5 by this ladder's own observed rate).

**The e-process is a secondary, invocable statistic for the mandatory-regardless M4b arm** (per
ADR-035, amended below) and the **primary statistic for every other successor rung this ADR
governs** — a distinction stated here because M4b's own registration deliberately kept the
40-item single-draw shape as primary, for reasons specific to that arm (it follows the ladder's
standing one-draw-per-architecture protocol rather than needing a confirmatory scale). This ADR
amends that choice — see § Which rungs it governs, below.

## Decision 2 — item supply, with a discrepancy uncovered and corrected

**Source: `adaptation-512` (train-derived), never `eval-200`/`holdout-100`.** This is the
unsurprising part of the decision — `adaptation-512` is already the ladder's designated
train-pool for exactly this purpose (Darwin fitness batches, per-generation audits), the mechanical
`eval-200`/`holdout-100` lock remains fully engaged (no `genome-frozen.json` receipt exists at the
time of this ADR — verified this session, unchanged from every prior status check), and drawing
successor-rung evidence from a locked, held-out split would be exactly the kind of leakage ADR-024
already built the 13-item exclusion rule to prevent for a different reason.

**The discrepancy, found while verifying this decision, not assumed:** ADR-030 states, in its
"Sampler, dataset, and split discipline" section, that "the frozen 40-item probe (a subset of
`adaptation-512` by construction, per ADR-023's own dataset-pin table) is the fixed evaluation
population for [the e-process]." **This is checked directly against the committed data this
session and found to be materially wrong**: intersecting the S1a probe's 40 item indices against
the committed `harness/latentmesh-live/data/adaptation-512.json` file gives an intersection of
**exactly 1 item** (index 1153), not 40. The probe's actual composition, verified: 22 of 40 items
sit inside `calibration-4000`, 17 of 40 sit in neither `calibration-4000` nor `adaptation-512` (they
were drawn by S1a's own independent ChaCha8 seed over the full `train.jsonl`, per ADR-023's own
"S1a's own 40-item probe already touched indices up to 7462 from a *different*, ChaCha8-seeded
40-item sample of the same file" disclosure — a fact ADR-023 itself already states plainly, which
makes ADR-030's later "subset... by construction" claim an internal contradiction with a document
it cites as its own source). **This ADR corrects the record rather than silently inheriting the
error**: the frozen 40-item probe is **not** a subset of `adaptation-512`; it is a separately-drawn
40-item sample sharing the same source file (`train.jsonl`) with limited, disclosed overlap.

**Consequence for this ADR's own e-process design, stated precisely:** an e-process that "starts"
from the frozen 40-item probe and "extends further into `adaptation-512`" (ADR-030's phrasing)
would therefore be drawing its first ~39 items from one population (the S1a-seeded 40-item draw,
minus the one item already in `adaptation-512`) and its extension items from a second, only
lightly-overlapping population (`adaptation-512` proper). **This ADR does not carry that ambiguity
forward.** The corrected item-supply rule for every rung this ADR governs:

1. **The e-process's item stream is drawn entirely from `adaptation-512`**, in the fixed order the
   file already commits (index order, per `harness/latentmesh-live/data/adaptation-512.json`), not
   from the frozen 40-item probe set at all. This is a clean break from treating the old probe as a
   starting population, made explicit because the "subset by construction" framing that motivated
   treating it as one is now known to be false.
2. **The 13-item leakage-exclusion list (ADR-024's original rule) still applies**: rows already
   used for M3-M4h's training-set fits are excluded from any successor rung's e-process stream on
   the same item-level basis, recorded per-item in the rung's receipt exactly as ADR-024 already
   requires for training data.
3. **Split discipline stays item-level, never token-level** — restating the existing rule, since
   nothing about the e-process changes it: each drawn item in the stream is a whole GSM8K problem,
   never a sub-item token span.
4. **No new set needs freezing before first use, because `adaptation-512` is already frozen** — its
   512 indices and their seed (`0x24C0_DE01` train-shuffle derivation, per `harness/latentmesh-live/src/gsm8k.rs`)
   were committed at S2, long before this ADR. What this ADR freezes is the *consumption order and
   rule* (fixed index order, sequential, leakage-excluded), not a new item selection.
5. **ADR-030's own text should be corrected to match**, since it currently asserts the false
   "subset by construction" premise as its own justification for reusing the frozen 40 as an
   e-process starting point. This ADR does not edit ADR-030's file (append-only discipline,
   ADR-031(a)) — it records the correction here and names it as an open item for whoever next
   amends ADR-030, the same way ADR-024's own correction sections handle a prior document's error
   without silently rewriting it.

## Decision 3 — comparability: two eras, never conflated

**Completed rungs (M3, M4 ×3, M4c, M4d, M4g, M4h Stage 1) were drawn under the frozen 40-item
sign-test protocol. Results under this ADR's e-process protocol are not directly comparable to
those results, and no completed rung may be re-drawn under the new protocol.** Stated plainly,
because the alternative reading — "the old rungs were underpowered, so let's re-run them properly
now" — is exactly the protocol-shopping ADR-032 forbids: "a null is never re-run under a changed
protocol to convert it into a pass." The distinction that makes this ADR legitimate rather than a
disguised re-run: this ADR governs *rungs that have not yet drawn*, not a re-evaluation of rungs
that already have a committed, adversarially-verified receipt under the old protocol.

**How a future write-up must present the two eras, side by side, without implying equivalence:**

- Report every completed rung's original exact-sign p, mid-p McNemar (where already retrofit,
  per ADR-024's own annotations), n_disc, and its power-floor status **exactly as originally
  recorded** — never recomputed under the e-process framework, since the e-process needs a
  sequential item stream the fixed-40 draws never had.
- Report every successor rung's e-process outcome as its **own statistic on its own scale**:
  either a wealth-crossing item count (if it PASSED before `N_max`) or a final wealth value at
  `N_max` (if it did not), never translated into an equivalent "p-value" for cross-comparison
  against the earlier era's sign-test p-values — the two statistics answer structurally different
  questions (a fixed-sample test's p vs. a sequential test's stopping wealth) and a false
  equivalence would misrepresent both.
- A write-up's headline claim about the ladder as a whole must state which rungs were tested under
  which protocol, explicitly, in the same sentence that reports the aggregate finding — mirroring
  `docs/research/041`'s own §5 "scope limits that must travel with every claim" discipline, applied
  to protocol identity rather than to model/task/site scope.
- **If a successor rung PASSES under the e-process where an architecturally similar completed rung
  NULLED under the old protocol** (e.g., a de-pooling Stage 2 rung passing where M4h Stage 1
  nulled), the write-up must state explicitly that this is not evidence the completed rung's null
  was wrong — it is evidence that a higher-powered instrument, applied to a comparable but not
  identical adapter, detected an effect the old instrument could not have detected regardless of
  whether it was present. This is the same distinction ADR-024's own annotations already draw for
  M4c/M4d/M4g's power-floor draws ("could not have detected an effect" vs. "no effect exists"),
  extended to the cross-protocol case.

## Decision 4 — which rungs it governs

| Rung | Governed by this ADR | Statistics used |
|---|---|---|
| **M4i** (inject at ordinary tokens, already pre-registered per ADR-024) | **Yes** | e-process primary, per this ADR; supersedes the "mid-p McNemar primary with exact-sign and n_disc-versus-power-floor reported" language in M4i's own ADR-024 pre-registration paragraph, which predates this ADR and assumed the old fixed-40 shape |
| **M4h Stage 2** (8 distinct per-slot vectors via attention-compression) | **Yes** | e-process primary |
| **M4b** (ADR-035, the receiver-scale arm) | **Yes, with an explicit amendment to ADR-035 §Statistics** — see below | e-process becomes primary (was: optional secondary); mid-p McNemar 40-item draw becomes secondary, reported alongside |
| Any later rung (M4f re-scoped, M4e continuous injection, a future M5) | **Yes, by default**, unless its own pre-registration explicitly opts into a different protocol with its own justification | e-process primary unless stated otherwise |
| **Completed rungs** (M3, M4, M4c, M4d, M4g, M4h Stage 1) | **No — explicitly out of scope**, per § Decision 3 | Original protocol stands; not retroactively re-governed |

**Amendment to ADR-035 §Statistics, stated explicitly per the task's own instruction to check
and say so:** ADR-035 registered the e-process as "an available option, not a requirement" for
M4b, reasoning that M4b follows the ladder's standing one-draw-per-architecture shape rather than
needing a confirmatory scale the way run 3 does. **This ADR's own finding — that the frozen
40-item protocol is now known, ladder-wide, to be systematically underpowered for exactly the class
of well-behaved payload M4b is likely to produce (a reconstruction-trained, on-manifold adapter, per
ADR-035's own adapter choice) — removes the basis for treating the e-process as merely optional for
M4b.** ADR-035's adapter choice was made specifically to avoid the off-manifold, actively-harmful
signature (M4c/M4d/M4g's family) — meaning M4b is, by ADR-035's own design, likely to land in
exactly the low-discordance regime M4h Stage 1 just demonstrated the fixed-40 protocol cannot
measure. **Amended rule for M4b**: the e-process (this ADR's Decision 1-2 machinery, `adaptation-512`
item stream) is the **primary** statistic for M4b's aligned-vs-random comparison; the original
40-item mid-p-McNemar-primary/exact-sign-secondary draw is retained as a **secondary, separately
reported** statistic — reversing which one is primary, not discarding either. This is recorded
here as an explicit amendment to ADR-035 rather than a silent contradiction between the two
documents; ADR-035's own file is not edited (append-only discipline).

## Decision 5 — cost, and what this does and does not fix

**Receipted precedent, re-verified this session, not retyped:**

| Reference point | Receipted value |
|---|---|
| One 40-item fixed-draw probe (M4c's, the task's own cited precedent) | `run2-m4c-receipt-cellL18toL14-mlp-taskloss-slots8-poolfull-rescaletrue-n40.json wall_clock_s = 422.48 s` |
| M4h Stage 1's draw (the cheapest fixed-40 draw in the ladder, per its own outcome section) | "395 s of GPU, no training, no capture" (ADR-024 §"M4h STAGE 1 OUTCOME," "Cost") |

**Scaling to the e-process's budget**, using the ~10.5 s/item rate these two reference points imply
(422-395 s ÷ 40 items ≈ 10-10.5 s/item, a range not a point estimate, stated as such): a
sequential draw that runs to the full `N_max≈300` budget costs on the order of **≈300 × 10.5 s ≈
3,150 s ≈ 0.88 GPU-hours** — comparable to, not dramatically larger than, a single M4c-style
training-plus-probe rung's existing cost, and well inside this ladder's per-rung budget norms.
A draw that crosses the wealth boundary early (the outcome a strong effect would produce) costs
proportionally less — potentially closer to the existing 40-item draw's ≈400 s if `W_t` crosses 20
within the first 40-50 items. Both bounds are extrapolated from receipted per-item rates, not
independently measured for the e-process's own item-by-item overhead (which may differ slightly
from the fixed-batch draw's overhead — flagged, not assumed away).

**Honest-fail path**: unchanged in spirit from every rung's existing discipline (ADR-024, ADR-032).
The e-process's full `W_t` trajectory is committed to the rung's receipt regardless of outcome — a
FAIL (no crossing by `N_max`) is reported with the complete wealth trajectory, not just a final
number, so a reader can see whether the process was trending toward significance and ran out of
budget versus staying flat throughout. **The e-process is never restarted, re-parametrized, or
re-run against the same rung after seeing `W_t`** — identical to ADR-030's own rule, restated here
because it is the single most important discipline this ADR inherits.

**What this ADR does and does not fix, stated plainly, per the task's own instruction:** this ADR
buys statistical power. **It does not create an effect that is not there.** A higher-powered
instrument applied to a truly null adapter will, correctly, fail to cross the wealth boundary by
`N_max` — that is the anytime-valid guarantee working as designed, not a failure of the protocol
change. This ADR's entire justification rests on the observation that the *old* instrument could
not distinguish "no effect" from "an effect too small to move 40 items' worth of discordance" —
it says nothing about which of those two explanations is true for any specific successor rung, and
this ADR does not predict an outcome for M4i, M4h Stage 2, or M4b. The instrument is being fixed
because it went blind at exactly the moment the science needed it most; whether there is anything
for the fixed instrument to see remains an open, rung-by-rung empirical question.

## Consequences

Correcting ADR-030's "subset by construction" claim before it propagates into a second document's
item-supply logic is the single most consequential decision in this ADR — building the e-process's
item stream on a false premise about dataset overlap would have created a leakage or double-counting
risk (drawing from a population partially, but not fully, already exhausted by the frozen probe)
that would only have surfaced once a successor rung's receipt was audited closely. Making the
`adaptation-512`-only rule explicit and clean avoids that risk at the cost of formally abandoning
the frozen 40 as anything but a closed, historical dataset for the completed-rung era — which this
ADR argues is the correct trade, since treating it as still-live risks exactly the kind of
population-ambiguity error this ADR's own item-supply investigation found. The comparability
section's explicit two-era framing is the direct application of ADR-032's own publishability
discipline to a protocol change rather than an architecture change — the first time that discipline
has had to be applied at the protocol level rather than the rung level, and worth stating as
precedent for any future protocol revision this ladder needs.

## What is frozen / what is pending

| Item | Status |
|---|---|
| Successor protocol (e-process, verbatim ADR-030 mechanics with the item-supply correction), item-supply rule (`adaptation-512`-only, fixed index order, 13-item exclusion, no new set needed), comparability rule (two eras, never conflated, exact presentation requirements), which rungs are governed, the ADR-035 statistics amendment, cost estimate and honest-fail path | **Frozen by this ADR** |
| ADR-030's own "subset by construction" text | **Not edited** (append-only discipline) — correction recorded here, flagged as an open item for whoever next amends ADR-030 |
| M4i, M4h Stage 2, M4b, and any later rung's actual e-process execution | **Not started** |
| The stratified-sampling escalation path, and the fixed-N mid-p-McNemar fallback | **Named as available secondary options, not built, not scheduled** |

## Implementation status

Not implemented. This ADR is a complete pre-registration for the successor protocol — item supply,
statistics, comparability, governed rungs, and cost are all specified in full, sufficient to execute
any of M4i, M4h Stage 2, or M4b's amended primary statistic directly from this document.
