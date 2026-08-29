# 030. Run 3 pre-registration: causally-gated TEXT communication

- **Status**: Proposed (pre-registration — frozen before any item is consumed, per ADR-023's own
  discipline). This is the results record once run 3 executes; a results section is appended here,
  not opened as a new ADR, mirroring ADR-023's own S6 pattern.
- **Date**: 2026-08-29.
- **Related**: [003](003-causal-edge-verification.md) (the five-control test this run applies to a
  new channel — text — for the first time), [009](009-online-causal-control-loop.md) (the online
  control loop; run 3 exercises the loop's causal-fitness/authority machinery independent of
  whether the "closed loop across live components" ADR-009 names remains only partially wired),
  [023](023-live-four-condition-run1-pre-registration.md) (the frozen 40-item probe and its
  discipline this run reuses verbatim, and the evidence-label/deviations-ledger conventions this
  ADR follows), [024](024-run2-trained-thought-adapter-ladder.md) (the ladder this run is
  independent of — run 3 does not require, wait for, or depend on any run-2 rung's outcome)
- **Evidence base**: [docs/research/030-economics-and-gate-standalone.md](../research/030-economics-and-gate-standalone.md)
  (the full basis for this ADR — Q1's compute economics and Q2's literature-verified novelty claim,
  read in full and cited by section below), [docs/research/031-statistical-power-and-design.md](../research/031-statistical-power-and-design.md)
  (a dedicated power/multiplicity/test-choice analysis of the ladder's own 10 valid probe draws,
  committed before this ADR froze and incorporated into the Acceptance criteria and Multiplicity
  sections below as an amendment, not a post-hoc correction), ADR-023's own receipts and
  split-discipline code (`harness/latentmesh-live/src/gsm8k.rs`,
  `crates/latentmesh-runtime/receipts/s1a-receipt-*.json`)

## Context

`docs/research/030` establishes two load-bearing findings, independently of whether latent transfer
ever works:

**Q1 (compute economics, §1a-1c):** the compute case for latent transfer at GPU-local/LAN tiers was
never the actual bottleneck. This repo's own receipts (`s2c-generated-dump-receipt.json`) measure
decode at **≈92× the per-token cost of prefill** for this model pair on this GPU — 7.305 ms/token
(Qwen2.5-3B greedy decode) versus 0.0797 ms/token (Qwen2.5-1.5B teacher-forced prefill), over the
identical 719,115-token generated span. Cache-to-Cache's reported 2.0-2.5× speedup over
text-to-text communication is, by the paper's own text, attributable to "eliminating intermediate
text generation" — decode avoidance, matching this repo's measured mechanism exactly, not payload
size. **Causal usefulness, not compute, is the binding constraint** on whether a communication
channel is worth using — which is exactly what ADR-023/024's three nulls (linear map, MLP,
FastGRNN) have been testing and falsifying, one architecture at a time, for the *latent* channel
specifically.

**Q2 (the gate's standalone value, §Q2):** a literature check — Zhang & Emu's causal audit
(arXiv:2607.26773, the paper ADR-023's own OPE/OME/CAG/SSG vocabulary mirrors) applies its
five-message-setting decoy methodology **exclusively to continuous internal representations**;
text appears only as a motivating comparison, never as a channel subjected to the same intervention
methodology, confirmed by direct quote from the paper's own text. "Agents that Matter"
(arXiv:2605.27621) attributes contribution via agent-level Leave-One-Out removal, not
content-level decoy substitution. **No source found combines: (a) text as the channel, (b)
content-level (not agent-level) controls, (c) a formal significance test against multiple decoy
conditions** — this is `docs/research/030`'s own summary of its search, at "reasonable confidence
given the search depth," not a claim of certainty.

**This run tests whether ADR-003's five-control causal gate has standalone value on the channel
agents actually use today — text — independent of whether latent transfer ever works.** It does
not retroactively change any run-1 or run-2 conclusion, and its own outcome, whatever it is, does
not retroactively validate or invalidate the latent-transfer research program either.

## Decision — the frozen registration

### Conditions (frozen, three conditions, mirroring ADR-023's four-condition frame with the two
latent conditions replaced)

| Condition | Channel | Controls tested |
|---|---|---|
| `StaticText` | Text | None — fixed hand-designed pipeline, cost/quality anchor (unchanged in spirit from ADR-023's own `StaticText`) |
| `StaticText+Gate` | Text | ADR-003's controls (four, not five — see below), measured but **not yet used as a Darwin fitness signal**: sender's real text summary vs. `zero`/`random`/`mismatched`/`self_generated` |
| `CausalDynamicText` | Text | Same four controls, used **as the Darwin fitness signal** (ADR-006/009), mirroring `CausalDynamicLatent`'s role in ADR-023 |

**No `DynamicText`-without-gate condition** in this three-condition frame (ADR-023's four-condition
frame had one) — `docs/research/030`'s own minimum-experiment table names exactly these three, and
this ADR follows it rather than inventing a fourth. If a future amendment wants an
ungated-Darwin-on-text condition for symmetry with ADR-023, that is named as a possible extension,
not built here.

**All conditions use the same candle BF16 weights, the same sampler settings, and the same
sender/receiver model pair (Qwen2.5-3B-Instruct → Qwen2.5-1.5B-Instruct) as ADR-023** — only the
channel content and the presence/absence/role of the causal gate vary between conditions.

### `text_equivalent` is dropped — stated explicitly, justified

ADR-003 defines five controls: `zero`, `random`, `mismatched`, `self_generated`, and
`text_equivalent` ("the same content as the real state, serialized to text and re-tokenized
instead of transferred as a latent frame — beating this specifically is what makes a surviving edge
a claim about **latent** communication, not merely communication-vs-silence," quoted from ADR-003's
own decision section). **This control is degenerate when the channel under test is already text —
there is no latent-vs-text distinction left to test, because the channel being tested *is* the text
serialization.** Running `text_equivalent` here would compare text against itself, which is not a
control, it's a tautology. The four remaining controls are unaffected by this reasoning and are
retained in full: `zero` (no message reaches the receiver), `random` (a random token sequence,
length-matched, not the sender's real content), `mismatched` (another episode's real message,
content-shaped but wrong task), `self_generated` (the receiver's own prior output fed back to
itself). This is stated here explicitly, per the coordinator's requirement, rather than silently
dropping a named control without explanation — a reader comparing this ADR's controls list against
ADR-003's five-control table should not have to guess why one is missing.

### Frozen probe reuse — a new experiment, not an extra draw against run 2

**This run reuses the exact 40-item S1a/S2b frozen probe items and seed** (`item_seed_chacha8:
20897`, the same GSM8K-train indices `[141, 150, 850, ..., 7462]` verbatim from
`s1a-receipt-slots8-block19-poolfull-rescaletrue-n40.json`). **This is legitimate and is not an
additional draw against run 2's probe budget, for a specific, stated reason: the channel under
test is different.** Run 1 and run 2's rungs (ADR-023, ADR-024) all test whether a *latent*
mid-layer signal, real or trained, carries causal content — every one of those probes measures the
same underlying question (does the injected vector matter) with a different mechanism producing
the vector. Run 3 tests a *categorically different channel* (text, consumed in-context, no
injection, no alignment transform, no adapter) against the *same* four applicable controls. Using
the same 40 items across both is analogous to running two different experiments on the same
population — legitimate precisely because it is a **separate pre-registration** (this ADR), with
**separate receipts** (a new namespace, `run3-*`, not appended to any `run2-*` or `s1a`/`s2b`
receipt), and **no adapter, transform, or injection code involved at all** — text consumed
in-context by the receiver's own prompt is not a mechanism run 1 or run 2's probe budget accounting
was ever scoped to cover. The frozen-probe item *set* is a reusable population identifier (which 40
GSM8K-train items, at which seed); the frozen-probe *budget* (one draw per registered rung) is
scoped per-ADR, not globally across every possible channel this repo might ever test. Stating this
explicitly, as required: reusing the item set is legitimate here because the question being asked
of those 40 items is new, not because the item-budget discipline is being relaxed.

### Acceptance criteria (frozen, testable exactly as stated, whatever the outcome)

**Amended 2026-08-29** against `docs/research/031-statistical-power-and-design.md` (a dedicated
power/design analysis of the ladder's own 10 valid probe draws to date, committed before this run
executes and therefore incorporated here as part of the freeze, not as a post-hoc correction).
Every change below is cited to that document by section.

- **Primary gate — statistic switched to mid-p McNemar** (`docs/research/031` §2.4, Fagerland,
  Lydersen & Laake 2013, *BMC Medical Research Methodology* 13:91): the primary test is the
  one-sided **mid-p McNemar** statistic on paired accuracy, α=0.05 — `StaticText+Gate`'s gated-text
  condition (`CausalDynamicText`'s frozen-genome evaluation, or `StaticText+Gate`'s fixed-pipeline
  evaluation — both are scored against this same test) **greater than the best (most favorable to
  the null) of the four controls**, mirroring ADR-003's own "significance against the worst
  control, not just the mean" discipline applied here at the condition level. The classic exact
  sign-test p is reported alongside every mid-p value, never in its place — this is strictly more
  powerful at identical data and identical practical Type-I error control (per `docs/research/031`
  §2.4's own citation of Fagerland et al.'s finding that mid-p tracks nominal α better than the
  conservative exact conditional test), at zero additional cost: both statistics are computed from
  the same collected pairs, no new items, no new receipts.
- **Power note — replaced with the structural fact, not a generic caveat** (`docs/research/031`
  §1, §2.1): a one-sided exact sign test's null distribution is `Binomial(n_disc, 0.5)`, and its
  minimum attainable p-value at a given discordant-pair count `n_disc` sets a hard floor on whether
  the test can reject at all — **at `n_disc=3`, the minimum attainable p is 0.125; at `n_disc=4`,
  it is 0.0625; both are above α=0.05, meaning the test is mathematically incapable of rejecting
  the null regardless of the true effect size.** This is not a modeling assumption; it is exact
  binomial arithmetic on the observed discordant-pair count. **Across the 9 cross-model draws in
  the ladder to date (S2b×4, M3×2, M4×3), discordant-pair counts were {3,3,3,4,4,4,4,5,5}
  (`docs/research/031` §1's table) — 6 of 9 landed at `n_disc∈{3,4}`, the dead zone where the exact
  test could not have passed no matter what the data showed.** Run 3's own 40-item arm inherits
  this exact structural risk: if its discordant-pair count also lands at 3 or 4, a "FAIL" verdict
  from the exact-sign statistic alone would be uninterpretable as evidence about the true effect —
  which is precisely why the primary statistic is mid-p McNemar (above, roughly halving the
  attainable p at the same `n_disc`) and why the confirmatory scale below is powered explicitly
  against this floor, not just described qualitatively as "detects only large effects."
- **Confirmatory scale — replaced with an anytime-valid e-process, not a second fixed-N look**
  (`docs/research/031` §3, Waudby-Smith & Ramdas 2023, *JRSS-B* 85(1), arXiv:2010.09686; Ramdas,
  Grünwald, Vovk & Shafer 2023, *Statistical Science* 38(4):576-601, arXiv:2210.01948). In place of
  the originally-proposed "40-item and eval-200 scales pre-registered together," the primary
  `CausalDynamicText`-vs-best-control comparison is evaluated as a **one-sided Bernoulli e-process**
  over the stream of discordant items drawn, in order, from `adaptation-512` (never
  `eval-200`/`holdout-100` — the existing split-discipline lock is unchanged). Verbatim registration
  text, from `docs/research/031` §3.2:

  > Wealth initializes at `W_0 = 1`. For each new item `i`: if concordant, `W_i = W_{i-1}` (no
  > update); if discordant, `W_i = W_{i-1} · (1 + λ(X_i − 0.5))`, `X_i = 1` if the gated-text
  > condition wins the pair, else 0, with betting fraction `λ` fixed in advance (`λ = 2θ−1 = 0.30`,
  > tuned to the smallest interesting effect `θ=0.65`; a mixture/universal betting strategy that
  > does not commit to one θ is a documented alternative if a single target is judged too
  > restrictive to freeze). **PASS** the instant `W_i ≥ 1/α = 20` (α=0.05). If `W_i` never crosses
  > 20 by a pre-registered budget `N_max ≈ 300` items (chosen so the ladder's own observed ~10%
  > discordance rate yields ≈30 discordant pairs — the count `docs/research/031` §2.2 shows is
  > needed for real power at a plausible effect size), the rung is a registered FAIL, its full
  > `W_t` trajectory is committed to the receipt regardless of outcome, and the ladder proceeds
  > exactly as today's no-rerun discipline requires. **The e-process is never restarted,
  > re-parametrized, or re-run against the same rung after seeing `W_t`.**

  This supersedes the original two-fixed-scales design specifically because it gets the
  peeking-safety property natively (Ville's inequality on nonnegative martingales — a real
  guarantee, not a heuristic) instead of by committing in advance to exactly two look-points, and
  it can stop early on an unambiguous win (a real θ near 1, like S1a's own signal, would very
  likely cross the boundary well before the full budget) or extend to the full `N_max≈300` on
  ambiguity rather than being declared FAIL at whatever fixed N the design happened to freeze at.
  **Cost accounting, both bounds pre-registered**: the minimum-item-cost bound is the same
  ≈0.26 GPU-hours as a 40-item draw (`docs/research/030`'s own receipted per-item rates: sender
  generation ≈2.05 s/item, 5 receiver generations ≈4.27 s each), if `W_t` crosses 20 early; the
  maximum-budget bound at `N_max≈300` is ≈1.3×(300/200)≈**≈1.95 GPU-hours**, scaled from
  `docs/research/030`'s own eval-200 estimate by item count, using `adaptation-512` items
  throughout (never `eval-200` — the e-process runs entirely within the existing train-pool split).
- **Secondary — uplift magnitude**: report the raw accuracy delta (gated-text minus best control)
  with its exact binomial confidence interval, regardless of whether the primary gate passes —
  matching ADR-023's A2-style "no threshold, reported to pre-empt an accounting-artifact objection"
  pattern.
- **Secondary — compute cost**: report GPU-seconds and wall-clock per condition, computed exactly
  as `docs/research/030`'s own Q1 cost estimate methodology (sender generation once per item,
  receiver generation once per condition-control combination) — not re-derived from scratch, since
  the methodology and its receipted per-token rates already exist.
- **Secondary — continuous-outcome (NLL) statistic explicitly rejected as co-primary**
  (`docs/research/031` §2.4): a paired Wilcoxon signed-rank test on per-item gold-token NLL was
  checked against the ladder's own receipted data, not assumed to be more sensitive on textbook
  grounds. The result is a negative finding, stated here rather than silently omitted: on the one
  draw with a real signal (S1a run 2, exact sign p=0.0312), the Wilcoxon-on-NLL statistic gives
  p=0.6834 — **the continuous outcome is blind to the one real effect the ladder has produced.**
  ADR-023 already flagged this for S1a specifically; `docs/research/031` confirmed it holds across
  every other draw checked (S2b gold L18→L14, M3 per-token, M4 r=256 — all show the same pattern:
  NLL tracks nothing the accuracy statistic tracks). The likely mechanism: gold-token NLL measures
  confidence in a single teacher-forced final-answer token, a different construct from whether the
  actual greedy generation reached the correct final answer through its own chain. **NLL is
  therefore not pre-registered as a co-primary or replacement statistic for run 3.** If a
  continuous outcome is wanted in a future run, `docs/research/031` names log-probability of the
  correct numeric answer marginalized over the sampled generation as a better candidate — it would
  need new instrumentation, not a re-read of existing receipts, and is named as follow-up work, not
  built here.
- **The "best control" selection — timing pinned down** (`docs/research/031` §5): the primary
  gate's comparison against "the best (most favorable to the null) of the four controls" is
  evaluated **post-hoc from the same draw**, chosen as whichever of `zero`/`random`/`mismatched`/
  `self_generated` scores highest on that draw's own data, not pre-ranked in advance from ADR-003's
  prior characterization of the controls. **This is stated explicitly because it makes the
  composite test slightly conservative in the useful direction** — comparing the gated-text
  condition against whichever control looks strongest in this specific draw is a harder bar to
  clear than comparing against a fixed a priori control would be, the opposite of the usual
  post-hoc-selection multiple-comparisons risk, and is disclosed here so a reviewer does not have
  to work this out independently.

### The eval-200 lock — handled explicitly, not glossed over

**ADR-023 created a mechanical lock on `eval-200`/`holdout-100`** (`harness/latentmesh-live/src/gsm8k.rs`'s
`eval_items()`/`holdout_items()`, which refuse until `receipts/genome-frozen.json` exists at
`harness/latentmesh-live`'s crate root — verified this session: no such file exists yet, the lock
is still fully engaged). **Run 3's `CausalDynamicText` condition is the milestone that satisfies
this requirement**, because it is the first run-3-or-earlier condition that actually runs a Darwin
loop to genome freeze (ADR-023's own `DynamicLatent`/`CausalDynamicLatent` conditions were
specified to do this too, but never ran — run 1 stopped at the S2b bridge probe, before any Darwin
generation executed). Concretely: `CausalDynamicText`'s Darwin loop (fitness = ADR-003's four
applicable controls on `adaptation-512`-drawn text-condition batches, mirroring
`CausalDynamicLatent`'s specified role in ADR-023 §4) runs to convergence, freezes its genome, and
writes `receipts/genome-frozen.json` — **this is the same file every future condition (run 1's own
unrun `DynamicLatent`/`CausalDynamicLatent`, or any future run's) would also check**, so writing it
here permanently unlocks `eval-200`/`holdout-100` for every subsequent stage of this repo's
experimental work, not just for run 3 itself.

**This is handled carefully, as instructed, with two explicit safeguards:**

1. **The genome-frozen receipt records which run and which condition froze it**, not just that
   freezing happened — `receipts/genome-frozen.json` must carry a `frozen_by` field naming
   `"run3-causaldynamictext"` and this ADR's number, so a future reader inspecting why the lock
   opened knows exactly which experiment's Darwin loop did it, and can distinguish "the lock opened
   because run 3's text-channel Darwin loop converged" from "the lock opened because a latent-channel
   Darwin loop converged" — a materially different provenance fact for anyone auditing what
   `eval-200`/`holdout-100` results downstream actually validate.
2. **Opening this lock does not retroactively grant run 1 or run 2 access to `eval-200`/`holdout-100`
   for anything already concluded.** Run 1 is concluded (ADR-023 § S6, negative result, receipts
   final). Run 2's rungs (ADR-024) that have already reported outcomes (M3, M4) used only
   `adaptation-512`; they are not reopened or re-run against `eval-200` merely because the lock is
   now technically satisfiable. Any *future* run-2 rung (M4b, M4c, M5, or any later addition) that
   wants to use `eval-200` must say so explicitly in its own receipt and cite this ADR's genome-freeze
   event as the reason access is possible — the lock opening is a mechanical fact about the harness,
   not an automatic grant of scope to every experiment that happens to run afterward.

### Sampler, dataset, and split discipline — reused verbatim from ADR-023

No new dataset pins, no new split-generation seeds. `calibration-4000`/`adaptation-512` supply
`CausalDynamicText`'s Darwin fitness batches (train-derived only, same rule as ADR-023's
adaptation-pool-from-train-only discipline) **and, per the amendment above, the entire e-process
item stream for the primary statistical test** — the frozen 40-item probe (a subset of
`adaptation-512` by construction, per ADR-023's own dataset-pin table) remains the fixed starting
population the e-process draws its first items from, extending further into `adaptation-512` up to
`N_max≈300` only if `W_t` has not crossed the significance boundary. **`eval-200` is not part of
the primary hypothesis test in any form after this amendment.** Its remaining role, unchanged from
before the amendment, is a single confirmatory scoring pass of the frozen genome after
`CausalDynamicText`'s Darwin loop converges and writes `receipts/genome-frozen.json` — reported as
descriptive evidence (accuracy, cost, per ADR-023's own A8-style "any eval-vs-adaptation-pool
discrepancy is reported, never grounds for a rerun" discipline), not as a second statistical test
subject to its own pass/fail threshold. Sampler: same paper-mirrored arm (T=0.6/top-p 0.95/1024-token
cap) and greedy witness arm as ADR-023, unless a receiver-generation step specifically requires
deterministic greedy decoding for a control condition (`self_generated` needs the receiver's own
prior deterministic output to be well-defined) — that exception is stated here, not discovered
mid-run.

**Darwin loop seed**: `latentmesh-run3-audit-run-seed` = `11796393239420137246`
(`0xa3b53526ba74ef1e`), derived by the identical `SHA-256(label)` → big-endian `u64` formula ADR-023
registered for its own (unused) audit-run seed — a fresh, explicitly-labeled derivation for run 3,
not a reuse of ADR-023's `darwin-genome`/`audit-run-seed` constants, which remain scoped to any
future latent-channel Darwin loop that might still use them.

### Multiplicity — the ladder's family, stated before this run's result exists

**Amended 2026-08-29, per `docs/research/031` §4** (a dedicated multiplicity audit of every valid
probe draw run so far). The frozen 40-item S1a/S2b protocol has produced **10 valid probe draws to
date** across ADR-023/024 (1 same-model mechanics check that passed, 9 cross-model transfer tests
that all failed) — this run's `CausalDynamicText` primary test is draw #11 in that lineage,
whatever channel it tests. Two facts govern how it must be reported:

- **The ladder's own between-architecture ordering (M3 → M4 → M5, run 1 → run 2 → run 3) is
  already a valid fixed-sequence/gatekeeping procedure and needs no multiple-comparisons
  correction between its steps** (`docs/research/031` §4.3, citing Maurer, Hothorn & Lehmacher
  1995 and Westfall & Krishen 2001 on fixed-sequence gatekeeping; §4.4 additionally frames the
  whole ladder as structurally closest to a multi-arm multi-stage platform-trial design with a
  shared control and arms added/dropped for futility). This holds specifically because each rung
  is evaluated only after its predecessor's result is reported, never in parallel with it — run 3
  satisfies this by construction, since it is authored and would execute after M3/M4's outcomes
  are already on record.
- **Within-rung parallel variants need Holm-Bonferroni**, not the gatekeeping exemption above. This
  run has no internal parallel variants comparable to S2b's two-cell or M3's two-variant structure
  — `StaticText+Gate` and `CausalDynamicText` are sequential stages of one condition set, not
  parallel candidates competing for the same probe draw — so no within-rung correction is needed
  here. If a future amendment adds a parallel variant to this run (e.g. testing two different
  `mismatched`-control constructions side by side), Holm-Bonferroni at that family's own α applies,
  per the same rule `docs/research/031` states for the rest of the ladder.
- **Whatever this run's outcome, it must be reported as "draw #11 in a family of N architecture/
  channel tests," not as if it were the only test ever run** — the same discipline
  `docs/research/031` §4.2 requires of any future run-2 rung that eventually passes. A pass here
  does not exempt this run from that family-size disclosure merely because it tests a different
  channel than the 10 prior draws.

## Positioning — independent of whether latent transfer works

**This run does not retroactively change any run-1 or run-2 conclusion, and does not require any
run-2 rung to have passed, failed, or even run at all.** It tests a structurally separate claim:
whether ADR-003's gate, applied to the channel every LLM agent pipeline already uses today, adds
measurable value over ungated text — a question with an answer independent of whether any latent
encoding scheme (linear, MLP, FastGRNN, or a future rung) ever clears its own frozen probe. A
negative result here says "the gate adds nothing measurable on this task/model pair for text";
it does not imply anything about latent transfer's prospects, any more than a positive result here
would imply latent transfer will eventually work. The two research programs are evaluated on their
own evidence.

## Novelty claim — hedged, not asserted as certain

**"Appears unclaimed, at reasonable confidence given the search depth in `docs/research/030`."**
Two nearest works, and exactly what each does differently:

- **Zhang & Emu** (arXiv:2607.26773) — the closest methodological relative (ADR-023's own
  OPE/OME/CAG/SSG vocabulary is mirrored from this paper). Applies decoy-controlled message
  replacement, but **exclusively to continuous internal representations** — confirmed by direct
  quote of the paper's own text, which frames text as a motivating comparison, never as a channel
  the audit methodology is applied to.
- **"Agents that Matter"** (arXiv:2605.27621) — the closest text-communication attribution work.
  Uses Leave-One-Out **agent** removal, not content-level decoy substitution — its causal claim is
  "this agent's presence/absence changes the outcome," never "this specific message content, versus
  content-matched noise, changes the outcome."

No source found combines content-level decoy controls with a text channel and a formal significance
test. This is stated as this ADR's honest confidence level, not as an assertion of priority —
`docs/research/030`'s own search was WebSearch/WebFetch-based, not an exhaustive literature review,
and a paper doing exactly this could exist and not have surfaced.

## What this experiment will NOT claim

1. **No claim about latent transfer's prospects, either direction** — per the Positioning section
   above, this run's outcome does not bear on ADR-023/024's latent-channel research program.
2. **Not a claim that ADR-003's gate is novel in the abstract** — only that its *application to a
   text channel with content-level decoy controls* appears unclaimed, per the hedged novelty
   section above.
3. **`radio-moe`'s fusion result (cited in ADR-029, corrected framing) is not a competing or
   preempting claim.** `radio-moe` verifies source provenance/independence, never applies a decoy
   condition to any expert's message content — a different, complementary question. This run is not
   redundant with it.
4. **A PASS reached quickly by the e-process (few discordant items, early wealth crossing) is not
   thereby weaker evidence than a PASS reached near the `N_max≈300` budget** — anytime-valid
   inference's whole guarantee (Ville's inequality on the wealth process) is that a crossing is
   equally valid whenever it happens; conversely, a FAIL at `N_max` does not mean "no effect
   exists," only that no effect large enough to cross the pre-registered betting boundary within
   budget was found — restating `docs/research/031` §2.1's "could not have passed" caveat for run
   1/run 2 in this run's own sequential terms, before any result exists to motivate hedging it.
5. **This run does not claim to have resolved which of the four remaining controls (having dropped
   `text_equivalent`) is doing the most work** — that decomposition, if wanted, is named as future
   analysis, not computed here as part of the primary/secondary criteria above.

## What is frozen / what is pending

| Item | Status |
|---|---|
| Three conditions, dropped `text_equivalent` (justified), primary statistic (mid-p McNemar) and its structural power floor, the e-process sequential test with its betting rule/λ/N_max, secondary criteria (uplift, compute cost), NLL explicitly rejected as co-primary, the "best control" selection-timing disclosure, multiplicity framing (gatekeeping between rungs, this run as draw #11), sampler/dataset/split reuse, the Darwin-loop seed, the genome-freeze provenance requirement, positioning, hedged novelty claim, will-not-claim list | **Frozen by this ADR (amended 2026-08-29 against `docs/research/031`)** |
| The frozen 40-item probe's item set and seed | **Reused verbatim from ADR-023** — not redrawn; now the e-process's starting population, extending into `adaptation-512` up to `N_max≈300` only if needed |
| `receipts/genome-frozen.json` | **Not yet written** — will be produced by `CausalDynamicText`'s Darwin loop converging, per the lock-handling discipline above |
| Run execution | **Not started** |

## Implementation status

Not implemented. This ADR is a complete pre-registration — conditions, controls, acceptance
criteria, dataset/split/sampler discipline, and the genome-freeze handling are specified in full,
sufficient to execute the experiment directly from this document. Execution, receipts, and a
results section (appended here, mirroring ADR-023's own S6 pattern, not opened as a new ADR) are
the next concrete step.
