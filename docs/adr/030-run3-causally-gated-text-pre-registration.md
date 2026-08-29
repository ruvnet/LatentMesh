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
  read in full and cited by section below), ADR-023's own receipts and split-discipline code
  (`harness/latentmesh-live/src/gsm8k.rs`, `crates/latentmesh-runtime/receipts/s1a-receipt-*.json`)

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

- **Primary gate**: one-sided exact sign test, α=0.05, paired accuracy, `StaticText+Gate`'s
  gated-text condition (`CausalDynamicText`'s frozen-genome evaluation, or `StaticText+Gate`'s
  fixed-pipeline evaluation — both are scored against this same test) **greater than the best
  (most favorable to the null) of the four controls** — mirroring ADR-003's own "significance
  against the worst control, not just the mean" discipline (`verify_edge`'s documented
  stricter-than-mean bar), applied here at the condition level.
- **Secondary — uplift magnitude**: report the raw accuracy delta (gated-text minus best control)
  with its exact binomial confidence interval, regardless of whether the primary gate passes —
  matching ADR-023's A2-style "no threshold, reported to pre-empt an accounting-artifact objection"
  pattern.
- **Secondary — compute cost**: report GPU-seconds and wall-clock per condition, computed exactly
  as `docs/research/030`'s own Q1 cost estimate methodology (sender generation once per item,
  receiver generation once per condition-control combination) — not re-derived from scratch, since
  the methodology and its receipted per-token rates already exist.
- **Power note, pre-registered**: a 40-item sign test detects only large effects (per ADR-023's own
  S6 discussion of S1a's p=0.03125 minimum-attainable-value at 5 discordant pairs on 40 items) —
  this is disclosed here as a limitation of the frozen-probe scale, not discovered after a null
  result. **Both scales are pre-registered together, not sequentially**:
  1. **The 40-item frozen-probe run** — `docs/research/030`'s own receipted cost estimate: 40 items
     × one sender generation (≈2.05 s at 281 mean tokens × 7.3 ms/token) + 40 items × 5 receiver
     generations (real + 4 controls, ≈4.27 s each) ≈ 936 s ≈ **≈0.26 GPU-hours**.
  2. **The `eval-200`-scale run** — the same per-item cost scaled by 200/40 ≈ **≈1.3 GPU-hours**.
  Both figures are receipted arithmetic from `docs/research/030` §"The minimum experiment," not
  fresh estimates made for this ADR.

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
adaptation-pool-from-train-only discipline); the frozen 40-item probe (a subset of `adaptation-512`
by construction, per ADR-023's own dataset-pin table) is the fixed evaluation population for both
scales above; `eval-200` is touched only by `CausalDynamicText`'s frozen-genome evaluation, after
its own genome freeze, per the lock discipline above. Sampler: same paper-mirrored arm
(T=0.6/top-p 0.95/1024-token cap) and greedy witness arm as ADR-023, unless a receiver-generation
step specifically requires deterministic greedy decoding for a control condition (`self_generated`
needs the receiver's own prior deterministic output to be well-defined) — that exception is stated
here, not discovered mid-run.

**Darwin loop seed**: `latentmesh-run3-audit-run-seed` = `11796393239420137246`
(`0xa3b53526ba74ef1e`), derived by the identical `SHA-256(label)` → big-endian `u64` formula ADR-023
registered for its own (unused) audit-run seed — a fresh, explicitly-labeled derivation for run 3,
not a reuse of ADR-023's `darwin-genome`/`audit-run-seed` constants, which remain scoped to any
future latent-channel Darwin loop that might still use them.

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
4. **A passing primary gate at 40-item scale is weaker evidence than a passing eval-200-scale
   result** — both are pre-registered together specifically so a 40-item pass is never
   over-interpreted as the final word; the eval-200 run is the confirmatory scale.
5. **This run does not claim to have resolved which of the four remaining controls (having dropped
   `text_equivalent`) is doing the most work** — that decomposition, if wanted, is named as future
   analysis, not computed here as part of the primary/secondary criteria above.

## What is frozen / what is pending

| Item | Status |
|---|---|
| Three conditions, dropped `text_equivalent` (justified), primary/secondary acceptance criteria, both pre-registered scales, sampler/dataset/split reuse, the Darwin-loop seed, the genome-freeze provenance requirement, positioning, hedged novelty claim, will-not-claim list | **Frozen by this ADR** |
| The frozen 40-item probe's item set and seed | **Reused verbatim from ADR-023** — not redrawn |
| `receipts/genome-frozen.json` | **Not yet written** — will be produced by `CausalDynamicText`'s Darwin loop converging, per the lock-handling discipline above |
| Run execution (both scales) | **Not started** |

## Implementation status

Not implemented. This ADR is a complete pre-registration — conditions, controls, acceptance
criteria, dataset/split/sampler discipline, and the genome-freeze handling are specified in full,
sufficient to execute the experiment directly from this document. Execution, receipts, and a
results section (appended here, mirroring ADR-023's own S6 pattern, not opened as a new ADR) are
the next concrete step.
