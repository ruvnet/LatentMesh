# 032. Negative-result publication contract

- **Status**: Proposed
- **Date**: 2026-08-29
- **Related**: [031](031-evidence-receipt-and-statistical-protocol-governance.md) (the receipt
  contract and statistical-protocol rules this ADR's publishability criteria are built on),
  [023](023-live-four-condition-run1-pre-registration.md) §S6 (the worked example this ADR
  formalizes), [024](024-run2-trained-thought-adapter-ladder.md) (M3/M4's honest-fail outcomes,
  the second and third worked examples), [014](014-benchmark-and-acceptance-method.md)
  (evidence-label discipline a null must still carry)
- **Evidence base**: ADR-023 § S6 — Results (read in full, cited by exact value), `docs/research/025-run1-negative-result.md`
  (the narrative writeup this ADR names as the format template), ADR-024's M3/M4 outcome sections
  and their cited receipts (`run2-m3-receipt-cellL18toL14-mlp-{pertoken,pooled}-*.json`,
  `run2-m4-receipt-cellL18toL14-fastgrnn-r{64,128,256}-*.json`)

## Context

This repository has now produced four adversarially-verified nulls: run-1's gold-calibration
bridge probe (ADR-023 S6, two cells — L18→L14 p=0.5000, L24→L19 p=0.8750), run-1's
generated-pairs contingency (same two cells, p=0.3125 and p=0.8750), M3's two trained-MLP variants
(ADR-024, per-token p=0.6875, pooled p=0.5000), and M4's three FastGRNN ranks (ADR-024, p=0.3125,
0.3125, 0.1875). Most research projects have no discipline for this outcome and either bury a null
(discarding the evidence that a direction doesn't work, so the next attempt repeats it) or report an
unattributed failure as if it settled a broader question than it actually tested. This repository
has, in practice, done neither — every null above was pre-registered, gated on integrity checks, and
adversarially re-verified before being written up — but the criteria that made each one trustworthy
have never been stated as a standing bar. This ADR states that bar.

## Decision

### What makes a null PUBLISHABLE here, versus merely recorded

A null is **publishable** — eligible to be written up as a negative-result artifact per the format
below, and cited elsewhere as settling the specific claim it tested — only when all four hold:

1. **Pre-registration existed before the outcome was known.** The threshold, the item set, the
   statistical test, and the calibration/training procedure were frozen (per [ADR-031](031-evidence-receipt-and-statistical-protocol-governance.md)(b)(1))
   before the receipt that reports the null existed. A result computed against a threshold chosen
   after seeing the data is not a null in this sense — it is an unregistered observation, and this
   ADR does not grant it the same status.
2. **Mechanics and integrity gates passed, so the null is attributable to the hypothesis under
   test, not to broken plumbing.** Every null cited above carries this explicitly: ADR-023's A7(a)
   (capture-path logits bit-identical) and A7(c) (zero-vector injection not catastrophic) both
   passed at every cell before A7(b) was even evaluated; M3 and M4's receipts record
   `hand_rolled_apply_matches_align_crate` / equivalent artifact-integrity gates passing alongside
   the null result. A null where the mechanics gates themselves failed is not publishable as a
   finding about the hypothesis — it is a bug report.
3. **Adversarial verification reproduced it.** ADR-023 S6 states plainly: "Adversarially verified:
   fresh GPU re-run bit-identical, all statistics independently recomputed." ADR-024's M4 outcome
   states the same for the r256 sub-rung: "r256 probe re-run bit-identical." A single, unreproduced
   run is not sufficient — the standing bar is a second, independent execution against the same
   frozen inputs producing the identical numbers.
4. **Confounds are named and scoped, not discovered after the fact and left unstated.** Every null
   above ships with its own registered confound: the receiver-scale threshold (ADR-024, Qwen2.5-1.5B
   sits below the ~1.7B threshold arXiv:2608.05164 reports as necessary for reliable cross-model
   steering transfer — named *before* M3's probe result was known), the loss-function confound
   (reconstruction-vs-task-loss, named before M4's verdict), and run-1's own generated-vs-gold
   calibration-distribution-shift caveat (tested directly, not merely disclosed). A null with no
   named confound is either a stronger claim than this repository's evidence actually supports, or
   an incomplete write-up.

A result that fails criterion 1 or 2 is **not publishable as a negative result** — it is either an
unregistered observation (criterion 1) or an inconclusive run (criterion 2), and must be labeled as
such rather than folded into the negative-result record. A result that fails criterion 3 or 4 is
publishable only with that gap explicitly disclosed as an open item, per the honesty discipline
ADR-014 and ADR-023 already apply to every evidence label.

### What a negative-result artifact must contain

Modeled directly on `docs/research/025` (the worked example named by this ADR) and ADR-023 § S6:

1. **The claim NOT supported, stated precisely and narrowly.** Not "latent transfer doesn't work" —
   ADR-023 S6 §3 states the actual scope: "a training-free linear (Procrustes-family) alignment of
   pooled residual-stream states, fit from either gold-teacher-forced or sender-generated calibration
   pairs, carries causally usable cross-model signal at L18→L14... or L24→L19... for
   Qwen2.5-3B→Qwen2.5-1.5B. At high confidence... no." Every noun in that sentence is load-bearing:
   the architecture, the pooling, the calibration source, the exact cells, the exact model pair. A
   negative-result artifact that generalizes beyond what was actually tested is a claim this
   contract does not license.
2. **An attribution chain that closes off the standing alternative explanations.** ADR-023 S6 §2
   is the template: (i) mechanics transmit information (rules out "the pathway is broken"), (ii) a
   hand-rolled reimplementation reproduces the reference crate bit-exactly (rules out "the
   reimplementation is wrong"), (iii) an identifiability check on the fit (rules out, or explicitly
   flags, a sample-size boundary case), (iv) a reproducibility check (rules out "this was a drifted
   re-run"). Each link names the alternative it closes and cites the receipt that closes it — an
   attribution chain that asserts a conclusion without walking through the alternatives it rules out
   does not meet this bar.
3. **The registered confounds that scope the null**, restated from the pre-registration, not
   invented at write-up time (see criterion 4 above).
4. **Full receipts**, per the ADR-031 contract, with every number in the write-up traced to a
   specific field in a specific committed file — ADR-023 S6's own stated rule ("nothing is
   estimated") and its one self-caught exception (the unreconciled "~5.7 GPU-h" vs. the
   receipt-summed 3.3 GPU-h, flagged rather than silently resolved) are both part of what this
   contract requires: **an unreconciled discrepancy is disclosed, never rounded away.**

### A null is never re-run under a changed protocol to convert it into a pass

This is the rule ADR-024's ladder already lives by and this ADR generalizes: a failing rung
escalates to the **next architecture** (M3 → M4 → M4c/M5), it is not retried with adjusted
hyperparameters against the same frozen probe. The distinction that matters: **escalating to a
different, pre-named contingency is allowed and expected** (M4c's task-loss ablation was registered
*before* M4's verdict was known, exactly the honest-fail-path discipline ADR-031(b)(4) requires) —
what is never allowed is re-running the *same* architecture-and-probe pairing with different
knobs until one draw happens to pass, or discarding a passed pre-registration criterion because the
result was inconvenient. A later arm that succeeds under a controlled confound (e.g. a future
scale-control arm at ≥1.7B receiver, or M4c's task-loss training) does not retroactively invalidate
or "unpublish" the earlier null — it produces a **new** receipt and a **new**, narrower-scoped
finding, exactly as ADR-031(a)'s append-only rule requires. The earlier null remains true of what it
actually tested.

## Boundary table — publishable vs. merely recorded

| Status | Criteria | Example from this repo |
|---|---|---|
| **Publishable negative result** | Pre-registered before outcome known; mechanics/integrity gates passed; adversarially reproduced; confounds named and scoped | Run-1 bridge probe (ADR-023 S6); M3 both variants; M4 all three ranks |
| **Recorded but not yet publishable** | Pre-registered, gates passed, but not yet independently re-verified, or a confound not yet named | None currently in this repo — named as a category so a future fast-turnaround result has a place to sit honestly before its adversarial-verification pass completes |
| **Inconclusive run, not a negative result** | Mechanics or integrity gates failed | S1a run-1 (RoPE bug + scoring bug) — correctly preserved as a *diagnosed bug*, not reported as a negative finding about the injection channel |
| **Unregistered observation, not a negative result** | No pre-registration, or threshold/test chosen after seeing the outcome | Not currently produced by this repo's practice; named as the category this contract exists to prevent |

## Consequences

This contract makes the cost of a null explicit and bounded rather than open-ended: a rung that
fails all four criteria's checks is expensive to write up honestly (attribution chain, adversarial
re-run, confound naming), which is the correct incentive — it should be at least as much work to
publish a trustworthy null as to publish a trustworthy pass. The "never re-run to convert a fail into
a pass" rule is the one that most directly protects the frozen-probe resource ADR-031(b)(3)
protects from the training side: without it, a large adapter-architecture search space (exactly the
one ADR-028 warns about) creates strong pressure to keep tweaking one architecture until the probe
happens to pass, which this ADR closes off by definition, not by discipline alone.

## Implementation status

Design contract only — no code, no new tooling. This ADR formalizes criteria that ADR-023 § S6 and
ADR-024's M3/M4 outcome sections already satisfy in practice; it requires no retroactive change to
either. It provides the checklist a future milestone's write-up (M4c, M5, a future scale-control arm,
or run 3) should be checked against before being committed as a negative-result artifact, and the
vocabulary ("publishable" vs. "recorded" vs. "inconclusive" vs. "unregistered") for a reviewer to use
when a future result's status is ambiguous.
