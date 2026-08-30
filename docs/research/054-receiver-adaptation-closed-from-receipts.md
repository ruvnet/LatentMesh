# 054. Receiver-side adaptation, closed — verifiable from receipts alone

**Companion to [052](052-decision-equivalence-from-receipts.md)**, which did this
for the **frozen** receiver. This one covers the **adapted** receiver. Same
discipline: every number below is a **field path into a committed receipt**.
Open the JSON, follow the path, read the value. Nothing here is derived from
prose, and no sentence in this repository needs to be true for the conclusion
to follow.

**Date**: 2026-08-30. Receipts under `crates/latentmesh-runtime/receipts/`,
named `run2-m5-receipt-cellL18toL14-lora{R1,R2,R4}-pertokenlast-fuse-questiontail-slots8-eprocess.json`.

Full registration, deviations and rulings: [ADR-045](../adr/045-m5-receiver-side-adaptation-pre-registration.md).

---

## Claim 1 — three registered draws, three failures

| field path | R1 | R2 | R4 |
|---|---|---|---|
| LoRA parameters | 3,072 | 6,144 | 12,288 |
| `e_process/final_wealth` | **1.1994** | **14.7481** | **6.0923** |
| `e_process/n_discordant` | 77 | 69 | 67 |
| `e_process/discordant_wins_aligned` | 42 | 46 | 42 |
| `e_process/wealth_threshold` | 20.0 | 20.0 | 20.0 |
| `e_process/crossed` | **false** | **false** | **false** |

**The bar is n-dependent — recompute it, do not read a fixed number.** At
λ = 0.30 and threshold 20.0, the wins required are
`ceil((ln20 − n·ln0.85) / (ln1.15 − ln0.85))`:

| realised n_disc | wins needed | wins observed | short by |
|---|---|---|---|
| 77 (R1) | 52 | 42 | 10 |
| 69 (R2) | 48 | 46 | **2** |
| 67 (R4) | 46 | 42 | 4 |

**None underpowered.** ADR-045 pre-committed `n_disc < 30` as *uninformative*;
the realised values are 67–77, and R1's 77 **exceeds** the registered
expectation of ≈65. This is the FAIL-with-real-power branch throughout.

## Claim 2 — capacity is NOT monotone, so there is no trend to extrapolate

Read `final_wealth` across the ranks: **1.20 → 14.75 → 6.09**. Rank 2 is the
peak; quadrupling the parameters from rank 2 to rank 4 moved wealth *down*.
Wins are 42 / 46 / 42.

Three points that rise then fall are **noise around a common rate**, not a
capacity trend. A two-point reading taken after R2 alone suggested a monotone
climb that N_max could not resolve; R4 retired it. Recorded in ADR-045
§"A pre-registered prediction, falsified".

## Claim 3 — four findings replicate across all three independent adapters

Each row is three separate adapters, three separate trainings, three separate
draws.

**(a) `aligned` vs `random` on likelihood is null every time.**
Path: `control_vs_control_battery/pairs/aligned_real_vs_random/nll_lower_is_better`

| | R1 | R2 | R4 |
|---|---|---|---|
| wins/losses | 147/153 | 156/144 | 146/154 |
| `p_one_sided` | 0.657 | 0.263 | 0.698 |

Two of the three point slightly the **wrong** way.

**(b) Likelihood movement against baseline is GENERIC, not content-specific.**
Paths: `.../aligned_real_vs_baseline_uninjected/nll_lower_is_better` and
`.../random_vs_baseline_uninjected/nll_lower_is_better`

| | R1 | R2 | R4 |
|---|---|---|---|
| aligned vs baseline | 200/100 | 202/98 | 239/61 |
| **random vs baseline** | **205/95** | 187/113 | 216/84 |

At R1 a **norm-matched random vector moves likelihood MORE than the aligned
payload** (p = 9.83e-11 against 4.01e-9). Read alone, that aligned-vs-baseline
p-value is a strong channel effect. It is not one.

**(c) Accuracy orders `baseline` > `aligned` > `random` at every rank.**
Path: `summary/accuracy`

| | baseline | aligned | random |
|---|---|---|---|
| R1 | 149 | 137 | 130 |
| R2 | 172 | 154 | 131 |
| R4 | 160 | 154 | 137 |

**Every injection hurts. The on-manifold payload hurts less.**

**(d) Operator correctness held.** `zerovec ≡ baseline` **exactly** — 0W/0L on
both endpoints in all three draws, `n_degenerate_capture` 0 throughout. Under
`fuse`, the zerovec condition is `h += 0`, so identity is the expected result
and its absence would have indicated a broken injection path.

## Claim 4 — what "aligned beats random" actually measures

R2's primary is 46W/23L on accuracy, the strongest of the three. Claim 3(c)
shows why that is **not** evidence of content:

> If every injection degrades the receiver and the on-manifold payload degrades
> it *least*, then "aligned beats random" is exactly what a **disruption-magnitude**
> difference predicts. No content need be transmitted for it to appear.

Claim 3(a) is what makes disruption the better-supported reading rather than
merely an available one: **content that helps decisions should also raise its
own likelihood, and across three adapters it never does.**

**The ambiguity is not resolvable from these receipts, and this document does
not resolve it.** `random` is a norm-matched Gaussian — *off*-manifold — so
`aligned` wins on manifold-conformity alone. Separating disruption from content
requires a control that is **on-manifold but content-wrong**: ADR-003's
`mismatched`. That control is **not registered for M5**, so no comparison
involving it exists here. **A larger N_max would not help** — it would only
sharpen whether the disruption difference is real.

## Claim 5 — the adapter works; the channel still does not

Path: `summary/generation_diagnostic` in the transfer receipts.

| | unadapted | R1 | R2 | R4 |
|---|---|---|---|---|
| GSM8K accuracy /64 | 31 | 34 | **39** | 36 |

Receiver-side adaptation **does** make the receiver a better solver — this is
precisely the general-fine-tune confound `research/034` predicted. It moves no
channel signal at any rank. The registered primary (`aligned` vs `random`, both
arms on the *same* adapted receiver) is what neutralises the confound: a general
competence gain raises both arms and cancels.

## Claim 6 — the endpoints are in tension (coordinator error #22)

Before the amendment, the adapter was trained on the probe's own likelihood
target (`"#### {gold}"`) — token for token, the thing the NLL endpoint scores.
Superseded receipts retained in-tree as `*-v1-superseded-goldanswerline.*`:

| | v1 (likelihood target) |
|---|---|
| holdout CE | 2.3172 → **0.6643** |
| transfer wins | **508W / 2L** of 510 |
| **GSM8K accuracy** | **31/64 → 5/64** |

**Optimising the likelihood endpoint destroyed the decision endpoint.** Run 2
established that injection is semantic at the likelihood level and non-semantic
at the decision level; this is the converse. A gradient on one endpoint is not
a gradient on the other.

The fix was the training target only (`render_gold`, the full solution). Both
probe endpoints stayed frozen.

## Two methodological results, recorded as results

**The control-vs-control battery earned its keep on its first real draw.**
Without `random_vs_baseline`, R1's `aligned_vs_baseline` at p = 4.01e-9 would
have been written up as a channel effect. That is coordinator error #21's exact
shape, caught by the mechanism made mandatory to prevent it.

**A derived field that can disagree with the decision rule will eventually
disagree with it.** `crossed_the_registered_bar` first compared a raw win count
against a bar derived for a *different* n_disc, reporting `true` for a failed
draw. Its replacement — a fixed-rate comparison — disagrees with the wealth rule
in 34 (n, wins) combinations over n ∈ [60, 85], **including n = 67, exactly
where R4 landed**: at 46 wins the wealth rule crosses (20.4126 ≥ 20) while the
rate form says it does not. The field is now `e_process.crossed` itself, not a
recomputation, so it cannot disagree with the authority.

---

## ⛔ Scope — read before citing this

**CLOSED, precisely and only:** receiver-side LoRA adaptation, at ranks 1/2/4,
at the L14 injection site, same-model, does not make **direct activation
injection** move **decisions** — on three powered draws with the baseline
re-measured on each adapted receiver.

**NOT closed, and must not be cited as such:**

- **Latent communication in general.** Three adapters, one site, one apparatus.
- **Learned-integration methods.** C2C's gated multi-layer Fuser trains a
  *fusion module*; nothing here tests that.
- **Cross-model transfer.** Every condition was same-model.
- **Whether the R2 movement is content.** Claim 4 — needs `mismatched`.
- **The frozen-receiver likelihood result.** M4i and M5X both store
  `nll_aligned_vs_random` **166W/134L** — *identical integers*, same stream,
  same site, same adapters, the only difference inert. That is **one
  observation seen twice** at one-sided p = 0.0367, not a replication, and it
  is not overturned by anything here. What Claim 3(b) adds is that on the
  **adapted** receiver the movement is not content-specific — a cross-receiver
  comparison, therefore **hypothesis-generating, not a controlled test**.

Combined with [052](052-decision-equivalence-from-receipts.md): under this
apparatus, **neither payload training nor receiver training makes direct
activation injection move decisions.** Text, on the same receiver family,
crossed its boundary at item 43 and beat another episode's genuine message
27W/1L at one-sided p = 1.08e-7.
