# 055. M6 — the content axis, and a null that turned out to be the control's

**Status**: Final for this rung. **Date**: 2026-08-30.
**Registration**: [ADR-047](../adr/047-m6-manifold-content-factorial.md).
**Receipt**: `crates/latentmesh-runtime/receipts/run2-m6-receipt-cellL18toL14-loraR2-contentaxis-questiontail-eprocess.json`.
**Predecessor**: [ADR-045](../adr/045-m5-receiver-side-adaptation-pre-registration.md) / [research/054](054-m5-receiver-adaptation-closed.md).

Every number below is a field path into that committed receipt. Follow the path,
read the value. Nothing here is derived from prose.

---

## 1. The verdict, first

**A powered null on the registered primary.**

| field path | value |
|---|---|
| `e_process/final_wealth` | **3.8153** |
| `e_process/max_wealth` | 5.8980 (at order 181) |
| `e_process/wealth_threshold` | 20.0 |
| `e_process/crossed` | **false** |
| `e_process/n_discordant` | **55** (34W / 21L) |
| `power_and_the_bar/wins_needed_at_the_realised_n_disc` | **40** |
| `power_and_the_bar/uninformative` | **false** |

`aligned` did not beat `mismatched` on the decision endpoint. n_disc = 55 falls
inside ADR-047 §6's registered expectation of 50–60, so the power model held:
this is a **powered null**, not a failure to measure.

## 2. What made this rung worth running

M5's rank-2 primary was 46W/23L — `aligned` beating `random` on decisions,
trending at the boundary. That looked like content transmission and was not
interpretable, because M5's own accuracy field ordered `baseline` > `aligned` >
`random` at every rank. Every injection hurts; the on-manifold payload hurts
least. **"Aligned beats random" is equally explained by `aligned` being a gentler
perturbation.**

The defect was the control. `random` is a norm-matched Gaussian: **content-free
AND off-manifold**, wrong on two axes at once, so beating it identifies neither.

M6 adds `mismatched` — the previous drawn item's genuine payload. On-manifold,
produced by the identical computation, rescaled by the identical rule to
identical L2, differing **only** in which episode's content it encodes. The term
and the control are **arXiv:2607.26773's "other-example message"**; cited, not
re-derived.

## 3. The apparatus is deterministic to the integer

This licenses everything that follows, so it is stated as a result.

| field path | M6 | M5 rank 2 |
|---|---|---|
| `summary/accuracy/aligned_real` | 154 | 154 |
| `summary/accuracy/baseline_uninjected` | 172 | 172 |
| `summary/accuracy/zerovec_injected` | 172 | 172 |
| `summary/accuracy/random` | 131 | 131 |
| `aligned_real_vs_random` accuracy | 46/23 | 46/23 |
| `aligned_real_vs_random` NLL | 156/144 | 156/144 |
| `aligned_real_vs_baseline_uninjected` NLL | 202/98 | 202/98 |
| `random_vs_baseline_uninjected` NLL | 187/113 | 187/113 |

**All 12 battery pairs the two rungs share are identical, integer for integer.**
The only difference between the draws is the added condition — so any difference
between them is attributable to the control set and to nothing else. This is not
a second sample; it is the same measurement re-run with one arm added.

`zerovec ≡ baseline` on all 300 items: 0 accuracy disagreements, 300
bit-identical NLLs, max |ΔNLL| exactly 0.0.

## 4. A registered prediction, and its falsification

Before the result was read, the coordinator registered: *if correction #23
generalises, the likelihood arm should stay null in M6 too.*

**It did not.**

| `control_vs_control_battery/pairs/…/nll_lower_is_better` | split | p |
|---|---|---|
| `aligned_real_vs_mismatched` | **181/119** | **2.05e-4** |
| `aligned_real_vs_random` | 156/144 | 0.263 |

The likelihood arm **is** sensitive to the content contrast. It needed an
on-manifold control to reveal it.

**This narrows correction #23 without retracting it.** #23 concluded from
`aligned` vs `random` alone that the adapted receiver's likelihood movement is
not content-specific. M6 reproduces that null with the same integers, and shows
on the same 300 items that the content contrast against an on-manifold control
is significant. **#23's null belongs to its control, not to the receiver.**

## 5. The caution — the likelihood picture is not clean

Stated at equal prominence, because a tidy content narrative is available here
and would be wrong.

**Every injected condition improves likelihood over `baseline`, and `random`
improves it more than `mismatched` does:**

| vs `baseline` | NLL split | p | mean NLL |
|---|---|---|---|
| `aligned` | 202/98 | 9.51e-10 | 3.8142 |
| `random` | 187/113 | 1.14e-5 | 3.8295 |
| `mismatched` | 176/124 | 1.59e-3 | 3.8901 |
| (`baseline`) | — | — | 3.9316 |

**And the pairwise contrasts do not compose**: `aligned` > `mismatched`
(p = 2.05e-4), `aligned` ≈ `random` (p = 0.263), `random` ≈ `mismatched`
(157/143, p = 0.227). A content-only account predicts `aligned` beats `random`.
It does not.

**One clarification so this is not over-read.** Paired sign tests count items,
not magnitudes, and are *not* required to be transitive — non-transitivity alone
is not evidence of a defect. The substantive anomaly is narrower and survives
that caveat: **`random` is better for likelihood than `mismatched` on the mean
as well as the count** (3.8295 vs 3.8901; 157/143). That is what a content-only
account fails to predict.

### The candidate mechanism — generated by this data, not established by it

**Off-manifold payloads may produce a larger generic likelihood shift.** Then
`random` gets a boost that offsets `aligned`'s content advantage and cancels
that contrast to null, while `mismatched`, being on-manifold, gets no such boost
and the content advantage shows. One mechanism explains #23's null and M6's
result together.

**This is a hypothesis, not a result, and must never be cited as the latter.**
It is directly testable: it predicts generic likelihood shift scales with
off-manifold-ness, and the phase-1 dose ladder already spans exactly that axis
with reproducible payloads.

## 6. The decision endpoint, stated without qualifiers

`aligned` vs `mismatched` on accuracy: **34/21, n_disc 55, p = 0.0524**. Not
"marginal", not "nearly significant" — the number, and the fact that **the
registered e-process is the only registered authority and did not cross**.

Accuracy ordering: `baseline` 172 > `aligned` 154 > `mismatched` 141 > `random`
131. Read as a decomposition, content accounts for 154-vs-141 and manifold
conformity for 141-vs-131. **This is DESCRIPTIVE, not inferential**: the
manifold contrast (`mismatched` vs `random`, 42/32) has p = 0.148 and is not
significant.

**`baseline` still leads.** Every injection continues to hurt decisions, as in
M5.

## 7. Phase 1 — the cell that was built, measured, and discarded

ADR-047 registered a 2×2 whose off-diagonal cell was `aligned_displaced`: right
content, wrong manifold. The CPU-only manipulation check built it and **the
registered gate passed 4/4**. It was withdrawn anyway, on evidence beyond the
gate.

**The gate could not have failed** (coordinator error #24, the coordinator's).
Its two arms — cosine within tolerance, and typicality strictly between
`aligned` and `random` — measure the same quantity: measured typicality equals
`c` × typicality(`aligned`) to within 0.0035 at every dose.

**The structural result, which is the real contribution of phase 1:**

> In 1536 dimensions a generic displacement direction is almost surely **both**
> off-manifold **and** content-free, so rotating toward one necessarily moves
> both factors in lockstep.

An exact rotation attenuates true content by exactly `c` — at the 0.90 dose,
signal 0.900 against noise 0.436, a 33% noise admixture by norm. ADR-047 §4's
"content is held exactly" is **false as written**. Every instrument agrees:

| | `aligned` | 0.99 | 0.95 | 0.90 | 0.75 | `random` |
|---|---|---|---|---|---|---|
| typicality | 0.6814 | 0.6741 | 0.6488 | 0.6139 | 0.5145 | −0.0006 |
| item-invariance | 0.6670 | 0.6539 | 0.6031 | 0.5390 | 0.3760 | −0.0000 |
| RMSNorm-lens entropy | 3.3199 | 3.3295 | 3.5242 | 3.6822 | 4.9180 | 5.4452 |

Monotone throughout. `aligned_displaced(c)` is a point on the
`aligned`→`random` segment — M5's diagonal, sampled at intermediate points — so
a dose-response along it cannot attribute. **That is the geometric reason the
off-diagonal cell resists construction at all**, and it is not obvious in
advance.

Calibration: `aligned`'s typicality (0.6814) slightly **exceeds** a genuine
un-pooled receiver L14 state (0.6670). The on-manifold column is sound; the
failure is specific to the off-manifold cell.

## 8. Scope — read before citing

**Established:** on this receiver (M5's rank-2 adapter), at the question-tail
site, the content contrast does not move the **decision** endpoint by the
registered rule, on a powered test. The apparatus is deterministic to the
integer across rungs. And correction #23's likelihood null was a **control
artefact**.

**NOT established:** that the likelihood effect is content-specific in general.
The off-manifold-inflation reading is unproven, and until it is tested the
likelihood picture stays ambiguous.

**NOT tested, and not to be inferred:** cross-model transfer. M6 is same-model
throughout and tests the apparatus. ADR-024's scope freeze and research/054's
scope section apply unaltered.

**A registration gap, recorded rather than smoothed over:** ADR-047 never named
which receiver M6 runs on (coordinator error #25). Rank 2 was ruled, for the
reason that §1's motivating number is rank 2's 46W/23L. A draw on a different
rank, or on the frozen receiver, would be a separate draw with a separate
comparator — reported separately, never pooled.

## 9. Two successors, neither implemented

1. **The subspace redesign.** Displace WITHIN the receiver's local L14 subspace
   versus ORTHOGONAL to it at matched magnitude. Both arms attenuate content
   equally, so the contrast is conformity alone — which is what phase 1 showed a
   generic rotation cannot deliver.
2. **The off-manifold-inflation test.** Does generic likelihood shift scale with
   off-manifold-ness? The phase-1 dose ladder is the instrument, already built.

Both need their own pre-registration with their own power calculation.
