# 054. M5 — receiver-side adaptation, closed

**Purpose**: record the outcome of the last untested axis of the
activation-injection ladder. Every number is a field in a committed receipt
under `crates/latentmesh-runtime/receipts/`; nothing here is inferred from
prose.

**Date**: 2026-08-29 · **Registration**:
[ADR-045](../adr/045-m5-receiver-side-adaptation-pre-registration.md) ·
**Protocol**: [ADR-036](../adr/036-successor-rung-evaluation-protocol.md)
e-process · **Publication**:
[ADR-032](../adr/032-negative-result-publication-contract.md), no softening.

---

## 1. The result

Three ranks registered, three trained, three drawn under the frozen design.
**None crossed.** Each draw is powered: realised `n_disc` 67–77, far above
ADR-045's uninformative floor of 30.

| rank | params | `final_wealth` | `n_disc` | wins | wins needed at THIS `n_disc` | crossed |
|---|---|---|---|---|---|---|
| 1 | 3,072 | 1.1994 | 77 | 42 | 52 | **no** |
| **2** | **6,144** | **14.7481** | 69 | 46 | 48 | **no** |
| 4 | 12,288 | 6.0923 | 67 | 42 | 46 | **no** |

Receipts: `run2-m5-receipt-cellL18toL14-loraR{1,2,4}-pertokenlast-fuse-questiontail-slots8-eprocess.json`.

**The bar is n-dependent, and the wealth rule is the authority.** ADR-045
registered "≥ 45 of 65 discordant wins (69.2%)". That is exact **at
`n_disc` = 65 and nowhere else**: the count solving
`(1+λ/2)^k (1−λ/2)^(n−k) ≥ 1/α` at λ = 0.30 is 45 of 65 (69.2%), 48 of 69
(69.6%), 52 of 77 (67.5%). Any fixed count *or* fixed rate derived from it
disagrees with the wealth rule at some `n`. The receipts' `crossed` field now
*is* `e_process.crossed`, so it cannot disagree with the decision rule; see §7.

## 2. Capacity is not the explanation, and the trend is not monotone

`final_wealth` runs **1.1994 → 14.7481 → 6.0923** and wins run **42 → 46 → 42**.
**Rank 2 is the peak; rank 4 falls back.**

After rank 2 the implementing owner offered the coordinator a reading — a
monotone capacity trend that `N_max` = 300 could not resolve — and **withdrew
it when rank 4 landed**. Three points that rise then fall are noise around a
common rate, not a trend. Recorded because the retraction is part of the
evidence: the reading was offered on two points and should not have been.

The "capacity was too low" objection to the rank-1 null is closed. **4× the
parameters does not approach the boundary better than 2×.** Holdout CE does
improve monotonically with capacity (0.5171 → 0.5094 → 0.5030), so the adapters
are learning more; that learning does not convert into channel signal.

## 3. Four findings that replicate across all three independent adapters

**1. The primary never crosses.** Three registered draws, three FAILs, all
powered.

**2. `aligned` vs `random` on NLL is null at every rank.**

| rank | W/L | one-sided p |
|---|---|---|
| 1 | 147/153 | 0.657 |
| 2 | 156/144 | 0.263 |
| 4 | 146/154 | 0.698 |

Three independent nulls; two point slightly the wrong way.

**3. Both `aligned` AND `random` move NLL hard against `baseline` at every
rank — the movement is generic, never content-specific.**

| rank | `aligned` vs `baseline` | `random` vs `baseline` |
|---|---|---|
| 1 | 200/100 (p = 4.01e-9) | 205/95 (p = 9.83e-11) |
| 2 | 202/98 (p = 9.51e-10) | 187/113 (p = 1.14e-5) |
| 4 | 239/61 (p = 2.34e-26) | 216/84 (p = 7.34e-15) |

**4. Accuracy orders `baseline` > `aligned` > `random` at every rank**:
149/137/130, 172/154/131, 160/154/137. Injection always hurts; the on-manifold
payload always hurts less.

Operator correctness held throughout: `zerovec` ≡ `baseline` **exactly** in all
three draws (0W/0L on both endpoints), 0 degenerate captures.

## 4. Disruption, not content — and why that reading is preferred

Finding 4 is what "aligned beats random" measures. Rank 2's primary was
46W/23L, its wealth rose to 14.75 and ended at its own maximum. Read alone,
that looks like a channel effect in waiting.

It is not, or at least the data does not say it is. The ordering
`baseline` > `aligned` > `random` is exactly what you expect if the
on-manifold payload merely **disrupts the receiver less** than a norm-matched
Gaussian — a **magnitude** effect, not a **content** effect.

**Finding 2 is what discriminates.** Content that helps decisions should also
raise its own likelihood. Across three independent adapters it never does:
147/153, 156/144, 146/154. The disruption reading is therefore not merely
*available* — it is **better supported** than the content reading.

**The remaining ambiguity, stated plainly.** `random` is **off-manifold**, so
`aligned` beats it on manifold-conformity alone. Separating disruption from
content needs a control that is **on-manifold but content-wrong** —
[ADR-003](../adr/003-causal-edge-verification.md)'s `mismatched`, another
episode's genuine payload. A magnitude effect predicts no advantage against it;
a content effect predicts one. See §8.

## 5. Correction #23 — the likelihood arm, now with 3/3 support

ADR-045's FAIL branch asserts *"the likelihood-level effect remains real and
unaffected either way."* **On the adapted receiver it does not hold**, and
finding 2 now supports that at all three ranks rather than the single rank the
correction was written on.

**This is a NEW FINDING, not a retraction.** The frozen-receiver result stands:
**M4i** reported `aligned` vs `random` on NLL at **166W/134L, one-sided
p = 0.0367** — a direct content-vs-content-free comparison, modest but real.

**M5X is not an independent replication of it.**
[research/052](052-decision-equivalence-from-receipts.md) Claim 1 establishes
that M4i and M5X are decision- *and* likelihood-equivalent — *"the same
integers, not a similar ratio"* — same stream, site, adapters and task, with
the only difference (a second injection site) inert. The frozen-receiver
evidence is **one observation of 166W/134L at p = 0.0367, seen twice through a
change that made no difference.**

> **The honest statement.** On the **frozen** receiver the likelihood movement
> was **weakly content-specific**. On the **adapted** receiver it is **not
> content-specific at all**. Receiver adaptation did not make the channel carry
> decisions; it appears to have **removed the content-specificity the
> likelihood arm previously had.**

**This must not be read as "the likelihood effect is dead."** It is: *at this
site, on this adapted receiver, the movement is generic.*

**Mandatory caveat.** This is a **cross-receiver** comparison — frozen versus
LoRA-adapted, and the adapted receiver's output distribution has shifted
substantially (mean generated length 252 vs 841 chars). It is therefore
**hypothesis-generating, not a controlled test**, and is exactly the class of
cross-rung comparison this rung's own `comparability_discipline` warns against.
It carries that warning wherever it is quoted.

## 6. Error #22 — the two endpoints are in tension

ADR-045 registered the training target as the **gold-answer continuation**
(`"#### {gold}"`). Trained on it, the rank-1 adapter produced:

| | with adapter | without |
|---|---|---|
| holdout gold-continuation CE | **0.6643** | 2.3172 |
| fused NLL, per-item W/L | **508 W / 2 L** of 510 | — |
| **baseline GSM8K accuracy** | **5 / 64** | **31 / 64** |

**Optimising the likelihood endpoint directly destroyed the decision
endpoint.** Run 2 established that injection is semantic at the likelihood
level and non-semantic at the decision level — the endpoints *disagree*. This
is the converse: **a gradient on one endpoint is a gradient against the other.**
They are not two views of one quantity.

The mechanism: the CE target was the final answer line only, so the cheapest
descent direction is to stop producing chain-of-thought and emit it
immediately — a target mismatch of the **same kind** as M4c's, which
[research/034](034-m5-receiver-side-adaptation-scout.md) §5.2 diagnosed and
which this target was introduced to fix.

It also **falsified the power model ADR-045 registers in the same document**:
`n_disc` ≈ 65 is anchored on M4i (66) and M5X (64), rungs whose receiver
answers ~47% of items; an 8% receiver cannot produce it. Two registered
elements were mutually inconsistent. Found **before any draw**; the amended
target is `render_gold` (the full solution, `<<…>>` stripped), training target
only, every probe endpoint frozen.

**Any future rung that trains against a likelihood target and reads an accuracy
endpoint must carry a decision-side diagnostic, or it is not measuring what it
reports.**

## 7. The confound is real, and the primary is what neutralises it

The adapters **do** make the receiver a better GSM8K solver: baseline accuracy
on holdout items goes **31/64 unadapted → 34 / 39 / 36** for ranks 1 / 2 / 4.
(Those three are within noise of each other on 64 paired items; the honest
reading is "at or slightly above the unadapted rate", not "rank 2 is best".)

This is precisely the general-fine-tuning confound ADR-045 anticipated. It is
neutralised **by construction**: `aligned` and `random` both run on the *same*
adapted receiver, so a general gain raises both and cancels.
`aligned` vs `baseline` is **not** immune and is read only against the freshly
measured adapted baseline — never against a frozen-receiver rung's.

**A derived field that can disagree with the decision rule will eventually
disagree with it.** The receipts' bar field was wrong twice: v1 compared the raw
win count against 45 and reported `true` for rank 2, a draw the e-process
FAILED; v2 compared the rate against 69.2% and reached the right verdict on all
three draws but disagrees with the wealth rule in **34** (n, wins) combinations
over n ∈ [60, 85], **in both directions** — including `n_disc` = 67, exactly
where rank 4 landed. v3 defers to `e_process.crossed`. No measurement was
changed by any correction.

## 8. Methodological result — the battery earned its keep on its first draw

ADR-045 made a **control-vs-control battery** mandatory after coordinator error
#21, in which a receipt tested one condition against one control and let an
unsupported rank ordering stand as a claim.

On rank 1, `aligned` vs `baseline` on NLL is **200W/100L at p = 4.01e-9**. Read
alone, that is a strong likelihood-level channel effect. The battery's
`random` vs `baseline` row is **205W/95L at p = 9.83e-11** — the norm-matched
Gaussian does it *more*. **Without that row, rank 1 would have been written up
as a channel effect.** That is error #21's exact shape, caught by the mechanism
made mandatory to prevent it, on its first real draw.

Two fields do the work and are emitted by the **receipt builder**, not added by
hand: the battery's `scope_limit_disclosed` and the accuracy block's "NOT
comparable to any prior rung" note. Mechanical inheritance is the difference
between a discipline and a habit someone has to remember.

**The battery is complete over the registered set, and thin.** ADR-045
registers four conditions; ADR-003's `mismatched` and `self_generated` are not
among them and were **not** added mid-flight. Under `InjectionMode::Fuse`,
`zerovec` ≡ `baseline` exactly, so every pair involving `zerovec` is an
operator-correctness check. **The twelve ordered pairs collapse to ONE
substantive control-vs-control comparison: `random` vs `baseline`.** A thin
battery that says it is thin is fine; one that reads as complete is #21 again.

## 9. What a successor should test — and what it should not

**Not more items.** Pooled across the three draws the primary is 130/213 =
61.0%, still under the bar. That number is **arithmetic, not a verdict**:
pooling across ranks is not a registered analysis and the three adapters are
different models. It is recorded and explicitly not offered as evidence.

Raising `N_max` would resolve only whether *"aligned disrupts less than random"*
is statistically real. It would **not** discriminate disruption from content,
because both current controls are wrong in the same way: `random` is
off-manifold, so `aligned` beats it on manifold-conformity alone.

**The discriminating control is `mismatched`** — on-manifold, content-wrong. A
magnitude effect predicts no advantage against it; a content effect predicts
one. Run 3 has already shown the control discriminates on this receiver family:
gated text beat `mismatched` **27W/1L, one-sided p = 1.08e-7**
(`run3-stageA-receipt-statictext-gate-eprocess.json`,
`summary.per_control[control="mismatched"]`).

A successor registration's content should therefore be **the control set, not
the item budget**. Deciding whether to open one is the operator's call.

## 10. Scope

Under **this apparatus**, neither payload training nor receiver training makes
direct activation injection move decisions — tested at three capacities, each
powered, each with `baseline` re-measured on its own adapted receiver.

**This closes receiver-side adaptation under this apparatus and nothing more.**
It says nothing about learned integration generally (C2C-style trained fusion
remains untested here), nothing about cross-model transfer — ADR-024's firewall
is unchanged, M5 is same-model and receiver-adapted and tests the apparatus,
never transfer — and it does not overturn the frozen-receiver likelihood result
of §5.

Reported without softening, per ADR-032.

## Sources

All receipts under `crates/latentmesh-runtime/receipts/`:
`run2-m5-receipt-cellL18toL14-loraR{1,2,4}-…-eprocess.json`,
`run2-m5-{training,transfer}-receipt-cellL18toL14-r{1,2,4}.json`,
`run2-m5-*-r1-v1-superseded-goldanswerline.*` (the error-#22 record),
`run2-m4i-receipt-…-eprocess.json`,
`run2-m5x-receipt-…-eprocess.json`,
`run3-stageA-receipt-statictext-gate-eprocess.json`.
Internal: [ADR-045](../adr/045-m5-receiver-side-adaptation-pre-registration.md)
(registration, errors #22/#23, outcome),
[ADR-036](../adr/036-successor-rung-evaluation-protocol.md),
[ADR-003](../adr/003-causal-edge-verification.md),
[ADR-032](../adr/032-negative-result-publication-contract.md),
[research/034](034-m5-receiver-side-adaptation-scout.md),
[research/052](052-decision-equivalence-from-receipts.md).
