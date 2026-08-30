# 045. M5 — receiver-side adaptation: pre-registration

- **Status**: **ACCEPTED — EXECUTING** (2026-08-29, on operator go-ahead).
  Everything below the "MANDATORY POWER CALCULATION" heading was written and
  committed (`9fe984e`, merged to main in PR #16) **before any item was drawn**
  and is **frozen**. Nothing in the registration may be edited now that
  execution has begun; outcomes go in a *separate* results section appended at
  the end, and any deviation is recorded as a numbered coordinator error.
- **Date**: 2026-08-29.
- **Promotes** [research/034](../research/034-m5-receiver-side-adaptation-scout.md)
  to a formal pre-registration, adding the power calculation
  [ADR-040](040-pc3-decision-change-endpoint-pre-registration.md) now mandates.
- **Protocol**: [ADR-036](036-successor-rung-evaluation-protocol.md) e-process ·
  **Publication**: [ADR-032](032-negative-result-publication-contract.md).

---

## Why this rung, and why it is the only one left

The activation-injection ladder is **closed** with a scoped negative
([research/052](../research/052-decision-equivalence-from-receipts.md)): direct
injection into a **frozen** receiver is decision-inert at one *and* two layers,
on a powered test. **Layer count is eliminated.**

Every rung to date trained **the payload** to suit a frozen receiver. **None
trained the receiver to accept the payload.** That asymmetry is the last
untested axis, and it is untested in the literature too:
**arXiv:2606.05711**'s Table 5 shows all 18 surveyed methods are either
training-free or use an **external trainable interface with the underlying LLMs
frozen** — *no surveyed method trains the receiver's own weights*.

*(Honest caveat: §6.2 names a third regime whose cited example,* Internalised
Debate*, does modify in-model weights — but the survey excludes it because
multi-agent structure disappears at inference, i.e. on architectural grounds,
not because it leaves receiver weights untouched.)*

## MANDATORY POWER CALCULATION (ADR-040), before any draw

Anchored on **measured** discordance for this exact stream, site and endpoint —
not assumed:

| source | n_disc |
|---|---|
| M4i (single-layer) | **66** |
| M5X (multi-layer) | **64** |
| **expected for M5** | **≈ 65** |

At λ = 0.30, threshold 20.0 (win ×1.15, loss ×0.85):

| quantity | value |
|---|---|
| **registered crossing bar** | **≥ 45 of 65 discordant wins (69.2%)** |
| max attainable wealth (all wins) | 8,818 ≫ 20 |
| unanimous-win crossing | 22 consecutive wins |
| **minimum attainable one-sided p** | **2.7 × 10⁻²⁰** |

**VERDICT: not power-blocked.** If the realised n_disc falls below **30**, the
rung is reported **uninformative** and the power model is recorded as wrong —
a finding about our estimation, not about the apparatus.

## Design

**Train**: rank-{1, 2, 4} additive LoRA on the **receiver**, at the L14
injection site. Sender and translator **frozen**. Loss: next-token CE on the
**gold-answer continuation** — *not* the sender's span, which
`research/034` names as the fix for M4c's diagnosed target mismatch.

**Draw**: ADR-036 e-process, `adaptation-512` stream, question-tail site,
N_max = 300. Conditions: `aligned`, `baseline`, `zerovec`, `random`
(norm-matched).

### The confound, and why the primary is immune to it

`research/034` flags it correctly: **a task-loss-adapted receiver could simply
become a better GSM8K solver**, improving regardless of injected content.

**Therefore `baseline` MUST be re-measured on the same adapted receiver.**
Reusing any frozen-receiver rung's baseline would attribute a general
fine-tuning gain to the channel.

**But note what this does *not* threaten.** The registered primary is
**`aligned` vs `random`**, and **both arms run on the same adapted receiver**.
A general fine-tuning effect raises both equally and **cancels**. So:

- **`aligned` vs `random`** — the primary — is **immune** to the confound.
- **`aligned` vs `baseline`** — secondary — is **not**, and must be read only
  against the freshly measured adapted baseline.

This is why the primary stays where it is rather than moving to a
baseline-relative comparison.

### ΔV is computed ONCE, post-training

`research/034` establishes that an **online** ΔV feedback loop costs **~3 GPU-h
per single feedback point** on permutation-test power grounds — one epoch of
online feedback would cost more than the entire run. **ΔV is a post-hoc
characterisation, never an online training signal.**

## Cost

**~1.2–2.0 GPU-h** for one rank, **~3.6–6.0 GPU-h** if all three are tried and
each escalates. Reuses existing candle infrastructure (`mlp.rs`/`fastgrnn.rs`
precedent); no new fusion module.

**Deliberately NOT chosen**: a C2C replication (multi-layer + Gumbel-sigmoid
gating + task loss, jointly). It is a faithful reproduction of the one published
recipe that works, but it stacks onto **two results we already have** — task
loss alone was *destructive* (0W/40L, off-manifold), multi-layer alone was a
*powered null* — and it needs a **new gated fusion module**. Worse expected
information per GPU-hour. Registered as a possible successor, not as this rung.

## Implementation path — surveyed at execution start, NOT part of the frozen registration

Recorded because it corrects a cost impression this ADR could otherwise leave.

**The differentiable receiver already exists.**
`crates/latentmesh-train/src/qwen2_c.rs` (352 lines) provides `TrainReceiver` —
a composed BF16 receiver forward built for M4c *specifically because* "the
vendored inference forward silently cuts the graph". It exposes
`forward_span_logits` and a `span_ce` loss, and five training binaries already
drive it (`train_m4c_taskloss`, `train_m4d_deploymatch`, `train_m4g_fuse`, …).
Backprop through the receiver is therefore **solved infrastructure, not new
work**. LoRA hooks into `TrainReceiver.layers: Vec<TrainLayer>`.

**But no LoRA implementation exists in this repository** — `grep -rl lora
crates/ --include=*.rs` returns nothing. This ADR's cost line said "reuses
existing candle infrastructure … no new fusion module", which is accurate
about *fusion* and about the 1.2–2.0 GPU-h figure, but it should not be read
as "no code to write". The LoRA layer, its optimiser wiring, and the M5 probe
are new. **The GPU estimate stands; an engineering estimate was never given
and should not be inferred from it.**

**The inherited hazard.** M4c/M4d/M4g trained the payload through this same
frozen-receiver task loss and produced a *reproducible NLL inversion* —
0W/40L against both controls, diagnosed as off-manifold rather than
destructive. M5 changes what is trained (receiver, not payload), so it is not
the same experiment; but if M5's aligned condition inverts the same way, the
first hypothesis is off-manifold input, **not** a channel effect, and it must
be reported that way.

## Ranked shortlist this was selected from

Everything the three SOTA lanes surfaced, ranked by expected information per
GPU-hour. **Only #1 is funded by this ADR.**

| # | candidate | cost | verdict |
|---|---|---|---|
| **1** | **receiver-side LoRA (this ADR)** | **1.2–2.0 GPU-h** | **BUILD** — only untested axis, pre-registered, no new infra |
| 2 | C2C replication (multi-layer + gating + task loss) | ~1–2 GPU-h **+ new fusion module** | HOLD — stacks on two known failures; registered as successor |
| 3 | cite-and-differentiate vs **arXiv:2607.26773** | ~0 (writing) | **DO NOW** — a reviewer who knows it reads our gate as re-derivation |
| 4 | reframe UtilityDensity as *measured* VoI censoring | ~0 (writing) | **DO NOW** — the ratio formalism is standard; only "measured, not predicted" is ours |
| 5 | 3/4/5 injection sites | ~1 GPU-h each | **REJECT** — layer count is eliminated; near-zero expected value |
| 6 | control-ordering follow-up (mismatched < random < zero) | ~1 GPU-h | **REJECT** — the ordering is noise at n=43 (error #21); nothing to chase |

Items 3 and 4 are **zero-GPU corrections to claims, not experiments**, and are
already recorded in
[research/053](../research/053-novelty-audit-and-the-remaining-experiment.md).

## Registered interpretation, both branches

**PASS** — receiver-side adaptation makes the channel carry decisions. This is
the **first positive decision-level result in the programme** and reopens
cross-model work. It would **not** retroactively validate frozen-receiver
injection; it would explain *why* that failed.

**FAIL with real power** — the last untested axis is closed. Combined with the
scoped negative, the honest statement becomes: *under this apparatus, neither
payload training nor receiver training makes direct activation injection move
decisions.* Reported without softening (ADR-032). The **likelihood-level effect
remains real** and unaffected either way.

**Underpowered** (n_disc < 30) — reported as uninformative, spun as neither.

## Mandatory co-reports

- **Likelihood arm** against `baseline`, `zerovec` **and norm-matched
  `random`**, with per-item sign tests. Accuracy alone is a **deaf** endpoint —
  proven twice.
- **Control-vs-control comparisons computed in the probe**, not left to whoever
  thinks to ask. Coordinator error #21 happened because Run 3's receipt only
  ever tested `gated_text` vs one control, letting an unsupported rank ordering
  stand as a claim.
- The full wealth trajectory's **shape**, not just its endpoint.

## Firewall

M5 is **same-model, receiver-adapted**. It tests **the apparatus, never
transfer.** Neither outcome may be cited for or against cross-model
transferability. The scope freeze in ADR-024's head applies unchanged.

## Single-owner rule

**Exactly one implementing agent**, and **rulings must be recorded on the branch
that owner can read** — coordinator errors #11 and #20 were a duplicate owner
and a wrong-branch delivery respectively.

---

# Deviations and outcomes — APPENDED AFTER EXECUTION BEGAN

Everything above this line is the frozen registration and is unedited. This
section is the append-only record ADR-045's head requires. Nothing here
changes the registered design retroactively; where the design itself is
amended, the amendment is numbered and its justification is stated.

## Coordinator error #22 — the registered training target falsifies the registered power model

**Discovered 2026-08-29, BEFORE any draw. No rank-1 draw existed when this was
found and none has been spent; the one-draw-per-rank budget is intact.**

### What was registered

Two things, in the same document:

1. **Training objective** (§Design): "Loss: next-token CE on the **gold-answer
   continuation** — *not* the sender's span". Implemented literally: the CE
   target is `"#### {gold}"`, token for token the probe's own teacher-forced
   NLL target.
2. **Power model** (§MANDATORY POWER CALCULATION): `n_disc ≈ 65`, anchored on
   **measured** discordance for this exact stream and site — M4i's 66 and
   M5X's 64 — and a registered crossing bar of ≥45 of 65.

### What it produced

Rank-1 adapter, 3,072 parameters, best of 10 epochs by holdout CE
(`receipts/run2-m5-training-receipt-cellL18toL14-r1.json`,
`receipts/run2-m5-transfer-receipt-cellL18toL14-r1.json`):

| quantity | with adapter | without |
|---|---|---|
| holdout gold-continuation CE (composed, training) | **0.6643** | 2.3172 |
| holdout gold-continuation NLL (vendored fused) | **0.6602** | 2.2353 |
| per-item NLL wins/losses | **508 W / 2 L** of 510 | — |
| **baseline GSM8K accuracy, 64 holdout items** | **5 / 64** | **31 / 64** |
| mean generated characters | **389** | 841 |

The likelihood endpoint improved by 1.58 nats and won 508 of 510 paired items.
The decision endpoint **collapsed**: 48% → 8%. Generated length halved. The
adapter learned to emit `#### N` early instead of reasoning.

### Why the two registered elements are inconsistent

M4i and M5X — the two rungs the power model is anchored on — ran a receiver
answering ~47% of items (M4i: baseline 140/300). The registered training
objective produces a receiver answering ~8%. **An 8% receiver cannot yield
n_disc ≈ 65**: `aligned` and `random` are concordantly wrong on nearly every
item, so almost no pair is discordant and the wealth process never moves.

The objective therefore **falsifies the power model the same document
registers**. No draw can satisfy both registered elements at once. That is a
defect in the registration, discovered before any item was drawn — exactly the
failure ADR-040's mandatory power calculation exists to catch — and not an
outcome of the experiment.

### Why it happened, mechanistically

The CE target was the **final answer line only**. Raising the likelihood of
`"#### N"` given the question does not require solving the problem; the
cheapest available descent direction is to stop producing chain-of-thought and
emit the answer line immediately. The objective rewards abandoning the very
reasoning the accuracy endpoint scores.

This is a target mismatch of the **same kind** as M4c's, which
[research/034](../research/034-m5-receiver-side-adaptation-scout.md) §5.2
diagnosed ("steer[ed] the receiver toward reproducing the sender's generated
token span... not the answer-format objective the probe scores") and which
ADR-045's gold-continuation target was introduced to fix. **The fix swapped one
mismatch for another.** M4c's target was too far from the endpoint; this one
was too close to a degenerate corner of it.

### AMENDMENT — training target only

**CHANGED**: the training CE target becomes **`render_gold`** — the full gold
solution with GSM8K's `<<a+b=c>>` calculator annotations stripped, already in
this repo as `examples/common/mod.rs::render_gold` and already used by PC1 —
so that **the target contains the reasoning whose absence is the pathology**.
It still ends with the `#### N` line.

**UNCHANGED, explicitly**: the probe's endpoints. The teacher-forced NLL target
stays `#### {gold}`. The accuracy endpoint stays as registered. The primary
stays `aligned` vs `random`. The bar stays ≥45 of 65. `baseline` is still
re-measured on the adapted receiver. The conditions stay the registered four.
The item stream, the statistic, λ, N_max, the site, the operator, the slot
count, the depth and the frozen sender/translator all stay as registered.

The retrained adapter receives rank 1's single draw. The v1 artifact, its
goldens and both receipts are committed unchanged as the record of this error.

## FINDING — the likelihood and decision endpoints are in TENSION

Recorded because it is a larger result than the rung was designed to produce,
and it arrives from the error above rather than from a draw.

Run 2 established that direct activation injection is **semantic at the
likelihood level and non-semantic at the decision level** — the two endpoints
disagree. The v1 rank-1 adapter demonstrates the **converse**: optimising the
likelihood endpoint *directly*, on its own target tokens, **destroys** the
decision endpoint. Likelihood 2.3172 → 0.6643 with 508W/2L, accuracy 31/64 →
5/64, in one training run.

So the two endpoints are not merely different measurements of one underlying
quantity that happen to disagree in sensitivity. **They are in tension: a
gradient on one is not a gradient on the other, and can be a gradient against
it.** Any future rung that trains against a likelihood target and reads an
accuracy endpoint must carry a decision-side diagnostic, or it is not measuring
what it reports.

This also retro-illuminates the ladder's standing NLL-vs-accuracy dissociation:
it need not indicate a weak channel measured two ways. Two endpoints that can
be driven in opposite directions by a single gradient are, to that extent,
measuring different things.

## Deviation — mandatory decision-side diagnostic in the transfer check

**Introduced by the implementing owner before the v1 training run; APPROVED by
the coordinator and now MANDATORY for every subsequent rank.**

The transfer check generates on **holdout items** (never draw items) with the
adapter installed and removed, under the draw's own greedy decoding, and
reports accuracy and mean generated length.

It is **deliberately NOT gating**. Gating the adapter on GSM8K accuracy would
select it against the very general-fine-tuning confound the registered primary
is designed to exclude. It is reported so that a null can be read correctly:
*"the receiver stopped answering"* is a different finding from *"the channel
carries nothing"*.

Without it, the v1 transfer check's 508W/2L would have waved a broken receiver
straight into an irreversible draw. This is the discipline §"Mandatory
co-reports" already calls for, applied one step earlier.

## Deviation — control-vs-control battery is COMPLETE but THIN, and says so

The probe computes **every ordered pair** among the four registered conditions
on both endpoints — twelve pairs — and stores them as receipt fields, per
§"Mandatory co-reports" and coordinator error #21.

**Disclosed consequence, recorded rather than left for a reader to notice.**
ADR-003's `mismatched` and `self_generated` controls are **not registered for
this rung**, so no comparison involving them exists; adding a fifth condition
mid-flight would be an unregistered design edit and was **not** done. And
because `zerovec ≡ baseline` **exactly** under `InjectionMode::Fuse` (`h += 0`
is an exact no-op), every pair involving `zerovec` is an operator-correctness
check rather than an independent control comparison.

**The twelve ordered pairs therefore collapse to ONE substantive
control-vs-control comparison: `random` vs `baseline`.** A thin battery that
states its own thinness is acceptable; a thin battery that reads as complete is
error #21 again. The receipt carries this text in its own
`control_vs_control_battery.scope_limit_disclosed` field.

## Deviation — the draw was withheld at v1

The implementing owner did not run the registered draw on the v1 adapter,
against the letter of the task brief. **Recorded as a justified deviation, not
an error.** ADR-045 permits one draw per rank and forbids retry; spending it on
a receiver answering 8% of items would have burned it irreversibly on a null
attributable to the objective rather than to the channel.

## Coordinator correction #23 — the FAIL branch's likelihood sentence does not hold on this rung

**Approved by the coordinator after the rank-1 draw. Recorded here; the frozen
registration, INCLUDING the FAIL-branch text itself, is not edited.**

### What the frozen text says

§"Registered interpretation, both branches", FAIL arm: *"The **likelihood-level
effect remains real** and unaffected either way."*

### What the rank-1 draw measured

On the **adapted** receiver, `aligned` vs `random` on teacher-forced NLL is
**147W/153L, one-sided p = 0.657** — dead null. And the control-vs-control pair
the battery exists to compute shows a norm-matched **random** vector beating
baseline *more* than the aligned payload does:

| pair (NLL, lower is better) | W/L | one-sided p |
|---|---|---|
| `aligned` vs `baseline` | 200/100 | 4.01 × 10⁻⁹ |
| **`random` vs `baseline`** | **205/95** | **9.83 × 10⁻¹¹** |
| `aligned` vs `random` | 147/153 | 0.657 |

So the `aligned`-vs-`baseline` result, which alone reads as a strong
likelihood-level effect, is **not content-specific**: it is a generic
perturbation effect at this site. On this rung the frozen sentence does not
hold.

### What is NOT overturned

The frozen-receiver rungs' likelihood finding stands. **M4i** reported
`aligned` vs `random` on NLL at **166W/134L, one-sided p = 0.0367** (mid-p
0.0325) — a direct content-vs-content-free comparison, and a real if modest
content-specific effect.

**CORRECTION TO THE COORDINATOR'S OWN SCOPING, recorded rather than quietly
adopted.** The ruling described this as *"replicated"* on the strength of M5X
reporting the same 166W/134L. **M5X is not an independent replication.**
[research/052](../research/052-decision-equivalence-from-receipts.md) Claim 1
already establishes that M4i and M5X are decision- *and* likelihood-equivalent
— *"the same integers, not a similar ratio"* — because both ran the same
300-item stream, the same site, the same adapters and the same task, and the
only difference (a second injection site) was inert. The frozen-receiver
evidence is therefore **one observation of 166W/134L at p = 0.0367, observed
twice through a change that made no difference** — not two independent
replications. A correction about an overstated claim must not itself overstate.

### The honest statement — a NEW finding, not a retraction

> On the **frozen** receiver the likelihood movement was **weakly
> content-specific** (166W/134L, p = 0.0367, single observation). On the
> **adapted** receiver it is **not content-specific at all** (147W/153L,
> p = 0.657, with random beating baseline more than aligned does). Receiver
> adaptation did not make the channel carry decisions — it appears to have
> **removed the content-specificity the likelihood arm previously had.**

**Do not write this as "the likelihood effect is dead."** It is not. It is: *at
this site, on this adapted receiver, the movement is generic.*

### Mandatory caveat on the comparison itself

This is a **between-rung comparison across DIFFERENT receivers** — frozen
versus LoRA-adapted, and per this ADR's own
`the_adapted_receiver_is_a_MATERIALLY_DIFFERENT_GENERATOR` caveat the adapted
receiver's output distribution has shifted substantially (252 vs 841 mean
generated chars). It is therefore **hypothesis-generating, not a controlled
test**. It is exactly the class of cross-rung comparison this rung's own
accuracy caveat warns against, and it must carry that warning wherever it is
quoted.

### Methodological result, recorded as one

The battery caught this on its **first real draw**. Without the
`random`-vs-`baseline` row, `aligned` vs `baseline` at p = 4 × 10⁻⁹ would have
been written up as a strong likelihood-level channel effect. That is coordinator
error #21's exact shape, prevented by the mechanism made mandatory to prevent
it. **The battery earning its keep is itself a result**, and the two fields
doing the work — the battery's `scope_limit_disclosed` and the accuracy block's
"NOT comparable to any prior rung" note — are carried into every subsequent
rank's receipt by the receipt builder, not by anyone remembering to add them.

## OUTCOME — the registered ladder {1, 2, 4}, complete

Three ranks registered, three trained, three drawn under the frozen design.
**None crossed.** Each draw is powered: realised `n_disc` 67–77, all far above
the registered uninformative floor of 30.

| rank | params | holdout CE | decision diag | W_final | n_disc | wins | wins needed | crossed |
|---|---|---|---|---|---|---|---|---|
| 1 | 3,072 | 0.5171 | 34/64 | 1.1994 | 77 | 42 | 52 | no |
| **2** | **6,144** | 0.5094 | 39/64 | **14.7481** | 69 | 46 | 48 | no |
| 4 | 12,288 | 0.5030 | 36/64 | 6.0923 | 67 | 42 | 46 | no |

*(unadapted receiver: 31/64 on the same holdout items. "Wins needed" is the
exact count reaching W ≥ 20 at that rung's realised `n_disc`; it is n-dependent
— see the power-accounting correction below.)*

**Capacity is not the explanation, and the trend is not monotone.** Rank 2 is
the peak; rank 4 falls back. Three points that rise then fall are noise around
a common rate, not a trend. The "capacity was too low" objection to the rank-1
null is closed: 4× the parameters does not approach the boundary better than 2×.

### Four findings replicate across all three independent adapters

1. **The primary never crosses.** Three registered draws, three FAILs.
2. **`aligned` vs `random` on NLL is null at every rank** — 147/153 (p = 0.66),
   156/144 (p = 0.26), 146/154 (p = 0.70). Three independent nulls, two
   pointing slightly the wrong way.
3. **Both `aligned` AND `random` move NLL hard against baseline at every rank**
   — aligned-vs-baseline 200/100, 202/98, 239/61; random-vs-baseline 205/95,
   187/113, 216/84. The likelihood movement is **generic at every capacity**,
   never content-specific. Correction #23 was written on rank 1 alone; it now
   has 3/3 support.
4. **Accuracy orders `baseline` > `aligned` > `random` at every rank** —
   149/137/130, 172/154/131, 160/154/137. Injection always hurts; the
   on-manifold payload always hurts less.

### The disruption reading, and why it is preferred

Finding 4 is what "aligned beats random" measures. That ordering is consistent
with the on-manifold payload merely **disrupting the receiver less** than a
norm-matched Gaussian — a magnitude effect, not a content effect. **Finding 2
is what discriminates**: content that helps decisions should also raise its own
likelihood, and across three adapters it never does.

This is the norm-matched-Gaussian floor control doing its designed job. A rung
reporting only the primary would have shown rank 2 at 46W/23L trending hard at
the boundary and called it a channel effect in waiting.

**The remaining ambiguity, stated plainly**: `random` is *off-manifold*, so
`aligned` beats it on manifold-conformity alone. Distinguishing disruption from
content needs a control that is **on-manifold but content-wrong** — ADR-003's
`mismatched`. That control is **not registered for M5** and was deliberately
not added mid-flight. Resolving this is a successor registration's job, and its
content should be **the control set, not the item budget**: more items would
only establish whether "aligned disrupts less" is real, never whether it is
content.

### A pre-registered prediction, falsified — recorded as such

Before ranks 2 and 4 were drawn, the implementing owner predicted to the
coordinator that both would **reproduce rank 1's null shape**, reasoning that
rank 1's holdout CE was flat from epoch 1 with its best checkpoint at epoch 4
of 10, so the adapter had unused budget and capacity was not the binding
constraint.

**Rank 2 falsified the shape prediction.** Wealth went 1.20 → 14.75; the
trajectory changed from oscillating around 1 to rising and ending at its own
maximum. Capacity plainly does something. The prediction's *conclusion* — that
no rank would cross — held at all three, but its *stated reasoning* was wrong
about what capacity does. Recorded because a pre-registered expectation that
fails is worth more than one that quietly succeeds.

### Power-accounting correction — a derived field that disagreed with the authority

`registered_power_accounting.crossed_the_registered_bar` was **wrong twice**.

- **v1** compared the **raw win count** against 45 — a bar ADR-045 derived *for
  n_disc = 65*. At rank 2's realised `n_disc` = 69 it reported **`true` for a
  draw the e-process FAILED**.
- **v2** compared the **rate** against 45/65 = 69.2%. It reached the right
  verdict on all three draws, but is also not exact: the required count is
  **n-dependent** (45 of 65 = 69.2%, 48 of 69 = 69.6%, 52 of 77 = 67.5%).
  Checked exhaustively over n ∈ [60, 85], the rate form disagrees with the
  wealth rule in **34** (n, wins) combinations, **in both directions** —
  including n_disc = 67, exactly where rank 4 landed.
- **v3**, current, **defers to the wealth rule**: `crossed` *is*
  `e_process.crossed`, so it cannot disagree with the authority. The exact
  required count for the realised n is reported as context only.

The lesson is general enough to record: **a derived field that can disagree
with the decision rule will eventually disagree with it.** No measurement was
changed by any correction; the receipts carry the superseded values and the
reasoning.

### Registered interpretation, discharged

The FAIL-with-real-power branch applies. Under this apparatus, **neither
payload training nor receiver training makes direct activation injection move
decisions** — tested at three capacities, each powered, each with `baseline`
re-measured on its own adapted receiver. Per correction #23, the FAIL branch's
"the likelihood-level effect remains real" does **not** hold on the adapted
receiver, with 3/3 support. The ADR-024 firewall is unchanged: M5 is
same-model, receiver-adapted, and tests the apparatus, never transfer.
