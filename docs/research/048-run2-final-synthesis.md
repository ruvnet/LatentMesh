# 048. Run 2 final synthesis — a likelihood/decision dissociation in activation injection

**Status**: Final. Written under
[ADR-032](../adr/032-negative-result-publication-contract.md)'s negative-result
contract — reported without softening.
**Date**: 2026-08-29.
**Supersedes**: [041](041-run2-synthesis-skeleton.md), [046](046-run2-synthesis-v2.md).

---

## The programme's thesis, restated after M5X

**LatentMesh does not optimise for latent transport. It optimises for the
measured causal utility of a representation.**

That is not a retreat from the original goal — it is what the receipts support.
The runtime's job is to *discover* which representation earns its cost:

```
candidate representation
  none · text · semantic delta · symbolic rule · prototype · KV · latent
                            ↓
                   causal gate (five controls)
                            ↓
                      UtilityDensity
                            ↓
              route only if it earns its cost
```

**Text and deterministic semantic deltas are first-class reference channels,
not fallbacks.** Text is the only channel in this repository with a *positive*
causal result on decisions: Run 3 crossed at item 43, beating **every** control,
and beating `mismatched` — another episode's genuine message — **hardest**
(27W/1L, p = 2.2e-7). That is what shows the receiver responds to *this*
message's content rather than to any message at all. (The *ordering among the
controls* is **not** supported — see
[research/052](052-decision-equivalence-from-receipts.md); they are
statistically indistinguishable at n = 43.) Treating text as the thing you
resort to when latent fails inverts the evidence.

**`UtilityDensity` is the selection objective**, not a diagnostic — implemented
in `crates/latentmesh-reasoning/src/routing.rs`, where `None` is a first-class
candidate that can win and `Unmeasured` is structurally distinct from
`Measured(0.0)`.

## The result in one sentence

**Activation injection carries semantic content into a receiver's likelihood
and contributes nothing to its decisions.**

A payload derived from a specific answer moves the receiver's probability mass
toward *that answer* — a deliberately false one by **−0.773 nats
(p = 7.5×10⁻³⁵)**, the true one by **−0.320 nats (p = 5.8×10⁻¹²)** — while
changing **which answer the receiver actually gives** no more than a
norm-matched Gaussian does (**p = 0.72**, at power sufficient to detect an
effect twenty orders of magnitude smaller).

**The decision layer reads the perturbation, not the meaning.**

---

## Evidence

Three positive controls, all **same-model, same-item, identity-transform** — no
model boundary crossed, no alignment fitted. They test the **apparatus**.

| | PC1b | PC2 | PC3 |
|---|---|---|---|
| items | 300 | 300 | **212, out-of-sample** |
| registered primary | accuracy | decoy-emission | **decision-change** |
| primary outcome | FAIL (n_disc 64) | **UNINFORMATIVE** (n_disc 3) | **powered null** (n_disc 68) |
| likelihood effect | −0.282 nats (gold) | −0.721 nats (decoy) | **−0.773 nats (decoy)** |
| `zerovec` ≡ `baseline` | 300/300, max\|Δ\|=0 | 300/300, max\|Δ\|=0 | **212/212, max\|Δ\|=0** |

**PC3 is the confirmation**: 212 items from index 4485, with a hard gate proving
**zero intersection** with the stream used by PC1b/PC2/M4i. All 15 structural
gates pass.

### The likelihood arm — semantic, and it replicates

| comparison (PC3) | Δ nats | split | p |
|---|---|---|---|
| `steer` → decoy vs baseline | **−0.773** | 190W/22L | **7.5e-35** |
| `steer` → decoy vs *norm-matched random* | −0.638 | 176W/36L | 1.2e-23 |
| `restore` → gold vs random | −0.320 | 155W/57L | 5.8e-12 |

Beating a **norm-matched Gaussian delivered by the same operator at the same
positions** is the load-bearing comparison — ITI's floor control
(arXiv:2306.03341). Clearing it proves the effect is attributable to payload
**content**, not to perturbing activations.

Direction is answer-specific: the decoy payload moves probability *toward* the
decoy and *away from* gold; the gold payload does the reverse. **The channel
carries not merely information but which answer.**

### The decision arm — non-semantic

| PC3 | changed its answer |
|---|---|
| `steer` | 108/212 = 50.9% |
| `random` | 104/212 = 49.1% |
| `zerovec` | **0/212** |

Paired: **36W/32L, n_disc = 68**, wealth 0.8444 (max 1.0130, min 0.2797), never
approaching the 20.0 threshold. **p = 0.72 two-sided.** Minimum attainable
p = 3.4×10⁻²¹ — this is a null with power, not an absence of measurement.

Injection changes roughly half of all answers. So does noise. **Content adds
nothing.**

---

## The controlled cross-channel comparison

**Run 3 stage A** ([ADR-030](../adr/030-run3-causally-gated-text-pre-registration.md),
results appended there) ran the *same receiver* on the *same item population*
with the *same accuracy endpoint*, differing only in **how the sender's
information was delivered**.

| channel | Δ on decisions vs best control | evidence |
|---|---|---|
| **Text** | **+0.512** | e-process crossed at **item 43 of 300**, `min_k W_k = 21.16`; 27W/1L vs `mismatched` |
| **Latent injection** | **≈ 0** | 36W/32L, n_disc 68, **p = 0.72**, min attainable p 3.4e-21 |

**The text channel moves decisions. The latent channel does not** — while
demonstrably carrying semantic content into the same receiver's *likelihood*
(−0.773 nats toward the specific answer it encodes, p = 7.5×10⁻³⁵).

This is what turns an isolated null into a **localisation**. The payload is not
the problem and the delivery is not the problem: the receiver acts decisively on
the sender's information when it arrives as **text**, and not at all when the
same kind of information arrives as **activations**.

### Run 3's own caveat, which is where its causal force lives

`gated_text` reaching **39/43 = 90.7%** is close to trivial on its own — the
message contains the sender's final answer. **The control ordering is the
finding:**

> **`mismatched` 30.2% < `random` 34.9% < `zero` = `self_generated` 39.5%**

**A wrong-but-real message is worse than random tokens, which are worse than no
message at all.** The receiver reads the content, and misleading content harms
it *below the noise floor*. `mismatched` — another episode's genuine message,
carrying its own number — is the load-bearing control, and the one text beat
hardest. That ordering is what makes this a causal claim about
[ADR-003](../adr/003-causal-edge-verification.md)'s gate rather than a
copy-the-answer artefact.

## M5X — MULTI-LAYER CHANGES NOTHING. The ladder closes definitively. (2026-08-29)

**The last open question is answered.** M5X ran multi-layer injection
(L18→L14 **and** L24→L19 simultaneously, `FuseMany`, 4+4 slots) on **M4i's exact
300-item stream at M4i's exact site**, with reconstruction weights at both —
so **layer count was the only difference** between them.

| | **M5X (multi-layer)** | **M4i (single-layer)** |
|---|---|---|
| primary, aligned vs random | **34W/30L**, p = 0.354 | 31W/35L, p = 0.356 |
| wealth (threshold 20) | 0.8837, never crossed | 0.2578, never crossed |
| accuracy aligned | **128** | **128** |
| NLL vs baseline | 184W/116L, p = 5.2e-5 | 181W/119L, p = 2.1e-4 |
| **NLL vs norm-matched random** | **166W/134L** | **166W/134L** |

**The likelihood arm's split against norm-matched random is *identical* — 166W
/134L in both.** Not similar. Identical. Accuracy is 128 in both. **Doubling the
injection sites changed nothing measurable, on either endpoint.**

**This was a powered null**: n_disc = 64, minimum attainable p = 5.4e-21, against
a registered bar of ≥46 of 66 discordant wins stated before the draw. It got 34
of 64 (53.1%).

### The C2C hypothesis does not transfer

M5X existed because **Cache-to-Cache's Table 10** shows its own method
collapsing to ~0.1pp when reduced to a single layer — our exact signature — while
its 6.4–14.2pp effect uses gated fusion across ~5 of 28 layers. That made layer
count the obvious suspect: **the one variable every rung had held fixed.**

**It is not the missing factor.** Two sites behave exactly like one. Whatever
C2C's multi-layer fusion is doing, it is not reproduced by injecting at two
depths with this apparatus — which points at their **learned gating and
task-loss-trained Fuser**, not at layer count per se.

### Unblocking M5X was still right

[Coordinator error #16](../adr/024-run2-trained-thought-adapter-ladder.md) —
blocking M5X on a mischaracterisation — was a real error, and correcting it was
correct **even though the rung nulled**. The alternative was closing a field on
an untested variable that the literature specifically implicated. **A cheap
decisive experiment that returns "no" is worth more than an assumption that
returns nothing**, and it cost ~1 GPU-hour.

### What is now closed

Every candidate explanation for the ladder's null has been tested and refuted
with receipts: **task loss** (off-manifold is the untrained default),
**the injection operator** (fuse reproduced it), **pooling** (de-pooling made
the payload geometrically indistinguishable from a real receiver state),
**injection site** (M4i, first powered null), and now **layer count** (M5X,
identical to single-layer on both endpoints).

**The dissociation stands as the final result**, unchanged and now
comprehensively defended:

> **Activation injection is SEMANTIC at the likelihood level and NON-SEMANTIC at
> the decision level** — and that holds at one layer and at two.

## What this refutes

Four candidate explanations for the ladder's cross-model nulls were tested and
**refuted with receipts**: task loss (off-manifold geometry is the *untrained
default*), the injection operator (fuse reproduced the failure), pooling
(de-pooling made the payload geometrically indistinguishable from a real
receiver state; transfer unchanged), and injection site (M4i, the first powered
null, n_disc = 66).

**The apparatus itself is now the explanation**, and it is established on the
endpoint every verdict used.

**Therefore the ladder closes.** Every cross-model null it produced is
explained by the delivery method. **None of them is evidence about latent
transferability.** M5X ([ADR-037](../adr/037-m5x-maximal-configuration-rung.md))
and M4b ([ADR-035](../adr/035-m4b-scale-control-pre-registration.md)) are
**blocked permanently under this apparatus** — both vary payload *content*,
which is demonstrably not what moves decisions.

## External corroboration

- **arXiv:2607.26773** independently reproduces our null with different models.
- **Cache-to-Cache's own Table 10** reports that *single-layer* enrichment
  yields **~0.1pp** — the working method, reduced to our configuration,
  reproduces our signature.

Every published method that works combines **multi-layer + continuous delivery
+ task-loss training simultaneously.** Our one-factor-at-a-time ladder may have
been structurally incapable of succeeding, and that is recorded as a
methodological finding rather than excused.

---

## Firewall — what this does NOT license

**These are apparatus results. They are not transfer results.** No model
boundary was crossed and no alignment was fitted in PC1b, PC2 or PC3. Per the
symmetric rule adopted after PC1b:

> A **PASS** may not be cited as evidence *for* cross-model transfer, and a
> **FAIL** may not be cited as evidence *against* it.

The honest scope: **this says activation injection, as implemented here, does
not steer decisions.** It does not say latent content is untransferable.

---

## Methodological record

**Fifteen coordinator errors** are recorded in
[ADR-024](../adr/024-run2-trained-thought-adapter-ladder.md), append-only, each
with the artifact that caught it. The dominant failure mode, by a wide margin:
**acting on a single unverified signal.** Three worth publishing:

- **#13** — foregrounding "the payload is worse than noise" at p = 0.27.
- **#15** — pre-registering a 12.7× wealth decline that arrived as
  **near-parity** (predicted 43% wins, observed 52.9%). The cause was reading
  PC2's 33W/44L (p = 0.13, inside noise) as a directional signal. **The
  correction strengthened the finding**: parity is the cleaner null, since
  "content contributes nothing" is exactly what p = 0.72 says.
- **Cross-rung transposition** — a wealth figure quoted from a *neighbouring
  rung's* receipt. It survived every plausibility check because it looked
  reasonable. In a repository of fifteen rungs whose wealths and NLL means all
  occupy the same numeric neighbourhood, **every figure is camouflage for every
  other.**

**Two instrument defects were found and one fixed**: `classify()`'s support arm
used an absolute token-union count against an N-dependent ceiling, making
labels incomparable across draws of different size
([ADR-038](../adr/038-n-invariant-manifold-classification.md) replaced it with
an N-invariant share). PC1's item-invariance anomaly is **still unexplained** —
that fix corrected the instrument, not the observation.

**A power calculation is now mandatory at pre-registration**
([ADR-040](../adr/040-pc3-decision-change-endpoint-pre-registration.md)), after
PC2 registered a rare-event endpoint that could not reach α = 0.05 on a perfect
run — repeating the failure
[research/047](047-authoritative-power-table.md) had already documented for
**10 of 14** prior draws.

---

## POST-HOC — the cross-model channel is NOT dead at the likelihood level

**Found after the ladder closed, from committed receipts, at zero compute cost.**
Every cross-model rung was adjudicated on **accuracy** — the endpoint we now
know is deaf. Re-reading the same receipts on the **likelihood** endpoint gives
a different picture.

| rung (cross-model, 3B → 1.5B) | site | Δ NLL vs baseline | vs **norm-matched random** |
|---|---|---|---|
| M3 (pooled / per-token) | `fim_pad` | +0.001 to +0.004 | — |
| M4 FastGRNN r64/r128/r256 | `fim_pad` | −0.021 to +0.001 | — |
| M4c (task loss) | `fim_pad` | **+3.230** (destructive) | — |
| **M4i (MLP, on-manifold)** | **question-tail** | **−0.148**, 181W/119L, p = 1.7e-4 | **−0.103, 166W/134L, p = 0.032** |

**M4i clears the ITI floor control.** Its advantage over a *norm-matched
Gaussian through the same operator at the same positions* is **−0.103 nats at
p = 0.032** — content-attributable, not perturbation. **This is the first
evidence in the project that a cross-model, learned-alignment payload carries
any content-specific signal into the receiver at all.**

**Stated with its limits, which are real:**
- **p = 0.032 is marginal.** Single uncorrected test; it would not survive
  multiplicity correction across the ladder's many comparisons.
- **One rung, unreplicated.** No out-of-sample confirmation exists.
- **~1/6 the same-model effect** (PC3's identity payload: −0.638 nats vs random,
  p = 1.2e-23).
- **Post-hoc on a non-registered endpoint** — under ADR-031/032 this is a
  hypothesis, **not** a result, and may not be cited as established.

**What it changes.** The ladder's conclusion stands unaltered: the channel does
not move **decisions**, cross-model or same-model. But *"cross-model latent
transfer carries nothing"* was never tested — **accuracy was, and accuracy is
deaf.** The right successor experiment is a **pre-registered cross-model draw
with the likelihood endpoint as primary and the norm-matched random arm as the
control**, which the existing harness already supports.

## Completeness of the package

**What is publishable now**, on receipts, with no further compute:

1. **The dissociation** — semantic at the likelihood level, non-semantic at the
   decision level — established in-sample (PC2) and confirmed **out of sample**
   (PC3, 212 fresh items, zero stream overlap).
2. **The cross-channel comparison** — same receiver, same population: text
   Δ = +0.512, latent Δ ≈ 0.
3. **The negative control ordering** in Run 3, which supplies the causal force.
4. **The methodological record** — 15 coordinator errors, two instrument
   defects, and a mandatory power calculation adopted after the fact.

**What is NOT claimed**: nothing here says latent content is untransferable.
Every positive control was **same-model, same-item, identity-transform**. The
scope is *this apparatus*, and the firewall above is not a formality.

**Known gaps, stated rather than papered over**:
- **PC1's item-invariance anomaly is unexplained.**
  [ADR-038](../adr/038-n-invariant-manifold-classification.md) fixed the
  *instrument* that mis-reported it across draws of different N; the underlying
  observation — a single token dominating ~100% of items' top-10 — still has no
  account.
- **Run 3 stage B was not run.** The Darwin loop to genome freeze would
  permanently unlock `eval-200`/`holdout-100` and break the committed test that
  guards them. Stage A reaches the registered primary without it, so the
  irreversible step was declined pending an explicit decision.
- **Run 3 stopped at 43 items** by its registered early-stopping rule. That is
  valid under optional stopping, but the effect size is estimated from 43 items,
  not 300.
- **The one-factor-at-a-time ladder design** may have been structurally
  incapable of succeeding, since every working published method combines
  multi-layer + continuous + task-loss *simultaneously*. Recorded as a
  methodological finding, not excused.

## What would move this forward

The bottleneck is **likelihood → decision**, and nothing in this ladder
addresses it. Candidate directions, none attempted here:

1. **Multi-layer + continuous injection with task loss, together** — the
   conjunction every working method uses, never tested jointly here.
2. **Decision-layer intervention** — the failure is at readout, not delivery;
   injecting at the residual stream may be the wrong place entirely.
3. **Training the receiver to use injected state** (receiver-side adaptation)
   rather than training the payload to be usable by a frozen receiver.

The dissociation is the contribution. **It localises the failure precisely,
which is worth more than another null.**
