# 048. Run 2 final synthesis — a likelihood/decision dissociation in activation injection

**Status**: Final. Written under
[ADR-032](../adr/032-negative-result-publication-contract.md)'s negative-result
contract — reported without softening.
**Date**: 2026-08-29.
**Supersedes**: [041](041-run2-synthesis-skeleton.md), [046](046-run2-synthesis-v2.md).

---

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
