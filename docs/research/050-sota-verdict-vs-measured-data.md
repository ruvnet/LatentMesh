# 050. The SOTA verdict meets our receipts

**Status**: Working note. **Date**: 2026-08-29.
**Confronts**: an external SOTA review of the latent-communication frontier
against [048](048-run2-final-synthesis.md)'s measured results.

---

## The review's "Key risk" is not hypothetical. We ran it.

> *"The correct sender state may fail to outperform a mismatched or receiver
> generated state at matched compute. In that case, LatentMesh remains useful
> as an efficient state cache and orchestrator, but the distributed cognition
> claim fails."*

**Measured, out-of-sample, fully powered:**

| | required by the review's acceptance test | measured (PC3, 212 fresh items) |
|---|---|---|
| correct vs random, decisions | **+10 absolute pp**, 95% CI above 0 | **36W/32L, p = 0.72** |
| | | min attainable p = 3.4e-21 — a null *with power* |

And in PC1b (300 items) the correct sender state is **behind** the random
control: aligned **127** vs random **133**, against a 140 baseline.

**So the risk materialized.** On our apparatus, at single-layer injection, the
distributed-cognition claim does not hold and the review's own fallback — an
efficient state cache and orchestrator — is the evidenced position.

## ⚠️ CORRECTION — half of it is prior art (added 2026-08-29, see [053](053-novelty-audit-and-the-remaining-experiment.md))

This section endorsed the review's framing that causal validation is *"probably
LatentMesh's most defensible research contribution."* **A literature check
deflates that.** arXiv:2607.26773 (submitted **2026-07-29**, three weeks before
our `causal.rs` first commit) already performs *"controlled message replacements
at the boundary where the sender-produced representation enters the receiver"*
and decomposes the effect into example-specific content versus other-agent
value — **that is ContentGain and AgentGain.** Verified directly from the
abstract.

What remains ours is narrower: a **hard admission gate** with **five** controls,
a formal significance test, and a **runtime routing consumer** — where that
paper is **diagnostic measurement only**, confirmed explicitly. Real, but a
paragraph of differentiation, not a headline.

## The proposed contribution already exists in this repo

The review names causal validation as *"probably LatentMesh's most defensible
research contribution."* **It is implemented and has already run.**
`crates/latentmesh-gate/src/causal.rs` carries all five controls —
`zero`, `random`, `mismatched`, `self_generated`, `text_equivalent` — with
admission requiring the **worst** control to be beaten, not the mean. Run 3
stage A used it and **passed**, crossing at item 43 of 300.

`ContentGain` and `AgentGain` are therefore not new instruments to build; they
are re-namings of comparisons this repo already computes. What is genuinely
missing is `UtilityDensity` — the per-resource normalisation.

## Two of the review's citations independently reproduce our result

- **arXiv:2607.14103** — 28x compression, **99.4% probe accuracy** vs 80.4% for
  text, and yet **no task-level win**. That is precisely our
  likelihood-live/decision-inert dissociation, found by a different group with
  different models.
- **arXiv:2608.04893** — correct and mismatched caches statistically equivalent
  within 2.8 points; zeroing costs 14.7 points while the *wrong example's* cache
  costs 0.4. Same shape as ours.

The review's own warning — *"high latent fidelity does not prove high task
value"* — is the sentence our mission spent 90 commits establishing with
receipts.

## Where our result does NOT bind

**Every rung we ran injected at a single layer.** The review's strongest
citations (XKV, OBF, C2C) are multi-layer, and C2C's own Table 10 ablation
collapses to ~0.1pp when reduced to one layer — our exact signature. So the
acceptance test is **not refuted for a multi-layer implementation**; it is
refuted for the configuration we tested. That is why M5X (ADR-037) was
unblocked.

**Firewall, unchanged**: our results are about the **apparatus**. A FAIL may not
be cited as evidence against cross-model transfer any more than a PASS may be
cited for it.

## What this changes about the plan

1. **Phase 1 "causal audit harness" (1–2 weeks) is largely already done** — the
   gate exists, has run, and produced a PASS on the text channel. Remaining work
   is `UtilityDensity` and wiring the gate as a *runtime* per-message check
   rather than offline topology fitness.
2. **The acceptance test needs a likelihood arm.** As written it targets
   accuracy alone — the endpoint we proved is deaf (p=0.72 while likelihood
   moved −0.773 nats at p=7.5e-35). A null on accuracy without the likelihood
   co-report is uninterpretable.
3. **The two-plane split is confirmed by measurement, not just argument**: the
   reasoning envelope's 282-byte floor is 79% identity digests, and at SF11 a
   signed delta costs ~an hour of duty-cycle budget on header alone.
