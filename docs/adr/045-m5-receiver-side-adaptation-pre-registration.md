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
