# 053. Novelty audit — what is actually ours, and the one experiment left

**Status**: Final for this pass. **Date**: 2026-08-29.
**Purpose**: deflate this project's novelty claims to what survives a literature
check, and name the single remaining experiment worth funding.

---

## 1. The causal gate is PARTLY SCOOPED — three weeks before our first commit

**This repo has been calling the causal gate its most defensible contribution.
Half of it is prior art.**

**arXiv:2607.26773** — *"Do Latent Channels Actually Communicate? A Causal Audit
of Latent Multi-Agent LLM"*, Huixiang Zhang & Mahzabeen Emu, **submitted
2026-07-29**. Verified directly from the abstract page:

> *"We introduce a causal audit that applies **controlled message replacements
> at the boundary where the sender-produced representation enters the
> receiver**."*
>
> *"…decomposes into a −6.17-point effect retained by an **other-example
> message** and a +5.17-point effect attributable to **example-specific
> content**."*
>
> *"**Self-substitution comparisons** further show that example-specific content
> and other-agent value are distinct."*

**That is `ContentGain` (real vs mismatched) and `AgentGain` (real vs
self-generated), by those definitions.** Our `crates/latentmesh-gate/src/causal.rs`
was first committed **2026-08-17 — three weeks later.** Independent
re-derivation, not priority.

**It also independently reproduces our null**: −1.00pp overall on GSM8K with
Qwen3-4B. We already cite it for that; we had not noticed it also anticipates
the decomposition.

### What remains ours

The paper is **diagnostic measurement only** — verified explicitly: it makes no
pass/fail determination. Our delta is narrow but real:

- a **hard admission gate** (pass/fail) rather than a reported effect size,
- requiring the **worst** of **five** controls to be beaten (they use four
  settings),
- with a **formal significance test** (`sign_flip_permutation_test`),
- **feeding a runtime routing decision** rather than a paper table.

**Cite and differentiate in a paragraph. Do not footnote it.** A reviewer who
knows this paper will otherwise read our gate as re-derivation — which, for the
decomposition itself, it is.

## 2. UtilityDensity is NOT novel — it is value-of-information censoring

Gain-over-normalised-cost transmission gating is an established subfield:

- **arXiv:2204.00474** and **arXiv:2211.11038** — value-of-information censoring
  in distributed filtering: transmit only when value exceeds cost.
- **arXiv:2305.08481** — task-oriented communication at scale, VoI-weighted
  transmission.
- **arXiv:2608.00458 (BANDMAS)** — transmits a packet only when *predicted*
  contribution exceeds resource cost, under joint bandwidth/latency/deadline
  constraints. Same shape as our `CostWeights`/`CostScale`.

**The one real distinction**: BANDMAS's contribution is **predicted**
(learned/replay-estimated); ours is **measured and hypothesis-tested**. That
difference is worth keeping — but the ratio formalism itself is standard.

## 3. Cross-representation routing exists, on a different criterion

**arXiv:2605.25422** selects between token and KV-cache transmission via a joint
media-selection and resource-allocation optimiser, and reports that *neither is
uniformly optimal across operating regimes* — **structurally the same claim as
our Run 2/3 result.** But its selection is driven by **predicted end-to-end
latency**, never by verified causal gain; it has no `None` option; and it is a
2-way choice against our 8-mode lattice.

## 4. Two properties I over-credited

- **`None` can win** — functionally implied by *every* VoI-censoring paper
  above. Not new. Only the **type-level** treatment (an explicit enum variant
  scored at 0.0 outside the cost formula, with ties favouring it) is unusual,
  and that is an engineering choice, not a research contribution.
- **`Unmeasured` ≠ `Measured(0.0)`** — the most locally novel piece; no
  published router in this space types them apart. But it is the **opposite** of
  the standard answer: bandit/UCB methods treat unmeasured arms as *worth
  exploring*, with an optimism bonus. Ours is **conservative by construction** —
  never explore automatically — which is defensible for a safety-gated runtime
  but is a **design decision, not an algorithmic advance**, and must be
  described that way.

## 5. The defensible claim, stated honestly

> No published system combines **(a)** a formal, statistically-tested,
> multi-control causal **admission gate**, feeding **(b)** a resource-normalised
> utility-density **router**, across **(c)** a representation lattice including
> `None` with `Unmeasured` structurally ineligible — and uses it to **discover**
> that the latent channel is decision-inert rather than assuming it works.

**Lead with the combination and the empirical result, not with the mechanisms.**
Every individual mechanism has close, recent prior art.

---

## 6. The one experiment left: receiver-side adaptation

### What C2C's Fuser actually is (arXiv:2510.03215, fetched directly)

Per-layer operation on **KV-cache pairs at every aligned layer** — terminal
alignment, walking backward — with: a **projection + feature-fusion** module, an
input-aware **dynamic weighting** head-modulation layer, and a **learnable
per-layer gate** through **Gumbel-sigmoid with temperature annealing
(1.0 → 0.001)**, which is what decides *which layers fuse at all*. Both models
frozen; **only the Fuser trains**, on plain next-token CE, 500K OpenHermes2.5
samples, batch 256, 1,929 steps. **Fuser parameter count is not stated anywhere
in the paper** — checked specifically.

**So the real C2C recipe is multi-layer + learned gating + task loss, jointly.**
Our rungs ran these **separately**: M4i/M5X were multi-layer with
**reconstruction** loss and no gating; M4c/M4d/M4g were task loss but
**single-layer** with no gating. **The conjunction has never been run here.**

### Receiver-side adaptation is genuinely unexplored — verified

**arXiv:2606.05711**'s Table 5 shows only two training regimes across all 18
surveyed methods: training-free, or an **external trainable interface with the
underlying LLMs frozen**. **No surveyed method trains the receiver's own
weights.**

One honest caveat rather than a flat claim: §6.2 names a third regime,
end-to-end training/distillation, whose cited example (*Internalised Debate*)
**does** modify weights inside a model — but the survey places it *adjacent to*
direct communication and excludes it because multi-agent structure disappears at
inference. **Excluded on architectural grounds, not because it leaves receiver
weights untouched.** State it that way.

### Recommendation: run (b), not (a)

| | (a) C2C recipe | **(b) receiver LoRA** |
|---|---|---|
| GPU | ~1–2 h | **~1.2–2.0 h** (one rank) |
| engineering | **new gated multi-layer fusion module** | reuses `mlp.rs`/`fastgrnn.rs` |
| pre-registered | no | **yes — [research/034](034-m5-receiver-side-adaptation-scout.md)** |
| prior signal | stacks on two nulls/failures | untested axis |

**(a) stacks onto results we already have**: task loss alone was *destructive*
(0W/40L, off-manifold), multi-layer alone was a *powered null*. Whether the
conjunction differs from two stacked failures is genuinely unknown — but its
expected information per GPU-hour is worse, and it needs new infrastructure.

**(b) is the one axis both the literature and this repo have not touched**, is
already fully pre-registered in `research/034`, and reuses existing candle
infrastructure.

**The confound `034` already flags, which must be honoured**: a task-loss-adapted
receiver could simply become *a general GSM8K fine-tune* that helps regardless of
injection content. **The zero-injection condition must be re-measured on the
same adapted receiver** — never reused from a frozen-receiver rung.

**Also from `034`**: an *online* ΔV feedback loop costs **~3 GPU-h per single
feedback point** on permutation-test power grounds, which makes one epoch of
online feedback more expensive than the entire run. ΔV is computed **once,
post-training, as characterisation** — not as an online signal.
