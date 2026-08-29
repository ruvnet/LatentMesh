# Research: the science of injection configuration (norm, placement, train/deploy match, diagnostics)

> **SUPERSEDED IN PART (2026-08-29) — read this before using this document.**
> This document characterises Cache-to-Cache as injecting "continuously, at
> every generation step, in lockstep". That is **wrong about C2C**: later
> passes fetched C2C's own equation and found it **fuses ONCE at prefill**
> (see [038](038-manifold-constrained-adapter-scout.md) and
> [039](039-bidirectional-latent-exchange.md)). The continuous-coupling
> description actually fits the **Bicameral Model**, which couples
> bidirectionally at every decode step. The rest of this document — the
> norm/rescale analysis and the diagnostic ranking — stands.
> This pointer exists because the append-only rule means the original text is
> never edited, and a reader consulting this file alone would otherwise carry
> the error forward.



* Purpose: informs the interpretation of M4d (ADR-024's registered train/deploy-configuration-match
  contingency, in flight as of this writing) and the design of any successor rung, by surveying what
  the activation-steering / activation-patching / cross-model-communication literature says about
  *how* an injected activation must be configured to carry causal effect — as distinct from whether
  it fits its target well. Does not re-report ground already covered by
  [docs/research/028](028-sota-continuous-sweep-1.md) (task-loss-vs-reconstruction-loss finding) or
  [031](031-statistical-power-and-design.md) (statistical power) — reads both as prior state.
* Date: 2026-08-29.
* Scope: arXiv 2023-2026, weighted toward activation-steering/RepE/activation-patching subfields not
  previously surveyed by 023/027/028. WebSearch/WebFetch only. Evidence grades as in 028: **primary**
  (paper fetched and the specific claim read in the fetched text), **inferred** (search-result
  synthesis only, not independently confirmed against the source text), **uncertain** (could not be
  verified — fetch blocked or ambiguous). Several sources below are graded down from what a search
  snippet implied, stated explicitly where that happens.
* Trigger: M4c (ADR-024) trained a receiver-task-loss adapter that measurably improved the receiver's
  own held-out next-token loss (mean fused NLL 0.2385→0.1570, 498/509 wins on the pre-registered
  transfer check) yet, at probe time, lost to random on that same NLL quantity 0/40 and to the zero
  vector 0/40 — a systematic all-40-item reversal of exactly what training improved. ADR-024 names
  this a train/deploy configuration mismatch and registers M4d (repeat the task-loss training with
  the deployment transform — 8-slot placement + rescale-to-natural-median — inside the training loop)
  before drawing further conclusions.

---

## 1. Norm/scale of injected vectors

**The steering literature treats norm/scale as a well-documented, load-bearing sensitivity, not a
solved or ignorable knob.** "Steering Language Models With Activation Engineering" (Turner et al.,
ActAdd, arXiv:2308.10248, **inferred** — search-synthesis of paper content, not independently
re-fetched) is described consistently across secondary sources as scale-sensitive: "steering quality
relies heavily on the intervention coefficient and the effective steering vector norm... overly
strong intervention can degrade fluency or introduce unintended side effects." This is corroborated
by "Minimizing Collateral Damage in Activation Steering" (arXiv:2605.01167, **primary**, fetched):
magnitude mismatched to natural per-layer activation norms produces collateral damage and fluency
degradation, and the paper's own framing implies practitioners should calibrate steering magnitude to
layer-specific activation statistics — which is, on its face, congenial to LatentMesh's
rescale-to-natural-median instinct.

**But the field does not treat norm-matching as a free, side-effect-free operation — it treats angle
and norm as two causally distinct levers that must not be conflated.** "A Geometric Account of
Activation Steering through Angle–Norm Decomposition" (arXiv:2606.06735, **primary**, fetched) is the
most directly relevant single source: across seven models, "concepts are represented primarily in
angular structure," but "norm remains important for the stability and downstream effects of
steering" — the paper's central prescription is that "activation steering should be parameterized by
interpretable angular and radial components of the intervention, rather than by a single additive
coefficient that entangles these two effects," because "interventions with similar concept-level
effects can behave differently" depending on how the two interact. Critically, the same source notes
that **post-hoc renormalization to a fixed norm is not a null operation — it is a different, named
method**: "if [a] matched CAA output is additionally renormalized to the original norm, the resulting
method is exactly equivalent to Spherical Steering" — i.e., adding a raw vector then rescaling its
norm is not "the same intervention, just calibrated better"; it is a categorically different
intervention (spherical rather than additive/affine) whose properties were derived for a
*direction-only* optimization target, not for a vector whose direction was fit under an unconstrained
(and, per M4c, ~2-3x larger) norm regime.

**How CAA and RepE actually handle norm, precisely — and why it differs from LatentMesh's pipeline
in a way that matters.** Contrastive Activation Addition (Rimsky et al., arXiv:2312.06681 / ACL
2024, **primary** for the norm-handling claim via the paper + the `nrimsky/CAA` reference
implementation, cross-checked against search synthesis): CAA normalizes the *extracted contrastive
vector's own norm* across behaviors, per layer, purely so that a single coefficient α is comparable
across different behaviors — but it deliberately does **not** normalize across layers, "to preserve
a 'natural norm' given the sampled activations," since residual-stream norm grows roughly
exponentially with depth. The key structural fact: **CAA fixes the vector's norm once, at
extraction time, before any downstream use — it never trains a vector under one norm regime and then
applies a different rescaling transform only at deployment.** Every steering-vector method surveyed
in this pass (CAA, RepE/LAT, ActAdd) computes its vector via a closed-form contrastive-pairs
difference-of-means — there is no gradient-trained network whose output norm could drift from a
calibration target the way M4c's MLP output (`aligned_l2_raw` 90–200 across the 40 probe items,
receipt-measured) drifted from the `natural_inject_block_norms` median (~40–56 per item, same
receipt) it gets rescaled to. **This is the single most important negative finding of this section:
no source found trains a raw activation-space vector end-to-end via gradient descent and *then*
applies a post-hoc renormalization to a statistic (a natural-distribution median) the training loss
never saw.** The combination LatentMesh's deploy path uses — gradient-optimized direction, frozen at
training's own (larger) norm, then forcibly rescaled at inference — is not a documented, previously-
validated configuration in any source this pass found; it is closer to an untested compound of two
separately-validated pieces (task-loss training per §2 of `docs/research/028`; norm-calibration to
natural activation statistics per this section) than to any single method whose interaction has been
characterized.

**One more concrete, receipt-derived observation worth naming: the "natural" reference distribution
itself is almost certainly dominated by the massive-activations/attention-sink phenomenon**, not a
tight, representative sample of ordinary content-token norms. `natural_inject_block_norms` in the
M4c receipt (`run2-m4c-receipt-cellL18toL14-mlp-taskloss-slots8-poolfull-rescaletrue-n40.json`) shows
median ≈ 40–56, p25/p75 within a few units of the median (i.e., most tokens sit in a narrow band),
but **max ≈ 11,000–11,057 in every single item** — a ~200× outlier, present in *every* item, which is
the textbook signature of "massive activations" (Sun et al. 2024, cross-checked via multiple 2026
follow-ups: "an activation is considered massive if its magnitude exceeds 100 and is at least 1,000×
greater than the median... concentrated in special tokens such as start and delimiter tokens,"
manifesting as "attention sinks... value-state drains... residual-state peaks," **primary** via
search synthesis of `arXiv:2605.08504`/`2606.20743`/`2505.21670`). This does not undermine the choice
of *median* over *mean* as the rescale target (median is the right robust statistic given this
skew, and mean would be dragged toward the sink-token outlier) — but it is worth flagging that the
"natural norm" being calibrated to is a statistic over a distribution with a well-characterized
architectural pathology at one tail, not an arbitrary empirical curiosity, and is a separate,
independently-known phenomenon from the injection question itself, not a candidate explanation for
the null.

---

## 2. Placement — site, position, and continuity of injection

**Single fixed-layer injection is the common baseline in the steering literature, but the strongest
positive cross-model transfer results (C2C, the Bicameral Model — both already primary-verified in
`docs/research/028` §2) deliberately reject it in favor of either learned multi-layer gating or
continuous per-step coupling — and this pass surfaces a placement dimension not previously named in
this repo's research chain.** CAA and RepE/LAT both inject at one empirically-chosen mid-to-late
layer, added to the residual stream at **every** token position after the prompt (not a small fixed
set of dedicated slots) — "applying the steering vector to all token positions is substantially more
effective than applying it only to the last prompt token" (**inferred**, search synthesis across
CAA/RepE ablations). "Where to Steer: Input-Dependent Layer Selection for Steering Improves LLM
Alignment" (arXiv:2604.03867, **inferred**) reports that a single fixed layer, chosen once and
applied uniformly regardless of input, underperforms an input-conditioned layer choice — mild
evidence that L14-fixed is a defensible baseline choice but a known-suboptimal one relative to what
the field's better methods do.

**Cache-to-Cache's own ablation is the most directly load-bearing citation here (re-confirmed and
extended from `docs/research/028` §2's fuser-architecture note): "enriching only selected layers is
better than enriching all layers, which later motivates a gating mechanism"** (**primary**, C2C
paper/repo, arXiv:2510.03215) — i.e., neither "one fixed layer" nor "every layer, uniformly" is
C2C's answer; a **learned, per-layer, per-position gate** deciding where to inject, trained jointly
with the projector, is what the strongest existing cross-model method actually does. LatentMesh's
single fixed L14 site is architecturally the simpler of the two options C2C's own ablation shows to
be worse than its learned-gate design — a real, previously-unstated placement gap, independent of the
norm question in §1.

**The more consequential placement fact this pass surfaces: whether injection is one-shot-upfront or
continuous-per-step.** Both C2C and the Bicameral Model inject **at every generation step, live,
during actual decoding** — the Bicameral Model's two models "run in lockstep... at every generation
step" (`docs/research/028` §3, re-cited here for this specific angle). LatentMesh's pipeline, by
contrast — 8 fixed placeholder-token slots, one pooled vector, injected once via
`forward_capture_multi`-style prefill before the receiver's own greedy generation proceeds unassisted
for up to 400 tokens — is a **single static injection consumed once at the start of generation**, not
a continuously-refreshed signal. No source in this pass or in `docs/research/028`/`027` documents a
cross-model method that injects once and then lets the receiver free-generate for hundreds of tokens
with no further signal; every positive result found (C2C, Bicameral) refreshes the injected content
at every step. **This is registered here as a placement candidate ADR-024's M4d contingency does not
currently name** — M4d's own text lists "8-slot placement and the greedy-decode context" as secondary
candidates alongside rescale, but frames placement narrowly as slot-position/count, not as
one-shot-vs-continuous delivery. Fixing the injection schedule (continuous per-step) is a materially
larger engineering change than fixing norm or slot count, and is not part of M4d as currently
registered — named here as a follow-up hypothesis for a future rung, not a retroactive change to
M4d's scope.

**Dedicated placeholder tokens vs. overwriting real token activations — the specific mechanism
LatentMesh uses (`<|fim_pad|>`, token id 151662) is a real but under-studied design choice with
named tradeoffs.** "Compositional Steering of Large Language Models with Steering Tokens"
(arXiv:2601.05062, **primary**, fetched) is the one source found that directly compares dedicated
placeholder-token injection against standard per-position activation addition: dedicated tokens
"isolate steering signals from genuine model computations," reducing interference with the model's
normal processing and supporting cleaner composition of multiple steering objectives — but "require
additional input space and may create artificial distributional shifts if the model hasn't
encountered such placeholders during training," which "could introduce brittleness when generalizing
to new contexts." `<|fim_pad|>` exists in Qwen2.5's vocabulary (a fill-in-middle padding token from
pretraining) but was almost certainly never paired with injected mid-layer activations during
pretraining — the exact "artificial distributional shift" this paper names as the placeholder
approach's known risk, independent of what vector is injected there.

---

## 3. Train/deploy configuration mismatch — is this a named phenomenon?

**Yes, unambiguously, under a different name than any prior LatentMesh document has used: exposure
bias / covariate shift, the general family M4d's own registered hypothesis is a specific instance
of.** "Exposure bias is a mismatch between training and inference conditions... during training, the
model learns to condition on ground-truth context... in contrast, during inference the model is
given [its own prior outputs]" (**primary**, cross-checked across the Emergent Mind synthesis,
arXiv:2606.12990 "Exposure Bias as Epistemic Underidentification in Recursive Forecasting", and
arXiv:1910.00292 "Generalization in Generation: A closer look at Exposure Bias"). The causal
mechanism named in the literature — "if the model makes an error, these errors are compounded into
subsequent steps... the next prediction will be made using an unusual input (one unlike those in the
training set)" — is precisely structurally analogous to M4d's own hypothesis, generalized one level:
**M4c's adapter was optimized against raw (unrescaled) vectors it would output; deployment feeds the
receiver a rescaled vector the training loss never produced or saw. The receiver's own forward pass
at injection time is therefore conditioning on an out-of-training-distribution input**, the same
shape of mismatch exposure bias names for token-level autoregressive generation, one representational
layer up (activation-space input distribution, not token-sequence distribution). Mitigations named in
this literature — scheduled sampling, propensity correction, "aligning training conditions with
inference scenarios" — converge on the same fix ADR-024 has already registered for M4d: **put the
deployment-time transform inside the training loop**, which is exactly scheduled sampling's own
structural move (expose the model to its actual deployment-time input distribution during training,
rather than only the clean one).

**A second, independently-arrived-at name for the identical structural problem, from outside NLP
entirely, worth citing because it generalizes M4d's framing beyond sequence generation specifically:
covariate shift / "training-serving skew" in the imitation-learning and applied-ML-systems
literature.** DAgger (Ross & Bagnell) is the canonical fix for exactly this shape of problem in
imitation learning: a policy trained on states visited under one distribution (an expert's) fails
when deployed, because its own errors — or, in M4d's case, the deployment-time rescale transform —
put it in states/inputs never visited during training; DAgger's fix is identical in structure to
scheduled sampling and to M4d's registered fix: **incorporate the deployment-time state distribution
into the training loop.** This is named here as convergent evidence, not a new citation graded
higher than primary-search-synthesis — the DAgger connection was not independently re-fetched this
pass, but it is standard enough that it is stated with confidence as background, not as a novel
finding requiring its own verification.

**Is the exact M4c/M4d dissociation — a signal that demonstrably improves a target loss during
training yet is causally inert or actively harmful at intervention time — independently documented
elsewhere in the activation-steering literature specifically (not just the generic exposure-bias
literature)?** The closest primary-verified match found this pass: "Decodable but Not Corrected by
Fixed Residual-Stream Linear Steering: Evidence from Medical LLM Failure Regimes"
(arXiv:2605.05715, **primary**, fetched) — its core finding is stated as "a signal can be cleanly
decoded via linear classifiers while remaining causally inert... even when residual stream
activations strongly encode which failure mode will occur, directly steering along that direction
fails to prevent the error," with the paper's own framing citing scale/magnitude mismatch and
"training-deployment divergence" (the "well-fit" direction identified in one context not
transferring when "applied uniformly across different contexts") as candidate mechanisms. This is
not the same experimental setup as M4c (single-model failure-mode steering, not cross-model transfer)
but it is the most directly on-point precedent found for the specific pattern "decodable/well-fit +
causally null or reversed," independent of the generic amnesic-probing/representational-vs-functional
framing `docs/research/028` §1 already established. **No source found documents the exact combination
M4c/M4d isolates — a vector trained to raw scale then deployed after forced renormalization — as a
previously-tested and named failure mode in its own right; the closest matches (this section, §1) are
each one level more general than LatentMesh's specific compound configuration.**

---

## 4. Diagnostics — telling "ignored" from "actively harmful" from "right direction, wrong content"

**The single most actionable finding of this section: a cheap, single-forward-pass diagnostic exists
that can be run today, on the existing M4c artifact, without any new training or live-model probe
draw.** "Predicting Where Steering Vectors Succeed" (arXiv:2604.15557, **primary**, fetched)
proposes **LAP (Linear Accessibility Profile)**: project a candidate steering/injection direction
through the model's own unembedding matrix (a logit-lens-style operation) and measure `A_lin` —
"whether a concept aligns with the model's own output projection." Their reported thresholds, across
24 concept families and five models (ρ = 0.86–0.91 correlation with actual measured steerability):
**peak `A_lin` > 0.1 → steering viable; peak `A_lin` < 0.05 → negligible effect regardless of
intervention strength.** This is directly computable against LatentMesh's own artifacts — the M4c
MLP's raw output vector, the rescaled/deployed vector, and (once it exists) M4d's trained output —
each is a `[1536]`-dim vector in the receiver's residual-stream basis at block 14; multiplying by the
receiver's unembedding matrix (`lm_head`, already loaded for every probe run) and checking `A_lin`
for both the raw and rescaled versions would test, in an afternoon and with zero GPU-hours beyond a
single forward pass' worth of matrix multiplication, **whether the rescale operation itself is what
destroys output-alignment** — the leading M4d hypothesis — before committing to a full M4d retrain-
and-reprobe cycle. If `A_lin` is high pre-rescale and collapses post-rescale, that is strong,
cheap, pre-registered-safe (it touches no probe items, no controls, no statistics ADR-028's protected
list covers) evidence for the norm-mismatch hypothesis specifically, isolable from the
placement/greedy-decode candidates named alongside it.

**A second, complementary diagnostic from the same paper: "Probe Gap" (Δ), which separates "the
information isn't there" from "it's there but not output-aligned."** Comparing a trained nonlinear
probe's accuracy (`A_mlp`) against the linear/unembedding-aligned accuracy (`A_lin`) yields a
three-regime framework: low `A_mlp` → concept genuinely absent, no method will work; high `A_mlp` but
low `A_lin` → information present but nonlinearly encoded, output-alignment-based steering will fail
regardless of magnitude; high `A_lin` → steering via simple addition should succeed. Applied to
LatentMesh's own diagnostics, this maps onto a real gap in what the M4c/M4d receipts currently report:
the training receipt shows the adapter fits its regression target well (M3's own holdout-MSE
framing) and the transfer receipt shows it improves the receiver's next-token loss on the sender's
own span — both are closer to `A_mlp`-style "is information present/usable for *some* task" checks —
but nothing in the current receipt schema checks `A_lin`-style output-alignment for the *probe's own
target* (the `#### <gold>` continuation) specifically, as opposed to the sender's generated span
(what task loss actually optimized). This is a genuinely new, cheap, protocol-safe addition worth
recommending: **compute the injected vector's unembedding-projected overlap with the gold-answer
token(s) specifically**, separately from its overlap with the sender's own generated tokens — this
would directly quantify the "target mismatch" mechanism M4c's own reading already proposes narratively
("the adapter learned to steer the receiver toward reproducing the sender's generated token
span... not the answer-format objective the probe scores") with a number computable from existing
artifacts, no new probe draw required.

**A third diagnostic, more specific to distinguishing "actively harmful" from "ignored": Perturbation
Sensitivity (λ)**, also from arXiv:2604.15557 — quantifies local representational instability around
the injection point; the paper reports high λ correlates (ρ = −0.73 to −0.99) with unpredictable
steering effects. A direction with low `A_lin` (steering "should" be negligible per the first
diagnostic) that nonetheless produces the M4c pattern (0/40 NLL wins, a *systematic* reversal rather
than noise) would be inconsistent with "ignored" and would point toward "actively harmful via a
different, unmeasured mechanism" — worth checking as a tie-breaker if `A_lin` alone doesn't cleanly
explain the observed 0/40 sweep.

**What LatentMesh already does that the literature treats as standard, load-bearing practice, not
optional extras:** the zero-vector and norm-matched-random controls already present in every M3/M4/
M4c receipt are precisely the "zero ablation" and "resample/random ablation" baselines the
activation-patching methodology literature treats as the minimum comparison set (**primary**,
`lesswrong.com/.../how-to-use-and-interpret-activation-patching`, cross-checked against
arXiv:2404.15255 — "zero ablation overwrites... to observe which components break behavior... mean
ablation overwrites with the dataset mean... resampling substitutes another input's variation...
these different baselines represent different causal assumptions about what should be 'removed'").
This is worth stating plainly since it's easy to read the M4c 0/40-vs-zerovec loss as merely "bad
luck" — the ablation-methodology literature is explicit that losing to *both* a null baseline and a
matched-random baseline, simultaneously and unanimously, is the strongest possible signature of "the
intervention is actively counterproductive," not merely "ignored" (an ignored intervention should
score statistically indistinguishably from these controls, not lose to both of them 0/40).

---

## 5. Ranked recommendations for a successor rung, if M4d also nulls

Ordered by expected value per unit of cost, cheapest/most-diagnostic first:

1. **(Near-zero cost, run today, no new training or probe draw)** Compute LAP's `A_lin` for the M4c
   adapter's raw output vs. its rescaled/deployed vector, against both the sender's-span tokens and
   the gold-answer tokens specifically. One matmul against the receiver's unembedding matrix per
   item. This directly tests the leading M4d hypothesis (does rescale destroy output-alignment) and
   the "target mismatch" reading (is the vector aligned with the sender's tokens but not the gold
   answer's) simultaneously, without touching the frozen probe, its controls, or its statistics —
   protocol-safe under ADR-028's own protected-list discipline, same category as the already-approved
   A6 permutation-null analysis in ADR-024/`docs/research/029`.
2. **(M4d itself, already registered and in flight)** — the DAgger/scheduled-sampling-style fix:
   train with the deployment transform inside the loop. This is the single highest-leverage
   structural fix the literature converges on for exposure-bias/covariate-shift-shaped mismatches
   (§3), and is already running; this document's contribution is confirming it is the theoretically
   correct next move, not merely an empirically-motivated guess.
3. **(Cheap — inference-time only, reuses M4c/M4d weights, no retraining)** Separate angle from norm
   explicitly, per the angle-norm decomposition paper's (§1) prescription: instead of one fixed
   post-hoc rescale-to-median, sweep a small grid of scale multipliers (e.g. {0.5×, 1×, 2×, 4×} of
   the natural median) applied to the *already-trained* direction, and re-run only the cheap
   diagnostics from §4 (not a fresh 40-item probe draw) across the grid to see whether some other
   scale — not necessarily "no rescale" or "rescale to median" — recovers alignment. This is
   explicitly informational/pre-probe, not a retry against the frozen protocol.
4. **(Moderate engineering cost — new injection architecture)** Move from one-shot-upfront
   8-placeholder-slot injection to continuous per-generation-step injection, closer to what both C2C
   and the Bicameral Model actually do (§2) — the one placement dimension this pass identifies as
   untested and un-named in ADR-024's own M4d registration. This targets a genuinely different
   candidate mechanism than norm or task-loss and would need its own pre-registration addendum before
   any probe draw, per ADR-024/028's standing discipline.
5. **(Highest engineering cost, most architecturally novel)** Multi-layer, learned-gate injection
   (C2C-style) rather than single fixed-site L14 injection. Ranked last because C2C's own ablation
   (§2, re-confirmed from `docs/research/028`) shows this adds a few points *on top of* an
   already-working task-loss single-mechanism baseline — it is not, per the field's own evidence, the
   primary lever, and should only be attempted after the cheaper diagnostics/fixes above are
   exhausted.
6. **(If 1-5 all null)** Report a joint negative result explicitly scoped as: task loss (M4c) +
   train/deploy configuration match (M4d) + norm/angle decomposition (item 3) all fail to produce
   probe-measured causal transfer at this receiver scale — this meaningfully narrows the remaining
   hypothesis space toward the already-mandatory receiver-scale confound (M4b) and/or a genuine
   representational non-transfer conclusion, rather than leaving "still the loss function" or "still
   the deployment transform" as live, unresolved alternatives.

---

## Sources

- ActAdd (Turner et al.): https://arxiv.org/abs/2308.10248 (inferred, search-synthesis only)
- Contrastive Activation Addition (CAA): https://arxiv.org/abs/2312.06681,
  https://aclanthology.org/2024.acl-long.828/, https://github.com/nrimsky/CAA (primary for
  norm-handling claim)
- Representation Engineering (RepE): https://arxiv.org/abs/2310.01405
- "A Geometric Account of Activation Steering through Angle–Norm Decomposition":
  https://arxiv.org/abs/2606.06735 (primary, fetched)
- "Minimizing Collateral Damage in Activation Steering": https://arxiv.org/abs/2605.01167 (primary,
  fetched)
- "Decodable but Not Corrected by Fixed Residual-Stream Linear Steering: Evidence from Medical LLM
  Failure Regimes": https://arxiv.org/abs/2605.05715 (primary, fetched)
- "Predicting Where Steering Vectors Succeed" (LAP): https://arxiv.org/abs/2604.15557 (primary,
  fetched)
- "Compositional Steering of Large Language Models with Steering Tokens":
  https://arxiv.org/abs/2601.05062 (primary, fetched)
- "Where to Steer: Input-Dependent Layer Selection for Steering Improves LLM Alignment":
  https://arxiv.org/abs/2604.03867 (inferred)
- "Activation Scaling for Steering and Interpreting Language Models": https://arxiv.org/abs/2410.04962
  (fetch attempted, content not resolvable at readable-text depth this pass — uncertain, named as a
  lead only)
- Activation patching / ablation-baseline methodology: https://arxiv.org/abs/2404.15255,
  https://www.lesswrong.com/posts/FhryNAFknqKAdDcYy/how-to-use-and-interpret-activation-patching
  (primary via search synthesis)
- Exposure bias: https://arxiv.org/abs/1910.00292, https://arxiv.org/abs/2606.12990 (primary via
  search synthesis)
- Massive activations / attention sinks: https://arxiv.org/abs/2605.08504,
  https://arxiv.org/abs/2606.20743, https://arxiv.org/abs/2505.21670 (primary via search synthesis)
- Cache-to-Cache, the Bicameral Model — already primary-verified in `docs/research/028` §2, re-cited
  here for the placement-specific reading, not re-fetched this pass
- "Understanding (Un)reliability of Steering Vectors" (OpenReview id JZiKuvIK1t) — fetch blocked
  again this pass (verification wall), still uncertain, carried forward from `docs/research/028`
  as an unresolved lead
- Internal: `docs/adr/024-run2-trained-thought-adapter-ladder.md` §"M4c outcome" and §"Registered
  contingency — M4d", `docs/research/028-sota-continuous-sweep-1.md` §1-2,
  `crates/latentmesh-runtime/receipts/run2-m4c-*.json`
