# Scout: M4f, the manifold-constrained adapter — from one paragraph to a pre-registerable spec

* Purpose: turn ADR-024's one-paragraph M4f registration ("constrain the adapter's output to the
  receiver's residual-stream manifold") into an executable, pre-registerable spec, at the standard
  the [M5 scout](034-m5-receiver-side-adaptation-scout.md) set (that scout found M5-as-written was
  infeasible and corrected it before any GPU time was spent; this document does the same due
  diligence for M4f).
* Date: 2026-08-30.
* Method / evidence label: **desk research + source-verified architecture facts + one already-run
  committed receipt's analysis** — read-only against this repo (no code written, no run executed).
  Literature claims are graded **primary** (fetched and the specific claim read in the fetched
  text), **inferred** (search-synthesis only), or **prior knowledge** (well-established ML fact,
  not independently re-fetched this pass, named as such rather than over-graded).
* Read first: [ADR-024](../adr/024-run2-trained-thought-adapter-ladder.md) §"DIAGNOSIS (2026-08-29):
  the adapter collapsed to a fixed OFF-MANIFOLD direction" and §"CORRECTION to the M4d
  registration"; [research/033](033-rescale-output-alignment-diagnostic.md) (the diagnostic that
  found the collapse); [research/032](032-injection-configuration-science.md) (injection-config
  science — angle/norm decomposition, exposure bias, LAP diagnostics); [ADR-028](../adr/028-evolutionary-adapter-search-anti-gaming.md)
  (evolvable/protected surfaces — the frozen probe is protected, architecture is evolvable).

---

## Answer, up front

**Recommend mechanism: attention over a frozen bank of real receiver states (convex-combination
output), trained with the SAME task loss as M4c/M4d, everything else held fixed.** Fallback:
residual/delta anchored to a nearest bank state, delta L2-penalized. Both are directly
implementable in candle 0.9.2's AdamW-only constraint, need no new data capture, and cost roughly
M4c's ~0.45 GPU-h for 10 epochs (§5).

**Overwrite-vs-fuse is very likely a second, independent root cause — verified from the C2C paper's
own equation, not inferred.** C2C's fuser is `𝒞_F = 𝒞_n(X) + ℱ_n(...)`: a **residual add** onto the
receiver's own cache, never a replacement (§4). LatentMesh's `inject.rs` performs a hard
`slice_assign` **overwrite** at the 8 placeholder positions (verified from source, §4). This is
orthogonal to the manifold-constraint question and is named here as a **candidate M4g**, not folded
into M4f — conflating the two would violate this ladder's own "one architecturally distinct change
per rung" discipline (ADR-024's own framing of M3 vs M4 vs M4c).

**The single biggest risk M4f carries**: a bank-attention adapter that is genuinely on-manifold
(passes the cheap pre-check) can still null against the frozen probe, for the same reason M3/M4
already did. This is not a hypothetical — it is the ladder's own recorded history. §1 below shows,
from an already-run committed receipt, that on-manifold-ness and task-loss training have so far
been *mutually exclusive properties* across this ladder's 8 trained adapters: M3/M4 (reconstruction
loss) are on-manifold and null; M4c/M4d (task loss) are off-manifold and null. **No rung has yet
tested the fourth cell — on-manifold AND task-loss-trained** — and M4f is exactly that cell, not a
guaranteed fix. If it also nulls, the parsimonious reading shifts to "manifold location was never
the bottleneck; item-specific *content* is" (testable pre-probe, §3), or to the still-open M4b
(receiver scale) / M4e (continuous injection) / M4g (fuse-not-overwrite) axes, none of which M4f
itself tests (§5, "what M4f does not test").

---

## 1. Why does training collapse to a constant off-manifold direction?

**This does not need to be answered from the SSL-collapse literature alone — it is already answered,
directly and cheaply, by a receipt that exists in this repo right now**:
`crates/latentmesh-runtime/receipts/run2-manifold-precheck-receipt.json`, produced by
`crates/latentmesh-runtime/examples/run2_manifold_precheck.rs` — this **is** the "cheap pre-check"
ADR-024's own M4f paragraph named ("re-run the same unembedding projection on M4d's artifact and on
the M3/M4 artifacts to establish whether off-manifold collapse is universal across the ladder or
specific to task-loss training"). It was already run this session (git commit `e7e36ff`,
CPU-only, no probe draw, `protocol_safety: "ANNOTATES ONLY"`), and its `verdict` field answers the
question directly:

```json
"verdict": {
  "answer": "TASK-LOSS-SPECIFIC — only the task-loss adapter collapsed; the reconstruction-trained M3/M4 nulls have a DIFFERENT cause",
  "reconstruction_trained": { "collapsed": 0, "n": 6 },
  "task_loss_trained":      { "collapsed": 2, "n": 2 },
  "training_free_affine":   { "collapsed": 0, "n": 2 }
}
```

Per-candidate detail (`classification`, cosine-to-natural-pooled-state — the on-manifold metric —
and mean-pairwise-cosine — the item-invariance metric):

| candidate | loss | classification | cos-to-natural | item-invar. cos |
|---|---|---|---|---|
| run1 affine (both) | closed-form, training-free | on-manifold | 0.974 / 0.996 | 0.966 / 0.969 |
| M3 MLP (pertoken, pooled) | reconstruction (MSE) | on-manifold | 0.989 / 0.975 | 0.970 / 0.964 |
| M4 FastGRNN (r64, r128, r256) | reconstruction (MSE) | on-manifold | 0.984 / 0.983 / 0.985 | 0.980 / 0.974 / 0.973 |
| M4 r64 superseded (bad init) | reconstruction (MSE) | on-manifold (weak) | 0.511 | 0.978 |
| **M4c-init (untrained, pre-training)** | n/a — random init | **COLLAPSED-OFF-MANIFOLD** | **−0.021** | 0.975 |
| **M4c trained** | task loss (CE) | **COLLAPSED-OFF-MANIFOLD** | **−0.018** | 0.881 |
| **M4d-init (untrained)** | n/a — identical to M4c-init | **COLLAPSED-OFF-MANIFOLD** | **−0.021** | 0.975 |
| **M4d trained** | task loss (CE) + deploy-transform-in-loop | **COLLAPSED-OFF-MANIFOLD** | **0.048** | 0.907 |
| reference: real receiver states (pooled) | — | on-manifold (trivially) | 1.000 | 0.962 |

**This is the load-bearing fact, and it reframes RQ1 away from "collapse during optimization" and
toward "never left an off-manifold init region, because nothing in the loss pushed it there."** The
untrained MLP init — before a single gradient step — is *already* at cosine ≈ −0.02 to the natural
manifold (m4c-init and m4d-init are numerically identical, confirming the same frozen seed).
Training under task loss moves that number by 0.003–0.07, staying deep in collapsed territory.
Training under reconstruction (MSE) loss, from a *different* random init not measured here,
converges to cosine 0.97–0.99. The mechanistic story this supports is simple and does not require
positing an active, adversarial shortcut:

- **Reconstruction loss is on-manifold by construction.** MSE against real receiver states *is*
  a distance-to-manifold penalty; minimizing it necessarily pulls the output toward the target
  distribution. There is no way to reduce MSE against real states without moving onto (or very near)
  the manifold those states define.
- **Task loss (next-token CE on the sender's span, delivered through the overwrite injection) is
  manifold-agnostic.** Nothing in the CE objective rewards "look like a real receiver state" — it
  only rewards "however I land in 1536-dim space, shift the receiver's downstream logits toward the
  target tokens." A random, item-invariant direction that happens to already exist at init (a
  generic property of ReLU-MLP inits in a much larger ambient space than the natural manifold
  occupies — `docs/research/032` §1 independently confirms the natural distribution is a narrow band
  except for massive-activation outliers) can partially reduce CE without ever needing to move.
  Gradient descent has no incentive to leave a region that already yields *some* loss reduction, even
  a bad one — and M4c's own probe result (aligned NLL 2.5× *worse* than baseline, §"M4c outcome")
  confirms the direction it settled on is not even a good local optimum for the objective it was
  trained on, just one gradient descent had no structural reason to escape.

**Relation to the SSL-collapse literature (representation collapse in BYOL/SimSiam, dimensional
collapse, VICReg):** this is a related but distinct failure shape. Canonical SSL collapse is a
*dynamic* phenomenon — an encoder with a shared target and no asymmetry (stop-gradient, a predictor
head) drifts to a trivial constant solution *during* training because the objective is satisfiable
by ignoring the input entirely. What this repo's own receipt shows is closer to **objective
underdetermination at init**: the loss (task CE via an overwrite channel) never uniquely determines
the output's *location* in the 1536-dim ambient space, only some coarse aggregate property of its
effect on logits, and a random init already sits somewhere loss-reducible. The fix families converge
regardless of which framing is more precise (§2) — both call for making "stay near the natural
manifold" a load-bearing part of the objective or the architecture, not an emergent property task
loss happens to select for. VICReg's variance/covariance regularization (Bardes, Ponce & LeCun,
ICLR 2022, arXiv:2105.04906, **primary**, confirmed via search of the published abstract/method) is
the standard SSL-collapse fix and is directly portable here as an auxiliary loss term (§2, mechanism
c) — but per the argument above, a *structural* fix (§2, mechanism a) is preferred over a *penalty*
fix for this specific failure shape, because the problem is that the objective doesn't care where the
output lands, not (only) that an unconstrained penalty-free optimum happens to be degenerate.

**Does the overwrite-not-add injection make collapse more likely?** Yes, plausibly, though this
pass did not isolate it as a controlled variable. Overwriting means the receiver's own downstream
processing at blocks 15–28 has *no* access to whatever the placeholder position's natural content
would have been — the adapter's output is the *entire* signal at that position, with no floor of
"at least it's a real state" to fall back on. An additive/residual formulation (§2 mechanism b, and
§4's C2C precedent) removes this specific risk by construction: even a near-zero-learned delta still
leaves a real (if content-free) state as the base.

---

## 2. Concrete mechanisms to constrain output to a learned manifold, ranked by candle 0.9.2 implementability

All of `latentmesh-train`'s existing rungs use candle 0.9.2, `AdamW`-only (no scheduler,
`candle-nn 0.9.2` ships exactly `SGD`/`AdamW` per ADR-024's own infra scout), batch=1, single-GPU
16 GB budget. Every mechanism below is scoped against that constraint.

**(a) Predict a convex combination / soft-attention over a bank of REAL receiver states — top
recommendation.** Freeze a bank `B ∈ ℝ^[K×1536]` of `K` real receiver-L14 per-token rows, sampled
once (seeded) from the existing `receiver_L14.tok.f32bin` per-token dump (already captured at M2,
no new data needed — `crates/latentmesh-train/src/dataset.rs::LayerMap::row(t)` is the existing
accessor). Content-hash and freeze `B` in the training receipt, same discipline as the golden pairs.
Per token: a small projector (the M3 MLP architecture, reused unchanged for comparability) maps the
sender's L18 state to a query `q ∈ ℝ^1536`; attention weights `w = softmax(q·Bᵀ/√d)`; output
`= w·B`. **Guarantees on-manifold by construction** — any convex combination of points in a bounded
cluster stays inside that cluster's convex hull, which per `docs/research/032` §1's massive-activation
finding is exactly the shape of the natural distribution (a tight band plus rare outliers a softmax
naturally down-weights unless attending directly to them). **Breaks item-invariance by construction
too** — the query is item-conditioned, so the weights (and hence the output) vary per item, which
directly targets the sharpest symptom `docs/research/033` found (77 distinct tokens, 3 dominant
across all 40 items). Cost: `K×1536` softmax + weighted sum per token — negligible next to a
1.5B-parameter receiver forward/backward (the actual cost driver in every rung to date). **Can it
express item-specific content at all?** Yes, and this is the mechanism's main advantage over (d):
the combination is continuous, not a hard nearest-neighbor pick, so it is not limited to the exact
`K` states in the bank — it can represent any point in their convex hull, a much larger, still
manifold-constrained space. Risk: if attention collapses to near-uniform (or a fixed few bank rows)
regardless of item, this degrades toward exactly the current on-manifold-but-item-invariant M3/M4
regime — a diagnostic §3 must check pre-probe.

**(b) Residual formulation: output = anchor + small delta, delta penalized — recommended fallback.**
Anchor = a bank state chosen by nearest-neighbor to the sender's query (or simply the item's own
natural mean, already computed as `natural_inject_block_norms`' companion statistic). Delta =
learned MLP output, penalized via `λ‖delta‖²` in the loss (λ a new frozen hyperparameter,
disclosed). Cheaper than (a) (no softmax/bank matmul, one nearest-neighbor lookup which can be
precomputed once per item since the anchor doesn't need to be differentiable). Directly analogous to
C2C's own verified residual fuser (§4) — the closest single-source precedent found in this pass.
Weaker than (a) on expressiveness (a small-norm delta around one anchor covers less of the manifold
than a combination over many), and does not structurally prevent delta-collapse to an item-invariant
direction the way (a)'s item-conditioned attention does — the same underlying problem (task loss
doesn't reward item-specificity) could reassert itself one level up, on the delta instead of the
whole vector. Implementable trivially in candle: an L2 penalty term is one `.sqr()?.sum_all()?` op
added to the CE loss before `backward()`.

**(c) Distributional penalties (VICReg-style variance/covariance, or Mahalanobis/energy-distance to
the natural distribution) — implementable, not preferred as primary.** A batch-level regularizer:
maintain running mean/covariance (or a frozen precomputed one) of the natural receiver-L14
distribution from the same bank as (a); add `λ·mahalanobis(output, μ, Σ)²` or a VICReg-style
variance-floor + covariance-decorrelation term across the batch to the CE loss. Fully implementable
in candle (mean/covariance are matmuls; Mahalanobis distance needs a precomputed inverse covariance,
computable host-side with `nalgebra`/`ndarray` once and loaded as a frozen constant — no candle
autodiff needed for the *inverse*, only for the quadratic form against the live output, which is a
plain matmul). **Named as not preferred over (a)/(b) for this specific problem** because, per §1's
diagnosis, the failure is that the objective never constrains *location* at all — a soft penalty
still leaves gradient descent free to trade CE reduction against manifold-distance at whatever λ is
chosen, reproducing the exact "task loss found a shortcut around the constraint" pattern §1 already
diagnosed once (this is the "regularizer vs. structural constraint" distinction the risk section
below returns to). Worth keeping as a *diagnostic* (compute Mahalanobis distance as a report metric)
even if not adopted as the training mechanism.

**(d) Nearest-neighbor projection / VQ-style quantization onto observed states.** Hard-assign the
predicted vector to its single nearest bank state (a discrete, non-differentiable op — needs a
straight-through estimator, `candle-nn` does not ship one, would need hand-rolling the gradient
copy-through). Structurally the tightest manifold guarantee of any mechanism here (output is
*literally* an observed state, not even a combination) but at real expressiveness cost: **a
constraint this tight may transmit nothing item-specific beyond which of `K` discrete states was
picked** — named explicitly, per the parent brief's instruction, as the sharpest version of the
"too-tight-to-carry-content" tradeoff every mechanism on this list carries to some degree. Ranked
last on implementability (straight-through estimators are a real but nontrivial candle 0.9.2 lift,
unlike (a)–(c) which are compositions of ops candle already has) and on expressiveness.

**(e) Train with the natural-state distribution as an explicit prior/regularizer.** This is (c)
restated at the population level (a KL-style or moment-matching prior over the whole training
distribution rather than a per-example penalty) — same implementability profile and same "soft
constraint, same failure shape possible" caveat as (c). Not separately ranked.

**Summary ranking for M4f: (a) primary, (b) fallback, (c) demoted to diagnostic-only, (d) ranked
last (implementability + tightness-vs-content tradeoff), (e) folded into (c).**

---

## 3. Diagnostics: on-manifold vs. item-varying vs. content-useful

All of the following are **already implemented and already run** in
`crates/latentmesh-runtime/examples/run2_manifold_precheck.rs` (extending
`docs/research/033`'s metric kit, factored into `common/lens.rs` so both diagnostics share one
implementation) — M4f's pre-probe gate is "run this same script against M4f's own artifact," not
new tooling:

- **On-manifold**: cosine between the emitted vector and the same-item natural receiver-L14 pooled
  state (registered threshold already in the precheck receipt: collapsed if `< 0.2`; M3/M4's
  passing range is 0.97–0.99, which M4f should target).
- **Item-invariance**: mean pairwise cosine between the 40 emitted vectors across items (registered
  threshold: collapsed if `≥ 0.95` — M4f should score noticeably *below* this, since low
  item-invariance combined with on-manifold-ness is the actual target signature, not merely
  clearing the on-manifold bar alone).
- **LAP `A_lin`-style output alignment** (`docs/research/032` §4, `docs/research/033` §4): gold-answer
  token rank and sender-span token rank through the receiver's real `W_U·RMSNorm(h)` readout. M4c
  scored gold at the 61st percentile (worse than chance); M4f's pre-probe gate should require a
  materially better gold-token rank, not just an on-manifold cosine, precisely because §1's finding
  that on-manifold-ness and content-usefulness are independent properties (M3/M4 were on-manifold
  and *still* uninformative by this same metric-family) is the whole reason M4f is risky (see
  "biggest risk," above and §5).
- **Participation ratio / effective dimensionality of the emitted 40-item set**: not yet in the
  precheck script; a cheap addition (eigenvalues of the 40×1536 emitted-vector covariance, PR =
  `(Σλ)²/Σλ²`). Low PR is a second, complementary signature of collapse to a low-rank/near-constant
  set, catching partial collapses the pairwise-cosine metric alone might miss (e.g. collapse to a
  2-3 dimensional subspace rather than a single direction).
- **Distance to k-nearest natural states**: a direct complement to the bank-attention mechanism's own
  training signal — if mechanism (a) is adopted, this is nearly free to compute (the attention
  weights over the bank already answer "how far / how concentrated" for every item) and should be
  reported per item, not just in aggregate.

---

## 4. Prior art: does any successful cross-model method constrain output to the receiver's manifold?

**No cross-model method surveyed constrains its projector's raw output to the receiver's manifold as
a first-class design goal — but the strongest one (C2C) avoids the problem structurally, by fusing
rather than replacing, which this pass verified directly from the paper's own method section rather
than from a search snippet.**

Fetched `arxiv.org/html/2510.03215` (C2C, "Cache-to-Cache: Direct Semantic Communication Between
Large Language Models") directly, **primary**: the fuser's integration equation (the paper's
Equation 3) is

```
𝒞_F = { 𝒞_n(X) + ℱ_n( 𝒞_n(X), 𝒞_{𝒢(n)}^𝒮(X) ) }
```

— the fused cache is the receiver's **own original cache plus** a learned fuser function's output, a
**residual add**, not a replacement. The paper's own framing: this "preserv[es] the receiver's
computed content through residual blending." Confirmed also: fusion happens **once, during prefill**
of the source context (not re-applied at every decode step — this nuance matters for how
`docs/research/032` §2 characterized C2C as injecting "continuously... in lockstep"; that framing may
describe the Bicameral Model more precisely than C2C, whose own equation is a one-shot prefill-time
residual add that then persists through the KV cache exactly the way LatentMesh's own injection
persists — worth flagging as a minor correction to research/032's characterization, though resolving
it fully is M4e's concern, not M4f's). Also confirmed: "selectively applying cache enrichment to the
top-performing layers... yields slightly higher accuracy than enriching all layers" — the
already-cited motivation for C2C's learned gate (research/032 §2, re-confirmed here).

**LatentMesh's `inject.rs` is verified, from source, to do the opposite of C2C's mechanism.**
`crates/latentmesh-runtime/src/models/qwen2_b.rs` lines 79–87 (`apply_edit`, the `LayerEdit::Inject`
arm): for each placeholder position, `xs = xs.slice_assign(&[0..1, pos..pos+1, 0..hidden], &v)` —
a hard overwrite of the residual row, discarding whatever the placeholder token's own forward pass
would have produced there. There is no `+` in this code path; `docs/research/033`'s own analysis
(quoted in ADR-024's DIAGNOSIS section) already established the residual row **is** `c·v` after
injection, not `h + c·v`.

**Is overwrite-vs-fuse the deeper root cause, more fundamental than the manifold question M4f
targets?** It is a *plausible and now source-verified* candidate, genuinely orthogonal to
manifold-constraint: even a perfectly on-manifold, perfectly item-varying output could still be
destructive if overwriting it destroys information blocks 15–28 would otherwise have needed (a
distinct mechanism from "the output points at the wrong place," closer to "the injection method
itself is lossy regardless of what's injected"). **This pass does not resolve which of the two
matters more** — that would require running both axes and cannot be inferred from either the C2C
citation or the manifold-precheck receipt alone. **Recommendation: keep it out of M4f's scope and
name it explicitly as a follow-on, M4g** (overwrite → residual-add at the injection site, requiring
its own architectural change to `LayerEdit` and its own pre-registration addendum, exactly as M4e's
continuous-injection change does) — conflating two architecturally distinct changes into one rung
would violate the very discipline that let this ladder cleanly attribute M3-vs-M4 and M4-vs-M4c to
single factors. If M4f (on-manifold, still overwrite) nulls, M4g becomes the best-motivated next
rung, now backed by a primary citation rather than an inferred one.

---

## 5. SPEC — M4f, pre-registerable

**Architecture (primary mechanism).** "Bank-attention MLP": sender L18 per-token state → M3-shaped
query projector (2048→512→1536 ReLU, same architecture as every prior rung, to keep the *only*
changed factor the output-constraint) → softmax attention over a frozen, seeded, content-hashed
bank `B` of `K=512` real receiver-L14 per-token rows (sampled once from the existing
`receiver_L14.tok.f32bin`, excluded of the same 13 leakage rows, disclosed in the receipt) →
output = `w·B` per token → mean-pool over the generated span (unchanged from every prior rung) →
rescale-to-natural-median (unchanged, frozen probe step) → 8-slot broadcast → inject via the
**existing, unchanged** overwrite `LayerEdit::Inject` at block 14 (M4g, not M4f, changes this) →
task-loss CE on the sender's generated span (C2C-style, unchanged from M4c/M4d — **not** switched to
gold-answer CE, per §1's note that DIAGNOSIS already retired the target-mismatch hypothesis; changing
the target here would reintroduce a second confounded factor).

**Fallback mechanism**, if bank-attention proves numerically unstable or the softmax collapses to a
near-constant weighting across items during a short pilot (checked at epoch 1-2 via the item-
invariance diagnostic, §3, before committing the full 10-epoch budget): residual/delta — output =
nearest-bank-neighbor(query) + MLP(query), with an added `λ‖delta‖²` term in the loss, λ a new
frozen hyperparameter disclosed in the receipt (start at λ=0.1, chosen to keep delta magnitude
comparable to natural per-position variance around the mean — a to-be-measured quantity from the
bank itself, not guessed).

**Objective**: identical to M4c/M4d — next-token CE on the sender's generated span, teacher-forced,
through the composed differentiable BF16 forward (`qwen2_c`), the same feasibility-validated path
M4c/M4d already use. **Only the output-constraint architecture changes.**

**Splits**: identical to every prior rung — `fit_holdout_split(2560, FIT_SPLIT_SEED=0x24C0_DE03)`,
then the same 13 probe-overlap rows dropped (`n_fit ≈ 2,035`, `n_holdout ≈ 509`, per-row disclosure
in the receipt, matching M3/M4/M4c/M4d exactly).

**Training config**: AdamW lr=1e-3 (unchanged), batch=1 (unchanged), 10 epochs with best-holdout-CE
epoch selection (M4c's own rule — selected epoch 4 there; M4f should use the same selection
discipline, not a fixed epoch count), `SEQ_CAP=256` (the measured VRAM envelope, unchanged), same
fresh seeded init family (a new, disclosed seed for the query-projector weights; the bank sample
seed is a second, separately disclosed seed).

**Pre-probe gate (new, specific to M4f, run before the one frozen draw)**:
1. `run2_manifold_precheck`-style check on M4f's own trained artifact: cosine-to-natural ≥ 0.7
   (a deliberately looser bar than M3/M4's 0.97–0.99, to allow for the bank-attention's convex-hull
   output sitting slightly inside rather than exactly on the reference cluster — disclosed as an
   authorial choice, not re-derived from any registered threshold, since no prior rung needed this
   exact bar).
2. Item-invariance strictly below the registered 0.95 collapse threshold, and ideally well below
   M3/M4's own 0.96–0.98 (which were on-manifold but still fairly item-invariant, and still nulled —
   §1's "fourth cell" framing).
3. Gold-answer token rank (LAP-style, `docs/research/033` §4's exact metric) materially better than
   M4c's 61st-percentile result — no fixed numeric bar is pre-registered here (naming one without
   a distribution to calibrate against would itself be an unprincipled threshold), but the receipt
   must report the number and the reader must be able to see whether it moved.
4. All three must be computed and disclosed in the training receipt **before** the frozen probe is
   invoked, exactly mirroring M4c's `transfer_check_passed_before_probe` gate discipline. **None of
   the three is a pass/fail gate on whether the probe gets drawn** — M4f gets its one frozen draw
   regardless (per ADR-028's promotion rule, one champion per rung), but the pre-probe numbers are
   the honest, protocol-safe context for interpreting whatever the probe says, computed the same way
   M4c's transfer check was computed before, not after, its own probe run.

**Gate (frozen probe, unchanged)**: aligned-real > random, one-sided exact sign test, p<0.05, on the
same 40-item S1a/S2b protocol every rung has used, single seeded run, training receipt frozen before
the probe invocation (the freeze point, per ADR-024's own numbers-rule discipline).

**GPU-hour estimate**: **~0.5-0.7 GPU-h for 10 epochs**, by direct analogy to M4c's receipted
**1,604 s (≈0.446 GPU-h)** for the identical receiver-forward/backward cost driver (same frozen 1.5B
receiver, same composed BF16 forward, same SEQ_CAP=256, same batch=1, same epoch count) — the added
cost is the bank-attention's own `K×1536` softmax+matmul per token, `K=512`, which is negligible
FLOPs next to a 1.5B-parameter forward+backward pass. **This is an inferred estimate by analogy, not
a measurement** — named as such, same discipline as the M0 scout's own capture-time extrapolation in
ADR-024. A short 1-epoch pilot run, timed before committing to the full 10-epoch budget, is the cheap
way to convert this from inferred to measured before spending the bulk of the GPU-hours.

**Receipts**: same format as every M2-M6 artifact per ADR-024 § Training receipts — `evidence_label`,
explicit seeds (training RNG, bank-sample RNG, `FIT_SPLIT_SEED`, the frozen probe's
`item_seed_chacha8: 20897`), the excluded-13-rows list, `git_commit`, GPU/nvcc environment,
wall-clock/GPU-seconds, plus the three new pre-probe-gate numbers from above and the bank's own
content hash (frozen exactly like the golden pairs).

**Honest-fail path**: identical to every prior rung — reported, preserved, receipt kept regardless of
outcome. If M4f nulls with a *good* pre-probe profile (on-manifold, item-varying, gold-rank
improved), that is the sharpest possible result this ladder can produce short of a pass: it would
mean location-on-manifold and content-specificity are BOTH achievable and STILL insufficient,
strengthening the case for M4b (receiver scale) and M4e/M4g (delivery mechanism) as the remaining
live axes over any further adapter-architecture search. If M4f nulls with a *bad* pre-probe profile
(still collapsed, despite the bank-attention design), that is itself informative — it would show the
mechanism proposed here is not sufficient to escape the failure §1 diagnosed, and M4f's own fallback
(§2 mechanism b) or a fresh design would be needed before any further probe draw, per ADR-028's
one-champion-per-rung promotion rule (a second M4f variant is a new rung, not a retry).

**What M4f does NOT test** (named explicitly, per the parent brief's instruction):
- **Receiver scale** (M4b, still mandatory, independent, unaffected by anything in this document).
- **One-shot-vs-continuous injection** (M4e) — M4f injects once, upfront, exactly like every prior
  rung; this axis is untouched.
- **Overwrite-vs-fuse at the injection site** (the new M4g candidate this scout names, §4) — M4f
  uses the existing, unchanged overwrite `LayerEdit::Inject`. If M4f nulls, M4g is the best-motivated
  next rung on the strength of the primary C2C citation this document adds.
- **Training target / loss shape** (sender-span CE vs. gold-answer CE vs. a weighted combination) —
  unchanged from M4c/M4d, per §1's note that DIAGNOSIS already retired target-mismatch as the leading
  explanation.
- **Whether the natural manifold itself is the right constraint target** — the bank is drawn from
  the *training* distribution's natural states; nothing here tests whether a differently-defined
  "on-manifold" (e.g., states specifically from correct-answer trajectories, or from a different
  layer) would do better. Named as a possible M4h if M4f's own pre-probe diagnostics (§3) suggest
  the bank itself is a poor proxy for the region the receiver's readout actually respects.

---

## Sources

- ADR-024 §"DIAGNOSIS", §"CORRECTION to the M4d registration", §"M4c outcome",
  §"M5 SUPERSEDED AND REDESIGNED" (house style for a scout correcting a one-paragraph registration)
- `docs/research/033-rescale-output-alignment-diagnostic.md` (the diagnostic that found the collapse)
- `docs/research/032-injection-configuration-science.md` §1 (angle/norm decomposition, massive
  activations), §2 (C2C placement/gating, re-confirmed here), §3 (exposure bias / DAgger), §4 (LAP)
- `docs/adr/028-evolutionary-adapter-search-anti-gaming.md` (evolvable/protected surfaces, promotion
  rule, one-champion-per-rung)
- `crates/latentmesh-runtime/receipts/run2-manifold-precheck-receipt.json` — **primary, already-run,
  committed receipt this scout reads rather than re-derives**; answers §1 directly
- `crates/latentmesh-runtime/examples/run2_manifold_precheck.rs` — the classifier/diagnostic script
  §3 recommends reusing unchanged
- `crates/latentmesh-runtime/src/models/qwen2_b.rs` lines 31-91 (`LayerEdit`, `apply_edit`) —
  source-verified overwrite mechanism, §4
- `crates/latentmesh-runtime/src/inject.rs` — `InjectionSpec`, source-verified rescale-is-a-scalar
  fact §1/§4 build on
- `crates/latentmesh-train/src/bin/train_m4c_taskloss.rs`, `train_m4d_deploymatch.rs`,
  `src/mlp.rs`, `src/dataset.rs` (`LayerMap::row`) — architecture/pipeline facts §2/§5 build on
- Cache-to-Cache: https://arxiv.org/abs/2510.03215, full text fetched at
  https://arxiv.org/html/2510.03215 (**primary**, this pass) — Equation 3, the residual-fuser
  mechanism verified in §4
- VICReg: Bardes, Ponce & LeCun, ICLR 2022, https://arxiv.org/abs/2105.04906 (**primary** via
  search-confirmed abstract/method, not independently fetched full-text this pass)
- Massive activations / attention sinks — already primary-verified in `docs/research/032` §1,
  re-cited here for the "natural manifold is a narrow band" claim §2/§5 rely on
