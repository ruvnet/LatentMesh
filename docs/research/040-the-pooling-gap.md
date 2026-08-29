# 040. The pooling gap — is mean-pooling the ladder's real bottleneck?

* **Purpose**: `docs/research/036` measured that a genuine receiver block-14 state and that same
  item's **pooled** state have cosine **0.667**, entropy **9.30 vs 3.36** nats, cross-item
  invariance **0.962 vs 0.635** — and that **every rung of the ladder, run-1's training-free affine
  bridges included, mean-pools a ~281-token generated span into ONE vector before injecting it into
  8 placeholder slots.** So every "on-manifold" verdict in this ladder means *on the manifold of
  pooled states*, which is itself well off the manifold the receiver actually carries. This
  document verifies, from primary sources, whether any successful cross-model method pools;
  surveys what pooling destroys; ranks de-pooling alternatives against this stack's own
  constraints; argues whether pooling is upstream of the ladder's other live hypotheses; and
  recommends the next rung.
* **Date**: 2026-08-28 (branch `feat/run2-thought-adapter`, read-only except this file — not
  committed).
* **Method / evidence grading**: WebFetch of primary sources (papers, specs, source code) plus
  direct repo source-reading (`crates/latentmesh-runtime/src/{inject.rs,models/qwen2_b.rs}`,
  `crates/latentmesh-train/src/dataset.rs`, `docs/adr/028`). **primary** = paper/spec/source
  fetched and the specific claim read in the fetched text. **secondary** = search-result synthesis
  citing but not independently re-fetched. **inferred** = this document's own reasoning from
  confirmed facts. WebSearch was unavailable for most of this pass (session budget exhausted before
  this task started); WebFetch against known URLs (arxiv abs pages, GitHub raw files, ar5iv full-text
  renders) was used throughout instead, and is graded identically to a WebSearch-sourced primary
  fetch — the grading is about what was read, not which tool read it.
* **Read first**: [docs/research/036](036-manifold-collapse-across-the-ladder.md) (the finding that
  prompted this document — §4 especially), the "CORRECTION to the M4f pre-check verdict" section of
  [ADR-024](../adr/024-run2-trained-thought-adapter-ladder.md), [docs/research/038](038-manifold-constrained-adapter-scout.md)
  (M4f's bank-attention mechanism, C2C's fuser verified from source), [docs/research/039](039-bidirectional-latent-exchange.md)
  (Bicameral's continuous coupling, LatentMAS's unidirectionality, C2C's one-shot-at-prefill
  correction), [ADR-028](../adr/028-evolutionary-adapter-search-anti-gaming.md) (evolvable/protected
  surfaces — load-bearing for §3's probe-compatibility verdict, and it contains an internal tension
  this document flags rather than resolves).

---

## Answer, up front

**No successful cross-model method pools a span into one vector, and this is the single largest
verified structural difference between this ladder's design and every method that works — larger
than overwrite-vs-fuse.** C2C transfers per-token KV-cache entries at full sequence length, every
layer (gated per-layer, never sequence-reduced). LatentMAS prepends full per-token, per-layer KV
caches with zero pooling — its "shared latent working memory" is a literal cache concatenation, not
a summary. The Bicameral Model couples per-token hidden states continuously, every decode step —
never a pre-computed static vector at all. AVP's cross-model path is a per-token vocabulary-mediated
projection with no pooling in its core algorithm (one caveat below). **MANTA is not a
latent-vector-transfer method and does not belong in this comparison at all** — verified from its
own abstract, it restructures multi-agent *topology* (roles, links, execution order, validation
pathways), not the representational shape of what agents exchange; treating it as a peer of
C2C/LatentMAS/Bicameral on the pooling axis would be a category error, and this document corrects
that framing rather than forcing a false answer.

**Pooling destroys exactly what the sentence-embedding literature has documented for a decade, now
reproduced at reasoning-trace scale.** Mean-pooling collapses distinct per-token content toward a
common, frequency/anisotropy-dominated centroid — the mechanism Li et al. (2020) trace to the
masked-LM objective's own geometry, and Ethayarajh (2019) traces to contextualized embeddings being
non-isotropic in every layer of every model tested. `docs/research/036`'s own numbers are this
mechanism's signature at a new scale: cross-item invariance 0.962 (pooled) vs 0.635 (real states) is
the "collapse toward a shared direction" the sentence-embedding literature already named; entropy
9.30 vs 3.36 nats is the "decommitment" that centroid-hood produces once read back through a real
unembedding, rather than compared by cosine alone as the classical literature mostly does.

**The cheapest possible next experiment is a near-zero-cost pilot: swap M3's existing pooled-mean
output for its last-token output, reusing the already-trained weights, already-captured data, and
already-frozen 8-slot injection mechanism, for one new frozen-probe draw.** The primary recommended
full rung is an 8-slot **sequence** injection — replace the single broadcast vector with 8
architecturally-distinct per-slot vectors, produced by a small attention-compression module over the
sender's own per-token stream (the M4f bank-attention mechanism, redirected from "attend over a
frozen receiver-state bank" to "compress T_i sender states into 8 outputs"), trained under the same
task loss as M4c/M4d/M4g. **Probe-compatibility verdict: yes, unchanged frozen probe, provided slot
count stays fixed at exactly 8** — this sidesteps an internal tension in ADR-028's own evolvable/
protected lists (§3) rather than silently picking a side of it. Pooling is **upstream of** the
manifold-membership framing (M4f as originally scoped) because "on-manifold" was measured the whole
time relative to a pooled reference that is itself 0.667-cosine off the real manifold; it is
**orthogonal to** overwrite-vs-fuse (M4g) and receiver scale (M4b); and it is **substantially
entangled with** one-shot-vs-continuous injection (M4e), because every method that avoids pooling
also avoids one-shot delivery — de-pooling and continuous delivery are two faces of nearly the same
underlying difference from what works.

---

## 1. Do any successful cross-model methods pool a span into a single vector?

### 1.1 Cache-to-Cache (C2C) — no. Per-token, full sequence, every layer (gated).

**[primary, fetched `arxiv.org/html/2510.03215` this pass, plus re-confirmed against
`docs/research/038`'s prior primary read of the same paper].** The transferred object is per-token
KV-cache entries, not a pooled vector: the paper's own cache notation is
`𝒞(X[0:n]) = [c₀,…,c_{n-1}] ∈ ℝ^{n×d}` — one row per token position, `n` the sequence length. The
fuser `ℱ_n` takes the receiver's own layer-`n` cache and the sender's corresponding layer-`𝒢(n)`
cache and produces `𝒞_F = 𝒞_n(X) + ℱ_n(𝒞_n(X), 𝒞_{𝒢(n)}^𝒮(X))` (Eq. 3) — a residual add, at full
token-sequence granularity, with **no reduction in token count during fusion**. Layer coverage: the
fuser architecture supports every layer, but a **learned per-layer gate** decides which layers
actually get enriched ("enriching only selected layers is better than enriching all layers" — this
paper's own ablation, already primary-verified in `docs/research/038` §4). Fusion happens once, at
prefill (already established in `docs/research/039` §1.3, correcting `docs/research/032`'s earlier
"continuous, in lockstep" characterization) — but "once at prefill" is orthogonal to pooling: the
one-shot fusion still operates on the full per-token cache, not a summary of it. **No pooling or
averaging across the sequence dimension occurs anywhere in the described pipeline.**

### 1.2 LatentMAS — no. Full per-token, per-layer KV cache, literally concatenated.

**[primary, fetched `arxiv.org/html/2511.20639v1` this pass].** The "shared latent working memory"
LatentMAS's framing evokes is, mechanically, a KV-cache prepend: agent A₂ receives A₁'s cache
`K^{(l)}_{A₁,cache} = [K^{(l)}_{A₁,1}, …, K^{(l)}_{A₁,t+m}]` for every layer `l` — the notation is
explicit that the cache grows along the sequence dimension (`t+m` positions, no dimensionality
reduction) — and A₂ updates its own cache by "layer-wise concatenation... prepending each
`K^{(l)}_{A₁,cache}` and `V^{(l)}_{A₁,cache}` to existing `K^{(l)}_{A₂,cache}`," across **all `L`
transformer layers, extracted once**. This is confirmed strictly unidirectional and one-pass
(`docs/research/039` §1.2) — but again, unidirectionality is orthogonal to pooling. **No averaging
or pooling occurs**; each successor agent receives the complete, sequence-length-preserving cache
from every predecessor layer.

### 1.3 The Bicameral Model — no, and not even a static object to pool.

Already established as primary fact in `docs/research/039` §1.1, re-read here for the pooling
question specifically: the coupling is **per-step, per-token, live** — `h_a ← (1−σ)h_a + σ·f(h_p)`
at every generation step, a translated-and-gated hidden state, never a pre-computed summary of a
whole span. There is structurally nothing in this design to pool: each step's translated signal is
the current step's own hidden state, consumed and discarded before the next step produces a fresh
one. This is the sharpest possible negative answer to "does a successful method pool" — Bicameral's
architecture makes pooling not just unused but incoherent with the method's own control flow.

### 1.4 AVP (Agent Vector Protocol) — no in its core algorithm, with one disclosed caveat.

**[primary, fetched `raw.githubusercontent.com/VectorArc/avp-spec/main/SPECIFICATION.md` this
pass].** AVP is a shipped, unpublished-as-a-paper protocol (`github.com/VectorArc/avp-{spec,python}`),
already flagged in `docs/research/035` §"AVP" as unverified against any benchmark suite — that
caveat is unaffected by this pass and is restated here, not resolved. The spec text for the
cross-size vocabulary-mediated projection path (its "Rosetta Stone v2" mode) is explicit and
per-token: `hidden @ W_src^T → logits [vocab_size] / softmax(logits) → token probabilities
[vocab_size] / probs @ W_tgt → target embedding [D_tgt]` — a token-by-token logit-lens-style
projection, one source token's hidden state mapping to one target token's embedding, sequence
structure preserved, **no pooling or averaging across tokens in the algorithm itself.** The
disclosed caveat: the same spec text separately describes a priming mode — "inject a single
projected embedding to prime the target model's context, then feed actual text tokens" — for short
structured prompts specifically. This is not pooling a generated span (nothing in the spec text
describes averaging multiple tokens' states into that single embedding); it reads as priming with
one token's projected state, not a summary statistic over many. Graded **primary but incomplete** —
the fetched spec excerpt does not resolve exactly how the "single projected embedding" is derived
when the source content spans more than one token, and this is named as an open gap rather than
assumed either way.

### 1.5 MANTA — not a comparable method; scoping correction

**[primary, fetched `arxiv.org/abs/2607.28527` abstract this pass].** MANTA's own abstract describes
"self-evolving topology" that can "modify agent roles, communication links, execution order,
information visibility, and validation pathways" — this is multi-agent orchestration-topology
adaptation, not a description of transferring hidden states or KV-cache content between models at
all. The fetched text gives no indication MANTA operates on latent vectors rather than
inter-agent text/task messages, and this repo's own prior citations of MANTA
(`docs/research/023-beyond-sota-roadmap.md` line 72, `docs/adr/009-online-causal-control-loop.md`)
consistently describe it as a **dynamic communication topology** method, cited for that property
specifically, never for a latent-transfer mechanism. **Correction to the research brief that
motivated this document**: MANTA was named alongside C2C/LatentMAS/AVP as a method to check for
pooling, but it is not a peer of those three on this axis — it answers a different question
("who talks to whom, and in what role") than "what object is transferred" (which is the pooling
question). This document treats MANTA as out of scope for §1's comparison, rather than force-fitting
an answer the fetched text does not support.

### 1.6 Verdict

| method | transferred object | pooled? | grade |
|---|---|---|---|
| C2C | per-token KV cache, full sequence, all layers (learned gate) | **no** | primary |
| LatentMAS | per-token, per-layer KV cache, full sequence, prepended | **no** | primary |
| Bicameral Model | live per-step, per-token hidden state (no static object to pool) | **no** (structurally incoherent with pooling) | primary |
| AVP | per-token vocabulary-mediated projection | **no** (core algorithm); priming-mode caveat undetermined | primary, incomplete |
| MANTA | agent roles/links/order/validation (topology, not latent content) | **not applicable** — not a latent-transfer method | primary (scoping correction) |
| **LatentMesh (this repo)** | one mean-pooled vector over ~281 generated tokens, broadcast to 8 slots | **yes, every rung including run-1's affine bridges and M4's FastGRNN "sequence" translator** | primary, this repo |

**Every method surveyed that demonstrably transfers cross-model causal signal preserves per-token
granularity through to the point of use.** LatentMesh is the outlier, not the field's typical
design — and it is an outlier on an axis (pooling) that has an independent, decades-old literature
predicting exactly the failure mode this ladder keeps observing (§2).

---

## 2. What does mean-pooling destroy?

### 2.1 Anisotropy — pooling averages over a space that is not isotropic to begin with

**[primary, fetched `arxiv.org/abs/1909.00512`, Ethayarajh 2019, "How Contextual are
Contextualized Word Representations?"].** The paper's headline finding: contextualized
representations "are not isotropic in any layer" of BERT, ELMo, or GPT-2 — the embedding space is
directionally biased, not uniformly distributed, in every layer of every model tested. A second
finding sharpens why this matters for pooling specifically: **less than 5% of the variance in a
word's contextualized representation is explained by a static embedding**, meaning the
context-specific component — exactly the information a mean over many tokens would need to preserve
to be useful — dominates the representation's variance, and averaging is precisely the operation
that suppresses variance along non-dominant directions in favor of whatever direction is shared
across the averaged set.

### 2.2 The mechanism, traced to the pretraining objective — and a direct quantitative anchor

**[primary, fetched `ar5iv.labs.arxiv.org/html/2011.05864`, Li et al. 2020, "On the Sentence
Embeddings from Pre-trained Language Models" — the BERT-flow paper].** This is the most directly
load-bearing citation in this section: it does not merely observe anisotropy, it derives it from the
masked-LM objective's own decomposition, `PMI(x,c) + log p(x) + λc`, and shows the `log p(x)` term
biases embedding *position* by word frequency — **high-frequency words cluster near the origin
(mean ℓ₂-norm 0.95) while low-frequency words scatter sparsely far away (mean ℓ₂-norm 1.45)**, with
low-frequency words 1.30–1.37 mean distance to their 3–7 nearest neighbors versus 0.77–0.87 for
frequent words — i.e., the space has dense, high-frequency regions and sparse "holes" where rare,
more semantically specific content lives. Because **mean-pooling is a convexity-preserving
operation**, a sentence embedding built by averaging inherits this frequency bias directly: it is
pulled toward whatever the token stream's frequency-weighted centroid happens to be, not toward the
specific meaning any individual rare, information-bearing token carried. Quantitatively, on STS-B:
**average BERT embeddings score 59.04 Spearman** (interestingly, well above raw BERT's CLS vector at
16.50 — pooling is not uniformly the worst option among untrained representations) **against
BERT-flow's 70.72** after an explicit isotropy-restoring transform — the gap between 59.04 and 70.72
is specifically the cost of not correcting the pooled representation's anisotropy, not the cost of
pooling per se versus not pooling at all.

### 2.3 The SBERT counter-case — pooling is not intrinsically the losing move, uncorrected geometry is

**[primary, fetched `ar5iv.labs.arxiv.org/html/1908.10084` for the ablation table, Reimers &
Gurevych 2019].** Table 6's ablation, trained end-to-end with a contrastive/regression objective
rather than used off-the-shelf: **MEAN pooling wins outright** — 87.44 (STS regression objective)
vs CLS 86.62 vs MAX 69.92; under the NLI classification objective the gap narrows (80.78 vs 79.80 vs
79.07) but MEAN still wins. **This is a genuine, disclosed tension this section does not resolve by
picking a side**: mean-pooling is the *worst-performing* untrained representation choice's neighbor
in one regime (Li et al.'s off-the-shelf BERT, where it still beats CLS but trails an
isotropy-corrected alternative by 11.7 points) and the *best*-performing choice in another (SBERT's
fine-tuned regime, where the pooling operator is trained jointly with an objective that shapes the
pre-pooling representations to survive averaging). **The operative variable is not "pool or don't
pool" in isolation — it is whether the representation being pooled was produced (or fine-tuned) with
awareness that it will be pooled.** This is directly relevant to reading M3's own reconstruction-MSE
training: M3's adapter is trained to match the receiver's own **pooled** target (per ADR-024's own
per-token-vs-pooled variant framing), so its landing "on the pooled manifold" is a case where the
representation *was* shaped with pooling in the loop — and it still nulled against the frozen probe,
which is itself evidence that fixing the geometry of the pooled target (what M3/M4's on-manifold
success achieves) is a different, and apparently insufficient, fix from fixing pooling itself.

### 2.4 Reading `docs/research/036`'s own numbers against this literature

`docs/research/036`'s measured facts — pooled-vs-real cosine **0.667**, entropy **9.30 vs 3.36**
nats, cross-item invariance **0.962 vs 0.635** — map onto this literature with unusual precision,
and in one respect sharpen it beyond what the classical sentence-embedding papers directly measured:

- **Cross-item invariance 0.962 (pooled) vs 0.635 (real states) is exactly Li et al.'s "collapse
  toward a shared direction," reproduced at reasoning-trace scale rather than word/sentence scale.**
  Forty different GSM8K items' pooled block-14 states are, on average, nearly the same direction
  (0.962 mean pairwise cosine) — the anisotropy/frequency-clustering mechanism operating not over
  individual words within one sentence, but over an entire ~281-token generated chain-of-thought,
  averaged into one point that ends up looking like every other item's average.
- **Entropy 9.30 (pooled) vs 3.36 (real, single-state) nats is a measurement the classical
  sentence-embedding literature mostly does not report directly** (STS/classification benchmarks
  measure downstream task correlation, not the induced-distribution entropy through a real
  unembedding matrix) — this is `docs/research/036`'s own methodological contribution, and it gives
  the anisotropy story a second, independent signature: not just "pooled states look alike across
  items" but "a pooled state, read through the receiver's own readout, commits to almost nothing" —
  entropy 9.30 nats against a ~11.93-nat uniform ceiling (cited in ADR-024's DIAGNOSIS section for
  M4c) means the pooled state induces a distribution close to maximally uncertain, exactly the
  behavior expected of a centroid sitting in the "holes" between committed token identities Li et
  al. describe, rather than at any one of them.
- **Cosine 0.667 is neither "the same object, slightly perturbed" nor "unrelated"** — it is the
  quantitative gap between a real state and its own item's centroid, and it is large enough
  (comparable in magnitude to the gap the ladder treats as diagnostic of collapse elsewhere: M4c's
  own item-invariance of 0.881 was flagged in `docs/research/036` §3 as anomalously *low* relative
  to on-manifold adapters' 0.96–0.98) to be a first-order effect, not measurement noise.

**No literature source found in this pass measures pooling's effect specifically on a *generated
reasoning trace* (as opposed to a single sentence or document) — this is named as a genuine gap.**
Every source in §2.1–2.3 pools short, single-utterance spans (sentences, sentence pairs). A
~281-token, multi-step chain-of-thought spans multiple distinct semantic phases (problem restatement,
intermediate arithmetic, a final answer line) that a sentence never does; whether pooling's damage
scales with the number of distinct "topics" or "steps" averaged together — which would predict
CoT-pooling is *worse* than sentence-pooling, not merely analogous to it — is a plausible extension
of this literature, not something any fetched source states directly. `docs/research/036`'s own
entropy gap (9.30 vs 3.36, a much larger gap in nats than the STS-correlation gaps §2.2/2.3 report in
their own units) is consistent with, but does not by itself prove, this extension.

---

## 3. Alternatives, ranked by cost against this stack

All four options below are assessed against concrete, source-verified facts about this repo's own
implementation, not in the abstract.

### 3.0 What the current injection mechanism already supports (source-verified, not assumed)

**[primary, read directly from `crates/latentmesh-runtime/src/models/qwen2_b.rs` and
`crates/latentmesh-runtime/src/inject.rs` this pass].** `LayerEdit::Inject` and `LayerEdit::Fuse`
(the two low-level operators every rung's forward pass ultimately calls) both take
`positions: &[usize]` and `vectors: &Tensor` of shape `(positions.len(), hidden)`, and assign each
position its **own row** — `vectors.narrow(0, row, 1)` per position, in a loop. **The core
injection mechanism already supports distinct per-slot vectors; it was built this way from the
start.** The only place pooling is structurally forced is one layer up, in `InjectionSpec`'s
`vectors_tensor()` helper (`crates/latentmesh-runtime/src/inject.rs`), which takes a single
`vector: Vec<f32>` field and **broadcasts it via `std::iter::repeat(v).take(n).flatten()`** — i.e.,
every rung to date (M3 through M4g) has fed the same one pooled vector into all 8 rows by
construction of this convenience wrapper, not because the underlying `LayerEdit` forward-pass code
requires it. **A de-pooling rung that fills 8 slots with 8 distinct vectors requires no change to
the model's forward pass at all** — only a new construction path that builds an `(8, hidden)` tensor
with genuinely different rows (bypassing or extending `InjectionSpec`), which is a small, contained
change against code that has not been touched by any prior rung. This is the single most important
implementability fact in this section: the ladder's own mechanism was never the barrier to
de-pooling — the convenience layer built on top of it was.

**A second load-bearing fact, also source-verified**: `crates/latentmesh-train/src/dataset.rs`'s
`LayerMap::row(t)` accessor already exposes the per-token capture at row granularity — this is the
same accessor `docs/research/038`'s bank-attention design (§2a there) already assumed would exist
for M4f. **No new data capture is needed for any variant below**; the 719,115-token per-token dumps
captured at M2 (per ADR-024) have been sitting fully available since M2, and every rung from M3
onward has used them only to fit a translator whose output is then thrown away down to one vector
immediately before injection (M4's FastGRNN is a genuine per-token *sequence* model internally, but
per ADR-024's own M3/M4 framing, its output still feeds "the existing pooled-injection path" — even
the ladder's one sequence-aware rung pools at the injection boundary).

### 3.1 (a) Sequence injection — a matching number of per-token-derived vectors into a matching number of slots

**Recommended primary mechanism.** Two sub-designs, both keeping slot count at exactly 8 (see §
probe-compatibility below for why this matters):

- **8-distinct-slot, attention-compressed** (top recommendation): a small learned query mechanism
  (8 learned queries, or an 8-head cross-attention) maps the sender's `T_i × 2048` per-token stream
  down to 8 distinct `1536`-dim outputs, trained end-to-end under task loss (identical objective to
  M4c/M4d/M4g). This is architecturally the M4f bank-attention mechanism (`docs/research/038` §2a)
  **redirected**: M4f's design compresses a query *against a frozen bank of real receiver states* to
  guarantee manifold membership; this variant compresses the *sender's own T_i states* down to 8
  outputs to guarantee content diversity across slots. The two are not mutually exclusive — a
  combined design (8 queries, each attending over a receiver-state bank *conditioned on* the
  sender's local context at that query's position) would jointly address the manifold-membership
  question M4f targets and the pooling-diversity question this document targets, at once, and is
  named here as a natural follow-on if this rung's simpler version nulls.
- **8-strided-slot** (cheaper fallback): sample 8 evenly-spaced positions from the generated span
  (or the 8 positions a small learned projector, trained per-token exactly as M3's per-token variant
  already is, is applied to) and inject those 8 translated per-token states directly, no attention
  module at all — closer to a direct extension of M3's *already-trained* per-token MLP (reuse its
  weights unchanged, per ADR-028's "one champion per rung" framing does not obviously forbid reusing
  an already-frozen artifact for a new injection-shape experiment, though this should be confirmed
  with whoever owns rung promotion before assuming it).

**Cost**: by direct analogy to M4c's receipted **1,604 s (≈0.446 GPU-h)** and M4f's own estimate of
**~0.5–0.7 GPU-h** for an architecturally comparable attention module (`docs/research/038` §5) — same
frozen 1.5B receiver forward/backward, same `SEQ_CAP=256`, same batch=1, same epoch count; the
8-query attention module's own FLOPs are negligible next to the receiver's. **Inferred by analogy,
not measured** — a 1-epoch pilot, as M4f's own spec already recommends for its own mechanism, is the
correct first move before committing the full budget.

**Data**: the existing per-token dumps, no new capture.

### 3.2 (b) Last-token-only transfer — the cheapest possible test

Inject the generated span's **final** per-token state (the mechanism `docs/research/036`'s own
"single-row" reference already used as its un-pooled comparison point: entropy 3.36 nats, cross-item
invariance 0.635) instead of the mean. This requires **no new training run at all** for the cheapest
version: M3's already-trained, already-frozen per-token MLP (its weights are already fit; its
architecture already processes per-token inputs) can have its **inference-time consumption** changed
from `mean(translated_rows)` to `translated_rows[-1]`, broadcast to the existing 8 slots exactly as
every prior rung has done. This is a pure pipeline change, zero new GPU-hours for training, one new
frozen-probe draw. Because M3's per-token variant already has a committed, hash-verified artifact
(`run2-m3-receipt-cellL18toL14-mlp-pertoken-*.json`), this pilot could plausibly run **before** any
new training-shaped rung, at near-zero marginal cost, as the single sharpest, cheapest available test
of "does the pooling operation itself matter, independent of everything else" — isolating pooling
from every other axis (loss, architecture, injection operator) more cleanly than any other option
in this section, because literally nothing else changes.

**Caveat, disclosed**: a single last-token state is a much narrower bottleneck than 8 slots' worth of
content — this variant tests "does moving off the pooled centroid help at all," not "does full
sequence-length transfer help," and a null result here would not rule out (a) above.

### 3.3 (c) Attention-based compression to k>1 — same as (a)'s primary mechanism, listed separately per the brief's framing

Already covered as the primary design in §3.1. Restated here only to note explicitly: this is **not**
a separate cost tier from (a) — it *is* (a)'s recommended sub-design. Ranked jointly with (a).

### 3.4 (d) Top-k selection by an informativeness criterion — ranked lowest, echoing M4f's own reasoning

A hard, discrete selection of the k most informative token positions (by some scoring rule) is
structurally the VQ/nearest-neighbor case `docs/research/038` §2d already ranked last for M4f, for
the identical reason: it needs a straight-through-estimator-style gradient path, which
`candle-nn` 0.9.2 does not ship (confirmed by the same source that established candle 0.9.2's
`AdamW`-only, no-scheduler constraint — `docs/adr/024`'s M0 training-infra scout). Implementable, but
a materially larger lift than (a)/(b) for a benefit (discrete, maximally interpretable content per
slot) that (a)'s continuous attention-compression already captures without the differentiability
problem. Ranked last on implementability, same as M4f's own analogous mechanism.

### 3.5 Probe-compatibility — the critical question, with a disclosed tension in the governing ADR

**ADR-028 names both "slot count" and "pooling scheme (per-token vs. pooled, and any pooling variant
in between)" as explicitly EVOLVABLE surfaces** — the same list that names architecture family,
injection depth/site, and loss shape as evolvable, i.e., exactly the axis every M3-through-M4g rung
has already varied under the unchanged frozen probe. **But the same ADR's PROTECTED list also names
"slot count" as part of "the frozen S1a/S2b probe protocol itself,"** alongside the 40 items, the
exact sign test, α=0.05, the rescale-to-median switch, and greedy/batch=1 decoding.

**This is a genuine, source-verified internal tension in ADR-028, not something this document
resolves by fiat.** Two readings are both defensible from the text: (1) "slot count" in the evolvable
list refers to an architectural design parameter a search process may propose, while "slot count" in
the protected list refers to the *specific frozen value* (8) that the S1a/S2b receipt already pins,
meaning a fixed choice of exactly 8 is permanently locked in and any change requires new
pre-registration comparable to M4e's; or (2) the protected-list mention is a drafting inconsistency
inherited from copying the probe's own configuration description wholesale, and the evolvable list's
explicit, separately-stated inclusion of "slot count" is the operative rule. **Recommendation: sidestep
the ambiguity rather than adjudicate it.** Every design in §3.1–3.2 above keeps slot count fixed at
exactly 8 — filling those 8 slots with distinct content is a change to what each slot *carries*
(squarely analogous to M4g's overwrite-vs-fuse change to what happens *at* each slot, which needed
its own registration but did not touch slot count, item count, or any other pinned protocol number),
not a change to *how many* slots exist. Under either reading of the tension above, **an 8-slot,
8-distinct-vector design does not touch the number ADR-028's protected list actually pins**, so it
should qualify as an evolvable architecture change requiring the same lightweight registration M4g
got, not a full ADR-030-style protocol redesign. **A future variant that wants MORE than 8 slots**
(e.g., one per generated token, ~281 on average) is a different case: it would change the S1a/S2b
receipt's own pinned "slots8" identity directly, and this document recommends that be named
explicitly as requiring new pre-registration, exactly as M4e's continuous-injection change already
is, rather than assumed to be covered by the evolvable list's "slot count" line without discussion.

**Whoever schedules a de-pooling rung should flag this tension to the ADR's own maintainers before
running it**, not silently pick a reading — this document surfaces the ambiguity as its own finding,
consistent with this ladder's stated discipline of recording coordinator errors and text tensions
rather than quietly working around them (see ADR-024's own "CORRECTION to the M4d registration").

---

## 4. Is pooling upstream of the ladder's other hypotheses?

Going through each live hypothesis from `docs/research/036` and ADR-024's DIAGNOSIS section:

- **Overwrite-vs-fuse (M4g, running).** **Orthogonal, not subsumed either direction.** M4g's own
  registered spec explicitly "retrain[s] the M3-shaped adapter" — M3's architecture pools by
  construction, so M4g tests whether the *write operator* (overwrite vs. residual-add) matters while
  leaving the pooled-payload question completely untouched. A pooled payload fused instead of
  overwritten is still a pooled payload; conversely, a de-pooled sequence injected via a hard
  overwrite is still de-pooled. The two axes compose independently — fixing one does not predict
  anything about the other, and both would need to be right simultaneously for a design that
  benefits from both fixes, which is exactly the reasoning M4g's own pre-registration used to justify
  running before the (then-unscoped) M4f: M4g addresses a mechanism verified to differ from what
  works (the write operator); this document's finding addresses a second, independently-verified
  mechanism (the payload shape) that also differs from what works. Neither implies the other.

- **Off-manifold collapse / manifold-membership framing (M4c/M4d, closed; M4f, registered but
  unscheduled).** **Pooling is upstream of this hypothesis as originally scoped**, and this is
  already conceded, in different words, by the ladder's own record: `docs/research/036` §5 states
  plainly that "M4f as sketched... would move M4c toward where M3 already is, and M3 is null" and
  that "M4f should be re-scoped before it is scheduled: manifold membership is necessary-looking but
  demonstrably not sufficient, and the pooled-vs-real-state gap... is the more promising target."
  `docs/research/038`'s own M4f spec (§5) already redirects its bank-attention mechanism toward
  breaking item-invariance, precisely because on-manifold-and-item-invariant (what pure manifold
  anchoring would produce) is the exact failure signature M3/M4 already demonstrated is insufficient.
  **This document's contribution is naming explicitly what was previously implicit: "on-manifold"
  was never measured against the receiver's real state distribution — it was measured against the
  pooled reference the whole time**, so every reconstruction-trained adapter's success at reaching
  cosine 0.97–0.99 to that reference is success at reaching a target that is itself only 0.667-cosine
  correlated with what the receiver actually carries. Fixing pooling does not merely *interact* with
  the manifold-membership question — it **redefines what "on-manifold" should mean**, which is a
  stronger claim than "these two hypotheses are related." An M4f-style bank-attention mechanism
  pointed at a bank of **real, per-token, un-pooled** receiver states (which `docs/research/038`'s
  own design already specifies — `receiver_L14.tok.f32bin`, per-token, not the pooled S2c dump) would
  automatically inherit a de-pooled target, meaning **a correctly-scoped M4f and this document's §3.1
  primary recommendation may already be close to the same experiment**, differing mainly in whether
  the sender's own query is a single pooled vector (M4f as currently spec'd) or 8 distinct per-slot
  queries (this document's proposal) feeding that same real-state bank. This convergence is named
  explicitly as a synthesis opportunity, not asserted as already-resolved: M4f's spec (§5,
  `docs/research/038`) still describes a single query producing a single output, mean-pooled over
  the generated span exactly like every prior rung, so the two documents' recommendations are
  adjacent, not identical, and reconciling them is a design decision for whoever schedules the next
  rung, not something this document decides unilaterally.

- **Receiver scale (M4b, mandatory, unaffected).** **Orthogonal, no interaction predicted.**
  Receiver scale governs whether the receiver has the capacity to *use* correctly-delivered content;
  pooling governs whether content is delivered correctly at all. A sub-threshold receiver could fail
  on both a pooled and a de-pooled payload for capacity reasons unrelated to payload shape; a
  larger receiver could still fail on a badly-pooled payload for the reasons in §2. No source or
  argument in this pass predicts these two axes interact, and M4b remains necessary regardless of
  what a de-pooling rung reports.

- **One-shot-vs-continuous injection (M4e, registered, unscheduled).** **Substantially entangled —
  the closest relationship among all four hypotheses.** Every method in §1 that avoids pooling also
  avoids one-shot static delivery: C2C's per-token cache persists through the receiver's own KV cache
  exactly as a one-shot injection would (this nuance is already correctly separated in
  `docs/research/039` §1.3 — C2C is one-shot-at-prefill *and* non-pooled, showing the two axes are
  logically separable), but LatentMAS's full-sequence prepend and Bicameral's per-step live coupling
  both deliver de-pooled content *because* they deliver it continuously — a per-step design has no
  natural single moment at which "pool everything so far into one vector" would even make sense.
  **The cleanest realization of both a de-pooled payload and continuous delivery is the same
  design**: feeding the receiver's own decode steps a stream of per-token translated sender states
  (via repeated KV-cache prepend/fuse, LatentMAS-style) rather than a fixed few slots filled once.
  This document's §3.1 8-slot design is a **partial, cheaper step toward that same direction**
  (de-pooled, but still one-shot) — a natural escalation path exists from "8 distinct slots, injected
  once" (this document) to "one slot per receiver decode step, injected continuously" (M4e, done
  correctly) that shares almost all of its engineering with a well-scoped M4e. **A combined
  de-pooling + continuous rung would test two of the ladder's three live hypotheses in one
  architecturally coherent step** — which ADR-024's own one-factor-per-rung discipline would require
  explicitly justifying, the same way M4g's coordinator explicitly justified testing "both null
  families" at once. This document names the possibility rather than recommending it as the next
  rung (§5's recommendation stays incremental, per that same discipline).

**Summary**: pooling is upstream of the manifold-membership framing (redefines what "on-manifold"
means), orthogonal to overwrite-vs-fuse and receiver scale (independent axes, no predicted
interaction), and closely entangled with one-shot-vs-continuous injection (the field's working
methods that avoid pooling also avoid one-shot delivery, for what looks like a structural, not
coincidental, reason).

---

## 5. Recommendation

**Yes — the next rung after M4g should be a de-pooling rung, run in two stages given the enormous
cost asymmetry between them.**

**Stage 1 (near-zero cost, could run immediately, before or alongside M4g's own report lands):**
swap M3's already-trained per-token MLP's inference-time consumption from mean-pool to last-token,
broadcast unchanged to the existing 8 slots. Zero new training, zero new capture, one new frozen-probe
draw, cleanest possible isolation of the pooling variable from every other axis this ladder has
touched. Given `docs/research/036`'s own entropy numbers (3.36 vs 9.30 nats — the sharpest, cheapest
signal this whole investigation surfaced), this is the highest information-per-GPU-hour experiment
available to the ladder right now, full stop.

**Stage 2 (primary rung, ~0.5–0.7 GPU-h by receipted analogy to M4c/M4f):** 8-distinct-slot sequence
injection via a small attention-compression module over the sender's own per-token stream — either
standalone, or, if Stage 1's result and the coordinator's judgment support it, merged with
`docs/research/038`'s M4f bank-attention design so the same rung compresses to 8 distinct queries
*and* anchors each to the real (per-token, un-pooled) receiver-state bank M4f already specifies —
addressing this document's finding and the manifold-membership question in the same architecturally
coherent step, per §4's synthesis. Trained under the same task loss as M4c/M4d/M4g, same fuse-or-
overwrite choice as whichever M4g reports (fuse if M4g passes; overwrite if it nulls, to keep
one-factor-per-rung discipline against M4g's own live result). No new per-token capture needed — the
719,115-token dumps captured at M2 have been available and unused at full granularity since before
M3's first training run.

**Probe-compatibility verdict**: **yes, the unchanged frozen 40-item probe survives, provided slot
count is held fixed at exactly 8** — both stages above satisfy this by construction. The one thing
this document does *not* resolve, and flags explicitly rather than silently deciding: ADR-028
contains a genuine internal tension between listing "slot count" as evolvable and as protected in the
same document, and whoever schedules a de-pooling rung should surface that to the ADR's maintainers
before running it, exactly as this ladder has recorded its own coordinator errors elsewhere rather
than quietly working around them.

---

## Sources

- Cache-to-Cache: [arxiv.org/abs/2510.03215](https://arxiv.org/abs/2510.03215),
  [full text](https://arxiv.org/html/2510.03215) (**primary**, fetched this pass and re-confirmed
  against `docs/research/038`'s prior primary read)
- LatentMAS: [arxiv.org/abs/2511.20639](https://arxiv.org/abs/2511.20639),
  [full text](https://arxiv.org/html/2511.20639v1) (**primary**, fetched this pass)
- The Bicameral Model: [arxiv.org/abs/2605.11167](https://arxiv.org/abs/2605.11167) — already
  primary-verified in `docs/research/039` §1.1, re-cited not re-fetched this pass
- AVP (Agent Vector Protocol): [github.com/VectorArc/avp-spec](https://github.com/VectorArc/avp-spec),
  [SPECIFICATION.md](https://raw.githubusercontent.com/VectorArc/avp-spec/main/SPECIFICATION.md)
  (**primary**, fetched this pass; unverified against any independent benchmark, per
  `docs/research/035`)
- MANTA: [arxiv.org/abs/2607.28527](https://arxiv.org/abs/2607.28527) (**primary**, abstract fetched
  this pass — scoping correction, not a peer comparison)
- Ethayarajh (2019), "How Contextual are Contextualized Word Representations?":
  [arxiv.org/abs/1909.00512](https://arxiv.org/abs/1909.00512) (**primary**, fetched this pass)
- Li et al. (2020), "On the Sentence Embeddings from Pre-trained Language Models" (BERT-flow):
  [arxiv.org/abs/2011.05864](https://arxiv.org/abs/2011.05864),
  [full text via ar5iv](https://ar5iv.labs.arxiv.org/html/2011.05864) (**primary**, fetched this pass)
- Reimers & Gurevych (2019), Sentence-BERT:
  [arxiv.org/abs/1908.10084](https://arxiv.org/abs/1908.10084),
  [full text via ar5iv](https://ar5iv.labs.arxiv.org/html/1908.10084) (**primary**, fetched this pass
  for the Table 6 pooling ablation)
- Internal: [docs/research/036](036-manifold-collapse-across-the-ladder.md) (the pooling finding this
  document extends), [docs/research/038](038-manifold-constrained-adapter-scout.md) (M4f
  bank-attention mechanism, C2C fuser verified from source), [docs/research/039](039-bidirectional-latent-exchange.md)
  (Bicameral/LatentMAS/C2C mechanism corrections), [ADR-024](../adr/024-run2-trained-thought-adapter-ladder.md)
  (M4g spec, M4c GPU-hour receipt reference), [ADR-028](../adr/028-evolutionary-adapter-search-anti-gaming.md)
  (evolvable/protected lists — the internal tension flagged in §3.5), source code read directly:
  `crates/latentmesh-runtime/src/models/qwen2_b.rs` (`LayerEdit::Inject`/`Fuse`),
  `crates/latentmesh-runtime/src/inject.rs` (`InjectionSpec::vectors_tensor`),
  `crates/latentmesh-train/src/dataset.rs` (`LayerMap::row`)

**Not written to any other file. Not committed — per this task's read-only constraint, only this
file was created.**
