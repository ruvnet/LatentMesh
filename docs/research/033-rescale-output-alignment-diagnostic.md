# Research: does the deployment rescale destroy M4c's output-token alignment?

* Purpose: execute ADR-024's **registered zero-GPU diagnostic** ("Registered zero-GPU diagnostic
  (protocol-safe, no probe draw)") — project M4c's raw and rescaled adapter vectors through the
  receiver's unembedding matrix and compare output-token alignment against the sender-span and
  gold-answer tokens. The leading M4d hypothesis is that
  `rescale_to_natural_median` is what breaks the trained adapter at deploy time; this resolves
  that *today*, off existing receipts.
* Date: 2026-08-28.
* Method / evidence label: **deterministic CPU analysis over committed artifacts — no probe draw,
  annotates only**. No live-model forward, no GPU (ADR-034 lane rule: the GPU is held by M4d). The
  only model-derived object touched is the receiver's unembedding matrix, loaded CPU-side from the
  HF cache. No probe item, control, or statistic on ADR-028's protected list is touched, and no
  recorded outcome changes.
* Artifacts: receipt
  `crates/latentmesh-runtime/receipts/run2-rescale-diagnostic-receipt.json`; code
  `crates/latentmesh-runtime/examples/run2_rescale_diagnostic.rs` (CPU-only, default features).
* Method source: `docs/research/032-injection-configuration-science.md` §4 / §5.1 — the LAP
  `A_lin` logit-lens check (arXiv:2604.15557), which §5 ranks first by expected value per cost.

---

## Answer

**No. Rescaling is harmless to output-token alignment, and it cannot be the mechanism behind
M4c's 0/40 reversal.** Top-k token overlap between the raw and rescaled vectors is **1.000** at
k = 10, 50 and 100 on every one of the 40 items in both lenses, the argmax is identical on every
item, the gold-answer token ranks are bit-unchanged, and the logit cosine is **1.0 to within
2 × 10⁻¹³**. Only the softmax *temperature* moves — and the receiver's actual readout applies a
scale-invariant RMSNorm before the unembedding, which removes even that.

This **redirects the M4d diagnosis away from the norm-mismatch hypothesis** toward the remaining
candidates ADR-024 named alongside it (one-shot-vs-continuous / greedy-decode context, 8-slot
placement) — plus one new candidate this diagnostic surfaced on its own (§4).

---

## 1. Why the answer is forced, and why it still had to be measured

Two facts about the frozen probe make the result analytic rather than empirical:

1. **The rescale is a positive scalar.** `examples/common/m3.rs::four_conditions` builds
   `InjectionSpec { vector: v, scale: Some(natural.median / ‖v‖) }`, and
   `InjectionSpec::effective_vector()` returns `v · c` with `c = median/‖v‖ > 0`.
2. **The injection overwrites, it does not add.** `crates/latentmesh-runtime/src/inject.rs` —
   "the residual rows at those positions, after `after_block` blocks, are **overwritten** with the
   (optionally rescaled) payload vector". So the residual row *is* `c·v`; there is no `h + c·v`
   term whose balance `c` could tilt.

Under any unembedding `W_U`, `W_U(c·v) = c·(W_U v)`: a positive scalar is exactly order-preserving.
Every rank-based `A_lin`-style statistic — top-k, argmax, token rank, cosine — is invariant by
construction. That is a proof, not a measurement, but it is worth *having measured* for three
reasons: it pins the claim to the probe's own code path rather than to a reading of it; it
quantifies the one thing that genuinely does change (entropy); and running the projection at all
produced the far more interesting §4 finding, which no amount of algebra would have handed over.

## 2. What was run

| | |
|---|---|
| Adapter | `run2-m4c-mlp-taskloss-cellL18toL14.f32bin`, content hash `fe4de15b…c15141`, **verified equal** to the M4c training receipt's frozen hash; hand-rolled apply re-verified against the 8 trained-network golden pairs (max relative L2 `1.89e-7` ≤ `1e-5`) |
| Rescale operator | **the probe's own function is called directly** — `latentmesh_runtime::inject::InjectionSpec::effective_vector`, constructed exactly as `four_conditions` constructs it. Additionally pinned against an independent recomputation on 8 seeded vectors across four magnitudes: max relative L2 error **0.0**, max relative norm error `4.0e-7` (tolerance `1e-6`) |
| Unembedding | Qwen2.5-1.5B-Instruct `model.embed_tokens.weight`, `[151936 × 1536]`, loaded F32 on CPU from the HF cache. `config.json` `tie_word_embeddings: true` is asserted, so this *is* the `lm_head`. `model.norm.weight` and `rms_norm_eps = 1e-6` loaded for the second lens |
| Device | `Device::Cpu`; the run **refuses to start** if the `cuda` feature is enabled |
| Items | 40 |

### Which items, and why not the probe's own 40

**The frozen probe's 40 items are not reconstructible from committed artifacts.** Their sender
captures are produced live inside `run2_m4c_probe` and never dumped; the M4c probe receipt records
only their *norms* (`pooled_l2_raw`, `aligned_l2_raw`, `natural_inject_block_norms`). Rebuilding
their raw adapter outputs needs a live sender forward — i.e. GPU, which this lane forbids.

The documented substitute: **the first 40 rows of the M4c training receipt's holdout split**
(`split.holdout_rows`, in receipt order). Same capture pipeline (sender L18 per-token states over
the sender's own greedy generated span, `run2-pertoken-dump-receipt.json`, sha256-verified before
use), same adapter input distribution, and held out of M4c's own fit.

The *deployment statistics* are the real ones: each sample slot is rescaled to the k-th natural
median **recorded by the M4c probe itself**. Those medians are tight (mean 52.45, range
51.36–53.28), and the scale factors deployment actually applied are

```
c = natural_median / ‖adapter output‖  ∈  [0.259, 0.579],  mean 0.386
```

— i.e. the probe's rescale **shrinks** the M4c vector by 1.7×–3.9×. A scale sweep at both
empirical extremes is included in the receipt: top-100 overlap 1.0 and cosine ≥ 0.9999999999998
against the raw vector at both ends.

## 3. Results

Two lenses are reported because they answer different questions. `plain` is `W_U·h`, the bare
logit lens `docs/research/032` names. `rmsnorm` is `W_U·RMSNorm(h)`, the receiver's *actual*
readout.

### (a), (e) Top-k overlap and logit cosine — invariant

| | plain lens | rmsnorm lens |
|---|---|---|
| top-10 / 50 / 100 overlap | **1.000** (min 1.000) | **1.000** (min 1.000) |
| argmax identical | 40/40 | 40/40 |
| cosine(raw logits, rescaled logits) | **0.99999999999987** (min 0.99999999999981) | same to 1e-13 |

### (b), (c) Gold-answer and sender-span token ranks — unchanged

Mean rank over the token set, out of a 151,936-token vocabulary:

| lens | token set | raw | rescaled | items changed |
|---|---|---|---|---|
| plain | gold `#### <gold>` | 80,558 | 80,558 | **0 / 40** |
| plain | sender span (unique) | 71,892 | 71,892 | 11 / 40, max Δmean-rank **0.029** |
| rmsnorm | gold | 93,460 | 93,460 | 3 / 40, max Δmean-rank **0.33** |
| rmsnorm | sender span | 81,154 | 81,154 | 16 / 40, max Δmean-rank **0.033** |

**Disclosed honestly:** the nonzero "items changed" counts are float rounding, not a rescale
effect. The two arms are computed through *independent* f32 matmuls (`W_U·(c·v)`, not
`c·(W_U·v)`), and the rmsnorm arm recomputes an f32 reciprocal square root per arm. A handful of
deep-tail tokens therefore move by ≤1 rank out of 151,936 — a mean-rank delta of 0.033 over a
~1,500-token span set is roughly *one* token moving *one* place. The top-k and argmax statistics,
which are what "alignment" means, are identical on every item.

### (d) Entropy — the only thing that moves, and it does not survive the real readout

| lens | entropy raw | entropy rescaled | ratio |
|---|---|---|---|
| plain | 6.021 nats | 11.062 nats | mean **2.00×** (range 1.26–4.06) |
| rmsnorm | 5.943 nats | **5.943 nats** | **1.00×** |

Under the bare logit lens, shrinking the vector by ~2.6× flattens the induced distribution toward
uniform (`ln 151936 = 11.93`). This is a pure temperature effect and it changes no ordering. Under
the receiver's real readout it is gone entirely, because RMSNorm divides out exactly the scalar
the rescale multiplied in. **Whatever the rescale does to the receiver, it does not do it through
the output projection.**

## 4. The finding that was not the question: `A_lin` is ~0 for *both* arms

The diagnostic also computes what `docs/research/032` §4 asked for in its own right — the injected
vector's unembedding-projected overlap with the gold-answer tokens **separately** from its overlap
with the sender's own generated tokens. The result is stark, and it holds identically for raw and
rescaled:

* **Gold-answer tokens sit at mean rank 93,460 / 151,936 — the 61st percentile of the vocabulary.**
  Worse than the middle. Best-ranked gold token per item: median 45,999 (min 3,897, max 109,866).
* **Sender-span tokens sit at mean rank 81,154 — the 53rd percentile.** Marginally better than
  gold, and the best single span token per item does reach median rank 2,554 — but that is the
  best of ~1,500 tokens, i.e. nothing.

LAP's reported thresholds are `peak A_lin > 0.1 → steering viable`, `< 0.05 → negligible effect
regardless of intervention strength`. The M4c vector is deep in the second regime **for both
targets**. It is not that the adapter learned the sender's tokens instead of the gold answer (the
"target mismatch" reading M4c's own receipt proposed narratively) — it learned neither, in the
output-aligned sense.

And yet the induced distribution is *confident*: entropy 5.94 nats against a uniform 11.93, a
perplexity of ~380 over 152k tokens. So the vector points somewhere, sharply. Decoding the top-10
tokens under the receiver's real readout shows where:

```
item 945:  ' Svens'  '#ab'  '/testify'  ' Lanc'  'rias'  'يان'  '上海证券'  ' Stake'  '.scalablytyped'  'rng'
item 3866: ' Lanc'  'rias'  ' Svens'  '上海证券'  '农历'  'stile'  ' concession'  'lider'  'DirectoryName'  ' Caval'
item 154:  'DirectoryName'  ' Svens'  '上海证券'  '代码'  ' Stake'  'antd'  'endregion'  'rias'  ' concession'  'UY'
```

Across all 40 items the top-10 sets draw from only **77 distinct tokens**, and the same handful
dominates: `DirectoryName` (37/40 items), `rias` (35), ` Svens` (34), ` Lanc` (34), ` concession`
(24), `上海证券` (21). These are rare, low-frequency embedding outliers — the classic signature of
a vector that is **off the receiver's residual-stream manifold**, whose unembedding projection is
therefore dominated by outlier embedding rows rather than by content.

The direction is also **nearly item-invariant**. Whatever the adapter emits, it emits substantially
the same off-manifold direction for every item, at a magnitude the probe then normalises to look
natural.

That is a much better explanation of M4c's recorded result than the rescale ever was. The
ablation-methodology literature `docs/research/032` §4 cites is explicit that losing to *both* the
zero vector and a norm-matched random control, unanimously and 0/40, is the signature of an
**actively counterproductive** intervention rather than an ignored one — and "a fixed off-manifold
direction, norm-matched into the residual stream at block 14 on every item" is exactly a
mechanism that would be actively counterproductive rather than ignored.

The magnitudes in M4c's own receipt corroborate this. Mean teacher-forced NLL of `#### <gold>`:
**aligned 5.359** against baseline 2.129, zero-vector 2.154, random 2.117. The aligned condition
is not marginally worse than the controls — it is **2.5× worse**, and worse on all 40 items
against both. An ignored injection lands on the controls; a mild misalignment lands near them.
A 2.5× NLL blow-up, unanimous, with a near-item-invariant off-manifold direction behind it, is one
coherent story.

## 5. What this does and does not license

**Discharged.** The norm-mismatch / output-alignment hypothesis for M4d. Rescaling to the natural
median cannot destroy output-token alignment, because it cannot change any ordering at all. Note
that `train_m4d_deploymatch.rs` had already recorded (in its `adr_premise_correction` field) that
M4c's training loop *did* contain the rescale, contrary to ADR-024's M4d registration text; this
diagnostic independently establishes the stronger statement — that even if it had not, it would
not have mattered for output alignment.

**Still open, and now the leading candidates.** ADR-024's secondary M4d candidates are untouched
by this result: the **one-shot-vs-continuous axis** (teacher-forced training context vs greedy
decode at probe time) and the **8-slot placement**. To those this adds a third:

**The off-manifold direction itself.** Two mechanisms remain in which the *magnitude* could still
matter, and neither is an output-alignment mechanism, so neither is refuted here:
(i) the residual at block 14 is overwritten with `c·v`, and blocks 15–28 then *add* branch outputs
to it — each block's own RMSNorm is scale-invariant, so the branch outputs are (near) fixed while
the carried `c·v` term scales, changing the balance downstream even though the block-14 readout
does not change; (ii) attention at blocks > 14 reads the slot rows as keys/values, where absolute
magnitude is not normalised away. Both are downstream-stack effects, testable — like this one —
without a probe draw.

The honest summary for the ladder: **M4d as registered is testing a hypothesis this diagnostic has
now largely ruled out.** An M4d null should not be read as evidence about configuration matching;
the parsimonious reading of M4c's 0/40 is that the adapter's output is off-manifold and
content-free at the receiver's readout, and that no amount of rescaling — in or out of the training
loop — changes that.

## 6. Reproduce

```bash
cd crates/latentmesh-runtime
cargo run --release --example run2_rescale_diagnostic     # CPU-only; ~12 s warm
```

The run refuses to start under `--features cuda`, verifies the adapter hash against the frozen M4c
training receipt, re-verifies the hand-rolled apply against the golden pairs, sha256-verifies the
sender dump / token streams / GSM8K inputs, and writes
`receipts/run2-rescale-diagnostic-receipt.json`. Unit tests covering the metric helpers (including
"a positive scalar preserves every rank metric" and "RMSNorm is scale-invariant") run under
`cargo test --all-targets`.
