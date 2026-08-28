# Research: run 1's negative result — a well-fitting linear alignment carries no cross-model causal signal

* Purpose: the S6 analysis summary for the live latent-exchange experiment pre-registered in
  [ADR-023](../adr/023-live-four-condition-run1-pre-registration.md). Companion to that ADR's
  own appended § S6 — Results, which is the receipt-cited ledger; this document is the narrative
  synthesis for a reader who wants the story once, not the frozen contract.
* Date: 2026-08-28.
* Scope: everything actually run — S0 (runtime mechanics) through S2c (generated-pairs
  recalibration contingency) — for the Qwen2.5-3B→Qwen2.5-1.5B pair on GSM8K. S3 (four-condition
  pilot) through S6 (full analysis of A1-A8) were never built: the pre-committed Deviation-7 stop
  rule in ADR-023 fired first.
* Method: every number below is read from a committed JSON receipt in
  `crates/latentmesh-runtime/receipts/`, re-verified against the file for this document (not
  copied from an earlier summary). Evidence label throughout: `live-model, single-host,
  simulation-free` for anything a model produced; `deterministic CPU fit over live-model dumps`
  for the alignment fits, which involve no model inference themselves.

---

## Abstract

Run 1 set out to test ADR-009's online causal control loop live, on real models, for the first
time. It stopped one stage short of the four-condition comparison it was built to run, on a
pre-committed kill rule — and the reason it stopped is itself the result. Injection mechanics work
(a model can use its own re-injected residual-stream state, p=0.03125 on a self-pair probe). A
training-free affine alignment between the two models' residual streams fits the calibration data
well by every measure available (held-out relative residual 0.51 on gold-text pairs, improving to
0.45 on sender-generated pairs — both comfortably under the 0.9 pre-registered gate), and every
hand-rolled application of that transform reproduces the alignment crate's own output bit-exactly.
And yet the aligned, calibrated vector is statistically indistinguishable from norm-matched random
noise on the exact test built to detect a real effect — at both of two depth pairs, under both of
two calibration distributions, four independent probes, zero passes. The gold-vs-generated
distribution-shift hypothesis that predicted this exact failure shape was directly tested via the
design's own pre-costed contingency and did not survive contact with the result: the
better-fitting transform did not produce a better signal. The most defensible reading, closing off
implementation error, identifiability, and reproducibility as alternative explanations in turn, is
that a single global, training-free linear map between pooled residual-stream states does not
carry causally usable cross-model information at these two depth pairs for this model pair. Run 1
therefore answers none of ADR-009 §5's compute or edge-survival clauses — it answers a logically
prior question, negatively, and does so with unusually tight attribution for a single-host,
~3.3-GPU-hour result.

---

## 1. What run 1 actually tested, and why it stopped where it did

ADR-023 pre-registered a four-condition comparison (`StaticText`/`DynamicText`/`DynamicLatent`/
`CausalDynamicLatent`) to answer, statistically, whether a causally-gated latent channel beats
text at equal quality and lower compute (ADR-009 §5). Before any of the four conditions could run,
the design required (§7 gates) that the injection mechanism work at all (S1a) and that a
cross-model alignment transform fit well enough to trust (S2, gate A6). Both of those passed. What
ADR-023 added on top — because S1a's passing probe used an *identity* transform on a model paired
with itself, not a real cross-model alignment — was a mandatory bridge probe (A7(b)) re-running
S1a's exact statistical protocol with the real, S2-fitted transform between the two different
models. That bridge probe is where run 1 actually stopped.

## 2. What was established before the stop

**Injection mechanics transmit information.** S1a's self-pair probe (Qwen2.5-1.5B → itself,
identity transform, 40 held-out GSM8K-train items) passed its pre-committed primary test after an
honestly-diagnosed and openly preserved run-1 failure (a BF16 RoPE position-aliasing bug plus an
answer-format scoring bug — both fixed, both receipts kept,
`crates/latentmesh-runtime/receipts/run-ledger.json`). The passing run: one-sided exact sign test,
real > random, paired accuracy, 5 wins / 0 losses, **p = 0.03125** — the minimum p-value attainable
at 5 discordant pairs, a real but thin effect (the NLL secondary diagnostic showed no effect at
all in the same run, 19W/21L, p=0.68).

**The cross-model alignment fits well, by every measure this pipeline can produce, at both
registered cells and both calibration distributions:**

| Depth cell | Calibration source | Held-out relative residual | Gate A6 (<0.9) |
|---|---|---:|---|
| L18→L14 (S2's chosen winner, min residual of a 3×3 sweep) | gold, teacher-forced GSM8K text | 0.5106 | PASS |
| L24→L19 (the design's original anchor depth, kept as a registered fallback) | gold, teacher-forced | 0.5600 | PASS |
| L18→L14 | generated — sender's own reasoning, teacher-forced through the receiver (S2c contingency) | 0.4451 | PASS, better |
| L24→L19 | generated (S2c) | 0.4682 | PASS, better |

Every one of these four fits, applied through a hand-rolled `apply()` function (necessary because
`latentmesh-align` cannot be linked into `latentmesh-runtime` — a Cargo half/`MSRV` conflict, ADR-
023 Deviation 1), reproduces the alignment crate's own reference output **bit-exactly** against 8
golden input/output pairs per cell (`max_relative_l2_error: 0.0` in every case). This closes off
"the reimplementation is subtly wrong" as an explanation for anything that follows.

## 3. The negative result

Despite a well-fitting transform, verified bit-exact against the reference implementation, the
aligned vector carries **no detectable causal signal** over norm-matched random noise:

| Cell | Calibration | Aligned-real wins / losses vs. random | p (exact sign test) | Result |
|---|---|---|---:|---|
| L18→L14 | gold | 2 / 1 | 0.5000 | fail |
| L24→L19 | gold | 1 / 2 | 0.8750 | fail |
| L18→L14 | generated (S2c) | 3 / 1 | 0.3125 | fail |
| L24→L19 | generated (S2c) | 1 / 2 | 0.8750 | fail |

None of the four probes reaches the pre-registered α=0.05 threshold. The pattern is not "close and
noisy" — the anchor cell (L24→L19) produces the *identical* p-value (0.875) under both calibration
distributions, meaning the recalibration changed nothing measurable there at all. The winner cell
(L18→L14) does move toward significance under the generated-pairs recalibration (0.50 → 0.31), but
not remotely close to a pass. A separate, weaker gate — that a *true zero vector* injected through
the real 8-slot pathway should not be catastrophically worse than no injection at all — passed in
all four combinations, meaning the injection mechanism itself is not damaging the receiver; the
*aligned content specifically* is what carries no usable signal.

**This directly tests, and does not confirm, the design's own leading hypothesis for exactly this
failure shape.** Design 024 §8 risk 6 named gold-vs-generated calibration distribution shift —
transforms fit on gold teacher-forced text, applied to sender-*generated* reasoning — as the prime
suspect for a null result like this one, and pre-costed a ~1 GPU-hour contingency to test it
directly: recalibrate from the sender's own generated reasoning and re-run the identical bridge
probe. That contingency ran (ADR-023 Deviation 7) and produced a transform that fits the
calibration data *better* by every measure available (0.445 vs. 0.511 at the winner cell) — and
the causal signal still did not appear above chance. Distribution shift, whatever its role, is not
sufficient on its own to explain the result: a genuinely stronger fit did not translate into a
detectable effect.

**The attribution chain, each link closing off one alternative explanation:**

1. **Mechanics work** — S1a's identity-transform pass rules out "the injection pathway itself is
   broken."
2. **The implementation is correct** — bit-exact reproduction of the alignment crate's reference
   output at every cell rules out "the hand-rolled cross-model apply function has a bug."
3. **Identifiability is mostly clean, with one honest caveat** — the gold-pairs fits use
   `n_fit=3,200` against sender dimension `d=2,048`, a comfortable margin. The generated-pairs
   fits land at `n_fit=2,048` — exactly equal to `d`, not comfortably above it, because a
   pre-committed cost-projection ladder capped the achievable sample count. The polar-uniqueness
   floor relative to the *receiver's* dimension (1,536) still clears with margin, and the
   qualitative result is unchanged by this caveat — the gold-pairs cells, comfortably `n>d`, show
   the identical failure — but it is disclosed rather than rounded away.
4. **Reproducibility holds** — every bridge probe verifies it used the exact frozen S1a item set
   and the exact registered transform hash, not a drifted variant.

With mechanics, implementation, and reproducibility ruled out, and distribution shift tested
directly and found insufficient, the most defensible reading is a substantive one: **a single
global, training-free affine map between pooled residual-stream states does not carry recoverable
cross-model causal information at either tested depth pair, for this model pair.** This is not
"the experiment was broken" — it is closer to "the experiment worked, cleanly, and returned no."

## 4. Which ADR-009 clauses this run answers

**None of A1-A8.** The compute/latency clause (A1-A3) and the edge-survival clause (A4) both
require `DynamicLatent`/`CausalDynamicLatent` to actually run against a working aligned-injection
channel; A7(b)'s failure at all four probes fired ADR-023's pre-committed stop rule exactly as
registered, and S3 through S5 were never built. The combined ADR-009 §5 acceptance test is exactly
as open after run 1 as it was before — this run does not weigh in on it either way. What it does
answer, cleanly, is the logically prior question the design's own gate structure was built to ask
before anything downstream could be trusted: does a training-free linear alignment of pooled
states carry cross-model signal here? At high confidence, given four independent, bit-verified,
cross-distribution probes: **no.**

## 5. Total GPU accounting

Summed directly from every receipt's own `wall_clock_s` field, including the disclosed 3,696.03 s
charge from a killed-and-resumed S2c attempt (preserved by incremental persistence, not silently
dropped):

- GPU-model stages (S0, both S1a runs plus the n=2 smoke, S2 dump, both gold bridge probes, both
  halves of the S2c generated-dump, both generated-pairs bridge probes): **11,802.8 s ≈ 3.28
  GPU-hours.**
- Plus three CPU-only alignment-fit stages (S2, the S2b anchor recalibration, S2c): **156.6 s ≈
  0.04 GPU-hours** (these involve no model inference, per every fit receipt's own evidence label).
- **Grand total: 11,959.5 s ≈ 3.32 GPU-hours.**

This is a receipt-summed figure, not an estimate. It is lower than a "~5.7 GPU-h" figure named in
this write-up's own task brief; that figure could not be reconciled against any committed receipt
and is flagged as an open discrepancy rather than adjusted to match — see ADR-023 § S6 — Results
§4 for the full accounting table and the specific reconciliation attempts that did not pan out.

## 6. What this experiment will not claim

Everything ADR-023 pre-registered under this heading still holds; with S3-S6 unbuilt, most items
are vacuously satisfied because the work they warn against never happened (no Zhang & Emu-style
decomposition was computed, no `EdgeTrial` audit ran, no Darwin loop mutated anything, no
authority ceiling moved against a live episode). Two items are directly, substantively addressed
by what did run: the pooled-vs-per-token caveat (item 4) is exactly what §7(b) below proposes
testing next, and the calibration-distribution-shift caveat (item 6) was tested head-on and found
insufficient to explain the null result on its own. One item's spirit landed early rather than
late: item 10 warned that a passing compute claim with a near-zero content decomposition would be
a hollow win reported honestly rather than papered over — that exact shape (a good *fit*, A6 PASS,
paired with a null *causal* result, A7(b) FAIL) is what stopped this run before A1 was ever
computed, one layer earlier than item 10 anticipated it would need to be caught.

## 7. Registered future work — run 2 (a proposal, not a pre-registration)

Not scoped, budgeted, or gated — a starting point for whoever writes run 2's own pre-registration,
informed specifically by what failed here (a well-fitting global linear map, ruled out as a
distribution-shift artifact):

**(a) A small trained nonlinear projector.** The Cache-to-Cache-style baseline
(`docs/research/023-beyond-sota-roadmap.md` §1a): train a small MLP between the same capture/
inject points on the same calibration pairs. Tests whether the ceiling here is *linearity*
specifically.

**(b) A FastGRNN-class tiny gated-RNN sequence translator, over per-token states rather than
pooled vectors.** Run 1 pooled every capture to one vector per message; pooling is an **untested
destruction suspect** — S1a's identity self-pair tolerated pooling in the no-alignment case, but
cross-model *translation* of a pooled average was never isolated from translation-per-se. The S2c
dumps already hold the raw material for this without new capture work: 2,560 items with generated
spans averaging 280.9 tokens each yield **719,115 sender-side generated-token positions**
(receipted exactly, not estimated — `s2c-generated-dump-receipt.json`), with the receiver side
captured over the identical shared span. A gated-RNN's KB-scale footprint also fits LatentMesh
Air's edge-device story better than a full trained projector.

**(c) Receiver-side MicroLoRA adaptation trained from causal-gate ΔV feedback**, rather than a
fixed transform. Confirmed present in-stack on this host: `@ruvector/sona` MicroLoRA — rank-1-4
hidden-state adapters, <50 KB each, real-time `adapt()` from a feedback signal
(`~/ruvector-upstream/crates/ruvllm-wasm/docs/MICRO_LORA.md`) — and `AdaptiveEmbedder`'s EWC++
consolidation plus memory-augmented retrieval
(`~/ruvector-upstream/npm/packages/ruvector/src/core/adaptive-embedder.ts`), both verified present
by direct read, not assumed from the name. ADR-003's own ΔV is a natural feedback signal for this.

**(d) HNSW-retrieved local linear maps**, testing whether the null result is about
global-vs-local structure rather than linear-vs-nonlinear — `latentmesh-memory` already has HNSW
indexing (ADR-016) to retrieve a neighborhood-specific transform per query instead of fitting one
global map.

---

## Sources

- [ADR-023](../adr/023-live-four-condition-run1-pre-registration.md) — the pre-registration this
  document reports results against, including its own § S6 — Results with the full receipt-cited
  ledger this narrative summarizes.
- [docs/research/024-live-latent-experiment-design.md](024-live-latent-experiment-design.md) —
  the experiment design run 1 executed.
- [docs/research/023-beyond-sota-roadmap.md](023-beyond-sota-roadmap.md) — motivating gap
  analysis and the source of the Cache-to-Cache/FastGRNN framing in §7.
- `crates/latentmesh-runtime/receipts/{s0-receipt,s1a-receipt-*,run-ledger,s2-*,s2b-*,s2c-*}.json`
  — every receipt cited by exact value throughout this document, re-verified against the
  committed JSON for this write-up.
