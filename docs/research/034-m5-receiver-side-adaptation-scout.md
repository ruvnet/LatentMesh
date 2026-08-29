# Research: M5 scout — receiver-side MicroLoRA adaptation from causal-gate ΔV feedback

* Purpose: turn ADR-024's M5 — explicitly flagged there as "the least fully specified rung," an
  "authorial extrapolation, not literally sourced" — into an executable, pre-registerable spec, per
  the coordinator's task brief. Reads [ADR-024](../adr/024-run2-trained-thought-adapter-ladder.md)
  (the ladder, M3/M4/M4c outcomes, M4d/M4e contingencies), [docs/research/032](032-injection-configuration-science.md)
  (norm/placement/train-deploy-match science), [docs/research/028](028-sota-continuous-sweep-1.md)
  (task-loss finding), [ADR-003](../adr/003-causal-edge-verification.md) (the five-control ΔV gate),
  [ADR-023](../adr/023-live-four-condition-run1-pre-registration.md) (the frozen probe M5 must face
  unchanged), [ADR-028](../adr/028-evolutionary-adapter-search-anti-gaming.md) (evolvable/protected
  split — directly load-bearing for M5's training-signal design), [docs/research/031](031-statistical-power-and-design.md)
  (power floor of the frozen probe — directly load-bearing for Q3 below).
* Date: 2026-08-29 (this session; branch `feat/run2-thought-adapter`, read-only against the repo
  except this one file).
* Evidence grades as elsewhere in this repo: **primary** (file/paper read directly, this session),
  **inferred** (arithmetic extrapolation from a primary number), **uncertain** (search-only, not
  independently confirmed).
* Repo state verified this session (primary, `git log --oneline -15`, `find`): M4c is discharged
  (honest fail, positive-but-mismatched training signal, commit `62469df`). M4d has code
  (`crates/latentmesh-train/src/bin/train_m4d_deploymatch.rs`, `crates/latentmesh-runtime/examples/
  run2_m4d_{probe,transfer_check}.rs`) and one golden-pair receipt
  (`receipts/run2-m4d-golden-mlp-deploymatch-init-cellL18toL14.json`) but **no training or probe
  receipt** — M4d has not yet reported an outcome. M5 has zero code. This scout does not require or
  wait for M4d's result (scouting spends no probe budget), but its **sequencing recommendation**
  (§6) does depend on it.

---

## 1. Prior art: who trains a RECEIVER-side adapter to accept injected/foreign activations?

**Headline finding, primary-verified this session via a fetched survey paper: nobody, in the
literature this pass could reach.** "Beyond Tokens: A Unified Framework for Latent Communication in
LLM-based Multi-Agent Systems" (arXiv:2606.05711v2, fetched and queried directly this session)
taxonomizes every latent-communication method it surveys along three axes — WHAT is communicated,
WHICH sender-receiver alignment is used, and **HOW** the communicated content is fused into the
receiver. Its HOW axis names exactly five fusion mechanisms — concatenation, prepending,
mathematical operations, cross-attention, cache restoration — and **every one keeps the receiver's
weights frozen at inference time**; the survey's own training discussion is scoped explicitly to
"sender-side" or "system-level" components (routing policy, fusion-function complexity), never to
adapting the receiver's own parameters to accept content. The one partial exception it names,
LRAgent, is architectural (reconstructing a shared-base-plus-LoRA cache via a fused kernel across
agents that already have per-agent LoRA adapters for unrelated reasons), not a method that *trains*
a new receiver-side adapter *to accept a sender's injected state* — the fetch's own conclusion:
"the framework deliberately excludes receiver-side parameter adaptation from its design space."
This is the single most load-bearing citation in this section, because it is a survey purpose-built
to enumerate exactly this design space and states its absence explicitly rather than by omission.

**Closest established analogs, graded individually:**

- **Prefix-tuning / soft-prompt tuning** — trains new parameters that condition a frozen backbone,
  which is architecturally receiver-side in spirit. But the standard use case is *task* adaptation
  (train a prefix so a frozen model does a new task better), not *cross-model injection acceptance*
  (train a prefix/adapter so a frozen model can consume a **different model's** live-generated
  latent content). ADR-027 (already in this repo, read this session) names Interlat
  (arXiv:2511.09149) as the nearest prefix-flavored cross-model precedent — but ADR-027's own text
  describes Interlat as involving "a fine-tuned receiver," which if accurate would make it the
  closest real precedent found anywhere in this repo's research chain for training the receiver
  itself, not just a translator. This repo's own citation of Interlat is graded **inferred** (§2 of
  `docs/research/028`, abstract-level only, loss equation not resolved) — worth a dedicated PDF-body
  fetch before M5's design is finalized, named here as a gap, not closed by this pass.
- **C2C's Fuser** (arXiv:2510.03215, primary via `docs/research/028`) is the strongest positive
  cross-model result and is architecturally receiver-adjacent — its Projection→Dynamic
  Weighting→Learnable Gating pipeline writes into the receiver's KV-cache — but it is an **external
  module bolted onto the cache**, not a LoRA on the receiver's own weight matrices. The receiver's
  own parameters are untouched; only its cache contents change. This is a materially different
  mechanism from M5's proposed "rank 1-4 adapter on the receiver's hidden states."
  The Bicameral Model (arXiv:2605.11167) is the same shape — task-loss-trained, but via a coupling
  mechanism between two frozen backbones, not a weight-level adapter on either.
- **"Steering Awareness" LoRA** (arXiv:2511.21399, found this session, **inferred** — search-summary
  only) is a genuine existence proof that a rank-32 LoRA *can* be trained (4 epochs, ~14K examples)
  to make a model do something useful with an injected residual-stream vector — but the task is
  *detecting and classifying* the injection (95.5% detection, 71.2% concept ID), not *using* it
  productively for a downstream task. It establishes LoRA-scale capacity is sufficient to learn
  *something* about injected activations quickly; it says nothing about whether that generalizes to
  productive causal use.
- **Steer2Adapt** (arXiv:2602.07276, found this session, **inferred**) composes a model's *own*
  previously-extracted steering vectors via a learned low-rank subspace for fast task adaptation —
  same-model, not cross-model, and the adapted object is a *combination of vectors*, not a receiver
  weight adapter.

**Verdict on Q1: this is genuinely underexplored territory, not "already tried and failed" or
"already tried and succeeded" — the same honest verdict `docs/research/028` §3 reached for
sequence-level transfer.** The absence is stated with real confidence for one specific reason this
pass has that `028`/`027` didn't: a purpose-built taxonomy survey was fetched and directly queried
about exactly this axis, and its own text states the exclusion, rather than this scout inferring
absence from search snippets alone. This raises M5's novelty (a genuine reason to run it) and its
risk (no existing recipe to borrow the *training mechanics* from, unlike M3/M4/M4c/M4d, which could
each borrow architecture and loss-type precedent from C2C/Bicameral almost directly).

---

## 2. The in-stack MicroLoRA option: usable, or a name only?

**Read directly, this session, primary: `~/ruvector-upstream/crates/ruvllm-wasm/src/micro_lora.rs`
(736 lines, full source, not the README).** Two findings, both decisive:

1. **It is a real Rust crate, not JS** — good news the ADR-024 citation slightly obscures (it cites
   `docs/MICRO_LORA.md`, a JS-facing doc). The actual implementation (`LoraAdapterInternal`) is
   plain Rust: `lora_a: Vec<f32>` (`in_features × rank`), `lora_b: Vec<f32>` (`rank × out_features`),
   `forward()` does `output += (x @ A @ B) * scaling` — architecturally exactly the rank-1-4
   additive adapter ADR-024 names. It is `wasm_bindgen`-wrapped for browser use, but the core
   `LoraAdapterInternal` struct and its math have no WASM dependency — porting the *architecture* to
   `latentmesh-train` (candle tensors instead of `Vec<f32>` loops) is a small, mechanical rewrite,
   not a redesign.
2. **Its `adapt()`/`accumulate_gradient()` is NOT usable as-is, and this matters more than (1).**
   Read verbatim, `micro_lora.rs:341-374`: gradient is *not* backpropagated from any loss function.
   The comment says it plainly — `"Simple gradient estimate: use quality as reward signal... For
   browser use, we use a lightweight update rule without full backprop"`. Concretely:
   `reward = (quality - 0.5) * 2.0`, then `grad_b[r,o] += intermediate[r] * reward * scaling * 0.01`
   and `grad_a[i,r] += input[i] * reward * scaling * 0.01` — an outer-product Hebbian/REINFORCE-style
   delta rule keyed on a single scalar quality score, with **no connection whatsoever to the
   receiver's own forward pass, task loss, or ΔV's actual value** (it doesn't even see the receiver
   model). It is designed for a fundamentally different regime: a browser-side per-request nudge
   from a thumbs-up/down signal, at $O(\text{rank} \times \text{dim})$ cost per update, with no
   autodiff engine available in that environment at all.

**Direct implication for M5**: this in-stack MicroLoRA is **not** a training-loop primitive to wire
ΔV into — it is, at best, a *reference for the adapter's forward-pass architecture and memory
footprint* (rank 1-4 additive residual delta, <50 KB, matches ADR-024's citation exactly), and
should be read that way, not as reusable training machinery. **What must actually be built**: a
`candle`-native `LoraAdapter` struct in `crates/latentmesh-train` (following `mlp.rs`/`fastgrnn.rs`
precedent already in that crate — primary, both files exist, `crates/latentmesh-train/src/{mlp,
fastgrnn}.rs`) trained by real backprop through `AdamW` (the only optimizer `candle-nn` 0.9.2 ships,
per ADR-024's own frozen-optimizer-constraint finding), with gradients flowing from an actual scalar
loss — **not** the browser crate's heuristic rule. **M4c already proved this is feasible at this
model scale**: ADR-024's M4c outcome section reports a live, seeded gradient path through the full
1.5B receiver's own forward pass (`"gradient path live"`, per `docs/research/028` §2's C2C-pattern
citation that M4c/M4d follow), 1,604 s GPU for 10 epochs over 2,035 fit items. M5's adapter would sit
in the identical position in that same pipeline — no new infrastructural risk here, just a different
(and smaller) parameter set to train, plus wiring it downstream of the injection point rather than
replacing the translator.

**Recommendation**: reimplement `LoraAdapterInternal`'s architecture (down-project A, up-project B,
`alpha/rank` scaling, zero-init B — standard LoRA init, correctly done in the source) as a `candle`
module in a new `crates/latentmesh-train/src/receiver_lora.rs`, and discard `accumulate_gradient`/
`apply_gradients` entirely in favor of `candle-nn::AdamW` over a real loss (§4 below specifies
which). This is a genuine build, comparable in scope to `mlp.rs` (M3) — not a wrapper around an
existing trainer.

---

## 3. The ΔV-feedback training loop: is it a trainable signal at our scale?

**No — not as a per-step or per-epoch training signal. This is the single most important finding of
this scout, and it is receipt-grounded, not a hunch.**

### 3.1 What one `verify_edge` call actually costs

`crates/latentmesh-gate/src/causal.rs` (read in full, primary, 248 lines) requires, per call: `n`
paired trial outcomes across **six** conditions (`real` + 5 controls: `zero`, `random`, `mismatched`,
`self_generated`, `text_equivalent`), then a sign-flip permutation test per control, admitting only
if the **worst** control's p-value clears α. The permutation resampling itself is CPU-cheap
(`resamples` reshuffles of already-collected numbers) — the real cost is **generating the `n × 6`
receiver rollouts** that produce those outcome values in the first place. This repo's own receipts
give a direct anchor: S1a's n=40, 2-condition (real vs. random) mechanics probe cost **512.22 s
GPU** (`s1a-receipt-slots8-block19-poolfull-rescaletrue-n40.json wall_clock_s`, primary). Scaling
linearly by condition count and item count (**inferred**, not separately measured, but the
receiver-generation-dominated cost structure this repo has established via 92× decode/prefill ratio
findings in `docs/research/030` makes linear scaling the defensible first-order model): a full
six-condition `verify_edge` draw at `n=40` is on the order of **(6/2) × 512 s ≈ 1,536 s ≈ 0.43
GPU-h** — and `docs/research/031` §2.1/§2.2 (primary, read in full this session) establishes that
**n=40 is nowhere near enough**: at this ladder's own observed ~10% discordance rate, n=40 yields
roughly 4 discordant pairs, which is *structurally incapable of reaching p<0.05 regardless of the
true effect size* (minimum attainable p at n_disc=4 is 0.0625 > α). `docs/research/031` §2.2/§2.3
independently derives that **~25-30 discordant pairs are needed for real power at a plausible effect
size (θ=0.75)** — at the ladder's observed 10% discordance rate, that means **roughly 250-300 total
items per single meaningfully-powered `verify_edge` draw**. Scaling the same anchor: **(6/2) ×
(280/40) × 512 s ≈ 10,752 s ≈ 3.0 GPU-h for ONE properly-powered ΔV measurement** — essentially
run 1's *entire* GPU budget (3.3 GPU-h, ADR-023 §S6 Table 4) spent on a single feedback point.

### 3.2 Why this rules out ΔV as an online/per-step signal

A gradient-trained adapter needs feedback on the order of hundreds to thousands of updates to
converge (M4c's own receipt: 10 epochs × 2,035 items = 20,350 gradient steps, 1,604 s total — i.e.,
**less time for the entire training run than one single well-powered `verify_edge` call would
cost**). Treating ΔV as a per-step or even per-epoch reward would require O(epochs) properly-powered
`verify_edge` draws, each costing more than the entire M4c training run — this is not merely
expensive, it is qualitatively the wrong shape of signal: high-variance (discrete, permutation-test
outcome, not a smooth scalar), extremely sample-hungry per measurement (§3.1), and — critically —
**forbidden as a training-loop signal by this repo's own already-frozen governance**: ADR-028
(`docs/adr/028-evolutionary-adapter-search-anti-gaming.md`, read in full this session) draws an
explicit evolvable/protected line and states, for *any* adapter search in this ladder's design
space, **"fitness for every evolvable-surface search is a metric computed on the adaptation-512 pool
ONLY... never the frozen 40-item probe."** ADR-024's M5 sketch ("train... using ADR-003's own ΔV as
the feedback signal") does not itself specify which item pool feeds `verify_edge` during training —
if read as implying repeated draws against the *frozen* probe, it directly conflicts with ADR-028's
later, more specific, and explicitly-frozen rule. If read as repeated draws against
`adaptation-512`, it survives ADR-028's rule but still costs ~3 GPU-h per properly-powered draw
(§3.1) — cheap items, expensive *test*, because the expense is receiver generation across six
conditions, which is identical regardless of which pool the items come from.

### 3.3 The credible alternative, and why it is not a compromise but the mechanistically better design

**Recommendation: train on task loss (as M4c/M4d already do), and use ΔV only as the promotion
gate** — exactly ADR-028's evolvable/protected split, which is not merely the cheaper option but the
one that is already this repo's frozen governance. Concretely for M5:

- **Training signal**: per-step next-token cross-entropy, differentiable, computed every step —
  cheap (§2's cost anchor: 20,350 steps in 1,604 s for a comparably-sized model/loss in M4c).
- **ΔV's role**: computed **once**, post-training, on `adaptation-512` (never the frozen probe), as
  a **characterization/diagnostic** of the trained adapter — this is what actually "exercises the
  causal admission machinery live" in ADR-024's own words, honestly satisfied by a single
  post-training draw rather than an online loop that is both governance-noncompliant and three
  orders of magnitude too expensive to run at the cadence gradient training needs.
- **The frozen 40-item probe** remains the *one* thing M5 is ultimately gated on, per ADR-024's own
  "one draw per rung" discipline (restated and generalized by ADR-028's promotion rule: "at most one
  champion per registered ladder rung... receives exactly one frozen-probe draw").

This reframing does not weaken M5's stated goal (exercise the causal-gate machinery) — it relocates
*where* ΔV enters, from an infeasible per-step reward to a legitimate, already-licensed
characterization + gate role, which is the only role the repo's existing statistical-power evidence
and its own anti-gaming ADR permit anyway.

---

## 4. Leakage danger — is a modified receiver a genuinely new contamination class, and what changes?

**Yes, and the coordinator's suspicion is correct: the frozen probe's baseline/zerovec/random arms
must be RE-MEASURED under the M5-adapted receiver, not reused from M3/M4/M4c/M4d's receipts.**
Reasoning, stated explicitly rather than assumed:

1. **M3/M4/M4c/M4d share one invariant M5 breaks**: across every prior rung, the receiver's own
   weights were bit-identical — only the *injected content* changed (a different translator's
   output). This is *why* those rungs' baseline/no-injection conditions are mutually comparable (in
   principle; whether they were in fact re-measured fresh each receipt or reused is not confirmed by
   this pass and is itself worth a follow-up check before M5, flagged here as **uncertain**, not
   assumed). **M5 changes the receiver's weights themselves** (a permanent, content-independent
   capability delta from the LoRA's mere presence) — this is categorically different from "a
   different vector was added at the same frozen site."
2. **The specific confound this introduces has a name in this repo's own foundational document**:
   ADR-001 §5's original risk — "an apparent gain... can come from message *presence*, extra
   *compute*, or a generic state effect — not from A's specific information" — reappears one layer
   up. A LoRA adapter trained via task loss could, in principle, become a **general GSM8K
   fine-tuning delta** that improves the receiver's answer accuracy regardless of what (if anything)
   is injected at inference time — exactly the mechanism ADR-003's five controls exist to rule out
   for message *content*, now needing to rule it out for a **standing weight change** that persists
   even under the `zero` control. If M5's `aligned_real` condition beats `random`/`zero` but the
   *same adapted receiver* under `zero` injection also beats M3/M4/M4c/M4d's frozen-receiver
   baseline, the result is evidence of "LoRA fine-tuning helped GSM8K," not "the adapter learned to
   use cross-model latent content" — a confound none of the ladder's prior rungs could produce,
   because none of them could change the receiver's zero-injection behavior at all.
3. **Consequently**: M5's own receipt must include a **six-condition-plus-one battery**, not the
   usual four/five: `aligned_real`, `zero`, `random`, `mismatched`, `self_generated` (all through the
   *adapted* receiver, per ADR-003), **plus** the original frozen (pre-M5) receiver's own
   `baseline`/`zero` numbers carried forward from the ladder's shared invariant, explicitly labeled
   as "prior-rung reference, different receiver weights, not a same-model control." The comparison
   that actually isolates M5's claim is **`aligned_real` (adapted receiver) vs. `zero` (SAME adapted
   receiver)** — not vs. any earlier rung's frozen-receiver baseline. This is a stricter, not looser,
   bar than earlier rungs faced, and it should be stated as such rather than silently inherited.
4. **Does a modified receiver invalidate comparisons to earlier rungs' baselines?** Yes, for
   causal-gate purposes specifically (point 3), but **not** for the ladder's overall accounting —
   M5's result is still reportable alongside M3/M4/M4c/M4d's as "rung N, receiver-adaptation family,"
   the same way M4 (sequence structure) is reported alongside M3 (pointwise) despite testing a
   materially different hypothesis (ADR-024's own stated convention for cross-rung comparison,
   "M4... tests a materially different hypothesis... a rung that fails the probe less badly than the
   previous rung is still a reportable finding").
5. **What must be excluded from training**: identical to every prior rung — the same 13
   probe-overlap items (ADR-024 § Leakage discipline, `(69,150) (683,1309) ...`), the same
   `fit_holdout_split(2560, FIT_SPLIT_SEED=0x24C0_DE03)` applied first, split by item never by
   token. No new exclusion category is needed for the *training data* — the new risk is entirely in
   how the *evaluation* battery must be structured (points 2-3), not in what data trains the adapter.

---

## 5. Spec

### 5.1 Architecture

- **Adapter**: rank-`r` (`r ∈ {1,2,4}`, ladder-style ordered sub-rungs mirroring M4's `r∈{64,128,256}`
  discipline — cheapest first, stop at first pass, all attempted ranks' receipts kept regardless of
  outcome) additive LoRA, `candle`-native, architecturally following `micro_lora.rs`'s
  `LoraAdapterInternal` (down-project `A: [d_h, r]`, up-project `B: [r, d_h]`, Kaiming init for `A`,
  zero init for `B`, `scaling = alpha/r`) but with `AdamW`-driven real backprop replacing the
  browser crate's heuristic delta rule (§2).
- **Attachment point**: on the receiver's residual stream **at and immediately after the injection
  site** (block 14 of 28, the frozen L18→L14 winner cell, per ADR-023 Deviation 6 / ADR-024's
  reused injection semantics) — added to the residual at L14 exactly where the translated vector is
  overwritten into the 8 placeholder slots, so the adapter's job is specifically "help this one
  layer's output make better use of whatever content the slots now hold," not a general capability
  patch elsewhere in the stack. **Escalation path, named but not built in this wave**: if the
  minimal single-site adapter's training curve underfits, extend to a small span of blocks (14-19,
  matching the injection→receiver-capture span M0/M2 already established), each with its own
  rank-`r` adapter — more expressive, bigger scope, registered as a contingency exactly the way
  ADR-027's prefix delivery is registered without being built.
- **Upstream translator**: **frozen**, not co-trained. Use M4d's trained output once it reports (the
  ladder's "same or best-so-far" convention, per M4c's own architecture-selection rule) — if M4d has
  not yet reported when M5 is scheduled, fall back to M4c's architecture (M3's MLP, 2048→512→1536
  ReLU, the current best-by-holdout-residual per ADR-024's M4c outcome section). Freezing the
  translator isolates M5's actual hypothesis (does receiver-side adaptation help) from re-testing
  the translator question M4d already owns.

### 5.2 Training objective — the one substantive deviation from ADR-024's original M5 sketch

**Recommendation: next-token cross-entropy on the probe's own scored target — the `#### <gold>`
continuation — not the sender's generated span.** This is a direct, named response to M4c's own
diagnosed failure mode (ADR-024 M4c outcome, point 5): task-loss training on the sender's span
"steer[ed] the receiver toward reproducing the sender's generated token span... not the answer-format
objective the probe scores," and `docs/research/032` §4 independently names the missing diagnostic
("nothing in the current receipt schema checks `A_lin`-style output-alignment for the probe's own
target... as opposed to the sender's generated span"). Training the **new LoRA parameters** (with the
frozen translator's output as their input) directly against the gold-answer continuation attacks
this mismatch at its source, rather than requiring the translator to solve two objectives (resemble
the sender's reasoning AND produce the right final answer) at once. **Deploy-transform-in-loop is
inherited from M4d**: the 8-slot placement and rescale-to-natural-median operator must sit inside
M5's training loop exactly as M4d registers, so M5 does not reintroduce the same exposure-bias
mismatch M4d exists to close.

**Pre-registered, run-once, zero-GPU diagnostic before any M5 training**: per
`docs/research/032` §4/§5's ranked recommendation #1, compute LAP's `A_lin` (arXiv:2604.15557) for
the frozen translator's raw and rescaled output against the gold-answer tokens specifically (not
just the sender's span, which M4c's own receipt already implies is well-aligned and answer-format is
not). If `A_lin` against gold-answer tokens is near zero even pre-rescale, that is evidence the
injection *site* geometrically cannot reach the answer-token subspace at all — in which case M5's
receiver-adaptation hypothesis (fix the receiver's *use* of content, not the content itself) is
still worth testing (a LoRA on downstream layers could in principle learn to route around a
geometrically awkward site), but the prior should shift toward expecting a harder climb, and this
should be stated in M5's own pre-registration addendum rather than discovered mid-run.

### 5.3 Splits, receipts, gate

- **Splits**: identical to every prior rung — `fit_holdout_split` first, 13-item leakage exclusion
  second, item-level (never token-level) split, `calibration-4000`/`adaptation-512` only,
  `eval-200`/`holdout-100` mechanically locked until M6's genome-freeze event (unaffected by M5).
- **Receipts**: same schema (`evidence_label`, training RNG seed, `FIT_SPLIT_SEED`, excluded-13 list,
  `git_commit`, GPU/nvcc environment, wall-clock/GPU-seconds — ADR-024 § Training receipts) **plus**
  the six-condition-plus-one battery specified in §4 point 3, explicitly labeled by which receiver
  weights each number was measured against.
- **Exact gate**: one frozen-probe draw, `aligned_real` (adapted receiver) vs. `random` (same
  adapted receiver, freshly measured — never reused from an earlier rung's receipt), one-sided,
  α=0.05. **Recommend adopting mid-p McNemar as the primary statistic** (ADR-030/`docs/research/031`
  §2.4's already-adopted, zero-cost upgrade for run 3) with the classic exact sign-test p reported
  alongside, rather than the exact-only statistic ADR-024's M3/M4/M4c text still names — this is not
  yet formally amended into ADR-024 itself, so M5's own addendum should either adopt it explicitly
  (recommended, for consistency with run 3's frozen design) or explain why it doesn't.
- **ΔV role**: computed once, post-training, on `adaptation-512` (§3.3) — reported as a
  characterization, not gated on, and never computed against the frozen probe during training.

### 5.4 Cost estimate (GPU-hours, from this repo's own receipted rates)

| Component | Estimate | Basis |
|---|---:|---|
| Zero-GPU LAP `A_lin` pre-check | ~0 (CPU matmul) | `docs/research/032` §5 item 1 |
| LoRA training, per rank, one seeded run | ~0.3-0.6 GPU-h | **inferred**, scaled from M4c's 1,604 s / 10 epochs / 2,035 items — comparable backprop-through-receiver cost structure, smaller parameter count (rank 1-4 vs. 1.8M-param MLP) should not materially change per-step cost, since the dominant cost is the receiver's own forward/backward pass, not the adapter's |
| Post-training ΔV diagnostic (one draw, `adaptation-512`, six conditions, `n≈64-80` per ADR-023's own audit-batch escalation ladder 48→64→80) | ~0.5-0.75 GPU-h | **inferred**, scaled from S1a's 512.22 s / 2-condition / n=40 receipt (§3.1's method) |
| Frozen-probe draw, full six-condition-plus-one battery, `n=40` | ~0.43-0.6 GPU-h | **inferred**, same scaling |
| **Total, one rank tried once** | **~1.2-2.0 GPU-h** | |
| **Total, all three ranks (r=1,2,4) if each fails and escalates** | **~3.6-6.0 GPU-h** | |

For comparison: run 1's entire GPU budget was 3.3 GPU-h (ADR-023 §S6 Table 4); M4c's single-rung
training+transfer-check was 1,604 s ≈ 0.45 GPU-h. **M5, done as specified (task loss + one-time
post-hoc ΔV, not an online ΔV loop), is cheap and comparable to the ladder's other rungs.** The
version ADR-024's original text could be read to imply — ΔV as a repeated online training
signal — would cost roughly **3 GPU-h per single properly-powered feedback point** (§3.1), which
would make even one epoch's worth of online feedback more expensive than this entire estimate.

### 5.5 Honest-fail path

Per ADR-024/032's standing discipline: every attempted rank's receipt is committed regardless of
outcome, no re-tuning against the frozen probe. If all attempted ranks fail: M5 is reported as a
third honest null in the receiver-adaptation family, alongside M3 (pointwise reconstruction), M4
(sequence reconstruction), M4c (pointwise task-loss, sender-span target), M4d (pointwise task-loss,
deploy-matched) — and the ladder proceeds per ADR-024's named next contingencies: M4e (continuous
per-step injection, already registered) or ADR-027's latent-prefix delivery (a different integration
mechanism entirely), or, if the cheaper diagnostics and M4b (receiver-scale arm, still mandatory and
independent of M5) are also exhausted, the joint negative-result report `docs/research/032` §5 item 6
already names as the ranked final step. **M4b is not superseded or made optional by M5** — a
receiver-scale threshold and a receiver-acceptance-capacity hypothesis are independent explanations,
and both remain open until each is tested on its own terms.

---

## 6. Sequencing recommendation (not one of the five numbered questions, but load-bearing)

**M5 should run after M4d reports, not in parallel with it, though this scout does not need to wait
for M4d to be written.** Reasoning: M4d tests whether the deploy-transform-in-loop fix alone
resolves M4c's dissociation (train/deploy configuration mismatch, exposure-bias-shaped, per
`docs/research/032` §3). If M4d **passes**, the parsimonious reading changes substantially — the
mismatch was sufficient explanation on its own, and M5 becomes optional comparative work (mirroring
ADR-024's own "if M4 passes, M4c is optional" convention, applied one step later in the chain). If
M4d **fails**, M5's hypothesis — that the bottleneck is the receiver's capacity to *use* correctly-
delivered content, not the content's fidelity — becomes the most mechanistically distinct untested
explanation remaining in the mid-layer-injection family, and is worth running as specified above.

---

## Sources

- Internal (all read in full or in relevant part this session): `docs/adr/024-run2-trained-thought-adapter-ladder.md`,
  `docs/adr/023-live-four-condition-run1-pre-registration.md`, `docs/adr/003-causal-edge-verification.md`,
  `docs/adr/027-latent-prefix-context-window-delivery.md`, `docs/adr/028-evolutionary-adapter-search-anti-gaming.md`,
  `docs/adr/030-run3-causally-gated-text-pre-registration.md` (partial), `docs/research/028-sota-continuous-sweep-1.md`,
  `docs/research/031-statistical-power-and-design.md`, `docs/research/032-injection-configuration-science.md`,
  `crates/latentmesh-gate/src/causal.rs`, `crates/latentmesh-train/` (directory listing + `mlp.rs`/`fastgrnn.rs`
  existence check), `~/ruvector-upstream/crates/ruvllm-wasm/src/micro_lora.rs` (736 lines, full read),
  `~/ruvector-upstream/crates/ruvllm-wasm/docs/MICRO_LORA.md`
- "Beyond Tokens: A Unified Framework for Latent Communication in LLM-based Multi-Agent Systems":
  https://arxiv.org/html/2606.05711v2 (primary, fetched and queried directly this session)
- Cache-to-Cache: https://arxiv.org/abs/2510.03215 (primary via `docs/research/028`, re-cited)
- The Bicameral Model: https://arxiv.org/abs/2605.11167 (primary via `docs/research/028`, re-cited)
- Interlat: https://arxiv.org/abs/2511.09149 (inferred, abstract-level only — flagged for a future
  PDF-body fetch)
- "Predicting Where Steering Vectors Succeed" (LAP): https://arxiv.org/abs/2604.15557 (primary via
  `docs/research/032`, re-cited)
- "Steering Awareness: Detecting Activation Steering from Within": https://arxiv.org/html/2511.21399
  (inferred, search-summary only, this session)
- Steer2Adapt: https://arxiv.org/abs/2602.07276 (inferred, search-summary only, this session)
- Fagerland, Lydersen & Laake 2013 (mid-p McNemar); Waudby-Smith & Ramdas 2023 (e-process) — both
  via `docs/research/031`, re-cited, not re-fetched this pass
