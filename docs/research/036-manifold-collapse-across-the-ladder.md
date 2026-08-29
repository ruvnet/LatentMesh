# Research: is M4c's off-manifold collapse universal across the ladder?

* Purpose: execute ADR-024's **registered M4f pre-check** — *"re-run the same unembedding
  projection on M4d's artifact and on the M3/M4 artifacts to establish whether off-manifold
  collapse is universal across the ladder or specific to task-loss training"*
  (ADR-024 § "DIAGNOSIS (2026-08-29): the adapter collapsed to a fixed OFF-MANIFOLD direction").
* Date: 2026-08-28.
* Method / evidence label: **deterministic CPU analysis over committed artifacts — no probe draw,
  annotates only**. No live-model forward, no GPU (ADR-034 lane rule). The only model-derived
  object touched is the receiver's unembedding matrix and final RMSNorm gain, loaded CPU-side
  from the HF cache. No probe item, control, or statistic on ADR-028's protected list is touched,
  and no recorded outcome changes.
* Artifacts: receipt `crates/latentmesh-runtime/receipts/run2-manifold-precheck-receipt.json`;
  code `crates/latentmesh-runtime/examples/run2_manifold_precheck.rs` (CPU-only, default
  features), measuring through `examples/common/lens.rs` — the metric kit
  [docs/research/033](033-rescale-output-alignment-diagnostic.md) built, extracted so both
  diagnostics use one implementation rather than two.
* Method source: [docs/research/033](033-rescale-output-alignment-diagnostic.md) §4, applied
  unchanged to every committed L18→L14 adapter artifact plus two references.

---

## Answer

**Neither — and the third option is the informative one.**

Off-manifold output is the **untrained default** of this adapter architecture. The fresh,
zero-optimiser-step initialisation that M4c and M4d both start from is *already* orthogonal to
the receiver's residual stream (cosine **−0.021** to the receiver's own block-14 state for the
same item). **Reconstruction training moves the output onto the manifold** — M3 and all three M4
FastGRNN ranks land at cosine **0.975 – 0.985**, with natural norms and natural entropy, as do
the training-free run-1 affine bridges (0.982 / 0.996). **Task-loss training does not** — M4c sits
at **−0.018** and M4d at **+0.048**, having grown the initialisation's norm 14× (9.3 → 135 / 144)
without rotating it anywhere near the receiver.

So the collapse is **not universal across the ladder** (seven of the eleven emitters measured are
on-manifold, one is intermediate) and it is **not caused by the task loss** (its own starting point was
already there). It is what task-loss training *failed to remove*, because nothing in a
next-token-CE objective over the receiver's output requires the injected vector to resemble a
receiver state.

**The consequence for the ladder is the important part: one mechanism does not explain all nine
nulls.** M3's and M4's nulls were produced by payloads that are on-manifold, natural-norm and
natural-entropy. Their cause is elsewhere.

This report also **corrects `docs/research/033` §4** on two of its three legs — see §3.

---

## 1. What was run

Every committed L18→L14 adapter artifact, plus two references, projected through the receiver's
readout on the same 40 items with the same code.

| | |
|---|---|
| Candidates | 13 (11 emitters + 2 references) |
| Hash gate | **13/13** content hashes verified against the training receipt (or, for the run-1 affine maps, the S2b/S2c probe receipt's `transform_hash_matches_registered` gate) that froze them |
| Forward gate | every adapter's hand-rolled forward re-verified against the golden pairs the trained network itself produced (relative L2 ≤ 1e-5) before any projection |
| Items | 40 — the first 40 rows of the holdout split **M3, M4 r64/r128/r256 and M4c all pin by the same sha256** (`02c1bf50…`), verified against all five receipts at run time |
| Lenses | `plain` = `W_U·h`; `rmsnorm` = `W_U·RMSNorm(h)`, the receiver's actual readout (headline numbers) |
| Device | `Device::Cpu`; the run refuses to start if the `cuda` feature is enabled |
| Wall clock | 20 s warm |

**M4d landed mid-flight.** Its trainer finished and wrote both its artifact and its training
receipt while this pre-check was being written, so M4d's **trained** adapter is measured here
with its hash pinned (`cd2a61da…`). Its *probe* was running on the GPU throughout; nothing in
this lane touched it, and M4d's probe verdict is neither read nor affected.

**M4c's and M4d's initialisations are byte-identical** — both training receipts pin the same
`init_content_hash_sha256` (`c424030e…`). They are therefore one candidate, not two, and the
identity is asserted at run time rather than assumed.

### The reference that disciplines the reading

The `receiver_L14.tok.f32bin` capture dump (sha256-verified, pinned by
`run2-pertoken-dump-receipt.json`) holds the receiver's **own** block-14 residual states over the
same token spans. Two references are built from it:

* **`reference-receiver-L14-pooled`** — mean-pooled exactly as every adapter's output is. This is
  the target the reconstruction rungs were fit to, and the only row whose "correct" values are
  known a priori.
* **`reference-receiver-L14-single-row`** — one genuine, un-pooled state (the span's last row),
  because a *mean* of states is not itself a state.

## 2. Results

Headline metrics, `rmsnorm` lens, 40 items, 151,936-token vocabulary. `manifold cos` is the mean
cosine between the emitted vector and the receiver's own pooled block-14 state **for the same
item** — the sharpest column.

| candidate | training | manifold cos | ‖v‖ | entropy (nats) | inv-cos | top-10 union | gold %ile | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `reference-receiver-L14-pooled` | none (the target) | **1.000** | 34.5 | 9.30 | 0.962 | 78 | 38.4 | on-manifold |
| `reference-receiver-L14-single-row` | none (real state) | 0.667 | 46.6 | 3.36 | 0.635 | 153 | 22.5 | on-manifold |
| `run1-affine-s2b` | training-**free** closed form | **0.982** | 34.6 | 9.25 | 0.969 | 65 | 44.3 | on-manifold |
| `run1-affine-s2c-genpairs` | training-**free** closed form | **0.996** | 34.5 | 9.30 | 0.966 | 78 | 39.3 | on-manifold |
| `m3-mlp-pertoken` | reconstruction | **0.989** | 34.3 | 9.34 | 0.970 | 70 | 39.4 | on-manifold |
| `m3-mlp-pooled` | reconstruction (variant ii) | **0.975** | 31.5 | 9.35 | 0.964 | 63 | 41.7 | on-manifold |
| `m4-fastgrnn-r64` | reconstruction, sequence | **0.984** | 32.9 | 9.30 | 0.980 | 59 | 44.2 | on-manifold |
| `m4-fastgrnn-r128` | reconstruction, sequence | **0.983** | 31.6 | 9.32 | 0.974 | 63 | 31.1 | on-manifold |
| `m4-fastgrnn-r256` | reconstruction, sequence | **0.985** | 32.8 | 9.40 | 0.973 | 65 | 36.9 | on-manifold |
| `m4-fastgrnn-r64-superseded` | reconstruction (superseded run) | 0.511 | 28.7 | 5.54 | 0.978 | 22 | 15.7 | **intermediate** |
| `m4c-m4d-shared-init` | **none — 0 optimiser steps** | **−0.021** | 9.3 | 6.31 | 0.975 | 38 | 39.2 | **off-manifold** |
| `m4c-mlp-taskloss` | task loss | **−0.018** | 134.6 | 5.94 | 0.881 | 77 | 61.5 | **off-manifold** |
| `m4d-mlp-taskloss-deploymatch` | task loss + deploy-match | **0.048** | 143.6 | 5.85 | 0.907 | 52 | 35.1 | **off-manifold** |

*`inv-cos` = mean pairwise cosine between the 40 emitted vectors (item-invariance).
`top-10 union` = distinct tokens across all 40 top-10 sets, out of a possible 400.
`gold %ile` = mean rank of the item's `#### <gold>` tokens as a percentage of the vocabulary.*

**The M4c anchor reproduces `docs/research/033` §4 exactly**: gold mean rank 93,460 (61.5th
percentile), sender-span 81,154 (53.4th), entropy 5.943 nats, 77 distinct tokens across the 40
top-10 sets, dominated by `DirectoryName` (37/40), `rias` (35), ` Lanc` (34), ` Svens` (34),
` concession` (24), `上海证券` (21). Same numbers, same tokens, same counts. The control is sound;
everything else in the table is measured on that footing.

### The three families the table separates

1. **On-manifold, natural-norm, natural-entropy** — the two run-1 affine bridges and all four
   reconstruction-trained adapters. Cosine 0.975–0.996 to the receiver's own state, ‖v‖ 31.5–34.6
   against the reference's 34.5, entropy 9.25–9.40 against the reference's 9.30. These emitters
   are, on every measurement available here, *emitting something that looks like a receiver state*.
2. **Orthogonal to the manifold** — the untrained initialisation (‖v‖ 9.3) and both task-loss
   adapters (‖v‖ 135, 144). Cosine −0.021, −0.018, +0.048: not "far from" the manifold,
   **orthogonal to it**, which is the generic position of a random direction in 1536 dimensions.
   Entropy 5.85–6.31 against the reference's 9.30.
3. **One intermediate** — the *superseded* r=64 window-zero-init FastGRNN run at cosine 0.511,
   entropy 5.54, and only 22 distinct top-10 tokens with six of them appearing in 35–40 of the 40
   items. Partially degenerate, consistent with its having been superseded. It is the only row
   the registered thresholds classify awkwardly (0.511 clears the 0.20 off-manifold bar), and it
   is reported as intermediate rather than forced into either family.

### What task-loss training actually did

The task-loss lane started at cosine −0.021 with ‖v‖ 9.3 and finished at cosine −0.018 / +0.048
with ‖v‖ 135 / 144. It **grew the norm ~14× and left the direction where it found it.** That
tracks the deployment arithmetic `docs/research/033` §2 recorded from the other side: the probe's
rescale factor `c = natural_median/‖v‖ ∈ [0.259, 0.579]`, and `134.6 × 0.386 ≈ 52`, the natural
median. The rescale is doing exactly what it says — shrinking a 4× oversized vector back to
natural magnitude — while the direction it is shrinking is orthogonal to everything the receiver
has at that block.

This is also the answer to why the task loss could improve and the probe still invert. The
objective (`init_holdout_task_ce 0.2546 → best 0.1595` for M4c; `0.2545 → 0.1583` for M4d) is
next-token CE at the receiver's output. Nothing in it constrains the *input* it injects to look
like a receiver state, and 1536 dimensions leave ample room to reduce that CE along directions
the receiver never produces.

## 3. Correction to `docs/research/033` §4

`docs/research/033` §4 rested its off-manifold conclusion on three observations. Measured against
the receiver's **own** pooled block-14 state over the same spans — which that diagnostic did not
have in front of it — two of the three do not survive.

| 033's observation | measured here on the natural pooled state | status |
|---|---|---|
| "across all 40 items the top-10 sets draw from only **77 distinct tokens**" | the receiver's own pooled state gives **78** | **not diagnostic** — a property of pooled mid-stack states, not of M4c |
| "the direction is **nearly item-invariant**" | natural pooled state: mean pairwise cosine **0.962**; M4c: **0.881** | **inverted** — M4c is the *least* item-invariant pooled emitter in the table; every on-manifold adapter is *more* item-invariant than it |
| "gold-answer tokens sit at mean rank 93,460 — the 61st percentile, worse than the middle" | natural pooled state puts gold at the **38.4th** percentile; a real single state at the **22.5th** | **survives only as a relative claim** — a middling gold percentile is normal at block 14, so LAP's `A_lin < 0.05` "negligible" band cannot be used to conclude the adapter learned nothing; the receiver's own genuine state is in that band too |

What replaces them is a single measurement that the reference does **not** trip: **cosine to the
receiver's own state for the same item**, on which the two families separate by roughly a full
unit (0.975–0.996 vs −0.021…+0.048) with no overlap. Entropy tracks it cleanly as a second,
correlated signal (9.25–9.40 vs 5.85–6.31).

`docs/research/033`'s **headline conclusion is unchanged**: M4c's adapter output is off the
receiver's residual-stream manifold, and rescaling remains exonerated. Its *evidence* for that
conclusion is narrower than it appeared, and one of its stated grounds pointed the wrong way.

## 4. The finding that was not the question: pooling is itself a large step off the manifold

The un-pooled reference exists to check that the pooled reference is a fair target. It is not
entirely.

A genuine receiver block-14 state and that same item's **pooled** state have cosine **0.667**.
The pooled state's induced distribution has entropy **9.30 nats** against the real state's
**3.36**; its top-10 sets draw on 78 distinct tokens across the 40 items against the real states'
153; its mean pairwise cosine across items is 0.962 against 0.635. Pooling roughly triples the
entropy, halves the token diversity and drives the cross-item directions together.

So "on-manifold" in §2 means **on the manifold of *pooled* states**, which is itself well off the
manifold of the states the receiver actually carries. Every rung of the ladder — run-1 affine
included — injects a pooled vector into 8 slots. The best-behaved adapter in this table is
faithfully reproducing an object the receiver never produces.

That is a mechanism for the M3/M4 nulls which is **independent of everything the ladder has
tested so far**, costs no GPU to state, and is testable the same way: it predicts that injecting
a real single receiver state should behave differently from injecting a pooled one. It sits
alongside — not instead of — ADR-024's live M4e (one-shot vs continuous injection) and M4b
(receiver scale) hypotheses, and it is arguably upstream of both.

## 5. What this does and does not license

**Answers the registered question.** Off-manifold collapse is **not** universal across the ladder,
and **not** caused by task-loss training. It is the untrained default; reconstruction loss is the
only thing in the ladder that removes it; the two task-loss rungs are the only trained artifacts
that keep it.

**Splits the nulls into two causes, which is the operational consequence.**

* **M4c and M4d** inject a payload orthogonal to the receiver's residual stream at 4× its natural
  norm. That is a sufficient and well-evidenced explanation of an *actively counterproductive*
  intervention, and it is now confirmed to apply to M4d's artifact as well as M4c's. An M4d null
  is therefore doubly expected: ADR-024 had already recorded that the norm hypothesis was
  exonerated, and this shows M4d's own adapter did not fix the property that actually differs.
* **M3, M4 r64/r128/r256 and the run-1 affine bridges** injected on-manifold, natural-norm,
  natural-entropy payloads and still produced nulls. **Whatever caused those nulls is not
  off-manifold collapse.** ADR-024's "one mechanism explains all nine nulls" reading is refuted.

**Registers a caveat about the strength of the on-manifold result.** The reconstruction rungs and
the affine bridges were *fit to* the receiver's L14 states, so their landing on that manifold is
close to tautological. The non-tautological content is that (a) it holds at deployment, on
held-out items, through the hand-rolled forward that the probe would use; (b) task-loss training
demonstrably does *not* get there; (c) the untrained default is orthogonal; and therefore (d) the
nine nulls do not share one mechanism.

**Does not license any claim about M4d's probe outcome.** That probe was running while this ran;
its result is not read here and is not affected by this.

**Disclosed.** The run-1 affine transforms were fit on run-1's own S2/S2c calibration pairs and
are **not** held out of these 40 rows, unlike every trained candidate. The measurement here is
output geometry, not generalisation, so the overlap does not bias the manifold cosine — but it
means the affine rows are not a held-out comparison and are not read as one.

**Feeds M4f directly.** ADR-024 registered M4f as "constrain the adapter's output to the
receiver's residual-stream manifold". This pre-check shows the *reconstruction* objective already
achieves that constraint and does not rescue the transfer — so M4f as sketched (anchor to real
receiver states, penalise distance to the natural distribution, predict a convex combination of
observed states) would move M4c toward where M3 already is, and M3 is null. **M4f should be
re-scoped before it is scheduled**: manifold membership is necessary-looking but demonstrably not
sufficient, and the pooled-vs-real-state gap in §4 is the more promising target.

## 6. Reproduce

```bash
cd crates/latentmesh-runtime
cargo run --release --example run2_manifold_precheck   # CPU-only; ~20 s warm
```

The run refuses to start under `--features cuda`, verifies all 13 candidate hashes against the
receipts that froze them, re-verifies every hand-rolled forward against the trained networks'
own golden pairs, verifies the 40-row holdout split against the sha256 all five training receipts
pin, sha256-verifies both capture dumps / the token streams / GSM8K, and writes
`receipts/run2-manifold-precheck-receipt.json`. Unit tests covering the metric kit, the
classification thresholds, the hash gate and the affine/pooling commutation run under
`cargo test --all-targets`.
