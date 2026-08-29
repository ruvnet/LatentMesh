# 024. Run 2: trained thought-adapter ladder

> ## ✅ THE LADDER IS CLOSED (2026-08-29)
>
> **Final result, confirmed out of sample by PC3**: activation injection is
> **semantic at the likelihood level** (a payload moves probability toward the
> specific answer it encodes — decoy −0.773 nats, p = 7.5e-35; gold −0.320
> nats, p = 5.8e-12) and **non-semantic at the decision level** (it changes
> which answer the receiver gives no more than a norm-matched Gaussian:
> 36W/32L, n_disc = 68, **p = 0.72**, min attainable p 3.4e-21).
>
> **The decision layer reads the perturbation, not the meaning.**
>
> Every cross-model null in this ladder is explained by the apparatus and
> **none is evidence about latent transferability**. M5X (ADR-037) and M4b
> (ADR-035) are **blocked permanently**. Full write-up:
> **[research/048](../research/048-run2-final-synthesis.md)**.
>
> Firewall: these are **apparatus** results. A FAIL may not be cited as
> evidence against transfer any more than a PASS may be cited for it.

> ## ⛔ STOP — DEFECTIVE PRE-REGISTRATION, READ BEFORE INHERITING ANYTHING (2026-08-29)
>
> **PC1b's pre-registered FAIL branch is LOGICALLY INVERTED and must not be
> copied into any successor rung.** It reads:
>
> > *"the ladder nulls would stand as evidence about **TRANSFER** rather than
> > about plumbing."*
>
> **This is backwards.** A positive control exists to show the apparatus
> converts a **known-present** signal into **the measured outcome**. PC1b
> **failed**. A failed positive control makes the **method** the *leading*
> explanation for every null; it cannot simultaneously explain the nulls and
> be ruled out so the nulls can be read as being about transfer. **Ruling out
> plumbing requires a PASS on the endpoint being measured.**
>
> The defective text appears in **three** places — this ADR's
> pre-registration block, the draw receipt's
> `pre_registered_interpretation_recorded_before_the_draw`, and the PC1b
> **workflow report's** outcome text. All three are preserved (append-only)
> and **all three are wrong on this point.**
>
> **The correct reading, by endpoint:**
> - **Likelihood endpoint — VALIDATED**: aligned beats a *norm-matched*
>   Gaussian through the same operator at the same positions by 0.237 nats,
>   198W/102L, p ≈ 2e-8.
> - **Accuracy endpoint — UNVALIDATED**: no control here has moved accuracy at
>   all. **Every rung's verdict rests on this endpoint.**
>
> **Consequences**: M5X (ADR-037) and M4b (ADR-035) **stay BLOCKED**;
> **a FAIL may not be cited as transfer evidence, just as a PASS may not**;
> and the headline is **"we have still not run a passing positive control."**
> See §"THE INVERSION IS REINSTATED" and §"PC1b FINAL".

> ## READING GUIDE (added 2026-08-29) — this document is append-only and out of order
>
> This ADR is the run-2 ladder's living ledger. Sections were appended by
> several concurrent agents and by the coordinator, each inserting above a
> marker, so **document order is not chronological** and a few topics have
> **two sections written independently**. Nothing is deleted (ADR-031's
> append-only rule); this guide is the map.
>
> **Duplicate-topic pairs — both are valid, read both:**
> - **M4g outcome**: the implementer's account (§"M4g outcome … fuse is not
>   the root cause") and the coordinator's (§"M4g OUTCOME … fuse REFUTED as
>   the root cause"). Same verdict; the second adds the power-floor caveat
>   (n_disc = 3, floor 0.125) and the control-semantics disclosure.
> - **M4g registration**: §"M4g PRE-REGISTRATION … fuse instead of overwrite"
>   and §"M4g REGISTERED … overwrite vs fuse — a separate root-cause
>   candidate". The latter was written first, as a candidate; the former is
>   the executed registration.
> - **M4h Stage 1 outcome**: the coordinator's (§"M4h Stage 1 OUTCOME —
>   mechanically successful, statistically unmeasurable") and the
>   implementer's (§"M4h STAGE 1 OUTCOME — honest fail; pooling REFUTED").
>   Same draw; the first emphasises the instrument's power floor, the second
>   the pooling refutation. Both conclusions hold.
>
> **Chronological reading order for the findings** (rather than document
> order): registered confounds → M3 outcome → M4 outcome → M4c outcome and
> engineering findings → M4d outcome → the DIAGNOSIS → M4f pre-check verdict
> → its CORRECTION → M4g registration → M4g outcome → MAJOR CORRECTION (the
> 0/40 inversion is off-manifold-only) → M4h registration → M4h Stage 1
> outcome → LAYER COVERAGE and the methodological reckoning → M4i and M5X
> pre-registrations.
>
> **The corrections are load-bearing, not incidental**: several sections
> correct claims made earlier *in this same document*, including three by the
> coordinator. Where a later section contradicts an earlier one, **the later
> one supersedes** — the earlier text is preserved so the reasoning trail
> stays auditable, per ADR-032's negative-result contract.



- **Status**: Proposed. Milestone M1 (this plan) — M0 (bootstrap scouting) is done and cited
  below; M2 through M6 are not started. This ADR embeds its own pre-registration section (§
  Frozen registration) rather than deferring it to a separate addendum, mirroring ADR-023's S2
  discipline directly in one document since nothing here is cleaner split across two files.
- **Date**: 2026-08-28.
- **Related**: [023](023-live-four-condition-run1-pre-registration.md) (the frozen probe protocol
  this run inherits unchanged, and the negative result this ladder responds to),
  [003](003-causal-edge-verification.md) (ΔV, the feedback signal M5 trains against),
  [002](002-latent-packet-protocol.md) (the align crate's `n>d`/polar-uniqueness reasoning this
  ADR explicitly says does not transfer to nonlinear adapters), [010](010-latentmesh-air-protocol.md)/
  [013](013-esp32-firmware.md) (the edge-device story a small trained sequence translator fits
  better than a full projector), [016](016-ruvector-persistent-latent-memory.md) (HNSW indexing —
  the local-linear-maps idea named but not built here)
- **Evidence base**: [docs/research/025-run1-negative-result.md](../research/025-run1-negative-result.md)
  §7 (the future-work list this ADR formalizes items a-c of), ADR-023's own receipts, and the
  M0 bootstrap scouts, preserved durably at
  [docs/research/026-run2-bootstrap-scouts.json](../research/026-run2-bootstrap-scouts.json)
  (copied into the repo this session — the original workflow-output file lives at
  `/tmp/claude-1000/-home-ruvultra-projects-LatentMesh-1/5e5545f1-6360-406b-b384-6806f2e012c9/tasks/wnunfca2y.output`,
  which is a per-session scratch path that does not survive a reboot; citing only that path would
  repeat exactly the failure mode commit `540c34e` existed to prevent for the S2c token streams,
  so this ADR's primary evidence is copied to a durable location rather than pointed at `/tmp`).
  Two evidence-graded scout reports, `dataShape` and `trainingInfra`, cited by claim throughout;
  every scout claim graded `primary` below was independently measured or fetched this session,
  `inferred` claims are arithmetic extrapolations from measured numbers, `uncertain` claims are
  flagged as such.

## Context

Run 1 (ADR-023, results in § S6) established that a training-free, single global affine map
between pooled residual-stream states carries no recoverable cross-model causal signal at either
registered depth pair for Qwen2.5-3B→1.5B, cross-checked against random noise across two
calibration distributions with bit-verified attribution. `docs/research/025` §7 proposed four
directions in response. This ADR formalizes three of them — **(a)** a small trained nonlinear
projector, **(b)** a FastGRNN-class tiny gated-RNN sequence translator over per-token states, and
**(c)** receiver-side MicroLoRA adaptation from causal-gate ΔV feedback — as an ordered ladder,
plus **(final)** finally computing ADR-023's original A1-A8 once a working adapter exists. Item
**(d)**, HNSW-retrieved local linear maps, is **explicitly deferred** — not built, not gated, not
part of this ladder's milestones; it remains a named future extension beyond run 2's scope,
testing global-vs-local structure only if the nonlinearity hypothesis this ladder tests turns out
insufficient on its own.

Before committing to an architecture or a training pipeline, two bootstrap scouts (M0, this
session) established the concrete facts this plan depends on: what data already exists and what
it would cost to get what doesn't, and whether the proposed training actually runs on this host's
hardware. Both are load-bearing to the decisions below and are cited by claim, not by vibe.

## Frozen registration — what is fixed before any M3+ probe runs

Per the S2-gate discipline ADR-023 established: the following are frozen now, before any trained
adapter is evaluated against the probe that decides pass/fail.

### The frozen probe protocol is inherited unchanged

Run 2 evaluates every ladder rung against **the exact same 40-item S1a/S2b protocol** ADR-023
already ran four times: the same 40 GSM8K-train indices (`item_seed_chacha8: 20897`, indices
`[141, 150, 850, ..., 7462]`, verbatim from `s1a-receipt-slots8-block19-poolfull-rescaletrue-n40.json`
and every `s2b-receipt-*` file), the same one-sided exact sign test at α=0.05, the same 8-slot
placeholder injection, the same rescale-to-natural-median switch, the same greedy/batch=1/max 400
tokens decoding. **The probe is never iterated, re-drawn, or re-tuned to make a rung pass** — this
is the single most important frozen fact in this ADR, restated because a trained adapter creates a
much larger temptation to retry against the eval set than a closed-form linear fit ever did. Any
rung that fails is reported, preserved, and the ladder moves to the *next architecture*, not a
retried hyperparameter sweep of the same one against the frozen probe (see § Ladder rungs below
for the one disciplined exception: FastGRNN's rank is treated as three ordered sub-rungs, tried
once each, all three receipts kept regardless of outcome).

### Data pipeline — no regeneration

**Finding (primary, M0 data-shape scout):** the existing S2c dumps are pooled, one vector per
item — `sender_L18/L24.f32bin` = 2,560 × 2,048 × 4 B, `receiver_L14/L19.f32bin` = 2,560 × 1,536 ×
4 B, confirmed by exact byte arithmetic against `receipts/s2c-manifest.json`. But the *per-token*
states these pooled vectors were averaged from already exist implicitly: `forward_capture_multi`
(`crates/latentmesh-runtime/src/capture.rs:85-123`) materializes a full `[T × dim]` rows tensor
before calling `rows.mean(1)` at line 123 to produce the pooled vector — the per-token capture
this ladder needs is a small capture-module change (return the rows tensor alongside the pooled
one), not a redesign.

**Finding (primary): no fresh generation is needed at all.** `harness/latentmesh-live/data/s2c-token-streams.jsonl`
(sha256 `ad5713116cb5acf663fe9c33ac58aaedb1fd64fcf629dd78b21439d244958539`, committed at `540c34e`,
verified by re-hashing the committed file this session) already holds every one of the 2,560
items' full prompt and generated token sequences (greedy, batch=1, max 400 new tokens). Per-token
paired capture is therefore pure **teacher-forced prefill over the saved streams through both
models** — the same convention the pooled S2c dump already used — and the per-item
tokenizer-parity gate (`token_streams_identical_across_models: pass`, every relevant receipt)
guarantees identical token count `T_i` across sender and receiver per item, which is what makes a
single shared offsets index valid for all four per-token dump files.

**Measured token totals** (primary, cross-checked against `s2c-generated-dump-receipt.json
rows.generated_tokens_total=719115`): 311,679 prompt tokens (mean 121.7), **719,115 generated
tokens** (mean 280.9 — not the ~200 a naive estimate might assume), 1,030,794 combined, across
2,560 items.

**Format decision (frozen):** capture the **generated span only** (matches both the existing
pooling convention and the live transmit distribution) — ragged, concatenated `[T_i × dim]` f32
blocks per layer (`sender_L18.tok.f32bin`, `sender_L24.tok.f32bin`, `receiver_L14.tok.f32bin`,
`receiver_L19.tok.f32bin`), plus **one shared** binary offsets index (cumulative `u64` token
offsets, item indices, prompt/gen lengths, per-item sha256s) — valid across all four files because
`T_i` is per-item identical across models. Disk cost: sender 719,115 × 2,048 × 4 B × 2 layers =
11.78 GB; receiver 719,115 × 1,536 × 4 B × 2 layers = 8.84 GB; **≈20.6 GB total** (347 GB free on
`/`, primary measurement). Consumption: `memmap2` zero-copy, byte offset = token offset × dim × 4
(always 4-byte aligned). Rejected: safetensors with one tensor per item (2,560×4 tensors, header
bloat, no benefit over the raw concatenated layout the repo's C/Rust codecs already use elsewhere).
**Estimated capture wall-clock: ~5-15 minutes** (inferred: the receiver teacher-forced phase for
all 2,560 items measured 57.3 s in the S2c dump; the sender phase scales ~2× for FLOPs, plus ~20.6
GB of large-buffered writes — this is an extrapolation, not a fresh measurement, and is named as
such). This is dramatically cheaper than the ~2.5 GPU-h the original S2c *generation* cost,
because generation is avoided entirely — only prefill is needed.

**Risk carried forward, not yet closed:** teacher-forced prefill states are not verified
bit-identical to true incremental-decode states in BF16 — the repo's existing parity gates compare
prefill-vs-prefill only. The pooled S2c asset already relies on this same convention, so per-token
prefill capture is internally consistent with everything upstream of it, but the live transmit
path at generation time uses incremental decode, and nobody has measured the gap. **Graded
uncertain** (M0 scout); flagged here, not resolved, and not blocking M2 — M3's actual probe
(evaluating the trained adapter live, injected during real generation) is itself the empirical test
of whether this gap matters.

### Leakage discipline — the load-bearing new rule this ADR adds

**Finding (primary):** exactly 13 of the 40 frozen probe items sit inside the S2c 2,560-row
dataset, at (row, item) pairs `(69,150) (683,1309) (824,1573) (1258,2365) (1281,2418) (1346,2540)
(1577,2958) (1586,2973) (1647,3084) (2052,3825) (2058,3844) (2078,3877) (2529,4746)` — computed by
exact set intersection of the receipts' `dataset.indices` against the S2c manifest's
`item_indices`, this session. 22 of 40 sit in the full `calibration-4000`.

ADR-023 already disclosed a related non-disjointness (§ Dataset pins) but ruled it low-risk
**only in the linear-fit context**, because a closed-form affine fit consumes no per-item
statistic the probe re-estimates. **That ruling does not transfer to run 2.** A trained adapter
evaluated by the same 40-item probe *would* have memorization-shaped leakage through the 13 shared
items if they were left in the training set — this is a genuinely different failure mode
(overfitting to items the probe will later re-test) than anything the linear fit could exhibit.

**Frozen rule:** for M3 onward,

1. Apply `fit_holdout_split(2560, FIT_SPLIT_SEED=0x24C0_DE03)` **first** — the exact seeded 80/20
   split `harness/latentmesh-live/src/calibrate.rs:34,115` already uses for S2/S2c, to preserve
   frozen-seed comparability with the prior calibration work.
2. **Then** drop the 13 probe-overlap rows from whichever side of the split they land in.
   Recorded, per row, in every M3+ training receipt.
3. Split **by item, never by token** — tokens within one item's generated span are correlated;
   a token-level split would itself be leakage.
4. **Do not draw a fresh 40-item probe set.** The frozen protocol from ADR-023 is reused verbatim
   (§ above). Changing the probe set is explicitly named as an ADR-023 deviation requiring its own
   registration, not a decision this ADR or any training run makes unilaterally.
5. Capture **all** 2,560 rows at M2 regardless — the 13 probe-overlap rows are excluded in the
   **training loop**, not at capture time, so the data asset stays complete and the exclusion
   stays auditable (anyone can re-derive the training set from the full capture plus the excluded
   list, rather than trusting a pre-filtered file no one can check).
6. **`eval-200`/`holdout-100` remain mechanically locked**, unchanged from ADR-023: `eval_items()`/
   `holdout_items()` refuse until a genome-frozen receipt exists (`receipts/s2-splits-receipt.json
   eval_holdout_lock`). Run 1 never produced that receipt — the Deviation-7 stop rule fired before
   any genome was frozen — so the lock is still fully engaged at the start of run 2. **M6 is the
   milestone that must produce the genome-frozen receipt** (it runs the four-condition Darwin loop
   ADR-023 specified but never reached); until M6 does, M2 through M5's training and evaluation
   data is `calibration-4000`/`adaptation-512`-derived only, by construction, same as run 1.

**Disclosed consequence:** excluding 13 rows drops `n_fit` to the scout's expected range of
**≈2,037-2,048** (the exact value depends on where the 13 rows land in the frozen 80/20 shuffle;
the true floor, if all 13 happened to land on the fit side, is 2,035 — stated for completeness,
not because it changes anything below), which is at or below `d_sender=2,048` — the exact boundary
ADR-023's § S6 already flagged for the S2c generated-pairs fit. **This ADR states explicitly: the
linear `n>d` identifiability gate does not carry the same meaning for the MLP or FastGRNN adapters
below** — it was a Procrustes-specific requirement for a unique semi-orthogonal solution, not a
general sample-size floor for gradient-trained nonlinear models, which routinely train with far
more parameters than paired examples. It is disclosed here as a fact about the dataset, not
imported as a gate.

**One freeze deferred to each rung's own training receipt, by design (mirroring ADR-023's S2
`content_hash` placeholder):** this ADR fixes each rung's *architecture* (M3's MLP shape, M4's
FastGRNN cell and rank ladder) but not its training seed, epoch budget, or stopping rule — "trained
once" is only meaningful once those are pinned. Each rung's training receipt must record its RNG
seed and its stopping criterion (e.g., early-stopping on the held-out 20% split, or a fixed epoch
count) **before** that rung's frozen probe is invoked, exactly as S2's `content_hash` was pending
at ADR-023's authoring and appended once known — the probe-invocation order is the actual freeze
point, not this document.

## Ladder rungs, with per-rung gates

Each rung is architecturally distinct (not a hyperparameter variant of the previous one), trained
once against a frozen configuration, evaluated once against the frozen probe, and reported
regardless of outcome — pass or fail, the receipt is kept, mirroring ADR-023's preserved S1a
run-1-failure receipt. A rung's failure escalates the ladder to the *next architecture*; it is not
grounds to retry the same architecture with different hyperparameters against the frozen probe.

### M3 — MLP projector (the Cache-to-Cache-style baseline)

**Architecture (frozen):** 2-layer MLP, 2048→512→1536, ReLU, **1,837,056 parameters**
(7.0 MB f32) — the exact configuration measured by the M0 training-infra scout, not a design-time
guess. **Measured training result (primary, synthetic teacher, this session):** AdamW lr=1e-3,
batch 256, MSE loss 2.817→0.593 over 300 steps and still decreasing; mean step time 0.94 ms (p50
0.92 ms, first 10 steps discarded for CUDA warmup/kernel-init noise); the 0.593 floor is a
512-hidden bottleneck against a full-rank *synthetic* linear teacher, not a training failure —
real hidden-state pairs have unknown structure and this number does not predict real-data
convergence quality (named as a limit of the scout's synthetic-data probe, not asserted as the
real result).

**Two evaluated variants**, both against the frozen probe: (i) the MLP as a **per-token
translator**, run per generated token, feeding the *existing* pooled-injection path (i.e., the
adapter translates each token's state, then the existing mean-pooling and 8-slot injection
mechanics from ADR-023 apply on top of the *translated* per-token stream); (ii) a **pooled-in/
pooled-out** variant — pool the sender's per-token states first (as run 1 already did), then run
the *pooled* vector through the MLP, matching run 1's pipeline shape exactly except for the
transform itself.

**Authorial choice, disclosed (the source brief specifies both variants but not whether they share
weights):** both variants use the **same** MLP, trained once on per-token pairs per (i)'s framing
— (ii) applies that per-token-trained network to a pooled input at inference time, rather than
training a second MLP on separately pooled pairs. The alternative (fit a second MLP on ~2,547
pooled pairs — one per training-set item, after the leakage exclusion) is badly data-starved
against 1,837,056 parameters and was rejected on that basis, not tested. **This choice does not
cleanly isolate "pooling destroyed run 1's signal" from "the network doesn't generalize to an
off-distribution input it never trained on"** — an MLP trained exclusively on per-token states,
then evaluated on a pooled vector, confounds the two: a variant-(ii) failure could mean pooling is
genuinely destructive, or could simply mean the pooled vector is out-of-distribution for a network
that never saw one during training. This ADR states the confound rather than claiming the two
variants cleanly separate pooling from linearity, and treats variant (ii)'s result as suggestive,
not decisive, on the pooling question — a properly isolated test (a second MLP trained on pooled
pairs, data constraints permitting) is named as follow-up work if variant (ii) alone proves
ambiguous.

**Gate:** aligned-real (MLP output) > random, one-sided exact sign test, p<0.05, on the frozen
40-item probe — for each variant separately. **Training data:** per-token pairs from M2's capture,
filtered per the leakage rule above (`n_fit ≈ 2,037-2,048` after exclusion, disclosed per-row in
the training receipt).

### M4 — FastGRNN sequence translator

**Cell equations and parameter formula (primary, Kusupati et al., NeurIPS 2018, arXiv:1901.02358;
cross-checked against the reference implementation `microsoft/EdgeML`
`pytorch/edgeml_pytorch/graph/rnn.py FastGRNNCell`, fetched and quoted this session):**

```
pre     = W x_t + U h_{t-1}                    (W, U SHARED between gate and candidate)
z       = sigmoid(pre + b_z)
h_tilde = tanh(pre + b_h)
h_t     = z * h_{t-1} + (sigmoid(zeta)*(1-z) + sigmoid(nu)) * h_tilde
```

with trainable scalars `zeta`, `nu`, and low-rank factorization `W = W1 @ W2` (`W1: [D_in, r_W]`,
`W2: [r_W, D_h]`), `U = U1 @ U2` (`U1: [D_h, r_U]`, `U2: [r_U, D_h]`). Parameter count formula:
`r_W(D_in + D_h) + 2·r_U·D_h + 2·D_h + 2`, independently verified this session — the fetched
formula and a from-scratch Rust implementation over candle ops agreed exactly at `D_in=2048,
D_h=1536, r=64`: **429,058 parameters, predicted and measured identically.**

**Measured training result (primary, this session, custom cell — no candle patches needed):**
`r=64` (429,058 params, 1.6 MB): loss 0.444→0.341 at 9.5 ms/step, batch 64 × sequence 16. `r=256`
(1,707,010 params, 6.5 MB): loss 0.570→0.396 at 9.8 ms/step, monotonically decreasing, no stall.
`r=128` (855,042 params, 3.3 MB) was computed by formula but not separately probed this session.
`candle-nn 0.9.2` also ships `GRU`/`LSTM` in `rnn.rs` as a documented fallback if the custom cell
proves troublesome downstream — not needed per the M0 evidence, kept as a named contingency.

**Frozen sub-rung order (the one disciplined exception to "one architecture, one probe run"):**
`r ∈ {64, 128, 256}`, tried in ascending order. Each rank is trained once and evaluated once
against the frozen probe; if `r=64` clears the gate, M4 stops there and reports it; if not,
`r=128` is tried; if not, `r=256`; **all attempted ranks' receipts are committed regardless of
outcome** — this is not a sweep-until-pass, it is three pre-declared architectures tried in a
pre-declared order with every result kept, the same discipline that kept S1a's run-1 failure
receipt in ADR-023.

**Gate:** same frozen probe, p<0.05, at whichever rank first passes (or a documented all-three-fail
report). **Comparison vs. M3:** report FastGRNN's best-passing (or best-failing, if none pass)
result alongside M3's, since M4 tests a materially different hypothesis (per-token sequence
structure vs. M3's per-token-or-pooled but still position-independent mapping) — a rung that fails
the probe less badly than the previous rung is still a reportable finding under ADR-023's A8-style
no-rerun-but-do-report discipline.

**Training data:** the same per-token capture as M3, but consumed as sequences (batch × seq-window)
rather than independent pairs, respecting item boundaries (no cross-item sequences).

### M5 — receiver-side MicroLoRA adaptation from causal-gate ΔV feedback

**In-stack prior art, verified present on this host by direct read (not assumed from a name):**
`@ruvector/sona` MicroLoRA — rank 1-4 hidden-state adapters, <50 KB each, real-time `adapt()`
update from a feedback signal (`~/ruvector-upstream/crates/ruvllm-wasm/docs/MICRO_LORA.md`); and
`AdaptiveEmbedder`'s EWC++ consolidation plus memory-augmented retrieval
(`~/ruvector-upstream/npm/packages/ruvector/src/core/adaptive-embedder.ts`, `class
AdaptiveEmbedder`, Fisher-information-based EWC regularization confirmed present in the source).

**Decision:** rather than a fixed offline-trained transform (M3/M4's paradigm), M5 trains a
small rank-1-4 adapter on the receiver side using ADR-003's own `ΔV` (from
`latentmesh-gate::causal::verify_edge`) as the online feedback signal — closer to the design's
causal-gate spirit than a static fit, and the first rung that actually exercises the causal
admission machinery live rather than as a downstream validation step.

**Gate (this ADR's own extrapolation of the mission's stated pattern, flagged as authorial, not
literally specified in the source brief):** the same frozen 40-item probe, with `aligned_real` now
meaning the MicroLoRA-adapted receiver's output, compared against M3/M4's best result as well as
against random. **M5 is explicitly the least fully specified rung in this ADR** — it is the only
one requiring a live `verify_edge` loop wired to an online adaptation step, which is more
architecturally novel than M3/M4's offline supervised fits and was not scouted by M0 the way M3/M4
were. A follow-up scouting pass on the ΔV-feedback training loop's mechanics (what triggers an
`adapt()` call, at what cadence, against what batch) is named as required work before M5 can be
pre-registered as tightly as M3/M4 are here.

### M6 — the original four-condition run, finally computed

If any rung (M3, M4, or M5) clears the frozen-probe gate, its adapter becomes the injection
mechanism for ADR-023's original four-condition comparison (`StaticText`/`DynamicText`/
`DynamicLatent`/`CausalDynamicLatent`), computing **A1-A8 for the first time** — the compute/
latency and edge-survival clauses run 1 never reached. If no rung clears the gate, run 2 reports
a second negative result, at the ladder's cost, with the same evidence-label and no-rerun
discipline as run 1's — this is the honest-fail path the mission brief requires for every rung,
extended to the ladder as a whole.

## Infrastructure

### New standalone crate `crates/latentmesh-train`

Same pattern as `latentmesh-runtime` (ADR-023 Deviation 1), for the identical reason: **workspace-
excluded**, own empty `[workspace]` table, `rust-version = "1.81"`, `cuda` feature off by default.
**Cargo.lock copied verbatim from `latentmesh-runtime`'s** (`candle-core`/`candle-nn` 0.9.2,
`cudarc` 0.17.8 + 0.19.9, the exact proven pin combination the M0 training-infra scout reused
without a single dependency-resolution error) — not re-resolved from scratch, to avoid pulling an
unproven point release against nvcc 12.8/sm_120. A `build.rs` nvcc-12.8 guard mirrors
`latentmesh-runtime`'s.

**Frozen optimizer constraint (primary, M0 training-infra scout, read directly from
`candle-nn-0.9.2/src/optim.rs`): `candle-nn` 0.9.2 ships exactly `SGD` and `AdamW`, no plain
`Adam`/`RMSprop`, and no learning-rate scheduler** — every rung above uses `AdamW` (the M0 probe's
own choice, matching the only reasonable option available); a warmup/decay schedule, if ever
wanted, has to be hand-rolled by reconstructing `ParamsAdamW` between steps, since `candle-nn`
0.9.2 exposes no `set_learning_rate` call.

**`latentmesh-train` path-depends on `latentmesh-runtime` directly.** Verified this session: root
`Cargo.lock` pins `half=2.4.1`, `latentmesh-runtime`'s own `Cargo.lock` pins `half>=2.5` — the
conflict is specifically between the root workspace and any candle-consuming crate, **not** between
two candle-consuming crates, which share the same `half>=2.5` requirement and don't conflict with
each other. **What is verified is pin compatibility across both lockfiles; the cross-crate path
dependency itself has not been built** — that build is M2's first concrete check, not asserted
here as already working. This is a materially different relationship than `harness/latentmesh-live`'s to
`latentmesh-runtime`: ADR-023's "Correction to the earlier workspace ruling" established that
`harness/latentmesh-live` **stays a root-workspace member** and therefore **must** cross to
`latentmesh-runtime` only via files and subprocess invocation, never a Cargo dependency.
`latentmesh-train` is **not** a root-workspace member — it is excluded exactly like
`latentmesh-runtime` — so it is free to link `latentmesh-runtime` directly for M6's live-injection
binary (which needs both model loading/generation *and* trained-adapter inference in one process,
to inject at each decode step). M2's per-token capture code, by contrast, lives in
`latentmesh-runtime` itself (extending `capture.rs`), since it only needs model forward passes, not
training.

### Optional MetaHarness gates at each milestone

Per ADR-022's established pattern: `metaharness_score`/`metaharness_threat_model` (the
`ruflo-metaharness` MCP surface) are invoked as **optional, subprocess/MCP-invoked gates**, never
a build or runtime dependency of `latentmesh-train` or `latentmesh-runtime`. Run once per milestone
(M2 through M6) when the tooling is available; a milestone's own receipt still stands without it if
the tooling is absent — the workspace stays hermetic (`cargo test --workspace` never depends on
this tooling's presence) exactly as ADR-022 requires for the Meshtastic/agentbbs/cognitum
integration wave.

### The codex auxiliary lane — a hard boundary, not a convenience

**Verified working this session:** `codex exec --sandbox read-only --skip-git-repo-check
"<prompt>"` (codex-cli 0.149.0; `codex exec`, not `codex -p`; `--skip-git-repo-check` required
outside a trusted/git directory; ChatGPT-login auth, not API-key billing — subject to plan rate
limits, not per-token cost; a no-op test returned `OK` using 7,179 tokens).

**Usable for**: code review assistance, boilerplate generation, drafting — anything where a wrong
or hallucinated output is caught by compilation, review, or a downstream test. **Never usable
for**: anything that touches receipts, statistics, seeds, or evidence labels. Codex output is
never the source of a number that appears in a training receipt, a probe p-value, a parameter
count reported as measured, or any A1-A8 computation. This is a hard boundary, analogous to
ADR-011's separation between a codec and permission to radiate — codex may help write the code
that produces evidence; it may never itself be the evidence.

### Training receipts

Every M2-M6 artifact gets a JSON receipt in the same style as ADR-023's: `evidence_label` (e.g.
`"live-model, single-host, simulation-free"` for anything a model touched, `"deterministic CPU
training over captured per-token pairs"` for the fit stages themselves), explicit seeds (training
RNG seed, `FIT_SPLIT_SEED=0x24C0_DE03` reused for the 80/20 split, the frozen probe's own
`item_seed_chacha8: 20897`), the excluded-13-rows list, `git_commit`, GPU/nvcc environment, and
wall-clock/GPU-seconds — nothing summarized without the underlying receipt to check it against,
matching § S6's own "nothing is estimated" discipline.

## Numbers rule

Every number in this ADR traces to either a committed receipt, a re-verified file hash, or a named
scout-evidence entry graded `primary`/`inferred`/`uncertain` in the M0 workflow output. No
remembered or approximated figure appears without that citation — this is the same horizon-tracker
drift guard ADR-023 §6 already applied to its own GPU accounting, extended here to every number in
a plan document, not just a results document.

## Milestones

| Milestone | Content | Status |
|---|---|---|
| M0 | Bootstrap scouting: data-shape scout, training-infra scout | **Done** — evidence cited throughout this ADR |
| M1 | This ADR (the plan, including its embedded pre-registration) | **This document** |
| M2 | Per-token paired capture (`capture.rs` extension, ragged f32 dump + shared offsets index); leakage-safe split machinery | Not started |
| M3 | MLP projector, both variants, against the frozen probe | Not started |
| M4 | FastGRNN sequence translator, `r∈{64,128,256}` ordered sub-rungs | Not started |
| M5 | Receiver-side MicroLoRA from ΔV feedback | Not started — least specified, needs its own scouting pass first |
| M6 | Full four-condition run, A1-A8 computed (or a second negative result) | Not started — gated on M3/M4/M5 producing a passing adapter |

## M3 outcome (2026-08-28, appended per ladder discipline)

**Honest fail, both variants, adversarially confirmed** (commit 14e2af1;
receipts `run2-m3-receipt-cellL18toL14-mlp-{pertoken,pooled}-*.json`).
The MLP learned substantial per-token structure (holdout MSE 0.179, relative
residual 0.461 vs the 0.843 mean-predictor baseline) yet the frozen probe
found no causal use: per-token p=0.6875 (aligned 21/40 vs random 21/40),
pooled p=0.5000 (22/40 vs 21/40); zero-vector and NLL secondaries normal;
all integrity gates (artifact hash, golden pairs at 4.5e-7, item-set
reproduction) passed. The anchor cell was deliberately NOT probed (an
unregistered extra draw). Reading: regression fit does not buy causal
transfer — now shown for a trained nonlinear map, scoped by the registered
receiver-scale confound above. Ladder proceeds to M4 (sequence translation)
per this ADR's fail path.

> **ANNOTATION (2026-08-29, appended post-hoc against `docs/research/031-statistical-power-and-design.md`
> — does NOT change the recorded verdict above).** Both M3 draws landed in the exact sign
> test's power-floor dead zone: per-token variant had `n_disc=4` (the minimum attainable p at
> `n_disc=4` is 0.0625, above α=0.05 — the test could not have rejected the null regardless of the
> true effect size); pooled variant had `n_disc=3` (minimum attainable p 0.125, likewise above
> α=0.05 unconditionally). **This means both M3 "FAIL" verdicts are more precisely described as
> "the test could not have detected an effect at this discordant-pair count" than as "no effect
> exists"** — a distinction this repo's own subsequent power analysis surfaced, not one apparent at
> the time M3's outcome was first reported. Secondary, non-authoritative re-analysis: recomputed
> mid-p McNemar statistic (Fagerland, Lydersen & Laake 2013) on the same collected pairs —
> per-token 0.6875 → **0.5000** mid-p; pooled 0.5000 → **0.3125** mid-p. Both remain clean fails
> under the mid-p statistic too; this recomputation does not flip either verdict and is reported as
> a post-hoc secondary annotation, not a re-test, not a changed verdict, and not grounds to re-probe
> M3. See ADR-030's amended Acceptance criteria section for how this structural floor is handled
> prospectively in run 3's own design.

## M4 outcome (2026-08-29, appended per ladder discipline)

**Honest fail, all three sub-rungs, adversarially confirmed** (receipts
`run2-m4-receipt-cellL18toL14-fastgrnn-r{64,128,256}-*.json`; r256 probe
re-run bit-identical). Primary: r=64 p=0.3125, r=128 p=0.3125, r=256
p=0.1875 (4W/1L, aligned 24/40 vs random 21/40 — least-bad, above α).
FastGRNN learned structure (holdout rel residual 0.704/0.663/0.633 vs
0.843 baseline, weaker fit than M3's MLP 0.461) and, like M3, fit bought
no causal transfer. One pre-probe superseded r=64 training run (diverging
window-zero-init scheme, caught by the holdout metric before any probe
invocation, preserved with -superseded suffix, disclosed in receipt).
**Per the pre-verdict interpretive rule above, this is NOT evidence against
sequence structure — the reconstruction-loss confound applies identically.
The registered M4c (task-loss ablation) is now MANDATORY.**

> **ANNOTATION (2026-08-29, appended post-hoc against `docs/research/031-statistical-power-and-design.md`
> — does NOT change the recorded verdicts above).** r=64 and r=128 both landed at `n_disc=4` — the
> same power-floor dead zone as M3's per-token draw (minimum attainable exact-sign p at `n_disc=4`
> is 0.0625, above α=0.05: the test could not have rejected the null at either rank regardless of
> the true effect size). r=256's `n_disc=5` sits just above that floor (minimum attainable p 0.0312,
> reachable only at 5/5 wins) but the observed 4/1 fell short of it. **Three of the ladder's ten
> valid probe draws to date are now known to have been structurally incapable of reaching
> significance (M3's per-token variant, M4 r=64, M4 r=128) — these "FAIL" verdicts are more
> precisely "the test could not have detected an effect at this discordant-pair count" than "no
> effect exists."** Secondary, non-authoritative re-analysis: recomputed mid-p McNemar statistic
> (Fagerland, Lydersen & Laake 2013) on the same collected pairs — r=64 0.3125 → **0.1875** mid-p;
> r=128 0.3125 → **0.1875** mid-p; r=256 0.1875 → **0.1094** mid-p. All three remain clean fails
> under the mid-p statistic too; none of these recomputations flip a verdict, and this is reported
> as a post-hoc secondary annotation, not a re-test, not a changed verdict, and not grounds to
> re-probe any M4 rank. The parallel discordant-pair floor across S2b's own four draws (`n_disc`
> 3, 3, 4, 3 — every one in the dead zone) is documented in full, alongside all 10 valid draws, in
> `docs/research/031` §1's table; that same document's §2.3 headline finding — that the ladder's
> own observed ~10% discordance rate implies roughly 25-30 discordant pairs are needed for real
> power at a plausible effect size, five to six times what any draw here has produced — is the
> honest scale context for reading every FAIL verdict in this ADR, past and future, and is why
> ADR-030 (run 3) adopts an anytime-valid sequential test rather than repeating this fixed-N
> 40-item design as its own primary statistic.

Execution
order decision: M4c runs before M4b (it reuses all existing capture and
machinery; M4b needs fresh 3B-receiver calibration + its own
pre-registration addendum; both remain mandatory).

## M4c outcome (2026-08-29, appended per ladder discipline)

**Honest fail on the frozen probe — but a POSITIVE training/transfer result, and
the sharpest dissociation this ladder has produced** (receipts
`run2-m4c-training-receipt-cellL18toL14.json`,
`run2-m4c-transfer-receipt-cellL18toL14.json`,
`run2-m4c-receipt-cellL18toL14-mlp-taskloss-slots8-poolfull-rescaletrue-n40.json`).

Architecture per this ADR's "same or best-so-far" rule: the M3 MLP
(2048→512→1536 ReLU), chosen as best-so-far by holdout fit (rel residual 0.461
vs FastGRNN's 0.633/0.663/0.704), from a FRESH seeded init so the ablation
isolates exactly one factor — the loss. Single seeded run, 10 epochs, batch 1,
AdamW lr 1e-3, 2,035 fit / 509 holdout items after the same frozen leakage rule
and the same 13 exclusions; 1,604 s GPU, 10,322 MiB process peak.

1. **The task loss trained, well.** Holdout task CE 0.2546 (seeded init) →
   **0.1595** at best epoch 4 (later epochs overfit: train 0.0884 / holdout
   0.1846 by epoch 9 — the best-holdout rule selected 4).
2. **The improvement TRANSFERS across the registered composed↔fused BF16 gap.**
   The pre-registered transfer check (inference-only, vendored fused forward,
   holdout items, no probe draw) measured mean fused NLL **0.1570 trained vs
   0.2385 init, 498 wins / 11 losses** (secondary sign p=8.1e-132). The
   registered caveat is therefore RETIRED, not merely disclosed: the numeric gap
   does not confound what follows.
3. **The frozen probe still nulls.** Aligned 23/40, baseline 22/40, zerovec
   24/40, random 21/40; primary aligned>random **p=0.3438** (4W/2L), α=0.05 —
   fail. Zerovec gate passed; all integrity gates (artifact hash, golden pairs,
   S1a item-set reproduction, transfer-gate-before-probe) passed.
4. **NOT the M3 power-floor artifact.** M4c drew `n_disc=6`, whose minimum
   attainable p is 2⁻⁶ = 0.0156 < α — unlike both M3 draws, this test *could*
   have rejected. The null is a real non-detection, not a structural dead zone.
5. **The dissociation, stated plainly.** The adapter's aligned injection made
   the receiver's gold-answer continuation dramatically *less* likely: mean NLL
   of `#### <gold>` was **5.359 aligned vs 2.129 baseline / 2.117 random /
   2.154 zerovec — 0 wins, 40 losses vs random (p=1.0)**. The adapter learned,
   verifiably and transferably, to steer the receiver toward reproducing the
   *sender's generated token span* — which is exactly what C2C-style task loss
   on that span optimizes — and that objective is not the answer-format
   objective the probe scores. Accuracy was untouched (23 vs 21) while the
   answer-token likelihood collapsed.

**Reading.** The registered hypothesis was "reconstruction loss was the wrong
loss." M4c shows that swapping in the task loss produces a genuinely, strongly
optimized channel (0.2546→0.1595 holdout CE, transferring at 498/11) that still
buys no probe-measured causal benefit — so the parsimonious reading is no longer
"still the loss function." The candidate that replaces it is a **target
mismatch**: next-token CE on the sender's span rewards mimicking the sender's
chain-of-thought surface, and the resulting steer actively fights the answer
format. This is a new, mechanistically specific finding, and it is the first
rung where the adapter demonstrably learned something causal about the
receiver's behaviour at all. It does not overturn either registered confound —
the receiver-scale threshold (M4b, still mandatory) and the epiphenomenal-
correctness framing both stand, and the sub-1.7B receiver scoping applies here
identically.

**Ladder position.** M4c is discharged (it was mandatory; it ran once, honestly,
with its transfer gate passing first). M4b — the ≥1.7B receiver scale-control
arm — remains mandatory and is the next rung; it needs fresh calibration and its
own pre-registration addendum. ADR-027's prefix-level delivery gains support
from the NLL dissociation above (a delivery path that cannot fight the answer
format is a different failure surface), but activating it still requires its own
addendum frozen before any probe.

## M5 SUPERSEDED AND REDESIGNED (2026-08-29) — the original sketch is infeasible

A dedicated scout ([docs/research/034](../research/034-m5-receiver-side-adaptation-scout.md))
found this ADR's own M5 paragraph describes something that cannot be built as
written. Recording the correction rather than quietly working around it:

- **ΔV cannot be an online training signal.** One properly-powered
  `verify_edge` draw (the ~25-30 discordant pairs research-031's power
  analysis requires) costs roughly **3 GPU-h — more than run 1's entire
  budget — for a single feedback point**, and repeated probe-style draws as
  a training fitness function are explicitly forbidden by
  [ADR-028](028-evolutionary-adapter-search-anti-gaming.md)'s protected list,
  which postdates this paragraph. **Corrected design**: task-loss training
  (cheap, per-step) with ΔV computed **once, post-training, as gate and
  diagnostic only** — never in the loop.
- **The cited in-stack prior art does not do what this ADR assumed.**
  `@ruvector/sona`'s MicroLoRA `adapt()` was read in full (736 lines of Rust,
  not the JS wrapper): it is a hand-rolled Hebbian/REINFORCE delta rule keyed
  on a scalar quality score, self-described as working "without full
  backprop", with no connection to any loss or to the receiver's forward
  pass. Architecturally right shape, unusable as training machinery. M5 needs
  a real candle LoRA module in `latentmesh-train` (small lift — M4c proved
  backprop through the full 1.5B receiver works).
- **M5 breaks an invariant every earlier rung relied on.** M3/M4/M4c/M4d all
  held the receiver's weights identical, so only injected content varied.
  M5 changes the receiver itself, so a LoRA trained on task loss could simply
  become a general GSM8K fine-tune that helps regardless of what is injected
  — ADR-001 §5's confound, one layer up. **Mandatory**: baseline, zero-vector
  and random conditions must be **re-measured fresh under the same adapted
  receiver** inside M5's own receipt; comparison against earlier rungs'
  frozen-receiver baselines is invalid for causal-gate purposes (cross-rung
  *reporting* comparisons remain fine, per the existing M3-vs-M4 convention).
- **Corrected spec**: rank-{1,2,4} sub-rungs, LoRA at the L14 injection site,
  frozen upstream translator (M4d's artifact if it reports, else M4c's),
  training target changed from sender-span CE to **gold-answer CE**, the
  deploy-transform-in-loop inherited from M4d, mid-p McNemar primary.
  ~1.2-2.0 GPU-h per rank, ~3.6-6.0 GPU-h for all three.
- **Prior-art status**: a purpose-built survey (arXiv:2606.05711) states
  receiver-side parameter adaptation is outside the design space of every
  latent-communication method it taxonomizes — raising M5's novelty *and* its
  risk, since there is no training recipe to borrow.
- **Sequencing**: after M4d reports. If M4d passes, M5 is optional
  comparative work; if M4d nulls, M5 is the most mechanistically distinct
  remaining hypothesis in the mid-layer family. M4b stays mandatory and
  independent either way. M5 requires its own pre-registration addendum
  before any probe draw.

## M4c engineering findings (recorded 2026-08-29 from the feasibility+implementation receipts)

Three facts worth preserving for anyone reproducing this work, and one that
narrows the M4c diagnosis:

1. **candle 0.9.2's inference forward silently cuts the gradient graph.**
   Measured: injecting a `Var`, running the vendored forward, computing CE and
   calling `backward()` succeeds but `grads.get(var)` returns `None`.
   Source-verified cause: `candle_nn::ops::softmax_last_dim`,
   `ops::rms_norm` and `rotary_emb::rope` are `apply_op*_no_bwd` — the op is
   never recorded — and the final `rms_norm` severs the residual skip path.
   `slice_assign` itself is differentiable. Fix used: a training-shaped
   forward variant substituting exactly three call sites (`rms_norm_slow`,
   a composed softmax with detached max, a composed rotate-half), no candle
   patching. F32 parity 128/128 argmax vs the vendored forward.
2. **F32 training of the 1.5B receiver does not fit 16 GB** — measured OOM at
   seq len 64, because candle 0.9.2's backward materialises frozen-weight
   gradients (no `requires_grad` gating). BF16 fits with ~4.7 GB headroom
   (9.9 GB process peak at L=256, 81 ms/step). candle 0.9.2 has no
   gradient-checkpointing API.
3. **Dropping a candle model does not return VRAM to the driver mid-process**
   (12,624 MiB retained after drop) — trainer and probe must be separate
   processes, which is why the harness runs them that way (see ADR-034).

**Why this narrows the M4c diagnosis**: training ran through the composed
BF16 forward while the frozen probe runs the vendored *fused* BF16 forward,
and those differ numerically (116/128 argmax agreement, max |Δlogit| 8.19 at
L=128). That confound was registered and then *closed* before the probe: the
`transfer_check_passed_before_probe` gate evaluated the trained vectors'
teacher-forced NLL **through the vendored fused forward** and measured
0.2385 → 0.1570. So the improvement demonstrably survives the numeric gap.
The 0/40 probe-time NLL inversion therefore cannot be attributed to
composed-vs-fused numerics — it is specific to the **deployment
configuration**, which is exactly what M4d isolates.

## M4d outcome (2026-08-29) — null, exactly as pre-registered

`run2-m4d-receipt-cellL18toL14-mlp-deploymatch-*-n40.json`. **Numbers
corrected 2026-08-29 from the workflow's full report** — my first entry
conflated two different comparisons. The **primary** gate (aligned > random)
is **5W/2L, n_disc = 7, exact p = 0.2266, mid-p 0.1445 — FAIL**. (The 0.3438
I originally recorded as "exact-sign" is the *aligned vs baseline*
comparison, not the primary.) Accuracy: aligned 24/40, baseline 22, zerovec
24, random 21.

**This null is NOT a power-floor artifact — and it is the ladder's first that
isn't.** At n_disc = 7 the minimum attainable one-sided p is 2⁻⁷ = 0.0078,
comfortably below α = 0.05, so this draw **could have rejected** and did not.
Every earlier null sat at n_disc ∈ {3,4,5} where rejection was wholly or
nearly unattainable (see the power annotations above). M4d is therefore the
first genuinely informative negative in the ladder. All integrity gates pass, including the fused-forward transfer
check (0.2385 → 0.1562). Two honest observations:

- This is the **best accuracy any rung has produced** (24/40) and the lowest
  p-value of the ladder — but it is not significant, and per the interpretive
  rule registered *before* this rung reported, an M4d null was **expected and
  weakly informative** because the diagnostic had already exonerated its
  hypothesis. It must not be read as evidence about configuration matching.
- The NLL inversion **persists unchanged**: 0W/40L against both random and
  zero, mean NLL 4.92 vs baseline 2.13. Deployment matching did not repair
  it, which is exactly what the pre-check predicts — the adapter was
  off-manifold *from its random initialisation*, and matching the deployment
  transform gives training no reason to return.

## M4g PRE-REGISTRATION (2026-08-29, before any run) — fuse instead of overwrite

**Further premise correction (2026-08-29, from the M4d implementer's report)**:
the earlier correction above understated the problem. `train_m4c_taskloss.rs`
(lines 106-124) already had **pooling, rescale-to-natural-median AND the
8-slot placement** in its training loop — not just the rescale. M4d's actual
delta versus M4c was therefore marginal (it sourced the rescale target from
the probe's literal code path, gate-verified to 3.2e-07). The registered
candidate was substantially discharged before M4d ever ran, which is
consistent with M4d reproducing M4c's NLL inversion exactly (0W/40L against
both controls, p = 1.0).

**Coordinator decision, deviating from the scout's recommended ordering**
(which put M4f first). Reasoning, recorded so it can be judged: M4g explains
**both** families of null, while M4f addresses only one. M3/M4 were
demonstrably **on-manifold and still useless** — if overwriting eight
residual positions is what makes injected content unusable, that is precisely
why manifold-correct adapters failed too, and it is the one mechanism
verified to differ from the method that demonstrably works (C2C fuses:
`C_F = C_n(X) + F_n(...)`, a residual ADD onto the receiver's own cache;
`inject.rs` hard-`slice_assign` OVERWRITES, `qwen2_b.rs:79-87`). M4f remains
registered and runs next if M4g nulls.

**Spec**: change the injection operator from overwrite to residual add
(`h[slot] += c·v`, preserving the receiver's own state at those positions),
retrain the M3-shaped adapter under task loss with the fuse operator in the
loop, single seeded run, same splits and the same 13 exclusions, training
receipt + artifact hash + ≥8 golden pairs frozen before the probe, ONE
registered frozen-probe draw under the unchanged protocol, mid-p McNemar
primary with exact-sign reported alongside. **This changes the injection
operator only** — an evolvable surface under ADR-028; the probe, controls,
items and statistics are untouched. Estimated ~0.5 GPU-h by analogy to M4c's
receipted 0.446.

**Registered interpretation, before the run**: a PASS identifies
overwrite-vs-fuse as the ladder's root cause and retroactively explains both
null families (M3/M4's on-manifold uselessness and M4c/M4d's off-manifold
harm), which would require annotating every earlier rung. A NULL leaves M4f
(structural on-manifold constraint) and M4b (receiver scale) as the live
hypotheses and makes the joint negative substantially stronger — spanning
loss function, deployment configuration, and injection operator.
**Honest-fail path unchanged**: full numbers either way, no protocol
iteration, no retry.

## M4g outcome (2026-08-29) — HONEST FAIL; fuse is not the root cause

Receipts `run2-m4g-training-receipt-cellL18toL14.json`,
`run2-m4g-transfer-receipt-cellL18toL14.json`,
`run2-manifold-precheck-m4g-receipt.json`,
`run2-m4g-receipt-cellL18toL14-mlp-fuse-slots8-poolfull-rescaletrue-n40.json`.
One changed factor, exactly as registered: `h[slot] = c·v` became
`h[slot] += c·v`. `LayerEdit::Fuse` is a NEW enum variant beside `Inject`, so
the overwrite arm's op sequence is untouched and every prior receipt stays
reproducible; every `InjectionSpec` construction site in the repo now names
its mode explicitly. The seeded init is **byte-identical to M4c's and M4d's**
(`c424030e…`, asserted against both training receipts, not assumed from the
shared seed constant), so M4c → M4d → M4g is a clean three-way ablation of
loss configuration → deployment configuration → injection operator.

1. **Training was the best of the three rungs.** Holdout task CE
   **0.2418 (seeded init) → 0.1560** at best epoch 4 (M4c 0.2546→0.1595,
   M4d 0.2545→0.1583). The lower init CE is itself the operator working as
   designed: an untrained random adapter does less damage when it is *added*
   to the receiver's own state than when it *replaces* it.
2. **The improvement transfers.** Pre-probe transfer check through the
   vendored fused forward: **0.2276 → 0.1536, 498 wins / 11 losses**
   (secondary sign p = 8.1e-132). Gate passed before the probe was invoked.
3. **The frozen probe nulls.** Aligned 23/40, baseline 22/40, zerovec 22/40,
   random 22/40. Primary aligned > random: **2W/1L, n_disc = 3, exact-sign
   p = 0.5000, mid-p McNemar 0.3125** — fail on both, and the mid-p value is
   the one this rung's pre-registration named as primary.
4. **This draw was structurally incapable of rejecting — say so plainly.**
   At `n_disc = 3` the minimum attainable one-sided p is 0.125 > α = 0.05.
   M4g is therefore back in the power-floor dead zone that M3, M4 r64 and
   M4 r128 sat in, and it is *not* comparable in strength to M4d's null,
   which at `n_disc = 7` could have rejected and did not. **The
   accuracy-side M4g null is weak evidence.**
5. **The NLL side is not power-limited, and it is decisive.** The 0/40
   inversion — the specific thing this rung was registered to fix —
   **persists essentially unchanged**: mean NLL of `#### <gold>` is
   **4.815 aligned vs 2.129 baseline / 2.129 zerovec / 2.170 random**, with
   **0 wins / 40 losses against both controls** (M4c 5.359, M4d 4.919 —
   fuse moved the mean by ~0.1 nats and moved the win count not at all).
   Preserving the receiver's own state at the eight rows did **not** repair
   the inversion. **Overwrite-vs-fuse is refuted as the root cause of the
   ladder's central anomaly.**
6. **Pre-probe manifold check (diagnostic only, not a gate, run before the
   draw).** M4g's emitted vector is still off-manifold: cosine-to-natural
   **−0.041** (M4c −0.018, M4d +0.048), item-invariance **0.869** — the
   lowest, i.e. most item-varying, of the task-loss family (M4c 0.881, M4d
   0.907) — and gold-answer token percentile **66.8%**, the *worst* of any
   candidate measured. Being slightly more item-varying bought nothing.
7. **Operator-correctness, measured twice.** In training, a zero payload
   under fuse reproduced the un-injected span CE exactly (0.195708 both
   ways) while the same payload under overwrite cost 0.0055 nats. In the
   probe, the zerovec condition was **bit-identical to baseline on all 40
   items** (0 accuracy disagreements, max |ΔNLL| = 0.0). The fuse operator
   does what the equation says.

**The control-semantics question, resolved and frozen BEFORE the draw**
(`control_semantics_under_fuse` in the training receipt, echoed into the
probe receipt). Under fuse a zero vector is a genuine no-op, so the
registered zero control's *meaning* changes even though its *definition*
does not. The resolution taken, and the deviation declared rather than
concealed:

- `aligned_real` and `random` — definitions and meanings both unchanged. The
  random control is still a per-item seeded Gaussian norm-matched to the
  effective aligned vector; under fuse it is an information-free
  *perturbation* of the receiver's own rows, which is exactly the comparator
  the primary statistic needs. **The primary statistic is unaffected.**
- `zerovec_injected` — definition unchanged (the true zero vector, `scale:
  None`, through the real 8-slot path), meaning changed: it collapses onto
  `baseline_uninjected`. It was still **run on all 40 items** and the
  collapse **measured** as the operator gate above. Not redefined, not
  replaced, not dropped.
- **No substitute control was added.** Keeping a destructive
  overwrite-with-zero condition inside an M4g draw would put two injection
  operators in a rung whose whole content is that exactly one operator
  changed, and would be an unregistered addition to an ADR-028-protected
  control set.
- **The registered zerovec gate is DEGENERATE under fuse** (`2 × zerovec ≥
  baseline` is trivially satisfied when the two are the same computation).
  It is still computed, still enters `gate_pass` unchanged for cross-rung
  comparability, and is labelled degenerate in the receipt so no reader
  mistakes it for evidence.
- Consequence: under fuse, `baseline_uninjected` is the meaningful "nothing
  delivered" reference, and `aligned_vs_baseline` (3W/2L, exact p 0.5000,
  mid-p 0.3438) is the informative "did adding anything help" contrast.

**Cross-rung comparison** (all three: same architecture, same seeded init,
same splits, same 13 exclusions, same loss, same schedule, one probe draw
each):

| | M4c (task loss) | M4d (+deploy match) | M4g (+fuse) |
|---|---|---|---|
| changed factor | loss → task CE | deployment configuration | **injection operator** |
| holdout CE init → best | 0.2546 → 0.1595 | 0.2545 → 0.1583 | **0.2418 → 0.1560** |
| transfer (fused NLL) | 0.2385 → 0.1570, 498W/11L | 0.2385 → 0.1562, 500W/9L | 0.2276 → 0.1536, 498W/11L |
| accuracy A/B/Z/R | 23/22/24/21 | 24/22/24/21 | 23/22/22/22 |
| primary W/L, n_disc | 4W/2L, 6 | 5W/2L, 7 | 2W/1L, **3** |
| exact-sign p | 0.3438 | 0.2266 | 0.5000 |
| mid-p McNemar | 0.2266 | 0.1445 | 0.3125 |
| could this draw have rejected? | yes (min p 0.0156) | yes (min p 0.0078) | **no (min p 0.125)** |
| mean NLL aligned / baseline | 5.359 / 2.129 | 4.919 / 2.129 | 4.815 / 2.129 |
| NLL vs random | **0W/40L** | **0W/40L** | **0W/40L** |
| NLL vs zero | **0W/40L** | **0W/40L** | **0W/40L** |
| cosine-to-natural (emitted) | −0.018 | +0.048 | −0.041 |
| item-invariance | 0.881 | 0.907 | 0.869 |
| gold-token percentile | 61.5% | 35.1% | 66.8% |
| GPU wall (train / probe) | 1603 s / 422 s | 1643 s / 409 s | 1656 s / 412 s |

**Reading, per the registered interpretation.** This is the NULL branch: M4f
(structural on-manifold constraint) and M4b (receiver scale) remain the live
hypotheses, and the joint negative now spans **loss function, deployment
configuration, and injection operator** — three architecturally distinct
factors, each isolated, each nulling. The registered PASS branch (annotating
every earlier rung) does not fire.

Two things this rung adds beyond "another null":

- **It refutes a specific, well-motivated, primary-sourced hypothesis.** The
  C2C-fuses-we-overwrite difference was verified from the paper's own Eq. 3
  (`docs/research/038` §4) and was the best mechanistic candidate on the
  board. Making the operator match C2C's, with everything else held, changed
  the NLL inversion by ~0.1 nats and 0 win/loss counts. That is a real
  finding about *this* system, not a null result about nothing.
- **A partially manifold-preserving delivery is still not enough.** Under
  fuse the row the receiver reads is `h + c·v` rather than `c·v`. Since the
  rescale forces ‖c·v‖ ≈ the natural per-position median ≈ ‖h‖, and the
  measured cosine between the emitted vector and the natural state is
  ≈ −0.04, the fused row retains cosine ≈ **0.69** to the natural state
  where the overwritten row retained ≈ −0.04. (Derived arithmetic from two
  measured quantities, **not** a fresh measurement — and the cosine it uses
  is emitted-vs-natural-*pooled*, a slightly different reference from the
  per-slot residual rows; treat it as an order-of-magnitude statement.) The
  gold-token likelihood collapsed anyway. Whatever is wrong is not fixed by
  making the injected row look more like a real state at the injection
  site — which is the same lesson M3/M4's on-manifold-and-still-useless
  nulls taught, now reproduced through a second, independent mechanism, and
  it sharpens rather than softens M4f's own pre-registered risk.

**Honest-fail path observed**: one training run, one transfer check, one
manifold pre-check, ONE frozen-probe draw. No retries, no protocol iteration,
no second variant. Full numbers reported either way (ADR-024 § Numbers rule,
ADR-032).

## CLARIFICATION — the permutation-null count (2026-08-29)

I have quoted the A6 permutation null as "0 of **160** permuted fits within
0.46 of any real fit". The **committed receipt records 80** (4 cells × 20
seeded permutations). The additional 80 come from the adversarial verifier's
independent re-run with 20 *fresh* seeds per cell, reported in its verdict but
**not** in the primary receipt. Both statements are true, but they are not the
same statement: **80 in the receipt, 160 across receipt + verification.**
Anything quoting 160 must say "across the original run and the independent
re-run". Flagged by the synthesis pass as the second brief-vs-receipt numeric
mismatch in this corpus (the first being the "~5.7 GPU-h" figure already
corrected to the receipt-summed 3.32 in ADR-023/research-025) — the pattern
itself is worth watching: **numbers restated in prose drift from numbers in
receipts.**

## M4g manifold pre-check — recorded BEFORE its probe verdict (2026-08-29)

`run2-manifold-precheck-m4g-receipt.json` (CPU-only, no probe draw) classifies
**M4g's fuse-trained adapter as COLLAPSED-OFF-MANIFOLD** — the same class as
M4c and M4d, and as their shared untrained initialisation. Changing the
injection operator to a residual add did **not** move the adapter back onto
the receiver's manifold. That is consistent with the standing diagnosis:
off-manifold output is the untrained *default*, and no task-loss objective —
under overwrite or under fuse — creates pressure to leave it.

**Interpretation registered before the probe reports:**
- If **M4g PASSES**, it passes *while off-manifold*. That would mean manifold
  membership is not the operative variable at all and the **injection
  operator** is — a strong, clean result that would also retroactively explain
  M3/M4 (on-manifold but overwriting) and M4c/M4d (off-manifold *and*
  overwriting).
- If **M4g NULLS**, the ladder has three task-loss rungs that are all
  off-manifold, and the two properties (on-manifold, and delivered by fuse)
  have never yet been combined in a single adapter. The next rung should
  therefore combine them — which is what **M4h Stage 1 delivers almost free**
  (M3's *already-trained, on-manifold* MLP, last-token instead of mean, and —
  if M4g's fuse path is available — delivered by fuse rather than overwrite).
  That combination is registered here as the preferred successor, before the
  verdict that would motivate it.

## M4h Stage 1 OUTCOME (2026-08-29) — mechanically successful, statistically unmeasurable

`run2-m4h-s1-receipt-...-pertokenlast-fuse-slots8-nopool-*.json`.

**The mechanism worked.** The pre-check classifies the de-pooled payload as
**"on-manifold-item-varying"** — the **first adapter in the ladder to be
both**. Every prior on-manifold adapter was *item-invariant*; de-pooling
(last-token instead of mean) fixed that, exactly as
[docs/research/040](../research/040-the-pooling-gap.md) predicted.

**And, as the MAJOR CORRECTION predicted, there is no NLL inversion**:
aligned 2.1599 vs baseline 2.1288 — and aligned actually **beats random**
on mean NLL (2.1599 vs 2.1680). Confirms that inversion belongs to
off-manifold payloads only.

**But the draw is uninformative, and the reason matters.** Accuracy: aligned
**24/40** (ties the ladder's best), baseline 22, random 22, zerovec 22 — and
**2W/0L versus baseline, no losses at all**. Yet the primary test gives
**n_disc = 2**, where the **minimum attainable one-sided p is 0.25** — five
times α. The receipt flags this itself (`power_limited: true`). This draw
**could not have rejected under any effect size whatsoever.**

**POOLING IS REFUTED — the pre-check makes this decisive.** Removing the mean
(not one trained weight changed) moved the emitted object onto the real
single-state values:

| candidate | invariance | top-10 union | cos-to-pooled-ref | entropy | class |
|---|---|---|---|---|---|
| M3 per-token (pooled) | 0.9702 | 70/400 | 0.9889 | 9.34 | item-invariant-but-on-manifold |
| **M4h S1 (de-pooled)** | **0.6670** | **133/400** | **0.6814** | **3.32** | **on-manifold-item-varying** |
| ref: receiver L14 **single row** | 0.6350 | 153/400 | 0.6670 | 3.36 | on-manifold-item-varying |
| ref: receiver L14 pooled | 0.9617 | 78/400 | 1.0000 | 9.30 | item-invariant-but-on-manifold |

The de-pooled payload is **statistically indistinguishable from a genuine
un-pooled receiver state** on every axis measured. And it still transfers
nothing — indeed NLL got slightly *worse*: +0.031 vs baseline, the largest
positive delta of any on-manifold rung, and 10W/30L against zerovec where
pooled M3 was an even 20W/20L. **A payload can be made geometrically
identical to a real receiver state and still carry nothing.** Pooling joins
task loss (M4c/M4d) and injection operator (M4g) as a refuted root cause —
the third to fall.

**Framing correction to my own earlier note**: I described the 2W/0L accuracy
result as "the best we've seen". The implementer's framing is more accurate
and I adopt it: *a clean 2W/0L sweep that cannot clear α is what a
power-limited design produces — it is not encouraging.* The NLL result, which
is not power-limited, is the one that carries weight here, and it points the
wrong way.

**THE BINDING CONSTRAINT IS NOW THE INSTRUMENT, NOT THE SCIENCE.** Discordant
counts across the ladder's recent draws: M4d **7** (the only non-limited
one), M4g **3** (floor 0.125), M4h-S1 **2** (floor 0.25). As payloads become
*better behaved* — on-manifold, non-destructive, item-varying — they perturb
fewer items, so discordance **falls** and the frozen 40-item sign test loses
what little power it had. The better our adapters get, the blinder the probe
becomes.

**Consequence, recorded now**: further rungs drawn against the frozen 40-item
protocol are **not capable of detecting the effects we are now looking for**.
Any future rung must either (a) report as descriptive evidence only, with the
power floor stated, or (b) come with its own pre-registration adopting a
higher-powered design — the anytime-valid e-process already specified in
[ADR-030](030-run3-causally-gated-text-pre-registration.md) §3.2 is the
ready-made instrument, and [docs/research/031](../research/031-statistical-power-and-design.md)
carries the power tables. This is a **protocol** change and therefore
requires its own pre-registration; it is **not** a licence to re-draw any
completed rung.

## LAYER COVERAGE — the strongest external evidence yet, and a methodological reckoning (2026-08-29)

[docs/research/044](../research/044-layer-coverage.md) found the citation this
ladder has been missing.

**C2C's own Table 10 ablates single-layer enrichment and reports ~0.1pp over
baseline** (58.42% → 58.45-58.52% for its two best individual layers; most
others net-negative). **The method that works, reduced to a single layer,
produces exactly our signature** — inert, not harmful, indistinguishable from
baseline. That is materially stronger support for a layer-coverage
explanation than M4i's current backing (two abstract-only pause-token
papers), and it comes from the strongest comparator in the field.

**No content-transfer method uses one layer**: C2C gates over ~top-5 of 28
layers; LatentMAS transfers the full per-token cache at **all** layers;
Bicameral couples at **4 fixed layer indices** (2 read + 2 write per
direction) found by sweeping **890 configurations**. *Correction to this
repo's own earlier docs (028, 039, 040), which described Bicameral as simply
"continuous per-step" without the 4-layer structure.*

**The distinction that saves the steering literature from contradicting
this**: CAA/ActAdd *do* work from a single layer — but they apply a steering
**direction** continuously at every generated token, nudging an
already-computed decision. We inject a one-shot **state** that must survive
14+ subsequent blocks unassisted. RepE, which transfers richer content rather
than a direction, explicitly uses many layers *"because changes made in
earlier layers propagate to later layers, diminishing the effect"* — a
primary-sourced statement of precisely the RMSNorm-dilution mechanism
[docs/research/033](../research/033-rescale-output-alignment-diagnostic.md) §5
flagged as un-refuted and which no rung has tested.

**Honest limit**: layer coverage explains the on-manifold family's *ceiling*
(why nothing gets through) but **not** the off-manifold family's *harm* —
M4c/M4d/M4g were also single-layer, and if single-layer alone caused damage,
single-layer on-manifold payloads would be damaging too. They are not. Same
partial-explanation shape as pooling.

**THE METHODOLOGICAL RECKONING, recorded because it may be the run's real
conclusion**: every working method combines **multi-layer + continuous
delivery + task-loss training simultaneously**. This ladder, by design, has
only ever varied **one factor at a time** — that discipline is what makes
each null attributable, and it may also be exactly why every rung nulls. If
transfer requires a *conjunction* of these properties, no single-factor rung
can ever succeed, and a one-factor-at-a-time ladder is structurally
guaranteed to produce the result we have. That is a testable claim (a
deliberate multi-factor rung, explicitly labelled as abandoning single-factor
attribution) and it must appear in any write-up as a limitation of the method,
not merely a finding about latent transfer.

**Implementation note for a future layer rung**: `LayerEdit` is single-site by
construction (only the read-only `Capture` has a multi-tap sibling); it needs
an `InjectMany`/`FuseMany` variant, bounded work of the same shape as M4g's
`Fuse`. **No new capture is required** — the dumps already hold sender
L18/L24 and receiver L14/L19, and **no rung has ever used the L24→L19 pair**.
Open tension to settle before pre-registering: 8 slots across 2 sites is
either 16 total or a 4+4 split, and ADR-028's slot-count discipline (already
self-contradictory and unadjudicated) does not cover it.

## ATTRIBUTION RESOLVED — there were two agents, both truthful, and I conflated them (2026-08-29)

I recorded peer attribution as "not determinable". **It is now determinable.**
The workflow journal for `wf_82867994-eef` lists **two** implementation agents:

- **`adf964e8b373cc990`** — the draw owner. Sent messages 1, 3, 4 and 5.
- **`a526911380dbf744a`** — the second agent. Sent message 2.

Both are type `general-purpose`; inbound messages carry **only the type**, so
both arrived as `from=general-purpose` and I treated five messages as coming
from one agent.

**Every apparent contradiction dissolves.** Message 2's *"I never killed any
process"* was **true of its author** — the workflow report confirms
`a526911380dbf744a` issued *"no kill, pkill or TaskStop at any point"*. Message
1's *"I killed MY duplicate"* was **true of its author**. **Neither agent was
inconsistent; I manufactured the inconsistency by merging them**, then
recorded a real agent's account as self-contradictory and briefly withdrew a
process kill that had actually happened.

`a526911380dbf744a` is also the author of the **rebuilt-binary/hash analysis**
and the **`lens.rs` N-invariance defect report** — the two findings
`adf964e8b373cc990` correctly refused credit for. Both were independently
verified before entering this record, so nothing rests on the attribution; but
**the refusal of unearned credit is what made the resolution findable.**

**Root cause is mine**: coordinator error #11 put two agents on one registered
rung. Everything downstream — the duplicate process, the concurrent source
edits, the merged message identity — follows from that single mistake.

## PC3 — THE DISSOCIATION IS CONFIRMED OUT OF SAMPLE. The ladder closes. (2026-08-29)

**Receipt**: `run2-pc3-receipt-identity-L19lasttoken-decoytf-fuse-questiontail-slots8-eprocess-answerchange-oos.json`.
**All 15 structural gates PASS**, including the one that matters most:
`OUT_OF_SAMPLE_intersection_with_pc2_m4i_is_empty = true`. N = 212 (the entire
remaining pool), items 4485+, zero overlap with any prior rung.

### The registered primary — a POWERED null

| | |
|---|---|
| `steer` changes its answer | **108 / 212 = 50.9%** |
| `random` changes its answer | **104 / 212 = 49.1%** |
| `restore` | 97 / 212 = 45.8% |
| `zerovec` | **0 / 212 = 0.0%** |
| e-process | **36W / 32L, n_disc = 68**, W = **0.8444** (max 1.0130, min 0.2797) |
| power | min attainable p = **3.4e-21**; floor of 30 cleared |
| exact sign test | **p = 0.358 one-sided, 0.716 two-sided** |

**`steer` and `random` are statistically indistinguishable.** Not "steer is
worse" — *indistinguishable*, at a discordant-pair count that could have
detected an effect twenty orders of magnitude smaller.

### The likelihood arm REPLICATES, decisively

| comparison | PC3 (out-of-sample) | PC2 (in-sample) |
|---|---|---|
| `steer` → decoy NLL vs baseline | **−0.773 nats**, 190W/22L, **p = 7.5e-35** | −0.721, p = 2.6e-44 |
| `steer` → decoy NLL vs *random* | **−0.638 nats**, 176W/36L, **p = 1.2e-23** | −0.237, p = 1.6e-8 |
| `restore` → gold NLL vs random | **−0.320 nats**, 155W/57L, **p = 5.8e-12** | −0.282, p = 1.6e-8 |
| `steer` → gold NLL vs baseline | −0.028, 110W/102L, p = 0.32 | +0.052 |

`zerovec` is bit-identical to `baseline` — **212/212 identical generated
texts, max |Δ NLL| = 0.0 on both targets**. The operator is proven for a third
time. Gaming guard clean (length ratio 1.015, 0/212 degenerate).

### THE RESULT, now registered and out-of-sample

> **The channel is SEMANTIC at the likelihood level and NON-SEMANTIC at the
> decision level.**
>
> A payload moves the receiver's distribution toward *whichever specific answer
> it encodes* — a deliberately false one by **−0.773 nats (p = 7.5e-35)**, the
> true one by **−0.320 nats (p = 5.8e-12)** — and contributes **nothing
> whatsoever** to which answer the receiver actually gives (**p = 0.72**, fully
> powered).

Injection perturbs ~half of all answers; a norm-matched Gaussian does the same.
**The decision layer reads the perturbation and not the meaning.**

### COORDINATOR ERROR #15 — I predicted the direction, and I was wrong

**Recorded prominently because I pre-registered the prediction and it failed.**

| | predicted (ADR-040, pre-draw) | observed |
|---|---|---|
| win rate | 43% | **52.9%** |
| final wealth | **0.0790** (12.7× decline) | **0.8444** |
| direction | steer *behind* random | steer marginally *ahead* |

**The sign flipped and the magnitude was off by an order of magnitude.**

The cause is diagnosable and it is my own over-reading: PC2's 33W/44L is a
43%/57% split on 77 pairs — **p = 0.13, comfortably inside sampling noise of
parity.** I treated a non-significant in-sample point estimate as a directional
signal, wrote a 12.7× decline into a pre-registration, and told the
implementer it made a FAIL "strongly expected rather than a coin flip." **It
was always a coin flip.** This is coordinator error #13's failure mode
(over-reading a directional lean at p = 0.27) repeated one level up, in a
document whose whole purpose is to bind me in advance.

**The correction strengthens the finding rather than weakening it.** Near-parity
is the *cleaner* null: "payload content contributes nothing" is exactly what
p = 0.72 says. My predicted decline would have been harder to explain — why
would real content be *worse* than noise? **The result I got is better evidence
for the claim than the result I predicted.** Recording that I was wrong in the
direction that helped me is the point of writing the prediction down first.

### Consequences, accepted in advance and now binding

- **M5X ([ADR-037](037-m5x-maximal-configuration-rung.md)) and M4b
  ([ADR-035](035-m4b-scale-control-pre-registration.md)) are BLOCKED
  PERMANENTLY under this apparatus.** Both vary payload *content*, which is now
  demonstrably not what moves decisions — confirmed out of sample.
- **The ladder closes.** Every cross-model null it produced is explained by the
  apparatus, and **none of them is evidence about latent transferability.**
- **This dissociation is the publishable result**, under
  [ADR-032](032-negative-result-publication-contract.md), reported without
  softening.
- **Firewall, unchanged**: PC3 is same-model, same-item, identity-transform. It
  tests the apparatus, **never transfer**, and this FAIL may not be cited as
  transfer evidence in either direction.

## ERROR CLASS — cross-rung number transposition (identified 2026-08-29)

Recorded as a **class**, not an incident, because it is the one that defeats a
plausibility check.

The PC3 owner stated the expected wealth decline as "around 0.19". Recomputing
from the split gives **W = 1.15³³ × 0.85⁴⁴ = 0.0790**, a 12.7× decline. The
owner then identified the source, and it is verifiable: **0.19 was PC1b's own
`final_wealth` (0.194939465)** — a figure carried across from a *different
rung's receipt* rather than computed from PC2's 33W/44L.

**Why this class is dangerous here.** It is not a formula error and not a typo.
A transposed number from a neighbouring rung **passes every sanity check**:
0.19 *is* a plausible declining wealth, it *is* the right order of magnitude,
and it *is* below 1.0 in the expected direction. Nothing about it looks wrong.
This repository now holds fifteen-plus rungs whose wealths, n_disc counts and
NLL means all live in the same numeric neighbourhood, so **every figure is
camouflage for every other figure.**

It was caught only by **recomputing from the underlying split** rather than
checking the magnitude for reasonableness.

**Standing rule, adopted**: *compute every reported number from the rung's own
trajectory and cite the receipt field it came from; never carry a figure across
rungs from memory.* This is the same discipline that
[research/047](../research/047-authoritative-power-table.md) had to impose on
power claims after the ladder accumulated four restated-number drifts, now
generalised beyond power.

Related and already recorded: coordinator error #4 (M4d's primary p conflated
aligned-vs-random with aligned-vs-baseline) and #8 ("6 of 9 power-incapable"
against the authoritative recount of 10 of 14) are **the same class**.

## THE DISSOCIATION, RESOLVED — "decision-inert" is WRONG; the apparatus is decision-PERTURBATIVE but NON-SEMANTIC (2026-08-29)

**Computed from PC2's committed receipt with zero new GPU time**, by reading
the per-item `extracted_answer` fields the probe already recorded. **This
answers the question PC2 was built to ask, at full power, where PC2's own
registered primary could not.**

### The dense decision endpoint

Instead of *"did the receiver emit the specific decoy"* (rate ~2%, n_disc = 3,
unmeasurable), ask *"did the injection change the answer **at all** versus
baseline"*:

| condition | answer differs from baseline |
|---|---|
| `random` | **151 / 300 = 50.3%** |
| `steer` | **140 / 300 = 46.7%** |
| `restore` | 135 / 300 = 45.0% |
| `zerovec` | **0 / 300 = 0.0%** |

`zerovec` at exactly 0/300 re-confirms the operator: this endpoint measures
injection, not harness noise.

**Paired `steer` vs `random`**: steer-changed-only **33**, random-changed-only
**44**, **n_discordant = 77** — minimum attainable p = **6.6e-24**, i.e.
**fully powered**, versus n_disc = 3 on the registered primary. One-sided exact
sign test: **p = 0.127**, and the direction slightly *favours random*.

### I was wrong to call it "decision-inert"

The apparatus is **not** inert at the decision level. **It changes the answer on
roughly half of all items.** What it does not do is change them *because of what
the payload says*: a norm-matched Gaussian moves decisions **at least as much**
(50.3% vs 46.7%), and the paired test cannot separate them at n_disc = 77.

**The corrected dissociation:**

> **Likelihood level — SEMANTIC.** The payload's content determines the
> direction with extreme fidelity: decoy NLL −0.721 nats, **p = 2.6e-44**; each
> payload moves toward the answer it encodes and away from the other.
>
> **Decision level — PERTURBATIVE but NON-SEMANTIC.** Injection changes ~47% of
> answers, but content contributes **nothing beyond what noise contributes**
> (33 vs 44, **p = 0.127**, fully powered).

The channel writes meaning into the distribution and the decision layer reads
only the perturbation. That is a far sharper claim than "the pathway is dead",
and it is the mission's central result.

### Status of this analysis — POST-HOC, and labelled as such

**This endpoint was not pre-registered.** Under
[ADR-031](031-evidence-receipt-and-statistical-protocol-governance.md) and
[ADR-032](032-negative-result-publication-contract.md) it is a **secondary
post-hoc analysis**, and it is **not** promoted to a registered result on the
strength of being interesting. It is powered and its direction is a clean null,
but **it must be pre-registered and confirmed on a fresh draw** before it can
be cited as established. That is exactly what ADR-040 does.

**It does not change PC2's verdict**: the registered primary remains
**UNINFORMATIVE**, and **M5X (ADR-037) and M4b (ADR-035) remain BLOCKED** —
strengthened, since a payload's content demonstrably contributes nothing to
decisions, and both rungs vary payload content.

## PC2 RESULT — primary UNINFORMATIVE (not a FAIL); the salvage is the mission's sharpest finding (2026-08-29)

**Receipt**: `run2-pc2-receipt-identity-L19lasttoken-decoytf-fuse-questiontail-slots8-eprocess.json`.
Leakage gate **did not trip** (baseline 1.33% < 2%), so the rung is **valid**.

### I am OVERRIDING the binary's auto-verdict

The receipt's verdict string reads *"FAIL — THE APPARATUS CANNOT MOVE A DECISION
BY ANY MEANS."* **That is not supported by its own data, and it ignores the
third branch ADR-039 registered.**

- **n_discordant = 3.** Minimum attainable p = 2⁻³ = **0.125**. The primary
  **cannot reach α = 0.05 even on a flawless run**. It is *structurally
  incapable*, exactly as [coordinator error #14](039-pc2-steering-control-pre-registration.md)
  recorded **mid-draw, before this outcome was known**.
- **wins 3, losses 0.** *Every* discordant pair favoured `steer`. A
  directionally perfect run — which is the opposite of "cannot move a decision
  by any means", and which a FAIL verdict actively misreports.

**The registered primary is therefore UNINFORMATIVE.** Per ADR-039 it is *"not
spun as either outcome"*, and per ADR-036 Decision 3 the power floor is stated
on its own scale. **PC2 does not discharge the gate. M5X (ADR-037) and M4b
(ADR-035) remain BLOCKED.**

### The salvage — fully powered, and it is the sharpest result of the mission

The decoy-string NLL is dense: 300 paired observations, no rarity problem.

**Mean NLL of the DECOY target**

| | steer | restore | baseline | zerovec | random |
|---|---|---|---|---|---|
| | **2.3097** | 2.9595 | 3.0308 | 3.0308 | 2.9000 |

**Mean NLL of the GOLD target**

| | steer | restore | baseline | zerovec | random |
|---|---|---|---|---|---|
| | 2.4446 | **2.1110** | 2.3928 | 2.3928 | 2.3481 |

**Each payload moves the likelihood toward the answer it encodes, and away from
the other:**

- `steer` → decoy NLL **−0.721 nats** vs baseline, **264W/36L, p = 2.6e-44**;
  vs norm-matched random **−0.590 nats**, 256W/44L, **p = 7.9e-38**.
- `steer` → gold NLL **+0.052 nats WORSE** (124W/176L) — it actively moves
  probability *off* the true answer.
- `restore` → gold NLL **−0.282 nats**, 198W/102L, p = 1.6e-8 — **reproducing
  PC1b's number exactly, on 300/300 comparable items.**
- `restore` → decoy NLL barely moves (3.031 → 2.960), as it must.

**This is semantically faithful, directional steering.** The channel does not
merely carry *information* — it carries **which answer**, and the receiver's
distribution follows it.

### And it still does not become a decision

Accuracy: `steer` **126**, `restore` 127, `baseline` **140**, `zerovec` 140,
`random` 130. Injecting any non-zero vector costs ~10-14 correct answers, on
now a **third** independent stream.

`zerovec` is bit-identical to `baseline` — **300/300 identical generated
texts**, `max |Δ NLL| = 0.0` on **both** targets. The operator is proven, so
none of this is an artefact of the injection machinery.

### The finding, stated precisely

> **The apparatus is LIKELIHOOD-STEERABLE and DECISION-INERT.** A payload can
> move the receiver's distribution toward *any specified answer* — the true one
> (PC1b, −0.28 nats) or a deliberately false one (PC2, **−0.72 nats**,
> p = 2.6e-44) — while **changing essentially none of its answers**, and while
> *reducing* accuracy.

This is stronger than PC1b alone, which showed only that *content* got through.
PC2 shows the channel is **steerable with semantic fidelity** and that the
failure is located precisely at the **likelihood → decision** step. The
gaming guard is clean (length ratio 1.005, 1/300 degenerate, no NLL collapse),
so this is not a degenerate-output artefact.

**Firewall, unchanged**: PC2 is same-model, same-item, identity-transform.
It tests **the apparatus, never transfer** — and, per the symmetric rule, this
result may not be cited as transfer evidence in **either** direction.

## COORDINATOR ERROR #13 — "worse than noise, reproducibly" is RETRACTED (2026-08-29)

**I foregrounded a finding that the data does not support.** I wrote that the
payload being beaten by the random control "deserves its own line — it is a
positive finding about harm, not just an absence." **It is not a finding at
all.** Exact one-sided sign tests on the discordant pairs, computed by me:

| comparison | split | one-sided p |
|---|---|---|
| PC1b aligned vs random | 29W/35L | **0.2662** |
| M4i aligned vs random | 31W/35L | **0.3561** |
| **pooled** | **60W/70L** | **0.2150** |

**Nothing approaches significance.** Two consistent directions at n_disc ≈ 65
each is not replication of an effect — it is two null results that happen to
lean the same way. Foregrounding it would have been precisely the over-reading
the e-process exists to prevent, and it would have been this ADR's **fourth
restated-number drift**. Caught by the implementer.

**What the data does support**, and what replaces it: *injecting a non-zero
vector at this site costs accuracy relative to baseline* — PC1b aligned vs
baseline **21W/34L, one-sided p = 0.0524**. **Marginal, not conclusive.** It is
a statement about **injection**, not about the payload beating or losing to
noise. `zerovec` being bit-identical to baseline (max |Δ| = 0.0) is what makes
it attributable to injected **content** rather than to the operator. For
completeness: **no accuracy comparison in either rung reaches p < 0.05.**

The contrast with the likelihood endpoint is the whole story: NLL
aligned-vs-baseline **213W/87L (p ≈ 1e-13)** and aligned-vs-random **198W/102L
(p ≈ 2e-8)** are overwhelming, while **every accuracy comparison is null.**

## THE INVERSION IS REINSTATED — and I was wrong to withdraw it (2026-08-29)

I argued the pre-registered FAIL branch is logically inverted, then
**withdrew** that on the implementer's norm-matched-random argument. **The
implementer has now withdrawn its refutation and I accept: my original
position was right, and my withdrawal was the error.** I flip-flopped; both
turns are preserved.

**The distinction that reconciles everything — endpoints:**

- **Likelihood endpoint: the apparatus is VALIDATED.** Aligned beats a
  norm-matched Gaussian through the same operator at the same positions by
  0.237 nats on 198/300. Delivery of content-dependent information is
  **proven**. This fact stands and is not in dispute.
- **Accuracy endpoint: the apparatus is UNVALIDATED.** **No control in this
  repository has shown it can move accuracy at all.**

**Every rung's verdict rests on the accuracy endpoint.** So for the endpoint
that decides every result, the apparatus cannot distinguish *"no effect"* from
*"cannot detect effects"* — and the method remains the **leading explanation**
for every null. Ruling plumbing out requires a **PASS on the endpoint being
measured**. Delivery working was never the gate; **conversion to the measured
outcome** is.

**The a fortiori argument supports BLOCKING, not proceeding.** *"A pathway that
cannot convert a gold-derived same-model identity payload will not convert a
weaker cross-model adapter output"* is a prediction that M5X/M4b **also fail
for apparatus reasons** — which is exactly why their nulls would be
uninformative. Both of us initially read this backwards.

**M5X (ADR-037) and M4b (ADR-035) STAY BLOCKED.** Headline, unhedged:
**we have still not run a passing positive control** — worse for the mission
than a clean transfer null.

### The pre-registered FAIL branch is itself DEFECTIVE — flagged at its source

This is not merely a bad verdict string repeating a good registration. **The
registration is wrong.** `pre_registered_interpretation_recorded_before_the_draw.fail_with_real_power`
— *"the ladder nulls would stand as evidence about TRANSFER rather than about
plumbing"* — **must be treated as defective wherever it is inherited.** Any
successor rung copying this pre-registration block inherits the inversion.
**Mark it at registration time, not at verdict time.**

**Symmetric firewall, now load-bearing**: `FIREWALL_liveness_not_transfer`
forbids citing a **PASS** as transfer evidence; **a FAIL may not be cited as
transfer evidence either.** It is evidence about the apparatus.

### The one finding that is stronger than I stated

| | n_disc | W/L | final wealth | **max wealth** | accuracy | baseline |
|---|---|---|---|---|---|---|
| M4i (trained MLP, **cross-model**) | 66 | 31/35 | 0.2578 | **1.0000** | 128 | 140 |
| PC1b (identity, gold, **same-model**) | 64 | 29/35 | 0.1949 | **1.0000** | 127 | 140 |

**Identical losses (35), adjacent wins, and neither wealth process ever rose
above its 1.0 starting value at any point across 300 items.** A trained
cross-model adapter and **the answer itself**, delivered through the same
apparatus, are indistinguishable. **This is the strongest single line of
evidence that the apparatus — not the payload — determines the outcome**, and
it is what makes "we have not run a passing positive control" the correct
headline rather than a hedge.

## PC1b FINAL — my "unvalidated apparatus" reading is WRONG; the bottleneck is LOCALIZED (2026-08-29)

**Supersedes my inversion argument in the section below.** The implementer
refuted it from the receipt's own numbers and it is right. Recorded here first
because the section below is now only partly correct.

### What refutes me: the norm-matched random control

I argued a failed positive control leaves the apparatus unvalidated, making
the method the leading explanation for every null. **That is wrong, and the
`random` condition is why.** It is a per-item seeded Gaussian, **norm-matched**
to the effective aligned vector, delivered by the **same operator at the same
8 positions** — ITI's floor control (arXiv:2306.03341). Aligned beats it by
**0.237 nats on 198 of 300 items.**

A payload carrying nothing the receiver lacked **could not do that**. The gain
is attributable to the payload's **content**, not to perturbing activations.
**The delivery mechanics are therefore validated as carrying content-dependent
information.** "The apparatus was never shown to carry signal" is **discharged**.

### The precise finding, which is sharper than either position I argued

> **The pathway is LIVE at the likelihood level and DEAD at the decision level.**

- **Live**: moves the gold answer's likelihood **0.282 nats on 71% of items**
  (213W/87L), and beats norm-matched noise by 0.237 nats (198W/102L).
- **Dead**: accuracy falls **140 → 127**, and **loses to the random control's
  133**.
- **Not answer-handover**: mean aligned NLL 2.1110 is **~21x above** the
  0.10-nat collapse threshold, 0/300 degenerate, length ratio 1.034. The
  payload *nudges* the distribution; it does not hand over the answer.

This is a **positive localization of the bottleneck**, not merely a null.

### My pre-draw objection: refuted, on stronger grounds than I conceded

I withdrew it on "the payload is gold-derived." The implementer's grounds are
better: the capture teacher-forces the receiver over the item's **actual GSM8K
gold solution** and taps block 19 at the **last token of that continuation**
(ending `#### <answer>`). At injection the receiver has **the question only**.
This is `docs/research/045` §2 candidate **(c)** — the strongest constructible
ceiling — and explicitly **not** candidates (a)/(b), the self-generated
designs against which my redundancy objection would have had real force.

### M5X and M4b STAY BLOCKED — but my reason was wrong

I blocked them because "the apparatus is unvalidated." That reason is void.
**The correct reason is stronger**: the bottleneck is now localized to
**delivery→decision**, which is **upstream of everything M5X (ADR-037) and
M4b (ADR-035) vary**. Both change payload construction and configuration;
neither touches conversion-to-decision. Running them would vary a factor that
is not the binding constraint.

The **a fortiori** inference holds: a pathway that cannot convert a
**gold-derived, own-model, identity-transformed** payload into correctness will
not convert a **weaker cross-model adapter output**.

### The registered wording is still wrong, on both our accounts

The pre-registration says a powered fail makes the nulls *"evidence about
**TRANSFER** rather than about plumbing."* Both the implementer and I reject
this independently. The nulls are evidence about **the delivery→decision
step** — a third category that is neither plumbing nor transfer. The receipt's
own firewall says it plainly: *"this FAIL says nothing directly about
cross-model transfer."* **The symmetric firewall rule stands: a FAIL may not be
cited as transfer evidence, just as a PASS may not.**

### The honest residual gap, and the next experiment

**No control in this repository has yet shown the pathway can move ACCURACY.**
PC1b shows nonzero bandwidth at the likelihood level and **zero** at the
decision level.

What does not exist is a control isolating **decision-level steering**: inject
a state encoding a **different** answer and test whether the receiver's answer
**flips**. PC1/PC1b are *restoration* controls; this would be a *steering*
control. **Registered as the next thing worth building — not claimed as done.**

### Correction: PC1's manifold anomaly is NOT resolved

I wrote that the item-invariance surprise "was an instrument artifact, not a
payload property." **The matched-N diagnostic
(`run2-pc1b-precheck-n-invariance-receipt.json`) shows I overstated.** At
**matched N=40**, PC1b's own payload classifies **`item-invariant-but-on-manifold`
— exactly as PC1's did — with a *lower* union (86 vs 98)**.

So: the **flip between the two rungs** is an N artifact (`flip_is_an_n_artifact
= true`), and `classify()`'s label is not comparable across N. But the
**underlying item-invariance at N=40 is real and reproducible in both
payloads**, and **remains unexplained**. Only the cross-N comparison is void.
`"PC1's surprise flag is NOT resolved by PC1b."`

## PC1b RESULT — the gating experiment FAILED, and the pre-registered interpretation of that failure is LOGICALLY INVERTED (2026-08-29)

**Receipt**: `run2-pc1b-receipt-identity-L19lasttoken-goldtf-fuse-questiontail-slots8-eprocess.json`
(773,964 B). Binary of record `4cda2ef1…`. e-process **pass = false**,
final wealth **0.1949**, never above its 1.0 start, threshold 20.0 never
approached.

### The numbers

| | aligned | baseline | zerovec | random |
|---|---|---|---|---|
| **accuracy /300** | **127** | 140 | 140 | 133 |
| NLL mean | **2.1110** | 2.3928 | 2.3928 | 2.3477 |

- e-process (aligned vs random): **29W / 35L, n_disc = 64** — minimum
  attainable p ≈ 5.4e-20. **The most powered null in the ladder.**
- NLL sign tests: aligned-vs-baseline **213W / 87L**, aligned-vs-zerovec
  213W/87L, aligned-vs-random 198W/102L.
- Operator correctness **proven**: `fuse_zero_is_noop` — 300/300 items
  bit-identical to baseline, max |Δ NLL| = **0.0**.
- Gaming guard **clean**: length ratio 1.034, 0/300 degenerate,
  `gaming_signature = false`.

### Three facts that matter more than the pass/fail

**1. The pathway is NOT inert — the auto-generated verdict is wrong on its own
numbers.** The receipt's verdict string says *"the pathway is inert."* Its own
NLL says otherwise: **0.282 nats better than baseline on 213W/87L over 300
items** — roughly **twice** M4i's 0.148 nats / 181W-119L, and two orders of
magnitude outside the ~0.004-nat inertness band. This payload registers more
strongly than anything else in the ladder. The accurate phrase remains the one
adopted earlier, now with far more current flowing: **the channel is
electrically connected and functionally dead.**

**2. The payload is WORSE THAN NOISE, reproducibly.** Aligned (127) < random
(133) < baseline (140), and the same ordering holds in M4i (128 < 132 < 140).
Since `zerovec` is proven bit-identical to `baseline`, injecting *anything* at
this site costs ~12-13 correct answers — and injecting the **gold-derived**
payload costs more than injecting Gaussian noise.

**3. A trained cross-model adapter and a gold-derived identity injection are
statistically indistinguishable.**

| | M4i (trained adapter) | PC1b (identity, gold-derived) |
|---|---|---|
| accuracy aligned / baseline | 128 / 140 | 127 / 140 |
| acc aligned-vs-baseline | 24W / 36L | 21W / 34L |
| e-process n_disc | 66 | 64 |
| e-process W / L | 31 / 35 | 29 / 35 |
| final wealth | 0.2578 | 0.1949 |

Whether the payload is a learned cross-model alignment or **the literal
answer**, the outcome is the same.

### My pre-draw concern was WRONG, and the receipt refutes it

Before the verdict I argued PC1b might be informationally incapable of
passing — that feeding the receiver its own state carries no information it
lacks, making a null the predicted outcome of a healthy pathway. **The receipt
refutes this.** The payload is `gold-TF` — captured with the gold answer
**teacher-forced** — and `FIREWALL_liveness_not_transfer` records the content
as **"gold-derived"**. It therefore carries information the receiver does not
have at inference time. **PC1b was informationally capable of passing and did
not pass.** My objection is withdrawn.

### THE INTERPRETATION REGISTERED BEFORE THE DRAW IS LOGICALLY INVERTED

`pre_registered_interpretation_recorded_before_the_draw.fail_with_real_power`
says a powered failure means:

> "the pathway is inert EVEN WHERE PAYLOADS DEMONSTRABLY REGISTER, and the
> ladder nulls would stand as evidence about **TRANSFER** rather than about
> plumbing."

**This is backwards, and I am recording it against my own pre-registration.**

A positive control exists to prove the apparatus can carry a signal it is
*known* to contain. **PC1b failed.** So we have **not** established that these
injection mechanics can carry usable signal — for gold content, same model,
same item, identity transform, at a site where delivery is proven. A failed
positive control makes **the method the leading explanation** for every null
in the ladder; it cannot simultaneously **rule the method out** so the nulls
can be read as being about transfer. Ruling out plumbing requires the control
to **pass**.

Steelmanned the other way — *"delivery is proven by the 0.282-nat shift, so the
failure is downstream, in the receiver's ability to use injected content"* —
the conclusion is still about the **method**, not about transfer: it says
*inject-a-vector-and-expect-use* fails for **any** payload, including the
answer itself. Either reading lands in the same place.

**Therefore, and contrary to what the binary printed:**
- The ladder's cross-model nulls remain **confounded with an unvalidated
  apparatus**. They are **not** upgraded to evidence about transfer.
- **M5X (ADR-037) and M4b (ADR-035) stay BLOCKED.** The gate was "a passing
  positive control," and it did not pass. Nothing about a failure unblocks
  them; a failure blocks them harder.
- The mission's honest position is **"we have still not run a passing positive
  control,"** which is a worse outcome than a clean transfer null and is the
  one the evidence supports.

`FIREWALL_liveness_not_transfer` already forbids citing a PASS as transfer
evidence. **The symmetric rule is now added: a FAIL may not be cited as
transfer evidence either.** It is evidence about the apparatus.

## The "source was overwritten" claim is WITHDRAWN — unsupported (2026-08-29)

Recorded because it goes **against** the agent that reported it: the draw
owner states both broken lookups are in **its own** source, that it never
wrote a hard gate there, and that no overwrite occurred. Verified, and it is
right on every checkable point:

- **There is no `ensure!` on the stream check anywhere in the probe.** The
  `m4i[...]` lookups sit at lines **363, 373, 375** and none is inside a gate.
  So **no hard stream-identity gate ever existed in the probe to be replaced.**
  My previous section said one was; that is withdrawn.
- Its account of its own source **matches the file exactly** — including the
  detail that line 373 reads `m4i["site_change"]["to"]` before falling back to
  `m4i["config"]["site"]`. Neither key exists in M4i.
- The single hard `ensure!(m4i_stream == stream)` is in the **capture** binary,
  which reads `m4i["items"]` correctly. **The two binaries disagree because one
  is wrong, not because either was tampered with.**
- **Authorship cannot be established either way**: `run2_pc1b_probe.rs` is
  **untracked** — no git history exists for it. Recorded as unverifiable
  rather than assigned.

**The binary difference stands, with its innocent explanation.** `/proc/2615231/exe`
resolves to a path marked **`(deleted)`** — the running inode was unlinked and
replaced, so the on-disk artifact (**13,102,888 B**, mtime 02:09) is
**definitively not the draw's binary**. The proposed cause — the 02:09 rebuild
omitted `--features cuda` (30.4 MB cuda build vs 13.1 MB non-cuda) — is
**plausible and unrefuted**, but I could not measure the running binary's size
(a deleted inode reports 0 via `/proc`), so it is recorded as an explanation,
not a verified fact. The load-bearing part needs neither: **the on-disk
artifact is not the draw's binary**, and `4cda2ef1…` remains the binary of
record.

**Confirmed for the comparison this unblocks**: M4i `n_disc = 66`,
`final_wealth = 0.25780744271218675`, `config.injection_site =
"question_tail_ordinary_tokens"` — and PC1b's capture stream equals M4i's 300
ids exactly, head `[4, 20, 23, 39, 42]`, tail `[4457, 4464, 4465]`.

## STRUCTURAL: peer message attribution is not determinable in this session

Five inbound messages arrived labelled only `from=general-purpose`. This
session has **two** agents of that type, and the header carries no identity.
The consequences are now concrete, not hypothetical:

- The draw owner says it sent **four**; I received **five**.
- One message denied a process kill that two others affirm — and the kill is
  **independently proven** by `pc1b_probe.log` on disk.
- The "hard gate was replaced" claim and its withdrawal both arrived attributed
  to the same agent, one contradicting the other.

**I cannot resolve which agent sent what, and I am not going to guess.** Every
attribution in the sections above should be read as *"reported by an agent of
type `general-purpose`"*.

**This is the durable lesson of the whole episode, and it outranks the science
here**: in a multi-agent session where messages carry only an agent *type*,
**provenance cannot be established, so no claim may enter this record on
authority — only on primary artifact.** Every finding that survived this
exchange survived because it was checked against a receipt, a log, or source:
the lens N-invariance defect, the stream identity, the determinism check, the
deleted-inode binary. Every claim that collapsed — the missing site tag, the
source overwrite, the replaced gate, "no second process" — collapsed because
it rested on a message. **The messages were not the evidence; they were only
ever pointers to where to look.**

## THE RETRACTION IS WITHDRAWN — error #12 was not an error, and my correction was the bigger mistake (2026-08-29)

**Read this before the section below it.** The retraction that follows is
**wrong** and is preserved only because this record is append-only. The
original claim — PC1b's stream is identical to M4i's — **is true and is now
verified from primary artifacts.**

The probe's alarming log line is a **false negative produced by two
JSON field-name lookup bugs.** Verified in source and receipts:

| probe reads | exists in M4i? | what M4i actually uses | value |
|---|---|---|---|
| `m4i["config"]["site"]` (line 375) | **no** | `config.injection_site` | `"question_tail_ordinary_tokens"` |
| `m4i["dataset"]["indices"]` (line 363) | **no** (no top-level `dataset`) | `items[*].item` | 300 ids |

Both resolve to null, so the tag prints `<unrecorded>` and the stream
comparison runs against an empty vector. The bug's origin is visible: the
probe **writes** its own receipt with a top-level `"dataset": {"indices": …}`
(line 711) and assumed M4i had the same shape. It does not.

**My own independent derivation, not the peer's and not the gate's
self-report:**
```
M4i  ids: n=300 head=[4, 20, 23, 39, 42] tail=[4457, 4464, 4465]
cap  ids: n=300 head=[4, 20, 23, 39, 42] tail=[4457, 4464, 4465]
cap_ids == m4i_ids -> True
```
And the **capture** binary — a *different* binary from the probe — reads the
correct key `m4i["items"]` (line 252) behind a **hard gate**,
`anyhow::ensure!(m4i_stream == stream)` (line 260). It passed; had it failed
the run would have aborted before writing any payload. Its committed gate:
`/gates/stream_identical_to_m4i → {"pass": true, "n": 300, "note": "…item for
item and in order — the site claim is checked, not asserted"}`.

**Restored, and the block I imposed is lifted**: PC1b runs at the same site,
same tag, same 300 items in the same order as M4i. **The comparison to M4i's
n_disc = 66 IS available.** There is no missing site tag in M4i's receipt and
no gap to flag in that rung — I withdraw that too.

### The process lesson, which is the opposite of the one I drew

I got this wrong **twice in one hour, in opposite directions, by the same
mechanism**:

1. I asserted *"site provenance proven"* on a peer's **prose**, without
   checking → wrong process, **right** conclusion.
2. I retracted it on a **single probe log line**, without checking → wrong
   process, **wrong** conclusion.

The failure was never "trusting a teammate." It is **acting on one
unverified signal**, and a refutation deserves exactly the scepticism an
assertion does. My retraction was the more damaging of the two: it would have
voided a valid comparison and manufactured a nonexistent defect in M4i's
receipt. **Scepticism applied in only one direction is not rigour.**

### The real finding underneath, which stands

The binary now running (`4cda2ef1…`) is **not built from the source the
capture author wrote.** Their version asserts stream identity as a *hard
gate*; the running version replaced it with a **soft, non-gating print**
carrying both broken lookups. Consequences:

1. **The draw is unaffected** — every pre-registered constant is identical
   (`SITE`, `INJECT_MODE`, `LAMBDA` 0.30, `E_ALPHA` 0.05, `N_MAX` 300, all
   three gaming-guard thresholds), verified before the flag was raised.
2. **The probe's receipt will understate its own provenance**, populating
   `site_provably_identical_to_m4i` from the two broken lookups. The
   corrected, independently-derived values are to be written in, with the
   binary's raw line preserved **verbatim beside them, labelled as a false
   negative from two field-name errors** — not as a finding.
3. **Two agents editing one rung's source while its draw is in flight** is the
   actual process failure, and it is the **same root cause as the duplicate
   launch**: I created multiple concurrent paths onto one registered rung.
   That is coordinator error #11's lesson recurring, not a new one.

## RETRACTION — coordinator error #12: I recorded "site provenance proven" and it is FALSE (2026-08-29)

**The most serious error of this mission, and it is entirely mine.** In
commit 8382f11 I wrote, under PC1b's pre-draw facts:

> **Site provenance proven at the strongest available level**: the item
> stream is **identical to M4i's committed stream** ... not argued in prose.

**This is false.** The probe binary prints the answer on its own stdout, in
both process logs, identically:

```
site provenance: tag question_tail_ordinary_tokens
  (M4i recorded <unrecorded>, match=false); stream identical to M4i's = false
```

`match=false`. `stream identical to M4i's = false`. I amplified an
implementer's prose claim into an ADR as **"proven"** without running the one
`grep` that would have falsified it — while the falsifying artifact sat in
the same log directory I had already read from twice.

**What is actually true, stated precisely so the retraction is not
over-corrected either.** M4i recorded **`<unrecorded>`** — it never committed
a site tag. So `match=false` means *"there is nothing to compare against"*,
**not** *"a mismatch was detected"*. The honest label is **unverifiable**,
which is exactly what `<unrecorded>` denotes. PC1b's site is the right *kind*
of site — the pre-flight resolves 300 items to 8 ordinary question-tail
positions with 0 tokenisation exclusions, and the decoded spans are genuine
question content (item 4 → `" How many pages does he write a year"`). What
cannot be claimed is **bit-level stream identity with M4i**, and therefore
PC1b is **not** the clean site-matched repeat of M4i I described it as.

**Consequence for interpreting PC1b, registered before the verdict lands**:
its result must be read as "the positive control at *a* question-tail site",
**not** "at M4i's site". Any conclusion of the form *"the same site that gave
M4i n_disc=66 does/doesn't carry the receiver's own state"* is **not
available** from this draw. M4i's missing site tag is the root cause and is
now a known gap in that rung's receipt.

**Process lesson**: the failure mode is not that a peer overstated something.
It is that **I promoted an unverified claim to "proven" in an append-only
record**, which is the strongest word available, on prose alone. Peer claims
about receipts get verified against the receipt before they enter an ADR — no
exceptions, and least of all for the claims that sound most reassuring.

## Coordinator error #11 — RESOLVED AFFIRMATIVELY: the second process existed

I recorded "whether a second process ever existed is unresolved". **It is now
resolved: it existed.** Durable evidence, verified directly:

- `scratchpad/pc1b_probe.log` — **1377 bytes, mtime 02:08**, a second log path
  distinct from the survivor's `pc1b.log`. It contains a full independent
  startup (build-env guard, payload verification, pre-check, model load, site
  pre-flight) and **terminates at `[2/300]`** — exactly where a process killed
  at ~35 s stops.
- Both logs carry the **same payload sha** `237f7bf4…2cde` and the **same**
  pre-check verdict.

So my `ps` reconstruction was right about the **end state** and wrong about
the **history** — I observed the survivor *after* the duplicate had already
been killed. The withdrawal of the kill in the previous section is itself
withdrawn; the implementer's original account was correct.

## A free determinism check on PC1b, obtained by accident

The two concurrent processes ran the same stream and **agree bit-identically
on every overlapping item**:

| item | aligned / baseline / zerovec / random NLL | W | wall-clock |
|---|---|---|---|
| 4 (survivor) | 2.872 / 2.965 / 2.965 / 3.291 | 1.0000 | 11s |
| 4 (duplicate) | 2.872 / 2.965 / 2.965 / 3.291 | 1.0000 | 15s |
| 20 (survivor) | 2.536 / 2.268 / 2.268 / 2.017 | 1.0000 | 24s |
| 20 (duplicate) | 2.536 / 2.268 / 2.268 / 2.017 | 1.0000 | 29s |

**Only wall-clock differs.** This is a determinism check on PC1b's own draw
that we did not design and did not pay for — the same accidental-replication
gift the unregistered PC1 re-draw produced. The duplicate launch was my error;
the determinism evidence is real regardless.

## Sharpening: PC1b's `surprise=false` is NOT a clean bill of health

I wrote that the instrument defect means "the condition for interpreting PC1b
is satisfied — the surprise was never real." That is too generous to PC1b.
The correct statement, adopted from the implementer:

> The surprise was never a payload property, **and PC1b's non-surprise is
> equally uninformative.**

It is the same defective instrument reporting differently because N changed.
On every N-invariant measure PC1's and PC1b's payloads are indistinguishable.
PC1b's pre-check provides **no independent assurance** about its payload — it
merely fails to raise an alarm it is structurally incapable of raising at
N=300.

## Provenance of these findings is AMBIGUOUS — recorded as such

Inbound peer messages are labelled only by agent **type**, and this session
has **two** `general-purpose` agents. The PC1b draw owner states it sent
neither the rebuilt-binary analysis nor the `lens.rs` defect report, and
declines attribution on the grounds that it cannot vouch for how they were
derived. **I cannot resolve authorship from the message headers.**

**This does not weaken either finding**: I verified the `lens.rs` defect
directly against source and both committed receipts, and the binary-hash
divergence against `/proc/2615231/exe`, before recording them. Both stand on
primary evidence, independent of who first reported them. Attribution in the
sections above should be read as **"reported by a general-purpose agent,
verified independently by me"** rather than as crediting a specific agent.

## INSTRUMENT DEFECT — `classify()`'s token-union arm is not N-invariant (2026-08-29)

**Found by the PC1b implementer, verified independently against source and
receipts.** I had hoped PC1b's `on-manifold-item-varying` classification
resolved the PC1 `item-invariant-but-on-manifold` anomaly I flagged as
blocking interpretation. **It does not. The flip is an artifact of the
instrument, and the anomaly is neither resolved nor real.**

`examples/common/lens.rs:38`:
```rust
let invariant = invariance >= COLLAPSE_COSINE || token_union <= COLLAPSE_TOKEN_UNION;
```
`COLLAPSE_TOKEN_UNION` is **120, an absolute count**, but the union's
attainable maximum is **10 x N**. The constant's own doc comment scopes it:
*"Distinct tokens in the union of the **40** top-10 sets ... (max possible
400)."* It was defined for N=40 and silently reused at N=300.

| | N | union | max | as % of max | dominant token in | inv-cos | manifold-cos | label |
|---|---|---|---|---|---|---|---|---|
| PC1 | 40 | 98 | 400 | **24.5%** | **40/40 items (100%)** | 0.8565 | 0.6869 | item-invariant-but-on-manifold |
| PC1b | 300 | 219 | 3000 | **7.3%** | **297/300 items (99%)** | 0.8696 | 0.6905 | on-manifold-item-varying |

Every N-invariant measure says these are **the same payload family**:
invariance cosine differs by 0.013, manifold cosine by 0.004, and the
dominant single token sits in the top-10 of **100% vs 99%** of items. Only the
absolute union count differs, and it differs *because N differs*. As a
fraction of attainable support PC1b is **more** concentrated, not less —
the label flipped to the *less* alarming value on the *more* concentrated
payload.

**Scope of the damage — deliberately narrow, and I checked rather than
assumed.** The defect is confined to the **invariance axis**, and only via
the `token_union` disjunct:
- `COLLAPSE_COSINE` compares a **mean pairwise cosine** — N-invariant. Unaffected.
- `OFF_MANIFOLD_COSINE` compares a **mean cosine** — N-invariant. Unaffected.

Therefore **the ladder's load-bearing conclusion survives intact**: the
on-manifold/off-manifold axis is sound, so *"on-manifold = inert (within
0.004 nats of baseline), off-manifold = actively destructive (NLL 4.8-5.4 vs
2.13)"* still stands on N-invariant measurements.

What does **not** survive: **any cross-rung comparison of the
`item-invariant` label where N differed** is void. Documents citing that
label — [research/033](../research/036-manifold-collapse-across-the-ladder.md),
036, 038, 039, 040, 041, 046 — are annotated by this section rather than
rewritten, per the append-only rule. Note that the receipts already carry the
tell in their own field names: PC1 wrote
`distinct_tokens_in_union_of_40_top10_sets`, PC1b wrote
`distinct_tokens_in_union_of_all_item_top10_sets`.

**Registered fix, deferred deliberately**: make the arm a fraction
(`token_union as f64 / (10*n) as f64 <= COLLAPSE_TOKEN_UNION_FRAC`) or use
the already-recorded N-invariant dominant-token share. **Not applied yet** —
a rebuild of `run2_pc1b_probe` already occurred *under* the live draw (below),
and I will not touch this crate again until PC1b lands. The pre-check is
ordering-only and gates nothing, so no result is at risk from the delay.

**This is the second instrument caveat on the same pre-check**, after PC1's
`manifold` metric caveat. The pre-check is now the least trustworthy
instrument in the ladder and its labels should not be quoted as findings.

## Binary changed under the live PC1b draw (2026-08-29)

`examples/run2_pc1b_probe.rs` was edited (mtime 02:09:09) and rebuilt at
**02:09:28 — after the draw started at 02:08:07**. The draw is unaffected:
Linux holds the original inode open, and the two differ:

- **binary that is producing the draw** (`/proc/2615231/exe`): `4cda2ef1d60c1df716f92991f98f356c8731c99973caf18f725e5ac4f5d11922`
- on-disk binary now: `a24c0d992a8ce3a9e2a9abb1603e77913198515d7c924cb413e55da01d9584ab`

The implementer checked every pre-registered constant in the new source —
`SITE`, `INJECT_MODE`, `LAMBDA` 0.30, `E_ALPHA` 0.05, `N_MAX` 300, and all
three gaming-guard thresholds — **all unchanged**; the edit is
prose/receipt-field only. **`4cda2ef1...` is the binary of record for PC1b.**
Recorded because a rebuild under a live registered draw is exactly the kind
of event that must not be discovered later from a hash mismatch.

## Correction to coordinator error #11 (same day)

My account of #11 credited the implementer with killing its own duplicate
process as a first-claim-wins tiebreak. **It disputes this and says it issued
no kill at all**, having launched exactly one probe. Its two messages to me
are mutually inconsistent on that point, so I record only what I verified
myself: `ps` shows **exactly one** `run2_pc1b_probe` process (2615231), whose
parent 2615230 is that implementer's own `nohup` launcher shell.

**Whether a second process ever existed is unresolved and I am not asserting
either way.** What stands unchanged is the part that matters and that is
entirely my fault: **I created two spawn paths for one registered rung** — an
agent resumed with an approval, plus a workflow whose implementation agent
does the same work. The process lesson is unaffected; only my narration of
who cleaned it up was wrong.

## Coordinator error #11 — a duplicate launch of one registered rung (2026-08-29)

**I launched PC1b twice.** I resumed the positive-control researcher with an
approval to proceed *and*, separately, launched workflow `wf_82867994` whose
implementation agent runs the same rung. Two processes
(`run2_pc1b_probe` pids 2615231 and 2615993) were executing concurrently
against **one binary, one payload and one receipt path** — a race on the
output file, and two concurrent draws of a single registered rung, which is
precisely the unregistered-re-draw problem recorded for PC1 an hour earlier.

**Caught and resolved by the implementer, not by me**: it killed **its own**
duplicate rather than another agent's process and let the first claim proceed.
**Ratified**: first-claim-wins; pid 2615231 is the authoritative draw. Both
draws are greedy, fixed-seed and deterministic over an identical sha-pinned
payload, so no evidence is at risk — but per
[ADR-032](032-negative-result-publication-contract.md) **the fact that a
duplicate launch occurred belongs in the record regardless**, and if a second
receipt exists it is to be preserved with a `-duplicate-launch` suffix rather
than deleted.

**Process lesson, recorded because it will recur**: resuming an agent to do
work *and* launching a workflow for the same work are two spawn paths that do
not see each other. Rungs need an explicit claim before launch when both
paths are in play.

## PC1b pre-draw facts already established (recorded before the verdict)

From `run2-pc1b-capture-receipt.json`, all verified before the draw:
- **The replacement gate I approved passed EXACTLY, not merely adequately**:
  bit-identity against PC1's committed vector at the single shared item
  (train 1153) returned **max |Δ| = 0.000e0**. The claim "same payload
  derivation" is now *demonstrated byte-for-byte*, not asserted.
- **Site provenance proven at the strongest available level**: the item stream
  is **identical to M4i's committed stream** (300 items, `adaptation-512`
  fixed index order, 13-item exclusion applied, 0 tokenisation exclusions) —
  not argued in prose. Sample injection sites are genuine question content
  (item 4 → " How many pages does he write a year").
- **No adapter weights constructed** in either binary.
- **The PC1 manifold anomaly may be resolved**: PC1b's payload classifies
  **on-manifold-item-varying, surprise = FALSE** — the registered expectation —
  where PC1's classified *item-invariant*. Since PC1b reuses PC1's derivation,
  the difference must come from the stream (300 items vs 40) or payload
  composition, not the derivation itself. **This matters**: I flagged the PC1
  anomaly as needing resolution *before* PC1b could be interpreted, and it
  appears to be a property of PC1's small item set rather than a pipeline
  defect.

## PRECISION CORRECTION — "fails to register" was too strong (2026-08-29)

PC1's full report refines my own correction one level further, and the
refinement matters.

I wrote that the `<|fim_pad|>` site is "where payloads **fail to register**".
**That is too strong.** PC1's receipt shows that at the placeholder site
**every one of the 40 items' aligned NLL differs from baseline** (15 wins +
25 losses = 40 discordant on the NLL arm — *not* power-limited), and the
generated text changes. The payload demonstrably **does** reach the forward
pass at `<|fim_pad|>`. The implementer's phrase is the accurate one and is
adopted: **"the channel is electrically connected and functionally dead."**

**What the site actually changes is the magnitude and sign of the effect, not
whether one exists:**

| site | aligned vs baseline NLL | direction | accuracy effect |
|---|---|---|---|
| `<|fim_pad|>` (PC1, identity payload) | 2.1539 vs 2.1288, 15W/25L | slightly **worse** | **0W/0L** — literally no answer changed |
| question-tail (M4i, M3 payload) | 2.2446 vs 2.3928, 181W/119L | **0.148 nats better** | none (128/300 vs 140/300) |

**The sharpest single fact in the PC1 receipt**: injecting the receiver's own
state changed **zero** of 40 answers relative to baseline, while an
**information-free norm-matched Gaussian moved three**. The pathway perturbs
the forward pass and never converts that perturbation into a different
answer. Anti-gaming gates are clean (generated-character ratio 1.026, zero
degenerate-short outputs, no NLL collapse), so this is not a degenerate-output
artifact.

**Two further facts from the receipt worth preserving:**
- **Capture-path parity was proven, not assumed**: on the 13 S1a items the
  committed dump covers, PC1's capture path reproduced the dump's block-19
  rows **bit-identically** — 5,483,520 elements, max |Δ| = 0.0 — which is
  what licenses using the same path for the 27 items the dump does not cover.
- **A pipeline surprise, flagged by the implementer**: PC1's payload — the
  receiver's *own* de-pooled state — classified **"item-invariant-but-
  on-manifold"** rather than the expected "on-manifold-item-varying"
  (inv-cos 0.8565, entropy 4.24). A real receiver state should be
  item-varying. That is a finding about the payload pipeline, not about
  transfer, and it is unexplained. It should be resolved before PC1b's result
  is interpreted, since PC1b uses the same derivation.

**PC1b's framing is unchanged** — it still separates "pathway inert" from
"placeholder site inert" — but the question it answers is now sharper:
**does moving the site convert a registering-but-useless perturbation into an
answer-changing one?**

## PC1b spec conflict (my tenth error) + an unregistered PC1 re-draw (2026-08-29)

**1. Coordinator error #10, caught by the implementer before any GPU time.**
My PC1b brief required both *"payload sha256 matches PC1"* and *"evaluate
under the e-process at N_max = 300"*. Those are **mutually exclusive**: PC1's
payload file holds exactly **40** vectors (S1a's items), while the e-process
stream draws 300 from `adaptation-512` — and **ADR-036's own Decision 2
already records that those sets intersect in exactly one item** (train index
1153). A byte-identical payload file would have confined PC1b to the same
40-item powerless regime PC1 already occupied (n_disc 3), which is precisely
what the rung exists to escape. I wrote a gate that made the experiment
impossible and did not notice.

**Approved resolution, which is stronger than my spec**: hold the payload
**derivation rule** byte-identical rather than the payload *file* — same code
path (`render_gold` + `forward_capture_multi_with_rows` at block 19, last
span row), reused verbatim from `run2_pc1_capture.rs`, not reimplemented —
and replace the unsatisfiable file-sha gate with three checkable ones:
(a) PC1's committed payload file still matches its receipt-pinned sha256;
(b) the derivation is provably the same code path; (c) **bit-identity on the
overlap item**: the freshly derived vector for train 1153 must equal PC1's
committed vector byte-for-byte. That is a genuine cross-artifact equality
check and it preserves what the gate was *for* — proving the payload is the
receiver's own state — while permitting the power the rung needs. The
no-adapter-weights gate is unchanged; the ~35 s capture is declared as a
receipt-level deviation exactly as PC1 declared its own.

**2. An unregistered PC1 re-draw occurred, and is recorded per
[ADR-032](032-negative-result-publication-contract.md).** A second PC1 run
(pid 2484574, finished 01:57:30) executed and **overwrote the committed PC1
receipt**. The implementer diffed it against HEAD before restoring: accuracy,
the primary comparison (1W/2L, n_disc 3, p = 0.875) and **all four NLL means
were bit-identical**; only three non-scientific lines (timing, env) differed.
The file has been restored to HEAD. **No evidence was corrupted**, and the
event doubles as an **unintended determinism check on PC1**, which passed.
Both facts belong in the record: an unregistered re-draw happened, and it
changed nothing.

## CORRECTION + a critical interaction between M4i and PC1 (2026-08-29)

**My M4i entry below understates the result, and the fuller receipt changes
what PC1 can be taken to prove.**

**1. The injection-site hypothesis is PARTIALLY SUPPORTED, not unsupported.**
I wrote that moving to ordinary tokens "did not rescue transfer" and that the
NLL improvement "must not be read as partial transfer". The first half is
wrong as stated. Measured over 300 items with the **same artifact and the
same payload derivation** as M4h Stage 1: aligned mean gold NLL **2.2446 vs
baseline 2.3928 — 0.148 nats better, 181W/119L**. Every prior on-manifold
configuration at the `<|fim_pad|>` site sat inside a **~0.004-nat inertness
band**. This is **two orders of magnitude outside it**. Moving off the
placeholder site **makes the payload register in the forward pass at all** —
direct support for [docs/research/043](../research/043-placeholder-token-choice.md)'s
mechanism. What it does *not* do is convert into accuracy (aligned 128/300 vs
baseline 140/300). The honest verdict is the implementer's: **the placeholder
site is a real but non-load-bearing structural difference** — real because it
changes the forward pass measurably, non-load-bearing because the change does
not become correctness.

**2. The interaction that matters: PC1 RAN AT THE PLACEHOLDER SITE.** PC1
used 8 `<|fim_pad|>` slots — precisely the site M4i has now shown is where
payloads *fail to register*. So PC1's failure is confounded: it may show
that **the current mechanics cannot carry signal**, or merely that
**`<|fim_pad|>` cannot carry signal**, which M4i independently demonstrates.
**PC1's conclusions above are hereby narrowed**: every caveat it triggers
applies to *the placeholder-site configuration*, and the sweeping claim that
"current mechanics have never been shown to carry signal" is **not
established** until a positive control runs at **ordinary tokens**.

**3. Consequent decision, registered now**: **PC1b** — repeat PC1 (receiver's
own L19 states, identity transform, fuse, de-pooled) at the **question-tail
ordinary-token site M4i used**, under the e-process. This is now the single
highest-value experiment available: it separates "our pathway is inert" from
"the placeholder token was the problem", and both of those readings currently
have real evidence. It remains **prerequisite to M5X and M4b**, which stay
blocked.

**4. What survives unchanged from PC1's outcome**: S1a's pass is still scoped
to retired mechanics, and the mechanical-discordance-suppression rival to my
"adapters improved" account still stands — PC1 had no adapter and n_disc 3.

## PC1 OUTCOME (2026-08-29) — THE POSITIVE CONTROL FAILED. This is the consequential branch.

`run2-pc1-receipt-identity-L19lasttoken-goldtf-fuse-slots8-nopool-*.json`.
The receiver was injected with **its own** gold teacher-forced block-19
states, via **identity transform** (gate confirms *no* adapter weights are
constructed anywhere in the binary), under **current mechanics** (fuse,
de-pooled, 8 `<|fim_pad|>` slots, rescale-to-median), scored with S1a's
original 40-item test.

**Result: aligned 22/40, baseline 22/40, random 23/40, zerovec 22/40.
`aligned_vs_baseline` = 0 wins, 0 losses — the model's own state, returned to
it, changed nothing at all. Primary vs random: 0.875 (n_disc 3, itself
power-limited).** NLL slightly worse than baseline (2.1539 vs 2.1288, 15W/25L).
Anti-gaming gates clean (no degenerate-output or NLL-collapse signature).

**Per the outcome rule registered before the run, and not softened:**
1. **Every null since S1a is now caveated.** S2b ×4, M3 ×2, M4 ×3, M4c, M4d,
   M4g, M4h S1 and M4i were all drawn under mechanics that have **never been
   shown capable of carrying signal — even from the model to itself**. They
   cannot be read as evidence about *transfer*; they are consistent with a
   delivery pathway that is simply inert.
2. **S1a's pass is scoped to retired mechanics.** It used overwrite + pooled
   + the 40-item test; nothing since M4c uses that configuration. Its p =
   0.031 remains valid *for those mechanics* and does not license any claim
   about the current ones.
3. **The rival explanation I owe the record**: my account that "discordance
   fell (7→3→2) because adapters improved" now has an unruled-out
   competitor — **fuse + de-pooling + the fim_pad site may suppress
   discordance mechanically**, independent of adapter quality. PC1 is exactly
   the case where adapter quality is not a variable (there is no adapter),
   and discordance was 3. That is evidence *for* the rival.
4. **Firewall unchanged**: PC1 was designed so a pass would prove liveness
   only. A *fail* proves the converse and is strictly more informative.

**What PC1 does NOT establish**: that latent transfer is impossible; that the
models lack shared structure (the permutation null already refuted that); or
which mechanic is responsible — fuse, de-pooling, the `<|fim_pad|>` site, or
the rescale. Isolating that is the necessary next work, and it is now
**prerequisite to every remaining rung**, including M5X (ADR-037) and M4b
(ADR-035). **Neither should run until a positive control passes.**

## M4i OUTCOME (2026-08-29) — the first properly-powered cross-model null

`run2-m4i-receipt-...-questiontail-slots8-eprocess.json`, the first rung under
ADR-036's e-process. It drew the **full N_max = 300** items and reached
**n_discordant = 66** — an order of magnitude more discordance than any
40-item draw, exactly as ADR-036 intended. Wins 31, losses 35. **Final wealth
0.258 against a threshold of 20.0; max wealth ever reached 1.0; never crossed.
FAIL.**

**This is the ladder's first cross-model null with genuine statistical
power** — the instrument could have detected a real effect and did not.
Under the pre-registered interpretation, the injection-site hypothesis
(`<|fim_pad|>` being near-vacant) is **not supported**: moving to ordinary
question-tail tokens did not rescue transfer.

**One observation recorded but not over-read**: aligned NLL **2.2446** beat
baseline **2.3928**, random 2.3474 and zerovec 2.3928 — the only rung where
the aligned payload improved likelihood over every control. Accuracy went the
other way (128 vs 140 over the drawn items). Given PC1's failure, **this must
not be read as partial transfer** — a pathway not shown capable of carrying
signal cannot be credited with carrying some. Recorded for the write-up as an
anomaly deserving explanation, not as evidence.

## RESOLVED — the authoritative power table (2026-08-29)

[docs/research/047](../research/047-authoritative-power-table.md) settled the
discrepancy below **from primary evidence**: every `n_disc` recomputed by
counting discordant items in each receipt's own per-item table, then
cross-checked against that receipt's summary. No number in it is copied from
prose in this ADR, 031, 041 or 046.

**The authoritative finding: 14 valid draws exist across both runs, and 10 of
them were structurally incapable of rejecting the null at any true effect
size.** Only **four** draws could ever have produced a significant result:
S1a (n_disc 5), M4 r256 (5), M4c (6), M4d (7).

**The "6 of 9" figure I have repeated throughout this ADR and in every
summary is WRONG.** The correct figure for that family is **8 of 9**.
research/046's independent recount (10 of 13 cross-model draws incapable) is
**confirmed at the receipt level**.

`n_disc` across all 14 draws: **{2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6, 7}**.

**What this changes**: it *strengthens* the instrument finding rather than
altering any verdict. The honest ladder-wide statement is now sharper —
**most of this ladder's draws could not have detected an effect had one been
present**, and only the four capable draws (one of which, S1a, *passed*)
carry evidential weight about absence. Every future write-up must cite
research/047's receipt-derived table rather than restating a remembered
figure; that is the third time tonight a restated number drifted from its
receipt (after the GPU-hour total and the permutation count), and the
pattern is now itself a recorded finding.

## UNRESOLVED DISCREPANCY — how many draws were power-incapable? (flagged 2026-08-29)

[docs/research/046](../research/046-run2-synthesis-v2.md) found an internal
arithmetic inconsistency in
[docs/research/041](../research/041-run2-synthesis-skeleton.md) §2.5, which
states both *"10 draws, 6 at n_disc ∈ {3,4}"* and *"12 draws, 8 capable /
4 incapable"* — those cannot both be right. Recomputing directly from this
ADR's own per-rung values (S2b 3, 3, 4, 3; M3 4, 3; M4 4, 4, 5) gives **9
pre-M4c draws of which 8 were incapable of rejecting**, and **11 draws total
after M4c/M4d** — not the "6 of 9" figure **I have repeated throughout this
ADR and in every summary**.

**Flagged, not adjudicated.** If the recomputation is right, the instrument
was blinder than the record says: nearly every pre-M4c draw could not have
rejected at any effect size, which would *strengthen* the
already-recorded instrument finding rather than weaken any conclusion — but
it changes a number that appears in several places, so it must be resolved
against the receipts before any write-up quotes either figure. **Whoever
resolves it should recompute n_disc from the per-item tables in the receipts
themselves, not from any prose in 031, 041 or this ADR** — all three restate
the number, and restated numbers have drifted from receipted ones twice
already tonight (the GPU-hour figure and the permutation count).

## PC1 PRE-REGISTRATION (2026-08-29) — the positive control this harness never had

[docs/research/045](../research/045-positive-control-design.md) identifies the
ladder's largest methodological gap: **every null since S1a is ambiguous
between "no effect exists" and "these mechanics cannot carry signal."**
Power analysis (ADR-036) settled whether the *statistics* could detect an
effect; nothing has established that the *injection pathway* can.

**The literature treats this as standard.** ROME's causal-tracing design
(arXiv:2202.05262) runs clean/corrupted/restored, with restoration as the
positive control validating the pathway *before* partial results are trusted;
ITI (arXiv:2306.03341) runs a random-direction floor **and** a probe ceiling,
and documents the exact gaming failure to guard against (an intervention that
appears to "restore the answer" while merely collapsing the model into a
degenerate response). Zhang & Nanda (arXiv:2404.15255) name unvalidated
negative patching results as an open problem. Not universal — DAS
(arXiv:2303.02536) runs only a floor — but the best work does it.

**We have exactly one instance, and it is stale**: S1a (self-pair, identity
transform, p = 0.03125) ran under **overwrite + pooled + the 40-item sign
test** — mechanics **nothing since M4c uses**. No positive control has ever
run under fuse, de-pooled payloads, or the e-process.

**PC1 spec** (cheapest item on the entire cost ledger, **~7-15 minutes GPU**,
zero training, zero new capture): inject the receiver's **own** gold
teacher-forced per-token block-19 state — already on disk in
`receiver_L19.tok.f32bin`, 719,115 tokens, verified — back into itself via
**identity transform**, under the **current** mechanics (fuse, de-pooled,
`<|fim_pad|>`, L18→L14 geometry), scored with S1a's original 40-item sign
test for direct comparability with the ladder's only pass.

**It runs BEFORE M5X (ADR-037, ~1.3-1.4 GPU-h) and M4b (ADR-035, ~1-2
GPU-h)** — spending hours on rungs whose nulls would be uninterpretable is
the wrong order.

**Registered outcome rule, before the run:**
- **PC1 PASSES** → the current mechanics demonstrably carry signal in the
  maximally favourable same-model case, and every subsequent null becomes
  interpretable as being about *transfer* rather than about *plumbing*.
- **PC1 FAILS** → this is the more consequential branch and must not be
  softened. Every null since S1a (S2b ×4, M3 ×2, M4 ×3, M4c, M4d, M4g,
  M4h S1, M4i) needs an explicit caveat that current mechanics were never
  shown capable of carrying signal even from a model to itself; **S1a's pass
  gains a footnote scoping it to mechanics no longer in use**; and — the part
  that corrects *my own* reasoning — ADR-024's "adapters improved, so
  discordance fell" explanation acquires an **unruled-out rival**: fuse +
  de-pooling + site may suppress discordance **mechanically**, independent of
  adapter quality.
- **Firewall**: a PC1 pass proves **liveness, not transfer**, and may never be
  cited as evidence for the transfer claim. docs/research/045 §3 carries the
  exact disclaimer language to reuse verbatim.

## M4i PRE-REGISTRATION (2026-08-29, before any run) — inject at ORDINARY tokens, not `<|fim_pad|>`

[docs/research/043](../research/043-placeholder-token-choice.md) identifies a
third structural difference from every working method, and it is the **only
one that predicts INERTNESS** rather than degraded or harmful transfer —
which is precisely the signature the correction above isolated.

**What was found** (primary sources): the base Qwen2.5 technical report
(2412.15115) **never mentions FIM**; FIM training is documented only in the
Qwen2.5-**Coder** report (2409.12186) as Coder-specific continued
pretraining. Our receiver is Qwen2.5-1.5B-**Instruct**, non-Coder — so
`<|fim_pad|>` (id 151662, `"special": false` in its own tokenizer_config)
is plausibly **experientially near-vacant for this model**: not a token it
learned to suppress, but one it has essentially **no circuitry for**. We have
been injecting eight copies of a state into a region of the vocabulary the
receiver may never have meaningfully trained on. **No surveyed method
(C2C, LatentMAS, Bicameral) injects at a placeholder token — all three
inject onto real token positions.**

Supporting literature (both abstract-only this pass, graded accordingly):
"Let's Think Dot by Dot" (2404.15758) — models *can* use filler positions for
hidden computation, but only with "specific, dense supervision to converge";
the pause-token work (2310.02226) — gains require the model to be *pretrained
and finetuned with* the tokens, not merely given them at inference. Neither
condition holds here.

**M4i spec** (no training, no capture — reuses M3's already-trained
on-manifold adapter): keep slot count **8** and depth **L14** (slot count is
protected; ADR-028's contradiction on that point remains flagged and
unadjudicated), deliver by **fuse** (so real content is preserved rather than
destroyed), and inject onto **8 already-present ordinary tokens** — the last
8 tokens of the item's question, or the fixed ANSWER_FORMAT instruction
tokens — instead of 8 `<|fim_pad|>` copies. One registered frozen-probe draw,
controls per M4g's frozen fuse definitions, mid-p McNemar primary with
exact-sign and n_disc-versus-power-floor reported.

**Registered interpretation, before the run**: a **PASS** would mean the
ladder's inertness was substantially an artifact of injecting into an
untrained embedding region — and every prior on-manifold null would need that
caveat attached. A **NULL** demotes this to a real-but-non-load-bearing
structural difference and points back to pooling and receiver scale as the
dominant candidates. **Honest limits recorded now**: the FIM-exposure claim is
inferred from *absence of mention*, not from a stated negative, and NLL alone
cannot distinguish "vacant embedding" from "mildly trained but unremarkable".

**Ordering**: runs after M4h Stage 1 reports (that rung is already probing and
also uses the fim_pad slots, so it does **not** test this).

## MAJOR CORRECTION (2026-08-29): the "0/40 NLL inversion" is NOT ladder-wide

Two zero-cost checks against **already-committed receipts** — no new runs —
refute a hypothesis and correct a claim I have repeated throughout this
ladder's write-ups.

**1. The 8-identical-rows hypothesis is REFUTED, for free.** Under
*overwrite*, the zero-vector control writes **8 literal zero rows** over real
content — a maximally unnatural duplicated block — and it is **harmless**:
NLL 2.1537 vs baseline 2.1288 (0.025 nats; 19W/21L, p = 0.68, statistically
indistinguishable). The *aligned* payload in the same rung costs **3.2 nats**.
If duplication or placement were doing the damage, eight identical zeros
would damage too. They do not. **The harm is content-dependent, not
placement-dependent.** (docs/research/042 proposed exactly this check and it
falsifies 042's own hypothesis — recorded as such.)

**2. My claim that the inversion "survived on-manifold AND off-manifold
payloads" is FALSE.** Measured from the committed receipts:

| rung | manifold | aligned NLL | baseline | aligned vs random |
|---|---|---|---|---|
| M3 per-token | on | 2.1328 | 2.1288 | 16W/24L |
| M3 pooled | on | 2.1302 | 2.1288 | 17W/23L |
| M4 r64 | on | 2.1171 | 2.1288 | 16W/24L |
| M4 r128 | on | 2.1295 | 2.1288 | 16W/24L |
| M4 r256 | on | **2.1074** | 2.1288 | 16W/24L |
| M4c | off | 5.3590 | 2.1288 | **0W/40L** |
| M4d | off | 4.9193 | 2.1288 | **0W/40L** |
| M4g | off | 4.8155 | 2.1288 | **0W/40L** |

**The 0/40 unanimous inversion occurs ONLY in the off-manifold task-loss
rungs.** Every on-manifold rung sits within ~0.004 nats of baseline — and
M4 r256 is *better* than baseline. The on-manifold family is not harmful at
all; it is simply **inert**.

**The two null families are therefore cleanly separated, and neither is what
I described:**
- **On-manifold (M3, M4, run-1 affine)**: payload is harmless and inert —
  NLL ≈ baseline, accuracy unmoved. Nothing transfers, nothing breaks.
- **Off-manifold (M4c, M4d, M4g)**: payload is actively destructive —
  NLL 2.3-2.5× baseline, unanimous 0/40 — and this survives both overwrite
  and fuse, so it is the *content* (an off-manifold vector at ~4x natural
  norm: 134-143 vs natural ~34) doing the damage, not the operator or the
  placement.

**Consequences**: (a) M4h Stage 1 uses M3's on-manifold adapter, so it should
**not** invert — its test is purely whether de-pooling adds *benefit*, not
whether it breaks an inversion; framing it as the latter (as the prior wakeup
prompt did) was wrong. (b) The off-manifold rungs' accuracy nulls are
confounded by active damage and are weak evidence about transfer per se.
(c) The honest ladder-wide statement is: **no configuration has produced
benefit; only off-manifold ones produced harm.**

## M4g OUTCOME (2026-08-29) — honest fail; fuse REFUTED as the root cause

`run2-m4g-receipt-cellL18toL14-mlp-fuse-*-n40.json`. Aligned 23/40, baseline
22, zerovec 22, random 22. Primary (aligned > random): **2W/1L, n_disc = 3,
exact-sign p = 0.5000, mid-p 0.3125 — FAIL**. **Power caveat, stated up
front**: at n_disc = 3 the minimum attainable one-sided p is 0.125 > α, so
this draw was **structurally incapable of rejecting** — the accuracy null is
therefore weak evidence, unlike M4d's (n_disc = 7).

**But the finding that matters is not power-limited.** The **0/40 NLL
inversion — the specific anomaly this rung was registered to fix — persists
completely unchanged**: mean NLL 4.815 aligned vs 2.129 baseline / 2.129
zerovec / 2.170 random, **0W/40L against both controls**. A unanimous 40-item
effect is not a small-sample artifact. **Fuse is refuted as the explanation
for the ladder's central anomaly**: the adapter actively harms the receiver
under a residual *add* exactly as it did under overwrite.

**Control semantics were resolved and frozen BEFORE the draw** (recorded in
both training and probe receipts), as required: `aligned_real` and `random`
keep both definition and meaning (random remains a norm-matched Gaussian —
under fuse an information-free perturbation of the receiver's own rows, which
is precisely the comparator the primary needs, so **the primary statistic's
meaning is unaffected**); `zerovec` keeps its definition but changes meaning —
under fuse it is a **true no-op**, verified empirically at 40/40 bit-identical
NLL and zero accuracy disagreements. Disclosed, not silently redefined.

**Engineering discipline worth preserving**: `LayerEdit::Fuse` was added
*beside* `Inject` and all 14 `InjectionSpec` construction sites now declare
their mode explicitly, so every prior receipt remains reproducible; the
shared init was asserted byte-identical against both M4c's and M4d's
committed receipts rather than assumed from a shared seed.

**What this leaves**: the inversion now survives overwrite AND fuse, and all
three task-loss adapters were off-manifold. The two properties that have
never been combined in one adapter remain **on-manifold** and **fuse-
delivered** — which is exactly what M4h Stage 1 tests, at near-zero cost,
from M3's already-trained on-manifold weights. Registered before this verdict
(commit 9007b97); proceeding to it now.

## M4h PRE-REGISTRATION (2026-08-29, before any run) — de-pooling

[docs/research/040](../research/040-the-pooling-gap.md) verified from primary
sources that **no successful cross-model method pools**: C2C transfers
per-token KV cache (full sequence, all layers); LatentMAS *concatenates* full
per-token per-layer caches (its "shared latent working memory" is literal, not
a summary); Bicameral couples live per-token states each decode step and has
no static object to pool; AVP's cross-model path is per-token. **LatentMesh is
the only surveyed design that pools** — and every rung, including run 1's
affine bridges and M4's nominally "sequence" FastGRNN, still pools at the
injection boundary. Mechanism from the literature (Ethayarajh 2019 anisotropy;
Li et al. 2020 BERT-flow, STS-B 59.04 pooled vs 70.72 isotropy-corrected),
reproduced at reasoning-trace scale by our own 036 numbers (cross-item
invariance 0.962 pooled vs 0.635 real; entropy 9.30 vs 3.36). Honest
counter-case recorded: SBERT's ablation has trained MEAN pooling *winning*
(87.44 vs CLS 86.62) — pooling is not intrinsically fatal; **uncorrected
pooling-unaware geometry** is.

**M4h, two stages, registered now:**
- **Stage 1 (near-zero cost, no new training or capture)**: take M3's
  *already-trained* per-token MLP and emit the **last-token** output instead
  of the mean, same 8-slot broadcast. One registered probe draw.
- **Stage 2 (~0.5-0.7 GPU-h)**: **8 distinct per-slot vectors** via
  attention-compression over the sender's own per-token stream — this
  redirects M4f's bank-attention mechanism at the right target. One
  registered probe draw.

**Implementability, source-verified**: `LayerEdit::Inject`/`Fuse` in
`qwen2_b.rs` already accept **distinct per-row vectors**; only
`InjectionSpec`'s broadcast-repeat wrapper forces pooling. No forward-pass
change, no new capture — the 719,115-position per-token dumps exist and have
never been used at full granularity.

**Probe compatibility**: preserved **only if slot count stays exactly 8**
(content per slot changes; slot count does not). M4h is specified that way
deliberately.

**Ordering and relationships**: upstream of M4f's manifold framing — our
"on-manifold" was measured against a *pooled* reference that is itself 0.667
cosine off the real manifold, so a correctly-scoped M4f and M4h Stage 2 may
converge on nearly the same experiment. Orthogonal to M4g (overwrite/fuse)
and M4b (receiver scale). Entangled with M4e (every non-pooling method also
avoids one-shot delivery) — a combined de-pool + continuous rung is named as a
future escalation, not the immediate step. **Runs after M4g reports.**

**Scoping correction to an earlier brief**: MANTA is **not** a latent-transfer
method — its own abstract describes multi-agent topology restructuring
(roles, links, order, validation), not hidden-state content. It should not be
cited alongside C2C/LatentMAS/Bicameral as a latent-transfer comparator; any
earlier text of mine that did so is wrong.

## M4h Stage 1 manifold pre-check — recorded BEFORE its probe verdict (2026-08-29)

`run2-m4h-s1-manifold-precheck-receipt.json` (CPU-only, no probe draw). The
de-pooled payload — M3's byte-identical trained MLP, applied per token, LAST
token taken instead of the mean — is the **first candidate in the entire
15-row kit that is neither collapsed nor pooled**. It classifies
**`on-manifold-item-varying`**, a cell only one other row occupies: the
un-pooled natural receiver state itself.

| candidate | invariance | top-10 union | cos-to-natural | entropy (nats) | class |
|---|---|---|---|---|---|
| M3 per-token (pooled) | 0.9702 | 70/400 | 0.9889 | 9.34 | item-invariant-but-on-manifold |
| **M4h S1 (de-pooled)** | **0.6670** | **133/400** | **0.6814** | **3.32** | **on-manifold-item-varying** |
| reference: receiver L14 pooled | 0.9617 | 78/400 | 1.0000 | 9.30 | item-invariant-but-on-manifold |
| reference: receiver L14 **single row** | 0.6350 | 153/400 | 0.6670 | 3.36 | on-manifold-item-varying |
| M4g (off-manifold, pooled) | 0.8689 | 78/400 | −0.0412 | 5.93 | COLLAPSED-OFF-MANIFOLD |

**Read the last two rows together.** On every registered metric the de-pooled
payload lands on the *un-pooled* receiver state, not on the pooled one:
invariance 0.667 vs 0.635, entropy 3.32 vs 3.36, gold-token percentile 23.0%
vs 22.5%, top-10 union 133 vs 153. Its `cos-to-natural` of 0.6814 is measured
against the **pooled** natural state — and a genuine single receiver state
scores 0.6670 on that same metric, so the de-pooled payload is, if anything,
marginally *closer* to the pooled mean than a real state is. This is the
scoping caveat the M4h registration already flagged, now quantified: against a
pooled reference, "0.68" is what being on the real manifold looks like.

**This is a direct confirmation of [docs/research/040](../research/040-the-pooling-gap.md)
at the representation level.** 036's numbers (invariance 0.962 pooled vs 0.635
real; entropy 9.30 vs 3.36) are reproduced, and removing the mean — changing
nothing else, not one trained weight — moves the emitted object from 0.970 /
9.34 to 0.667 / 3.32. Pooling, not the adapter, was destroying the geometry.

**Interpretation registered before the probe reports:**
- If **M4h Stage 1 PASSES**, pooling-induced geometric damage was the
  operative variable and the fix is nearly free.
- If **M4h Stage 1 NULLS**, then a payload can be geometrically
  indistinguishable from a real receiver state and still transfer nothing —
  which would separate *representational fidelity* from *transfer* and rule
  out the pooling explanation for the ladder's nulls.

## M4h STAGE 1 OUTCOME (2026-08-29) — honest fail; pooling REFUTED as the cause

`run2-m4h-s1-receipt-cellL18toL14-mlp-pertokenlast-fuse-slots8-nopool-rescaletrue-n40.json`.
One registered draw, no retry. Aligned **24/40**, baseline 22, zerovec 22,
random 22 — the aligned condition is the top row, and it lost **zero** items
to any control on accuracy.

**Primary (aligned > random): 2W/0L, n_disc = 2, exact-sign p = 0.2500,
mid-p 0.1250 — FAIL.**

**Power, stated first and plainly.** At n_disc = 2 the minimum attainable
one-sided p is **0.25 > α**, so this draw was **structurally incapable of
rejecting** the accuracy null at any outcome. It is the **weakest draw in the
ladder**: M4d's n_disc = 7 (floor 0.0078) remains the only non-power-limited
rung; M4g's n_disc = 3 (floor 0.125) was already incapable; this is worse
still. A clean 2W/0L sweep that cannot clear α is exactly what a
power-limited design produces, and it must not be read as encouraging.

**The informative result is the NLL, and it is negative.** Against the
on-manifold family — the correct comparator, per the MAJOR CORRECTION above,
*not* M4g:

| rung | manifold | pooled | aligned NLL | Δ vs baseline | vs zerovec | vs random |
|---|---|---|---|---|---|---|
| M3 per-token | on | yes | 2.1328 | +0.004 | 20W/20L | 16W/24L |
| M4 r256 | on | yes | **2.1074** | **−0.021** | 21W/19L | 16W/24L |
| **M4h S1** | **on** | **NO** | 2.1599 | **+0.031** | **10W/30L** | 15W/25L |
| M4g | off | yes | 4.8155 | +2.687 | 0W/40L | 0W/40L |

De-pooling made the NLL **worse**, not better: +0.031 nats is the largest
positive deviation of any on-manifold rung, and 10W/30L against both baseline
and zerovec is the worst win/loss of the family (M3 was an even 20W/20L). The
payload stays firmly inside the on-manifold "inert" band in magnitude — two
orders below the off-manifold rungs' +2.7 nats — but sits on the wrong side
of it. **No benefit was added.**

**What this refutes.** The pre-check and the probe, read together, are a
clean dissociation:

> A payload can be made **geometrically indistinguishable from a real,
> un-pooled receiver state** — invariance, entropy, token support and
> manifold cosine all matching the single-row reference — and still transfer
> **nothing**, while costing slightly *more* NLL than the pooled version of
> the same weights.

**Pooling is therefore refuted as the explanation for the ladder's nulls**,
in the same way and on the same evidential footing that M4g refuted the
injection operator. Research 040's diagnosis was correct about the
*representation* — the pre-check confirms pooling was destroying the geometry
— and wrong about the *consequence*: restoring the geometry buys nothing.
This is the third named root-cause candidate to fall (task loss → M4f;
injection operator → M4g; pooling → M4h S1).

**Scope of the refutation, honestly bounded.** Stage 1 removes the mean but
still delivers **one vector broadcast to 8 identical slots**. What the
literature actually does is deliver **many distinct per-token vectors**
(C2C's full KV cache, LatentMAS's concatenated caches). So this rung refutes
"*pooling-induced geometric damage* explains the nulls"; it does **not**
refute "*one-vector bottleneck* explains the nulls". That distinction is
precisely **M4h Stage 2** (8 distinct per-slot vectors by attention
compression), which is now the better-motivated successor than it was before
this draw: with geometry eliminated as the variable, the remaining
non-pooling property the literature has and we lack is *channel capacity*.

**Two changed factors, disclosed.** Stage 1 differs from M3's committed
per-token receipt in both payload derivation (mean → last token) and
injection operator (overwrite → fuse), as the registration specified. A null
therefore cannot apportion blame between them. The fuse half is not a live
suspect — M4g characterised it and this run re-verified the operator at
**0 accuracy disagreements, 40/40 bit-identical zerovec-vs-baseline NLL,
max |ΔNLL| = 0.0** — but the confound is recorded, not argued away.

**Controls were M4g's, reused rather than restated**: the same shared
`common::m3::four_conditions` code path under the same `InjectionMode::Fuse`,
with M4g's frozen `control_semantics_under_fuse` block read out of M4g's
training receipt and echoed into this one. `zerovec` is again a true no-op,
so the registered `2 × zerovec ≥ baseline` gate is again **degenerate** and
carries no evidential weight. Slot count stayed at **8** throughout, keeping
clear of ADR-028's flagged contradiction.

**Cost**: 395 s of GPU, no training, no capture — the cheapest rung in the
ladder, and it eliminated a root-cause candidate.

**Receipt prose amended after the draw, disclosed.** This rung's probe
receipt was written with an `nll_inversion_status` block whose prose asserted
the 0/40 inversion was ladder-wide — the false premise the MAJOR CORRECTION
section above refutes, inherited from this rung's task brief and written
before that correction landed. The block was renamed to `nll_harm_accounting`
and its prose corrected, with a `post_draw_amendment` block recording exactly
what changed. **No measured value, count, statistic or gate was touched, and
the probe was not re-run.** Recorded rather than adjudicated.

## ADR-028 INTERNAL CONTRADICTION (found 2026-08-29, flagged not adjudicated)

[ADR-028](028-evolutionary-adapter-search-anti-gaming.md) lists **"slot
count"** on **both** sides of its own boundary: as an *evolvable* surface
(alongside pooling scheme, injection depth/site) **and** as part of the
*protected* frozen probe protocol. Those cannot both hold. Recorded here
rather than silently resolved; M4h sidesteps it by keeping exactly 8 slots.
ADR-028's owner should adjudicate before any rung proposes changing slot
count.

## CORRECTION to M4e's premise + bidirectional exchange deprioritised (2026-08-29)

[docs/research/039](../research/039-bidirectional-latent-exchange.md) corrects
a claim this ADR's M4e registration relied on, and settles the bidirectional
axis:

- **M4e's premise was overstated.** The M4e section states that "every
  externally successful cross-model method injects continuously, at every
  generation step". That is **not accurate**: **C2C fuses ONCE at prefill**,
  not continuously (correcting docs/research/032's characterisation, and
  consistent with 038's reading of C2C's fuser). **Only the Bicameral Model**
  (arXiv:2605.11167) couples continuously — and it does so *simultaneously in
  both directions* at every decode step, so it is not a clean instance of
  one-way continuous injection either. M4e remains a legitimate untested
  axis, but the "everyone successful does this" support behind it is reduced
  to a single, structurally different method. Registered correction; M4e's
  priority is unchanged but its justification is weaker than written.
- **Independent corroboration of the scale confound**: Bicameral **degrades
  on GSM8K** (49.6% → ~40%) when the capability gap between the coupled
  models is small — our exact task, and a second independent reason to keep
  **M4b mandatory**.
- **No one has demonstrated turn-based latent dialogue** (receiver replies,
  sender revises) anywhere in the literature. Bicameral's only ablation swaps
  in a no-op auxiliary rather than a unidirectional-continuous control, so
  bidirectionality is never isolated from "continuous" or from "having a real
  second model". LatentMAS is confirmed **strictly unidirectional** despite
  its shared-working-memory framing.
- **The frozen probe cannot score a bidirectional protocol at all** — not as a
  matter of degree: every clause (one fixed injected vector, unassisted
  free-running receiver, one scored output, controls defined relative to one
  vector, no stopping rule) is violated once the sender's state also changes.
  This is the first axis that changes the **protocol** rather than the
  mechanism, so it needs its own pre-registration comparable in scope to
  ADR-030 — not a drop-in rung.
- **Hard ordering blocker**: under the current **overwrite** injection, a
  second round would clobber the first round's content before the receiver
  integrates it. Bidirectional exchange is not even well-posed until M4g
  (fuse) lands.
- **Decision: deprioritised to last, and out of run 2.** It layers on an axis
  (continuous injection) this ladder has not tested even unidirectionally,
  violating one-factor-per-rung twice over. If the ladder still nulls after
  M4g/M4f-rescoped/M4b, spin continuous-then-bidirectional coupling into its
  own ADR — scoped "continuous unidirectional first, bidirectional as a
  contingent follow-on".

## CORRECTION to the M4f pre-check verdict and to the DIAGNOSIS (2026-08-29)

The full pre-check report refines and partly refutes what the section below
records. Corrections, per [ADR-031](031-evidence-receipt-and-statistical-protocol-governance.md)'s
append-only rule — the original text stays, this supersedes it:

**1. "Task-loss-specific" is wrong. Off-manifold is the UNTRAINED DEFAULT.**
The shared zero-step initialisation (byte-identical across M4c and M4d,
asserted not assumed) is already orthogonal to the receiver's block-14
residual stream (cosine −0.021). **Reconstruction training MOVES the output
onto the manifold** (M3 0.989/0.975; M4 r64/r128/r256 0.984/0.983/0.985) —
**as do run 1's training-free affine bridges** (0.982/0.996), all at natural
norms (31.5-34.6 vs reference 34.5) and natural entropy. **Task-loss training
does not**: M4c −0.018 at norm 134.6, M4d +0.048 at norm 143.6 — it grew the
initialisation's norm ~14x and left the direction where it found it. So the
collapse is **not caused by** task loss; it is what task loss **failed to
remove**. Neither universal nor task-loss-specific: 7 of 11 emitters are
on-manifold, 1 intermediate.

**2. Two of docs/research/033 §4's three grounds fail** (measured against the
receiver's OWN pooled L14 state — a reference 033 did not have):
- "77 distinct tokens across 40 items" is **not diagnostic**: the natural
  pooled state gives 78.
- "nearly item-invariant" **inverts**: M4c's 0.881 mean pairwise cosine makes
  it the *least* item-invariant pooled emitter; every on-manifold adapter is
  *more* invariant (0.96-0.98).
- "gold at the 61st percentile, worse than the middle" survives only as a
  *relative* claim: the natural pooled state puts gold at 38.4% and a real
  single state at 22.5%, so LAP's "negligible" band contains the receiver's
  own genuine state.
033's headline (off-manifold; rescale exonerated) **stands, on one
measurement rather than three**: cosine to the receiver's own state for the
same item, where the two families separate by ~a full unit with no overlap.

**3. The unasked finding, potentially the most consequential: POOLING is
itself a large step off the manifold.** A genuine receiver block-14 state and
that same item's pooled state have cosine **0.667**, entropy 9.30 vs 3.36,
cross-item invariance 0.962 vs 0.635. **Every rung — run 1's affine included
— injects a POOLED vector**, so "on-manifold" above means *on the manifold of
pooled states*, which is itself well off the manifold of states the receiver
actually carries. This is independent of M4e (continuous injection) and M4b
(receiver scale), and arguably upstream of both. Testable the same zero-GPU
way.

**4. M4f MUST BE RE-SCOPED before scheduling.** This ADR registered M4f as
"constrain the adapter's output to the receiver's residual-stream manifold".
**The reconstruction objective already achieves that** (0.975-0.989) and did
not rescue transfer — manifold membership is necessary-looking but
demonstrably **not sufficient**. The pooled-vs-real-state gap is the better
target. M4f as sketched is superseded; a re-scoped version must name the
pooling gap, not manifold membership, as its object.

**Disclosed caveat** (from the report, not to be lost): the reconstruction
rungs and affine bridges were *fit to* the receiver's L14 states, so landing
on that manifold is near-tautological. The non-tautological content is that
it holds at deployment on held-out items, that task loss demonstrably does
not get there, and that the untrained default is orthogonal — hence the nulls
do **not** share one mechanism.

## M4f PRE-CHECK VERDICT (2026-08-29): collapse is TASK-LOSS-SPECIFIC, and it was there from init

The registered zero-probe pre-check ran across every ladder artifact
(receipt `run2-manifold-precheck-receipt.json`; scout analysis
[docs/research/038](../research/038-manifold-constrained-adapter-scout.md)).
Verdict, and it is sharper than the question asked:

- **All six reconstruction-trained candidates are ON-manifold** — M3 (both
  variants) and M4 (all three ranks, including the deliberately-bad
  superseded init) sit at cosine-to-natural **0.97-0.99**.
- **Both task-loss candidates are OFF-manifold** — M4c and M4d at cosine
  **-0.02 / 0.05**.
- **The decisive detail**: M4c-init and M4d-init — *the same untrained random
  MLP initialisation, before a single gradient step* — are **already** at
  cosine -0.02. **Task-loss training never learned to leave the manifold; it
  simply never had to return to it.** MSE reconstruction loss structurally
  forces manifold proximity (it regresses toward real receiver states);
  cross-entropy through an *overwrite* channel imposes no such pressure, so a
  random init that starts off-manifold can stay there while still reducing
  the loss.

**Consequences for the ladder**: the nulls now separate into two mechanically
distinct families rather than one. M3/M4 nulls are **on-manifold and still
useless** — manifold location is demonstrably NOT sufficient. M4c/M4d nulls
are **off-manifold and actively harmful**. Any write-up must report these
separately; a single "nothing transferred" narrative would misrepresent both.

**M4f is therefore the fourth cell of a 2x2** (reconstruction/task-loss x
on-/off-manifold), and the pre-registered risk is explicit: it may produce a
genuinely on-manifold, item-varying adapter that **still nulls**, because
manifold location and content-usefulness are independent properties — M3/M4
already prove on-manifold-ness alone buys nothing. M4f's pre-probe gate
(re-running this same pre-check plus a gold-token-rank check) exists to catch
that *before* spending the frozen probe draw, not to guarantee a pass.

**M4f recommended mechanism** (structural, not a penalty): attention over a
frozen bank of ~512 real receiver-L14 states sampled from the existing
`receiver_L14.tok.f32bin` dump — query from sender L18 through the M3-shaped
MLP, softmax weights, output = convex combination of bank rows. On-manifold
**by construction**, item-varying **by construction** — the two symptoms
together. A soft VICReg-style penalty is explicitly rejected: it would add
one more term for gradient descent to trade against, reproducing the same
shortcut. Fallback: residual/delta anchored to the nearest bank state with an
L2-penalised delta. ~0.5-0.7 GPU-h for 10 epochs by analogy to M4c's
receipted 0.446 GPU-h; no new data capture required.

## M4g REGISTERED (2026-08-29): overwrite vs fuse — a separate root-cause candidate

Verified from Cache-to-Cache's own method section (arxiv.org/html/2510.03215):
its fuser computes `C_F = C_n(X) + F_n(...)` — a **residual ADD onto the
receiver's own cache**. LatentMesh's `inject.rs` performs a hard
`slice_assign` **OVERWRITE** of the residual rows (verified in
`qwen2_b.rs:79-87`). These are opposite mechanisms, and overwriting plausibly
gives training a shortcut: make the injected slots *harmless* rather than
useful. Registered as **M4g** rather than folded into M4f, to preserve the
ladder's one-changed-factor-per-rung discipline. If M4f nulls, M4g is the
best-motivated successor. Own pre-registration required before any probe draw.

## CORRECTION to the M4d registration (2026-08-29) — a coordinator error, recorded

The M4d contingency text above asserts that "no training rung has ever had
[the rescale] in its loop — training optimizes a raw vector that deployment
then rescales." **That premise was factually wrong.** The M4d implementer
found and recorded it independently (`adr_premise_correction` field in
`crates/latentmesh-train/src/bin/train_m4d_deploymatch.rs`): M4c's training
loop **already contained the rescale**. The error was mine, in a
pre-registration, and it is left in place above with this correction
attached rather than edited away — per [ADR-031](031-evidence-receipt-and-statistical-protocol-governance.md)'s
append-only rule.

Consequence: M4d was registered to fix something that was not broken, on a
premise that was not true. The diagnostic below independently establishes the
stronger statement — rescaling cannot affect output alignment at all — so no
conclusion drawn anywhere depended on the false premise, but the record must
show the mistake was made and caught by an implementer reading the code
rather than by the coordinator who wrote it.

**Why the diagnostic's answer was analytically forced** (and still worth
measuring): `inject.rs` **overwrites** the residual rows at the 8 placeholder
positions rather than adding to them, so the residual *is* `c·v` with
`c = natural_median/‖v‖ > 0`; `W_U(c·v) = c·(W_U v)` is exactly
order-preserving, making every rank-based statistic invariant by
construction. Measured anyway, which pinned the claim to the probe's own code
path and surfaced the off-manifold finding.

**Two magnitude mechanisms survive and are NOT refuted** (neither is an
output-alignment mechanism, so the diagnostic says nothing about them):
(i) blocks 15-28 *add* branch outputs to the carried `c·v` term while each
block's own RMSNorm is scale-invariant, so the downstream balance shifts even
though the block-14 readout does not; (ii) attention above block 14 reads the
slot rows as keys/values, where absolute magnitude is not normalised away.
Both are testable without a probe draw and are registered here as open.
Measured deployment scale, for the record: `c ∈ [0.259, 0.579]`, mean 0.386 —
the rescale *shrinks* M4c's vector by 1.7-3.9x.

## DIAGNOSIS (2026-08-29): the adapter collapsed to a fixed OFF-MANIFOLD direction

The registered zero-GPU diagnostic ([docs/research/033](../research/033-rescale-output-alignment-diagnostic.md))
answered its own question negatively and then found the real mechanism.

**Rescaling is exonerated.** It is a positive scalar, and the receiver applies
a scale-invariant RMSNorm before the unembedding: top-k token overlap between
raw and rescaled vectors is **1.000** at k=10/50/100 on all 40 items, argmax
identical on every item, gold-token ranks bit-unchanged, logit cosine 1.0 to
2e-13. The norm-mismatch hypothesis M4d was registered to test **cannot be
the mechanism behind M4c's 0/40 reversal.**

**What is**: M4c's adapter emits a nearly **item-invariant, off-manifold**
direction. Projected through the receiver's real readout, the induced
distribution is confident (entropy 5.94 nats vs 11.93 uniform, perplexity
~380) but points at rare embedding outliers: across all 40 items the top-10
token sets draw from only **77 distinct tokens**, dominated by
`DirectoryName` (37/40 items), `rias` (35), ` Svens` (34), ` Lanc` (34).
Gold-answer tokens sit at mean rank 93,460/151,936 (61st percentile — worse
than the middle); sender-span tokens at 81,154 (53rd). Both are deep inside
LAP's "negligible effect regardless of intervention strength" regime
(peak A_lin < 0.05). The adapter did not learn the sender's tokens instead of
the gold answer — **it learned neither**, and collapsed to substantially the
same direction for every item.

This explains the previously puzzling shape of the result: losing 0/40 to
*both* the zero vector and a norm-matched random control is the ablation
literature's signature of an **actively counterproductive** intervention, not
an ignored one — and "a fixed off-manifold direction normalised to look
natural and injected at block 14 on every item" is precisely such a
mechanism.

**Interpretive consequences, registered before M4d reports:**
- An **M4d null is now expected and weakly informative** — it changes the
  norm handling, and norm is exonerated. It remains worth completing (already
  running, ~0 marginal cost) but must NOT be read as evidence about
  configuration matching in general.
- **An M4d pass would be surprising** and would have to be attributed to slot
  placement or the training path, not to the rescale.
- The ladder's live hypotheses are now: **off-manifold collapse** (new, best
  supported), **one-shot vs continuous injection** (M4e), and **receiver
  scale** (M4b). The loss-function axis (M3/M4 vs M4c) is no longer the
  leading explanation, because M4c's task-loss training *did* optimise its
  objective and still produced a degenerate, item-invariant solution — which
  is itself evidence that the injection channel admits a degenerate shortcut.

**M4f registered (new, unscheduled, own pre-registration required before any
probe draw)**: constrain the adapter's output to the receiver's residual-stream
manifold — candidate mechanisms include initialising/anchoring to real
receiver states, penalising distance to the natural state distribution during
training, or predicting a convex combination of observed receiver states
rather than an unconstrained vector. Cheap pre-check available with no probe
draw: re-run the same unembedding projection on M4d's artifact and on the
M3/M4 artifacts to establish whether off-manifold collapse is universal
across the ladder or specific to task-loss training.

## Registered hypothesis — M4e, continuous per-step injection (added 2026-08-29, BEFORE M4d's verdict)

Injection-configuration research ([docs/research/032](../research/032-injection-configuration-science.md))
surfaced a configuration difference no earlier rung, and not M4d either,
controls for: **every externally successful cross-model method injects
continuously, at every generation step, "in lockstep"** (Cache-to-Cache; the
Bicameral Model). LatentMesh injects **one-shot into 8 placeholder slots and
then free-runs up to 400 tokens unassisted**. No positive cross-model result
was found anywhere in the literature using one-shot-then-free-run. This is
independent of both the loss-function axis (M3/M4 vs M4c) and the
norm/rescale axis (M4d), and it is therefore a THIRD live explanation for
the ladder's nulls.

The same research also names the specific mechanism M4d tests: M4c's adapter
was gradient-trained to a raw output norm of roughly 90-200 and then
deployed through a rescale to the natural-median target of roughly 40-56 —
a 2-4x change the training loss never saw. Precedent check: **no source was
found that trains a raw activation vector and then applies a post-hoc
rescale to a statistic the loss never saw**; CAA, RepE and ActAdd all fix
norm *before* training or extraction and never touch it after. Norm and
direction are causally distinct levers (arXiv:2606.06735), and post-hoc
renormalising an additive vector converts it into a different, separately
named method (Spherical Steering) — so our deployment path was an
unvalidated compound, not a calibration detail.

**M4e (registered now, unscheduled)**: continuous per-step injection —
the translated payload is applied at each generation step rather than once
upfront. Requires new engineering and its OWN pre-registration addendum
before any probe draw; it is named here so that a later decision to run it
is on record as pre-planned rather than post-hoc. Ordering: after M4d and
M4b report, and only if the cheaper diagnostics (below) do not already
explain the ladder.

**Registered zero-GPU diagnostic (protocol-safe, no probe draw)**: project
M4c's raw and rescaled vectors through the receiver's unembedding matrix and
compare output-token alignment against sender-span and gold-answer tokens
(one matmul over committed artifacts; the LAP A_lin-style check the research
ranks first by expected value per cost). It resolves *today*, off existing
receipts, whether the rescale destroys output alignment. Annotates; never
changes a recorded outcome.

## Registered contingency — M4d, train/deploy configuration match (added 2026-08-29, BEFORE any M4d run)

M4c's receipt shows a pattern no earlier rung produced: task-loss training
worked *as training* (train CE 0.193→0.133; pre-probe transfer check: mean
fused NLL 0.2385→0.1570, a 34% reduction on held-out data — the adapter
demonstrably improves the receiver's own next-token loss), yet at probe time
the aligned condition loses on NLL to random **0W/40L** and to the zero
vector **0W/40L** (p=1.0 both). A systematic all-40-item reversal of the
exact quantity training improved is not noise and is not explained by "task
loss is insufficient"; it is the signature of a **train/deploy configuration
mismatch**.

Named candidate: the probe applies `rescale_to_natural_median` to the
injected vector, and no training rung has ever had that operator in its
loop — training optimizes a raw vector that deployment then rescales.
Secondary candidates: the 8-slot placement and the greedy-decode context
differing from the teacher-forced training context.

**M4d (registered now, before any run)**: repeat M4c's task-loss training
with the probe's exact deployment transform inside the training loop (slot
placement + rescale applied to the adapter output before the receiver
forward), so the trained object is the object actually deployed. Single
seeded run, training receipt frozen before the probe, ONE registered probe
draw under the unchanged frozen protocol. **This changes training only — the
probe protocol, controls, items, and statistics are untouched** (see
ADR-028's protected list; the deployment transform is an *evolvable* surface,
the probe is not). Honest-fail path unchanged. Interpretation rule, also
registered now: an M4d pass would mean prior nulls were configuration
artifacts rather than evidence about latent transfer, and every earlier rung
would need that caveat attached; an M4d null strengthens the joint negative
across loss functions AND configuration matching.

## Registered contingency — task-loss training rung (added 2026-08-28, M4 verdict NOT yet known)

SOTA sweep [docs/research/028](../research/028-sota-continuous-sweep-1.md)
establishes that every external method reporting real cross-model transfer
gains trains its projector through the receiver's TASK loss (C2C: "standard
next-token prediction loss on the Receiver's response predictions",
per-token, all layers), while run 1 and M3 — the field's only
reconstruction-trained, single-layer, pooled data points — both nulled.
Independent theory backs the distinction: arXiv:2605.05029 proves
reconstruction-risk minimizers can be structurally misaligned with the
causally-relevant subspace (capacity does not fix it), and arXiv:2605.23315
names the phenomenon ("epiphenomenal correctness") for exactly our
A6-pass/A7(b)-fail pattern. Registered NOW, before M4's probe result is
known: **if M4 nulls, a task-loss ablation rung (M4c) runs before any
conclusion that the adapter ladder is exhausted** — same or best-so-far
architecture, trained through the receiver's next-token loss on the
generated spans (C2C-style), single seeded run, the same frozen probe, its
own receipt. If M4 passes, M4c is optional comparative work. **Interpretive
pre-commitment (also registered pre-verdict)**: M4 trains on the same
reconstruction loss as M3, so an M4 null must NOT be read as evidence
against sequence structure per se — the loss-function confound applies to
M4 identically, and the parsimonious reading of a joint M3+M4 null is
"still the loss function" (the sweep notes both externally successful
methods are pointwise-per-token, mild evidence sequence structure is not
the lever either way). Either way the
M3/run-1 nulls now carry a candidate mechanistic explanation and a citable
name, scoped alongside (not replacing) the receiver-scale confound above —
the two confounds are independent and M4b remains mandatory.
**Registered analysis (protocol-safe, no probe draw)**: an A6
permutation-null baseline — compute what an uncorrelated cross-model
mapping scores on the same held-out residual metric (row-shuffled pairs,
same fit machinery) — because the PRH-critique literature
(arXiv:2602.14486) shows representation-similarity metrics can be inflated
by depth/width artifacts, and our A6 "pass" thresholds were never
calibrated against chance. CPU-only, runs on the existing dumps; results
annotate (never retroactively change) the recorded A6 outcomes.

## Registered confound — receiver-scale threshold (added 2026-08-28, while M3 was in flight)

arXiv:2608.05164 ("Cross-Architecture Steering Transfer", surveyed in
[docs/research/027](../research/027-global-ambient-intelligence-track.md))
reports cross-model latent-steering transfer works reliably only above a
~1.7B receiver-parameter threshold. This run's receiver (Qwen2.5-1.5B) sits
just below it. Consequences, registered before any M3 probe result was
known: (1) every run-1 and run-2 null result is scoped to "at a sub-threshold
receiver" until controlled; (2) a **mandatory scale-control arm** follows
M3/M4 regardless of outcome — repeat the best (or least-bad) adapter with a
receiver ≥1.7B (Qwen2.5-3B as receiver fits the 16 GB card alongside the
sender; a fresh calibration/capture and its own pre-registration addendum
are required, since receiver layers/dims change); (3) M3/M4 probe outcomes
must not be interpreted as verdicts on linear-vs-trained translation until
that arm reports. The frozen probe protocol itself is unchanged.

## Future work / out of scope — registrations, not commitments

Three directions are named here because they bear on run 2's eventual shape, but **none of them
expand run 2's current scope** — nothing below is scheduled, gated, or built in this wave. Each
now has its own full design-contract ADR (025, 026, 027) rather than staying only a bullet here,
per a follow-on coordinator request; this section stays as the pointer and the one-line framing
for why each was worth naming from inside run 2's own context.

1. **Distributed latent-data fabric.** `ruvector-replication` (crates.io 0.1.1, Rust 1.77+ —
   notably MSRV-*compatible* with this workspace, unlike candle) is the natural replication layer
   if ADR-016's embedded RuVector backend or run 2's own per-token shards ever need to distribute
   across more than one host. Formalized in [ADR-025](025-distributed-latent-data-fabric.md).
2. **Verified-edge federation wire contract.** `agentdb`'s QUIC sync architecture already defines
   a near-exact wire shape (`CausalEdgeSync`: edge id, uplift, confidence, vector-clock version)
   for federating gate-verified causal edges between nodes — reference prior art for a future
   federation ADR, not a run-2 dependency. Formalized in
   [ADR-026](026-verified-edge-federation-wire-contract.md).
3. **M4.5 — latent-prefix delivery, registered by name only.** If M3/M4's mid-layer injection
   gates fail, prefix-level delivery into the receiver's context window is the pre-named next
   contingency: `latentmesh-gate`'s existing `LatentPrefix` authority tier is already designed for
   exactly this lower-trust mode, and prefix conditioning is a strictly easier transfer problem
   than the mid-layer geometry run 1 falsified (usefulness-as-input, not valid mid-layer
   activation structure). **This is not part of run 2's current ladder** — M3 through M6 above are
   unchanged by this registration — and activating it requires its own pre-registration addendum,
   frozen before any probe runs, exactly as ADR-023 required for run 1. Formalized in
   [ADR-027](027-latent-prefix-context-window-delivery.md).

## Consequences

The data pipeline decision (per-token capture from existing streams, no regeneration) makes M2 a
minutes-scale, not hours-scale, milestone — a direct dividend of run 1's own discipline of
preserving every intermediate artifact (the token streams would have been silently lost to `cargo
clean` without the explicit preservation commit this session found already in place). The leakage
rule is the one genuinely new methodological commitment this ADR makes beyond restating ADR-023:
a trained adapter's evaluation integrity depends on train/probe disjointness in a way a closed-form
fit's did not, and this ADR states the rule before any training happens, not after a suspiciously
good M3 result would have made it tempting to skip. The `latentmesh-train`/`latentmesh-runtime`
path-dependency decision keeps M6's live-injection binary simple (one process, both concerns)
without reopening the MSRV conflict that motivated excluding either crate from the root workspace
in the first place — verified, not assumed, against both crates' actual `Cargo.lock` files this
session.

## What is frozen / what is pending

| Item | Status |
|---|---|
| Frozen probe protocol (unchanged from ADR-023), leakage-exclusion rule, per-token capture format, M3/M4 architectures and hyperparameters, `latentmesh-train`'s crate layout, the codex boundary | **Frozen by this ADR** |
| M5's exact training-loop mechanics and gate | **Not fully specified** — needs its own scouting pass; flagged explicitly above, not silently assumed |
| `crates/latentmesh-train` itself | **Not implemented** |
| `capture.rs`'s per-token extension | **Not implemented** |
| M2 through M6 | **Not started** |

## Implementation status

Not implemented beyond M0 (the two bootstrap scouts, cited throughout). This ADR is the plan
milestone M1; M2 (data pipeline) is the natural next concrete step, since it is cheap, well-scoped,
and every downstream milestone depends on its output.

## M4i OUTCOME (2026-08-29) — HONEST FAIL on the registered primary; but the site change is NOT inert, and the instrument is no longer blind

Receipts `run2-m4i-receipt-cellL18toL14-mlp-pertokenlast-fuse-questiontail-slots8-eprocess.json`
and `run2-m4i-manifold-precheck-receipt.json`. **First rung evaluated under
[ADR-036](036-successor-rung-evaluation-protocol.md)'s successor protocol.** Every number below is
an e-process/`adaptation-512` number and is **not comparable** to any frozen-40-item-protocol
result; no p-value translation is offered for the primary, per ADR-036 Decision 3.

**The single changed factor, and its comparator.** Injection **SITE only**, measured against
**M4h Stage 1** (`run2-m4h-s1-receipt-...-pertokenlast-fuse-slots8-nopool-rescaletrue-n40.json`,
asserted at run time to carry the same artifact hash, the same `pertokenlast` derivation, the same
`fuse` operator and the placeholder site). Held fixed: M3's artifact byte-for-byte
(`a864e518…d73c91`, hash-gated against M3's training receipt, 8 golden pairs, max relative L2 error
4.53e-7), the de-pooled `apply_last_row` payload derivation, `InjectionMode::Fuse`, 8 slots, L14,
rescale-to-natural-median, greedy/400, and M4g's frozen control semantics. Changed: 8 `<|fim_pad|>`
copies → **the last 8 tokens of the item's own question**, with the slot sentence and its bracket
removed entirely (`docs/research/043` §4 — keeping the bracket would reintroduce a textual
placeholder cue). Site chosen over 043's ANSWER_FORMAT variant because those tokens are
item-invariant boilerplate; fusing an item-varying payload onto positions carrying no item
information would partially reproduce the very confound this rung removes. Recorded before the draw.

**Positions were resolved from the tokeniser's own offset map, after the obvious method failed.**
Re-encoding `header + question` and requiring it to be a prefix of the canonical tokenisation
**rejected every item**: Qwen2.5's pre-tokeniser groups trailing punctuation with the newlines that
follow it, so a GSM8K question ending in `?` produces a final token spanning `"?\n\n"` that straddles
the question/answer-format boundary. The 8 positions are therefore the last 8 tokens lying *wholly
inside* `item.question`; gated per item as contiguous, ending at the last such token, and decoding
back to question text. 300/300 items resolved, 0 tokenisation exclusions. Every per-item row carries
`injection_site.positions`, `.position_token_ids` and `.positions_decoded` (e.g. item 4 →
positions 32-39 → `" How many pages does he write a year"`).

**PRIMARY (ADR-036 e-process): FAIL.** `λ=0.30`, PASS at `W ≥ 1/α = 20`, `N_max=300`, items drawn
from `adaptation-512` in fixed index order with ADR-024's 13-item leakage exclusion applied (**0 of
the 13 are present in `adaptation-512`** — computed, not assumed; that split is also fully disjoint
from the S2c training pool, so the exclusion is a measured no-op here). The stream ran the **full
300 items**; `W` **never crossed**, and never even rose above its starting value: `W_max = 1.0000`,
`W_min = 0.1474`, `W_final = 0.2578`. The first discordant pair (item order 3) was a loss and the
process was under 1.0 from then on. **`n_disc = 66` (31W/35L).**

**THE INSTRUMENT FIX WORKED.** 66 discordant pairs, against the frozen protocol's M4d 7 / M4g 3 /
M4h-S1 **2**. This draw *could* have rejected; it did not. That is a materially different epistemic
object from M4h Stage 1's `power_limited: true`, and it is the first rung in this ladder where a
null on the primary means "no effect was detected by an instrument capable of detecting one" rather
than "the instrument could not have spoken." Cost: **3,397 s ≈ 0.94 GPU-h**, against ADR-036's
pre-registered ≈0.88 GPU-h full-budget estimate — inside the stated band, no training, no capture.

**But the site change is NOT inert, and that is the rung's real finding.** Every prior on-manifold
configuration sat within ~0.004 nats of baseline. Here, at ordinary question tokens:

| | aligned | baseline | zerovec | random |
|---|---|---|---|---|
| accuracy (of 300) | **128** | 140 | 140 | 132 |
| mean gold NLL | **2.2446** | 2.3928 | 2.3928 | 2.3474 |

Aligned NLL is **0.148 nats below baseline** and wins **181/300** items against it — two orders of
magnitude outside the inertness band, on the same artifact and the same payload derivation that were
inert at the placeholder site. `docs/research/043`'s mechanistic claim therefore gets **partial
support**: moving off `<|fim_pad|>` makes the payload *register* in the forward pass.

**It registers, and it does not help — a dissociation that must be stated as such.** Accuracy moves
the *wrong* way: aligned 128 vs baseline 140 (24W/36L), and aligned loses the e-process to random
(31W/35L). Two honest qualifications on the NLL number, both from the same receipt: (i) `random`
*also* beats baseline on NLL (2.3474 vs 2.3928), so part of the gain is a generic consequence of
fusing *any* norm-matched vector onto real content positions, not payload-specific — aligned's
margin over random is the smaller 0.103 nats at 166W/134L; and (ii) accuracy at this site is mildly
*damaged* by the operation itself (random 132 and aligned 128, both below baseline's 140), while
`zerovec` is bit-identical to baseline on **300/300** items (max |ΔNLL| exactly 0.0), confirming the
fuse operator is a true no-op and that the damage is content-driven, not operator-driven. NLL was
explicitly rejected as a co-primary by ADR-030 §3.2 because it was blind to the one real accuracy
effect this ladder produced; M4i is the exact mirror — NLL showing a large effect where accuracy
shows none. Both directions of that dissociation are now on record, and neither licenses reading
the other's number as transfer.

**Manifold pre-check (diagnostic, ordering gate only, run before the draw).** Payload derivation is
byte-identical to M4h Stage 1's, so the pre-check row is the same object re-measured into this
rung's own receipt: **`on-manifold-item-varying`**, cosine-to-natural 0.6814 against the un-pooled
reference row's own 0.6670, entropy 3.32 against the reference's 3.36. The payload is a well-formed,
item-varying, on-manifold receiver-like state. It still does not transfer.

**Registered interpretation, applied as written.** ADR-024's own M4i pre-registration said a NULL
"demotes this to a real-but-non-load-bearing structural difference and points back to pooling and
receiver scale as the dominant candidates," and `docs/research/043` §5(4) said the same. That is the
verdict — with one sharpening this rung earns: **pooling is already excluded for this payload** (it
is M4h Stage 1's de-pooled derivation), so the live candidates are now **receiver scale** (M4b),
**one-shot vs. continuous delivery**, and **the single-vector-broadcast-to-8-positions structure**
(M4h Stage 2's 8 distinct per-slot vectors). The placeholder site is real — it demonstrably
suppressed the payload's effect on the forward pass — but removing it is not sufficient.

**Scope of the null, stated narrowly (ADR-032 criterion 1).** M3's reconstruction-trained MLP at
L18→L14, de-pooled to its last translated token, broadcast to 8 positions and residual-added onto
the last 8 ordinary tokens of the receiver's own question, rescaled to the per-item natural median,
does **not** carry causally usable signal for Qwen2.5-3B-Instruct → Qwen2.5-1.5B-**Instruct** on
GSM8K accuracy, over 300 `adaptation-512` items under ADR-036's e-process. Every noun is
load-bearing. It says nothing about a different receiver scale, a per-slot payload, continuous
delivery, or a different task.

**Declared confounds, from the pre-registration, not invented now.** (1) Removing the slot sentence
changes the baseline prompt, so **accuracy levels are not comparable to any prior rung** — the
primary is paired within this rung, where all four conditions share one prompt, and is unaffected.
(2) `random` is a **stronger** comparator here than at the placeholder site because it perturbs
genuine question content; that was declared before the draw and the primary's loss is therefore more
informative, not less. (3) The FIM-exposure claim underpinning the hypothesis remains inferred from
*absence of mention* in the base Qwen2.5 report, not from a stated negative.

**Honest-fail path honoured.** ONE draw, no retry, no restart, no re-parametrisation. The complete
300-step `W_t` trajectory is committed in the receipt regardless of outcome, and `adaptation-512`'s
items 4…4465 were consumed in the file's own fixed index order. `eval-200`/`holdout-100` remain
mechanically locked and untouched. `adaptation-512` shares exactly **one** index (1153) with the
historical frozen-40 probe set — disclosed rather than silently kept; it is not training leakage,
and ADR-036's registered exclusion rule names only the 13-item list, so removing it would itself
have been an unregistered protocol change.

**One unreconciled wording discrepancy inside the committed receipt, disclosed rather than rounded
away (ADR-032 criterion 4).** The receipt's
`gates.site_change_recorded_with_exact_token_positions.note` says the 8 positions were "gated to
decode to a **suffix** of the item's own question." The operative rule — correctly stated in the
same receipt's `site_choice_justification.position_resolution` and enforced in code — is
**containment**: contiguous, ending at the last token *wholly inside* the question, decoding back to
text contained in it. The two differ only because the straddling `"?\n\n"` boundary token is
excluded, so the window is a suffix of the question's *token-aligned* span rather than of its raw
text. The gate note in the probe source is corrected for future runs; the committed receipt is left
byte-for-byte as drawn, and this note is the reconciliation.
