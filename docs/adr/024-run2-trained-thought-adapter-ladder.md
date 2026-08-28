# 024. Run 2: trained thought-adapter ladder

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
