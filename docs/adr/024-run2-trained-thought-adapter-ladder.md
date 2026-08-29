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
