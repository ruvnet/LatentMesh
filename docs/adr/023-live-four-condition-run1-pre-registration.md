# 023. Pre-registration for the live latent-exchange experiment (run 1)

- **Status**: **Concluded — negative result, pre-registration honored.** Run 1 is finished. The
  pre-committed Deviation-7 stop rule fired: the A7(b) mechanics-with-alignment gate failed at
  both registered cells under both calibration distributions (§ S6 — Results, below), so S3-S5
  were never built and A1-A8's compute/quality/edge-survival criteria were never measured. The
  frozen registration above was honored unmodified through the point of failure — no threshold,
  seed, or statistical test was changed after any outcome was observed. See § S6 — Results for
  the full account, receipt-cited, and `docs/research/025-run1-negative-result.md` for the
  narrative writeup.
- **Date**: 2026-08-28 (pre-registration authored and frozen); results appended 2026-08-28.
- **Related**: [009](009-online-causal-control-loop.md) (the combined acceptance test this run
  partially answers), [003](003-causal-edge-verification.md) (the five-control test this run
  exercises live), [014](014-benchmark-and-acceptance-method.md) (evidence-label discipline this
  ADR inherits), [018](018-metaharness-darwin-topology-loop.md) (the receipt/harness pattern this
  run follows), [002](002-latent-packet-protocol.md) (the affine mean-centering amendment §5
  below depends on), [006](006-self-evolving-topology.md)/[008](008-capability-governed-execution.md)
  (topology mutation and authority ceiling this run exercises)
- **Evidence base**: [docs/research/024-live-latent-experiment-design.md](../research/024-live-latent-experiment-design.md)
  (the design this ADR pre-registers), [docs/research/023-beyond-sota-roadmap.md](../research/023-beyond-sota-roadmap.md)
  (the motivating gap analysis), `crates/latentmesh-runtime/receipts/{s0-receipt.json,
  s1a-receipt-slots8-block19-poolfull-rescaletrue-n40.json, run-ledger.json}` (S0/S1a evidence,
  read in full and cited by exact value below, not summarized)

## Context

ADR-009 §5 named one combined acceptance test and one vertical experiment and left both unrun.
`docs/research/023` independently confirmed the gap is real and specific: LatentMesh has run
**zero** live experiments combining a continuous causal-verification gate with real heterogeneous
models, while at least three external groups now ship exactly the raw-latent-transfer half of
this claim on real models. `docs/research/024` is the resulting implementation-ready design — a
judged synthesis (mvp-first winner, three grafts, one scope correction) — and its own §7 states
the rule this ADR exists to satisfy: **statistics, thresholds, deviations ledger, injection
semantics, and the norm-switch setting must be frozen before any eval or holdout item is
consumed.** S0 (runtime mechanics) and S1a (self-pair injection probe) have already run against
this design, live, on this host, with receipts committed; S1b's crate patches (align affine
mean-centering, gate configurable thresholds) are already in `crates/latentmesh-align` and
`crates/latentmesh-gate`. S2 (calibration) is running concurrently with this ADR's authoring —
its one load-bearing artifact, the fitted transform's `content_hash`, is not yet known and is
registered below as an explicit pending field. Nothing else in this registration depends on that
value; it exists here for provenance, not as an input to any frozen threshold.

## Decision — the frozen registration

### Stage status this ADR builds on (recap, not reopened by this ADR)

| Stage | Gate | Result | Evidence |
|---|---|---|---|
| S0 — runtime mechanics | build green under nvcc 12.8; capture shape 2048; capture logits bit-identical; injected logits finite and different from baseline; zero-slot ≈ baseline; injected norm within 3× natural band | **PASS**, all gates green on 3 GSM8K test items | `receipts/s0-receipt.json`, commit `9a676a5d53f66425e40e3ca6f8c3bf6fb1aa9379` |
| S1a — self-pair injection probe | real distinguishable from random, p<0.05, one-sided exact sign test | **PASS on the second run** — see Deviation 3 below for the honest run-1 failure | `receipts/s1a-receipt-slots8-block19-poolfull-rescaletrue-n40.json`, `receipts/run-ledger.json` |
| S1b — crate patches | `cargo test --workspace` green; align affine mean-centering + cached matrix + hash-once; gate configurable dV thresholds | **Landed** — verified in this session against `crates/latentmesh-align/src/lib.rs` (`mu_s`/`mu_r` fields, `content_hash`) and `crates/latentmesh-gate/src/lib.rs` (`CeilingThresholds { action_influencing_dv: 0.15, latent_prefix_dv: 0.05 }`) | current tree |
| S2 — calibration | held-out relative residual < 0.9 (A6) | **PASS** — all 9 sweep cells pass; winner **L18→L14** (50%/50% relative depth), held-out relative residual **0.5106**; transform `content_hash` **eb3f42edde853824642a2b811577e2c767f73c2c179fe03a05ac8dac23704457** (artifact `receipts/transform-L18-to-L14.json`, file sha256 == content_hash); registered via `Policy::trust_transform` | `receipts/s2-calibration-receipt.json`, `receipts/s2-dump-receipt.json`, `receipts/s2-splits-receipt.json` |
| S2b — bridge probe (A7 resolution) | aligned-real > random, p<0.05, at winner cell, else registered fallback | **FAILED AT BOTH CELLS — kill-path signal** (winner L18→L14 p=0.5000; anchor L24→L19 p=0.8750; exact S1a protocol, knobs untouched). A7(c) true zero-vector gate **PASS** (through the real 8-slot path, at/above baseline both cells). Attribution clean: S1a identity-transform passed and the affine apply verified bit-exact against golden pairs, so injection mechanics transmit but the gold-teacher-forced affine alignment carries no usable signal. Adversarially verified: fresh GPU re-run bit-identical, all statistics independently recomputed. | `receipts/s2b-receipt-cellL18toL14-*.json`, `receipts/s2b-receipt-cellL24toL19-*.json`, `receipts/transform-L24-to-L19.json`, golden-pair files |
| S3-S6 | per design §7 | **GATED on the Deviation-7 contingency below** | — |

### S2-completion coordinator rulings (2026-08-28, before any eval item consumed)

- **Deviation 5 — split-seed collision, resolved toward implementation.** This ADR
  (authored concurrently with S2) registered train/test shuffle seeds derived from
  `SHA-256("latentmesh-live-run1-<label>")` (`0x970340d0faa39992` / `0xa6109ba8198fbb17`).
  The S2 implementation, running in parallel, generated the actual committed splits with
  its own pre-run seeds (`0x24C0DE01` train shuffle, `0x24C0DE02` test shuffle,
  `0x24C0DE03` 80/20 fit split; `data/splits-receipt.json`). Both seed sets were fixed
  before any outcome-relevant item was consumed, and eval/holdout access is mechanically
  refused until a genome-frozen receipt exists (unit-tested). Ruling: the **as-implemented
  seeds are authoritative**; the SHA-derived train/test-shuffle values are void. The
  `darwin-genome` and `audit-run-seed` registrations above remain authoritative (unused yet).
- **Deviation 6 — winning calibration cell is L18→L14, not the design §2 anchor L24→L19.**
  Selection followed the registered rule (minimum held-out residual, §5.3): L18→L14 = 0.5106
  vs anchor L24→L19 = 0.5600 (anchor also passes A6). S1a's injection-mechanics evidence is
  at receiver L19; no mechanics probe has run at L14. Ruling: the mandatory S2→S3 bridge
  probe (A7 resolution above) runs at the winner (aligned 3B-L18 → transform → 1.5B-L14,
  plus true zero-vector injection through the real path). **Registered fallback**: if the
  bridge probe fails at L14, run 1 carries the anchor cell L24→L19 instead — its A6 pass
  and L19 mechanics evidence are already on record — and the fallback use is disclosed in
  the results section. Either way the choice closes before S3 opens.
- **Correction to the earlier workspace ruling**: `harness/latentmesh-live` was built as a
  root-workspace **member** after all — it crosses to `latentmesh-runtime` exclusively via
  files and subprocess invocation (never a cargo dependency), so the half/MSRV conflict the
  earlier ledger entry feared does not arise. The exclusion requirement stands only for
  crates that link candle.

- **Deviation 7 — design-named generated-pairs contingency fired (2026-08-28, post-S2b).**
  Design 024 §5.4/§8 risk 6 pre-names gold-vs-generated calibration distribution shift as
  the prime suspect for exactly this failure shape and pre-costs a ~1 GPU-h contingency:
  recalibrate on states captured from each model's OWN generated reasoning (same
  calibration-4000 items, same sweep/protocol machinery, same A6 gate) and re-run the S2b
  bridge probe once against the new transform. This is a disclosed pre-registered
  contingency, not protocol iteration: the S1a/S2b statistical protocol stays frozen.
  **Pre-committed outcome rule**: if the generated-pairs aligned-real gate also fails at
  both cells, run 1 STOPS — S3-S5 are not built — and the negative result is written up
  with full receipts per design §7 S4's kill-switch and §10's honesty contract.

### Four conditions (frozen, as design §4)

- **StaticText** — fixed hand-designed 2-agent pipeline, text channel, no mutation, no gate. Cost/quality anchor.
- **DynamicText** — same channel, genome `{sender prompt variant ×3, message token budget, edge on/off}` mutated by an in-harness (1+3) hill-climb, 6 generations, fitness on adaptation-pool batches only.
- **DynamicLatent** — same genome/loop; channel = `LatentFrame` (injection semantics below); receiver prompt carries the question only, the latent is the sole inter-model channel.
- **CausalDynamicLatent** — DynamicLatent + the ADR-003 five-control causal loop as fitness and governance; `set_ceiling_from_verdict` live each generation; one full six-way audit of the frozen final genome on eval-200 is the sole source of the reported OPE/OME/CAG/SSG decomposition and survival CIs.

All four run on identical candle BF16 weights, identical sampler code, both models CUDA-resident
concurrently, batch=1 — channel and topology-mutation are the only varied factors.

### Injection semantics (frozen)

| Parameter | Frozen value | Source |
|---|---|---|
| Sender / capture point | `Qwen/Qwen2.5-3B-Instruct`, after block 24 of 36, mean-pooled over the generated-reasoning token span | `s0-receipt.json config.sender_capture_block`; design §2 |
| Receiver / injection point | `Qwen/Qwen2.5-1.5B-Instruct`, block 19 of 28, placeholder-slot overwrite | `s0-receipt.json config.receiver_inject_block` |
| Slot count | 8 | `s0-receipt.json config.n_slots`, `s1a` `config.slots` |
| Placeholder token | `<|fim_pad|>`, id 151662 | `s0-receipt.json config.placeholder_token/placeholder_id` |
| Norm band factor (G6 gate) | 3.0 | `s0-receipt.json config.norm_band_factor` |
| Norm-rescale switch | **ON** — injected vector rescaled to the natural layer-19 **median** L2 norm at the target position, per item | `s0-receipt.json config.injection_vector`; `s1a` `config.rescale_to_natural_median: true`; every S0/S1a receipt's `G6_injected_norm_within_band` compares against `natural_median`, never a mean |
| LatentFrame construction | `Payload::encode` F16; `transform_hash` from the S2-fitted mean-centered 2048→1536 transform; `confidence` from held-out fit residual; `provenance.context_hash = sha256(prompt)`; `authority = ContextInject`; passes `Gate::admit` before injection | design §4 DynamicLatent |

**Registered discrepancy (flagged, not silently resolved):** design doc §4 describes the
rescale target in prose as "the natural layer-19 **mean** norm." The implemented config field is
`rescale_to_natural_median`, and every S0/S1a receipt's gate compares the injected vector against
`natural_median`, computed as the literal median of the natural-norm distribution (`s0-receipt.json`
also reports `mean`/`median`/`p25`/`p75` separately per item — they are materially different
numbers, e.g. item 0: mean 136.97 vs. median 63.47). **This ADR freezes the implemented
behavior — median — as the registered norm-switch setting**, since S0/S1a's empirical evidence is
what actually ran; the design doc's prose is corrected here as an editorial note, not re-derived.

### Sampler settings (frozen)

- **Paper-mirrored arm** (used for eval-200/holdout-100 scoring): temperature 0.6, top-p 0.95, 1024-token cap.
- **Greedy witness arm**: batch=1, on a 32-item slice, per-step logits hashes recorded for rerun agreement (A7).
- S1a's probe used greedy/batch=1/`max_new_tokens=400` — a **probe-only** budget for the
  self-pair mechanics test, not the eval sampler above; it is not reused for S3-S6.

### Dataset pins and split discipline (frozen)

| Item | Value |
|---|---|
| Test source | `https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl`, 1,319 problems |
| Test sha256 | `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14` (verbatim from `s0-receipt.json dataset.sha256`) |
| Train source | `.../grade_school_math/data/train.jsonl` |
| Train sha256 | `17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465` (verbatim from `s1a` receipt `dataset.sha256`) |
| Parsing | `serde_json` line-parse, `'#### n'` gold-answer normalization — no Python, per design constraint; S0/S1a receipts confirm this parsing path already runs |

**Splits**: `calibration-4000` (train), `adaptation-512` (train), `eval-200` (test),
`holdout-100` (test). Disjointness is guaranteed **by construction**, not by hoping four
independent samples don't collide: one seeded full permutation of each source file's item
indices, then contiguous non-overlapping slices taken in this fixed order —

- Train permutation (seed below) → first 4,000 indices = `calibration-4000`; next 512 =
  `adaptation-512`. (4,512 of the train file's ~7,473 items; S1a's own 40-item probe already
  touched indices up to 7462 from a *different*, ChaCha8-seeded 40-item sample of the same file —
  that probe draw is **not** guaranteed disjoint from these splits under the fixed-seed
  permutation above, so up to 40 of the 4,512 `calibration-4000`/`adaptation-512` items may
  overlap with items S1a already used. Registered as a known, low-risk overlap: the mechanism
  that makes it low-risk is that S1a's outcome informed only *mechanics* decisions already locked
  before this ADR — slot count, injection block, the rescale-to-median switch — none of which is
  a per-item statistic any A1-A8 criterion re-estimates from those same 40 items; S1a produced no
  alignment transform, no calibration fit, and no Darwin fitness signal that calibration/adaptation
  would otherwise re-derive from the same rows. The overlap is disclosed, not excluded, because
  excluding it would require a harness that doesn't exist yet to special-case 40 indices; A8's
  no-rerun discipline applies here too — this is reported, not patched around.)
- Test permutation (seed below) → first 200 = `eval-200`; next 100 = `holdout-100`. (300 of 1,319 test items.)

**Seed registration**: `harness/latentmesh-live` does not exist yet (see Implementation status),
so this ADR freezes the deterministic **generator** rather than literal index lists — any
conforming implementation of the formula below against the pinned sha256 source files reproduces
byte-identical splits, which is what "frozen" has to mean when the harness that would emit index
files hasn't been written. Formula: `ChaCha8Rng::seed_from_u64(seed)`, where `seed` is the
big-endian `u64` formed from the first 8 bytes of `SHA-256(label)`. **Authorship note**: these
four label strings and their resulting seed integers do not appear anywhere in design doc 024 or
in any committed receipt — they were chosen and computed at this ADR's authoring time (this
session, 2026-08-28) specifically to satisfy the S2-gate freezing requirement in the absence of a
harness that could otherwise emit them. They are frozen as of this ADR's merge, not sourced from
prior work; report this as an authorial decision, not a design-doc citation.

| Label | Frozen seed (u64) |
|---|---:|
| `latentmesh-live-run1-train-shuffle` | `10881612390959651218` (`0x970340d0faa39992`) |
| `latentmesh-live-run1-test-shuffle` | `11966235356209068823` (`0xa6109ba8198fbb17`) |
| `latentmesh-live-run1-darwin-genome` (the (1+3) hill-climb's mutation RNG) | `2327475860294564945` (`0x204cdaf6a4277c51`) |
| `latentmesh-live-run1-audit-run-seed` (base for `verify_edge`'s `seed = hash(run_seed, edge, generation)`) | `14877963171984959178` (`0xce7924ad9781e6ca`) |

`harness/latentmesh-live/src/gsm8k.rs` must (a) implement exactly this derivation, (b) commit the
resulting index lists to `harness/latentmesh-live/data/` once generated, and (c) **mechanically
refuse to yield an `eval-200` or `holdout-100` index until a genome-frozen receipt exists** — this
refusal is a required implementation behavior of that module, not a documented policy a caller
could bypass; S5's gate ("harness verifiably refused eval/holdout indices pre-freeze") is only
satisfiable if this is enforced in code.

**Adaptation-pool-from-train-only rule** (the mandatory graft, §1 of the design doc): Darwin
fitness batches and every per-generation causal audit draw **only** from `adaptation-512`
(train-derived). `eval-200` and `holdout-100` are touched only after the genome is frozen, and
only for scoring — never for fitness, never for audit-driven authority changes.

### S2 calibration protocol (frozen parameters; result pending)

- 4,000 GSM8K-train items (`calibration-4000` above), raised from the base design's 2,000
  specifically because 1,600 fit rows < sender dim 2,048 trips `latentmesh-align`'s own n<d
  arbitrary-null-space-rotation gap; 3,200 fit rows clears it. Cost ≈ +15 min prefill (Deviation 4
  below).
- Teacher-forced pairing: the same text (question + gold solution) through both models as
  prefill-only passes, mean-pooled over the solution-token span at each of a 3×3 relative-depth
  grid {50%, 66%, 80%} per side.
- Fit: affine mean-center each side (`mu_s`, `mu_r` in the hashed transform struct, per the
  landed S1b patch); fit on 80%; **report held-out (20%) relative residual as the honest quality
  number, never the crate's on-train confidence.** Pick the depth pair minimizing held-out
  residual.
- **A6 gate**: held-out relative residual < 0.9, else escalate calibration N or invoke the
  budgeted (~1 GPU-h) generated-pairs contingency (regenerate calibration pairs from
  sender-*generated* reasoning instead of gold text, to close the registered gold-vs-generated
  distribution-shift caveat) before S3 may proceed.
- Dense SVD at 2048×1536 timed explicitly, gated < 5 min, else fall back to the 1536×1536 `MᵀM`
  eigendecomposition path behind the same fit API (already implemented per S1b).

> **S2 transform `content_hash`: PENDING.** S2 is running concurrently with this ADR's authoring.
> The realized hash, chosen depth pair, and held-out residual will be appended to this section
> verbatim from the S2 calibration receipt when S2 completes, alongside `Policy::trust_transform`
> registration. No frozen threshold or seed above depends on this value.

### Acceptance criteria A1-A8 (frozen, testable exactly as stated, whatever the outcome)

| # | Type | Statement | Test | Threshold |
|---|---|---|---|---|
| A1 | Compute, primary | CausalDynamicLatent is cheaper than StaticText at eval time | One-sided paired bootstrap, 10,000 resamples, on the per-problem ratio `CDL eval-phase compute / StaticText eval-phase compute` (compute = GPU-seconds; FLOPs proxy `2·P·tokens` reported alongside) | PASS iff the 95% CI upper bound < 0.80 |
| A2 | Compute, secondary (reported, no pass/fail) | Amortized cost including adaptation and audit | `(CDL adaptation compute + all audit compute + CDL eval compute) / CDL eval episodes ÷ (StaticText eval compute / StaticText eval episodes)` | No threshold — reported with CI, pre-empts a "cheapness is an accounting artifact" objection |
| A3 | Quality | CausalDynamicLatent is not meaningfully worse than StaticText | One-sided non-inferiority test, margin −7pp, N=200×2 | PASS iff non-inferior at that margin (wider than the full-scale −5pp/500×3 design — the price of this run's cut, stated not hidden) |
| A4 | Edge survival — **descriptive by pre-registration, no hypothesis test claimed** | Fraction of admitted-edge events surviving all five ADR-003 controls | Exact binomial (Clopper–Pearson) CI on the observed fraction, expected ~6-12 admitted-edge events | No pass/fail — the statistical ≥80%-survival test (n≥50 events, reject if ≥46 survive) is explicitly deferred to the pre-costed 3-model extension |
| A5 | Audit validity gate | Every control's difference vector has enough discordant pairs to be a meaningful sign test | Discordance count at batch 48, escalating 48→64→80 | PASS requires ≥5 discordant pairs on **every** control at whichever batch size is reached; S5 may not start until this passes |
| A6 | Calibration gate | Alignment transform is usable | Held-out relative residual | PASS iff < 0.9, else escalate/contingency per S2 above |
| A7 | Mechanics gates | Injection pipeline does not corrupt inference | (a) capture-path logits bit-identical to unpatched forward; (b) self-pair **aligned-real** distinguishable from random at p<0.05 (design §6, verbatim); (c) zero-injection ≈ no-injection baseline; (d) greedy-arm logits-hash agreement across reruns | (a) **PASS** per S0 (`G3_*_logits_parity_bit_identical`). (b) **Registered ambiguity, not resolved as passed**: S1a's passing run used `transform: "identity"` (`s1a` receipt `config.transform`) — it proves raw re-injection transmits information, not that an S2-*aligned* vector does; whether design §6's "aligned-real" means this identity probe or requires a post-S2 re-check with the real cross-model transform is genuinely unclear from the design doc and is left open here rather than assumed satisfied. (c) **Weaker than the stated gate, flagged**: S0's `G5_zero_slot_equals_baseline` note reads "bit-identical BY CONSTRUCTION (empty position list is a no-op branch)" — that exercised the no-injection *code path* (an empty slot list, skipping injection entirely), not the real injection path fed a zero *vector* at all 8 slots; A7(c) as stated needs that second, stronger check before S5. (d) checked at S3-S5. **Coordinator resolution (2026-08-28, pre-S3, before any eval item consumed)**: (b) and (c) resolve in the rigorous direction — a mandatory S2→S3 bridge probe re-runs the S1a protocol with the S2-fitted cross-model transform (3B L24 → aligned → 1.5B L19, same pre-committed sign test), and the same run injects a true zero *vector* through the real 8-slot injection path; both must pass before S3 opens. |
| A8 | Reporting discipline | Any eval-vs-holdout discrepancy is reported, never grounds for a rerun | N/A — a standing rule, not a statistical test | Violating this rule (rerunning because holdout disagreed with eval) invalidates the run's evidence label |

All statistics above (bootstrap resample count, permutation-test alpha=0.05/resamples=2000 for
`verify_edge` per ADR-003's implementation, non-inferiority margin, Clopper-Pearson CIs) are
frozen as stated; none may be chosen or adjusted after any eval or holdout item is observed.

## Deviations ledger (to date)

1. **Runtime crate excluded from the root Cargo workspace (half/MSRV conflict).** `candle-core`
   0.9.2's dependency tree requires `half >= 2.5`, whose own `rust-version` is 1.81; the
   workspace's `latentmesh-core` pins `half = 2.4.1` under the workspace MSRV floor of 1.77. The
   two cannot unify in one `Cargo.lock`. Fix (already committed, root `Cargo.toml`):
   `latentmesh-runtime` is `exclude`d from the workspace, declares its own empty `[workspace]`
   (a standalone single-crate workspace, own lockfile, `rust-version = "1.81"`), and is built and
   driven as a subprocess from the harness — the same precedent `harness/evolve`/`harness/air`
   already use for shelling out to a `cargo`-built binary.
2. **candle-transformers 0.9.2 qwen2 BF16 RoPE fix — output-changing, applied uniformly.** The
   vendored 0.9.2 original casts rotary-embedding position indices to the model dtype *before*
   the outer product with `inv_freq`; in BF16 (8 mantissa bits) integer positions above ~256 alias
   to identical values, corrupting rotary angles deep into any sequence longer than that. Fixed by
   building the sin/cos tables in F32 and casting only the final result (`qwen2_a.rs`, declared
   Deviation 4, matching the pattern candle's own llama implementation already uses). This is
   **output-changing** — it measurably changes generated text — but it is applied identically to
   every model, every condition, and every stage from S0 onward (S0's mechanics gates were re-run
   and stayed green after the change); it is a bug fix to the vendored numerics, not a
   condition-specific intervention, so it does not introduce a between-condition confound.
3. **S1a run-1 failure, diagnosed, fixed, and both receipts preserved.** Run 1 of the S1a
   self-pair probe measured GATE FAIL (primary test p=0.75; first-pass, uninjected accuracy 5/40
   against an expected ~55-65%). Diagnosis (both preserved verbatim in
   `receipts/run-ledger.json`): (a) the RoPE BF16 aliasing bug above, confirmed live by comparing
   the same prompt at F32 (clean) vs. BF16 (degenerate, e.g. `"48 + 24 = 72"` → `"7288"`, token
   duplication), with stepwise-vs-prefill KV parity at 64/64 ruling out the decode loop as the
   cause; (b) a scoring bug — 22 of 40 run-1 generations never emitted `'####'`, so correct
   prose answers were scored wrong, measuring format compliance rather than the injection channel.
   Both fixes (RoPE F32 tables; an anchored-example answer-format instruction; identical flexible
   scoring applied to every condition and the first pass) were applied, and **the primary
   statistical test, alpha, item set, decoding, slots, block, pooling, and rescale switch were
   left unchanged from the pre-committed run-1 configuration** — only the mechanics bugs were
   fixed. Run 2 passed: one-sided exact sign test, real > random, paired accuracy, 5 wins / 0
   losses, p=0.03125 (< α=0.05). Both receipts are committed
   (`s1a-receipt-run1-buggy-rope-noncompliant-prompt.json` and the passing
   `s1a-receipt-slots8-block19-poolfull-rescaletrue-n40.json`), and the ledger records honest
   caveats about run 2 that this ADR repeats rather than hides: p=0.03125 is the minimum
   attainable p-value at 5 discordant pairs (2⁻⁵) — a real pass, not an overwhelming one, on a
   40-item mechanics probe, not an effect-size estimate; the NLL secondary diagnostic showed *no*
   effect in run 2 (19W/21L, p=0.68) and the run-1 NLL "effect" is best read as a numerics
   artifact of the RoPE bug, not a real signal.
4. **Calibration size raised 2,000 → 4,000 items (n<d fix).** 1,600 fit rows (80% of 2,000) is
   below sender dim 2,048 — `latentmesh-align`'s own documented arbitrary-null-space-rotation gap
   when fit rows < dimension. 3,200 fit rows (80% of 4,000) clears it. Cost: prefill-only, ≈+15
   minutes. No judge caught this in the design doc's own review process; it is declared here as a
   deviation from the base (mvp-first-verbatim) design, not a silent change.
5. **New, found this session: `harness/latentmesh-live` cannot be a root-workspace member as
   design §3 literally lists it, and this ADR corrects that.** Design doc §3's crate layout marks
   `harness/latentmesh-live/` as "NEW Rust bin crate (workspace member)." But that harness must
   depend on `latentmesh-runtime` (a path dependency, to call `load`/`generate`/`capture`/`inject`)
   — and `latentmesh-runtime` is precisely the crate Deviation 1 excludes from the root workspace
   for an MSRV/`half` conflict that a path dependency does not route around; adding
   `harness/latentmesh-live` to the root `members` list while it depends on `latentmesh-runtime`
   would reintroduce the exact conflict the exclusion exists to prevent. **This ADR registers the
   corrected layout**: `harness/latentmesh-live` must live outside the root workspace exactly like
   `latentmesh-runtime` (its own standalone crate/workspace, own lockfile, invoked as a
   subprocess from `harness/run.mjs` or directly), not a root `members` entry. Flagged as a
   conflict between the design doc and the coordinator's already-committed workspace-exclusion
   ruling — not silently resolved by quietly picking one.
6. **Design-doc prose vs. implemented norm-rescale target ("mean" vs. "median").** Covered in the
   injection-semantics table above; registered here for visibility in the ledger too, since it is
   exactly the kind of doc/code drift design §8 risk #12 already names as a standing category.

## Scope declaration — which ADR-009 clauses run 1 answers, and how

Per design §1/§10, **run 1 does not claim ADR-009 §5's combined acceptance test.** It answers:

- The **≥20%-cheaper-at-equal-quality clause** — **statistically** (A1 primary, A2 secondary
  accounting, A3 non-inferiority).
- The **>80%-edge-survival clause** — **descriptively only** (A4: expected ~6-12 admitted-edge
  events, far below the ≈50 events the binomial test needs for a real hypothesis test). No
  hypothesis test on edge survival is claimed by this run; the statistical version is deferred to
  the pre-costed 3-model extension named in the design doc.

The combined claim (both halves, both statistically significant, on the same run) **remains
open** after run 1 regardless of outcome.

## What this experiment will NOT claim

Carried forward from design §10, frozen as part of this pre-registration (every result row is
labeled `"live-model, single-host, simulation-free"`):

1. No claim of ADR-009 §5's combined acceptance test (see Scope declaration above).
2. Not a reproduction of Zhang & Emu — method mirrored (audit grid, decoding params, OPE/OME/CAG/SSG
   vocabulary), checkpoints not (Qwen2.5-3B/1.5B, not Qwen3-4B/8B); their published sign-flips
   between model sizes mean this run's effect signs need not match theirs.
3. K=1 mismatched decoy (the crate's actual `EdgeTrial` shape), not the paper's K=4 — a declared
   extension, not built here.
4. Pooled per-message vectors, not per-token latent streaming — nothing here speaks to
   LatentMAS-style token-level KV relay.
5. Single host, single GPU, single task domain (GSM8K) — no generalization claim beyond
   grade-school math word problems; the alignment transform is trustworthy only on the calibrated
   GSM8K subspace.
6. Calibration distribution shift is real and disclosed: transforms fit on gold-solution
   teacher-forced states, applied to sender-*generated* reasoning states; the held-out residual is
   honestly measured but same-distribution-of-text, not same-distribution-of-generation.
7. Reduced statistical scale: −7pp non-inferiority at N=200×2 is coarser evidence than the
   full-scale −5pp/500×3 design would provide.
8. "Darwin-mutated topology" is minimal by design: a (1+3) hill-climb over a 3-field genome on
   one structural edge — the honest minimum satisfying the condition definitions, not open-ended
   topology evolution.
9. The authority-consequence mechanism is demonstrated live, not proven necessary: run 1 shows
   ceilings moving with verdicts; it does not ablate whether the recalibrated thresholds
   (0.05/0.15, an experimental parameter already landed in S1b) are the *right* ones.
10. A passing A1 with a near-zero OPE/OME/CAG/SSG decomposition is a hollow win and will be
    reported as such — if `CAG ≈ 0` while the compute claim passes, the writeup states plainly
    that the latent channel contributed nothing measurable; the worst-of-five-controls gate exists
    precisely so this cannot be papered over.

## Consequences

Freezing this registration before S2 completes (rather than after) means the calibration result
itself cannot influence any threshold, seed, or statistical test chosen above — the one number
this ADR explicitly leaves open (the S2 `content_hash`) is provenance metadata, not an input to
any decision. The cost of writing this ADR from real S0/S1a evidence rather than the design doc's
prose alone is that this ADR surfaces two things the design doc got wrong or left ambiguous
(Deviations 5 and 6) before they became silent implementation choices made under time pressure
during S3-S5. The Deviations ledger being append-only and preserved-not-erased (both S1a receipts
committed, not just the passing one) is itself a load-bearing methodological choice: a
pre-registration that only shows the run that passed is not a pre-registration.

## S6 — Results (appended 2026-08-28)

S6 per design §7: compute A1-A8 exactly as registered, append results to this ADR, state
explicitly which ADR-009 clauses were answered statistically vs. descriptively. Run 1 stopped at
the pre-committed Deviation-7 kill point before A1-A8 could be computed — this section reports
what *did* run (S0 through S2c) and states plainly that A1-A8 were never reached, per A8's own
no-rerun/report-honestly discipline. Every number below is cited to a committed receipt in
`crates/latentmesh-runtime/receipts/`; none is retyped from a prior summary without
re-verification against the JSON.

### 1. What was established

- **Injection mechanics transmit information.** S1a's self-pair, identity-transform probe (run 2,
  after the RoPE/scoring fixes in Deviation 3) passed its pre-committed primary test: one-sided
  exact sign test, real > random, paired accuracy, 5 wins / 0 losses, **p = 0.03125**
  (`s1a-receipt-slots8-block19-poolfull-rescaletrue-n40.json`, `summary.primary_real_vs_random`).
  Honest decomposition, stated in the same receipt and repeated here rather than hidden: p=0.03125
  is the minimum attainable value at 5 discordant pairs (2⁻⁵) on a 40-item probe — a real pass,
  not an overwhelming one — and the NLL secondary diagnostic showed *no* effect in the same run
  (19 wins / 21 losses, p=0.68). The honest reading: raw re-injection of a model's own state into
  itself measurably helps accuracy on this probe, but the signal is thin, not the NLL-scale effect
  run 1 (buggy) originally over-measured before the RoPE fix.
- **The affine cross-model alignment fits well by the crate's own held-out metric, at both
  registered cells, under both calibration distributions:**

  | Cell | Calibration source | Held-out relative residual | A6 (< 0.9) | Receipt |
  |---|---|---:|---|---|
  | L18→L14 (S2 winner) | gold, teacher-forced | 0.5106 | PASS | `s2-calibration-receipt.json` |
  | L24→L19 (Deviation-6 anchor) | gold, teacher-forced | 0.5600 | PASS | `s2b-anchor-recalibration-receipt.json` |
  | L18→L14 (S2 winner) | generated, sender's own reasoning (S2c) | 0.4451 | PASS, **improved** | `s2c-calibration-receipt.json` |
  | L24→L19 (Deviation-6 anchor) | generated (S2c) | 0.4682 | PASS, **improved** | `s2c-calibration-receipt.json` |

  Every fit's hand-rolled `apply()` (required because `latentmesh-align` cannot be a path
  dependency of `latentmesh-runtime` — Deviation 1's half/MSRV conflict) reproduces the crate's
  own `AlignmentTransform::apply` bit-exactly against 8 golden pairs per cell
  (`max_relative_l2_error: 0.0`, all four `s2b*golden-affine*`/`s2c*golden-affine*` files),
  asserted before any model ran.

### 2. The central negative finding

Despite (1), the aligned, calibrated vector is **statistically indistinguishable from
norm-matched random noise** on A7(b)'s pre-committed primary test (one-sided exact sign test,
aligned_real > random, α=0.05), at both registered cells, under both calibration distributions:

| Cell | Calibration | wins / losses | p (one-sided exact sign) | A7(b) |
|---|---|---|---:|---|
| L18→L14 (winner) | gold | 2 / 1 | 0.5000 | **FAIL** |
| L24→L19 (anchor) | gold | 1 / 2 | 0.8750 | **FAIL** |
| L18→L14 (winner) | generated (S2c) | 3 / 1 | 0.3125 | **FAIL** |
| L24→L19 (anchor) | generated (S2c) | 1 / 2 | 0.8750 | **FAIL** |

(`s2b-receipt-cellL18toL14-slots8-poolfull-rescaletrue-n40.json`,
`s2b-receipt-cellL24toL19-slots8-poolfull-rescaletrue-n40.json`,
`s2b-receipt-cellL18toL14-slots8-poolfull-rescaletrue-n40-genpairs.json`,
`s2b-receipt-cellL24toL19-slots8-poolfull-rescaletrue-n40-genpairs.json`, all
`gates.A7b_aligned_real_vs_random`.) None reaches α=0.05. The winner cell moves toward
significance under the generated-pairs recalibration (p: 0.5000 → 0.3125) while the anchor cell
shows no movement at all (0.8750 both times) — a small, inconsistent shift, not a near-miss.
A7(c) (true zero-vector injected through the real 8-slot path, required "not catastrophically
below baseline") **passed in all four combinations** — the injection mechanism itself does not
harm the receiver; the aligned *content* specifically carries no usable signal.

**Design §8 risk 6 (gold-vs-generated calibration distribution shift as the failure cause) is
falsified as a full explanation, not confirmed.** The S2c contingency followed the design's own
pre-committed recipe exactly (Deviation 7), produced a *better*-fitting transform by the crate's
own residual metric at both cells (0.445 < 0.511 at the winner; 0.468 < 0.560 at the anchor), and
the causal signal still did not clear the pre-committed bar at either cell. If distribution shift
contributes at all, it is not the dominant or sole cause of the A7(b) failure.

**Attribution chain** (each link closes off one alternative explanation, receipt-cited):

1. Injection mechanics transmit information (S1a, p=0.03125) — rules out "the injection pathway
   is broken."
2. The hand-rolled `apply()` reproduces `latentmesh-align`'s own transform bit-exactly against
   golden pairs at every cell (`max_relative_l2_error: 0.0`) — rules out "the reimplementation
   forced by the half/MSRV split is wrong."
3. Identifiability: the gold-pairs fits use `n_fit=3200 > d_sender=2048` (comfortable margin).
   The generated-pairs fits use `n_fit=2048`, which **equals** `d_sender=2048` exactly — a
   boundary case, not comfortably `n>d`, even though the polar-uniqueness floor relative to
   `d_receiver=1536` clears with margin (`s2c-generated-dump-receipt.json gates.n_over_d`,
   `s2c-calibration-receipt.json fit.n_over_d_gate`). **Flagged, not silently passed over**: the
   S2c preplan's own stated goal was strict `n>d` (`s2c-preplan.json budget_ladder.quota_rule`),
   but the realized quota lands exactly *at* `d`, not above it, because the budget ladder's cost
   projection capped it there. This does not change the qualitative conclusion — the gold-pairs
   cells, comfortably `n>d`, show the same A7(b) failure — but it is a genuine residual caveat on
   the generated-pairs cells specifically, and this ADR states it rather than rounding "n=d" to
   "n>d."
4. Reproducibility: every bridge-probe receipt records `s1a_item_set_reproduced: pass=true` and
   `transform_hash_matches_registered: pass=true` — the exact frozen S1a item set and the exact
   registered transform were used each time, not a drifted re-implementation.

Given (1)-(4), the most defensible attribution is that a training-free, single global affine
(mean-centered semi-orthogonal Procrustes) map does not carry recoverable cross-model causal
information at either registered depth pair for this model pair, under pooled per-message
injection — not that the experiment's mechanics, hand-rolled apply implementation, or calibration
distribution were at fault.

### 3. Which ADR-009 clauses this run answers

**None of A1-A8 were computed.** The pre-registered scope declaration (above) named the
≥20%-cheaper-at-equal-quality clause (statistical) and the >80%-edge-survival clause (descriptive)
as run 1's intended targets. Both require `DynamicLatent`/`CausalDynamicLatent` to actually run,
and both conditions require a working aligned-injection channel to mean anything — A7(b)'s
failure at all four probe combinations fired the pre-committed Deviation-7 stop rule exactly as
registered ("if the generated-pairs aligned-real gate also fails at both cells, run 1 STOPS — S3-
S5 are not built"). `StaticText`/`DynamicText`/`DynamicLatent`/`CausalDynamicLatent` were never
built or run; S3 (pilot), S4 (audit pilot), and S5 (full run) do not exist. **The compute/latency
clause is NOT answered. The edge-survival clause is NOT answered. The combined ADR-009 §5
acceptance test remains exactly as open after run 1 as before it** — this run neither confirms
nor denies it; it establishes that a training-free linear alignment is not a viable vehicle to
test it with, at these two depth pairs, for this model pair.

What run 1 *does* answer, honestly: a narrower, logically prior question the design's own S1a/S2b
gate structure was built to ask — whether a training-free linear (Procrustes-family) alignment of
pooled residual-stream states, fit from either gold-teacher-forced or sender-generated calibration
pairs, carries causally usable cross-model signal at L18→L14 (50%/50% sender/receiver relative
depth, of 36/28 layers respectively) or L24→L19 (67%/68% sender/receiver relative depth) for
Qwen2.5-3B→Qwen2.5-1.5B. At high confidence, given the multi-distribution,
multi-cell, bit-verified cross-check above: **no.**

### 4. Total GPU accounting

Every wall-clock figure below is read directly from its receipt's `wall_clock_s` field (or, for
the killed S2c attempt, from the resumed receipt's own disclosure of the prior charge); nothing is
estimated.

| Stage | Seconds | Receipt |
|---|---:|---|
| S0 mechanics | 2.51 | `s0-receipt.json` |
| S1a run 1 (buggy RoPE/scoring, GPU time spent before diagnosis) | 542.57 | `s1a-receipt-run1-buggy-rope-noncompliant-prompt.json` |
| S1a run 2 (passing, n=40) | 512.22 | `s1a-receipt-slots8-block19-poolfull-rescaletrue-n40.json` |
| S1a n=2 smoke | 12.23 | `s1a-receipt-slots8-block19-poolfull-rescaletrue-n2.json` |
| S2 dump (4,000-item multi-tap capture) | 156.83 | `s2-dump-receipt.json` |
| S2b bridge probe, gold, L18→L14 | 390.41 | `s2b-receipt-cellL18toL14-...-n40.json` |
| S2b bridge probe, gold, L24→L19 | 393.06 | `s2b-receipt-cellL24toL19-...-n40.json` |
| S2c generated-dump: killed attempt (preserved, disclosed, charged) | 3696.03 | `s2c-generated-dump-receipt.json resumed.prior_gpu_s_charged` |
| S2c generated-dump: resumed process | 5310.37 | `s2c-generated-dump-receipt.json wall_clock_s.total` |
| S2b bridge probe, generated, L18→L14 | 391.35 | `s2b-receipt-...-n40-genpairs.json` |
| S2b bridge probe, generated, L24→L19 | 395.26 | `s2b-receipt-...-n40-genpairs.json` |
| **GPU-model subtotal** | **11,802.84 s ≈ 3.28 GPU-h** | |
| S2 calibration fit (CPU, 9-cell sweep) | 118.58 | `s2-calibration-receipt.json` |
| S2b anchor recalibration fit (CPU, 1 cell) | 13.51 | `s2b-anchor-recalibration-receipt.json` |
| S2c calibration fit (CPU, 2 cells) | 24.53 | `s2c-calibration-receipt.json` |
| **Grand total (GPU + CPU fit stages)** | **11,959.45 s ≈ 3.32 GPU-h** | |

**Discrepancy flagged, not silently resolved**: the task brief for this S6 write-up cited "~5.7
GPU-h" as the expected total. Summing every `wall_clock_s` field committed to
`crates/latentmesh-runtime/receipts/` — including the disclosed 3,696.03 s killed-dump charge —
gives **≈3.3 GPU-h**, not 5.7. No committed receipt records a standing aggregate total, and no
additional GPU-consuming stage was found beyond the eleven listed above. Two candidate
reconciliations were considered and neither is receipt-supported enough to assert: (a) the
real-world elapsed span from the first S0 receipt timestamp to the last S2c-genpairs receipt
timestamp is ≈4.6-5.0 hours (file mtimes 13:01→17:39), which includes idle/coordination gaps, not
GPU compute; (b) no per-stage double-counting convention (e.g., crediting both resident models
separately) reproduces 5.7 from 3.3 by a documented rule. This ADR reports the receipt-summed
**≈3.3 GPU-h** as the defensible number and flags the "~5.7" figure as unreconciled — a fact for
the coordinator to explain or correct, not to paper over with an invented adjustment.

### 5. Will-not-claim list (restated; most items are now vacuously true because S3-S6 never ran)

The design §10 list was frozen as part of this pre-registration. With S3-S6 unbuilt, items about
`DynamicLatent`/`CausalDynamicLatent` behavior (5, 7-10 below) never had anything to claim in the
first place — restated for completeness, marked where the item concerns work that did not occur:

1. No claim of ADR-009 §5's combined acceptance test. **Confirmed above (§3): unanswered.**
2. Not a reproduction of Zhang & Emu — moot; no condition-level decomposition was ever computed.
3. K=1 mismatched decoy vs. the paper's K=4 — moot; no `EdgeTrial` audit ran.
4. Pooled per-message vectors, not per-token streaming — **still the live, relevant caveat**: §6
   below proposes exactly the per-token relaxation this run never tested.
5. Single host, single GPU, single task domain (GSM8K) — **applies to everything that did run**
   (S0 through S2c).
6. Calibration distribution shift is real and disclosed — **directly tested this run** (§2): shown
   to improve the fit metric without closing the causal gap; not the dominant explanation.
7. Reduced statistical scale (−7pp/N=200×2) — moot; A3 was never computed.
8. "Darwin-mutated topology is minimal" — moot; no Darwin loop ran.
9. Authority-consequence mechanism demonstrated live, not proven necessary — moot; no gate/ceiling
   mutation ran against live episodes.
10. A passing A1 with a near-zero decomposition would be a hollow win — moot in the form written
    (A1 was never computed), but its spirit is exactly what happened one layer earlier: a
    passing-looking signal (well-fitting alignment, A6 PASS) paired with a null causal result
    (A7(b) FAIL) is precisely the "good fit, no content" failure mode item 10 warned about, just
    caught before S3 rather than after.

### 6. Registered future work — run 2 (proposal, not a pre-registration)

Not frozen, not binding, no acceptance criteria yet — a scoped starting point for whoever designs
run 2, given run 1's finding that a training-free linear map is the wrong tool at these cells:

- **(a) Small trained nonlinear projector (MLP).** The Cache-to-Cache-style baseline
  (`docs/research/023-beyond-sota-roadmap.md` §1a): replace the closed-form affine fit with a
  small trained MLP between the same capture/inject points, trained on the same calibration pairs
  (gold and/or generated). Cheapest way to test whether the ceiling on cross-model transfer here
  is *linearity* specifically, not alignment in general.
- **(b) A FastGRNN-class tiny gated-RNN sequence translator over per-token states, not pooled
  vectors.** This run pooled every capture to a single vector per message; pooling itself is an
  **untested destruction suspect** — S1a's identity-transform self-pair passed *with* pooling, so
  mechanics tolerate pooling in the no-alignment case, but cross-model translation of a pooled
  average was never isolated from translation-per-se. The S2c dumps already hold the raw material
  for a per-token sequence-translation experiment without new capture work: 2,560 items ×
  generated spans averaging 280.9 tokens = **719,115 sender-side generated-token positions**
  (`s2c-generated-dump-receipt.json rows.generated_tokens_mean/generated_tokens_total`; the
  receiver side is captured over the identical shared span per the S2c pairing rule, so the
  usable paired-sequence count is of the same order — the ~500K figure in earlier framing was an
  estimate, this is the receipted number). A tiny gated-RNN's KB-scale footprint also fits
  LatentMesh Air's edge-device story (ADR-010/013) better than a full trained projector would.
- **(c) Receiver-side MicroLoRA adaptation trained from causal-gate ΔV feedback.** In-stack prior
  art confirmed present on this host: `@ruvector/sona` MicroLoRA — rank 1-4 hidden-state adapters,
  <50 KB each, real-time `adapt()` update from feedback
  (`~/ruvector-upstream/crates/ruvllm-wasm/docs/MICRO_LORA.md`, verified read); and
  `AdaptiveEmbedder`'s EWC++ consolidation + memory-augmented retrieval
  (`~/ruvector-upstream/npm/packages/ruvector/src/core/adaptive-embedder.ts`, verified read,
  `class AdaptiveEmbedder`, EWC Fisher-information consolidation). Rather than a fixed transform,
  the receiver would adapt a small rank-1-4 adapter using ADR-003's own ΔV as the feedback signal
  — closer to the design's causal-gate spirit than a static alignment fit.
- **(d) HNSW-retrieved local linear maps (sublinear-selection variant).** A single global affine
  map may be the wrong granularity even before nonlinearity is considered — `latentmesh-memory`
  already has HNSW indexing (ADR-016); a local-linear-maps approach would retrieve/fit a
  neighborhood-specific transform per query rather than one global map, testing whether the null
  result is about global-vs-local structure rather than linear-vs-nonlinear.

None of (a)-(d) is scoped, budgeted, or gated here — that is run 2's own pre-registration to
write, informed by run 1's specific, receipt-verified failure mode (well-fitting global affine
map, null causal content, ruled out as a distribution-shift artifact).

## What is frozen / what is pending

| Item | Status |
|---|---|
| Four conditions, injection semantics, sampler settings, dataset pins, split-generation seeds, A1-A8 statistics and thresholds, deviations ledger through S2-completion, scope declaration, will-not-claim list | **Frozen by this ADR, honored through the point of stop** |
| S2 calibration result (depth pair, held-out residual, transform `content_hash`) | **Resolved** — winner L18→L14, residual 0.5106, hash `eb3f42edde853824642a2b811577e2c767f73c2c179fe03a05ac8dac23704457` |
| A7(b)/(c) mechanics-with-alignment bridge probe (the ambiguity this ADR originally flagged rather than resolving) | **Resolved — FAILED** at both cells, both calibration distributions (§ S6 — Results) |
| S3 pilot, S4 audit pilot, S5 full run | **Will not run for run 1** — Deviation-7 stop rule fired |
| A1-A8 | **Not computed** — see § S6 — Results §3 |
| `harness/latentmesh-live` crate | Built to the extent needed to reach S2c (per the corrected workspace-membership ruling above); `conditions/*`, `darwin.rs`, `audit.rs` (the S3-S5 machinery) **not implemented**, not needed for a stopped run |

## Implementation status

S0, S1a, S1b, S2, S2b, and S2c are implemented and their receipts are committed, cited throughout
this ADR. S3 (pilot), S4 (audit pilot), and S5 (full run) are **not implemented and will not be
built for run 1** — the Deviation-7 pre-committed stop rule fired before they were needed. This
ADR's frozen registration was honored unmodified from authoring through the point of failure; run
1 is concluded. Run 2, if pursued, is a new pre-registration informed by § S6 — Results §6, not an
amendment to this file.
