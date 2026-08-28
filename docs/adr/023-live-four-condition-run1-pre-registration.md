# 023. Pre-registration for the live latent-exchange experiment (run 1)

- **Status**: Proposed. This is a **pre-registration ADR**, not a results ADR: it is the S2-gate
  freeze required before any eval or holdout item may be consumed (design doc §7, S2 gate). It
  becomes the results record when S6 appends A1-A8 outcomes to this file — S6 amends this ADR
  with results, it does not open a new one, and nothing frozen below may change once amended.
- **Date**: 2026-08-28.
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
| S2 — calibration | held-out relative residual < 0.9 (A6) | **Running concurrently with this ADR** — result and transform `content_hash` pending | — |
| S3-S6 | per design §7 | **Not started** | — |

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

## What is frozen / what is pending

| Item | Status |
|---|---|
| Four conditions, injection semantics, sampler settings, dataset pins, split-generation seeds, A1-A8 statistics and thresholds, deviations ledger through this date, scope declaration, will-not-claim list | **Frozen by this ADR** |
| S2 calibration result (depth pair, held-out residual, transform `content_hash`) | **Pending** — running concurrently; appended verbatim when S2 completes; does not affect any frozen value above |
| `harness/latentmesh-live` crate (gsm8k.rs, calibrate.rs, conditions/*, darwin.rs, audit.rs, metrics.rs, receipts.rs, stats.rs) | **Not implemented** |
| Committed split index-list files (`calibration-4000`, `adaptation-512`, `eval-200`, `holdout-100`) | **Not implemented** — generated deterministically from the frozen seeds above once the harness exists; this ADR freezes the generator, not hand-written literal lists |
| S3 pilot, S4 audit pilot, S5 full run, S6 analysis | **Not started** |

## Implementation status

Not implemented beyond S0, S1a, and S1b, each already landed and cited by receipt above. This ADR
is the S2-gate contract those later stages must execute against unmodified; S3-S6 land in
follow-up work, and S6 amends this same file with A1-A8 outcomes rather than opening a new ADR.
