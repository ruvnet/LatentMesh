# LatentMesh Live Latent-Exchange Experiment — Final Design (Run 1)

**Status:** Implementation-ready synthesis. Base = judges' consensus winner **mvp-first** (judges 1 and 3 of 3; highest mean score 76.3), with every judge-endorsed idea from rigor-first and risk-first grafted in, and one judge overridden explicitly (below).

---

## 1. Decision Summary

**Build the mvp-first architecture — Qwen2.5-3B → Qwen2.5-1.5B on vendored candle 0.9.2/CUDA 12.8, four ADR-009 conditions, ≤16 GPU-hours — with three mandatory grafts and one scope declaration:**

1. **Mandatory graft (the price of the win):** Darwin adaptation moves entirely to a GSM8K-**train**-derived adaptation pool. Both winning judges conditioned their votes on this fix; mvp-first verbatim drew fitness batches from eval-set indices, which is adaptation leakage into the headline numbers. Topology/genome is frozen — receipt-enforced — before any eval or holdout item is consumed.
2. **Grafted probes (from risk-first, endorsed by all three judges):** Stage 1a self-pair injection-mechanics probe (identity transform), capture-path logits-parity gate, kill-switch economics with a pre-committed publish-the-negative-result exit under 5 GPU-hours.
3. **Grafted reporting discipline (from rigor-first, endorsed by judges 1 and 3):** dual compute accounting as explicit formulas, eval-vs-holdout discrepancy reported never rerun, stratified edge-survival reporting reserved for the multi-edge extension, W=40 floor for any future rolling re-verification.
4. **Scope declaration (resolves judge 2's structural objection):** Run 1 does **not** claim ADR-009 §5's combined acceptance test. It answers the compute/latency clause statistically and the >80% edge-survival clause descriptively, and pre-registers exactly that.

**Judge disagreements resolved explicitly:**

- **Judge 2 (voted rigor-first) is overridden on the winner.** Their two objections were (a) eval-set adaptation leakage and (b) the edge-survival clause being statistically unevaluable in a 2-model/6-generation run. (a) is fixed by grafting rigor-first's own adaptation-set design — the exact fix all three judges converged on. (b) is resolved by scope, not rebutted: run 1 pre-registers the survival clause as descriptive and defers the statistical version to the pre-costed 3-model extension. What cannot be fixed by grafting is rigor-first's cost of winning: an unverified llama.rs sender on the critical path of an sm_120 GPU where only qwen2 has been empirically run end-to-end, sequential persist-and-swap machinery forced by 12.6/14.3 GB marginal VRAM, a third-party mirror dependency with no HF token on host, and a 65–90 GPU-h commitment before any live four-condition result. Feasibility-on-this-hardware is the binding constraint; two of three judges scored it that way.
- **Judge 2 / risk-first on the model pair is also overridden.** The "same-family alignment ≈ identity" objection does not apply to the chosen pair: Qwen2.5-3B→1.5B is a genuine **rectangular 2048→1536** alignment that exercises latentmesh-align's dense semi-orthogonal path exactly as a cross-family pair would. Cross-family Llama-3.2-3B is extension arm 1, not run 1.
- **W=20 (risk-first) is rejected** per its own cited evidence (wiring finding #7, min p ≈ 2⁻ᵏ): run 1 uses per-generation audit batches of 48 with a discordance-gated escalation ladder; any future rolling re-verification uses W≥40 (the crate's reference n).
- **One internal fix no judge caught:** mvp-first's 2,000-item calibration yields 1,600 fit rows < sender dim 2,048 — the alignment scout's own headline gap (n<d ⇒ arbitrary null-space rotation). Calibration is raised to 4,000 train items (3,200 fit rows). Declared deviation from mvp-first verbatim; cost ≈ +15 min of prefill.

---

## 2. Runtime + Model Pair (with deciding evidence)

**Runtime: candle 0.9.2 (candle-core/candle-nn, `cuda` feature) with one vendored model file — candle-transformers 0.9.2's `qwen2.rs`.**

Deciding evidence (all empirical, this session):
- candle 0.9.2 + cuda built AND ran on this host's RTX 5080 (sm_120): CUDA matmul smoke test plus a full BF16 end-to-end qwen2-architecture generation (Qwen2.5-0.5B-Instruct, correct output), exercising the full custom-kernel set (rope, rmsnorm, softmax, cast) on Blackwell. **qwen2 is the only architecture with this evidence grade on this GPU.**
- The verified build pinned `PATH=/usr/local/cuda-12.8/bin` (nvcc 12.8.93). Host default nvcc is CUDA 13.0, **untested** with candle 0.9.2/cudarc — the pin is load-bearing and is enforced by a build-env guard, not just documentation (judge-2-endorsed graft).
- Rejected: mistral.rs (only gpt-oss path, no documented hidden-state surface, own candle fork unverified on sm_120); llama-cpp-2 (last-layer embeddings only; mid-layer capture = C++ patching); ollama (no hidden states; using it even for text arms would introduce a quantization/runtime confound — **all four conditions run through the same candle BF16 weights and sampler; ollama appears nowhere in measured runs**).

**Model pair: Qwen/Qwen2.5-3B-Instruct (sender, d=2048, 36 layers, ~6.2 GB BF16) → Qwen/Qwen2.5-1.5B-Instruct (receiver, d=1536, 28 layers, ~3.1 GB BF16; config.json verified this session).**

The unique pair satisfying every hard constraint simultaneously:
1. Single empirically-verified vendored architecture covers both models.
2. Both repos ungated — no HF token exists on this host; meta-llama 403s; no third-party mirror provenance risk.
3. **Concurrent** BF16 residency ~9.3 of 14.3 GB free — no sequential persist-and-swap subsystem.
4. Genuine rectangular 2048→1536 alignment exercising latentmesh-align's dense path honestly.
5. Heterogeneous strength (~79% vs ~60% GSM8K) gives the latent channel measurable headroom — Zhang & Emu's near-zero OPE on equal-strength 4B→4B relays shows equal pairs can null out.
6. Same tokenizer → trivial calibration pairing and text_equivalent construction.

Capture at sender layer 24/36, inject at receiver layer 19/28 (~2/3 relative depth), subject to the {50%, 66%, 80%} calibration sweep. **Extension arms (deferred, named):** cross-family Llama-3.2-3B (alignment stress), gpt-oss:20b via mistral.rs, 3-model topology for statistical edge survival, Qwen3-4B checkpoint mirror.

---

## 3. Crate Layout

```
crates/latentmesh-runtime/            NEW library crate
  src/lib.rs                          public API: load, generate, capture, inject;
                                      build-env guard asserting nvcc 12.8 toolchain
  src/models/qwen2_a.rs, qwen2_b.rs   vendored candle-transformers 0.9.2 qwen2.rs,
                                      split for the <500-line rule, version-pinned
  src/capture.rs                      forward_capture: pooled residual after block k
                                      over a token span (logits-parity guaranteed)
  src/inject.rs                       forward_inject: placeholder-position overwrite
                                      at layer k (matched depth; RoPE/KV coherent)
  src/sampler.rs                      T=0.6/top-p 0.95 arm + greedy batch=1 witness arm
  src/norms.rs                        injected vs natural layer-k norm logging
  deps: candle-core/candle-nn 0.9.2 (cuda), hf-hub, tokenizers

crates/latentmesh-align/              EXISTING, patched (with ADR-002 amendment):
                                      affine mean-centering (mu_s, mu_r in hashed struct);
                                      cached DMatrix (no ~25 MB rebuild per apply);
                                      content_hash once per fit;
                                      bench: dense SVD at 2048x1536, MᵀM eig fallback

crates/latentmesh-gate/               EXISTING, patched: ceiling_from_verdict dV
                                      thresholds policy-configurable (run-1 values
                                      0.05→LatentPrefix, 0.15→ActionInfluencing on the
                                      accuracy scale; defaults preserved so crate tests
                                      pass); risk_threshold=0.8; ADR-003 prose 4→5 controls

crates/latentmesh-core/               EXISTING, unmodified (LatentFrame, Payload,
                                      wire_bytes, Provenance, Authority)

harness/latentmesh-live/              NEW Rust bin crate (workspace member)
  src/main.rs                         CLI: s0-smoke, s1a-probe, calibrate, pilot,
                                      audit-pilot, run, audit, report
  src/gsm8k.rs                        raw JSONL loader (sha256-verified), '#### n'
                                      normalization, ChaCha8 subset selection,
                                      committed index lists; REFUSES eval AND holdout
                                      indices until a genome-frozen receipt exists
  src/calibrate.rs                    teacher-forced prefill pairing, mean-centering,
                                      80/20 held-out residual, 3x3 depth sweep
  src/conditions/{static_text,dynamic_text,dynamic_latent,causal_dynamic_latent}.rs
  src/darwin.rs                       (1+3) hill-climb, 6 generations, genome =
                                      {prompt variant x3, message budget, edge on/off};
                                      fitness batches drawn ONLY from adaptation pool
  src/audit.rs                        EdgeTrial: real/zero/random/mismatched(K=1)/
                                      self_generated(compute-matched Mself)/text_equivalent;
                                      OPE/OME/CAG/SSG
  src/metrics.rs                      accuracy, wall-clock, GPU-seconds, tokens,
                                      wire_bytes, FLOPs proxy, NVML 1 Hz samples
  src/receipts.rs                     ADR-014/018 evidence-labelled JSONL
  src/stats.rs                        paired bootstrap, non-inferiority, exact binomial CIs
harness/run.mjs                       optional thin Node stage sequencer only

harness/latentmesh-live/data/         gsm8k test.jsonl (sha256 3730d312...) +
                                      committed index lists: calibration-4000 (train),
                                      adaptation-512 (train), eval-200 (test),
                                      holdout-100 (test)

docs/adr/0XX-live-four-condition-run1.md   pre-registration ADR (frozen at S2 gate,
                                      BEFORE any eval item is consumed)
```

---

## 4. Per-Condition Implementation Spec

All four conditions: same candle BF16 weights, same sampler, both models CUDA-resident concurrently, batch=1.

**StaticText** — fixed hand-designed 2-agent pipeline. Sender (3B) generates a text reasoning summary under a fixed prompt and token budget; receiver (1.5B) gets question + sender text, answers `#### n`. No mutation, no gate. Cost/quality anchor. Message cost = UTF-8 bytes + token count.

**DynamicText** — identical channel; genome {sender prompt variant (3), message token budget, edge on/off} mutated by the in-harness (1+3) hill-climb, 6 generations, fitness = accuracy on 32-item batches **drawn from the adaptation pool (train-derived) only**; best genome frozen, then evaluated on eval-200×2. The Darwin loop is byte-identical across all three Dynamic* conditions so channel and fitness are the only variables. (latentmesh-evolve is simulation-only per ADR-018; this loop is the honest minimum for "Darwin-mutated".)

**DynamicLatent** — same genome/loop; channel = LatentFrame. Sender generates reasoning; `forward_capture` mean-pools layer-24 hidden states over the generated reasoning tokens; harness builds LatentFrame (Payload::encode F16, transform_hash from the fitted mean-centered 2048→1536 transform, confidence from held-out fit, provenance.context_hash = sha256(prompt), authority = ContextInject); `Gate::admit`; on admit: decode → transform.apply() → `forward_inject` at receiver layer 19 (matched depth, placeholder-slot overwrite). Receiver prompt contains the question only — the latent is the sole inter-model channel. Wire cost = `frame.wire_bytes()` exactly. A pre-declared norm-rescaling switch (to the natural layer-19 mean norm) exists in config; its setting is frozen in the ADR.

**CausalDynamicLatent** — DynamicLatent + the ADR-003 causal loop as fitness and governance. Each generation, the candidate edge runs the 6-way paired audit on a 48-item **adaptation-pool** batch: real / zero (**placeholders left unreplaced, mechanics otherwise byte-identical** — proving splice inertness inside the control itself, risk-first graft) / random norm-matched / mismatched (K=1, other-item aligned state, length-matched — the crate's actual EdgeTrial shape; K=4 is a declared trial-struct extension) / self_generated (**declared as Zhang & Emu's compute-matched Mself**: receiver generates its own reasoning through the same interface, pooled identically) / text_equivalent (sender reasoning serialized to text via the shared tokenizer). `verify_edge(alpha=0.05, resamples=2000, seed=hash(run_seed, edge, generation))` → dV, verdict; fitness = dV (Reject ⇒ fitness floor, edge culled); `set_ceiling_from_verdict` with recalibrated thresholds updates authority each generation — the authority-consequence mechanism live. **Additionally, after genome freeze: one full 6-way audit of the frozen final genome on eval-200** — this, not the adaptation-phase audits, is the source of the reported OPE/OME/CAG/SSG decomposition and survival CIs (otherwise the headline decomposition is measured on the data the genome was selected on).

---

## 5. Calibration Protocol

1. **Data:** 4,000 GSM8K-**train** items (index list committed; disjoint from adaptation/eval/holdout). Raised from mvp-first's 2,000 because 1,600 fit rows < d_sender=2,048 trips the n<d arbitrary-rotation gap; 3,200 rows > 2,048 clears it. Cost: prefill-only, ~30 min total.
2. **Pairing:** teacher-force the same text (question + gold solution) through both models as prefill-only passes; mean-pool over the solution-token span at each of a 3×3 relative-depth grid {50%, 66%, 80%} per side.
3. **Fit:** mean-center each side (store μ_s, μ_r in the hashed transform struct — the align-crate patch + ADR-002 amendment); fit AlignmentTransform on 80%; **report held-out (20%) relative residual as the honest quality number — never the crate's on-train confidence**. Pick the depth pair minimizing held-out residual.
4. **Registered caveat + contingency:** transmitted vectors pool over sender-*generated* reasoning while calibration pools over *gold* text — a logged distribution shift. If held-out residual is fine but live confidence collapses, spend one sender-generation pass per calibration item to regenerate pairs (budgeted contingency, ~1 GPU-h).
5. **SVD:** 2048×1536 dense SVD timed explicitly at S1 (<5 min gate) with the 1536×1536 MᵀM eigendecomposition fallback implemented behind the same fit API.
6. Transform content_hash registered via `Policy::trust_transform`; calibration receipt (indices, residuals, hash) committed.

---

## 6. Eval Protocol — Pre-Registered Acceptance Criteria

**Dataset:** GSM8K. Test split: raw JSONL, 1,319 problems, sha256 `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`, serde_json line-parse, no Python. Splits (seeded ChaCha8, index lists committed in-repo): **calibration-4000** (train) / **adaptation-512** (train; Darwin fitness + per-generation audits only) / **eval-200** (test; frozen-genome evaluation) / **holdout-100** (test; sealed). The harness mechanically refuses eval and holdout indices until a genome-frozen receipt exists.

**N and seeds:** eval-200 × 2 seeds {0, 1} for all four conditions; holdout-100 × 2 seeds, final genomes only, evaluated once — first look is the reported look. Decoding: T=0.6/top-p 0.95/1024 cap (paper-mirrored arm) + greedy batch=1 witness arm on a 32-item slice with per-step logits hashes.

**Metrics:** exact-match accuracy (normalized `#### n`); wall-clock per episode; GPU-seconds (+ NVML 1 Hz power/utilization cross-check receipts — judge-2 graft); prefill+decode tokens per agent; FLOPs proxy 2·P·tokens (P = 3.09B/1.54B, dense); bytes transferred (wire_bytes() for latent, UTF-8 length for text); per-edge dV, per-control p-values, discordant-pair counts, verdicts; OPE/OME/CAG/SSG in Zhang & Emu's vocabulary from the frozen-genome eval-200 audit; held-out alignment residual; injected-vs-natural norm distributions; genome edit distance across seeds/generations.

**Acceptance criteria (testable statements, computed exactly as registered, whatever the outcome):**

- **A1 (compute, primary accounting):** PASS iff the 95% CI upper bound (one-sided paired bootstrap, 10,000 resamples) of the per-problem ratio `eval-phase CausalDynamicLatent compute / eval-phase StaticText compute` is **< 0.80**. Compute = GPU-seconds (FLOPs proxy reported alongside).
- **A2 (compute, secondary accounting — dual-accounting formula, rigor-first graft):** report the amortized ratio `(CDL adaptation compute + all audit compute + CDL eval compute) / (number of CDL eval episodes) ÷ (StaticText eval compute / number of StaticText eval episodes)`. No pass threshold; reported with CI to pre-empt "cheapness is an accounting artifact."
- **A3 (quality):** PASS iff CausalDynamicLatent accuracy is one-sided non-inferior to StaticText at margin **−7pp**, N=200×2 (margin wider than the full-scale −5pp/500×3 design; the price of the run-1 cut, stated not hidden).
- **A4 (edge survival — DESCRIPTIVE by pre-registration):** report admitted-edge events (expected ~6–12) and the fraction surviving all five controls, with exact binomial (Clopper–Pearson) CIs. **No hypothesis test is claimed**; the statistical >80% test (n≥50 events, reject if ≥46 survive) is deferred to the 3-model extension.
- **A5 (audit validity gate):** the S4 audit pilot must observe **≥5 discordant pairs on every control's difference vector at batch 48**, escalating 48→64→80 until met, before S5 may start.
- **A6 (calibration gate):** held-out relative residual **< 0.9** for the chosen depth pair, else escalate calibration N / invoke the generated-pairs contingency before proceeding.
- **A7 (mechanics gates):** capture-path logits bit-identical to unpatched forward (parity); self-pair aligned-real distinguishable from random at p<0.05 (S1a); zero-injection ≈ no-injection baseline; greedy-arm logits-hash agreement across reruns.
- **A8:** any eval-vs-holdout discrepancy is itself a reported finding, never grounds for rerunning (rigor-first graft, verbatim in the ADR).

All parameters live in the pre-registration ADR + manifest, frozen at the S2 gate — none hard-coded, none chosen after data.

---

## 7. Staged Plan with Verification Gates

**S0 — Runtime spike + mechanics gates (0.5 day, ~0.5 GPU-h).** Stand up latentmesh-runtime (vendored qwen2, CUDA-12.8-pinned, build-env guard); download both models; on 3 GSM8K items run forward_capture (L24) and forward_inject (L19). **GATE:** build green under pinned nvcc 12.8; capture shape = 2048; **forward_capture logits bit-identical to unpatched forward**; injected logits finite and different from baseline; **zero-slot injection ≈ no-injection baseline**; injected-vector norms within ~3× of natural layer-19 distribution; receipts emitted.

**S1a — Self-pair injection-mechanics probe (~1 GPU-h; grafted from risk-first, endorsed by all three judges).** Qwen2.5-1.5B → itself, transform = identity: capture own pooled layer-19 state, re-inject on 40 held-out train items; paired sign-flip real vs zero vs random. **GATE:** real distinguishable from random (p<0.05) — the injection mechanism transmits information. **KILL-SWITCH:** if a model cannot use its own re-injected state, fix slot count/layer/scaling before any alignment work; a later null must be attributable.

**S1b — Crate patches (0.5–1 day, CPU).** Align: affine mean-centering + cached matrix + hash-once (+ ADR-002 amendment). Gate: configurable dV thresholds + ADR-003 five-control prose fix. **GATE:** `cargo test --workspace` green with existing crate tests unmodified; dense SVD 2048×1536 timed < 5 min (else switch to named MᵀM fallback and re-gate).

**S2 — Calibration + ADR freeze (0.5 day, ~1.5 GPU-h).** 4,000-item teacher-forced calibration, 3×3 sweep, held-out residual. **GATE:** A6 passes; transform hash registered; **pre-registration ADR merged — statistics, thresholds, deviations ledger, injection semantics, norm-switch setting all frozen BEFORE any eval item is consumed** (rigor-first's Stage-3 discipline, judge-2's core demand, adopted).

**S3 — Four-condition pilot (1 day, ~1.5 GPU-h).** N=32 adaptation items, 1 seed, Darwin capped at 2 generations; plus **receiver-alone baseline** (question only). **GATE:** all four conditions produce complete receipts; StaticText receiver accuracy ≥45%; DynamicLatent > receiver-alone floor; ≥1 admitted frame, zero panics.

**S4 — Causal-audit pilot (0.5 day, ~1.5 GPU-h).** One full 6-way EdgeTrial on a 48-item adaptation batch. **GATE:** A5 (discordance ladder); p-values non-degenerate; OPE/OME/CAG/SSG computed; ceiling rises above ContextInject at least once (recalibrated ladder is live); per-episode cost re-estimated against the S5 budget. **KILL-SWITCH (pre-committed):** if aligned injection is indistinguishable from random across the depth grid here and in S2 diagnostics, STOP — publish the negative result with full receipts for <5 GPU-h total; the condition harness's remaining scope is not built.

**S5 — Full run (2–3 nights, ~9–11 GPU-h).** 6 Darwin generations on adaptation-512 (fitness batches + 48-item audits); genome-frozen receipt written; then eval-200 × 2 seeds × 4 conditions; **frozen-genome 6-way audit on eval-200** (headline decomposition source); holdout-100 × 2 once; greedy witness slice. **GATE:** receipts complete for every episode; both seeds finish; harness verifiably refused eval/holdout indices pre-freeze.

**S6 — Analysis + report (1 day, CPU).** A1–A8 computed exactly as registered; results appended to the ADR; every headline number traceable to receipt rows by hash; deviations ledger complete; explicit statement of which ADR-009 clauses were answered statistically vs descriptively.

---

## 8. Risk Register

| # | Risk | Mitigation |
|---|------|------------|
| 1 | Sign-flip starvation (min p ≈ 2⁻ᵏ in discordant pairs): small audit batches auto-reject every edge, faking a negative | Batch 48 (> crate reference n=40) with 48→64→80 escalation gated at ≥5 discordant pairs per control (A5); discordance counts in every receipt; W≥40 floor codified for any future rolling window (W=20 rejected across the board) |
| 2 | Authority ladder dead on 0/1 outcomes (stock dV>1.0/0.5 unreachable) | S1b policy-configurable thresholds; run-1 values 0.05/0.15, risk_threshold 0.8, pre-declared; S4 gate proves a ceiling rises above ContextInject |
| 3 | Adaptation leakage into headline numbers | Train-derived adaptation-512; genome-frozen receipt; harness mechanically refuses eval AND holdout indices pre-freeze; frozen-genome eval-200 audit is the sole decomposition source |
| 4 | CUDA toolkit drift (only nvcc 12.8 verified; host default 13.0) | Build-env guard asserts 12.8 at compile and harness start; toolkit recorded in every receipt |
| 5 | Injection distribution mismatch → uninformative null | Matched-depth injection; S0 norm band + logging; pre-declared rescaling switch frozen in ADR; S1a self-pair probe attributes mechanics vs alignment; receiver-alone baseline separates useless from harmful |
| 6 | Calibration gold-vs-generated distribution shift; poor transfer trips RiskTooHigh, conflating alignment with governance | Caveat registered; A6 residual gate; budgeted generated-pairs contingency; gate rejections logged with reason codes |
| 7 | Dense SVD 2048×1536 unbenchmarked | S1b <5 min timing gate; MᵀM eigendecomposition fallback behind the same API |
| 8 | Compute-accounting attack on the ≥20% claim | Dual accounting pre-registered as formulas (A1 primary excl. audit, A2 amortized incl. adaptation+audit); audit episodes tagged at capture |
| 9 | Reduced scale weakens quality claim (−7pp at 200×2) | Margin declared prospectively; sealed holdout; harness N-parameterized so full-scale (500×3, −5pp) is a config change |
| 10 | VRAM regression (~9.3 GB + KV toward 14.3 GB) | 1024-token cap, batch=1, KV reset per episode, NVML peak receipts; documented sequential-residency fallback |
| 11 | Checkpoint non-mirroring vs Zhang & Emu | Deviations ledger: method mirrored (audit grid, decoding, OPE/OME/CAG/SSG vocabulary), checkpoint not; Qwen3 arm named as first extension |
| 12 | Doc/code drift (ADR-003 4-vs-5 controls, ADR-008 phantom `now`, non-serde Policy) | Implement against code; S1b lands ADR-003 fix; harness serde wrapper for Policy; drift filed in ADR amendments |
| 13 | Edge-survival descriptively answered could be misread as the full claim | Honesty section (below) + explicit clause-by-clause statement in the S6 report |

---

## 9. Compute Budget

**Ceiling ≤16 GPU-hours; expected ~12–14** on the RTX 5080 (both models CUDA-resident, batch=1, ~3–4 s/episode).

| Stage | GPU-h |
|---|---|
| S0 spike + mechanics gates | ~0.5 |
| S1a self-pair probe (grafted) | ~1.0 |
| S2 calibration (2 × 4,000 prefill-only passes + sweep) | ~1.5 |
| S3 four-condition pilot + receiver-alone baseline | ~1.5 |
| S4 audit pilot (48-item 6-way, escalation reserve) | ~1.5 |
| S5 full run: Darwin (6 gen × fitness 32×3 arms + 6×288 audit episodes) + eval 4×200×2×~2 agents + frozen-genome eval audit 200×2×6 + holdout 4×100×2 | ~9–11 |
| Contingency: generated-pairs calibration regen | ~1.0 |
| S6 analysis | CPU only |

Kill-switch economics: a negative alignment/mechanics result costs <5 GPU-h with publishable receipts. This is ~4× cheaper than the 50–90 GPU-h full-scale designs, which remain the pre-costed scale-up path (500×3 seeds, 3-model topology, K=4 decoys, statistical edge survival) reachable by config + one trial-struct change, not rearchitecture. Wall-clock ~4–6 working days; S5 as 2–3 unattended nights.

---

## 10. What This Experiment Will NOT Claim

Per this repo's evidence-label culture (ADR-014/018), every result row carries the grade **"live-model, single-host, simulation-free"**, and the writeup states explicitly:

1. **No claim of ADR-009 §5's combined acceptance test.** Run 1 answers the ≥20%-cheaper-at-equal-quality clause statistically (A1/A3) and the >80% edge-survival clause **descriptively only** (A4: ~6–12 admitted-edge events, far below the ~50 the binomial test needs). The combined claim remains open until the 3-model extension.
2. **Not a reproduction of Zhang & Emu.** Method mirrored (audit grid, decoding params, OPE/OME/CAG/SSG vocabulary); checkpoints not (Qwen2.5-3B/1.5B, not Qwen3-4B/8B), and their published sign-flips between model sizes mean our effect signs need not match theirs.
3. **K=1 mismatched decoy, not the paper's K=4** — the crate's EdgeTrial shape; K=4 is a declared extension.
4. **Pooled per-message vectors, not per-token latent streaming.** Nothing here speaks to token-level latent relay (LatentMAS-style KV relay is not what was tested).
5. **Single host, single GPU, single task domain (GSM8K).** No generalization claim beyond grade-school math word problems; the alignment transform is trustworthy only on the calibrated GSM8K subspace.
6. **Calibration distribution shift is real:** transforms fitted on gold-solution teacher-forced states, applied to sender-generated reasoning states; the residual number is held-out but same-distribution-of-text, not same-distribution-of-generation.
7. **Reduced statistical scale:** −7pp non-inferiority at N=200×2 is coarser than the full-scale −5pp/500×3 design; a pass here is weaker evidence than the full-scale run would provide.
8. **"Darwin-mutated topology" is minimal:** a (1+3) hill-climb over a 3-field genome on one structural edge — the honest minimum satisfying the condition definitions, not open-ended topology evolution.
9. **The authority-consequence mechanism is demonstrated live, not proven necessary:** run 1 shows ceilings moving with verdicts; it does not ablate whether recalibrated thresholds (0.05/0.15, an experimental parameter) are the right ones.
10. **A passing A1 with a near-zero decomposition is a hollow win and will be reported as such:** if CAG ≈ 0 while the compute claim passes, the writeup says the latent channel contributed nothing measurable — the worst-of-five-controls gate and the decomposition exist precisely so this cannot be papered over.

---

**Deviations-from-base ledger (this synthesis vs mvp-first verbatim):** calibration 2,000→4,000 (n<d fix); Darwin fitness batches eval-indices→adaptation-512 (judge-mandated); + frozen-genome eval-200 audit as decomposition source; + S1a probe, parity gate, zero≈baseline gate, kill-switch, dual-accounting formulas, NVML cross-check, eval-index mechanical refusal, A8 no-rerun rule; budget ≤15→≤16 GPU-h.