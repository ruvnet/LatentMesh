# 041. Run 2 results synthesis — skeleton

* **Purpose**: assemble the scattered findings of run 2's trained thought-adapter ladder (ADR-024)
  and its supporting research lane into one coherent, precisely-scoped account — the structure a
  results write-up will fill in, not the results write-up itself. **The run has not concluded**:
  M4g is executing as this document is written; M4h and M4b are registered and unrun. Every claim
  below is dated to what is established as of the most recent committed receipt, not projected
  forward.
* **Date**: 2026-08-29. Branch `feat/run2-thought-adapter`. Read-only against the repo except this
  one file — not committed.
* **Status**: skeleton for a future results record, built to satisfy [ADR-032](../adr/032-negative-result-publication-contract.md)'s
  publishability criteria section-by-section. Where a section cannot yet be filled (M4g/M4h/M4b
  outcomes), it says so explicitly rather than guessing.
* **Corpus read for this document**: [ADR-023](../adr/023-live-four-condition-run1-pre-registration.md)
  (in full, including § S6 and its external-corroboration annotation), [ADR-024](../adr/024-run2-trained-thought-adapter-ladder.md)
  (in full — the ladder's living ledger), [ADR-030](../adr/030-run3-causally-gated-text-pre-registration.md),
  [ADR-031](../adr/031-evidence-receipt-and-statistical-protocol-governance.md), [ADR-032](../adr/032-negative-result-publication-contract.md),
  [ADR-029](../adr/029-autogenous-governed-promotion-adoption.md) (§ provenance discrepancy only),
  [research/025](025-run1-negative-result.md), [research/028](028-sota-continuous-sweep-1.md),
  [research/029](029-a6-permutation-null.md), [research/031](031-statistical-power-and-design.md),
  [research/033](033-rescale-output-alignment-diagnostic.md), [research/035](035-probe-task-selection.md),
  [research/036](036-manifold-collapse-across-the-ladder.md), [research/038](038-manifold-constrained-adapter-scout.md),
  [research/039](039-bidirectional-latent-exchange.md), [research/040](040-the-pooling-gap.md).

---

## 1. The one-paragraph claim

**Across two runs and twelve adversarially-verified probe draws against a frozen 40-item GSM8K
protocol (Qwen2.5-3B-Instruct → Qwen2.5-1.5B-Instruct, block 18/24 → block 14/19), no architecture
tested — a training-free affine map (run 1), a trained MLP, a trained FastGRNN sequence translator,
or a task-loss-trained MLP with and without deployment-transform matching (run 2's M3, M4, M4c,
M4d) — has produced a statistically detectable causal effect of injected content on the receiver's
downstream accuracy, despite the alignment/reconstruction geometry itself being real and decisively
non-chance (§2.1).** The twelve nulls are not one failure mode: mechanistic analysis (§2.3-2.4)
splits them into two families with different, mutually exclusive signatures — reconstruction-loss
adapters that land convincingly on the manifold of *pooled* receiver states and still transmit
nothing usable, and task-loss adapters that remain orthogonal to that manifold from their untrained
initialization onward and actively harm the receiver (0/40 against every control on next-token
likelihood, unanimous). Three registered confounds — receiver sub-threshold scale, one-shot vs.
continuous delivery, and a newly-identified pooling/manifold-definition gap that is itself
independent of and upstream of the earlier manifold framing — remain untested at time of writing;
one rung (M4g, overwrite-vs-fuse) is executing and two (M4h de-pooling, M4b receiver scale) are
registered but not started. **This claim is scoped exactly to the tested architectures, the tested
model pair, the tested injection site, pooled payloads, and a sub-1.7B receiver** (§5); it does not
generalize to per-token/de-pooled transfer, continuous injection, a larger receiver, or any other
model pair, task, or channel — each of those is a live, separately-registered hypothesis, not a
closed question.

---

## 2. What is established

Each item below carries its receipt(s)/commit and an evidence grade in this repo's own vocabulary
(`primary` = directly measured/fetched and read this session or a prior cited session; `inferred` =
arithmetic/reasoning from primary numbers; `uncertain`/`disclosed caveat` = named limitation).

### 2.1 The alignment/reconstruction geometry is real, decisively non-chance, and mechanistically dissociated from causal usefulness

- **Run-1's training-free affine map fits well by held-out residual at both registered depth cells,
  under both calibration distributions**: gold L18→L14 0.5106, gold L24→L19 0.5600, generated
  L18→L14 0.4451, generated L24→L19 0.4682 — all clear the A6 gate (<0.9) by a wide margin.
  (`primary`, ADR-023 § S6 Table 1, receipts `s2-calibration-receipt.json`,
  `s2b-anchor-recalibration-receipt.json`, `s2c-calibration-receipt.json`.)
- **This fit is decisively better than chance, not a depth/width artifact.** An item-shuffle
  permutation null (20 seeded permutations per cell, 4 cells = 80 total permuted fits, same fit
  machinery) scores 1.02–1.03 on the identical metric — 0 of 80 permuted fits came within 0.46 of
  any real residual, z-scores of −550 to −885. (`primary`, research/029 §3, receipt
  `run2-a6-permnull-receipt.json`.) *(Note: an earlier framing of this finding cited "0/160
  permuted"; the receipted permutation count is 80 — 4 cells × 20 permutations each — not 160. This
  document uses the receipted number.)*
- **And yet the well-fitting, non-chance geometric alignment carries no detectable causal effect**
  on the receiver's behavior, at either cell, under either calibration distribution: aligned-real vs.
  random, one-sided exact sign test, p ∈ {0.5000, 0.8750, 0.3125, 0.8750} — none below α=0.05.
  (`primary`, ADR-023 § S6 Table 2.) This is the run's central dissociation, and it recurs in every
  subsequent architecture (below).
- **Independent external corroboration.** arXiv:2607.26773 (Zhu et al., a structurally similar
  OPE/OME/CAG/SSG causal decomposition, different model family — Qwen3-4B/8B) finds a
  content-attributable effect (CAG) near zero on GSM8K that **flips sign across model scale**
  (+5.17pp at 4B, −2.29pp at 8B). Their ARC-Challenge (knowledge, not reasoning) arm shows the same
  near-zero pattern, ruling out "it's specifically arithmetic reasoning" as the explanation.
  (`primary`, ADR-023 § S6 annotation; research/035 §2.)

### 2.2 The two mechanically distinct null families

`research/036`'s manifold pre-check (13 candidates: 11 trained/fitted emitters + 2 real-state
references, all L18→L14, all hash-verified against their freezing receipt) established that the
ladder's nulls do **not** share one mechanism:

| Family | Members | Manifold cosine (to receiver's own pooled state) | Norm | Signature |
|---|---|---:|---:|---|
| **On-manifold, still useless** | run-1 affine (both), M3 (both variants), M4 (r64/r128/r256) | 0.975–0.996 | 31.5–34.6 (natural: 34.5) | Reconstruction-loss training is a distance-to-manifold penalty by construction; it reliably lands there and still nulls |
| **Off-manifold, actively harmful** | M4c, M4d | −0.018 / +0.048 | 134.6 / 143.6 (14× the natural init's 9.3) | The untrained MLP init is *already* orthogonal (cosine −0.021); task-loss CE never constrained location, only downstream logits |
| **Intermediate** | M4 r=64 superseded (bad init, discarded pre-probe) | 0.511 | — | Not forced into either bucket |

(`primary`, research/036 §2, receipt `run2-manifold-precheck-receipt.json`; ADR-024 §"DIAGNOSIS"
and §"M4f PRE-CHECK VERDICT".) **Any write-up that reports "nothing transferred" as a single
narrative misrepresents both families** — this is stated explicitly in the source document and
repeated here because it is the single most load-bearing structural fact for how run 2's results
must be organized.

### 2.3 The M4c/M4d dissociation — the sharpest single result in the ladder

M4c (task-loss MLP) is the first rung with a **positive** training/transfer result and a probe
null at once:

- Task loss trained well: holdout CE 0.2546→0.1595 (best epoch 4). (`primary`, ADR-024 §"M4c
  outcome", receipt `run2-m4c-training-receipt-cellL18toL14.json`.)
- The improvement **transfers** across the composed↔fused BF16 numerical gap: mean fused NLL
  0.2385→0.1570, 498 wins/11 losses (secondary sign p=8.1e-132), measured **before** the probe ran.
  (`primary`, same receipt.)
- The frozen probe still nulls: aligned 23/40 vs. random 21/40, p=0.3438 (n_disc=6 — not a
  power-floor artifact; the test *could* have rejected at this discordant count, minimum attainable
  p=0.0156). (`primary`, receipt `run2-m4c-receipt-cellL18toL14-mlp-taskloss-slots8-poolfull-rescaletrue-n40.json`.)
- **The dissociation**: mean NLL of the gold-answer continuation is 5.359 (aligned) vs. 2.129
  (baseline) / 2.117 (random) / 2.154 (zerovec) — 0 wins / 40 losses vs. random, unanimous,
  p=1.0. Accuracy is untouched; answer-token likelihood collapses by 2.5×. (`primary`, same
  receipt; corroborated by the output-alignment diagnostic below.)
- M4d (task loss + deployment-transform-in-loop, registered to test whether rescale/slot-placement
  mismatch explained the reversal) reproduces the same pattern: primary gate 5W/2L, n_disc=7, exact
  p=0.2266, mid-p 0.1445 — **the first null in the ladder that is not a power-floor artifact at
  all** (minimum attainable p at n_disc=7 is 0.0078). NLL inversion persists unchanged: 0W/40L,
  mean NLL 4.92 vs. 2.13 baseline. (`primary`, ADR-024 §"M4d outcome", receipt
  `run2-m4d-receipt-cellL18toL14-mlp-deploymatch-*-n40.json`.)
- **Mechanistic explanation, closed by a zero-GPU diagnostic before M4d even reported**: the deploy
  rescale is a positive scalar through an overwrite (not additive) injection, so it is provably
  order-preserving on every unembedding-projected rank statistic (top-k overlap 1.000, argmax
  identical 40/40, gold-token rank bit-unchanged). Rescaling is exonerated; the mechanism is a
  fixed, item-near-invariant, off-manifold direction whose top-10 output tokens draw from only 77
  distinct vocabulary entries across all 40 items, dominated by rare embedding outliers
  (`DirectoryName`, `rias`, ` Svens`, ` Lanc`). (`primary`, research/033, receipt
  `run2-rescale-diagnostic-receipt.json`.)

### 2.4 The pooling gap — independent-of and upstream-of the manifold framing

- A genuine receiver block-14 state and that same item's mean-pooled state (the object every rung
  injects) have cosine **0.667**, entropy **9.30 vs. 3.36** nats, cross-item invariance **0.962 vs.
  0.635**. (`primary`, research/036 §4, same receipt as §2.2.)
- **No externally successful cross-model method pools a span into one vector.** C2C transfers
  per-token KV-cache entries at full sequence length; LatentMAS prepends full per-token/per-layer
  caches; the Bicameral Model couples live per-token states with no static object to pool at all;
  AVP's core algorithm is per-token. LatentMesh — every rung, including run-1's affine bridges and
  M4's nominally-sequential FastGRNN — is the only surveyed design that pools at the injection
  boundary. (`primary`, research/040 §1, direct primary fetches of C2C/LatentMAS/Bicameral/AVP
  source or full text.)
- This mechanism has decades of precedent (Ethayarajh 2019 anisotropy; Li et al. 2020 BERT-flow,
  STS-B 59.04 pooled vs. 70.72 isotropy-corrected) with a disclosed honest counter-case (SBERT's
  trained MEAN pooling *wins*, 87.44 vs. CLS 86.62 — pooling is not intrinsically fatal;
  *uncorrected pooling-unaware geometry* is). (`primary`, research/040 §2.)
- **This redefines, not merely complicates, what "on-manifold" means for every reconstruction-loss
  rung.** M3/M4's 0.975-0.996 cosine was measured against a pooled reference that is itself only
  0.667-cosine correlated with what the receiver actually carries — success at reaching that target
  is success at reaching an object the receiver never produces. (`primary`, research/036 §5,
  research/040 §4.)

### 2.5 The power finding — which draws could not have rejected, and which could

- The one-sided exact sign test's null distribution is `Binomial(n_disc, 0.5)`; at `n_disc≤4` the
  minimum attainable p-value (0.0625 at n_disc=4, 0.125 at n_disc=3) is **above** α=0.05 — the test
  is mathematically incapable of rejecting regardless of true effect size. (`primary`, research/031
  §2.1, exact binomial enumeration.)
- **Of the 10 valid cross-model draws through M4 (S2b×4, M3×2, M4×3, excluding S1a's same-model
  sanity check), 6 landed at `n_disc∈{3,4}`** — structurally incapable of significance. Adding M4c
  (n_disc=6, could have rejected) and M4d (n_disc=7, could have rejected) brings the total to 12
  valid draws, 8 of which were structurally capable of detecting an effect and did not, 4 of which
  (all pre-M4c) could not have rejected under any true effect size. (`primary`, research/031 §1
  table; ADR-024's own post-hoc power annotations on the M3/M4 outcome sections.)
- **M4d is the first genuinely informative negative in the ladder** — every earlier null except
  M4c sat at or near the power floor. (`primary`, ADR-024 §"M4d outcome".)
- At the ladder's own observed ~9.5% discordance rate (range 7.5-12.5%, stable across 9 draws
  regardless of injected content — including literal random noise), roughly 25-30 discordant pairs
  are needed for 80% power at a plausible effect size (θ=0.75) — five to six times what any 40-item
  draw has produced. (`primary`, research/031 §2.2-2.3.) This is why ADR-030 (run 3) adopts an
  anytime-valid e-process sequential test rather than repeating the fixed-40-item design.

---

## 3. The correction log

ADR-032's publishability contract requires confounds be "named and scoped, not discovered after the
fact and left unstated," and its own boundary table treats a documented correction as *evidence of
reliability*, not a liability. This ladder's append-only discipline (ADR-031(a)) means every error
below is preserved with its original text intact, superseded by an explicit correction note — the
list here is a reading guide to that trail, not a new disclosure.

| # | What was wrong | What caught it | Where recorded |
|---|---|---|---|
| 1 | M3 and M4's early "FAIL" verdicts were read as clean negatives | Post-hoc power analysis (research/031) showed 3 of the 5 M3/M4 draws landed at n_disc∈{3,4}, structurally incapable of rejecting the null at any effect size | ADR-024 M3/M4 outcome annotations (2026-08-29, appended) |
| 2 | M4d's registration asserted "no training rung has ever had [the deployment rescale] in its loop" | The M4d implementer read `train_m4c_taskloss.rs` directly and found the rescale was already present | ADR-024 §"CORRECTION to the M4d registration" |
| 3 | The correction in #2 was itself incomplete — it implied only the rescale was pre-existing | A second, deeper read found M4c's training loop already had pooling, rescale-to-median, **and** 8-slot placement all present; M4d's actual delta vs. M4c was therefore marginal | ADR-024 §"M4g PRE-REGISTRATION", opening paragraph ("Further premise correction... the earlier correction above understated the problem") |
| 4 | The M4d receipt was first summarized with the aligned-vs-baseline comparison (p=0.3438) reported as the primary statistic | A re-check against the "workflow's full report" found the actual primary gate (aligned vs. random) is p=0.2266, n_disc=7 | ADR-024 §"M4d outcome" ("Numbers corrected 2026-08-29... my first entry conflated two different comparisons") |
| 5 | research/033 attributed M4c's off-manifold reading to task-loss training specifically ("task-loss-specific" collapse) | research/038's M4f scout re-ran the identical unembedding projection against every ladder artifact and found the **untrained, zero-optimizer-step initialization** already sits at cosine −0.021 — the collapse predates any gradient step | ADR-024 §"CORRECTION to the M4f pre-check verdict and to the DIAGNOSIS", point 1 |
| 6 | Two of research/033 §4's three grounds for "off-manifold" ("77 distinct tokens," "nearly item-invariant," "gold at the 61st percentile") were read as diagnostic of M4c specifically | research/036 measured the same three statistics against the receiver's *own* natural pooled state and found two invert (natural state gives 78 distinct tokens, not diagnostic; natural state's item-invariance 0.962 makes M4c's 0.881 the *least* invariant pooled emitter, not the most) and one survives only as a relative claim | ADR-024 §"CORRECTION..." point 2; research/036 §3 |
| 7 | ADR-024's M4e registration claimed "every externally successful cross-model method injects continuously, at every generation step" | research/039 verified from C2C's own equation that its fuser applies **once, at prefill** — only the Bicameral Model injects continuously, and it does so bidirectionally, not as a clean instance of the claimed pattern | ADR-024 §"CORRECTION to M4e's premise", research/039 §1.3 |
| 8 | An early research brief cited MANTA alongside C2C/LatentMAS/Bicameral as a comparable latent-transfer method | research/040 (and independently ADR-024's own text) verified from MANTA's own abstract that it restructures multi-agent *topology* (roles, links, order, validation), not hidden-state content — a category error to compare it on the pooling axis | ADR-024 §"M4h PRE-REGISTRATION", "Scoping correction to an earlier brief"; research/040 §1.5 |
| 9 | ADR-028 lists "slot count" as both an evolvable surface (a search process may propose changing it) and part of the protected frozen-probe protocol | Found while scoping M4h; not adjudicated, only flagged | ADR-024 §"ADR-028 INTERNAL CONTRADICTION"; research/040 §3.5 (independently re-derives the same tension while scoping a de-pooling rung, sidesteps rather than resolves it) |
| 10 | `latentmesh-gate`'s design-origin attribution is disputed between two sources | autogenous's own README claims to be the design source for LatentMesh's causal edge-verification gate; `latentmesh-gate/src/lib.rs`'s own header says it was "ported in shape from `cognitum-one/slack`'s AGL admission model" — both read directly, not resolved | ADR-029 §"Attribution discrepancy" |
| 11 | The task brief motivating this synthesis (and an earlier internal framing) stated the permutation null beat 0/160 permuted fits | The receipted permutation count is 4 cells × 20 permutations = 80, not 160 | This document, §2.1 — corrected against `run2-a6-permnull-receipt.json` directly |
| 12 | Run-1's own GPU accounting: a "~5.7 GPU-h" figure appears in a task brief with no matching receipt | Summing every `wall_clock_s` field in every committed receipt gives 3.32 GPU-h; two reconciliation attempts (elapsed wall-clock span; a double-counting convention) were tried and neither reproduces 5.7 from 3.3 | ADR-023 § S6 §4 ("Discrepancy flagged, not silently resolved"); research/025 §5 |

---

## 4. What is NOT established

The run has three live, registered, unrun hypotheses and one executing rung. None of the following
is settled by anything above.

| Hypothesis | Status | What it would need | Why it survives the current evidence |
|---|---|---|---|
| **Overwrite-vs-fuse (M4g)** | **Executing now** | Retrain the M3-shaped adapter under task loss with the injection operator changed from hard `slice_assign` overwrite to a C2C-style residual add (`h[slot] += c·v`), one registered frozen-probe draw | C2C's own Eq. 3 is a residual add onto the receiver's own cache, never a replacement (`primary`, arXiv:2510.03215 fetched directly); LatentMesh's `inject.rs` verified from source to overwrite (`qwen2_b.rs:79-87`). A PASS would retroactively explain both null families; a NULL leaves M4f/M4b as live and strengthens the joint negative |
| **Receiver scale (M4b)** | **Registered, mandatory, unrun** — needs its own calibration/capture and pre-registration addendum | Repeat the best/least-bad adapter with a ≥1.7B receiver (Qwen2.5-3B fits alongside the sender on the 16GB card) | arXiv:2608.05164 reports cross-model steering transfer reliably works only above ~1.7B receiver parameters; Qwen2.5-1.5B sits just below it. Independently corroborated: Bicameral degrades on GSM8K specifically (49.6%→~40%) when the capability gap between coupled models is small — the same task, a second independent reason this arm is mandatory (`primary`, research/039 §1.1) |
| **De-pooling (M4h)** | **Registered, two-stage, unrun** | Stage 1 (near-zero cost): swap M3's already-trained per-token MLP to emit last-token instead of mean, same 8-slot broadcast, one probe draw. Stage 2 (~0.5-0.7 GPU-h): 8 distinct per-slot vectors via attention-compression over the sender's per-token stream | Every externally successful method preserves per-token granularity to the point of use (§2.4); LatentMesh's own injection mechanism already supports distinct per-slot vectors at the `LayerEdit` level — the pooling was only ever forced by a convenience wrapper (`InjectionSpec::vectors_tensor`), source-verified, not architecturally necessary |
| **One-shot vs. continuous injection (M4e)** | **Registered, deprioritized to after M4g/M4f/M4b, unscheduled** | New engineering, own pre-registration addendum; blocked until overwrite→fuse lands (a second round under overwrite would clobber the first round's content before integration) | The only confirmed continuous method (Bicameral) is also bidirectional, so continuous-alone is untested even by proxy; its justification ("everyone successful does this") was weakened by correction #7 above — only Bicameral actually does it |
| **Bidirectional/multi-round exchange** | **Named, explicitly deprioritized to last, likely out of run 2 entirely** | A wholesale new evaluation protocol (the frozen 40-item probe cannot score it in any form — every clause assumes a fixed one-shot injected vector and a single scored output) | One confirmed working precedent (Bicameral), no ablation isolating bidirectionality from continuity or from "having a real second model" at all; layered on an axis (continuous) this ladder has not tested even unidirectionally (`primary`, research/039, in full) |

**The run has not concluded.** M4g's outcome is not yet known at the time of this document. Whatever
it reports, M4f (manifold-constrained adapter, bank-attention over real receiver states, re-scoped
per research/038 to target item-invariance rather than mere manifold membership) remains registered
but is not sequenced ahead of M4h in the current ordering; M4b and M4h are both mandatory regardless
of M4g's outcome. **This document takes no position on which hypothesis will turn out to matter.**

---

## 5. Scope limits that must travel with every claim above

Every number in §2 is scoped, without exception, to:

- **A sub-1.7B receiver** (Qwen2.5-1.5B-Instruct). The receiver-scale confound (§4) means no claim
  above may be read as "cross-model latent transfer doesn't work" — only "at this receiver scale."
- **GSM8K, with two independent reasons its discordance rate may be structurally low**: (a) item
  difficulty is correlated across model scale (hard items are hard for both models), which
  suppresses discordant pairs independent of any injection mechanism — corroborated by an
  external paper's ARC-Challenge arm showing the same pattern despite being a knowledge, not
  reasoning, task (`primary`, research/035 §2-3); (b) the ladder's own observed 90-92.5% item
  concordance across 9 cross-model draws is consistent with, not merely asserted about, this
  mechanism (research/031 §1, research/035 §0).
- **One model pair** (Qwen2.5-3B-Instruct → Qwen2.5-1.5B-Instruct) — no cross-family claim is
  licensed.
- **One injection site** (block 18/24 sender → block 14/19 receiver, the S2-selected winner cell)
  — the anchor cell (L24→L19) was tested in run 1 only, not in any run-2 rung.
- **Pooled payloads throughout** (§2.4) — the single largest verified structural difference from
  every externally successful method, and itself untested as a variable until M4h.
- **A 40-item probe with a documented power floor** (§2.5) — 4 of the 12 valid draws to date could
  not have rejected the null under any true effect size; the remaining 8 could have and did not.

---

## 6. Figure/table list for a future results write-up

| # | Figure/table | Content | Receipt(s) supplying it |
|---|---|---|---|
| 1 | 2×2 null taxonomy | Manifold location (on/off) × loss type (reconstruction/task), all 8 trained candidates plus 2 training-free affine bridges plus 2 references, cosine/norm/entropy per candidate | `run2-manifold-precheck-receipt.json` (research/036 §2 table) |
| 2 | A6 residual vs. permutation null | Real residuals (4 cells) overlaid on their 20-permutation null distributions, with z-scores | `run2-a6-permnull-receipt.json` (research/029 §3) |
| 3 | Power floor vs. observed draws | Minimum attainable p by n_disc, overlaid with all 12 valid draws' actual (n_disc, p) points, marking which could/could not have rejected | research/031 §1-2 tables; ADR-024 M3/M4c/M4d outcome sections |
| 4 | Full ladder draw history table | All valid draws (S1a, S2b×4, M3×2, M4×3, M4c, M4d, [M4g pending]) with wins/losses/n_disc/exact-p/mid-p/verdict | research/031 §1 table, extended with M4c/M4d from ADR-024 |
| 5 | Pooling gap | Cosine/entropy/cross-item-invariance, pooled vs. single real state, bar or paired-point chart | `run2-manifold-precheck-receipt.json` (research/036 §4); STS-B literature comparators (research/040 §2.2-2.3) |
| 6 | M4c/M4d NLL dissociation | Accuracy (near-flat: 23/22/24/21, 24/22/24/21) vs. gold-answer NLL (5.359/2.129 and 4.92/2.13) across conditions, paired bars | `run2-m4c-receipt-*.json`, `run2-m4d-receipt-*.json` |
| 7 | Ladder decision tree | M3→M4(r64/128/256)→M4c→M4d→{M4f, M4g, M4h, M4b} with pass/fail/pending state per node | ADR-024 milestones table + all outcome/pre-registration sections |
| 8 | Correction-log table | §3 above, formatted for the results record | ADR-024 append-only annotations, cross-referenced |
| 9 | External corroboration comparison | This repo's 12-draw null pattern vs. Zhu et al.'s Qwen3-4B/8B CAG sign-flip (+5.17→−2.29pp) and ARC-C near-zero result | ADR-023 § S6 annotation; research/035 §2 |
| 10 | Cross-document contradiction table | §7 below, formatted for the results record | This document |

---

## 7. Open provenance/consistency items

Hunted for across the full corpus read for this document, per the assignment's explicit instruction
to look across documents rather than within any single one.

1. **ADR-028's slot-count contradiction (already flagged, restated for completeness).** ADR-028
   lists "slot count" on both its evolvable list (alongside pooling scheme, injection depth/site)
   and its protected list (as part of the frozen S1a/S2b probe protocol). Not adjudicated anywhere
   in the corpus; M4h sidesteps it by construction (keeps slot count fixed at exactly 8) rather than
   resolving it. **Recommendation carried forward from research/040 §3.5**: ADR-028's owner should
   adjudicate before any rung proposes changing slot count, not just before M4h specifically.

2. **ADR-029's autogenous-vs-cognitum-slack provenance conflict.** autogenous's own README states,
   in its "Related" section, that it is the design source for LatentMesh's causal edge-verification
   gate ("cognitum's admission-gate model"). `latentmesh-gate/src/lib.rs`'s own header states the
   gate was "ported in shape from `cognitum-one/slack`'s AGL admission model." Both were read
   directly this session (per ADR-029) and disagree on which repository the pattern originated in.
   Flagged as open in ADR-029 itself; not resolved anywhere else in the corpus.

3. **The GPU-hour reconciliation gap.** A "~5.7 GPU-h" figure, present in at least one task brief
   for this repo's own work, does not match the receipt-summed total of 3.32 GPU-h for everything
   S0 through S2c (ADR-023 § S6 §4; research/025 §5). Two candidate reconciliations (real-world
   elapsed span including idle time; a double-counting convention for two resident models) were
   checked and neither reproduces 5.7 from 3.3 under any documented rule. This is disclosed, not
   resolved, in both source documents — repeating that disclosure here because it is exactly the
   kind of unreconciled-number-flagged-not-rounded-away discipline ADR-032 requires, and because
   this document's own task brief (§2.1) inherited a related, distinct numeric error (0/160 vs.
   0/80 permutations) — two independent instances of a received-brief number not matching a
   receipt, in the same corpus, worth noting as a pattern rather than two unrelated typos.

4. **research/032's characterization of C2C as "continuous... in lockstep" is superseded but not
   corrected in place.** research/032 (not read in full for this document, but cited and quoted by
   038/039/040) originally described C2C as injecting continuously, at every generation step.
   research/038 §4 and research/039 §1.3, both fetching C2C's own equation directly, established
   this describes the Bicameral Model, not C2C — C2C's fuser applies once, at prefill. Per this
   corpus's append-only norm, research/032's original text was not edited; the correction lives only
   in the two later documents that cite it. A reader who consults research/032 alone (rather than
   038/039) would carry the wrong characterization forward. **Recommend research/032 gain a
   pointer-style annotation** (per the existing house style — see ADR-024's own "CORRECTION to..."
   sections) rather than leaving the correction discoverable only via forward citation.

5. **No numeric inconsistency was found in the core statistical record itself.** The n_disc/wins/
   losses/p-value quadruples for every draw were cross-checked between ADR-023 § S6, ADR-024's
   per-rung outcome sections, and research/031's independently-tabulated 10-draw table — all agree
   to the printed digit. This is stated explicitly because the assignment asked for contradictions
   found, and a clean cross-check (finding nothing wrong) is itself worth recording as a positive
   consistency result, not silently omitted because it isn't a finding.

---

## Sources

Every document listed in the header's "Corpus read for this document" line; no number above is
retyped from memory of a prior summary — each was re-read from the cited file this session.
