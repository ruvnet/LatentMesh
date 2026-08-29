# 046. Run 2 results synthesis v2 — supersedes 041 in six specific places

* **Purpose**: [docs/research/041](041-run2-synthesis-skeleton.md) was written 2026-08-29, before six
  major developments landed the same day and the day after. This document is a **superseding
  update, not a replacement** — it states up front which parts of 041 it supersedes (§0 below), then
  gives the full seven-section account in 041's own shape so the two can be read side by side. 041 is
  **append-only and not edited**, per this corpus's own discipline (ADR-031(a)); nothing here retypes
  041's numbers from memory — every figure below is re-read from its cited source this session.
* **Date**: 2026-08-29/30 (repo clock reads 2026-08-29 for every source document; this document's own
  authoring session began after the clock had advanced to 2026-08-29 per the system context). Branch
  `feat/run2-thought-adapter`. Read-only against the repo except this one file — not committed.
* **Status**: v2 synthesis, built to the same ADR-032 publishability-criteria shape as 041. **The run
  still has not concluded.** M4i and PC1 are both registered and unrun; M5X and M4b are both fully
  pre-registered but unstarted (M4b queued behind whatever currently holds the GPU lane). **PC1's
  outcome is the single most consequential unresolved fact in this document** — a FAIL would require
  attaching a caveat to every null since S1a, including several this document otherwise reports as
  established.
* **Corpus read for this document, beyond everything 041 already cites**: [ADR-024](../adr/024-run2-trained-thought-adapter-ladder.md)
  in full (all 1,863 lines, including its 2026-08-29 READING GUIDE — the document is append-only and
  materially out of chronological order; the guide's own chronological reading order was followed),
  [ADR-035](../adr/035-m4b-scale-control-pre-registration.md), [ADR-036](../adr/036-successor-rung-evaluation-protocol.md),
  [ADR-037](../adr/037-m5x-maximal-configuration-rung.md), [research/044](044-layer-coverage.md),
  [research/045](045-positive-control-design.md). research/042 and research/043 were read via
  ADR-024's own citation and summary of each (their findings are load-bearing to §2 below and are
  cited through ADR-024's sections, per the READING GUIDE's own convention of treating ADR-024 as the
  living ledger for these findings).

---

## 0. What this document supersedes in 041, and what still stands

041's seven sections are **not uniformly stale** — most of §5 (scope limits) and roughly half of §6
(figure list) are untouched by anything below. What changed:

| 041 section | Status | Why |
|---|---|---|
| §1 (one-paragraph claim) | **Superseded** | 041's claim describes "the twelve nulls" as splitting into two families by *loss function and manifold location combined*; the MAJOR CORRECTION (§2 below) establishes the split is *manifold location alone* — task loss, injection operator, and pooling are each individually refuted as the mechanism, and the correct two-family split is on-manifold-inert vs. off-manifold-harmful, full stop. 041 also predates M4g, M4h Stage 1, M4i's registration, M5X, M4b's full pre-registration, and PC1. |
| §2.1–2.2 | **Stands, with one addition** | The alignment/reconstruction dissociation and the original two-family framing are still true observations; §2 below restates them and layers the corrected framing on top rather than deleting them. |
| §2.3 (M4c dissociation) | **Stands** | Unaffected by anything since. |
| §2.4 (pooling gap) | **Superseded by outcome, not by finding** | 041 correctly identified the pooling gap as a live, untested hypothesis. It has since been tested (M4h Stage 1) and **refuted as sole cause** — the finding 041 reported (pooled vs. real-state geometry differs) is confirmed, not overturned; what changed is that fixing it was tried and didn't help. |
| §2.5 (power finding) | **Stands, extended, and one arithmetic point flagged** | The floor mechanics are unchanged; two more draws (M4g, M4h-S1) landed in the dead zone since, and this document's recount of the running total does not reproduce 041's own "10"/"12" tallies — see §7 item 1. |
| §3 (correction log) | **Extended, not replaced** | 041's 12 items all still stand as written. This document adds items #13–#17 for corrections that landed after 041 was authored, and promotes two items 041 had filed only under its own §7 into the correction log proper, per this document's own task brief. |
| §4 (not established) | **Fully superseded** | Every row in 041's table has moved: M4g and M4h Stage 1 (both "executing"/unlisted in 041) have reported; M4i has been registered (041 didn't know it existed); M5X and PC1 did not exist when 041 was written. |
| §5 (scope limits) | **Stands almost unchanged** | Still the correct scope envelope; the pooled-payloads bullet needs a one-line update (pooling is no longer "untested until M4h" — it was tested and refuted). |
| §6 (figure list) | **Extended** | 041's ten rows are still each individually valid; this document adds rows for the two-family NLL table, the instrument-power collapse, the layer-coverage comparator table, and the PC1 design. |
| §7 (open items) | **Extended, with new items found this session** | 041's five items stand. This document adds new items surfaced by reading ADR-024/036/037/044/045 in full — see §7. |

---

## 1. The claim as defensible right now

**Across two runs and fourteen adversarially-verified probe draws against the frozen 40-item GSM8K
protocol (thirteen cross-model plus one same-model sanity check), no architecture tested — a
training-free affine map, a trained MLP, a trained FastGRNN sequence translator, or task-loss-trained
MLPs under four independently-varied delivery mechanics (overwrite vs. residual-add injection, pooled
vs. de-pooled payload, standard vs. deployment-matched training) — has produced a statistically
detectable *beneficial* causal effect of injected content on the receiver's downstream accuracy.** The
alignment/reconstruction geometry is real and decisively non-chance (§2.1). Mechanistic analysis now
separates every completed rung into exactly **two families, cleanly, by one variable alone — manifold
location, not loss function, not injection operator, not pooling**: on-manifold payloads (run-1's
affine bridges, M3, M4) are **inert** — NLL within ~0.03 nats of baseline, several rungs producing
literally zero accuracy loss — while off-manifold payloads (M4c, M4d, M4g) are **actively harmful** —
an unrelenting 0/40 next-token-likelihood reversal that survives both injection operators tested
(§2.2). Three of the four single-factor root-cause candidates this ladder has now tried against the
on-manifold family — task loss, injection operator, and pooling — are each **individually refuted**:
none of them, once isolated, moves an on-manifold payload out of "inert" (§2.3). A fourth
(placeholder-token choice, M4i) and a fifth (receiver scale, M4b) remain untested; a sixth,
genuinely new candidate — **single-layer injection**, supported by the strongest external evidence
this ladder has found (C2C's own ablation reproduces the on-manifold signature almost exactly) —
raises the possibility that no single-factor rung on this design can ever pass, because every working
external method combines multiple of these axes at once (§2.4). **This claim is now qualified by a
methodological finding that did not exist when 041 was written and that could scope everything above:
no positive control has ever validated that this harness's *current* delivery mechanics (residual-add,
de-pooled, `<|fim_pad|>`) can carry signal at all, even in the maximally favorable same-model case**
(§4, PC1). **M4i and PC1 are both unresolved as this document is written; M5X (a deliberate four-factor
conjunction rung) and M4b (receiver scale) are both fully pre-registered but unstarted.** This claim is
scoped exactly to the tested architectures, delivery mechanics, model pair, injection site(s), payload
shapes, and a sub-1.7B receiver (§5); it does not generalize beyond what was actually drawn.

---

## 2. Established

Evidence grades follow this corpus's own vocabulary (`primary` = directly measured/fetched and read
this session or a prior cited session; `inferred` = arithmetic/reasoning from primary numbers;
`uncertain`/`disclosed caveat` = named limitation).

### 2.1 The alignment/reconstruction geometry is real and decisively non-chance — unchanged from 041

041 §2.1 stands without correction: the training-free affine map fits well by held-out residual at
both registered depth cells (`primary`, ADR-023 §S6), the fit clears an item-shuffle permutation null
by 80 seeded permutations with z-scores of −550 to −885 (`primary`, research/029 §3 — see §7 item 2
below for the 80-vs-160 count clarification since 041 was written), and the fit carries no detectable
causal effect at either cell (`primary`, ADR-023 §S6 Table 2). External corroboration
(arXiv:2607.26773) is unchanged.

### 2.2 MAJOR CORRECTION: the two null families split by manifold location alone, not by the framing 041 used

**This is the single largest correction to 041's own account.** 041 §2.2 described the split as
reconstruction-loss/pooled vs. task-loss/off-manifold — a framing that conflated two separable axes
(loss function and manifold location) that ADR-024's 2026-08-29 "MAJOR CORRECTION" section explicitly
disentangles, using two zero-cost checks against already-committed receipts, no new runs:

1. **The 8-identical-rows hypothesis is refuted, for free.** Under overwrite, the zero-vector control
   writes eight literal zero rows over real content and is harmless (NLL 2.1537 vs. baseline 2.1288,
   19W/21L, p=0.68) while the *aligned* payload in the same rung costs 3.2 nats. If duplication or
   placement caused the harm, zeros would harm too. They do not. **The harm is content-dependent, not
   placement-dependent.** (`primary`, ADR-024 §"MAJOR CORRECTION"; this falsifies research/042's own
   repeated-slots hypothesis, which research/042 itself proposed the check that falsified it — recorded
   as such, not as a failure of that document.)
2. **The claim that the inversion "survived on-manifold and off-manifold payloads" — 041's own framing
   — is false, measured directly:**

   | rung | manifold | aligned NLL | baseline | aligned vs. random |
   |---|---|---:|---:|---|
   | M3 per-token | on | 2.1328 | 2.1288 | 16W/24L |
   | M3 pooled | on | 2.1302 | 2.1288 | 17W/23L |
   | M4 r64 | on | 2.1171 | 2.1288 | 16W/24L |
   | M4 r128 | on | 2.1295 | 2.1288 | 16W/24L |
   | M4 r256 | on | **2.1074** (better than baseline) | 2.1288 | 16W/24L |
   | M4c | off | 5.3590 | 2.1288 | **0W/40L** |
   | M4d | off | 4.9193 | 2.1288 | **0W/40L** |
   | M4g | off | 4.8155 | 2.1288 | **0W/40L** |

   (`primary`, ADR-024 §"MAJOR CORRECTION", cross-checked against each rung's own outcome section.)
   **The 0/40 unanimous inversion occurs only in off-manifold task-loss rungs. Every on-manifold rung
   sits within ~0.03 nats of baseline — M4 r256 is actually better than baseline.** The on-manifold
   family is not harmful at all; it is inert.

**Corrected two-family statement, replacing 041's §2.2 table**: on-manifold payloads (run-1 affine,
M3, M4) are harmless and inert — nothing transfers, nothing breaks. Off-manifold payloads (M4c, M4d,
M4g) are actively destructive — NLL 2.3–2.5× baseline, unanimous 0/40, and **this survives both the
overwrite and the fuse (residual-add) injection operator** (§2.3), so it is the off-manifold *content*
itself doing the damage (norm ~134–144 vs. natural ~34), not the delivery mechanism.

### 2.3 Three named root-cause candidates for the on-manifold family's inertness, each individually refuted

Three architecturally distinct hypotheses for *why the on-manifold family transfers nothing* have now
each been tested in isolation and each refuted — none of them, once controlled for, moves an
on-manifold payload out of "inert":

1. **Task loss was refuted as the cause of off-manifold collapse specifically (not as a cause of
   inertness, since M4c/M4d/M4g are the off-manifold family, but as the mechanism the ladder had
   suspected before this correction).** The M4f pre-check found the shared M4c/M4d untrained
   initialization was already off-manifold at cosine −0.021, **before a single gradient step** —
   reconstruction training (MSE) structurally forces manifold proximity by construction, and
   task-loss CE imposes no such pressure, so a random init that starts off-manifold simply never has
   to return. (`primary`, ADR-024 §"CORRECTION to the M4f pre-check verdict", §"M4f PRE-CHECK
   VERDICT".)
2. **Injection operator (overwrite vs. fuse) — refuted by M4g.** C2C's own Eq. 3 (`C_F = C_n(X) +
   F_n(...)`) is a residual add; LatentMesh's `inject.rs` performed a hard `slice_assign` overwrite.
   M4g retrained the M3-shaped task-loss adapter under a residual-add operator (`LayerEdit::Fuse`,
   added beside `Inject` so every prior receipt stayed reproducible). Training was the best of the
   three task-loss rungs (holdout CE 0.2418→0.1560, transfer 0.2276→0.1536 fused NLL, 498W/11L). **The
   0/40 NLL inversion persisted essentially unchanged**: 4.815 aligned vs. 2.129 baseline, 0W/40L
   against both controls (M4c 5.359, M4d 4.919 — fuse moved the mean by ~0.1 nats, moved the win count
   not at all). The primary accuracy statistic itself was structurally uninformative (n_disc=3, floor
   0.125), but the NLL result is not power-limited and is decisive. (`primary`, ADR-024 §"M4g
   OUTCOME".)
3. **Pooling — refuted by M4h Stage 1, the most striking single result since 041.** research/040's
   finding (041 §2.4, still valid as a *geometric* finding) is that pooled and real receiver states
   differ sharply (cosine 0.667, entropy 9.30 vs. 3.36, cross-item invariance 0.962 vs. 0.635). M4h
   Stage 1 took M3's already-trained per-token MLP and emitted the last token instead of the mean —
   zero new training, zero new capture. The pre-check classified the resulting payload
   `on-manifold-item-varying`, the first candidate in the entire 15-row kit to be **statistically
   indistinguishable from a genuine single receiver row**:

   | candidate | invariance | top-10 union | cos-to-pooled-ref | entropy (nats) | class |
   |---|---:|---:|---:|---:|---|
   | M3 per-token (pooled) | 0.9702 | 70/400 | 0.9889 | 9.34 | item-invariant-but-on-manifold |
   | **M4h S1 (de-pooled)** | **0.6670** | **133/400** | **0.6814** | **3.32** | **on-manifold-item-varying** |
   | ref: receiver L14 single row | 0.6350 | 153/400 | 0.6670 | 3.36 | on-manifold-item-varying |
   | ref: receiver L14 pooled | 0.9617 | 78/400 | 1.0000 | 9.30 | item-invariant-but-on-manifold |

   (`primary`, ADR-024 §"M4h Stage 1 manifold pre-check", cross-checked against the outcome section's
   identical table.) The probe draw: aligned 24/40 (ties the ladder's best), 2W/0L against baseline
   (zero losses), but `n_disc=2` — minimum attainable one-sided p is 0.25, five times α, "the weakest
   draw in the ladder" by its own receipt (`power_limited: true`). **The informative result is the
   NLL, and it is negative**: +0.031 nats vs. baseline, the largest positive deviation of any
   on-manifold rung, and 10W/30L against zerovec (M3 pooled was an even 20W/20L). **A payload can be
   made geometrically indistinguishable from a real receiver state and still carry nothing.**
   (`primary`, ADR-024 §"M4h STAGE 1 OUTCOME".) 041's own §2.4 framed pooling as an untested,
   independent-and-upstream confound; that is now half-superseded — the geometric finding stands, the
   consequence 041 implied (fixing it would matter) does not.

### 2.4 Layer coverage — the strongest external evidence yet, and a methodological reckoning that did not exist at 041

research/044 (2026-08-29, not read by 041's author) found the citation this ladder had been missing:
**C2C's own Table 10 ablates single-layer content enrichment and finds it statistically
indistinguishable from doing nothing** — 58.42% baseline vs. 58.45–58.52% for the two best individual
layers (≈0.1pp), most other individual layers net-negative (`primary`, arXiv:2510.03215 fetched and
quoted). **No content-transfer method in the literature uses one layer**: C2C gates over roughly the
top-5 of 28 layers; LatentMAS transfers the full per-token cache at every layer; the Bicameral Model
couples at exactly 4 fixed layer indices, found by sweeping 890 configurations — a **correction to
this repository's own prior characterization** of Bicameral (research/028, 039, 040 all called it
merely "continuous," silent on layer count; research/044 fetched the paper's own equation and found
"the interface reads and writes at four layer indices... two read/write layer pairs, one per coupling
direction" (`primary`, arXiv:2605.11167). This is a genuinely new finding, distinct from 041's
correction-log item 7 (which corrected the *continuity* claim, not the *layer-count* claim) — see §3
item 13.

**The distinction that keeps this consistent with the steering-vector literature (CAA/ActAdd, which
do work from one layer)**: those methods add a fixed *direction*, continuously, at every generated
token, nudging an already-computed decision. LatentMesh injects a one-shot *state* that must survive
14+ subsequent blocks unassisted. RepE (the one steering-family method that transfers richer content,
not just a direction) explicitly uses many layers and names the mechanism: "the potential cascading
effect when simultaneously altering representations across multiple layers... changes made in earlier
layers may propagate to later layers, diminishing the effect" (`primary`, arXiv:2310.01405) — a direct
primary-sourced statement of the RMSNorm-dilution mechanism research/033 §5 named as surviving and
untested.

**Honest limit, stated by research/044 itself and worth restating here**: layer coverage explains the
on-manifold family's *ceiling* (why nothing beneficial gets through) but **not** the off-manifold
family's *harm* — M4c/M4d/M4g were also single-layer, and if single-layer injection alone caused harm,
single-layer on-manifold payloads would be harmful too. They are not (§2.2).

**The methodological reckoning, quoted because ADR-024 itself flags it as potentially the run's real
conclusion**: every working external method combines multi-layer **and** continuous delivery **and**
task-loss training **simultaneously**. This ladder, by design, has varied exactly one factor per rung
— the discipline that makes each null attributable is also, possibly, exactly why every rung nulls. If
transfer requires a *conjunction*, no single-factor rung can succeed by construction, and this is
testable only by a rung that deliberately abandons single-factor attribution — which is what M5X
(§4) now does. (`primary`, research/044 §5; ADR-024 §"LAYER COVERAGE".)

### 2.5 The power finding — extended past 041, with a recount discrepancy flagged

041 §2.5's mechanics are unchanged: the one-sided exact sign test cannot reject at `n_disc≤4`
(minimum attainable p 0.0625 at n_disc=4, 0.125 at n_disc=3), and this repository's own observed
~9.5% discordance rate implies ~25–30 discordant pairs are needed for 80% power at a plausible effect
size — five to six times what any 40-item draw has produced (`primary`, research/031 §2).

**Two more draws have landed in the dead zone since 041**: M4g at `n_disc=3` (floor 0.125) and M4h
Stage 1 at `n_disc=2` (floor 0.25, "the weakest draw in the ladder"). ADR-024 names the resulting
pattern directly, quoted because it motivated the entire protocol replacement in §4 below:

> **THE BINDING CONSTRAINT IS NOW THE INSTRUMENT, NOT THE SCIENCE.** Discordant counts across the
> ladder's recent draws: M4d **7** (the only non-limited one), M4g **3** (floor 0.125), M4h-S1 **2**
> (floor 0.25). As payloads become *better behaved* — on-manifold, non-destructive, item-varying —
> they perturb fewer items, so discordance **falls** and the frozen 40-item sign test loses what
> little power it had. The better our adapters get, the blinder the probe becomes.

Recomputing the running tally directly from each rung's own receipted `n_disc` (rather than retyping
041's summary numbers) across all 13 valid cross-model draws to date (S2b×4: n_disc 3,3,4,3; M3×2:
4,3; M4×3: 4,4,5; M4c: 6; M4d: 7; M4g: 3; M4h-S1: 2) gives **10 of 13 structurally incapable of
rejecting at any true effect size, and 3 capable-and-nulled (M4 r256 at the floor, M4c, M4d)**. This
recount does not reproduce 041's own "10 valid draws through M4" / "12 valid draws, 8 capable" tally
for the equivalent pre-M4g subset — see §7 item 1 for the discrepancy, disclosed rather than silently
resolved in either direction.

---

## 3. The correction log — now 17 items, cross-document corrections included per this document's own remit

041's 12 items (see [041 §3](041-run2-synthesis-skeleton.md#3-the-correction-log) for full text) all
stand unedited. This section reproduces them by one-line summary for continuity, then adds five new
items (#13–#17) surfaced since 041, including two that correct claims across documents rather than
within one — the strongest form of evidence ADR-032 treats a documented correction as.

| # | What was wrong | What caught it | Where recorded |
|---|---|---|---|
| 1–12 | *(carried forward from 041 unedited — see 041 §3 for full text: M3/M4 power-floor reframing; M4d's rescale-premise error; its own incomplete correction; M4d's primary-vs-secondary p mixup; the "task-loss-specific" collapse claim; two of three off-manifold diagnostic grounds inverting; M4e's "everyone injects continuously" overstatement; the MANTA mischaracterization; ADR-028's slot-count self-contradiction; the autogenous/cognitum-slack attribution dispute; the 0/160-vs-0/80 permutation count; the 5.7-vs-3.32 GPU-h discrepancy)* | — | 041 §3 |
| **13** | **research/028, 039, and 040 all characterized the Bicameral Model as merely "continuous... every generation step," silent on layer count** | research/044, fetching the Bicameral paper's own equation directly, found it couples at exactly 4 fixed layer indices (2 read + 2 write per direction), swept over 890 configurations — a genuinely new, previously-uncaptured structural fact, not merely a restatement of the continuity correction (041's item 7, corr. log entry for M4e, addressed *when* Bicameral injects; this addresses *where*) | ADR-024 §"LAYER COVERAGE"; research/044 §2.3 |
| **14** | ADR-030 states "the frozen 40-item probe... is a subset of `adaptation-512` by construction" | ADR-036, while designing the e-process item stream, intersected the S1a probe's 40 item indices directly against the committed `adaptation-512.json` and found the true intersection is **exactly 1 item** (index 1153), not 40 — 22 of 40 sit in `calibration-4000` instead, and 17 of 40 sit in neither file, drawn by S1a's own independent ChaCha8 seed. ADR-030 itself already discloses this fact elsewhere ("S1a's own 40-item probe already touched indices... from a *different*... sample"), making its later "subset by construction" claim an internal contradiction with a document it cites as its own source, not merely a stale fact | ADR-036 §"Decision 2 — item supply, with a discrepancy uncovered and corrected" |
| **15** | An M4g probe receipt's `nll_inversion_status` block asserted the 0/40 inversion was ladder-wide | Written before the MAJOR CORRECTION (§2.2 above) landed, inheriting the false premise from the rung's own task brief. The block was renamed `nll_harm_accounting` and its prose corrected; **no measured value, count, statistic, or gate was touched**, and the probe was not re-run | ADR-024 §"M4g outcome... fuse is not the root cause", "Receipt prose amended after the draw, disclosed" |
| **16** | The coordinator's own framing of M4h Stage 1's 2W/0L accuracy result as "the best we've seen" | The implementer's framing — "a clean 2W/0L sweep that cannot clear α is what a power-limited design produces — it is not encouraging" — was judged more accurate and adopted in place of the coordinator's | ADR-024 §"M4h Stage 1 OUTCOME... Framing correction to my own earlier note" |
| **17** | research/042 proposed the 8-identical-rows repeated-slot-injection hypothesis as a candidate explanation for the ladder's nulls | Its own proposed check (zero-vector control under overwrite: 8 literal zero rows, harmless, NLL 0.025 nats from baseline) falsifies the hypothesis it was designed to test — recorded as research/042 correctly identifying the decisive check, not as an error in that document | ADR-024 §"MAJOR CORRECTION", point 1 |

**Two 041 items promoted from "open provenance item" to correction-log entry, per this document's own
task brief** (the brief specifically named these two as cross-document corrections that belong in the
log, not the open-items appendix): 041's §7 items 1 (ADR-028's slot-count self-contradiction) and 4
(research/032's C2C "continuous... lockstep" mischaracterization, corrected only by forward citation in
038/039) are **already present in the numbered list above as items 9 and 7 respectively** (041's own
correction-log table, not its §7) — re-checked this session and confirmed 041 in fact filed the
slot-count item in its §7 (open items) *and* referenced a related M4e correction in its §3 table (item
7) separately. The C2C mischaracterization specifically (research/032's "continuous... in lockstep")
is filed in 041 only under §7 item 4, not §3 — **this document promotes it to the correction log
explicitly, as item 7's twin**, since research/039's finding (C2C fuses once, at prefill) is the same
underlying correction 041's own item 7 already logs for the *ladder's* premise; 041 simply never
cross-referenced that its own §7 item 4 and §3 item 7 are the same correction viewed from two
documents. Recorded here, not resolved by editing 041.

---

## 4. Not established — live hypotheses, ranked

| Hypothesis | Status | What it would need | Why it survives / where it stands |
|---|---|---|---|
| **PC1 — mechanics liveness (positive control)** | **Registered 2026-08-29, unrun, gates M5X and M4b** | One ~7–15 minute GPU draw: inject the receiver's own gold-teacher-forced per-token block-19 state back into itself via identity transform, under current mechanics (fuse, de-pooled, `<|fim_pad|>`, L18→L14), scored with S1a's original 40-item sign test | **This is the highest-priority unresolved item in the entire corpus.** No positive control has run under the mechanics used since M4c (fuse, de-pooled). S1a's own PASS (p=0.03125) ran under overwrite+pooled+the old test — mechanics nothing since M4c uses. A FAIL here would mean every null since S1a needs a caveat that current mechanics were never shown capable of carrying signal even from a model to itself, would reopen ADR-024's own "adapters improved, so discordance fell" explanation with an unruled-out rival (mechanics suppress discordance mechanically, independent of adapter quality), and would require footnoting S1a's PASS to scope it to retired mechanics. A PASS licenses **liveness only, never transfer** — this firewall is stated explicitly in research/045 §3 and must be reused verbatim in any write-up. (`primary`, research/045, ADR-024 §"PC1 PRE-REGISTRATION".) |
| **M4i — ordinary-token injection site** | **Registered 2026-08-29, unrun** | One registered e-process draw (per ADR-036, superseding the mid-p-McNemar-primary language in M4i's own original registration): reuse M3's already-trained on-manifold adapter, deliver by fuse, inject onto 8 already-present ordinary tokens instead of 8 `<\|fim_pad\|>` copies | The base Qwen2.5 technical report never mentions FIM; FIM training is Coder-specific continued pretraining per the Qwen2.5-Coder report. The receiver is non-Coder Instruct, so `<\|fim_pad\|>` (id 151662) is plausibly experientially near-vacant. No surveyed method injects at a placeholder token. Supporting literature is two abstract-only citations, graded weaker than research/044's C2C ablation. A PASS would mean the ladder's inertness was substantially an artifact of an untrained embedding region; a NULL demotes this to real-but-non-load-bearing and points back to layer coverage and receiver scale. (`primary`, research/043; ADR-024 §"M4i PRE-REGISTRATION".) |
| **Layer coverage / the conjunction hypothesis (M5X)** | **Fully pre-registered (ADR-037), unrun — needs `FuseMany` engineering (not yet implemented) and a fresh L24→L19 adapter** | A deliberate four-factor rung: multi-layer (both depth pairs simultaneously) + de-pooled payload + fuse delivery + task-loss training, evaluated once under the e-process. Explicitly excludes M4i's axis and continuous injection. ~1.3–1.4 GPU-h | The strongest, best-evidenced remaining single hypothesis (§2.4) — but a PASS cannot attribute the result to any one of its four combined factors, and this ADR pre-registers the exact decomposition ladder (drop one factor at a time, reverse order) a PASS would trigger, not built yet. A NULL would mean every named single-axis structural difference — task loss, injection operator, pooling, and now layer coverage — has been refuted individually **and** in this specific conjunction, leaving only receiver scale (M4b) and continuous+bidirectional delivery (untested even alone) as live. (`primary`, ADR-037 in full.) |
| **Receiver scale (M4b)** | **Fully pre-registered (ADR-035), unrun — queued behind the GPU lane** | 3B-self-pair (Qwen2.5-3B as both sender and receiver), reconstruction-trained M3-shaped adapter, fresh capture/calibration required. e-process now **primary** (amended by ADR-036, reversing ADR-035's original "optional" framing), old 40-item draw retained as secondary | The one contingency registered as mandatory regardless of every other rung's outcome. Independently corroborated twice (a >1.7B threshold paper; Bicameral's own GSM8K degradation when coupled models are close in scale). **Explicitly pre-registered as ambiguous even on a PASS**: a same-checkpoint self-pair PASS could mean "scale fixes transfer" or merely "same-checkpoint pairs transfer more easily than cross-checkpoint pairs" — ADR-035 pre-commits that a PASS here licenses only the narrower claim. A NULL removes the ladder's most-cited confound outright. (`primary`, ADR-035 in full; ADR-036's amendment.) |
| **Continuous injection (M4e)** | **Registered, deprioritized, unscheduled** | New engineering; blocked until M4g (fuse) landed — now unblocked, but still not scheduled | 041's correction (research/039: only Bicameral is confirmed continuous, and it is also bidirectional and 4-layer) still stands and is now reinforced by research/044's finding that Bicameral's continuity was never a clean single-axis instance of anything — it is multi-site *and* continuous *and* bidirectional. M5X deliberately excludes this axis rather than folding it in (§4 above), naming it as the natural next escalation only if M5X nulls. |
| **Bidirectional/multi-round exchange** | **Deprioritized, likely out of run 2 entirely** | A wholesale new evaluation protocol — the frozen 40-item probe and the e-process alike assume a fixed one-shot injected vector; neither can score a protocol where the sender's state also changes | Unchanged from 041. No new evidence since. |

**The run has not concluded.** Per ADR-036's own explicit ruling: the e-process is now the default
primary statistic for every ungoverned successor rung (M4i, M4h Stage 2, M4b by amendment, M5X, and
any future rung) unless a rung's own pre-registration opts out with its own justification. **This is a
protocol replacement, not a re-run** — no completed rung (M3 through M4h Stage 1) may be re-drawn
under it, and any future write-up comparing eras must state which protocol produced which result in
the same sentence as any aggregate claim (§5 below; ADR-036 §"Decision 3").

---

## 5. Scope limits that must travel with every claim above

041's five bullets stand essentially unchanged, with one line updated:

- **A sub-1.7B receiver.** Unchanged — M4b is still unrun.
- **GSM8K**, with the same two independent low-discordance mechanisms 041 named. Unchanged.
- **One model pair** (Qwen2.5-3B→1.5B). Unchanged — M4b's 3B-self-pair, when it runs, will not by
  itself license a cross-architecture claim even on a PASS (§4 above).
- **Injection site(s).** No longer "one injection site" without qualification — M5X (unrun) will be
  the first rung to use both depth pairs simultaneously; every *completed* rung remains L18→L14 only,
  and the anchor cell (L24→L19) was tested in run 1 only, never in any completed run-2 rung.
- **Payload shape — updated from 041.** 041 scoped every claim to "pooled payloads throughout... itself
  untested as a variable until M4h." **This is now stale**: pooling was tested (M4h Stage 1) and
  refuted as sole cause of the on-manifold family's inertness (§2.3). The corrected scope: every
  *completed* rung except M4h Stage 1 used pooled payloads; M4h Stage 1 used a de-pooled,
  single-broadcast-vector payload; no completed rung has used the multi-vector, per-slot payload every
  externally successful method actually uses (M4h Stage 2, unrun, is the first rung that would).
- **A 40-item probe with a documented power floor** — for every *completed* rung (through M4h Stage
  1). Successor rungs (M4i, M4h Stage 2, M4b, M5X) are governed by the e-process instead (§4); their
  eventual results carry the e-process's own scope limits (drawn from `adaptation-512` only, in fixed
  index order, per ADR-036), not the fixed-40 protocol's.
- **New scope limit, not present in 041**: **every claim above about the on-manifold family's
  inertness (as opposed to the off-manifold family's harm) is additionally scoped to mechanics never
  validated by a positive control.** PC1 has not run. If it fails, every null since S1a — including
  several this document reports as "established" in §2 — requires the caveat named in §4's PC1 row.

---

## 6. Figure/table list for a future results write-up

041's ten rows (see [041 §6](041-run2-synthesis-skeleton.md#6-figuretable-list-for-a-future-results-write-up))
all remain individually valid and are not reproduced here. This document adds:

| # | Figure/table | Content | Receipt(s) supplying it |
|---|---|---|---|
| 11 | Two-family NLL table (corrected) | The 8-row on-manifold-vs-off-manifold NLL comparison in §2.2 above, replacing any single "nothing transferred" framing | ADR-024 §"MAJOR CORRECTION"; per-rung outcome sections |
| 12 | Instrument power collapse | n_disc trajectory across the ladder's most recent draws (M4d=7 → M4g=3 → M4h-S1=2) against the minimum-attainable-p floor curve, illustrating "the better our adapters get, the blinder the probe becomes" | ADR-024 §"M4h Stage 1 OUTCOME"; ADR-036 §Context |
| 13 | Layer-coverage cross-method comparator | C2C single-layer ablation (58.42%→58.45-58.52%) vs. LatentMAS (all layers) vs. Bicameral (4 layers, 890-config sweep) vs. LatentMesh (1 layer, inert) | research/044 §2.4 table |
| 14 | M4c/M4d/M4g cross-rung ablation table | loss/deployment/operator held constant pairwise, one factor changed per column, all outcome metrics aligned in one table | ADR-024 §"M4g outcome", "Cross-rung comparison" table |
| 15 | PC1 design and decision tree | The floor/measurement/ceiling three-part control structure (ROME-style), applied to this stack, with the pre-committed PASS/FAIL branches | research/045 §1, §5 |
| 16 | Two-era comparability panel | Fixed-40-item sign-test results (through M4h Stage 1) vs. e-process wealth trajectories (M4i onward), presented side by side per ADR-036's explicit non-equivalence rule | ADR-036 §"Decision 3" |

---

## 7. Open provenance/consistency items

Hunted for across the full corpus read for this document and 041 together, per the assignment's
standing instruction to look across documents rather than within any single one.

1. **A draw-count arithmetic discrepancy in 041 §2.5 itself, found while recomputing §2.5 above for
   this document.** 041 states "Of the 10 valid cross-model draws through M4 (S2b×4, M3×2, M4×3...),
   6 landed at n_disc∈{3,4}," then states that adding M4c and M4d "brings the total to 12 valid draws,
   8 of which were structurally capable of detecting an effect and did not, 4 of which (all pre-M4c)
   could not have rejected." Recomputing directly from each rung's own receipted n_disc, as quoted
   verbatim in ADR-024's own per-rung annotations (S2b: 3,3,4,3; M3: 4,3; M4: 4,4,5): S2b×4+M3×2+M4×3
   is **9** draws, not 10, of which **8** (not 6) sit at n_disc∈{3,4} and only M4 r256 (n_disc=5)
   clears the floor. Adding M4c and M4d gives **11** valid draws (not 12), of which **8** are
   floor-incapable and **3** (r256, M4c, M4d) are floor-capable-and-null — the inverse of 041's own
   "8 capable / 4 incapable" split. **Not resolved here** — this document's own §2.5 uses the
   directly-recomputed 13-draw total (adding M4g and M4h-S1) rather than either of 041's two
   internally-inconsistent counts, but the underlying raw JSON receipts were not independently
   re-summed this session (only ADR-024's own prose annotations were read), so this is disclosed as a
   found discrepancy requiring receipt-level reconciliation, not adjudicated as which number is
   correct. This is the same "numbers restated in prose drift from numbers in receipts" pattern
   ADR-024's own §"CLARIFICATION" section flags for the permutation-null count (item 2 below) — a
   third instance of the same pattern in this corpus, worth escalating as a recurring risk rather than
   three unrelated slips.
2. **The permutation-null count (041's own item 11) has a further wrinkle 041 didn't have access to.**
   041 corrected an inherited "0/160" claim to the receipted "0/80." ADR-024's own later
   "CLARIFICATION" section (2026-08-29, after 041 was authored) explains where "160" actually comes
   from: the adversarial verifier ran an *independent* re-verification with 20 *fresh* seeds per cell,
   reported in its verdict but not in the primary receipt — so **80 is correct for the primary
   receipt, and 160 is correct "across the original run and the independent re-run,"** and both
   framings are true statements about different things, not a contradiction. 041's correction stands
   but is incomplete without this addendum. (`primary`, ADR-024 §"CLARIFICATION — the permutation-null
   count".)
3. **research/032's C2C mischaracterization and 041's own §7 item 4 are the same finding, filed in two
   places without cross-reference — flagged, not an error, but worth fixing in any future edit.** See
   §3 above; this document promotes the correction into its own log while noting 041 never linked its
   §3 item 7 (M4e's premise) to its own §7 item 4 (research/032's characterization) despite both
   describing the identical underlying fact (C2C fuses once, at prefill, not continuously).
4. **No numeric inconsistency was found between ADR-035, ADR-036, and ADR-037's shared cost-derivation
   inputs.** All three cite the identical receipted `wall_clock_s` figures for M4c's training
   (1603.48s) and probe (422.48s) and M4h Stage 1's probe (395s), and derive the same ≈10–10.5s/item
   e-process rate and ≈0.88 GPU-h N_max budget independently. Stated explicitly, as 041 §7 item 5 did,
   because a clean cross-check is itself worth recording, not silently omitted for not being a finding.
5. **ADR-035's citation of "docs/research/024 §2, §9 risk 10" was checked against the possibility that
   it is a typo for ADR-024 (this repository's ADR and research numbering both use "024," for different
   documents) and found to be correct as written** — `docs/research/024-live-latent-experiment-design.md`
   is a real, distinct document (the run-1 live-experiment design, cited three times in ADR-035 for its
   own VRAM-budget arithmetic). Recorded because the shared "024" digit across the ADR and research
   sequences is exactly the kind of numbering collision ADR-035's own header flags for its own number
   ("035" is shared between this ADR and an unrelated research document) — checked here specifically
   so it is not silently assumed to be an error in a future read.
6. **041's own five open items (§7) are not re-litigated here** — the ADR-028 slot-count contradiction
   remains unadjudicated (M5X's own author explicitly declines to adjudicate it generally, ruling only
   narrowly for M5X's own 4+4 split, per ADR-037 §"Slot-count ruling"); the autogenous/cognitum-slack
   attribution dispute and the GPU-hour reconciliation gap are untouched by anything read for this
   document.

---

## Sources

Every document listed in this header's "Corpus read for this document" line, plus every document 041
cites (041 itself, read in full, since this document's every claim about what 041 said or didn't say
is checked against 041's actual text, not a memory of a prior summary of it). No number above is
retyped from memory of a prior summary of a prior summary — each was re-read from its cited file this
session.
