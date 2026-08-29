# 045. The positive control this harness does not have

* **Purpose**: after twelve-plus null draws (ADR-032's own accounting) every result is ambiguous
  between "no transfer effect exists" and "this harness cannot detect an effect under current
  mechanics." Power analysis (`docs/research/031`, ADR-036's e-process) answers whether the
  *statistics* could detect an effect; it says nothing about whether the *injection pathway*
  itself can move the receiver's answers under the mechanics M4c onward actually use (fuse
  delivery, de-pooled per-token payloads, `<|fim_pad|>` slots, the e-process). This document
  surveys how the interpretability/steering literature validates an intervention pipeline before
  trusting a null, designs concrete positive controls runnable on this stack, and recommends one
  to run before M4b/M5X.
* **Date**: 2026-08-29 (branch `feat/run2-thought-adapter`, read-only except this file — not
  committed).
* **Method / evidence grading**: WebFetch of primary sources (arXiv abstracts + ar5iv full-text
  renders) plus direct repo/receipt reading. **primary** = paper text fetched and the specific
  claim read in the fetched text. **secondary** = abstract/metadata only, claim not independently
  verified in body text. **inferred** = this document's own reasoning from confirmed facts.
  WebSearch was unavailable for this pass (session budget exhausted by concurrent agents before
  this task started); WebFetch against known arXiv/ar5iv URLs was used throughout instead and is
  graded identically to a WebSearch-sourced fetch — grading is about what was read, not which tool
  read it.
* **Read first**: ADR-023 (S1a — the ladder's only PASS, and the mechanics this document asks
  whether the current harness still reproduces), ADR-024's MAJOR CORRECTION and M4i sections
  (what changed since S1a), `docs/research/031` (power-floor arithmetic), ADR-036 (the e-process
  this document's cost estimates reuse), ADR-035/037 (M4b/M5X — the two queued rungs this
  document's recommendation gates).

---

## Answer, up front

**Yes, the literature treats this as standard practice, though inconsistently applied even in
canonical work** — the single clearest precedent is ROME's own three-run causal-tracing design
(clean run = ceiling, corrupted run = floor, corrupted-with-restoration = the positive control
that validates the pathway before any partial result is trusted) (§1). This repository already
has one instance of exactly this pattern — S1a — but it was frozen under mechanics (overwrite,
pooled, `<|fim_pad|>`, the old 40-item sign test) that nothing since M4c actually uses. **No
positive control has ever been run under the mechanics the ladder currently uses to produce
nulls.** The recommended fix (§5) is close to free: inject the receiver's own gold-teacher-forced
per-token state (already captured on disk, zero new GPU work beyond the injection+scoring pass)
back into itself, under M4b/M5X's actual frozen mechanics (fuse, de-pooled, `<|fim_pad|>`, L18→L14
depth), scored with S1a's original 40-item sign test for direct comparability. Estimated cost:
well under 10 minutes GPU time, no training, no new capture — the cheapest item on the entire
run-2 cost ledger. **Recommendation: run it before M4b and before M5X**, both of which cost
1-2 GPU-hours each and would be uninterpretable nulls if the mechanics themselves turn out to be
inert.

---

## 1. What makes a good positive control here — literature survey (RQ1)

### 1.1 ROME's three-run causal tracing: the canonical instance of exactly this pattern

Meng et al. 2022, "Locating and Editing Factual Associations in GPT" (arXiv:2202.05262, primary,
fetched via ar5iv full text) structures every causal-tracing experiment as three runs, not two:

- **Clean run** — the model processes the true factual prompt and outputs the correct answer.
  This is the ceiling.
- **Corrupted run** — the subject tokens are noised so the model outputs an incorrect answer.
  This is the floor.
- **Corrupted-with-restoration** — the corrupted run, except a handful of hidden states at one
  layer/token are forcibly overwritten with the clean run's activations at that location; "future
  computations execute without further intervention."

The paper's own framing of why this third run exists is exactly the positive-control argument
this task asks about, quoted directly: **"the ability of a few clean states to recover the correct
fact, despite many other states being corrupted by the obfuscated subject, will indicate their
causal importance in the computation graph."** The clean run is not merely a reference point
plotted alongside the results — it is the ceiling that restoration is judged against (Indirect
Effect = P(correct | restoration) − P(correct | corrupted)), and no claim about *where* the fact
lives is trusted until this ceiling is shown to be reachable by *some* restoration set. That
"ability to recover" check is the positive control: it demonstrates the causal pathway (noised
embedding → forward pass → output) can be steered by injected activations at all, before any
claim is made about which specific layer matters most.

### 1.2 Zhang & Nanda 2024, "Best Practices for Activation Patching" — the field names the same
gap this document exists to close

arXiv:2404.15255 (primary for the denoising/noising distinction, fetched via ar5iv; the
sanity-check section was not found verbatim in the fetched excerpt, graded secondary for that
specific claim) formalizes ROME's two directions as **denoising** (patch clean→corrupted, "which
patch restores the clean-prompt behaviour") and **noising** (patch corrupted→clean, "which patch
breaks the clean-prompt behaviour"), and states explicitly that **"these two directions can be
very different, and are not just symmetric mirrors of each other."** Load-bearing for RQ4 here:
the paper flags that negative/ambiguous patching results are an **"unsolved problem"** in the
field — components whose noising *increases* performance ("negative components") make null
results structurally hard to interpret without independent validation that the patching mechanism
itself is live. This is the same asymmetry this repository has already found empirically (ADR-024's
MAJOR CORRECTION: on-manifold payloads are inert, off-manifold payloads are actively harmful,
and neither result alone tells you the pathway is *capable* of helping).

### 1.3 Inference-Time Intervention (Li et al. 2023) — two-tier validation, and the gaming failure
mode this task's RQ3 warns about

arXiv:2306.03341 (primary, ar5iv full text) validates its truthfulness-steering intervention with
**two separate controls, not one**:

- **A floor control**: random intervention directions score 31.2% (true*informative) versus 42.3%
  for the probe-derived (mass-mean-shift) direction (their Table 3) — demonstrating the
  intervention's effect is attributable to the *direction*, not merely to perturbing activations
  at all.
- **A ceiling reference**: linear-probe validation accuracy on held-out truthfulness labels (top
  single attention head: 83.3%) — an upper bound on how much truthfulness information is even
  linearly present to intervene on, reported *before* the main steering result, so a reader can
  judge how much headroom the intervention could plausibly have captured.

ITI also documents, directly relevant to RQ3's "honest but strong" constraint, the failure mode of
an over-strong positive control: at high intervention strength α the truthfulness score keeps
rising toward a ceiling, but the paper flags this rise as **partly illusory** — "it is trivial to
attain a perfect truthfulness score simply by answering 'no comment'" — i.e., an intervention can
look like it is "restoring the right answer" while actually just suppressing the model's willingness
to answer at all. This is the literature's own worked example of a positive control that passes
for the wrong reason, and it is the exact failure mode §3 below rules out for each candidate here.

### 1.4 Causal abstraction / DAS (Geiger et al.) — the contrasting, weaker-validated case

arXiv:2303.02536 (primary, ar5iv full text) validates Distributed Alignment Search with a
**floor-only** control: DAS applied to a frozen, randomly initialized network achieves only
chance accuracy at small hidden dimensions and cannot "construct entirely new behaviors from
random structure." Notably — and this is the contrast worth naming for RQ1 — **the paper does not
report a ceiling/oracle validation** (e.g., confirming DAS recovers ground truth when interchange
interventions are constructed directly from known task-relevant variables before reporting partial
or negative alignment results elsewhere). This is evidence that even well-cited interpretability
methodology is inconsistent about running both halves of the control (floor *and* ceiling); it is
not evidence that skipping the ceiling half is best practice. ROME and ITI both run both halves;
DAS runs one. This repository's own zero-vector gate (ADR-023's A7(c)) is already a floor control
in this sense — a positive/ceiling control is the missing half, precisely the gap this task names.

### 1.5 What this establishes for RQ1

The pattern across all three well-instrumented cases (ROME, Zhang & Nanda, ITI) is the same
three-part structure this repository already half-has: **a floor** (corrupted/random/zero — this
repo has this, ADR-023 A7(c)), **the measurement under test** (aligned cross-model payload — this
repo has this, it is the entire M3-M4i history), **and a ceiling that proves the pathway can carry
signal at all** (clean restoration / probe-derived direction / oracle labels — **this repo does
not have this under current mechanics**, and S1a is the only historical instance, frozen under
mechanics nothing since M4c uses).

---

## 2. Candidate positive controls runnable on this stack (RQ2)

All four candidates below are **inference-only or near-zero-training** and reuse artifacts already
on disk. Costs are read from receipted rates, not estimated from scratch: M4c task-loss training
`wall_clock_s = 1603.48 s` (≈0.446 GPU-h, ADR-035/037), one frozen 40-item sign-test draw
`wall_clock_s = 422.48 s` (≈7 min, ADR-035/036/037), e-process full budget to `N_max≈300` ≈3,150 s
≈**0.88 GPU-h** (ADR-036).

| # | Candidate | Mechanism | Training? | New capture? | Cost | What it proves | What it does NOT prove |
|---|---|---|---|---|---|---|---|
| **(c)** | **Gold-teacher-forced self-pair oracle** | Inject the receiver's own per-token block-19 state, captured while teacher-forced on the *gold* solution text for the same item, into a fresh (non-teacher-forced) generation of that item. Identity transform (self-pair, no alignment needed). | **No** | **No** — `receiver_L14.tok.f32bin` / `receiver_L19.tok.f32bin` already exist on disk, teacher-forced over `calibration-4000` (`run2-pertoken-dump-receipt.json`, verified this session: 719,115 tokens each, dim 1536) | **≈7 min** (one 40-item draw, no training, no dump) — the cheapest candidate | Maximal ceiling: if the injection pathway (fuse, de-pooled, current site, current norm-rescale) is mechanically capable of moving the receiver's answer *at all* when the injected content is literally the correct-answer trajectory, this will show it. Directly analogous to ROME's clean-run restoration. | **Nothing about cross-model transfer.** The content here was never produced by an alignment transform, never crossed models, and is definitionally the easiest possible payload — this is a liveness/plumbing test, not a capability test, and must be labeled as such in any write-up (see §3). |
| **(a)** | **Self-pair, receiver's own *natural* (non-gold) later-layer state, same item** | Same as (c) mechanically, but the captured state comes from the receiver's own unforced generation of the item (correct or not), not a gold-teacher-forced pass. | No | **Yes** — no such capture exists yet (existing dumps are teacher-forced-only, `run2-pertoken-dump-receipt.json design` field: "teacher-forced prefill only, no generation") | ≈7 min draw + one new generation-capture pass (order of minutes, no receipted precedent to cite directly — smallest addition on the ladder) | A weaker, more honest ceiling than (c): tests whether the receiver can use its *own* natural reasoning trajectory re-injected into itself, without smuggling in the gold answer. Closer analogue to ITI's probe-derived (not label-derived) direction. | Still not a transfer test — self-pair only. Slightly less informative than (c) as a pure liveness test, since a natural trajectory of an item the receiver got *wrong* would carry no correct-answer signal to recover; would need item-correctness stratification to interpret cleanly, adding scope M4b's own registration explicitly avoided ("new measurement work with its own leakage and pre-registration requirements," ADR-036). |
| **(d)** | **Re-run S1a's exact identity-transform self-pair probe, swapping in ONLY the current-mechanics axes one at a time** (overwrite→fuse; pooled→de-pooled; keep `<|fim_pad|>` fixed, since M5X explicitly excludes the ordinary-token axis) | Same self-pair identity mechanism as S1a | No | Depends on ablation depth — de-pooled dumps already exist; a fuse-only variant needs zero new capture | ≈7 min per ablation cell (2-3 cells to isolate which axis broke S1a, if any) | **Diagnostic, not just confirmatory**: if the combined control (c) fails, this isolates *which* mechanics change (fuse vs de-pooling vs both) is responsible, rather than leaving "the current harness" as an undifferentiated black box. | Same self-pair-only scope limit as (a)/(c). A multi-cell ablation reopens (in miniature) the multiplicity concerns `docs/research/031` §4 already raised for the ladder — worth Holm-correcting if more than one cell is run and reported jointly. |
| **(b)** | **Within-model oracle: inject receiver's own state from an item it answered correctly into a run of the same item where it (under some perturbation) answered incorrectly** | Requires first stratifying items by correct/incorrect under two decoding conditions (e.g., greedy vs. sampled, or truncated CoT budget) to find flip pairs | No | **Yes**, and it is the expensive kind — a full item-stratification pass across `calibration-4000` under two decode conditions before any injection experiment can even be designed | Stratification pass: order of a full 4,000-item inference sweep (no receipted precedent — likely the single most expensive candidate here, plausibly comparable to or exceeding M4b's ≈1-2 GPU-h just for the stratification step) | The most "naturally occurring" oracle — the injected content is a state the model actually produced while succeeding, not an artificially derived gold-answer state, which sits closer to (a) than (c) on the honesty axis while still being a real ceiling. | Same self-pair-only scope limit. Materially more expensive and more complex to pre-register cleanly (what counts as a "flip pair," how many are needed for the frozen 40-item test) than (a)/(c)/(d) — this is the candidate most likely to introduce exactly the kind of ad hoc, post-hoc-flavored item selection ADR-036 already flagged as a live risk when it rejected stratified sampling as a *primary* route for the same reason. |

**Ranking by strength × cost**: (c) > (d) > (a) > (b). (c) is both the cheapest and the strongest
ceiling; it should run first. (d)'s ablation cells are the natural follow-up if (c) fails, to
localize the cause. (a) and (b) add real information (a more honest, "natural" ceiling) but at
higher cost and, for (b), meaningfully higher design risk — neither is needed to answer the
binary question this task exists to resolve ("is the current mechanics pathway alive at all"),
so both are named here as available escalations, not part of the primary recommendation.

---

## 3. Separating the positive control from the transfer claim (RQ3)

Every candidate above shares one structural property that must be stated plainly in any write-up
that uses one: **all four are same-model, same-item constructions.** None of them cross model
checkpoints, none of them pass information through a trained or fitted alignment (identity
transform only), and (c)/(a) in particular hand the receiver a state that already, by construction,
encodes the item's own solution trajectory (gold-derived for (c), self-generated for (a)/(b)).

Concretely, the separation each write-up must carry:

- **A PASS on any of these controls licenses exactly one claim: "the injection pathway, under the
  mechanics tested, is capable of moving the receiver's output when the injected content already
  contains usable task signal."** It does **not** license "cross-model transfer works," "the
  adapter architecture is sound," or "M4b/M5X's eventual result is more likely to be a real
  effect." M4b's own pre-registration already models this discipline correctly for a different
  axis (a same-checkpoint 3B self-pair PASS "does not, by itself, license the claim" that receiver
  scale fixes transfer, ADR-035) — the same sentence structure applies here, with "mechanics
  liveness" substituted for "receiver scale."
- **Guard against the ITI failure mode (§1.3):** a positive control that passes because the
  injection collapses the receiver toward a degenerate, easy-to-get-right output (e.g., always
  emitting the gold answer's exact token sequence, or collapsing entropy so hard the model just
  echoes back injected content) is not evidence the pathway is *usefully* live — it is evidence the
  injection can overwhelm the receiver's own computation. This repository already has the
  instrument to check for this: ADR-024's own NLL-inversion diagnostic (aligned NLL vs. baseline
  NLL) and the manifold-precheck's item-invariance/entropy metrics (`docs/research/036`'s cosine,
  entropy, cross-item-invariance triple) should be run on **every** positive-control draw, not
  skipped because the control is "supposed to" pass — a PASS with NLL collapsed to near-zero and
  entropy collapsed toward the injected content's own decoded tokens is the ITI "no comment"
  pattern in this repo's own vocabulary, and should be reported as a gamed pass, not a clean one.
- **Explicit non-claim, stated once and reused verbatim in every rung's write-up**: "This control
  used a same-model, same-item, identity-transform injection with gold-adjacent content. It tests
  whether this repository's injection mechanics (delivery operator, payload shape, injection site,
  norm-rescale) are capable of carrying signal at all. It does not test, and must not be cited as
  evidence for or against, whether a cross-model-derived, learned-alignment payload can transfer
  reasoning content — that is exactly the separate question M3 through M5X exist to answer."

---

## 4. What a FAILING positive control would mean (RQ4) — unsparing

If even candidate (c) — the maximal ceiling, gold-derived, same-model, identity-transform, under
the exact mechanics M4b and M5X are about to spend 1-2 GPU-hours each on — fails to move the
receiver's answers above chance/baseline, the implications are severe and specific, not a vague
"the project needs more work":

1. **Every cross-model null since S1a becomes uninterpretable as evidence about transfer**, because
   there is no longer any demonstration that *this specific mechanics configuration* (fuse,
   de-pooled, `<|fim_pad|>`, current norm-rescale, current site) can carry signal under any
   condition, including the easiest one constructible. S2b (4 draws), M3 (2), M4 (3), M4c/M4d/M4g,
   M4h Stage 1, and M4i would all need a caveat reading approximately: "this null was produced
   under mechanics never independently confirmed to be capable of transmitting a signal, even in
   the maximally favorable same-model case."
2. **ADR-024's MAJOR CORRECTION would need to be reopened, not just caveated.** Its central causal
   story — "discordance *falls* as adapters improve... the better our adapters get, the blinder
   the probe becomes" — currently has exactly one explanation on record (adapter quality). A
   failing positive control adds a second, mechanically prior explanation the correction does not
   currently rule out: **the fuse+de-pooled+current-site combination itself suppresses discordance**,
   independent of adapter quality, because the injection is inert *before* any adapter-specific
   content is even considered. These two explanations are not currently distinguished anywhere in
   the ladder's write-ups, and a failing (c) is exactly the evidence that would force the
   distinction.
3. **ADR-023's S1a PASS itself would need a footnote**, not a retraction — S1a's own result (self-
   pair, identity transform, p=0.03125, mid-p 0.0156) stands as a valid, adversarially-verified
   finding *under the mechanics it actually used* (overwrite, pooled, `<|fim_pad|>`, the original
   40-item sign test). But it would no longer be citable, as it currently implicitly is throughout
   ADR-024, as evidence that "the injection mechanism can transmit information" in the present
   tense — it would need to read "could, as of 2026-08-28, under overwrite+pooled mechanics; not
   re-confirmed under the fuse+de-pooled mechanics used since M4c."
4. **M4b and M5X should not be reported on their own terms until this is resolved.** Both are
   pre-registered to test real, distinct hypotheses (receiver-scale threshold; conjunction of
   factors) — but a null on either, reported today, inherits the same ambiguity this entire task
   exists to name. Running them before resolving liveness would mean spending ≈1-2 and ≈1.3-1.4
   GPU-hours respectively on results that a subsequent failing positive control could retroactively
   invalidate.
5. **What a failing positive control would NOT mean**: it would not mean LatentMesh's injection
   *code* is buggy in the S0-mechanics sense (shape mismatches, non-finite values, wrong layer
   indices) — those integrity gates (A7(a), G6, `hand_rolled_apply_matches_align_crate`) are
   checked independently on every rung and have passed throughout. A failing (c) would isolate the
   problem to the *combination* of delivery operator, payload shape, injection site, and
   norm-rescale setting — i.e., a real finding about which of ADR-023's frozen "injection
   semantics" choices stopped being sufficient once other axes moved, not a claim that nothing in
   the runtime works.

---

## 5. Recommendation (RQ5)

**Run candidate (c) — the gold-teacher-forced self-pair oracle — first, before M4b and before
M5X.**

Reasoning, weighed directly against the two queued rungs:

- **Cost asymmetry is stark.** (c) costs ≈7 minutes of GPU time with zero new training and zero
  new capture (the per-token dumps it needs already exist, receipted, on disk). M4b costs an
  estimated ≈1-2 GPU-h (ADR-035 §Cost) and M5X costs ≈1.3-1.4 GPU-h (ADR-037 §Cost). Spending under
  10 minutes to determine whether either multi-GPU-hour investment can even produce an
  interpretable result is a clearly dominant move under any reasonable cost-of-information
  argument.
- **It answers a strictly prior question.** M4b tests "does receiver scale fix transfer" and M5X
  tests "does the conjunction of factors fix transfer" — both presuppose the injection pathway
  itself is live under current mechanics. Neither pre-registration (ADR-035, ADR-037) currently
  contains a liveness check; both would, on a null, currently be reported as evidence about their
  named hypothesis, when a large part of a null's likely cause (per §4) has never been ruled out.
- **It is honest by construction, not by discipline alone.** Because (c) is explicitly scoped as a
  same-model, same-item, identity-transform, gold-adjacent construction (§3), there is no risk of
  it being mistaken for, or informally cited as, evidence for the transfer claim — the separation
  is structural, not just a sentence in the write-up.

### Pre-registerable spec

- **Item source**: `calibration-4000` (train-derived, per ADR-036's item-supply rule), a fresh
  40-item ChaCha8 draw, seed to be fixed at authoring time and disclosed, checked for overlap
  against S1a's original 40 items and any items already consumed by M3/M4/M4c/M4d/M4g/M4h/M4i
  probe draws (disclose, don't exclude, per ADR-023's own overlap precedent — this is a mechanics
  probe, not a scale/comparability probe, so exact non-overlap is not load-bearing the way it is
  for eval-200/holdout-100).
- **Capture**: none new — read per-token receiver states directly from
  `crates/latentmesh-runtime/target/latentmesh-runs/run2/receiver_L14.tok.f32bin` and
  `receiver_L19.tok.f32bin` (already committed via `run2-pertoken-dump-receipt.json`, teacher-forced
  over `calibration-4000`, dim 1536, verified this session).
- **Injection mechanics**: identity transform (self-pair, receiver→receiver, no alignment fit
  needed); delivery = **fuse** (M4g's operator, matching M4b/M5X's actual mechanics, not S1a's
  overwrite); payload = **de-pooled**, full per-token span (matching M4h Stage 1 onward, not
  S1a's pooled payload); site = **`<|fim_pad|>`, 8 slots, block 19** (frozen, unchanged — M5X
  explicitly excludes the ordinary-token axis pending M4i, so this control should not confound
  that open question either); norm-rescale = current default (rescale-to-natural-median, unchanged
  since ADR-023).
- **Statistic**: primary = the original frozen 40-item one-sided exact sign test with mid-p
  correction (Fagerland, Lydersen & Laake 2013, already the ladder's own citation) — chosen over
  the e-process specifically so this control is comparable to S1a's own PASS on identical footing,
  isolating mechanics changes as the only variable. The e-process (ADR-036) may be reported as
  secondary if the sign-test result is ambiguous (n_disc ≤ 4, per `docs/research/031`'s power
  floor).
- **Integrity/gaming gates, mandatory before the primary statistic is trusted (per §3)**: NLL
  inversion check (aligned vs. baseline, per ADR-024's own diagnostic), item-invariance/entropy
  check (`docs/research/036`'s triple), and a manual read of 3-5 transcripts checking for the ITI
  "no comment" pattern (degenerate echoing of injected content rather than genuine task
  completion).
- **Pre-committed outcome rule**: PASS (p<0.05, gates clean, no gaming signature) → mechanics are
  confirmed live under current conditions; M4b and M5X proceed as separately registered, and every
  null since S1a can be read as evidence about transfer specifically, not plumbing. FAIL or
  gated-out (gaming signature present) → M4b and M5X are **held**, not run, until candidate (d)'s
  ablation cells (fuse-only, de-pooled-only) localize which mechanics axis broke liveness; the
  finding is written up per ADR-032's negative-result contract with the four caveats in §4 above
  attached to every affected prior rung.

**Estimated total cost of the recommended sequence**: ≈7-15 minutes GPU time for (c) alone;
≈20-30 minutes if the FAIL branch's 2-3 ablation cells from (d) are needed. Either outcome is
resolved before M4b or M5X would have consumed their first GPU-hour.
