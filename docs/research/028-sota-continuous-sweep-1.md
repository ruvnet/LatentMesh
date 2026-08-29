# Research: continuous SOTA re-sweep, post-M3-null (sweep #1)

* Purpose: updates `docs/research/023-beyond-sota-roadmap.md` and
  `docs/research/027-global-ambient-intelligence-track.md` in light of two now-established
  adversarially-verified negative results (ADR-023 S6: linear Procrustes map, zero causal
  transfer; ADR-024 M3-outcome: trained nonlinear MLP, also zero causal transfer). Does not
  re-report ground already covered by 023/027 — reads them as prior state, not source material.
* Date: 2026-08-28/29.
* Scope: arXiv 2025-2026, weighted toward the last ~3 months; GitHub repos with working code.
  Evidence grades as in 023/027: **primary** (paper/repo fetched and the actual method/equation/
  code read), **inferred** (search-result summary only), **uncertain** (could not be verified).
* Method: WebSearch/WebFetch only, no repo writes beyond this file. Where a fetch returned only
  an abstract or a search snippet, that is stated — several sources below are graded down from
  what a first search summary implied, on purpose, rather than silently upgraded.

---

## 1. Does anyone explain or predict our specific result?

**Framing, precisely**: two of our adapters (a linear Procrustes map, ADR-023 S6; a trained
2-layer MLP, ADR-024 M3) both fit their target well by a reconstruction/similarity metric (A6
residual 0.44-0.56 for the linear map; holdout MSE 0.179 / relative residual 0.461 for the MLP,
well below the 0.843 mean-predictor floor) and both carry **zero** measurable causal effect on
the live receiver (sign-test p between 0.31 and 0.875 across six independent cell/calibration/
architecture combinations, none near α=0.05). We asked: has anyone named or predicted exactly
this split — good fit, no causal use?

**No single unifying name exists**, but three independent literatures converge on the identical
point without citing each other, which is itself the finding:

1. **Probing-classifier critique (interpretability, well-established, primary via multiple
   sources)** — "the presence of decodable information does not conclusively demonstrate it plays
   a causal role in the model's outputs... the classifier's high accuracy may indicate that the
   model encodes information that is not causally efficacious in model behavior." This is the
   generic, decades-old-by-ML-standards form of our exact split, developed for single-model probing
   (Belinkov's survey, *Probing Classifiers: Promises, Shortcomings, and Advances*, MIT Press;
   Elazar et al.'s **amnesic probing**, TACL). It transfers cleanly to our cross-model case: our
   alignment/MLP fit is structurally a probe (can we predict/reconstruct the target's hidden
   state?), and the causal gate (ADR-003's five controls) is structurally an amnesic-probing-style
   causal follow-up (does removing/substituting the signal change behavior?). The field's own
   caution on amnesic probing itself — "one has to be cautious of causal interpretations... the
   information removed in practice should be seen only as an approximation" — is a symmetric
   caution worth carrying into how our own gate's p-values are read.
2. **The representational-vs-computational/functional alignment distinction (brain-LLM alignment
   literature, primary via search synthesis of ICLR 2026 Re-Align workshop framing)** — this
   field draws the split explicitly: "representational alignment asks whether model features
   predict [target] activity... computational [functional] alignment examines whether the
   [target] uses the same algorithms... encoding models test representational alignment,
   [only] causal work" tests the computational claim. This is precisely our A6 (representational:
   does the fit predict well) vs. A7(b) (functional/causal: does the live receiver actually use
   it) split, independently arrived at in a completely different subfield (comparing LLMs to
   brains, not to each other). No paper found applies this named distinction to cross-*model*
   (not brain) latent transfer specifically — that gap is itself worth noting as unclaimed
   framing LatentMesh could adopt for its own writeup.
3. **PRH-similarity-metric artifact critique — the single most load-bearing new finding of this
   sub-question.** "Revisiting the Platonic Representation Hypothesis: An Aristotelian View"
   (arXiv:2602.14486, **primary**, fetched) shows that the global similarity/alignment metrics
   used to support Platonic convergence claims are themselves confounded: "increasing model depth
   or width can systematically inflate representational similarity scores," and after a
   permutation-based null-calibration correction, "the apparent convergence reported by global
   spectral measures largely disappears." Only **local neighborhood similarity** (not local
   distances, not global structure) survives calibration — their proposed "Aristotelian"
   replacement for PRH. **This directly threatens the interpretability of our own A6 residual
   metric**: a 0.44-0.56 relative-residual "PASS" was never checked against a null/permutation
   baseline for what an *uncorrelated* mapping between two models of these specific depths/widths
   would score by chance. If depth/width alone inflate apparent fit the way this paper documents,
   part of A6's "PASS" could be exactly this artifact, which would make the good-fit/no-causal-use
   split even less surprising than it already is — the fit metric itself may be measuring less
   than it appears to.

**Secondary, weaker findings, each adding texture but not the headline**:

- **"Representation Alignment Rests on Linear Structure"** (arXiv:2605.28870, primary, previously
  cited in 027 §1.3, re-read here specifically for the MLP-relevance angle): Platonic alignment is
  attributed to a signal/bias/noise decomposition where the *signal* term specifically requires
  representations to encode object-attribute relationships **linearly** (per the Linear
  Representation Hypothesis). This paper does not directly test what a *nonlinear* (MLP-style) fit
  recovers when linear structure is weak or absent, and that gap was not closed by any other
  source found this pass — **this remains an open, not a resolved, question for interpreting M3**.
  It is graded here as suggestive context, not confirmation, that M3's MLP may have fit
  idiosyncratic higher-order correlations rather than exploitable structure — plausible, not
  demonstrated.
- **Steering-vector transfer literature is genuinely mixed, not uniformly negative** — a nuance
  worth stating precisely rather than cherry-picking the negative half. "Transferring Linear
  Features Across Language Models With Model Stitching" (arXiv:2506.06609, primary, fetched)
  reports affine mappings *can* transfer SAE features/probes/steering vectors "effectively," with
  cost savings from cross-size initialization — a **positive** counter-example to a blanket "linear
  transfer never works" reading, though the abstract-level fetch did not resolve whether their
  setup used same-family/scale-adjacent models or matched LatentMesh's specific S18/S24-depth,
  pooled-injection regime, so this is not a clean apples-to-apples contradiction of our result —
  flagged as a genuine open comparison, not resolved this pass. "Linear Representation
  Transferability Hypothesis" (arXiv:2506.00653, inferred, search-summary only) similarly reports
  successful small-to-large steering transfer under some conditions. Against these,
  "Understanding (Un)reliability of Steering Vectors" (OpenReview id JZiKuvIK1t) could not be
  fetched this pass (403/verification wall) — **uncertain**, named here only as a lead for a
  future pass, not used as evidence.
- **Adragna's "Against the Platonic Representation Hypothesis"** — still **uncertain**: found only
  as a talk/event page (Luma), not a resolvable arXiv preprint, in this pass or the prior one
  (027). The argument (convergence reflects shared training-data structure, not discoverable
  universal geometry) is consistent with and arguably subsumed by the Aristotelian-view paper
  above, which *is* primary-verified. Treat the Aristotelian-view paper as the load-bearing citation
  going forward; keep Adragna as an unconfirmed pointer, not a citable source.

**Bottom line for Q1**: no single paper says "regression fit without causal transfer, cross-model,
named phenomenon X" — but the amnesic-probing critique, the representational/computational-
alignment split, and the Aristotelian-PRH similarity-artifact finding are three independently-
derived statements of the same underlying caution, from interpretability, brain-alignment, and
representation-geometry subfields respectively, none of which cite each other. That convergence
is itself evidence the pattern is real and general, not an artifact of LatentMesh's specific setup.

---

## 2. Trained-projector mechanics — task loss vs. reconstruction loss (highest-priority finding)

**This is the single most actionable finding of this sweep.** M3's MLP was trained on **MSE
reconstruction loss** against captured target hidden states (ADR-024 M3 outcome section — the
frozen probe evaluates the *causal* effect separately, after training is already fixed). We
directly read the training code of every trained cross-model latent-communication method found in
this pass that reports a real positive result against heterogeneous or different-size models, and
in every case the answer was the same:

| Method | Loss type | Grade | Evidence |
|---|---|---|---|
| **Cache-to-Cache (C2C)**, arXiv:2510.03215, github.com/thu-nics/C2C | **TASK LOSS** — standard causal-LM next-token cross-entropy | **primary** — read `script/train/SFT_train.py` directly: both the baseline and "Rosetta" (fused) training paths call `outputs = model.forward(..., labels=labels); loss = outputs.loss` — a full forward pass through the frozen **target/receiver** model, with the fuser's output wired into its KV-cache, and `outputs.loss` is `AutoModelForCausalLM`'s built-in next-token loss. Gradients flow end-to-end from the receiver's own prediction loss into the fuser's weights. No separate MSE/reconstruction stage exists in the script. The README's "both source and target models remain frozen" refers to frozen *weights*, not a frozen/detached gradient path. |
| **The Bicameral Model**, arXiv:2605.11167 | **TASK LOSS** — "the sum of masked cross-entropy losses on both models' outputs" | **primary**, fetched. Confirms per-token pointwise transfer (not sequential/recurrent — see §3), same-family same-size pairs in the headline results (Qwen2.5-0.5B×2, Qwen3-0.6B×2, Qwen3-4B×2), +60.3pp on arithmetic, 1.7x on ZebraLogic. |
| **Interlat** (Du et al. 2026 / "Enabling Agents to Communicate Entirely in Latent Space"), arXiv:2511.09149, ACL 2026 | Task-loss-flavored, **inferred** only — abstract-level fetch could not resolve the exact loss equation, but describes "a fine-tuned receiver" and "genuine utilization of latent information," which is inconsistent with a frozen-receiver reconstruction-only regime. Not upgraded to primary; flagged for a follow-up fetch of the PDF body. | inferred |
| **AVP** (Agent Vector Protocol), github.com/VectorArc/avp-{spec,python} | **NO TRAINING** — same-family cross-size path is a deterministic vocabulary-mediated (logit-lens-style) projection, not a fitted network with any loss | primary (repo read) |
| **LatentMAS**, arXiv:2511.20639, github.com/Gen-Verse/LatentMAS | **NO TRAINING** — explicitly "training-free," direct use of existing hidden states, realignment toggled as a hyperparameter, not a learned map | primary (repo read) |

**Verdict: the trained-vs-untrained methods split cleanly, and within the trained methods, the
loss-type pattern is unanimous across every primary-verified source.** Every method that (a)
actually trains a network and (b) reports a real positive cross-model or cross-size result trains
that network on **task loss, end-to-end through the receiver's own forward pass** — never on
reconstruction/MSE against captured target activations in isolation. The two zero-training methods
(AVP, LatentMAS) sidestep the loss-type question entirely rather than supporting reconstruction
loss.

**Direct implication for M3/M4/M5**: M3's MLP is described as "the Cache-to-Cache-style MLP
baseline" (ADR-024 §"Ladder rungs"), but its training signal is **not** Cache-to-Cache's actual
training signal — the architecture (2-layer MLP projecting hidden-dim to hidden-dim) matches:
the loss function does not. This is a real, fixable design gap, not a confirmation that trained
nonlinear adapters can't work here. **Before M4 (FastGRNN) is read as a verdict on sequence
structure, or M5 (MicroLoRA) is scoped, this gap should be closed or explicitly controlled for**:
a rung trained end-to-end on the receiver's own next-token loss (with the receiver frozen but the
gradient path live, exactly C2C's SFT_train.py pattern) is a materially different, currently-
untested experiment from what M3 ran, and it is the single highest-value next architecture to try
before concluding these two model's representations "don't share exploitable structure at any
polynomial-order alignment" (the kill-criterion language in `docs/research/027` §4.4 Stage 1).

**Secondary detail on C2C's fuser architecture** (relevant to §3 below): the Projection → Dynamic
Weighting → Learnable Gating pipeline operates **per-layer, per-cache-position independently** —
concatenate-then-project, a per-layer Gumbel-sigmoid gate deciding whether to inject at that layer
— with no described cross-token recurrent or attention mixing inside the fuser itself. C2C's own
ablation: pure projection (discarding the receiver's cache) is much worse; residual-preserving
fusion +24.18pp; adding the per-layer gate +3.07pp further. The big wins in C2C come from **fusion
architecture and gating**, not from any sequential/temporal component — worth weighing against
M4's bet that sequence structure specifically is the missing piece (§3).

---

## 3. Sequence-level transfer prior art

**Genuinely underexplored territory, with one important negative-context data point.** No paper
was found (positive or negative) that tests RNN/SSM/attention-based **cross-model** hidden-state
*sequence* translation as its own ablated variable against a pointwise/per-token-independent
baseline. What was found:

- **C2C's fuser is pointwise per-token/per-layer** (§2 above) — the strongest existing positive
  result in this space does **not** use sequential/recurrent structure across tokens, and still
  succeeds (given task loss). This is at minimum weak evidence *against* "sequential structure is
  necessary for cross-model transfer to work at all" — C2C didn't need it. It does not rule out
  sequential structure *helping further*, only that it is not a precondition for any signal.
- **The Bicameral Model is also pointwise-per-token** ("both models run in lockstep... at every
  generation step," §2) — a second independently-built successful trained method, also without
  recurrent structure across the communicated stream.
- **Vision Wormhole's temporal-mixing design could not be confirmed either way this pass** — the
  abstract-level fetch found only "hub-and-spoke topology" and "Universal Visual Codec" language,
  not enough to state whether its per-agent encoder/decoder mixes across token positions. Graded
  **uncertain**, not resolved from 027's original pass either — a genuine gap, not settled by this
  sweep, and worth a dedicated fetch of the paper body if M4's result is ambiguous.
- **No SSM/Mamba-based cross-model activation-transfer adapter was found.** Searches surfaced only
  same-model architecture-conversion work (TransMamba, Mamba-3's own cross-architecture
  distillation) — converting one model's weights into an SSM, not translating one model's live
  activations into a second, different model's activations. This is adjacent, not on point.
- **No paper was found claiming sequence-level cross-model transfer was tried and failed.** The
  absence is as informative as it is uninformative: nobody has published a negative result here,
  but nobody has published this exact test at all, positive or negative.

**Answer to the framing question**: this is **unexplored territory**, not a case of "already
tried and failed" or "already tried and succeeded." The one directly relevant data point — C2C and
Bicameral both succeed *without* sequential structure — is a mild caution against over-attributing
M3's failure to "pointwise mapping is inherently insufficient" and M4's eventual result to "adding
recurrence fixed it," **if** M4 is trained on the same reconstruction loss M3 used: per §2, the
loss-function gap is present in M4 exactly as it was in M3 (ADR-024's M4 section describes
training the FastGRNN cell on captured per-token pairs with the same per-token-pair data pipeline
as M3, with no stated task-loss objective). If M4 fails, the most parsimonious reading given this
section's evidence is "the loss function, not the architecture, is still the confound" — not
"sequential structure doesn't help either." This should be stated as an explicit alternative
hypothesis before M4's result is interpreted.

---

## 4. Implications for M4/M4b/M5 while they run

1. **Highest-priority, concrete, and actionable**: the task-loss-vs-reconstruction-loss gap (§2)
   applies identically to M4 (in flight) as it did to M3. If M4 also fails the frozen probe, the
   ladder's own discipline (ADR-024: "a rung's failure escalates to the next architecture... not
   grounds to retry the same architecture with different hyperparameters") would currently move to
   M5 (MicroLoRA) — but M5, per ADR-024, is *already* the rung that "actually exercises the causal
   admission machinery live rather than as a downstream validation step," using ADR-003's own ΔV
   as online feedback. That is closer to task-loss training than M3/M4's offline captured-pair
   fits are. **Recommendation for the coordinator to weigh**: before spending M5's build effort,
   consider a cheap, cleanly-isolated control — retrain M3's exact MLP architecture (or M4's, if it
   fails first) on task loss (frozen receiver, live gradient path, next-token cross-entropy, C2C's
   `SFT_train.py` pattern) rather than escalating architecture again. This is not a proposal to
   violate the "no retry the same architecture" rule in spirit — it is a proposal to test whether
   the ladder has been probing the right *objective*, not just the right *architecture family*,
   before the ladder's own kill-criterion (`docs/research/027` §4.4 Stage 1) is invoked. Framed as
   a registration question for the coordinator, not a unilateral scope change.
2. **The receiver-scale confound (registered in ADR-024) is unresolved by this sweep** — no v2 or
   follow-up to arXiv:2608.05164 was found; the 1.7B threshold claim stands exactly as previously
   registered, still unconfirmed/uncontradicted by anything newer.
3. **The Aristotelian-PRH finding (§1) suggests A6's residual metric itself may deserve a
   permutation-null baseline** — not a change to the frozen probe protocol (which ADR-024
   correctly keeps unchanged), but a candidate diagnostic to run *alongside* M4/M5's own A6-style
   fit checks, to see whether "PASS" is being cleared partly by a depth/width artifact rather than
   real structure. Worth a cheap add-on, not a blocker.
4. **M4's sequential-structure hypothesis is untested by prior art either way (§3)** — proceed as
   planned, but interpret a pass or fail against the loss-function confound named in point 1, not
   in isolation as confirming or denying "sequence structure matters."
5. **Two new trained methods (Bicameral Model, Interlat) were not in LatentMesh's existing threat
   map (023/027) and are worth a citation update, not a scope change** — see ADR-drift list below.

---

## 5. ADR-drift list (read-only — no ADR files modified)

- **ADR-file: `docs/adr/024-run2-trained-thought-adapter-ladder.md`, §"Registered confound —
  receiver-scale threshold" + claim: arXiv:2608.05164's 1.7B threshold is cited as the live,
  unrevised finding. Correction: none found — no v2, errata, or follow-up located this pass. The
  citation stands as-is; this is a confirmation, not a drift, but is listed because the ADR
  explicitly asked for this check.**
- **ADR-file: `docs/research/023-beyond-sota-roadmap.md`, §1a table + claim: Cache-to-Cache is
  cited only for its accuracy/speedup numbers and architecture (projector + gating), with no
  statement about its training objective. Correction: C2C's Fuser is trained on end-to-end task
  loss (causal-LM cross-entropy via the receiver's own forward pass), not reconstruction loss —
  worth adding as a fact when 023 is next revised, since it is directly load-bearing for how any
  future LatentMesh adapter work is scoped (see §2 above).**
- **ADR-file: none currently — new finding, not a correction: `docs/research/023` and `027`'s
  citation lists do not include "The Bicameral Model" (arXiv:2605.11167, bidirectional hidden-
  state coupling between parallel frozen LLMs, task-loss-trained, pointwise-per-token, +60.3pp
  arithmetic) or "Interlat"/"Enabling Agents to Communicate Entirely in Latent Space"
  (arXiv:2511.09149, ACL 2026, latent-prefix-flavored communication adapter, heterogeneous models,
  directly relevant to ADR-027's latent-prefix design). Neither supersedes an existing claim, but
  both are close enough to LatentMesh's own space (and Interlat specifically close enough to
  ADR-027's latent-prefix contingency) that a future revision of 023/027 should cite them.**
- **ADR-file: `docs/research/027-global-ambient-intelligence-track.md`, §1.3 ("Platonic
  Representation Hypothesis... limitations and counterexamples were not retrievable at the depth
  fetched in this pass — a genuine gap") + claim: 027 flags this as an open gap and names an
  unconfirmed "Adragna" critique. Correction: a **primary-verified** critique now exists and
  should replace/supplement the Adragna pointer — "Revisiting the Platonic Representation
  Hypothesis: An Aristotelian View" (arXiv:2602.14486): global similarity-metric convergence is
  shown to be a depth/width artifact that "largely disappears" after permutation-null calibration,
  with only local-neighborhood similarity surviving. This is a stronger, resolvable citation than
  the still-unconfirmed Adragna talk, and directly informs how much weight PRH should carry as
  LatentMesh's theoretical justification for expecting shared structure to exist at all.**
- **ADR-file: none — no drift found for Zhang & Emu (arXiv:2607.26773) or MANTA (arXiv:2607.28527)
  — no follow-up, v2, or citing paper materially extending either was found this pass. Both stand
  as previously cited.**
- **New, unrelated to the 6 items above**: C2C (arXiv:2510.03215) is confirmed accepted at
  ICLR 2026 (was an in-review preprint when 023 was written) — a status upgrade, not a content
  change, worth reflecting if 023 is revised for publication-readiness framing.

---

## Sources

- Cache-to-Cache (ICLR 2026, accepted): https://arxiv.org/abs/2510.03215,
  https://github.com/thu-nics/C2C, `script/train/SFT_train.py` (training loss, primary)
- LatentMAS: https://github.com/Gen-Verse/LatentMAS, https://arxiv.org/abs/2511.20639
- AVP: https://github.com/VectorArc/avp-spec, https://github.com/VectorArc/avp-python
- The Bicameral Model: https://arxiv.org/abs/2605.11167
- Interlat / "Enabling Agents to Communicate Entirely in Latent Space":
  https://arxiv.org/abs/2511.09149 (ACL 2026)
- "Revisiting the Platonic Representation Hypothesis: An Aristotelian View":
  https://arxiv.org/abs/2602.14486
- "Representation Alignment Rests on Linear Structure": https://arxiv.org/abs/2605.28870
- "Transferring Linear Features Across Language Models With Model Stitching":
  https://arxiv.org/abs/2506.06609
- "Linear Representation Transferability Hypothesis": https://arxiv.org/abs/2506.00653 (inferred)
- Amnesic probing (Elazar et al., TACL); "Probing Classifiers: Promises, Shortcomings, and
  Advances" (Belinkov, MIT Press) — general interpretability grounding, primary via search
  synthesis
- "Cross-Architecture Steering Transfer in Language Models" (1.7B threshold):
  https://arxiv.org/abs/2608.05164 — no update found this pass
- "Do Latent Channels Actually Communicate?" (Zhang & Emu): https://arxiv.org/abs/2607.26773 —
  no update found this pass
- MANTA: https://arxiv.org/abs/2607.28527 — no update found this pass
- "Understanding (Un)reliability of Steering Vectors" (OpenReview id JZiKuvIK1t) — fetch blocked
  (403), graded uncertain, named as a lead for a future pass
- "Adragna, Against the Platonic Representation Hypothesis" — still uncertain, only found as a
  talk/event page (luma.com/n5d2q50u), not a resolvable preprint
- Internal: `docs/research/023-beyond-sota-roadmap.md`, `docs/research/027-global-ambient-
  intelligence-track.md`, `docs/adr/023-live-four-condition-run1-pre-registration.md` §S6,
  `docs/adr/024-run2-trained-thought-adapter-ladder.md` §"M3 outcome" and §"Registered confound"
