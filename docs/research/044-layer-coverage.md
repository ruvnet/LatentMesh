# 044. Layer coverage — is single-site injection structurally too weak to matter?

* **Purpose**: the last untested structural difference between LatentMesh's design and every
  cross-model method that reports a real positive result. We inject once, at **one** site
  (sender L18 → receiver L14, ~50% relative depth), 8 slots, once. C2C/LatentMAS/Bicameral all
  touch **multiple layers** of the receiver. This document asks whether that difference plausibly
  explains the ladder's on-manifold "inert, not harmful" family, surveys what the literature says
  about single-layer vs multi-layer intervention strength, ablates what the three comparator
  methods actually do, and scopes the cheapest test on this stack.
* **Date**: 2026-08-29 (branch `feat/run2-thought-adapter`, read-only except this file — not
  committed).
* **Read first**: ADR-024's "MAJOR CORRECTION", "M4h Stage 1 OUTCOME", "M4i PRE-REGISTRATION"
  sections; ADR-036 (successor e-process protocol); `docs/research/032` §1-2 (norm/placement
  survey, corrected in part by 038/039), `docs/research/033` §5 (the two surviving downstream-stack
  mechanisms), `docs/research/040` (the pooling gap — the sibling structural-difference document
  this one is modeled on).
* **Method / evidence grading**: WebFetch of primary sources (arXiv HTML full-text renders) plus
  direct repo source-reading (`crates/latentmesh-runtime/src/{inject.rs,models/qwen2_b.rs}`).
  **primary** = paper fetched and the specific claim read in the fetched text. WebSearch was
  unavailable this pass (session budget exhausted at 200/200 before this task started) — every
  citation below is a direct WebFetch of an arXiv abstract or `arxiv.org/html/...` full-text page,
  graded primary where the quoted text was actually read, and flagged explicitly wherever a fetch
  came back abstract-only.

---

## Answer, up front

**No successful cross-model method transfers content at a single layer, and the one method that
directly ablates single- vs multi-layer content injection shows single-layer injection is
statistically indistinguishable from doing nothing** — a striking quantitative echo of
LatentMesh's own "on-manifold and inert" finding. C2C's own Table 10 single-layer-enrichment
ablation moves accuracy from a 58.42% baseline to 58.45–58.52% for the two best individual layers
(≈0.1pp) while most individual layers are net-negative; its real ~24pp gain comes only from the
gated multi-layer configuration. LatentMAS transfers the full per-layer KV cache at every layer,
with no reduction. The Bicameral Model — corrected from an earlier LatentMesh document that called
it "continuous injection" — actually couples at exactly **4 fixed layer indices** (2 read + 2 write
per direction), swept over 890 configurations to find the working combination, never at one.
**Zero of the three surveyed methods use one layer for content transfer.**

The steering-vector literature (CAA, ActAdd, RepE) tells a more nuanced story that this document
did not expect going in: CAA and ActAdd both steer effectively at a **single, well-chosen layer**
— but they add a fixed *direction* on top of an otherwise-intact residual stream, at every token
position, continuously through generation. RepE, the one method in that family that transfers
richer content (a "concept" read out and pushed back in, not just a linear direction), explicitly
uses **many layers, iteratively**, and names the mechanism directly: single-layer changes get
"diminished" by "cascading" through the network before they reach the readout. That mechanism —
RMSNorm's per-block scale-invariance re-normalizing away a fixed-magnitude residual contribution as
later blocks add their own (undiminished) branch outputs — is the same one `docs/research/033` §5
named as a surviving, untested candidate for LatentMesh's own null.

**Layer coverage is a genuinely strong, literature-congruent candidate for explaining the *ceiling*
on the on-manifold family** (why a geometrically correct, item-varying, non-destructive payload
still moves nothing) — better supported by direct ablation evidence than any prior candidate except
pooling. **It is not, and cannot be, an explanation for the off-manifold family's active harm**
(M4c/M4d/M4g were also single-layer; if single-layer injection alone caused harm, on-manifold
single-layer injection should be harmful too, and it measurably is not — see the MAJOR CORRECTION
table). Layer coverage explains *why nothing gets through*, not *why the wrong content does
damage*; those remain two separate mechanisms, exactly as ADR-024's MAJOR CORRECTION already
established for pooling. The cheapest test on this stack is real but not free: it needs one new
`LayerEdit` variant (`InjectMany`/`FuseMany`, precedented by how `Fuse` was added beside `Inject`)
and one new adapter trained on the sender-L24→receiver-L19 pair — but **zero new capture**, because
both depth pairs are already dumped.

---

## 1. What does the evidence say about single-site vs multi-layer intervention strength?

**The honest finding here is a split, not a clean "everyone uses many layers."** Steering-*direction*
methods (add a fixed vector, at a fixed coefficient, on top of the model's own ongoing computation)
and content-*transfer* methods (carry a payload of information from one model/context into another)
behave differently in this literature, and conflating them would overstate the case.

### 1.1 CAA — single layer, all positions, continuous through generation (primary, fetched)

Contrastive Activation Addition (arXiv:2312.06681) steers at **one** empirically-chosen layer per
model — "In the 7B model, this corresponds to layer 13 and adjacent layers. The optimal layer in
the 13B model is usually 14 or 15" — but adds the vector "at all token positions after the user's
prompt," which means every newly generated token during decoding receives the same addition again.
CAA's own layer-sweep (their Figure 3) shows a clear single peak, not a plateau across many layers,
and their cross-layer transfer test (Figure 8) found the effect "even more significant for some
earlier layers" when a layer-13-derived vector is applied elsewhere — evidence that CAA's own
finding is about which single layer to pick, not that single-layer suffices only marginally. No
ablation comparing simultaneous multi-layer application against single-layer was found in this
paper.

### 1.2 ActAdd — single layer, subset of positions, empirically tuned (primary, fetched)

ActAdd (arXiv:2308.10248) is explicit: "we add the resulting activation vector to the input of
layer *l* and allow the forward pass to continue" — one layer, chosen by hyperparameter search
("intervening at the middle layers is most effective," peak around layer 6 of GPT-2-XL). The paper
does not test or discuss multi-layer simultaneous injection at all — it is a non-finding on that
axis, not a rejection of it.

### 1.3 RepE — many layers, deliberately, with the mechanism named explicitly (primary, fetched)

Representation Engineering (arXiv:2310.01405) is the one steering-family source that both (a) uses
many layers and (b) states why. For their strongest read/control results: "We use the middle 20
layers, which exhibit the strongest reading performance" — and for the control/injection side, they
apply the contrast vector **iteratively, layer by layer**: "modifying each target layer starting
from the earliest layer, computing the contrast vector for the next target layer, and repeating
this procedure" — specifically because of **"the potential cascading effect when simultaneously
altering representations across multiple layers. Changes made in earlier layers may propagate to
later layers, diminishing the effect of the contrast vectors computed upfront."** This is a direct,
primary-sourced statement that a residual-stream perturbation injected at one layer does not
reliably survive to later layers undiminished — precisely the mechanism this document's Q3 asks
about, from the one source in this pass that names it in so many words.

### 1.4 The content-transfer methods: no single-layer transfer exists, and where it was tested it was near-null

See §2 below for the per-method detail. The summary relevant to this question: C2C, LatentMAS, and
the Bicameral Model all transfer *content* (a cache, a coupled state) rather than a *direction*, and
none of the three does so at one layer. Where the comparison was directly run — C2C's own Table 10
— single-layer content injection was **statistically indistinguishable from baseline**, the same
qualitative signature (inert, not harmful, not helpful) that LatentMesh's own on-manifold family
shows.

**Reading the split honestly**: a *direction* — a linear nudge toward a known concept, added
continuously at every step — can work from one layer, because it only has to tip an
already-present, already-computed decision boundary, and it gets to keep nudging at every
subsequent token. A *state* — a payload meant to carry information the receiver did not already
compute, delivered once and left to survive 14+ subsequent blocks unassisted — is the harder case,
and it is exactly the case no surveyed content-transfer method attempts at a single layer. This is
not a proof that single-layer *state* injection is impossible; it is the honest shape of what the
literature has and has not tried, and it lines up with which category LatentMesh's design actually
falls into (state, not direction; one-shot, not continuous — see `docs/research/032` §2 and
`docs/research/040`'s continuous-injection entanglement note).

---

## 2. What exactly does C2C do across layers, and does it ablate this? Same for LatentMAS and Bicameral.

### 2.1 Cache-to-Cache (C2C) — gated subset of layers; single-layer ablated and found near-null (primary, fetched `arxiv.org/html/2510.03215`)

The fuser architecture is built to support every layer, but **a learned per-layer gate decides which
layers actually get enriched** — already established in `docs/research/038`/`040` and re-confirmed
here: "enriching only selected layers is better than enriching all layers." This pass adds the
number: Figure 4 shows "selectively applying cache enrichment to the top-performing layers (e.g.,
top-5) yields slightly higher accuracy than enriching all layers, while targeting the worst-
performing layers leads to accuracy decline" — against a 28-layer Qwen3 backbone (Table 10 spans
layers 0–27), so the gate typically lands on roughly a sixth to a fifth of the stack, not one layer
and not all of it.

**The directly decisive number for this document's question**: Table 10's single-layer-enrichment
ablation. Baseline (no enrichment) is 58.42%. The two best individual layers tested reach 58.52%
(layer 4) and 58.45% (layer 16) — gains of 0.10pp and 0.03pp — and "most other layers" fall *below*
baseline. **A single enriched layer, chosen well, buys C2C essentially nothing**, and a badly-chosen
one costs a little. Compare this to the fuser-component ablation in the same paper (Table 8): pure
projection alone reaches 20.70%, adding residual fusion reaches 44.88% (+24.18pp), adding the
learned gate reaches 47.95% (+3.07pp further) — these are a different metric/task setup than
Table 10's 58% baseline, so the two tables are not directly subtractable, but the qualitative
picture from both is the same: **single-layer content injection is where C2C's own ablation puts
the null**, and its real effect comes entirely from the multi-layer, gated configuration.

### 2.2 LatentMAS — all layers, full per-token cache, no reduction (primary, from `docs/research/040`, re-confirmed here)

LatentMAS (arXiv:2511.20639) prepends the full per-token, per-layer KV cache "across all `L`
transformer layers, extracted once" — this pass found no ablation in the fetched text (nor in
`docs/research/040`'s prior read) that reduces layer count below "all." It is the one comparator
method for which "does layer coverage matter" was never tested at all, because it was never varied
— all-layers is the only configuration reported.

### 2.3 The Bicameral Model — corrected: exactly 4 fixed layers, never continuous-at-every-layer, swept over hundreds of configurations (primary, fetched `arxiv.org/html/2605.11167`)

This is a genuine new finding for this pass, and it corrects how the Bicameral Model has been
described in this repository's prior research documents (`docs/research/028`, `039`, `040` all call
it "continuous... every generation step" without specifying layer count, which is accurate about
*when* but silent about *where*). The paper's own text: **"The interface reads and writes at four
layer indices (read and write for each direction): ℓ_{r,p→a}, ℓ_{w,p→a}, ℓ_{r,a→p}, ℓ_{w,a→p}"** —
i.e., two read/write layer pairs, one per coupling direction, **four layers total**, not a
continuous sweep across the stack. Critically, the paper reports sweeping **890 sampled
configurations, varying which coupling layers are selected** — meaning *which* four layers matters
enough to warrant an 890-point search, not an incidental detail. **Bicameral is therefore multi-site
by design and its own ablation practice treats layer selection as a first-class, expensively-tuned
hyperparameter** — a second, independent instance (alongside C2C's gate) of a successful method
finding that layer choice/count is load-bearing enough to search over.

### 2.4 Cross-method verdict

| Method | Layers touched | Ablates single-layer? | Result |
|---|---|---|---|
| C2C | Learned gate, ~top-5 of 28 | **Yes** (Table 10) | Single layer ≈ baseline (+0.03 to +0.10pp); most layers net-negative |
| LatentMAS | All layers | No (never varied) | N/A |
| Bicameral | Exactly 4, searched over 890 configs | Implicitly — 890-config sweep treats layer *choice* as critical, never reports 1-layer | Not directly comparable (no single-layer number found) |
| **LatentMesh (current)** | **1 (L18→L14)** | — | On-manifold family: inert (Δ ≈ 0.004 to −0.021 nats) |

No method in this survey transfers content at a single layer. Where the comparison was run
directly, the result mirrors LatentMesh's own on-manifold family almost exactly in kind: not harmful,
just absent.

---

## 3. Is there a theoretical reason one layer is not enough?

**Partial answer, sourced from two places, neither of which is a clean quantitative "half-life"
result.** RepE's own stated mechanism (§1.3) is the most direct primary citation found this pass:
a perturbation introduced at an early/mid layer is progressively "diminished" as later layers
"propagate" and add their own (unperturbed) contributions on top. This is qualitatively consistent
with `docs/research/033` §5's two named-but-untested mechanisms for LatentMesh's own stack,
restated here because they are the concrete, model-specific version of RepE's generic claim:

> "(i) the residual at block 14 is overwritten with `c·v`, and blocks 15–28 then *add* branch
> outputs to it — each block's own RMSNorm is scale-invariant, so the branch outputs are (near)
> fixed while the carried `c·v` term scales, changing the balance downstream even though the
> block-14 readout does not change; (ii) attention at blocks > 14 reads the slot rows as keys/values,
> where absolute magnitude is not normalised away."

Mechanism (i) is the RMSNorm-scale-invariance argument: each of Qwen2.5-1.5B's remaining ~14 blocks
after L14 normalizes its own input before computing a branch output, so that branch output's
magnitude is set by the *model's own* typical activation scale at that block, not by whatever
magnitude the injected payload carries. The injected content is carried forward additively in the
residual stream, but its *relative* share of the residual's total norm shrinks every time a
full-magnitude branch output is added on top of it — a form of dilution, not erasure, but one that
compounds over 14 blocks. This is consistent with, and gives a mechanistic story for, RepE's
observed need to keep re-injecting layer by layer rather than trusting one early injection to
survive.

**What this pass could not find**: no source fetched quantifies *how many blocks* it takes for an
injected perturbation of a given relative magnitude to become sub-threshold for downstream
behavior — i.e., there is no "washout half-life in blocks" number in the literature surveyed here,
for this model family or any other. The RepE citation and `docs/research/033`'s own mechanism (i)
together give the qualitative direction (dilution grows with distance from the injection site, and
14 blocks is a lot of distance relative to CAA/ActAdd/RepE's own effective layers, all chosen in the
middle third of their host models) but not a number that would let this document say "L14 in a
28-block model is [inside/outside] the surviving range." This is named as an open quantification
gap, not resolved here.

---

## 4. The cheapest test on our stack

**(a) `LayerEdit` is single-site by construction; multi-layer injection needs one new variant, not
a rewrite.** Read directly from `crates/latentmesh-runtime/src/models/qwen2_b.rs`: `LayerEdit` is
an enum with `Capture`, `CaptureMany`, `Inject`, and `Fuse` variants, and `Model::forward_with_edit`
takes exactly `edit: Option<&mut LayerEdit<'_>>` — **one** edit reference per forward pass, applied
via `apply_edit(xs, blocks_run, edit)` which matches `after_block == blocks_run` for whichever
single variant was passed. `Inject`/`Fuse` each carry one `after_block: usize` — there is no
`InjectMany`/`FuseMany` counterpart to the read-only `CaptureMany` that already exists precisely for
this purpose ("S2 calibration sweeps 3 depths per model; three single-tap passes would triple the
prefill cost" — the same cost argument applies to injection). The precedent for adding this cleanly
already exists in the same file: M4g added `Fuse` "beside" `Inject` as a wholly separate variant
rather than a flag, specifically so every existing receipt stayed reproducible and the overwrite
arm's op sequence stayed untouched. An `InjectMany`/`FuseMany` variant carrying
`Vec<(after_block, positions, vectors)>` (or a small fixed-size array, since only 2 depths are in
play) is the same shape of change, bounded to `qwen2_b.rs`'s edit-application match arm plus the
harness call sites that currently construct one `InjectionSpec`.

**(b) We already have the sender-side and receiver-side captures needed for a 2-layer test — no
new capture.** Confirmed directly from ADR-024 §"per-token dump" text: the per-token dumps exist at
**both** depth pairs already — `sender_L18.tok.f32bin`, `sender_L24.tok.f32bin`,
`receiver_L14.tok.f32bin`, `receiver_L19.tok.f32bin`, all 2,560 items, sharing one offsets index.
No rung in the ladder has ever used the L24/L19 pair — every rung to date, M3 through M4i, has used
L18→L14 exclusively (confirmed by grep across ADR-024: the only other depth mentions are the raw
dump-manifest line itself). A 2-layer rung is therefore purely a training-time and harness-time
cost; the data pipeline is already in place.

**(c) Minimal 2-layer rung, cost estimate.** Reuse M3's already-trained L18→L14 on-manifold MLP
(byte-identical, as M4h Stage 1 did) for the shallow site; train one new M3-shaped MLP on the
L24→L19 pair, same reconstruction-loss recipe, same per-token dump pipeline, no architecture
change — this is a same-shaped training run to M3's own, whose cost is already receipted elsewhere
in the ladder (on the order of the M3/M4 training rungs, not a M4c/M4d/M4g task-loss run). Add the
`InjectMany`/`FuseMany` `LayerEdit` variant (a `qwen2_b.rs` change plus harness plumbing — bounded,
single-file-plus-call-sites, same size as the `Fuse` addition M4g already made). Fuse both sites
simultaneously, under M4g's now-standard fuse semantics (preserves the receiver's own content at
both sites rather than destroying it), and evaluate under ADR-036's e-process — this would be a
successor rung, so ADR-036 Decision 4's default table applies ("any later rung... governed by this
ADR unless its own pre-registration explicitly opts into a different protocol").

**One genuine tension to flag before this is pre-registered, not resolved here**: M4h Stage 1 and
M4i both kept slot count fixed at exactly 8 specifically to preserve frozen-probe compatibility
(per ADR-028's slot-count discipline, itself internally contradictory and unadjudicated — see
ADR-024's "ADR-028 INTERNAL CONTRADICTION" section). A 2-site rung most naturally wants 8 slots
*per site* (16 total) or a split (4+4) — either choice changes the object being injected in a way
the existing 8-slot probe-compatibility framing does not obviously cover, and ADR-028's owner should
weigh in before this is pre-registered, the same way that ADR already flags for slot count generally.
Under ADR-036 this is moot for the *statistic* (the e-process governs successor rungs regardless of
slot count) but not for the *architecture's* comparability to the completed 8-slot-single-site era.

---

## 5. Honest assessment: does layer coverage plausibly explain inertness specifically?

**Yes, and on stronger direct-ablation footing than any prior candidate in this ladder except
pooling — but it explains a different half of the data than M4i does, not a competing account of
the same half.**

**Where it fits relative to the refuted candidates.** Task loss (M4f), injection operator (M4g),
and pooling (M4h Stage 1) are refuted for the *on-manifold* family specifically — none of them, once
controlled for, moves the on-manifold payload out of "inert." Layer coverage was never varied by
any completed rung; it sits alongside M4i (placeholder-token choice) as the two live, untested
candidates that specifically predict **inertness** (as opposed to M4b's receiver-scale hypothesis,
which predicts a capability-gap-dependent null, or M4e's continuous-injection hypothesis, which is
partly conflated with layer coverage per `docs/research/040`'s own "substantially entangled" note).

**Where it is stronger than M4i's current evidence base**: M4i's supporting literature (`docs/
research/043`) is two abstract-only, inferred-grade citations ("Let's Think Dot by Dot," the
pause-token paper). This document's central citation — C2C's own Table 10 — is a primary-graded,
directly-fetched, directly-quantified ablation showing single-layer content injection lands within
0.1pp of baseline on the comparator method's own task. That is a materially stronger form of
evidence than either of M4i's citations, though it comes at materially higher cost to test (new
training + a harness change, vs. M4i's zero-training, zero-capture reuse of M3's existing weights).

**Where it would fail to explain the data, stated plainly**: layer coverage says nothing about why
the off-manifold family (M4c/M4d/M4g) is actively *harmful* rather than merely inert. All three of
those rungs were also single-layer, single-site L18→L14 — if "single layer is too weak to have any
effect" were the whole story, off-manifold single-layer injection should be as inert as on-manifold
single-layer injection, and it measurably is not (0/40 unanimous NLL inversion, 2.5–2.7 nats worse
than baseline, vs. the on-manifold family's ≤0.03-nat deviations). The parsimonious reading,
consistent with the MAJOR CORRECTION's own two-family split: **layer coverage is a hypothesis about
ceiling, not about direction** — it can explain why a well-behaved payload's positive effect never
clears the noise floor, but it cannot explain why a badly-behaved payload's negative effect does
clear it. Those need separate mechanisms (content/direction for harm, coverage/dose for missing
benefit), the same structural split this ladder has already had to draw for task loss vs. pooling.

**If both M4i and a layer-coverage rung also null**, the honest remaining conclusion is this: every
named single-axis structural difference from the working methods (task loss, injection operator,
pooling, placeholder-token choice, and now layer coverage) will have been tested and refuted for the
on-manifold family, one factor at a time, per this ladder's own one-factor-per-rung discipline. Two
non-exclusive readings would remain live, and neither is closable by a further single-factor rung:

1. **Conjunction, not disjunction.** Every method that actually works (C2C, Bicameral, LatentMAS)
   combines multiple of these axes *simultaneously* — C2C is multi-layer **and** task-loss-trained
   **and** operates on the full per-token cache (never pooled); Bicameral is multi-layer **and**
   continuous **and** bidirectional **and** task-loss-trained. This ladder has, by design, tested
   each axis in isolation against an otherwise-inert on-manifold baseline. A ladder that nulls on
   every single factor is fully consistent with a world where transfer requires several of these
   axes *together* — e.g., multi-layer *and* continuous *and* task-loss, which is architecturally
   close to reimplementing a scoped C2C rather than any remaining single-axis LatentMesh rung. That
   would not be a new finding so much as a return to the choice ADR-024 has deferred throughout:
   whether to keep isolating single factors on a design that has never worked, or to build the
   closest-to-working comparator directly and work backward from it.
2. **Receiver scale remains the one mandatory, not-yet-run arm** (M4b) that this document's
   argument does not touch at all — every rung discussed here, on-manifold and off-manifold alike,
   used the same ≤1.5B-parameter receiver, and `docs/research/035`'s external corroboration (a
   capability-gap threshold, and Bicameral's own GSM8K degradation when the paired models are
   close in scale) is untouched by anything in this document. If layer coverage also nulls, M4b
   becomes the only remaining named candidate this ladder has not yet drawn against, and the honest
   ladder-wide statement would need to say so explicitly rather than defaulting to "transfer doesn't
   work here" — the same distinction ADR-036 already requires between "an effect wasn't detected"
   and "no effect exists," extended from statistical power to hypothesis coverage.

---

## Sources

- Cache-to-Cache (C2C): https://arxiv.org/abs/2510.03215, https://arxiv.org/html/2510.03215
  (primary, fetched this pass — Table 10 single-layer ablation, Figure 4 top-5-vs-all, Table 8 fuser
  ablation)
- LatentMAS: https://arxiv.org/abs/2511.20639 (primary via `docs/research/040`'s prior fetch,
  re-confirmed here; no new fetch this pass)
- The Bicameral Model: https://arxiv.org/abs/2605.11167, https://arxiv.org/html/2605.11167
  (primary, fetched this pass — 4-layer coupling-index equation, 890-config sweep; corrects the
  layer-coverage silence in `docs/research/028`/`039`/`040`'s prior characterizations)
- Contrastive Activation Addition (CAA): https://arxiv.org/abs/2312.06681,
  https://arxiv.org/html/2312.06681 (primary, fetched this pass — single-layer-per-model-size
  choice, all-positions-after-prompt application, Figure 3/8 layer-sweep and transfer results)
- Steering Language Models With Activation Engineering (ActAdd):
  https://arxiv.org/abs/2308.10248, https://arxiv.org/html/2308.10248 (primary, fetched this pass —
  single-layer injection, middle-layer peak, no multi-layer ablation found)
- Representation Engineering (RepE): https://arxiv.org/abs/2310.01405,
  https://arxiv.org/html/2310.01405 (primary, fetched this pass — middle-20-layers reading,
  iterative multi-layer control, explicit cascading-diminishment mechanism)
- Locating and Editing Factual Associations in GPT (ROME): https://arxiv.org/abs/2202.05262,
  https://arxiv.org/html/2202.05262 (primary, fetched this pass — cited for contrast, not support:
  ROME's single-layer edits work because they modify persistent MLP *weights* at one layer, changing
  what that layer computes for every future forward pass, which is a structurally different
  intervention from a one-shot residual-stream *state* injection that must survive passive
  propagation through 14+ subsequent blocks with no repeated "power source")
- Activation-patching methodology survey: https://arxiv.org/abs/2404.15255 (abstract-only this
  pass — fetch did not surface layer-depth-curve specifics; carried forward from `docs/research/032`
  as a general methodology citation, not upgraded)
- Internal: `docs/adr/024-run2-trained-thought-adapter-ladder.md` (MAJOR CORRECTION, M4f/M4g/M4h/M4i
  sections, ADR-028 internal-contradiction flag), `docs/adr/036-successor-rung-evaluation-protocol.md`
  (e-process primary statistic for any successor rung this document's proposal would fall under),
  `docs/research/032` §1-2, `docs/research/033` §5 (the two surviving downstream-stack mechanisms
  this document's §3 builds on), `docs/research/040` (the pooling gap — sibling document, same
  method), `crates/latentmesh-runtime/src/models/qwen2_b.rs` (`LayerEdit` enum, single-site-per-pass
  confirmed by direct read), `crates/latentmesh-runtime/src/inject.rs` (`InjectionSpec`, single-site
  by construction)
