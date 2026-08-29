# 039. Bidirectional / multi-round latent exchange — is the last untouched axis worth touching?

* **Purpose**: survey whether anyone has demonstrated working multi-round or bidirectional latent
  exchange between models, argue both sides of "is one-shot handoff the hardest possible case,"
  sketch what a bidirectional rung would concretely cost this stack, check whether it interacts
  with the ladder's own diagnosed failure modes, and recommend where it sits in priority against
  the ladder's registered-but-unrun rungs (M4b, M4e, M4f) and the just-launched M4g.
* **Date**: 2026-08-29.
* **Scope/method**: WebSearch + WebFetch only, this session. No repo code run, no probe draw — this
  document is read-only against the repo except for itself.
* **Evidence grading**: **[confirmed-primary]** = paper fetched, specific claim read in the fetched
  text this session. **[confirmed-secondary]** = search-result synthesis that itself cites/quotes
  the source, not independently re-fetched in full. **[inferred]** = this document's own reasoning
  from confirmed facts, not itself sourced. **[absent]** = explicitly checked for and not found.
* **Read first**: [ADR-024](../adr/024-run2-trained-thought-adapter-ladder.md) § DIAGNOSIS, § M4f
  PRE-CHECK VERDICT, § M4g REGISTERED, § M4d outcome, § "Registered hypothesis — M4e"; § "Registered
  confound — receiver-scale threshold"; [research/036](036-manifold-collapse-across-the-ladder.md);
  [research/038](038-manifold-constrained-adapter-scout.md); [research/032](032-injection-configuration-science.md);
  [research/035](035-probe-task-selection.md).

---

## Answer, up front

**Bidirectional latent exchange has been demonstrated to work, exactly once, by one paper — and it
does not look like the "multi-round dialogue" the mission brief's framing suggests.** The Bicameral
Model (arXiv:2605.11167, **confirmed-primary**) couples two frozen models' hidden states
*continuously and simultaneously, at every single decoding step*, through a trained gate — 36%→96%
on arithmetic, 37.5%→64.7% on ZebraLogic. This is real, working, bidirectional influence. But it is
architecturally a **fused joint computation**, not a turn-based exchange where a receiver replies
to a sender's message and the sender revises. No paper found anywhere in this pass implements that
latter shape — genuine round-trip dialogue in latent space, receiver-state informing a *later,
separate* sender step — in either direction, successfully or not. LatentMAS, the paper whose
"shared latent working memory" framing sounds most dialogue-like, is confirmed
**[confirmed-primary]** strictly unidirectional and one-pass: agents never revisit each other.

**The frozen probe cannot score a bidirectional exchange, in any form, without a redesign.** It is
built to answer one question — did injecting X into a receiver change its P(correct) on 40 fixed
items relative to a control — and every clause of that design (single receiver, single injection,
free-run to a scored answer) is violated by a protocol where the sender's own state also changes
mid-exchange. This is not a close call; see § 3.

**Priority recommendation: last.** Bidirectional exchange does not test any of the ladder's three
live hypotheses (off-manifold collapse, overwrite-destroys-state, receiver scale) — it is orthogonal
to all three (§ 4) — and its one working precedent (Bicameral) requires *continuous* per-step
coupling as a structural precondition, which this ladder has not yet tested even unidirectionally
(M4e, registered, unscheduled). Testing bidirectional before continuous-unidirectional would violate
the ladder's own one-changed-factor-per-rung discipline twice over. It belongs behind M4b/M4e/M4f/M4g
in this run, and probably outside run 2 entirely — as a named, unscheduled future direction with its
own ADR, the same treatment M4.5/ADR-025/026/027 already received, not a run-2 milestone.

---

## 1. Does anyone do multi-round / bidirectional latent exchange, and does it work?

### 1.1 The one confirmed working case: The Bicameral Model (arXiv:2605.11167)

**[confirmed-primary]**, full text fetched. Two frozen models (e.g., twin Qwen2.5-0.5B-Instruct, or
twin Qwen3-0.6B / Qwen3-4B) coupled through a small trainable interface (~1% of combined parameters)
operating **at every generation step**: model A reaches a "forward-coupling" layer, its hidden state
is translated and gated into model B (`h_a ← (1−σ)h_a + σ·f(h_p)`, a learned suppression gate, not
an unconditional overwrite or unconditional add); both continue; model B's state feeds back into
model A **symmetrically**, at a "reverse-coupling" layer, within the *same token step*. This repeats
every step through generation.

Results: arithmetic (Qwen2.5-0.5B pair + calculator on the auxiliary) 36.2%→96.5%; ZebraLogic
(Qwen3-0.6B pair + Z3 solver) 37.5%→64.7% (1.7×); Python-math (Qwen3-4B pair + code sandbox) 62.5% on
MATH, *below* the 81.6% thinking-alone baseline — a genuine failure case reported alongside the
successes, per the paper's own framing.

**What this is not**: turn-based dialogue. There is no point where model A "sends," model B
"replies," and A revises based on B's reply as a discrete later step. The coupling is continuous and
simultaneous — closer to two coupled oscillators or a fused two-headed forward pass than to a
conversation. The paper's own claimed contrast is with "turn-based or sequential" prior
approaches — it explicitly positions itself against one-shot/unidirectional coupling, not against
multi-turn text dialogue.

**No ablation isolates bidirectionality itself.** **[confirmed-primary]**, checked directly: the
paper's only relevant ablation replaces the auxiliary model's coupling with a no-op round-trip
(`M_p → ∅ → M_p`), and performance collapses (arithmetic −50.5pp, ZebraLogic −57.2pp). That shows
*having a real second model matters enormously*; it does not isolate "bidirectional" from
"unidirectional-but-continuous," because no unidirectional-continuous control condition is reported.
**This is the single most load-bearing gap in the evidence base for this document's Q1/Q2**: the
strongest positive result conflates "continuous" and "bidirectional," and this pass found no source
that separates them.

**A directly relevant negative finding, buried in the same paper**: on GSM8K, where the base model
already handles most items well, bicameral coupling *degrades* performance (49.6%→~40%) — the
paper's own reading is that "hidden-state perturbations inject more noise than signal" when the
capability gap between what the receiver already can do and what the sender offers is small. This is
independently corroborating evidence for something LatentMesh's own ladder has repeatedly run into
from a different angle: a 3B→1.5B sender/receiver gap on GSM8K specifically (the same task) may
simply be too narrow a gap for latent transfer of any shape — continuous, one-shot, or bidirectional
— to show a clean positive effect, independent of injection mechanism. `docs/research/035` already
flags GSM8K's low cross-model discordance as a probe-sensitivity problem from the statistics side;
this is the same concern arriving independently from the steering literature's own failure mode.

**No VRAM/compute-overhead numbers were locatable in the abstract/available text this pass** for
running two coupled models simultaneously — flagged as **[absent]**, not measured, rather than
assumed favorable or unfavorable.

### 1.2 LatentMAS (arXiv:2511.20639) — the paper whose framing sounds most like a dialogue, and isn't one

**[confirmed-primary]**, full text fetched. "Shared latent working memory" is the framing that most
resembles bidirectional/multi-round exchange in the mission brief's sense, but the mechanism is
confirmed strictly **unidirectional and one-pass**: agent A₂ prepends agent A₁'s cached KV state and
generates forward; **there is no path for A₁ to receive anything back from A₂**, and no agent
revisits an earlier agent's state. Two topologies are used — a four-agent sequential pipeline
(planner→critic→refiner→solver) and a hierarchical fan-out/aggregate structure — neither of which is
a loop. Results are strong (up to +14.6pp accuracy, 70.8-83.7% token reduction, 4-4.3× speedup on
Qwen3-4B/8B/14B across 9 benchmarks), but they are evidence *for* one-shot, forward-only latent
transfer working well **when both agents are drawn from the same model family/checkpoint acting in
different roles** — a materially easier setting than LatentMesh's cross-checkpoint, cross-capability
3B→1.5B transfer, and worth naming as a scope difference rather than treating LatentMAS's success as
evidence about LatentMesh's own null.

**[confirmed-primary]**: the paper contains no discussion of iterative multi-round exchange between
the same agent pair, and its own efficiency framing (fewer tokens, faster inference) would be
undermined by adding revisit loops — the design explicitly optimizes against that shape.

### 1.3 C2C, again, for completeness on the "does it iterate" question

Already established as primary fact in `docs/research/038` §4 (re-cited here, not re-fetched): C2C's
fusion (`𝒞_F = 𝒞_n(X) + ℱ_n(...)`) happens **once, during prefill** of the source context — one-shot,
unidirectional, receiver-to-sender-and-back never occurs. C2C is neither multi-round nor
bidirectional, despite `docs/research/032`'s earlier characterization (since corrected in 038) of it
as injecting "continuously... in lockstep" — that description fits the Bicameral Model, not C2C.

### 1.4 Everything else surveyed

- **Latent Cache Flow** (arXiv:2605.22863, **confirmed-secondary**): a compressed-adapter successor
  to C2C's cache-translation idea (adapter ~4% of C2C's size). No indication of multi-round or
  bidirectional exchange — it is a cheaper one-shot translator, same shape as C2C.
- **Cache Merging / CanonicalMerge** (arXiv:2607.01308, **confirmed-secondary**): multiple agents'
  KV-caches are *merged* into one context for a final agent (order-invariant merge, not iterative
  exchange). This is a different primitive — parallel fan-in, not sequential rounds or bidirectional
  coupling — named here because "convergent replicated state" sounds dialogue-like but is not: it is
  a one-shot merge operation with a commutativity guarantee, no loop.
- **Mixture-of-Agents** (arXiv:2406.04692, **confirmed-secondary**): text-based, not latent. Its own
  original design *does* describe multi-round iterative synthesis (aggregators re-aggregating
  aggregated answers), but **[confirmed-secondary]** single-round aggregation is the commonly
  deployed variant "due to latency and cost constraints" — i.e., the field's own experience is that
  even in the *text* modality, where round-trip cost is far cheaper than a fused-forward-pass latent
  coupling, multi-round is more often abandoned than adopted in practice. No latent analog of
  multi-round MoA was found.
- **No paper found** (explicitly checked, **[absent]**) implements a genuine turn-based latent
  dialogue: sender emits a latent packet, receiver processes it to completion (or to some
  checkpoint) and emits its own state back, sender revises and emits a second packet informed by the
  receiver's reply. This is the shape the mission brief's Q1 asks about most directly, and this pass
  found zero examples of it, working or failing, anywhere in the latent-communication literature
  surveyed across this document plus the prior citations in `docs/research/028`/`032`/`035`/`038`.

### 1.5 Grading summary

| Claim | Grade | Source |
|---|---|---|
| Bicameral Model achieves working bidirectional, continuous coupling | confirmed-primary | arXiv:2605.11167 |
| No ablation in Bicameral isolates bidirectionality from continuity/from having a real 2nd model | confirmed-primary | same, verified by direct check |
| Bicameral degrades on GSM8K when capability gap is small | confirmed-primary | same |
| LatentMAS is strictly unidirectional, one-pass, no revisit loop | confirmed-primary | arXiv:2511.20639 |
| C2C fuses once at prefill, not continuously, not bidirectionally | confirmed-primary | already established, `docs/research/038` §4 |
| MoA's own multi-round design is rarely deployed due to latency/cost even in text | confirmed-secondary | arXiv:2406.04692 + synthesis |
| No genuine turn-based latent dialogue (receiver replies, sender revises) exists in the surveyed literature | absent (explicitly checked) | this pass |

---

## 2. Is one-shot handoff the theoretically hardest case?

**No clean answer exists in the literature — the honest reading is "conditionally yes, and the
condition is exactly the one this ladder's own diagnostics say we fail."**

**The case for** (a single vector must carry everything, unrecoverable if misaligned) has intuitive
force and a real information-theoretic flavor — a one-shot handoff is a hard information bottleneck
with zero opportunity for query-and-clarify. This document did not find a source proving one-shot is
provably the worst case in general (it is a plausible but unverified intuition, **[inferred]**, not
a cited result), but it did find literature on the "more rounds help" side that is directly
conditional rather than unconditionally positive:

- **"Correction and Corruption: A Two-Rate View of Error Flow in LLM Protocols"**
  (arXiv:2604.18245, **confirmed-primary**, fetched): frames multi-round correction as a race between
  a *correction rate* (how fast errors get fixed per round) and a *corruption rate* (how fast new
  errors get introduced per round). **Iteration is net-positive only when correction rate exceeds
  corruption rate; when corruption ≥ correction, more rounds actively degrade outcomes** — a
  threshold-based finding, not a blanket endorsement of "more rounds = better." The paper's own
  scope is text-based protocols, not latent/hidden-state coordination — this document does not claim
  the threshold transfers numerically to a latent channel, only that the qualitative shape (rounds
  help conditionally, not unconditionally) is the right prior to bring in.
- **"Detection Without Correction: A Two-Parameter Decomposition of Multi-Stage LLM Pipelines"**
  (arXiv:2605.27559, **confirmed-secondary**, found via search synthesis, not independently
  fetched in full this pass): reports accuracy plateaus and *reversals* across correction rounds in
  multi-agent debate/self-correction pipelines — direct evidence that adding rounds is not
  monotonically beneficial even when the mechanism (text-based self-correction) is much cheaper and
  better-understood than a latent round-trip would be.
- **A concrete negative datapoint surfaced in search synthesis, not independently verified this
  pass** (**[confirmed-secondary]**, flagged as the weakest-sourced claim in this section): a
  five-role multi-agent system's GSM8K accuracy reported as dropping from 75% to 45% due to error
  accumulation across roles — cited here as illustrative of the failure shape, not as a rigorously
  checked number.

**The case against** (more rounds multiply drift; if one round transmits nothing, N rounds transmit
N × nothing) is therefore the better-evidenced reading for *this specific ladder's situation* — and
it maps onto our own diagnosed mechanisms unusually precisely. Nine of nine cross-model draws in
this ladder are nulls (`docs/research/036`). The two mechanically distinct null families are
"on-manifold and useless" (M3/M4) and "off-manifold and actively harmful" (M4c/M4d). Neither family
is a case where a single round *almost* transmits something a second round could plausibly finish —
they are, respectively, a channel that carries a faithful-looking but content-free signal, and a
channel that carries an actively counterproductive one. **Correction-and-Corruption's own framework
predicts exactly the outcome we should expect from adding rounds to either family**: for the
on-manifold-useless family, there is nothing to correct (correction rate is not meaningfully
positive because there was no error to fix, just an absent signal), so added rounds cost compute for
no gain; for the off-manifold-harmful family, a second round of the *same broken adapter* has no
principled reason to have a higher correction rate than corruption rate — it is the identical
degenerate mapping applied again, not a differently-trained corrective step.

**The theoretically interesting case where bidirectionality *could* plausibly beat one-shot** is
narrower than "more rounds": it requires the *later* round's injection to be conditioned on
information the earlier round's injection did not have — e.g., the sender seeing the receiver's
actual intermediate state and adjusting. That is closer to what Bicameral's simultaneous coupling
achieves (each step's translation is conditioned on the current joint state, not a fixed pre-computed
vector) than to "run the same fixed translator twice." **This is the load-bearing distinction this
document surfaces for Q2**: the literature does not support "iterate a fixed one-shot adapter
multiple times," but it does support (with one confirmed example) "couple two models' *live*
computation continuously so each step's signal is state-conditioned." Those are different claims,
and only the second has positive evidence.

---

## 3. What would a bidirectional rung cost, concretely — and can the frozen probe even score it?

**Compute/VRAM sketch, against this stack:**

- Both models already sit resident together for the *existing* one-way pipeline at **~9.3 GB VRAM
  baseline, toward ~14.3 GB with KV cache** (`docs/research/024` §10 VRAM-regression line item,
  **[confirmed-primary, this repo]**). A bidirectional protocol does not require a second copy of
  either model — the same two resident models suffice — so the base VRAM floor does not obviously
  change.
- What does change: a second adapter (1.5B→3B) alongside the existing 3B→1.5B translator (M3-shaped
  MLPs are ~7 MB each, per ADR-024 — negligible next to the base models), or one shared bidirectional
  module if a single architecture is trained to translate both directions (untested design choice,
  not scoped further here). Adapter VRAM/parameter cost is not the bottleneck.
- The real cost is **compute and engineering, not memory**: each round requires a forward (and, if
  trained online, backward) pass through *both* models with injection applied at each direction's
  site, for however many rounds the protocol runs. If bidirectional coupling is trained end-to-end
  like Bicameral's gate (jointly, with gradients flowing through both frozen backbones at every
  step), this is a materially larger training-infrastructure lift than any rung in the current
  ladder — every M3-M4g rung trains one small network against one frozen, pre-captured dataset;
  Bicameral-style joint training needs both models' forward/backward live, at every step, which is
  the same VRAM-for-backprop problem M4c/M4d's engineering findings already document being tight
  (candle 0.9.2's BF16 training fits the 1.5B receiver alone with ~4.7 GB headroom on a 16 GB card —
  see ADR-024 § M4c engineering findings). Fitting *both* models' training-mode forward/backward
  simultaneously is a materially harder VRAM-budget problem than anything solved so far, and this
  document does not have a measured answer for whether it fits.
- **The overwrite-vs-fuse question (M4g, currently running) is a hard prerequisite, not a parallel
  concern.** `inject.rs` currently overwrites 8 residual positions per injection
  (`crates/latentmesh-runtime/src/models/qwen2_b.rs:79-87`, verified in `docs/research/038` §4). In
  a bidirectional protocol, round 2's injection into the receiver would **overwrite round 1's still-
  unintegrated content at the same 8 slots** before the receiver's own forward computation at blocks
  15-28 had a chance to use it — a strictly worse version of the exact destructive mechanism M4g was
  registered to test. **Bidirectional exchange is not well-posed under the current overwrite
  mechanism at all**; it depends on M4g (or an equivalent fuse-not-overwrite fix) landing first, as
  a structural precondition, not merely a nice-to-have.

**Can the frozen probe score a bidirectional exchange? No — plainly, and not as a matter of degree.**
Every clause of the S1a/S2b protocol this ladder has run nine times (`ADR-024` § "The frozen probe
protocol is inherited unchanged") assumes a shape a bidirectional exchange breaks:

1. **"aligned_real" is defined as one fixed injected vector, computed once, from the sender alone.**
   A bidirectional protocol has no single such vector — the sender's own state is itself modified by
   what comes back from the receiver, so "the sender's message" is not a fixed pre-registerable
   object the way M3-M4g's adapter outputs are.
2. **The receiver "free-generates" for up to 400 tokens after injection, unassisted.** In a
   bidirectional protocol, by definition, the receiver is *not* unassisted during generation — that
   is the entire point of the axis being tested. The probe's scoring window (a single, isolated,
   post-injection generation trace) has no way to represent an ongoing exchange during that window.
3. **There is one scored output — the receiver's answer.** A bidirectional protocol raises a genuine
   design question the probe was never built to answer: does correctness get scored on the receiver
   alone (as now), on whichever model reaches an answer first, on both, or on some fused output? No
   choice here is forced by anything in the existing protocol; a new protocol has to make this call
   explicitly.
4. **Controls (zero-vector, norm-matched random) are defined relative to one injected vector.** A
   bidirectional protocol's natural control set is different in kind — e.g., "receiver gets no
   reply signal at all vs. gets a reply" — not a drop-in substitution of the existing controls.
5. **The stopping rule (how many rounds) is entirely unspecified by anything in this ladder.** Every
   existing rung injects once. A bidirectional protocol needs a pre-registered round count or
   stopping criterion before any evaluation is meaningful, and nothing in ADR-023/024/028 supplies
   one.

**Conclusion, stated as plainly as the brief asked for: this axis needs its own, separately
pre-registered evaluation. It cannot be scored by reusing the frozen 40-item probe with a
bidirectional adapter substituted in — the substitution is not well-typed.** This is a different
situation from every other registered-but-unrun rung (M4b changes the receiver's scale but keeps the
protocol; M4e changes injection cadence but keeps a single sender→receiver direction and a single
scored output; M4f/M4g change the adapter/injection mechanism but not the protocol shape). Bidirectional
exchange is the first axis in this ladder that changes the *protocol*, not just the *mechanism*.

---

## 4. Interaction with the ladder's diagnosed failure modes — orthogonal, stated honestly

Going through each live/closed hypothesis from `docs/research/036` and ADR-024's DIAGNOSIS section:

- **Off-manifold collapse (M4c/M4d).** Orthogonal, and if anything **compounded, not rescued**. The
  mechanism `docs/research/036` established is that task-loss training through an overwrite channel
  never had a structural reason to keep its output near the receiver's natural manifold, because the
  loss only cared about downstream logits, not the injected vector's location. Running the same
  degenerate, item-invariant, off-manifold adapter a second time (in the reverse direction, or as a
  second round in the same direction) does not change *why* it collapsed — the training objective
  that produced the collapse is unaffected by how many times its frozen output gets injected.
  Bidirectionality is a delivery-schedule change, not a training-objective change, and the diagnosed
  cause lives entirely in the training objective.
- **Overwrite destroys the receiver's own state (M4g's hypothesis).** Not rescued — **actively
  worsened**, per § 3 above. A second round's overwrite compounds exactly the mechanism M4g exists to
  test, before M4g's own single-round result is even known.
- **Item-invariance (the adapter emits ~the same vector regardless of the item).** Orthogonal. A
  second round of an item-invariant adapter is still item-invariant; nothing about running it twice,
  in either direction, introduces item-specific content that wasn't there in round one.
- **Receiver-scale threshold (M4b, mandatory, unaffected by anything in this ladder to date).**
  Orthogonal in the narrow sense — bidirectionality doesn't change which side of the ~1.7B threshold
  the receiver sits on — but there is a real, if speculative, interaction worth naming: Bicameral's
  own reported GSM8K degradation with a *small capability gap* (§1.1) suggests that even a
  successfully-coupled bidirectional channel may need a *large enough* capability or scale gap to
  show a clean positive effect, which is a variant of the same receiver-scale concern the ladder
  already has registered, arriving from a different literature. This is not evidence that
  bidirectionality interacts constructively with the scale confound — it is evidence that the scale
  confound may generalize beyond this ladder's specific injection mechanism, which is a reason to
  resolve M4b first, not a reason to prioritize bidirectionality.

**Honest summary: bidirectional exchange does not test, and could not plausibly rescue, any of the
three live diagnosed failure mechanisms.** It is a genuinely different axis — closer in kind to
"replace the entire injection protocol with continuous joint computation" (Bicameral's actual
mechanism) than to "add a second pass of the same broken translator." The only way bidirectionality
becomes a candidate *fix* rather than an orthogonal axis is if it is understood as adopting
Bicameral's architecture wholesale (jointly-trained, continuous, gated, simultaneous coupling) —
which is not an incremental rung on top of M3-M4g's adapter-translator paradigm, it is a different
paradigm, sharing essentially no infrastructure with the current ladder except the two base models.

---

## 5. Recommendation

**Bidirectional/multi-round exchange should sit last among all named directions in this ladder's
current view, and probably does not belong inside run 2 at all.**

Reasoning, in order of weight:

1. **It is layered on top of an axis this ladder has not yet tested even unidirectionally.** The one
   confirmed working precedent (Bicameral) is continuous *and* bidirectional; this ladder has not
   run continuous-unidirectional (M4e, registered, unscheduled) yet. Testing bidirectional before
   continuous-unidirectional would mean changing two architecturally distinct factors from the
   current ladder's baseline at once — a direct violation of the "one architecturally distinct
   change per rung" discipline ADR-024 has held to since M3 vs M4.
2. **It has a hard structural precondition (M4g or equivalent) that has not yet reported.** Per § 3,
   the current overwrite injection mechanism makes a bidirectional protocol actively self-destructive
   before any adapter-quality question is even reached.
3. **It cannot be scored by the frozen probe in any form.** Every other registered-but-unrun rung
   (M4b, M4e, M4f) is a drop-in mechanism change against the *same* protocol and the *same* 40-item
   evaluation; bidirectional exchange requires a wholesale new evaluation design — comparable in
   scope to run 3's own pre-registration effort (ADR-030), not to a single ladder rung.
4. **It is orthogonal to every diagnosed failure mode** (§ 4) — it does not address off-manifold
   collapse, overwrite-destructiveness, or item-invariance, and plausibly compounds the second of
   those three.
5. **The literature's positive evidence is thin and specific.** One paper, no ablation isolating
   bidirectionality from continuity or from "having a real second model" at all, and a directly
   analogous negative result (GSM8K degradation at a small capability gap) on the exact task this
   ladder also probes.

**Concretely**: finish M4g (running now) and M4f/M4b as already ordered; if the joint ladder
ultimately reports a second full negative result (per ADR-024's own honest-fail framing), name
bidirectional/continuous exchange as a **named future research programme with its own ADR** —
following the precedent already set for ADR-025 (distributed data fabric), ADR-026 (federation wire
contract), and ADR-027 (latent-prefix delivery), each spun out of run 2's context without expanding
run 2's own scope. That ADR's natural scope is **continuous unidirectional injection first** (i.e.,
formalize M4e properly, since it is the one piece of the Bicameral recipe this repo has any existing
infrastructure for), with **bidirectional coupling registered as a follow-on contingent on M4e
actually clearing the (redesigned) probe** — mirroring exactly the sequencing discipline this ladder
already applies to M4c-before-M4d-before-M4f. Recommend the working title
"run-4: continuous and bidirectional latent coupling (Bicameral-class)" rather than treating it as a
run-2 milestone, since it is not an adapter-architecture variant of anything M3-M4g built — it is a
different training and delivery paradigm requiring its own infrastructure, its own VRAM feasibility
scout (joint-training two models' forward/backward, unverified as fitting a 16 GB card, § 3), and its
own pre-registered evaluation design, none of which exist today.

---

## Sources

- The Bicameral Model: Bidirectional Hidden-State Coupling Between Parallel Language Models —
  [arXiv:2605.11167](https://arxiv.org/abs/2605.11167),
  [full text](https://arxiv.org/html/2605.11167v1) (**confirmed-primary**, fetched)
- Latent Collaboration in Multi-Agent Systems (LatentMAS) —
  [arXiv:2511.20639](https://arxiv.org/abs/2511.20639),
  [full text](https://arxiv.org/html/2511.20639v1) (**confirmed-primary**, fetched)
- Cache-to-Cache (C2C) — [arXiv:2510.03215](https://arxiv.org/abs/2510.03215) — already
  primary-verified in `docs/research/038` §4, re-cited not re-fetched this pass
- Cache Merging as a Convergent Replicated State for Multi-Agent Latent Reasoning —
  [arXiv:2607.01308](https://arxiv.org/abs/2607.01308) (**confirmed-secondary**)
- Latent Cache Flow: Model-to-Model Communication Without Text —
  [arXiv:2605.22863](https://arxiv.org/abs/2605.22863) (**confirmed-secondary**)
- Mixture-of-Agents Enhances Large Language Model Capabilities —
  [arXiv:2406.04692](https://arxiv.org/abs/2406.04692) (**confirmed-secondary**)
- Correction and Corruption: A Two-Rate View of Error Flow in LLM Protocols —
  [arXiv:2604.18245](https://arxiv.org/pdf/2604.18245) (**confirmed-primary**, fetched)
- Detection Without Correction: A Two-Parameter Decomposition of Multi-Stage LLM Pipelines —
  [arXiv:2605.27559](https://arxiv.org/abs/2605.27559) (**confirmed-secondary**)
- Do Latent Channels Actually Communicate? A Causal Audit of Latent Multi-Agent LLM Communication —
  [arXiv:2607.26773](https://arxiv.org/abs/2607.26773) — already primary-verified in
  `docs/research/035` §2, re-cited for background, not re-fetched this pass
- Internal: [ADR-024](../adr/024-run2-trained-thought-adapter-ladder.md) (frozen probe protocol,
  M4c/M4d/M4f/M4g registrations and outcomes, receiver-scale confound, M4e registration),
  [docs/research/024](024-live-latent-experiment-design.md) §10 (9.3 GB / 14.3 GB VRAM baseline),
  [docs/research/032](032-injection-configuration-science.md) (injection configuration science, C2C
  placement correction), [docs/research/035](035-probe-task-selection.md) (GSM8K probe-sensitivity
  concerns), [docs/research/036](036-manifold-collapse-across-the-ladder.md) (the two mechanically
  distinct null families), [docs/research/038](038-manifold-constrained-adapter-scout.md) (C2C
  fuse-vs-overwrite, verified from source and from the paper's own equation)

**Not written to any other file. Not committed — per this task's read-only constraint, only this
file was created.**
