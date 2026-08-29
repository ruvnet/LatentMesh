# 042. The 8-identical-rows injection pattern — is repeated-slot placement itself the pathology?

* **Purpose**: every rung of the ladder (M3, M4, M4c, M4d, M4g, and M4h Stage 1 as it runs) shares
  one unquestioned constant: `InjectionSpec::vectors_tensor` (`crates/latentmesh-runtime/src/inject.rs`
  lines 85-91) takes ONE pooled/last-token vector and broadcasts it via
  `std::iter::repeat(v).take(n).flatten()` into `n=8` **identical** rows, written into 8 **consecutive**
  placeholder token positions (`"<|fim_pad|>".repeat(N_SLOTS)`, `crates/latentmesh-runtime/examples/common/m3.rs`
  lines 35, 280 — a single special token repeated with no separator, so it tokenizes to 8 consecutive
  positions sharing one token id before injection). This document asks whether that placement — 8
  identical residual rows at consecutive positions, a configuration a transformer never encounters in
  natural text — is a plausible root cause of the ladder's central, unexplained anomaly: injected
  content makes the receiver assign **lower** probability to the gold answer, unanimously, **0 wins /
  40 losses** against both a zero-vector control and a norm-matched random control, across M4c, M4d,
  and M4g (task-loss training, on- and off-manifold payloads, overwrite and fuse delivery, MLP and
  FastGRNN architectures — ADR-024 lines 507, 641-642, 726-728).
* **Date**: 2026-08-29 (branch `feat/run2-thought-adapter`, read-only except this file — not committed).
* **Method / evidence grading**: direct repo source-reading (`crates/latentmesh-runtime/src/{inject.rs,models/qwen2_b.rs,lib.rs}`,
  `crates/latentmesh-runtime/examples/common/m3.rs`, `docs/adr/024`) plus WebFetch of primary sources.
  **WebSearch was unavailable for this entire pass** — the session's search budget was already
  exhausted (200/200) before this task started, so every external claim below is either (a) **primary**
  — read directly from a fetched arXiv/transformer-circuits page, quoted, (b) **primary, abstract-only**
  — the full-text fetch returned only abstract-level content despite requesting the HTML full text (noted
  per-source), or (c) **inferred** — this document's own mechanistic reasoning from confirmed facts,
  clearly labeled. No claim below is asserted beyond what its grade supports.
* **Read first**: the teammate brief that prompted this document (reproduced in the task); ADR-024's
  M4c/M4d/M4g outcome sections (lines 473-900); `docs/research/036` and `docs/research/040` (the
  pooling-gap finding, a related but distinct structural difference from what works).

---

## Answer, up front

**Repeated identical rows at consecutive positions is a real, literature-grounded pathology
candidate, but the primary sources fetched this pass do not directly test "duplicate a payload N
times" — they establish the surrounding mechanics precisely enough to make a strong inferred case,
not a directly-confirmed one.** Three fetched primary sources triangulate on the same mechanism from
different angles:

1. **Softmax must spend 100% of its mass every query, regardless of whether anything deserves it**
   (StreamingLLM, quoted below) — this is the load-bearing fact that makes duplication dangerous:
   attention is a zero-sum budget, not an additive resource.
2. **A small number of *fixed, input-agnostic* positions can absorb attention mass disproportionate to
   their semantic content, acting as a constant bias term the model has learned to rely on** (Massive
   Activations, quoted below) — this is a real, naturally-occurring instance of "positions the model
   treats specially regardless of what's actually there," which is structurally what an artificial
   8-identical-row block would also be, except unlearned and out-of-distribution.
3. **No surveyed steering method (ActAdd, CAA, RepE) creates a block of literally-identical injected
   content at literally-identical, mutually-adjacent positions in a single forward pass.** CAA comes
   closest — the same delta vector is added at every position after the prompt — but it is added
   sequentially, one decode step at a time, on top of genuinely different underlying per-token content,
   and even at its most aggressive coefficients CAA's own paper reports only "degradation in quality,"
   never the clean, unanimous 0/40 inversion this ladder observes.

**This predicts the specific shape of the ladder's signature reasonably well**: an artificial,
out-of-distribution attention sink competing for the softmax budget against the actual question tokens
would produce a **content-independent, systematic** drop in gold-answer probability — exactly what
"0 wins across every payload the ladder has tried, including on-manifold, off-manifold, overwrite, and
fuse" looks like. It also predicts the harmless-zero-vector observation **under fuse specifically**,
though for a subtler reason than "zero is a naturally-safe magnitude" (§2.4): under fuse, `c=0` is a
literal identity operation on the receiver's own already-present state (ADR-024 line 897: "verified
empirically at 40/40 bit-identical NLL"), so the zero condition never creates a duplicated block *at
all* — it leaves whatever the receiver's own forward pass would have produced at those 8 positions,
which is a much weaker, much more plausible near-duplicate (the pad token computed through context) than
a rescaled-to-natural-median, high-informativeness payload copied 8 times.

**The single cheapest confirming experiment, runnable with zero new training**: take an already-trained
adapter's output vector, and instead of writing it into all 8 slots, write it into **1** slot and leave
the other 7 slot positions untouched (fuse-mode; equivalently, under overwrite, leave 7 of 8 positions as
their natural pad-token forward pass). If the 0/40 NLL inversion **weakens or disappears** with 1 slot
vs 8, placement (duplication count) is doing real work, independent of content. If it **persists
unchanged** at 1 slot, the pathology is in the payload's content/direction, not the repetition — and the
repeated-slots hypothesis is refuted as cleanly as M4g refuted fuse-vs-overwrite.

---

## 1. What does a transformer do with repeated identical hidden states at consecutive positions?

### 1.1 The load-bearing mechanical fact: softmax is a fixed budget, not an additive resource

**[primary, quoted from `arxiv.org/html/2309.17453`, StreamingLLM, Xiao et al.]**

> "the Softmax operation, which requires attention scores to sum up to one for all contextual tokens.
> Thus, even when the current query does not have a strong match in many previous tokens, the model
> still needs to allocate these unneeded attention values somewhere so it sums up to one."

This is the single fact every downstream argument in this document depends on: attention mass is
**conserved, not conjured**. A position does not simply "receive some extra attention" when it is
duplicated — every unit of attention mass it gains is a unit some other position (in this ladder's
case, plausibly the actual question tokens the receiver needs to read to answer correctly) loses. The
paper further attributes *which* tokens become sinks to visibility, not semantic content: "initial
tokens are visible to almost all subsequent tokens because of the autoregressive language modeling
nature, making them more readily trained to serve as attention sinks" — i.e., a token becomes a sink
because of a **structural property of its position** (how many future queries can see it), not because
of what it means. This is directly relevant to the ladder's slots: 8 consecutive positions sit inside
the prompt, visible to every later prompt token and to the entire generated answer span, i.e.
structurally well-placed to become a sink if their content gives queries a reason to attend there.

**What the paper does not test, disclosed as a gap**: StreamingLLM does not run an experiment that
duplicates a token or a content vector N times and measures the resulting attention distribution. Its
sink discussion is about the model's *learned* preference for specific low-content anchor tokens
(the true first token, or linebreak tokens as a substitute), not about what happens when a
*high-content* vector is artificially copied across several positions the model has never seen behave
that way. Bridging from "sinks exist and are visibility-driven" to "our 8 duplicated payload rows are
sink-like" is this document's own inference (§1.4), not a directly fetched finding.

### 1.2 Duplicate keys mechanically inflate collective attention share — basic softmax arithmetic, not a citation

**[inferred, from first-principles reasoning over the standard attention equation, no single citation
found this pass that runs exactly this experiment]**

If two positions `i` and `j` carry (near-)identical keys `k_i ≈ k_j` (which 8 rows built from the same
source vector, differing only through post-injection RoPE rotation, approximately are — RoPE rotates
identical pre-rotation content by different angles per position, so the 8 keys are related but not
bit-identical after the rotary transform is applied inside attention), then for any query `q`,
`q·k_i ≈ q·k_j`, so the two positions receive **approximately equal** softmax weight. Critically, this
does not mean each individual position's share shrinks to compensate — it means the *combined* share
claimed by the duplicated block scales with how many near-identical copies exist, up to the point where
the block's combined logit-weighted share saturates. A single position with a given content-match score
`s` claims attention proportional to `exp(s)`; eight near-duplicate positions with that same score each
jointly claim approximately `8·exp(s)` before renormalization — an eightfold inflation of that content's
share of the total softmax budget relative to what a single, non-duplicated occurrence of the same
content would have claimed. This is the direct mechanical link between "duplication count" and
"attention budget stolen from everything else," and it requires no new mechanism beyond the softmax
identity already quoted in §1.1 — it is arithmetic, not a separately-discovered phenomenon, which is
why no dedicated citation for it was sought or needed.

### 1.3 Neural text degeneration — repetition literature is about a different failure direction, flagged as a non-fit

**[primary, abstract-only, from `arxiv.org/abs/1904.09751`, Holtzman et al., "The Curious Case of
Neural Text Degeneration"]** The paper documents that machine-generated text is "bland and strangely
repetitive," and that repetition arises from the model's own high-confidence regions of its output
distribution being over-relied-upon (motivating nucleus sampling). **This is the opposite failure
direction from the ladder's finding**: degeneration literature describes a model becoming *more*
confident in repeating already-emitted content; the ladder's anomaly is the model becoming *less*
confident in the correct gold answer when fed a duplicated non-textual payload. This citation is
recorded because the brief asked for a survey of "the neural-text-degeneration literature," but graded
here explicitly as **not a mechanistic fit** — it is evidence that repetition and language models
interact in known, studied ways, not evidence for this ladder's specific inversion.

### 1.4 Induction heads — the closest "repeated sequence" primary source, and what it does not cover

**[primary, abstract/summary-level, from `transformer-circuits.pub/2022/in-context-learning-and-induction-heads`,
Olsson et al.]** Induction heads implement "prefix matching" (attend back to a previous occurrence of
the current token) and "copying" (boost the logit of whatever followed that prior occurrence),
formalized as completing `[A][B]...[A] → [B]`. The paper's own test harness specifically uses "repeated
random sequences of tokens" to isolate genuine induction from surface heuristics — meaning **the
literature's standard tool for probing repeated-content behavior in transformers already relies on
literal duplication**, which lends indirect support to the idea that duplicated content is a
meaningfully different regime worth isolating experimentally, exactly as this document's §4 proposes.
**What the fetched summary does not cover, disclosed as a gap**: it does not describe what induction
heads (or the residual stream more broadly) do when the *duplicated* content is not two separated
occurrences of a single token (the `[A]...[A]` pattern induction heads are built for) but 8
simultaneously-present, mutually-adjacent copies with no distance between them for a "prefix match" to
even be a meaningful operation over. This is a structurally different regime from what induction-head
research characterizes, and this document does not claim induction heads explain the ladder's finding
— only that the induction-head literature's own methodological choice (using literal duplication to
create an unusual, diagnostic regime) is suggestive that duplication is understood field-wide as doing
something structurally different from ordinary content.

---

## 2. Attention sinks and massive activations — do they predict this ladder's exact signature?

### 2.1 Massive activations: fixed, input-agnostic positions that absorb attention as a bias term

**[primary, quoted from `arxiv.org/html/2402.17762`, Sun et al., "Massive Activations in Large
Language Models"]**

> "LLMs use massive activations to allocate substantial attention at certain tokens. These tokens are
> then utilized to form a constant bias term."

> "their values largely stay constant regardless of the input... input agnostic."

> "Attention is Concentrated on Massive Activations... attention is mostly concentrated on the two
> tokens associated with massive activations."

This is a **naturally-occurring**, model-learned instance of exactly the structural role this document
is asking whether the 8 slots accidentally take on: a small number of positions whose activation profile
causes disproportionate attention regardless of semantic content. The paper's ablations sharpen the
picture: **zeroing** massive activations causes "significant degradation in model performance, e.g.,
exploding perplexity," while **setting them to their mean value** causes "negligible changes in
perplexity and zero-shot accuracy." The model has learned to *depend* on these positions carrying a
specific, narrow-range value; removing that value (zero) breaks things, while a value the model
recognizes as "normal for that position" (mean) does not.

### 2.2 Why this is the closest known analog to the ladder's harmless-zero-under-fuse finding — and where the analogy breaks

**[inferred, bridging §2.1 to ADR-024's own reported result]** ADR-024 records, under fuse mode
specifically, that the zero-vector control is "a true no-op, verified empirically at 40/40
bit-identical NLL" (line 897-898) — adding zero changes nothing because fuse is `h[slot] += c·v` and
`c=0` recovers `h[slot]` exactly, algebraically, not just empirically-close. This is a **stronger and
simpler** claim than anything in the massive-activations paper: it is not that zero happens to be a
"safe" magnitude the network tolerates (as mean-substitution is for a naturally-occurring massive
activation) — it is that fuse's zero condition is a mathematical identity, so no experiment is even
needed to predict it. **The massive-activations analogy is therefore illustrative, not load-bearing,
for the fuse case** — it is a real, independently-documented example of "some positions are
special and value-sensitive," but the ladder's own zero-under-fuse result follows from arithmetic, not
from a network having "tolerated" a natural value.

**Where the analogy is load-bearing**: for the **non-zero** conditions. Massive activations show that
LLMs can and do treat a handful of fixed positions as attention magnets that absorb mass "as a bias
term" independent of content — this is direct primary evidence that the *mechanism* this document is
worried about (a block of positions siphoning disproportionate softmax share away from the tokens that
actually need to be read) is not hypothetical machinery invented for this investigation; it is an
already-documented thing transformers do, just normally at 1-2 model-chosen positions rather than 8
externally-imposed ones.

### 2.3 Does this predict the ladder's specific 0/40 signature?

**[inferred]** Partially, and with an important caveat about *scale* the fetched sources don't resolve.
The predicted mechanism (§1.1-§1.2 + §2.1) is: 8 duplicated, natural-median-norm, high-informativeness
rows create a block that competes for attention share against the true question/answer-relevant tokens,
and because the effect is a property of duplication-plus-salience rather than of any specific payload
direction, it should fire **regardless of what the payload vector actually encodes** — consistent with
0/40 holding across on-manifold and off-manifold payloads, MLP and FastGRNN architectures, overwrite and
fuse. What is **not** resolved by any fetched source: whether an 8-fold duplication is large enough, on
its own, to produce a full *inversion* (gold answer becoming *less* likely than baseline, not just
somewhat less certain) rather than a milder confidence dilution. StreamingLLM and the massive-activations
paper both study sinks that are 1-2 positions among thousands of context tokens (Qwen's context here is
much shorter — the probe's own sequence cap is `SEQ_CAP=256` per `docs/research/040`'s citation of ADR-024
— so 8 out of ~256 positions, roughly 3%, is a larger fractional footprint than a typical natural sink's
1-2 out of thousands). Whether 3% of positions, all high-salience and mutually reinforcing, is enough to
flip a gold-vs-wrong-answer comparison specifically (rather than just add noise) is a magnitude question
neither source answers, and this document does not claim more precision than that.

### 2.4 The zero-vector control being harmless is a weaker test of the "artificial sink" hypothesis than it first appears

**[inferred]** A possible objection: "if repetition itself were the problem, wouldn't 8 identical
*zero* rows also be an unnatural repeated block, and therefore also harmful?" Two responses, both
already grounded in facts established above: (a) under fuse, zero is not "8 identical rows of content" —
it is 8 unmodified natural rows, whatever the receiver's own forward pass over 8 `<|fim_pad|>` tokens
happened to produce, which are themselves *already* similar to each other (same token id, only RoPE
differs) in the **unpatched baseline the receiver was presumably at least partially exposed to during
pretraining or generic prompt-following**, i.e., not an out-of-distribution event the way a rescaled,
high-informativeness payload duplicated 8 times is; (b) under overwrite, a zero-vector *would* place 8
literal zero rows into the residual stream — a genuinely unusual configuration — and it would be a
direct test of "does duplication alone, independent of salience, cause harm." **This document does not
have overwrite-mode zero-vector NLL numbers to check that prediction against** — ADR-024's quoted
zero-vector language (line 897) is specific to the M4g fuse rung; the overwrite-mode zero-vector NLL
result, if it exists in the M4c/M4d receipts, was not located in this pass and is flagged as a concrete,
already-available check someone with receipt access should run before the new experiment in §4: **if
overwrite-mode zero-vector NLL is ALSO harmless (unlike what the repeated-content-salience story would
predict, since 8 literal zeros is still a duplicated out-of-distribution block), that weakens this
document's hypothesis; if overwrite-mode zero-vector NLL shows a partial inversion (worse than baseline,
better than a non-zero payload), that strengthens it.**

---

## 3. Do any working methods write one vector into multiple positions?

### 3.1 Survey verdict: no surveyed method creates a same-forward-pass block of identical content at adjacent positions

| Method | Same vector at multiple positions? | Underlying content at those positions | Timing |
|---|---|---|---|
| ActAdd | Abstract-only fetch did not resolve position count (§ gap, below) | unknown from this pass | unknown from this pass |
| **CAA** | **Yes** — "steering vectors are added at all token positions after the user's prompt" (quoted below) | **Different per position** — each position's own real, distinct generated-token hidden state | **Sequential, one decode step at a time**, not one simultaneous block in a single forward pass |
| RepE | Abstract-only fetch did not resolve position count (§ gap, below) | unknown from this pass | unknown from this pass |
| C2C / LatentMAS / Bicameral (per `docs/research/039`/`040`, not re-fetched this pass) | No — per-token, content-distinct transfer at every position; the opposite of broadcast-repeat | distinct per position by construction | one-shot (C2C/LatentMAS) or continuous (Bicameral) |
| **LatentMesh (this repo)** | **Yes** — literally `std::iter::repeat(v)`, bit-identical pre-RoPE content at all 8 rows | **Identical** — same source vector at every one of the 8 rows | **Simultaneous** — one prefill forward pass, all 8 rows present together before any generation happens |

**[primary, quoted from `arxiv.org/html/2312.06681`, CAA]**:

> "steering vectors are added at all token positions after the user's prompt with either a positive or
> negative coefficient"

This is the one surveyed method that does add the *same* vector repeatedly — but the comparison table's
right two columns are where it diverges sharply from this ladder's design. CAA's repeated addition lands
on top of **genuinely different underlying content each time** (each generated token has its own real
hidden state; the steering vector is a shared *perturbation*, not the *entirety* of what's at that
position), and it happens **sequentially across the autoregressive decode**, so at no single point in a
forward pass does the model process a block of several positions whose content is mutually identical
the way it processes this ladder's 8 slots during one prefill pass. **On the magnitude side**, CAA's own
paper reports that pushing the *coefficient* (not the position count) too high causes "degradation in
the quality of the open-ended text" — never a clean, unanimous, gold-answer-specific inversion — which
is weak but real evidence that CAA's regime (shared delta, distinct base content, sequential
application) is a materially gentler perturbation than this ladder's regime (shared *entire content*,
identical base content, simultaneous block).

**Disclosed gap**: the ActAdd and RepE WebFetch calls this pass returned abstract-level summaries only —
neither resolved to the full-text mechanism description requested. ActAdd's known public description
(not independently re-verified this pass, so graded **secondary** at best) is a single-position or
short-prefix addition rather than a full-generation broadcast; RepE's control-vector application is
commonly described in secondary sources as an addition at every layer/position during generation,
similar in spirit to CAA. Neither claim is asserted as primary fact here — both are named as open
verification items rather than filled in from memory.

### 3.2 Verdict

**No surveyed method — confirmed for CAA, C2C, LatentMAS, and Bicameral; unresolved for ActAdd and
RepE — writes literally identical content into multiple simultaneously-present positions within a
single forward pass.** This is a second structural difference from what works, alongside pooling
(`docs/research/040`), and it is related but not identical to it, exactly as the brief's framing
anticipated: pooling is about *what* is sent (one vector summarizing many tokens' worth of content);
repetition-into-multiple-slots is about *how* that one vector is then *placed* (copied into several
positions rather than placed once). A design could fix pooling (send 8 genuinely distinct vectors, per
`docs/research/040`'s own §3.1 recommendation) while still leaving the placement question in §4 below
completely untested, because 8 *distinct* vectors at 8 positions is not a repeated-content block at all
— which is worth naming explicitly, since a reader could otherwise conflate "de-pooling" (040's fix) with
"de-duplicating placement" (this document's question) as the same experiment. They are not: 040's
recommended 8-distinct-slot design is simultaneously a de-pooling fix AND a de-duplication fix, but a
design that only de-pools without changing placement is not conceivable in this codebase's current
mechanism (8 distinct vectors are, by definition, not duplicated) — meaning **040's own recommended
next rung, if it nulls, would not cleanly distinguish whether de-pooling or de-duplication (or both) was
the active ingredient**, which is exactly why §4's cheaper, narrower diagnostic (repetition count alone,
content held fixed) is valuable to run first or alongside it.

---

## 4. Cheap diagnostics — isolating placement from content with no new training

### 4.1 The proposed experiment

**Inject an already-trained, already-frozen adapter's output vector (M4c's or M4d's artifact, content
hash already verified per `docs/research/033`) into 1 slot instead of 8, leaving the other 7 placeholder
positions as their natural, un-injected forward-pass content** (fuse mode: skip the `+=` at those 7
positions entirely; overwrite mode: skip the `slice_assign` at those 7 positions, letting the pad token's
own computed row pass through, per `LayerEdit::Inject`'s own documented empty-`positions`-is-a-no-op
semantics at `crates/latentmesh-runtime/src/models/qwen2_b.rs` lines 47-51 — a 7-of-8-empty `positions`
list is the same mechanism, just partially applied). This requires:

- **No new training** — reuse M4c's or M4d's frozen weights and already-produced adapter outputs.
- **No new data capture** — the same holdout items, same per-token dumps.
- **One code change**: `InjectionSpec::vectors_tensor` currently forces `n = self.positions.len()` copies
  of one vector (`crates/latentmesh-runtime/src/inject.rs` lines 85-91); a 1-slot variant either
  constructs a spec with `positions: vec![single_pos]` (trivial — the existing `LayerEdit` machinery
  already handles arbitrary position lists per-row, confirmed by direct read of
  `crates/latentmesh-runtime/src/models/qwen2_b.rs` lines 81-124: each position gets its own
  `vectors.narrow(0, row, 1)` row, looped, so nothing about the low-level edit assumes 8 rows) or a
  minimal new constructor bypassing the broadcast-repeat helper.
- **One new frozen-probe draw** — same 40 items, same statistics, same sign test, same α, per
  ADR-028's protocol (this is a placement-only change, analogous in scope to M4g's overwrite-vs-fuse
  change, which needed lightweight registration but not a full protocol redesign, per
  `docs/research/040` §3.5's own reasoning about what counts as touching the protected slot-count
  number — 1-of-8 populated slots does not change how many slots *exist*, only how many are filled).

### 4.2 What each outcome would mean

- **0/40 persists unchanged at 1 slot**: placement/duplication is refuted as a contributing cause,
  cleanly, the same way M4g refuted overwrite-vs-fuse. The pathology is in the payload's content or
  direction, not its repetition — strengthening the case that `docs/research/040`'s de-pooling
  hypothesis (content-only) is the more promising remaining lever, and that this document's hypothesis
  should be closed out as a discharged, negative-but-informative branch.
- **0/40 weakens (e.g., 30/10 or better against one or both controls) at 1 slot**: duplication count is
  doing real, partial work — consistent with the softmax-budget mechanism in §1.1-§1.2, and motivating a
  natural follow-up sweep (2, 4, 8 slots populated) to find where the effect turns on, which would be
  the first quantitative dose-response curve this ladder has produced for any hypothesis.
- **0/40 fully disappears (parity or wins) at 1 slot**: strong confirmation that 8-fold duplication
  specifically, not the payload's content, is the dominant cause — the highest-value outcome, since it
  would redirect the entire ladder's remaining budget toward a placement-only fix (e.g., §3.1's
  8-distinct-vectors design from `docs/research/040`, which fixes both pooling and duplication at once,
  would already be justified; but a much cheaper interim fix — inject the existing pooled vector into
  fewer, or non-adjacent, slots — would also become worth testing before committing to a new training
  run).

### 4.3 A second, complementary diagnostic named in the brief: 8 distinct *natural* receiver states

The brief also proposes injecting 8 **distinct**, genuinely on-manifold receiver states (e.g., 8 real
per-token hidden states sampled from the receiver's own forward pass on unrelated content) that carry
**no sender information at all**. If this null-content, non-repeated, on-manifold control *also*
inverts NLL, that would indict the general shape "8 slots, populated with something" independent of both
repetition and content — a different, and more surprising, finding than this document's central
hypothesis, effectively ruling out both pooling (040) and repetition (042) as the operative mechanism and
pointing instead at the 8-slot mechanism or its position-19 injection site itself. This document names
it as a valuable complementary run (it isolates repetition from §4.1 by removing it entirely rather than
reducing it to 1, and isolates content-correctness from §4.1 by using genuinely natural, non-payload
content) but does not rank it above §4.1: §4.1 is strictly cheaper (reuses an existing trained artifact
end-to-end; this variant needs a fresh capture-and-select step for "8 distinct natural states" even
though no training is needed) and answers a more surgical version of the same question. Slot count
itself (asked by the brief to flag, not adjudicate) is untouched by either diagnostic — both keep exactly
8 slots as positions, varying only how many are *populated* and with *what*, which per
`docs/research/040` §3.5's own reasoning about ADR-028's evolvable/protected tension, should not trip
the protected slot-count number either.

---

## 5. Recommendation

**Repeated identical rows at consecutive positions is a plausible, literature-grounded contributing
cause of the NLL inversion, ranked as at least as promising as pooling (`docs/research/040`) and
cheaper to test in isolation** — the mechanism (§1.1's softmax-budget conservation plus §1.2's
duplication-inflates-collective-share arithmetic, reinforced by §2.1's directly-documented case of
transformers learning to treat specific positions as content-independent attention magnets) is grounded
in fetched primary sources, and no surveyed working method (§3) creates this exact configuration —
identical content at simultaneously-present, mutually-adjacent positions in one forward pass. **What
this document cannot claim, honestly**: no fetched source runs the literal experiment "duplicate a
high-salience vector N times and measure downstream task probability," so the connection from
"duplication mechanically inflates attention share" to "this specific 0/40, content-independent
inversion" is this document's own inference, not a confirmed transfer from the literature.

**The single cheapest confirming experiment is §4.1**: reuse M4c's or M4d's already-trained, already-frozen
adapter output, inject it into 1 slot instead of 8 (leaving the other 7 positions as their natural
forward-pass content), and redraw the frozen 40-item probe. Zero new GPU-hours for training, one new
probe draw, and — because it changes exactly one variable (duplication count) while holding payload
content, training procedure, and injection operator fixed — it is the cleanest single-factor isolation
available to the ladder right now, directly analogous in cost and rigor to `docs/research/040`'s own
top-ranked "swap mean-pool for last-token" diagnostic. **Recommend running §4.1 before or alongside
`docs/research/040`'s Stage 1 last-token pilot**, since the two together (repetition count varied
independently of pooling method) would let the next report distinguish which of the ladder's two live
structural-difference hypotheses — pooling, or repeated placement — is doing the work, rather than
conflating them the way a combined "8 distinct vectors" rung (which fixes both at once) necessarily
would.

---

## Sources

- StreamingLLM (Xiao et al.): [arxiv.org/abs/2309.17453](https://arxiv.org/abs/2309.17453),
  [full text](https://arxiv.org/html/2309.17453) (**primary**, fetched this pass, full text; explicit
  quotes on softmax mass conservation and visibility-driven sink formation; duplication/repetition
  experiment confirmed **not present** in this paper)
- Massive Activations in Large Language Models (Sun et al. 2024):
  [arxiv.org/abs/2402.17762](https://arxiv.org/abs/2402.17762),
  [full text](https://arxiv.org/html/2402.17762) (**primary**, fetched this pass, full text; explicit
  quotes on fixed/input-agnostic sink positions, attention-as-bias-term mechanism, zero-vs-mean ablation)
- Neural Text Degeneration (Holtzman et al. 2019):
  [arxiv.org/abs/1904.09751](https://arxiv.org/abs/1904.09751) (**primary, abstract-only**; full-text
  fetch requested but returned abstract-level content; graded as a non-fit for this ladder's failure
  direction, not a supporting citation)
- In-context Learning and Induction Heads (Olsson et al. 2022, Anthropic):
  [transformer-circuits.pub/2022/in-context-learning-and-induction-heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)
  (**primary, summary-level**; prefix-matching/copying mechanism and the paper's own use of repeated
  random sequences as a diagnostic tool quoted; behavior on simultaneously-adjacent duplicates confirmed
  **not covered** by the fetched content)
- CAA / Steering Llama 2 via Contrastive Activation Addition (Rimsky et al.):
  [arxiv.org/abs/2312.06681](https://arxiv.org/abs/2312.06681),
  [full text](https://arxiv.org/html/2312.06681) (**primary**, fetched this pass, full text; explicit
  quote on all-positions-after-prompt application and large-coefficient quality degradation)
- ActAdd (Turner et al.): [arxiv.org/abs/2308.10248](https://arxiv.org/abs/2308.10248) (**abstract
  only** — full-text fetch did not resolve position-count mechanism; named as an open verification gap,
  not filled from memory)
- RepE / Representation Engineering (Zou et al.):
  [arxiv.org/abs/2310.01405](https://arxiv.org/abs/2310.01405) (**abstract only** — same gap as ActAdd)
- Internal: `docs/adr/024-run2-trained-thought-adapter-ladder.md` (M4c/M4d/M4g outcome sections, the
  0/40 NLL numbers and the fuse-mode zero-vector no-op verification this document builds on),
  `docs/research/033-rescale-output-alignment-diagnostic.md` (M4c artifact hash, deployment-rescale
  properties), `docs/research/036-manifold-collapse-across-the-ladder.md`,
  `docs/research/040-the-pooling-gap.md` (the related, distinct pooling finding), source code read
  directly: `crates/latentmesh-runtime/src/inject.rs` (`InjectionSpec::vectors_tensor`'s
  broadcast-repeat), `crates/latentmesh-runtime/src/models/qwen2_b.rs` (`LayerEdit::Inject`/`Fuse`,
  confirming per-row application already supports arbitrary position lists),
  `crates/latentmesh-runtime/examples/common/m3.rs` (`N_SLOTS=8`, the literal
  `"<|fim_pad|>".repeat(N_SLOTS)` construction, `QwenRuntime::placeholder_positions` in
  `crates/latentmesh-runtime/src/lib.rs`)

**Not written to any other file. Not committed — per this task's read-only constraint, only this file
was created.**
