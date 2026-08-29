# 043. The placeholder-token choice — is `<|fim_pad|>` a channel the receiver was trained to mute?

* **Purpose**: `docs/research/032` flagged and left unchased a specific literature thread —
  "the placeholder-token tradeoff literature, arXiv:2601.05062, for the `<|fim_pad|>` slot mechanism
  specifically" (line 161-172). ADR-024's 2026-08-29 correction (lines 874-925) now makes this thread
  the live question: on-manifold payloads (M3 both variants, M4 all three ranks, run-1's affine
  bridges) are measurably **inert** — aligned NLL within 0.004 nats of baseline, accuracy unmoved —
  while off-manifold payloads (M4c/M4d/M4g) are **actively harmful**, and the harm is content-driven
  (042's own placement/duplication hypothesis was refuted for free: 8 literal zero rows over real
  content cost only 0.025 nats). This document asks whether the receiver simply not attending to /
  not using a well-formed, information-bearing state — inertness, not damage — is explained by the
  specific choice of injection site: 8 copies of `<|fim_pad|>` (`crates/latentmesh-runtime/examples/common/m3.rs`
  line 280, `"<|fim_pad|>".repeat(N_SLOTS)`), a fill-in-the-middle padding token.
* **Date**: 2026-08-29 (branch `feat/run2-thought-adapter`, read-only except this file — not
  committed, per instruction).
* **Method / evidence grading**: **primary** = paper/spec/source fetched (WebFetch) and the specific
  claim read in the fetched text. **inferred** = this document's own reasoning from confirmed facts,
  or a claim reconstructed from an abstract-only fetch. **abstract-only** = the full-text fetch did
  not resolve past the arXiv abstract page despite requesting HTML/full text, noted per-source.
  **WebSearch was unavailable for this entire pass** (session budget exhausted at 200/200 before this
  task started — the same constraint `docs/research/042` and `040` hit) — every external claim below
  is WebFetch against a known URL, not search-discovered.
* **Read first**: `docs/research/032` §2 (first pass at this token, abstract-level); `docs/research/042`
  (the sibling placement hypothesis, now refuted); ADR-024's "MAJOR CORRECTION (2026-08-29)" section
  (the inertness/harm split this document responds to); `docs/research/040` (the pooling gap — the
  other standing structural-difference hypothesis, read together with this one in §5).

---

## Answer, up front

**`<|fim_pad|>` is very likely not "a token the receiver learned to ignore" in the trained-suppression
sense the brief hypothesized — it is more precisely a token the receiver (Qwen2.5-1.5B-**Instruct**,
not Coder) probably never meaningfully saw at all.** The base Qwen2.5 technical report documents no
FIM training anywhere; FIM pretraining is documented only in the **Qwen2.5-Coder** report, as a
continued-pretraining addition on top of the base checkpoint. The four `fim_*` tokens exist in the
receiver's vocabulary (ids 151659-151662, confirmed from its own `tokenizer_config.json`) — Qwen2.5
shares one tokenizer across the whole family — but they are marked `"special": false`, unlike the
chat-control tokens (`<|im_start|>`, vision/image/video pad) that the receiver's own instruct-tuning
demonstrably drives hard. This is a **different and arguably worse** failure mode than "trained to
mute": a *learned* suppression circuit is a real circuit that *could*, in principle, be routed around;
an **experientially near-vacant embedding** sits in a region of the residual stream the receiver has
no dedicated circuitry for at all, trained-to-ignore or otherwise. Both readings predict the same
inertness signature, so this pass cannot distinguish them by NLL alone — but it can name which is more
likely, and design the experiment that would tell them apart.

No surveyed cross-model method (C2C, LatentMAS, Bicameral) injects at a dedicated placeholder or pad
token — all three inject onto or alongside the receiver's **own real token positions** (C2C: residual
fuse onto the receiver's per-token KV cache at the receiver's actual prompt positions; LatentMAS:
prepend the sender's real per-token, per-layer cache; Bicameral: couple live per-token hidden states
at the receiver's current decode position). This is a **third structural difference** from every
working method, alongside pooling (`docs/research/040`) and one-shot-vs-continuous delivery
(`docs/research/032` §2) — and unlike those two, it is the one that specifically predicts *inertness
rather than harm*, which is the one signature this ladder has never yet flipped.

The cheapest decisive experiment is specified in §4: **fuse** (not overwrite) the identical
already-trained M3 payload onto 8 **real, already-present** token positions instead of 8 `fim_pad`
positions, holding payload, slot count, injection depth, and operator fixed. A near-zero-cost,
capture-only (no generation) pre-check is also specified, using code that already exists and already
runs per item (`forward_capture`'s `per_position_l2`, `crates/latentmesh-runtime/examples/common/m3.rs`
lines 292-301) — cheap, not free, since the per-position array is computed but not currently
persisted in receipts.

---

## 1. What is `<|fim_pad|>` in Qwen2.5, and did Qwen2.5-1.5B-Instruct ever see FIM training?

**Vocabulary fact — primary, fetched directly from the receiver's own tokenizer config**
(`huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/raw/main/tokenizer_config.json`). Four FIM tokens exist,
consecutively numbered:

| token | id | `"special"` |
|---|---|---|
| `<\|fim_prefix\|>` | 151659 | `false` |
| `<\|fim_middle\|>` | 151660 | `false` |
| `<\|fim_suffix\|>` | 151661 | `false` |
| `<\|fim_pad\|>` | 151662 | `false` |

For contrast, the chat-control tokens the receiver's own instruct-tuning is built around —
`<\|vision_pad\|>` (151654), `<\|image_pad\|>` (151655), `<\|video_pad\|>` (151656) — are marked
`"special": true` in the same file. `special` governs tokenizer-level behavior (skip-on-decode,
normalization exemption) rather than directly proving training exposure, so this is suggestive, not
dispositive on its own — but it is consistent with the fim tokens being vocabulary-reserved slots the
tokenizer treats as ordinary text rather than control tokens the model's own template machinery is
built to lean on.

**Training-exposure fact — primary, fetched from both technical reports directly (WebSearch
unavailable; both fetches went straight to `arxiv.org/html/<id>`).** The base **Qwen2.5** technical
report (arXiv:2412.15115) contains **no mention of Fill-in-the-Middle, and no mention of any `fim_*`
token**, anywhere the fetch could reach. Its own tokenizer description: "we utilize Qwen's
tokenizer... with a vocabulary of 151,643 regular tokens. We have expanded the set of control tokens
from 3 to 22 compared to previous Qwen versions, adding two new tokens for tool functionality" — FIM
is not among the enumerated control-token additions, and the reported regular-token count (151,643)
plus 22 control tokens does not obviously reserve a named slot for FIM padding as a *trained*
element. The **Qwen2.5-Coder** technical report (arXiv:2409.12186), by contrast, is explicit and
specific: "`<\|fim_prefix\|>`, `<\|fim_middle\|>`, and `<\|fim_suffix\|>` tokens are used to implement
the Fill-in-the-Middle (FIM)... `<\|fim_pad\|>` is used for padding during FIM operations,"
under §3.2.1 "File-Level Pretraining": "the training objectives include next token prediction and
fill-in-the-middle (FIM)," extended in §3.2.2 to repo-level FIM per Lozhkov et al. (2024). Read
together, the natural inference (**inferred**, not directly stated as a negative claim by either
report) is that FIM pretraining is a Qwen2.5-**Coder**-specific continued-pretraining addition on top
of the shared base checkpoint and shared tokenizer, not a base-series-wide objective. The Coder
report's fetch could not resolve whether `fim_pad` is masked from the loss, appears as a real
training target, or which of PSM/SPM format is used (Figure references only, no extractable body
text at the depth this fetch reached) — flagged as unresolved, not asserted.

**What this means for the receiver specifically.** `Qwen2.5-1.5B-Instruct` — the receiver in every
rung of this ladder (`crates/latentmesh-runtime/examples/common/m3.rs` line 27) — is a base-series
instruct model, not a Coder variant. If FIM training is Coder-specific as the two reports' asymmetric
coverage suggests, the receiver's `<|fim_pad|>` embedding is a vocabulary slot that was almost
certainly **never a real training target and rarely-to-never a real training *input*** for this
specific checkpoint — present because the family shares one tokenizer across base/Coder/Math variants
(**inferred** — sharing a tokenizer across variants is standard Qwen practice but this document did
not independently re-verify the claim for 2.5 specifically), not because this model's own pretraining
corpus used it. This is the precise, falsifiable-if-wrong version of the brief's hypothesis, and it is
a meaningfully different claim than "trained to ignore": an embedding that received minimal gradient
signal sits wherever weight initialization plus incidental optimization (weight decay, any output-
embedding tying) left it — not necessarily in a region the receiver's circuitry treats as
"disregard this," just a region nothing downstream was ever shaped to read.

---

## 2. What does the literature say about the choice of injection-site token?

**arXiv:2601.05062, "Compositional Steering of Large Language Models with Steering Tokens"** — the
paper `docs/research/032` named and this brief asked to be chased. **This pass could not get past the
abstract page** on either `arxiv.org/abs/2601.05062` or `arxiv.org/html/2601.05062` (the second fetch
did resolve additional body text via a different rendering path — see below — but neither reached the
"artificial distributional shift" passage `docs/research/032` quoted). What the HTML fetch did surface
(**primary**, body text read): steering tokens here are **newly initialized trainable embeddings**
("we introduce a trainable steering token ⟨b⟩, that is, an embedding e_b ∈ ℝ^d"), initialized
semantically as the mean of the frozen LM's embeddings of the behavior's instruction tokens, operating
in **input embedding space** with the base LM frozen — not existing reserved vocabulary, and not a
mid-layer residual-stream intervention the way LatentMesh's injection is. This paper's own steering
tokens are the opposite of `<|fim_pad|>` on the exact axis this document cares about: **freshly
trained for the purpose**, vs. **incidentally present and probably never trained for any purpose**.
The paper reports an ablation consistent with tokens-need-training-to-matter (without the composition
token `<and>`, accuracy drops from 93-95% to 73.6%), but has no experiment comparing a placeholder
position to an ordinary content-token position, and no inertness-vs-harm framing. `docs/research/032`'s
quoted lines — "isolate steering signals from genuine model computations... require additional input
space and may create artificial distributional shifts if the model hasn't encountered such
placeholders during training" — were read from the arXiv abstract page directly by that pass's fetch
(032 line 164 cites it as "primary, fetched"); this pass's independent re-fetch of the same URL
returned only metadata, not that sentence, so it is **corroborated by an earlier fetch of the same
source but not independently re-confirmed in this pass** — noted rather than silently re-asserted at
full confidence.

**Two closely-related literatures, fetched directly this pass, bear on "does a model spontaneously use
a token position it wasn't trained to use," which is the more precise question underlying the
brief's hypothesis:**

- **Pause tokens** (Goyal et al., "Think before you speak: Training Language Models With Pause
  Tokens," arXiv:2310.02226, **abstract-only** — full text not reached). The finding most relevant
  here: "inference-time delays show gains when the model is both **pre-trained and finetuned** with
  delays" — i.e. pause tokens only help when the model was trained with them from the start;
  inserting them only at inference (into a model never trained with them) is reported elsewhere in
  this literature as **not** reliably helping, which is the abstract's own implicit contrast. This is
  structurally the closest published analogue to LatentMesh's situation: `<|fim_pad|>` is inserted at
  *inference* (well, at frozen-adapter-probe time) into a receiver whose own training likely never
  paired that position with useful signal.
- **Filler tokens** (Pfau et al., "Let's Think Dot by Dot: Hidden Computation in Transformer Language
  Models," arXiv:2404.15758, **abstract-only**). Directly on point: transformers *can* use
  meaningless repeated filler tokens (e.g. `......`) to solve algorithmic tasks they cannot solve
  zero-shot with no intermediate tokens — **but** "learning to use filler tokens is difficult and
  requires specific, dense supervision to converge." This is the single strongest piece of evidence
  this pass found for the brief's underlying mechanism, restated precisely: a token position can carry
  useful hidden computation, but **only if the model was explicitly, densely supervised to route
  information through that position** — a frozen, off-the-shelf model given a novel filler/placeholder
  position for the first time at inference has no established reason to spend attention or MLP
  capacity there. `<|fim_pad|>` in Qwen2.5-1.5B-Instruct is exactly this case: a position never
  supervised for this purpose, receiving well-formed content for the first time at probe time.

Neither pause-token nor filler-token paper compares "dedicated special/placeholder token" against
"ordinary content token" injection sites directly (both use tokens that are either genuinely novel
learnable parameters or genuinely un-special repeated punctuation, not a family-shared-but-unused
special token like `<|fim_pad|>`) — so the analogy is **inferred**, not a direct precedent, but it
converges from two independent angles (steering-token literature's own initialization choice; the
filler-token literature's supervision requirement) on the same conclusion: **novelty of the position
relative to training, not "specialness" of the token per se, is what the literature says predicts
whether a model uses it.**

---

## 3. Do any working cross-model methods inject at a dedicated pad/placeholder token?

**No** — re-confirmed directly against `docs/research/040`'s already-primary-sourced survey, read
here specifically for token-identity-at-injection-site rather than pooling:

- **C2C** (arXiv:2510.03215): the fused object is `𝒞_F = 𝒞_n(X) + ℱ_n(...)` — a residual add **onto
  the receiver's own KV-cache entries at the receiver's own real prompt-token positions** `X`. There
  is no dedicated placeholder token anywhere in the mechanism; every position that gets enriched is a
  position the receiver would have populated with its own genuine token regardless.
- **LatentMAS** (arXiv:2511.20639): prepends the **sender's real per-token, per-layer KV cache** —
  content the sender actually generated at real decode positions — to the receiver's cache. Not a
  placeholder in either direction: the injected positions carry the sender's genuine per-token
  content, and they are prepended (extending the sequence with real cache), not overwritten onto an
  inert slot.
- **Bicameral Model**: couples live hidden states at the receiver's **own current decode position**,
  every step — there is no separate slot at all, placeholder or otherwise.

**This is a third structural difference from every surveyed working method, distinct from and additive
to pooling (`docs/research/040`) and one-shot-vs-continuous delivery (`docs/research/032` §2).** It is
also the first of the three that specifically predicts the signature ADR-024's correction just
isolated — **inertness on-manifold, not harm** — rather than a generic "why doesn't this work" story.
Pooling and one-shot-delivery are structural differences that plausibly explain degraded or noisy
transfer; neither obviously predicts *zero measurable effect within 0.004 nats of baseline across
five separate on-manifold configurations*. A receiver-side circuit (or absence of one) at the specific
injection site does predict exactly that: if the position itself is functionally inert to the
receiver regardless of what's written there, on-manifold payloads should look statistically identical
to baseline — which is precisely what M3, M4 (all three ranks), and run-1's affine bridges show.

---

## 4. Cheapest decisive experiment

**Design.** Reuse M3's already-trained per-token MLP adapter (`crates/latentmesh-runtime/examples/common/m3.rs`,
`Variant::PerToken`) and its already-derived aligned payload — no new training. Change exactly one
variable: the identity of the 8 injected-into token positions, from 8 `<|fim_pad|>` copies to 8
**already-present, ordinary content-token positions**, holding everything else fixed:

- **Slot count**: stays exactly 8 (ADR-028 lists slot count on both the evolvable and the protected
  side of its own boundary — flagged in ADR-024's 2026-08-29 correction, not adjudicated here; this
  design sidesteps the contradiction the same way M4h Stage 1 does, by keeping the count unchanged).
- **Injection depth/site**: unchanged, receiver block 14 (`RECEIVER_BLOCK`), same as every prior rung.
- **Delivery operator**: **fuse** (residual add), not overwrite. This is the load-bearing change beyond
  the token-identity swap itself: overwriting real question-token activations would destroy the
  receiver's own reading of the question, confounding "the position is inert" with "we deleted content
  the receiver needed." `LayerEdit::Fuse` already exists and is already validated as a true no-op for
  the zero-vector control (M4g, 40/40 bit-identical NLL) — the same fuse path used there isolates the
  token-identity variable cleanly, adding the payload on top of the receiver's own real-token state
  rather than replacing it.
- **Payload**: identical aligned vector M3 already produced per item — no re-derivation.
- **Positions**: the concrete recommendation is the **last 8 tokens of `item.question`** itself
  (already present in every prompt, requires no added text, and is the closest available analogue to
  C2C's "fuse onto the receiver's own real prompt positions" without inventing a new phrase whose own
  novelty could reintroduce a placeholder-like confound). A secondary variant worth registering
  alongside it: the 8 tokens of the fixed `ANSWER_FORMAT` instruction text that already follows the
  hint sentence in every prompt (`crates/latentmesh-runtime/examples/common/m3.rs` line 284) — also
  already-present, also ordinary, and lets the two variants bracket "content the receiver must read
  carefully" (question tail) against "content the receiver has seen identically across thousands of
  training-adjacent instruction-formatted prompts" (a fixed formatting instruction).
- **What changes in prompt construction**: `slots = "<|fim_pad|>".repeat(N_SLOTS)` (line 280) and the
  bracketed "stored in these slots: [{slots}]" sentence are removed entirely for this variant — there
  is no reason to keep an artificial "[...]" bracket once the injected-into positions are ordinary
  question tokens; keeping it would reintroduce a textual placeholder cue even if the *token* identity
  changed. `QwenRuntime::placeholder_positions(&inj_tokens, pad_id)` (line 289) is replaced with a
  direct position range computed from the tokenized question (e.g. the last 8 indices of the question
  span before the answer-format text begins), since there is no longer a `pad_id` to search for.
- **What the controls mean under the change**: `zerovec` becomes a true no-op automatically under
  fuse (already established empirically by M4g, 40/40 bit-identical), so it remains a clean baseline
  arm rather than needing redefinition. `random` (norm-matched Gaussian, fused onto real-token
  activations) is now a **perturbation of genuine question content**, not a perturbation of an inert
  slot — this is a **meaning change worth flagging explicitly before the draw**, per ADR-024's own
  fuse-semantics discipline (the same discipline applied to M4g's `aligned_real`/`random`/`zerovec`
  triple): under this design, `random` tests whether *any* norm-matched perturbation of real content
  at these 8 positions measurably moves NLL, which is now a **stronger**, not weaker, comparator for
  the primary `aligned vs random` statistic than it was against an inert placeholder — if `aligned`
  still loses or ties against `random` here, that is more informative than the placeholder-site result
  was, because `random` is no longer "perturbing nothing."

**Near-zero-cost, capture-only pre-check (proposed, not yet run).** `four_conditions`
(`crates/latentmesh-runtime/examples/common/m3.rs` lines 292-301) already calls `forward_capture` on
every item's **un-injected** prompt to get `nat_cap.per_position_l2` — the natural residual-stream L2
norm at every position, computed purely to derive the rescale target (`natural.median`). This array
already exists in memory per item; the receipt only persists the reduced `NormStats` (median etc.,
`crates/latentmesh-runtime/src/norms.rs`), not the full per-position vector, so this is **not
literally free from already-committed receipts** — it requires re-running the existing capture-only
call (no generation, no injection, the cheapest forward pass in the whole pipeline) and reading
`per_position_l2` at the known `fim_pad` positions vs. ordinary content-token positions in the same
prompt, for the same 40 items already used throughout the ladder. If the natural (zero-injection)
residual norm at the 8 `fim_pad` positions is anomalously low, anomalously uniform across items
(low cross-item variance where content positions vary naturally), or otherwise degenerate relative to
real-token positions in the identical prompt, that is evidence the position is already "quiet" before
any payload arrives — obtainable without a probe draw, a training run, or even a generation call.

---

## 5. Honest assessment — does this explain inertness better than the standing alternatives?

**Where it fits well.** Pooling (`docs/research/040`) and one-shot-vs-continuous delivery
(`docs/research/032` §2) are both real, literature-grounded structural gaps from what works — but
neither has an obvious mechanism for producing *exactly* the signature ADR-024's correction isolated:
five separate on-manifold configurations (M3 both variants, M4 all three ranks) landing within 0.004
nats of baseline, essentially indistinguishable from doing nothing. Pooling predicts *degraded,
noisy, or wrong-direction* transfer — a collapsed, anisotropic summary vector should still perturb the
forward pass *somehow* once written into the residual stream, just not usefully. One-shot delivery
predicts transfer that *fades* over the 400-token generation rather than *never registering at all*.
A receiver-side injection site with little to no dedicated circuitry, by contrast, predicts close to
**zero measurable effect regardless of payload content** — which is the one candidate mechanism that
naturally produces "harmless AND inert" rather than "harmful" or "weakly beneficial." That is a better
qualitative match to the specific finding this document was asked to explain.

**Where it is weaker or unresolved.** (1) The FIM-training-exposure claim for the base Qwen2.5 series
is **inferred from an absence** — the base technical report simply doesn't mention FIM — not from a
positive statement that base-series models are never exposed to `fim_*` tokens; an absence-of-mention
is weaker evidence than a stated negative, and this pass could not independently verify pretraining
corpus composition. (2) This document could not distinguish, and flags as an open question, whether
`<|fim_pad|>` is (a) genuinely near-untrained/vacant for this receiver, or (b) present in small
quantities (e.g. via any shared data-mixing across Qwen2.5 variants, or incidental appearance in web
text discussing FIM) and mildly trained-but-unremarkable — both predict inertness, and NLL alone
cannot tell them apart; only the §4 experiment (or a direct embedding-norm/embedding-neighbor analysis
against other rarely-used vocabulary, not attempted this pass) could. (3) The arXiv:2601.05062
"artificial distributional shift" quote that motivated this whole thread could not be independently
re-confirmed in this pass's own fetch (§2) — it stands on `docs/research/032`'s earlier fetch, not a
fresh one. (4) Even if §4's experiment shows injection at ordinary tokens *also* produces inertness,
that would not refute this hypothesis on its own — it would instead point toward pooling or receiver
scale as the dominant factor and placeholder-choice as a real but non-load-bearing structural
difference; conversely, if ordinary-token fuse *does* produce a measurable, non-inert effect (in
either direction), that is a strong positive result for this hypothesis and would make it the leading
explanation of the three standing candidates. This document does not adjudicate which outcome is more
likely — only that the experiment in §4 is cheap enough, and specific enough, to settle it.

---

## Sources

- Qwen2.5-1.5B-Instruct `tokenizer_config.json`:
  https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/raw/main/tokenizer_config.json (primary, fetched)
- Qwen2.5 Technical Report: https://arxiv.org/abs/2412.15115,
  https://arxiv.org/html/2412.15115v2 (primary, fetched; no FIM/`fim_*` mention found)
- Qwen2.5-Coder Technical Report: https://arxiv.org/abs/2409.12186,
  https://arxiv.org/html/2409.12186v2 (primary, fetched; FIM/`fim_pad` confirmed Coder-specific)
- "Compositional Steering of Large Language Models with Steering Tokens":
  https://arxiv.org/abs/2601.05062, https://arxiv.org/html/2601.05062 (primary for steering-token
  mechanism/initialization; the "artificial distributional shift" quote is carried from
  `docs/research/032`'s earlier fetch, not independently re-confirmed this pass)
- "Think before you speak: Training Language Models With Pause Tokens":
  https://arxiv.org/abs/2310.02226 (abstract-only)
- "Let's Think Dot by Dot: Hidden Computation in Transformer Language Models":
  https://arxiv.org/abs/2404.15758 (abstract-only)
- Cache-to-Cache (C2C): https://arxiv.org/abs/2510.03215 (primary, re-cited via `docs/research/040`)
- LatentMAS: https://arxiv.org/abs/2511.20639 (primary, re-cited via `docs/research/040`)
- Repo sources read directly: `crates/latentmesh-runtime/examples/common/m3.rs`,
  `crates/latentmesh-runtime/src/{norms.rs,capture.rs,inject.rs}`, `docs/adr/024`, `docs/adr/028`,
  `docs/research/032`, `docs/research/040`, `docs/research/042`
