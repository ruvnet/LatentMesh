# 049. Adjacent-areas survey — what the run-2 apparatus is actually good for

**Status**: In progress. Four independent lanes; results appended as they land.
**Date**: 2026-08-29.
**Foundation**: [048](048-run2-final-synthesis.md) — activation injection is
semantic at the likelihood level and non-semantic at the decision level.

---

## Lane 1 — likelihood-as-instrument (scoring, reranking, routing): **NEGATIVE**

**Verdict: do not build a scoring or reranking product on this channel.** It is
already solved better, more cheaply, and training-free by reading a
log-likelihood off a plain **text-conditioned** prompt.

### The technique is not new — only our delivery mechanism is

"Read a likelihood as a semantic score instead of steering a decision" is an
established IR tradition, delivered via text rather than activations:

- **UPR** — *Improving Passage Retrieval with Zero-Shot Question Generation*
  (**arXiv:2204.07496**, Sachan, Lewis, Joshi, Aghajanyan, Yih, Pineau,
  Zettlemoyer). **Verified independently by the coordinator** against the arXiv
  API: the abstract states the reranker *"uses a pre-trained language model to
  compute the probability of the input question conditioned on a retrieved
  passage"*, **zero-shot**, requiring *"no domain- or task-specific training"*.
  Reported 6–18pp absolute gains over unsupervised baselines.
- **arXiv:2405.20654** — same family; treats question-log-likelihood-per-passage
  as an established recent pattern.
- Activation patching in IR is positioned as **interpretability, not scoring**:
  arXiv:2504.07898, arXiv:2505.02154, arXiv:2510.06728.
- Closest cross-model analogues: **model stitching** (arXiv:2106.07682;
  arXiv:2603.12433) trains a connector and reads *task accuracy* — the endpoint
  our own result proves is deaf; and **CKA** (arXiv:1905.00414) compares
  activations directly with **zero forward passes through the second model**.

**Nobody has named our exact technique** — but only because the field either
gets the same readout far more cheaply via text, or, for cross-model
compatibility, uses methods needing no forward pass at all.

### Why injection loses, stated bluntly

Both UPR and injection cost **one forward pass per candidate**, so injection is
not worse *there*. It loses on everything else:

| | injection | UPR-style text |
|---|---|---|
| access required | **white-box weights, both models** | log-probs, works over an API |
| per-pair setup | **a trained alignment adapter** | none |
| apparatus | Capture/Inject/Fuse machinery | none |
| evidence | **−0.103 nats, p = 0.032**, one rung, uncorrected, unreplicated | mature, replicated, double-digit pp gains |

Neither beats a bi-encoder dot product on latency — that trade-off is old news
and nothing in our result changes it.

**The one place injection could add something text cannot** is scoring
compatibility between representations where **no text interface exists** —
non-linguistic latents, cross-modal states, control signals. Our proven regime
is two *text-native* LLMs, so that case **is not licensed by anything we have
shown**, and it is not what reranking or routing actually needs.

### What IS novel

**The dissociation itself** — a payload that moves likelihood with extreme
fidelity (−0.773 nats, p = 7.5e-35) while moving decisions no more than noise
(p = 0.72, fully powered, out-of-sample confirmed). The repo's earlier SOTA
sweep ([028](028-sota-continuous-sweep-1.md)) found no paper naming this split for
cross-model transfer.

**So the contribution is the finding, not a product built on the channel.**

### Narrow exception, flagged and not oversold

Same-model injection as an **interpretability/audit** tool — *"does the model's
internal state agree with this candidate claim?"* — is not excluded by the
above. But it is a research/debugging framing, not a serving path, and even
there it competes with cheaper self-consistency and entailment prompting.

### Method note

The researcher's session-wide web-search budget was exhausted (200/200), so all
citations came from direct arXiv API fetches, listed in full for verifiability.
The coordinator independently re-fetched **arXiv:2204.07496** and confirmed
title, authors and the zero-shot no-training claim. **Delivering this negative
early — before any build — is exactly what was asked for.**

---

## Lane 2 — receiver-side adaptation: **MY HYPOTHESIS IS PARTLY REFUTED, and it found the null's likely cause**

### The correction to me, first

I hypothesised the missing factor was *"nobody trained the receiver to listen."*
**The literature refutes the strict form of that.** Two methods move decisions
with a **frozen** receiver whose weights are never touched:

- **Cache-to-Cache** (**arXiv:2510.03215**) — **6.4–14.2% accuracy gain**,
  cross-model, receiver weights unchanged; a Fuser writes into the KV-cache.
- **ITI** (**arXiv:2306.03341**) — no receiver training at all; TruthfulQA
  32.5% → 65.1%. But its direction is probed from the *same model's own*
  representations, so there is no cross-model translation gap to close.

So receiver-weight-training is **not** the shared ingredient. What every working
method shares is **task-loss training of something, across multiple
layers, with learned gating.**

### The finding that probably explains our entire null

> **C2C's own Table 10: reduced to a SINGLE layer, C2C yields ~58.42% → 58.45–58.52%, about 0.1pp.**

**That is our null signature exactly.** Every rung in our ladder injected at a
**single** depth pair. C2C's 6.4–14.2pp requires gated fusion over roughly the
**top 5 of 28 layers**. The published method, restricted to our configuration,
reproduces our result.

**Our null may not be a fact about activation injection. It may be a fact about
single-layer activation injection.**

Corroborating: the **"Beyond Tokens" survey** (**arXiv:2606.05711**) states as
its own design-space boundary that every surveyed latent-communication method
keeps receiver weights frozen — so receiver-parameter adaptation is genuinely
unexplored, but it is *not* what the working methods rely on.
**Flamingo** (**arXiv:2204.14198**) is the one exception that also trains
receiver-side parameters — frozen LM blocks plus new gated cross-attention with
the gate initialised to zero, so it starts as an exact no-op. Its Table 3 shows
*unfreezing* the LM **hurts** (−8.0%). But it needed ~1.8B image-text pairs;
nothing in the literature says what a narrow single-task adapter needs.

### And the rung was already written

`docs/research/034-m5-receiver-side-adaptation-scout.md` (32KB) is a
**ready-to-run pre-registration** for a rank-{1,2,4} receiver-side LoRA.
**Verified: no M5 receipts exist — it was never executed.** The ladder pivoted
to site/pooling/operator ablations and closed before M5 ever drew. Costed from
our own M4c receipts at **~1.2–2.0 GPU-h for all three ranks** — a weekend
build, with candle's gradient-graph pitfalls already solved and receipted.

## COORDINATOR ERROR #16 — I blocked M5X for a reason that does not apply

**I permanently blocked the one rung that tests the factor the literature says
is load-bearing.** My stated reason (research/048, ADR-037):

> *"Both vary payload **content**, which is demonstrably not what moves decisions."*

**M5X's factor #1 is not content — it is multi-layer injection**, described in
ADR-037 as *"the single largest evidence gap this ladder has left untested,"*
with the L24→L19 pair never used by any rung.

**The precise flaw**: PC3 held the delivery architecture **fixed** (single site,
8 slots) and varied content. Its null is therefore *conditional on single-layer
delivery*. I generalised it to "content doesn't matter," then blocked a rung
whose primary variable is **architecture, not content**. C2C's Table 10 is
direct external evidence that the generalisation is wrong: same content, more
layers, 0.1pp → 6.4–14.2pp.

**Consequence: M5X is UNBLOCKED.** M4b's block is untouched — its variable is
receiver scale and its own rationale is unaffected by this.

---

## Lane 3 — low-bandwidth mesh: **send terse text or scores, not compressed thoughts**

Two corrections to my own framing, both verified:

- **The 211-byte budget is optimistic.** LMS1/LMAD envelope tax is ~48B header
  + 52B fixed (+64B if signed), so real content room is **~106B unsigned /
  ~42B signed**.
- **Duty cycle — ADR-019's open row is now resolvable.** EU868 is
  **sub-band dependent**: the L/M band where Meshtastic operates is **1%**
  (not the 0.1%/10% figures the ADR flags as conflicting). At SF11/BW125,
  ~4.18s airtime per full frame gives **≈8–9 packets/hour — one packet every
  ~7 minutes.**

**That kills raw latent transmission on economics alone**, independent of
whether it works: a 1.5–6KB activation needs 8–30 fragments = 33–125s of
airtime = **93–100% of the entire hourly budget on one message.**

**The slot is the scarce resource, not the byte.** A 2-byte codebook index and a
100-byte text message cost the same single slot — so pack score + retrieval key
+ terse text into each rare slot rather than optimising any one field's
compression ratio. Ranked by decision-relevance per slot: routing hint/score
(1–4B) > retrieval key (8–20B) > terse text (the only channel with positive
causal evidence, Δ=+0.512) > VQ codes > raw activations (dead on arrival).

Relevant literature exists but has not merged: wireless VQ semantic comms
(arXiv:2602.15045, 2508.08686, 2510.02646, 2510.18604, 2504.11709, 2606.26398)
is image/sensor reconstruction, not LLM reasoning; the LLM-agent side
(**BANDMAS arXiv:2608.00458**, 53–77% byte reduction while matching accuracy;
arXiv:2607.16133; 2605.25422; 2604.13349; 2608.25277) is routing and selection,
not codecs.

---

## Lane 4 — safety/detection: **the machinery already exists; point it at the text channel**

- **ADR-003's five-control admission test is already implemented in code** —
  verified at `crates/latentmesh-gate/src/causal.rs`:
  `sign_flip_permutation_test`, controls `zero/random/mismatched/self_generated/
  text_equivalent`, *"Admission requires the WORST."* **`mismatched` is
  precisely the independent variable of our misinformation finding.** It is
  offline topology-fitness tooling today; the proposal is to reframe it as a
  **runtime per-message gate**.
- **agentbbs-bridge already signs authorship (Ed25519)** but checks **no content
  plausibility** — a validly-signed *wrong* message passes unchanged.
- **Point detection at the TEXT channel.** Run 3 shows text is what moves
  decisions; a latent-only NLL monitor would defend a surface our own data shows
  is decision-inert.
- **A likelihood check cannot stand alone** — PC3's p=0.72 null means a PASS
  proves content was *read*, not that the outcome is safe. It must be paired
  with ADR-003's ΔV outcome check, and needs a norm-matched null or it is
  trivially defeated by any perturbation of matching magnitude.
- **Our misinformation ordering may be novel.** The general phenomenon is
  documented (**arXiv:2606.16710**, **2506.00509**, **2410.07283** prompt
  infection), but the **three-way ordering** — plausible-wrong **<** random
  noise **<** no message, isolating *plausibility itself* rather than mere
  wrongness — was not found already published. Nearest analogue is the human
  processing-fluency / illusory-truth literature. **Flagged as tentative**
  (arXiv-API-only search reach), and worth writing up separately.

---

# RANKED SHORTLIST — all four lanes closed

Costs are **verified against this repo**, not estimated: GPU figures anchor on
`wall_clock_s` from committed receipts (M4c training 1,604 s; M4i draw 3,397 s
for 300 items × 4 conditions).

## 1. M5X — multi-layer injection · **HIGHEST DECISIVENESS** · ~1.5–2 GPU-h + engineering

**Spec already exists**: [ADR-037](../adr/037-m5x-maximal-configuration-rung.md),
now unblocked with three mandatory amendments.

**Why first**: it tests the one variable C2C's Table 10 identifies as
load-bearing — layer count — and every rung we ran held it at one. A PASS
re-scopes the entire mission's null from *"activation injection is
decision-inert"* to *"single-layer activation injection is decision-inert."*
That is the difference between closing a field and closing a configuration.

**Verified cost drivers:**
- `FuseMany` **does not exist** — must be added to the vendored Qwen2 forward
  pass. Engineering, no GPU. **This is the real cost.**
- L24/L19 dumps **are present** (`sender_L24.tok.f32bin`,
  `receiver_L19.tok.f32bin`) — ADR-037's *"zero new capture required"* claim
  **verified true**.
- One new MLP for the L24→L19 pair, M3's recipe: **~0.45 GPU-h**.
- The draw, anchored on M4i's identical shape: **~0.95 GPU-h**.

**Head-to-head is free**: M4i is the single-layer counterpart on the identical
stream, so layer count is the only difference.

## 2. Misinformation-ordering write-up · **~0 cost** · possibly novel

Our three-way ordering — **plausible-wrong (30.2%) < random noise (34.9%) <
no message (39.5%)** — isolates *plausibility itself*, not mere wrongness, as
the damaging factor. The general phenomenon is documented
(arXiv:2606.16710, 2506.00509, 2410.07283) but **the random-noise-controlled
triangulation was not found published**. Nearest analogue is the human
processing-fluency / illusory-truth literature.

**No GPU, data already collected.** Flagged tentative — the search reach was
arXiv-API-only, so a proper literature check is a precondition, not an
afterthought.

## 3. M5 — receiver-side LoRA · ~1.2–2.0 GPU-h · **prior LOWERED by lane 2**

Spec is ready to run at `research/034` (32 KB) and **verified never executed**
(no M5 receipts). But lane 2 **lowered its prior**: C2C and ITI both move
decisions with a **frozen** receiver, so receiver-weight-training is *not* what
working methods share. It remains genuinely unexplored — the "Beyond Tokens"
survey (arXiv:2606.05711) states receiver-parameter adaptation is outside every
surveyed method's design space — but it is no longer the leading hypothesis.
**Run it after M5X, not instead of.**

## 4. Runtime plausibility gate · engineering only · **needs a new ADR if built**

`latentmesh-gate::causal` already implements the five-control test with
`mismatched` — our finding's exact independent variable — at
`crates/latentmesh-gate/src/causal.rs`. Today it is offline topology-fitness
tooling. Reframing it as a **runtime per-message gate**, pointed at the **text**
channel (where decisions actually move) and paired with ΔV rather than
likelihood alone, is the one product-shaped item here. **The only shortlist
entry lacking a spec.**

## Not proceeding

- **Lane 1 (scoring/reranking)** — closed negative. UPR (arXiv:2204.07496) does
  it better, cheaper, training-free, over an API.
- **Latent payloads over LoRa** — dead on duty-cycle economics before the
  science even matters: at SF11 a 1.5–6 KB activation consumes 93–100% of the
  hourly budget in one message.

**Standing rule for every item above**: ADR-040's power calculation before any
draw, and no accuracy-only endpoint — PC1b/PC2/PC3 proved accuracy can be deaf
while likelihood carries the signal.
