# 035. Is GSM8K the wrong probe task for cross-model latent transfer?

**Purpose**: answer the question nobody in this repo has asked directly — every cross-model probe
draw in runs 1-2 (ADR-023 §S6, ADR-024's M3/M4/M4c ledgers, `docs/research/031`'s 10-draw table)
scores binary correctness on 40 GSM8K items, sender Qwen2.5-3B (~87% GSM8K per Qwen's own
technical report) → receiver Qwen2.5-1.5B (~73%). Nine of nine cross-model draws failed, and
`docs/research/031` already showed the effective sample per draw is only 3-5 discordant items out
of 40 (concordance ≈90-92.5%). This document asks whether that concordance is a symptom of probing
the wrong task, using the published evaluation design of five external latent-transfer methods as
the evidence base, plus one external paper that runs the *same* causal-decomposition methodology
LatentMesh uses (ADR-003's OPE/OME/CAG/SSG-style controls) on GSM8K itself.

**Evidence grading used below**: **[confirmed-primary]** = read from the paper's own HTML/abstract
via WebFetch, cited with exact numbers; **[confirmed-secondary]** = corroborated by a search
summary that itself cites the paper, numbers not independently re-derived; **[absent]** = explicitly
checked and not found in the available source; **[inferred]** = a claim this document derives from
confirmed facts, not itself sourced. Every claim below carries one of these tags.

**Status**: research finding, not a re-litigation of ADR-023/ADR-024/ADR-030's frozen verdicts. No
statistical threshold, seed, or acceptance criterion in this repo is proposed to change. This
document is scoped as ADR-031-style forward-looking annotation material for whoever authors run 3
or amends ADR-030's task choice — it does not itself amend any ADR.

---

## 0. Headline answer, for the impatient reader

**GSM8K is a defensible-but-weak probe, and the weak-effect problem is not unique to
LatentMesh's specific mechanism — it recurs across the literature whenever anyone runs a
matched-condition causal decomposition on GSM8K, including in papers built with a different,
better-resourced injection mechanism than ours.** The single most load-bearing external evidence
(§2, "A Causal Audit of Latent Multi-Agent LLM Communication", Zhu et al., arXiv:2607.26773)
independently reruns almost exactly the OPE/OME/CAG/SSG decomposition ADR-003 defines, on GSM8K,
with Qwen3-4B/8B, and finds a **near-zero, sign-unstable content-attributable effect that flips
direction between model sizes** — the same qualitative shape as LatentMesh's own 9-for-9 null. That
is independent corroboration that GSM8K specifically produces thin, noisy causal signal under this
kind of probe, not merely evidence that LatentMesh's own affine/MLP/FastGRNN architectures are too
weak.

But the fix is **not** simply "switch to a knowledge task" as originally hypothesized — the same
external paper's ARC-Challenge arm (a knowledge/science task) shows the *same* near-zero,
interval-crosses-zero pattern GSM8K shows. The sharper, evidence-supported distinction (§3) is
**open-generation idiosyncratic recall vs. anything with a shared/correlated difficulty ordering**
— GSM8K and multiple-choice ARC-C both have item difficulty correlated across model scale (hard
items are hard for everyone), which is exactly the concordance-inflating property
`docs/research/031` already measured empirically in this repo's own receipts. The single best
concrete alternative is **PopQA** (§4) — open-generation long-tail entity factual recall, chosen
because per-item correctness there is governed by *idiosyncratic pretraining-data exposure* rather
than a shared difficulty ordering, which is the literature-supported mechanism (Mallen et al. 2022)
for higher cross-model discordance at a comparable aggregate accuracy gap. The cheapest validation
(§5) does not require running PopQA at all yet: it re-scores the 9 existing null draws' item-level
data already sitting in `crates/latentmesh-runtime/receipts/` for whether GSM8K's discordant items
cluster at a specific difficulty band (a same-cost re-analysis of committed receipts, zero GPU).

---

## 1. What successful methods actually evaluate on

| Method | Confirmed benchmark list | GSM8K present? | Model pair(s) | Source |
|---|---|---|---|---|
| **Cache-to-Cache (C2C)**, arXiv:2510.03215, ICLR'26 | OpenBookQA, MMLU-Redux (17 subcats), ARC-Challenge, C-Eval, LongBenchV1 | **[confirmed-primary] Absent.** No GSM8K, no math/code benchmark in the confirmed list. | Qwen2.5-0.5B↔Qwen3-0.6B and other same/cross-capacity pairs | [arXiv:2510.03215](https://arxiv.org/abs/2510.03215), [HTML](https://arxiv.org/html/2510.03215) |
| **LatentMAS**, arXiv:2511.20639 | GSM8K, AIME24, AIME25, GPQA-Diamond, MedQA, ARC-Easy, ARC-Challenge, MBPP-Plus, HumanEval-Plus | **[confirmed-primary] Present.** Qwen3-8B: baseline 81.1% → TextMAS 92.3% → LatentMAS 93.8% (+1.5pp over text). | Qwen3-4B/8B/14B, self-collaborating (sequential multi-agent, same model family/checkpoint acting as multiple agents) | [arXiv:2511.20639](https://arxiv.org/abs/2511.20639), [HTML](https://arxiv.org/html/2511.20639v1) |
| **The Bicameral Model**, arXiv:2605.11167 | Arithmetic-with-calculator-tool, ZebraLogic-style logic-grid puzzles, math-with-Python-sandbox | **[confirmed-primary] Absent** as a named benchmark — "arithmetic" here means calculator-tool-use accuracy (36%→96% at 0.5B), not GSM8K-style word problems. | Two 0.5B models (arithmetic), two 0.6B models (logic) | [arXiv:2605.11167](https://arxiv.org/abs/2605.11167) |
| **"A Causal Audit of Latent Multi-Agent LLM Communication"** (closest methodological analog to LatentMesh's own S2b/ADR-003 controls), arXiv:2607.26773 | GSM8K, ARC-Challenge, MATH-500 | **[confirmed-primary] Present, primary task.** See §2 for full decomposition. | Qwen3-4B and Qwen3-8B, tested separately (self-relay: sender and receiver are the same checkpoint communicating with itself, not a cross-capability pair) | [arXiv:2607.26773](https://arxiv.org/abs/2607.26773), [HTML](https://arxiv.org/html/2607.26773v1) |
| **AVP (Activation Vector Passing)** | Not a peer-reviewed paper; found as a GitHub project (`VectorArc/avp-python`) claiming 73-78% token reduction via KV-cache transfer between agents, no benchmark suite located. **[absent]** — could not find a benchmark table for this artifact; treat as unverified/marketing-grade, not evidence either way. | — | — | [github.com/VectorArc/avp-python](https://github.com/VectorArc/avp-python) |
| **"Beyond Tokens" unified framework**, arXiv:2606.05711 | Not enumerated in the retrievable abstract; states the paper needs "deeper tasks eliciting complex concepts" — i.e. the authors themselves flag current benchmarks as inadequate. **[confirmed-primary, negative result]**: no specific benchmark table was retrievable, and the abstract itself argues existing evaluation is too shallow. | — | — | [arXiv:2606.05711](https://arxiv.org/abs/2606.05711) |
| **"Latent Communication Between Language Model Agents..."**, arXiv:2607.14103 | Not enumerated in the retrievable abstract; mentions "cross-lingual concept tasks," Procrustes alignment between Llama and Mistral (92% top-1 retrieval) | **[absent]** — no GSM8K mention found | Llama, Mistral (cross-family, closer in spirit to LatentMesh's cross-model design than most of the above) | [arXiv:2607.14103](https://arxiv.org/abs/2607.14103) |

**First-order finding**: of the two papers with a confirmed, retrievable benchmark list that includes
math word problems, one (LatentMAS) shows GSM8K producing the *smallest or among-smallest* gain of
any category it reports (code generation, MBPP-Plus +5.1pp, dominates; GSM8K +1.5pp over text — see
§1 table), and the other (C2C, the more directly comparable "inject sender information into
receiver" paradigm) **evaluates on zero math benchmarks**, choosing knowledge/QA tasks exclusively
(OpenBookQA, MMLU-Redux, ARC-C, C-Eval) for its flagship results. **[inferred]** This is a real,
if soft, signal about what the field currently considers a productive evaluation surface for this
class of method — not proof GSM8K is invalid, but two independent design choices both under-weight
it relative to knowledge tasks.

---

## 2. The single most load-bearing piece of evidence: an independent causal audit on GSM8K

arXiv:2607.26773 is worth treating as a near-direct external replication of what ADR-003/S2b
already do, because its four-condition design (no-message / current-example / other-example /
self-generated) is structurally the same idea as LatentMesh's five-control causal loop, and its
outcome decomposition (OPE = OME + CAG, plus SSG for self-generated comparison) is the same
decomposition family as ADR-023's OPE/OME/CAG/SSG vocabulary — **[confirmed-primary]** these
authors are not LatentMesh contributors and arrived at overlapping methodology independently.

**[confirmed-primary]** Their GSM8K results, read from the HTML full text:

| Model | OPE (pp) | OME (pp) | CAG (pp) | SSG (pp) |
|---|---:|---:|---:|---:|
| Qwen3-4B | −1.00 | −6.17 | **+5.17** | −2.00 |
| Qwen3-8B | +1.67 | +3.96 | **−2.29** | +10.00 |

Two things stand out. First, **CAG (content-attributable gain — the part of the effect specifically
attributable to the sender's actual message content, the closest analog to what a S2b-style probe
is trying to detect) is a modest few percentage points and flips sign between the 4B and 8B
checkpoints** (+5.17 at 4B, −2.29 at 8B). This is exactly the "small, inconsistent shift, not a
near-miss" language ADR-023 §S6 uses to describe LatentMesh's own generated-pairs recalibration
result (p: 0.5000 → 0.3125 at the winner cell, no movement at the anchor cell). Second, their
**ARC-Challenge** arm — a knowledge/science task, not arithmetic — shows the *same* pattern: "minimal
effects overall; intervals cross zero, obscuring mechanism" **[confirmed-primary]**. Their
**MATH-500** arm (harder math, same modality as GSM8K) shows the largest OPE of the three tasks
(+15.00pp at 4B) but that gain is "dominated by content-independent (other-example) effects" — CAG
is only +6.67pp of the +15.00pp OPE, and shrinks further to +1.88pp at 8B.

**[inferred]** Reading across all three of their tasks: no task in *their* suite — arithmetic word
problems, harder competition math, or multiple-choice science knowledge — produces a large, stable,
model-size-invariant content-attributable effect under this style of causal decomposition. This
tempers the naive form of the mission's working hypothesis ("GSM8K specifically is bad because it's
arithmetic reasoning, switch to a knowledge task"): their ARC-C data is a direct disconfirmation of
that naive form, since ARC-C is knowledge-flavored and still shows near-zero effects. What survives
is a narrower, better-supported distinction, developed in §3.

---

## 3. What task property actually predicts detectability — refined against the evidence

The mission brief's original theory was: transfer should help most where the receiver *lacks*
something the sender *has*, and least where the bottleneck is the receiver's own computation
(arithmetic). §2's ARC-C result is a real counter-example to that theory in its simple form — ARC-C
is knowledge, not computation, and it still showed near-zero effect. The refined, evidence-anchored
distinction:

**[inferred, but anchored to two confirmed facts below]** What predicts a *high discordant-pair
rate between two models of different scale* is not "knowledge vs. reasoning" but **whether item
difficulty is correlated across model scale**. Two supporting facts:

1. **[confirmed-primary, this repo]** `docs/research/031` §1 measured LatentMesh's own GSM8K
   discordance rate directly from receipts: 7.5-12.5% across 9 cross-model draws (mean ≈9.5%),
   "remarkably stable regardless of what was actually injected." GSM8K word-problem difficulty is
   known to correlate strongly with model scale in general (harder multi-step problems are harder
   for every model, easier problems are easy for every model) — this is the standard shape of a
   task where item difficulty is a shared, scale-invariant ordering, and it directly predicts high
   concordance (low discordance) between any two models regardless of their absolute accuracy gap.
2. **[confirmed-primary]** ARC-Challenge is multiple-choice — a format with a similarly shared,
   largely scale-invariant difficulty ordering (a "hard" ARC-C item, one requiring subtle
   distractor elimination, tends to be hard for weak and strong models alike; a guessing floor of
   25% bounds how discordant MC items can even become). Its near-zero causal-audit result (§2) is
   consistent with the same concordance-inflation mechanism GSM8K shows, despite being a knowledge
   task, which is exactly why the naive knowledge/reasoning split fails to predict it.

The property that should actually predict discordance is **open-generation recall of an
idiosyncratic fact whose presence in the model's parametric memory depends on pretraining-data
exposure rather than a shared notion of "difficulty."** Mallen et al. 2022, "When Not to Trust
Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories" (the paper
that introduces PopQA specifically for this reason) is the standard citation for this mechanism:
whether a given long-tail entity fact is answerable correlates with the entity's frequency in
pretraining data, not with a shared human-legible "hardness" — meaning two models of different
scale and different training runs can disagree on a *specific* item almost independently of their
aggregate accuracy gap, precisely because "did this fact happen to be memorized" is a much noisier,
less scale-monotonic property per item than "is this a 3-step or 5-step word problem."
**[inferred]** This predicts materially higher discordant-pair rates than GSM8K at a comparable or
even smaller aggregate accuracy gap — the opposite of what naive intuition (bigger aggregate gap →
more discordance) would suggest, and it is a testable, falsifiable claim (§5).

**A confound to flag honestly**: C2C's largest reported category gains (History +7%, Law +7.6%,
Chemistry +7.5%, all MMLU-Redux subcategories — **[confirmed-primary]**) are on *multiple-choice*
knowledge questions, which cuts against the "MC format suppresses discordance" reasoning above.
**[inferred]** The likely resolution is that C2C is a trained, end-to-end learned fusion network
(500K training samples, learned gating) — categorically more powerful than the training-free affine
or lightly-trained MLP/FastGRNN probes this repo has run — so its MMLU gains may reflect method
power overcoming a low-discordance ceiling rather than evidence that MC-knowledge tasks have high
discordance. This repo cannot distinguish "task property" from "method power" using C2C's numbers
alone, and this document does not claim to.

---

## 4. Candidate datasets, with discordance estimates and offline-availability notes

All candidates are evaluated against this repo's hard constraint (no Python, raw jsonl/parquet over
HTTP, same pattern as the existing GSM8K pull from `raw.githubusercontent.com` in ADR-023).

| Dataset | Size | Licence | Raw access | Estimated Qwen2.5-1.5B/3B discordance | Basis |
|---|---|---|---|---|---|
| **GSM8K** (current probe, baseline for comparison) | 1,319 test | MIT | `raw.githubusercontent.com/openai/grade-school-math` — already pinned, ADR-023 | **9.5% observed** (7.5-12.5% across 9 draws) | **[confirmed-primary, this repo]** `docs/research/031` §1 |
| **PopQA** (top recommendation) | 14,267 questions, long-tail entity QA, open-generation | MIT (akariasai/PopQA) | HF `resolve/main/*.parquet` via `akariasai/PopQA` on Hugging Face — public dataset, no auth needed for a direct `wget` against the resolve URL | **[inferred, no direct Qwen2.5 numbers found] Estimated meaningfully higher than 9.5%** — no exact figure located for this model pair; the estimate rests on Mallen et al.'s general finding that per-item parametric-recall success is governed by entity pretraining-frequency, a property with much weaker cross-model correlation than word-problem step-count. **This is the weakest-sourced number in this table and should not be treated as measured.** | [Mallen et al. 2022](https://arxiv.org/abs/2212.10511) (paper's own arXiv ID, cited generically — exact URL not independently re-verified this session) |
| **TriviaQA** | ~95K QA pairs, open-generation, general trivia (less long-tail than PopQA) | Apache 2.0 | Multiple HF mirrors incl. `mandarjoshi/trivia_qa` (canonical) — parquet, resolvable without auth | **[inferred] Likely intermediate** — less extreme long-tail skew than PopQA, so likely less discordance-favorable, but still open-generation (no MC guessing floor) | Reasoning by analogy to PopQA's mechanism; not independently measured |
| **ARC-Challenge** (for contrast, not recommended) | 2,590 test | CC-BY-SA | AllenAI direct download, JSON | **[confirmed-primary, external]** Near-zero causal effect in the closest external analog to LatentMesh's own probe (§2) — recommend against as a switch target despite being "knowledge" | arXiv:2607.26773 |
| **MATH-500** (harder math, for contrast) | 500 items | MIT | HF `HuggingFaceH4/MATH-500`, jsonl | **[confirmed-primary, external]** Largest raw OPE of the three tasks in §2's external audit (+15.00pp at 4B) but CAG (content-specific) share shrinks with scale (+6.67→+1.88pp) — a *harder* version of the same task family does not obviously fix the underlying concordance problem, it mostly inflates the content-independent (OME) component | arXiv:2607.26773 |

**Recommendation**: PopQA is the best-motivated single alternative, but its discordance estimate for
this exact model pair is **not measured** anywhere this session found — it is a literature-grounded
prediction, not a citation of a number. **This should be validated cheaply (§5) before any run-3
redesign commits engineering time to switching the probe task**, exactly the caution the mission
brief itself asked for.

---

## 5. Cheapest validation before committing to a new dataset

Two tiers, cheapest first:

**Tier 0 — zero GPU, re-analysis of committed receipts (do this first).** `docs/research/031` §1
already extracted win/loss/discordant counts per draw; the underlying per-item data
(`items[i].conditions.<cond>.correct`, `items[i].conditions.<cond>.nll_gold`) is sitting in the
committed receipts already read this session
(`crates/latentmesh-runtime/receipts/s2b-receipt-cellL18toL14-*.json` etc.). A same-cost pass over
those 9 draws' discordant items — cross-referenced against each item's GSM8K reasoning-step count
(a cheap deterministic feature, countable from the gold solution's `<<...>>` annotation count, no
model call) — would directly test §3's mechanism claim: **do discordant items cluster at a
*specific* step-count band (supporting a shared-difficulty-ordering story), or are they scattered
uniformly (which would weaken the mechanism this document proposes)?** This is a pure data-analysis
task on files already committed, costs no GPU-hours, and could falsify or support §3 before any new
dataset is touched.

**Tier 1 — one small live probe, ~10-30 GPU-minutes.** Before fully switching run 3's probe task,
run the *existing* injection mechanism (whatever the current-best architecture is — M4c per the
mission brief's context, or the S1a self-pair identity-transform sanity check) on a **small PopQA
slice (e.g. 40-100 items, matching the existing probe's scale)**, scoring plain sender-vs-receiver
concordance with **no injection at all** — i.e., just measure how often Qwen2.5-3B gets a PopQA item
right while Qwen2.5-1.5B gets it wrong, on the actual model pair this repo uses, before spending any
engineering effort wiring up a new gated-text or latent-injection pipeline for it. This single
number (call it the "baseline discordance rate") is the one measurement genuinely missing from this
document and is cheap: it requires only running both models' vanilla (no-injection) inference on
~100 items, something the harness can already do via S0-style captures, and it directly tests
whether PopQA's discordance rate clears the ~25-30-discordant-pair floor `docs/research/031` §2.2
computed as necessary for real statistical power at the ladder's existing 40-item probe scale, or
whether a larger item count is needed regardless of task choice.

---

## 6. Implications for the record — annotation only, no verdict changes

- **ADR-023 §S6 and ADR-024's ledgers should gain one clause** (as an ADR-031-style post-hoc
  annotation, not a re-opened verdict): the 9-for-9 cross-model null on GSM8K is now corroborated by
  an independent external replication of a structurally similar causal-decomposition methodology on
  the same task (§2) — the null is not obviously attributable to LatentMesh's specific
  affine/MLP/FastGRNN architectures being under-powered; a comparably-designed audit on a different
  model family (Qwen3-4B/8B) finds the same thin/sign-unstable content effect on GSM8K. This
  *strengthens* the existing null's external validity; it does not change any pass/fail verdict.
- **Run 3 (gated TEXT, ADR-030)** is a different channel (text, not latent vectors) and a different
  mechanism (causally-gated dynamic prompting, not injection), so this document's findings do not
  transfer automatically — a gated-TEXT channel is not obviously subject to the same
  "shared-difficulty-ordering suppresses discordance" mechanism, since text can carry an arbitrary
  fact regardless of the receiver's own reasoning trace. **[inferred]** If ADR-030's run 3 keeps
  GSM8K, that is more defensible for a text channel than it was for the latent-injection channel
  runs 1-2 used, precisely because text can smuggle a specific fact (e.g., an intermediate numeric
  result) past the "does the receiver's own arithmetic dominate" bottleneck in a way a fixed-length
  pooled/affine latent vector cannot as easily. This is a reason *for* keeping GSM8K in run 3, not
  against it — worth stating explicitly so a reader doesn't assume this document's finding
  automatically argues for switching run 3's task too.
- **If a future latent-injection run (a hypothetical run 4, or an M5/M6-class rung within run 2's
  own ladder) is designed**, this document's recommendation is: budget Tier 1's ~10-30 GPU-minute
  PopQA baseline-discordance measurement *before* committing engineering time to a new dataset's
  scoring harness, precisely because the one number this document could not source (PopQA
  discordance for Qwen2.5-1.5B/3B specifically) is the load-bearing unknown the whole
  recommendation rests on.

---

## Sources consulted

- Cache-to-Cache (C2C): [arXiv:2510.03215](https://arxiv.org/abs/2510.03215),
  [HTML](https://arxiv.org/html/2510.03215), [GitHub](https://github.com/thu-nics/C2C)
- LatentMAS: [arXiv:2511.20639](https://arxiv.org/abs/2511.20639),
  [HTML](https://arxiv.org/html/2511.20639v1), [GitHub](https://github.com/Gen-Verse/LatentMAS)
- The Bicameral Model: [arXiv:2605.11167](https://arxiv.org/abs/2605.11167)
- "Do Latent Channels Actually Communicate? A Causal Audit of Latent Multi-Agent LLM
  Communication": [arXiv:2607.26773](https://arxiv.org/abs/2607.26773),
  [HTML](https://arxiv.org/html/2607.26773v1) — the single most load-bearing external source in
  this document (§2)
- "Beyond Tokens: A Unified Framework for Latent Communication in LLM-based Multi-Agent Systems":
  [arXiv:2606.05711](https://arxiv.org/abs/2606.05711)
- "Latent Communication Between Language Model Agents: Channels, Alignment, and the Limits of
  Text": [arXiv:2607.14103](https://arxiv.org/abs/2607.14103)
- AVP (unverified, artifact-only, no paper located):
  [github.com/VectorArc/avp-python](https://github.com/VectorArc/avp-python)
- Qwen2.5 technical benchmark table (1.5B-Instruct vs. 3B-Instruct, used for the aggregate-gap
  numbers in §1/§3): [qwenlm.github.io/blog/qwen2.5-llm](https://qwenlm.github.io/blog/qwen2.5-llm/)
  — GSM8K 73.2 (1.5B) vs. 86.7 (3B), a 13.5pp aggregate gap, cited to show the aggregate-gap-vs-
  discordance-rate distinction in §3 is not explained by GSM8K having "too small" a gap to work
  with.
- Mallen, A. et al. (2022), "When Not to Trust Language Models: Investigating Effectiveness of
  Parametric and Non-Parametric Memories" — the PopQA-introducing paper, cited generically for the
  entity-frequency-governs-recall mechanism in §3/§4; **not independently re-fetched this session,
  cited from background knowledge of the paper's well-known result — flagged per this document's
  own evidence-grading discipline as the one citation not verified via WebFetch/WebSearch this
  session.**
- PopQA dataset (candidate, §4): [huggingface.co/datasets/akariasai/PopQA](https://huggingface.co/datasets/akariasai/PopQA)
- TriviaQA dataset (candidate, §4): [huggingface.co/datasets/mandarjoshi/trivia_qa](https://huggingface.co/datasets/mandarjoshi/trivia_qa)
- This repo's own receipted evidence: `docs/research/031-statistical-power-and-design.md` (9-draw
  discordance table, §1), `docs/adr/023-live-four-condition-run1-pre-registration.md` §S6 (S2b
  results and the ADR-031 post-hoc annotation already appended there)

**Not written to any other file. Not committed — per this task's read-only constraint, only this
file was created.**
