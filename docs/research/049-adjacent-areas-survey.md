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
sweep ([028](028-sota-sweep.md)) found no paper naming this split for
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
