# Research: what would push LatentMesh beyond current state of the art

* Purpose: primary-source input for a future ADR (023+) proposing beyond-SOTA experiments.
* Date: 2026-08-28
* Scope: external 2024-2026 literature in latent agent communication, cross-model alignment,
  causal verification of communication, semantic radio, and federated world models — diffed
  against ADR-001's own prior-art survey and the repo's current implementation.
* Method: WebSearch across five sub-areas, WebFetch on the highest-signal papers to check
  method/claims beyond abstract, cross-referenced against ADR-001 §3/§10's citation list and
  ADR-009's second-pass table. Every load-bearing claim below carries an evidence grade:
  **primary** (paper/repo fetched and read), **inferred** (search-result summary only, not
  independently fetched), **uncertain** (could not be located/confirmed despite searching).

---

## 0. Headline correction to ADR-001/009's own citation list

ADR-001 §10 and ADR-009 both flag their prior-art citations as "requester-supplied, not
independently verified." This research pass independently verified most of them and found two
that could **not** be located:

- **"StateBridge"** (ADR-001 §2, dated 2026-08-13, the training-free orthogonal hidden-state
  alignment paper the ADR leans on to justify `latentmesh-align`'s Procrustes approach) — not
  found under that name in arXiv, OpenReview, or general search across multiple query
  phrasings. **Uncertain — possibly a different title, a non-indexed source, or miscited.**
  This matters because ADR-002's whole justification for training-free orthogonal alignment
  cites this paper as the closest analog; the closest *verified* analog is actually
  "Platonic Representations in the Human Brain" (§2c below), which does the same
  unsupervised-orthogonal-rotation trick but between human fMRI subjects, not LLMs.
- **"E2 Explainer"** (ADR-001 §3, ADR-009 §1, dated 2026-08-13, the causal-attribution paper
  that ADR-009 says "took" the causal-attribution novelty claim four days before ADR-001 was
  written) — not found under that name. **Uncertain.** The closest verified match doing
  essentially that job is Zhang & Emu's causal audit (§3a below), dated 2026-07-29, which
  *predates* ADR-001 by three weeks, not four days — so the underlying worry ADR-009 raises
  (someone already published channel-masking causal attribution for latent MAS) is **confirmed
  true**, just attached to a different, correctly-dated paper.

Net effect: ADR-001/009's map of the threat is directionally right even where a specific
citation doesn't resolve. The rest of this document cites only sources independently found.

## 1. Where external SOTA is now, by claim area

### 1a. Direct latent/activation communication between LLM agents

| Claim | Evidence |
|---|---|
| **Cache-to-Cache (C2C)**: a trained neural projector + learnable gating fuses one LLM's KV-cache directly into a different LLM's KV-cache, no intermediate text. 6.4-14.2% higher accuracy than individual models, 3.1-5.4% over text communication, 2.5x speedup. Oracle ablations suggest cache *content* (not just presence) drives the gain, though no formal content-vs-presence significance test is described. ICLR 2026, Fu et al., Tsinghua/SJTU. | **Primary** (fetched). arXiv:2510.03215 |
| **LatentMAS**: agents generate "latent thoughts" via last-layer hidden embeddings and share a common latent working memory (KV-cache), zero-training. +14.6pp accuracy, 70.8-83.7% fewer output tokens, 4-4.3x faster. ICML 2026 Spotlight, live public repo with code. | **Primary** (repo + abstract confirmed). github.com/Gen-Verse/LatentMAS, arXiv:2511.20639 |
| **AVP (Agent Vector Protocol)**: a shipped, open, binary wire protocol (not just a paper) for KV-cache/hidden-state transfer. Same-model: direct transfer. Same-family, different size: vocabulary-mediated projection. Fully incompatible: automatic JSON fallback. 73-78% fewer tokens, 2-4x faster; on Llama-3.2-3B specifically, 2.1x faster and +5pp accuracy at 74% fewer tokens. Works today with HuggingFace Transformers and vLLM. | **Primary** (repo + benchmark doc fetched). github.com/VectorArc/avp-spec, avp-python |
| **KVCOMM**: cross-context KV-cache reuse with context-aware offset correction for multi-agent pipelines; ~6.7x prefill speedup at 3 agents on one H100. NeurIPS 2025. | **Inferred**. arXiv:2510.12872 |
| **DroidSpeak**: KV-cache sharing for cross-LLM serving. NSDI 2026. | **Inferred**. arXiv:2411.02820 |
| A dedicated taxonomy survey ("Beyond tokens", Liu, arXiv:2606.05711) catalogs **18 methods (2024-2026)** along three axes — WHAT is sent (embeddings/hidden-states/KV-caches), WHICH alignment (space + layer), HOW fused (concat/prepend/math-op/cross-attn/cache-restore) — with a companion GitHub awesome-list. This is the survey ADR-001 §3 references as "a July 2026 survey reportedly catalogs 18 representative latent-communication methods" — confirmed real, correctly cited in substance. | **Primary** (fetched). arXiv:2606.05711, github.com/enochliu98/Awesome-Latent-Communication |

**Verdict**: this is now a crowded, well-funded category with at least three groups (Tsinghua/SJTU, Gen-Verse, VectorArc) shipping working code against real heterogeneous or same-family models, not simulations. LatentMesh has run **zero** live experiments in this category. This is the single biggest gap between LatentMesh's claims and the field's evidence bar.

### 1b/2. Cross-model representation alignment — direct competitor to `latentmesh-align`

| Claim | Evidence |
|---|---|
| **vec2vec** ("Harnessing the Universal Geometry of Embeddings", Jha/Zhang/Shmatikov/Morris, NeurIPS 2025): translates embeddings between *arbitrary* encoders **with zero paired data** — no calibration pairs, no known correspondence — via a CycleGAN-style adversarial architecture built on the Strong Platonic Representation Hypothesis. | **Primary** (fetched). arXiv:2505.12540 |
| **mini-vec2vec** (2026): scales the same unpaired universal-geometry alignment using cheaper *linear* transformations instead of adversarial training. | **Primary** (fetched abstract). arXiv:2510.02348 |
| **"Platonic Representations in the Human Brain"** (2026): recovers a shared geometry across independently-trained fMRI encoders using **unsupervised orthogonal rotation**, no paired cross-subject samples — methodologically the closest verified relative of `latentmesh-align`'s Procrustes/SVD approach, just applied to neuroscience, not LLM hidden states. | **Primary** (fetched abstract). arXiv:2605.20496 |
| **Relative representations** (Moschella et al., ICLR 2023, arXiv:2209.15430): the foundational "zero-shot latent space communication" result — anchor-based relative coordinates make different latent spaces directly comparable/stitchable without training. Older (2022) but the conceptual root of this entire line, with a maintained repo (`lucmos/relreps`). | **Primary** (fetched). arXiv:2209.15430 |

**Verdict — this is the sharpest concrete gap found in this pass.** `latentmesh-align` requires ~16-64 *paired calibration examples* to fit its orthogonal transform (see README's own benchmark table). The vec2vec lineage (2505.12540, 2510.02348) already solves the harder, more general version of this problem — alignment with **zero paired data** — and does so at production embedding-model scale. `latentmesh-align`'s contribution (the `O(d²n)` fast-path reformulation, 430-964x speedup over dense SVD at LLM hidden dims) is a genuine, verified engineering result, but it optimizes a weaker starting assumption (paired calibration exists) than what the field's current best already does without that assumption. If real cross-model calibration pairs are hard to source for the one-vertical experiment, this is a direct risk, not just a novelty gap.

### 3. Causal verification of communication value

| Claim | Evidence |
|---|---|
| **"Do Latent Channels Actually Communicate? A Causal Audit of Latent Multi-Agent LLM Communication"** (Zhang & Emu, 2026-07-29): the paper that best matches ADR-001 §5's "July causal audit." Controlled message replacement at the sender→receiver boundary: an **other-example** (mismatched-task) condition and a **self-substitution** condition decompose aggregate accuracy change into "generic channel presence" vs. "example-specific content" components. On GSM8K/Qwen3-4B: overall −1.00pp decomposes into −6.17pp from presence alone vs. +5.17pp from real content — and **both signs flip at 8B scale**. On MATH-500: +15.00pp splits 8.33/6.67 between the same two components. Core finding: **aggregate accuracy cannot tell you whether a channel's content specifically mattered — you need the decomposition.** This is a **one-shot audit methodology**, not a continuously-running admission gate, and it does not attach an authority/execution consequence to the result. | **Primary** (fetched). arXiv:2607.26773 |
| **MANTA** (2026-07): self-evolving topology at inference time — bounded structural updates to roles/links/order/visibility/validation while preserving task interface and budget. +5.8pp over strongest baseline across 5 benchmarks (info-seeking, tool-use, planning, workflow, math). Live-tested on real models, not simulation. | **Primary** (fetched). arXiv:2607.28527 |
| **"When Latent Agents Lie: KV-Cache Integrity in Multi-Agent LLM Collaboration"** (2026-06-27): a malicious specialist injects a poisoned KV-cache into a split-evidence reasoning pipeline. Tests visible-verifier filtering, latent-state quarantine, and learned sanitizers — most fail against adaptive white-box attacks; only transport authentication (MAC-gated fail-closed rejection) gives a concrete positive systems boundary. **Directly relevant to whether ADR-003's causal test is adversarially robust**: a poisoned/substituted cache is structurally similar to this paper's `mismatched` control — meaning ADR-003's gate might catch this attack class as a side effect of testing for causal value, which nobody in this literature has claimed or tested. | **Inferred** (search summary; not independently fetched in full). arXiv:2606.28958 |
| **LCGuard**: adversarial KV-cache transformation to strip reconstructable sensitive information before transmission — privacy, not integrity. Confirms ADR-001's citation. | **Inferred**. arXiv:2605.22786 |
| **"Out of Sight, Not Out of Mind"**: latent-channel attacks in latent-based MAS generally — broader threat-model survey adjacent to the above. | **Inferred**. arXiv:2605.28214 |

**Verdict**: ADR-003's actual distinguishing claim survives this pass — nobody found ties a causal decoy test to a **live, rolling, authority-consequential admission gate**. Zhang & Emu's audit is a research diagnostic run once per paper; MANTA adapts topology on live signal but not on a formal counterfactual test; the KV-integrity papers propose static/one-shot defenses, not a continuously re-verified trust ceiling. This is real, defensible whitespace **as of this pass** — but ADR-009's own "months, not years" window estimate looks right: Zhang & Emu (causal decomposition) and MANTA (live continuous topology adaptation) are both real, both dated within three weeks of each other, and combining them is an obvious next paper for a well-resourced group.

### 4. Semantic/goal-oriented communication over radio

| Claim | Evidence |
|---|---|
| DeepSC lineage (LSTM/Transformer joint source-channel coding for text) remains the dominant reference architecture cited across 6G semantic-comms surveys; the field frames the shift as "bit-centric → meaning-aware, goal-oriented." | **Inferred**, well-established prior art (pre-2026). |
| **3GPP concretely timed 6G standardization** at its June 2026 Singapore plenary: study phase through 2026, **Release 21 specs span March 2027-end 2028**; ITU-R IMT-2030 coordinates 3GPP/ETSI. Semantic communication is explicitly named as a Release-21-era topic, **not yet standardized**, and current work targets cellular/O-RAN infrastructure, not amateur/embedded bands. | **Primary** (fetched search synthesis, cross-checked against 3GPP news page). |
| No paper or product found combining: bounded (\<256-byte) deterministic-critical-facts-first envelope + learned-assist bounded by CRC/FEC fallback + amateur/LoRa/HF/VHF physical layer + cross-language (C/embedded + Rust) golden-vector conformance. The DeepSC/6G lineage targets cellular infrastructure and joint neural source-channel coding as the *primary* signal path, with no deterministic fallback discipline resembling ADR-010's invariant 2 ("learned latents may carry residual evidence but never silently replace a critical symbolic value"). | Search found no counter-example across multiple query phrasings — treat as **inferred absence**, not proven absence. |

**Verdict**: LatentMesh Air's niche (amateur/embedded radio, bounded deterministic facts, learned-assist strictly subordinate) is genuinely unoccupied — it sits below and orthogonal to where 6G/DeepSC standardization is aimed, and it is years ahead of the earliest possible 6G semantic-comms deployment (2028+ at best). This is LatentMesh's cleanest, least-contested whitespace, and it is also the one area where the repo already has portable-C infrastructure and cheap real hardware ($60 LoRa boards) sitting on this machine's shopping list.

### 5. Agent memory/world-model federation

| Claim | Evidence |
|---|---|
| **FedWorld**: "scope-aware federation of agent world models" — exchanges structured, checked abstract transition rules across clients instead of naive parameter aggregation, explicitly because a client's unchecked rules may not generalize. This is the paper ADR-001 §10 already cites by name, confirmed real and substantively matching the citation. `latentmesh-federation` (ADR-007/017) implements essentially the same idea (bounded `TransitionRule`s + decoy-controlled local-scope validation) — **duplicative territory**, differentiated (if at all) only by LatentMesh's reuse of the causal permutation test for scope validation, which FedWorld's own scope-checking may or may not do (not independently confirmed). | **Primary** (fetched search synthesis). arXiv:2608.01561 |
| **"Cache Merging as a Convergent Replicated State for Multi-Agent Latent Reasoning"** (Baquero & Brito, 2026-07): applies **CRDT semantics** (commutative, associative, idempotent, causally consistent merge) directly to KV-caches/latent states across agents — convergence without coordination. This is a capability `latentmesh-memory`'s fidelity continuum (raw→compressed→prototype→rule) does **not** have: there is no conflict-free merge algebra for latent state arriving from multiple, possibly-partitioned Radio nodes. Directly actionable gap for ADR-005/007. | **Primary** (fetched). arXiv:2607.01308 |
| AgentCRDT / `crdt-merge` — engineering tools applying CRDT merge to agent state generally, not latent-specific but confirms the pattern is becoming a recognized building block. | **Inferred**. |

**Verdict**: FedWorld is close enough to `latentmesh-federation` that the "we're first" framing in ADR-007/017 does not hold up; the CRDT-merge idea (Baquero & Brito) is a genuinely missing piece LatentMesh could adopt rather than compete against.

## 2. LatentMesh's defensible novelty vs. where it's behind — scored adversarially

| Area | Verdict | Why |
|---|---|---|
| Continuous causal re-verification gating **execution authority** (not a one-shot research audit) | **Ahead, still defensible** | Nobody found combines Zhang & Emu's decoy-decomposition methodology with a live, rolling admission gate that caps authority (`ObserveOnly`→`ActionInfluencing`) and re-tests on drift. This is ADR-003/008's actual claim and it survives this pass. |
| Five-control test including `text_equivalent` as a mandatory baseline | **Ahead, narrow** | The causal-audit literature found uses 2-4 controls (other-example, self-substitution, noise/scramble variants) but nothing found requires beating a *text* baseline specifically as a condition of admission — LatentMesh's framing ("is latent better than text," not just "is signal present") is a real, if small, differentiator. |
| Causal gate as incidental defense against KV-poisoning (mismatched-control overlap with "When Latent Agents Lie"'s attack class) | **Ahead, unclaimed by anyone** | This is a genuine, currently-unexploited connection between two literatures (causal verification and KV integrity) that neither side has published. Untested even inside this repo — no adversarial-injection test exists in `latentmesh-gate` today. |
| Deterministic-critical-facts-first radio envelope with learned-assist strictly bounded by CRC/FEC, targeting amateur/LoRa bands | **Ahead, clean whitespace** | 6G semantic-comms work targets cellular infrastructure years out; nobody found is doing this at ham/embedded scale with this invariant structure. |
| Cross-language (C11 + Rust) golden-vector conformance discipline and staged simulation/hardware evidence labeling (ADR-014) | **Ahead on rigor, not on results** | This staging discipline is unusually careful compared to the academic papers (which report single benchmark numbers without a hardware-acceptance ladder), but rigor without a passed hardware gate is not itself a result — see Honest Status. |
| `latentmesh-align`'s paired-calibration orthogonal Procrustes | **Behind** | vec2vec/mini-vec2vec already solve the harder zero-paired-data version of this problem; `latentmesh-align`'s real contribution is a speed optimization on a *weaker* assumption, not new alignment capability. |
| Raw latent/KV-cache handoff between real heterogeneous models | **Behind, by a lot** | Cache-to-Cache, LatentMAS, and AVP all have live numbers on real models today; LatentMesh has none. `latentmesh-bench` explicitly measures only wire bytes and alignment wall-clock, never task accuracy. |
| Self-evolving communication topology | **Behind on evidence, competitive on design** | MANTA already shows +5.8pp on live benchmarks with inference-time bounded structural adaptation; `latentmesh-evolve` is deterministic simulation only (ADR-018's own status line says so). |
| Federated world models over compatible transition rules | **Duplicative** | FedWorld already does this; `latentmesh-federation`'s differentiator (reusing the causal permutation test for scope validation) is real but narrow, and unconfirmed against FedWorld's own method. |
| CRDT-style conflict-free merge for latent memory across partitioned nodes | **Missing entirely** | Baquero & Brito already published this pattern; LatentMesh's memory continuum has no analog. |
| Standalone latent wire format (`LatentFrame`) as a competing protocol | **Correctly not claimed** | ADR-001 §3 already concedes this to AVP; this pass confirms AVP is a shipped, working implementation with real fallback semantics, which strengthens rather than weakens that concession. |

## 3. Beyond-SOTA proposals, ranked

### #1 — Run the one-vertical live experiment, in a form directly comparable to Zhang & Emu's methodology, and publish the harness

**What**: Wire `latentmesh-core` + `latentmesh-align` + `latentmesh-gate` into an actual generation loop between two real open-weight models available on this host today: `llama3.2:3b` (fast, local) and `gpt-oss:20b` (via ollama, local RTX GPU or the tailnet Mac once the ADR-150 blocker clears). Reproduce ADR-009 §5's four-condition comparison (`StaticText`/`DynamicText`/`DynamicLatent`/`CausalDynamicLatent`) on a **GSM8K-style task subset**, deliberately mirroring Zhang & Emu's exact evaluation setup (same benchmark family, same decomposition into "channel presence" vs. "example-specific content" components) so the result is a **direct, citable comparison point** against a real published paper rather than an isolated internal number.

**Why it clears SOTA**: this is the single thing every competitor is missing. Zhang & Emu ran a one-shot audit; MANTA adapts topology without a formal counterfactual test; Cache-to-Cache/LatentMAS/AVP have live numbers but no continuous causal gate. Running the exact ADR-003 five-control test (including `text_equivalent`) on real models, continuously, with authority consequences, and reporting it in Zhang & Emu's own decomposition language, is a genuinely new combination nobody has shown.

**Requires**: RTX-class GPU (available), ollama with `llama3.2:3b` + `gpt-oss:20b` (available per host inventory), a GSM8K-subset harness (new, small effort — the causal-audit paper's setup is public enough to approximate), and closing ADR-009 §6's own named blocker (dense SVD too slow at dim=4096 — use the already-shipped `O(d²n)` fast path, or restrict to a smaller/truncated hidden-dim slice for the first run, exactly as ADR-009 §6 already recommends).

**Measurable claim**: ">80% of edges admitted by `CausalDynamicLatent` individually survive Zhang & Emu-style mismatched/other-example decoy controls on a rolling re-verification window across N task batches, at compute or latency at least 20% below `StaticText`" — this is literally ADR-009's own combined acceptance test, just finally run.

**Lands in**: ADR-009's experiment plan; touches `latentmesh-core`, `latentmesh-align`, `latentmesh-gate`, `latentmesh-stream`, `latentmesh-memory`.

### #2 — Causal gate as a demonstrated defense against KV-cache poisoning

**What**: Replicate a "When Latent Agents Lie"-style attack (a malicious specialist substitutes a poisoned/mismatched KV-cache into a split-evidence pipeline) inside `latentmesh-gate`'s existing test harness, and measure whether ADR-003's `mismatched` control — designed to catch task-irrelevant state, not built as a security control — catches the poisoned injection as a side effect, without any purpose-built defense.

**Why it clears SOTA**: neither literature has published this connection. If the causal gate catches poisoning at a meaningful rate, this is a defense-in-depth result nobody claimed; if it doesn't, that's an equally publishable negative result about the limits of causal verification as a security boundary, and it directly strengthens or corrects ADR-008's honest "risk score is a placeholder" admission.

**Requires**: the models/harness from #1, plus a small adversarial-injection test suite modeled on the attack description in arXiv:2606.28958 (worth reading in full before implementing — this pass only fetched a search summary).

**Measurable claim**: detection/rejection rate of injected poisoned caches under the existing `mismatched` control, reported against the attack paper's own reported false-negative rates for its visible-verifier and learned-sanitizer baselines.

**Lands in**: `latentmesh-gate::causal`, ADR-003, ADR-008.

### #3 — Real LoRa hardware run for LatentMesh Air

**What**: Buy the two ~$60 LoRa boards already scoped in the mission brief, run ADR-014's stage-gate ladder for real (conducted modem → over-the-air channel), and produce actual BER/PER/semantic-reduction numbers instead of the current deterministic-simulation-only evidence.

**Why it clears SOTA**: this niche (bounded deterministic-facts-first semantic radio on amateur/embedded bands) has zero external competition found in this pass — 6G semantic-comms work is 2+ years from any deployment and targets cellular infrastructure, not $60 hobby boards. Being first to *any* real RF evidence here, however modest, is a genuine claim nobody else is racing for.

**Requires**: $60 hardware (per mission brief), the already-implemented portable C codec (`c/**`), ESP32 or SDR bridge work already scoped in ADR-011/013.

**Measurable claim**: ADR-014's own staged thresholds — semantic reduction ≥10x at equivalent task accuracy, ≥0.99 critical-WorldGraph agreement, on a real conducted or over-the-air link, with a 95% bootstrap CI, on at least one named blind channel condition. Even partial progress against gate 1 of ADR-014's table (conducted modem) would be the first non-simulated result in the Air stack.

**Lands in**: ADR-010/011/013/014.

### #4 — A public benchmark others can compete on, seeded by #1's harness

**What**: Package #1's four-condition experiment (task corpus, model pair, causal-gate harness, evidence-label discipline from ADR-014) as a submittable benchmark, in the spirit of AgentBench/HAL, where a submission is a new alignment method or gating policy scored on the same causal-survival metric.

**Why it clears SOTA**: none of the academic papers found (Zhang & Emu, Cache-to-Cache, LatentMAS, MANTA) ship a standing, re-runnable, third-party-submittable benchmark for *causally-verified* latent communication specifically — they report one-shot numbers in a paper. LatentMesh's own staged evidence-label discipline (simulation vs. hardware, ADR-014) is unusually well-suited to formalizing this as a benchmark contract.

**Requires**: #1 done first; otherwise this is a benchmark with no reference result.

**Measurable claim**: n/a directly — the claim is the existence of a reproducible, adversarially-submittable leaderboard, evaluated by adoption (external submissions) rather than a single number.

**Lands in**: ADR-014, a new `harness/latent-bench` (or similar) alongside the existing `harness/evolve` and `harness/air`.

### #5 (defensive, lower priority) — Adopt unpaired alignment as a fallback to close the vec2vec gap

**What**: Add a vec2vec/mini-vec2vec-style unpaired alignment path to `latentmesh-align` as an alternative to the current paired-calibration Procrustes fit, for cases where calibration pairs are unavailable (which will be the common case for arbitrary heterogeneous models in the wild, not the curated case the current benchmark table assumes).

**Why it matters**: this is not a beyond-SOTA claim — it's closing a gap identified in §2 before it becomes a liability in the #1 experiment, where sourcing real calibration pairs between `llama3.2:3b` and `gpt-oss:20b` may itself be nontrivial.

**Requires**: implementing a linear (mini-vec2vec-style) or adversarial (vec2vec-style) unpaired fit as a second `AlignmentTransform` variant; the existing fast-path SVD infrastructure and test discipline (`fast_path_matches_dense_reference`) is directly reusable as a correctness harness for the new path.

**Lands in**: `latentmesh-align`, ADR-002.

## 4. Publishability

There is a venue-worthy claim here, but it is narrower than "LatentMesh the system" — it is specifically **#1 combined with #2**: a continuous causal-verification control loop, run on real heterogeneous open models with Zhang & Emu-comparable methodology, that (a) reproduces and extends a causal-decomposition result on a live authority-gated system rather than a one-shot audit, and (b) shows an incidental security property (poisoning resistance) that neither the causal-verification nor the KV-integrity literature has connected. That combination, with real numbers, is a plausible NeurIPS/ICML workshop paper or an MLSys systems paper — not because any single piece is unclaimed (none are), but because the *combination*, continuously running with execution consequences, is what ADR-009 always said was the actual bet, and this pass didn't find anyone who has done it.

The minimum experiment that supports the claim is exactly proposal #1: two real open models, GSM8K-subset task, the existing `latentmesh-core`/`align`/`gate` code wired into one real generation loop, four conditions, reported in the same decomposition language Zhang & Emu used. Everything else (the LoRa hardware run, the public benchmark, the vec2vec fallback) strengthens the story but is not required to produce the first defensible result.

## Sources

- Cache-to-Cache (ICLR 2026): https://arxiv.org/pdf/2510.03215
- KVCOMM (NeurIPS 2025): https://arxiv.org/pdf/2510.12872
- DroidSpeak (NSDI 2026): https://arxiv.org/pdf/2411.02820
- LatentMAS (ICML 2026 Spotlight): https://arxiv.org/abs/2511.20639, https://github.com/Gen-Verse/LatentMAS
- AVP / Agent Vector Protocol: https://github.com/VectorArc/avp-spec, https://github.com/VectorArc/avp-python
- vec2vec / "Harnessing the Universal Geometry of Embeddings" (NeurIPS 2025): https://arxiv.org/pdf/2505.12540
- mini-vec2vec: https://arxiv.org/pdf/2510.02348
- "Platonic Representations in the Human Brain": https://arxiv.org/abs/2605.20496
- Relative representations (ICLR 2023): https://arxiv.org/abs/2209.15430
- "Do Latent Channels Actually Communicate? A Causal Audit..." (Zhang & Emu, 2026-07-29): https://arxiv.org/abs/2607.26773
- MANTA (2026-07): https://arxiv.org/abs/2607.28527
- "When Latent Agents Lie: KV-Cache Integrity in Multi-Agent LLM Collaboration": https://arxiv.org/pdf/2606.28958
- LCGuard: https://arxiv.org/abs/2605.22786
- "Out of Sight, Not Out of Mind": https://arxiv.org/pdf/2605.28214
- FedWorld: https://arxiv.org/abs/2608.01561
- "Cache Merging as a Convergent Replicated State for Multi-Agent Latent Reasoning" (Baquero & Brito, 2026-07): https://arxiv.org/pdf/2607.01308
- "Beyond tokens: a unified framework for latent communication in LLM-based multi-agent systems" (Liu): https://arxiv.org/abs/2606.05711, https://github.com/enochliu98/Awesome-Latent-Communication
- 3GPP 6G Release 21 timeline: https://www.3gpp.org/news-events/3gpp-news/partner-pr-6g
- ADR-001, ADR-003, ADR-009, ADR-010, ADR-014 (this repo, `docs/adr/`)
