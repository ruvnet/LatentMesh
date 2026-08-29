# Activation Transfer Channel Qualification

**Status:** research synthesis and experiment recommendation, 2026-08-28  
**Scope:** cross-model activation transfer, hidden-state injection, model stitching, shared activation interfaces, and causal qualification of LatentMesh thought adapters  
**Related:** [ADR-024](../adr/024-run2-trained-thought-adapter-ladder.md), [run-1 negative result](025-run1-negative-result.md), [live experiment design](024-live-latent-experiment-design.md)

## Executive decision

Pause M4 FastGRNN until the receiver channel is qualified.

Run a small **M3.5 channel qualification** first: a 40-item self-pair identity probe, matched random and zero controls, a pre-registered gain sweep over the exact current pooled 8-slot delivery path, and a Qwen2.5-3B receiver scale oracle. If a model cannot causally use its own correctly located state, additional cross-model adapter capacity has low expected value. If the 1.5B receiver fails but the 3B receiver passes, register a genuinely cross-model 3B receiver arm before moving the adapter ladder. If both fail, redesign delivery before training another translator.

The research is useful even if activation transfer remains null. The defensible output is a **Causal Activation Interface qualification protocol and compatibility registry**, not a claim of universal model telepathy.

## Evidence grading

| Grade | Meaning |
|---|---|
| A | Peer-reviewed primary paper or accepted conference paper, with sufficient protocol detail to evaluate the claim |
| B | Primary preprint with explicit methods and quantitative results, not yet independently replicated |
| C | Primary code or project evidence without a peer-reviewed outcome, or a narrow result that does not directly test causal task transfer |
| D | Hypothesis, extrapolation, or this project's proposed interpretation |

No paper below is treated as proof of universal latent communication. “Alignment,” “steering,” “state execution,” and “task communication” are distinct outcomes.

## Current project evidence

Run 2 M3 trained a 2-layer 2048 to 512 to 1536 ReLU MLP with approximately 1.84M parameters on 572,424 generated-token pairs from 2,037 leakage-excluded fit items. The regression fit was material: holdout MSE 0.179 and relative residual 0.461, versus 0.843 for the mean-predictor baseline. The causal probe was null:

| M3 path | Aligned | Random | Primary result |
|---|---:|---:|---:|
| Translate each token, then pool | 21/40 | 21/40 | p = 0.6875 |
| Pool, then translate | 22/40 | 21/40 | p = 0.5000 |

Artifact hashes, leakage exclusion, golden-pair reproduction, the fixed probe set, and all integrity gates passed. The anchor cell was correctly left unprobed. The result therefore supports a narrow statement: **a well-fitting nonlinear cross-model regression did not produce measurable task-useful causal transfer through the tested mean-pooled, 8-slot, middle-layer overwrite path into Qwen2.5-1.5B**.

It does not establish that nonlinear translation is generally ineffective. Receiver scale, state position, injection operator, gain, sequence preservation, and receiver conditioning remain unresolved.

M3 variant ii is additionally confounded. The network was trained on per-token inputs and evaluated on a pooled input outside that training distribution. It cannot distinguish information destroyed by pooling from failure caused by out-of-distribution adapter input. Retire this variant as evidence about pooling unless the adapter is retrained directly on pooled inputs.

## What the literature establishes

### 1. Representational alignment is not causal task transfer

The strongest directly relevant result is Piepereit's architecture-dependent transfer study [B]. It separates three levels: representational similarity, retrieval after projection, and causal output change after injection. Learned MLP projections reached 45 to 50% top-1 retrieval in a 20-way pool for three decoder-only model pairs, versus 5% chance. Yet end-to-end injection was significant for only Qwen2-0.5B to Phi-3-mini, at 23.3% versus 0% for a negative control, FDR-corrected p = 0.0469. Both paths targeting Mistral-7B were null despite strong hidden-state retrieval. The authors explicitly scope the positive result to transfer of a “representational vehicle,” not transfer of meaning. This is almost the same failure pattern as M3: regression or retrieval success does not show that the receiver can causally consume the translated state.

Zhang and Xin's Pythia study [B] is a closer negative analogue. A Pythia-160M to Pythia-410M linear bridge achieved normalized cosine similarity near 0.97 across seeds, while downstream multi-hop answers did not improve. Replacement injection was destructive, low-strength additive injection remained near baseline, and receiver-norm rescaling did not rescue it.

The practical invariant is:

> A map can reconstruct the receiver's activation distribution while missing the small, context-dependent directions that the receiver's downstream computation actually uses.

MSE, cosine similarity, CKA, Procrustes fit, neighborhood retrieval, probe reuse, steering, next-token identity, and held-out task uplift must therefore be reported separately.

### 2. State execution is possible, but only under compatibility constraints

Kim, Mun, and Han's Universal Activation Bus [B] is the closest prior art to a reusable activation interface. It trains one linear encoder-decoder pair per model around a shared 3,072-dimensional space. Four source models were trained on 1.42M matched positions; an unseen OLMo-2-7B model was attached by training only its adapter pair. Shared probes, a shared SAE, and a carrier-specific natural-language activation interpreter could then be reused.

Its execution test is important but narrower than LatentMesh task transfer. A final-position activation from model A was translated into model B at normalized depth 0.5 while all other context states came from B. The frozen upper half of B agreed with B's native next token on 72.4 to 87.7% of 1,000 matched prefixes, 7.5 to 27.2 percentage points above native A/B agreement. This demonstrates matched-context state compatibility. It does not show that an independently generated sender message provides new task information to a receiver with a different context.

The Bus also screened candidate models before training and retained only those exceeding 50% recall at 1 in 4,096-way cosine retrieval; retained candidates scored at least 90% while rejected candidates scored at most 5%. A universal interface is therefore a contract for pre-qualified compatible models, not arbitrary architectures.

### 3. Steering transfer is easier than transmitting a reasoning state

Activation steering changes behavior along a low-dimensional, human-selected direction. It does not require reconstructing another model's full reasoning trajectory.

Stolfo et al. [A, ICLR 2025] derive instruction vectors from activation differences and show inference-time control of format, length, and word-inclusion constraints across four models. They also transfer vectors from instruction-tuned models to base models. CAST [A, ICLR 2025] conditionally applies steering based on detected input categories. Oozeer et al. [A, ICML 2025] map activation-space safety interventions across Llama, Qwen, and Gemma for refusal and backdoor mitigation. These works establish that selected causal directions can transfer. They do not establish lossless hidden-state communication.

Agarwal's cross-architecture steering study [B] reports a 71.0% positive-effect win rate for cross-model vectors versus 68.0% for native vectors and 65.7% for a naive baseline across five models and 15 concepts. The 3 percentage point cross-model/native difference is not statistically significant, paired t(4) = 1.85, p = 0.14. The reported discontinuity near 1.7B parameters is based on one model per tier below 7B and is confounded by positional encoding, which the paper acknowledges. It is a valid reason to run a scale control, not evidence of a universal 1.7B threshold.

### 4. Model stitching tests functional compatibility, not just geometry

Bansal, Nakkiran, and Barak [A, NeurIPS 2021] formalized model stitching as a trainable connector between the frozen lower layers of model A and frozen upper layers of model B. Stitching performance can reveal functional compatibility that representational metrics such as CKA miss. Traft [A, UniReps 2026] further shows, in vision, that nonlinear and bottleneck connectors plus spatial reconciliation can stitch distant layers and different architectures.

The lesson for this project is methodological: compare a connector on the downstream computation it is meant to drive. Reconstruction loss is a diagnostic, not the objective. The caveat is domain: image classifiers have fixed feed-forward spatial structure; autoregressive LLMs add tokenizer, position, cache, normalization, prompt, and generation-regime dependencies.

### 5. KV-cache channels currently have stronger task evidence than one-vector overwrites

C2C [A, ICLR 2026] projects and fuses source and receiver KV caches, with learned layer gates. It reports 3.1 to 5.4 percentage points higher task accuracy than text communication and an average 2.5 times latency speedup. Latent Cache Flow [B] compresses and translates joint K/V summaries, reporting a 13 MB adapter and different-context gains over text communication. XKV [B] jointly conditions on both caches, reconciles layer count and KV geometry, and reports results over 45 dataset/model-pair settings; it surpasses text on four of five datasets and reports 6.8 times lower end-to-end latency than text communication.

These results do not invalidate hidden-state injection. They show why the current 8-slot mean-pooled channel is a weak implementation point: KV methods preserve token, layer, key/value, position, and receiver-context structure, then learn where and how strongly to fuse it. If M3.5 cannot qualify a hidden-state channel, a compact gated KV residual is the best-supported architectural pivot.

### 6. Aggregate gains require causal decomposition

Zhang and Emu [B] audit Qwen3-4B and Qwen3-8B latent relays using real, absent, other-example, self-generated, and related controls. They show that aggregate task uplift can combine example-specific content with nonspecific effects, and that those components can reverse across receiver scale. This directly supports ADR-024's frozen controls and argues against treating “accuracy improved” as proof of useful other-agent communication.

## Novelty assessment

### Not novel by itself

| Element | Prior art | Evidence |
|---|---|---|
| Linear or MLP mapping between model hidden spaces | Model stitching, activation intervention transfer, architecture-dependent causal transfer | A/B |
| Injecting a translated state into a frozen receiver | Model stitching, Universal Activation Bus, Piepereit | A/B |
| Shared model-wise activation adapters | Universal Activation Bus, KV-cache shared spaces | B |
| Cross-model steering vectors | Stolfo, Oozeer, Agarwal | A/B |
| Latent multi-agent communication | Interlat, C2C, LCF, XKV | A/B |
| Random, mismatched, self-generated, and absent controls | Zhang and Emu causal audit | B |

### Potentially novel and useful as a system

The following combination was not found in the reviewed primary literature. This is a bounded prior-art statement, not a claim of exhaustive novelty:

1. A **channel-first stopping rule** that qualifies self-pair identity delivery before spending compute on cross-model adapter capacity.
2. A frozen, leakage-excluded probe reused across an ordered adapter ladder, with negative outcomes preserved rather than silently retuned.
3. A compatibility artifact that binds exact model and tokenizer identities, layers, normalization, injection operator, gain envelope, task domain, causal controls, effect interval, safety regressions, and signed execution receipts.
4. Runtime routing that uses that certificate to choose latent transfer only for qualified edges and falls back to structured semantic deltas or text otherwise.
5. Capability enforcement and provenance around an opaque activation channel, rather than treating an adapter file as sufficient authority to write into a model's residual stream.

The valuable product is therefore **activation-edge certification and governed routing**. Successful transfer increases its value, but null results also populate the compatibility graph and prevent repeated waste.

## M3.5: pre-registered channel qualification

### Hypothesis

The current nulls are dominated by receiver delivery mechanics, not insufficient adapter capacity.

### Fixed inputs

- Frozen 40-item probe already used by M3; no item replacement or prompt changes.
- Exact Qwen2.5 checkpoint, tokenizer hash, layer, slot positions, generation settings, and scoring logic from the existing receipts.
- Identity self-pair state captured from the receiver itself.
- Zero vector and random vector matched on norm, position count, dtype, and mechanics.
- Gains selected before any task result is opened: 0.25, 0.5, 1.0, 2.0, and 4.0.
- Delivery arm: the current mean-pooled 8-slot overwrite at the exact registered receiver depth. Sequence-preserving and additive delivery remain separately registered follow-up experiments rather than being mixed into the first channel gate.
- Receiver profiles: Qwen2.5-1.5B at block 14 and Qwen2.5-3B at matched half-depth block 18. The 3B self-pair profile is a scale oracle, not a substitute for ADR-024's still-required cross-model scale-control arm.

### Primary comparison

For each gain and receiver profile, compare identity against matched random injection item by item. Use the exact paired sign or McNemar test over discordant outcomes and report ties. Do not select the best gain and then test it without correcting for the sweep. The committed registration designates gain 1.0 as primary and uses Holm correction for the two adjacent stability gains, 0.5 and 2.0; gains 0.25 and 4.0 are mechanism diagnostics only.

### Proceed gate

Proceed to cross-model adapters on a receiver only if all conditions hold:

| Gate | Required result |
|---|---|
| Task effect | At gain 1.0, identity beats matched random at one-sided paired p < 0.01 on the frozen probe |
| Magnitude | At least 15 percentage points absolute advantage over matched random |
| Stability | Positive effect at two adjacent gains, with no generation collapse |
| Specificity | Identity beats zero and mismatched-item controls; NLL movement alone is insufficient |
| Safety | No material regression on the existing unrelated-output checks |

A shorthand target of 29 favorable items out of 40 corresponds to a one-sample two-sided exact binomial p below 0.01 only when all 40 are informative and ties are absent. The paired discordant count is the actual denominator.

Self-pair performance is not a mathematical upper bound on every learned cross-model direction. An adapter could theoretically land in a more controllable direction than the native state. It is nevertheless the correct economic gate: a receiver that cannot use its own state through the chosen operator offers little reason to fund a more complex translator on the same path.

### Decision table

| 1.5B | 3B | Decision |
|---|---|---|
| Pass | Pass | Continue on the cheaper receiver; use 3B as replication |
| Fail | Pass | Keep M4 blocked on 1.5B, label its nulls receiver-scoped, and preregister a genuine cross-model 3B receiver arm |
| Pass | Fail | Treat the proposed scale threshold as falsified for this path; investigate layer calibration |
| Fail | Fail | Stop M4; redesign delivery or pivot to gated KV/cache fusion |

## Experiment portfolio ranked by information gain per cost

The scores are planning estimates on the current 16 GB GPU setup, not measured runtimes. Information gain is 1 to 5. Cost includes setup and evaluation compute, rounded to GPU-hour bands. The ratio is used only for ordering.

| Rank | Experiment | Question isolated | Information | Cost | Info/GPU-hour | Decision unlocked |
|---:|---|---|---:|---:|---:|---|
| 1 | M3.5 self-pair gain and sham sweep, 1.5B | Can the current channel carry causally useful native state? | 5.0 | 1 to 2 | 3.3 | Stop or retain the entire adapter ladder |
| 2 | M3.5 Qwen2.5-3B scale control | Is the null receiver-scale dependent? | 4.5 | 2 to 4 | 1.5 | Choose receiver before more training |
| 3 | Per-token aligned identity versus 8-slot pooling | Does pooling or slot compression destroy the usable signal? | 4.0 | 1 to 3 | 2.0 | Select sequence-preserving delivery |
| 4 | Pooled-input-trained MLP versus translate-then-pool | Was M3 variant ii merely out of distribution? | 2.5 | 1 to 2 | 1.7 | Produce an interpretable pooling result |
| 5 | Injection operator sweep: replacement, additive, gated residual | Is overwrite erasing receiver context? | 4.0 | 3 to 6 | 0.9 | Select a causal fusion operator |
| 6 | FastGRNN sequence translator after a channel pass | Does temporal token structure improve cross-model transfer? | 3.0 | 4 to 8 | 0.5 | Test the registered M4 hypothesis |
| 7 | Compact KV residual prototype against text | Does a structure-preserving channel produce task and latency value? | 5.0 | 12 to 30 | 0.2 | Decide hidden-state versus KV product direction |

M4 ranks below the controls because it changes the adapter while retaining the suspect delivery path. It becomes high value only after an identity channel passes.

## Stack integration

| Component | Useful role | Required artifact or behavior |
|---|---|---|
| LatentMesh | Transport only qualified latent edges; retain semantic-delta and text fallback | Negotiated channel mode, payload budget, receiver certificate hash, fallback reason |
| RuLLM | Expose deterministic capture and injection hooks | Exact checkpoint/tokenizer/layer identity, position semantics, gain limits, zero/random/self controls |
| RuVector | Store and retrieve compatibility evidence | Graph keyed by sender, receiver, layer pair, operator, task, quantization, and effect interval; local-neighborhood adapters remain experimental |
| Ruflo | Route by measured utility rather than adapter availability | Policy chooses latent, structured delta, or text using certified uplift, latency, byte cost, and risk |
| MetaHarness | Run the ordered ladder and enforce stop conditions | Frozen probe manifest, pre-registration hash, multiplicity correction, promotion/rejection decision |
| RVM | Enforce least privilege around activation writes | Default-deny `ContextInject` capability scoped to receiver, layer, operator, max norm, and expiry |
| RVF | Package adapters and evidence as signed cognitive artifacts | Weights, architecture, model hashes, normalization, allowed gains, causal results, safety regressions, receipts, revocation metadata |
| Core Memory | Preserve authoritative decisions and provenance | Immutable experiment summary, linked receipts, hypothesis status, unresolved confounds, next gate; raw private activations stored separately |

The certificate should be an input to routing, not an assertion by the adapter author. Runtime admission must verify the exact model identity and execution geometry. A certificate for Qwen2.5-1.5B layer 14, one quantization, and one injection operator must not authorize a different checkpoint or operator.

## Security and governance requirements

Activation channels are opaque, high-bandwidth inputs to privileged internal state. They can bypass text moderation, carry training-data remnants, encode prompt injection, or trigger unstable generation without producing an auditable natural-language message.

Minimum deployment controls:

1. Default-deny receiver injection and bind authority to exact model, layer, norm, operator, and certificate hash.
2. Authenticate and integrity-check every payload and adapter; reject replay outside its episode and context hash.
3. Enforce norm, dtype, length, finite-value, and generation-stability bounds before injection.
4. Preserve a structured semantic summary and causal receipt for audit without claiming that the summary fully describes the latent payload.
5. Run matched random, mismatched, zero, self, and text controls during certification; monitor unrelated-task regressions.
6. Keep text or structured semantic deltas as the safe fallback when the edge is unknown, expired, or outside its certified task domain.
7. Treat raw activation stores as potentially sensitive model telemetry. Minimize retention, encrypt at rest, and separate them from Core Memory's durable decision record.

## Success criteria for a game-changing result

Activation transfer becomes strategically differentiating when one implementation satisfies all of the following on held-out tasks:

1. At least 10 percentage points absolute task improvement over receiver alone and matched sham injection.
2. Replication across at least three model pairs and two architecture families.
3. At least 30% reduction in end-to-end reasoning latency or compute versus a quality-matched text relay.
4. Quantized wire payload below approximately 2 KB per transfer, or a measured byte/latency advantage over the text baseline.
5. Less than 1% absolute regression on unrelated capabilities and no generation-collapse increase.
6. Signed, reproducible receipts and a runtime revocation path.

Until then, the honest product claim is **causal compatibility certification for activation interfaces**, not universal thought transfer.

## Falsifiable conclusions

1. **Current evidence:** M3 falsifies the claim that a well-fitting 1.84M-parameter per-token MLP is sufficient on the tested 1.5B receiver path.
2. **Strongest uncertainty:** the channel has not been qualified at n = 40 under self-pair identity injection with a gain sweep and matched sham.
3. **Near-term decision:** M3.5 should precede M4 because it changes the suspected variable at lower cost and higher information gain.
4. **Architecture implication:** if hidden-state delivery fails, preserve sequence, receiver context, and fusion structure through a gated KV residual before increasing translator capacity again.
5. **Stack implication:** package causal evidence and execution authority with the adapter. A regression fit alone must never promote a latent edge.

## Acceptance test

This research is implemented correctly when the frozen 40-item M3.5 run produces a signed receipt for every receiver profile, gain, and control; multiplicity handling is fixed before results are read; and MetaHarness makes the same deterministic proceed or stop decision from those receipts on three independent replays. M4 remains blocked on a receiver unless that receiver passes the pre-registered identity-versus-sham channel gate.

## Primary sources

All URLs below were resolved and title-checked on 2026-08-28.

1. Piepereit, “Architecture-Dependent Causal Transfer of Activation States Across Large Language Models,” arXiv:2608.16347, 2026. https://arxiv.org/abs/2608.16347
2. Kim, Mun, and Han, “One Adapter Pair per Model: A Universal Activation Interface for Language Models,” arXiv:2608.09521, 2026. https://arxiv.org/abs/2608.09521
3. Agarwal, “Cross-Architecture Steering Transfer in Language Models: A Systematic Empirical Study,” arXiv:2608.05164, 2026. https://arxiv.org/abs/2608.05164
4. Zhang and Emu, “Do Latent Channels Actually Communicate? A Causal Audit of Latent Multi-Agent LLM,” arXiv:2607.26773, 2026. https://arxiv.org/abs/2607.26773
5. Zhang and Xin, “A Negative Result on Cross-Model Activation Transfer in a Pythia Multi-Hop Setting,” arXiv:2606.03280v2, 2026. https://arxiv.org/abs/2606.03280
6. Stolfo et al., “Improving Instruction-Following in Language Models through Activation Steering,” ICLR 2025. https://proceedings.iclr.cc/paper_files/paper/2025/hash/8c3262a4c965ba9888f120d4f9e13478-Abstract-Conference.html
7. Lee et al., “Programming Refusal with Conditional Activation Steering,” ICLR 2025. https://proceedings.iclr.cc/paper_files/paper/2025/hash/e2dd53601de57c773343a7cdf09fae1c-Abstract-Conference.html
8. Oozeer et al., “Activation Space Interventions Can Be Transferred Between Large Language Models,” ICML 2025. https://proceedings.mlr.press/v267/oozeer25a.html
9. Bansal, Nakkiran, and Barak, “Revisiting Model Stitching to Compare Neural Representations,” NeurIPS 2021. https://proceedings.neurips.cc/paper/2021/hash/01ded4259d101feb739b06c399e9cd9c-Abstract.html
10. Traft, “Bridging Large Gaps in Neural Network Representations with Model Stitching,” UniReps 2026. https://proceedings.mlr.press/v322/traft26a.html
11. Huh et al., “Position: The Platonic Representation Hypothesis,” ICML 2024. https://proceedings.mlr.press/v235/huh24a.html
12. Fu et al., “Cache-to-Cache: Direct Semantic Communication Between Large Language Models,” ICLR 2026. https://proceedings.iclr.cc/paper_files/paper/2026/hash/474ada926b331d78f06d95e8913111cc-Abstract-Conference.html
13. Dery et al., “Latent Space Communication via K-V Cache Alignment,” arXiv:2601.06123, 2026. https://arxiv.org/abs/2601.06123
14. Rossi, Raghunath, and Wu, “Latent Cache Flow: Model-to-Model Communication Without Text,” arXiv:2605.22863v2, 2026. https://arxiv.org/abs/2605.22863
15. Liu et al., “Dual-Cache Latent Space Communication between Heterogeneous Language Models,” arXiv:2608.20617, 2026. https://arxiv.org/abs/2608.20617
16. Du et al., “Enabling Agents to Communicate Entirely in Latent Space,” ACL 2026. https://aclanthology.org/2026.acl-long.1248/
17. Karvonen et al., “Activation Oracles: Training and Evaluating LLMs as General-Purpose Activation Explainers,” arXiv:2512.15674v2, 2026. https://arxiv.org/abs/2512.15674
