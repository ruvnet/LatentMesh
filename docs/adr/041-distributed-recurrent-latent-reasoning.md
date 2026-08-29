# 041. Distributed recurrent latent reasoning over LatentMesh

Status: Proposed

> **Renumbered 023 → 041 before merge (2026-08-29).** This ADR was authored
> against `main`, where 023 was free. It is not: `main` now carries
> **023 — Pre-registration for the live latent-exchange experiment (run 1)**,
> and the run-2 branch adds **024–040**. 041 is the next free number.
>
> ### ⚠️ Evidence this ADR was written without
>
> Run 2 concluded after this ADR was drafted
> ([research/048](../research/048-run2-final-synthesis.md)). Two of its findings
> bear directly on the design below and **must be read before Phase 1 is built
> on it**:
>
> 1. **"Require identical model family and checkpoint on both sides" does NOT
>    de-risk the transfer.** That is precisely the case run 2 tested. PC1b, PC2
>    and PC3 were **same-model, same-item, identity-transform** positive
>    controls — and all three were **decision-inert** (PC3: p = 0.72, n_disc 68,
>    fully powered, out-of-sample). Same checkpoint is not the safe starting
>    point this ADR assumes it is.
> 2. **What DID move was the likelihood, not the decision** — −0.773 nats toward
>    the specific answer encoded (p = 7.5e-35), while accuracy moved no more
>    than a norm-matched Gaussian. So the acceptance test below
>    ("match baseline task accuracy via latent deltas") is measuring the
>    endpoint run 2 found deaf, and **must co-report the likelihood arm against
>    a norm-matched random control** or a null will be uninterpretable.
>
> **This does not refute the design.** A recurrent architecture's latent state
> is its *own native workspace*, not a foreign activation pushed into a frozen
> transformer, and every run-2 rung injected at a **single layer** — where
> Cache-to-Cache's own Table 10 ablation (arXiv:2510.03215) also collapses to
> ~0.1pp. But the burden of proof sits higher than this ADR currently assumes,
> and ADR-040's **mandatory power calculation** applies to every experiment it
> proposes.

Date: 2026-08-29

Owners: LatentMesh, RuVector, RVM, MetaHarness maintainers

Related: [001](001-latentmesh-architecture-and-prior-art.md), [003](003-causal-edge-verification.md), [008](008-capability-governed-execution.md), [010](010-latentmesh-air-protocol.md), [014](014-benchmark-and-acceptance-method.md), [015](015-live-midstream-latent-streaming.md), [016](016-ruvector-persistent-latent-memory.md), [018](018-metaharness-darwin-topology-loop.md)

## 1. Decision summary

LatentMesh will add a governed recurrent latent reasoning layer in which a node can:

1. Update a persistent task context from demonstrations or observations without changing model weights.
2. Perform multiple internal reasoning iterations in a continuous latent workspace before decoding language or actions.
3. Allocate more or fewer recurrent iterations according to uncertainty, task structure, risk, and marginal utility.
4. Persist useful recurrent context and latent trajectories through RuVector.
5. Transmit compact, compatibility checked reasoning deltas between nodes through LatentMesh.
6. Admit a remote latent delta only if provenance, compatibility, integrity, policy, and causal value checks succeed.
7. Keep all latent state non authoritative. RVM and RVF remain the final execution boundary.

The design is inspired by the architectural pattern reported by BDH-CQ, which separates recurrent contextual memory from iterative latent reasoning. This ADR does not attempt to reproduce BDH-CQ internals because the paper explicitly states that exact dimensions, update rules, and the complete training recipe are proprietary [1]. We instead adopt the public system level separation and define an independent LatentMesh implementation contract.

The target system is:

```text
observations or demonstrations
          |
          v
   RuVector retrieval
          |
          v
 recurrent context S
          |
          v
 latent workspace H
          |
   recurrent iterations
          |
          +------------------------------+
          |                              |
          v                              v
 local verifier                  latent delta encoder
          |                              |
          v                              v
 symbolic output                    LatentMesh
          |                              |
          v                              v
      RVM gate                    remote node S,H
          |                              |
          +--------------+---------------+
                         v
                    governed action
```

## 2. Why this decision exists

Most current agent systems pay for reasoning through generated text. A model produces tokens that represent intermediate thought, those tokens are serialized, stored, transmitted, tokenized again, and then consumed by another model or a later step. That creates three avoidable costs:

1. Token generation latency.
2. Repeated serialization and tokenization.
3. Bandwidth spent on linguistic form rather than task sufficient state.

BDH-CQ demonstrates a different system level organization. Demonstrations update recurrent memory:

```text
S_t = U_theta(S_(t minus 1), D_t)
```

After demonstrations are ingested, the query initializes a separate latent workspace:

```text
H_0 = E_theta(x_star, S_K)
H_(r plus 1) = F_theta(H_r, S_K)
y_hat = G_theta(H_R)
```

The paper explicitly assigns different roles to these states: S changes as evidence is encountered and supports in context learning, while H carries the ongoing computation used to answer the current query [1]. The public 150M parameter configuration reports 29.5 percent pass at 2 on ARC AGI 1 at approximately 0.85 H200 GPU seconds and a computed cost of 0.00070 USD per task [1].

The paper also reports that increasing latent reasoning effort improves accuracy, while lower effort reduces cost. HIGH reports 29.5 percent pass at 2, MEDIUM 27 percent, and LOW 21 percent, with reported cost reductions of 0, 11, and 22 percent relative to HIGH [1]. This supports treating recurrent depth as an explicit runtime budget rather than a fixed architectural constant.

A separate line of work, Coconut, feeds a model hidden state back as the next input embedding instead of decoding it to language, showing that continuous thought can represent multiple candidate continuations and can outperform tokenized chain of thought on selected reasoning tasks [2]. Recurrent depth models similarly show that test time compute can be increased by repeatedly applying a shared block instead of generating longer reasoning traces [3].

LatentMesh already treats hidden state as a transportable and governable object. ADR 016 provides RuVector backed persistent latent memory. ADR 015 provides streaming latent frames. ADR 003 requires causal counterfactual validation. ADR 008 makes execution default deny. The missing layer is a precise contract for recurrent reasoning itself and for transporting a compact delta of that reasoning state between nodes.

## 3. Scope

This ADR defines the architecture and algorithms for:

1. Recurrent task context.
2. Iterative latent reasoning.
3. Structural demonstration retrieval.
4. Adaptive reasoning budgets.
5. Latent reasoning delta extraction.
6. Same model and cross model compatibility checks.
7. Remote latent merge.
8. Causal admission.
9. Persistence and replay.
10. Governance and fallback.

This ADR does not claim:

1. Reproduction of BDH-CQ.
2. Compatibility with proprietary BDH latent states.
3. Cross model latent interoperability without a learned or verified adapter.
4. That latent communication is always better than text or symbolic messages.
5. That opaque latent state may bypass RVM policy or RVF provenance.
6. That ARC benchmark efficiency transfers automatically to language, tool use, robotics, RF sensing, or production agents.

## 4. Architectural state model

Each reasoning session has five logically distinct states.

### 4.1 Canonical world state W

W is the deterministic, inspectable state used for critical facts and action policy. WorldGraph, LMAD symbolic fields, signed RVF artifacts, and RVM capability state live here.

Latent state may suggest updates to W but cannot mutate W directly.

### 4.2 Recurrent context S

S is a compact context dependent state updated as demonstrations, observations, tool results, or recalled prototypes arrive.

Public BDH-CQ defines the abstract update:

```text
S_t = U_theta(S_(t minus 1), D_t)
```

LatentMesh defines the interface rather than a single implementation:

```rust
trait ContextUpdater {
    fn update(
        &self,
        previous: &ContextState,
        evidence: &Evidence,
        budget: &ContextBudget,
    ) -> ContextUpdate;
}
```

A conforming ContextUpdate must return:

```text
new_state
parent_hash
result_hash
model_fingerprint
adapter_fingerprint
confidence
reconstruction_error if compressed
provenance_refs
```

### 4.3 Latent workspace H

H is an ephemeral or checkpointable reasoning workspace. It is not canonical memory and it is not authority.

```text
H_0 = E_theta(query, S)
H_(r plus 1) = F_theta(H_r, S)
```

A verifier evaluates each iteration and may stop early.

### 4.4 Reasoning delta DeltaH

DeltaH is the smallest transmitted representation that produces measurable downstream task value relative to controls.

It is not defined as raw tensor subtraction in all implementations. A codec may use sparse residuals, learned bottlenecks, low rank projections, vector quantization, prototypes, or symbolic summaries, provided reconstruction error and causal utility are measured.

### 4.5 Canonical action A

Actions are generated only after a latent or symbolic result crosses the governance boundary:

```text
execute(A) iff
    provenance_valid(A)
    and signature_valid(A)
    and capability_allows(A)
    and risk(A) < threshold
    and policy_allows(A)
```

Remote latent state never satisfies this predicate by itself.

## 5. Algorithm 1: recurrent context update

### 5.1 Goal

Absorb new task evidence into a bounded recurrent memory without appending an unbounded explicit prompt or KV cache.

### 5.2 Abstract algorithm

```text
INPUT
    previous recurrent state S
    evidence item D
    model theta
    retention coefficient lambda
    update budget B

OUTPUT
    updated recurrent state S_prime
    update receipt R

PROCEDURE
    e = encode_evidence(D)
    candidate = U_theta(S, e, B)

    if candidate contains NaN or Inf
        reject

    novelty = 1 minus similarity(candidate, S)
    consistency = verify_against_canonical_state(candidate)

    if consistency fails
        reject or quarantine candidate

    S_prime = normalize(candidate)

    R = hash(
        parent_state_hash,
        evidence_hash,
        result_state_hash,
        model_fingerprint,
        adapter_fingerprint,
        update_budget,
        novelty,
        timestamp
    )

    return S_prime, R
```

### 5.3 Bounded memory option

For implementations that need explicit retention control, use a gated update:

```text
G_t = sigmoid(g_theta(S_(t minus 1), D_t))
C_t = U_theta(S_(t minus 1), D_t)
S_t = normalize((1 minus G_t) elementwise S_(t minus 1) plus G_t elementwise C_t)
```

This equation is a LatentMesh implementation proposal, not a formula reported by BDH-CQ.

### 5.4 Invariant

Adding evidence may change recurrent context but may not expand execution authority.

## 6. Algorithm 2: iterative latent reasoning with adaptive stopping

### 6.1 Goal

Spend additional compute on hard tasks without requiring additional natural language tokens.

### 6.2 Algorithm

```text
INPUT
    query x
    recurrent state S
    minimum iterations R_min
    maximum iterations R_max
    verifier V
    convergence threshold epsilon
    confidence threshold tau
    risk class rho

OUTPUT
    final workspace H
    decoded candidate y
    reasoning receipt

PROCEDURE
    H = E_theta(x, S)
    best = null

    for r in 1 to R_max
        H_next = F_theta(H, S)

        delta = norm(H_next minus H) / max(norm(H), tiny)
        candidate = G_theta(H_next)
        score = V(candidate, H_next, S)

        record(r, delta, score)

        if r >= R_min
            if score.confidence >= tau
               and delta <= epsilon
               and score.policy_safe
               and risk_budget_satisfied(rho, score)
                best = candidate
                H = H_next
                break

        H = H_next

    if best is null
        best = G_theta(H)

    return H, best, signed_reasoning_receipt
```

### 6.3 Why adaptive stopping is required

The BDH-CQ paper reports a monotonic point estimate improvement across LOW, MEDIUM, and HIGH latent reasoning effort [1]. Recurrent depth work also reports that additional unrolled test time depth can improve reasoning [3]. Therefore fixed depth wastes compute on easy tasks and under allocates compute to hard tasks.

### 6.4 Stop policy

A production stop policy should combine at least:

```text
latent convergence
verifier confidence
constraint satisfaction
risk class
remaining latency budget
remaining energy budget
marginal expected task value
```

The controller must not use latent convergence alone because a model can converge to a wrong attractor.

## 7. Algorithm 3: structure aware demonstration retrieval with RuVector

### 7.1 Motivation

BDH-CQ reports that demonstration coverage materially changes generalization. In its controlled tests, supported demonstrations recover depth five nesting from 19 of 24 to 24 of 24 exact outputs at pass at 2, and length eight ordering from 0 of 24 to 13 of 24 [1]. The paper also reports sharp degradation as ordering length and nesting depth increase [1].

Nearest neighbor semantic retrieval alone is therefore insufficient. Retrieval should match structural complexity.

### 7.2 Retrieval descriptor

Each stored demonstration or prototype receives:

```text
semantic_embedding
operator_family
composition_depth
relation_depth
dependency_depth
object_count
output_shape_class
demonstrated_parameter_range
model_fingerprint
historical_success_rate
causal_value
```

### 7.3 Scoring function

For query q and candidate d:

```text
score(q,d) =
    w_s times semantic_similarity(q,d)
  + w_o times operator_match(q,d)
  + w_c times complexity_match(q,d)
  + w_p times parameter_coverage(q,d)
  + w_h times historical_success(d)
  + w_v times causal_value(d)
  minus w_k times retrieval_cost(d)
```

Where complexity_match can be:

```text
complexity_match = exp(
    minus absolute(composition_depth_q minus composition_depth_d)
    minus absolute(relation_depth_q minus relation_depth_d)
    minus absolute(dependency_depth_q minus dependency_depth_d)
)
```

Weights are initially configured by benchmark and later tunable by MetaHarness Darwin.

### 7.4 Retrieval rule

At least one retrieved example should cover the estimated structural difficulty of the query when such an example exists. If only simpler demonstrations exist, the reasoning budget controller should raise expected risk and allocate extra verification.

## 8. Algorithm 4: reasoning budget controller

### 8.1 Goal

Convert difficulty and risk into a bounded recurrent compute budget.

### 8.2 Inputs

```text
uncertainty u in [0,1]
structural difficulty d in [0,1]
novelty n in [0,1]
risk r in [0,1]
latency pressure l in [0,1]
energy pressure e in [0,1]
historical failure probability f in [0,1]
```

### 8.3 Budget score

```text
b = clamp(
      alpha_u times u
    + alpha_d times d
    + alpha_n times n
    + alpha_r times r
    + alpha_f times f
    minus alpha_l times l
    minus alpha_e times e,
    0,
    1
)
```

Map b into iterations:

```text
R_target = R_min plus round(b times (R_max minus R_min))
```

For safety critical tasks, latency and energy pressure may not reduce verification below a configured floor.

### 8.4 Marginal utility refinement

After each iteration estimate:

```text
MU_r = expected_value_improvement_r / incremental_compute_cost_r
```

Continue while:

```text
MU_r > minimum_marginal_utility
```

unless risk policy requires further verification.

## 9. Algorithm 5: latent reasoning delta extraction

### 9.1 Goal

Transmit only the portion of reasoning state that creates downstream value.

### 9.2 Same checkpoint residual mode

For identical model family, checkpoint, latent schema, and adapter version:

```text
Delta_raw = H_sender minus H_receiver_prediction
```

Then:

```text
indices = top_k_by_absolute_value(Delta_raw, k)
values = quantize_int8(Delta_raw[indices], scale)
DeltaH = encode(indices, values, scale)
```

This creates a sparse residual with bounded payload.

### 9.3 Low rank mode

For matrix shaped workspace H:

```text
Delta_raw approximately U_k Sigma_k V_k_transpose
```

Transmit rank k factors if:

```text
bytes(low_rank) < bytes(sparse)
```

and reconstruction error stays below policy.

### 9.4 Learned bottleneck mode

```text
z = C_phi(H_sender, receiver_context)
H_reconstructed = D_psi(z, receiver_context)
```

The learned codec is accepted only if it beats deterministic baselines on both reconstruction and downstream task utility.

### 9.5 Selection objective

Choose codec c that minimizes:

```text
J(c) =
    beta_b times transmitted_bytes(c)
  + beta_l times added_latency(c)
  + beta_e times energy(c)
  + beta_r times reconstruction_error(c)
  + beta_p times privacy_risk(c)
```

subject to:

```text
downstream_accuracy(c) >= required_accuracy
causal_value(c) >= causal_floor
governance_checks(c) = pass
```

## 10. Algorithm 6: compatibility handshake

Latent states are not assumed interoperable.

Every reasoning delta carries a CompatibilityDescriptor:

```text
model_family
model_fingerprint
checkpoint_digest
latent_schema_version
layer_or_workspace_id
hidden_dimension
normalization_scheme
position_encoding_id
adapter_id
adapter_digest
quantizer_id
codec_version
training_domain_tag optional
```

Receiver policy:

```text
if exact model fingerprint and schema match
    use identity adapter
else if verified adapter exists
    use adapter
else if semantic fallback exists
    request semantic delta
else
    reject latent delta
```

No heuristic dimension matching is permitted.

## 11. Algorithm 7: remote latent merge

### 11.1 Same model merge

```text
H_remote = decode(DeltaH)
quality = reconstruction_quality(DeltaH)
causal = cached_or_measured_causal_value(edge)

weight = clamp(
    sender_confidence
    times quality
    times causal,
    0,
    1
)

H_merged = normalize(H_local plus weight times H_remote)
```

### 11.2 Cross model merge

With verified adapter A:

```text
H_aligned = A(H_remote)
H_merged = M_phi(H_local, H_aligned, S_local)
```

The adapter must have an immutable fingerprint and benchmark record. If its benchmark expires or the sender checkpoint changes, the receiver falls back to semantic or symbolic transport.

### 11.3 Merge receipt

Every merge records:

```text
sender
receiver
edge_id
parent_state_hash
remote_state_hash
adapter_digest
codec_digest
causal_score
merge_weight
result_state_hash
policy_decision
```

## 12. Algorithm 8: causal admission of reasoning deltas

Latent communication can appear useful because of extra compute, message presence, or generic perturbation rather than sender specific information. ADR 003 already makes this a first class failure mode.

For candidate edge A to B define task value V.

```text
DeltaV_specific = V(B given A_state) minus V(B given mismatched_state)
DeltaV_zero = V(B given A_state) minus V(B given zero_state)
DeltaV_random = V(B given A_state) minus V(B given random_state)
DeltaV_self = V(B given A_state) minus V(B given B_self_generated_state)
```

An edge is admitted only when configured statistical tests show positive incremental value against all required controls.

Suggested first gate:

```text
mean(DeltaV_specific) > delta_min
p_value < 0.05 after multiple comparison correction
confidence_interval_lower_bound > 0
no material safety regression
```

The exact test depends on metric type. Binary task success should use a paired test such as McNemar or a paired bootstrap. Continuous reward should use paired permutation or bootstrap confidence intervals.

Causal value is periodically re measured because model updates, task drift, adapter changes, and compression changes can invalidate an edge.

## 13. ReasoningDeltaEnvelope

The transport level object is deliberately separate from ADR 010 Air frames. ADR 010 remains the bounded physical and semantic transport. This envelope is an upper layer payload that can be fragmented by Air or streamed by MidStream.

Proposed logical schema:

```rust
struct ReasoningDeltaEnvelope {
    version: u16,
    session_id: [u8; 16],
    edge_id: [u8; 16],
    sender_id: [u8; 32],
    model_fingerprint: [u8; 32],
    checkpoint_digest: [u8; 32],
    latent_schema: u16,
    adapter_digest: [u8; 32],
    codec: CodecId,
    iteration: u32,
    parent_state_hash: [u8; 32],
    result_state_hash: [u8; 32],
    causal_score_q15: u16,
    confidence_q15: u16,
    risk_class: u8,
    reconstruction_error_q15: u16,
    provenance_root: [u8; 32],
    body: Vec<u8>,
    signature: Option<[u8; 64]>,
}
```

Embedded profiles may use a reduced representation and dictionary coded identifiers, but semantic meaning must remain equivalent.

## 14. Persistence through RuVector

ADR 016 already defines persistent latent records with fidelity, reconstruction error, lineage, and causal value. Recurrent reasoning extends that record with:

```text
session_id
iteration
context_state_hash
workspace_state_hash
query_family
structural_descriptor
verifier_score
stop_reason
reasoning_budget
adapter_fingerprint
codec_fingerprint
```

Persistence policy:

1. Do not persist every recurrent step by default.
2. Persist checkpoints when causal value exceeds a configured floor.
3. Persist failure trajectories when they provide unique diagnostic value.
4. Compress older trajectories through ADR 016 fidelity tiers.
5. Promote repeatedly useful trajectories into prototypes.
6. Promote only inspectable and verified behavior into symbolic rules.

The retention objective is utility per storage byte, not archival completeness.

## 15. MetaHarness integration

MetaHarness controls four independent dimensions:

```text
retrieval policy
reasoning depth
codec choice
communication topology
```

A Darwin genome may contain:

```text
retrieval weights
R_min
R_max
convergence epsilon
confidence threshold
codec family
sparsity k
quantization bits
causal floor
merge threshold
fallback threshold
```

Fitness must include:

```text
primary task success
minus compute cost
minus latency cost
minus transmitted bytes cost
minus energy cost
minus safety penalty
minus causal control failure penalty
```

A candidate genome may not relax RVM authority or cryptographic policy. Governance invariants are outside the mutation space.

## 16. RuView and spatial intelligence extension

For RuView class workloads, recurrent context may summarize temporal scene state while the latent workspace reasons over short horizon sensor evidence.

Example:

```text
CSI plus BLE plus LiDAR observations
          |
          v
 temporal RuVector context S
          |
          v
 latent workspace H
          |
    iterative refinement
          |
          v
 semantic scene delta
          |
          v
 LatentMesh Air
```

Only compact scene or reasoning deltas cross constrained links. Raw CSI, video, LiDAR point clouds, and full hidden tensors remain out of band unless a higher bandwidth profile explicitly permits them.

## 17. Security and governance invariants

1. Latent state never grants authority.
2. Remote latent state is untrusted input until policy checks complete.
3. Model and checkpoint identity are cryptographically bound to the envelope.
4. Adapter identity is cryptographically bound to the envelope.
5. Unknown adapters are rejected.
6. Reconstruction quality is measured, not asserted.
7. Causal value is measured against controls, not inferred from correlation.
8. Safety critical actions require deterministic or inspectable verification before execution.
9. High consequence actions preserve a symbolic audit trail even when internal reasoning remains latent.
10. A receiver may discard any latent state without corrupting canonical world state.
11. Replay protection applies to reasoning deltas independently of transport CRC.
12. Compression and quantization may not remove provenance or authority metadata.
13. A latent payload must have explicit size, iteration, and memory bounds.
14. Failure to decode, align, verify, or merge results in fallback or rejection, never silent coercion.
15. Private latent state should be treated as potentially reconstructable sensitive information unless a tested privacy transform proves otherwise.

## 18. Failure modes and mitigations

### 18.1 Wrong attractor convergence

Failure: latent state converges numerically but represents an incorrect solution.

Mitigation: require external verifier confidence, constraint checks, or independent decoding. Never stop on latent norm alone.

### 18.2 Cross model semantic mismatch

Failure: equal dimension vectors from two models do not encode compatible semantics.

Mitigation: exact fingerprint match by default. Use only benchmarked adapters. Fall back to semantic messages.

### 18.3 Context contamination

Failure: recurrent context accumulates stale or adversarial evidence.

Mitigation: provenance aware evidence weighting, bounded context epochs, checkpoint rollback, anomaly detection, and canonical state reconciliation.

### 18.4 Causal illusion

Failure: downstream performance improves because any extra state perturbs computation.

Mitigation: ADR 003 controls: zero, random, mismatched, and self generated state.

### 18.5 Compression destroys task critical structure

Failure: low reconstruction error does not imply task equivalence.

Mitigation: benchmark downstream task success, not only vector distortion.

### 18.6 Hidden state exfiltration

Failure: transmitted latent state leaks private or proprietary information.

Mitigation: data classification, allowlisted fields or subspaces, optional privacy transforms, payload minimization, destination policy, and explicit research on reconstructability.

### 18.7 Reasoning budget runaway

Failure: adaptive depth consumes unbounded compute.

Mitigation: hard R_max, wall clock deadline, energy budget, per session quota, and cancellation token.

### 18.8 Silent canonical state mutation

Failure: a latent merge directly changes critical facts.

Mitigation: latent state can only propose canonical updates. World state changes require deterministic reconciliation and RVM policy.

## 19. Benchmark plan

The architecture is accepted only if it beats simpler baselines on measurable system objectives.

### 19.1 Baselines

```text
single pass model
verbal chain of thought
fixed recurrent depth
adaptive recurrent depth
text agent communication
JSON semantic communication
full latent tensor communication
compressed reasoning delta communication
```

### 19.2 Task families

At minimum:

```text
ARC style abstraction
Sudoku or constraint satisfaction
maze or graph reachability
tool planning with verifiable outcomes
multi agent information fusion
RuView synthetic temporal scene reconciliation
```

### 19.3 Metrics

```text
exact task success
pass at 1
pass at 2 if applicable
latency
GPU or accelerator seconds
energy joules when measurable
peak memory
bytes transmitted
compression ratio
reconstruction error
tokens emitted
causal value against controls
safety gate failures
fallback rate
```

### 19.4 Utility density

For LatentMesh Air and constrained links define:

```text
UtilityDensity =
    successful_task_value
    divided by
    transmitted_bits times energy_joules times elapsed_seconds
```

Because units can become numerically unstable near zero, production reports must also publish the raw numerator and denominator metrics.

### 19.5 Primary acceptance gate

A first implementation is considered successful only if all of the following hold on at least two non ARC task families:

1. Task accuracy is no worse than 2 percentage points below the best matched baseline, or is statistically indistinguishable within a preregistered margin.
2. At least one of compute cost, latency, emitted reasoning tokens, or transmitted bytes improves by at least 3 times.
3. Reasoning delta communication uses at least 3 times fewer bytes than text or JSON communication at equivalent downstream task success.
4. The causal edge passes zero, random, mismatched, and self generated controls.
5. All RVM authorization tests remain unchanged and passing.
6. No high consequence action can be triggered directly by a ReasoningDeltaEnvelope.

### 19.6 Stretch gate

For constrained Air profiles:

```text
at least 10 times fewer semantic or reasoning bytes
at equivalent task accuracy
```

This aligns with ADR 010 and ADR 014 rather than inventing a second incompatible transport target.

## 20. Implementation phases

### Phase 1: deterministic runtime scaffold

Create a crate tentatively named `latentmesh-reasoning` with:

```text
ContextState
ReasoningWorkspace
ReasoningBudget
ReasoningDeltaEnvelope
CompatibilityDescriptor
ReasoningReceipt
```

Implement in memory reference algorithms and deterministic test fixtures.

### Phase 2: RuVector integration

Add structure aware retrieval and recurrent trajectory persistence on top of ADR 016.

### Phase 3: adaptive recurrent model

Integrate one open weight recurrent or looped model and expose controllable iteration depth.

### Phase 4: same checkpoint distributed reasoning

Run two nodes with the identical model and checkpoint. Compare text, JSON, full latent, and sparse latent delta communication.

### Phase 5: causal edge verification

Apply ADR 003 controls and reject edges that do not show sender specific value.

### Phase 6: cross model adapters

Only after same checkpoint success, test orthogonal or learned adapters between different model families. Keep semantic fallback mandatory.

### Phase 7: LatentMesh Air compression

Map compact reasoning deltas onto ADR 010 fragmentation, prioritization, authentication, replay defense, and bounded frame profiles.

## 21. Alternatives considered

### 21.1 Keep all reasoning as language

Rejected as the only architecture. Language remains required for audit, interoperability, and some tool interfaces, but using it for every internal step unnecessarily couples reasoning compute to token generation.

### 21.2 Transmit full hidden states

Rejected for constrained links. Full tensors can be tens of kilobytes or more per step and violate the bounded Air model. They remain useful as an upper bound baseline on ordinary networks.

### 21.3 Use only semantic JSON deltas

Retained as a fallback and canonical path for critical state. Rejected as the sole representation because it may discard useful sub symbolic structure before downstream reasoning.

### 21.4 Trust latent communication if benchmark accuracy improves

Rejected. ADR 003 requires counterfactual controls because generic state presence can masquerade as useful information transfer.

### 21.5 Reproduce BDH-CQ directly

Rejected. The paper does not disclose sufficient internal implementation detail to support a faithful reproduction claim [1]. The architecture defined here is an independent LatentMesh design inspired by public principles.

## 22. Consequences

Positive consequences:

1. Reasoning compute becomes an explicit schedulable resource.
2. Agents can potentially exchange task sufficient cognitive state without verbalizing every intermediate step.
3. RuVector gains a principled role as recurrent context and trajectory memory rather than generic RAG storage.
4. MetaHarness can optimize depth, retrieval, codec, and topology jointly.
5. LatentMesh Air can optimize useful knowledge per bit rather than raw tensor throughput.
6. RVM remains the authority boundary, preserving inspectability where it matters.

Negative consequences:

1. Latent states are harder to debug than text.
2. Cross model compatibility is a research problem, not a solved transport problem.
3. Causal verification multiplies benchmark compute because each candidate edge needs controls.
4. Privacy risk may increase because hidden states can encode more information than intended.
5. Adaptive depth can increase tail latency if budgets are poorly calibrated.
6. Recurrent state can accumulate contamination without explicit epoch and rollback policy.

## 23. Key uncertainty

The largest uncertainty is whether recurrent latent reasoning and compressed latent communication preserve their efficiency advantage on useful agentic workloads outside ARC style visual reasoning.

The fix path is deliberately narrow:

```text
first prove local recurrent reasoning
then prove same checkpoint latent delta transfer
then prove causal sender specific value
then optimize compression
then test cross model adapters
then test constrained radio transport
```

Each stage has a falsifiable benchmark and does not depend on success at later stages.

## 24. Acceptance test

The ADR may move from Proposed to Accepted only when an implementation demonstrates, on at least two non ARC task families, that adaptive recurrent reasoning plus ReasoningDeltaEnvelope transport matches the selected baseline within the preregistered accuracy margin while reducing either end to end latency, compute, emitted reasoning tokens, or transmitted bytes by at least 3 times; the communication edge must also pass ADR 003 causal controls, and RVM authorization behavior must remain unchanged.

## 25. References

[1] Bjorn Engdahl, Adrian Kosowski, Jan Chorowski, Zuzanna Stamirowska, Przemyslaw Uznanski, Junlin Jiang, Rohan Phadke, Remigiusz Kinas, Richard Zhong. “BDH-CQ: In-Context Learning with Recurrent Latent Reasoning.” arXiv:2608.09888, 2026. https://arxiv.org/abs/2608.09888

[2] Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, Yuandong Tian. “Training Large Language Models to Reason in a Continuous Latent Space.” arXiv:2412.06769, 2024. https://arxiv.org/abs/2412.06769

[3] Jonas Geiping, Sean McLeish, Neel Jain, John Kirchenbauer, Siddharth Singh, Brian R. Bartoldson, Bhavya Kailkhura, Abhinav Bhatele, Tom Goldstein. “Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach.” arXiv:2502.05171, 2025. https://arxiv.org/abs/2502.05171

[4] Gautier Wang, Jin Li, Yu Sun, et al. “Hierarchical Reasoning Model.” arXiv:2506.21734, 2025. https://arxiv.org/abs/2506.21734

[5] Alexia Jolicoeur-Martineau. “Less is More: Recursive Reasoning with Tiny Networks.” arXiv:2510.04871, 2025. https://arxiv.org/abs/2510.04871

[6] N. Saunshi, N. Dikkala, Z. Li, S. Kumar, S. J. Reddi. “Reasoning with Latent Thoughts: On the Power of Looped Transformers.” arXiv:2502.17416, 2025. https://arxiv.org/abs/2502.17416

[7] H. Zhu, S. Hao, Z. Hu, J. Jiao, S. Russell, Y. Tian. “Reasoning by Superposition: A Theoretical Perspective on Chain of Continuous Thought.” NeurIPS, 2025. Reference listed in BDH-CQ [1].

[8] K. Xu, I. Sato. “A Formal Comparison Between Chain of Thought and Latent Thought.” ICML, 2026. Reference listed in BDH-CQ [1].

[9] Ye Yu, Heming Liu, Haibo Jin, Xiaopeng Yuan, Peng Kuang, Haohan Wang. “Learning to Communicate: Toward End-to-End Optimization of Multi-Agent Language Systems.” arXiv:2604.21794, 2026. https://arxiv.org/abs/2604.21794
