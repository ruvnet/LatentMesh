# 042. Causal activation interface qualification before further adapter training

> ## ⚠️ PRESERVED PROVENANCE — superseded before it executed (2026-08-29)
> 
> Renumbered **028 → 042** because `028` was already taken on main
> (`028-evolutionary-adapter-search-anti-gaming.md` / `028-sota-continuous-sweep-1.md`).
> 
> This document came from PR #13, which set out to *qualify the activation
> delivery channel **before** M4*. **M4 ran, and the entire ladder has since
> closed** — see [research/048](../research/048-run2-final-synthesis.md). The
> stronger evidence lineage was produced by PC1b/PC2/PC3 and Run 3 and merged
> in PR #15, so this qualification never executed.
> 
> **It is preserved for provenance, not as live plan.** Its design reasoning —
> same-model self-pairs, identity and random controls, pre-registration, a
> fail-closed gate — is sound and was independently arrived at again by the
> positive-control ladder. Read it as the road not taken, whose destination
> was reached another way.



- **Status**: Accepted for implementation. M4 is paused. No M3.5 outcome exists yet.
- **Date**: 2026-08-29.
- **Decision owner**: LatentMesh thought-adapter research lead.
- **Authoritative pre-registration**:
  `crates/latentmesh-runtime/receipts/run2-m35-channel-preregistration.json`, sha256
  `ebe4e76947fdd514d3759c4b02e8c9189696e635cd14a8c03c3bf8488d445915`.
- **Related**: [003](003-causal-edge-verification.md) (causal admission),
  [014](014-benchmark-and-acceptance-method.md) (evidence labels),
  [023](023-live-four-condition-run1-pre-registration.md) (live-probe discipline),
  [024](024-run2-trained-thought-adapter-ladder.md) (M3 result and the M4 ladder this decision
  pauses), and [027](027-latent-prefix-context-window-delivery.md) (fallback if the mid-layer
  channel cannot be qualified).
- **Evidence**: `crates/latentmesh-runtime/receipts/run2-m3-training-receipt-cellL18toL14.json`,
  `run2-m3-receipt-cellL18toL14-mlp-{pertoken,pooled}-slots8-poolfull-rescaletrue-n40.json`,
  `s1a-receipt-slots8-block19-poolfull-rescaletrue-n40.json`, and `run-ledger.json`.

## Outcome and decision

Pause M4 FastGRNN training and qualify the delivery channel first. Introduce M3.5, a same-model
identity-state probe over the existing, frozen 40 items. It measures the current mean-pooled,
eight-slot, mid-layer overwrite path at five pre-registered gains. Run both registered profiles
before M4:

1. Qwen2.5-1.5B-Instruct, capture L14 and inject L14. This is the exact receiver depth used by
   M3's failed L18 to L14 cross-model path.
2. Qwen2.5-3B-Instruct, capture L18 and inject L18. This is the matched half-depth receiver-scale
   oracle.

M4 stays blocked until the 1.5B profile passes and the 3B profile has produced a valid receipt.
The 3B result is mandatory even when the 1.5B profile passes. A 3B-only pass does not authorize
the existing 3B to 1.5B M4 path; it justifies a separately pre-registered, genuinely cross-model
experiment with a 3B receiver.

This decision changes the order of ADR-024's ladder. It does not silently change M4's model pair,
layer, injection operator, pooling, slots, task, or statistics.

## Why the channel is now the suspected variable

Run 1's affine maps fit their representational objective but failed the causal probe. M3 then
trained a 2-layer, 2048 to 512 to 1536 ReLU projector with 1,837,056 parameters on 572,424 token
pairs from 2,037 leakage-excluded fit items. Its holdout MSE was 0.179 and its relative residual
was 0.461 versus a 0.843 mean-predictor baseline. The causal result was still null:

| M3 path | Aligned | Random | Exact paired result |
|---|---:|---:|---:|
| Translate each token, then mean-pool | 21/40 | 21/40 | 2 wins, 2 losses, one-sided `p=0.6875` |
| Mean-pool, then translate | 22/40 | 21/40 | 2 wins, 1 loss, one-sided `p=0.5000` |

Artifact hash, golden-pair reproduction, frozen item-set reproduction, true-zero, and finite-value
integrity gates passed. Increasing transform capacity therefore improved activation prediction
without producing task-useful causal transfer through the tested channel.

The only positive channel evidence is S1a at the 1.5B L19 self-pair: identity scored 25/40 and
random scored 20/40, with five discordant wins and zero losses, one-sided `p=0.03125`. The run did
evaluate 40 items, but an exact paired sign test uses the five discordant outcomes as its effective
sample. It does not meet M3.5's stricter primary `p<0.01` gate.

### Representational fit is not causal use

These are separate claims:

- **Representational fit** asks whether a function predicts receiver activations under MSE,
  residual, cosine, or retrieval loss.
- **Causal use** asks whether injecting that state changes receiver behavior in the intended
  direction relative to matched random and zero controls.

A low-MSE map can reproduce common or high-variance directions that downstream layers ignore. It
can also miss sparse task-causal directions, erase token order during pooling, or write an
otherwise plausible state through an ineffective operator. Regression fit is a useful diagnostic;
it is not an adapter-promotion criterion.

## Business value

M3.5 converts an open-ended adapter search into a causal activation interface gate. It avoids
training FastGRNN ranks 64, 128, and 256, from 429,058 to 1,707,010 parameters, on a path that may
not carry the receiver's own state. M3's 40-item arms took about 390 seconds each, so three doomed
M4 probes alone would consume roughly 20 GPU-minutes, excluding capture, training, debugging, and
interpretation.

The larger value is a compatibility record keyed by model, layer, injection operator, gain, task,
and causal outcome. Ruflo can route through latent transfer only for a qualified tuple. RVM can
deny uncertified activation writes. RVF can package a successful adapter with the exact interface
and receipts. RuVector can index positive and null edges instead of repeatedly relearning them.

## SPARC specification

### Actors

| Actor | Responsibility |
|---|---|
| Research lead | Owns the frozen protocol, deviation decisions, and ladder outcome |
| M3.5 runner | Loads the authoritative registration, runs all controls, and writes receipts |
| Qwen runtime | Captures same-model states and performs isolated receiver generations |
| Statistics gate | Computes exact paired tests, Holm correction, effect sizes, and pass reasons |
| Ladder guard | Keeps M4 closed unless the exact receipt requirements are satisfied |
| MetaHarness | May record the outcome; it cannot change frozen statistics or thresholds |
| Ruflo router | Selects only an activation tuple with compatible evidence |
| RVM policy layer | Enforces model, layer, operator, gain, and authority constraints |
| Human reviewer | Approves any deviation or successor experiment |

### Inputs

1. The authoritative JSON pre-registration with the exact sha256 above.
2. Pinned GSM8K train JSONL, sha256
   `17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465`.
3. The exact 40 indices already frozen by S1a and repeated by M3.
4. The committed S1a receipt, sha256
   `4cc7c17521a46d23104185a0a7710e6c788f1a30973010a086374ee1baf05c36`.
5. Qwen2.5-1.5B-Instruct and Qwen2.5-3B-Instruct local model assets.
6. Gains `{0.25, 0.5, 1.0, 2.0, 4.0}` applied after natural-median norm matching.
7. The current full-generated-span arithmetic mean, eight placeholder slots, greedy decoding,
   batch size 1, and 400-token maximum.

### Outputs

1. One immutable receipt for each model profile, including the registration hash.
2. Raw outcomes for baseline, true zero, identity, and matched random at every gain and item.
3. Paired wins, losses, ties, raw exact p-values, Holm-adjusted stability p-values, and accuracy
   deltas for all gains.
4. A machine-readable profile decision with explicit integrity and statistical reason codes.
5. A ladder decision: proceed on 1.5B, require a new 3B-receiver registration, or stop mid-layer
   adapter work and activate the ADR-027 design path.
6. A null receipt when a profile fails. Failure is a completed research outcome.

### Assumptions

1. Same-model identity injection is a practical positive control for this delivery path. It is not
   a mathematical upper bound: a learned adapter could theoretically synthesize a more useful
   direction than the native pooled state.
2. Greedy decoding plus cache clearing makes condition reruns deterministic on the pinned runtime.
3. Exact-answer correctness is the primary task outcome. Gold-answer NLL is diagnostic only.
4. Repeating the same 40 items preserves direct comparability with S1a and M3.
5. The 3B self-pair profile measures whether that receiver can use its own state through the same
   class of channel. It does not prove cross-model transfer or identify parameter count as the
   cause of any difference.

### Constraints

1. The registration file must hash to
   `ebe4e76947fdd514d3759c4b02e8c9189696e635cd14a8c03c3bf8488d445915` before any model runs.
2. M4 must not train or invoke its probe until both M3.5 profile receipts exist and the ladder
   guard authorizes the exact original path.
3. The item set, order derivation, gain list, profile depth, slot count, pooling, normalization,
   prompt, token budget, and thresholds cannot change after either profile starts.
4. No item may be dropped or replaced. Any incomplete outcome makes the profile fail integrity.
5. Baseline, zero, identity, and random conditions clear KV state and use identical prompts,
   decoding, and receiver weights.
6. Identity and matched random vectors must have the same effective L2 norm at each item and gain.
7. Gain 1.0 is primary. Gains 0.5 and 2.0 are the registered adjacent stability family. Gains
   0.25 and 4.0 are diagnostics only.
8. The delivery mechanism is full-span arithmetic-mean pooling broadcast into eight slots.
9. No L19, L24, segment-preserving, per-token, additive, gated-residual, or best-observed-gain arm
   may be introduced into this registration.
10. Missing assets, hash mismatch, wrong model geometry, non-finite values, cache contamination,
    or analysis mismatch fail closed.

### Non-goals

1. Claiming an independent untouched confirmation from these reused items.
2. Claiming a universal latent protocol from one model family and GSM8K.
3. Proving that receiver parameter count causes a difference between 1.5B and 3B.
4. Promoting M3 from regression fit, NLL, or a diagnostic gain.
5. Separating token pooling from adapter input-distribution shift in M3 variant ii.
6. Testing per-token alignment, eight-segment delivery, L19/L24, additive or gated injection,
   mismatched-item controls, KV-cache transfer, or fresh confirmation items.
7. Activating ADR-027 prefix delivery or M5 MicroLoRA in this change.
8. Allowing latent state to authorize tools, network access, radio transmission, or other actions.

### Sequential evidence limitation

The registration is frozen before any M3.5 outcome is observed, but after S1a and M3 outcomes on
the same 40 items were observed. M3.5 is therefore a pre-registered sequential follow-up, not an
independent confirmation set. Its thresholds and gain family protect this run from post-outcome
tuning, but they cannot erase prior human exposure to the probe.

Any publication-level confirmation must freeze fresh, disjoint items before model inference. A
160-item fresh probe would reduce interval width by roughly half relative to 40 items, at about
four times the inference cost per arm. This is future work, not part of the authoritative M3.5
registration.

### Security and trust boundaries

Activation payloads are untrusted data even when captured from a local model. M3.5 affects only a
sandboxed offline inference process. It has no tool credentials, network authority after model
acquisition, radio authority, or action-influencing capability. RVM must cap an uncertified or
failed tuple at `ObserveOnly`; a passing receipt may authorize only the already governed context
injection experiment. `LatentPrefix` and `ActionInfluencing` remain separate authority decisions.

The runner verifies registration, dataset, S1a artifact, model identity, layer count, hidden size,
and required cache files before inference. A receipt hash proves artifact integrity, not author
identity; future signed-evidence claims require a real signature and key identity. Receipts must
not copy private prompts, credentials, or raw user data. Raw activations should remain ephemeral
unless a separate retention decision is approved.

Random controls are deterministic and norm matched. Every condition clears KV state. A frozen
witness subset reruns after the arm and must reproduce outputs or logits hashes before the receipt
is admissible. Any cache-order effect, non-finite activation, dimension mismatch, or norm mismatch
fails integrity rather than being converted into a skipped item.

## Frozen pre-registration

### Probe and provenance

The following exact indices are authoritative:

```
141, 150, 850, 1153, 1309, 1435, 1573, 1602, 2365, 2418,
2436, 2489, 2540, 2723, 2958, 2973, 3084, 3165, 3741, 3772,
3825, 3844, 3877, 4255, 4305, 4344, 4746, 4888, 5422, 5610,
6031, 6146, 6378, 6452, 6552, 6629, 7293, 7375, 7431, 7462
```

They derive from ChaCha8 item seed `20897` over the pinned GSM8K train source. The matched-random
seed base is `1369571328`. The runner must reproduce the list from the source and seed, compare it
with both the registration and S1a receipt, and stop on any difference.

### Profiles

| Profile | Model | Layers | Hidden size | Capture | Inject | Purpose |
|---|---|---:|---:|---:|---:|---|
| `qwen2.5-1.5b-exact-channel` | Qwen/Qwen2.5-1.5B-Instruct | 28 | 1536 | L14 | L14 | Qualify the exact M3 receiver depth |
| `qwen2.5-3b-scale-oracle` | Qwen/Qwen2.5-3B-Instruct | 36 | 2048 | L18 | L18 | Mandatory half-depth receiver-scale oracle |

The 3B profile is an oracle-capacity diagnostic, not ADR-024's still-required genuinely
cross-model receiver-scale arm. A 3B pass authorizes design of that arm; it does not make the M3
cross-model result positive.

### Delivery and conditions

Capture the receiver model's own generated-span residual rows at the profile's capture block.
Arithmetic-mean pool the full generated span to one vector. Rescale that vector to the receiver's
natural median per-position L2 norm at the injection block, then apply gain. Broadcast the same
effective vector to all eight `<|fim_pad|>` placeholder positions.

Run conditions in the registered order:

1. Baseline with no residual edit on the same slotted prompt.
2. True zero through all eight real overwrite positions.
3. Identity and matched random at gains 0.25, 0.5, 1.0, 2.0, and 4.0, in that order.

Matched random is a per-item seeded Gaussian direction with the exact effective L2 norm of the
corresponding identity condition. Greedy decoding, batch size 1, and `max_new_tokens=400` are fixed.

### Primary statistic

For item `i` and gain `g`, define exact-answer correctness `I_i(g)` for identity and `R_i(g)` for
matched random. Define:

```
w_g = count(I_i(g) = 1 and R_i(g) = 0)
l_g = count(I_i(g) = 0 and R_i(g) = 1)
n_eff_g = w_g + l_g
p_g = P[X >= w_g], where X follows Binomial(n_eff_g, 0.5)
delta_g = 100 * (sum_i I_i(g) - sum_i R_i(g)) / 40
```

The exact paired sign test is one-sided because the directional alternative, identity greater
than matched random, is frozen before M3.5 outcomes. Ties do not enter `n_eff_g`. All 40 items
remain in accuracy denominators.

Do not use a raw `identity_correct >= 29` gate. Twenty-nine correct of 40 has a two-sided
one-sample binomial p-value below 0.01 only under a different null and only when treating all 40 as
informative. M3.5's causal contrast is paired; its denominator is the discordant count. For
example, seven wins and zero losses gives one-sided `p=0.0078125`, while five wins and zero losses
gives `p=0.03125`.

### Multiplicity and exact profile gate

Gain 1.0 is the sole primary gain. It passes its task-specificity gate only when:

1. raw paired one-sided `p_1.0 < 0.01`; and
2. `delta_1.0 >= 15.0` accuracy percentage points.

Gains 0.5 and 2.0 form the two-hypothesis adjacent-gain stability family. Apply Holm-Bonferroni
to their two raw paired p-values. Stability passes when at least one of the two adjacent gains has:

1. Holm-adjusted `p < 0.05`; and
2. identity minus random accuracy `>= 10.0` percentage points.

This yields a positive result at gain 1.0 plus at least one neighboring gain. Gains 0.25 and 4.0
are reported but never participate in the profile gate.

The zero-vector mechanics gate passes when:

```
2 * zero_correct >= baseline_correct
```

A profile passes only when the primary task-specificity gate, adjacent-gain stability gate,
zero-vector gate, and every integrity check pass. NLL is diagnostic and cannot substitute for any
accuracy gate. Report all raw and adjusted p-values regardless of outcome.

### Exact proceed and stop matrix

| 1.5B profile | 3B profile | Decision |
|---|---|---|
| Pass | Pass | M4 may resume on the original 1.5B path; retain 3B as supporting scale evidence |
| Pass | Fail | M4 may resume on 1.5B, but no positive receiver-scale claim is allowed |
| Fail | Pass | Keep existing M4 blocked; pre-register a genuine cross-model adapter into the qualified 3B receiver |
| Fail | Fail | Stop the mid-layer adapter ladder and move to ADR-027 or a separately registered delivery redesign |

Both profile receipts are mandatory before applying this matrix. A partial, unavailable, or
integrity-failed 3B arm is not equivalent to a 3B statistical fail; either state keeps M4 blocked.

## SPARC pseudocode

```text
function load_frozen_registration(path):
    bytes = read(path)
    require sha256(bytes) == AUTHORITATIVE_REGISTRATION_SHA256
    registration = parse_strict_schema(bytes)
    require registration.protocol_id == EXPECTED_PROTOCOL_ID
    return registration

function validate_inputs(registration, profile):
    frozen = registration.frozen_source
    require sha256(gsm8k_train) == frozen.dataset.sha256
    require sha256(s1a_receipt) == frozen.s1a_receipt.sha256
    require derive_indices(gsm8k_train, seed=20897) == frozen.dataset.indices
    require s1a_receipt.indices == frozen.dataset.indices
    require loaded_model.id == profile.model_id
    require loaded_model.layers == profile.expected_layers
    require loaded_model.hidden_size == profile.hidden_size
    require all_required_cache_files_exist()

function run_profile(registration, profile):
    validate_inputs(registration, profile)
    outcomes = []
    for item in registration.frozen_source.dataset.indices:  # no skip or replacement branch
        source = generate_and_capture_same_model(item, profile.capture_block)
        require source.generated_span_nonempty and finite(source.states)
        pooled = arithmetic_mean(source.generated_span_states)
        natural_median = measure_natural_median(profile.inject_block, slotted_prompt)

        baseline = decode_after_clear_cache(no_injection)
        zero = decode_after_clear_cache(true_zero_through_real_path)
        record(outcomes, item, baseline, zero)

        for gain in [0.25, 0.5, 1.0, 2.0, 4.0]:
            identity = normalize_to(pooled, natural_median) * gain
            random = seeded_gaussian(item, RANDVEC_SEED_BASE, gain)
            random = normalize_to(random, l2(identity))
            require relative_norm_error(identity, random) <= NORM_TOLERANCE
            id_result = decode_after_clear_cache(broadcast8(identity))
            random_result = decode_after_clear_cache(broadcast8(random))
            record(outcomes, item, gain, id_result, random_result)

    require exactly_40_complete_rows_per_condition(outcomes)
    require frozen_witness_rerun_matches(outcomes)
    decision = analyze_exactly_as_registered(outcomes)
    write_immutable_receipt(registration_hash, profile, outcomes, decision)
    return decision

function analyze_exactly_as_registered(outcomes):
    primary = paired_sign_test(identity_1_0, random_1_0)
    primary_pass = primary.p_raw < 0.01 and primary.delta_pp >= 15.0

    adjacent = [paired_sign_test(identity_0_5, random_0_5),
                paired_sign_test(identity_2_0, random_2_0)]
    holm_adjust(adjacent)
    stability_pass = any(x.p_holm < 0.05 and x.delta_pp >= 10.0
                         for x in adjacent)

    zero_pass = 2 * zero_correct >= baseline_correct
    return primary_pass and stability_pass and zero_pass and integrity_all_pass

registration = load_frozen_registration(PREREGISTRATION_PATH)
q1 = run_profile(registration, QWEN_1_5B_L14)
q2 = run_profile(registration, QWEN_3B_L18)            # mandatory regardless of q1

if q1.pass:
    unblock_original_M4_only_after(q2.receipt_exists_and_is_valid)
else if q2.pass:
    keep_original_M4_blocked_and_require_new_3B_cross_model_registration()
else:
    stop_mid_layer_ladder_and_open_ADR_027_design_path()
```

### Success walk-through

The 1.5B profile verifies every frozen input and completes 40 rows for every condition. At gain
1.0, identity versus random has raw paired `p<0.01` and at least 15 points of accuracy advantage.
Gain 0.5 has Holm-adjusted `p<0.05` and at least 10 points of advantage. True zero is not
catastrophic and witnesses reproduce, so the profile passes. The 3B profile then runs and emits a
valid receipt. Only after that receipt exists may the guard allow original M4 to start.

### Failure walk-through

At gain 1.0, identity beats random by 15 points but the paired p-value is 0.03125. Gain 4.0 is
strong, but it is diagnostic. The 1.5B profile fails and M4 remains blocked. If the 3B profile
passes, the system records a receiver-specific channel result and requires a new cross-model
3B-receiver registration. It does not reinterpret M3 or silently reverse the model pair.

### Invariants

1. **Registration first**: the authoritative registration hash is verified before model loading.
2. **Exact item reuse**: the runner reproduces the S1a indices and never replaces an item.
3. **No post-outcome tuning**: gain roles, thresholds, profiles, pooling, and controls are fixed.
4. **Single content variable**: within a gain, identity and random differ only in direction.
5. **Norm equivalence**: identity and random have equal effective L2 norm.
6. **Cache isolation**: every condition begins from cleared receiver state.
7. **Paired inference**: primary p-values use wins and losses, not total correct as trial count.
8. **Multiplicity integrity**: only 0.5 and 2.0 enter the registered Holm family.
9. **No diagnostic promotion**: gains 0.25, 4.0, and NLL never rescue a failed profile.
10. **Evidence immutability**: failed, partial, and invalid receipts remain preserved.
11. **Scale-control completion**: M4 stays blocked until a valid 3B receipt exists.
12. **Authority monotonicity**: experiment success cannot grant action-influencing authority.
13. **Claim discipline**: reused items are sequential evidence; 3B self-pair is not cross-model.

## SPARC architecture

### Components and interfaces

| Component | Interface | Owns |
|---|---|---|
| Registration loader | `load_and_validate(path, expected_hash, profile)` | Strict schema, artifact hash, frozen constants |
| Input validator | `validate_s1a_receipt()` and model-geometry checks | Dataset, prior receipt, indices, model assets |
| Capture runtime | `forward_capture(model, block)` | Same-model generated-span state |
| Delivery encoder | `mean_pool()` and existing `InjectionSpec` broadcast | Current one-vector, eight-slot path |
| Control generator | `matched_random(vector, seed)` | Deterministic norm-matched sham |
| Evaluator | exact answer plus diagnostic gold NLL | Raw item outcomes only |
| Statistics gate | `decide(results) -> Pass or Fail` | Exact pairing, Holm correction, effect floors |
| Receipt writer | `write_receipt()` | Immutable evidence and reason codes |
| Ladder guard | `require_channel_qualification()` | Fail-closed M4 enforcement |

No multi-vector injection API is required for M3.5. The current `InjectionSpec` intentionally
broadcasts one vector to all registered positions and is the exact channel under test.

### Data lifecycle

The registration and prior receipt are immutable inputs. States exist in process memory for one
item and need not be retained after its conditions complete. Raw activation dumps are unnecessary
for the decision and increase prompt-reconstruction and storage risk. Receipts retain hashes,
indices, norms, outputs, correctness, statistics, provenance, and timing. The summary decision is
recomputed from item rows rather than trusted as an unaudited boolean.

### Deployment and trust model

M3.5 is an offline, single-host research harness. It is not linked into the default LatentMesh
radio or production path. Production routing may later consume a compact compatibility tuple and
receipt hash, never raw experiment activations. A pass is scoped to the exact model, layer,
operator, gain envelope, prompt protocol, task, runtime, and evidence revision.

### Observability

Every profile receipt records:

- authoritative registration hash, protocol ID, runtime commit, CUDA and GPU metadata;
- model ID, layer count, hidden size, cache-file validation, and tokenizer identity when available;
- dataset and S1a hashes, exact indices, item and random seeds;
- source span, block, slot count, pooling, normalization, gain, and condition;
- natural, raw, and effective norms plus identity/random relative norm error;
- exact answer, correctness, gold NLL, token count, failure reason, and wall time per condition;
- paired wins, losses, ties, effective n, raw p, adjusted p where applicable, and accuracy delta;
- zero versus baseline counts, witness reproducibility, profile decision, and reason codes.

Alert conditions are registration mismatch, input artifact mismatch, wrong model geometry, missing
rows, duplicate indices, non-finite values, norm mismatch, cache-witness mismatch, analysis drift,
and an M4 invocation without the required receipts.

### Rollback

Implementation is additive and experiment-only. Rollback removes or disables the M3.5 runner and
analysis module while preserving all registrations and receipts. Rolling back code does not reopen
M4; only a superseding ADR may change that policy. If a schema or statistics defect is found, mark
the receipt invalid, preserve it, version and freeze a corrected registration, then rerun. Never
overwrite evidence or alter this registration after viewing an M3.5 outcome.

## Alternatives and quantified tradeoffs

| Alternative | Delivery cost | Information value | Main risk | Decision |
|---|---:|---:|---|---|
| Continue directly to M4 ranks 64, 128, 256 | About 20 GPU-minutes for probes plus training | Low if the channel is null | Optimizing the adapter while delivery is causal bottleneck | Rejected now |
| Reuse the exact 40 with registered gain controls | Roughly 1 to 2 GPU-hours for 1.5B | High path comparability | Sequential reuse limits independent confirmation | Selected for Q1 |
| Add the mandatory 3B oracle | Roughly 2 to 4 additional GPU-hours | High receiver-feasibility information | Same-model result confounds scale with checkpoint identity | Selected for Q2 |
| Fresh 160-item confirmation | About four times a 40-item arm | Stronger power and generalization | Higher cost before the engineering stop decision | Deferred to publication stage |
| Per-token, segment8, additive, gated, or alternate-layer sweep | Roughly 1 to 6 GPU-hours per design family | High after current path fails | Multiple-comparison and operator-selection inflation | Requires a new registration |
| Train a pooled-specific 1.84M-parameter MLP now | About 2,547 item pairs plus a new probe | Medium on the OOD question | Severe capacity-to-item ratio and same channel uncertainty | Deferred |
| Move directly to ADR-027 prefix delivery | Moderate new runtime and probe work | None about current mid-layer path | Abandons a cheap falsifiable channel check | Contingency |

Time figures are planning estimates anchored to observed M3 wall time. Live receipts must report
actual GPU-seconds and replace estimates in all result claims.

## M3 variant ii and the pooled OOD confound

M3 variant ii applies a network trained exclusively on per-token activation pairs to a mean-pooled
input it never saw during training. Its failure cannot distinguish:

1. pooling destroys causally useful token structure;
2. pooled inputs are out of distribution for the per-token-trained MLP;
3. the mean-pooled eight-slot overwrite channel is weak regardless of adapter quality.

M3.5 tests only hypothesis 3 for native same-model states under the exact current channel. It does
not settle hypotheses 1 or 2. A valid pooling study must compare a pooled-input-trained adapter
against translate-then-pool under leakage-safe training and its own frozen probe. A token-delivery
study must pre-register per-token or bounded segment mechanics. Until those exist, M3 variant ii
is evidence that the implemented composite pipeline failed, not proof that pooling destroyed the
signal.

## RuV stack integration

| Component | Integration contract | Value |
|---|---|---|
| RuLLM | Expose capture and injection coordinates plus an activation-interface descriptor | Makes latent interchange explicit rather than assumed |
| MetaHarness | Verify the receipt and enforce the SPARC completion gate | Stops low-information training and preserves honest nulls |
| RVM | Default-deny a tuple without matching evidence; enforce model, layer, operator, and gain scope | Prevents opaque states from bypassing policy |
| RVF | Package weights, interface descriptor, allowed gains, tests, and receipt hash | Produces a portable, reproducible cognitive artifact |
| RuVector | Index compatibility outcomes by model, layer, task, operator, and causal effect | Retrieves viable and known-null paths |
| Ruflo | Choose latent transfer only for qualified tuples; otherwise use text or semantic deltas | Converts evidence into a routing signal |
| LatentMesh | Bind payloads to the receiver's interface and certificate hash | Rejects model, layer, norm, and operator mismatch |
| Core Memory | Preserve the ADR, receipt hashes, decision, uncertainty, and next allowed action | Maintains provenance across agents and runs |
| Cognitum | Use qualification in placement and cost policy, never as implicit safety approval | Connects research evidence to deployment governance |

## Refinement plan

1. Implement a strict schema matching the authoritative JSON and pin its sha256 in the runner.
2. Validate the S1a artifact, dataset, exact indices, profile geometry, and required model assets.
3. Implement the frozen five-gain identity, matched-random, zero, and baseline protocol with the
   existing broadcast injection path.
4. Implement exact paired tests, the two-member Holm family, effect floors, and reason codes.
5. Add unit fixtures that would catch use of total n instead of discordant n and any raw
   `identity_correct >= 29` promotion rule.
6. Add the M4 ladder guard before any M3.5 GPU outcome is consumed.
7. Run Q1 once and retain its receipt regardless of outcome.
8. Run Q2 once regardless of Q1 and retain its receipt regardless of outcome.
9. Apply the proceed and stop matrix without changing thresholds.
10. Append results to this ADR, mapping every claim to exact receipt fields.

Each increment must preserve existing S0, S1a, S2b, and M3 reproduction behavior.

## Completion and requirement-to-acceptance mapping

| Requirement | Acceptance test | Evidence required |
|---|---|---|
| R1: M4 is paused | Invoke M4 without M3.5 receipts | Nonzero exit with `CHANNEL_NOT_QUALIFIED` |
| R2: Authoritative registration | Change one registration byte | Hash validation rejects before model load |
| R3: Exact historical probe | Change, duplicate, or reorder one index | Strict validation rejects the input |
| R4: S1a provenance | Change the S1a receipt or dataset hash | Validation rejects before model load |
| R5: Exact profile geometry | Substitute L19, L24, or a lookalike model | Validation rejects the profile |
| R6: No item replacement | Inject one controlled per-item failure | Profile fails integrity and retains the failed row |
| R7: Norm-matched sham | Evaluate all identity/random pairs | Relative norm error stays within the implementation tolerance |
| R8: Cache isolation | Repeat frozen witnesses | Outputs or logits hashes reproduce |
| R9: Paired denominator | Analyze 7W/0L and 5W/0L fixtures | Returns `0.0078125` and `0.03125` |
| R10: No 29-correct shortcut | Supply 29 identity correct with a null paired contrast | Profile remains failed |
| R11: Primary gate | Supply `p<0.01` with delta below 15 points | Profile remains failed |
| R12: Holm stability | Pass primary while both neighboring gains fail corrected tests | Profile remains failed |
| R13: Diagnostic non-promotion | Make only gain 4.0 pass | Profile remains failed |
| R14: Zero mechanics | Make `2 * zero_correct < baseline_correct` | Profile remains failed |
| R15: Mandatory scale oracle | Make Q1 pass with no valid Q2 receipt | M4 remains blocked |
| R16: 3B-only result | Make Q1 fail and Q2 pass | Existing M4 remains blocked; new registration required |
| R17: Security ceiling | Load a passing receipt into policy | Authority cannot exceed the governed context-injection scope |
| R18: OOD claim discipline | Generate M3 summary | Pooled variant ii is labeled composite-pipeline null, not pooling proof |
| R19: Honest failure | Make both profiles fail | Stop receipt exists and no automatic rerun starts |

SPARC completion requires R1 through R19 to pass, both live profile receipts to exist, old probe
behavior to remain reproducible, and the outcome appendix to map each live claim to receipt fields.

## Residual risks and owners

| Risk | Impact | Mitigation | Owner |
|---|---|---|---|
| Reusing observed items biases sequential evidence | Apparent effect may not generalize | Fresh 160-item preregistered confirmation before publication | Research lead |
| Forty items produce coarse effect estimates | A modest real channel can fail | Treat this as an economic stop gate, not a universal scientific verdict | Statistics owner |
| One random direction per item is noisy | Specificity effect can vary with sham draw | Preserve seed; use new preregistered seeds only in confirmation | Statistics owner |
| 3B self-pair is not a pure scale experiment | Difference may be over-attributed to size | Label it oracle feasibility; require a genuine cross-model 3B arm | Experiment designer |
| Mean pooling may erase useful order | Current channel fails although richer delivery works | Pre-register per-token or segment delivery only after M3.5 | Runtime owner |
| Opaque latent input bypasses text inspection | Safety and audit gap | Default deny, exact scope, norm bounds, semantic fallback, revocation | Security owner |

## Acceptance test

Run the M4 entry point after a passing 1.5B receipt but before a valid 3B receipt exists. It must
fail with `CHANNEL_NOT_QUALIFIED`. After both receipts exist, original M4 may proceed only if the
1.5B L14 profile passes the primary gain 1.0 gate, at least one Holm-corrected adjacent gain gate,
the zero-vector gate, and every integrity check. A raw total of 29 correct, a gain 4.0 result, NLL
movement, regression fit, or a 3B-only pass must not unblock it.
