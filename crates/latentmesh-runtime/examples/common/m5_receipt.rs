//! The M5 receipt (ADR-045), split out of `common/m5_draw.rs` for file-size
//! discipline: one large `serde_json::json!` literal, auditable top to bottom,
//! plus the context struct that carries everything the draw itself does not
//! produce.
//!
//! Declared from `run2_m5_probe.rs` rather than from `common/mod.rs`, because
//! that module is `#[path]`-included into every example and this literal's
//! expansion exceeds the default macro recursion limit.

use crate::common::m3::{N_SLOTS, RECEIVER, RECEIVER_BLOCK, SENDER, SENDER_BLOCK};
use crate::common::m5::{condition_battery, LoadedAdapter};
use crate::m5_draw::{
    DrawOutcome, E_ALPHA, FUSE_NOOP_TOL, LAMBDA, N_MAX, REGISTERED_BAR_OF, REGISTERED_BAR_WINS,
    UNINFORMATIVE_BELOW_N_DISC,
};
use latentmesh_runtime::inject::InjectionMode;

/// Everything the receipt needs that the draw itself does not produce.
pub struct ReceiptCtx<'a> {
    pub rank: usize,
    pub env: serde_json::Value,
    pub comparator_receipt: &'a str,
    pub payload_training_receipt: &'a str,
    pub payload_artifact: String,
    pub payload_hash: &'a str,
    pub payload_golden_file: String,
    pub payload_golden_pairs: usize,
    pub payload_golden_seed: u64,
    pub payload_golden_max_rel: f32,
    pub payload_golden_tol: f32,
    pub adapter: &'a LoadedAdapter,
    pub m5_training: &'a serde_json::Value,
    pub transfer: &'a serde_json::Value,
    pub site_tag: &'a str,
    pub site_description: &'a str,
    pub inject_mode: InjectionMode,
    pub randvec_seed_base: u64,
    pub pad_id: u32,
    pub train_sha: String,
    pub leakage_exclusions: Vec<usize>,
    pub excluded_present: Vec<usize>,
    pub eligible: usize,
    pub tokenization_excluded: Vec<serde_json::Value>,
    pub site_samples: Vec<serde_json::Value>,
}

/// The M5 receipt, one auditable object.
pub fn receipt(c: &ReceiptCtx<'_>, o: &DrawOutcome) -> serde_json::Value {
    let (n, n_disc, e_pass) = (o.n(), o.n_disc(), o.e_pass());
    let (noop_dis, noop_exact, noop_max, noop_pass) = o.noop_stats();
    let (real_c, base_c, zero_c, rand_c) = (
        o.count(|q| q.real.0),
        o.count(|q| q.base.0),
        o.count(|q| q.zero.0),
        o.count(|q| q.rand.0),
    );
    let zerovec_pass = 2 * zero_c >= base_c;
    let (inv_base, inv_zero, inv_rand) = (
        o.count(|q| q.real.1 > q.base.1),
        o.count(|q| q.real.1 > q.zero.1),
        o.count(|q| q.real.1 > q.rand.1),
    );
    let w_threshold = 1.0 / E_ALPHA;
    let uninformative = o.uninformative();
    let a = &c.adapter;

    serde_json::json!({
        "stage": "run2-M5-receiver-side-adaptation-probe",
        "design": "docs/adr/045-m5-receiver-side-adaptation-pre-registration.md (ACCEPTED — EXECUTING; everything below its MANDATORY POWER CALCULATION heading was frozen before any item was drawn). Evaluation protocol = docs/adr/036-successor-rung-evaluation-protocol.md (e-process, adaptation-512 stream). Scout: docs/research/034.",
        "env": c.env,
        "pre_committed": true,
        "rank": c.rank,
        "variant": "pertokenlast-fuse-questiontail-receiveradapted",
        "protocol_identity": {
            "statistics": "ADR-036 anytime-valid Bernoulli e-process (ADR-030 §3.2 mechanics verbatim)",
            "item_supply": "adaptation-512, fixed index order, ADR-024's 13-item leakage exclusion applied",
            "era": "SUCCESSOR-RUNG ERA. Not the frozen 40-item sign-test protocol that governed M3 through M4h Stage 1.",
            "must_travel_with_every_headline_number": "Per ADR-036 Decision 3, any write-up quoting a number from this receipt must state in the same sentence that it was produced under the e-process protocol on the adaptation-512 stream, on a RECEIVER-ADAPTED model.",
        },
        "the_single_changed_factor": {
            "factor": "THE RECEIVER — it carries a trained additive low-rank adapter",
            "from": "unadapted Qwen2.5-1.5B-Instruct (every rung through M4i)",
            "to": format!("the same weights plus a rank-{} LoRA on the residual stream after block {RECEIVER_BLOCK}", c.rank),
            "comparator_receipt": c.comparator_receipt,
            "held_fixed_against_the_comparator": [
                "payload artifact: M3's trained on-manifold reconstruction MLP, byte-identical (hash asserted against M3's training receipt AND the comparator's recorded hash)",
                "payload derivation: Variant::PerTokenLast (apply_last_row, de-pooled)",
                "injection operator: InjectionMode::Fuse, h[pos] += c*v",
                "injection site: the last 8 tokens wholly inside the item's own question",
                "slot count: 8 (ADR-028-protected; asserted at run time)",
                "depth: receiver block 14, cell L18->L14",
                "rescale: to the per-item natural median at the inject block, captured on the residual BEFORE the adapter",
                "decoding: greedy, batch=1, max_new_tokens=400",
                "item stream and statistic: adaptation-512 in fixed index order; lambda=0.30, N_max=300, PASS at W>=20 — all asserted equal to the comparator's own recorded values",
            ],
            "adapter_order_declared": "block 14 -> injection edit -> LoRA. The adapter sees the injected content, which is its purpose. A Capture tap at the same block runs BEFORE the adapter, so the rescale target is the base receiver's and is unchanged from M4i.",
        },
        "receiver_adaptation": {
            "training_receipt": format!("run2-m5-training-receipt-cellL18toL14-r{}.json", c.rank),
            "transfer_receipt": format!("run2-m5-transfer-receipt-cellL18toL14-r{}.json", c.rank),
            "adapter_content_hash": a.lora.content_hash,
            "adapter_init_content_hash": a.init_content_hash,
            "rank": a.lora.rank, "alpha": a.lora.alpha,
            "scaling": a.lora.scaling(), "param_count": a.param_count,
            "equation": "h' = h + ((h @ A) @ B) * (alpha / rank), F32 matmuls, delta cast to BF16 before the add",
            "golden_verification": {
                "pairs": a.golden_pairs, "max_relative_l2_error": a.golden_max_rel,
                "tolerance": c.payload_golden_tol, "pass": true,
                "what_it_pins": "the adapter this probe loaded is bit-for-bit the one the trainer wrote, and the probe-side ResidualLora computes the same function the trainer's LoraAdapter did",
            },
            "trained_on": "the leakage-safe FIT side of the S2c calibration pool, asserted DISJOINT from adaptation-512 in the training receipt; loss = next-token CE on the gold-answer continuation '#### {gold}', NOT the sender's generated span",
            "frozen_during_training": "the sender, the M3 translator, and every receiver weight; only A and B were optimised",
            "transfer_check_gate": {
                "pass": true,
                "criterion": c.transfer["gates"]["transfer"]["criterion"].clone(),
                "mean_fused_train_target_ce_adapter_on": c.transfer["summary"]["mean_fused_train_target_ce_adapter_on"].clone(),
                "mean_fused_train_target_ce_adapter_off": c.transfer["summary"]["mean_fused_train_target_ce_adapter_off"].clone(),
                "probe_endpoint_reported_not_gating": c.transfer["summary"]["probe_endpoint_reported_not_gating"].clone(),
                "n_evaluated": c.transfer["summary"]["n_evaluated"].clone(),
                "scope": "composed->fused BF16 transfer only, on holdout items. It says nothing about the channel: an adapter that merely made the receiver a better GSM8K solver would pass it too.",
                "decision_side_diagnostic_carried_forward": c.transfer["summary"]["generation_diagnostic"].clone(),
                "why_carried_forward": "ADR-045 § Deviations (coordinator error #22) makes the decision-side diagnostic mandatory from v2 onward. It is copied into THIS receipt so a reader of the draw alone can see whether the adapted receiver still reasons — without it, 'the receiver stopped answering' and 'the channel carries nothing' are indistinguishable.",
            },
        },
        "the_confound_and_why_the_primary_is_immune": {
            "confound": "ADR-045: a task-loss-adapted receiver could simply become a better GSM8K solver, improving regardless of injected content.",
            "baseline_re_measured_on_this_receiver": true,
            "no_frozen_receiver_baseline_is_reused_anywhere": "every number in this receipt's accuracy and NLL blocks was measured on the ADAPTED receiver in this run. M4i's numbers are referenced only to assert the held-fixed factors, never as an arm of a comparison.",
            "primary": "aligned vs random — both arms on the same adapted receiver, so a general fine-tuning effect raises both and cancels",
            "secondary_and_confounded": "aligned vs baseline. It is reported, and it is NOT immune; it may only be read against the freshly measured adapted baseline above.",
        },
        "inherited_hazard_declared_before_the_draw": {
            "history": "M4c/M4d/M4g trained the PAYLOAD through a frozen-receiver task loss and produced a reproducible 0W/40L NLL inversion against both controls, diagnosed as OFF-MANIFOLD rather than destructive.",
            "why_this_rung_is_not_that_experiment": "M5 trains the RECEIVER and leaves the payload at M3's on-manifold reconstruction weights — the same payload M4i ran, which was inert rather than harmful.",
            "registered_reading_if_aligned_inverts": "ADR-045: if the aligned condition inverts the same way here, the FIRST hypothesis is off-manifold input, NOT a channel effect, and it must be reported that way.",
            "measured_this_draw": {
                "items_where_aligned_nll_is_WORSE_than_baseline": inv_base,
                "items_where_aligned_nll_is_WORSE_than_zerovec": inv_zero,
                "items_where_aligned_nll_is_WORSE_than_random": inv_rand,
                "n_items": n,
                "unanimous_inversion_vs_baseline": inv_base == n && n > 0,
            },
        },
        "delta_v": {
            "computed_here": false,
            "why": "docs/research/034 §3 prices ONE properly powered verify_edge draw at ~3 GPU-h — more than this entire rung — and ADR-028 forbids frozen-probe fitness for any adapter search. ADR-045 registers ΔV as a single post-hoc characterisation, never an online training signal. It was not used during training and is not used here.",
        },
        "config": {
            "sender": SENDER, "receiver": RECEIVER, "receiver_is_adapted": true,
            "sender_capture_block": SENDER_BLOCK, "receiver_inject_block": RECEIVER_BLOCK,
            "cell": "L18->L14 (S2 winner; M5 registers on this cell only)",
            "slots": N_SLOTS,
            "injection_site": c.site_tag, "site_description": c.site_description,
            "placeholder_token": serde_json::Value::Null,
            "resolved_pad_id_unused": c.pad_id,
            "pool_span": "NONE — de-pooled; the payload is the last translated token of the generated span, broadcast to the 8 positions",
            "rescale_to_natural_median": true,
            "natural_norm_source": "receiver forward_capture over this rung's prompt at the inject block, per item — captured BEFORE the adapter runs",
            "decoding": "greedy, batch=1, max_new_tokens=400",
            "randvec_seed_base": c.randvec_seed_base,
            "injection_operator": {"mode": c.inject_mode.tag(), "equation": c.inject_mode.equation()},
            "transform": {
                "kind": "M3's ALREADY-TRAINED reconstruction MLP 2048->512->1536 ReLU — byte-identical weights, no retraining",
                "file": c.payload_artifact,
                "content_hash": c.payload_hash,
                "training_receipt": c.payload_training_receipt,
                "payload_derivation": "apply_last_row on the LAST generated-span token state",
                "hand_rolled_apply_verification": {
                    "golden_file": c.payload_golden_file,
                    "golden_pairs": c.payload_golden_pairs,
                    "golden_input_seed_chacha8": c.payload_golden_seed,
                    "max_relative_l2_error": c.payload_golden_max_rel,
                    "tolerance": c.payload_golden_tol, "pass": true,
                },
            },
            "conditions": {
                "aligned_real": "sender per-token capture -> M3 MLP per token -> LAST token -> fused at the 8 question-tail positions, rescaled to natural median, on the ADAPTED receiver",
                "random": "per-item seeded gaussian, norm-matched to the effective aligned vector, same positions, same operator, same ADAPTED receiver",
                "zerovec_injected": "TRUE ZERO VECTOR through the same path (scale: None). Under fuse this is h += 0, an exact no-op, so it collapses onto baseline_uninjected — on the adapted receiver as on any other.",
                "baseline_uninjected": "no injection (spec=None), same prompt, SAME ADAPTED RECEIVER — re-measured here, never inherited",
            },
        },
        "e_process": {
            "registered_source": "ADR-036 Decision 1, quoting ADR-030 §3.2 verbatim",
            "rule": "W_0 = 1. Concordant item: W_i = W_{i-1}. Discordant item: W_i = W_{i-1} * (1 + lambda*(X_i - 0.5)), X_i = 1 iff aligned wins the pair. PASS the instant W_i >= 1/alpha.",
            "lambda": LAMBDA, "alpha": E_ALPHA, "wealth_threshold": w_threshold, "n_max": N_MAX,
            "comparison": "aligned_real vs random on paired per-item accuracy, both on the adapted receiver",
            "degenerate_item_rule": "an item whose sender capture pass is degenerate yields no pair and no wealth update, but still CONSUMES one of the N_max budget items — registered, not chosen after seeing the data",
            "stopping": "the draw stops at the first item where W_i >= the threshold; otherwise it runs the full N_max",
            "never_restarted": "this is the one and only draw for this rank",
            "outcome": if e_pass { "PASS" } else { "FAIL" },
            "crossed": e_pass, "crossed_at_item_count": o.crossed_at,
            "items_drawn": o.items_drawn,
            "final_wealth": o.wealth, "max_wealth_reached": o.max_wealth,
            "n_discordant": n_disc,
            "discordant_wins_aligned": o.wins, "discordant_losses_aligned": o.losses,
            "trajectory": o.trajectory_json(),
            "trajectory_is_complete": "the full W_t path is committed regardless of outcome, per ADR-045's mandatory co-report on the trajectory's SHAPE — a reader can see whether the process trended toward the boundary and ran out of budget, or stayed flat.",
        },
        "registered_power_accounting": {
            "adr": "ADR-045 § MANDATORY POWER CALCULATION, frozen before any item was drawn",
            "expected_n_discordant": REGISTERED_BAR_OF,
            "registered_crossing_bar": format!(">= {REGISTERED_BAR_WINS} of {REGISTERED_BAR_OF} discordant wins (69.2%)"),
            "realised_n_discordant": n_disc,
            "realised_discordant_wins": o.wins,
            "crossed_the_registered_bar": o.wins >= REGISTERED_BAR_WINS,
            "uninformative_threshold": UNINFORMATIVE_BELOW_N_DISC,
            "uninformative": uninformative,
            "if_uninformative": "ADR-045: with n_disc < 30 the rung is reported UNINFORMATIVE and the power model is recorded as wrong — a finding about our estimation, not about the apparatus.",
        },
        "control_vs_control_battery": condition_battery(&o.paired),
        "item_supply": {
            "adr": "ADR-036 Decision 2",
            "source": "harness/latentmesh-live/data/adaptation-512.json",
            "source_sha256_of_train_jsonl": c.train_sha,
            "consumption_order": "the file's own fixed ascending index order, sequential, never shuffled",
            "eval_holdout_lock": "eval-200 / holdout-100 untouched and still mechanically locked",
            "leakage_exclusion": {
                "rule": "ADR-024's 13 probe-overlap items, kept in force by ADR-036 Decision 2(2)",
                "excluded_item_indices": c.leakage_exclusions,
                "of_those_present_in_adaptation_512": c.excluded_present,
            },
            "disclosed_overlap_with_the_historical_probe_set": {
                "items": [1153],
                "note": "index 1153 is the ONE item adaptation-512 shares with the frozen 40-item probe set. It is NOT training leakage; ADR-036's registered exclusion rule names only the 13-item list, so it stays in the stream. Disclosed rather than silently kept.",
            },
            "disjoint_from_the_adapters_training_set": c.m5_training["split"]["item_stream_disjointness"].clone(),
            "eligible_pool_size": c.eligible,
            "tokenization_preflight_exclusions": c.tokenization_excluded,
            "sample_positions": c.site_samples,
            "split_discipline": "item-level, never token-level",
        },
        "items": o.rows,
        "summary": {
            "headline": if uninformative {
                "UNINFORMATIVE — fewer than 30 discordant pairs; ADR-045 records the power model as wrong rather than reading a verdict"
            } else if e_pass {
                "E-PROCESS PASS on a RECEIVER-ADAPTED model under ADR-036's successor protocol (adaptation-512 stream) — NOT comparable to any frozen-40-item-protocol result"
            } else {
                "E-PROCESS FAIL on a RECEIVER-ADAPTED model under ADR-036's successor protocol (adaptation-512 stream) — the wealth boundary was not crossed within N_max"
            },
            "n_evaluated": n, "n_degenerate_capture": o.degenerate,
            "e_process": {
                "outcome": if e_pass { "PASS" } else { "FAIL" },
                "crossed_at_item_count": o.crossed_at, "items_drawn": o.items_drawn,
                "final_wealth": o.wealth, "max_wealth_reached": o.max_wealth,
                "wealth_threshold": w_threshold,
                "n_discordant": n_disc, "wins": o.wins, "losses": o.losses,
            },
            "accuracy": {"aligned_real": real_c, "baseline_uninjected": base_c,
                          "zerovec_injected": zero_c, "random": rand_c,
                          "note": "raw counts on the ADAPTED receiver; levels are NOT comparable to any prior rung"},
            "nll_mean": {"aligned_real": o.mean(|q| q.real.1), "baseline_uninjected": o.mean(|q| q.base.1),
                          "zerovec_injected": o.mean(|q| q.zero.1), "random": o.mean(|q| q.rand.1)},
            "likelihood_arm": {
                "why_co_reported": "ADR-045 mandatory: accuracy alone is a DEAF endpoint, proven twice on this programme. The per-item sign tests against baseline, zerovec AND norm-matched random are in control_vs_control_battery.pairs, under the aligned_real_vs_* keys.",
                "see": "control_vs_control_battery",
            },
            "fuse_zero_is_noop_vs_baseline": {
                "expected": "EXACT identity — under fuse the zerovec condition is h += 0, at any site and on an adapted receiver too",
                "accuracy_disagreements": noop_dis, "nll_bit_identical_items": noop_exact,
                "n_items": n, "max_abs_nll_delta": noop_max, "tolerance": FUSE_NOOP_TOL,
                "pass": noop_pass,
                "gating": "NONE — operator-correctness diagnostic. It is also a check that installing the adapter did not break the injection path.",
            },
            "zerovec_vs_baseline": {
                "degenerate_under_fuse": true,
                "criterion": "2 x zerovec_correct >= baseline_correct", "pass": zerovec_pass,
                "note": "trivially satisfied because zerovec == baseline under fuse; retained for cross-rung continuity, carries NO evidential weight"},
        },
        "comparability_discipline": {
            "adr": "ADR-036 Decision 3",
            "no_p_value_translation": "This receipt reports NO exact-sign or mid-p p-value for the PRIMARY accuracy comparison. The e-process outcome is reported on its own scale and is not translated into an equivalent p-value.",
            "no_completed_rung_redrawn": "No completed rung was re-drawn. M3 through M4i stand exactly as recorded.",
            "cross_rung_levels_not_comparable": "accuracy and NLL LEVELS here are on a different model (an adapted receiver) than every prior rung's. Only the within-rung paired comparisons are interpretable.",
            "the_adapted_receiver_is_a_MATERIALLY_DIFFERENT_GENERATOR": {
                "measured": "mean generated length is 252 chars with the adapter installed vs 841 without — a 3.3x reduction. See receiver_adaptation.transfer_check_gate.decision_side_diagnostic_carried_forward for the per-item rows.",
                "what_it_is": "concision, not truncation. Accuracy is PRESERVED across the same shift (34/64 adapted vs 31/64 unadapted), and ~252 chars is about the length of a GSM8K reference solution, which is what the amended render_gold target trains toward.",
                "why_length_is_not_the_diagnostic": "the v1 adapter, which DID collapse (5/64), shortened output only 2.2x (841 -> 389). The v2 adapter shortens it MORE, 3.3x, and does not collapse. A larger length reduction therefore does not predict the decision-side failure — accuracy does. Length is reported as a description of the generator, not as a proxy for its health.",
                "CONSEQUENCE_FOR_CROSS_RUNG_USE": "the M5 adapted receiver's output distribution has shifted substantially from the one M4i and M5X ran. Any cross-rung comparison of M5 against M4i/M5X is NOT apples-to-apples on generation behaviour. M5's accuracy and NLL numbers MUST NOT be presented side by side with M4i's as if one receiver produced both; a table that lists them together must carry this caveat on the row.",
                "what_it_does_NOT_threaten": "the M5 primary (aligned vs random) is paired WITHIN this rung, with both arms on this same adapted receiver. It is unaffected — that is the point of the design.",
            },
            "firewall": "ADR-045: M5 is same-model, receiver-adapted. It tests THE APPARATUS, never transfer. Neither outcome may be cited for or against cross-model transferability.",
        },
        "gates": {
            "payload_artifact_hash_matches_m3_training_receipt": {"pass": true, "hash": c.payload_hash},
            "hand_rolled_apply_matches_trained_network": {"pass": true, "max_relative_l2_error": c.payload_golden_max_rel},
            "comparator_receipt_matches_on_every_held_fixed_factor": {"pass": true, "receipt": c.comparator_receipt},
            "m5_adapter_hash_matches_its_training_receipt": {"pass": true, "hash": a.lora.content_hash},
            "m5_adapter_matches_its_golden_pairs": {"pass": true, "max_relative_l2_error": a.golden_max_rel},
            "m5_transfer_check_passed_before_this_draw": {"pass": true},
            "adapter_installed_at_the_injection_block": {"pass": true, "block": RECEIVER_BLOCK},
            "baseline_re_measured_on_the_adapted_receiver": {"pass": true},
            "fuse_mode_recorded": {"pass": true, "mode": c.inject_mode.tag()},
            "slot_count_unchanged": {"pass": true, "slots": N_SLOTS},
            "item_supply_matches_adr_036": {"pass": true, "source": "adaptation-512", "order": "fixed index order"},
            "control_vs_control_computed_in_the_probe": {"pass": true,
                "note": "every ordered pair among the four REGISTERED conditions, on both endpoints; ADR-003's mismatched/self_generated are not registered for this rung and are absent, which the battery states in its own scope_limit_disclosed field"},
            "M5_e_process": {"pass": e_pass, "crossed_at_item_count": o.crossed_at,
                "final_wealth": o.wealth, "items_drawn": o.items_drawn, "n_discordant": n_disc},
            "power_model_informative": {"pass": !uninformative, "n_discordant": n_disc,
                "threshold": UNINFORMATIVE_BELOW_N_DISC},
            "zerovec_not_catastrophic": {"pass": zerovec_pass, "degenerate_under_fuse": true},
            "fuse_zero_payload_is_a_noop": {"pass": noop_pass, "gating": "none — diagnostic"},
        },
        "gate_pass": e_pass && zerovec_pass,
        "honest_fail_contract": "ADR-032 + ADR-036 + ADR-045: ONE registered draw per rank, no retry, no restart, no re-parametrisation. The complete W_t trajectory is committed above whatever the outcome. A powered null closes the last untested axis and is reported without softening.",
    })
}
