//! The M6 receipt literal (ADR-047), split out of `run2_m6_probe.rs` for
//! file-size discipline. The probe keeps the gates and the draw loop, which is
//! what a reader audits against the ADR.

use crate::common::m3::{RECEIVER, RECEIVER_BLOCK, SENDER};
use crate::common::m5::LoadedAdapter;
use crate::common::m6::CONDITIONS;
use crate::common::m6_battery::condition_battery;
use crate::m6_draw::{
    wins_needed, DrawOutcome, E_ALPHA, FUSE_NOOP_TOL, LAMBDA, N_MAX, UNINFORMATIVE_BELOW_N_DISC,
};
use latentmesh_runtime::inject::InjectionMode;

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
    pub priming_item: usize,
    pub tokenization_excluded: Vec<serde_json::Value>,
    pub site_samples: Vec<serde_json::Value>,
    pub phase1_receipt: &'a str,
}

pub fn receipt(c: &ReceiptCtx<'_>, o: &DrawOutcome) -> serde_json::Value {
    let (n, n_disc, e_pass) = (o.n(), o.n_disc(), o.e_pass());
    let (noop_dis, noop_exact, noop_max, noop_pass) = o.noop_stats();
    let (real_c, mism_c, base_c, zero_c, rand_c) = (
        o.count(|q| q.real.0),
        o.count(|q| q.mism.0),
        o.count(|q| q.base.0),
        o.count(|q| q.zero.0),
        o.count(|q| q.rand.0),
    );
    let w_threshold = 1.0 / E_ALPHA;
    let a = &c.adapter;

    serde_json::json!({
        "stage": "run2-M6-manifold-content-factorial-probe",
        "design": "docs/adr/047-m6-manifold-content-factorial.md (everything above '## Outcomes' frozen before any item was drawn; deviations are numbered coordinator errors). Evaluation protocol = docs/adr/036-successor-rung-evaluation-protocol.md (e-process, adaptation-512 stream).",
        "env": c.env,
        "pre_committed": true,
        "rank": c.rank,
        "variant": "pertokenlast-fuse-questiontail-receiveradapted-contentaxis",

        "what_this_rung_asks": {
            "question": "M5's aligned-beats-random result is ambiguous: `random` is a norm-matched Gaussian, which is wrong on TWO axes at once — content-free AND off-manifold — so beating it identifies neither factor. At every M5 rank accuracy ordered baseline > aligned > random, so 'aligned beats random' is equally explained by the on-manifold payload being a GENTLER perturbation, with no content transmitted.",
            "the_fix": "add `mismatched` — another episode's GENUINE payload. On-manifold, produced by the identical computation, norm-matched by the identical rescale rule, and wrong ONLY in which episode's content it encodes. Beating it cannot be explained by disruption magnitude.",
            "registered_primary": "aligned_real vs mismatched, on the decision endpoint. This is NOT M5's primary and its numbers are not comparable to M5's.",
        },

        "five_conditions_not_six": {
            "registered": "ADR-047 §2 registered a 2x2 — {aligned, aligned_displaced} x {mismatched, random} — with TWO primaries, one per axis, and §8 asked for a 30-pair battery at six conditions.",
            "withdrawn": "aligned_displaced, and with it the MANIFOLD primary at the 0.90 dose.",
            "coordinator_error": 24,
            "why": "The §5 manipulation check ran (CPU-only) and the doses PASSED the registered gate 4/4 — the cell was withdrawn on evidence BEYOND the gate, not because it failed. Measured typicality equalled c x typicality(aligned) to within 0.0035 at every dose, so gate arm (b) 'strictly between aligned and random' was satisfied by construction rather than by the cell being sound: the two gate arms measured the same quantity. Under every instrument in the lens kit a displaced payload was a partial step toward `random` (typicality 0.6814 -> 0.5145 -> -0.0006, item-invariance 0.6670 -> 0.3760 -> 0.0000, entropy 3.32 -> 4.92 -> 5.45), because an exact rotation attenuates true content by exactly c: at the 0.90 dose, signal 0.900 against noise 0.436. So aligned_displaced(c) is a point on the aligned->random SEGMENT — M5's existing diagonal sampled at intermediate points — and a dose-response along it cannot attribute, since both factors move together.",
            "structural_result": "In 1536 dimensions a generic displacement direction is almost surely BOTH off-manifold AND content-free, so rotating toward one necessarily moves both factors in lockstep. That is the geometric reason the off-diagonal 'right content, wrong manifold' cell resists construction at all.",
            "phase1_receipt": c.phase1_receipt,
            "unaffected": "The CONTENT axis is untouched by this: `aligned` and `mismatched` are both on-manifold and both genuine payloads, so nothing above bears on it. Its power anchor and its n-dependent bar stand as registered.",
            "battery_pairs": "20, not 30 — five conditions rather than six.",
        },

        "protocol_identity": {
            "statistics": "ADR-036 anytime-valid Bernoulli e-process (ADR-030 §3.2 mechanics verbatim)",
            "item_supply": "adaptation-512, fixed index order, ADR-024's 13-item leakage exclusion applied",
            "era": "SUCCESSOR-RUNG ERA. Not the frozen 40-item sign-test protocol that governed M3 through M4h Stage 1.",
            "must_travel_with_every_headline_number": "Per ADR-036 Decision 3, any write-up quoting a number from this receipt must state in the same sentence that it was produced under the e-process protocol on the adaptation-512 stream, on a RECEIVER-ADAPTED model, with the primary being aligned vs MISMATCHED.",
        },

        "the_single_changed_factor": {
            "factor": "THE CONTROL SET — a fifth condition, `mismatched`, and with it a different registered primary",
            "comparator_receipt": c.comparator_receipt,
            "held_fixed_against_the_comparator": [
                "the payload artifact (M3's trained reconstruction MLP, byte-for-byte)",
                "the payload derivation (Variant::PerTokenLast, via common::m6::capture_payload, whose body is m3.rs's PerTokenLast branch)",
                "the operator (InjectionMode::Fuse)",
                "the site (Site::QuestionTail)",
                "the slot count, the depth (block 14, cell L18->L14), and the rescale rule",
                "the decoding (greedy, batch = 1, <= 400 new tokens)",
                "the item stream (adaptation-512, fixed index order) and the statistic (lambda = 0.30, PASS at W >= 20, N_max = 300)",
                "the receiver, including its trained adapter",
            ],
            "what_is_NOT_claimed": "M6 is not a re-run of M5 with one more arm bolted on: because the primary changed, its e-process is a different statistic on the same stream and its wealth is NOT comparable to M5's. The four M5 conditions are re-measured here on this receiver in this run; none is copied from M5's receipt.",
        },

        "mismatched_control": {
            "definition": "the PREVIOUS drawn item's aligned payload, carried forward.",
            "priming": format!("stream order 0 has no predecessor, so it is primed from item {} — the last eligible index, asserted to be OUTSIDE the drawn stream. Ported from run3_gated_text_probe.rs:313-317, 361-363, 516-517.", c.priming_item),
            "no_lookahead": "the carry-forward never reads an item the draw has not already evaluated, so early stopping stays honest: nothing about item i+1 can influence item i's outcome.",
            "norm_matching": "free, not a separate mechanism. M5's rescale rule scales every injected vector to the receiver's natural inject-block median, so aligned, mismatched and random all arrive at identical L2 by construction. Each item's row stores all three realised norms so a reader can verify rather than trust it.",
            "degenerate_items": "a degenerate sender pass produces no payload, so the carried-forward control is left at the last item that produced one. There is no alternative that does not either fabricate a payload or look ahead.",
            "prior_art": "arXiv:2607.26773's 'other-example message'. The term and the control are theirs; cited, not re-derived.",
        },

        "receiver_adaptation": {
            "training_receipt": format!("run2-m5-training-receipt-cellL18toL14-r{}.json", c.rank),
            "transfer_receipt": format!("run2-m5-transfer-receipt-cellL18toL14-r{}.json", c.rank),
            "adapter_content_hash": a.lora.content_hash,
            "adapter_init_content_hash": a.init_content_hash,
            "rank": a.lora.rank, "alpha": a.lora.alpha,
            "scaling": a.lora.scaling(), "param_count": a.param_count,
            "golden_verification": {
                "pairs": a.golden_pairs, "max_relative_l2_error": a.golden_max_rel,
                "tolerance": c.payload_golden_tol, "pass": true,
            },
            "training_receipt_carried": c.m5_training["artifact"].clone(),
            "transfer_check_gate": {
                "pass": true,
                "receipt_is_M5s_committed_one_not_re_run": "the transfer check measures the composed->fused BF16 agreement of THIS adapter. M6 changes no adapter, so re-running it would reproduce the same numbers under a name that would overwrite a frozen M5 receipt. It is read and gated on, not regenerated.",
                "criterion": c.transfer["gates"]["transfer"]["criterion"].clone(),
                "mean_fused_train_target_ce_adapter_on": c.transfer["summary"]["mean_fused_train_target_ce_adapter_on"].clone(),
                "mean_fused_train_target_ce_adapter_off": c.transfer["summary"]["mean_fused_train_target_ce_adapter_off"].clone(),
                "n_evaluated": c.transfer["summary"]["n_evaluated"].clone(),
                "decision_side_diagnostic_carried_forward": c.transfer["summary"]["generation_diagnostic"].clone(),
                "why_carried_forward": "ADR-045 error #22 made the decision-side generation diagnostic mandatory and NON-gating, and ADR-047 §8 carries that forward. It is copied into THIS receipt so a reader of the draw alone can see whether the adapted receiver still reasons — without it, 'the receiver stopped answering' and 'the channel carries nothing' are indistinguishable.",
            },
        },

        "the_confound_and_why_the_primary_is_immune": {
            "confound": "a task-loss-adapted receiver could simply be a better GSM8K solver, improving regardless of injected content.",
            "baseline_re_measured_on_this_receiver": true,
            "no_frozen_receiver_baseline_is_reused_anywhere": "every accuracy and NLL number in this receipt was measured on the adapted receiver in THIS run.",
            "primary": "aligned vs mismatched — both arms are genuine payloads injected into the same adapted receiver by the same operator at the same site with the same norm, so a general fine-tuning effect raises both and cancels. It also cancels the disruption-magnitude explanation, which aligned-vs-random could not.",
            "secondary_and_confounded": "aligned vs baseline. Reported, not immune, and readable only against the freshly measured adapted baseline above.",
        },

        "config": {
            "sender": SENDER, "receiver": RECEIVER, "receiver_is_adapted": true,
            "receiver_block": RECEIVER_BLOCK,
            "injection_site": c.site_tag,
            "injection_site_description": c.site_description,
            "injection_operator": {"mode": c.inject_mode.tag()},
            "conditions": CONDITIONS,
            "randvec_seed_base": c.randvec_seed_base,
            "pad_id_resolved_but_unused_at_this_site": c.pad_id,
            "transform": {
                "training_receipt": c.payload_training_receipt,
                "artifact": c.payload_artifact,
                "content_hash": c.payload_hash,
                "golden": {
                    "file": c.payload_golden_file,
                    "pairs": c.payload_golden_pairs,
                    "seed_chacha8": c.payload_golden_seed,
                    "max_relative_l2_error": c.payload_golden_max_rel,
                    "tolerance": c.payload_golden_tol,
                },
            },
        },

        "item_supply": {
            "source": "adaptation-512 over GSM8K train.jsonl",
            "train_sha256": c.train_sha,
            "leakage_exclusions": c.leakage_exclusions,
            "excluded_present_on_this_split": c.excluded_present,
            "eligible": c.eligible,
            "priming_item_outside_the_stream": c.priming_item,
            "n_max": N_MAX,
            "tokenization_excluded_before_any_forward_pass": c.tokenization_excluded,
            "site_samples": c.site_samples,
        },

        "e_process": {
            "primary": "aligned_real vs mismatched, decision endpoint",
            "lambda": LAMBDA, "alpha": E_ALPHA, "wealth_threshold": w_threshold,
            "n_max": N_MAX,
            "items_drawn": o.items_drawn,
            "degenerate_items": o.degenerate,
            "n_discordant": n_disc,
            "discordant_wins_aligned": o.wins,
            "discordant_losses": o.losses,
            "final_wealth": o.wealth,
            "max_wealth": o.max_wealth,
            "crossed": e_pass,
            "crossed_at_item": o.crossed_at,
            "trajectory": o.trajectory_json(),
        },

        "power_and_the_bar": {
            "authority": "the wealth rule, and only the wealth rule. `e_process.crossed` IS the verdict.",
            "why_no_stored_bar_field": "in M5 a derived bar field was wrong TWICE, in both directions: first as a fixed count valid only at n_disc = 65, then as a fixed rate that disagreed with the wealth rule in 34 (n, wins) combinations. The bar is n-DEPENDENT, so it is recomputed from the same lambda and alpha the wealth process uses and reported as a diagnostic beside the verdict, never as it.",
            "wins_needed_at_the_realised_n_disc": wins_needed(n_disc),
            "realised_n_disc": n_disc,
            "registered_expectation": "50-60, anchored on M5's MEASURED aligned-vs-baseline discordance (60/50/58 at ranks 1/2/4) — the nearest on-manifold analogue, since aligned-vs-mismatched had never been drawn.",
            "uninformative_below": UNINFORMATIVE_BELOW_N_DISC,
            "uninformative": o.uninformative(),
            "if_uninformative": "ADR-047 §7.5: reported as uninformative for this pair and the power model recorded as wrong. NEVER reported as a null.",
        },

        "summary": {
            "n_items_paired": n,
            "accuracy": {
                "aligned_real": real_c,
                "mismatched": mism_c,
                "baseline_uninjected": base_c,
                "zerovec_injected": zero_c,
                "random": rand_c,
            },
            "mean_nll_gold": {
                "aligned_real": o.mean(|q| q.real.1),
                "mismatched": o.mean(|q| q.mism.1),
                "baseline_uninjected": o.mean(|q| q.base.1),
                "zerovec_injected": o.mean(|q| q.zero.1),
                "random": o.mean(|q| q.rand.1),
            },
            "mean_generated_chars": {
                "why": "ADR-045's non-gating degenerate-output instrument, made permanent. A condition that 'wins' by collapsing into a terse answer is not a channel effect.",
                "aligned_real": mean_field(o, "aligned_real"),
                "mismatched": mean_field(o, "mismatched"),
                "baseline_uninjected": mean_field(o, "baseline_uninjected"),
                "zerovec_injected": mean_field(o, "zerovec_injected"),
                "random": mean_field(o, "random"),
            },
            "zerovec_is_baseline_under_fuse": {
                "why": "under Fuse the zero payload is h += 0, an exact no-op, so this pair is an operator-correctness check rather than an independent control. It also means one of the 20 battery pairs is degenerate by construction, which is declared here rather than discovered by a reader.",
                "accuracy_disagreements": noop_dis,
                "bit_identical_nlls": noop_exact,
                "max_abs_nll_delta": noop_max,
                "tolerance": FUSE_NOOP_TOL,
                "pass": noop_pass,
            },
        },

        "control_vs_control_battery": condition_battery(&o.paired),

        "delta_v": {
            "computed_here": false,
            "why": "docs/research/034 §3 prices one properly powered verify_edge draw at ~3 GPU-h, more than this entire rung. Registered as a single post-hoc characterisation, never an online signal; it was not used anywhere in this rung.",
        },

        "firewall": {
            "same_model": true,
            "scope": "M6 tests the APPARATUS, never transfer. Neither outcome may be cited for or against cross-model transferability. ADR-024's scope freeze and research/054's scope section apply unaltered.",
        },

        "items": o.rows,
    })
}

/// Mean generated length for one condition, read back out of the per-item rows
/// (they are the stored source; this is a view over them).
fn mean_field(o: &DrawOutcome, key: &str) -> f64 {
    let vals: Vec<f64> = o
        .rows
        .iter()
        .filter_map(|r| r["generated_chars"][key].as_f64())
        .collect();
    if vals.is_empty() {
        return 0.0;
    }
    vals.iter().sum::<f64>() / vals.len() as f64
}
