//! The M5 transfer-check receipt (ADR-045), split out of
//! `run2_m5_transfer_check.rs` for file-size discipline.
//!
//! Declared from the transfer check rather than from `common/mod.rs`: that
//! module is `#[path]`-included into every example, and this literal's `json!`
//! expansion is large.

/// Everything the transfer receipt records.
pub struct TransferCtx<'a> {
    pub rank: usize,
    pub env: serde_json::Value,
    pub criterion: &'a str,
    pub receiver: &'a str,
    pub inject_block: usize,
    pub n_slots: usize,
    pub site: &'a str,
    pub mode_tag: &'a str,
    pub mode_equation: &'a str,
    pub seq_cap: usize,
    pub min_target: usize,
    pub adapter_hash: &'a str,
    pub adapter_rank: usize,
    pub adapter_alpha: f64,
    pub adapter_scaling: f64,
    pub adapter_params: usize,
    pub adapter_golden_pairs: usize,
    pub adapter_golden_max_rel: f32,
    pub payload_hash: &'a str,
    pub payload_golden_max_rel: f32,
    pub rows: Vec<serde_json::Value>,
    pub n_eval: usize,
    pub skipped: Vec<usize>,
    pub mean_on: f64,
    pub mean_off: f64,
    pub mean_covered: f64,
    pub mean_probe_on: f64,
    pub mean_probe_off: f64,
    pub wins: usize,
    pub losses: usize,
    pub p_sign: f64,
    pub probe_wins: usize,
    pub probe_losses: usize,
    pub p_probe: f64,
    pub composed_improvement: serde_json::Value,
    pub gen_n: usize,
    pub acc_on: usize,
    pub acc_off: usize,
    pub chars_on: f64,
    pub chars_off: f64,
    pub gen_rows: Vec<serde_json::Value>,
    pub pass: bool,
    pub smoke: bool,
    pub wall_clock_s: f32,
}

/// The decision-side diagnostic, MANDATORY since ADR-045 coordinator error #22.
///
/// Generates on HOLDOUT items only (never draw items) with the adapter
/// installed and removed, under the draw's own greedy decoding, in the
/// **baseline** condition — the cleanest read of "is the adapted receiver
/// still a GSM8K solver at all".
///
/// It deliberately does **not** gate. Gating the adapter on GSM8K accuracy
/// would select it against the very general-fine-tuning confound the
/// registered primary is designed to exclude. It is reported so that a null
/// can be read correctly: *the receiver stopped answering* is a different
/// finding from *the channel carries nothing*. Without it, the v1 adapter's
/// 508W/2L likelihood win would have waved a receiver scoring 5/64 straight
/// into an irreversible draw.
///
/// Returns `(correct_on, correct_off, total_chars_on, total_chars_off, rows)`.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn generation_diagnostic(
    receiver: &mut latentmesh_runtime::QwenRuntime,
    lora: &latentmesh_runtime::lora::ResidualLora,
    all_items: &[crate::common::Gsm8kItem],
    row_to_item: &[usize],
    rows: Vec<usize>,
    pad_id: u32,
    max_new_tokens: usize,
) -> anyhow::Result<(usize, usize, f64, f64, Vec<serde_json::Value>)> {
    use crate::common::m3::{build_site_prompt, Site};
    use latentmesh_runtime::sampler::{Sampler, Sampling};
    let e = anyhow::Error::msg;
    let (mut acc_on, mut acc_off, mut chars_on, mut chars_off) = (0usize, 0usize, 0f64, 0f64);
    let mut out = Vec::with_capacity(rows.len());
    println!(
        "degenerate-generation diagnostic over {} holdout items (baseline condition, adapter on \
         vs off)...",
        rows.len()
    );
    for row in rows {
        let item = &all_items[row_to_item[row]];
        let sp = build_site_prompt(receiver, item, pad_id, Site::QuestionTail)?;
        let mut gen = |on: bool| -> anyhow::Result<(bool, usize)> {
            receiver.model.set_residual_lora(on.then(|| lora.clone()));
            let mut s = Sampler::new(Sampling::Greedy, 0);
            let g = receiver
                .generate(&sp.tokens, None, &mut s, max_new_tokens, false)
                .map_err(e)?;
            let ok = crate::common::extract_answer(&g.text)
                .is_some_and(|a| crate::common::answers_equal(&a, &item.gold));
            Ok((ok, g.text.chars().count()))
        };
        let (ok_on, len_on) = gen(true)?;
        let (ok_off, len_off) = gen(false)?;
        acc_on += usize::from(ok_on);
        acc_off += usize::from(ok_off);
        chars_on += len_on as f64;
        chars_off += len_off as f64;
        out.push(serde_json::json!({
            "row": row, "item": item.index,
            "correct_adapter_on": ok_on, "correct_adapter_off": ok_off,
            "generated_chars_adapter_on": len_on, "generated_chars_adapter_off": len_off,
        }));
    }
    receiver.model.set_residual_lora(None);
    println!(
        "generation diagnostic: baseline accuracy adapter-on {acc_on}/{} vs off {acc_off}/{}; \
         mean generated chars {:.0} vs {:.0}",
        out.len(),
        out.len(),
        chars_on / out.len() as f64,
        chars_off / out.len() as f64
    );
    Ok((acc_on, acc_off, chars_on, chars_off, out))
}

/// The transfer-check receipt, one auditable object.
pub fn receipt(c: TransferCtx<'_>) -> serde_json::Value {
    serde_json::json!({
        "stage": "run2-m5-transfer-check",
        "design": "ADR-045 M5, inheriting ADR-024 M4c's registered composed-vs-fused caveat mitigation. The criterion was frozen in the M5 training receipt BEFORE this check ran. Inference-only: no draw items, no generation, no e-process.",
        "rank": c.rank,
        "env": c.env,
        "criterion_frozen_in_training_receipt": c.criterion,
        "config": {
            "receiver": c.receiver,
            "forward": "vendored FUSED BF16 — the forward the draw runs",
            "inject_after_block": c.inject_block, "n_slots": c.n_slots,
            "site": c.site,
            "operator": {"mode": c.mode_tag, "equation": c.mode_equation},
            "delivery": "M3's frozen hand-rolled MLP on the item's LAST generated-span dump row -> rescale to the VENDORED natural block-14 median -> fuse at the 8 question-tail positions (the draw's delivery path exactly)",
            "arms": "the SAME receiver with the trained adapter installed and removed. B is zero-initialised, so 'removed' and 'at init' are the same function; the init artifact and its goldens are committed regardless.",
            "span_rule": format!("train target = min(render_gold_len, {} - prompt_len), item skipped only if < {} (M4c's rule, matching the trainer); the recomputed skip set is asserted equal to the training receipt's", c.seq_cap, c.min_target),
            "training_target": "render_gold — the rendered gold SOLUTION, per ADR-045 § Deviations coordinator error #22. The v1 target ('#### {gold}' alone) produced a receiver that answered 5/64.",
            "items": "the leakage-safe holdout side only — never trained on, and disjoint from the draw's adaptation-512 stream",
            "adapter": {
                "content_hash": c.adapter_hash,
                "rank": c.adapter_rank, "alpha": c.adapter_alpha,
                "scaling": c.adapter_scaling, "param_count": c.adapter_params,
                "golden_pairs": c.adapter_golden_pairs,
                "golden_max_relative_l2_error": c.adapter_golden_max_rel,
            },
            "payload_adapter": {"content_hash": c.payload_hash, "golden_max_rel": c.payload_golden_max_rel},
        },
        "items": c.rows,
        "summary": {
            "n_evaluated": c.n_eval, "n_skipped": c.skipped.len(), "skipped_rows": c.skipped,
            "gate_is_on_the_training_target": true,
            "mean_fused_train_target_ce_adapter_on": c.mean_on,
            "mean_fused_train_target_ce_adapter_off": c.mean_off,
            "fused_improvement_nats": c.mean_off - c.mean_on,
            "train_target_mean_covered_fraction": c.mean_covered,
            "probe_endpoint_reported_not_gating": {
                "target": "#### {gold} — the probe's own teacher-forced NLL target, frozen and unchanged by the error-#22 amendment",
                "mean_fused_nll_adapter_on": c.mean_probe_on,
                "mean_fused_nll_adapter_off": c.mean_probe_off,
                "improvement_nats": c.mean_probe_off - c.mean_probe_on,
                "wins_adapter_on_lower": c.probe_wins, "losses": c.probe_losses,
                "sign_test_p_one_sided": c.p_probe,
                "comparable_with": "the v1 receipt's mean_fused_nll_adapter_on/off, which measured this same quantity when it was also the training target",
            },
            "composed_improvement_nats_from_training_receipt": c.composed_improvement,
            "wins_adapter_on_lower": c.wins, "losses": c.losses,
            "sign_test_p_one_sided_secondary": c.p_sign,
            "generation_diagnostic": {
                "gating": "NONE",
                "why": "an adapter trained on an answer-shaped target can learn the FORMAT and stop reasoning — measured, not hypothetical: the v1 adapter trained on '#### {gold}' alone scored 5/64 against 31/64 (ADR-045 error #22). That collapse makes the draw uninformative — almost no discordant pairs, because every condition is wrong — for a reason that has nothing to do with the channel. Measured here, before the draw, on HOLDOUT items only.",
                "why_not_gating": "gating the adapter on GSM8K accuracy would select it against the very confound ADR-045 keeps out of the primary. This is reported so a null can be read correctly: 'the receiver stopped answering' is a different finding from 'the channel carries nothing'.",
                "condition": "baseline (no injection), greedy, batch=1, max_new_tokens=400 — the draw's own decoding",
                "n_items": c.gen_n,
                "accuracy_adapter_on": c.acc_on, "accuracy_adapter_off": c.acc_off,
                "mean_generated_chars_adapter_on": c.chars_on / c.gen_n as f64,
                "mean_generated_chars_adapter_off": c.chars_off / c.gen_n as f64,
                "items": c.gen_rows,
            },
            "what_this_does_NOT_establish": "nothing about the channel. This measures only that a training improvement survives the composed->fused BF16 crossing, on items the draw never sees. An adapter that merely made the receiver a better GSM8K solver would pass this check too — which is exactly why ADR-045's primary is aligned vs random on the same adapted receiver.",
        },
        "gates": {
            "m3_payload_hash_matches_its_training_receipt": {"pass": true},
            "m5_adapter_hash_matches_the_m5_training_receipt": {"pass": true},
            "m5_adapter_matches_its_golden_pairs": {"pass": true, "max_relative_l2_error": c.adapter_golden_max_rel},
            "prompt_parity_vs_streams": {"pass": true},
            "skip_set_matches_training_receipt": {"pass": true},
            "transfer": {"pass": c.pass, "criterion": "mean fused CE of the TRAINING target (render_gold) with the adapter on < with it off"},
        },
        "gate_pass": c.pass,
        "verdict": if c.pass {
            "the training improvement TRANSFERS across the composed->fused BF16 gap; the M5 draw may run"
        } else {
            "the training improvement does NOT transfer — the M5 draw must NOT be invoked (a null would be confounded by the numeric gap); this receipt is the honest M5 outcome for that branch"
        },
        "smoke_run": c.smoke,
        "wall_clock_s": c.wall_clock_s,
    })
}
