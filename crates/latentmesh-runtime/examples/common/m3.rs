//! Run-2 probe internals (ADR-024): the frozen per-item four-condition
//! execution, shared constants, and the M3 eval variants.
//! Split out of `run2_m3_probe.rs` purely for file-size discipline — the
//! protocol here is ADR-023's frozen S1a/S2b protocol, byte-for-byte the
//! same mechanics as `s2b_bridge_probe.rs`'s `run_item`, with only the
//! transform application swapped for the trained adapter.
//!
//! M4 reuse (ADR-024 sub-rung ladder): the sender solve + per-token rows
//! capture ([`sender_solve_capture_rows`]) and the receiver-side frozen
//! four-condition block ([`four_conditions`]) are the SAME code paths M3's
//! per-token variant ran — extracted, not duplicated, so the frozen
//! protocol exists in exactly one place and cannot silently diverge
//! between rungs.

use super::mlp::MlpTransform;
use latentmesh_runtime::{
    capture::{forward_capture, forward_capture_multi_with_rows},
    inject::{teacher_forced_nll, InjectionMode, InjectionSpec},
    norms,
    sampler::{Sampler, Sampling},
    QwenRuntime,
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::ops::Range;

pub const SENDER: &str = "Qwen/Qwen2.5-3B-Instruct";
pub const RECEIVER: &str = "Qwen/Qwen2.5-1.5B-Instruct";
/// S2 winner cell L18→L14 (ADR-023 Deviation 6; ADR-024 M3/M4's cell).
pub const SENDER_BLOCK: usize = 18;
pub const RECEIVER_BLOCK: usize = 14;
pub const ITEM_SEED: u64 = 0x51A1;
pub const RANDVEC_SEED_BASE: u64 = 0x51A2_0000;
pub const MAX_NEW_TOKENS: usize = 400;
pub const N_SLOTS: usize = 8;
pub const ALPHA: f64 = 0.05;
pub const SYSTEM: &str = "You are a careful math tutor.";
pub const GOLDEN_REL_TOL: f32 = 1e-5;

/// The two ADR-024-registered M3 eval variants, plus ADR-024 M4h Stage 1's
/// de-pooled payload derivation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Variant {
    /// (i) MLP per generated-span token state, then mean-pooling over the
    /// TRANSLATED stream, then the frozen 8-slot injection.
    PerToken,
    /// (ii) pool the sender's per-token states first (run-1 pipeline
    /// shape), then the SAME per-token-trained MLP on the pooled vector.
    Pooled,
    /// ADR-024 M4h Stage 1: MLP per generated-span token state exactly as in
    /// [`Variant::PerToken`], but the payload is the **LAST token's** output
    /// instead of the mean over the span. Same 8-slot broadcast, same slot
    /// count (protected), same trained weights — the mean is the only thing
    /// removed (`docs/research/040`).
    PerTokenLast,
}

impl Variant {
    pub fn tag(self) -> &'static str {
        match self {
            Variant::PerToken => "pertoken",
            Variant::Pooled => "pooled",
            Variant::PerTokenLast => "pertokenlast",
        }
    }
}

/// Per-condition (correct, nll_gold) quad — identical to S2b's.
pub struct Quad {
    pub real: (bool, f32),
    pub base: (bool, f32),
    pub zero: (bool, f32),
    pub rand: (bool, f32),
}

pub type QuadRow = (serde_json::Value, Quad);

/// Sender first-pass outcome (step 1), carried into the per-item row JSON.
pub struct SenderPass {
    pub first_pass_correct: bool,
    pub first_pass_answer: Option<String>,
    pub generated_tokens: usize,
}

/// Capture metadata for the per-item row JSON.
pub struct CaptureMeta {
    pub hidden_size: usize,
    pub pooled_l2_raw: f32,
    pub span: Range<usize>,
    pub variant: &'static str,
}

/// Sender solve + per-token rows capture (steps 1–2a of the frozen
/// protocol): greedy solve, then teacher-forced re-prefill with the tap
/// after the sender sweep block, returning the raw `[n_rows × hidden]`
/// generated-span rows (plus the natural pooled vector for diagnostics).
/// `Ok(None)` = degenerate capture pass (no generated tokens), skipped.
pub struct SenderRows {
    pub pass: SenderPass,
    pub rows: Vec<f32>,
    pub n_rows: usize,
    pub pooled: Vec<f32>,
    pub hidden_size: usize,
    pub span: Range<usize>,
}

pub fn sender_solve_capture_rows(
    sender: &mut QwenRuntime,
    item: &super::Gsm8kItem,
    device: &candle_core::Device,
) -> anyhow::Result<Option<SenderRows>> {
    let e = anyhow::Error::msg;
    let fmt = super::ANSWER_FORMAT;
    let mut greedy = Sampler::new(Sampling::Greedy, 0);
    let cap_prompt = QwenRuntime::chat_prompt(SYSTEM, &format!("{}\n\n{fmt}", item.question));
    let cap_tokens = sender.encode(&cap_prompt).map_err(e)?;
    let gen = sender
        .generate(&cap_tokens, None, &mut greedy, MAX_NEW_TOKENS, false)
        .map_err(e)?;
    if gen.tokens.is_empty() {
        return Ok(None);
    }
    let full: Vec<u32> = cap_tokens
        .iter()
        .chain(gen.tokens.iter())
        .copied()
        .collect();
    let span = cap_tokens.len()..full.len();
    let (_, mut caps) =
        forward_capture_multi_with_rows(&mut sender.model, &full, &[SENDER_BLOCK], span, device)
            .map_err(e)?;
    let cwr = caps.remove(0);
    let first_pass_answer = super::extract_answer(&gen.text);
    let first_pass_correct = first_pass_answer
        .as_deref()
        .is_some_and(|a| super::answers_equal(a, &item.gold));
    let n_rows = cwr.capture.span.len();
    Ok(Some(SenderRows {
        pass: SenderPass {
            first_pass_correct,
            first_pass_answer,
            generated_tokens: gen.tokens.len(),
        },
        rows: cwr.rows,
        n_rows,
        pooled: cwr.capture.pooled,
        hidden_size: cwr.capture.hidden_size,
        span: cwr.capture.span,
    }))
}

/// One frozen-protocol item for the M3 MLP: sender solve + capture,
/// variant-specific MLP application, then the four paired receiver
/// conditions.
// The frozen protocol's parameter list, plus ADR-024 M4g's injection-mode
// selector. Kept flat and explicit rather than bundled into a struct: every
// one of these is a registered protocol element, and a reader auditing a
// probe against the ADR should see them named at the call site.
#[allow(clippy::too_many_arguments)]
pub fn run_item(
    sender: &mut QwenRuntime,
    receiver: &mut QwenRuntime,
    transform: &MlpTransform,
    item: &super::Gsm8kItem,
    pad_id: u32,
    variant: Variant,
    mode: InjectionMode,
    device: &candle_core::Device,
) -> anyhow::Result<Option<QuadRow>> {
    let e = anyhow::Error::msg;

    // 1)+2) Sender capture pass + variant-specific trained-MLP application.
    let (aligned, sender_pass, meta) = match variant {
        Variant::PerToken | Variant::PerTokenLast => {
            let Some(sr) = sender_solve_capture_rows(sender, item, device)? else {
                return Ok(None);
            };
            anyhow::ensure!(sr.hidden_size == super::mlp::D_IN);
            // The ONE difference M4h Stage 1 introduces: mean over the
            // translated span, or that span's last translated token.
            let aligned = if variant == Variant::PerTokenLast {
                transform.apply_last_row(&sr.rows, sr.n_rows)
            } else {
                transform.apply_rows_then_pool(&sr.rows, sr.n_rows)
            };
            let meta = CaptureMeta {
                hidden_size: sr.hidden_size,
                pooled_l2_raw: norms::l2(&sr.pooled),
                span: sr.span.clone(),
                variant: variant.tag(),
            };
            (aligned, sr.pass, meta)
        }
        Variant::Pooled => {
            let fmt = super::ANSWER_FORMAT;
            let mut greedy = Sampler::new(Sampling::Greedy, 0);
            let cap_prompt =
                QwenRuntime::chat_prompt(SYSTEM, &format!("{}\n\n{fmt}", item.question));
            let cap_tokens = sender.encode(&cap_prompt).map_err(e)?;
            let gen = sender
                .generate(&cap_tokens, None, &mut greedy, MAX_NEW_TOKENS, false)
                .map_err(e)?;
            if gen.tokens.is_empty() {
                return Ok(None);
            }
            let full: Vec<u32> = cap_tokens
                .iter()
                .chain(gen.tokens.iter())
                .copied()
                .collect();
            let span = cap_tokens.len()..full.len();
            let (_, cap) =
                forward_capture(&mut sender.model, &full, SENDER_BLOCK, span, device).map_err(e)?;
            anyhow::ensure!(cap.hidden_size == super::mlp::D_IN);
            let aligned = transform.apply(&cap.pooled);
            let first_pass_answer = super::extract_answer(&gen.text);
            let first_pass_correct = first_pass_answer
                .as_deref()
                .is_some_and(|a| super::answers_equal(a, &item.gold));
            let meta = CaptureMeta {
                hidden_size: cap.hidden_size,
                pooled_l2_raw: norms::l2(&cap.pooled),
                span: cap.span.clone(),
                variant: variant.tag(),
            };
            (
                aligned,
                SenderPass {
                    first_pass_correct,
                    first_pass_answer,
                    generated_tokens: gen.tokens.len(),
                },
                meta,
            )
        }
    };
    four_conditions(
        receiver,
        item,
        pad_id,
        &aligned,
        &sender_pass,
        &meta,
        mode,
        device,
    )
    .map(Some)
}

/// Steps 3–5 of the frozen protocol (receiver side), shared by every run-2
/// rung: placeholder-slot prompt (S1a wording), natural inject-block norms,
/// the four paired conditions, and the per-item row JSON.
///
/// `mode` selects the residual operator (ADR-024 M4g's single changed
/// factor). It is the ONLY thing that varies here between rungs: the three
/// `InjectionSpec`s below — the aligned payload, the per-item seeded Gaussian
/// norm-matched to the effective aligned vector, and the true zero vector —
/// are constructed identically under both operators, so no control is
/// redefined by changing it. What DOES change is what the zero payload means:
/// under `Overwrite` it zeroes the eight rows (destructive), under `Fuse` it
/// is `h += 0` and therefore an exact no-op equal to the uninjected baseline.
/// That consequence is declared in M4g's training receipt before the draw and
/// measured in its probe receipt; it is not papered over with a substitute
/// control.
#[allow(clippy::too_many_arguments)]
pub fn four_conditions(
    receiver: &mut QwenRuntime,
    item: &super::Gsm8kItem,
    pad_id: u32,
    aligned: &[f32],
    sender_pass: &SenderPass,
    meta: &CaptureMeta,
    mode: InjectionMode,
    device: &candle_core::Device,
) -> anyhow::Result<QuadRow> {
    let e = anyhow::Error::msg;
    let fmt = super::ANSWER_FORMAT;

    // 3) Receiver injection prompt with placeholder slots (S1a wording).
    let slots = "<|fim_pad|>".repeat(N_SLOTS);
    let inj_prompt = QwenRuntime::chat_prompt(
        SYSTEM,
        &format!(
            "{}\n\nA compressed latent hint from a previous solver is stored in these slots: [{slots}]\n\n{fmt}",
            item.question
        ),
    );
    let inj_tokens = receiver.encode(&inj_prompt).map_err(e)?;
    let positions = QwenRuntime::placeholder_positions(&inj_tokens, pad_id);
    anyhow::ensure!(positions.len() == N_SLOTS, "slot count mismatch");

    // 4) Natural inject-block norms of the receiver (rescale target).
    let (_, nat_cap) = forward_capture(
        &mut receiver.model,
        &inj_tokens,
        RECEIVER_BLOCK,
        0..inj_tokens.len(),
        device,
    )
    .map_err(e)?;
    let natural = norms::stats(nat_cap.per_position_l2.clone());

    let aligned_l2 = norms::l2(aligned);
    let real = InjectionSpec {
        after_block: RECEIVER_BLOCK,
        positions: positions.clone(),
        vector: aligned.to_vec(),
        scale: Some(natural.median / aligned_l2),
        mode,
    };
    let target_l2 = norms::l2(&real.effective_vector());
    let mut vrng = ChaCha8Rng::seed_from_u64(RANDVEC_SEED_BASE + item.index as u64);
    let gauss = super::gaussian_vec(&mut vrng, aligned.len());
    let random = InjectionSpec {
        after_block: RECEIVER_BLOCK,
        positions: positions.clone(),
        vector: gauss.clone(),
        scale: Some(target_l2 / norms::l2(&gauss)),
        mode,
    };
    let zerovec = InjectionSpec {
        after_block: RECEIVER_BLOCK,
        positions,
        vector: vec![0f32; aligned.len()],
        scale: None,
        mode,
    };

    // 5) Paired conditions (identical to S2b).
    let mut outcome = |spec: Option<&InjectionSpec>| -> anyhow::Result<(bool, f32, String)> {
        let mut s = Sampler::new(Sampling::Greedy, 0);
        let out = receiver
            .generate(&inj_tokens, spec, &mut s, MAX_NEW_TOKENS, false)
            .map_err(e)?;
        let correct =
            super::extract_answer(&out.text).is_some_and(|a| super::answers_equal(&a, &item.gold));
        let answer_toks = receiver.encode(&format!("#### {}", item.gold)).map_err(e)?;
        let nll_tokens: Vec<u32> = inj_tokens
            .iter()
            .chain(answer_toks.iter())
            .copied()
            .collect();
        let nll = teacher_forced_nll(
            &mut receiver.model,
            &nll_tokens,
            inj_tokens.len()..nll_tokens.len(),
            spec,
            device,
        )
        .map_err(e)?;
        Ok((correct, nll, out.text))
    };
    let (real_ok, real_nll, real_text) = outcome(Some(&real))?;
    let (base_ok, base_nll, _) = outcome(None)?;
    let (zero_ok, zero_nll, _) = outcome(Some(&zerovec))?;
    let (rand_ok, rand_nll, _) = outcome(Some(&random))?;

    let row = serde_json::json!({
        "item": item.index,
        "gold": item.gold,
        "sender_first_pass": {"correct": sender_pass.first_pass_correct,
                               "answer": sender_pass.first_pass_answer,
                               "generated_tokens": sender_pass.generated_tokens},
        "capture": {
            "hidden_size": meta.hidden_size,
            "pooled_l2_raw": meta.pooled_l2_raw,
            "aligned_l2_raw": aligned_l2,
            "injected_l2": target_l2,
            "natural_inject_block_norms": natural,
            "span": [meta.span.start, meta.span.end],
            "variant": meta.variant,
            "injection_mode": mode.tag(),
        },
        "conditions": {
            "aligned_real": {"correct": real_ok, "nll_gold": real_nll},
            "baseline_uninjected": {"correct": base_ok, "nll_gold": base_nll},
            "zerovec_injected": {"correct": zero_ok, "nll_gold": zero_nll},
            "random": {"correct": rand_ok, "nll_gold": rand_nll},
        },
        "aligned_answer_tail": real_text.chars().rev().take(60).collect::<String>().chars().rev().collect::<String>(),
    });
    Ok((
        row,
        Quad {
            real: (real_ok, real_nll),
            base: (base_ok, base_nll),
            zero: (zero_ok, zero_nll),
            rand: (rand_ok, rand_nll),
        },
    ))
}
