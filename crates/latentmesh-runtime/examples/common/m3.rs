//! Run-2 M3 probe internals (ADR-024): the frozen per-item four-condition
//! execution, shared constants, and the two registered eval variants.
//! Split out of `run2_m3_probe.rs` purely for file-size discipline — the
//! protocol here is ADR-023's frozen S1a/S2b protocol, byte-for-byte the
//! same mechanics as `s2b_bridge_probe.rs`'s `run_item`, with only the
//! transform application swapped for the trained MLP.

use super::mlp::MlpTransform;
use latentmesh_runtime::{
    capture::{forward_capture, forward_capture_multi_with_rows},
    inject::{teacher_forced_nll, InjectionSpec},
    norms,
    sampler::{Sampler, Sampling},
    QwenRuntime,
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

pub const SENDER: &str = "Qwen/Qwen2.5-3B-Instruct";
pub const RECEIVER: &str = "Qwen/Qwen2.5-1.5B-Instruct";
/// S2 winner cell L18→L14 (ADR-023 Deviation 6; ADR-024 M3's cell).
pub const SENDER_BLOCK: usize = 18;
pub const RECEIVER_BLOCK: usize = 14;
pub const ITEM_SEED: u64 = 0x51A1;
pub const RANDVEC_SEED_BASE: u64 = 0x51A2_0000;
pub const MAX_NEW_TOKENS: usize = 400;
pub const N_SLOTS: usize = 8;
pub const ALPHA: f64 = 0.05;
pub const SYSTEM: &str = "You are a careful math tutor.";
pub const GOLDEN_REL_TOL: f32 = 1e-5;

/// The two ADR-024-registered M3 eval variants.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Variant {
    /// (i) MLP per generated-span token state, then mean-pooling over the
    /// TRANSLATED stream, then the frozen 8-slot injection.
    PerToken,
    /// (ii) pool the sender's per-token states first (run-1 pipeline
    /// shape), then the SAME per-token-trained MLP on the pooled vector.
    Pooled,
}

impl Variant {
    pub fn tag(self) -> &'static str {
        match self {
            Variant::PerToken => "pertoken",
            Variant::Pooled => "pooled",
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

/// One frozen-protocol item: sender solve + capture, variant-specific MLP
/// application, then the four paired receiver conditions.
pub fn run_item(
    sender: &mut QwenRuntime,
    receiver: &mut QwenRuntime,
    transform: &MlpTransform,
    item: &super::Gsm8kItem,
    pad_id: u32,
    variant: Variant,
    device: &candle_core::Device,
) -> anyhow::Result<Option<QuadRow>> {
    let e = anyhow::Error::msg;
    let fmt = super::ANSWER_FORMAT;
    let mut greedy = Sampler::new(Sampling::Greedy, 0);

    // 1) Sender capture pass: solve, then teacher-forced re-prefill with the
    //    tap after the sender sweep block over the generated span.
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

    // 2) Variant-specific trained-MLP application (ADR-024 M3).
    let (aligned, cap_pooled_l2, cap_hidden, cap_span) = match variant {
        Variant::PerToken => {
            let (_, mut caps) = forward_capture_multi_with_rows(
                &mut sender.model,
                &full,
                &[SENDER_BLOCK],
                span.clone(),
                device,
            )
            .map_err(e)?;
            let cwr = caps.remove(0);
            anyhow::ensure!(cwr.capture.hidden_size == super::mlp::D_IN);
            let n_rows = cwr.capture.span.len();
            let aligned = transform.apply_rows_then_pool(&cwr.rows, n_rows);
            (
                aligned,
                norms::l2(&cwr.capture.pooled),
                cwr.capture.hidden_size,
                cwr.capture.span.clone(),
            )
        }
        Variant::Pooled => {
            let (_, cap) =
                forward_capture(&mut sender.model, &full, SENDER_BLOCK, span, device).map_err(e)?;
            anyhow::ensure!(cap.hidden_size == super::mlp::D_IN);
            let aligned = transform.apply(&cap.pooled);
            (
                aligned,
                norms::l2(&cap.pooled),
                cap.hidden_size,
                cap.span.clone(),
            )
        }
    };
    let first_pass_answer = super::extract_answer(&gen.text);
    let first_pass_correct = first_pass_answer
        .as_deref()
        .is_some_and(|a| super::answers_equal(a, &item.gold));

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

    let aligned_l2 = norms::l2(&aligned);
    let real = InjectionSpec {
        after_block: RECEIVER_BLOCK,
        positions: positions.clone(),
        vector: aligned.clone(),
        scale: Some(natural.median / aligned_l2),
    };
    let target_l2 = norms::l2(&real.effective_vector());
    let mut vrng = ChaCha8Rng::seed_from_u64(RANDVEC_SEED_BASE + item.index as u64);
    let gauss = super::gaussian_vec(&mut vrng, aligned.len());
    let random = InjectionSpec {
        after_block: RECEIVER_BLOCK,
        positions: positions.clone(),
        vector: gauss.clone(),
        scale: Some(target_l2 / norms::l2(&gauss)),
    };
    let zerovec = InjectionSpec {
        after_block: RECEIVER_BLOCK,
        positions,
        vector: vec![0f32; aligned.len()],
        scale: None,
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
        "sender_first_pass": {"correct": first_pass_correct, "answer": first_pass_answer,
                               "generated_tokens": gen.tokens.len()},
        "capture": {
            "hidden_size": cap_hidden,
            "pooled_l2_raw": cap_pooled_l2,
            "aligned_l2_raw": aligned_l2,
            "injected_l2": target_l2,
            "natural_inject_block_norms": natural,
            "span": [cap_span.start, cap_span.end],
            "variant": variant.tag(),
        },
        "conditions": {
            "aligned_real": {"correct": real_ok, "nll_gold": real_nll},
            "baseline_uninjected": {"correct": base_ok, "nll_gold": base_nll},
            "zerovec_injected": {"correct": zero_ok, "nll_gold": zero_nll},
            "random": {"correct": rand_ok, "nll_gold": rand_nll},
        },
        "aligned_answer_tail": real_text.chars().rev().take(60).collect::<String>().chars().rev().collect::<String>(),
    });
    Ok(Some((
        row,
        Quad {
            real: (real_ok, real_nll),
            base: (base_ok, base_nll),
            zero: (zero_ok, zero_nll),
            rand: (rand_ok, rand_nll),
        },
    )))
}
