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

/// **WHERE** the payload is delivered — ADR-024 M4i's single changed factor
/// (`docs/research/043` §4).
///
/// The site is an ADR-028 *evolvable* surface (it is neither the slot count
/// nor the statistics); the slot count stays 8 under both variants and is
/// asserted at run time exactly as before.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Site {
    /// Every rung up to and including M4h Stage 1: the S1a prompt wording
    /// inserts `N_SLOTS` copies of `<|fim_pad|>` inside a bracketed
    /// "stored in these slots: [..]" sentence, and the payload is delivered
    /// at those placeholder positions.
    FimPadSlots,
    /// ADR-024 M4i: the last `N_SLOTS` tokens of the item's **own question** —
    /// ordinary, already-present content tokens. The slot sentence and its
    /// bracket are removed entirely (`docs/research/043` §4: keeping the
    /// bracket would reintroduce a textual placeholder cue even once the
    /// token identity changed), so the receiver prompt is exactly
    /// `{question}\n\n{ANSWER_FORMAT}` under the same chat template.
    ///
    /// Only meaningful under [`InjectionMode::Fuse`]: overwriting real
    /// question tokens would destroy content the receiver needs and confound
    /// "the position is inert" with "we deleted the question".
    QuestionTail,
}

impl Site {
    pub fn tag(self) -> &'static str {
        match self {
            Site::FimPadSlots => "fim_pad_slots",
            Site::QuestionTail => "question_tail_ordinary_tokens",
        }
    }

    /// The site, written out, for receipts.
    pub fn description(self) -> &'static str {
        match self {
            Site::FimPadSlots => {
                "8 <|fim_pad|> placeholder tokens inserted by the S1a slot sentence \
                 (ADR-023 original; every rung through M4h Stage 1)"
            }
            Site::QuestionTail => {
                "the last 8 tokens of the item's own question — ordinary content tokens \
                 already present in the prompt; no slot sentence, no bracket, no placeholder \
                 token anywhere (docs/research/043 §4)"
            }
        }
    }
}

/// The receiver prompt for one item plus the exact absolute token positions
/// the payload is delivered at, under the selected [`Site`].
pub struct SitePrompt {
    pub tokens: Vec<u32>,
    pub positions: Vec<usize>,
    /// The token ids at `positions` (recorded per item, so the site is
    /// auditable from the receipt alone).
    pub position_token_ids: Vec<u32>,
    /// Those tokens decoded back to text.
    pub positions_decoded: String,
}

/// Whitespace-insensitive suffix check used by the `QuestionTail` gate.
fn squeeze(s: &str) -> String {
    s.chars().filter(|c| !c.is_whitespace()).collect()
}

/// Build the receiver's injection prompt and resolve the `N_SLOTS` delivery
/// positions for `site`.
///
/// `FimPadSlots` reproduces ADR-023's frozen wording byte-for-byte. For
/// `QuestionTail` the positions are read off the **canonical tokenisation's
/// own offset map**: the eight positions are the last eight tokens whose byte
/// span lies wholly inside `item.question`.
///
/// Reading offsets rather than re-encoding a prefix is what makes this exact.
/// Qwen2.5's pre-tokeniser groups trailing punctuation with the newlines that
/// follow it, so for a GSM8K question ending in `?` the final token spans
/// `"?\n\n"` and straddles the question/answer-format boundary. That token is
/// not part of the question and is excluded, which is why the decode gate
/// below checks containment rather than a strict suffix. The gates — at least
/// eight tokens wholly inside, contiguous, ending at the last such token, and
/// decoding back to text that is part of the question — are all mechanical
/// properties of the prompt, resolved before any generation for the item.
pub fn build_site_prompt(
    receiver: &QwenRuntime,
    item: &super::Gsm8kItem,
    pad_id: u32,
    site: Site,
) -> anyhow::Result<SitePrompt> {
    let e = anyhow::Error::msg;
    let fmt = super::ANSWER_FORMAT;
    let (tokens, positions) = match site {
        Site::FimPadSlots => {
            let slots = "<|fim_pad|>".repeat(N_SLOTS);
            let prompt = QwenRuntime::chat_prompt(
                SYSTEM,
                &format!(
                    "{}\n\nA compressed latent hint from a previous solver is stored in these slots: [{slots}]\n\n{fmt}",
                    item.question
                ),
            );
            let tokens = receiver.encode(&prompt).map_err(e)?;
            let positions = QwenRuntime::placeholder_positions(&tokens, pad_id);
            (tokens, positions)
        }
        Site::QuestionTail => {
            let user = format!("{}\n\n{fmt}", item.question);
            let full = QwenRuntime::chat_prompt(SYSTEM, &user);
            let q_start = full
                .find(&user)
                .ok_or_else(|| anyhow::anyhow!("chat template did not contain the user turn"))?;
            let q_end = q_start + item.question.len();
            // The canonical tokenisation of the WHOLE prompt, with the
            // tokeniser's own offset map. Selecting by offsets rather than by
            // re-encoding a prefix is what makes this exact: Qwen2.5's
            // pre-tokeniser groups trailing punctuation with the newlines that
            // follow it (` ?[^\s\p{L}\p{N}]+[\r\n]*`), so for a GSM8K question
            // ending in "?" the final token spans "?\n\n" and STRADDLES the
            // question/answer-format boundary. Such a token is not part of the
            // question and is excluded here.
            let enc = receiver
                .tokenizer
                .encode(full.as_str(), false)
                .map_err(|err| anyhow::anyhow!("encode: {err}"))?;
            let tokens = enc.get_ids().to_vec();
            let inside: Vec<usize> = enc
                .get_offsets()
                .iter()
                .enumerate()
                .filter(|(_, &(s, t))| t > s && s >= q_start && t <= q_end)
                .map(|(i, _)| i)
                .collect();
            anyhow::ensure!(
                inside.len() >= N_SLOTS,
                "item {}: only {} tokens lie wholly inside the question, fewer than the {N_SLOTS} \
                 required",
                item.index,
                inside.len()
            );
            let positions: Vec<usize> = inside[inside.len() - N_SLOTS..].to_vec();
            anyhow::ensure!(
                positions.windows(2).all(|w| w[1] == w[0] + 1),
                "item {}: the resolved question-tail positions are not contiguous ({positions:?})",
                item.index
            );
            anyhow::ensure!(
                positions[N_SLOTS - 1] == inside[inside.len() - 1],
                "item {}: the tail window does not end at the last token wholly inside the question",
                item.index
            );
            (tokens, positions)
        }
    };
    anyhow::ensure!(
        positions.len() == N_SLOTS,
        "slot count mismatch: {} positions resolved, expected {N_SLOTS}",
        positions.len()
    );
    let position_token_ids: Vec<u32> = positions.iter().map(|&p| tokens[p]).collect();
    let positions_decoded = receiver.decode(&position_token_ids).map_err(e)?;
    if site == Site::QuestionTail {
        // The positions must decode back to text that really is part of the
        // item's own question. Not a strict suffix: the boundary token (e.g.
        // "?\n\n") is excluded by construction above, so the window ends at
        // the last token wholly inside the question, which may leave a trailing
        // character or two uncovered.
        anyhow::ensure!(
            squeeze(&item.question).contains(&squeeze(&positions_decoded)),
            "item {}: the {N_SLOTS} injected positions decode to {positions_decoded:?}, which is \
             not part of the item's own question — the site gate refuses",
            item.index
        );
    }
    Ok(SitePrompt {
        tokens,
        positions,
        position_token_ids,
        positions_decoded,
    })
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
    run_item_at(
        sender,
        receiver,
        transform,
        item,
        pad_id,
        variant,
        mode,
        Site::FimPadSlots,
        device,
    )
}

/// [`run_item`] with the delivery [`Site`] named explicitly (ADR-024 M4i).
/// `run_item` is the `Site::FimPadSlots` specialisation, so every rung
/// through M4h Stage 1 keeps its exact historical code path.
#[allow(clippy::too_many_arguments)]
pub fn run_item_at(
    sender: &mut QwenRuntime,
    receiver: &mut QwenRuntime,
    transform: &MlpTransform,
    item: &super::Gsm8kItem,
    pad_id: u32,
    variant: Variant,
    mode: InjectionMode,
    site: Site,
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
    four_conditions_at(
        receiver,
        item,
        pad_id,
        &aligned,
        &sender_pass,
        &meta,
        mode,
        site,
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
    four_conditions_at(
        receiver,
        item,
        pad_id,
        aligned,
        sender_pass,
        meta,
        mode,
        Site::FimPadSlots,
        device,
    )
}

/// [`four_conditions`] with the delivery [`Site`] named explicitly.
///
/// `site` selects **where** the payload lands and is ADR-024 M4i's single
/// changed factor. Nothing else about the block moves: the three
/// `InjectionSpec`s, the rescale-to-natural-median rule, the four paired
/// conditions and the per-item row schema are constructed by exactly the same
/// statements under either site, so no control is redefined by changing it.
/// What the change *means* for `random` IS different and is declared before
/// the draw rather than papered over: under `QuestionTail` + `Fuse` the
/// norm-matched Gaussian perturbs the receiver's own genuine question
/// content, not an inert placeholder row — a **stronger** comparator for the
/// primary, per `docs/research/043` §4.
#[allow(clippy::too_many_arguments)]
pub fn four_conditions_at(
    receiver: &mut QwenRuntime,
    item: &super::Gsm8kItem,
    pad_id: u32,
    aligned: &[f32],
    sender_pass: &SenderPass,
    meta: &CaptureMeta,
    mode: InjectionMode,
    site: Site,
    device: &candle_core::Device,
) -> anyhow::Result<QuadRow> {
    let e = anyhow::Error::msg;

    // 3) Receiver injection prompt + the delivery positions for this site.
    let sp = build_site_prompt(receiver, item, pad_id, site)?;
    let inj_tokens = sp.tokens;
    let positions = sp.positions;
    let positions_recorded = positions.clone();

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
    let (base_ok, base_nll, base_text) = outcome(None)?;
    let (zero_ok, zero_nll, zero_text) = outcome(Some(&zerovec))?;
    let (rand_ok, rand_nll, rand_text) = outcome(Some(&random))?;

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
        "injection_site": {
            "site": site.tag(),
            "prompt_tokens": inj_tokens.len(),
            "positions": positions_recorded,
            "position_token_ids": sp.position_token_ids,
            "positions_decoded": sp.positions_decoded,
        },
        "conditions": {
            "aligned_real": {"correct": real_ok, "nll_gold": real_nll},
            "baseline_uninjected": {"correct": base_ok, "nll_gold": base_nll},
            "zerovec_injected": {"correct": zero_ok, "nll_gold": zero_nll},
            "random": {"correct": rand_ok, "nll_gold": rand_nll},
        },
        "aligned_answer_tail": real_text.chars().rev().take(60).collect::<String>().chars().rev().collect::<String>(),
        // RECORDING-ONLY instrumentation (added for ADR-024's PC1 rung; it
        // changes no mechanic and no statistic). ADR-024's PC1 registration
        // and `docs/research/045` §3 require a degenerate-output check on
        // every positive-control draw — the ITI "no comment" failure mode,
        // where an intervention appears to restore the answer while actually
        // collapsing the model into a trivial response. Generated length per
        // condition is the cheapest instrument for that, and it is worth
        // having on every rung.
        "generated_chars": {
            "aligned_real": real_text.chars().count(),
            "baseline_uninjected": base_text.chars().count(),
            "zerovec_injected": zero_text.chars().count(),
            "random": rand_text.chars().count(),
        },
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
