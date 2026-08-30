//! M6 (ADR-047): the **five**-condition receiver block and its battery.
//!
//! ## Why five and not six
//!
//! ADR-047 registered a 2×2 — {`aligned`, `aligned_displaced`} ×
//! {`mismatched`, `random`} — plus `baseline` and `zerovec`, and §9 asked for
//! `six_conditions_at`. The phase-1 manipulation check killed the
//! `aligned_displaced` cell and the coordinator withdrew it (**coordinator
//! error #24**, recorded in ADR-047's outcomes): in 1536 dimensions a generic
//! displacement direction is almost surely BOTH off-manifold AND content-free,
//! so rotating toward one moves both factors in lockstep, and
//! `aligned_displaced(c)` measured as a point on the `aligned`→`random`
//! segment under every instrument in the lens kit. Five conditions, one
//! primary — the CONTENT axis — is what survives.
//!
//! ## Why this file exists rather than an edit to `m3.rs`
//!
//! `four_conditions_at` is called by M4c/M4d/M4g/M4i/M5/M5X. `m3.rs:365-367`
//! states the house doctrine that each rung keeps its exact historical code
//! path, so the four-condition block is not touched; this is the five-arm
//! sibling. Everything it shares with that block — the site prompt, the
//! natural-median rescale target, the `InjectionSpec` construction, the greedy
//! sampler, the teacher-forced NLL against `#### {gold}` — is called through
//! the same functions, so the only registered difference is the added arm.
//!
//! ## The `mismatched` payload is supplied, not derived
//!
//! [`five_conditions_at`] takes the mismatched payload as an argument and does
//! not know where it came from. That is deliberate: ADR-047 §4.4 requires it to
//! be the **previous stream item's** aligned payload with **no lookahead**, and
//! that is a property of the caller's iteration order, not of this block. The
//! probe primes it from a designated item outside the drawn stream and carries
//! it forward, following `run3_gated_text_probe.rs:313-317, 361-363, 516-517`.
//! Both the source item index and the no-lookahead assertion are the probe's
//! responsibility and land in its receipt.

use super::m3::{
    build_site_prompt, sender_solve_capture_rows, CaptureMeta, SenderPass, Site, Variant,
    MAX_NEW_TOKENS, RANDVEC_SEED_BASE, RECEIVER_BLOCK,
};
use latentmesh_runtime::{
    capture::forward_capture,
    inject::{teacher_forced_nll, InjectionMode, InjectionSpec},
    norms,
    sampler::{Sampler, Sampling},
    QwenRuntime,
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// The five registered conditions, in receipt order.
///
/// M5's four keep their relative order; `mismatched` is inserted directly
/// after the treatment arm because it is the registered CONTENT contrast, not
/// a background control.
pub const CONDITIONS: [&str; 5] = [
    "aligned_real",
    "mismatched",
    "baseline_uninjected",
    "zerovec_injected",
    "random",
];

/// The registered conditions as a type. See [`super::m5`]'s `Cond` for why
/// this is an enum and not a `usize`: the accessors it replaced ended in a
/// catch-all that would have read a newly added condition as `random` on both
/// endpoints, writing a wrong number into a stored receipt field with no
/// compile error and no runtime signal. ADR-047 §9 named that hazard and
/// required it fixed before this arm was added; it was.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Cond {
    AlignedReal,
    Mismatched,
    BaselineUninjected,
    ZerovecInjected,
    Random,
}

impl Cond {
    /// In receipt order — the same order and labels as [`CONDITIONS`].
    pub const ALL: [Cond; 5] = [
        Cond::AlignedReal,
        Cond::Mismatched,
        Cond::BaselineUninjected,
        Cond::ZerovecInjected,
        Cond::Random,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Cond::AlignedReal => "aligned_real",
            Cond::Mismatched => "mismatched",
            Cond::BaselineUninjected => "baseline_uninjected",
            Cond::ZerovecInjected => "zerovec_injected",
            Cond::Random => "random",
        }
    }

    /// Everything except the treatment arm.
    pub fn is_control(self) -> bool {
        self != Cond::AlignedReal
    }
}

/// One item's outcome under all five conditions: `(correct, nll_gold)`.
#[derive(Clone, Copy)]
pub struct Quint {
    pub real: (bool, f32),
    pub mism: (bool, f32),
    pub base: (bool, f32),
    pub zero: (bool, f32),
    pub rand: (bool, f32),
}

impl Quint {
    /// Exhaustive by construction — no catch-all arm, so adding a condition
    /// to [`Cond`] fails to compile here rather than silently aliasing onto
    /// `random`.
    pub fn endpoints(&self, which: Cond) -> (bool, f32) {
        match which {
            Cond::AlignedReal => self.real,
            Cond::Mismatched => self.mism,
            Cond::BaselineUninjected => self.base,
            Cond::ZerovecInjected => self.zero,
            Cond::Random => self.rand,
        }
    }

    pub fn correct(&self, which: Cond) -> bool {
        self.endpoints(which).0
    }

    pub fn nll(&self, which: Cond) -> f32 {
        self.endpoints(which).1
    }
}

pub type QuintRow = (serde_json::Value, Quint);

/// Steps 3–5 of the frozen protocol with M6's fifth arm, at an explicit
/// delivery [`Site`].
///
/// `mismatched` is another episode's genuine payload (see the module header).
/// It is rescaled by the SAME rule as `random` — to the effective aligned
/// vector's L2, which the natural-median rule has already fixed at
/// `natural.median` — so `aligned`, `mismatched` and `random` all arrive at
/// identical norm by construction (ADR-047 §4.4). Norm-matching is therefore
/// not a separate mechanism that could drift; it falls out of the rescale rule
/// M5 already ran, and the receipt records all three realised norms so a
/// reader can check rather than trust it.
#[allow(clippy::too_many_arguments)]
pub fn five_conditions_at(
    receiver: &mut QwenRuntime,
    item: &super::Gsm8kItem,
    pad_id: u32,
    aligned: &[f32],
    mismatched: &[f32],
    mismatched_from: usize,
    sender_pass: &SenderPass,
    meta: &CaptureMeta,
    mode: InjectionMode,
    site: Site,
    device: &candle_core::Device,
) -> anyhow::Result<QuintRow> {
    let e = anyhow::Error::msg;
    anyhow::ensure!(
        mismatched.len() == aligned.len(),
        "mismatched payload is {} wide but aligned is {} — the mismatched control must be another \
         item's payload from the SAME transform, not a differently shaped vector",
        mismatched.len(),
        aligned.len()
    );
    anyhow::ensure!(
        mismatched_from != item.index,
        "the mismatched payload for item {} is item {}'s own payload — that is the aligned \
         condition wearing a control's name",
        item.index,
        mismatched_from
    );

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

    // The fifth arm. Same operator, same positions, same target norm; the
    // ONLY thing that differs from `aligned` is which episode's content the
    // vector encodes. That is the whole point of the rung.
    let mismatched_l2 = norms::l2(mismatched);
    anyhow::ensure!(
        mismatched_l2 > 0.0,
        "the mismatched payload has zero norm — it cannot be rescaled, and a zero vector is the \
         zerovec condition, not a content control"
    );
    let mism = InjectionSpec {
        after_block: RECEIVER_BLOCK,
        positions: positions.clone(),
        vector: mismatched.to_vec(),
        scale: Some(target_l2 / mismatched_l2),
        mode,
    };

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

    // 5) Paired conditions, strictly serial, in receipt order.
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
    let (mism_ok, mism_nll, mism_text) = outcome(Some(&mism))?;
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
        "mismatched_control": {
            "payload_from_item": mismatched_from,
            "raw_l2": mismatched_l2,
            // All three injected arms land here by construction; stored so a
            // reader can verify the norm match rather than take §4.4 on trust.
            "injected_l2": norms::l2(&mism.effective_vector()),
            "random_injected_l2": norms::l2(&random.effective_vector()),
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
            "mismatched": {"correct": mism_ok, "nll_gold": mism_nll},
            "baseline_uninjected": {"correct": base_ok, "nll_gold": base_nll},
            "zerovec_injected": {"correct": zero_ok, "nll_gold": zero_nll},
            "random": {"correct": rand_ok, "nll_gold": rand_nll},
        },
        "aligned_answer_tail": tail(&real_text),
        "mismatched_answer_tail": tail(&mism_text),
        // RECORDING-ONLY, carried from `four_conditions_at`: the ITI "no
        // comment" failure mode, where an intervention appears to restore the
        // answer while collapsing the model into a trivial response.
        "generated_chars": {
            "aligned_real": real_text.chars().count(),
            "mismatched": mism_text.chars().count(),
            "baseline_uninjected": base_text.chars().count(),
            "zerovec_injected": zero_text.chars().count(),
            "random": rand_text.chars().count(),
        },
    });
    Ok((
        row,
        Quint {
            real: (real_ok, real_nll),
            mism: (mism_ok, mism_nll),
            base: (base_ok, base_nll),
            zero: (zero_ok, zero_nll),
            rand: (rand_ok, rand_nll),
        },
    ))
}

fn tail(s: &str) -> String {
    let rev: String = s.chars().rev().take(60).collect();
    rev.chars().rev().collect()
}

/// Steps 1–2 — the sender's capture pass and the trained MLP applied to it —
/// returning the payload instead of consuming it.
///
/// `run_item_at` bundles these with the four-condition block and hands back
/// only the row and the `Quad`. M6 cannot use it, because the payload itself
/// is the thing that must survive the call: item *i*'s aligned payload IS item
/// *i+1*'s `mismatched` control. So the capture half is exposed here, with the
/// body kept literally identical to `m3.rs`'s `Variant::PerTokenLast` branch —
/// same `sender_solve_capture_rows`, same width assertion, same
/// `apply_last_row`, same [`CaptureMeta`] fields — so the payload M6 injects
/// is the payload M4i and M5 injected, not a re-derivation that happens to
/// look the same.
///
/// `None` is a degenerate sender pass (no answer span to capture), which the
/// caller must record as a consumed budget item with no pair, exactly as the
/// prior rungs do.
pub fn capture_payload(
    sender: &mut QwenRuntime,
    transform: &super::mlp::MlpTransform,
    item: &super::Gsm8kItem,
    variant: Variant,
    device: &candle_core::Device,
) -> anyhow::Result<Option<(Vec<f32>, SenderPass, CaptureMeta)>> {
    anyhow::ensure!(
        variant == Variant::PerTokenLast,
        "M6 holds the payload derivation fixed at PerTokenLast (M4i's, and M5's); {} is a second \
         changed factor",
        variant.tag()
    );
    let Some(sr) = sender_solve_capture_rows(sender, item, device)? else {
        return Ok(None);
    };
    anyhow::ensure!(sr.hidden_size == super::mlp::D_IN);
    let aligned = transform.apply_last_row(&sr.rows, sr.n_rows);
    let meta = CaptureMeta {
        hidden_size: sr.hidden_size,
        pooled_l2_raw: norms::l2(&sr.pooled),
        span: sr.span.clone(),
        variant: variant.tag(),
    };
    Ok(Some((aligned, sr.pass, meta)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quint(
        r: (bool, f32),
        m: (bool, f32),
        b: (bool, f32),
        z: (bool, f32),
        n: (bool, f32),
    ) -> Quint {
        Quint {
            real: r,
            mism: m,
            base: b,
            zero: z,
            rand: n,
        }
    }

    #[test]
    fn labels_match_the_registered_order() {
        assert_eq!(Cond::ALL.len(), CONDITIONS.len());
        for (c, name) in Cond::ALL.iter().zip(CONDITIONS.iter()) {
            assert_eq!(c.label(), *name);
        }
        assert!(!Cond::ALL[0].is_control());
        assert!(Cond::ALL[1..].iter().all(|c| c.is_control()));
    }

    /// The hazard ADR-047 §9 named, stated as a test rather than a comment:
    /// every condition must read its OWN field. Under the catch-all this
    /// replaced, `mismatched` would have returned `random`'s numbers.
    #[test]
    fn every_condition_reads_its_own_field() {
        let q = quint(
            (true, 1.0),
            (false, 2.0),
            (true, 3.0),
            (false, 4.0),
            (true, 5.0),
        );
        let seen: Vec<(bool, f32)> = Cond::ALL.iter().map(|&c| q.endpoints(c)).collect();
        assert_eq!(
            seen,
            vec![
                (true, 1.0),
                (false, 2.0),
                (true, 3.0),
                (false, 4.0),
                (true, 5.0)
            ]
        );
        // No two conditions alias onto the same field.
        for (i, a) in Cond::ALL.iter().enumerate() {
            for b in &Cond::ALL[i + 1..] {
                assert_ne!(q.endpoints(*a), q.endpoints(*b), "{a:?} aliases {b:?}");
            }
        }
    }
}
