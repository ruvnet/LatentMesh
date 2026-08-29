//! `forward_inject` — placeholder-position residual edit at block k
//! (design §3). The receiver prompt carries placeholder tokens; during
//! prefill the residual rows at those positions, after `after_block` blocks,
//! are edited with the (optionally rescaled) payload vector. Decode
//! steps inherit the edit through the KV cache of blocks > k, positions
//! unchanged — RoPE/KV coherent.
//!
//! Two operators exist, selected by [`InjectionMode`] and recorded explicitly
//! at every construction site and in every receipt:
//!
//! * [`InjectionMode::Overwrite`] — the ADR-023 original and the operator
//!   every run-1 and run-2 M3/M4/M4c/M4d receipt was produced with:
//!   `h[slot] = c·v`, a hard `slice_assign` that discards whatever the
//!   placeholder position's own forward pass produced. **This path is frozen
//!   and behaviourally untouched**, so every prior receipt stays reproducible.
//! * [`InjectionMode::Fuse`] — ADR-024's M4g rung: `h[slot] += c·v`, a
//!   residual ADD that preserves the receiver's own state at those positions.
//!   This mirrors Cache-to-Cache's own fuser equation
//!   `C_F = C_n(X) + F_n(...)` (arXiv:2510.03215 Eq. 3, verified in
//!   `docs/research/038` §4). The injection operator is an ADR-028
//!   **evolvable** surface; the probe protocol around it is protected.

use crate::models::{LayerEdit, ModelForCausalLM};
use candle_core::{DType, Device, Result, Tensor};
use std::ops::Range;

/// Which residual-stream operator delivers the payload at the slot rows.
///
/// Recorded explicitly in every receipt: it is the single changed factor
/// between ADR-024's M4d rung (`Overwrite`) and its M4g rung (`Fuse`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize)]
pub enum InjectionMode {
    /// `h[slot] = c·v` — hard overwrite (ADR-023 original; the default, so an
    /// unannotated call site keeps the historical semantics exactly).
    #[default]
    Overwrite,
    /// `h[slot] += c·v` — residual add / fuse (ADR-024 M4g; C2C Eq. 3).
    Fuse,
}

impl InjectionMode {
    /// Short receipt tag.
    pub fn tag(self) -> &'static str {
        match self {
            InjectionMode::Overwrite => "overwrite",
            InjectionMode::Fuse => "fuse",
        }
    }

    /// The operator, written out, for receipts and error messages.
    pub fn equation(self) -> &'static str {
        match self {
            InjectionMode::Overwrite => "h[slot] = c*v (slice_assign overwrite; ADR-023 original)",
            InjectionMode::Fuse => {
                "h[slot] += c*v (residual add; C2C Eq.3 C_F = C_n(X) + F_n(...))"
            }
        }
    }
}

/// A single-vector injection: the pooled payload broadcast to every slot.
#[derive(Debug, Clone)]
pub struct InjectionSpec {
    /// 1-based block count after which the edit happens ("L19/28" => 19).
    pub after_block: usize,
    /// Absolute token positions of the placeholder slots in the prompt.
    pub positions: Vec<usize>,
    /// Payload vector, f32, length = receiver hidden size.
    pub vector: Vec<f32>,
    /// Optional multiplier applied to `vector` before the edit (the
    /// pre-declared norm-rescaling switch; `None` = raw).
    pub scale: Option<f32>,
    /// Residual operator: overwrite (historical) or fuse (ADR-024 M4g).
    pub mode: InjectionMode,
}

impl InjectionSpec {
    /// The vector actually delivered (post-scale). Under `Overwrite` this is
    /// what the slot row becomes; under `Fuse` it is what is ADDED to it.
    pub fn effective_vector(&self) -> Vec<f32> {
        let s = self.scale.unwrap_or(1.0);
        self.vector.iter().map(|x| x * s).collect()
    }

    fn vectors_tensor(&self, device: &Device) -> Result<Tensor> {
        let v = self.effective_vector();
        let h = v.len();
        let n = self.positions.len();
        let rows: Vec<f32> = std::iter::repeat(v).take(n.max(1)).flatten().collect();
        Tensor::from_vec(rows, (n.max(1), h), device)
    }

    /// The `LayerEdit` this spec's [`InjectionMode`] selects.
    fn layer_edit<'a>(&'a self, vectors: &'a Tensor) -> LayerEdit<'a> {
        match self.mode {
            InjectionMode::Overwrite => LayerEdit::Inject {
                after_block: self.after_block,
                positions: &self.positions,
                vectors,
            },
            InjectionMode::Fuse => LayerEdit::Fuse {
                after_block: self.after_block,
                positions: &self.positions,
                vectors,
            },
        }
    }
}

/// Prefill `tokens` with the injection applied (or `None` for the baseline),
/// returning last-position logits `(1, 1, vocab)`. Clears the KV cache
/// first; the cache is left populated so a caller can continue decoding.
pub fn prefill_with_injection(
    model: &mut ModelForCausalLM,
    tokens: &[u32],
    spec: Option<&InjectionSpec>,
    device: &Device,
) -> Result<Tensor> {
    model.clear_kv_cache();
    let input = Tensor::new(tokens, device)?.unsqueeze(0)?;
    match spec {
        None => model.forward(&input, 0),
        Some(spec) => {
            let vectors = spec.vectors_tensor(device)?;
            let mut edit = spec.layer_edit(&vectors);
            model.forward_with_edit(&input, 0, Some(&mut edit))
        }
    }
}

/// Teacher-forced mean NLL (nats/token) of `tokens[target_span]` given the
/// preceding context, with an optional injection applied during the single
/// full-sequence pass. Continuous per-item diagnostic for the S1a sign-flip
/// analysis (secondary to paired accuracy, pre-committed in the receipt).
pub fn teacher_forced_nll(
    model: &mut ModelForCausalLM,
    tokens: &[u32],
    target_span: Range<usize>,
    spec: Option<&InjectionSpec>,
    device: &Device,
) -> Result<f32> {
    assert!(target_span.start >= 1 && target_span.end <= tokens.len());
    model.clear_kv_cache();
    let input = Tensor::new(tokens, device)?.unsqueeze(0)?;
    let logits = match spec {
        None => model.forward_full_logits(&input, 0, None)?,
        Some(spec) => {
            let vectors = spec.vectors_tensor(device)?;
            let mut edit = spec.layer_edit(&vectors);
            model.forward_full_logits(&input, 0, Some(&mut edit))?
        }
    };
    // Logits at position p predict token p+1.
    let start = target_span.start - 1;
    let len = target_span.end - target_span.start;
    let rows = logits.narrow(1, start, len)?.to_dtype(DType::F32)?;
    let logp = candle_nn::ops::log_softmax(&rows, candle_core::D::Minus1)?
        .squeeze(0)?
        .to_vec2::<f32>()?;
    let mut nll = 0f32;
    for (i, target_pos) in target_span.clone().enumerate() {
        nll -= logp[i][tokens[target_pos] as usize];
    }
    Ok(nll / len as f32)
}
