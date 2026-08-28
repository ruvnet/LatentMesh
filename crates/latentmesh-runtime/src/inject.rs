//! `forward_inject` — placeholder-position residual overwrite at block k
//! (design §3). The receiver prompt carries placeholder tokens; during
//! prefill the residual rows at those positions, after `after_block` blocks,
//! are overwritten with the (optionally rescaled) payload vector. Decode
//! steps inherit the edit through the KV cache of blocks > k, positions
//! unchanged — RoPE/KV coherent.

use crate::models::{LayerEdit, ModelForCausalLM};
use candle_core::{DType, Device, Result, Tensor};
use std::ops::Range;

/// A single-vector injection: the pooled payload broadcast to every slot.
#[derive(Debug, Clone)]
pub struct InjectionSpec {
    /// 1-based block count after which the overwrite happens ("L19/28" => 19).
    pub after_block: usize,
    /// Absolute token positions of the placeholder slots in the prompt.
    pub positions: Vec<usize>,
    /// Payload vector, f32, length = receiver hidden size.
    pub vector: Vec<f32>,
    /// Optional multiplier applied to `vector` before the overwrite (the
    /// pre-declared norm-rescaling switch; `None` = raw).
    pub scale: Option<f32>,
}

impl InjectionSpec {
    /// The vector actually written (post-scale).
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
            let mut edit = LayerEdit::Inject {
                after_block: spec.after_block,
                positions: &spec.positions,
                vectors: &vectors,
            };
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
            let mut edit = LayerEdit::Inject {
                after_block: spec.after_block,
                positions: &spec.positions,
                vectors: &vectors,
            };
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
