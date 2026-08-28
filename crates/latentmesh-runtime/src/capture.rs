//! `forward_capture` — pooled residual capture after block k (design §3).
//!
//! Runs one teacher-forced prefill over `tokens` with a read-only tap after
//! block `after_block`, mean-pools the tapped residual over a token span, and
//! returns both the pooled vector and the pass's last-position logits so the
//! caller can verify logits parity against an unpatched forward (S0 gate A7:
//! bit-identical, verified on live GPU — the tap only clones a tensor, but
//! the gate measures rather than trusts that).

use crate::models::{LayerEdit, ModelForCausalLM};
use candle_core::{DType, Device, Result, Tensor, D};
use std::ops::Range;

/// One pooled residual capture plus its instrumentation.
#[derive(Debug, Clone)]
pub struct Capture {
    /// 1-based block count the tap ran after (design "L24/36" => 24).
    pub after_block: usize,
    /// Token span (within `tokens`) that was mean-pooled.
    pub span: Range<usize>,
    /// Mean-pooled residual over the span, f32, length = hidden_size.
    pub pooled: Vec<f32>,
    /// Hidden size of the captured residual (S0 gate: 2048 at sender L24).
    pub hidden_size: usize,
    /// Per-position L2 norms of the *natural* residual over the span —
    /// the reference distribution for the injected-norm band gate.
    pub per_position_l2: Vec<f32>,
}

/// Teacher-forced prefill with a capture tap.
///
/// Clears the KV cache first (fresh pass), returns
/// `(last_position_logits, capture)`. The logits tensor is `(1, 1, vocab)`
/// in the model dtype, untouched — compare with [`logits_bit_identical`].
pub fn forward_capture(
    model: &mut ModelForCausalLM,
    tokens: &[u32],
    after_block: usize,
    span: Range<usize>,
    device: &Device,
) -> Result<(Tensor, Capture)> {
    assert!(
        span.end <= tokens.len() && span.start < span.end,
        "capture span {span:?} out of range for {} tokens",
        tokens.len()
    );
    model.clear_kv_cache();
    let input = Tensor::new(tokens, device)?.unsqueeze(0)?;
    let mut tapped: Option<Tensor> = None;
    let mut edit = LayerEdit::Capture {
        after_block,
        out: &mut tapped,
    };
    let logits = model.forward_with_edit(&input, 0, Some(&mut edit))?;
    let tapped = tapped.expect("capture tap did not fire: after_block beyond layer count?");
    // (1, seq, hidden) -> span rows in f32.
    let rows = tapped
        .narrow(1, span.start, span.end - span.start)?
        .to_dtype(DType::F32)?;
    let hidden_size = rows.dim(2)?;
    let pooled = rows.mean(1)?.squeeze(0)?.to_vec1::<f32>()?;
    let per_position_l2 = rows
        .sqr()?
        .sum(D::Minus1)?
        .sqrt()?
        .squeeze(0)?
        .to_vec1::<f32>()?;
    Ok((
        logits,
        Capture {
            after_block,
            span,
            pooled,
            hidden_size,
            per_position_l2,
        },
    ))
}

/// Unpatched reference forward over the same tokens (fresh KV cache),
/// returning last-position logits via the vendored-original path.
pub fn forward_unpatched(
    model: &mut ModelForCausalLM,
    tokens: &[u32],
    device: &Device,
) -> Result<Tensor> {
    model.clear_kv_cache();
    let input = Tensor::new(tokens, device)?.unsqueeze(0)?;
    model.forward(&input, 0)
}

/// Bit-exact logits comparison. Both tensors are converted to f32 (lossless
/// from BF16) and compared on their IEEE-754 bit patterns — any difference,
/// including NaN-payload or signed-zero drift, fails.
pub fn logits_bit_identical(a: &Tensor, b: &Tensor) -> Result<bool> {
    let a = a.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let b = b.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    Ok(a.len() == b.len()
        && a.iter()
            .zip(b.iter())
            .all(|(x, y)| x.to_bits() == y.to_bits()))
}
