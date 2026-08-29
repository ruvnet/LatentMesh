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

/// Teacher-forced prefill with taps after EACH of `after_blocks` in a single
/// pass (S2 calibration: 3 depths per model per item). Returns the pass's
/// last-position logits plus one [`Capture`] per tapped block, in
/// `after_blocks` order. Same parity property as [`forward_capture`] — each
/// tap only clones the residual — and the S2 dump receipt measures it.
pub fn forward_capture_multi(
    model: &mut ModelForCausalLM,
    tokens: &[u32],
    after_blocks: &[usize],
    span: Range<usize>,
    device: &Device,
) -> Result<(Tensor, Vec<Capture>)> {
    assert!(
        span.end <= tokens.len() && span.start < span.end,
        "capture span {span:?} out of range for {} tokens",
        tokens.len()
    );
    model.clear_kv_cache();
    let input = Tensor::new(tokens, device)?.unsqueeze(0)?;
    let mut tapped: Vec<(usize, Tensor)> = Vec::new();
    let mut edit = LayerEdit::CaptureMany {
        after_blocks,
        out: &mut tapped,
    };
    let logits = model.forward_with_edit(&input, 0, Some(&mut edit))?;
    let mut captures = Vec::with_capacity(after_blocks.len());
    for &block in after_blocks {
        let t = tapped
            .iter()
            .find(|(b, _)| *b == block)
            .map(|(_, t)| t)
            .unwrap_or_else(|| panic!("tap after block {block} did not fire (layer count?)"));
        captures.push(pool_capture(t, block, span.clone())?);
    }
    Ok((logits, captures))
}

/// One pooled capture plus the pre-pooling per-token span rows (run 2:
/// per-token paired dumps train sequence adapters on exactly the rows the
/// pooled S2c asset averaged away).
#[derive(Debug, Clone)]
pub struct CaptureWithRows {
    /// The pooled capture — identical to what [`forward_capture_multi`]
    /// returns for the same pass (same op sequence on the same tensor).
    pub capture: Capture,
    /// Row-major `[span_len x hidden_size]` f32 — the exact rows
    /// `capture.pooled` is the mean of.
    pub rows: Vec<f32>,
}

/// [`forward_capture_multi`] variant that ALSO returns the per-token span
/// rows each tap pooled over. The pooled values are computed by the same op
/// sequence as [`forward_capture_multi`] (narrow → f32 → mean), so pooled
/// output is unchanged; the only addition is a host copy of the span rows.
pub fn forward_capture_multi_with_rows(
    model: &mut ModelForCausalLM,
    tokens: &[u32],
    after_blocks: &[usize],
    span: Range<usize>,
    device: &Device,
) -> Result<(Tensor, Vec<CaptureWithRows>)> {
    assert!(
        span.end <= tokens.len() && span.start < span.end,
        "capture span {span:?} out of range for {} tokens",
        tokens.len()
    );
    model.clear_kv_cache();
    let input = Tensor::new(tokens, device)?.unsqueeze(0)?;
    let mut tapped: Vec<(usize, Tensor)> = Vec::new();
    let mut edit = LayerEdit::CaptureMany {
        after_blocks,
        out: &mut tapped,
    };
    let logits = model.forward_with_edit(&input, 0, Some(&mut edit))?;
    let mut captures = Vec::with_capacity(after_blocks.len());
    for &block in after_blocks {
        let t = tapped
            .iter()
            .find(|(b, _)| *b == block)
            .map(|(_, t)| t)
            .unwrap_or_else(|| panic!("tap after block {block} did not fire (layer count?)"));
        let rows_t = span_rows_f32(t, &span)?;
        let capture = pool_rows(&rows_t, block, span.clone())?;
        // (1, span, hidden) flattened row-major = [span x hidden] rows.
        let rows = rows_t.flatten_all()?.to_vec1::<f32>()?;
        captures.push(CaptureWithRows { capture, rows });
    }
    Ok((logits, captures))
}

/// Narrow a tapped `(1, seq, hidden)` residual to the span rows in f32.
fn span_rows_f32(tapped: &Tensor, span: &Range<usize>) -> Result<Tensor> {
    tapped
        .narrow(1, span.start, span.end - span.start)?
        .to_dtype(DType::F32)
}

/// Pool one tapped `(1, seq, hidden)` residual tensor into a [`Capture`].
fn pool_capture(tapped: &Tensor, after_block: usize, span: Range<usize>) -> Result<Capture> {
    let rows = span_rows_f32(tapped, &span)?;
    pool_rows(&rows, after_block, span)
}

/// Pool already-narrowed f32 span rows `(1, span, hidden)` into a
/// [`Capture`] — the exact op sequence the pooled S2/S2c dumps used.
fn pool_rows(rows: &Tensor, after_block: usize, span: Range<usize>) -> Result<Capture> {
    let hidden_size = rows.dim(2)?;
    let pooled = rows.mean(1)?.squeeze(0)?.to_vec1::<f32>()?;
    let per_position_l2 = rows
        .sqr()?
        .sum(D::Minus1)?
        .sqrt()?
        .squeeze(0)?
        .to_vec1::<f32>()?;
    Ok(Capture {
        after_block,
        span,
        pooled,
        hidden_size,
        per_position_l2,
    })
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
