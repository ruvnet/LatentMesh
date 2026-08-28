//! Vendored candle-transformers 0.9.2 `src/models/qwen2.rs` — part B of 2.
//!
//! Carries `DecoderLayer`, `Model` and `ModelForCausalLM` (source lines
//! 217–402 of the original); part A ([`super::qwen2_a`]) carries the rest.
//! Provenance and the verbatim-deviation ledger are in part A's header.
//!
//! Additional deviation (the point of vendoring, design doc 024 §3):
//! `Model::forward` is re-expressed as `forward_with_edit(.., None)`, where
//! [`LayerEdit`] optionally captures or overwrites the residual stream
//! immediately after block k. With `edit = None` the executed tensor-op
//! sequence is identical to the vendored original, and with a `Capture` edit
//! the only extra op is a `Tensor::clone` of the residual (no arithmetic
//! touches the forward path) — the S0 logits-parity gate verifies this
//! bit-identically on live GPU rather than trusting the argument.

use super::qwen2_a::{rms_norm, Attention, Config, RotaryEmbedding, MLP};
use candle_core::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{Linear, RmsNorm, VarBuilder};
use std::sync::Arc;

/// A single residual-stream edit applied during one forward pass.
///
/// `after_block` counts executed decoder blocks, 1-based: `after_block = 24`
/// edits the residual stream after 24 of the sender's 36 blocks have run
/// (design notation "L24/36"). Positions are absolute token positions within
/// the *current* pass's sequence axis, so injection is meaningful during
/// prefill (seqlen_offset 0); decode steps see the edit only through the KV
/// cache of blocks > `after_block`, which is exactly the intended
/// RoPE/KV-coherent persistence.
#[derive(Debug)]
pub enum LayerEdit<'a> {
    /// Read-only tap: clone the residual hidden state `(b, seq, hidden)`.
    Capture {
        after_block: usize,
        out: &'a mut Option<Tensor>,
    },
    /// Read-only multi-tap: clone the residual hidden state after EACH listed
    /// block in one pass (S2 calibration sweeps 3 depths per model; three
    /// single-tap passes would triple the prefill cost). Like `Capture`, the
    /// only extra op per tap is a `Tensor::clone` — the S2 dump receipt
    /// re-measures logits parity against the unpatched forward rather than
    /// trusting this argument.
    CaptureMany {
        after_blocks: &'a [usize],
        out: &'a mut Vec<(usize, Tensor)>,
    },
    /// Overwrite the residual rows at `positions` with the rows of `vectors`
    /// (`(positions.len(), hidden)`, any float dtype; cast to model dtype).
    /// An empty `positions` list is a no-op *by construction*: the tensor
    /// passes through untouched (receipts must label the zero-slot gate as
    /// such, per the evidence-honesty rule).
    Inject {
        after_block: usize,
        positions: &'a [usize],
        vectors: &'a Tensor,
    },
}

fn apply_edit(xs: Tensor, blocks_run: usize, edit: &mut LayerEdit<'_>) -> Result<Tensor> {
    match edit {
        LayerEdit::Capture { after_block, out } if *after_block == blocks_run => {
            **out = Some(xs.clone());
            Ok(xs)
        }
        LayerEdit::CaptureMany { after_blocks, out } if after_blocks.contains(&blocks_run) => {
            out.push((blocks_run, xs.clone()));
            Ok(xs)
        }
        LayerEdit::Inject {
            after_block,
            positions,
            vectors,
        } if *after_block == blocks_run => {
            if positions.is_empty() {
                return Ok(xs);
            }
            let hidden = xs.dim(2)?;
            let mut xs = xs;
            for (row, &pos) in positions.iter().enumerate() {
                let v = vectors
                    .narrow(0, row, 1)?
                    .reshape((1, 1, hidden))?
                    .to_dtype(xs.dtype())?
                    .to_device(xs.device())?;
                xs = xs.slice_assign(&[0..1, pos..pos + 1, 0..hidden], &v)?;
            }
            Ok(xs)
        }
        _ => Ok(xs),
    }
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: MLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(rotary_emb: Arc<RotaryEmbedding>, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let self_attn = Attention::new(rotary_emb, cfg, vb.pp("self_attn"))?;
        let mlp = MLP::new(cfg, vb.pp("mlp"))?;
        let input_layernorm =
            rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, attention_mask, seqlen_offset)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = xs.apply(&self.post_attention_layernorm)?.apply(&self.mlp)?;
        residual + xs
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache()
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    sliding_window: usize,
    device: Device,
    dtype: DType,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let rotary_emb = Arc::new(RotaryEmbedding::new(vb.dtype(), cfg, vb_m.device())?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = DecoderLayer::new(rotary_emb.clone(), cfg, vb_l.pp(layer_idx))?;
            layers.push(layer)
        }
        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            sliding_window: cfg.sliding_window,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    fn prepare_causal_attention_mask(
        &self,
        b_size: usize,
        tgt_len: usize,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        // Sliding window mask?
        let mask: Vec<_> = (0..tgt_len)
            .flat_map(|i| {
                (0..tgt_len).map(move |j| {
                    if i < j || j + self.sliding_window < i {
                        f32::NEG_INFINITY
                    } else {
                        0.
                    }
                })
            })
            .collect();
        let mask = Tensor::from_slice(&mask, (tgt_len, tgt_len), &self.device)?;
        let mask = if seqlen_offset > 0 {
            let mask0 = Tensor::zeros((tgt_len, seqlen_offset), self.dtype, &self.device)?;
            Tensor::cat(&[&mask0, &mask], D::Minus1)?
        } else {
            mask
        };
        mask.expand((b_size, 1, tgt_len, tgt_len + seqlen_offset))?
            .to_dtype(self.dtype)
    }

    fn prepare_attention_mask(&self, attn_mask: &Tensor) -> Result<Tensor> {
        let (b_sz, sql_len) = attn_mask.dims2()?;
        let mut mask: Vec<Tensor> = vec![];
        for b in 0..b_sz {
            mask.push(attn_mask.i((b, ..))?.expand((1, 1, sql_len, sql_len))?);
        }
        let mask = Tensor::cat(&mask, 0)?;
        let on_true = mask.zeros_like()?.to_dtype(self.dtype)?;
        let on_false = Tensor::new(f32::NEG_INFINITY, &self.device)?
            .broadcast_as(mask.shape())?
            .to_dtype(self.dtype)?;
        mask.where_cond(&on_true, &on_false)
    }

    /// Vendored-original forward: identical op sequence to
    /// candle-transformers 0.9.2 (`edit = None` adds no operation).
    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        attn_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        self.forward_with_edit(input_ids, seqlen_offset, attn_mask, None)
    }

    /// Forward pass with an optional residual-stream edit after block k.
    pub fn forward_with_edit(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        attn_mask: Option<&Tensor>,
        mut edit: Option<&mut LayerEdit<'_>>,
    ) -> Result<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;
        let attention_mask: Option<Tensor> = match attn_mask {
            Some(mask) => Some(self.prepare_attention_mask(mask)?),
            None => {
                if seq_len <= 1 {
                    None
                } else {
                    Some(self.prepare_causal_attention_mask(b_size, seq_len, seqlen_offset)?)
                }
            }
        };
        let mut xs = self.embed_tokens.forward(input_ids)?;
        for (idx, layer) in self.layers.iter_mut().enumerate() {
            xs = layer.forward(&xs, attention_mask.as_ref(), seqlen_offset)?;
            if let Some(e) = edit.as_deref_mut() {
                xs = apply_edit(xs, idx + 1, e)?;
            }
        }
        xs.apply(&self.norm)
    }

    pub fn embed_tokens(&self) -> &candle_nn::Embedding {
        &self.embed_tokens
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache()
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelForCausalLM {
    base_model: Model,
    lm_head: Linear,
}

impl ModelForCausalLM {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let base_model = Model::new(cfg, vb.clone())?;
        let lm_head = if vb.contains_tensor("lm_head.weight") {
            candle_nn::linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        } else {
            Linear::new(base_model.embed_tokens().embeddings().clone(), None)
        };
        Ok(Self {
            base_model,
            lm_head,
        })
    }

    /// Vendored-original forward: last-position logits `(b, 1, vocab)`.
    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        self.forward_with_edit(input_ids, seqlen_offset, None)
    }

    /// Last-position logits with an optional residual edit after block k.
    pub fn forward_with_edit(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        edit: Option<&mut LayerEdit<'_>>,
    ) -> Result<Tensor> {
        let (_b_size, seq_len) = input_ids.dims2()?;
        self.base_model
            .forward_with_edit(input_ids, seqlen_offset, None, edit)?
            .narrow(1, seq_len - 1, 1)?
            .apply(&self.lm_head)
    }

    /// Full-sequence logits `(b, seq, vocab)` with an optional edit —
    /// used for teacher-forced NLL diagnostics, not by the vendored API.
    pub fn forward_full_logits(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        edit: Option<&mut LayerEdit<'_>>,
    ) -> Result<Tensor> {
        self.base_model
            .forward_with_edit(input_ids, seqlen_offset, None, edit)?
            .apply(&self.lm_head)
    }

    pub fn clear_kv_cache(&mut self) {
        self.base_model.clear_kv_cache()
    }
}
