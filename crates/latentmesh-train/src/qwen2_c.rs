//! qwen2_c — the training-shaped differentiable Qwen2 forward (ADR-024 M4c).
//!
//! The vendored inference forward (`latentmesh-runtime` qwen2_a/qwen2_b)
//! CANNOT backprop to an injected vector: candle-nn 0.9.2's
//! `softmax_last_dim`, `ops::rms_norm` and `rotary_emb::rope` are all
//! `apply_op*_no_bwd` (custom op never recorded on the tape), so the graph is
//! silently cut — measured in the M4c feasibility probe (grads.get == None).
//! This module is the minimal-change fix that probe validated: the SAME
//! function with exactly three call-site substitutions —
//!   1. `ops::rms_norm`        → `ops::rms_norm_slow` (composed, has bwd)
//!   2. `ops::softmax_last_dim`→ composed softmax with DETACHED max (F32)
//!   3. `rotary_emb::rope`     → composed rotate-half (cos/sin tables built
//!      F32 then cast, matching the vendored deviation-4 convention)
//!
//! Parity vs the vendored forward, measured by the feasibility probe on this
//! host: F32 128/128 argmax agreement (max|dlogit| 0.119 — same function);
//! BF16 composed-vs-fused 116/128 (max|dlogit| 8.19, pure rounding
//! amplification — the registered M4c numeric caveat).
//!
//! No KV cache: this forward exists for teacher-forced training passes only.
//! Injection mirrors `qwen2_b::apply_edit`'s `Inject`/`Fuse` arms op-for-op
//! (per-position `slice_assign` of a narrowed row — or of that row ADDED to
//! the receiver's own row, under [`InjectionMode::Fuse`] — cast to model
//! dtype), so an F32 adapter output feeds the BF16 model with gradients
//! intact (`ToDType` backward exists in candle 0.9.2). The operator is
//! selected explicitly per call so the training loop and the frozen probe
//! can be pinned to the same one (ADR-024 M4g).

use crate::receiver_lora::LoraAdapter;
use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::{linear, linear_no_bias, Activation, Linear, VarBuilder};
use latentmesh_runtime::inject::InjectionMode;
use latentmesh_runtime::Config;

struct TrainLayer {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    gate: Linear,
    up: Linear,
    down: Linear,
    input_ln_w: Tensor,
    post_ln_w: Tensor,
}

/// The frozen receiver, loaded for differentiable teacher-forced passes.
pub struct TrainReceiver {
    embed: candle_nn::Embedding,
    layers: Vec<TrainLayer>,
    norm_w: Tensor,
    lm_head: Linear,
    cos: Tensor, // (max_seq, head_dim/2), model dtype
    sin: Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    hidden: usize,
    eps: f32,
    act: Activation,
    dtype: DType,
    device: Device,
}

/// Fetch and parse the model's `config.json` from the hf-hub cache.
pub fn load_config(model_id: &str) -> anyhow::Result<Config> {
    let api = hf_hub::api::sync::Api::new()?;
    let repo = api.model(model_id.to_string());
    Ok(serde_json::from_str(&std::fs::read_to_string(
        repo.get("config.json")?,
    )?)?)
}

impl TrainReceiver {
    /// Load from the same hf-hub cache the runtime uses (single-file
    /// `model.safetensors` repos only — Qwen2.5-1.5B-Instruct is one).
    pub fn load(
        model_id: &str,
        cfg: &Config,
        dtype: DType,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let api = hf_hub::api::sync::Api::new()?;
        let repo = api.model(model_id.to_string());
        let weights = vec![repo.get("model.safetensors")?];
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weights, dtype, device)? };
        let vb_m = vb.pp("model");
        let embed = candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            let vb_i = vb_l.pp(i);
            let vb_a = vb_i.pp("self_attn");
            let vb_p = vb_i.pp("mlp");
            layers.push(TrainLayer {
                q: linear(
                    cfg.hidden_size,
                    cfg.num_attention_heads * head_dim,
                    vb_a.pp("q_proj"),
                )?,
                k: linear(
                    cfg.hidden_size,
                    cfg.num_key_value_heads * head_dim,
                    vb_a.pp("k_proj"),
                )?,
                v: linear(
                    cfg.hidden_size,
                    cfg.num_key_value_heads * head_dim,
                    vb_a.pp("v_proj"),
                )?,
                o: linear_no_bias(
                    cfg.num_attention_heads * head_dim,
                    cfg.hidden_size,
                    vb_a.pp("o_proj"),
                )?,
                gate: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb_p.pp("gate_proj"))?,
                up: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb_p.pp("up_proj"))?,
                down: linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb_p.pp("down_proj"))?,
                input_ln_w: vb_i.pp("input_layernorm").get(cfg.hidden_size, "weight")?,
                post_ln_w: vb_i
                    .pp("post_attention_layernorm")
                    .get(cfg.hidden_size, "weight")?,
            });
        }
        let norm_w = vb_m.pp("norm").get(cfg.hidden_size, "weight")?;
        let lm_head = if vb.contains_tensor("lm_head.weight") {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        } else {
            Linear::new(embed.embeddings().clone(), None)
        };
        // Rope tables: F32 build, cast to model dtype (vendored deviation 4).
        let max_seq = 4096usize.min(cfg.max_position_embeddings);
        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / head_dim as f64) as f32)
            .collect();
        let n_if = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, n_if), device)?;
        let t = Tensor::arange(0u32, max_seq as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_seq, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            embed,
            layers,
            norm_w,
            lm_head,
            cos: freqs.cos()?.to_dtype(dtype)?,
            sin: freqs.sin()?.to_dtype(dtype)?,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim,
            hidden: cfg.hidden_size,
            eps: cfg.rms_norm_eps as f32,
            act: cfg.hidden_act,
            dtype,
            device: device.clone(),
        })
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden
    }

    fn causal_mask(&self, l: usize) -> candle_core::Result<Tensor> {
        let mask: Vec<f32> = (0..l)
            .flat_map(|i| (0..l).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 }))
            .collect();
        Tensor::from_vec(mask, (1, 1, l, l), &self.device)?.to_dtype(self.dtype)
    }

    /// Composed rotate-half rope (substitution 3).
    fn rope(&self, x: &Tensor, l: usize) -> candle_core::Result<Tensor> {
        let d2 = self.head_dim / 2;
        let cos = self.cos.narrow(0, 0, l)?.reshape((1, 1, l, d2))?;
        let sin = self.sin.narrow(0, 0, l)?.reshape((1, 1, l, d2))?;
        let x1 = x.narrow(3, 0, d2)?;
        let x2 = x.narrow(3, d2, d2)?;
        let o1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
        let o2 = (x1.broadcast_mul(&sin)? + x2.broadcast_mul(&cos)?)?;
        Tensor::cat(&[o1, o2], 3)
    }

    fn repeat_kv(&self, x: Tensor) -> candle_core::Result<Tensor> {
        let n_rep = self.num_heads / self.num_kv_heads;
        if n_rep == 1 {
            return Ok(x);
        }
        let (b, h, t, d) = x.dims4()?;
        Tensor::cat(&vec![&x; n_rep], 2)?.reshape((b, h * n_rep, t, d))
    }

    /// Composed softmax with detached max, F32 upcast (substitution 2).
    fn softmax_f32(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let xf = x.to_dtype(DType::F32)?;
        let m = xf.max_keepdim(D::Minus1)?.detach();
        let z = xf.broadcast_sub(&m)?.exp()?;
        let s = z.sum_keepdim(D::Minus1)?;
        z.broadcast_div(&s)?.to_dtype(x.dtype())
    }

    /// Composed rms_norm (substitution 1).
    fn rmsnorm(&self, x: &Tensor, w: &Tensor) -> candle_core::Result<Tensor> {
        candle_nn::ops::rms_norm_slow(x, w, self.eps)
    }

    /// One decoder block (attention + MLP with residuals), no KV cache.
    fn block(&self, xs: &Tensor, layer: &TrainLayer, mask: &Tensor) -> candle_core::Result<Tensor> {
        let l = xs.dim(1)?;
        let scale = 1f64 / (self.head_dim as f64).sqrt();
        let residual = xs.clone();
        let h = self.rmsnorm(xs, &layer.input_ln_w)?;
        let q = h
            .apply(&layer.q)?
            .reshape((1, l, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = h
            .apply(&layer.k)?
            .reshape((1, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = h
            .apply(&layer.v)?
            .reshape((1, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let q = self.rope(&q, l)?;
        let k = self.rope(&k, l)?;
        let k = self.repeat_kv(k)?.contiguous()?;
        let v = self.repeat_kv(v)?.contiguous()?;
        let att = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let att = att.broadcast_add(mask)?;
        let att = self.softmax_f32(&att)?;
        let out = att
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((1, l, self.hidden))?
            .apply(&layer.o)?;
        let xs = (out + residual)?;
        let residual = xs.clone();
        let h2 = self.rmsnorm(&xs, &layer.post_ln_w)?;
        let mlp = (h2.apply(&layer.gate)?.apply(&self.act)? * h2.apply(&layer.up)?)?
            .apply(&layer.down)?;
        residual + mlp
    }

    /// Full differentiable forward with optional injection AFTER
    /// `after_block` (1-based, `qwen2_b::apply_edit` convention). `inject` =
    /// `(vectors (n_positions, hidden), positions, after_block, mode)`, where
    /// `mode` selects the SAME residual operator the frozen probe will use —
    /// [`InjectionMode::Overwrite`] (M3/M4/M4c/M4d) or
    /// [`InjectionMode::Fuse`] (ADR-024 M4g). Returns logits for hidden rows
    /// `[span_start, span_start + t_len)` — narrowed BEFORE lm_head (vocab
    /// work only on the loss span; the measured VRAM envelope depends on
    /// this).
    pub fn forward_span_logits(
        &self,
        tokens: &Tensor,
        inject: Option<(&Tensor, &[usize], usize, InjectionMode)>,
        span_start: usize,
        t_len: usize,
    ) -> candle_core::Result<Tensor> {
        self.forward_span_logits_with_lora(tokens, inject, None, span_start, t_len)
    }

    /// [`Self::forward_span_logits`] with an ADR-045 M5 receiver adapter
    /// applied after `lora.after_block` blocks.
    ///
    /// ORDER, identical to the deployed `latentmesh_runtime::models::Model`
    /// forward and recorded in every M5 receipt: **block → injection edit →
    /// LoRA**. The adapter therefore sees the injected content, while
    /// [`Self::natural_per_position_l2`] (which never runs the adapter)
    /// still reports the base receiver's norms, so the rescale target is the
    /// same one every frozen-receiver rung used.
    pub fn forward_span_logits_with_lora(
        &self,
        tokens: &Tensor,
        inject: Option<(&Tensor, &[usize], usize, InjectionMode)>,
        lora: Option<(&LoraAdapter, usize)>,
        span_start: usize,
        t_len: usize,
    ) -> candle_core::Result<Tensor> {
        let (_b, l) = tokens.dims2()?;
        let mask = self.causal_mask(l)?;
        let mut xs = self.embed.forward(tokens)?;
        for (idx, layer) in self.layers.iter().enumerate() {
            xs = self.block(&xs, layer, &mask)?;
            if let Some((vectors, positions, after_block, mode)) = inject {
                if idx + 1 == after_block {
                    // Mirror apply_edit's Inject / Fuse arm op-for-op.
                    for (row, &pos) in positions.iter().enumerate() {
                        let v = vectors
                            .narrow(0, row, 1)?
                            .reshape((1, 1, self.hidden))?
                            .to_dtype(xs.dtype())?
                            .to_device(xs.device())?;
                        let write = match mode {
                            InjectionMode::Overwrite => v,
                            InjectionMode::Fuse => (xs.narrow(1, pos, 1)? + v)?,
                        };
                        xs = xs.slice_assign(&[0..1, pos..pos + 1, 0..self.hidden], &write)?;
                    }
                }
            }
            if let Some((adapter, after_block)) = lora {
                if idx + 1 == after_block {
                    xs = adapter.apply(&xs)?;
                }
            }
        }
        let xs = self.rmsnorm(&xs, &self.norm_w)?;
        let rows = xs.narrow(1, span_start, t_len)?;
        rows.apply(&self.lm_head)
    }

    /// Per-position L2 norms of the natural (un-injected) residual after
    /// `after_block` blocks over `tokens` — the training-side analogue of the
    /// probe's `forward_capture(..).per_position_l2` rescale source, computed
    /// through THIS composed forward (disclosed deviation: the frozen probe
    /// recomputes its own norms through the vendored fused forward at eval).
    pub fn natural_per_position_l2(
        &self,
        tokens: &Tensor,
        after_block: usize,
    ) -> candle_core::Result<Vec<f32>> {
        let (_b, l) = tokens.dims2()?;
        assert!(
            (1..=self.layers.len()).contains(&after_block),
            "after_block {after_block} out of range"
        );
        let mask = self.causal_mask(l)?;
        let mut xs = self.embed.forward(tokens)?;
        for layer in self.layers.iter().take(after_block) {
            xs = self.block(&xs, layer, &mask)?;
        }
        let rows = xs.detach().to_dtype(DType::F32)?.squeeze(0)?;
        let norms = rows.sqr()?.sum(D::Minus1)?.sqrt()?;
        norms.to_vec1::<f32>()
    }
}

/// Teacher-forced mean CE (nats/token) from span logits `(1, t_len, vocab)`
/// against `targets`. Differentiable composition (detached max, F32).
pub fn span_ce(logits: &Tensor, targets: &[u32], device: &Device) -> candle_core::Result<Tensor> {
    let (_b, t_len, vocab) = logits.dims3()?;
    assert_eq!(t_len, targets.len(), "span/target length mismatch");
    let lf = logits.to_dtype(DType::F32)?.reshape((t_len, vocab))?;
    let m = lf.max_keepdim(D::Minus1)?.detach();
    let z = lf.broadcast_sub(&m)?;
    let lse = z.exp()?.sum_keepdim(D::Minus1)?.log()?;
    let logp = z.broadcast_sub(&lse)?;
    let tgt = Tensor::from_vec(targets.to_vec(), (t_len, 1), device)?;
    let picked = logp.gather(&tgt, 1)?;
    picked.mean_all()?.neg()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn span_ce_matches_hand_computation() {
        // 2 positions, 3-token vocab, known logits.
        let dev = Device::Cpu;
        let logits =
            Tensor::from_vec(vec![1.0f32, 0.0, -1.0, 0.5, 0.5, 0.5], (1, 2, 3), &dev).unwrap();
        let ce = span_ce(&logits, &[0, 2], &dev)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        // pos 0: -log softmax([1,0,-1])[0]; pos 1: -log(1/3).
        let e: f32 = (1f32.exp() + 1.0 + (-1f32).exp()).ln() - 1.0;
        let expected = (e + (3f32).ln()) / 2.0;
        assert!((ce - expected).abs() < 1e-5, "ce {ce} vs {expected}");
    }
}
