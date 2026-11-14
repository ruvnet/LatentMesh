//! Inference port of memorilla/memory.py at d3e3f8f1b6dd2560971dfc989ffd017c40d5b825.
use crate::weights::{file_hash, Weights};
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{LayerNorm, Linear};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::Path;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryConfig {
    pub embedding_dim: usize,
    pub output_dim: usize,
    pub num_memories: usize,
    pub num_heads: usize,
    pub num_self_attn_layers: usize,
    pub num_cross_attn_layers: usize,
    pub dropout: f64,
    pub retrieval_init: bool,
}
impl MemoryConfig {
    pub fn validate(&self) -> Result<()> {
        if self.embedding_dim == 0
            || self.embedding_dim > 16384
            || self.output_dim == 0
            || self.output_dim > 16384
            || self.num_heads == 0
            || !self.embedding_dim.is_multiple_of(self.num_heads)
            || self.num_memories == 0
            || self.num_memories > 1024
            || self.num_cross_attn_layers == 0
            || self.num_cross_attn_layers > 32
            || self.num_self_attn_layers > 32
            || !self.dropout.is_finite()
            || !(0.0..=1.0).contains(&self.dropout)
        {
            candle_core::bail!("invalid or excessive memory configuration");
        }
        Ok(())
    }
    pub fn supports_cache(&self) -> bool {
        !self.retrieval_init && self.num_self_attn_layers == 0
    }
}

struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    heads: usize,
    dim: usize,
}
impl Attention {
    fn load(w: &mut Weights, name: &str, dim: usize, heads: usize) -> Result<Self> {
        let weight = w.tensor(&format!("{name}.in_proj_weight"), &[3 * dim, dim])?;
        let bias = w.tensor(&format!("{name}.in_proj_bias"), &[3 * dim])?;
        let projection = |i| -> Result<Linear> {
            Ok(Linear::new(
                weight.narrow(0, i * dim, dim)?.contiguous()?,
                Some(bias.narrow(0, i * dim, dim)?.contiguous()?),
            ))
        };
        Ok(Self {
            q: projection(0)?,
            k: projection(1)?,
            v: projection(2)?,
            out: w.linear(&format!("{name}.out_proj"), dim, dim)?,
            heads,
            dim,
        })
    }
    fn split(&self, x: Tensor) -> Result<Tensor> {
        let (b, n, _) = x.dims3()?;
        x.reshape((b, n, self.heads, self.dim / self.heads))?
            .transpose(1, 2)?
            .contiguous()
    }
    fn project(&self, docs: &Tensor) -> Result<(Tensor, Tensor)> {
        Ok((
            self.split(self.k.forward(docs)?)?
                .transpose(2, 3)?
                .contiguous()?,
            self.split(self.v.forward(docs)?)?,
        ))
    }
    fn attend(
        &self,
        slots: &Tensor,
        keys: &Tensor,
        values: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (batch, count, _) = slots.dims3()?;
        let q = self.split(self.q.forward(slots)?)?;
        let mut scores = (q.broadcast_matmul(keys)?.to_dtype(DType::F32)?
            * (self.dim as f64 / self.heads as f64).sqrt().recip())?;
        if let Some(mask) = mask {
            let mask = mask
                .unsqueeze(1)?
                .unsqueeze(1)?
                .broadcast_as(scores.shape())?;
            scores = mask.where_cond(
                &Tensor::full(f32::NEG_INFINITY, scores.shape(), scores.device())?,
                &scores,
            )?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?.to_dtype(values.dtype())?;
        let attended = probs
            .broadcast_matmul(values)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, count, self.dim))?;
        self.out.forward(&attended.to_dtype(slots.dtype())?)
    }
    fn forward(&self, slots: &Tensor, docs: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (k, v) = self.project(docs)?;
        self.attend(slots, &k, &v, mask)
    }
}
struct Block {
    attn: Attention,
    attn_norm: LayerNorm,
    ffn_norm: LayerNorm,
    up: Linear,
    down: Linear,
}
struct EncoderBlock {
    attn: Attention,
    norm1: LayerNorm,
    norm2: LayerNorm,
    up: Linear,
    down: Linear,
}

pub struct MemoryModule {
    config: MemoryConfig,
    checkpoint_hash: String,
    device: Device,
    dtype: DType,
    queries: Option<Tensor>,
    query_attn: Attention,
    query_norm: LayerNorm,
    doc_norm: LayerNorm,
    encoder: Vec<EncoderBlock>,
    blocks: Vec<Block>,
    norm: LayerNorm,
    output: Linear,
}

#[derive(Clone)]
pub struct MemoryCache {
    layers: Vec<(Tensor, Tensor)>,
    mask: Option<Tensor>,
    checkpoint_hash: String,
    pub collections: usize,
    pub documents: usize,
    pub nbytes: usize,
}

impl MemoryModule {
    pub fn load(directory: impl AsRef<Path>, device: &Device, dtype: DType) -> Result<Self> {
        let dir = directory.as_ref();
        let config_bytes = std::fs::read(dir.join("config.json"))?;
        let config: MemoryConfig =
            serde_json::from_slice(&config_bytes).map_err(candle_core::Error::wrap)?;
        config.validate()?;
        let path = if dir.join("memory.safetensors").exists() {
            dir.join("memory.safetensors")
        } else {
            dir.join("memory.pt")
        };
        let checkpoint_hash = format!(
            "{:x}",
            Sha256::digest(format!(
                "{}:{}",
                file_hash(&path)?,
                String::from_utf8_lossy(&config_bytes)
            ))
        );
        let mut w = Weights::load(&path, device, dtype)?;
        let d = config.embedding_dim;
        let h = config.num_heads;
        let queries = if config.retrieval_init {
            None
        } else {
            Some(w.tensor("memory_queries", &[1, config.num_memories, d])?)
        };
        let query_attn = Attention::load(&mut w, "query_cond_attn", d, h)?;
        let query_norm = w.norm("query_cond_norm", d)?;
        let doc_norm = w.norm("doc_norm", d)?;
        let mut encoder = Vec::new();
        for i in 0..config.num_self_attn_layers {
            let p = format!("encoder.layers.{i}");
            encoder.push(EncoderBlock {
                attn: Attention::load(&mut w, &format!("{p}.self_attn"), d, h)?,
                norm1: w.norm(&format!("{p}.norm1"), d)?,
                norm2: w.norm(&format!("{p}.norm2"), d)?,
                up: w.linear(&format!("{p}.linear1"), d, 4 * d)?,
                down: w.linear(&format!("{p}.linear2"), 4 * d, d)?,
            });
        }
        let mut blocks = Vec::new();
        for i in 0..config.num_cross_attn_layers {
            blocks.push(Block {
                attn: Attention::load(&mut w, &format!("cross_attn_layers.{i}"), d, h)?,
                attn_norm: w.norm(&format!("cross_attn_norms.{i}"), d)?,
                ffn_norm: w.norm(&format!("ffn_norms.{i}"), d)?,
                up: w.linear(&format!("ffn_layers.{i}.0"), d, 4 * d)?,
                down: w.linear(&format!("ffn_layers.{i}.3"), 4 * d, d)?,
            });
        }
        let norm = w.norm("layer_norm", d)?;
        let output = w.linear("memory_projection", d, config.output_dim)?;
        w.finish()?;
        Ok(Self {
            config,
            checkpoint_hash,
            device: device.clone(),
            dtype,
            queries,
            query_attn,
            query_norm,
            doc_norm,
            encoder,
            blocks,
            norm,
            output,
        })
    }
    pub fn config(&self) -> &MemoryConfig {
        &self.config
    }
    pub fn checkpoint_hash(&self) -> &str {
        &self.checkpoint_hash
    }
    pub fn device(&self) -> &Device {
        &self.device
    }
    pub fn dtype(&self) -> DType {
        self.dtype
    }
    fn input(&self, t: &Tensor) -> Result<Tensor> {
        t.to_device(&self.device)?.to_dtype(self.dtype)
    }
    fn validate_docs(&self, docs: &Tensor, mask: Option<&Tensor>) -> Result<(usize, usize)> {
        let (b, n, d) = docs.dims3()?;
        if b == 0 || n == 0 || d != self.config.embedding_dim {
            candle_core::bail!("documents must be nonempty [batch, documents, embedding_dim]");
        }
        if let Some(mask) = mask {
            if mask.dims() != [b, n] {
                candle_core::bail!("document mask shape mismatch");
            }
            for row in mask.to_dtype(DType::U8)?.to_vec2::<u8>()? {
                if row.iter().any(|x| *x > 1) || row.iter().all(|x| *x != 0) {
                    candle_core::bail!(
                        "mask must be boolean with at least one visible document per collection"
                    );
                }
            }
        }
        Ok((b, n))
    }
    fn validate_questions(&self, questions: &Tensor) -> Result<usize> {
        let (b, d) = questions.dims2()?;
        if b == 0 || d != self.config.embedding_dim {
            candle_core::bail!("question shape mismatch");
        }
        Ok(b)
    }
    fn initial(
        &self,
        docs: Option<&Tensor>,
        mask: Option<&Tensor>,
        questions: &Tensor,
    ) -> Result<Tensor> {
        let (b, d) = questions.dims2()?;
        let m = self.config.num_memories;
        let slots = if let Some(q) = &self.queries {
            q.broadcast_as((b, m, d))?.contiguous()?
        } else {
            let docs = docs.ok_or_else(|| {
                candle_core::Error::Msg("retrieval initialization needs documents".into())
            })?;
            let source = docs.to_dtype(DType::F32)?.to_vec3::<f32>()?;
            let qs = questions.to_dtype(DType::F32)?.to_vec2::<f32>()?;
            let masks = mask.map(|x| x.to_vec2::<u8>()).transpose()?;
            let mut selected = Vec::with_capacity(b * m * d);
            for i in 0..b {
                let normq = qs[i].iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
                let mut scores: Vec<_> = source[i]
                    .iter()
                    .enumerate()
                    .map(|(j, v)| {
                        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
                        let score = if masks.as_ref().is_some_and(|x| x[i][j] != 0) {
                            f32::NEG_INFINITY
                        } else {
                            v.iter().zip(&qs[i]).map(|(a, b)| a * b).sum::<f32>() / (norm * normq)
                        };
                        (j, score)
                    })
                    .collect();
                scores.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
                for (j, _) in scores.iter().take(m) {
                    selected.extend_from_slice(&source[i][*j]);
                }
                selected.resize((i + 1) * m * d, 0.0);
            }
            Tensor::from_vec(selected, (b, m, d), &self.device)?.to_dtype(self.dtype)?
        };
        let question = questions.unsqueeze(1)?;
        self.query_norm
            .forward(&(&slots + self.query_attn.forward(&slots, &question, None)?)?)
    }
    pub fn forward(
        &self,
        docs: &Tensor,
        mask: Option<&Tensor>,
        questions: &Tensor,
    ) -> Result<Tensor> {
        let (b, _) = self.validate_docs(docs, mask)?;
        if self.validate_questions(questions)? != b {
            candle_core::bail!("forward needs one collection per question");
        }
        let mut docs = self.input(docs)?;
        let questions = self.input(questions)?;
        let mask = mask
            .map(|x| x.to_device(&self.device)?.to_dtype(DType::U8))
            .transpose()?;
        let mut slots = self.initial(Some(&docs), mask.as_ref(), &questions)?;
        for layer in &self.encoder {
            docs = layer
                .norm1
                .forward(&(&docs + layer.attn.forward(&docs, &docs, mask.as_ref())?)?)?;
            docs = layer
                .norm2
                .forward(&(&docs + layer.down.forward(&layer.up.forward(&docs)?.relu()?)?)?)?;
        }
        docs = self.doc_norm.forward(&docs)?;
        for layer in &self.blocks {
            slots = (&slots
                + layer
                    .attn
                    .forward(&layer.attn_norm.forward(&slots)?, &docs, mask.as_ref())?)?;
            slots = (&slots
                + layer.down.forward(
                    &layer
                        .up
                        .forward(&layer.ffn_norm.forward(&slots)?)?
                        .gelu_erf()?,
                )?)?;
        }
        self.output.forward(&self.norm.forward(&slots)?)
    }
    pub fn write_cache(&self, docs: &Tensor, mask: Option<&Tensor>) -> Result<MemoryCache> {
        if !self.config.supports_cache() {
            candle_core::bail!("caching requires learned slots and no document self attention");
        }
        let (b, n) = self.validate_docs(docs, mask)?;
        let docs = self.doc_norm.forward(&self.input(docs)?)?;
        let layers = self
            .blocks
            .iter()
            .map(|l| l.attn.project(&docs))
            .collect::<Result<Vec<_>>>()?;
        let nbytes = layers
            .iter()
            .map(|(k, v)| (k.elem_count() + v.elem_count()) * self.dtype.size_in_bytes())
            .sum();
        Ok(MemoryCache {
            layers,
            mask: mask
                .map(|m| m.to_device(&self.device)?.to_dtype(DType::U8))
                .transpose()?,
            checkpoint_hash: self.checkpoint_hash.clone(),
            collections: b,
            documents: n,
            nbytes,
        })
    }
    pub fn read_cache(&self, cache: &MemoryCache, questions: &Tensor) -> Result<Tensor> {
        let batch = self.validate_questions(questions)?;
        if cache.checkpoint_hash != self.checkpoint_hash
            || (cache.collections != 1 && cache.collections != batch)
            || cache.layers.len() != self.blocks.len()
        {
            candle_core::bail!("cache model or collection count mismatch");
        }
        let questions = self.input(questions)?;
        let mut slots = self.initial(None, None, &questions)?;
        for (layer, (k, v)) in self.blocks.iter().zip(&cache.layers) {
            slots = (&slots
                + layer.attn.attend(
                    &layer.attn_norm.forward(&slots)?,
                    k,
                    v,
                    cache.mask.as_ref(),
                )?)?;
            slots = (&slots
                + layer.down.forward(
                    &layer
                        .up
                        .forward(&layer.ffn_norm.forward(&slots)?)?
                        .gelu_erf()?,
                )?)?;
        }
        self.output.forward(&self.norm.forward(&slots)?)
    }
}
