//! Local Qwen3 text encoder, Memorilla memory, and input embedding generation.
use crate::{qwen3, DType, Device, MemoryModule, MemoryService, Result, Tensor};
use candle_core::{IndexOp, Module};
use candle_nn::{Linear, VarBuilder};
use std::path::Path;
use tokenizers::{AddedToken, Tokenizer};

pub struct QwenBackend {
    pub config: qwen3::Config,
    model: qwen3::Model,
    head: Option<Linear>,
    tokenizer: Tokenizer,
    device: Device,
    dtype: DType,
}
impl QwenBackend {
    pub fn load(directory: &Path, encoder: bool, device: &Device, dtype: DType) -> Result<Self> {
        let config: qwen3::Config =
            serde_json::from_slice(&std::fs::read(directory.join("config.json"))?)
                .map_err(candle_core::Error::wrap)?;
        if config.hidden_size == 0
            || config.num_attention_heads == 0
            || config.num_key_value_heads == 0
            || !config
                .num_attention_heads
                .is_multiple_of(config.num_key_value_heads)
            || config.head_dim == 0
            || !config.head_dim.is_multiple_of(2)
            || config.max_position_embeddings == 0
            || config.use_sliding_window
        {
            candle_core::bail!("invalid or unsupported Qwen3 configuration");
        }
        let mut files = Vec::new();
        if directory.join("model.safetensors").exists() {
            files.push(directory.join("model.safetensors"));
        } else {
            let index: serde_json::Value = serde_json::from_slice(&std::fs::read(
                directory.join("model.safetensors.index.json"),
            )?)
            .map_err(candle_core::Error::wrap)?;
            let map = index["weight_map"]
                .as_object()
                .ok_or_else(|| candle_core::Error::Msg("invalid shard index".into()))?;
            for name in map.values() {
                let name = name
                    .as_str()
                    .ok_or_else(|| candle_core::Error::Msg("invalid shard filename".into()))?;
                if Path::new(name).components().count() != 1 || !name.ends_with(".safetensors") {
                    candle_core::bail!("invalid shard path");
                }
                files.push(directory.join(name));
            }
            files.sort();
            files.dedup();
        }
        // Model files must remain immutable for the lifetime of the mapping.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, dtype, device)? };
        let vb = if vb.contains_tensor("model.embed_tokens.weight") {
            vb
        } else {
            vb.rename_f(|name| name.strip_prefix("model.").unwrap_or(name).to_owned())
        };
        let model = qwen3::Model::new(&config, vb.clone())?;
        let head = if encoder {
            None
        } else if config.tie_word_embeddings {
            Some(Linear::new(
                vb.get(
                    (config.vocab_size, config.hidden_size),
                    "model.embed_tokens.weight",
                )?,
                None,
            ))
        } else {
            Some(candle_nn::linear_no_bias(
                config.hidden_size,
                config.vocab_size,
                vb.pp("lm_head"),
            )?)
        };
        let mut tokenizer = Tokenizer::from_file(directory.join("tokenizer.json"))
            .map_err(candle_core::Error::wrap)?;
        tokenizer.with_padding(None);
        tokenizer
            .with_truncation(None)
            .map_err(candle_core::Error::wrap)?;
        if !encoder {
            tokenizer.add_special_tokens(&[AddedToken::from("<|memory|>", true)]);
        }
        Ok(Self {
            config,
            model,
            head,
            tokenizer,
            device: device.clone(),
            dtype,
        })
    }
    fn tokens(&self, text: &str) -> Result<Vec<u32>> {
        let ids = self
            .tokenizer
            .encode(text, true)
            .map_err(candle_core::Error::wrap)?
            .get_ids()
            .to_vec();
        if ids.is_empty() || ids.len() > self.config.max_position_embeddings {
            candle_core::bail!("empty text or context limit exceeded");
        }
        Ok(ids)
    }
    pub fn encode(&mut self, text: &str) -> Result<Vec<f32>> {
        let ids = self.tokens(text)?;
        self.model.clear_kv_cache();
        let result = (|| {
            let input = Tensor::from_slice(&ids, (1, ids.len()), &self.device)?;
            let hidden = self
                .model
                .forward(&input, 0)?
                .i((0, ids.len() - 1))?
                .to_dtype(DType::F32)?;
            let norm = hidden.sqr()?.sum_all()?.sqrt()?;
            // Upstream embed_texts stores normalized pooled vectors as float16.
            hidden
                .broadcast_div(&norm)?
                .to_dtype(DType::F16)?
                .to_dtype(DType::F32)?
                .to_vec1()
        })();
        self.model.clear_kv_cache();
        result
    }
    pub fn generate(
        &mut self,
        question: &str,
        system: &str,
        memory: &Tensor,
        max_tokens: usize,
    ) -> Result<String> {
        if max_tokens == 0 || max_tokens > 4096 {
            candle_core::bail!("max tokens must be in 1..=4096");
        }
        let (batch, slots, hidden) = memory.dims3()?;
        if batch != 1 || hidden != self.config.hidden_size {
            candle_core::bail!("decoder memory shape mismatch");
        }
        let memory_id = self
            .tokenizer
            .token_to_id("<|memory|>")
            .ok_or_else(|| candle_core::Error::Msg("missing memory token".into()))?;
        let placeholders = "<|memory|>".repeat(slots);
        let prompt=format!("<|im_start|>system\n{system}\nMemory{placeholders}Memory<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",question.trim());
        let ids = self.tokens(&prompt)?;
        if ids.len().saturating_add(max_tokens) > self.config.max_position_embeddings {
            candle_core::bail!("prompt plus generation exceeds context limit");
        }
        let positions: Vec<_> = ids
            .iter()
            .enumerate()
            .filter_map(|(i, id)| (*id == memory_id).then_some(i))
            .collect();
        if positions.len() != slots {
            candle_core::bail!("reserved memory placeholder in user or system text");
        }
        // Added placeholders may be outside the model vocabulary. Substitute a
        // valid id before embedding and replace exactly those positions below.
        let safe_ids: Vec<_> = ids
            .iter()
            .map(|id| if *id == memory_id { 0 } else { *id })
            .collect();
        let input = Tensor::from_slice(&safe_ids, (1, ids.len()), &self.device)?;
        let embeddings = self.model.embeddings(&input)?;
        let memory = memory.to_device(&self.device)?.to_dtype(self.dtype)?;
        let mut pieces = Vec::new();
        let mut start = 0;
        for (slot, position) in positions.iter().enumerate() {
            if *position > start {
                pieces.push(embeddings.narrow(1, start, position - start)?);
            }
            pieces.push(memory.narrow(1, slot, 1)?);
            start = position + 1;
        }
        if start < ids.len() {
            pieces.push(embeddings.narrow(1, start, ids.len() - start)?);
        }
        let embeddings = Tensor::cat(&pieces, 1)?;
        self.model.clear_kv_cache();
        let result = (|| {
            let head = self.head.as_ref().ok_or_else(|| {
                candle_core::Error::Msg("encoder has no language model head".into())
            })?;
            let mut h = self.model.forward_embeds(&embeddings, 0)?;
            let mut generated = Vec::new();
            for step in 0..max_tokens {
                let last = h.dim(1)? - 1;
                let logits = head.forward(&h.narrow(1, last, 1)?)?;
                let token = logits.flatten_all()?.argmax(0)?.to_scalar::<u32>()?;
                if [
                    self.tokenizer.token_to_id("<|im_end|>"),
                    self.tokenizer.token_to_id("<|endoftext|>"),
                ]
                .contains(&Some(token))
                {
                    break;
                }
                generated.push(token);
                if step + 1 < max_tokens {
                    h = self
                        .model
                        .forward(&Tensor::new(&[[token]], &self.device)?, ids.len() + step)?;
                }
            }
            self.tokenizer
                .decode(&generated, true)
                .map_err(candle_core::Error::wrap)
        })();
        self.model.clear_kv_cache();
        result
    }
}

pub struct MemorillaAgent {
    pub memory: MemoryService,
    encoder: QwenBackend,
    decoder: QwenBackend,
}
impl MemorillaAgent {
    pub fn load(
        memory_dir: &Path,
        encoder_dir: &Path,
        decoder_dir: &Path,
        scope: &str,
        budget: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let model = MemoryModule::load(memory_dir, device, dtype)?;
        let encoder = QwenBackend::load(encoder_dir, true, device, dtype)?;
        let decoder = QwenBackend::load(decoder_dir, false, device, dtype)?;
        if encoder.config.hidden_size != model.config().embedding_dim
            || decoder.config.hidden_size != model.config().output_dim
        {
            candle_core::bail!("encoder, memory and decoder dimensions disagree");
        }
        Ok(Self {
            memory: MemoryService::new(model, scope, budget)?,
            encoder,
            decoder,
        })
    }
    pub fn write(&mut self, id: &str, documents: &[String]) -> Result<usize> {
        if documents.is_empty() {
            candle_core::bail!("documents cannot be empty");
        }
        let mut embeddings = Vec::new();
        for document in documents {
            embeddings.extend(self.encoder.encode(document)?);
        }
        self.memory.write(id, &embeddings)
    }
    pub fn answer(
        &mut self,
        id: &str,
        question: &str,
        system: &str,
        max_tokens: usize,
    ) -> Result<String> {
        let query = self.encoder.encode(question)?;
        let result = self.memory.read(id, &query)?;
        let tokens = Tensor::from_vec(
            result.values,
            result.shape.as_slice(),
            self.memory.model.device(),
        )?;
        self.decoder.generate(question, system, &tokens, max_tokens)
    }
}
