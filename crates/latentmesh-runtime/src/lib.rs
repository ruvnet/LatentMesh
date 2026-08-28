//! latentmesh-runtime — live Qwen2 runtime for the latent-exchange
//! experiment (design doc `docs/research/024-live-latent-experiment-design.md`
//! §2–§3).
//!
//! Public surface: [`QwenRuntime::load`] (hf-hub BF16 safetensors),
//! [`QwenRuntime::generate`], [`capture::forward_capture`],
//! [`inject::prefill_with_injection`], [`sampler`], [`norms`], and the
//! build-env guard [`assert_cuda_build_env`].
//!
//! Evidence honesty: this crate performs live single-host GPU inference —
//! no simulation. Everything measured through it must be labelled
//! "live-model, single-host, simulation-free" in receipts (ADR-014/018).
//! Probe receipts (S0/S1a examples) are written under
//! `target/latentmesh-runs/` — inside the gitignored `target/` tree by
//! design, so run artifacts never enter the source tree.

pub mod capture;
pub mod inject;
mod models;
pub mod norms;
pub mod sampler;

pub use models::{Config, LayerEdit, Model, ModelForCausalLM};

use candle_core::{DType, Device, Error, Result, Tensor};
use candle_nn::VarBuilder;
use tokenizers::Tokenizer;

/// nvcc release the crate was compiled against (`"none"` for CPU builds).
/// Set by build.rs, which refuses to compile the `cuda` feature against
/// anything but CUDA 12.8 (the only toolchain verified with candle 0.9.2 on
/// this host's RTX 5080; design §2, risk #4).
pub const NVCC_RELEASE_AT_BUILD: &str = env!("LATENTMESH_NVCC_RELEASE");

/// Runtime half of the build-env guard: re-asserts at process start that the
/// `nvcc` on PATH is still the verified 12.8 toolchain, and returns the
/// string to record in receipts. Errors when the crate was built without the
/// `cuda` feature (live runs must not silently fall back to CPU) or when the
/// toolchain drifted after the build (cargo may reuse cached candle-kernels
/// artifacts, so the compile-time check alone is not sufficient).
pub fn assert_cuda_build_env() -> Result<String> {
    if NVCC_RELEASE_AT_BUILD == "none" {
        return Err(Error::Msg(
            "latentmesh-runtime built without the `cuda` feature; live runs require \
             `--features cuda` under PATH=/usr/local/cuda-12.8/bin:$PATH"
                .to_string(),
        ));
    }
    let out = std::process::Command::new("nvcc")
        .arg("--version")
        .output()
        .map_err(|e| Error::Msg(format!("nvcc not runnable at harness start: {e}")))?;
    let text = String::from_utf8_lossy(&out.stdout).to_string();
    let release = text
        .lines()
        .find_map(|l| l.split("release ").nth(1))
        .map(|r| r.split([',', ' ']).next().unwrap_or("").to_string())
        .unwrap_or_default();
    if !release.starts_with("12.8") {
        return Err(Error::Msg(format!(
            "nvcc on PATH reports release {release:?}; only 12.8 is verified (built \
             against {NVCC_RELEASE_AT_BUILD})"
        )));
    }
    Ok(format!(
        "nvcc {release} (build-time {NVCC_RELEASE_AT_BUILD})"
    ))
}

/// The device live runs use. With the `cuda` feature: CUDA:0; otherwise CPU
/// (for CPU-only unit tests — never for measured runs).
pub fn device() -> Result<Device> {
    #[cfg(feature = "cuda")]
    {
        Device::new_cuda(0)
    }
    #[cfg(not(feature = "cuda"))]
    {
        Ok(Device::Cpu)
    }
}

/// A loaded Qwen2.5 chat model plus its tokenizer.
pub struct QwenRuntime {
    pub model: ModelForCausalLM,
    pub tokenizer: Tokenizer,
    pub config: Config,
    pub device: Device,
    /// hf-hub repo id, recorded in receipts.
    pub model_id: String,
}

impl QwenRuntime {
    /// Download (into the default `~/.cache/huggingface` hf-hub cache) and
    /// load a BF16 safetensors Qwen2 checkpoint. Handles both single-file
    /// (`model.safetensors`) and sharded (`model.safetensors.index.json`)
    /// repos.
    pub fn load(model_id: &str, device: &Device, dtype: DType) -> Result<Self> {
        let files = fetch_model_files(model_id)?;
        let config: Config =
            serde_json::from_str(&std::fs::read_to_string(&files.config).map_err(Error::wrap)?)
                .map_err(Error::wrap)?;
        let tokenizer = Tokenizer::from_file(&files.tokenizer)
            .map_err(|e| Error::Msg(format!("tokenizer load: {e}")))?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files.weights, dtype, device)? };
        let model = ModelForCausalLM::new(&config, vb)?;
        Ok(Self {
            model,
            tokenizer,
            config,
            device: device.clone(),
            model_id: model_id.to_string(),
        })
    }

    /// Qwen2.5 chat-template prompt (ChatML), ending at the assistant turn
    /// opener so generation continues as the assistant.
    pub fn chat_prompt(system: &str, user: &str) -> String {
        format!(
            "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
        )
    }

    /// Encode text to token ids (no additional special tokens).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self
            .tokenizer
            .encode(text, false)
            .map_err(|e| Error::Msg(format!("encode: {e}")))?
            .get_ids()
            .to_vec())
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(tokens, true)
            .map_err(|e| Error::Msg(format!("decode: {e}")))
    }

    /// Token ids that end a chat turn: `<|im_end|>` and `<|endoftext|>`.
    pub fn eos_ids(&self) -> Vec<u32> {
        ["<|im_end|>", "<|endoftext|>"]
            .iter()
            .filter_map(|t| self.tokenizer.token_to_id(t))
            .collect()
    }

    /// Positions of `placeholder` token ids within `tokens`.
    pub fn placeholder_positions(tokens: &[u32], placeholder: u32) -> Vec<usize> {
        tokens
            .iter()
            .enumerate()
            .filter_map(|(i, &t)| (t == placeholder).then_some(i))
            .collect()
    }

    /// Greedy/sampled generation with an optional prefill-time injection.
    ///
    /// Clears the KV cache, prefills `prompt_tokens` (applying `inject` if
    /// given), then decodes up to `max_new_tokens` or EOS. Returns the
    /// generated token ids (EOS excluded) and, when `hash_logits` is set,
    /// the SHA-256 of each step's logits (greedy witness receipts).
    pub fn generate(
        &mut self,
        prompt_tokens: &[u32],
        inject_spec: Option<&inject::InjectionSpec>,
        sampler: &mut sampler::Sampler,
        max_new_tokens: usize,
        hash_logits: bool,
    ) -> Result<GenerateOutput> {
        let eos = self.eos_ids();
        let mut logits = inject::prefill_with_injection(
            &mut self.model,
            prompt_tokens,
            inject_spec,
            &self.device,
        )?;
        let mut tokens: Vec<u32> = Vec::new();
        let mut hashes: Vec<String> = Vec::new();
        for offset in (prompt_tokens.len()..).take(max_new_tokens) {
            if hash_logits {
                hashes.push(sampler::hash_logits(&logits)?);
            }
            let next = sampler.sample(&logits)?;
            if eos.contains(&next) {
                break;
            }
            tokens.push(next);
            let input = Tensor::new(&[next], &self.device)?.unsqueeze(0)?;
            logits = self.model.forward(&input, offset)?;
        }
        let text = self.decode(&tokens)?;
        Ok(GenerateOutput {
            tokens,
            text,
            logits_hashes: hash_logits.then_some(hashes),
        })
    }
}

/// Output of one generation episode.
#[derive(Debug, Clone)]
pub struct GenerateOutput {
    pub tokens: Vec<u32>,
    pub text: String,
    pub logits_hashes: Option<Vec<String>>,
}

struct ModelFiles {
    config: std::path::PathBuf,
    tokenizer: std::path::PathBuf,
    weights: Vec<std::path::PathBuf>,
}

fn fetch_model_files(model_id: &str) -> Result<ModelFiles> {
    let api = hf_hub::api::sync::Api::new().map_err(Error::wrap)?;
    let repo = api.model(model_id.to_string());
    let config = repo.get("config.json").map_err(Error::wrap)?;
    let tokenizer = repo.get("tokenizer.json").map_err(Error::wrap)?;
    let weights = match repo.get("model.safetensors.index.json") {
        Ok(index_path) => {
            let index: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(&index_path).map_err(Error::wrap)?)
                    .map_err(Error::wrap)?;
            let map = index["weight_map"]
                .as_object()
                .ok_or_else(|| Error::Msg("index.json without weight_map".to_string()))?;
            let mut shards: Vec<String> = map
                .values()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect();
            shards.sort();
            shards.dedup();
            shards
                .iter()
                .map(|s| repo.get(s).map_err(Error::wrap))
                .collect::<Result<Vec<_>>>()?
        }
        Err(_) => vec![repo.get("model.safetensors").map_err(Error::wrap)?],
    };
    Ok(ModelFiles {
        config,
        tokenizer,
        weights,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_prompt_shape() {
        let p = QwenRuntime::chat_prompt("sys", "user");
        assert!(p.starts_with("<|im_start|>system\nsys<|im_end|>"));
        assert!(p.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn placeholder_scan() {
        let toks = [1u32, 7, 7, 3, 7];
        assert_eq!(QwenRuntime::placeholder_positions(&toks, 7), vec![1, 2, 4]);
    }

    #[test]
    fn cpu_build_guard_refuses_live_runs() {
        if NVCC_RELEASE_AT_BUILD == "none" {
            assert!(assert_cuda_build_env().is_err());
        }
    }
}
