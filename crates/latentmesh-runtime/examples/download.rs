//! Prefetch both run-1 checkpoints into the default hf-hub cache
//! (`~/.cache/huggingface`), so the GPU probes start without network I/O.
//! CPU-only; no CUDA feature required.
//!
//! Usage: `cargo run -p latentmesh-runtime --example download`

use hf_hub::api::sync::Api;

const MODELS: &[(&str, &[&str])] = &[
    (
        "Qwen/Qwen2.5-3B-Instruct",
        &[
            "config.json",
            "tokenizer.json",
            "model.safetensors.index.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ],
    ),
    (
        "Qwen/Qwen2.5-1.5B-Instruct",
        &["config.json", "tokenizer.json", "model.safetensors"],
    ),
];

fn main() -> anyhow::Result<()> {
    let api = Api::new()?;
    for (model_id, files) in MODELS {
        let repo = api.model(model_id.to_string());
        for file in *files {
            let path = repo.get(file)?;
            let size = std::fs::metadata(&path)?.len();
            println!("{model_id}/{file}: {size} bytes at {}", path.display());
        }
    }
    println!("all model files present in hf-hub cache");
    Ok(())
}
