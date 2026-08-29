//! Vendored Qwen2 model (candle-transformers 0.9.2), split across two files
//! for the repo's <500-line rule (design doc 024 §3).
//!
//! Provenance: `candle-transformers` v0.9.2 (crates.io), `src/models/qwen2.rs`
//! (402 lines) plus the two internal helpers it depends on
//! (`src/models/with_tracing.rs` wrappers, `src/utils.rs::repeat_kv`).
//! Modifications from verbatim are listed at the top of each file.

mod qwen2_a;
mod qwen2_b;

pub use qwen2_a::Config;
pub use qwen2_b::{LayerEdit, Model, ModelForCausalLM};
