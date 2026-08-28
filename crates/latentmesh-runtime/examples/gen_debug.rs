//! Diagnostic (not a design-doc stage): decode-path verification after the
//! S1a run-1 quality anomaly (first-pass GSM8K accuracy 5/40, duplicated
//! tokens in outputs).
//!
//! Checks, on live GPU:
//!   1. KV parity: greedy stepwise decode vs one teacher-forced full prefill
//!      over prompt+generation — the argmax at every position must agree
//!      (bitwise logit equality is NOT expected across batch shapes in BF16;
//!      argmax agreement is the correctness signal for greedy decode).
//!   2. Full-text inspection of two prompts (one trivial, one GSM8K-style).
//!
//! Run: PATH=/usr/local/cuda-12.8/bin:$PATH \
//!      cargo run --release --features cuda --example gen_debug

#[path = "common/mod.rs"]
#[allow(dead_code)]
mod common;

use candle_core::{DType, Tensor};
use latentmesh_runtime::sampler::{Sampler, Sampling};
use latentmesh_runtime::QwenRuntime;

const MODEL: &str = "Qwen/Qwen2.5-1.5B-Instruct";
const SYSTEM: &str = "You are a careful math tutor.";

fn main() -> anyhow::Result<()> {
    let nvcc = latentmesh_runtime::assert_cuda_build_env().map_err(anyhow::Error::msg)?;
    println!("build-env guard: {nvcc}");
    let device = latentmesh_runtime::device().map_err(anyhow::Error::msg)?;
    let mut rt = QwenRuntime::load(
        MODEL,
        &device,
        if std::env::var("LM_F32").is_ok() {
            DType::F32
        } else {
            DType::BF16
        },
    )
    .map_err(anyhow::Error::msg)?;

    // --- 1) KV parity: stepwise greedy vs teacher-forced prefill ----------
    let prompt = QwenRuntime::chat_prompt(
        SYSTEM,
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\n\nThink step by step, then give the final answer on the last line as '#### <number>'.",
    );
    let ptoks = rt.encode(&prompt).map_err(anyhow::Error::msg)?;
    let mut greedy = Sampler::new(Sampling::Greedy, 0);
    let gen = rt
        .generate(&ptoks, None, &mut greedy, 64, false)
        .map_err(anyhow::Error::msg)?;
    println!("stepwise 64-token prefix: {:?}", gen.tokens);

    let mut full = ptoks.clone();
    full.extend_from_slice(&gen.tokens);
    rt.model.clear_kv_cache();
    let input = Tensor::new(full.as_slice(), &device)
        .and_then(|t| t.unsqueeze(0))
        .map_err(anyhow::Error::msg)?;
    let logits = rt
        .model
        .forward_full_logits(&input, 0, None)
        .and_then(|l| l.to_dtype(DType::F32))
        .map_err(anyhow::Error::msg)?;
    let mut mismatches = 0usize;
    for (i, &tok) in gen.tokens.iter().enumerate() {
        // Position ptoks.len()-1+i predicts generated token i.
        let row = logits
            .get(0)
            .and_then(|l| l.get(ptoks.len() - 1 + i))
            .and_then(|r| r.to_vec1::<f32>())
            .map_err(anyhow::Error::msg)?;
        let argmax = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(j, _)| j as u32)
            .unwrap_or(u32::MAX);
        if argmax != tok {
            mismatches += 1;
            println!("  MISMATCH at gen pos {i}: stepwise {tok} vs prefill argmax {argmax}");
        }
    }
    println!(
        "KV parity: {} / {} positions agree (stepwise greedy vs teacher-forced prefill argmax)",
        gen.tokens.len() - mismatches,
        gen.tokens.len()
    );

    // --- 2) Full-text inspection ------------------------------------------
    for (tag, user, cap) in [
        ("trivial", "What is 2+2? Reply with just the number.", 16),
        (
            "gsm8k-style",
            "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\n\nSolve the problem step by step. Then write the final line in exactly this format:\n#### <numeric answer>\nFor example, if the answer were 5, the last line must be: #### 5",
            512,
        ),
    ] {
        let p = QwenRuntime::chat_prompt(SYSTEM, user);
        let t = rt.encode(&p).map_err(anyhow::Error::msg)?;
        let mut s = Sampler::new(Sampling::Greedy, 0);
        let out = rt
            .generate(&t, None, &mut s, cap, false)
            .map_err(anyhow::Error::msg)?;
        println!(
            "--- {tag} ({} tokens) ---\n{}\n--- end (extracted: {:?}) ---",
            out.tokens.len(),
            out.text,
            common::extract_final_answer(&out.text)
        );
    }
    Ok(())
}
