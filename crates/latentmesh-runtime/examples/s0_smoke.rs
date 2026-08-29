//! S0 — runtime spike + mechanics gates (design doc 024 §7 S0).
//!
//! On 3 GSM8K test items (raw JSONL, sha256-verified), against live BF16
//! CUDA-resident Qwen2.5-3B-Instruct (sender) and Qwen2.5-1.5B-Instruct
//! (receiver), this probe checks every S0 gate:
//!   G1 build green under pinned nvcc 12.8 (build.rs + runtime guard)
//!   G2 sender capture shape = 2048 at L24/36
//!   G3 forward_capture logits BIT-IDENTICAL to unpatched forward (both models)
//!   G4 injected logits finite and different from baseline (receiver L19,
//!      receiver-native 1536-dim vector — no cross-model transform exists at
//!      S0; alignment is S2 scope)
//!   G5 zero-slot injection == no-injection baseline (bit-identical BY
//!      CONSTRUCTION: empty position list short-circuits; measured anyway)
//!   G6 injected-vector norm within ~3x of the natural L19 per-position
//!      norm distribution (rescale switch ON, recorded)
//!   G7 JSON receipt emitted (evidence: live GPU, deterministic — no
//!      sampling occurs in S0)
//!
//! Receipts: `crates/latentmesh-runtime/target/latentmesh-runs/s0/`
//! (gitignored by being under target/).
//!
//! Run: PATH=/usr/local/cuda-12.8/bin:$PATH \
//!      cargo run --release --features cuda --example s0_smoke

#[path = "common/mod.rs"]
#[allow(dead_code)]
mod common;

use latentmesh_runtime::{
    capture::{forward_capture, forward_unpatched, logits_bit_identical},
    inject::{prefill_with_injection, InjectionMode, InjectionSpec},
    norms, QwenRuntime,
};

const SENDER: &str = "Qwen/Qwen2.5-3B-Instruct";
const RECEIVER: &str = "Qwen/Qwen2.5-1.5B-Instruct";
const SENDER_CAPTURE_BLOCK: usize = 24; // L24/36
const RECEIVER_INJECT_BLOCK: usize = 19; // L19/28
const N_SLOTS: usize = 8;
const NORM_BAND_FACTOR: f32 = 3.0;
const SYSTEM: &str = "You are a careful math tutor.";

fn main() -> anyhow::Result<()> {
    let t0 = std::time::Instant::now();
    // G1 (runtime half): the toolchain guard must pass before any GPU work.
    let nvcc = latentmesh_runtime::assert_cuda_build_env().map_err(anyhow::Error::msg)?;
    println!("build-env guard: {nvcc}");

    let dir = common::run_dir("s0");
    let data = dir.join("gsm8k-test.jsonl");
    let sha = common::fetch(common::GSM8K_TEST_URL, &data)?;
    anyhow::ensure!(
        sha == common::GSM8K_TEST_SHA256,
        "gsm8k test.jsonl sha256 {sha} != pinned {}",
        common::GSM8K_TEST_SHA256
    );
    let items = common::load_gsm8k(&data)?;
    let items = &items[..3];
    println!("gsm8k test.jsonl verified ({sha}), using items 0..3");

    let device = latentmesh_runtime::device().map_err(anyhow::Error::msg)?;
    println!("loading {SENDER} + {RECEIVER} (BF16, concurrent-resident)...");
    let mut sender =
        QwenRuntime::load(SENDER, &device, candle_core::DType::BF16).map_err(anyhow::Error::msg)?;
    let mut receiver = QwenRuntime::load(RECEIVER, &device, candle_core::DType::BF16)
        .map_err(anyhow::Error::msg)?;
    println!("models loaded in {:.1}s", t0.elapsed().as_secs_f32());

    let pad_id = receiver
        .tokenizer
        .token_to_id("<|fim_pad|>")
        .ok_or_else(|| anyhow::anyhow!("<|fim_pad|> not in tokenizer"))?;

    let mut rows = Vec::new();
    let mut all_pass = true;
    for item in items {
        let row = run_item(&mut sender, &mut receiver, item, pad_id, &device)?;
        all_pass &= row["gates"]
            .as_object()
            .unwrap()
            .values()
            .all(|v| v["pass"].as_bool().unwrap_or(false));
        println!(
            "item {}: {}",
            item.index,
            serde_json::to_string(&row["gates"])?
        );
        rows.push(row);
    }

    let receipt = serde_json::json!({
        "stage": "S0",
        "design": "docs/research/024-live-latent-experiment-design.md section 7 S0",
        "env": common::env_info(&nvcc),
        "decoding": "none (S0 is prefill-only mechanics; deterministic, no sampling)",
        "dataset": {
            "source": common::GSM8K_TEST_URL,
            "sha256": sha,
            "items": items.iter().map(|i| i.index).collect::<Vec<_>>(),
        },
        "config": {
            "sender": SENDER, "receiver": RECEIVER,
            "sender_capture_block": SENDER_CAPTURE_BLOCK,
            "receiver_inject_block": RECEIVER_INJECT_BLOCK,
            "n_slots": N_SLOTS, "norm_band_factor": NORM_BAND_FACTOR,
            "placeholder_token": "<|fim_pad|>", "placeholder_id": pad_id,
            "injection_vector": "receiver-native pooled own-L19 state, rescaled to natural median norm (no cross-model transform exists at S0; alignment is S2 scope)",
        },
        "items": rows,
        "all_gates_pass": all_pass,
        "wall_clock_s": t0.elapsed().as_secs_f32(),
    });
    common::write_receipt(&dir, "s0-receipt.json", &receipt)?;
    println!("S0 all gates pass: {all_pass}");
    anyhow::ensure!(all_pass, "S0 gate failure — see receipt");
    Ok(())
}

fn run_item(
    sender: &mut QwenRuntime,
    receiver: &mut QwenRuntime,
    item: &common::Gsm8kItem,
    pad_id: u32,
    device: &candle_core::Device,
) -> anyhow::Result<serde_json::Value> {
    let e = anyhow::Error::msg;
    let fmt = common::ANSWER_FORMAT;

    // --- Sender side: capture shape + logits parity at L24 -----------------
    let s_prompt = QwenRuntime::chat_prompt(SYSTEM, &format!("{}\n\n{fmt}", item.question));
    let s_tokens = sender.encode(&s_prompt).map_err(e)?;
    let span = 0..s_tokens.len();
    let (cap_logits, s_cap) = forward_capture(
        &mut sender.model,
        &s_tokens,
        SENDER_CAPTURE_BLOCK,
        span,
        device,
    )
    .map_err(e)?;
    let s_ref = forward_unpatched(&mut sender.model, &s_tokens, device).map_err(e)?;
    let sender_parity = logits_bit_identical(&cap_logits, &s_ref).map_err(e)?;
    let shape_ok = s_cap.hidden_size == 2048;

    // --- Receiver side: natural L19 stats + parity on the slotted prompt --
    let slots = "<|fim_pad|>".repeat(N_SLOTS);
    let r_prompt = QwenRuntime::chat_prompt(
        SYSTEM,
        &format!(
            "{}\n\nA compressed latent hint from another solver is stored in these slots: [{slots}]\n\n{fmt}",
            item.question
        ),
    );
    let r_tokens = receiver.encode(&r_prompt).map_err(e)?;
    let positions = QwenRuntime::placeholder_positions(&r_tokens, pad_id);
    anyhow::ensure!(
        positions.len() == N_SLOTS,
        "expected {N_SLOTS} placeholder slots, found {}",
        positions.len()
    );
    let (r_cap_logits, r_cap) = forward_capture(
        &mut receiver.model,
        &r_tokens,
        RECEIVER_INJECT_BLOCK,
        0..r_tokens.len(),
        device,
    )
    .map_err(e)?;
    let baseline = forward_unpatched(&mut receiver.model, &r_tokens, device).map_err(e)?;
    let receiver_parity = logits_bit_identical(&r_cap_logits, &baseline).map_err(e)?;
    let natural = norms::stats(r_cap.per_position_l2.clone());

    // --- G4: injected logits finite and != baseline ------------------------
    let pooled_l2 = norms::l2(&r_cap.pooled);
    let spec = InjectionSpec {
        after_block: RECEIVER_INJECT_BLOCK,
        positions: positions.clone(),
        vector: r_cap.pooled.clone(),
        scale: Some(natural.median / pooled_l2),
        mode: InjectionMode::Overwrite,
    };
    let injected =
        prefill_with_injection(&mut receiver.model, &r_tokens, Some(&spec), device).map_err(e)?;
    let inj_vec = injected
        .to_dtype(candle_core::DType::F32)
        .map_err(e)?
        .flatten_all()
        .map_err(e)?
        .to_vec1::<f32>()
        .map_err(e)?;
    let injected_finite = inj_vec.iter().all(|x| x.is_finite());
    let injected_differs = !logits_bit_identical(&injected, &baseline).map_err(e)?;

    // --- G5: zero-slot injection == baseline --------------------------------
    let zero_spec = InjectionSpec {
        positions: Vec::new(),
        ..spec.clone()
    };
    let zero = prefill_with_injection(&mut receiver.model, &r_tokens, Some(&zero_spec), device)
        .map_err(e)?;
    let zero_identical = logits_bit_identical(&zero, &baseline).map_err(e)?;

    // --- G6: norm band -------------------------------------------------------
    let injected_l2 = norms::l2(&spec.effective_vector());
    let norm_in_band = norms::within_band(injected_l2, natural.median, NORM_BAND_FACTOR);

    Ok(serde_json::json!({
        "item": item.index,
        "sender_capture": {
            "hidden_size": s_cap.hidden_size,
            "after_block": s_cap.after_block,
            "span_tokens": s_cap.span.end - s_cap.span.start,
            "pooled_l2": norms::l2(&s_cap.pooled),
            "natural_norms": norms::stats(s_cap.per_position_l2.clone()),
        },
        "receiver": {
            "prompt_tokens": r_tokens.len(),
            "slot_positions": positions,
            "natural_l19_norms": natural,
            "pooled_own_l19_l2_raw": pooled_l2,
            "injected_vector_l2": injected_l2,
        },
        "gates": {
            "G2_capture_shape_2048": {"pass": shape_ok, "measured": s_cap.hidden_size},
            "G3_sender_logits_parity_bit_identical": {"pass": sender_parity},
            "G3_receiver_logits_parity_bit_identical": {"pass": receiver_parity},
            "G4_injected_logits_finite": {"pass": injected_finite},
            "G4_injected_logits_differ_from_baseline": {"pass": injected_differs},
            "G5_zero_slot_equals_baseline": {
                "pass": zero_identical,
                "note": "bit-identical BY CONSTRUCTION (empty position list is a no-op branch); measured equality confirms it",
            },
            "G6_injected_norm_within_band": {
                "pass": norm_in_band,
                "injected_l2": injected_l2,
                "natural_median": natural.median,
                "factor": NORM_BAND_FACTOR,
            },
        },
    }))
}
