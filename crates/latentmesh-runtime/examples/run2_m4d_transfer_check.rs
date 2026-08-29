//! Run-2 M4d — registered transfer check (ADR-024 M4d caveat mitigation),
//! run BETWEEN training and the frozen probe, gating the probe.
//!
//! The M4d adapter was trained through the COMPOSED differentiable BF16
//! forward (`latentmesh-train::qwen2_c`); the frozen probe runs the vendored
//! FUSED BF16 forward. The measured numeric gap between the two (116/128
//! argmax agreement, max|dlogit| 8.19 at L=128 — pure BF16 rounding
//! amplification; F32 parity 128/128 proves same function) means a probe
//! null could be confounded unless the training improvement is first shown
//! to TRANSFER across the gap. This check does exactly that, inference-only,
//! with NO probe items, NO generation, NO frozen-probe draw:
//!
//! For every evaluable holdout item (same leakage-safe holdout side, same
//! seq-cap/min-target span rule as training — recomputed here and asserted
//! against the training receipt), compute the teacher-forced NLL of the
//! item's own sender-generated span through the VENDORED fused forward,
//! with the probe's exact delivery path (hand-rolled MLP per-token →
//! mean-pool → rescale to the vendored natural inject-block median → 8-slot
//! injection), once with the TRAINED adapter and once with the SEEDED-INIT
//! adapter (the training receipt's frozen baseline).
//!
//! This is byte-for-byte M4c's check with M4d's artifact paths substituted,
//! deliberately: the two rungs' `mean_fused_nll_{init,trained}` numbers are
//! only comparable if the measurement is literally the same one. Note that
//! for M4d this delivery path is now also the TRAINING path's configuration
//! (M4d change 1 sources the rescale target from this same vendored fused
//! `forward_capture`), so the check additionally reports whether that made
//! the composed→fused transfer tighter than M4c's.
//!
//! Frozen pass criterion (quoted from the training receipt, frozen there
//! BEFORE this check ran): mean fused NLL(trained) < mean fused NLL(init);
//! per-item wins/losses + sign test reported as secondary, not gating.
//! On fail: the frozen probe is NOT invoked (run2_m4d_probe refuses to run).
//!
//! Run: PATH=/usr/local/cuda-12.8/bin:$PATH \
//!      cargo run --release --features cuda --example run2_m4d_transfer_check

#[path = "common/mod.rs"]
#[allow(dead_code)]
mod common;

use common::m3::{RECEIVER, RECEIVER_BLOCK, SYSTEM};
use common::mlp::MlpTransform;
use latentmesh_runtime::{
    capture::forward_capture,
    inject::{teacher_forced_nll, InjectionMode, InjectionSpec},
    norms, QwenRuntime,
};
use sha2::Digest as _;
use std::io::{Read as _, Seek as _};
use std::path::{Path, PathBuf};

const TRAINING_RECEIPT: &str = "receipts/run2-m4d-training-receipt-cellL18toL14.json";
const DUMP_RECEIPT: &str = "receipts/run2-pertoken-dump-receipt.json";
const N_SLOTS: usize = 8;
const SENDER_DIM: usize = 2048;
/// Frozen span rule — MUST match `train_m4d_deploymatch.rs` (`SEQ_CAP`,
/// `MIN_TARGET`); the recomputed skip set is asserted against the training
/// receipt's `holdout_skipped`, so silent divergence fails loudly.
const SEQ_CAP: usize = 256;
const MIN_TARGET: usize = 8;
/// Streams pins, verbatim from `run2_pertoken_dump.rs`.
const STREAMS_SHA256: &str = "ad5713116cb5acf663fe9c33ac58aaedb1fd64fcf629dd78b21439d244958539";
const GSM8K_TRAIN_SHA256: &str = "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465";

fn crate_path(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

#[derive(serde::Deserialize)]
struct StreamRow {
    row: usize,
    item: usize,
    prompt_tokens: Vec<u32>,
    gen_tokens: Vec<u32>,
    #[allow(dead_code)]
    sender_first_pass_correct: bool,
}

/// Read one item's `[n_rows × SENDER_DIM]` block from the ragged dump file.
fn read_rows(f: &mut std::fs::File, tok0: u64, n_rows: usize) -> anyhow::Result<Vec<f32>> {
    f.seek(std::io::SeekFrom::Start(tok0 * SENDER_DIM as u64 * 4))?;
    let mut buf = vec![0u8; n_rows * SENDER_DIM * 4];
    f.read_exact(&mut buf)?;
    Ok(buf
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn main() -> anyhow::Result<()> {
    let t0 = std::time::Instant::now();
    let nvcc = latentmesh_runtime::assert_cuda_build_env().map_err(anyhow::Error::msg)?;
    println!("build-env guard: {nvcc}");
    let e = anyhow::Error::msg;

    // ---- Training receipt: hashes, holdout rows, frozen criterion ---------
    let tr: serde_json::Value =
        serde_json::from_slice(&std::fs::read(crate_path(TRAINING_RECEIPT))?)?;
    let trained_hash = tr["artifact"]["content_hash_sha256"]
        .as_str()
        .unwrap()
        .to_string();
    let init_hash = tr["artifact"]["init_content_hash_sha256"]
        .as_str()
        .unwrap()
        .to_string();
    let holdout_rows: Vec<usize> = tr["split"]["holdout_rows"]
        .as_array()
        .expect("training receipt split.holdout_rows")
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let receipt_skipped: Vec<usize> = tr["split"]["holdout_skipped"]
        .as_array()
        .expect("training receipt split.holdout_skipped")
        .iter()
        .map(|v| v["row"].as_u64().unwrap() as usize)
        .collect();
    let criterion = tr["registered_caveat_bf16_composed_vs_fused"]
        ["transfer_pass_criterion_frozen"]
        .as_str()
        .expect("frozen transfer criterion in training receipt")
        .to_string();
    println!("frozen criterion (from training receipt): {criterion}");

    // ---- Adapters: hash gates + golden verification (both) ----------------
    let trained = MlpTransform::load(&crate_path(
        "receipts/run2-m4d-mlp-deploymatch-cellL18toL14.f32bin",
    ))?;
    anyhow::ensure!(
        trained.content_hash == trained_hash,
        "trained artifact hash mismatch"
    );
    let (gn, gmax, _) = common::mlp::verify_against_golden(
        &trained,
        &crate_path("receipts/run2-m4d-golden-mlp-deploymatch-cellL18toL14.json"),
        common::m3::GOLDEN_REL_TOL,
    )?;
    let init = MlpTransform::load(&crate_path(
        "receipts/run2-m4d-mlp-deploymatch-init-cellL18toL14.f32bin",
    ))?;
    anyhow::ensure!(
        init.content_hash == init_hash,
        "init artifact hash mismatch"
    );
    let (gni, gmaxi, _) = common::mlp::verify_against_golden(
        &init,
        &crate_path("receipts/run2-m4d-golden-mlp-deploymatch-init-cellL18toL14.json"),
        common::m3::GOLDEN_REL_TOL,
    )?;
    println!("adapters verified: trained ({gn} goldens, max rel {gmax:.2e}), init ({gni} goldens, max rel {gmaxi:.2e})");

    // ---- Dump: index + sender bin sha-verified against the M2 receipt -----
    let dr: serde_json::Value = serde_json::from_slice(&std::fs::read(crate_path(DUMP_RECEIPT))?)?;
    let run_dir = PathBuf::from(dr["run_dir"].as_str().unwrap());
    let index_path = run_dir.join("run2-pertoken-index.json");
    let index_sha = format!("{:x}", sha2::Sha256::digest(std::fs::read(&index_path)?));
    anyhow::ensure!(
        index_sha == dr["index"]["sha256"].as_str().unwrap(),
        "index sha mismatch"
    );
    let index: serde_json::Value = serde_json::from_slice(&std::fs::read(&index_path)?)?;
    let token_offsets: Vec<u64> = index["token_offsets"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap())
        .collect();
    let sender_path = run_dir.join("sender_L18.tok.f32bin");
    {
        println!("hashing sender_L18.tok.f32bin (integrity gate)...");
        let mut f = std::fs::File::open(&sender_path)?;
        let mut hasher = sha2::Sha256::new();
        let mut buf = vec![0u8; 8 << 20];
        loop {
            let n = f.read(&mut buf)?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
        }
        let sha = format!("{:x}", hasher.finalize());
        let expect = dr["files"]["sender_L18.tok.f32bin"]["sha256"]
            .as_str()
            .unwrap();
        anyhow::ensure!(sha == expect, "sender bin sha {sha} != receipt {expect}");
        println!(
            "  sender bin sha256 OK ({:.0}s)",
            t0.elapsed().as_secs_f32()
        );
    }
    let mut sender_bin = std::fs::File::open(&sender_path)?;

    // ---- Streams (CE targets) + questions, sha-gated ----------------------
    let streams_path = crate_path("../../harness/latentmesh-live/data/s2c-token-streams.jsonl");
    let sbytes = std::fs::read(&streams_path)?;
    anyhow::ensure!(format!("{:x}", sha2::Sha256::digest(&sbytes)) == STREAMS_SHA256);
    let streams: Vec<StreamRow> = String::from_utf8(sbytes)?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).map_err(anyhow::Error::from))
        .collect::<anyhow::Result<_>>()?;
    for (i, r) in streams.iter().enumerate() {
        anyhow::ensure!(r.row == i, "stream row {i} out of order (row={})", r.row);
    }
    let gsm_path = crate_path("../../harness/latentmesh-live/data/gsm8k-train.jsonl");
    let gbytes = std::fs::read(&gsm_path)?;
    anyhow::ensure!(format!("{:x}", sha2::Sha256::digest(&gbytes)) == GSM8K_TRAIN_SHA256);
    let all_items = common::load_gsm8k(&gsm_path)?;

    // ---- Receiver only (the sender's work is already in the dump) ---------
    let device = latentmesh_runtime::device().map_err(e)?;
    println!("loading {RECEIVER} (BF16, vendored fused forward)...");
    let mut receiver = QwenRuntime::load(RECEIVER, &device, candle_core::DType::BF16).map_err(e)?;
    let pad_id = receiver
        .tokenizer
        .token_to_id("<|fim_pad|>")
        .ok_or_else(|| anyhow::anyhow!("<|fim_pad|> not in tokenizer"))?;

    // ---- Per-item fused-forward NLL, trained vs init ----------------------
    let fmt = common::ANSWER_FORMAT;
    let slots = "<|fim_pad|>".repeat(N_SLOTS);
    let mut rows_json = Vec::new();
    let mut skipped: Vec<usize> = Vec::new();
    let mut nll_trained_sum = 0f64;
    let mut nll_init_sum = 0f64;
    let (mut wins, mut losses) = (0usize, 0usize);
    let mut n_eval = 0usize;
    for (done, &row) in holdout_rows.iter().enumerate() {
        let sr = &streams[row];
        let q = &all_items[sr.item].question;
        // Prompt-parity gate vs the stream (same gate the trainer ran).
        let cap_prompt = QwenRuntime::chat_prompt(SYSTEM, &format!("{q}\n\n{fmt}"));
        anyhow::ensure!(
            receiver.encode(&cap_prompt).map_err(e)? == sr.prompt_tokens,
            "row {row}: prompt-parity gate failed"
        );
        let inj_prompt = QwenRuntime::chat_prompt(
            SYSTEM,
            &format!("{q}\n\nA compressed latent hint from a previous solver is stored in these slots: [{slots}]\n\n{fmt}"),
        );
        let inj_tokens = receiver.encode(&inj_prompt).map_err(e)?;
        let positions = QwenRuntime::placeholder_positions(&inj_tokens, pad_id);
        anyhow::ensure!(positions.len() == N_SLOTS, "slot count mismatch");
        let target_fit = SEQ_CAP
            .saturating_sub(inj_tokens.len())
            .min(sr.gen_tokens.len());
        if target_fit < MIN_TARGET {
            skipped.push(row);
            continue;
        }
        let nll_tokens: Vec<u32> = inj_tokens
            .iter()
            .chain(sr.gen_tokens[..target_fit].iter())
            .copied()
            .collect();
        let span = inj_tokens.len()..nll_tokens.len();
        // Vendored natural inject-block norms (probe's rescale source).
        let (_, nat_cap) = forward_capture(
            &mut receiver.model,
            &inj_tokens,
            RECEIVER_BLOCK,
            0..inj_tokens.len(),
            &device,
        )
        .map_err(e)?;
        let natural = norms::stats(nat_cap.per_position_l2.clone());
        let rows = read_rows(&mut sender_bin, token_offsets[row], sr.gen_tokens.len())?;
        let mut nll_for = |t: &MlpTransform| -> anyhow::Result<f32> {
            let pooled = t.apply_rows_then_pool(&rows, sr.gen_tokens.len());
            let spec = InjectionSpec {
                after_block: RECEIVER_BLOCK,
                positions: positions.clone(),
                vector: pooled.clone(),
                scale: Some(natural.median / norms::l2(&pooled)),
                mode: InjectionMode::Overwrite,
            };
            teacher_forced_nll(
                &mut receiver.model,
                &nll_tokens,
                span.clone(),
                Some(&spec),
                &device,
            )
            .map_err(e)
        };
        let nll_t = nll_for(&trained)?;
        let nll_i = nll_for(&init)?;
        nll_trained_sum += nll_t as f64;
        nll_init_sum += nll_i as f64;
        if nll_t < nll_i {
            wins += 1;
        } else if nll_t > nll_i {
            losses += 1;
        }
        n_eval += 1;
        if (done + 1) % 64 == 0 {
            println!(
                "[{}/{}] running means: trained {:.4}, init {:.4} ({:.0}s)",
                done + 1,
                holdout_rows.len(),
                nll_trained_sum / n_eval as f64,
                nll_init_sum / n_eval as f64,
                t0.elapsed().as_secs_f32()
            );
        }
        rows_json.push(serde_json::json!({
            "row": row, "item": sr.item, "target_tokens": target_fit,
            "natural_median": natural.median,
            "nll_trained": nll_t, "nll_init": nll_i,
        }));
    }
    anyhow::ensure!(
        skipped == receipt_skipped,
        "recomputed skip set {skipped:?} != training receipt's {receipt_skipped:?} — span rule drifted"
    );
    let mean_trained = nll_trained_sum / n_eval as f64;
    let mean_init = nll_init_sum / n_eval as f64;
    let pass = mean_trained < mean_init;
    let p_sign = common::sign_test_one_sided(wins, losses);
    println!(
        "transfer check: fused NLL trained {mean_trained:.6} vs init {mean_init:.6} over {n_eval} \
         holdout items => pass={pass}; wins {wins} losses {losses} (secondary sign p={p_sign:.2e})"
    );

    let receipt = serde_json::json!({
        "stage": "run2-m4d-transfer-check",
        "design": "ADR-024 M4d registered caveat mitigation, criterion frozen in the training receipt BEFORE this check ran; inference-only, no probe items, no generation, no frozen-probe draw",
        "env": common::env_info(&nvcc),
        "criterion_frozen_in_training_receipt": criterion,
        "config": {
            "receiver": RECEIVER, "forward": "vendored FUSED BF16 (the frozen probe's forward)",
            "inject_after_block": RECEIVER_BLOCK, "n_slots": N_SLOTS,
            "delivery": "hand-rolled MLP per-token over the item's FULL generated-span dump rows -> mean-pool TRANSLATED rows -> rescale to VENDORED natural inject-block median -> 8-slot injection (probe delivery path exactly)",
            "span_rule": format!("target = min(gen_len, {SEQ_CAP} - inj_len), skip if < {MIN_TARGET} (asserted equal to the training receipt's holdout skip set)"),
            "items": "leakage-safe holdout side only (never trained on, no probe overlap)",
            "artifacts": {"trained": trained.content_hash, "init": init.content_hash},
            "golden_verification": {"trained_max_rel": gmax, "init_max_rel": gmaxi},
        },
        "items": rows_json,
        "summary": {
            "n_evaluated": n_eval, "n_skipped": skipped.len(), "skipped_rows": skipped,
            "mean_fused_nll_trained": mean_trained,
            "mean_fused_nll_init": mean_init,
            "fused_improvement_nats": mean_init - mean_trained,
            "composed_improvement_nats_from_training_receipt":
                tr["results"]["composed_forward_improvement_nats"],
            "wins_trained_lower": wins, "losses": losses,
            "sign_test_p_one_sided_secondary": p_sign,
        },
        "gates": {
            "trained_artifact_hash_matches_training_receipt": {"pass": true},
            "init_artifact_hash_matches_training_receipt": {"pass": true},
            "prompt_parity_vs_streams": {"pass": true},
            "skip_set_matches_training_receipt": {"pass": true},
            "transfer": {"pass": pass, "criterion": "mean fused NLL(trained) < mean fused NLL(init)"},
        },
        "gate_pass": pass,
        "verdict": if pass {
            "improvement TRANSFERS across the composed->fused BF16 gap; the frozen probe may run"
        } else {
            "improvement does NOT transfer — the frozen probe must NOT be invoked (a null would be confounded by the numeric gap); this receipt is the honest M4d outcome for this branch"
        },
        "wall_clock_s": t0.elapsed().as_secs_f32(),
    });
    common::write_receipt(
        &crate_path("receipts"),
        "run2-m4d-transfer-receipt-cellL18toL14.json",
        &receipt,
    )?;
    Ok(())
}
