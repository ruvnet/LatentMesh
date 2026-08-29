//! Run-2 **M5** — registered transfer check (ADR-045's inherited M4c caveat),
//! run BETWEEN training and the draw, and gating the draw.
//!
//! The M5 adapter is trained through the COMPOSED differentiable BF16 forward
//! (`latentmesh-train::qwen2_c`); the draw runs the vendored FUSED BF16
//! forward. The measured gap between the two (116/128 argmax agreement,
//! max|dlogit| 8.19 at L=128 — pure rounding amplification; F32 parity
//! 128/128 proves same function) means a null could be confounded unless the
//! training improvement is first shown to survive the crossing.
//!
//! For every evaluable holdout item — same leakage-safe holdout side, same
//! skip rule as training, recomputed here and asserted against the training
//! receipt — this computes the teacher-forced NLL of `"#### {gold}"` through
//! the vendored forward under the probe's exact delivery path (M3's frozen
//! adapter on the item's LAST dump row → rescale to the vendored natural
//! block-14 median → fuse at the 8 question-tail positions), once with the
//! trained adapter installed and once with it removed.
//!
//! "Removed" and "at init" are the same function here: B is zero-initialised,
//! so the init adapter is the exact identity. Both artifacts are committed
//! either way, and this check verifies the trained one against its goldens
//! before running anything.
//!
//! Frozen pass criterion, quoted from the training receipt (written there
//! BEFORE this ran): mean fused NLL(trained) < mean fused NLL(off). Per-item
//! wins/losses and a sign test are secondary, not gating. On fail the draw is
//! NOT invoked and this receipt is the honest M5 outcome for that branch.
//!
//! Run: PATH=/usr/local/cuda-12.8/bin:$PATH \
//!      cargo run --release --features cuda --example run2_m5_transfer_check -- [rank]

#[path = "common/mod.rs"]
#[allow(dead_code)]
mod common;

use common::m3::{build_site_prompt, Site, GOLDEN_REL_TOL, N_SLOTS, RECEIVER, RECEIVER_BLOCK};
use common::m5;
use common::mlp::MlpTransform;
use latentmesh_runtime::sampler::{Sampler, Sampling};
use latentmesh_runtime::{
    capture::forward_capture,
    inject::{teacher_forced_nll, InjectionMode, InjectionSpec},
    norms, QwenRuntime,
};
use sha2::Digest as _;
use std::io::{Read as _, Seek as _};
use std::path::{Path, PathBuf};

const DUMP_RECEIPT: &str = "receipts/run2-pertoken-dump-receipt.json";
const M3_ARTIFACT: &str = "receipts/run2-m3-mlp-cellL18toL14.f32bin";
const M3_GOLDEN: &str = "receipts/run2-m3-golden-mlp-cellL18toL14.json";
const M3_TRAINING_RECEIPT: &str = "receipts/run2-m3-training-receipt-cellL18toL14.json";
const SENDER_DIM: usize = 2048;
const HIDDEN: usize = 1536;
/// Frozen span rule — MUST match `train_m5_receiver_lora.rs`'s `SEQ_CAP`; the
/// recomputed skip set is asserted against the training receipt's.
const SEQ_CAP: usize = 256;
const INJECT_MODE: InjectionMode = InjectionMode::Fuse;
/// Holdout items used for the degenerate-generation diagnostic. NON-GATING,
/// and never a draw item — see `GEN_DIAG_WHY` in the receipt.
const GEN_DIAG_ITEMS: usize = 64;
const MAX_NEW_TOKENS: usize = 400;
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

/// The item's LAST generated-span row from the ragged dump — the only row
/// `apply_last_row` reads.
fn read_last_row(f: &mut std::fs::File, tok0: u64, n_rows: usize) -> anyhow::Result<Vec<f32>> {
    let at = (tok0 + n_rows as u64 - 1) * SENDER_DIM as u64 * 4;
    f.seek(std::io::SeekFrom::Start(at))?;
    let mut buf = vec![0u8; SENDER_DIM * 4];
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
    let rank: usize = match std::env::args().nth(1) {
        Some(a) => a.parse()?,
        None => 1,
    };
    // SMOKE MODE (see the trainer): `LM_M5_SMOKE` is set, so read the M5
    // artifacts from — and write this receipt into — the gitignored target/
    // tree. The registered path never sets it and never touches that tree.
    let smoke = std::env::var("LM_M5_SMOKE").is_ok();
    let m5_dir = if smoke {
        let d = crate_path("target/latentmesh-runs/run2-m5-smoke");
        std::fs::create_dir_all(&d)?;
        println!(
            "SMOKE MODE: M5 artifacts read from and receipt written to {}",
            d.display()
        );
        d
    } else {
        crate_path("receipts")
    };

    // ---- Training receipt: hashes, holdout rows, frozen criterion ---------
    let tr: serde_json::Value = serde_json::from_slice(&std::fs::read(m5_dir.join(format!(
        "run2-m5-training-receipt-cellL18toL14-r{rank}.json"
    )))?)?;
    anyhow::ensure!(
        tr["rank"].as_u64() == Some(rank as u64),
        "the training receipt is for a different rank"
    );
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
        .expect("frozen transfer criterion in the training receipt")
        .to_string();
    println!("frozen criterion (from the training receipt): {criterion}");

    // ---- The two frozen/trained artifacts, both verified ------------------
    let m3_receipt: serde_json::Value =
        serde_json::from_slice(&std::fs::read(crate_path(M3_TRAINING_RECEIPT))?)?;
    let transform = MlpTransform::load(&crate_path(M3_ARTIFACT))?;
    anyhow::ensure!(
        Some(transform.content_hash.as_str())
            == m3_receipt["artifact"]["content_hash_sha256"].as_str(),
        "M3 artifact hash != M3 training receipt"
    );
    let (gn, gmax, _) =
        common::mlp::verify_against_golden(&transform, &crate_path(M3_GOLDEN), GOLDEN_REL_TOL)?;
    println!(
        "M3 payload adapter {} verified ({gn} goldens, max rel {gmax:.2e})",
        transform.content_hash
    );

    let device = latentmesh_runtime::device().map_err(e)?;
    let loaded = m5::load_adapter(&m5_dir, rank, &tr, &device)?;
    println!(
        "M5 adapter r{rank} {} verified against {} goldens (max rel L2 {:.2e}); {} params",
        loaded.lora.content_hash, loaded.golden_pairs, loaded.golden_max_rel, loaded.param_count
    );

    // ---- Dump integrity ---------------------------------------------------
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

    // ---- Streams + items, sha-gated ---------------------------------------
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
    anyhow::ensure!(
        format!("{:x}", sha2::Sha256::digest(std::fs::read(&gsm_path)?)) == GSM8K_TRAIN_SHA256
    );
    let all_items = common::load_gsm8k(&gsm_path)?;

    // ---- Receiver ---------------------------------------------------------
    println!("loading {RECEIVER} (BF16, vendored fused forward)...");
    let mut receiver = QwenRuntime::load(RECEIVER, &device, candle_core::DType::BF16).map_err(e)?;
    let pad_id = receiver
        .tokenizer
        .token_to_id("<|fim_pad|>")
        .ok_or_else(|| anyhow::anyhow!("<|fim_pad|> not in tokenizer"))?;

    let mut rows_json = Vec::new();
    let mut skipped: Vec<usize> = Vec::new();
    let (mut sum_on, mut sum_off) = (0f64, 0f64);
    let (mut wins, mut losses) = (0usize, 0usize);
    let mut n_eval = 0usize;
    for (done, &row) in holdout_rows.iter().enumerate() {
        let sr = &streams[row];
        let item = &all_items[sr.item];
        // The probe's OWN prompt builder and site gate, not a copy.
        let sp = build_site_prompt(&receiver, item, pad_id, Site::QuestionTail)?;
        anyhow::ensure!(
            sp.tokens == sr.prompt_tokens,
            "row {row}: prompt-parity gate failed — the question-tail prompt is not the stream's \
             stored capture prompt"
        );
        let target = receiver.encode(&format!("#### {}", item.gold)).map_err(e)?;
        if sp.tokens.len() + target.len() > SEQ_CAP {
            skipped.push(row);
            continue;
        }
        let nll_tokens: Vec<u32> = sp.tokens.iter().chain(target.iter()).copied().collect();
        let span = sp.tokens.len()..nll_tokens.len();

        // Natural block-14 norms with the adapter OFF — a Capture tap runs
        // before the adapter by construction, but removing it here makes the
        // rescale target unambiguously the base receiver's.
        receiver.model.set_residual_lora(None);
        let (_, nat) = forward_capture(
            &mut receiver.model,
            &sp.tokens,
            RECEIVER_BLOCK,
            0..sp.tokens.len(),
            &device,
        )
        .map_err(e)?;
        let median = norms::stats(nat.per_position_l2.clone()).median;

        let last = read_last_row(&mut sender_bin, token_offsets[row], sr.gen_tokens.len())?;
        let aligned = transform.apply_last_row(&last, 1);
        anyhow::ensure!(aligned.len() == HIDDEN);
        let spec = InjectionSpec {
            after_block: RECEIVER_BLOCK,
            positions: sp.positions.clone(),
            vector: aligned.clone(),
            scale: Some(median / norms::l2(&aligned)),
            mode: INJECT_MODE,
        };
        let mut nll = |on: bool| -> anyhow::Result<f32> {
            receiver
                .model
                .set_residual_lora(on.then(|| loaded.lora.clone()));
            teacher_forced_nll(
                &mut receiver.model,
                &nll_tokens,
                span.clone(),
                Some(&spec),
                &device,
            )
            .map_err(e)
        };
        let on = nll(true)?;
        let off = nll(false)?;
        sum_on += on as f64;
        sum_off += off as f64;
        if on < off {
            wins += 1;
        } else if on > off {
            losses += 1;
        }
        n_eval += 1;
        if (done + 1) % 64 == 0 {
            println!(
                "[{}/{}] running means: adapter-on {:.4}, adapter-off {:.4} ({:.0}s)",
                done + 1,
                holdout_rows.len(),
                sum_on / n_eval as f64,
                sum_off / n_eval as f64,
                t0.elapsed().as_secs_f32()
            );
        }
        rows_json.push(serde_json::json!({
            "row": row, "item": sr.item, "target_tokens": target.len(),
            "natural_median": median,
            "nll_adapter_on": on, "nll_adapter_off": off,
        }));
    }
    receiver.model.set_residual_lora(None);
    anyhow::ensure!(
        skipped == receipt_skipped,
        "recomputed skip set {skipped:?} != the training receipt's {receipt_skipped:?} — the span \
         rule drifted between training and this check"
    );
    // ---- Degenerate-generation diagnostic (NON-GATING) --------------------
    // A rank-1 adapter trained hard on `"#### {gold}"` can in principle learn
    // to emit the answer format and stop reasoning — a format collapse that
    // would make the draw uninformative (every condition wrong, so almost no
    // discordant pairs) for a reason that has nothing to do with the channel.
    // ADR-024's PC1 registration already requires a degenerate-output check on
    // any draw, and `examples/common/m3.rs` records generated length per
    // condition for exactly this failure mode. This is the same instrument,
    // run BEFORE the draw, on HOLDOUT items only.
    //
    // It does NOT gate, and deliberately so: gating the adapter on GSM8K
    // accuracy would select it against the very confound ADR-045 exists to
    // keep out of the primary. It is reported so that a null can be read
    // correctly — "the receiver stopped answering" is a different finding from
    // "the channel carries nothing".
    let mut gen_rows = Vec::new();
    let (mut acc_on, mut acc_off, mut chars_on, mut chars_off) = (0usize, 0usize, 0f64, 0f64);
    let gen_n = GEN_DIAG_ITEMS.min(holdout_rows.len());
    println!("degenerate-generation diagnostic over {gen_n} holdout items (baseline condition, adapter on vs off)...");
    for &row in holdout_rows.iter().take(gen_n) {
        let item = &all_items[streams[row].item];
        let sp = build_site_prompt(&receiver, item, pad_id, Site::QuestionTail)?;
        let mut gen = |on: bool| -> anyhow::Result<(bool, usize)> {
            receiver
                .model
                .set_residual_lora(on.then(|| loaded.lora.clone()));
            let mut s = Sampler::new(Sampling::Greedy, 0);
            let out = receiver
                .generate(&sp.tokens, None, &mut s, MAX_NEW_TOKENS, false)
                .map_err(e)?;
            let ok = common::extract_answer(&out.text)
                .is_some_and(|a| common::answers_equal(&a, &item.gold));
            Ok((ok, out.text.chars().count()))
        };
        let (ok_on, len_on) = gen(true)?;
        let (ok_off, len_off) = gen(false)?;
        acc_on += usize::from(ok_on);
        acc_off += usize::from(ok_off);
        chars_on += len_on as f64;
        chars_off += len_off as f64;
        gen_rows.push(serde_json::json!({
            "row": row, "item": item.index,
            "correct_adapter_on": ok_on, "correct_adapter_off": ok_off,
            "generated_chars_adapter_on": len_on, "generated_chars_adapter_off": len_off,
        }));
    }
    receiver.model.set_residual_lora(None);
    println!(
        "generation diagnostic: baseline accuracy adapter-on {acc_on}/{gen_n} vs off \
         {acc_off}/{gen_n}; mean generated chars {:.0} vs {:.0}",
        chars_on / gen_n as f64,
        chars_off / gen_n as f64
    );

    let mean_on = sum_on / n_eval as f64;
    let mean_off = sum_off / n_eval as f64;
    let pass = mean_on < mean_off;
    let p_sign = common::sign_test_one_sided(wins, losses);
    println!(
        "transfer check: fused gold-continuation NLL adapter-on {mean_on:.6} vs off {mean_off:.6} \
         over {n_eval} holdout items => pass={pass}; wins {wins} losses {losses} (secondary sign \
         p={p_sign:.2e})"
    );

    let receipt = serde_json::json!({
        "stage": "run2-m5-transfer-check",
        "design": "ADR-045 M5, inheriting ADR-024 M4c's registered composed-vs-fused caveat mitigation. The criterion was frozen in the M5 training receipt BEFORE this check ran. Inference-only: no draw items, no generation, no e-process.",
        "rank": rank,
        "env": common::env_info(&nvcc),
        "criterion_frozen_in_training_receipt": criterion,
        "config": {
            "receiver": RECEIVER,
            "forward": "vendored FUSED BF16 — the forward the draw runs",
            "inject_after_block": RECEIVER_BLOCK, "n_slots": N_SLOTS,
            "site": Site::QuestionTail.tag(),
            "operator": {"mode": INJECT_MODE.tag(), "equation": INJECT_MODE.equation()},
            "delivery": "M3's frozen hand-rolled MLP on the item's LAST generated-span dump row -> rescale to the VENDORED natural block-14 median -> fuse at the 8 question-tail positions (the draw's delivery path exactly)",
            "arms": "the SAME receiver with the trained adapter installed and removed. B is zero-initialised, so 'removed' and 'at init' are the same function; the init artifact and its goldens are committed regardless.",
            "span_rule": format!("skip if prompt_len + gold_continuation_len > {SEQ_CAP}; the recomputed skip set is asserted equal to the training receipt's"),
            "items": "the leakage-safe holdout side only — never trained on, and disjoint from the draw's adaptation-512 stream",
            "adapter": {
                "content_hash": loaded.lora.content_hash,
                "rank": loaded.lora.rank, "alpha": loaded.lora.alpha,
                "scaling": loaded.lora.scaling(), "param_count": loaded.param_count,
                "golden_pairs": loaded.golden_pairs,
                "golden_max_relative_l2_error": loaded.golden_max_rel,
            },
            "payload_adapter": {"content_hash": transform.content_hash, "golden_max_rel": gmax},
        },
        "items": rows_json,
        "summary": {
            "n_evaluated": n_eval, "n_skipped": skipped.len(), "skipped_rows": skipped,
            "mean_fused_nll_adapter_on": mean_on,
            "mean_fused_nll_adapter_off": mean_off,
            "fused_improvement_nats": mean_off - mean_on,
            "composed_improvement_nats_from_training_receipt": tr["results"]["composed_forward_improvement_nats"],
            "wins_adapter_on_lower": wins, "losses": losses,
            "sign_test_p_one_sided_secondary": p_sign,
            "generation_diagnostic": {
                "gating": "NONE",
                "why": "a rank-1 adapter trained on '#### {gold}' can learn the answer FORMAT and stop reasoning. That collapse would make the draw uninformative — almost no discordant pairs, because every condition is wrong — for a reason that has nothing to do with the channel. Measured here, before the draw, on HOLDOUT items only.",
                "why_not_gating": "gating the adapter on GSM8K accuracy would select it against the very confound ADR-045 keeps out of the primary. This is reported so a null can be read correctly: 'the receiver stopped answering' is a different finding from 'the channel carries nothing'.",
                "condition": "baseline (no injection), greedy, batch=1, max_new_tokens=400 — the draw's own decoding",
                "n_items": gen_n,
                "accuracy_adapter_on": acc_on, "accuracy_adapter_off": acc_off,
                "mean_generated_chars_adapter_on": chars_on / gen_n as f64,
                "mean_generated_chars_adapter_off": chars_off / gen_n as f64,
                "items": gen_rows,
            },
            "what_this_does_NOT_establish": "nothing about the channel. This measures only that a training improvement survives the composed->fused BF16 crossing, on items the draw never sees. An adapter that merely made the receiver a better GSM8K solver would pass this check too — which is exactly why ADR-045's primary is aligned vs random on the same adapted receiver.",
        },
        "gates": {
            "m3_payload_hash_matches_its_training_receipt": {"pass": true},
            "m5_adapter_hash_matches_the_m5_training_receipt": {"pass": true},
            "m5_adapter_matches_its_golden_pairs": {"pass": true, "max_relative_l2_error": loaded.golden_max_rel},
            "prompt_parity_vs_streams": {"pass": true},
            "skip_set_matches_training_receipt": {"pass": true},
            "transfer": {"pass": pass, "criterion": "mean fused NLL(adapter on) < mean fused NLL(adapter off)"},
        },
        "gate_pass": pass,
        "verdict": if pass {
            "the training improvement TRANSFERS across the composed->fused BF16 gap; the M5 draw may run"
        } else {
            "the training improvement does NOT transfer — the M5 draw must NOT be invoked (a null would be confounded by the numeric gap); this receipt is the honest M5 outcome for that branch"
        },
        "smoke_run": smoke,
        "wall_clock_s": t0.elapsed().as_secs_f32(),
    });
    common::write_receipt(
        &m5_dir,
        &format!("run2-m5-transfer-receipt-cellL18toL14-r{rank}.json"),
        &receipt,
    )?;
    Ok(())
}
