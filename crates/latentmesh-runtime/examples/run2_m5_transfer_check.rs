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

// Declared HERE rather than in `common/mod.rs`, which is `#[path]`-included
// into every example: this module carries the receipt literal.
#[path = "common/m5_transfer.rs"]
mod m5_transfer;

use common::m3::{build_site_prompt, Site, GOLDEN_REL_TOL, N_SLOTS, RECEIVER, RECEIVER_BLOCK};
use common::m5;
use common::mlp::MlpTransform;
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
/// Frozen span rule — MUST match `train_m5_receiver_lora.rs`'s `SEQ_CAP` and
/// `MIN_TARGET`; the recomputed skip set is asserted against the training
/// receipt's, so silent divergence fails loudly.
const SEQ_CAP: usize = 384;
const MIN_TARGET: usize = 8;
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

    // TWO arms, kept distinct on purpose (ADR-045 error #22 amendment):
    //   * the GATE is the fused CE of the TRAINING target (the rendered gold
    //     solution) — the actual composed->fused transfer question, measured
    //     on the quantity training optimised;
    //   * the probe's own endpoint NLL ("#### {gold}") is reported unchanged
    //     and comparably with the v1 receipt, but does NOT gate. Measuring
    //     transfer on a quantity training did not optimise would conflate
    //     transfer with generalisation.
    let mut rows_json = Vec::new();
    let mut skipped: Vec<usize> = Vec::new();
    let (mut sum_on, mut sum_off) = (0f64, 0f64);
    let (mut probe_on, mut probe_off) = (0f64, 0f64);
    let (mut wins, mut losses) = (0usize, 0usize);
    let (mut probe_wins, mut probe_losses) = (0usize, 0usize);
    let mut covered_sum = 0f64;
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
        // The TRAINING target, under the trainer's cap rule verbatim.
        let train_full = receiver
            .encode(&common::render_gold(&item.answer_text))
            .map_err(e)?;
        let target_fit = SEQ_CAP
            .saturating_sub(sp.tokens.len())
            .min(train_full.len());
        if target_fit < MIN_TARGET {
            skipped.push(row);
            continue;
        }
        let train_tokens: Vec<u32> = sp
            .tokens
            .iter()
            .chain(train_full[..target_fit].iter())
            .copied()
            .collect();
        let train_span = sp.tokens.len()..train_tokens.len();
        // The PROBE's endpoint, frozen and unchanged.
        let probe_target = receiver.encode(&format!("#### {}", item.gold)).map_err(e)?;
        let nll_tokens: Vec<u32> = sp
            .tokens
            .iter()
            .chain(probe_target.iter())
            .copied()
            .collect();
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
        let mut nll = |on: bool, toks: &[u32], sp: std::ops::Range<usize>| -> anyhow::Result<f32> {
            receiver
                .model
                .set_residual_lora(on.then(|| loaded.lora.clone()));
            teacher_forced_nll(&mut receiver.model, toks, sp, Some(&spec), &device).map_err(e)
        };
        let on = nll(true, &train_tokens, train_span.clone())?;
        let off = nll(false, &train_tokens, train_span.clone())?;
        let p_on = nll(true, &nll_tokens, span.clone())?;
        let p_off = nll(false, &nll_tokens, span.clone())?;
        sum_on += on as f64;
        sum_off += off as f64;
        probe_on += p_on as f64;
        probe_off += p_off as f64;
        if on < off {
            wins += 1;
        } else if on > off {
            losses += 1;
        }
        if p_on < p_off {
            probe_wins += 1;
        } else if p_on > p_off {
            probe_losses += 1;
        }
        covered_sum += target_fit as f64 / train_full.len().max(1) as f64;
        n_eval += 1;
        if (done + 1) % 64 == 0 {
            println!(
                "[{}/{}] running means (train target): on {:.4}, off {:.4}; (probe endpoint): \
                 on {:.4}, off {:.4} ({:.0}s)",
                done + 1,
                holdout_rows.len(),
                sum_on / n_eval as f64,
                sum_off / n_eval as f64,
                probe_on / n_eval as f64,
                probe_off / n_eval as f64,
                t0.elapsed().as_secs_f32()
            );
        }
        rows_json.push(serde_json::json!({
            "row": row, "item": sr.item,
            "train_target_tokens_used": target_fit,
            "train_target_tokens_total": train_full.len(),
            "probe_target_tokens": probe_target.len(),
            "natural_median": median,
            "train_target_ce_adapter_on": on, "train_target_ce_adapter_off": off,
            "probe_endpoint_nll_adapter_on": p_on, "probe_endpoint_nll_adapter_off": p_off,
        }));
    }
    receiver.model.set_residual_lora(None);
    anyhow::ensure!(
        skipped == receipt_skipped,
        "recomputed skip set {skipped:?} != the training receipt's {receipt_skipped:?} — the span \
         rule drifted between training and this check"
    );
    // ---- Degenerate-generation diagnostic (NON-GATING) --------------------
    // MANDATORY from ADR-045 error #22 onward. See `m5_transfer::generation_diagnostic`
    // for what it measures and why it deliberately does not gate.
    let gen_n = GEN_DIAG_ITEMS.min(holdout_rows.len());
    let diag = m5_transfer::generation_diagnostic(
        &mut receiver,
        &loaded.lora,
        &all_items,
        &streams.iter().map(|r| r.item).collect::<Vec<_>>(),
        holdout_rows.iter().take(gen_n).copied().collect::<Vec<_>>(),
        pad_id,
        MAX_NEW_TOKENS,
    )?;
    let (acc_on, acc_off, chars_on, chars_off, gen_rows) = diag;

    let mean_on = sum_on / n_eval as f64;
    let mean_off = sum_off / n_eval as f64;
    let mean_probe_on = probe_on / n_eval as f64;
    let mean_probe_off = probe_off / n_eval as f64;
    let mean_covered = covered_sum / n_eval as f64;
    let pass = mean_on < mean_off;
    let p_sign = common::sign_test_one_sided(wins, losses);
    let p_probe = common::sign_test_one_sided(probe_wins, probe_losses);
    println!(
        "transfer check (GATE, training target): fused CE adapter-on {mean_on:.6} vs off \
         {mean_off:.6} over {n_eval} holdout items => pass={pass}; {wins}W/{losses}L (sign \
         p={p_sign:.2e}); mean gold-solution coverage {mean_covered:.3}"
    );
    println!(
        "transfer check (REPORTED, probe endpoint '#### gold'): fused NLL adapter-on \
         {mean_probe_on:.6} vs off {mean_probe_off:.6}; {probe_wins}W/{probe_losses}L (sign \
         p={p_probe:.2e}) — NOT gating"
    );

    let receipt = m5_transfer::receipt(m5_transfer::TransferCtx {
        rank,
        env: common::env_info(&nvcc),
        criterion: &criterion,
        receiver: RECEIVER,
        inject_block: RECEIVER_BLOCK,
        n_slots: N_SLOTS,
        site: Site::QuestionTail.tag(),
        mode_tag: INJECT_MODE.tag(),
        mode_equation: INJECT_MODE.equation(),
        seq_cap: SEQ_CAP,
        min_target: MIN_TARGET,
        adapter_hash: &loaded.lora.content_hash,
        adapter_rank: loaded.lora.rank,
        adapter_alpha: loaded.lora.alpha,
        adapter_scaling: loaded.lora.scaling(),
        adapter_params: loaded.param_count,
        adapter_golden_pairs: loaded.golden_pairs,
        adapter_golden_max_rel: loaded.golden_max_rel,
        payload_hash: &transform.content_hash,
        payload_golden_max_rel: gmax,
        rows: rows_json,
        n_eval,
        skipped: skipped.clone(),
        mean_on,
        mean_off,
        mean_covered,
        mean_probe_on,
        mean_probe_off,
        wins,
        losses,
        p_sign,
        probe_wins,
        probe_losses,
        p_probe,
        composed_improvement: tr["results"]["composed_forward_improvement_nats"].clone(),
        gen_n,
        acc_on,
        acc_off,
        chars_on,
        chars_off,
        gen_rows,
        pass,
        smoke,
        wall_clock_s: t0.elapsed().as_secs_f32(),
    });
    common::write_receipt(
        &m5_dir,
        &format!("run2-m5-transfer-receipt-cellL18toL14-r{rank}.json"),
        &receipt,
    )?;
    Ok(())
}
