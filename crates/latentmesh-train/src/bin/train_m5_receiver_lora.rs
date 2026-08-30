//! M5 trainer (ADR-045 — receiver-side adaptation): train an additive
//! rank-`r` LoRA on the **receiver's own residual stream** at the L14
//! injection site, with the sender and the translator FROZEN.
//!
//! Every rung before this one trained the payload to suit a bit-identical
//! frozen receiver. None trained the receiver to accept the payload — the
//! asymmetry ADR-045 identifies as the last untested axis, and one
//! arXiv:2606.05711's Table 5 finds untested in the literature too.
//!
//! Per training step (batch = 1):
//!   sender L18 state of the item's LAST generated-span token (from the
//!   verified M2 dump) → M3's **already-trained, byte-identical**
//!   reconstruction MLP (frozen, hash-asserted against M3's training receipt)
//!   → rescale to the receiver's natural block-14 median → broadcast to the 8
//!   **question-tail** positions → `InjectionMode::Fuse` after block 14 →
//!   **LoRA after the edit** → teacher-forced CE on the **gold-answer
//!   continuation** `"#### {gold}"` → AdamW step on the two LoRA Vars only.
//!
//! Two choices are worth naming because they are corrections, not defaults:
//!
//! * **the target is the gold continuation, not the sender's span.**
//!   `docs/research/034` §5.2 identifies the sender-span target as M4c's
//!   diagnosed mismatch (it steers the receiver toward reproducing the
//!   sender's tokens, not the answer the probe scores). The target here is
//!   the probe's OWN likelihood target, token for token.
//! * **the payload is M3's on-manifold reconstruction adapter**, delivered
//!   exactly as M4i delivered it (`apply_last_row`, fuse, question tail,
//!   8 positions, rescale-to-natural-median). That makes M4i the comparator
//!   and "the receiver carries a trained LoRA" the single changed factor.
//!   Using an M4c/M4d/M4g task-loss adapter instead would reintroduce the
//!   off-manifold payload ADR-045 names as the inherited hazard.
//!
//! ΔV is NEVER computed here: `docs/research/034` §3 prices one properly
//! powered `verify_edge` draw at ~3 GPU-h, more than this entire run, and
//! ADR-045 registers ΔV as a one-off post-hoc characterisation.
//!
//! Run: PATH=/usr/local/cuda-12.8/bin:$PATH \
//!      cargo run --release --features cuda --bin train_m5_receiver_lora -- [rank]

use latentmesh_train::dataset::{expected_from_receipt, open_verified, sha256_file};
use latentmesh_train::deploy::deploy_slot_vectors;
use latentmesh_train::m5receipt;
use latentmesh_train::mlp::{Mlp, D_IN, D_OUT};
use latentmesh_train::qwen2_c::{load_config, span_ce, TrainReceiver};
use latentmesh_train::receiver_lora::{golden_pairs, LoraAdapter};
use latentmesh_train::split::{leakage_safe_split, PROBE_OVERLAP_ROW_ITEMS};
use latentmesh_train::taskdata::{self, M5Item};

use candle_core::{DType, Device, Tensor};
use candle_nn::Optimizer;
use latentmesh_runtime::inject::InjectionMode;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::path::{Path, PathBuf};

const RECEIVER: &str = "Qwen/Qwen2.5-1.5B-Instruct";
const SENDER_FILE: &str = "sender_L18.tok.f32bin";
const RECEIVER_FILE: &str = "receiver_L14.tok.f32bin";
const CELL: &str = "L18->L14";
const INJECT_AFTER_BLOCK: usize = 14;
const N_SLOTS: usize = 8;
const INJECT_MODE: InjectionMode = InjectionMode::Fuse;
/// M3's frozen reconstruction adapter — the payload, unchanged from M4i.
const M3_ARTIFACT: &str = "run2-m3-mlp-cellL18toL14.f32bin";
const M3_TRAINING_RECEIPT: &str = "run2-m3-training-receipt-cellL18toL14.json";
const ADAPTATION_512: &str = "../../harness/latentmesh-live/data/adaptation-512.json";
/// Training RNG seed (LoRA init + per-epoch shuffle), frozen here.
const TRAIN_SEED: u64 = 0x4D35_0001;
/// Golden-pair input seed (probe-side verification), frozen here.
const GOLDEN_SEED: u64 = 0x4D35_061D;
const GOLDEN_PAIRS: usize = 8;
const LR: f64 = 1e-3;
const EPOCHS: usize = 10;
/// Max receiver sequence. MEASURED on this 16 GB card by the M5 v1 run
/// (9,842 MiB peak at prompt+answer-line lengths under 256) and re-measured by
/// the v2 smoke at this value before the real run; the receipt records the
/// peak actually observed.
const SEQ_CAP: usize = 384;
/// Minimum CE-target tokens for an item to train/evaluate — M4c's rule,
/// verbatim. Items are TRUNCATED to the cap, not dropped, so this only
/// excludes items whose prompt alone nearly fills the window.
const MIN_TARGET: usize = 8;

fn crate_rel(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

/// One item's gold-continuation CE through the composed forward, with the
/// frozen payload fused at the question tail and the LoRA after the edit.
fn item_loss(
    model: &TrainReceiver,
    lora: &LoraAdapter,
    it: &M5Item,
    payload: &[f32],
    dev: &Device,
) -> anyhow::Result<Tensor> {
    let row = Tensor::from_vec(payload.to_vec(), (1, D_OUT), dev)?;
    let vectors = Tensor::cat(&[&row; N_SLOTS], 0)?;
    let tokens = Tensor::new(&it.full_tokens[..], dev)?.unsqueeze(0)?;
    let logits = model.forward_span_logits_with_lora(
        &tokens,
        Some((&vectors, &it.positions, INJECT_AFTER_BLOCK, INJECT_MODE)),
        Some((lora, INJECT_AFTER_BLOCK)),
        it.span_start,
        it.target_tokens.len(),
    )?;
    Ok(span_ce(&logits, &it.target_tokens, dev)?)
}

fn eval_ce(
    model: &TrainReceiver,
    lora: &LoraAdapter,
    items: &[M5Item],
    payloads: &[Vec<f32>],
    dev: &Device,
) -> anyhow::Result<f64> {
    let mut sum = 0f64;
    for (i, it) in items.iter().enumerate() {
        sum += item_loss(model, lora, it, &payloads[i], dev)?.to_scalar::<f32>()? as f64;
    }
    Ok(sum / items.len() as f64)
}

fn main() -> anyhow::Result<()> {
    let t0 = std::time::Instant::now();
    let nvcc = latentmesh_runtime::assert_cuda_build_env().map_err(anyhow::Error::msg)?;
    println!("build-env guard: {nvcc}");
    let rank: usize = match std::env::args().nth(1) {
        Some(a) => a.parse()?,
        None => 1,
    };
    anyhow::ensure!(
        [1usize, 2, 4].contains(&rank),
        "ADR-045 registers ranks {{1,2,4}} only; got {rank}"
    );
    println!("M5 receiver LoRA, rank {rank} (ADR-045 ladder: cheapest first)");
    let receipts_dir = crate_rel("../latentmesh-runtime/receipts");
    // SMOKE MODE — pipeline proof only, quarantined from the registered run.
    // `LM_M5_SMOKE=n` caps the fit/holdout item counts and runs ONE epoch, and
    // writes every artifact and receipt into the gitignored `target/` tree
    // instead of `receipts/`. The registered path is the one with the variable
    // unset; a smoke artifact can never be picked up by it, because the draw
    // reads `receipts/` and a smoke run never writes there.
    let smoke: Option<usize> = std::env::var("LM_M5_SMOKE")
        .ok()
        .map(|v| v.parse::<usize>())
        .transpose()?;
    let out_dir = match smoke {
        None => receipts_dir.clone(),
        Some(n) => {
            let d = crate_rel("../latentmesh-runtime/target/latentmesh-runs/run2-m5-smoke");
            std::fs::create_dir_all(&d)?;
            println!(
                "SMOKE MODE: {n} fit / {n} holdout items, 1 epoch, output -> {}",
                d.display()
            );
            d
        }
    };
    let epochs = if smoke.is_some() { 1 } else { EPOCHS };

    // ---- Dataset integrity (all four bins + index) vs the M2 receipt ------
    let dump_receipt = receipts_dir.join("run2-pertoken-dump-receipt.json");
    let (expected, run_dir) = expected_from_receipt(&dump_receipt)?;
    let (ds, verified) = open_verified(&run_dir, SENDER_FILE, RECEIVER_FILE, &expected)?;
    for v in &verified {
        println!("  {}: sha256 {} == receipt OK", v.file, v.sha256);
    }
    anyhow::ensure!(ds.sender.dim == D_IN && ds.receiver.dim == D_OUT);
    for (row, item) in PROBE_OVERLAP_ROW_ITEMS {
        anyhow::ensure!(
            ds.index.item_indices[row] == item,
            "probe-overlap row {row}"
        );
    }
    let mut split = leakage_safe_split(ds.index.n_items);
    anyhow::ensure!((2035..=2048).contains(&split.fit.len()));
    if let Some(n) = smoke {
        // Truncate AFTER the split-arithmetic assert, so the leakage rule is
        // still the one that was checked.
        split.fit.truncate(n);
        split.holdout.truncate(n);
    }

    // ---- Streams + questions/golds + tokenizer; M5 items ------------------
    let streams = taskdata::load_streams(&crate_rel(taskdata::STREAMS_REL))?;
    anyhow::ensure!(streams.len() == ds.index.n_items);
    for (row, sr) in streams.iter().enumerate() {
        anyhow::ensure!(
            sr.item == ds.index.item_indices[row] && sr.gen_tokens.len() == ds.index.gen_len[row],
            "row {row}: stream/index mismatch"
        );
    }
    let gsm = taskdata::load_gsm8k_items(&crate_rel(taskdata::GSM8K_TRAIN_REL))?;
    let tok = taskdata::load_tokenizer(RECEIVER)?;
    let build = |rows: &[usize]| {
        taskdata::build_m5_items(
            rows,
            &streams,
            &gsm,
            &tok,
            &ds.index.token_offsets,
            N_SLOTS,
            SEQ_CAP,
            MIN_TARGET,
        )
    };
    let (fit_items, fit_skipped) = build(&split.fit)?;
    let (holdout_items, holdout_skipped) = build(&split.holdout)?;
    // Truncation accounting for the amended render_gold target: how much of
    // each item's reasoning the CE span actually covers. Reported, because a
    // silently truncated target is how error #22 would recur in another form.
    let coverage = |items: &[M5Item]| -> (f64, usize) {
        let full = items.iter().filter(|i| i.covered_fraction() >= 1.0).count();
        let mean = items.iter().map(|i| i.covered_fraction()).sum::<f64>() / items.len() as f64;
        (mean, full)
    };
    let (fit_cov, fit_full) = coverage(&fit_items);
    let (hold_cov, hold_full) = coverage(&holdout_items);
    println!(
        "split: fit {} items ({} skipped by cap), holdout {} ({} skipped); prompt-parity and \
         question-tail site gates passed on all {} built items",
        fit_items.len(),
        fit_skipped.len(),
        holdout_items.len(),
        holdout_skipped.len(),
        fit_items.len() + holdout_items.len()
    );
    println!(
        "gold-solution coverage under SEQ_CAP {SEQ_CAP}: fit mean {:.3} ({fit_full}/{} items \
         complete), holdout mean {:.3} ({hold_full}/{} complete)",
        fit_cov,
        fit_items.len(),
        hold_cov,
        holdout_items.len()
    );

    // ---- Disjointness from the probe's item stream, MEASURED --------------
    let adaptation: serde_json::Value =
        serde_json::from_slice(&std::fs::read(crate_rel(ADAPTATION_512))?)?;
    anyhow::ensure!(adaptation["split"].as_str() == Some("adaptation-512"));
    let adapt: std::collections::HashSet<usize> = adaptation["indices"]
        .as_array()
        .expect("adaptation-512 indices")
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let trained_items: std::collections::HashSet<usize> =
        fit_items.iter().map(|it| it.item).collect();
    let overlap: Vec<usize> = trained_items.intersection(&adapt).copied().collect();
    anyhow::ensure!(
        overlap.is_empty(),
        "the M5 fit set intersects adaptation-512 (the draw's item stream) at {overlap:?}"
    );
    println!(
        "item-stream disjointness: 0 of the {} trained items appear in adaptation-512",
        trained_items.len()
    );

    // ---- Receiver (composed differentiable BF16) + frozen M3 translator ---
    let dev = Device::new_cuda(0).map_err(anyhow::Error::msg)?;
    let cfg = load_config(RECEIVER)?;
    println!("loading {RECEIVER} (BF16, composed qwen2_c forward)...");
    let model = TrainReceiver::load(RECEIVER, &cfg, DType::BF16, &dev)?;
    anyhow::ensure!(model.hidden_size() == D_OUT);
    let m3_receipt: serde_json::Value =
        serde_json::from_slice(&std::fs::read(receipts_dir.join(M3_TRAINING_RECEIPT))?)?;
    let m3_expected = m3_receipt["artifact"]["content_hash_sha256"]
        .as_str()
        .expect("M3 training receipt artifact.content_hash_sha256");
    let (m3, m3_hash) = Mlp::load_artifact(&receipts_dir.join(M3_ARTIFACT), &dev)?;
    anyhow::ensure!(
        m3_hash == m3_expected,
        "M3 artifact hash {m3_hash} != M3 training receipt's {m3_expected}"
    );
    println!("frozen translator: M3 MLP {m3_hash} (byte-identical to M4i's payload)");

    // ---- Per-item constants: natural median + the frozen aligned payload --
    // Both are independent of the LoRA (the natural median is captured on the
    // BASE receiver, exactly as the probe's forward_capture does), so they are
    // computed once and reused every epoch.
    println!("precomputing natural block-14 medians and frozen aligned payloads...");
    let payloads_for = |items: &[M5Item]| -> anyhow::Result<(Vec<Vec<f32>>, Vec<f32>)> {
        let mut payloads = Vec::with_capacity(items.len());
        let mut medians = Vec::with_capacity(items.len());
        for (n, it) in items.iter().enumerate() {
            let prompt = Tensor::new(&it.full_tokens[..it.prompt_len], &dev)?.unsqueeze(0)?;
            let l2 = model.natural_per_position_l2(&prompt, INJECT_AFTER_BLOCK)?;
            let median = latentmesh_runtime::norms::stats(l2).median;
            let x = Tensor::from_vec(ds.sender.row(it.last_row_tok).to_vec(), (1, D_IN), &dev)?;
            let y = m3.forward(&x)?.detach().squeeze(0)?;
            let v = deploy_slot_vectors(&y, median, 1)?
                .squeeze(0)?
                .to_vec1::<f32>()?;
            payloads.push(v);
            medians.push(median);
            if (n + 1) % 512 == 0 {
                println!(
                    "  {}/{} ({:.0}s)",
                    n + 1,
                    items.len(),
                    t0.elapsed().as_secs_f32()
                );
            }
        }
        Ok((payloads, medians))
    };
    let (fit_payloads, fit_medians) = payloads_for(&fit_items)?;
    let (holdout_payloads, holdout_medians) = payloads_for(&holdout_items)?;
    let med_stats = latentmesh_runtime::norms::stats(
        fit_medians
            .iter()
            .chain(&holdout_medians)
            .copied()
            .collect(),
    );
    println!("natural medians across items: {med_stats:?}");

    // ---- LoRA: fresh seeded init; INIT artifact + goldens saved first -----
    let lora = LoraAdapter::new_seeded(TRAIN_SEED, rank, D_OUT, &dev)?;
    let name = |s: &str| format!("run2-m5-lora-r{rank}-{s}cellL18toL14");
    let init_artifact = out_dir.join(format!("{}.f32bin", name("init-")));
    let init_hash = lora.save_artifact(&init_artifact)?;
    let (init_golden_sha, init_in, init_out) = golden_pairs(
        &init_artifact,
        D_OUT,
        INJECT_AFTER_BLOCK,
        GOLDEN_SEED,
        GOLDEN_PAIRS,
    )?;
    anyhow::ensure!(init_golden_sha == init_hash);
    let init_golden = out_dir.join(format!(
        "run2-m5-golden-lora-r{rank}-init-cellL18toL14.json"
    ));
    std::fs::write(
        &init_golden,
        serde_json::to_string(&serde_json::json!({
            "artifact_file_sha256": init_hash, "input_seed_chacha8": GOLDEN_SEED,
            "n_pairs": GOLDEN_PAIRS, "inputs": init_in, "outputs": init_out,
            "outputs_are": "the adapter's DELTA for a single (1,1,hidden) residual row, not h+delta",
            "note": "SEEDED-INIT adapter. B is zero-initialised, so every output here is EXACTLY zero and the init adapter is the identity — that is the property, not a bug, and it is what makes 'LoRA at init' and 'no LoRA' the same function for the transfer check's baseline arm.",
        }))?,
    )?;
    println!(
        "init artifact: {} ({init_hash}), {} params",
        init_artifact.display(),
        lora.param_count()
    );

    // ---- Step-0 gradient gate --------------------------------------------
    // With B = 0 the gradient on A is EXACTLY zero (d(delta)/dA carries a
    // factor of B). Registered here rather than discovered: the gate requires
    // grads to EXIST and be finite on both Vars, and to be nonzero on B.
    let (a_grad0, b_grad0, loss0) = {
        let loss = item_loss(&model, &lora, &fit_items[0], &fit_payloads[0], &dev)?;
        let lv = loss.to_scalar::<f32>()?;
        let grads = loss.backward()?;
        let norm = |name: &str, v: &candle_core::Var| -> anyhow::Result<f32> {
            let g = grads.get(v.as_tensor()).ok_or_else(|| {
                anyhow::anyhow!("step-0 gate: NO gradient on {name} — graph silently cut")
            })?;
            let gn = g
                .to_dtype(DType::F32)?
                .sqr()?
                .sum_all()?
                .to_scalar::<f32>()?
                .sqrt();
            anyhow::ensure!(gn.is_finite(), "step-0 gate: {name} grad not finite ({gn})");
            Ok(gn)
        };
        let (ga, gb) = (norm("a", &lora.a)?, norm("b", &lora.b)?);
        anyhow::ensure!(
            gb > 0.0,
            "step-0 gate: gradient on b is zero — no learning signal"
        );
        println!(
            "step-0 grad gate: a {ga:.6e} (expected 0 at zero-B init), b {gb:.6e} (loss {lv:.4})"
        );
        (ga, gb, lv)
    };

    let init_holdout_ce = eval_ce(&model, &lora, &holdout_items, &holdout_payloads, &dev)?;
    println!("holdout gold-continuation CE at init: {init_holdout_ce:.6} nats/token");

    // ---- Training loop ----------------------------------------------------
    let mut opt = candle_nn::AdamW::new_lr(lora.vars(), LR)?;
    let mut curve: Vec<serde_json::Value> = Vec::new();
    let mut best: Option<(usize, f64, Vec<f32>)> = None;
    let mut order: Vec<usize> = (0..fit_items.len()).collect();
    let mut peak_vram = 0u64;
    for epoch in 0..epochs {
        let te = std::time::Instant::now();
        let mut rng = ChaCha8Rng::seed_from_u64(TRAIN_SEED ^ (0x5EED_0000 + epoch as u64));
        order.shuffle(&mut rng);
        let mut train_sum = 0f64;
        for (step, &i) in order.iter().enumerate() {
            let loss = item_loss(&model, &lora, &fit_items[i], &fit_payloads[i], &dev)?;
            train_sum += loss.to_scalar::<f32>()? as f64;
            opt.backward_step(&loss)?;
            if step % 64 == 0 {
                if let Some(v) = taskdata::process_vram_mib() {
                    peak_vram = peak_vram.max(v);
                }
            }
            if step % 500 == 0 {
                println!(
                    "epoch {epoch} step {step}/{}: mean train CE so far {:.6} ({:.0}s)",
                    order.len(),
                    train_sum / (step + 1) as f64,
                    t0.elapsed().as_secs_f32()
                );
            }
        }
        let train_ce = train_sum / order.len() as f64;
        let holdout_ce = eval_ce(&model, &lora, &holdout_items, &holdout_payloads, &dev)?;
        let (na, nb) = lora.factor_norms()?;
        println!(
            "epoch {epoch}: train CE {train_ce:.6}, holdout CE {holdout_ce:.6}, ||A|| {na:.5} \
             ||B|| {nb:.5} ({:.1}s, peak vram {peak_vram} MiB)",
            te.elapsed().as_secs_f32()
        );
        curve.push(serde_json::json!({
            "epoch": epoch, "train_ce_mean": train_ce, "holdout_ce": holdout_ce,
            "lora_a_l2": na, "lora_b_l2": nb, "wall_s": te.elapsed().as_secs_f32(),
        }));
        if best
            .as_ref()
            .map(|(_, b, _)| holdout_ce < *b)
            .unwrap_or(true)
        {
            best = Some((epoch, holdout_ce, lora.to_flat()?));
        }
    }
    let (best_epoch, best_holdout_ce, best_flat) = best.unwrap();
    println!(
        "best epoch {best_epoch}: holdout CE {best_holdout_ce:.6} (init {init_holdout_ce:.6})"
    );

    // ---- Trained artifact + goldens ---------------------------------------
    lora.set_flat(&best_flat, &dev)?;
    let artifact = out_dir.join(format!("{}.f32bin", name("")));
    let content_hash = lora.save_artifact(&artifact)?;
    let (golden_sha, inputs, outputs) = golden_pairs(
        &artifact,
        D_OUT,
        INJECT_AFTER_BLOCK,
        GOLDEN_SEED,
        GOLDEN_PAIRS,
    )?;
    anyhow::ensure!(golden_sha == content_hash, "artifact hash drift");
    let golden_path = out_dir.join(format!("run2-m5-golden-lora-r{rank}-cellL18toL14.json"));
    std::fs::write(
        &golden_path,
        serde_json::to_string(&serde_json::json!({
            "artifact_file_sha256": content_hash, "input_seed_chacha8": GOLDEN_SEED,
            "n_pairs": GOLDEN_PAIRS, "inputs": inputs, "outputs": outputs,
            "outputs_are": "the adapter's DELTA for a single (1,1,hidden) residual row, not h+delta",
            "note": "outputs produced by the TRAINED adapter itself, reloaded from disk through the RUNTIME's ResidualLora (the probe-side type) on CPU — so verifying them pins the probe's adapter to the trained one across the crate boundary",
        }))?,
    )?;
    println!("trained artifact: {} ({content_hash})", artifact.display());

    // ---- Training receipt (BEFORE the transfer check and the draw) --------
    let receipt = m5receipt::build(
        &m5receipt::TrainingReceipt {
            rank,
            cell: CELL,
            env: m5receipt::env_info(&nvcc),
            m3_hash: &m3_hash,
            m3_training_receipt: M3_TRAINING_RECEIPT,
            inject_mode: INJECT_MODE,
            n_slots: N_SLOTS,
            inject_after_block: INJECT_AFTER_BLOCK,
            hidden: D_OUT,
            seq_cap: SEQ_CAP,
            lr: LR,
            epochs,
            train_seed: TRAIN_SEED,
            golden_seed: GOLDEN_SEED,
            golden_pairs: GOLDEN_PAIRS,
            run_dir: run_dir.display().to_string(),
            verified: &verified,
            fit_items: fit_items.len(),
            fit_skipped: &fit_skipped,
            holdout_items: holdout_items.len(),
            holdout_skipped: &holdout_skipped,
            natural_median_stats: serde_json::json!(med_stats),
            min_target: MIN_TARGET,
            fit_mean_covered_fraction: fit_cov,
            fit_items_fully_covered: fit_full,
            holdout_mean_covered_fraction: hold_cov,
            holdout_items_fully_covered: hold_full,
            step0_loss: loss0,
            step0_grad_a: a_grad0,
            step0_grad_b: b_grad0,
            peak_vram_mib: peak_vram,
            curve,
            init_holdout_ce,
            best_epoch,
            best_holdout_ce,
            artifact_file: format!("{}.f32bin", name("")),
            content_hash,
            golden_file: format!("run2-m5-golden-lora-r{rank}-cellL18toL14.json"),
            golden_file_sha256: sha256_file(&golden_path)?,
            init_file: format!("{}.f32bin", name("init-")),
            init_hash,
            init_golden_file: format!("run2-m5-golden-lora-r{rank}-init-cellL18toL14.json"),
            init_golden_file_sha256: sha256_file(&init_golden)?,
            smoke,
            wall_clock_s: t0.elapsed().as_secs_f64(),
        },
        &lora,
        &split,
        &ds,
    );
    let receipt_path = out_dir.join(format!(
        "run2-m5-training-receipt-cellL18toL14-r{rank}.json"
    ));
    std::fs::write(&receipt_path, serde_json::to_string_pretty(&receipt)?)?;
    println!("training receipt: {}", receipt_path.display());
    println!("total wall clock: {:.1}s", t0.elapsed().as_secs_f64());
    Ok(())
}
