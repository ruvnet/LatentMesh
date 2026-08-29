//! Run-2 M4 — frozen-probe evaluation of the trained FastGRNN sequence
//! translator (ADR-024 § M4), run2_m3_probe lineage.
//!
//! The probe protocol is ADR-023's, ABSOLUTELY FROZEN and inherited
//! unchanged (ADR-024 § Frozen registration): the same 40 GSM8K-train items
//! (ChaCha8 seed 0x51A1, asserted equal to the committed S1a receipt), the
//! same one-sided exact sign test at α=0.05 (primary gate: aligned_real >
//! random), 8 slots, rescale-to-natural-median, greedy/batch=1/max 400
//! tokens, and the same four conditions — via the SAME extracted
//! `common::m3::{sender_solve_capture_rows, four_conditions}` code paths M3
//! ran. The ONLY change from the M3 per-token run is the transform: the
//! trained FastGRNN (h_0 = 0 over the full generated-span sequence, then
//! mean-pool of the TRANSLATED output sequence — the payload derivation
//! frozen in the M4 training receipt's eval plan) replaces the MLP.
//!
//! Sub-rung discipline (ADR-024's registered exception): `--rank` selects
//! one of the three pre-declared ranks {64, 128, 256}, probed in ascending
//! order, one probe run each, stopping at the first rank that clears the
//! gate; every attempted rank's receipt is kept regardless of outcome.
//!
//! The FastGRNN apply is HAND-ROLLED (`common::fastgrnn` — latentmesh-train
//! cannot be a path dep here) and verified against golden sequence pairs
//! from the trained network itself (every per-step output AND the pooled
//! payload, ≤1e-5 relative L2) before any model runs; the artifact hash is
//! asserted against that rank's M4 training receipt, written BEFORE any
//! invocation of this probe (the freeze point).
//!
//! Run: PATH=/usr/local/cuda-12.8/bin:$PATH \
//!      cargo run --release --features cuda --example run2_m4_probe -- --rank 64

#[path = "common/mod.rs"]
#[allow(dead_code)]
mod common;

use common::fastgrnn::FastGrnnTransform;
use common::m3::{
    four_conditions, sender_solve_capture_rows, CaptureMeta, Quad, ALPHA, GOLDEN_REL_TOL,
    ITEM_SEED, N_SLOTS, RANDVEC_SEED_BASE, RECEIVER, RECEIVER_BLOCK, SENDER, SENDER_BLOCK,
};
use latentmesh_runtime::{norms, QwenRuntime};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::path::{Path, PathBuf};

const GSM8K_TRAIN_SHA256: &str = "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465";
const S1A_RECEIPT: &str = "receipts/s1a-receipt-slots8-block19-poolfull-rescaletrue-n40.json";
/// The M3 probe/training receipts, loaded for the ADR-024-required
/// report-alongside-M3 comparison block.
const M3_PROBE_RECEIPTS: [&str; 2] = [
    "receipts/run2-m3-receipt-cellL18toL14-mlp-pertoken-slots8-poolfull-rescaletrue-n40.json",
    "receipts/run2-m3-receipt-cellL18toL14-mlp-pooled-slots8-poolfull-rescaletrue-n40.json",
];
const M3_TRAINING_RECEIPT: &str = "receipts/run2-m3-training-receipt-cellL18toL14.json";
/// Payload-derivation tag recorded per item (see the M4 training receipt's
/// frozen eval plan).
const VARIANT_TAG: &str = "seq-translate-then-pool";

#[derive(Debug, Clone)]
struct ProbeConfig {
    rank: usize,
    artifact: PathBuf,
    golden: PathBuf,
    n_items: usize,
}

fn crate_path(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

fn parse_args() -> ProbeConfig {
    let mut rank: Option<usize> = None;
    let mut artifact: Option<PathBuf> = None;
    let mut golden: Option<PathBuf> = None;
    let mut n_items = 40usize;
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        let next = |i: usize| args.get(i + 1).cloned().expect("missing arg value");
        match args[i].as_str() {
            "--rank" => rank = Some(next(i).parse().expect("--rank 64|128|256")),
            "--artifact" => artifact = Some(PathBuf::from(next(i))),
            "--golden" => golden = Some(PathBuf::from(next(i))),
            "--items" => n_items = next(i).parse().expect("--items N"),
            other => panic!("unknown arg {other}"),
        }
        i += 2;
    }
    let rank = rank.expect(
        "--rank 64|128|256 is required (no default: ADR-024's sub-rungs are probed in a pre-declared ascending order, one run each)",
    );
    assert!(
        [64usize, 128, 256].contains(&rank),
        "--rank must be 64|128|256"
    );
    ProbeConfig {
        rank,
        artifact: artifact.unwrap_or_else(|| {
            crate_path(&format!(
                "receipts/run2-m4-fastgrnn-r{rank}-cellL18toL14.f32bin"
            ))
        }),
        golden: golden.unwrap_or_else(|| {
            crate_path(&format!(
                "receipts/run2-m4-golden-fastgrnn-r{rank}-cellL18toL14.json"
            ))
        }),
        n_items,
    }
}

/// Summary extraction from a committed M3 probe receipt for the comparison
/// block (numbers come from the receipts themselves, never re-typed).
fn m3_summary(path: &Path) -> anyhow::Result<serde_json::Value> {
    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
    Ok(serde_json::json!({
        "receipt": path.file_name().and_then(|n| n.to_str()),
        "variant": v["variant"],
        "accuracy": v["summary"]["accuracy"],
        "primary_aligned_vs_random": v["summary"]["primary_aligned_vs_random"],
        "gate_pass": v["gate_pass"],
    }))
}

fn main() -> anyhow::Result<()> {
    let t0 = std::time::Instant::now();
    let cfg = parse_args();
    let nvcc = latentmesh_runtime::assert_cuda_build_env().map_err(anyhow::Error::msg)?;
    println!("build-env guard: {nvcc}\nconfig: {cfg:?}");

    // ---- Trained artifact: hash gate against THIS RANK's frozen training
    //      receipt (written before this probe — the freeze point) ----------
    let training_receipt_rel = format!(
        "receipts/run2-m4-training-receipt-cellL18toL14-r{}.json",
        cfg.rank
    );
    let train_receipt: serde_json::Value =
        serde_json::from_slice(&std::fs::read(crate_path(&training_receipt_rel))?)?;
    let expected_hash = train_receipt["artifact"]["content_hash_sha256"]
        .as_str()
        .expect("training receipt artifact.content_hash_sha256")
        .to_string();
    let transform = FastGrnnTransform::load(&cfg.artifact)?;
    anyhow::ensure!(
        transform.rank == cfg.rank,
        "artifact rank {} != requested {}",
        transform.rank,
        cfg.rank
    );
    anyhow::ensure!(
        transform.content_hash == expected_hash,
        "artifact {} content_hash {} != training-receipt hash {expected_hash}",
        cfg.artifact.display(),
        transform.content_hash
    );
    let (golden_seqs, golden_len, golden_max_rel, golden_seed) =
        common::fastgrnn::verify_against_golden(&transform, &cfg.golden, GOLDEN_REL_TOL)?;
    let pre_committed = cfg.n_items == 40;
    println!(
        "FastGRNN r={} artifact {} (content_hash {}): hand-rolled sequential apply verified \
         against {golden_seqs} golden sequences x {golden_len} steps + pooled payloads, max \
         relative L2 error {golden_max_rel:.3e} <= {GOLDEN_REL_TOL:.0e} (pre_committed={pre_committed})",
        cfg.rank,
        cfg.artifact.display(),
        transform.content_hash
    );

    // ---- Dataset: pinned sha + the exact S1a item set ---------------------
    let dir = common::run_dir("run2-m4");
    let data = dir.join("gsm8k-train.jsonl");
    let train_sha = common::fetch(common::GSM8K_TRAIN_URL, &data)?;
    anyhow::ensure!(train_sha == GSM8K_TRAIN_SHA256);
    let all_items = common::load_gsm8k(&data)?;
    let mut rng = ChaCha8Rng::seed_from_u64(ITEM_SEED);
    let mut indices = rand::seq::index::sample(&mut rng, all_items.len(), cfg.n_items).into_vec();
    indices.sort_unstable();
    let mut s1a_indices_match = false;
    if cfg.n_items == 40 {
        let s1a: serde_json::Value =
            serde_json::from_slice(&std::fs::read(crate_path(S1A_RECEIPT))?)?;
        let s1a_idx: Vec<usize> = s1a["dataset"]["indices"]
            .as_array()
            .expect("S1a receipt indices")
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        anyhow::ensure!(
            indices == s1a_idx,
            "derived indices differ from the S1a receipt"
        );
        s1a_indices_match = true;
        println!("item set: 40 indices identical to the committed S1a receipt");
    }

    let device = latentmesh_runtime::device().map_err(anyhow::Error::msg)?;
    println!("loading {SENDER} + {RECEIVER} (BF16, concurrent-resident)...");
    let mut sender =
        QwenRuntime::load(SENDER, &device, candle_core::DType::BF16).map_err(anyhow::Error::msg)?;
    let mut receiver = QwenRuntime::load(RECEIVER, &device, candle_core::DType::BF16)
        .map_err(anyhow::Error::msg)?;
    let pad_id = receiver
        .tokenizer
        .token_to_id("<|fim_pad|>")
        .ok_or_else(|| anyhow::anyhow!("<|fim_pad|> not in tokenizer"))?;
    println!("models loaded in {:.1}s", t0.elapsed().as_secs_f32());

    let mut rows = Vec::new();
    let mut paired: Vec<Quad> = Vec::new();
    for (done, &idx) in indices.iter().enumerate() {
        let item = &all_items[idx];
        match sender_solve_capture_rows(&mut sender, item, &device)? {
            Some(sr) => {
                anyhow::ensure!(sr.hidden_size == common::fastgrnn::D_IN);
                let aligned = transform.translate_seq_then_pool(&sr.rows, sr.n_rows);
                let meta = CaptureMeta {
                    hidden_size: sr.hidden_size,
                    pooled_l2_raw: norms::l2(&sr.pooled),
                    span: sr.span.clone(),
                    variant: VARIANT_TAG,
                };
                let (row, q) = four_conditions(
                    &mut receiver,
                    item,
                    pad_id,
                    &aligned,
                    &sr.pass,
                    &meta,
                    &device,
                )?;
                println!(
                    "[{}/{}] item {idx}: aligned={} baseline={} zerovec={} random={} (nll {:.3}/{:.3}/{:.3}/{:.3}) {:.0}s",
                    done + 1, indices.len(), q.real.0, q.base.0, q.zero.0, q.rand.0,
                    q.real.1, q.base.1, q.zero.1, q.rand.1, t0.elapsed().as_secs_f32()
                );
                rows.push(row);
                paired.push(q);
            }
            None => {
                println!(
                    "[{}/{}] item {idx}: degenerate sender capture pass, skipped",
                    done + 1,
                    indices.len()
                );
                rows.push(
                    serde_json::json!({"item": idx, "skipped": "degenerate sender capture pass"}),
                );
            }
        }
    }

    // ---- Pre-committed analysis (verbatim S2b/S1a/M3 frozen protocol) -----
    let n = paired.len();
    let count = |f: &dyn Fn(&Quad) -> bool| paired.iter().filter(|q| f(q)).count();
    let real_c = count(&|q| q.real.0);
    let base_c = count(&|q| q.base.0);
    let zero_c = count(&|q| q.zero.0);
    let rand_c = count(&|q| q.rand.0);
    let wins_rr = count(&|q| q.real.0 && !q.rand.0);
    let loss_rr = count(&|q| !q.real.0 && q.rand.0);
    let p_primary = common::sign_test_one_sided(wins_rr, loss_rr);
    let primary_pass = p_primary < ALPHA;
    let zb_base_wins = count(&|q| q.base.0 && !q.zero.0);
    let zb_zero_wins = count(&|q| q.zero.0 && !q.base.0);
    let zerovec_pass = 2 * zero_c >= base_c;
    let wins_rz = count(&|q| q.real.0 && !q.zero.0);
    let loss_rz = count(&|q| !q.real.0 && q.zero.0);
    let wins_rb = count(&|q| q.real.0 && !q.base.0);
    let loss_rb = count(&|q| !q.real.0 && q.base.0);
    let nll_sign = |a: &dyn Fn(&Quad) -> f32, b: &dyn Fn(&Quad) -> f32| {
        let w = paired.iter().filter(|q| a(q) < b(q)).count();
        let l = paired.iter().filter(|q| a(q) > b(q)).count();
        (w, l, common::sign_test_one_sided(w, l))
    };
    let nll_rr = nll_sign(&|q| q.real.1, &|q| q.rand.1);
    let nll_rz = nll_sign(&|q| q.real.1, &|q| q.zero.1);
    let nll_zb = nll_sign(&|q| q.zero.1, &|q| q.base.1);
    let mean = |f: &dyn Fn(&Quad) -> f32| paired.iter().map(f).sum::<f32>() / n.max(1) as f32;

    // ---- M3 comparison block (ADR-024 M4: report alongside M3's result) ---
    let m3_train: serde_json::Value =
        serde_json::from_slice(&std::fs::read(crate_path(M3_TRAINING_RECEIPT))?)?;
    let comparison_vs_m3 = serde_json::json!({
        "note": "ADR-024 M4 requires FastGRNN's result reported alongside M3's — M4 tests per-token sequence structure vs M3's position-independent mapping; numbers below are read from the committed M3 receipts, not re-typed",
        "m3_probe_results": M3_PROBE_RECEIPTS
            .iter()
            .map(|r| m3_summary(&crate_path(r)))
            .collect::<anyhow::Result<Vec<_>>>()?,
        "m3_training_best_holdout_rel_residual": m3_train["results"]["best_holdout_rel_residual_vs_fit_mean"],
        "m4_training_best_holdout_rel_residual": train_receipt["results"]["best_holdout_rel_residual_vs_fit_mean"],
        "m4_training_receipt": training_receipt_rel,
    });

    let receipt = serde_json::json!({
        "stage": "run2-M4-fastgrnn-probe",
        "design": "docs/adr/024-run2-trained-thought-adapter-ladder.md M4; probe protocol = ADR-023's frozen S1a/S2b protocol inherited unchanged (never iterated, re-drawn, or re-tuned); sub-rung ladder r in {64,128,256} ascending, one probe run per rank, stop at first pass, all receipts kept",
        "env": common::env_info(&nvcc),
        "pre_committed": pre_committed,
        "rank": cfg.rank,
        "config": {
            "sender": SENDER, "receiver": RECEIVER,
            "sender_capture_block": SENDER_BLOCK, "receiver_inject_block": RECEIVER_BLOCK,
            "cell": "L18->L14 (S2 winner, matching M3; anchor cell deliberately not probed — unregistered extra draw)",
            "slots": N_SLOTS, "placeholder_token": "<|fim_pad|>", "placeholder_id": pad_id,
            "pool_span": "full generated span",
            "rescale_to_natural_median": true,
            "natural_norm_source": "receiver forward_capture over the slotted injection prompt at the inject block, per item (S0 cross-model precedent)",
            "decoding": "greedy, batch=1, max_new_tokens=400",
            "item_seed_chacha8": ITEM_SEED, "randvec_seed_base": RANDVEC_SEED_BASE,
            "transform": {
                "kind": format!("trained M4 FastGRNN sequence translator, low-rank r={} (D_in 2048 -> D_h 1536, shared-W/U gated cell, Kusupati et al. arXiv:1901.02358; ADR-024 frozen architecture)", cfg.rank),
                "file": cfg.artifact.display().to_string(),
                "content_hash": transform.content_hash,
                "training_receipt": training_receipt_rel,
                "payload_derivation": "sequence-translate-then-pool (frozen in the training receipt's eval plan before this probe): FastGRNN consumes the sender's full generated-span per-token L18 sequence from h0=0, emits the translated sequence h_1..h_T, and the 8-slot payload is the mean-pool (f64 accumulation) of that TRANSLATED sequence — sequence processing upstream of the pool, the sequence analog of M3's variant (i), which is the pooling-hypothesis contrast M4 exists to test",
                "apply": "hand-rolled plain-Rust sequential cell forward (latentmesh-train cannot be a path dep of runtime examples)",
                "hand_rolled_apply_verification": {
                    "golden_file": cfg.golden.display().to_string(),
                    "golden_seqs": golden_seqs,
                    "golden_seq_len": golden_len,
                    "golden_input_seed_chacha8": golden_seed,
                    "max_relative_l2_error": golden_max_rel,
                    "tolerance": GOLDEN_REL_TOL,
                    "pass": true,
                    "note": "every per-step output AND the pooled injection payload verified against the trained network's own golden sequences (candle CPU forward at training time); asserted before any model ran",
                },
            },
            "conditions": {
                "aligned_real": "sender per-token capture -> trained FastGRNN over the sequence -> mean-pool of translated outputs -> 8-slot injection, rescaled to natural median",
                "random": "per-item seeded gaussian, norm-matched to the effective aligned_real vector, real path",
                "zerovec_injected": "TRUE ZERO VECTOR through the real 8-slot injection path (scale: None)",
                "baseline_uninjected": "no injection (spec=None), same prompt",
            },
            "primary_test": "one-sided exact sign test, paired accuracy, aligned_real > random, alpha 0.05",
            "zerovec_gate": "pre-committed: pass iff 2 x zerovec accuracy >= baseline accuracy; numbers reported either way",
            "secondary_diagnostic": "one-sided sign tests on paired teacher-forced NLL of '#### <gold>'",
        },
        "dataset": {"source": common::GSM8K_TRAIN_URL, "sha256": train_sha, "indices": indices,
                     "s1a_indices_match": s1a_indices_match},
        "items": rows,
        "summary": {
            "n_evaluated": n,
            "accuracy": {"aligned_real": real_c, "baseline_uninjected": base_c,
                          "zerovec_injected": zero_c, "random": rand_c},
            "primary_aligned_vs_random": {"wins": wins_rr, "losses": loss_rr,
                "p_one_sided": p_primary, "alpha": ALPHA, "pass": primary_pass},
            "zerovec_vs_baseline": {
                "baseline_wins": zb_base_wins, "zerovec_wins": zb_zero_wins,
                "p_baseline_gt_zerovec": common::sign_test_one_sided(zb_base_wins, zb_zero_wins),
                "p_zerovec_gt_baseline": common::sign_test_one_sided(zb_zero_wins, zb_base_wins),
                "criterion": "2 x zerovec_correct >= baseline_correct",
                "pass": zerovec_pass},
            "aligned_vs_zerovec": {"wins": wins_rz, "losses": loss_rz,
                "p_one_sided": common::sign_test_one_sided(wins_rz, loss_rz)},
            "aligned_vs_baseline": {"wins": wins_rb, "losses": loss_rb,
                "p_one_sided": common::sign_test_one_sided(wins_rb, loss_rb)},
            "nll_mean": {"aligned_real": mean(&|q| q.real.1), "baseline_uninjected": mean(&|q| q.base.1),
                          "zerovec_injected": mean(&|q| q.zero.1), "random": mean(&|q| q.rand.1)},
            "nll_aligned_vs_random": {"wins": nll_rr.0, "losses": nll_rr.1, "p_one_sided": nll_rr.2},
            "nll_aligned_vs_zerovec": {"wins": nll_rz.0, "losses": nll_rz.1, "p_one_sided": nll_rz.2},
            "nll_zerovec_vs_baseline": {"wins": nll_zb.0, "losses": nll_zb.1, "p_one_sided": nll_zb.2},
        },
        "comparison_vs_m3": comparison_vs_m3,
        "gates": {
            "artifact_hash_matches_training_receipt": {"pass": true, "hash": transform.content_hash},
            "hand_rolled_apply_matches_trained_network": {"pass": true, "max_relative_l2_error": golden_max_rel},
            "s1a_item_set_reproduced": {"pass": s1a_indices_match},
            "M4_aligned_real_vs_random": {"pass": primary_pass, "p": p_primary},
            "zerovec_not_catastrophic": {"pass": zerovec_pass,
                "zerovec_correct": zero_c, "baseline_correct": base_c},
        },
        "gate_pass": primary_pass && zerovec_pass,
        "wall_clock_s": t0.elapsed().as_secs_f32(),
    });
    // Receipts are written STRAIGHT into the committed receipts/ dir — the
    // crate target/ (where run_dir lives) is gitignored.
    let name = format!(
        "run2-m4-receipt-cellL18toL14-fastgrnn-r{}-slots8-poolfull-rescaletrue-n{}.json",
        cfg.rank, cfg.n_items
    );
    common::write_receipt(&crate_path("receipts"), &name, &receipt)?;
    println!(
        "M4[r={}]: acc aligned {real_c}/{n} baseline {base_c}/{n} zerovec {zero_c}/{n} random {rand_c}/{n}",
        cfg.rank
    );
    println!(
        "primary p={p_primary:.4} (wins {wins_rr}, losses {loss_rr}) => pass={primary_pass}; \
         zerovec {zero_c} vs baseline {base_c} => pass={zerovec_pass}"
    );
    Ok(())
}
