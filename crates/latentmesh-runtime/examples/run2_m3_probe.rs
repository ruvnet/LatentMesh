//! Run-2 M3 — frozen-probe evaluation of the trained MLP projector
//! (ADR-024 § M3), s2b_bridge_probe lineage.
//!
//! The probe protocol is ADR-023's, ABSOLUTELY FROZEN and inherited
//! unchanged (ADR-024 § Frozen registration): the same 40 GSM8K-train items
//! (ChaCha8 seed 0x51A1, asserted equal to the committed S1a receipt), the
//! same one-sided exact sign test at α=0.05 (primary gate: aligned_real >
//! random), 8 slots, rescale-to-natural-median, greedy/batch=1/max 400
//! tokens, and the same four conditions. The ONLY change from s2b is the
//! transform: the trained M3 MLP (2048→512→1536, ReLU) replaces the affine
//! map, in one of the two ADR-024-registered eval variants:
//!   --variant pertoken  (i)  MLP per generated-span token state, then the
//!                            existing pooling + 8-slot injection on the
//!                            TRANSLATED stream
//!   --variant pooled    (ii) pool first (run-1 shape), then the SAME
//!                            per-token-trained MLP on the pooled vector
//! Cell: the S2 winner L18→L14 only (ADR-024's M3 gate names variants, not
//! cells). The MLP apply is HAND-ROLLED (`common::mlp` — latentmesh-train
//! cannot be a path dep here) and verified against golden pairs from the
//! trained network itself (≤1e-5 relative L2) before any model runs; the
//! artifact hash is asserted against the M3 training receipt, written
//! BEFORE any invocation of this probe (the freeze point). Per-item
//! mechanics live in `common::m3` (file-size discipline only).
//!
//! Run: PATH=/usr/local/cuda-12.8/bin:$PATH \
//!      cargo run --release --features cuda --example run2_m3_probe -- --variant pertoken

#[path = "common/mod.rs"]
#[allow(dead_code)]
mod common;

use common::m3::{
    run_item, Quad, Variant, ALPHA, GOLDEN_REL_TOL, ITEM_SEED, N_SLOTS, RANDVEC_SEED_BASE,
    RECEIVER, RECEIVER_BLOCK, SENDER, SENDER_BLOCK,
};
use common::mlp::MlpTransform;
use latentmesh_runtime::QwenRuntime;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::path::{Path, PathBuf};

const GSM8K_TRAIN_SHA256: &str = "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465";
const S1A_RECEIPT: &str = "receipts/s1a-receipt-slots8-block19-poolfull-rescaletrue-n40.json";
const TRAINING_RECEIPT: &str = "receipts/run2-m3-training-receipt-cellL18toL14.json";

#[derive(Debug, Clone)]
struct ProbeConfig {
    variant: Variant,
    artifact: PathBuf,
    golden: PathBuf,
    n_items: usize,
}

fn crate_path(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

fn parse_args() -> ProbeConfig {
    let mut variant: Option<Variant> = None;
    let mut cfg = ProbeConfig {
        variant: Variant::PerToken,
        artifact: crate_path("receipts/run2-m3-mlp-cellL18toL14.f32bin"),
        golden: crate_path("receipts/run2-m3-golden-mlp-cellL18toL14.json"),
        n_items: 40,
    };
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        let next = |i: usize| args.get(i + 1).cloned().expect("missing arg value");
        match args[i].as_str() {
            "--variant" => {
                variant = Some(match next(i).as_str() {
                    "pertoken" => Variant::PerToken,
                    "pooled" => Variant::Pooled,
                    other => panic!("--variant must be pertoken|pooled, got {other}"),
                })
            }
            "--artifact" => cfg.artifact = PathBuf::from(next(i)),
            "--golden" => cfg.golden = PathBuf::from(next(i)),
            "--items" => cfg.n_items = next(i).parse().expect("--items N"),
            other => panic!("unknown arg {other}"),
        }
        i += 2;
    }
    cfg.variant = variant.expect(
        "--variant pertoken|pooled is required (no default: the two ADR-024 variants are evaluated separately)",
    );
    cfg
}

fn main() -> anyhow::Result<()> {
    let t0 = std::time::Instant::now();
    let cfg = parse_args();
    let nvcc = latentmesh_runtime::assert_cuda_build_env().map_err(anyhow::Error::msg)?;
    println!("build-env guard: {nvcc}\nconfig: {cfg:?}");

    // ---- Trained artifact: hash gate against the FROZEN training receipt --
    let train_receipt: serde_json::Value =
        serde_json::from_slice(&std::fs::read(crate_path(TRAINING_RECEIPT))?)?;
    let expected_hash = train_receipt["artifact"]["content_hash_sha256"]
        .as_str()
        .expect("training receipt artifact.content_hash_sha256")
        .to_string();
    let transform = MlpTransform::load(&cfg.artifact)?;
    anyhow::ensure!(
        transform.content_hash == expected_hash,
        "artifact {} content_hash {} != training-receipt hash {expected_hash}",
        cfg.artifact.display(),
        transform.content_hash
    );
    let (golden_n, golden_max_rel, golden_seed) =
        common::mlp::verify_against_golden(&transform, &cfg.golden, GOLDEN_REL_TOL)?;
    let pre_committed = cfg.n_items == 40;
    println!(
        "MLP artifact {} (content_hash {}): hand-rolled apply verified against {golden_n} \
         trained-network golden pairs, max relative L2 error {golden_max_rel:.3e} <= {GOLDEN_REL_TOL:.0e} \
         (pre_committed={pre_committed})",
        cfg.artifact.display(),
        transform.content_hash
    );

    // ---- Dataset: pinned sha + the exact S1a item set ---------------------
    let dir = common::run_dir("run2-m3");
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
        match run_item(
            &mut sender,
            &mut receiver,
            &transform,
            item,
            pad_id,
            cfg.variant,
            &device,
        )? {
            Some((row, q)) => {
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

    // ---- Pre-committed analysis (verbatim S2b/S1a frozen protocol) --------
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

    let variant_desc = match cfg.variant {
        Variant::PerToken => "(i) per-token translator: trained MLP applied to each generated-span token state, then mean-pooling over the TRANSLATED stream, then the frozen 8-slot injection",
        Variant::Pooled => "(ii) pooled-in/pooled-out: sender per-token states mean-pooled first (run-1 pipeline shape), then the SAME per-token-trained MLP on the pooled vector (ADR-024 authorial choice: shared weights; off-distribution confound disclosed in ADR-024)",
    };
    let receipt = serde_json::json!({
        "stage": "run2-M3-mlp-probe",
        "design": "docs/adr/024-run2-trained-thought-adapter-ladder.md M3; probe protocol = ADR-023's frozen S1a/S2b protocol inherited unchanged (never iterated, re-drawn, or re-tuned)",
        "env": common::env_info(&nvcc),
        "pre_committed": pre_committed,
        "variant": cfg.variant.tag(),
        "config": {
            "sender": SENDER, "receiver": RECEIVER,
            "sender_capture_block": SENDER_BLOCK, "receiver_inject_block": RECEIVER_BLOCK,
            "cell": "L18->L14 (S2 winner; ADR-024 M3 registers variants, not an anchor-cell run)",
            "slots": N_SLOTS, "placeholder_token": "<|fim_pad|>", "placeholder_id": pad_id,
            "pool_span": "full generated span",
            "rescale_to_natural_median": true,
            "natural_norm_source": "receiver forward_capture over the slotted injection prompt at the inject block, per item (S0 cross-model precedent)",
            "decoding": "greedy, batch=1, max_new_tokens=400",
            "item_seed_chacha8": ITEM_SEED, "randvec_seed_base": RANDVEC_SEED_BASE,
            "transform": {
                "kind": "trained M3 MLP projector 2048->512->1536 ReLU (ADR-024 frozen architecture)",
                "file": cfg.artifact.display().to_string(),
                "content_hash": transform.content_hash,
                "training_receipt": TRAINING_RECEIPT,
                "variant": variant_desc,
                "apply": "hand-rolled plain-Rust forward relu(x@W1+b1)@W2+b2 (latentmesh-train cannot be a path dep of runtime examples)",
                "hand_rolled_apply_verification": {
                    "golden_file": cfg.golden.display().to_string(),
                    "golden_pairs": golden_n,
                    "golden_input_seed_chacha8": golden_seed,
                    "max_relative_l2_error": golden_max_rel,
                    "tolerance": GOLDEN_REL_TOL,
                    "pass": true,
                    "note": "golden outputs produced by the trained network itself (candle CPU forward at training time); asserted before any model ran",
                },
            },
            "conditions": {
                "aligned_real": "sender per-token capture -> trained MLP (per this run's variant) -> 8-slot injection, rescaled to natural median",
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
        "gates": {
            "artifact_hash_matches_training_receipt": {"pass": true, "hash": transform.content_hash},
            "hand_rolled_apply_matches_trained_network": {"pass": true, "max_relative_l2_error": golden_max_rel},
            "s1a_item_set_reproduced": {"pass": s1a_indices_match},
            "M3_aligned_real_vs_random": {"pass": primary_pass, "p": p_primary},
            "zerovec_not_catastrophic": {"pass": zerovec_pass,
                "zerovec_correct": zero_c, "baseline_correct": base_c},
        },
        "gate_pass": primary_pass && zerovec_pass,
        "wall_clock_s": t0.elapsed().as_secs_f32(),
    });
    let name = format!(
        "run2-m3-receipt-cellL18toL14-mlp-{}-slots8-poolfull-rescaletrue-n{}.json",
        cfg.variant.tag(),
        cfg.n_items
    );
    common::write_receipt(&dir, &name, &receipt)?;
    println!(
        "M3[{}]: acc aligned {real_c}/{n} baseline {base_c}/{n} zerovec {zero_c}/{n} random {rand_c}/{n}",
        cfg.variant.tag()
    );
    println!(
        "primary p={p_primary:.4} (wins {wins_rr}, losses {loss_rr}) => pass={primary_pass}; \
         zerovec {zero_c} vs baseline {base_c} => pass={zerovec_pass}"
    );
    Ok(())
}
