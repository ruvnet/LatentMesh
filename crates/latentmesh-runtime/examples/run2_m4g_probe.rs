//! Run-2 M4g — frozen-probe evaluation of the FUSE-trained task-loss MLP
//! adapter (ADR-024 § "M4g PRE-REGISTRATION (2026-08-29, before any run) —
//! fuse instead of overwrite"), run2_m4d_probe lineage.
//!
//! The probe protocol is ADR-023's, ABSOLUTELY FROZEN and inherited
//! unchanged (the same 40 GSM8K-train items — ChaCha8 seed 0x51A1, asserted
//! equal to the committed S1a receipt — one-sided exact sign test at α=0.05,
//! primary gate aligned_real > random, 8 slots, rescale-to-natural-median,
//! greedy/batch=1/max 400 tokens, four conditions). The per-item mechanics
//! are `common::m3`'s SHARED code paths, so the frozen protocol exists in
//! exactly one place and cannot silently diverge between rungs.
//!
//! **The ONE changed factor: the injection operator.** M3/M4/M4c/M4d (and all
//! of run 1) OVERWROTE the receiver's residual rows at the 8 placeholder
//! positions. M4g performs a **residual ADD**, `h[slot] += c·v`, preserving
//! the receiver's own state there — Cache-to-Cache's own fuser equation
//! (arXiv:2510.03215 Eq. 3). ADR-028 lists the injection operator as an
//! EVOLVABLE surface and this protocol as PROTECTED; the items, the four
//! condition definitions, the rescale switch, the decoding and the statistics
//! are untouched. ONE probe invocation, no second draw.
//!
//! **What the operator does to the controls, and what this probe measures
//! about it** (resolved and frozen in the training receipt's
//! `control_semantics_under_fuse` BEFORE this draw, echoed into this receipt,
//! and then measured here):
//!   * `aligned_real` and `random` — definitions AND meanings unchanged; the
//!     random control is still a norm-matched, information-free perturbation,
//!     so the PRIMARY statistic is unaffected.
//!   * `zerovec_injected` — definition unchanged (the true zero vector
//!     through the real 8-slot path), meaning CHANGED: `h += 0` is an exact
//!     no-op, so this condition collapses onto `baseline_uninjected`. The
//!     condition is still RUN, on all 40 items, and the collapse is MEASURED
//!     (`fuse_zero_is_noop_vs_baseline`) as an operator-correctness
//!     diagnostic. No substitute control is introduced — that would be an
//!     unregistered redefinition and a second changed factor.
//!   * the registered zerovec gate (`2 × zerovec ≥ baseline`) is therefore
//!     DEGENERATE under fuse. It is still computed, still reported, and
//!     explicitly labelled so no reader mistakes it for evidence.
//!
//! REPORTING: every sign-test line carries both the frozen exact sign p and
//! the one-sided **mid-p McNemar** value on the same pairs. ADR-024's M4g
//! pre-registration names mid-p as this rung's primary; the machine
//! `gate_pass` field stays on the ADR-028-protected exact sign test so it
//! remains comparable with every prior rung. Both are reported; neither is
//! selected after seeing the other.
//!
//! ORDERING GATES (refused before any model loads):
//!   1. artifact hash == the M4g training receipt's;
//!   2. the training receipt records the SAME injection operator this probe
//!      will use;
//!   3. the registered transfer check receipt exists with gate_pass=true and
//!      the SAME artifact hash.
//!
//! Run: PATH=/usr/local/cuda-12.8/bin:$PATH \
//!      cargo run --release --features cuda --example run2_m4g_probe

#[path = "common/mod.rs"]
#[allow(dead_code)]
mod common;

use common::m3::{
    run_item, Quad, Variant, ALPHA, GOLDEN_REL_TOL, ITEM_SEED, N_SLOTS, RANDVEC_SEED_BASE,
    RECEIVER, RECEIVER_BLOCK, SENDER, SENDER_BLOCK,
};
use common::mlp::MlpTransform;
use latentmesh_runtime::inject::InjectionMode;
use latentmesh_runtime::QwenRuntime;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::path::{Path, PathBuf};

const GSM8K_TRAIN_SHA256: &str = "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465";
const S1A_RECEIPT: &str = "receipts/s1a-receipt-slots8-block19-poolfull-rescaletrue-n40.json";
const TRAINING_RECEIPT: &str = "receipts/run2-m4g-training-receipt-cellL18toL14.json";
const TRANSFER_RECEIPT: &str = "receipts/run2-m4g-transfer-receipt-cellL18toL14.json";
const N_ITEMS: usize = 40;
/// **THE ONE CHANGED FACTOR** — pinned here, asserted against the training
/// receipt before any model loads, and threaded into the shared frozen
/// four-condition block.
const INJECT_MODE: InjectionMode = InjectionMode::Fuse;
/// Tolerance for the fuse zero-payload no-op diagnostic (nats). Reported,
/// never gating: `h += 0` is expected to be EXACTLY the uninjected forward.
const FUSE_NOOP_TOL: f32 = 1e-6;

fn crate_path(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

fn main() -> anyhow::Result<()> {
    let t0 = std::time::Instant::now();
    let nvcc = latentmesh_runtime::assert_cuda_build_env().map_err(anyhow::Error::msg)?;
    println!("build-env guard: {nvcc}");

    // ---- Gate 1: artifact hash vs the FROZEN M4d training receipt ---------
    let train_receipt: serde_json::Value =
        serde_json::from_slice(&std::fs::read(crate_path(TRAINING_RECEIPT))?)?;
    let expected_hash = train_receipt["artifact"]["content_hash_sha256"]
        .as_str()
        .expect("training receipt artifact.content_hash_sha256")
        .to_string();
    let train_mode = train_receipt["training"]["injection_operator"]["mode"]
        .as_str()
        .expect("training receipt training.injection_operator.mode");
    anyhow::ensure!(
        train_mode == INJECT_MODE.tag(),
        "training receipt records injection operator '{train_mode}' but this probe would run \
         '{}' — the adapter must be probed through the operator it was trained for",
        INJECT_MODE.tag()
    );
    let control_semantics = train_receipt["control_semantics_under_fuse"].clone();
    anyhow::ensure!(
        control_semantics["registered_before_the_probe"].as_bool() == Some(true),
        "the training receipt does not carry the control-semantics registration that must be \
         frozen BEFORE this draw"
    );
    println!(
        "injection operator: {} ({}) — asserted equal to the training receipt's",
        INJECT_MODE.tag(),
        INJECT_MODE.equation()
    );
    let artifact = crate_path("receipts/run2-m4g-mlp-fuse-cellL18toL14.f32bin");
    let transform = MlpTransform::load(&artifact)?;
    anyhow::ensure!(
        transform.content_hash == expected_hash,
        "artifact {} content_hash {} != training-receipt hash {expected_hash}",
        artifact.display(),
        transform.content_hash
    );
    let golden = crate_path("receipts/run2-m4g-golden-mlp-fuse-cellL18toL14.json");
    let (golden_n, golden_max_rel, golden_seed) =
        common::mlp::verify_against_golden(&transform, &golden, GOLDEN_REL_TOL)?;
    println!(
        "M4g artifact ({}): hand-rolled apply verified against {golden_n} trained-network \
         golden pairs, max relative L2 error {golden_max_rel:.3e} <= {GOLDEN_REL_TOL:.0e}",
        transform.content_hash
    );

    // ---- Gate 2: the registered transfer check must have PASSED -----------
    let transfer: serde_json::Value =
        serde_json::from_slice(&std::fs::read(crate_path(TRANSFER_RECEIPT)).map_err(|e| {
            anyhow::anyhow!(
                "transfer receipt {TRANSFER_RECEIPT} unreadable ({e}) — the frozen ordering is \
                 training -> transfer check -> probe; run run2_m4d_transfer_check first"
            )
        })?)?;
    anyhow::ensure!(
        transfer["gate_pass"].as_bool() == Some(true),
        "transfer check did NOT pass — per the frozen mitigation the probe must not run \
         (a null would be confounded by the composed<->fused BF16 gap)"
    );
    anyhow::ensure!(
        transfer["config"]["artifacts"]["trained"].as_str() == Some(expected_hash.as_str()),
        "transfer receipt evaluated a different artifact"
    );
    println!(
        "transfer gate: PASSED (fused NLL trained {} vs init {})",
        transfer["summary"]["mean_fused_nll_trained"], transfer["summary"]["mean_fused_nll_init"]
    );

    // ---- Dataset: pinned sha + the exact S1a item set ---------------------
    let dir = common::run_dir("run2-m4g");
    let data = dir.join("gsm8k-train.jsonl");
    let train_sha = common::fetch(common::GSM8K_TRAIN_URL, &data)?;
    anyhow::ensure!(train_sha == GSM8K_TRAIN_SHA256);
    let all_items = common::load_gsm8k(&data)?;
    let mut rng = ChaCha8Rng::seed_from_u64(ITEM_SEED);
    let mut indices = rand::seq::index::sample(&mut rng, all_items.len(), N_ITEMS).into_vec();
    indices.sort_unstable();
    let s1a: serde_json::Value = serde_json::from_slice(&std::fs::read(crate_path(S1A_RECEIPT))?)?;
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
    println!("item set: 40 indices identical to the committed S1a receipt");

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
            Variant::PerToken,
            INJECT_MODE,
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
    // Mid-p McNemar on the SAME pairs — reported only, gates nothing.
    let mid_primary = common::mid_p_one_sided(wins_rr, loss_rr);
    let nll_sign = |a: &dyn Fn(&Quad) -> f32, b: &dyn Fn(&Quad) -> f32| {
        let w = paired.iter().filter(|q| a(q) < b(q)).count();
        let l = paired.iter().filter(|q| a(q) > b(q)).count();
        (
            w,
            l,
            common::sign_test_one_sided(w, l),
            common::mid_p_one_sided(w, l),
        )
    };
    let nll_rr = nll_sign(&|q| q.real.1, &|q| q.rand.1);
    let nll_rz = nll_sign(&|q| q.real.1, &|q| q.zero.1);
    let nll_zb = nll_sign(&|q| q.zero.1, &|q| q.base.1);
    let mean = |f: &dyn Fn(&Quad) -> f32| paired.iter().map(f).sum::<f32>() / n.max(1) as f32;

    let receipt = serde_json::json!({
        "stage": "run2-M4d-deploymatch-probe",
        "design": "docs/adr/024-run2-trained-thought-adapter-ladder.md § 'Registered contingency — M4d, train/deploy configuration match' (registered 2026-08-29 BEFORE any M4d run); probe protocol = ADR-023's frozen S1a/S2b protocol inherited unchanged (never iterated, re-drawn, or re-tuned). M4d changed TRAINING only.",
        "env": common::env_info(&nvcc),
        "pre_committed": true,
        "variant": "pertoken-deploymatch",
        "config": {
            "sender": SENDER, "receiver": RECEIVER,
            "sender_capture_block": SENDER_BLOCK, "receiver_inject_block": RECEIVER_BLOCK,
            "cell": "L18->L14 (S2 winner; M4d registers the train/deploy configuration-match rung on this cell only)",
            "slots": N_SLOTS, "placeholder_token": "<|fim_pad|>", "placeholder_id": pad_id,
            "pool_span": "full generated span",
            "rescale_to_natural_median": true,
            "natural_norm_source": "receiver forward_capture over the slotted injection prompt at the inject block, per item (S0 cross-model precedent)",
            "decoding": "greedy, batch=1, max_new_tokens=400",
            "item_seed_chacha8": ITEM_SEED, "randvec_seed_base": RANDVEC_SEED_BASE,
            "transform": {
                "kind": "DEPLOY-MATCHED task-loss-trained M4d MLP 2048->512->1536 ReLU (M4c's architecture, seed, split, loss and schedule; the one changed factor is that training's rescale target came from the probe's OWN vendored-fused forward_capture + norms::stats and used InjectionSpec's operator order, gate-verified to <=1e-6)",
                "file": artifact.display().to_string(),
                "content_hash": transform.content_hash,
                "training_receipt": TRAINING_RECEIPT,
                "transfer_receipt": TRANSFER_RECEIPT,
                "variant": "per-token pathway ONLY (the pathway training optimized; frozen in the training receipt — no pooled-variant second draw)",
                "apply": "hand-rolled plain-Rust forward relu(x@W1+b1)@W2+b2 (latentmesh-train cannot be a path dep of runtime examples)",
                "hand_rolled_apply_verification": {
                    "golden_file": golden.display().to_string(),
                    "golden_pairs": golden_n,
                    "golden_input_seed_chacha8": golden_seed,
                    "max_relative_l2_error": golden_max_rel,
                    "tolerance": GOLDEN_REL_TOL,
                    "pass": true,
                },
            },
            "conditions": {
                "aligned_real": "sender per-token capture -> task-trained MLP per token -> pool -> 8-slot injection, rescaled to natural median",
                "random": "per-item seeded gaussian, norm-matched to the effective aligned_real vector, real path",
                "zerovec_injected": "TRUE ZERO VECTOR through the real 8-slot injection path (scale: None)",
                "baseline_uninjected": "no injection (spec=None), same prompt",
            },
            "primary_test": "one-sided exact sign test, paired accuracy, aligned_real > random, alpha 0.05 — THE recorded verdict statistic, frozen and unchanged",
            "reported_secondary_statistic": "one-sided mid-p McNemar (exact_p - 0.5*P(X=wins); Fagerland, Lydersen & Laake 2013; docs/research/031 §2.4, ADR-030's run-3 primary) computed on the SAME collected pairs and reported alongside every sign-test line. It gates nothing and changes no recorded verdict here; run 2's protocol statistic is ADR-028-protected.",
            "zerovec_gate": "pre-committed: pass iff 2 x zerovec accuracy >= baseline accuracy; numbers reported either way",
            "secondary_diagnostic": "one-sided sign tests on paired teacher-forced NLL of '#### <gold>'",
        },
        "dataset": {"source": common::GSM8K_TRAIN_URL, "sha256": train_sha, "indices": indices,
                     "s1a_indices_match": true},
        "items": rows,
        "summary": {
            "n_evaluated": n,
            "accuracy": {"aligned_real": real_c, "baseline_uninjected": base_c,
                          "zerovec_injected": zero_c, "random": rand_c},
            "primary_aligned_vs_random": {"wins": wins_rr, "losses": loss_rr,
                "n_discordant": wins_rr + loss_rr,
                "p_one_sided": p_primary, "alpha": ALPHA, "pass": primary_pass,
                "mid_p_one_sided_reported_only": mid_primary,
                "mid_p_pass_if_it_were_primary": mid_primary < ALPHA,
                "mid_p_note": "reported per ADR-030/research-031; NOT the gate — 'pass' above is the recorded verdict and is computed from p_one_sided alone",
                "min_attainable_p_at_this_n_disc": if wins_rr + loss_rr == 0 { 1.0 } else { 0.5f64.powi((wins_rr + loss_rr) as i32) }},
            "zerovec_vs_baseline": {
                "baseline_wins": zb_base_wins, "zerovec_wins": zb_zero_wins,
                "p_baseline_gt_zerovec": common::sign_test_one_sided(zb_base_wins, zb_zero_wins),
                "p_zerovec_gt_baseline": common::sign_test_one_sided(zb_zero_wins, zb_base_wins),
                "criterion": "2 x zerovec_correct >= baseline_correct",
                "pass": zerovec_pass},
            "aligned_vs_zerovec": {"wins": wins_rz, "losses": loss_rz,
                "p_one_sided": common::sign_test_one_sided(wins_rz, loss_rz),
                "mid_p_one_sided_reported_only": common::mid_p_one_sided(wins_rz, loss_rz)},
            "aligned_vs_baseline": {"wins": wins_rb, "losses": loss_rb,
                "p_one_sided": common::sign_test_one_sided(wins_rb, loss_rb),
                "mid_p_one_sided_reported_only": common::mid_p_one_sided(wins_rb, loss_rb)},
            "nll_mean": {"aligned_real": mean(&|q| q.real.1), "baseline_uninjected": mean(&|q| q.base.1),
                          "zerovec_injected": mean(&|q| q.zero.1), "random": mean(&|q| q.rand.1)},
            "nll_aligned_vs_random": {"wins": nll_rr.0, "losses": nll_rr.1,
                "p_one_sided": nll_rr.2, "mid_p_one_sided_reported_only": nll_rr.3},
            "nll_aligned_vs_zerovec": {"wins": nll_rz.0, "losses": nll_rz.1,
                "p_one_sided": nll_rz.2, "mid_p_one_sided_reported_only": nll_rz.3},
            "nll_zerovec_vs_baseline": {"wins": nll_zb.0, "losses": nll_zb.1,
                "p_one_sided": nll_zb.2, "mid_p_one_sided_reported_only": nll_zb.3},
        },
        "gates": {
            "artifact_hash_matches_training_receipt": {"pass": true, "hash": transform.content_hash},
            "hand_rolled_apply_matches_trained_network": {"pass": true, "max_relative_l2_error": golden_max_rel},
            "transfer_check_passed_before_probe": {"pass": true,
                "mean_fused_nll_trained": transfer["summary"]["mean_fused_nll_trained"],
                "mean_fused_nll_init": transfer["summary"]["mean_fused_nll_init"]},
            "s1a_item_set_reproduced": {"pass": true},
            "M4d_aligned_real_vs_random": {"pass": primary_pass, "p": p_primary},
            "zerovec_not_catastrophic": {"pass": zerovec_pass,
                "zerovec_correct": zero_c, "baseline_correct": base_c},
        },
        "gate_pass": primary_pass && zerovec_pass,
        "wall_clock_s": t0.elapsed().as_secs_f32(),
    });
    common::write_receipt(
        &crate_path("receipts"),
        "run2-m4d-receipt-cellL18toL14-mlp-deploymatch-slots8-poolfull-rescaletrue-n40.json",
        &receipt,
    )?;
    println!(
        "M4d[deploymatch/pertoken]: acc aligned {real_c}/{n} baseline {base_c}/{n} zerovec {zero_c}/{n} random {rand_c}/{n}"
    );
    println!(
        "primary p={p_primary:.4} (wins {wins_rr}, losses {loss_rr}) => pass={primary_pass} \
         [mid-p {mid_primary:.4}, reported only]; \
         zerovec {zero_c} vs baseline {base_c} => pass={zerovec_pass}"
    );
    Ok(())
}
