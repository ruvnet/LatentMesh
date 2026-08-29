//! Run-2 **M4h Stage 1** — frozen-probe evaluation of M3's already-trained,
//! on-manifold MLP with the payload **DE-POOLED** (last translated token
//! instead of the mean over the generated span) and delivered by **FUSE**
//! (ADR-024 § "M4h PRE-REGISTRATION (2026-08-29, before any run) —
//! de-pooling", Stage 1). `run2_m4g_probe` lineage.
//!
//! **Why this rung exists.** Three task-loss adapters (M4c, M4d, M4g) are all
//! off-manifold; three reconstruction adapters (M3, M4 r64/r128/r256) are
//! on-manifold but were delivered by OVERWRITE and were pooled. **On-manifold
//! AND fuse-delivered AND de-pooled has never been combined in one payload.**
//! Stage 1 combines all three at near-zero cost: NO new training and NO new
//! capture — M3's committed artifact, byte-for-byte, hash-asserted against the
//! M3 training receipt that was frozen long before this probe existed.
//!
//! **The changed factors, stated plainly.** Two things differ from M3's
//! committed `pertoken` receipt, and the ADR registers both together as one
//! rung because neither is testable alone at zero cost:
//!   1. **payload derivation** — `apply_rows_then_pool` → `apply_last_row`
//!      (`docs/research/040`: no externally successful cross-model method
//!      pools; C2C, LatentMAS, Bicameral and AVP are all per-token);
//!   2. **injection operator** — overwrite → `InjectionMode::Fuse`, the
//!      operator M4g built and M4g's receipts already characterise.
//!
//! Everything else is the ADR-023 frozen protocol, inherited unchanged: the
//! same 40 GSM8K-train items (ChaCha8 0x51A1, asserted equal to the committed
//! S1a receipt), **8 slots** (ADR-028-protected slot count — deliberately NOT
//! touched; see the note on ADR-028's own flagged contradiction below),
//! rescale-to-natural-median, greedy/batch=1/max 400 tokens, four conditions,
//! one-sided tests at α=0.05. The per-item mechanics are `common::m3`'s
//! SHARED code paths, so the frozen protocol exists in exactly one place.
//!
//! **Slot count is untouched, on purpose.** ADR-024 records that ADR-028 lists
//! "slot count" on BOTH sides of its own evolvable/protected boundary, and
//! that the contradiction is flagged but not adjudicated. M4h sidesteps it
//! entirely: the CONTENT of each slot changes, the NUMBER of slots does not
//! (8, asserted at run time by `four_conditions`).
//!
//! **Controls are M4g's, verbatim.** The three `InjectionSpec`s are built by
//! the same shared `common::m3::four_conditions` M4g ran, under the same
//! `InjectionMode::Fuse`, so `aligned_real` / `random` / `zerovec_injected` /
//! `baseline_uninjected` carry M4g's registered definitions AND M4g's
//! registered meanings without restatement. In particular `zerovec` is a
//! **true no-op** under fuse (`h += 0`) and therefore collapses onto
//! `baseline_uninjected`; that was declared before M4g's draw, verified there
//! at 40/40 bit-identical NLL, is re-measured here, and makes the registered
//! `2 × zerovec ≥ baseline` gate DEGENERATE — it is still computed, still
//! reported, and explicitly labelled as carrying no evidential weight. This
//! probe reads M4g's frozen `control_semantics_under_fuse` block out of M4g's
//! training receipt and echoes it into its own receipt rather than authoring a
//! second, possibly-drifting copy.
//!
//! **No transfer check, and why.** M4c/M4d/M4g each needed one because they
//! were TRAINED through a composed path that had to be reconciled with the
//! deployed fused BF16 path. Stage 1 performs **no training at all**: the
//! artifact is M3's, and M3's own probe had no transfer gate either. There is
//! nothing composed-vs-deployed to reconcile, so asserting one would be
//! theatre. What IS gated is the manifold pre-check (below), which must exist
//! and must have measured THIS payload derivation on THIS artifact hash.
//!
//! **The manifold pre-check runs BEFORE this probe and gates nothing.** Per
//! ADR-024's M4f framing the pre-check is diagnostic. Its receipt is required
//! to exist and to carry a row for this exact (artifact hash, derivation)
//! pair — that is an ORDERING gate, so the geometry is on record before the
//! verdict — but its classification never decides whether this draw happens.
//!
//! REPORTING: ADR-024's M4h registration names **one-sided mid-p McNemar** as
//! this rung's primary, with the exact sign test alongside; both are computed
//! on the same collected pairs and both appear below, neither selected after
//! seeing the other. The machine `gate_pass` field stays on the
//! ADR-028-protected exact sign test so this rung remains comparable with
//! every prior one. **`n_discordant` and the power floor `2^-n_disc` are
//! reported at the top of the summary**: M4d's n_disc=7 was the ladder's only
//! non-power-limited draw; M4g's n_disc=3 could not have rejected at any
//! outcome. Which category this draw falls into is stated in the receipt.
//!
//! ONE probe invocation. No second draw, no retry (ADR-032 honest-fail).
//!
//! Run: PATH=/usr/local/cuda-12.8/bin:$PATH \
//!      cargo run --release --features cuda --example run2_m4h_s1_probe

// The receipt is one large `serde_json::json!` literal (the frozen-receipt
// house style: one object, auditable top to bottom). It expands past rustc's
// default 128-deep macro recursion limit.
#![recursion_limit = "512"]

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
/// M3's training receipt — the artifact's freeze point, written 2026-08-28,
/// long before this rung was conceived.
const TRAINING_RECEIPT: &str = "receipts/run2-m3-training-receipt-cellL18toL14.json";
/// M4g's training receipt, read ONLY for its frozen control-semantics block.
const M4G_TRAINING_RECEIPT: &str = "receipts/run2-m4g-training-receipt-cellL18toL14.json";
/// The manifold pre-check receipt this rung's payload must already appear in.
const PRECHECK_RECEIPT: &str = "receipts/run2-m4h-s1-manifold-precheck-receipt.json";
/// The pre-check candidate label for this rung's payload derivation.
const PRECHECK_LABEL: &str = "m4h-s1-m3-mlp-lasttoken-depooled";
const ARTIFACT: &str = "receipts/run2-m3-mlp-cellL18toL14.f32bin";
const GOLDEN: &str = "receipts/run2-m3-golden-mlp-cellL18toL14.json";
const N_ITEMS: usize = 40;
/// **CHANGED FACTOR 1** — de-pooled payload derivation.
const VARIANT: Variant = Variant::PerTokenLast;
/// **CHANGED FACTOR 2** — M4g's residual-add operator.
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

    // ---- Gate 1: artifact hash vs M3's FROZEN training receipt ------------
    let train_receipt: serde_json::Value =
        serde_json::from_slice(&std::fs::read(crate_path(TRAINING_RECEIPT))?)?;
    let expected_hash = train_receipt["artifact"]["content_hash_sha256"]
        .as_str()
        .expect("training receipt artifact.content_hash_sha256")
        .to_string();
    let artifact = crate_path(ARTIFACT);
    let transform = MlpTransform::load(&artifact)?;
    anyhow::ensure!(
        transform.content_hash == expected_hash,
        "artifact {} content_hash {} != M3 training-receipt hash {expected_hash}",
        artifact.display(),
        transform.content_hash
    );
    let golden = crate_path(GOLDEN);
    let (golden_n, golden_max_rel, golden_seed) =
        common::mlp::verify_against_golden(&transform, &golden, GOLDEN_REL_TOL)?;
    println!(
        "M3 artifact ({}): hand-rolled apply verified against {golden_n} trained-network golden \
         pairs, max relative L2 error {golden_max_rel:.3e} <= {GOLDEN_REL_TOL:.0e}",
        transform.content_hash
    );
    println!(
        "payload derivation: {} (LAST translated token, no mean) | injection operator: {} ({})",
        VARIANT.tag(),
        INJECT_MODE.tag(),
        INJECT_MODE.equation()
    );

    // ---- Gate 2: M4g's frozen control semantics, reused verbatim ----------
    let m4g: serde_json::Value = serde_json::from_slice(
        &std::fs::read(crate_path(M4G_TRAINING_RECEIPT)).map_err(|e| {
            anyhow::anyhow!(
                "M4g training receipt {M4G_TRAINING_RECEIPT} unreadable ({e}) — this rung reuses \
                 M4g's frozen fuse control semantics rather than authoring a second copy"
            )
        })?,
    )?;
    let control_semantics = m4g["control_semantics_under_fuse"].clone();
    anyhow::ensure!(
        control_semantics["registered_before_the_probe"].as_bool() == Some(true),
        "M4g's training receipt does not carry the control-semantics registration"
    );
    anyhow::ensure!(
        m4g["training"]["injection_operator"]["mode"].as_str() == Some(INJECT_MODE.tag()),
        "M4g's receipt records a different injection operator than this probe runs"
    );
    println!(
        "control semantics: inherited VERBATIM from {M4G_TRAINING_RECEIPT} \
         (frozen before M4g's draw; zerovec is a true no-op under fuse)"
    );

    // ---- Gate 3 (ORDERING, not evidential): the manifold pre-check --------
    // The pre-check must already have measured THIS artifact under THIS
    // derivation, so the payload's geometry is on the record before the
    // verdict is drawn. Its classification decides nothing.
    let precheck: serde_json::Value =
        serde_json::from_slice(&std::fs::read(crate_path(PRECHECK_RECEIPT)).map_err(|e| {
            anyhow::anyhow!(
                "manifold pre-check receipt {PRECHECK_RECEIPT} unreadable ({e}) — ADR-024 M4h \
                 orders the pre-check BEFORE this draw; run run2_manifold_precheck first"
            )
        })?)?;
    let precheck_row = precheck["candidates"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("pre-check receipt has no candidates array"))?
        .iter()
        .find(|c| c["label"].as_str() == Some(PRECHECK_LABEL))
        .ok_or_else(|| {
            anyhow::anyhow!("pre-check receipt carries no '{PRECHECK_LABEL}' candidate row")
        })?
        .clone();
    anyhow::ensure!(
        precheck_row["gate"]["content_hash_sha256"].as_str() == Some(expected_hash.as_str()),
        "the pre-check measured a different artifact than this probe loads"
    );
    let precheck_class = precheck_row["classification"]
        .as_str()
        .unwrap_or("<unclassified>")
        .to_string();
    let precheck_manifold_cos =
        precheck_row["manifold"]["mean_cosine_to_same_item_natural_receiver_L14_pooled"].clone();
    let precheck_invariance =
        precheck_row["item_invariance"]["mean_pairwise_cosine_between_emitted_vectors"].clone();
    let precheck_entropy = precheck_row["entropy_nats"]["rmsnorm_lens_mean"].clone();
    println!(
        "manifold pre-check (diagnostic, non-gating): {precheck_class} — cosine-to-natural \
         {precheck_manifold_cos}, item-invariance {precheck_invariance}, entropy {precheck_entropy}"
    );

    // ---- Dataset: pinned sha + the exact S1a item set ---------------------
    let dir = common::run_dir("run2-m4h-s1");
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
            VARIANT,
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
    let n_disc = wins_rr + loss_rr;
    let p_primary = common::sign_test_one_sided(wins_rr, loss_rr);
    let primary_pass = p_primary < ALPHA;
    let mid_primary = common::mid_p_one_sided(wins_rr, loss_rr);
    let zb_base_wins = count(&|q| q.base.0 && !q.zero.0);
    let zb_zero_wins = count(&|q| q.zero.0 && !q.base.0);
    let zerovec_pass = 2 * zero_c >= base_c;
    let wins_rz = count(&|q| q.real.0 && !q.zero.0);
    let loss_rz = count(&|q| !q.real.0 && q.zero.0);
    let wins_rb = count(&|q| q.real.0 && !q.base.0);
    let loss_rb = count(&|q| !q.real.0 && q.base.0);

    // Power floor: the smallest one-sided p attainable at this many
    // discordant pairs. Reported first, exactly as ADR-024's M4h task frames
    // it, so a null is never read as stronger than the draw could support.
    let min_attainable_p = if n_disc == 0 {
        1.0
    } else {
        0.5f64.powi(n_disc as i32)
    };
    let power_limited = min_attainable_p >= ALPHA;

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
    let nll_rb = nll_sign(&|q| q.real.1, &|q| q.base.1);
    let nll_zb = nll_sign(&|q| q.zero.1, &|q| q.base.1);
    let mean = |f: &dyn Fn(&Quad) -> f32| paired.iter().map(f).sum::<f32>() / n.max(1) as f32;

    // ---- Fuse operator diagnostic (M4g's, unchanged) ----------------------
    let noop_acc_disagreements = count(&|q| q.zero.0 != q.base.0);
    let noop_max_abs_nll_delta = paired
        .iter()
        .map(|q| (q.zero.1 - q.base.1).abs())
        .fold(0f32, f32::max);
    let noop_exact_items = paired.iter().filter(|q| q.zero.1 == q.base.1).count();
    let noop_pass = noop_acc_disagreements == 0 && noop_max_abs_nll_delta <= FUSE_NOOP_TOL;

    // ---- NLL harm accounting -----------------------------------------------
    // CORRECTED FRAMING (ADR-024 § "MAJOR CORRECTION (2026-08-29): the '0/40
    // NLL inversion' is NOT ladder-wide"): the unanimous 0/40 inversion occurs
    // ONLY in the three OFF-MANIFOLD task-loss rungs (M4c 0W/40L, M4d 0W/40L,
    // M4g 0W/40L). Every ON-MANIFOLD rung — M3 per-token 20W/20L vs zerovec,
    // M4 r64 23W/17L, r256 21W/19L, the latter's mean NLL actually BELOW
    // baseline — is inert, not harmful. M4h Stage 1 runs M3's on-manifold
    // weights, so it was never in the inverting family: its question is
    // whether de-pooling adds BENEFIT, not whether it breaks an inversion.
    // These counts are therefore reported as harm accounting against the
    // on-manifold family's own band, not as an inversion test.
    let inversion_vs_baseline_items = paired.iter().filter(|q| q.real.1 > q.base.1).count();
    let inversion_vs_zerovec_items = paired.iter().filter(|q| q.real.1 > q.zero.1).count();
    let inversion_vs_random_items = paired.iter().filter(|q| q.real.1 > q.rand.1).count();
    let inversion_unanimous_vs_baseline = inversion_vs_baseline_items == n && n > 0;

    let receipt = serde_json::json!({
        "stage": "run2-M4h-stage1-depooled-fuse-probe",
        "design": "docs/adr/024-run2-trained-thought-adapter-ladder.md § 'M4h PRE-REGISTRATION (2026-08-29, before any run) — de-pooling', Stage 1; probe protocol = ADR-023's frozen S1a/S2b protocol inherited unchanged (never iterated, re-drawn, or re-tuned). Evidence basis for de-pooling: docs/research/040-the-pooling-gap.md.",
        "env": common::env_info(&nvcc),
        "pre_committed": true,
        "variant": "pertokenlast-fuse",
        "what_this_rung_combines": {
            "claim": "First payload in the ladder that is simultaneously ON-MANIFOLD, DE-POOLED and FUSE-DELIVERED.",
            "on_manifold": "M3's reconstruction-trained artifact — the manifold pre-check classifies the M3 family on-manifold and this rung's own derivation is measured in the pre-check receipt below",
            "de_pooled": "payload = LAST translated token of the generated span (apply_last_row), not the mean (apply_rows_then_pool). docs/research/040: C2C transfers per-token KV cache, LatentMAS concatenates full per-token per-layer caches, Bicameral couples live per-token states, AVP's cross-model path is per-token — no surveyed successful method pools; LatentMesh is the only surveyed design that does",
            "fuse_delivered": "LayerEdit::Fuse, h[slot] += c*v (C2C Eq.3) — the operator M4g built",
            "never_combined_before": "M3/M4 were on-manifold but pooled and overwrite-delivered; M4c/M4d/M4g were fuse- or overwrite-delivered but off-manifold and pooled",
        },
        "changed_factors_vs_m3_pertoken_receipt": {
            "count": 2,
            "factors": ["payload derivation: mean-pool over the translated span -> LAST translated token",
                         "injection operator: overwrite -> fuse (h[slot] += c*v)"],
            "why_two_and_not_one": "ADR-024 registers M4h Stage 1 as this combination, because the ladder's open question is precisely whether the three properties CO-OCCURRING matter; each half alone is already covered by a committed receipt (M3 pertoken = on-manifold+pooled+overwrite; M4g = fuse+off-manifold+pooled). Disclosed, not concealed: a null here cannot attribute blame between the two factors.",
        },
        "no_training_performed": {
            "trained_this_rung": false,
            "artifact_origin": "M3's committed adapter, unchanged; its training receipt was frozen before this rung was conceived",
            "no_transfer_check_and_why": "M4c/M4d/M4g each carried a transfer check because they were TRAINED through a composed path needing reconciliation with the deployed fused BF16 path. Stage 1 trains nothing, so there is no composed-vs-deployed gap to reconcile; M3's own probe likewise had no transfer gate. Asserting one here would be theatre.",
        },
        "slot_count_protected": {
            "slots": N_SLOTS,
            "changed": false,
            "note": "The CONTENT of each slot changes; the NUMBER does not. ADR-024 records that ADR-028 lists 'slot count' on BOTH sides of its evolvable/protected boundary and that the contradiction is flagged, not adjudicated. M4h stays clear of it by construction, and four_conditions asserts positions.len() == 8 at run time.",
        },
        "manifold_precheck_run_before_this_draw": {
            "receipt": PRECHECK_RECEIPT,
            "candidate_label": PRECHECK_LABEL,
            "artifact_hash_matches_this_probe": true,
            "classification": precheck_class,
            "mean_cosine_to_same_item_natural_receiver_L14_pooled": precheck_manifold_cos,
            "item_invariance_mean_pairwise_cosine": precheck_invariance,
            "entropy_nats_rmsnorm_lens": precheck_entropy,
            "gating": "NONE — ORDERING ONLY. Per ADR-024's M4f framing the pre-check is diagnostic; it is required to EXIST and to have measured this exact (artifact hash, derivation) pair before the draw, and it decides nothing about the verdict.",
            "scoping_caveat": "cosine-to-natural is measured against the POOLED natural receiver state, which ADR-024 notes is itself ~0.667 cosine from a real un-pooled receiver state. The pre-check's own un-pooled reference row (reference-receiver-L14-single-row) is the like-for-like comparator for a de-pooled payload.",
        },
        "control_semantics_under_fuse_inherited_verbatim_from_m4g": {
            "source_receipt": M4G_TRAINING_RECEIPT,
            "reused_not_restated": "This rung constructs its controls through the SAME shared common::m3::four_conditions code path M4g ran, under the same InjectionMode::Fuse. M4g's frozen block is echoed below rather than re-authored, so the definitions cannot drift between rungs.",
            "frozen_block": control_semantics,
        },
        "config": {
            "sender": SENDER, "receiver": RECEIVER,
            "sender_capture_block": SENDER_BLOCK, "receiver_inject_block": RECEIVER_BLOCK,
            "cell": "L18->L14 (S2 winner; M4h Stage 1 registers on this cell only)",
            "slots": N_SLOTS, "placeholder_token": "<|fim_pad|>", "placeholder_id": pad_id,
            "pool_span": "NONE — de-pooled. The payload is the last translated token of the generated span; the 8-slot broadcast of that single vector is unchanged.",
            "rescale_to_natural_median": true,
            "natural_norm_source": "receiver forward_capture over the slotted injection prompt at the inject block, per item (S0 cross-model precedent)",
            "decoding": "greedy, batch=1, max_new_tokens=400",
            "item_seed_chacha8": ITEM_SEED, "randvec_seed_base": RANDVEC_SEED_BASE,
            "injection_operator": {
                "mode": INJECT_MODE.tag(),
                "equation": INJECT_MODE.equation(),
                "prior_rungs": "overwrite, h[slot] = c*v (all of run 1, and run-2 M3 / M4 / M4c / M4d); fuse first used by M4g",
                "asserted_equal_to_m4g_receipt": true,
            },
            "transform": {
                "kind": "M3's ALREADY-TRAINED reconstruction MLP 2048->512->1536 ReLU — byte-identical weights, no retraining",
                "file": artifact.display().to_string(),
                "content_hash": transform.content_hash,
                "training_receipt": TRAINING_RECEIPT,
                "payload_derivation": "apply_last_row: relu(x@W1+b1)@W2+b2 on the LAST generated-span token state only",
                "apply": "hand-rolled plain-Rust forward (latentmesh-train cannot be a path dep of runtime examples)",
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
                "inherited": "All four condition definitions are M4g's, unchanged, produced by the same shared code path — see control_semantics_under_fuse_inherited_verbatim_from_m4g above.",
                "aligned_real": "sender per-token capture -> M3 MLP per token -> LAST token -> 8-slot delivery, rescaled to natural median, delivered by residual ADD.",
                "random": "per-item seeded gaussian, norm-matched to the effective aligned_real vector, real path. Under fuse a norm-matched, information-free PERTURBATION of the receiver's own rows — exactly the comparator the primary needs.",
                "zerovec_injected": "TRUE ZERO VECTOR through the real 8-slot path (scale: None). Under fuse this is h += 0, an exact NO-OP, so it collapses onto baseline_uninjected. Registered by M4g before M4g's draw, verified there, re-measured here.",
                "baseline_uninjected": "no injection (spec=None), same prompt.",
            },
            "primary_test": "ADR-024's M4h registration names one-sided MID-P McNEMAR as this rung's primary, with the frozen exact sign test reported alongside. BOTH are computed on the same collected pairs; neither was selected after seeing the other. gate_pass is computed from the ADR-028-protected EXACT sign test so this rung stays comparable with every prior one.",
            "zerovec_gate": "pre-committed: pass iff 2 x zerovec accuracy >= baseline accuracy. DEGENERATE UNDER FUSE (zerovec == baseline computation), so it is trivially satisfied and carries NO evidential weight; retained unchanged for cross-rung comparability and labelled here.",
            "secondary_diagnostic": "one-sided sign tests on paired teacher-forced NLL of '#### <gold>'",
        },
        "dataset": {"source": common::GSM8K_TRAIN_URL, "sha256": train_sha, "indices": indices,
                     "s1a_indices_match": true},
        "items": rows,
        "summary": {
            "n_evaluated": n,
            "power": {
                "n_discordant_primary": n_disc,
                "min_attainable_one_sided_p": min_attainable_p,
                "alpha": ALPHA,
                "power_limited": power_limited,
                "reading": if power_limited {
                    "POWER-LIMITED: at this many discordant pairs the smallest attainable one-sided p is >= alpha, so this draw was STRUCTURALLY INCAPABLE of rejecting the accuracy null regardless of outcome. Same category as M4g (n_disc=3). The accuracy result is therefore weak evidence; read the NLL numbers instead."
                } else {
                    "NOT power-limited: the smallest attainable one-sided p is below alpha, so this draw COULD have rejected the accuracy null. Same category as M4d (n_disc=7), the ladder's only prior non-power-limited draw."
                },
                "ladder_context": "M4d n_disc=7 (not power-limited); M4g n_disc=3 (power-limited, min p = 0.125 > alpha).",
            },
            "accuracy": {"aligned_real": real_c, "baseline_uninjected": base_c,
                          "zerovec_injected": zero_c, "random": rand_c},
            "primary_aligned_vs_random": {"wins": wins_rr, "losses": loss_rr,
                "n_discordant": n_disc,
                "p_one_sided": p_primary, "alpha": ALPHA, "pass": primary_pass,
                "mid_p_one_sided": mid_primary,
                "mid_p_pass": mid_primary < ALPHA,
                "adr_named_primary": "mid_p_one_sided",
                "gate_pass_computed_from": "p_one_sided (exact sign test, ADR-028-protected, for cross-rung comparability)",
                "min_attainable_p_at_this_n_disc": min_attainable_p},
            "nll_harm_accounting": {
                "question": "Does de-pooling an ON-MANIFOLD payload add BENEFIT on teacher-forced NLL? (NOT 'does it break the 0/40 inversion' — see corrected_framing.)",
                "corrected_framing": "ADR-024 § 'MAJOR CORRECTION (2026-08-29): the 0/40 NLL inversion is NOT ladder-wide' establishes from the committed receipts that the unanimous 0/40 inversion occurs ONLY in the three OFF-MANIFOLD task-loss rungs (M4c/M4d/M4g, all 0W/40L, aligned NLL 2.3-2.5x baseline). Every ON-MANIFOLD rung is inert, not harmful: M3 per-token 20W/20L vs zerovec at 2.1328 vs 2.1288 baseline; M4 r64 23W/17L; r256 21W/19L at 2.1074, BELOW baseline. M4h Stage 1 runs M3's on-manifold weights and so was never in the inverting family. Its comparator is the on-manifold family, not M4g.",
                "items_where_aligned_nll_is_WORSE_than_baseline": inversion_vs_baseline_items,
                "items_where_aligned_nll_is_WORSE_than_zerovec": inversion_vs_zerovec_items,
                "items_where_aligned_nll_is_WORSE_than_random": inversion_vs_random_items,
                "n_items": n,
                "unanimously_worse_than_baseline": inversion_unanimous_vs_baseline,
                "verdict": if inversion_unanimous_vs_baseline {
                    "UNANIMOUS HARM — aligned loses to baseline on every item. That would put this on-manifold payload in the off-manifold rungs' destructive class, which would itself be a new finding."
                } else if inversion_vs_baseline_items * 2 > n {
                    "NO BENEFIT, MILD HARM — aligned loses to baseline on a majority of items. Within the on-manifold family's inert band in magnitude, but on the wrong side of it: de-pooling did not add benefit."
                } else {
                    "BENEFIT ON THE MAJORITY — aligned beats baseline on more items than not, which no prior on-manifold rung achieved against BOTH baseline and zerovec."
                },
            },
            "fuse_zero_is_noop_vs_baseline": {
                "expected": "EXACT identity — under fuse the zerovec condition is h += 0",
                "accuracy_disagreements": noop_acc_disagreements,
                "nll_bit_identical_items": noop_exact_items,
                "n_items": n,
                "max_abs_nll_delta": noop_max_abs_nll_delta,
                "tolerance": FUSE_NOOP_TOL,
                "pass": noop_pass,
                "gating": "NONE — operator-correctness diagnostic only",
            },
            "zerovec_vs_baseline": {
                "degenerate_under_fuse": true,
                "baseline_wins": zb_base_wins, "zerovec_wins": zb_zero_wins,
                "p_baseline_gt_zerovec": common::sign_test_one_sided(zb_base_wins, zb_zero_wins),
                "p_zerovec_gt_baseline": common::sign_test_one_sided(zb_zero_wins, zb_base_wins),
                "criterion": "2 x zerovec_correct >= baseline_correct",
                "pass": zerovec_pass},
            "aligned_vs_zerovec": {"wins": wins_rz, "losses": loss_rz,
                "p_one_sided": common::sign_test_one_sided(wins_rz, loss_rz),
                "mid_p_one_sided": common::mid_p_one_sided(wins_rz, loss_rz)},
            "aligned_vs_baseline": {"wins": wins_rb, "losses": loss_rb,
                "p_one_sided": common::sign_test_one_sided(wins_rb, loss_rb),
                "mid_p_one_sided": common::mid_p_one_sided(wins_rb, loss_rb)},
            "nll_mean": {"aligned_real": mean(&|q| q.real.1), "baseline_uninjected": mean(&|q| q.base.1),
                          "zerovec_injected": mean(&|q| q.zero.1), "random": mean(&|q| q.rand.1)},
            "nll_aligned_vs_random": {"wins": nll_rr.0, "losses": nll_rr.1,
                "p_one_sided": nll_rr.2, "mid_p_one_sided": nll_rr.3},
            "nll_aligned_vs_zerovec": {"wins": nll_rz.0, "losses": nll_rz.1,
                "p_one_sided": nll_rz.2, "mid_p_one_sided": nll_rz.3},
            "nll_aligned_vs_baseline": {"wins": nll_rb.0, "losses": nll_rb.1,
                "p_one_sided": nll_rb.2, "mid_p_one_sided": nll_rb.3},
            "nll_zerovec_vs_baseline": {"wins": nll_zb.0, "losses": nll_zb.1,
                "p_one_sided": nll_zb.2, "mid_p_one_sided": nll_zb.3},
        },
        "gates": {
            "artifact_hash_matches_m3_training_receipt": {"pass": true, "hash": transform.content_hash},
            "hand_rolled_apply_matches_trained_network": {"pass": true, "max_relative_l2_error": golden_max_rel},
            "s1a_item_set_reproduced": {"pass": true},
            "manifold_precheck_ran_before_this_draw": {"pass": true, "receipt": PRECHECK_RECEIPT,
                "classification": precheck_class, "gating": "ordering only"},
            "m4g_control_semantics_inherited": {"pass": true, "source": M4G_TRAINING_RECEIPT},
            "slot_count_unchanged": {"pass": true, "slots": N_SLOTS},
            "M4h_s1_aligned_real_vs_random": {"pass": primary_pass, "p": p_primary,
                "mid_p": mid_primary, "n_discordant": n_disc,
                "min_attainable_p": min_attainable_p, "power_limited": power_limited},
            "zerovec_not_catastrophic": {"pass": zerovec_pass,
                "zerovec_correct": zero_c, "baseline_correct": base_c,
                "degenerate_under_fuse": true,
                "note": "trivially satisfied because zerovec == baseline under fuse; no evidential weight"},
            "fuse_zero_payload_is_a_noop": {"pass": noop_pass,
                "accuracy_disagreements": noop_acc_disagreements,
                "max_abs_nll_delta": noop_max_abs_nll_delta,
                "gating": "none — diagnostic"},
        },
        "gate_pass": primary_pass && zerovec_pass,
        "honest_fail_contract": "ADR-032: ONE registered draw, no retry, full numbers reported either way. This receipt is written before any interpretation is added to the ADR.",
        "wall_clock_s": t0.elapsed().as_secs_f32(),
    });
    common::write_receipt(
        &crate_path("receipts"),
        "run2-m4h-s1-receipt-cellL18toL14-mlp-pertokenlast-fuse-slots8-nopool-rescaletrue-n40.json",
        &receipt,
    )?;
    println!(
        "M4h-S1[depooled/fuse]: acc aligned {real_c}/{n} baseline {base_c}/{n} zerovec {zero_c}/{n} random {rand_c}/{n}"
    );
    println!(
        "primary p={p_primary:.4} (wins {wins_rr}, losses {loss_rr}, n_disc {n_disc}) => pass={primary_pass} \
         [mid-p {mid_primary:.4}, the ADR-named primary]; min attainable p at n_disc={n_disc} is \
         {min_attainable_p:.4} => power_limited={power_limited}"
    );
    println!(
        "fuse zero-payload no-op diagnostic: {noop_acc_disagreements} accuracy disagreements, \
         {noop_exact_items}/{n} bit-identical NLLs, max |dNLL| {noop_max_abs_nll_delta:.3e} \
         => {noop_pass} (reported, not gating)"
    );
    println!(
        "NLL means: aligned {:.4} baseline {:.4} zerovec {:.4} random {:.4}",
        mean(&|q| q.real.1),
        mean(&|q| q.base.1),
        mean(&|q| q.zero.1),
        mean(&|q| q.rand.1)
    );
    println!(
        "NLL sign tests: aligned-vs-baseline {}W/{}L, aligned-vs-zerovec {}W/{}L, \
         aligned-vs-random {}W/{}L; inversion vs baseline on {inversion_vs_baseline_items}/{n} items",
        nll_rb.0, nll_rb.1, nll_rz.0, nll_rz.1, nll_rr.0, nll_rr.1
    );
    Ok(())
}
