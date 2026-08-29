//! Run-2 **PC1** — the positive control this harness never had.
//!
//! ADR-024 § "PC1 PRE-REGISTRATION (2026-08-29) — the positive control this
//! harness never had"; design `docs/research/045-positive-control-design.md`
//! §2 candidate **(c)** and §5.
//!
//! **Why this rung exists.** Every null since S1a is ambiguous between "no
//! transfer effect exists" and "these mechanics cannot carry signal". S1a —
//! the ladder's only PASS (accuracy 25/20, 5W/0L, p = 0.03125) — ran under
//! **overwrite + pooled + inject-at-block-19**, mechanics **nothing since M4c
//! uses**. No positive control has ever run under **fuse**, **de-pooled
//! payloads**, or the current **L18→L14** injection site. This rung runs one,
//! at the cheapest point on the whole cost ledger, and it gates the expensive
//! rungs (M4b ≈1-2 GPU-h, M5X ≈1.3-1.4 GPU-h) that would otherwise produce
//! uninterpretable nulls.
//!
//! **What is injected.** The receiver's **own** block-19 residual state,
//! captured while teacher-forced over the item's **GSM8K gold solution**,
//! de-pooled to the span's LAST token, delivered back into the receiver by an
//! **IDENTITY transform** — no adapter, no fitted alignment, no weights of any
//! kind are loaded by this binary. Mechanics are the CURRENT ones, unchanged:
//! **fuse** (`h[slot] += c·v`), **8 `<|fim_pad|>` slots**, **injection at
//! receiver block 14** (the L18→L14 cell's receiver half),
//! rescale-to-natural-median, greedy, batch 1, max 400 new tokens, the same
//! four paired conditions produced by the SAME shared
//! `common::m3::four_conditions_at` every rung since M3 has used.
//!
//! **FIREWALL — `docs/research/045` §3, verbatim, and repeated in the
//! receipt:** *"This control used a same-model, same-item, identity-transform
//! injection with gold-adjacent content. It tests whether this repository's
//! injection mechanics (delivery operator, payload shape, injection site,
//! norm-rescale) are capable of carrying signal at all. It does not test, and
//! must not be cited as evidence for or against, whether a cross-model-derived,
//! learned-alignment payload can transfer reasoning content — that is exactly
//! the separate question M3 through M5X exist to answer."*
//! **A PASS proves LIVENESS ONLY, NEVER TRANSFER.**
//!
//! **A FAIL is the more consequential outcome and is reported plainly.**
//! ADR-024's registered FAIL branch: every null since S1a (S2b ×4, M3 ×2,
//! M4 ×3, M4c, M4d, M4g, M4h S1, M4i) needs an explicit caveat that the
//! current mechanics were never shown capable of carrying signal even from a
//! model to itself; S1a's PASS gains a footnote scoping it to mechanics no
//! longer in use; and ADR-024's "adapters improved, so discordance fell"
//! explanation acquires an unruled-out rival — that fuse + de-pooling + site
//! may suppress discordance **mechanically**, independent of adapter quality.
//!
//! **The statistic is S1a's, deliberately.** ADR-024's PC1 registration names
//! the original frozen **40-item one-sided exact sign test** — chosen for
//! direct comparability with the ladder's only PASS on identical footing, NOT
//! because the e-process (ADR-036) is unavailable. Mid-p McNemar is reported
//! alongside on the same collected pairs, and `n_discordant` is reported
//! against the `2^-n_disc` power floor at the top of the summary, so a null is
//! never read as stronger than the draw could support.
//!
//! **Gaming guard (`docs/research/045` §3, mandatory on every positive-control
//! draw).** ITI (arXiv:2306.03341) documents the failure mode where an
//! intervention appears to restore the answer while merely collapsing the
//! model into a degenerate response ("it is trivial to attain a perfect
//! truthfulness score simply by answering 'no comment'"). Two pre-committed
//! flags below detect this repo's version of it — degenerate-short outputs and
//! an NLL collapse that means the payload handed over the answer rather than
//! the receiver computing it. **A PASS carrying either flag is reported as a
//! GAMED pass, not a clean one.**
//!
//! ONE probe invocation. No second draw, no retry (ADR-032 honest-fail).
//!
//! Run: PATH=/usr/local/cuda-12.8/bin:$PATH \
//!      cargo run --release --features cuda --example run2_pc1_probe

#![recursion_limit = "512"]

#[path = "common/mod.rs"]
#[allow(dead_code)]
mod common;

use common::m3::{
    four_conditions_at, CaptureMeta, Quad, SenderPass, Site, ALPHA, ITEM_SEED, N_SLOTS, RECEIVER,
    RECEIVER_BLOCK,
};
use latentmesh_runtime::inject::InjectionMode;
use latentmesh_runtime::{norms, QwenRuntime};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::path::{Path, PathBuf};

const GSM8K_TRAIN_SHA256: &str = "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465";
const S1A_RECEIPT: &str = "receipts/s1a-receipt-slots8-block19-poolfull-rescaletrue-n40.json";
const CAPTURE_RECEIPT: &str = "receipts/run2-pc1-capture-receipt.json";
const PRECHECK_RECEIPT: &str = "receipts/run2-pc1-manifold-precheck-receipt.json";
const PRECHECK_LABEL: &str = "pc1-payload-receiver-L19-lasttoken-goldteacherforced";
const D_RECEIVER: usize = 1536;
const N_ITEMS: usize = 40;
const VECS_PER_ITEM: usize = 3;
/// Current mechanics, unchanged from M4h Stage 1 onward.
const INJECT_MODE: InjectionMode = InjectionMode::Fuse;
const SITE: Site = Site::FimPadSlots;
/// Tolerance for the fuse zero-payload no-op diagnostic (nats).
const FUSE_NOOP_TOL: f32 = 1e-6;

// --- Pre-committed gaming-guard thresholds (docs/research/045 §3) ----------
/// An aligned generation shorter than this fraction of its OWN baseline
/// generation is a candidate degenerate/echo output.
const DEGENERATE_LEN_RATIO: f64 = 0.25;
/// If at least this fraction of items are degenerate-short, the draw carries
/// the degenerate-output signature.
const DEGENERATE_ITEM_FRACTION: f64 = 0.25;
/// Mean aligned teacher-forced NLL of `"#### <gold>"` below this (nats) means
/// the payload handed the receiver the answer outright — roughly >90%
/// probability mass on the gold string — rather than the receiver computing
/// it. Deliberately strict: for a gold-derived payload this is the expected
/// direction, and §3 requires it to be FLAGGED, not excused.
const NLL_COLLAPSE_NATS: f32 = 0.10;

fn crate_path(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

#[allow(clippy::too_many_lines)]
fn main() -> anyhow::Result<()> {
    let t0 = std::time::Instant::now();
    let e = anyhow::Error::msg;
    let nvcc = latentmesh_runtime::assert_cuda_build_env().map_err(e)?;
    println!("build-env guard: {nvcc}");

    // ---- Gate 1: the capture receipt and its sha-pinned payload file ------
    let cap: serde_json::Value =
        serde_json::from_slice(&std::fs::read(crate_path(CAPTURE_RECEIPT)).map_err(|err| {
            anyhow::anyhow!(
                "capture receipt {CAPTURE_RECEIPT} unreadable ({err}) — run run2_pc1_capture first"
            )
        })?)?;
    for gate in [
        "receiver_L19_dump_sha256_verified",
        "s1a_item_set_reproduced",
        "identity_transform_no_adapter_weights_loaded",
        "capture_path_reproduces_the_committed_dump",
        "all_captured_states_finite",
    ] {
        anyhow::ensure!(
            cap["gates"][gate]["pass"].as_bool() == Some(true),
            "capture receipt gate {gate} did not pass"
        );
    }
    let dump_sha = cap["gates"]["receiver_L19_dump_sha256_verified"]["measured"]
        .as_str()
        .unwrap_or_default()
        .to_string();
    let payload_path = PathBuf::from(
        cap["payload_file"]["path"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("capture receipt does not name the payload file"))?,
    );
    let want_payload_sha = cap["payload_file"]["sha256"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("capture receipt does not pin the payload sha256"))?
        .to_string();
    let bytes = std::fs::read(&payload_path)?;
    let payload_sha = common::sha256_hex(&bytes);
    anyhow::ensure!(
        payload_sha == want_payload_sha,
        "payload sha256 {payload_sha} != capture-receipt-pinned {want_payload_sha}"
    );
    anyhow::ensure!(
        bytes.len() == N_ITEMS * VECS_PER_ITEM * D_RECEIVER * 4,
        "payload file size mismatch"
    );
    let flat: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    // The PAYLOAD is vector 0 of each item's triple: the block-19 last row.
    let payloads: Vec<Vec<f32>> = (0..N_ITEMS)
        .map(|i| {
            let b = i * VECS_PER_ITEM * D_RECEIVER;
            flat[b..b + D_RECEIVER].to_vec()
        })
        .collect();
    println!("payload VERIFIED: sha {payload_sha} ({N_ITEMS} x {D_RECEIVER} f32, identity)");

    // ---- Gate 2 (ORDERING, not evidential): the manifold pre-check --------
    let precheck: serde_json::Value =
        serde_json::from_slice(&std::fs::read(crate_path(PRECHECK_RECEIPT)).map_err(|err| {
            anyhow::anyhow!(
                "manifold pre-check receipt {PRECHECK_RECEIPT} unreadable ({err}) — ADR-024 orders \
                 the pre-check BEFORE this draw; run run2_pc1_manifold_precheck first"
            )
        })?)?;
    anyhow::ensure!(
        precheck["payload_source"]["sha256"].as_str() == Some(payload_sha.as_str()),
        "the pre-check measured a different payload than this probe loads"
    );
    let precheck_row = precheck["candidates"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("pre-check receipt has no candidates array"))?
        .iter()
        .find(|c| c["label"].as_str() == Some(PRECHECK_LABEL))
        .ok_or_else(|| anyhow::anyhow!("pre-check carries no '{PRECHECK_LABEL}' row"))?
        .clone();
    let precheck_class = precheck_row["classification"]
        .as_str()
        .unwrap_or("<unclassified>")
        .to_string();
    let precheck_surprise = precheck["registered_expectation_stated_before_the_numbers"]
        ["surprise_flag"]
        .as_bool()
        .unwrap_or(true);
    println!(
        "manifold pre-check (diagnostic, non-gating): {precheck_class} — surprise={precheck_surprise}"
    );

    // ---- Gate 3: S1a's exact 40-item set ----------------------------------
    let dir = common::run_dir("run2-pc1");
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
        "derived indices differ from S1a's receipt"
    );
    let cap_idx: Vec<usize> = cap["dataset"]["indices"]
        .as_array()
        .expect("capture receipt indices")
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    anyhow::ensure!(
        indices == cap_idx,
        "the capture receipt's items differ from this probe's"
    );
    println!("item set: 40 indices identical to the committed S1a receipt");

    // ---- Model: receiver ONLY (PC1 is a same-model self-pair) -------------
    let device = latentmesh_runtime::device().map_err(e)?;
    println!("loading {RECEIVER} (BF16)... [no sender, no adapter]");
    let mut receiver = QwenRuntime::load(RECEIVER, &device, candle_core::DType::BF16).map_err(e)?;
    let pad_id = receiver
        .tokenizer
        .token_to_id("<|fim_pad|>")
        .ok_or_else(|| anyhow::anyhow!("<|fim_pad|> not in tokenizer"))?;
    println!("model loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // ---- The draw ---------------------------------------------------------
    let mut rows = Vec::new();
    let mut paired: Vec<Quad> = Vec::new();
    for (done, &idx) in indices.iter().enumerate() {
        let item = &all_items[idx];
        let payload = &payloads[done];
        // PC1 has NO sender. The shared frozen block's `SenderPass` slot is
        // filled with an explicit not-applicable marker and the per-item row's
        // `sender_first_pass` object is REPLACED below, so no receipt reader
        // can mistake a placeholder for a measurement.
        let sender_pass = SenderPass {
            first_pass_correct: false,
            first_pass_answer: Some("<PC1: no sender — same-model self-pair>".to_string()),
            generated_tokens: 0,
        };
        // The span the payload was taken from, read back out of the capture
        // receipt so the per-item row records the real provenance rather than
        // a placeholder: the payload is the LAST of these gold-continuation
        // rows, so the recorded span is `[gold_len - 1, gold_len)`.
        let gold_len = cap["items"][done]["gold_continuation_tokens"]
            .as_u64()
            .ok_or_else(|| anyhow::anyhow!("capture receipt item {done} has no span length"))?
            as usize;
        anyhow::ensure!(
            cap["items"][done]["item"].as_u64() == Some(idx as u64),
            "capture receipt item order differs from this probe's"
        );
        let meta = CaptureMeta {
            hidden_size: D_RECEIVER,
            pooled_l2_raw: norms::l2(payload),
            span: gold_len - 1..gold_len,
            variant: "pc1-identity-L19-lasttoken-goldteacherforced",
        };
        let (mut row, q) = four_conditions_at(
            &mut receiver,
            item,
            pad_id,
            payload,
            &sender_pass,
            &meta,
            INJECT_MODE,
            SITE,
            &device,
        )?;
        row["sender_first_pass"] = serde_json::json!({
            "not_applicable": "PC1 is a same-model self-pair; no sender model is loaded or run",
        });
        row["pc1_payload"] = serde_json::json!({
            "source": "receiver's OWN block-19 state, gold-teacher-forced, LAST span token",
            "transform": "identity",
            "l2_raw": norms::l2(payload),
            "gold_continuation_tokens": gold_len,
            "span_note": "capture.span records [gold_len-1, gold_len) — the single row that IS the payload",
        });
        println!(
            "[{}/{}] item {idx}: aligned={} baseline={} zerovec={} random={} (nll {:.3}/{:.3}/{:.3}/{:.3}) {:.0}s",
            done + 1, indices.len(), q.real.0, q.base.0, q.zero.0, q.rand.0,
            q.real.1, q.base.1, q.zero.1, q.rand.1, t0.elapsed().as_secs_f32()
        );
        rows.push(row);
        paired.push(q);
    }

    // ---- Pre-committed analysis (S1a's frozen protocol, verbatim) ---------
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
    let zerovec_pass = 2 * zero_c >= base_c;
    let wins_rz = count(&|q| q.real.0 && !q.zero.0);
    let loss_rz = count(&|q| !q.real.0 && q.zero.0);
    let wins_rb = count(&|q| q.real.0 && !q.base.0);
    let loss_rb = count(&|q| !q.real.0 && q.base.0);
    let zb_base_wins = count(&|q| q.base.0 && !q.zero.0);
    let zb_zero_wins = count(&|q| q.zero.0 && !q.base.0);

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
    let mean_real_nll = mean(&|q| q.real.1);

    // Fuse no-op diagnostic (M4g's, unchanged).
    let noop_acc_disagreements = count(&|q| q.zero.0 != q.base.0);
    let noop_max_abs_nll_delta = paired
        .iter()
        .map(|q| (q.zero.1 - q.base.1).abs())
        .fold(0f32, f32::max);
    let noop_exact_items = paired.iter().filter(|q| q.zero.1 == q.base.1).count();
    let noop_pass = noop_acc_disagreements == 0 && noop_max_abs_nll_delta <= FUSE_NOOP_TOL;

    // NLL harm/benefit accounting, the same shape every rung reports.
    let worse_than_baseline = paired.iter().filter(|q| q.real.1 > q.base.1).count();
    let worse_than_zerovec = paired.iter().filter(|q| q.real.1 > q.zero.1).count();
    let worse_than_random = paired.iter().filter(|q| q.real.1 > q.rand.1).count();

    // ---- Gaming guard (docs/research/045 §3), pre-committed ---------------
    let chars =
        |r: &serde_json::Value, k: &str| -> f64 { r["generated_chars"][k].as_f64().unwrap_or(0.0) };
    let ratios: Vec<f64> = rows
        .iter()
        .map(|r| {
            let b = chars(r, "baseline_uninjected");
            if b > 0.0 {
                chars(r, "aligned_real") / b
            } else {
                1.0
            }
        })
        .collect();
    let degenerate_short_items = ratios.iter().filter(|&&x| x < DEGENERATE_LEN_RATIO).count();
    let mean_len_ratio = ratios.iter().sum::<f64>() / ratios.len().max(1) as f64;
    let degenerate_signature =
        (degenerate_short_items as f64) >= DEGENERATE_ITEM_FRACTION * n as f64;
    let nll_collapse_signature = mean_real_nll < NLL_COLLAPSE_NATS;
    let gaming_signature = degenerate_signature || nll_collapse_signature;

    // ---- Registered verdict ----------------------------------------------
    // ADR-024 / docs/research/045 §5: PASS = p < alpha AND gates clean AND no
    // gaming signature. A PASS carrying a signature is a GAMED pass.
    let verdict = if primary_pass && !gaming_signature {
        "PASS — mechanics LIVE (liveness only, never transfer)"
    } else if primary_pass && gaming_signature {
        "GAMED PASS — the primary rejected, but a pre-committed degenerate-output/NLL-collapse signature is present; docs/research/045 §3 requires this be reported as gamed, not clean"
    } else {
        "FAIL — the positive control did not move the receiver's answers under current mechanics. This is the more consequential branch (ADR-024's registered FAIL rule) and is reported without softening."
    };

    let receipt = serde_json::json!({
        "stage": "run2-PC1-positive-control-probe",
        "design": "docs/adr/024-run2-trained-thought-adapter-ladder.md § 'PC1 PRE-REGISTRATION (2026-08-29) — the positive control this harness never had'; docs/research/045-positive-control-design.md §2 candidate (c) + §5. Probe protocol = ADR-023's frozen S1a/S2b four-condition protocol, inherited unchanged through common::m3::four_conditions_at.",
        "env": common::env_info(&nvcc),
        "pre_committed": true,
        "variant": "pc1-identity-L19lasttoken-goldteacherforced-fuse",

        "FIREWALL_liveness_not_transfer": {
            "verbatim_research_045_section_3": "This control used a same-model, same-item, identity-transform injection with gold-adjacent content. It tests whether this repository's injection mechanics (delivery operator, payload shape, injection site, norm-rescale) are capable of carrying signal at all. It does not test, and must not be cited as evidence for or against, whether a cross-model-derived, learned-alignment payload can transfer reasoning content — that is exactly the separate question M3 through M5X exist to answer.",
            "rule": "A PASS on this rung proves LIVENESS ONLY, NEVER TRANSFER, and may NEVER be cited as evidence for the transfer claim.",
            "structural_scope": "same-model, same-item, identity transform, gold-derived content. No model boundary is crossed and no alignment is fitted anywhere in this rung.",
        },

        "what_this_rung_answers": {
            "question": "Under the mechanics the ladder has used since M4c — fuse delivery, de-pooled payload, <|fim_pad|> slots, the L18->L14 cell's receiver site, rescale-to-natural-median — can the injection pathway move the receiver's answers AT ALL, given the easiest constructible payload?",
            "why_it_was_never_asked": "S1a, the ladder's only PASS, ran under overwrite + pooled + inject-at-block-19. Nothing since M4c uses those mechanics, so no positive control has ever been run under the current ones.",
            "ordering": "ADR-024 registers PC1 to run BEFORE M5X (ADR-037, ~1.3-1.4 GPU-h) and M4b (ADR-035, ~1-2 GPU-h): spending hours on rungs whose nulls would be uninterpretable is the wrong order.",
        },

        "registered_outcome_rule_before_the_run": {
            "pass": "the current mechanics demonstrably carry signal in the maximally favourable same-model case, and every subsequent null becomes interpretable as being about TRANSFER rather than about PLUMBING.",
            "fail": "the more consequential branch, not to be softened: every null since S1a (S2b x4, M3 x2, M4 x3, M4c, M4d, M4g, M4h S1, M4i) needs an explicit caveat that current mechanics were never shown capable of carrying signal even from a model to itself; S1a's PASS gains a footnote scoping it to mechanics no longer in use; and ADR-024's 'adapters improved, so discordance fell' explanation acquires an unruled-out rival — fuse + de-pooling + site may suppress discordance MECHANICALLY, independent of adapter quality.",
            "gamed_pass": "docs/research/045 §3: a control that passes because the injection collapses the receiver toward a degenerate output is not evidence the pathway is USEFULLY live. Flagged below and reported as gamed, not clean.",
        },

        "declared_deviation_from_the_registered_payload_source": cap["declared_deviation_from_the_registered_capture_source"].clone(),

        "config": {
            "receiver": RECEIVER,
            "sender": "NONE — same-model self-pair",
            "transform": "IDENTITY — no adapter artifact is loaded, constructed or applied anywhere in this binary",
            "payload": "the receiver's OWN block-19 residual state, teacher-forced over the item's GSM8K gold solution, LAST token of the span (de-pooled)",
            "payload_file": payload_path.display().to_string(),
            "payload_sha256": payload_sha,
            "capture_receipt": CAPTURE_RECEIPT,
            "receiver_inject_block": RECEIVER_BLOCK,
            "cell": "L18->L14 geometry — the injection site is the L18->L14 cell's receiver half, unchanged; PC1 replaces only the capture side (with the receiver's own L19 state) since there is no sender",
            "slots": N_SLOTS,
            "site": SITE.tag(),
            "site_description": SITE.description(),
            "placeholder_token": "<|fim_pad|>",
            "placeholder_id": pad_id,
            "pool_span": "NONE — de-pooled. The payload is a single real receiver state; the 8-slot broadcast of that one vector is unchanged.",
            "rescale_to_natural_median": true,
            "decoding": "greedy, batch=1, max_new_tokens=400",
            "item_seed_chacha8": ITEM_SEED,
            "injection_operator": {"mode": INJECT_MODE.tag(), "equation": INJECT_MODE.equation()},
            "conditions": {
                "produced_by": "common::m3::four_conditions_at — the SAME shared code path M4g/M4h/M4i ran, under the same InjectionMode::Fuse, so no control is redefined by this rung",
                "aligned_real": "the identity payload, rescaled to the natural median, delivered by residual ADD at the 8 <|fim_pad|> rows",
                "random": "per-item seeded gaussian, norm-matched to the effective aligned vector — under fuse an information-free perturbation of the receiver's own rows, which is exactly the comparator the primary needs",
                "zerovec_injected": "TRUE ZERO VECTOR through the real 8-slot path. Under fuse this is h += 0, an exact NO-OP that collapses onto baseline_uninjected (M4g's frozen semantics, verified again below). The 2 x zerovec >= baseline gate is therefore DEGENERATE and carries no evidential weight; retained for cross-rung comparability.",
                "baseline_uninjected": "no injection, same prompt",
            },
            "primary_test": "ADR-024's PC1 registration names the ORIGINAL frozen 40-item one-sided EXACT SIGN TEST as primary — chosen for direct comparability with S1a, the ladder's only PASS, on identical footing, NOT because the e-process (ADR-036) is unavailable. Mid-p McNemar is computed on the same pairs and reported alongside; neither was selected after seeing the other.",
            "secondary_diagnostic": "one-sided sign tests on paired teacher-forced NLL of '#### <gold>'",
        },

        "comparability_with_s1a": {
            "s1a_receipt": S1A_RECEIPT,
            "same": ["the 40 GSM8K-train items (ChaCha8 0x51A1, asserted equal)",
                      "identity transform / self-pair",
                      "8 <|fim_pad|> slots",
                      "rescale-to-natural-median",
                      "greedy, max 400 new tokens",
                      "the 40-item one-sided exact sign test at alpha 0.05"],
            "changed_to_the_current_mechanics": ["delivery operator: overwrite -> FUSE",
                      "payload: pooled over the full generated span -> DE-POOLED (single last-token state)",
                      "injection site: block 19 -> block 14 (the L18->L14 cell's receiver half)",
                      "payload capture text: the receiver's own greedy generation -> the item's GOLD solution (a STRONGER ceiling; docs/research/045 §2 candidate (c))",
                      "rescale reference: the capture pass's own L19 norms -> the injection prompt's block-14 norms (current mechanics)"],
            "s1a_result_for_reference": s1a["summary"].clone(),
        },

        "manifold_precheck_run_before_this_draw": {
            "receipt": PRECHECK_RECEIPT,
            "candidate_label": PRECHECK_LABEL,
            "payload_sha_matches_this_probe": true,
            "classification": precheck_class,
            "surprise_flag": precheck_surprise,
            "expected": "on-manifold-item-varying — trivially, because the payload IS a real receiver state",
            "if_surprised": "a finding about the PIPELINE (the pre-check's reference metric), not about PC1 — see the pre-check receipt's registered_metric_caveat.",
            "gating": "NONE — ORDERING ONLY, per ADR-024's M4f framing.",
            "row": precheck_row,
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
                    "POWER-LIMITED: at this many discordant pairs the smallest attainable one-sided p is >= alpha, so this draw was STRUCTURALLY INCAPABLE of rejecting regardless of outcome. Same category as M4g (n_disc=3). A null here is weak evidence about the accuracy question; read the NLL numbers."
                } else {
                    "NOT power-limited: the smallest attainable one-sided p is below alpha, so this draw COULD have rejected. Same category as M4d (n_disc=7) and S1a (n_disc=5)."
                },
                "ladder_context": "S1a n_disc=5 (5W/0L, p=0.03125 — the ladder's only PASS); M4d n_disc=7; M4g n_disc=3 (power-limited).",
            },
            "accuracy": {"aligned_real": real_c, "baseline_uninjected": base_c,
                          "zerovec_injected": zero_c, "random": rand_c},
            "primary_aligned_vs_random": {"wins": wins_rr, "losses": loss_rr,
                "n_discordant": n_disc, "p_one_sided": p_primary, "alpha": ALPHA,
                "pass": primary_pass, "mid_p_one_sided": mid_primary,
                "mid_p_pass": mid_primary < ALPHA,
                "gate_pass_computed_from": "p_one_sided (exact sign test — S1a's own statistic, for direct comparability)",
                "min_attainable_p_at_this_n_disc": min_attainable_p},
            "aligned_vs_zerovec": {"wins": wins_rz, "losses": loss_rz,
                "p_one_sided": common::sign_test_one_sided(wins_rz, loss_rz),
                "mid_p_one_sided": common::mid_p_one_sided(wins_rz, loss_rz)},
            "aligned_vs_baseline": {"wins": wins_rb, "losses": loss_rb,
                "p_one_sided": common::sign_test_one_sided(wins_rb, loss_rb),
                "mid_p_one_sided": common::mid_p_one_sided(wins_rb, loss_rb)},
            "zerovec_vs_baseline": {
                "degenerate_under_fuse": true,
                "baseline_wins": zb_base_wins, "zerovec_wins": zb_zero_wins,
                "criterion": "2 x zerovec_correct >= baseline_correct",
                "pass": zerovec_pass,
                "note": "trivially satisfied because zerovec == baseline under fuse; no evidential weight"},
            "nll_mean": {"aligned_real": mean_real_nll,
                          "baseline_uninjected": mean(&|q| q.base.1),
                          "zerovec_injected": mean(&|q| q.zero.1),
                          "random": mean(&|q| q.rand.1)},
            "nll_aligned_vs_random": {"wins": nll_rr.0, "losses": nll_rr.1,
                "p_one_sided": nll_rr.2, "mid_p_one_sided": nll_rr.3},
            "nll_aligned_vs_zerovec": {"wins": nll_rz.0, "losses": nll_rz.1,
                "p_one_sided": nll_rz.2, "mid_p_one_sided": nll_rz.3},
            "nll_aligned_vs_baseline": {"wins": nll_rb.0, "losses": nll_rb.1,
                "p_one_sided": nll_rb.2, "mid_p_one_sided": nll_rb.3},
            "nll_zerovec_vs_baseline": {"wins": nll_zb.0, "losses": nll_zb.1,
                "p_one_sided": nll_zb.2, "mid_p_one_sided": nll_zb.3},
            "nll_harm_accounting": {
                "ladder_context": "ADR-024's MAJOR CORRECTION: the unanimous 0/40 NLL inversion occurs ONLY in the three OFF-MANIFOLD task-loss rungs (M4c/M4d/M4g); every ON-MANIFOLD rung sits within ~0.004 nats of the 2.1288 baseline. PC1's payload is a real receiver state, so it belongs to the on-manifold family by construction.",
                "items_where_aligned_nll_is_WORSE_than_baseline": worse_than_baseline,
                "items_where_aligned_nll_is_WORSE_than_zerovec": worse_than_zerovec,
                "items_where_aligned_nll_is_WORSE_than_random": worse_than_random,
                "n_items": n,
                "unanimously_worse_than_baseline": worse_than_baseline == n && n > 0,
            },
            "fuse_zero_is_noop_vs_baseline": {
                "expected": "EXACT identity — under fuse the zerovec condition is h += 0",
                "accuracy_disagreements": noop_acc_disagreements,
                "nll_bit_identical_items": noop_exact_items, "n_items": n,
                "max_abs_nll_delta": noop_max_abs_nll_delta, "tolerance": FUSE_NOOP_TOL,
                "pass": noop_pass, "gating": "NONE — operator-correctness diagnostic only"},
            "gaming_guard_research_045_section_3": {
                "why": "ITI (arXiv:2306.03341) documents an intervention that appears to restore the answer while merely collapsing the model into a degenerate response ('it is trivial to attain a perfect truthfulness score simply by answering \"no comment\"'). docs/research/045 §3 makes this check mandatory on EVERY positive-control draw, including one 'supposed to' pass.",
                "thresholds_pre_committed": {
                    "degenerate_len_ratio": DEGENERATE_LEN_RATIO,
                    "degenerate_item_fraction": DEGENERATE_ITEM_FRACTION,
                    "nll_collapse_nats": NLL_COLLAPSE_NATS,
                },
                "mean_aligned_over_baseline_generated_char_ratio": mean_len_ratio,
                "degenerate_short_items": degenerate_short_items,
                "degenerate_output_signature": degenerate_signature,
                "mean_aligned_nll": mean_real_nll,
                "nll_collapse_signature": nll_collapse_signature,
                "nll_collapse_reading": "For a GOLD-derived payload a large NLL drop is the expected direction — the payload encodes the answer. §3 requires it be FLAGGED rather than excused: an NLL collapsed below the threshold means the receiver was handed the answer, not that it computed it, and any accuracy PASS resting on that is a liveness result about the CHANNEL's bandwidth, still never about transfer.",
                "gaming_signature": gaming_signature,
            },
        },

        "gates": {
            "receiver_L19_dump_sha256_verified": {"pass": true, "sha256": dump_sha,
                "note": "verified in run2-pc1-capture-receipt.json against run2-pertoken-dump-receipt.json; see declared_deviation_from_the_registered_payload_source for why the dump could not supply the payload itself"},
            "payload_sha256_matches_capture_receipt": {"pass": true, "sha256": payload_sha},
            "identity_transform_no_adapter_weights_loaded": {"pass": true,
                "note": "no MlpTransform / FastGrnnTransform / AffineTransform is constructed anywhere in this binary; the injected vector IS a captured receiver state, byte-for-byte"},
            "injection_mode_recorded": {"pass": true, "mode": INJECT_MODE.tag(),
                "equation": INJECT_MODE.equation()},
            "s1a_item_set_reproduced": {"pass": true, "n": N_ITEMS},
            "manifold_precheck_ran_before_this_draw": {"pass": true, "receipt": PRECHECK_RECEIPT,
                "classification": precheck_class, "gating": "ordering only"},
            "slot_count_unchanged": {"pass": true, "slots": N_SLOTS},
            "PC1_aligned_real_vs_random": {"pass": primary_pass, "p": p_primary,
                "mid_p": mid_primary, "n_discordant": n_disc,
                "min_attainable_p": min_attainable_p, "power_limited": power_limited},
            "zerovec_not_catastrophic": {"pass": zerovec_pass, "degenerate_under_fuse": true},
            "fuse_zero_payload_is_a_noop": {"pass": noop_pass, "gating": "none — diagnostic"},
            "no_gaming_signature": {"pass": !gaming_signature,
                "degenerate_output_signature": degenerate_signature,
                "nll_collapse_signature": nll_collapse_signature},
        },
        "gate_pass": primary_pass && zerovec_pass && !gaming_signature,
        "verdict": verdict,
        "honest_fail_contract": "ADR-032: ONE registered draw, no retry, full numbers reported either way. A FAILING positive control is the MORE consequential outcome and is stated plainly here, not softened. This receipt is written before any interpretation is added to the ADR.",
        "wall_clock_s": t0.elapsed().as_secs_f32(),
    });
    common::write_receipt(
        &crate_path("receipts"),
        "run2-pc1-receipt-identity-L19lasttoken-goldtf-fuse-slots8-nopool-rescaletrue-n40.json",
        &receipt,
    )?;

    println!("\n=== PC1 (positive control — LIVENESS ONLY, NEVER TRANSFER) ===");
    println!(
        "acc aligned {real_c}/{n} baseline {base_c}/{n} zerovec {zero_c}/{n} random {rand_c}/{n}"
    );
    println!(
        "primary p={p_primary:.5} (wins {wins_rr}, losses {loss_rr}, n_disc {n_disc}) => pass={primary_pass} \
         [mid-p {mid_primary:.5}]; min attainable p at n_disc={n_disc} is {min_attainable_p:.5} \
         => power_limited={power_limited}"
    );
    println!(
        "NLL means: aligned {:.4} baseline {:.4} zerovec {:.4} random {:.4}",
        mean_real_nll,
        mean(&|q| q.base.1),
        mean(&|q| q.zero.1),
        mean(&|q| q.rand.1)
    );
    println!(
        "NLL sign tests: aligned-vs-baseline {}W/{}L, aligned-vs-zerovec {}W/{}L, aligned-vs-random {}W/{}L",
        nll_rb.0, nll_rb.1, nll_rz.0, nll_rz.1, nll_rr.0, nll_rr.1
    );
    println!(
        "gaming guard: mean len ratio {mean_len_ratio:.3}, degenerate-short items \
         {degenerate_short_items}/{n} => degenerate={degenerate_signature}; \
         nll_collapse={nll_collapse_signature} => gaming_signature={gaming_signature}"
    );
    println!("VERDICT: {verdict}");
    Ok(())
}
