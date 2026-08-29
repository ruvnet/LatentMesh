//! Run-2 **PC1b** — the positive control, repeated at the site where payloads
//! demonstrably register.
//!
//! ADR-024 § "CORRECTION + a critical interaction between M4i and PC1
//! (2026-08-29)" ¶3 registers this rung: *"repeat PC1 (receiver's own L19
//! states, identity transform, fuse, de-pooled) at the question-tail
//! ordinary-token site M4i used, under the e-process. This is now the single
//! highest-value experiment available: it separates 'our pathway is inert'
//! from 'the placeholder token was the problem', and both of those readings
//! currently have real evidence."*
//!
//! # The confound this rung exists to break
//!
//! PC1 handed the receiver its **own** gold-teacher-forced block-19 states via
//! identity transform and produced **0 wins / 0 losses** against baseline —
//! apparently proving the current pathway inert. But PC1 ran at the **8
//! `<|fim_pad|>` placeholder slots**, and M4i then showed that this is exactly
//! the site where payloads fail to register: moving to ordinary question-tail
//! tokens produced a **0.148-nat** NLL improvement (2.2446 vs 2.3928,
//! 181W/119L over 300 items) — two orders of magnitude outside the ~0.004-nat
//! inertness band every placeholder-site configuration sat in.
//!
//! So PC1's failure is **confounded**. It may mean *"our mechanics cannot
//! carry signal"* or merely *"the placeholder token cannot carry signal"*.
//! PC1b separates them by changing **exactly one factor: the site.**
//!
//! # ONE changed factor
//!
//! | | PC1 | PC1b |
//! |---|---|---|
//! | payload | receiver's own gold-TF block-19 last-token state | **same derivation** (bit-identity gated on the shared item) |
//! | transform | identity, no adapter weights | identity, no adapter weights |
//! | delivery | fuse (`h[slot] += c·v`) | fuse |
//! | pooling | de-pooled | de-pooled |
//! | slots | 8 | 8 |
//! | depth | receiver block 14 | receiver block 14 |
//! | rescale | to natural median | to natural median |
//! | **site** | **8 `<|fim_pad|>` slots** | **8 ordinary question-tail tokens** |
//! | statistic | S1a's frozen 40-item sign test | **ADR-036 e-process** |
//!
//! The site is changed by passing `Site::QuestionTail` to the very same
//! `common::m3::four_conditions_at` PC1 called with `Site::FimPadSlots` — one
//! argument, one shared code path, so no control is silently redefined. The
//! positions themselves come from `build_site_prompt`, reused verbatim from
//! M4i rather than reimplemented, which is what makes the site **provably
//! identical to M4i's** rather than merely similar.
//!
//! The statistic changes too, and that is not a second free variable: ADR-036
//! Decision 4 governs every successor rung by default, and PC1's 40-item sign
//! test is precisely the instrument that could not measure PC1 (`n_disc` = 3,
//! minimum attainable one-sided p = 0.125). M4i reached **`n_disc` = 66** at
//! this site, so this rung should finally have real power.
//!
//! # FIREWALL — `docs/research/045` §3, verbatim, and repeated in the receipt
//!
//! *"This control used a same-model, same-item, identity-transform injection
//! with gold-adjacent content. It tests whether this repository's injection
//! mechanics (delivery operator, payload shape, injection site, norm-rescale)
//! are capable of carrying signal at all. It does not test, and must not be
//! cited as evidence for or against, whether a cross-model-derived,
//! learned-alignment payload can transfer reasoning content — that is exactly
//! the separate question M3 through M5X exist to answer."*
//!
//! **A PASS proves LIVENESS ONLY, NEVER TRANSFER, and may never be cited as
//! evidence for the transfer claim.**
//!
//! # PRE-REGISTERED INTERPRETATION, recorded BEFORE the draw
//!
//! - **PASS** → the pathway **can** carry signal when the site is right. That
//!   narrows every prior null to *"the placeholder site was inert"* and makes
//!   M5X (ADR-037) and M4b (ADR-035) worth running.
//! - **FAIL, with real power** → far stronger than PC1: the pathway is inert
//!   **even where payloads demonstrably register**, and the ladder's nulls
//!   would stand as evidence about **transfer** rather than about plumbing.
//!
//! PC1b is the mission's gating experiment; M5X and M4b stay blocked behind
//! it. ONE registered draw, no retry, full numbers either way (ADR-032).
//!
//! Run: PATH=/usr/local/cuda-12.8/bin:$PATH \
//!      cargo run --release --features cuda --example run2_pc1b_probe

#![recursion_limit = "512"]

#[path = "common/mod.rs"]
#[allow(dead_code)]
mod common;

use common::m3::{
    build_site_prompt, four_conditions_at, CaptureMeta, Quad, SenderPass, Site, N_SLOTS, RECEIVER,
    RECEIVER_BLOCK,
};
use latentmesh_runtime::inject::InjectionMode;
use latentmesh_runtime::{norms, QwenRuntime};
use std::path::{Path, PathBuf};

const GSM8K_TRAIN_SHA256: &str = "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465";
const ADAPTATION_512: &str = "../../harness/latentmesh-live/data/adaptation-512.json";
const CAPTURE_RECEIPT: &str = "receipts/run2-pc1b-capture-receipt.json";
const PRECHECK_RECEIPT: &str = "receipts/run2-pc1b-manifold-precheck-receipt.json";
const PRECHECK_LABEL: &str = "pc1b-payload-receiver-L19-lasttoken-goldteacherforced";
/// PC1's committed receipt — the rung this one is the single-factor successor
/// to. Read for the side-by-side comparison block, never to gate.
const PC1_RECEIPT: &str =
    "receipts/run2-pc1-receipt-identity-L19lasttoken-goldtf-fuse-slots8-nopool-rescaletrue-n40.json";
/// M4i's committed receipt — the rung whose SITE this one reuses.
const M4I_RECEIPT: &str =
    "receipts/run2-m4i-receipt-cellL18toL14-mlp-pertokenlast-fuse-questiontail-slots8-eprocess.json";

const D_RECEIVER: usize = 1536;
const VECS_PER_ITEM: usize = 3;

/// Current mechanics, unchanged from PC1.
const INJECT_MODE: InjectionMode = InjectionMode::Fuse;
/// **The ONE changed factor vs PC1** (PC1 used `Site::FimPadSlots`).
const SITE: Site = Site::QuestionTail;

/// ADR-024's 13 probe-overlap items, verbatim from M4i's own probe.
/// ADR-036 Decision 2(2) keeps the exclusion in force for the e-process.
const LEAKAGE_EXCLUSIONS: [usize; 13] = [
    150, 1309, 1573, 2365, 2418, 2540, 2958, 2973, 3084, 3825, 3844, 3877, 4746,
];

// ---- ADR-036 Decision 1 / ADR-030 §3.2 e-process parameters, frozen -------
/// Betting fraction, `λ = 2θ−1` tuned to the smallest interesting effect
/// θ=0.65. Fixed in advance; never re-parametrised after seeing `W_t`.
const LAMBDA: f64 = 0.30;
/// α for the wealth boundary. PASS at `W_i ≥ 1/α`.
const E_ALPHA: f64 = 0.05;
/// The registered budget.
const N_MAX: usize = 300;
/// Tolerance for the fuse zero-payload no-op diagnostic (nats). Reported,
/// never gating.
const FUSE_NOOP_TOL: f32 = 1e-6;

// --- Pre-committed gaming-guard thresholds (docs/research/045 §3) ----------
// PC1's values, unchanged, so the two rungs' guards are directly comparable.
const DEGENERATE_LEN_RATIO: f64 = 0.25;
const DEGENERATE_ITEM_FRACTION: f64 = 0.25;
const NLL_COLLAPSE_NATS: f32 = 0.10;

fn crate_path(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

/// One step of the registered wealth process.
struct EStep {
    order: usize,
    item: usize,
    aligned_correct: bool,
    random_correct: bool,
    discordant: bool,
    x: Option<u8>,
    wealth: f64,
}

#[allow(clippy::too_many_lines)]
fn main() -> anyhow::Result<()> {
    let t0 = std::time::Instant::now();
    let e = anyhow::Error::msg;
    let nvcc = latentmesh_runtime::assert_cuda_build_env().map_err(e)?;
    println!("build-env guard: {nvcc}");
    let w_threshold = 1.0 / E_ALPHA;

    // ---- Gate 1: the capture receipt and its sha-pinned payload file ------
    let cap: serde_json::Value =
        serde_json::from_slice(&std::fs::read(crate_path(CAPTURE_RECEIPT)).map_err(|err| {
            anyhow::anyhow!(
                "capture receipt {CAPTURE_RECEIPT} unreadable ({err}) — run run2_pc1b_capture first"
            )
        })?)?;
    for gate in [
        "pc1_reference_payload_sha256_intact",
        "derivation_is_the_shared_code_path",
        "pc1_overlap_item_bit_identical",
        "identity_transform_no_adapter_weights_loaded",
        "stream_identical_to_m4i",
        "all_captured_states_finite",
    ] {
        anyhow::ensure!(
            cap["gates"][gate]["pass"].as_bool() == Some(true),
            "capture receipt gate {gate} did not pass"
        );
    }
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
        bytes.len() == N_MAX * VECS_PER_ITEM * D_RECEIVER * 4,
        "payload file size mismatch"
    );
    let flat: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    // The PAYLOAD is vector 0 of each item's triple: the block-19 last row.
    let payloads: Vec<Vec<f32>> = (0..N_MAX)
        .map(|i| {
            let b = i * VECS_PER_ITEM * D_RECEIVER;
            flat[b..b + D_RECEIVER].to_vec()
        })
        .collect();
    // The bit-identity result, carried forward into this receipt so the
    // payload-provenance argument is readable from the probe receipt alone.
    let bit_identity = cap["bit_identity_vs_pc1"].clone();
    println!("payload VERIFIED: sha {payload_sha} ({N_MAX} x {D_RECEIVER} f32, identity)");

    // ---- Gate 2 (ORDERING, not evidential): the manifold pre-check --------
    let precheck: serde_json::Value =
        serde_json::from_slice(&std::fs::read(crate_path(PRECHECK_RECEIPT)).map_err(|err| {
            anyhow::anyhow!(
                "manifold pre-check receipt {PRECHECK_RECEIPT} unreadable ({err}) — ADR-024 orders \
                 the pre-check BEFORE this draw; run run2_pc1b_manifold_precheck first"
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

    // ---- Item supply: ADR-036 Decision 2, re-derived independently --------
    let dir = common::run_dir("run2-pc1b");
    let data = dir.join("gsm8k-train.jsonl");
    let train_sha = common::fetch(common::GSM8K_TRAIN_URL, &data)?;
    anyhow::ensure!(train_sha == GSM8K_TRAIN_SHA256);
    let all_items = common::load_gsm8k(&data)?;

    let adaptation: serde_json::Value =
        serde_json::from_slice(&std::fs::read(crate_path(ADAPTATION_512))?)?;
    anyhow::ensure!(
        adaptation["split"].as_str() == Some("adaptation-512"),
        "the item-supply file is not adaptation-512"
    );
    anyhow::ensure!(
        adaptation["source_sha256"].as_str() == Some(GSM8K_TRAIN_SHA256),
        "adaptation-512 was drawn from a different train.jsonl than this probe loaded"
    );
    let adaptation_indices: Vec<usize> = adaptation["indices"]
        .as_array()
        .expect("adaptation-512 indices")
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    anyhow::ensure!(adaptation_indices.len() == 512);
    anyhow::ensure!(
        adaptation_indices.windows(2).all(|w| w[0] < w[1]),
        "adaptation-512 is not in ascending index order — the registered consumption order is the \
         file's own fixed index order"
    );
    let excluded_present: Vec<usize> = LEAKAGE_EXCLUSIONS
        .iter()
        .copied()
        .filter(|i| adaptation_indices.contains(i))
        .collect();
    let eligible: Vec<usize> = adaptation_indices
        .iter()
        .copied()
        .filter(|i| !LEAKAGE_EXCLUSIONS.contains(i))
        .collect();
    anyhow::ensure!(
        eligible.len() >= N_MAX,
        "eligible pool ({}) is smaller than N_max ({N_MAX})",
        eligible.len()
    );

    // ---- Model: receiver ONLY (PC1b is a same-model self-pair) ------------
    let device = latentmesh_runtime::device().map_err(e)?;
    println!("loading {RECEIVER} (BF16)... [no sender, no adapter]");
    let mut receiver = QwenRuntime::load(RECEIVER, &device, candle_core::DType::BF16).map_err(e)?;
    // Resolved but UNUSED under `Site::QuestionTail`: there is no placeholder
    // token anywhere in this rung's prompt. Kept so the shared code path is
    // literally the same function every prior rung called.
    let pad_id = receiver
        .tokenizer
        .token_to_id("<|fim_pad|>")
        .ok_or_else(|| anyhow::anyhow!("<|fim_pad|> not in tokenizer"))?;
    println!("model loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // ---- Tokenisation pre-flight, BEFORE any generation -------------------
    // M4i's rule verbatim: resolve the site for the whole stream up front so
    // no item can be dropped mid-draw for a reason that could correlate with
    // its outcome.
    let mut stream: Vec<usize> = Vec::new();
    let mut tokenization_excluded: Vec<serde_json::Value> = Vec::new();
    let mut site_samples: Vec<serde_json::Value> = Vec::new();
    for &idx in &eligible {
        if stream.len() == N_MAX {
            break;
        }
        match build_site_prompt(&receiver, &all_items[idx], pad_id, SITE) {
            Ok(sp) => {
                if site_samples.len() < 5 {
                    site_samples.push(serde_json::json!({
                        "item": idx,
                        "prompt_tokens": sp.tokens.len(),
                        "positions": sp.positions,
                        "position_token_ids": sp.position_token_ids,
                        "positions_decoded": sp.positions_decoded,
                    }));
                }
                stream.push(idx);
            }
            Err(err) => tokenization_excluded.push(serde_json::json!({
                "item": idx, "reason": err.to_string(),
            })),
        }
    }
    anyhow::ensure!(
        stream.len() == N_MAX,
        "pre-flight resolved only {} of the required {N_MAX} items",
        stream.len()
    );

    // The capture must have walked EXACTLY this stream, or the payload at
    // position i is not the state for item i.
    let cap_stream: Vec<usize> = cap["dataset"]["indices"]
        .as_array()
        .expect("capture receipt indices")
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    anyhow::ensure!(
        stream == cap_stream,
        "the capture receipt's item stream differs from this probe's independently-derived one — \
         the payloads would not correspond to the items"
    );

    // The site is provably M4i's: same resolver, same stream. Asserted against
    // M4i's committed receipt rather than argued in prose.
    let m4i: serde_json::Value = serde_json::from_slice(
        &std::fs::read(crate_path(M4I_RECEIPT))
            .map_err(|err| anyhow::anyhow!("M4i receipt {M4I_RECEIPT} unreadable ({err})"))?,
    )?;
    let m4i_stream: Vec<usize> = m4i["dataset"]["indices"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_u64())
                .map(|v| v as usize)
                .collect()
        })
        .unwrap_or_default();
    let stream_identical_to_m4i = !m4i_stream.is_empty() && m4i_stream == stream;
    let m4i_site = m4i["site_change"]["to"]
        .as_str()
        .or_else(|| m4i["config"]["site"].as_str())
        .unwrap_or("<unrecorded>")
        .to_string();
    let site_tag_matches_m4i = m4i_site == SITE.tag();
    println!(
        "site provenance: tag {} (M4i recorded {m4i_site}, match={site_tag_matches_m4i}); \
         stream identical to M4i's = {stream_identical_to_m4i}",
        SITE.tag()
    );
    println!(
        "site pre-flight: {} items resolved to 8 ordinary question-tail positions, {} excluded on \
         tokenisation grounds (recorded before any forward pass)",
        stream.len(),
        tokenization_excluded.len()
    );
    for s in &site_samples {
        println!(
            "  item {} -> positions {} decode to {}",
            s["item"], s["positions"], s["positions_decoded"]
        );
    }

    // ---- THE DRAW: ADR-036 e-process over the fixed stream ----------------
    let mut wealth = 1.0f64;
    let mut max_wealth = 1.0f64;
    let mut trajectory: Vec<EStep> = Vec::new();
    let mut rows: Vec<serde_json::Value> = Vec::new();
    let mut paired: Vec<Quad> = Vec::new();
    let mut wins = 0usize;
    let mut losses = 0usize;
    let mut crossed_at: Option<usize> = None;

    for (order, &idx) in stream.iter().enumerate() {
        let item = &all_items[idx];
        let payload = &payloads[order];
        // PC1b has NO sender. The shared frozen block's `SenderPass` slot is
        // filled with an explicit not-applicable marker and the per-item row's
        // `sender_first_pass` object is REPLACED below, so no receipt reader
        // can mistake a placeholder for a measurement.
        let sender_pass = SenderPass {
            first_pass_correct: false,
            first_pass_answer: Some("<PC1b: no sender — same-model self-pair>".to_string()),
            generated_tokens: 0,
        };
        // The span the payload was taken from, read back out of the capture
        // receipt so the per-item row records real provenance: the payload is
        // the LAST gold-continuation row, so the span is `[gold_len-1, gold_len)`.
        let gold_len = cap["items"][order]["gold_continuation_tokens"]
            .as_u64()
            .ok_or_else(|| anyhow::anyhow!("capture receipt item {order} has no span length"))?
            as usize;
        anyhow::ensure!(
            cap["items"][order]["item"].as_u64() == Some(idx as u64),
            "capture receipt item order differs from this probe's"
        );
        let meta = CaptureMeta {
            hidden_size: D_RECEIVER,
            pooled_l2_raw: norms::l2(payload),
            span: gold_len - 1..gold_len,
            variant: "pc1b-identity-L19-lasttoken-goldteacherforced",
        };
        // The ONE changed argument vs PC1: `SITE` is `Site::QuestionTail`.
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
            "not_applicable": "PC1b is a same-model self-pair; no sender model is loaded or run",
        });
        row["pc1b_payload"] = serde_json::json!({
            "source": "receiver's OWN block-19 state, gold-teacher-forced, LAST span token",
            "transform": "identity",
            "l2_raw": norms::l2(payload),
            "gold_continuation_tokens": gold_len,
        });

        let discordant = q.real.0 != q.rand.0;
        let x = if discordant {
            Some(u8::from(q.real.0))
        } else {
            None
        };
        if discordant {
            if q.real.0 {
                wins += 1;
            } else {
                losses += 1;
            }
            let xv = f64::from(x.unwrap());
            wealth *= 1.0 + LAMBDA * (xv - 0.5);
            max_wealth = max_wealth.max(wealth);
        }
        println!(
            "[{}/{N_MAX}] item {idx}: aligned={} baseline={} zerovec={} random={} \
             (nll {:.3}/{:.3}/{:.3}/{:.3}) | disc={discordant} W={wealth:.4} {:.0}s",
            order + 1,
            q.real.0,
            q.base.0,
            q.zero.0,
            q.rand.0,
            q.real.1,
            q.base.1,
            q.zero.1,
            q.rand.1,
            t0.elapsed().as_secs_f32()
        );
        trajectory.push(EStep {
            order: order + 1,
            item: idx,
            aligned_correct: q.real.0,
            random_correct: q.rand.0,
            discordant,
            x,
            wealth,
        });
        rows.push(row);
        paired.push(q);

        if wealth >= w_threshold {
            crossed_at = Some(order + 1);
            println!(
                "e-process CROSSED the boundary: W = {wealth:.4} >= {w_threshold} at item {} of \
                 the stream — stopping, per the registered rule",
                order + 1
            );
            break;
        }
    }

    let e_pass = crossed_at.is_some();
    let n_disc = wins + losses;
    let n = paired.len();

    // ---- Secondary diagnostics (every rung reports these) -----------------
    let count = |f: &dyn Fn(&Quad) -> bool| paired.iter().filter(|q| f(q)).count();
    let real_c = count(&|q| q.real.0);
    let base_c = count(&|q| q.base.0);
    let zero_c = count(&|q| q.zero.0);
    let rand_c = count(&|q| q.rand.0);
    let wins_rz = count(&|q| q.real.0 && !q.zero.0);
    let loss_rz = count(&|q| !q.real.0 && q.zero.0);
    let wins_rb = count(&|q| q.real.0 && !q.base.0);
    let loss_rb = count(&|q| !q.real.0 && q.base.0);
    let zerovec_pass = 2 * zero_c >= base_c;

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
    let mean_base_nll = mean(&|q| q.base.1);

    // Fuse no-op diagnostic (M4g's, unchanged).
    let noop_acc_disagreements = count(&|q| q.zero.0 != q.base.0);
    let noop_max_abs_nll_delta = paired
        .iter()
        .map(|q| (q.zero.1 - q.base.1).abs())
        .fold(0f32, f32::max);
    let noop_exact_items = paired.iter().filter(|q| q.zero.1 == q.base.1).count();
    let noop_pass = noop_acc_disagreements == 0 && noop_max_abs_nll_delta <= FUSE_NOOP_TOL;

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
    let verdict = if e_pass && !gaming_signature {
        "PASS — mechanics LIVE at the question-tail site (liveness only, never transfer). The pathway CAN carry signal when the site is right, which narrows every prior null to 'the placeholder site was inert' and makes M5X and M4b worth running."
    } else if e_pass && gaming_signature {
        "GAMED PASS — the e-process crossed, but a pre-committed degenerate-output/NLL-collapse signature is present; docs/research/045 §3 requires this be reported as gamed, not clean"
    } else {
        "FAIL — the positive control did not move the receiver's answers even at the site where payloads demonstrably register. Under the pre-registered interpretation this is FAR STRONGER than PC1: the pathway is inert where M4i showed payloads DO register, so the ladder's nulls stand as evidence about TRANSFER rather than about plumbing. Reported without softening (ADR-032)."
    };

    let receipt = serde_json::json!({
        "stage": "run2-PC1b-positive-control-probe-at-question-tail",
        "design": "docs/adr/024-run2-trained-thought-adapter-ladder.md § 'CORRECTION + a critical interaction between M4i and PC1 (2026-08-29)' para 3, which registers PC1b. Payload + mechanics = PC1's (docs/research/045 §2 candidate (c)); SITE = M4i's; statistic = docs/adr/036-successor-rung-evaluation-protocol.md's e-process, which ADR-036 Decision 4 makes primary for every successor rung by default.",
        "env": common::env_info(&nvcc),
        "pre_committed": true,
        "variant": "pc1b-identity-L19lasttoken-goldtf-fuse-questiontail-eprocess",

        "FIREWALL_liveness_not_transfer": {
            "verbatim_research_045_section_3": "This control used a same-model, same-item, identity-transform injection with gold-adjacent content. It tests whether this repository's injection mechanics (delivery operator, payload shape, injection site, norm-rescale) are capable of carrying signal at all. It does not test, and must not be cited as evidence for or against, whether a cross-model-derived, learned-alignment payload can transfer reasoning content — that is exactly the separate question M3 through M5X exist to answer.",
            "rule": "A PASS on this rung proves LIVENESS ONLY, NEVER TRANSFER, and may NEVER be cited as evidence for the transfer claim.",
            "structural_scope": "same-model, same-item, identity transform, gold-derived content. No model boundary is crossed and no alignment is fitted anywhere in this rung.",
            "unchanged_from_pc1": true,
        },

        "pre_registered_interpretation_recorded_before_the_draw": {
            "pass": "the pathway CAN carry signal when the site is right — which narrows every prior null to 'the placeholder site was inert' and makes M5X (ADR-037) and M4b (ADR-035) worth running.",
            "fail_with_real_power": "far stronger than PC1: the pathway is inert EVEN WHERE PAYLOADS DEMONSTRABLY REGISTER, and the ladder nulls would stand as evidence about TRANSFER rather than about plumbing.",
            "gamed_pass": "docs/research/045 §3: a control that passes because the injection collapses the receiver toward a degenerate output is not evidence the pathway is USEFULLY live. Flagged below and reported as gamed, not clean.",
            "gating_role": "PC1b is the mission's gating experiment; M5X (ADR-037) and M4b (ADR-035) stay blocked behind it.",
        },

        "the_confound_this_rung_breaks": {
            "pc1_result": "receiver's own gold-TF L19 states, identity transform, at the 8 <|fim_pad|> slots: aligned 22/40, baseline 22/40, aligned_vs_baseline 0 wins / 0 losses — apparently proving the pathway inert.",
            "why_pc1_was_confounded": "PC1 ran at the placeholder site, and M4i then showed that site is exactly where payloads fail to register: moving to ordinary question-tail tokens produced a 0.148-nat NLL improvement (2.2446 vs 2.3928, 181W/119L over 300 items), two orders of magnitude outside the ~0.004-nat inertness band every placeholder-site configuration sat in.",
            "what_pc1b_separates": "'our mechanics cannot carry signal' from 'the placeholder token cannot carry signal'. Both readings currently have real evidence; only the site differs between PC1 and PC1b.",
        },

        "one_changed_factor_vs_pc1": {
            "changed": {
                "site": {"from": Site::FimPadSlots.tag(), "to": SITE.tag(),
                          "description": SITE.description()},
                "statistic": {"from": "S1a's frozen 40-item one-sided exact sign test",
                               "to": "ADR-036 e-process",
                               "why_this_is_not_a_second_free_variable": "ADR-036 Decision 4 governs every successor rung by default, and the 40-item sign test is precisely the instrument that could not measure PC1 (n_disc 3, minimum attainable one-sided p 0.125). M4i reached n_disc 66 at this site."},
            },
            "held_identical_to_pc1": {
                "payload": "the receiver's OWN block-19 residual state, teacher-forced over the item's GSM8K gold solution, LAST token of the span (de-pooled)",
                "payload_derivation": "run2_pc1_capture's code path, reused verbatim via common::render_gold + forward_capture_multi_with_rows at blocks [14,19]",
                "transform": "IDENTITY — no adapter artifact is loaded, constructed or applied anywhere in this binary",
                "delivery": INJECT_MODE.tag(),
                "pooling": "de-pooled",
                "slots": N_SLOTS,
                "receiver_inject_block": RECEIVER_BLOCK,
                "rescale_to_natural_median": true,
                "decoding": "greedy, batch=1, max_new_tokens=400",
                "conditions": "produced by the SAME shared common::m3::four_conditions_at PC1 called — the site is ONE ARGUMENT to that function, so no control is redefined by this rung",
            },
        },

        "site_provenance": {
            "resolver": "common::m3::build_site_prompt, reused verbatim — NOT reimplemented, which is what makes the site provably identical to M4i's rather than merely similar",
            "site_tag": SITE.tag(),
            "m4i_receipt": M4I_RECEIPT,
            "m4i_recorded_site": m4i_site,
            "site_tag_matches_m4i": site_tag_matches_m4i,
            "item_stream_identical_to_m4i": stream_identical_to_m4i,
            "samples": site_samples,
            "tokenization_excluded": tokenization_excluded,
        },

        "payload_provenance": {
            "capture_receipt": CAPTURE_RECEIPT,
            "payload_file": payload_path.display().to_string(),
            "payload_sha256": payload_sha,
            "file_sha_gate_was_replaced": {
                "registered_gate": "payload sha256 matches PC1",
                "why_unsatisfiable": "PC1's payload artifact holds exactly 40 vectors (S1a's frozen probe set); the e-process stream is 300 items from adaptation-512, and the two sets intersect in exactly ONE item (train index 1153). The payload is per-item by construction, so 299 stream items have no PC1 payload anywhere on disk. Holding the FILE fixed would confine PC1b to 40 items — the powerless regime PC1 already occupied (n_disc 3).",
                "replacement": "the payload DERIVATION RULE is held byte-identical instead, and verified by BIT-IDENTITY against PC1's committed vector on the shared item — a real cross-artifact equality check that a file-level sha over disjoint item sets could not have been.",
                "approved_by_the_coordinator_before_any_capture_ran": true,
                "see": "run2-pc1b-capture-receipt.json § declared_deviation_from_the_registered_payload_gate",
            },
            "bit_identity_vs_pc1": bit_identity,
        },

        "manifold_precheck_run_before_this_draw": {
            "receipt": PRECHECK_RECEIPT,
            "candidate_label": PRECHECK_LABEL,
            "payload_sha_matches_this_probe": true,
            "classification": precheck_class,
            "surprise_flag": precheck_surprise,
            "expected": "on-manifold-item-varying — trivially, because the payload IS a real receiver state",
            "gating": "NONE — ORDERING ONLY, per ADR-024's M4f framing.",
            "row": precheck_row,
        },

        "e_process": {
            "protocol": "docs/adr/036-successor-rung-evaluation-protocol.md Decision 1 (ADR-030 §3.2's betting rule, adopted verbatim)",
            "wealth_rule": "W_0 = 1; on each DISCORDANT pair W <- W * (1 + lambda * (x - 0.5)) where x = 1 if the aligned condition is correct and the random condition is not, x = 0 if the reverse. Concordant items produce no update.",
            "lambda": LAMBDA,
            "lambda_note": "lambda = 2*theta - 1 tuned to the smallest interesting effect theta = 0.65. Fixed in advance; never re-parametrised after seeing W_t.",
            "alpha": E_ALPHA,
            "wealth_threshold": w_threshold,
            "n_max": N_MAX,
            "primary_comparison": "aligned_real vs random",
            "stopping_rule": "stop and PASS at the first item where W >= 1/alpha; otherwise consume the full N_max and report the final wealth.",
            "never_restarted": "This is ONE registered draw over the fixed stream, run once. No item was re-drawn, no parameter re-tuned, and the process was never restarted (ADR-032 honest-fail).",
            "items_drawn": n,
            "n_discordant": n_disc,
            "wins": wins,
            "losses": losses,
            "crossed_at_item": crossed_at,
            "final_wealth": wealth,
            "max_wealth_reached": max_wealth,
            "pass": e_pass,
            "power_context": "M4i reached n_disc 66 at this same site over the same stream — an order of magnitude more discordance than any 40-item draw, which is why this rung is expected to have real power where PC1 (n_disc 3) had none.",
            "no_p_value_translation": "This receipt deliberately reports NO exact-sign or mid-p McNemar p-value for the primary accuracy comparison. The e-process outcome is reported on its own scale (crossing item count, or final wealth at N_max) and is NOT translated into an equivalent p-value: a fixed-sample test's p and a sequential test's stopping wealth answer structurally different questions, and a false equivalence would misrepresent both (ADR-036 Decision 3).",
            "full_trajectory": trajectory.iter().map(|s| serde_json::json!({
                "order": s.order, "item": s.item,
                "aligned_correct": s.aligned_correct, "random_correct": s.random_correct,
                "discordant": s.discordant, "x": s.x, "wealth": s.wealth,
            })).collect::<Vec<_>>(),
        },

        "item_supply": {
            "source": ADAPTATION_512,
            "source_split": "adaptation-512",
            "consumption_order": "the file's own fixed ascending index order (ADR-036 Decision 2(1))",
            "leakage_exclusions": LEAKAGE_EXCLUSIONS,
            "leakage_exclusions_present_in_split": excluded_present,
            "eligible_pool": eligible.len(),
            "must_travel_with_every_headline_number": "Per ADR-036 Decision 3, any write-up quoting a number from this receipt must state in the same sentence that it was produced under the e-process protocol on the adaptation-512 stream, not under the frozen 40-item protocol. PC1's numbers were produced under the 40-item protocol and are NOT directly comparable.",
        },

        "dataset": {"source": common::GSM8K_TRAIN_URL, "sha256": train_sha, "indices": stream},
        "items": rows,

        "summary": {
            "n_evaluated": n,
            "accuracy": {"aligned_real": real_c, "baseline_uninjected": base_c,
                          "zerovec_injected": zero_c, "random": rand_c},
            "primary_e_process_aligned_vs_random": {
                "wins": wins, "losses": losses, "n_discordant": n_disc,
                "final_wealth": wealth, "max_wealth_reached": max_wealth,
                "threshold": w_threshold, "crossed_at_item": crossed_at, "pass": e_pass},
            "aligned_vs_zerovec": {"wins": wins_rz, "losses": loss_rz},
            "aligned_vs_baseline": {"wins": wins_rb, "losses": loss_rb},
            "zerovec_vs_baseline": {
                "degenerate_under_fuse": true,
                "criterion": "2 x zerovec_correct >= baseline_correct",
                "pass": zerovec_pass,
                "note": "trivially satisfied because zerovec == baseline under fuse; no evidential weight"},
            "nll_mean": {"aligned_real": mean_real_nll,
                          "baseline_uninjected": mean_base_nll,
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
            "nll_note": "The NLL sign tests are SECONDARY diagnostics on the same collected pairs. They are not the registered primary and carry no e-process interpretation.",
            "nll_harm_accounting": {
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
                "why": "ITI (arXiv:2306.03341) documents an intervention that appears to restore the answer while merely collapsing the model into a degenerate response. docs/research/045 §3 makes this check mandatory on EVERY positive-control draw, including one 'supposed to' pass.",
                "thresholds_pre_committed": {
                    "degenerate_len_ratio": DEGENERATE_LEN_RATIO,
                    "degenerate_item_fraction": DEGENERATE_ITEM_FRACTION,
                    "nll_collapse_nats": NLL_COLLAPSE_NATS,
                    "note": "PC1's values, unchanged, so the two rungs' guards are directly comparable",
                },
                "mean_aligned_over_baseline_generated_char_ratio": mean_len_ratio,
                "degenerate_short_items": degenerate_short_items,
                "degenerate_output_signature": degenerate_signature,
                "mean_aligned_nll": mean_real_nll,
                "nll_collapse_signature": nll_collapse_signature,
                "nll_collapse_reading": "For a GOLD-derived payload a large NLL drop is the expected direction — the payload encodes the answer. §3 requires it be FLAGGED rather than excused: an NLL collapsed below the threshold means the receiver was handed the answer, not that it computed it, and any PASS resting on that is a liveness result about the CHANNEL's bandwidth, still never about transfer.",
                "gaming_signature": gaming_signature,
            },
        },

        "comparison_pc1_vs_pc1b": {
            "pc1_receipt": PC1_RECEIPT,
            "note": "PC1 ran the 40-item sign-test protocol at the fim_pad site; PC1b runs the e-process at the question-tail site. Per ADR-036 Decision 3 the two statistics are NOT directly comparable and neither is translated into the other's scale. The ACCURACY and NLL columns are reported side by side because both rungs used the same four conditions and the same payload derivation.",
            "pc1_summary": serde_json::from_slice::<serde_json::Value>(
                &std::fs::read(crate_path(PC1_RECEIPT)).unwrap_or_default())
                .map(|v| v["summary"].clone()).unwrap_or(serde_json::Value::Null),
            "m4i_summary_for_site_context": m4i["summary"].clone(),
        },

        "gates": {
            "payload_sha256_matches_capture_receipt": {"pass": true, "sha256": payload_sha},
            "payload_bit_identical_to_pc1_on_shared_item": {
                "pass": cap["gates"]["pc1_overlap_item_bit_identical"]["pass"].as_bool() == Some(true),
                "note": "the replacement for the unsatisfiable file-sha gate; see payload_provenance.file_sha_gate_was_replaced"},
            "identity_transform_no_adapter_weights_loaded": {"pass": true,
                "note": "no MlpTransform / FastGrnnTransform / AffineTransform is constructed anywhere in this binary; the injected vector IS a captured receiver state, byte-for-byte"},
            "site_provably_identical_to_m4i": {
                "pass": site_tag_matches_m4i && stream_identical_to_m4i,
                "site_tag_matches": site_tag_matches_m4i,
                "item_stream_matches": stream_identical_to_m4i,
                "note": "build_site_prompt is reused verbatim from M4i, and the resolved item stream is asserted equal to M4i's committed one"},
            "injection_mode_recorded": {"pass": true, "mode": INJECT_MODE.tag(),
                "equation": INJECT_MODE.equation()},
            "slot_count_unchanged": {"pass": true, "slots": N_SLOTS},
            "manifold_precheck_ran_before_this_draw": {"pass": true, "receipt": PRECHECK_RECEIPT,
                "classification": precheck_class, "gating": "ordering only"},
            "eprocess_stream_matches_capture": {"pass": true, "n": n},
            "eprocess_never_restarted": {"pass": true},
            "PC1b_aligned_real_vs_random_eprocess": {"pass": e_pass,
                "final_wealth": wealth, "threshold": w_threshold,
                "n_discordant": n_disc, "crossed_at_item": crossed_at},
            "zerovec_not_catastrophic": {"pass": zerovec_pass, "degenerate_under_fuse": true},
            "fuse_zero_payload_is_a_noop": {"pass": noop_pass, "gating": "none — diagnostic"},
            "no_gaming_signature": {"pass": !gaming_signature,
                "degenerate_output_signature": degenerate_signature,
                "nll_collapse_signature": nll_collapse_signature},
        },
        "gate_pass": e_pass && zerovec_pass && !gaming_signature,
        "verdict": verdict,
        "honest_fail_contract": "ADR-032: ONE registered draw, no retry, full numbers reported either way. A FAILING positive control WITH REAL POWER is the MORE consequential outcome and is stated plainly here, not softened. This receipt is written before any interpretation is added to the ADR.",
        "wall_clock_s": t0.elapsed().as_secs_f32(),
    });
    common::write_receipt(
        &crate_path("receipts"),
        "run2-pc1b-receipt-identity-L19lasttoken-goldtf-fuse-questiontail-slots8-eprocess.json",
        &receipt,
    )?;

    println!("\n=== PC1b (positive control at the question-tail site — LIVENESS ONLY, NEVER TRANSFER) ===");
    println!(
        "acc aligned {real_c}/{n} baseline {base_c}/{n} zerovec {zero_c}/{n} random {rand_c}/{n}"
    );
    println!(
        "e-process: wins {wins}, losses {losses}, n_discordant {n_disc}; final W {wealth:.4} \
         (max ever {max_wealth:.4}) vs threshold {w_threshold} => pass={e_pass} \
         (crossed_at={crossed_at:?})"
    );
    println!(
        "NLL means: aligned {mean_real_nll:.4} baseline {mean_base_nll:.4} zerovec {:.4} random {:.4}",
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
    println!("manifold pre-check: {precheck_class} (surprise={precheck_surprise})");
    println!("VERDICT: {verdict}");
    Ok(())
}
