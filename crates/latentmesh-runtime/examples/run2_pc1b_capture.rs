//! Run-2 **PC1b capture** — PC1's payload derivation, applied to the
//! ADR-036 e-process item stream.
//!
//! ADR-024 § "CORRECTION + a critical interaction between M4i and PC1
//! (2026-08-29)" ¶3, which registers PC1b: *"repeat PC1 (receiver's own L19
//! states, identity transform, fuse, de-pooled) at the question-tail
//! ordinary-token site M4i used, under the e-process."*
//!
//! # Why this binary exists at all — the declared deviation, stated up front
//!
//! PC1b's brief says "reuse EXACTLY PC1's payload (sha256 must match)" AND
//! "evaluate under the ADR-036 e-process (adaptation-512, N_max = 300)".
//! **Those two instructions are arithmetically incompatible, and the
//! incompatibility is a fact about the data, not a judgement call:**
//!
//! - PC1's payload artifact holds exactly **40** vectors — one per item of
//!   S1a's frozen probe set.
//! - The e-process stream is drawn from `adaptation-512` in fixed index
//!   order (ADR-036 Decision 2), and S1a's 40 items intersect
//!   `adaptation-512` in **exactly one** item, train index **1153**.
//!   ADR-036 Decision 2 records this itself, and it is re-derived here by
//!   direct set intersection rather than quoted.
//! - The payload is **per-item by construction** (it is *that item's* own
//!   gold-teacher-forced state), so 299 of the 300 stream items have no
//!   payload anywhere on disk.
//!
//! Holding the payload *file* fixed would therefore confine PC1b to 40 items
//! — precisely the powerless regime PC1 already occupied (`n_disc = 3`,
//! minimum attainable one-sided p = 0.125) — and would defeat the entire
//! reason PC1b exists, which is to ask the liveness question **with real
//! power** at the site M4i showed payloads actually register at.
//!
//! **The resolution: hold the payload DERIVATION RULE byte-identical instead
//! of the payload FILE, and replace the unsatisfiable file-sha gate with
//! three strictly checkable ones**, all enforced below:
//!
//! 1. `pc1_reference_payload_sha256_intact` — PC1's committed payload file
//!    still matches the sha256 its own capture receipt pins. The reference
//!    artifact is verified present and unmodified before it is used.
//! 2. `derivation_is_the_shared_code_path` — the gold text comes from
//!    `common::render_gold` (PC1's own rule, promoted into `common` for this
//!    rung) and the tap is the same `forward_capture_multi_with_rows` call at
//!    the same `PAYLOAD_BLOCK = 19`, last row of the gold span.
//! 3. `pc1_overlap_item_bit_identical` — **the load-bearing one.** For train
//!    index 1153, the item present in BOTH PC1's 40 and this stream's 300,
//!    the freshly-derived payload must equal PC1's committed vector
//!    **byte-for-byte**. This is a real cross-artifact equality check on the
//!    whole derivation (gold rendering, tokenisation, teacher forcing, tap
//!    depth, row selection) and is the strongest evidence available that this
//!    binary reproduces PC1's payload rather than something merely similar.
//!
//! The cost is ~300 teacher-forced prefills (PC1's 40 receipted 4.65 s), zero
//! training, zero new *kind* of measurement. PC1 itself shipped an analogous
//! declared deviation — its registered capture source did not exist as
//! written — and this receipt follows that precedent rather than inventing
//! one.
//!
//! # What is captured
//!
//! Per stream item, the same three vectors PC1 stored, in the same layout, so
//! the pre-check reads both artifacts through identical code:
//! `[L19_last (THE PAYLOAD) | L14_pooled (reference) | L14_last (reference)]`.
//!
//! **No adapter of any kind is constructed here.** No `MlpTransform`,
//! `FastGrnnTransform` or `AffineTransform` is named, loaded or applied. The
//! payload IS a captured receiver state.
//!
//! **FIREWALL (`docs/research/045` §3):** a PC1b PASS proves **LIVENESS
//! ONLY, NEVER TRANSFER**, and may never be cited as evidence for the
//! transfer claim.
//!
//! Run: PATH=/usr/local/cuda-12.8/bin:$PATH \
//!      cargo run --release --features cuda --example run2_pc1b_capture

#![recursion_limit = "512"]

#[path = "common/mod.rs"]
#[allow(dead_code)]
mod common;

use common::m3::{build_site_prompt, Site, N_SLOTS, RECEIVER, RECEIVER_BLOCK, SYSTEM};
use latentmesh_runtime::capture::forward_capture_multi_with_rows;
use latentmesh_runtime::{norms, QwenRuntime};
use std::path::{Path, PathBuf};

const GSM8K_TRAIN_SHA256: &str = "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465";
const ADAPTATION_512: &str = "../../harness/latentmesh-live/data/adaptation-512.json";
const PC1_CAPTURE_RECEIPT: &str = "receipts/run2-pc1-capture-receipt.json";
const M4I_RECEIPT: &str =
    "receipts/run2-m4i-receipt-cellL18toL14-mlp-pertokenlast-fuse-questiontail-slots8-eprocess.json";

/// The payload tap: S1a's capture block, unchanged through PC1.
const PAYLOAD_BLOCK: usize = 19;
const D_RECEIVER: usize = 1536;
/// `[L19_last | L14_pooled | L14_last]` — PC1's layout, unchanged.
const VECS_PER_ITEM: usize = 3;

/// PC1b's site. The ONLY factor that differs from PC1.
const SITE: Site = Site::QuestionTail;

/// ADR-024's 13 probe-overlap items, verbatim from M4i's own copy.
const LEAKAGE_EXCLUSIONS: [usize; 13] = [
    150, 1309, 1573, 2365, 2418, 2540, 2958, 2973, 3084, 3825, 3844, 3877, 4746,
];
const N_MAX: usize = 300;

fn crate_path(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

#[allow(clippy::too_many_lines)]
fn main() -> anyhow::Result<()> {
    let t0 = std::time::Instant::now();
    let e = anyhow::Error::msg;
    let nvcc = latentmesh_runtime::assert_cuda_build_env().map_err(e)?;
    println!("build-env guard: {nvcc}");

    // ---- Gate 1: PC1's reference payload, verified intact -----------------
    let pc1: serde_json::Value =
        serde_json::from_slice(&std::fs::read(crate_path(PC1_CAPTURE_RECEIPT))?)?;
    let pc1_path =
        PathBuf::from(pc1["payload_file"]["path"].as_str().ok_or_else(|| {
            anyhow::anyhow!("PC1 capture receipt does not name its payload file")
        })?);
    let pc1_want_sha = pc1["payload_file"]["sha256"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("PC1 capture receipt does not pin a payload sha256"))?
        .to_string();
    let pc1_bytes = std::fs::read(&pc1_path).map_err(|err| {
        anyhow::anyhow!(
            "PC1's reference payload {} is unreadable ({err}) — PC1b's bit-identity gate cannot \
             run without it",
            pc1_path.display()
        )
    })?;
    let pc1_got_sha = common::sha256_hex(&pc1_bytes);
    anyhow::ensure!(
        pc1_got_sha == pc1_want_sha,
        "PC1's reference payload sha256 {pc1_got_sha} != its own capture-receipt-pinned \
         {pc1_want_sha} — the reference artifact has changed and no comparison against PC1 is valid"
    );
    let pc1_n: usize = pc1["payload_file"]["n_items"].as_u64().unwrap_or(0) as usize;
    anyhow::ensure!(
        pc1_bytes.len() == pc1_n * VECS_PER_ITEM * D_RECEIVER * 4,
        "PC1 reference payload size mismatch"
    );
    let pc1_flat: Vec<f32> = pc1_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let pc1_indices: Vec<usize> = pc1["dataset"]["indices"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("PC1 capture receipt has no dataset.indices"))?
        .iter()
        .filter_map(|v| v.as_u64())
        .map(|v| v as usize)
        .collect();
    anyhow::ensure!(pc1_indices.len() == pc1_n, "PC1 index count mismatch");
    println!(
        "PC1 reference payload VERIFIED intact: sha {pc1_got_sha} ({pc1_n} items x {VECS_PER_ITEM} \
         x {D_RECEIVER} f32)"
    );

    // ---- Item supply: ADR-036 Decision 2, resolved exactly as M4i does ----
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
        "adaptation-512 was drawn from a different train.jsonl than this capture loaded"
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
        "adaptation-512 is not in ascending index order"
    );
    let eligible: Vec<usize> = adaptation_indices
        .iter()
        .copied()
        .filter(|i| !LEAKAGE_EXCLUSIONS.contains(i))
        .collect();
    anyhow::ensure!(
        eligible.len() >= N_MAX,
        "eligible pool ({}) < N_max ({N_MAX})",
        eligible.len()
    );

    // The overlap between S1a's frozen 40 and this stream — computed, not quoted.
    let overlap: Vec<usize> = pc1_indices
        .iter()
        .copied()
        .filter(|i| eligible.contains(i))
        .collect();
    println!(
        "item supply: adaptation-512, fixed index order, 13-item exclusion applied; {} eligible. \
         Overlap with PC1's 40-item payload set: {overlap:?}",
        eligible.len()
    );

    // ---- Model: receiver ONLY (PC1b is a same-model self-pair) ------------
    let device = latentmesh_runtime::device().map_err(e)?;
    println!("loading {RECEIVER} (BF16)... [no sender, no adapter]");
    let mut rt = QwenRuntime::load(RECEIVER, &device, candle_core::DType::BF16).map_err(e)?;
    let pad_id = rt
        .tokenizer
        .token_to_id("<|fim_pad|>")
        .ok_or_else(|| anyhow::anyhow!("<|fim_pad|> not in tokenizer"))?;
    println!("model loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // ---- Site pre-flight — the SAME resolution M4i performed --------------
    // Run here as well as in the probe so the capture covers exactly the
    // items the probe will draw, in exactly the probe's order. The probe
    // re-resolves independently and asserts equality with this list.
    let mut stream: Vec<usize> = Vec::new();
    let mut tokenization_excluded: Vec<serde_json::Value> = Vec::new();
    for &idx in &eligible {
        if stream.len() == N_MAX {
            break;
        }
        match build_site_prompt(&rt, &all_items[idx], pad_id, SITE) {
            Ok(_) => stream.push(idx),
            Err(err) => tokenization_excluded.push(serde_json::json!({
                "item": idx, "reason": err.to_string(),
            })),
        }
    }
    anyhow::ensure!(
        stream.len() == N_MAX,
        "site pre-flight resolved only {} of {N_MAX}",
        stream.len()
    );

    // The stream must be M4i's stream, item for item. This is what makes
    // "the site is provably identical to M4i" a checked fact rather than a
    // claim about intent: same population, same order, same 8 positions.
    let m4i: serde_json::Value = serde_json::from_slice(&std::fs::read(crate_path(M4I_RECEIPT))?)?;
    let m4i_stream: Vec<usize> = m4i["items"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("M4i receipt has no items array"))?
        .iter()
        .filter_map(|r| r["item"].as_u64())
        .map(|v| v as usize)
        .collect();
    anyhow::ensure!(
        m4i_stream == stream,
        "the resolved stream differs from M4i's committed 300-item stream — the site would not be \
         the same site"
    );
    println!(
        "site pre-flight: {} items resolved to {N_SLOTS} ordinary question-tail positions, {} \
         excluded on tokenisation grounds; stream is IDENTICAL to M4i's committed stream",
        stream.len(),
        tokenization_excluded.len()
    );

    // ---- Payload capture: gold-teacher-forced, per item -------------------
    // PC1's loop, unchanged except for which items it walks.
    let mut flat: Vec<f32> = Vec::with_capacity(N_MAX * VECS_PER_ITEM * D_RECEIVER);
    let mut rows = Vec::new();
    let mut overlap_checks = Vec::new();
    for (done, &idx) in stream.iter().enumerate() {
        let item = &all_items[idx];
        let gold_text = common::render_gold(&item.answer_text);
        let prompt = QwenRuntime::chat_prompt(
            SYSTEM,
            &format!("{}\n\n{}", item.question, common::ANSWER_FORMAT),
        );
        let ptoks = rt.encode(&prompt).map_err(e)?;
        let ctoks = rt.encode(&gold_text).map_err(e)?;
        anyhow::ensure!(!ctoks.is_empty(), "item {idx}: empty gold continuation");
        let full: Vec<u32> = ptoks.iter().chain(ctoks.iter()).copied().collect();
        let span = ptoks.len()..full.len();
        let (_, caps) = forward_capture_multi_with_rows(
            &mut rt.model,
            &full,
            &[RECEIVER_BLOCK, PAYLOAD_BLOCK],
            span,
            &device,
        )
        .map_err(e)?;
        let n = ctoks.len();
        let l14 = &caps[0];
        let l19 = &caps[1];
        anyhow::ensure!(
            l19.rows.len() == n * D_RECEIVER,
            "item {idx}: L19 row shape"
        );
        let l19_last = l19.rows[(n - 1) * D_RECEIVER..].to_vec();
        let l14_pooled = l14.capture.pooled.clone();
        let l14_last = l14.rows[(n - 1) * D_RECEIVER..].to_vec();
        anyhow::ensure!(
            l19_last.iter().all(|v| v.is_finite())
                && l14_pooled.iter().all(|v| v.is_finite())
                && l14_last.iter().all(|v| v.is_finite()),
            "item {idx}: non-finite captured state"
        );

        // ---- Gate 3: bit-identity against PC1, where both cover the item --
        if let Some(k) = pc1_indices.iter().position(|&p| p == idx) {
            let base = k * VECS_PER_ITEM * D_RECEIVER;
            let pc1_l19 = &pc1_flat[base..base + D_RECEIVER];
            let identical = pc1_l19 == l19_last.as_slice();
            let max_abs = pc1_l19
                .iter()
                .zip(l19_last.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            println!(
                "  bit-identity vs PC1 at item {idx}: identical={identical} (max|d| {max_abs:.3e})"
            );
            overlap_checks.push(serde_json::json!({
                "item": idx,
                "pc1_payload_row": k,
                "pc1b_stream_order": done + 1,
                "bit_identical": identical,
                "max_abs_elementwise_delta": max_abs,
                "what_it_proves": "the whole derivation — gold rendering, tokenisation, teacher forcing, tap depth 19, last-row selection — reproduces PC1's committed payload exactly for this item, so PC1b's 300 payloads are PC1's payload rule applied to more items, not a different construction.",
            }));
            anyhow::ensure!(
                identical,
                "item {idx}: the re-derived payload is NOT bit-identical to PC1's committed vector \
                 (max|d| {max_abs:.3e}) — PC1b's derivation has drifted from PC1's and the rung \
                 must not proceed"
            );
        }

        rows.push(serde_json::json!({
            "item": idx,
            "gold": item.gold,
            "prompt_tokens": ptoks.len(),
            "gold_continuation_tokens": n,
            "payload_l19_last_l2": norms::l2(&l19_last),
            "reference_l19_pooled_l2": norms::l2(&l19.capture.pooled),
            "reference_l14_pooled_l2": norms::l2(&l14_pooled),
            "reference_l14_last_l2": norms::l2(&l14_last),
        }));
        flat.extend_from_slice(&l19_last);
        flat.extend_from_slice(&l14_pooled);
        flat.extend_from_slice(&l14_last);
        if (done + 1) % 50 == 0 || done + 1 == stream.len() {
            println!(
                "[{}/{N_MAX}] item {idx}: {n} gold tokens, |L19_last| {:.2}  {:.0}s",
                done + 1,
                norms::l2(&l19_last),
                t0.elapsed().as_secs_f32()
            );
        }
    }
    anyhow::ensure!(
        !overlap_checks.is_empty(),
        "no item was shared with PC1's payload set — the bit-identity gate could not run, so the \
         claim that this is PC1's derivation is unverified and the rung must not proceed"
    );

    // ---- Write the payload artifact ---------------------------------------
    let out = dir.join("run2-pc1b-payloads.f32bin");
    let mut bytes = Vec::with_capacity(flat.len() * 4);
    for v in &flat {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(&out, &bytes)?;
    let payload_sha = common::sha256_hex(&bytes);
    println!(
        "wrote {} ({} bytes, sha {payload_sha})",
        out.display(),
        bytes.len()
    );

    let receipt = serde_json::json!({
        "stage": "run2-PC1b-capture",
        "design": "docs/adr/024-run2-trained-thought-adapter-ladder.md § 'CORRECTION + a critical interaction between M4i and PC1 (2026-08-29)' ¶3, which registers PC1b: repeat PC1 (receiver's own L19 states, identity transform, fuse, de-pooled) at the question-tail ordinary-token site M4i used, under the e-process. Payload derivation = docs/research/045 §2 candidate (c), unchanged from PC1.",
        "env": common::env_info(&nvcc),
        "pre_committed": true,

        "FIREWALL_liveness_not_transfer": {
            "verbatim_research_045_section_3": "This control used a same-model, same-item, identity-transform injection with gold-adjacent content. It tests whether this repository's injection mechanics (delivery operator, payload shape, injection site, norm-rescale) are capable of carrying signal at all. It does not test, and must not be cited as evidence for or against, whether a cross-model-derived, learned-alignment payload can transfer reasoning content — that is exactly the separate question M3 through M5X exist to answer.",
            "rule": "A PASS on PC1b proves LIVENESS ONLY, NEVER TRANSFER, and may NEVER be cited as evidence for the transfer claim.",
        },

        "declared_deviation_from_the_brief": {
            "registered_text": "PC1b brief: 'Reuse EXACTLY PC1 payload ... (payload sha256 recorded in run2-pc1-capture-receipt.json and the PC1 probe receipt — assert it matches)' together with 'Evaluate under ADR-036 e-process (adaptation-512 fixed order, 13 exclusions, registered ... N_max)'.",
            "why_it_is_not_executable_as_written": [
                "COVERAGE: PC1's payload artifact holds exactly 40 vectors, one per item of S1a's frozen probe set. The e-process stream is adaptation-512 in fixed index order. Those two sets intersect in EXACTLY ONE item, train index 1153 — re-derived here by direct set intersection, not quoted from ADR-036. 299 of the 300 stream items therefore have no payload on disk.",
                "PER-ITEM BY CONSTRUCTION: the payload is THAT item's own gold-teacher-forced state. There is no item-independent payload to reuse; holding the file fixed necessarily holds the ITEM SET fixed at 40.",
                "SELF-DEFEATING: a 40-item PC1b would reproduce PC1's own powerless regime (n_disc 3, min attainable one-sided p 0.125) at a new site, which is exactly the failure PC1b was registered to escape. The brief's own goal — 'this rung should finally have real power' — requires the 300-item stream."
            ],
            "what_was_done_instead": "The payload DERIVATION RULE is held byte-identical instead of the payload FILE, and the unsatisfiable file-sha gate is replaced by three checkable ones: (1) PC1's committed payload file is verified still to match its own receipt-pinned sha256; (2) the derivation runs through the shared common::render_gold plus the same forward_capture_multi_with_rows tap at block 19, last row of the gold span; (3) BIT-IDENTITY — for every item shared by both sets (train index 1153), the freshly-derived payload must equal PC1's committed vector byte-for-byte, or this binary aborts.",
            "direction_of_the_deviation": "TOWARD the registered design (the e-process at N_max=300 with real power, at M4i's site), not away from it. The cost is 300 teacher-forced prefills — PC1's 40 receipted 4.65 s — with zero training and no new KIND of measurement.",
            "precedent": "PC1 itself shipped a declared deviation of the same class (its registered capture source did not exist as described). This receipt follows that precedent rather than silently re-specifying the rung.",
            "what_is_NOT_deviated_from": "the payload semantics (receiver's OWN block-19 state, gold-teacher-forced, de-pooled to the span's LAST token), the identity transform, the absence of any adapter, the mechanics (fuse, 8 slots, L14 delivery, rescale-to-median), and the e-process parameters.",
        },

        "config": {
            "receiver": RECEIVER,
            "sender": "NONE — same-model self-pair",
            "transform": "IDENTITY — no MlpTransform / FastGrnnTransform / AffineTransform is constructed, loaded or applied anywhere in this binary",
            "payload_block": PAYLOAD_BLOCK,
            "reference_block": RECEIVER_BLOCK,
            "payload": "the receiver's OWN block-19 residual state, teacher-forced over the item's GSM8K gold solution, LAST token of the span (de-pooled)",
            "gold_rendering_rule": "common::render_gold — PC1's own pre-committed rule, promoted into common for this rung so both binaries share ONE definition. Behavioural identity is proved empirically by the bit-identity gate, not asserted.",
            "site_this_payload_is_for": {"tag": SITE.tag(), "description": SITE.description()},
            "slots": N_SLOTS,
        },

        "pc1_reference": {
            "capture_receipt": PC1_CAPTURE_RECEIPT,
            "payload_file": pc1_path.display().to_string(),
            "sha256_pinned": pc1_want_sha,
            "sha256_measured_now": pc1_got_sha,
            "n_items": pc1_n,
            "indices": pc1_indices,
        },

        "item_supply": {
            "adr": "ADR-036 Decision 2",
            "source": "harness/latentmesh-live/data/adaptation-512.json",
            "source_split": "adaptation-512",
            "consumption_order": "the file's own fixed ascending index order, sequential, never shuffled and never re-seeded",
            "leakage_exclusion": {"excluded_item_indices": LEAKAGE_EXCLUSIONS,
                "present_in_this_split": LEAKAGE_EXCLUSIONS.iter().filter(|i| adaptation_indices.contains(i)).count()},
            "eligible_pool_size": eligible.len(),
            "n_max": N_MAX,
            "tokenization_preflight_exclusions": tokenization_excluded,
            "stream_is_identical_to_m4i": true,
            "m4i_receipt_compared_against": M4I_RECEIPT,
            "overlap_with_pc1_payload_set": overlap,
        },

        "payload_file": {
            "path": out.display().to_string(),
            "sha256": payload_sha,
            "n_items": N_MAX,
            "dim": D_RECEIVER,
            "vecs_per_item": VECS_PER_ITEM,
            "layout": "300 items in the e-process stream order; per item three contiguous f32 vectors of 1536, row-major little-endian: [L19_last (THE PAYLOAD) | L14_pooled (on-manifold reference) | L14_last (un-pooled on-manifold reference)] — PC1's layout, unchanged, so the pre-check reads both artifacts through identical code",
        },

        "dataset": {"source": common::GSM8K_TRAIN_URL, "sha256": train_sha, "indices": stream},
        "items": rows,

        "gates": {
            "pc1_reference_payload_sha256_intact": {"pass": true,
                "pinned": pc1_want_sha, "measured": pc1_got_sha,
                "note": "PC1's committed payload artifact is present and unmodified; every comparison below is against a verified reference"},
            "derivation_is_the_shared_code_path": {"pass": true,
                "note": "common::render_gold + forward_capture_multi_with_rows at blocks [14, 19], last row of the gold span — the same statements run2_pc1_capture executes"},
            "pc1_overlap_item_bit_identical": {"pass": true,
                "checks": overlap_checks,
                "note": "THE substantive gate replacing the unsatisfiable file-sha match. Any non-identity aborts the binary before a payload file is written."},
            "identity_transform_no_adapter_weights_loaded": {"pass": true,
                "note": "no adapter type is named, constructed or applied anywhere in this binary; the payload IS a captured receiver state, byte-for-byte"},
            "stream_identical_to_m4i": {"pass": true, "n": N_MAX,
                "note": "the 300 items resolved here equal M4i's committed 300-item stream, item for item and in order — the site claim is checked, not asserted"},
            "all_captured_states_finite": {"pass": true},
            "slot_count_unchanged": {"pass": true, "slots": N_SLOTS},
        },
        "gate_pass": true,
        "wall_clock_s": t0.elapsed().as_secs_f32(),
    });
    common::write_receipt(
        &crate_path("receipts"),
        "run2-pc1b-capture-receipt.json",
        &receipt,
    )?;
    println!(
        "PC1b capture complete: {N_MAX} payloads, sha {payload_sha}, in {:.1}s",
        t0.elapsed().as_secs_f32()
    );
    Ok(())
}
