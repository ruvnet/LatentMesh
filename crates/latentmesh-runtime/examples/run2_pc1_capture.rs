//! Run-2 **PC1 capture** — build the positive control's payloads.
//!
//! ADR-024 § "PC1 PRE-REGISTRATION (2026-08-29) — the positive control this
//! harness never had"; design `docs/research/045-positive-control-design.md`
//! §2 candidate **(c)** ("gold-teacher-forced self-pair oracle") and §5's
//! pre-registerable spec.
//!
//! **What PC1 is.** Every null since S1a is ambiguous between "no transfer
//! effect exists" and "these mechanics cannot carry signal". S1a — the
//! ladder's only PASS (p = 0.03125) — ran under **overwrite + pooled +
//! inject-at-L19**, mechanics nothing since M4c uses. This rung re-runs a
//! positive control under the **current** mechanics (fuse, de-pooled,
//! `<|fim_pad|>` slots, the L18→L14 cell's receiver site) with the strongest
//! constructible payload: the receiver's **own block-19 state, teacher-forced
//! on the item's GOLD solution**, delivered back into itself by an
//! **identity transform** (no adapter of any kind).
//!
//! **FIREWALL — reused verbatim from `docs/research/045` §3, and repeated in
//! every receipt this rung writes:** *"This control used a same-model,
//! same-item, identity-transform injection with gold-adjacent content. It
//! tests whether this repository's injection mechanics (delivery operator,
//! payload shape, injection site, norm-rescale) are capable of carrying
//! signal at all. It does not test, and must not be cited as evidence for or
//! against, whether a cross-model-derived, learned-alignment payload can
//! transfer reasoning content — that is exactly the separate question M3
//! through M5X exist to answer."* A PASS proves **liveness, never transfer**.
//!
//! **DECLARED DEVIATION from the brief and from `docs/research/045` §5's
//! "Capture: none new" (disclosed, not silently taken).** §5 assumed the
//! committed per-token dump could supply the payload. Two facts, both
//! measured here and recorded in the receipt, make that impossible for the
//! registered item set:
//!   1. `receiver_L19.tok.f32bin` covers the **calibration-4000** split
//!      (2,560 items, gsm8k-train indices 1..4797). **Only 13 of S1a's 40
//!      items are in it.** An n = 13 draw is far below the power floor
//!      `docs/research/031` sets, so the dump cannot supply this rung's
//!      registered 40-item test.
//!   2. The dump is **not** gold-teacher-forced. Its receipt's own `spans`
//!      field reads "sender-GENERATED tokens only": the receiver states were
//!      captured while teacher-forced over the **3B sender's greedy
//!      solution**, not over the GSM8K gold solution. The brief's and
//!      ADR-024's phrase "gold teacher-forced ... from `receiver_L19...`"
//!      describes an artifact that does not exist on disk.
//!
//! The deviation is therefore **toward** the registered design, not away from
//! it: this capture teacher-forces the receiver over the item's **actual GSM8K
//! gold solution**, which is what §2 candidate (c) literally specifies
//! ("captured while teacher-forced on the *gold* solution text for the same
//! item"), and it does so for **all 40 of S1a's items** so the registered
//! statistic is the one that runs.
//!
//! **The committed dump is still verified and still load-bearing**, in the
//! only way it can be: its sha256 is checked against
//! `run2-pertoken-dump-receipt.json`, and on the 13 overlapping items this
//! capture path is asserted to **reproduce the committed dump's block-19 rows
//! from the stored token stream**. That turns "new capture" into a *verified
//! extension of a receipted artifact* rather than an unaudited second path.
//!
//! Writes `run2-pc1-payloads.f32bin` (40 × [L19_last | L14_pooled | L14_last]
//! f32) into the run dir and `receipts/run2-pc1-capture-receipt.json`.
//!
//! Run: PATH=/usr/local/cuda-12.8/bin:$PATH \
//!      cargo run --release --features cuda --example run2_pc1_capture

#![recursion_limit = "512"]

#[path = "common/mod.rs"]
#[allow(dead_code)]
mod common;

use common::m3::{ITEM_SEED, RECEIVER, RECEIVER_BLOCK, SYSTEM};
use latentmesh_runtime::capture::forward_capture_multi_with_rows;
use latentmesh_runtime::{norms, QwenRuntime};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

const GSM8K_TRAIN_SHA256: &str = "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465";
const S1A_RECEIPT: &str = "receipts/s1a-receipt-slots8-block19-poolfull-rescaletrue-n40.json";
const DUMP_RECEIPT: &str = "receipts/run2-pertoken-dump-receipt.json";
const RUN_DIR: &str = "target/latentmesh-runs/run2";
const INDEX_JSON: &str = "run2-pertoken-index.json";
const STREAMS: &str = "../../harness/latentmesh-live/data/s2c-token-streams.jsonl";
/// The payload tap: S1a's own capture block, unchanged.
const PAYLOAD_BLOCK: usize = 19;
const D_RECEIVER: usize = 1536;
const N_ITEMS: usize = 40;
/// Vectors stored per item: `[L19_last | L14_pooled | L14_last]`.
const VECS_PER_ITEM: usize = 3;
pub const PAYLOAD_FILE: &str = "run2-pc1-payloads.f32bin";
/// Parity tolerance for the dump-reproduction gate, **fixed here before the
/// run**. The expectation is bit-identical — same code path, same weights,
/// same tokens, same device, deterministic shapes — and whether that held is
/// reported separately and unconditionally. The tolerance exists so the gate
/// is not brittle to a sub-BF16 float difference: these states have L2 norm
/// ~30 over 1536 dims, i.e. per-element magnitudes ~1, where BF16's own
/// representable resolution is ~1e-2. A 1e-3 bound is an order of magnitude
/// *below* the storage format's granularity and so cannot hide a real
/// capture-path divergence. Chosen now rather than after seeing the numbers.
const PARITY_TOL: f32 = 1e-3;

fn crate_path(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

fn sha256_file(path: &Path) -> anyhow::Result<String> {
    use sha2::{Digest, Sha256};
    let mut f = std::fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 22];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}

/// `[n_rows x dim]` rows for one dump row, read by byte offset.
fn read_dump_rows(
    dump: &Path,
    dim: usize,
    token_offset: usize,
    n_rows: usize,
) -> anyhow::Result<Vec<f32>> {
    let mut f = std::fs::File::open(dump)?;
    f.seek(SeekFrom::Start((token_offset * dim * 4) as u64))?;
    let mut bytes = vec![0u8; n_rows * dim * 4];
    f.read_exact(&mut bytes)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// **Pre-committed gold rendering rule.** GSM8K's `answer` field is the human
/// reference solution; it already ends with the `#### <n>` line the receiver's
/// own `ANSWER_FORMAT` instruction asks for. The only edit is removal of the
/// dataset's `<<a+b=c>>` calculator annotations, which are an artifact of the
/// GSM8K authoring tool and appear in no model's output distribution. Nothing
/// else is added — in particular no EOS token — so the captured span's LAST
/// row is the state at the final answer token.
fn render_gold(answer_text: &str) -> String {
    let mut out = String::with_capacity(answer_text.len());
    let mut rest = answer_text;
    while let Some(i) = rest.find("<<") {
        out.push_str(&rest[..i]);
        match rest[i..].find(">>") {
            Some(j) => rest = &rest[i + j + 2..],
            None => {
                rest = "";
                break;
            }
        }
    }
    out.push_str(rest);
    out
}

/// One saved S2c stream row (schema of `s2c-token-streams.jsonl`).
#[derive(serde::Deserialize)]
struct StreamRow {
    row: usize,
    item: usize,
    prompt_tokens: Vec<u32>,
    gen_tokens: Vec<u32>,
}

#[allow(clippy::too_many_lines)]
fn main() -> anyhow::Result<()> {
    let t0 = std::time::Instant::now();
    let e = anyhow::Error::msg;
    let nvcc = latentmesh_runtime::assert_cuda_build_env().map_err(e)?;
    println!("build-env guard: {nvcc}");

    // ---- Gate 1: the committed per-token dump, sha256-verified ------------
    let dump_receipt: serde_json::Value =
        serde_json::from_slice(&std::fs::read(crate_path(DUMP_RECEIPT))?)?;
    let l19 = &dump_receipt["files"]["receiver_L19.tok.f32bin"];
    let want_sha = l19["sha256"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("dump receipt does not pin receiver_L19 sha256"))?
        .to_string();
    let want_tokens = l19["tokens"].as_u64().unwrap_or(0);
    let dump_path = crate_path(RUN_DIR).join("receiver_L19.tok.f32bin");
    anyhow::ensure!(
        dump_path.exists(),
        "{} missing — PC1 verifies the committed dump even though (see the module docs) it \
         cannot supply this rung's payloads",
        dump_path.display()
    );
    println!(
        "hashing {} ({} tokens)...",
        dump_path.display(),
        want_tokens
    );
    let got_sha = sha256_file(&dump_path)?;
    anyhow::ensure!(
        got_sha == want_sha,
        "receiver_L19.tok.f32bin sha256 {got_sha} != receipt-pinned {want_sha}"
    );
    anyhow::ensure!(
        l19["dim"].as_u64() == Some(D_RECEIVER as u64),
        "dump receipt records a different receiver dim"
    );
    println!("dump sha256 VERIFIED: {got_sha}");

    let index_path = crate_path(RUN_DIR).join(INDEX_JSON);
    let index_sha = sha256_file(&index_path)?;
    anyhow::ensure!(
        Some(index_sha.as_str()) == dump_receipt["index"]["sha256"].as_str(),
        "run2-pertoken-index.json sha256 drifted from the dump receipt"
    );
    let index: serde_json::Value = serde_json::from_slice(&std::fs::read(&index_path)?)?;
    let as_usize = |v: &serde_json::Value| -> Vec<usize> {
        v.as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_u64())
                    .map(|x| x as usize)
                    .collect()
            })
            .unwrap_or_default()
    };
    let dump_items = as_usize(&index["item_indices"]);
    let dump_gen_len = as_usize(&index["gen_len"]);
    let dump_offsets = as_usize(&index["token_offsets"]);

    // ---- Gate 2: S1a's exact 40-item set ----------------------------------
    let dir = common::run_dir("run2-pc1");
    let data = dir.join("gsm8k-train.jsonl");
    let train_sha = common::fetch(common::GSM8K_TRAIN_URL, &data)?;
    anyhow::ensure!(train_sha == GSM8K_TRAIN_SHA256, "gsm8k-train sha drifted");
    let all_items = common::load_gsm8k(&data)?;
    let mut rng = ChaCha8Rng::seed_from_u64(ITEM_SEED);
    let mut indices = rand::seq::index::sample(&mut rng, all_items.len(), N_ITEMS).into_vec();
    indices.sort_unstable();
    let s1a: serde_json::Value = serde_json::from_slice(&std::fs::read(crate_path(S1A_RECEIPT))?)?;
    let s1a_idx = as_usize(&s1a["dataset"]["indices"]);
    anyhow::ensure!(
        indices == s1a_idx,
        "derived indices differ from the committed S1a receipt"
    );
    println!("item set: 40 indices identical to the committed S1a receipt");

    // Which of them the committed dump actually covers (the parity subset).
    let overlap: Vec<usize> = indices
        .iter()
        .copied()
        .filter(|i| dump_items.contains(i))
        .collect();
    println!(
        "dump coverage of the S1a item set: {}/{N_ITEMS} — the reason PC1 captures rather than \
         reads (module docs, deviation 1)",
        overlap.len()
    );

    // ---- Model ------------------------------------------------------------
    let device = latentmesh_runtime::device().map_err(e)?;
    println!("loading {RECEIVER} (BF16)... [receiver only — PC1 has no sender]");
    let mut rt = QwenRuntime::load(RECEIVER, &device, candle_core::DType::BF16).map_err(e)?;
    anyhow::ensure!(rt.config.hidden_size == D_RECEIVER);
    println!("loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // ---- Gate 3: capture-path parity against the committed dump -----------
    // For every S1a item the dump does cover, rebuild the exact stored token
    // stream and re-run THIS capture path; the block-19 rows must reproduce
    // the receipted file. This is what licenses using the same path to
    // capture the 27 items the dump does not cover.
    let streams_path = crate_path(STREAMS);
    let streams_bytes = std::fs::read(&streams_path)?;
    let streams_sha = common::sha256_hex(&streams_bytes);
    anyhow::ensure!(
        Some(streams_sha.as_str()) == dump_receipt["streams"]["sha256_pinned"].as_str(),
        "s2c-token-streams.jsonl sha256 drifted from the dump receipt"
    );
    let stream_rows: Vec<StreamRow> = String::from_utf8(streams_bytes)?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).map_err(anyhow::Error::from))
        .collect::<anyhow::Result<_>>()?;

    let mut parity_rows = Vec::new();
    let mut parity_max_abs = 0f32;
    let mut parity_bit_identical_items = 0usize;
    let mut parity_elems = 0usize;
    for &item_idx in &overlap {
        let pos = dump_items.iter().position(|&i| i == item_idx).unwrap();
        let sr = stream_rows
            .iter()
            .find(|r| r.item == item_idx)
            .ok_or_else(|| anyhow::anyhow!("item {item_idx} absent from the streams file"))?;
        anyhow::ensure!(sr.row == pos, "streams row order != index item order");
        let prompt = QwenRuntime::chat_prompt(
            SYSTEM,
            &format!(
                "{}\n\n{}",
                all_items[item_idx].question,
                common::ANSWER_FORMAT
            ),
        );
        let enc = rt.encode(&prompt).map_err(e)?;
        anyhow::ensure!(
            enc == sr.prompt_tokens,
            "item {item_idx}: prompt re-encoding differs from the stored stream"
        );
        let full: Vec<u32> = sr
            .prompt_tokens
            .iter()
            .chain(sr.gen_tokens.iter())
            .copied()
            .collect();
        let span = sr.prompt_tokens.len()..full.len();
        let n_rows = sr.gen_tokens.len();
        anyhow::ensure!(
            n_rows == dump_gen_len[pos],
            "gen_len mismatch for {item_idx}"
        );
        let (_, caps) = forward_capture_multi_with_rows(
            &mut rt.model,
            &full,
            &[RECEIVER_BLOCK, PAYLOAD_BLOCK],
            span,
            &device,
        )
        .map_err(e)?;
        let mine = &caps[1].rows;
        let theirs = read_dump_rows(&dump_path, D_RECEIVER, dump_offsets[pos], n_rows)?;
        anyhow::ensure!(mine.len() == theirs.len(), "parity: row-count mismatch");
        let mut max_abs = 0f32;
        let mut bit_identical = true;
        for (a, b) in mine.iter().zip(&theirs) {
            if a.to_bits() != b.to_bits() {
                bit_identical = false;
            }
            max_abs = max_abs.max((a - b).abs());
        }
        parity_elems += mine.len();
        parity_max_abs = parity_max_abs.max(max_abs);
        parity_bit_identical_items += usize::from(bit_identical);
        parity_rows.push(serde_json::json!({
            "item": item_idx, "dump_row": pos, "span_rows": n_rows,
            "bit_identical": bit_identical, "max_abs_diff": max_abs,
        }));
        println!(
            "  parity item {item_idx}: {n_rows} rows, bit_identical={bit_identical}, max|d|={max_abs:.3e}"
        );
    }
    let parity_all_bit_identical =
        !overlap.is_empty() && parity_bit_identical_items == overlap.len();
    let parity_pass = !overlap.is_empty() && parity_max_abs <= PARITY_TOL;
    println!(
        "capture-path parity vs the committed dump: {parity_bit_identical_items}/{} items \
         bit-identical (all={parity_all_bit_identical}) over {parity_elems} elements, \
         max|d| {parity_max_abs:.3e} <= {PARITY_TOL:.0e} => {parity_pass}",
        overlap.len()
    );

    // ---- Payload capture: gold-teacher-forced, per item -------------------
    let mut flat: Vec<f32> = Vec::with_capacity(N_ITEMS * VECS_PER_ITEM * D_RECEIVER);
    let mut rows = Vec::new();
    for (done, &idx) in indices.iter().enumerate() {
        let item = &all_items[idx];
        let gold_text = render_gold(&item.answer_text);
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
        let l19_pooled = l19.capture.pooled.clone();
        rows.push(serde_json::json!({
            "item": idx,
            "gold": item.gold,
            "prompt_tokens": ptoks.len(),
            "gold_continuation_tokens": n,
            "payload_l19_last_l2": norms::l2(&l19_last),
            "reference_l19_pooled_l2": norms::l2(&l19_pooled),
            "reference_l14_pooled_l2": norms::l2(&l14_pooled),
            "reference_l14_last_l2": norms::l2(&l14_last),
            "gold_continuation_tail": gold_text.chars().rev().take(48).collect::<String>()
                .chars().rev().collect::<String>(),
        }));
        flat.extend_from_slice(&l19_last);
        flat.extend_from_slice(&l14_pooled);
        flat.extend_from_slice(&l14_last);
        println!(
            "[{}/{N_ITEMS}] item {idx}: {n} gold tokens, |L19_last| {:.2}  {:.0}s",
            done + 1,
            norms::l2(&l19_last),
            t0.elapsed().as_secs_f32()
        );
    }
    anyhow::ensure!(flat.len() == N_ITEMS * VECS_PER_ITEM * D_RECEIVER);

    let payload_path = dir.join(PAYLOAD_FILE);
    let mut bytes = Vec::with_capacity(flat.len() * 4);
    for v in &flat {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(&payload_path, &bytes)?;
    let payload_sha = common::sha256_hex(&bytes);

    let receipt = serde_json::json!({
        "stage": "run2-PC1-capture",
        "design": "docs/adr/024-run2-trained-thought-adapter-ladder.md § 'PC1 PRE-REGISTRATION (2026-08-29) — the positive control this harness never had'; docs/research/045-positive-control-design.md §2 candidate (c) + §5",
        "env": common::env_info(&nvcc),
        "pre_committed": true,
        "firewall_verbatim_research_045_section_3": "This control used a same-model, same-item, identity-transform injection with gold-adjacent content. It tests whether this repository's injection mechanics (delivery operator, payload shape, injection site, norm-rescale) are capable of carrying signal at all. It does not test, and must not be cited as evidence for or against, whether a cross-model-derived, learned-alignment payload can transfer reasoning content — that is exactly the separate question M3 through M5X exist to answer.",
        "liveness_not_transfer": "A PASS on this rung proves LIVENESS ONLY, NEVER TRANSFER, and may never be cited as evidence for the transfer claim.",
        "declared_deviation_from_the_registered_capture_source": {
            "registered_text": "docs/research/045 §5 'Capture: none new — read per-token receiver states directly from receiver_L14.tok.f32bin and receiver_L19.tok.f32bin'; ADR-024's PC1 section repeats it as 'the receiver's own gold teacher-forced per-token block-19 state — already on disk in receiver_L19.tok.f32bin'.",
            "why_it_is_not_executable_as_written": [
                "COVERAGE: the dump covers the calibration-4000 split (2560 items, gsm8k-train indices 1..4797). Only 13 of S1a's 40 registered items appear in it; n=13 is far below docs/research/031's power floor, so the dump cannot supply the registered 40-item statistic.",
                "SEMANTICS: the dump is NOT gold-teacher-forced. run2-pertoken-dump-receipt.json's own `spans` field reads 'sender-GENERATED tokens only' — the receiver states were captured while teacher-forced over the 3B SENDER's greedy solution, not over the GSM8K gold solution. The phrase 'gold teacher-forced ... from receiver_L19.tok.f32bin' describes an artifact that does not exist on disk.",
            ],
            "what_was_done_instead": "The receiver is teacher-forced over the item's ACTUAL GSM8K gold solution and block 19 is tapped — which is what docs/research/045 §2 candidate (c) literally specifies ('captured while teacher-forced on the gold solution text for the same item') — for ALL 40 of S1a's items, so the registered item set AND the registered statistic both survive.",
            "direction_of_the_deviation": "TOWARD the registered design (literally gold, full registered item set), not away from it. The cost is 40 teacher-forced prefills, seconds of GPU time.",
            "how_the_committed_dump_is_still_load_bearing": "Its sha256 is verified against run2-pertoken-dump-receipt.json, and on the 13 overlapping items THIS capture path is asserted to reproduce the committed block-19 rows from the stored token stream. That makes the fresh capture a verified extension of a receipted artifact rather than an unaudited second path.",
        },
        "config": {
            "receiver": RECEIVER,
            "sender": "NONE — PC1 is a same-model self-pair; no sender is loaded",
            "transform": "IDENTITY — no adapter, no weights, no fitted alignment of any kind",
            "payload_tap_block": PAYLOAD_BLOCK,
            "reference_tap_block": RECEIVER_BLOCK,
            "payload_derivation": "LAST row of the block-19 span (de-pooled, per-token last-token — M4h Stage 1's derivation)",
            "teacher_forced_over": "the item's GSM8K gold solution, rendered by the pre-committed rule below",
            "gold_rendering_rule": "GSM8K `answer` verbatim with the dataset's <<a+b=c>> calculator annotations removed; nothing added, no EOS appended, so the span's LAST row is the state at the final answer token. The gold text already ends with the '#### <n>' line ANSWER_FORMAT asks for.",
            "capture_span": "the gold-continuation tokens only (prompt excluded) — the same convention the committed dump uses for its generated span",
            "item_seed_chacha8": ITEM_SEED,
        },
        "dataset": {"source": common::GSM8K_TRAIN_URL, "sha256": train_sha,
                     "indices": indices, "s1a_indices_match": true},
        "payload_file": {
            "path": payload_path.display().to_string(),
            "sha256": payload_sha,
            "layout": "40 items in ascending index order; per item three contiguous f32 vectors of 1536, row-major little-endian: [L19_last (THE PAYLOAD) | L14_pooled (on-manifold reference) | L14_last (un-pooled on-manifold reference)]",
            "n_items": N_ITEMS, "vecs_per_item": VECS_PER_ITEM, "dim": D_RECEIVER,
        },
        "items": rows,
        "gates": {
            "receiver_L19_dump_sha256_verified": {
                "pass": true, "measured": got_sha, "pinned_by": DUMP_RECEIPT,
                "tokens": want_tokens,
            },
            "pertoken_index_sha256_verified": {"pass": true, "measured": index_sha},
            "streams_sha256_verified": {"pass": true, "measured": streams_sha},
            "s1a_item_set_reproduced": {"pass": true, "n": N_ITEMS},
            "dump_coverage_of_the_s1a_item_set": {
                "covered": overlap.len(), "of": N_ITEMS, "items": overlap,
                "gating": "NONE — disclosed. This is the measurement that forced the declared deviation above.",
            },
            "capture_path_reproduces_the_committed_dump": {
                "pass": parity_pass,
                "items_checked": overlap.len(),
                "items_bit_identical": parity_bit_identical_items,
                "all_items_bit_identical": parity_all_bit_identical,
                "elements_compared": parity_elems,
                "max_abs_diff": parity_max_abs,
                "tolerance": PARITY_TOL,
                "tolerance_rationale": "Fixed before the run. Bit-identity is the expectation and is reported unconditionally above; the numeric bound keeps the gate from being brittle to a sub-BF16 float difference. These states have L2 ~30 over 1536 dims (per-element magnitudes ~1) where BF16's own resolution is ~1e-2, so a 1e-3 bound sits an order of magnitude BELOW the storage granularity and cannot hide a real divergence.",
                "per_item": parity_rows,
                "why_it_matters": "It is what licenses using this capture path for the 27 registered items the committed dump does not cover.",
            },
            "identity_transform_no_adapter_weights_loaded": {
                "pass": true,
                "note": "No MlpTransform / FastGrnnTransform / AffineTransform is constructed anywhere in this binary; the payload IS a captured receiver state, byte-for-byte.",
            },
            "all_captured_states_finite": {"pass": true},
        },
        "gate_pass": parity_pass,
        "wall_clock_s": t0.elapsed().as_secs_f32(),
    });
    common::write_receipt(
        &crate_path("receipts"),
        "run2-pc1-capture-receipt.json",
        &receipt,
    )?;
    println!(
        "PC1 capture complete: {N_ITEMS} gold-teacher-forced payloads, sha {payload_sha}, {:.0}s",
        t0.elapsed().as_secs_f32()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::render_gold;

    #[test]
    fn gold_rendering_strips_only_calculator_annotations() {
        let a = "Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.\nSo she earned 0.2 x 50 = \
                 $<<0.2*50=10>>10.\n#### 10";
        let r = render_gold(a);
        assert!(!r.contains("<<") && !r.contains(">>"));
        assert!(r.ends_with("#### 10"));
        assert!(r.starts_with("Weng earns 12/60 = $0.2 per minute."));
        // No annotations => byte-identical passthrough.
        assert_eq!(render_gold("plain\n#### 4"), "plain\n#### 4");
        // Unterminated annotation truncates rather than panicking.
        assert_eq!(render_gold("a<<b"), "a");
    }
}
