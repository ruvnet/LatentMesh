//! Run-2 **PC2 capture** — PC1b's payload derivation with **exactly one
//! value changed**: the teacher-forced solution's final numeric answer is
//! replaced by a pre-committed decoy `d`.
//!
//! Registered by [`docs/adr/039-pc2-steering-control-pre-registration.md`]:
//!
//! > *"For each item, teacher-force the receiver over its own question and a
//! > solution **whose final numeric answer has been replaced by a decoy `d`**,
//! > then tap block 19 at the last token — identical derivation to PC1b in
//! > every respect except the answer value."*
//!
//! # What "identical derivation except the answer value" means here, checked
//!
//! This binary derives **two** payloads per item and keeps both:
//!
//! 1. **The PC2 payload** — block-19, last token, teacher-forced over
//!    `render_decoy(item)`: PC1b's rendered gold solution with the trailing
//!    `#### g` replaced by `#### d`, and **nothing else touched**.
//! 2. **A gold payload, re-derived** — block-19, last token, teacher-forced
//!    over `common::render_gold(item)`, i.e. byte-for-byte PC1b's own input.
//!
//! (2) exists solely to be compared against **PC1b's committed payload file**
//! for the same item, and the binary **aborts** unless every one of the 300 is
//! bit-identical. That turns "PC2 changes exactly one thing" from a claim
//! about intent into a checked property of this run: the same code, on the
//! same stream, on the same GPU, reproduces PC1b's payloads exactly, so the
//! only thing separating PC2's payload from PC1b's is the answer token.
//!
//! # Decoy construction — pre-committed BEFORE any capture forward pass
//!
//! ADR-039 § "Decoy construction": `d` is derived deterministically from the
//! gold answer `g` — never sampled, never equal to `g` — by a fixed ChaCha8
//! stream (seed `0x5732`) selecting a perturbation from `{g+1, g-1, g+10,
//! 2g}`, re-drawing on collision with `g` or on a non-positive result.
//!
//! The full `(item, gold, decoy)` table is computed and written to its **own**
//! receipt, `run2-pc2-decoy-commitment-receipt.json`, before a single payload
//! forward pass runs. Nothing about the decoys can be a function of anything
//! this rung later measures.
//!
//! # Item supply
//!
//! ADR-036 Decision 2, unchanged from PC1b: `adaptation-512` in the file's own
//! fixed ascending index order, ADR-024's 13-item leakage exclusion,
//! `N_max = 300`, question-tail site pre-flight.
//!
//! **The M4i stream gate reads `m4i["items"][*]["item"]`.** PC1b's own probe
//! read `m4i["dataset"]["indices"]`, a key that **does not exist** in M4i's
//! committed receipt, and so recorded `item_stream_identical_to_m4i: false`
//! for a stream that is in fact identical — a self-inflicted false negative.
//! This binary reads the key that exists and **hard-gates** on it.
//!
//! # FIREWALL (ADR-039 § Firewall, inherited from `docs/research/045` §3)
//!
//! PC2 is same-model, same-item, identity-transform, with gold-adjacent
//! content. It tests **the apparatus, never transfer**. A PASS proves the
//! pathway can steer a decision and says **nothing** about cross-model
//! transfer; **a FAIL may not be cited as transfer evidence either.**
//!
//! Run: PATH=/usr/local/cuda-12.8/bin:$PATH \
//!      cargo run --release --features cuda --example run2_pc2_capture

#![recursion_limit = "512"]

#[path = "common/mod.rs"]
#[allow(dead_code)]
mod common;

use candle_core::Device;
use common::m3::{build_site_prompt, Site, N_SLOTS, RECEIVER, RECEIVER_BLOCK, SYSTEM};
use latentmesh_runtime::capture::forward_capture_multi_with_rows;
use latentmesh_runtime::{norms, QwenRuntime};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::path::{Path, PathBuf};

const GSM8K_TRAIN_SHA256: &str = "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465";
const ADAPTATION_512: &str = "../../harness/latentmesh-live/data/adaptation-512.json";
/// PC1b's capture receipt — the reference this rung must reproduce exactly on
/// the gold arm, and the source of the `restore` payload the probe replays.
const PC1B_CAPTURE_RECEIPT: &str = "receipts/run2-pc1b-capture-receipt.json";
const M4I_RECEIPT: &str =
    "receipts/run2-m4i-receipt-cellL18toL14-mlp-pertokenlast-fuse-questiontail-slots8-eprocess.json";

/// The payload tap: S1a's capture block, unchanged through PC1, PC1b, PC2.
const PAYLOAD_BLOCK: usize = 19;
const D_RECEIVER: usize = 1536;
/// `[L19_last (THE PAYLOAD) | L14_pooled | L14_last]` — PC1b's layout.
const VECS_PER_ITEM: usize = 3;

/// PC2's site — PC1b's, unchanged. The site is NOT a changed factor here.
const SITE: Site = Site::QuestionTail;

/// ADR-024's 13 probe-overlap items, verbatim from PC1b's copy.
const LEAKAGE_EXCLUSIONS: [usize; 13] = [
    150, 1309, 1573, 2365, 2418, 2540, 2958, 2973, 3084, 3825, 3844, 3877, 4746,
];
const N_MAX: usize = 300;

/// ADR-039 § "Decoy construction": the pre-committed decoy stream seed.
const DECOY_SEED: u64 = 0x5732;
/// The pre-committed perturbation menu, in the ADR's own order.
const PERTURBATIONS: [&str; 4] = ["g+1", "g-1", "g+10", "2g"];

fn crate_path(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

/// Apply perturbation `k` of the pre-committed menu to gold `g`.
fn perturb(g: i64, k: usize) -> i64 {
    match k {
        0 => g + 1,
        1 => g - 1,
        2 => g + 10,
        _ => g * 2,
    }
}

/// One teacher-forced tap: PC1b's capture statements, factored so the decoy
/// arm and the gold arm provably run the SAME code.
struct Tap {
    l19_last: Vec<f32>,
    l14_pooled: Vec<f32>,
    l14_last: Vec<f32>,
    prompt_tokens: usize,
    continuation_tokens: usize,
    l19_pooled_l2: f32,
}

/// Teacher-force `continuation` after the item's own question and tap block
/// [`PAYLOAD_BLOCK`] at the LAST row of the continuation span — PC1b's
/// derivation verbatim, with the continuation text as the only argument that
/// differs between PC2's two arms.
fn tap(
    rt: &mut QwenRuntime,
    device: &Device,
    question: &str,
    continuation: &str,
) -> anyhow::Result<Tap> {
    let e = anyhow::Error::msg;
    let prompt =
        QwenRuntime::chat_prompt(SYSTEM, &format!("{question}\n\n{}", common::ANSWER_FORMAT));
    let ptoks = rt.encode(&prompt).map_err(e)?;
    let ctoks = rt.encode(continuation).map_err(e)?;
    anyhow::ensure!(!ctoks.is_empty(), "empty continuation");
    let full: Vec<u32> = ptoks.iter().chain(ctoks.iter()).copied().collect();
    let span = ptoks.len()..full.len();
    let (_, caps) = forward_capture_multi_with_rows(
        &mut rt.model,
        &full,
        &[RECEIVER_BLOCK, PAYLOAD_BLOCK],
        span,
        device,
    )
    .map_err(e)?;
    let n = ctoks.len();
    let l14 = &caps[0];
    let l19 = &caps[1];
    anyhow::ensure!(l19.rows.len() == n * D_RECEIVER, "L19 row shape");
    let l19_last = l19.rows[(n - 1) * D_RECEIVER..].to_vec();
    let l14_pooled = l14.capture.pooled.clone();
    let l14_last = l14.rows[(n - 1) * D_RECEIVER..].to_vec();
    anyhow::ensure!(
        l19_last.iter().all(|v| v.is_finite())
            && l14_pooled.iter().all(|v| v.is_finite())
            && l14_last.iter().all(|v| v.is_finite()),
        "non-finite captured state"
    );
    let l19_pooled_l2 = norms::l2(&l19.capture.pooled);
    Ok(Tap {
        l19_last,
        l14_pooled,
        l14_last,
        prompt_tokens: ptoks.len(),
        continuation_tokens: n,
        l19_pooled_l2,
    })
}

/// One committed decoy draw.
struct Decoy {
    item: usize,
    gold: i64,
    decoy: i64,
    /// Index into [`PERTURBATIONS`] that was finally accepted.
    perturbation: usize,
    /// How many draws were rejected (collision with `g`, or non-positive)
    /// before the accepted one. Recorded so the stream is fully replayable.
    redraws: usize,
}

/// PC1b's gold rendering with **only** the trailing `#### g` replaced by
/// `#### d`.
///
/// Everything else — the whole worked solution, its whitespace, the removal of
/// GSM8K's `<<a+b=c>>` calculator annotations by `common::render_gold` — is
/// byte-identical to what PC1b teacher-forced. The returned tuple carries the
/// gold rendering too, so the caller can assert the two differ only where they
/// are supposed to.
fn render_decoy(answer_text: &str, gold: &str, decoy: &str) -> anyhow::Result<(String, String)> {
    let gold_text = common::render_gold(answer_text);
    let i = gold_text
        .rfind("####")
        .ok_or_else(|| anyhow::anyhow!("rendered gold solution has no '####' final-answer line"))?;
    let head = &gold_text[..i];
    let tail = &gold_text[i + 4..];
    let ws_len = tail.len() - tail.trim_start().len();
    let (ws, after) = tail.split_at(ws_len);
    let tok_len = after.find(char::is_whitespace).unwrap_or(after.len());
    let (tok, rest) = after.split_at(tok_len);
    anyhow::ensure!(
        common::extract_final_answer(&format!("#### {tok}")).as_deref() == Some(gold),
        "the token after '####' ({tok:?}) does not normalise to the item's gold answer ({gold})"
    );
    let decoy_text = format!("{head}####{ws}{decoy}{rest}");
    anyhow::ensure!(
        common::extract_final_answer(&decoy_text).as_deref() == Some(decoy),
        "the rebuilt decoy solution does not read back as the decoy"
    );
    anyhow::ensure!(
        common::extract_final_answer(&gold_text).as_deref() == Some(gold),
        "the rendered gold solution does not read back as the gold answer"
    );
    anyhow::ensure!(
        decoy_text != gold_text,
        "the decoy rendering is identical to the gold rendering"
    );
    // The edit is confined to the final-answer token: the prefix up to and
    // including "####" plus its whitespace, and the suffix after the number,
    // are byte-identical between the two renderings.
    let prefix_len = i + 4 + ws_len;
    anyhow::ensure!(
        gold_text.as_bytes()[..prefix_len] == decoy_text.as_bytes()[..prefix_len]
            && gold_text.ends_with(rest)
            && decoy_text.ends_with(rest),
        "the decoy edit is not confined to the final-answer token"
    );
    Ok((decoy_text, gold_text))
}

#[allow(clippy::too_many_lines)]
fn main() -> anyhow::Result<()> {
    let t0 = std::time::Instant::now();
    let e = anyhow::Error::msg;
    let nvcc = latentmesh_runtime::assert_cuda_build_env().map_err(e)?;
    println!("build-env guard: {nvcc}");

    // ---- Gate 1: PC1b's reference payload, verified intact ----------------
    let pc1b: serde_json::Value =
        serde_json::from_slice(&std::fs::read(crate_path(PC1B_CAPTURE_RECEIPT))?)?;
    let pc1b_path =
        PathBuf::from(pc1b["payload_file"]["path"].as_str().ok_or_else(|| {
            anyhow::anyhow!("PC1b capture receipt does not name its payload file")
        })?);
    let pc1b_want_sha = pc1b["payload_file"]["sha256"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("PC1b capture receipt does not pin a payload sha256"))?
        .to_string();
    let pc1b_bytes = std::fs::read(&pc1b_path).map_err(|err| {
        anyhow::anyhow!(
            "PC1b's reference payload {} is unreadable ({err}) — PC2's one-changed-factor gate \
             cannot run without it",
            pc1b_path.display()
        )
    })?;
    let pc1b_got_sha = common::sha256_hex(&pc1b_bytes);
    anyhow::ensure!(
        pc1b_got_sha == pc1b_want_sha,
        "PC1b's reference payload sha256 {pc1b_got_sha} != its own capture-receipt-pinned \
         {pc1b_want_sha} — the reference artifact has changed and no comparison against PC1b is \
         valid"
    );
    anyhow::ensure!(
        pc1b_bytes.len() == N_MAX * VECS_PER_ITEM * D_RECEIVER * 4,
        "PC1b reference payload size mismatch"
    );
    let pc1b_flat: Vec<f32> = pc1b_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let pc1b_stream: Vec<usize> = pc1b["dataset"]["indices"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("PC1b capture receipt has no dataset.indices"))?
        .iter()
        .filter_map(|v| v.as_u64())
        .map(|v| v as usize)
        .collect();
    anyhow::ensure!(pc1b_stream.len() == N_MAX, "PC1b stream length mismatch");
    println!("PC1b reference payload VERIFIED intact: sha {pc1b_got_sha} ({N_MAX} items)");

    // ---- Item supply: ADR-036 Decision 2, resolved exactly as PC1b does ---
    let dir = common::run_dir("run2-pc2");
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

    // ---- Model: receiver ONLY (PC2 is a same-model self-pair) -------------
    let device = latentmesh_runtime::device().map_err(e)?;
    println!("loading {RECEIVER} (BF16)... [no sender, no adapter]");
    let mut rt = QwenRuntime::load(RECEIVER, &device, candle_core::DType::BF16).map_err(e)?;
    let pad_id = rt
        .tokenizer
        .token_to_id("<|fim_pad|>")
        .ok_or_else(|| anyhow::anyhow!("<|fim_pad|> not in tokenizer"))?;
    println!("model loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // ---- Site pre-flight — PC1b's, unchanged ------------------------------
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

    // ---- HARD GATE: the stream is M4i's committed stream -------------------
    // Read from `items[*].item`, the key M4i's receipt actually carries.
    // PC1b read `dataset.indices` — absent from that receipt — and recorded a
    // false negative. Verified here rather than reported.
    let m4i: serde_json::Value = serde_json::from_slice(&std::fs::read(crate_path(M4I_RECEIPT))?)?;
    let m4i_items = m4i["items"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("M4i receipt has no items array"))?;
    let m4i_stream: Vec<usize> = m4i_items
        .iter()
        .filter_map(|r| r["item"].as_u64())
        .map(|v| v as usize)
        .collect();
    anyhow::ensure!(
        m4i_stream.len() == N_MAX,
        "M4i receipt items[*].item yielded {} entries, expected {N_MAX} — the gate cannot be \
         evaluated and the rung must not proceed",
        m4i_stream.len()
    );
    anyhow::ensure!(
        m4i_stream == stream,
        "the resolved stream differs from M4i's committed 300-item stream — the site would not be \
         the same site"
    );
    // ... and it is also PC1b's stream, so payload row i corresponds item-for-item.
    anyhow::ensure!(
        pc1b_stream == stream,
        "the resolved stream differs from PC1b's committed stream — the restore payload at row i \
         would not belong to item i"
    );
    println!(
        "site pre-flight: {N_MAX} items at {N_SLOTS} question-tail positions, {} excluded on \
         tokenisation grounds; stream is IDENTICAL to M4i's AND to PC1b's committed streams \
         (m4i read from items[*].item)",
        tokenization_excluded.len()
    );

    // ---- DECOY COMMITMENT — computed and written BEFORE any capture pass ---
    // ADR-039: "Decoys are committed to the capture receipt BEFORE the draw."
    // Nothing below this block can influence the table above it.
    let mut rng = ChaCha8Rng::seed_from_u64(DECOY_SEED);
    let mut decoys: Vec<Decoy> = Vec::with_capacity(N_MAX);
    for &idx in &stream {
        let gold: i64 = all_items[idx].gold.parse().map_err(|_| {
            anyhow::anyhow!(
                "item {idx}: gold answer {:?} is not an integer; the pre-committed perturbation \
                 menu {{g+1, g-1, g+10, 2g}} is defined on integers",
                all_items[idx].gold
            )
        })?;
        let mut redraws = 0usize;
        let (perturbation, decoy) = loop {
            let k = rng.gen_range(0..PERTURBATIONS.len());
            let d = perturb(gold, k);
            if d != gold && d > 0 {
                break (k, d);
            }
            redraws += 1;
            anyhow::ensure!(
                redraws < 1000,
                "item {idx}: the decoy draw did not terminate for gold {gold}"
            );
        };
        decoys.push(Decoy {
            item: idx,
            gold,
            decoy,
            perturbation,
            redraws,
        });
    }
    anyhow::ensure!(
        decoys.iter().all(|d| d.decoy != d.gold && d.decoy > 0),
        "a committed decoy collides with its gold or is non-positive"
    );
    let commitment_rows: Vec<serde_json::Value> = decoys
        .iter()
        .enumerate()
        .map(|(order, d)| {
            serde_json::json!({
                "stream_order": order + 1,
                "item": d.item,
                "gold": d.gold,
                "decoy": d.decoy,
                "perturbation": PERTURBATIONS[d.perturbation],
                "redraws_before_acceptance": d.redraws,
            })
        })
        .collect();
    let mut menu_histogram = [0usize; PERTURBATIONS.len()];
    for d in &decoys {
        menu_histogram[d.perturbation] += 1;
    }
    let commitment = serde_json::json!({
        "stage": "run2-PC2-decoy-commitment",
        "design": "docs/adr/039-pc2-steering-control-pre-registration.md § 'Decoy construction (pre-committed, to foreclose gaming)'.",
        "env": common::env_info(&nvcc),
        "pre_committed": true,
        "written_before": "any payload forward pass. The full (item, gold, decoy) table below is a pure function of the committed item stream and the registered seed; no measurement of any kind precedes it.",
        "rule_verbatim_from_adr_039": "d is derived deterministically from the gold answer g — not sampled, and never equal to g: a fixed per-item ChaCha8 stream (seed 0x5732) selects a perturbation from {g+1, g-1, g+10, 2g}, re-drawing on collision with g or on a non-positive result.",
        "implementation": {
            "seed": DECOY_SEED,
            "seed_hex": format!("{DECOY_SEED:#x}"),
            "stream": "ONE ChaCha8 stream seeded at 0x5732, consumed in the committed item-stream order — the ADR names 0x5732 as THE seed, so the seed is not re-derived per item. The consumption order is itself pre-committed (adaptation-512 fixed index order, gated identical to M4i's stream), so every draw is reproducible; `stream_order` and `redraws_before_acceptance` are recorded per item so the whole stream can be replayed by hand.",
            "menu": PERTURBATIONS,
            "draw": "k = rng.gen_range(0..4); d = menu[k](g); accept iff d != g and d > 0, else redraw",
            "menu_histogram": {"g+1": menu_histogram[0], "g-1": menu_histogram[1],
                                "g+10": menu_histogram[2], "2g": menu_histogram[3]},
            "total_redraws": decoys.iter().map(|d| d.redraws).sum::<usize>(),
        },
        "item_stream_source": "adaptation-512, fixed ascending index order, ADR-024's 13-item leakage exclusion, N_max = 300; gated equal to M4i's committed stream (read from items[*].item) and to PC1b's",
        "commitments": commitment_rows,
        "n": decoys.len(),
    });
    common::write_receipt(
        &crate_path("receipts"),
        "run2-pc2-decoy-commitment-receipt.json",
        &commitment,
    )?;
    println!(
        "decoys COMMITTED before any forward pass: {} items, menu histogram g+1={} g-1={} g+10={} \
         2g={}, {} redraws total",
        decoys.len(),
        menu_histogram[0],
        menu_histogram[1],
        menu_histogram[2],
        menu_histogram[3],
        decoys.iter().map(|d| d.redraws).sum::<usize>()
    );

    // ---- Payload capture --------------------------------------------------
    // Two teacher-forced prefills per item: the DECOY solution (PC2's payload)
    // and the GOLD solution (PC1b's input, re-derived only to be gated against
    // PC1b's committed vector).
    let mut flat: Vec<f32> = Vec::with_capacity(N_MAX * VECS_PER_ITEM * D_RECEIVER);
    let mut rows = Vec::new();
    let mut gold_identity_failures: Vec<serde_json::Value> = Vec::new();
    let mut max_gold_delta = 0f32;
    let mut steer_vs_restore_cos: Vec<f64> = Vec::new();

    for (done, d) in decoys.iter().enumerate() {
        let item = &all_items[d.item];
        let (decoy_text, gold_text) =
            render_decoy(&item.answer_text, &item.gold, &d.decoy.to_string())?;

        // (1) THE PC2 PAYLOAD — teacher-forced over the decoy solution.
        let t_steer = tap(&mut rt, &device, &item.question, &decoy_text)?;
        let steer = t_steer.l19_last;
        let l14_pooled = t_steer.l14_pooled;
        let l14_last = t_steer.l14_last;
        let ptoks_n = t_steer.prompt_tokens;
        let decoy_toks = t_steer.continuation_tokens;
        let l19_pooled_l2 = t_steer.l19_pooled_l2;

        // (2) The gold arm, re-derived ONLY to be gated against PC1b.
        let t_gold = tap(&mut rt, &device, &item.question, &gold_text)?;
        let gold_vec = t_gold.l19_last;
        let gold_toks = t_gold.continuation_tokens;
        let base = done * VECS_PER_ITEM * D_RECEIVER;
        let pc1b_l19 = &pc1b_flat[base..base + D_RECEIVER];
        let identical = pc1b_l19 == gold_vec.as_slice();
        let delta = pc1b_l19
            .iter()
            .zip(gold_vec.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        max_gold_delta = max_gold_delta.max(delta);
        if !identical {
            gold_identity_failures.push(serde_json::json!({
                "item": d.item, "stream_order": done + 1, "max_abs_delta": delta,
            }));
        }

        // How far the one changed token actually moves the payload. Recorded,
        // never gating: if this were ~0 the rung would be measuring nothing,
        // and a reader must be able to see it from the receipt.
        let dot: f64 = steer
            .iter()
            .zip(gold_vec.iter())
            .map(|(a, b)| f64::from(*a) * f64::from(*b))
            .sum();
        let cos = dot / (f64::from(norms::l2(&steer)) * f64::from(norms::l2(&gold_vec))).max(1e-12);
        steer_vs_restore_cos.push(cos);

        rows.push(serde_json::json!({
            "item": d.item,
            "gold": d.gold,
            "decoy": d.decoy,
            "perturbation": PERTURBATIONS[d.perturbation],
            "prompt_tokens": ptoks_n,
            "decoy_continuation_tokens": decoy_toks,
            "gold_continuation_tokens": gold_toks,
            "payload_l19_last_l2": norms::l2(&steer),
            "regold_l19_last_l2": norms::l2(&gold_vec),
            "reference_l19_pooled_l2": l19_pooled_l2,
            "reference_l14_pooled_l2": norms::l2(&l14_pooled),
            "reference_l14_last_l2": norms::l2(&l14_last),
            "gold_arm_bit_identical_to_pc1b": identical,
            "gold_arm_max_abs_delta_vs_pc1b": delta,
            "cosine_steer_vs_restore": cos,
        }));
        flat.extend_from_slice(&steer);
        flat.extend_from_slice(&l14_pooled);
        flat.extend_from_slice(&l14_last);
        if (done + 1) % 25 == 0 || done + 1 == decoys.len() {
            println!(
                "[{}/{N_MAX}] item {} gold {} -> decoy {} ({}): {decoy_toks} decoy tokens, \
                 |L19_last| {:.2}, cos(steer,restore) {cos:.4}, gold-arm identical={identical}  \
                 {:.0}s",
                done + 1,
                d.item,
                d.gold,
                d.decoy,
                PERTURBATIONS[d.perturbation],
                norms::l2(&steer),
                t0.elapsed().as_secs_f32()
            );
        }
    }
    anyhow::ensure!(
        gold_identity_failures.is_empty(),
        "the re-derived GOLD payload is not bit-identical to PC1b's committed vector for {} of \
         {N_MAX} items (max|d| {max_gold_delta:.3e}) — PC2's derivation has drifted from PC1b's \
         and 'exactly one changed factor' would be false; the rung must not proceed. First \
         failures: {}",
        gold_identity_failures.len(),
        serde_json::to_string(&gold_identity_failures[..gold_identity_failures.len().min(5)])?
    );
    let mean_cos = steer_vs_restore_cos.iter().sum::<f64>() / steer_vs_restore_cos.len() as f64;
    let min_cos = steer_vs_restore_cos
        .iter()
        .copied()
        .fold(f64::MAX, f64::min);

    // ---- Write the payload artifact ---------------------------------------
    let out = dir.join("run2-pc2-payloads.f32bin");
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
        "stage": "run2-PC2-capture",
        "design": "docs/adr/039-pc2-steering-control-pre-registration.md § Decision — PC2, a steering control: 'For each item, teacher-force the receiver over its own question and a solution whose final numeric answer has been replaced by a decoy d, then tap block 19 at the last token — identical derivation to PC1b in every respect except the answer value.'",
        "env": common::env_info(&nvcc),
        "pre_committed": true,

        "FIREWALL_apparatus_not_transfer": {
            "verbatim_adr_039_firewall": "PC2 is same-model, same-item, identity-transform with gold-adjacent content. It tests the apparatus, never transfer. A PASS proves the pathway can steer a decision; it says nothing about whether a cross-model, learned-alignment payload can carry reasoning. A FAIL may not be cited as transfer evidence either — the symmetric rule added after PC1b.",
            "rule": "Neither branch of this rung is evidence about transfer.",
        },

        "the_one_changed_factor_vs_pc1b": {
            "changed": "the final numeric answer of the teacher-forced solution: '#### g' becomes '#### d'. Nothing else in the rendered solution is touched.",
            "held_identical": [
                "the item stream (adaptation-512 fixed order, 13-item exclusion, N_max 300 — gated equal to BOTH M4i's and PC1b's committed streams)",
                "the gold rendering rule (common::render_gold), applied first and then edited at exactly one token",
                "the prompt (SYSTEM + question + ANSWER_FORMAT under the same chat template)",
                "the tap (forward_capture_multi_with_rows at blocks [14, 19], LAST row of the continuation span)",
                "the payload file layout ([L19_last | L14_pooled | L14_last], 300 x 3 x 1536 f32 LE)",
                "the absence of any adapter — no MlpTransform / FastGrnnTransform / AffineTransform is named, constructed or applied anywhere in this binary",
            ],
            "how_it_is_PROVED_rather_than_asserted": "this binary re-derives the GOLD payload for every one of the 300 items through the same code path and gates on it being BIT-IDENTICAL to PC1b's committed vector. All 300 matched. The only thing that can therefore separate PC2's payload from PC1b's is the answer token.",
            "gold_arm_bit_identity": {
                "items_checked": N_MAX,
                "items_bit_identical": N_MAX,
                "failures": gold_identity_failures,
                "max_abs_elementwise_delta_over_all_items": max_gold_delta,
            },
            "how_far_the_one_token_moves_the_payload": {
                "metric": "cosine(steer_payload, re-derived gold payload) per item",
                "mean": mean_cos,
                "min": min_cos,
                "gating": "NONE — diagnostic. A cosine of ~1.0 everywhere would mean the decoy edit barely perturbs the tapped state, which a reader must be able to see when interpreting the outcome.",
            },
        },

        "decoy_construction": {
            "commitment_receipt": "receipts/run2-pc2-decoy-commitment-receipt.json",
            "written_before_any_forward_pass": true,
            "rule_verbatim_from_adr_039": "d is derived deterministically from the gold answer g — not sampled, and never equal to g: a fixed per-item ChaCha8 stream (seed 0x5732) selects a perturbation from {g+1, g-1, g+10, 2g}, re-drawing on collision with g or on a non-positive result.",
            "seed": DECOY_SEED,
            "menu": PERTURBATIONS,
            "menu_histogram": {"g+1": menu_histogram[0], "g-1": menu_histogram[1],
                                "g+10": menu_histogram[2], "2g": menu_histogram[3]},
            "total_redraws": decoys.iter().map(|d| d.redraws).sum::<usize>(),
            "edit_scope": "render_decoy() replaces ONLY the whitespace-delimited token following the LAST '####' in common::render_gold's output, and asserts (a) that token normalises to the item's gold, (b) the rebuilt text reads back as the decoy, (c) the byte prefix through '####'+whitespace and the byte suffix after the number are identical between the two renderings.",
            "leakage_check_is_the_PROBE's_job": "ADR-039 requires the rate at which BASELINE already emits d to be recorded, with the rung declared VOID above 2%. That is measured in run2_pc2_probe, not here.",
        },

        "config": {
            "receiver": RECEIVER,
            "sender": "NONE — same-model self-pair",
            "transform": "IDENTITY — the payload IS a captured receiver state",
            "payload_block": PAYLOAD_BLOCK,
            "reference_block": RECEIVER_BLOCK,
            "payload": "the receiver's OWN block-19 residual state, teacher-forced over the item's GSM8K solution WITH THE FINAL ANSWER REPLACED BY THE DECOY, LAST token of the span (de-pooled)",
            "site_this_payload_is_for": {"tag": SITE.tag(), "description": SITE.description()},
            "slots": N_SLOTS,
        },

        "pc1b_reference": {
            "capture_receipt": PC1B_CAPTURE_RECEIPT,
            "payload_file": pc1b_path.display().to_string(),
            "sha256_pinned": pc1b_want_sha,
            "sha256_measured_now": pc1b_got_sha,
            "role": "(a) the reference the gold arm is gated bit-identical against, and (b) the payload the probe replays as the `restore` condition — byte-for-byte PC1b's own vectors, not a re-derivation",
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
            "m4i_receipt_compared_against": M4I_RECEIPT,
            "m4i_stream_key_read": "items[*].item",
            "m4i_stream_key_note": "PC1b's probe read m4i['dataset']['indices'], which does NOT exist in M4i's receipt, and therefore recorded item_stream_identical_to_m4i: false for a stream that is in fact identical. This binary reads the key that exists and HARD-GATES on equality.",
            "stream_identical_to_m4i": true,
            "stream_identical_to_pc1b": true,
        },

        "payload_file": {
            "path": out.display().to_string(),
            "sha256": payload_sha,
            "n_items": N_MAX,
            "dim": D_RECEIVER,
            "vecs_per_item": VECS_PER_ITEM,
            "layout": "300 items in the e-process stream order; per item three contiguous f32 vectors of 1536, row-major little-endian: [L19_last over the DECOY solution (THE PC2 PAYLOAD) | L14_pooled | L14_last] — PC1b's layout, unchanged",
        },

        "dataset": {"source": common::GSM8K_TRAIN_URL, "sha256": train_sha, "indices": stream},
        "items": rows,

        "gates": {
            "pc1b_reference_payload_sha256_intact": {"pass": true,
                "pinned": pc1b_want_sha, "measured": pc1b_got_sha},
            "stream_identical_to_m4i": {"pass": true, "n": N_MAX,
                "read_from": "items[*].item",
                "note": "hard gate — the binary aborts on mismatch"},
            "stream_identical_to_pc1b": {"pass": true, "n": N_MAX},
            "decoys_committed_before_any_forward_pass": {"pass": true,
                "receipt": "receipts/run2-pc2-decoy-commitment-receipt.json"},
            "no_decoy_equals_its_gold": {"pass": true},
            "all_decoys_positive": {"pass": true},
            "decoy_edit_confined_to_the_final_answer_token": {"pass": true,
                "note": "asserted per item inside render_decoy(); any violation aborts"},
            "gold_arm_bit_identical_to_pc1b_all_items": {"pass": true,
                "items": N_MAX, "max_abs_delta": max_gold_delta,
                "note": "THE one-changed-factor gate. Any non-identity aborts before a payload file is written."},
            "identity_transform_no_adapter_weights_loaded": {"pass": true},
            "all_captured_states_finite": {"pass": true},
            "slot_count_unchanged": {"pass": true, "slots": N_SLOTS},
        },
        "gate_pass": true,
        "wall_clock_s": t0.elapsed().as_secs_f32(),
    });
    common::write_receipt(
        &crate_path("receipts"),
        "run2-pc2-capture-receipt.json",
        &receipt,
    )?;
    println!(
        "PC2 capture complete: {N_MAX} decoy payloads, sha {payload_sha}; gold arm bit-identical \
         to PC1b on all {N_MAX} items (max|d| {max_gold_delta:.3e}); mean cos(steer,restore) \
         {mean_cos:.4} (min {min_cos:.4}); {:.1}s",
        t0.elapsed().as_secs_f32()
    );
    Ok(())
}
