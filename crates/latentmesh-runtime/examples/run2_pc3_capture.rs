//! Run-2 **PC3 capture** — PC2's payload derivation, replayed on the
//! **out-of-sample** tail of `adaptation-512`.
//!
//! Registered by [`docs/adr/040-pc3-decision-change-endpoint-pre-registration.md`],
//! which changes exactly **two** things from PC2 and nothing else:
//!
//! 1. **The item stream** — the eligible `adaptation-512` items **not used by
//!    PC2 / M4i / PC1b**. This binary hard-gates that the intersection with
//!    PC2's committed stream is **EMPTY**.
//! 2. **The registered primary endpoint** — measured in the probe, not here.
//!
//! Decoy construction is **unchanged from PC2** (ChaCha8 seed `0x5732` over
//! `{g+1, g-1, g+10, 2g}`), per ADR-040 § REVERSAL: the clean-floor argument
//! applied to the decoy-emission endpoint, which is no longer primary, and
//! moving decoys off natural-slip space would have lowered `steer` along with
//! `baseline` and made power **worse**.
//!
//! # ⚠️ TWO RECORDED DEVIATIONS FROM PC2, forced by out-of-sample items
//!
//! **(a) `N_max` is 212, not the registered 300 — the pool is exhausted.**
//! `adaptation-512` holds 512 indices and **none** of ADR-024's 13 exclusions
//! fall inside it, so the eligible pool is 512. PC2 consumed the first 300.
//! **212 remain.** ADR-040's "next 300 eligible items NOT used by PC2/M4i"
//! is arithmetically unattainable, and ADR-036 Decision 2 confines the stream
//! to `adaptation-512` (its stratified-resample escalation is explicitly
//! "named, not built"). This rung therefore draws the **entire** remaining
//! out-of-sample pool — the maximal draw the registered item-supply rule
//! permits. Power survives it: at PC2's measured ~26% discordance on the new
//! endpoint, `n_disc` ≈ 55, well clear of ADR-040's own 30-pair floor.
//!
//! **(b) The `restore` payload is DERIVED here, not replayed.** PC2's
//! `restore` arm replayed PC1b's committed payload file byte-for-byte, and
//! PC2's capture gated its gold arm bit-identical against that file. **No
//! PC1b or PC2 artifact covers these items**, so neither is available. This
//! binary derives the gold payload through the *same* [`tap`] call as the
//! decoy payload, with the continuation text as the only differing argument.
//! The one-changed-factor claim stays provable **within** the run —
//! [`render_decoy`] asserts the edit is confined to the final-answer token —
//! but it is no longer cross-validated against a prior committed file. That
//! is a real loss of evidence and is recorded as such, not glossed.
//!
//! # FIREWALL (ADR-040 § Firewall)
//!
//! PC3 is same-model, same-item, identity-transform. It tests **the
//! apparatus, never transfer**. Per the symmetric rule established after
//! PC1b, **neither a PASS nor a FAIL may be cited as evidence for or against
//! cross-model transfer.**
//!
//! Run: PATH=/usr/local/cuda-12.8/bin:$PATH \
//!      cargo run --release --features cuda --example run2_pc3_capture

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
/// PC2's probe receipt — the source of the 300 items PC3 must **not** reuse.
const PC2_PROBE_RECEIPT: &str =
    "receipts/run2-pc2-receipt-identity-L19lasttoken-decoytf-fuse-questiontail-slots8-eprocess.json";
/// M4i's receipt — read for the same exclusion, from the key it actually
/// carries (`items[*].item`). PC1b's probe read `dataset.indices`, absent from
/// that receipt, and recorded a false negative for an identical stream.
const M4I_RECEIPT: &str =
    "receipts/run2-m4i-receipt-cellL18toL14-mlp-pertokenlast-fuse-questiontail-slots8-eprocess.json";

/// The payload tap: S1a's capture block, unchanged through PC1, PC1b, PC2, PC3.
const PAYLOAD_BLOCK: usize = 19;
const D_RECEIVER: usize = 1536;
/// `[L19_last_decoy (STEER) | L19_last_gold (RESTORE) | L14_pooled | L14_last]`.
///
/// **Four**, not PC2's three: PC2's `restore` vectors lived in PC1b's file,
/// which does not cover these items, so the gold arm is carried here. See
/// deviation (b) in the module docs.
const VECS_PER_ITEM: usize = 4;

/// PC3's site — PC2's, unchanged. The site is NOT a changed factor here.
const SITE: Site = Site::QuestionTail;

/// ADR-024's 13 probe-overlap items, verbatim from PC2's copy.
const LEAKAGE_EXCLUSIONS: [usize; 13] = [
    150, 1309, 1573, 2365, 2418, 2540, 2958, 2973, 3084, 3825, 3844, 3877, 4746,
];
/// ADR-040's registered ceiling. The pool cannot supply it — see deviation (a).
const N_MAX_REGISTERED: usize = 300;
/// ADR-040's own floor: below this many discordant pairs the rung is reported
/// uninformative **and the power model is recorded as wrong**.
const N_DISC_FLOOR: usize = 30;

/// ADR-040 § REVERSAL: PC2's decoy seed, deliberately unchanged.
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

/// One teacher-forced tap: PC2's capture statements, factored so the decoy arm
/// and the gold arm provably run the SAME code.
struct Tap {
    l19_last: Vec<f32>,
    l14_pooled: Vec<f32>,
    l14_last: Vec<f32>,
    prompt_tokens: usize,
    continuation_tokens: usize,
    l19_pooled_l2: f32,
}

/// Teacher-force `continuation` after the item's own question and tap block
/// [`PAYLOAD_BLOCK`] at the LAST row of the continuation span — PC2's
/// derivation verbatim, with the continuation text as the only argument that
/// differs between PC3's two arms.
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

/// PC2's gold rendering with **only** the trailing `#### g` replaced by
/// `#### d`, verbatim.
///
/// Everything else — the whole worked solution, its whitespace, the removal of
/// GSM8K's `<<a+b=c>>` calculator annotations by `common::render_gold` — is
/// byte-identical between the two arms. The returned tuple carries the gold
/// rendering too, so the caller can assert the two differ only where they are
/// supposed to.
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

/// Read a committed stream out of a receipt, trying `dataset.indices` first and
/// falling back to `items[*].item`.
///
/// Both keys exist across this ladder's receipts and **neither is universal**:
/// M4i carries only the second, PC2 carries both. PC1b's probe read only the
/// first and recorded `item_stream_identical_to_m4i: false` for an identical
/// stream. Reading both is how that false negative stays fixed.
fn committed_stream(receipt: &serde_json::Value) -> Vec<usize> {
    if let Some(a) = receipt["dataset"]["indices"].as_array() {
        return a
            .iter()
            .filter_map(|v| v.as_u64())
            .map(|v| v as usize)
            .collect();
    }
    receipt["items"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|r| r["item"].as_u64())
                .map(|v| v as usize)
                .collect()
        })
        .unwrap_or_default()
}

#[allow(clippy::too_many_lines)]
fn main() -> anyhow::Result<()> {
    let t0 = std::time::Instant::now();
    let e = anyhow::Error::msg;
    let nvcc = latentmesh_runtime::assert_cuda_build_env().map_err(e)?;
    println!("build-env guard: {nvcc}");

    // ---- Item supply: ADR-036 Decision 2 + ADR-040's out-of-sample rule ----
    let dir = common::run_dir("run2-pc3");
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

    // ---- THE OUT-OF-SAMPLE GATE: PC2's and M4i's items are removed ---------
    let pc2: serde_json::Value = serde_json::from_slice(
        &std::fs::read(crate_path(PC2_PROBE_RECEIPT)).map_err(|err| {
            anyhow::anyhow!(
                "PC2 probe receipt {PC2_PROBE_RECEIPT} unreadable ({err}) — PC3's \
                             out-of-sample gate cannot be evaluated without it"
            )
        })?,
    )?;
    let pc2_stream = committed_stream(&pc2);
    anyhow::ensure!(
        pc2_stream.len() == N_MAX_REGISTERED,
        "PC2's committed stream has {} items, expected {N_MAX_REGISTERED}",
        pc2_stream.len()
    );
    let m4i: serde_json::Value = serde_json::from_slice(
        &std::fs::read(crate_path(M4I_RECEIPT))
            .map_err(|err| anyhow::anyhow!("M4i receipt {M4I_RECEIPT} unreadable ({err})"))?,
    )?;
    let m4i_stream = committed_stream(&m4i);
    anyhow::ensure!(
        m4i_stream.len() == N_MAX_REGISTERED,
        "M4i receipt yielded {} stream entries, expected {N_MAX_REGISTERED}",
        m4i_stream.len()
    );
    anyhow::ensure!(
        m4i_stream == pc2_stream,
        "M4i's and PC2's committed streams differ — PC3's exclusion set is ambiguous and the rung \
         must not proceed"
    );
    let used: std::collections::BTreeSet<usize> = pc2_stream
        .iter()
        .chain(m4i_stream.iter())
        .copied()
        .collect();
    let remaining: Vec<usize> = eligible
        .iter()
        .copied()
        .filter(|i| !used.contains(i))
        .collect();
    println!(
        "item supply: adaptation-512 = {} indices, {} of the 13 exclusions present, eligible {}, \
         used by PC2/M4i {}, REMAINING OUT-OF-SAMPLE {}",
        adaptation_indices.len(),
        excluded_present.len(),
        eligible.len(),
        used.len(),
        remaining.len()
    );
    anyhow::ensure!(
        remaining.len() >= N_DISC_FLOOR,
        "only {} out-of-sample items remain — below even the n_disc floor; the rung cannot be run",
        remaining.len()
    );
    if remaining.len() < N_MAX_REGISTERED {
        println!(
            "⚠️  DEVIATION (a): ADR-040 registers N_max = {N_MAX_REGISTERED} out-of-sample items. \
             The pool holds {}. Drawing the ENTIRE remaining out-of-sample pool instead — the \
             maximal draw ADR-036 Decision 2's adaptation-512-only rule permits. Recorded as a \
             first-class gate field, not buried.",
            remaining.len()
        );
    }
    let n_max = remaining.len().min(N_MAX_REGISTERED);

    // ---- Model: receiver ONLY (PC3 is a same-model self-pair) -------------
    let device = latentmesh_runtime::device().map_err(e)?;
    println!("loading {RECEIVER} (BF16)... [no sender, no adapter]");
    let mut rt = QwenRuntime::load(RECEIVER, &device, candle_core::DType::BF16).map_err(e)?;
    let pad_id = rt
        .tokenizer
        .token_to_id("<|fim_pad|>")
        .ok_or_else(|| anyhow::anyhow!("<|fim_pad|> not in tokenizer"))?;
    println!("model loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // ---- Site pre-flight — PC2's, unchanged -------------------------------
    let mut stream: Vec<usize> = Vec::new();
    let mut tokenization_excluded: Vec<serde_json::Value> = Vec::new();
    for &idx in &remaining {
        if stream.len() == n_max {
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
        stream.len() >= N_DISC_FLOOR,
        "site pre-flight resolved only {} items — below the n_disc floor",
        stream.len()
    );
    let n = stream.len();

    // ---- HARD GATE: the intersection with PC2's stream is EMPTY ------------
    let overlap: Vec<usize> = stream
        .iter()
        .copied()
        .filter(|i| used.contains(i))
        .collect();
    anyhow::ensure!(
        overlap.is_empty(),
        "PC3's stream intersects PC2/M4i's committed stream on {} items ({:?}) — this rung is an \
         OUT-OF-SAMPLE confirmation and must not reuse a single item",
        overlap.len(),
        &overlap[..overlap.len().min(10)]
    );
    println!(
        "site pre-flight: {n} items at {N_SLOTS} question-tail positions, {} excluded on \
         tokenisation grounds; OUT-OF-SAMPLE GATE PASSED — intersection with PC2/M4i's committed \
         300-item stream is EMPTY",
        tokenization_excluded.len()
    );

    // ---- DECOY COMMITMENT — computed and written BEFORE any capture pass ---
    // ADR-040 § REVERSAL: construction unchanged from PC2 (seed 0x5732, menu
    // {g+1, g-1, g+10, 2g}). Nothing below this block can influence the table
    // above it.
    let mut rng = ChaCha8Rng::seed_from_u64(DECOY_SEED);
    let mut decoys: Vec<Decoy> = Vec::with_capacity(n);
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
        "stage": "run2-PC3-decoy-commitment",
        "design": "docs/adr/040-pc3-decision-change-endpoint-pre-registration.md § REVERSAL — 'Decoy construction is therefore unchanged from PC2 (ChaCha8 seed 0x5732 over {g+1, g-1, g+10, 2g}), committed before the draw.'",
        "env": common::env_info(&nvcc),
        "pre_committed": true,
        "written_before": "any payload forward pass. The full (item, gold, decoy) table below is a pure function of the committed OUT-OF-SAMPLE item stream and the registered seed; no measurement of any kind precedes it.",
        "why_decoys_stay_in_natural_slip_space_verbatim_adr_040": "ADR-039's closing note said a successor must draw decoys AWAY from natural-slip space to recover a genuine chance floor. ADR-040 REVERSES that: (1) the clean-floor argument applied to the decoy-emission endpoint, which is no longer the primary and is moot here; (2) moving decoys away from plausible answers would lower the baseline rate AND lower steer with it, making power WORSE; (3) natural-slip decoys are the STRONGER test of steering, being answers the model would plausibly accept.",
        "implementation": {
            "seed": DECOY_SEED,
            "seed_hex": format!("{DECOY_SEED:#x}"),
            "stream": "ONE ChaCha8 stream seeded at 0x5732, consumed in the committed item-stream order — PC2's construction, unchanged. Because PC3's item stream differs from PC2's, the realised decoy VALUES differ; the RULE and the SEED do not.",
            "menu": PERTURBATIONS,
            "draw": "k = rng.gen_range(0..4); d = menu[k](g); accept iff d != g and d > 0, else redraw",
            "menu_histogram": {"g+1": menu_histogram[0], "g-1": menu_histogram[1],
                                "g+10": menu_histogram[2], "2g": menu_histogram[3]},
            "total_redraws": decoys.iter().map(|d| d.redraws).sum::<usize>(),
        },
        "item_stream_source": "adaptation-512, fixed ascending index order, ADR-024's 13-item leakage exclusion, OUT-OF-SAMPLE tail only — gated to have EMPTY intersection with PC2's and M4i's committed 300-item stream",
        "commitments": commitment_rows,
        "n": decoys.len(),
    });
    common::write_receipt(
        &crate_path("receipts"),
        "run2-pc3-decoy-commitment-receipt.json",
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
    // Two teacher-forced prefills per item through the SAME `tap` call: the
    // DECOY solution (the STEER payload) and the GOLD solution (the RESTORE
    // payload). PC2 got its gold arm from PC1b's committed file; no such file
    // covers these items — deviation (b).
    let mut flat: Vec<f32> = Vec::with_capacity(n * VECS_PER_ITEM * D_RECEIVER);
    let mut rows = Vec::new();
    let mut steer_vs_restore_cos: Vec<f64> = Vec::new();

    for (done, d) in decoys.iter().enumerate() {
        let item = &all_items[d.item];
        let (decoy_text, gold_text) =
            render_decoy(&item.answer_text, &item.gold, &d.decoy.to_string())?;

        // (1) THE STEER PAYLOAD — teacher-forced over the decoy solution.
        let t_steer = tap(&mut rt, &device, &item.question, &decoy_text)?;
        // (2) THE RESTORE PAYLOAD — same call, gold continuation.
        let t_gold = tap(&mut rt, &device, &item.question, &gold_text)?;

        let steer = t_steer.l19_last;
        let restore = t_gold.l19_last;

        // How far the one changed token actually moves the payload. Recorded,
        // never gating: if this were ~1.0 everywhere the rung would be
        // measuring almost nothing, and a reader must see it from the receipt.
        let dot: f64 = steer
            .iter()
            .zip(restore.iter())
            .map(|(a, b)| f64::from(*a) * f64::from(*b))
            .sum();
        let cos = dot / (f64::from(norms::l2(&steer)) * f64::from(norms::l2(&restore))).max(1e-12);
        steer_vs_restore_cos.push(cos);

        rows.push(serde_json::json!({
            "item": d.item,
            "gold": d.gold,
            "decoy": d.decoy,
            "perturbation": PERTURBATIONS[d.perturbation],
            "prompt_tokens": t_steer.prompt_tokens,
            "decoy_continuation_tokens": t_steer.continuation_tokens,
            "gold_continuation_tokens": t_gold.continuation_tokens,
            "payload_l19_last_l2": norms::l2(&steer),
            "restore_l19_last_l2": norms::l2(&restore),
            "reference_l19_pooled_l2": t_steer.l19_pooled_l2,
            "reference_l14_pooled_l2": norms::l2(&t_steer.l14_pooled),
            "reference_l14_last_l2": norms::l2(&t_steer.l14_last),
            "cosine_steer_vs_restore": cos,
        }));
        flat.extend_from_slice(&steer);
        flat.extend_from_slice(&restore);
        flat.extend_from_slice(&t_steer.l14_pooled);
        flat.extend_from_slice(&t_steer.l14_last);
        if (done + 1) % 25 == 0 || done + 1 == decoys.len() {
            println!(
                "[{}/{n}] item {} gold {} -> decoy {} ({}): {} decoy tokens, |L19_last| {:.2}, \
                 cos(steer,restore) {cos:.4}  {:.0}s",
                done + 1,
                d.item,
                d.gold,
                d.decoy,
                PERTURBATIONS[d.perturbation],
                t_steer.continuation_tokens,
                norms::l2(&steer),
                t0.elapsed().as_secs_f32()
            );
        }
    }
    let mean_cos = steer_vs_restore_cos.iter().sum::<f64>() / steer_vs_restore_cos.len() as f64;
    let min_cos = steer_vs_restore_cos
        .iter()
        .copied()
        .fold(f64::MAX, f64::min);
    let max_cos = steer_vs_restore_cos
        .iter()
        .copied()
        .fold(f64::MIN, f64::max);

    // ---- Write the payload artifact ---------------------------------------
    let out = dir.join("run2-pc3-payloads.f32bin");
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
        "stage": "run2-PC3-capture",
        "design": "docs/adr/040-pc3-decision-change-endpoint-pre-registration.md. PC2's derivation, replayed on the OUT-OF-SAMPLE tail of adaptation-512. Two things change from PC2: the item stream and the registered primary endpoint (measured in the probe). Site, operator, slots, decoy construction are unchanged.",
        "env": common::env_info(&nvcc),
        "pre_committed": true,

        "FIREWALL_apparatus_not_transfer": {
            "verbatim_adr_040": "PC3 is same-model, same-item, identity-transform. It tests the apparatus, never transfer. Per the symmetric rule established after PC1b, NEITHER a PASS nor a FAIL may be cited as evidence for or against cross-model transfer.",
            "rule": "Neither branch of this rung is evidence about transfer.",
        },

        "DO_NOT_INHERIT_PC1B_FAIL_WORDING": {
            "quarantined_text": "PC1b's registered FAIL branch said a powered failure would make the ladder nulls 'evidence about TRANSFER rather than about plumbing.'",
            "why_it_is_wrong": "That is logically inverted (quarantined at the head of ADR-024). A failed positive control makes the METHOD the leading explanation for every null; ruling plumbing out requires a PASS on the endpoint being measured.",
            "this_receipt": "does not use it anywhere.",
        },

        "RECORDED_DEVIATIONS_FROM_THE_PRE_REGISTRATION": {
            "a_n_max_reduced_pool_exhausted": {
                "registered": N_MAX_REGISTERED,
                "actually_drawn": n,
                "arithmetic": {
                    "adaptation_512_indices": adaptation_indices.len(),
                    "adr_024_exclusions_present_in_the_split": excluded_present.len(),
                    "eligible_pool": eligible.len(),
                    "consumed_by_pc2_and_m4i": used.len(),
                    "remaining_out_of_sample": remaining.len(),
                    "excluded_on_tokenisation_grounds": tokenization_excluded.len(),
                },
                "why": "ADR-040 registers 'the next 300 eligible items NOT used by PC2/M4i'. adaptation-512 holds 512 indices and NONE of ADR-024's 13 exclusions fall inside it, so the eligible pool is 512, not 499. PC2/M4i consumed the first 300. The registered 300 out-of-sample items DO NOT EXIST.",
                "what_was_done_instead": "the ENTIRE remaining out-of-sample pool was drawn — the maximal draw ADR-036 Decision 2's adaptation-512-only item-supply rule permits. Its stratified-resample escalation into a further tranche is explicitly 'named as an available secondary option, not built, not scheduled'.",
                "power_consequence": "at PC2's measured ~26% discordance on the new endpoint, the expected n_disc at this N is well above ADR-040's own 30-pair floor, so the rung remains POWERED. The realised n_disc is reported by the probe and is what actually settles this.",
                "not_a_protocol_shop": "the endpoint, lambda, alpha, wealth threshold, direction, site, operator, slots and decoy construction are all exactly as ADR-040 froze them. Only the attainable N changed, and it changed because of arithmetic fixed before this rung existed.",
            },
            "b_restore_is_derived_not_replayed": {
                "pc2_did": "replayed PC1b's committed payload FILE byte-for-byte as the `restore` arm, and gated its own gold arm bit-identical against that file over all 300 items.",
                "pc3_does": "derives the gold payload itself, through the SAME `tap()` call as the decoy payload, with the continuation text as the only differing argument.",
                "why": "no PC1b or PC2 artifact covers these out-of-sample items, so neither the byte-replay nor the bit-identity gate is available.",
                "what_is_lost": "the cross-run bit-identity check. PC2 could PROVE its derivation had not drifted from PC1b's; PC3 cannot, because there is nothing to compare against on these items.",
                "what_is_retained": "the one-changed-factor claim remains provable WITHIN the run: both arms call the same `tap()` at the same blocks with the same last-row tap, and render_decoy() asserts per item that the two continuation texts differ ONLY in the whitespace-delimited token after the final '####'.",
                "gating": "this deviation is DISCLOSED, not gated away.",
            },
        },

        "the_two_changed_factors_vs_pc2": {
            "changed": [
                "the item stream — the out-of-sample tail of adaptation-512, gated to share NOT ONE item with PC2/M4i",
                "the registered primary endpoint — answer-differs-from-own-baseline, evaluated in the probe",
            ],
            "held_identical": [
                "the site (question-tail) and the slot count",
                "the injection operator (fuse) — applied in the probe",
                "the gold rendering rule (common::render_gold), applied first and then edited at exactly one token",
                "the decoy construction (ChaCha8 seed 0x5732, menu {g+1, g-1, g+10, 2g}) — ADR-040 § REVERSAL keeps decoys in natural-slip space deliberately",
                "the prompt (SYSTEM + question + ANSWER_FORMAT under the same chat template)",
                "the tap (forward_capture_multi_with_rows at blocks [14, 19], LAST row of the continuation span)",
                "the absence of any adapter — no MlpTransform / FastGrnnTransform / AffineTransform is named, constructed or applied anywhere in this binary",
            ],
            "how_far_the_one_token_moves_the_payload": {
                "metric": "cosine(steer payload, restore payload) per item — the same diagnostic PC2 recorded",
                "mean": mean_cos,
                "min": min_cos,
                "max": max_cos,
                "gating": "NONE — diagnostic. A cosine of ~1.0 everywhere would mean the decoy edit barely perturbs the tapped state, which a reader must be able to see when interpreting the outcome.",
            },
        },

        "decoy_construction": {
            "commitment_receipt": "receipts/run2-pc3-decoy-commitment-receipt.json",
            "written_before_any_forward_pass": true,
            "rule_unchanged_from_pc2": "d is derived deterministically from the gold answer g — not sampled, and never equal to g: a fixed ChaCha8 stream (seed 0x5732) selects a perturbation from {g+1, g-1, g+10, 2g}, re-drawing on collision with g or on a non-positive result.",
            "seed": DECOY_SEED,
            "menu": PERTURBATIONS,
            "menu_histogram": {"g+1": menu_histogram[0], "g-1": menu_histogram[1],
                                "g+10": menu_histogram[2], "2g": menu_histogram[3]},
            "total_redraws": decoys.iter().map(|d| d.redraws).sum::<usize>(),
            "edit_scope": "render_decoy() replaces ONLY the whitespace-delimited token following the LAST '####' in common::render_gold's output, and asserts (a) that token normalises to the item's gold, (b) the rebuilt text reads back as the decoy, (c) the byte prefix through '####'+whitespace and the byte suffix after the number are identical between the two renderings.",
            "leakage_check_is_the_PROBE's_job": "the rate at which BASELINE already emits d is measured in run2_pc3_probe, not here.",
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

        "item_supply": {
            "adr": "ADR-036 Decision 2 item-supply rule + ADR-040's out-of-sample restriction",
            "source": ADAPTATION_512,
            "source_split": "adaptation-512",
            "consumption_order": "the file's own fixed ascending index order, sequential, never shuffled and never re-seeded",
            "leakage_exclusion": {"excluded_item_indices": LEAKAGE_EXCLUSIONS,
                "present_in_this_split": excluded_present},
            "eligible_pool_size": eligible.len(),
            "consumed_by_pc2_m4i": used.len(),
            "remaining_out_of_sample": remaining.len(),
            "n_max_registered": N_MAX_REGISTERED,
            "n_drawn": n,
            "first_item": stream.first(),
            "tokenization_preflight_exclusions": tokenization_excluded,
            "pc2_receipt_compared_against": PC2_PROBE_RECEIPT,
            "m4i_receipt_compared_against": M4I_RECEIPT,
            "intersection_with_pc2_m4i_stream": overlap,
        },

        "payload_file": {
            "path": out.display().to_string(),
            "sha256": payload_sha,
            "n_items": n,
            "dim": D_RECEIVER,
            "vecs_per_item": VECS_PER_ITEM,
            "layout": "n items in the e-process stream order; per item FOUR contiguous f32 vectors of 1536, row-major little-endian: [L19_last over the DECOY solution (STEER) | L19_last over the GOLD solution (RESTORE) | L14_pooled | L14_last]. PC2's layout carried THREE because its restore vectors lived in PC1b's file, which does not cover these items — deviation (b).",
        },

        "dataset": {"source": common::GSM8K_TRAIN_URL, "sha256": train_sha, "indices": stream},
        "items": rows,

        "gates": {
            "OUT_OF_SAMPLE_intersection_with_pc2_m4i_is_empty": {"pass": true, "overlap": 0,
                "pc2_stream_items": pc2_stream.len(), "pc3_stream_items": n,
                "note": "HARD GATE — the binary aborts on any overlap. This is the defining property of PC3."},
            "pc2_and_m4i_streams_agree": {"pass": true, "n": N_MAX_REGISTERED},
            "decoys_committed_before_any_forward_pass": {"pass": true,
                "receipt": "receipts/run2-pc3-decoy-commitment-receipt.json"},
            "no_decoy_equals_its_gold": {"pass": true},
            "all_decoys_positive": {"pass": true},
            "decoy_edit_confined_to_the_final_answer_token": {"pass": true,
                "note": "asserted per item inside render_decoy(); any violation aborts"},
            "identity_transform_no_adapter_weights_loaded": {"pass": true},
            "all_captured_states_finite": {"pass": true},
            "slot_count_unchanged": {"pass": true, "slots": N_SLOTS},
            "n_max_reduced_from_300_pool_exhausted": {
                "pass": true,
                "is_a_deviation": n < N_MAX_REGISTERED,
                "registered": N_MAX_REGISTERED, "drawn": n,
                "remaining_out_of_sample_pool": remaining.len(),
                "note": "NOT a silent pass — see RECORDED_DEVIATIONS_FROM_THE_PRE_REGISTRATION.a for the full arithmetic. Marked pass because drawing the entire remaining pool is the maximal compliant response to an unattainable N, not because the deviation is immaterial."},
            "restore_derived_not_replayed": {
                "pass": true, "is_a_deviation": true,
                "note": "NOT a silent pass — see RECORDED_DEVIATIONS_FROM_THE_PRE_REGISTRATION.b. The cross-run bit-identity check PC2 could run is UNAVAILABLE here and is not claimed."},
        },
        "gate_pass": true,
        "wall_clock_s": t0.elapsed().as_secs_f32(),
    });
    common::write_receipt(
        &crate_path("receipts"),
        "run2-pc3-capture-receipt.json",
        &receipt,
    )?;
    println!(
        "PC3 capture complete: {n} out-of-sample items (registered {N_MAX_REGISTERED}, pool held \
         {}), sha {payload_sha}; mean cos(steer,restore) {mean_cos:.4} (min {min_cos:.4}, max \
         {max_cos:.4}); {:.1}s",
        remaining.len(),
        t0.elapsed().as_secs_f32()
    );
    Ok(())
}
