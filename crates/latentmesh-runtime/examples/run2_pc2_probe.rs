//! Run-2 **PC2** — the steering control.
//!
//! Registered by [`docs/adr/039-pc2-steering-control-pre-registration.md`],
//! which asks the strictly easier question PC1 and PC1b could not:
//!
//! > *"can the apparatus move the receiver's answer at all — in any direction,
//! > including a wrong one?"*
//!
//! # Why restoration could not answer it
//!
//! PC1 and PC1b are both **restoration** controls: they hand the receiver the
//! right answer and ask whether it gets more answers right. PC1b failed at the
//! ladder's highest power (`n_disc` 64, final wealth 0.1949, never once above
//! its 1.0 start) — but a restoration null is consistent with *"the pathway
//! cannot steer decisions"* **and** with *"the receiver's errors are not of a
//! kind the gold state repairs."* It cannot separate them.
//!
//! PC2 replaces restoration with **steering**: the payload is the receiver's
//! own block-19 state teacher-forced over its own solution **whose final
//! numeric answer has been replaced by a pre-committed decoy `d`**. The
//! primary endpoint is the rate at which the receiver **emits `d`**.
//!
//! - `random` hits `d` at chance **by construction** — a clean floor that
//!   restoration never had.
//! - The effect ceiling is far higher: restoration is bounded by the 160/300
//!   items the receiver already gets wrong; steering can move **any** item.
//! - Exactly one thing varies from PC1b: the answer value. Same site, same
//!   operator, same slots, same derivation, same stream — and the capture
//!   binary **proves** it by re-deriving PC1b's gold payload for all 300 items
//!   and gating on bit-identity.
//!
//! # Conditions (ADR-039 § Conditions)
//!
//! | condition | payload |
//! |---|---|
//! | `steer` | L19 last-token state, teacher-forced over the **decoy** solution |
//! | `restore` | **PC1b's own committed vectors**, replayed byte-for-byte |
//! | `baseline` | no injection |
//! | `zerovec` | `h += 0` — must be bit-identical to baseline |
//! | `random` | per-item seeded Gaussian, norm-matched to the **effective steer** vector |
//!
//! # Endpoint and statistic
//!
//! **Primary**: decoy-emission rate, `steer` vs `random`, paired per item,
//! under ADR-036's e-process (λ = 0.30, α = 0.05, wealth threshold 20.0,
//! `N_max` = 300, `adaptation-512` fixed order, ADR-024's 13-item exclusion).
//!
//! **LEAKAGE GATE, evaluated before any interpretation**: the rate at which
//! `baseline` already emits `d`. Expected ≈ 0. **Above 2% the decoy
//! construction is declared leaky and THE RUNG IS VOID** — no PASS/FAIL
//! reading is issued.
//!
//! # PRE-REGISTERED INTERPRETATION (ADR-039), recorded before the draw
//!
//! - **PASS** — the apparatus **can steer decisions**. PC1b's failure is
//!   re-read as *restoration being the wrong probe*, not as a dead pathway.
//!   M5X (ADR-037) and M4b (ADR-035) **UNBLOCK**.
//! - **FAIL with real power** — the apparatus **cannot move a decision by any
//!   means**, not even toward an answer it was explicitly handed. With PC1b
//!   that is decisive about the **method**: this injection paradigm is
//!   **decision-inert while remaining likelihood-live**. M5X and M4b stay
//!   blocked permanently, the ladder closes, and this becomes the publishable
//!   result, reported without softening (ADR-032).
//! - **Underpowered** — reported as uninformative, on its own scale, and **not
//!   spun as either outcome** (ADR-036 Decision 3).
//!
//! ## ⛔ PC1b's FAIL wording is NOT inherited
//!
//! PC1b registered that a powered failure would make the ladder's nulls
//! *"evidence about TRANSFER rather than about plumbing."* **That is logically
//! inverted** and is quarantined at the head of ADR-024: a failed positive
//! control makes the **method** the leading explanation; ruling plumbing out
//! requires a **PASS**. The branches above are ADR-039's and avoid it.
//!
//! # FIREWALL (ADR-039 § Firewall, extended after PC1b)
//!
//! PC2 is same-model, same-item, identity-transform, gold-adjacent. It tests
//! **the apparatus, never transfer**. **A PASS may not be cited as transfer
//! evidence, and a FAIL may not either** — the symmetric rule.
//!
//! Run: PATH=/usr/local/cuda-12.8/bin:$PATH \
//!      cargo run --release --features cuda --example run2_pc2_probe

#![recursion_limit = "512"]

#[path = "common/mod.rs"]
#[allow(dead_code)]
mod common;

use common::m3::{
    build_site_prompt, Site, MAX_NEW_TOKENS, N_SLOTS, RANDVEC_SEED_BASE, RECEIVER, RECEIVER_BLOCK,
};
use latentmesh_runtime::capture::forward_capture;
use latentmesh_runtime::inject::{teacher_forced_nll, InjectionMode, InjectionSpec};
use latentmesh_runtime::sampler::{Sampler, Sampling};
use latentmesh_runtime::{norms, QwenRuntime};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::path::{Path, PathBuf};

const GSM8K_TRAIN_SHA256: &str = "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465";
const ADAPTATION_512: &str = "../../harness/latentmesh-live/data/adaptation-512.json";
const CAPTURE_RECEIPT: &str = "receipts/run2-pc2-capture-receipt.json";
const COMMITMENT_RECEIPT: &str = "receipts/run2-pc2-decoy-commitment-receipt.json";
/// PC1b's capture receipt — source of the `restore` payload file.
const PC1B_CAPTURE_RECEIPT: &str = "receipts/run2-pc1b-capture-receipt.json";
/// PC1b's draw receipt — read for the side-by-side block and for the
/// mechanical-identity cross-check on the `restore` arm. Never gates.
const PC1B_PROBE_RECEIPT: &str =
    "receipts/run2-pc1b-receipt-identity-L19lasttoken-goldtf-fuse-questiontail-slots8-eprocess.json";
const M4I_RECEIPT: &str =
    "receipts/run2-m4i-receipt-cellL18toL14-mlp-pertokenlast-fuse-questiontail-slots8-eprocess.json";

const D_RECEIVER: usize = 1536;
const VECS_PER_ITEM: usize = 3;

/// Mechanics, unchanged from PC1b.
const INJECT_MODE: InjectionMode = InjectionMode::Fuse;
const SITE: Site = Site::QuestionTail;

/// ADR-024's 13 probe-overlap items, verbatim from PC1b's probe.
const LEAKAGE_EXCLUSIONS: [usize; 13] = [
    150, 1309, 1573, 2365, 2418, 2540, 2958, 2973, 3084, 3825, 3844, 3877, 4746,
];

// ---- ADR-036 Decision 1 / ADR-030 §3.2 e-process parameters, frozen -------
const LAMBDA: f64 = 0.30;
const E_ALPHA: f64 = 0.05;
const N_MAX: usize = 300;
/// Tolerance for the fuse zero-payload no-op diagnostic (nats).
const FUSE_NOOP_TOL: f32 = 1e-6;

/// ADR-039 § "Decoy construction": above this baseline decoy-emission rate the
/// construction is leaky and the rung is VOID.
const LEAKAGE_VOID_RATE: f64 = 0.02;

// --- Pre-committed gaming-guard thresholds (docs/research/045 §3) ----------
// PC1b's values, UNCHANGED, so the two rungs' guards are directly comparable.
const DEGENERATE_LEN_RATIO: f64 = 0.25;
const DEGENERATE_ITEM_FRACTION: f64 = 0.25;
const NLL_COLLAPSE_NATS: f32 = 0.10;

fn crate_path(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

/// One condition's measured outcome.
#[derive(Clone)]
struct Cond {
    /// The receiver emitted the item's GOLD answer.
    correct: bool,
    /// The receiver emitted the item's committed DECOY — the primary endpoint.
    emits_decoy: bool,
    answer: Option<String>,
    nll_gold: f32,
    nll_decoy: f32,
    chars: usize,
    text: String,
}

/// The five conditions for one item.
struct Five {
    steer: Cond,
    restore: Cond,
    base: Cond,
    zero: Cond,
    rand: Cond,
}

/// One step of the registered wealth process (primary = decoy emission).
struct EStep {
    order: usize,
    item: usize,
    steer_emits_decoy: bool,
    random_emits_decoy: bool,
    discordant: bool,
    x: Option<u8>,
    wealth: f64,
}

fn read_payloads(path: &Path, want_sha: &str) -> anyhow::Result<Vec<Vec<f32>>> {
    let bytes = std::fs::read(path)?;
    let sha = common::sha256_hex(&bytes);
    anyhow::ensure!(
        sha == want_sha,
        "payload sha256 {sha} != receipt-pinned {want_sha} for {}",
        path.display()
    );
    anyhow::ensure!(
        bytes.len() == N_MAX * VECS_PER_ITEM * D_RECEIVER * 4,
        "payload file size mismatch for {}",
        path.display()
    );
    let flat: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Ok((0..N_MAX)
        .map(|i| {
            let b = i * VECS_PER_ITEM * D_RECEIVER;
            flat[b..b + D_RECEIVER].to_vec()
        })
        .collect())
}

#[allow(clippy::too_many_lines)]
fn main() -> anyhow::Result<()> {
    let t0 = std::time::Instant::now();
    let e = anyhow::Error::msg;
    let nvcc = latentmesh_runtime::assert_cuda_build_env().map_err(e)?;
    println!("build-env guard: {nvcc}");
    let w_threshold = 1.0 / E_ALPHA;

    // ---- Gate 1: PC2's capture receipt and its sha-pinned payload ---------
    let cap: serde_json::Value =
        serde_json::from_slice(&std::fs::read(crate_path(CAPTURE_RECEIPT)).map_err(|err| {
            anyhow::anyhow!(
                "capture receipt {CAPTURE_RECEIPT} unreadable ({err}) — run run2_pc2_capture first"
            )
        })?)?;
    for gate in [
        "pc1b_reference_payload_sha256_intact",
        "stream_identical_to_m4i",
        "stream_identical_to_pc1b",
        "decoys_committed_before_any_forward_pass",
        "no_decoy_equals_its_gold",
        "all_decoys_positive",
        "decoy_edit_confined_to_the_final_answer_token",
        "gold_arm_bit_identical_to_pc1b_all_items",
        "identity_transform_no_adapter_weights_loaded",
        "all_captured_states_finite",
    ] {
        anyhow::ensure!(
            cap["gates"][gate]["pass"].as_bool() == Some(true),
            "capture receipt gate {gate} did not pass"
        );
    }
    let steer_path = PathBuf::from(
        cap["payload_file"]["path"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("capture receipt does not name the payload file"))?,
    );
    let steer_sha = cap["payload_file"]["sha256"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("capture receipt does not pin the payload sha256"))?
        .to_string();
    let steer_payloads = read_payloads(&steer_path, &steer_sha)?;
    println!("STEER payload VERIFIED: sha {steer_sha} ({N_MAX} x {D_RECEIVER} f32, identity)");

    // ---- Gate 2: PC1b's payload, replayed byte-for-byte as `restore` ------
    let pc1b_cap: serde_json::Value =
        serde_json::from_slice(&std::fs::read(crate_path(PC1B_CAPTURE_RECEIPT))?)?;
    let restore_path = PathBuf::from(
        pc1b_cap["payload_file"]["path"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("PC1b capture receipt does not name its payload"))?,
    );
    let restore_sha = pc1b_cap["payload_file"]["sha256"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("PC1b capture receipt does not pin a payload sha256"))?
        .to_string();
    let restore_payloads = read_payloads(&restore_path, &restore_sha)?;
    println!("RESTORE payload VERIFIED (PC1b's own file, replayed): sha {restore_sha}");

    // ---- Gate 3: the decoy commitment, read back independently ------------
    let commit: serde_json::Value = serde_json::from_slice(
        &std::fs::read(crate_path(COMMITMENT_RECEIPT)).map_err(|err| {
            anyhow::anyhow!("decoy commitment receipt {COMMITMENT_RECEIPT} unreadable ({err})")
        })?,
    )?;
    let commitments = commit["commitments"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("commitment receipt has no commitments array"))?;
    anyhow::ensure!(
        commitments.len() == N_MAX,
        "commitment receipt covers {} items, expected {N_MAX}",
        commitments.len()
    );

    // ---- Item supply: ADR-036 Decision 2, re-derived independently --------
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
    anyhow::ensure!(
        eligible.len() >= N_MAX,
        "eligible pool ({}) is smaller than N_max ({N_MAX})",
        eligible.len()
    );

    // ---- Model: receiver ONLY (PC2 is a same-model self-pair) -------------
    let device = latentmesh_runtime::device().map_err(e)?;
    println!("loading {RECEIVER} (BF16)... [no sender, no adapter]");
    let mut receiver = QwenRuntime::load(RECEIVER, &device, candle_core::DType::BF16).map_err(e)?;
    // Resolved but UNUSED under `Site::QuestionTail` — kept so the shared site
    // resolver is called with exactly the arguments every prior rung used.
    let pad_id = receiver
        .tokenizer
        .token_to_id("<|fim_pad|>")
        .ok_or_else(|| anyhow::anyhow!("<|fim_pad|> not in tokenizer"))?;
    println!("model loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // ---- Tokenisation pre-flight, BEFORE any generation -------------------
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

    // The capture, the commitment and PC1b must all have walked EXACTLY this
    // stream, or row i is not item i in one of the three artifacts.
    let cap_stream: Vec<usize> = cap["dataset"]["indices"]
        .as_array()
        .expect("capture receipt indices")
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    anyhow::ensure!(
        stream == cap_stream,
        "the PC2 capture receipt's stream differs from this probe's independently-derived one"
    );
    let commit_stream: Vec<usize> = commitments
        .iter()
        .filter_map(|c| c["item"].as_u64())
        .map(|v| v as usize)
        .collect();
    anyhow::ensure!(
        stream == commit_stream,
        "the decoy commitment's stream differs from this probe's — a decoy would be paired with \
         the wrong item"
    );
    let pc1b_stream: Vec<usize> = pc1b_cap["dataset"]["indices"]
        .as_array()
        .expect("PC1b capture indices")
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    anyhow::ensure!(
        stream == pc1b_stream,
        "PC1b's stream differs from this probe's — the restore payload would not belong to the item"
    );

    // ---- HARD GATE: the stream is M4i's committed stream -------------------
    // Read from `items[*].item` — the key M4i's receipt actually carries.
    // PC1b's probe read `dataset.indices`, absent from that receipt, and
    // recorded a false negative for an identical stream.
    let m4i: serde_json::Value = serde_json::from_slice(
        &std::fs::read(crate_path(M4I_RECEIPT))
            .map_err(|err| anyhow::anyhow!("M4i receipt {M4I_RECEIPT} unreadable ({err})"))?,
    )?;
    let m4i_stream: Vec<usize> = m4i["items"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("M4i receipt has no items array"))?
        .iter()
        .filter_map(|r| r["item"].as_u64())
        .map(|v| v as usize)
        .collect();
    anyhow::ensure!(
        m4i_stream == stream,
        "the resolved stream differs from M4i's committed 300-item stream"
    );
    println!(
        "site pre-flight: {N_MAX} items at {N_SLOTS} question-tail positions, {} excluded on \
         tokenisation grounds; stream HARD-GATED identical to M4i's (items[*].item), PC1b's, the \
         PC2 capture's and the decoy commitment's",
        tokenization_excluded.len()
    );
    for s in &site_samples {
        println!(
            "  item {} -> positions {} decode to {}",
            s["item"], s["positions"], s["positions_decoded"]
        );
    }

    // PC1b's per-item aligned_real outcomes — used ONLY for the mechanical
    // identity cross-check on the `restore` arm. Never gates.
    let pc1b_probe: serde_json::Value =
        serde_json::from_slice(&std::fs::read(crate_path(PC1B_PROBE_RECEIPT)).unwrap_or_default())
            .unwrap_or(serde_json::Value::Null);
    let pc1b_aligned: Vec<Option<bool>> = (0..N_MAX)
        .map(|i| pc1b_probe["items"][i]["conditions"]["aligned_real"]["correct"].as_bool())
        .collect();

    // ---- THE DRAW: ADR-036 e-process on DECOY EMISSION --------------------
    let mut wealth = 1.0f64;
    let mut max_wealth = 1.0f64;
    let mut trajectory: Vec<EStep> = Vec::new();
    let mut rows: Vec<serde_json::Value> = Vec::new();
    let mut five: Vec<Five> = Vec::new();
    let mut decoys: Vec<i64> = Vec::new();
    let mut wins = 0usize;
    let mut losses = 0usize;
    let mut crossed_at: Option<usize> = None;
    let mut restore_matches_pc1b = 0usize;
    let mut restore_comparable = 0usize;

    for (order, &idx) in stream.iter().enumerate() {
        let item = &all_items[idx];
        let decoy = commitments[order]["decoy"]
            .as_i64()
            .ok_or_else(|| anyhow::anyhow!("commitment row {order} has no decoy"))?;
        anyhow::ensure!(
            commitments[order]["gold"].as_i64().map(|g| g.to_string()) == Some(item.gold.clone()),
            "commitment row {order} gold does not match the dataset's gold for item {idx}"
        );
        let decoy_s = decoy.to_string();
        anyhow::ensure!(
            !common::answers_equal(&decoy_s, &item.gold),
            "item {idx}: the committed decoy equals the gold answer"
        );
        decoys.push(decoy);

        // --- site prompt + rescale target (PC1b's statements, unchanged) ---
        let sp = build_site_prompt(&receiver, item, pad_id, SITE)?;
        let inj_tokens = sp.tokens.clone();
        let positions = sp.positions.clone();
        let (_, nat_cap) = forward_capture(
            &mut receiver.model,
            &inj_tokens,
            RECEIVER_BLOCK,
            0..inj_tokens.len(),
            &device,
        )
        .map_err(e)?;
        let natural = norms::stats(nat_cap.per_position_l2.clone());

        let steer_vec = &steer_payloads[order];
        let restore_vec = &restore_payloads[order];
        let steer_l2 = norms::l2(steer_vec);
        let restore_l2 = norms::l2(restore_vec);
        let steer = InjectionSpec {
            after_block: RECEIVER_BLOCK,
            positions: positions.clone(),
            vector: steer_vec.clone(),
            scale: Some(natural.median / steer_l2),
            mode: INJECT_MODE,
        };
        // ADR-039: `random` is norm-matched to the EFFECTIVE STEER vector.
        let target_l2 = norms::l2(&steer.effective_vector());
        let restore = InjectionSpec {
            after_block: RECEIVER_BLOCK,
            positions: positions.clone(),
            vector: restore_vec.clone(),
            scale: Some(natural.median / restore_l2),
            mode: INJECT_MODE,
        };
        let mut vrng = ChaCha8Rng::seed_from_u64(RANDVEC_SEED_BASE + item.index as u64);
        let gauss = common::gaussian_vec(&mut vrng, steer_vec.len());
        let random = InjectionSpec {
            after_block: RECEIVER_BLOCK,
            positions: positions.clone(),
            vector: gauss.clone(),
            scale: Some(target_l2 / norms::l2(&gauss)),
            mode: INJECT_MODE,
        };
        let zerovec = InjectionSpec {
            after_block: RECEIVER_BLOCK,
            positions,
            vector: vec![0f32; steer_vec.len()],
            scale: None,
            mode: INJECT_MODE,
        };

        // --- the five paired conditions ------------------------------------
        let gold_toks = receiver.encode(&format!("#### {}", item.gold)).map_err(e)?;
        let decoy_toks = receiver.encode(&format!("#### {decoy_s}")).map_err(e)?;
        let mut outcome = |spec: Option<&InjectionSpec>| -> anyhow::Result<Cond> {
            let mut s = Sampler::new(Sampling::Greedy, 0);
            let out = receiver
                .generate(&inj_tokens, spec, &mut s, MAX_NEW_TOKENS, false)
                .map_err(e)?;
            let answer = common::extract_answer(&out.text);
            let correct = answer
                .as_deref()
                .is_some_and(|a| common::answers_equal(a, &item.gold));
            let emits_decoy = answer
                .as_deref()
                .is_some_and(|a| common::answers_equal(a, &decoy_s));
            let mut nll_of = |tgt: &[u32]| -> anyhow::Result<f32> {
                let toks: Vec<u32> = inj_tokens.iter().chain(tgt.iter()).copied().collect();
                teacher_forced_nll(
                    &mut receiver.model,
                    &toks,
                    inj_tokens.len()..toks.len(),
                    spec,
                    &device,
                )
                .map_err(e)
            };
            let nll_gold = nll_of(&gold_toks)?;
            let nll_decoy = nll_of(&decoy_toks)?;
            Ok(Cond {
                correct,
                emits_decoy,
                answer,
                nll_gold,
                nll_decoy,
                chars: out.text.chars().count(),
                text: out.text,
            })
        };
        let c_steer = outcome(Some(&steer))?;
        let c_restore = outcome(Some(&restore))?;
        let c_base = outcome(None)?;
        let c_zero = outcome(Some(&zerovec))?;
        let c_rand = outcome(Some(&random))?;

        // Mechanical-identity cross-check: `restore` replays PC1b's exact
        // payload through the same operator at the same site, so its accuracy
        // should reproduce PC1b's `aligned_real` item for item. Diagnostic.
        if let Some(prev) = pc1b_aligned[order] {
            restore_comparable += 1;
            if prev == c_restore.correct {
                restore_matches_pc1b += 1;
            }
        }

        // --- the registered wealth update: DECOY EMISSION, steer vs random --
        let discordant = c_steer.emits_decoy != c_rand.emits_decoy;
        let x = if discordant {
            Some(u8::from(c_steer.emits_decoy))
        } else {
            None
        };
        if discordant {
            if c_steer.emits_decoy {
                wins += 1;
            } else {
                losses += 1;
            }
            let xv = f64::from(x.unwrap());
            wealth *= 1.0 + LAMBDA * (xv - 0.5);
            max_wealth = max_wealth.max(wealth);
        }

        println!(
            "[{}/{N_MAX}] item {idx} gold {} decoy {decoy}: decoy-emit \
             steer={} restore={} base={} zero={} rand={} | correct \
             s={} r={} b={} z={} n={} | disc={discordant} W={wealth:.4} {:.0}s",
            order + 1,
            item.gold,
            c_steer.emits_decoy,
            c_restore.emits_decoy,
            c_base.emits_decoy,
            c_zero.emits_decoy,
            c_rand.emits_decoy,
            c_steer.correct,
            c_restore.correct,
            c_base.correct,
            c_zero.correct,
            c_rand.correct,
            t0.elapsed().as_secs_f32()
        );

        let cond_json = |c: &Cond| {
            serde_json::json!({
                "correct": c.correct, "emits_decoy": c.emits_decoy,
                "extracted_answer": c.answer, "nll_gold": c.nll_gold,
                "nll_decoy": c.nll_decoy, "generated_chars": c.chars,
            })
        };
        rows.push(serde_json::json!({
            "item": idx,
            "gold": item.gold,
            "decoy": decoy,
            "perturbation": commitments[order]["perturbation"],
            "capture": {
                "steer_l2_raw": steer_l2,
                "restore_l2_raw": restore_l2,
                "injected_l2": target_l2,
                "natural_inject_block_norms": natural,
                "injection_mode": INJECT_MODE.tag(),
            },
            "injection_site": {
                "site": SITE.tag(),
                "prompt_tokens": inj_tokens.len(),
                "positions": sp.positions,
                "position_token_ids": sp.position_token_ids,
                "positions_decoded": sp.positions_decoded,
            },
            "conditions": {
                "steer": cond_json(&c_steer),
                "restore": cond_json(&c_restore),
                "baseline_uninjected": cond_json(&c_base),
                "zerovec_injected": cond_json(&c_zero),
                "random": cond_json(&c_rand),
            },
            "steer_answer_tail": c_steer.text.chars().rev().take(60).collect::<String>()
                .chars().rev().collect::<String>(),
            "generated_chars": {
                "steer": c_steer.chars, "restore": c_restore.chars,
                "baseline_uninjected": c_base.chars, "zerovec_injected": c_zero.chars,
                "random": c_rand.chars,
            },
            "restore_reproduces_pc1b_aligned_real": pc1b_aligned[order].map(|p| p == c_restore.correct),
        }));

        trajectory.push(EStep {
            order: order + 1,
            item: idx,
            steer_emits_decoy: c_steer.emits_decoy,
            random_emits_decoy: c_rand.emits_decoy,
            discordant,
            x,
            wealth,
        });
        five.push(Five {
            steer: c_steer,
            restore: c_restore,
            base: c_base,
            zero: c_zero,
            rand: c_rand,
        });

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
    let n = five.len();
    let nf = n.max(1) as f64;

    // ---- THE LEAKAGE GATE — evaluated BEFORE any interpretation ----------
    let count = |f: &dyn Fn(&Five) -> bool| five.iter().filter(|q| f(q)).count();
    let leak_base = count(&|q| q.base.emits_decoy);
    let leak_rate = leak_base as f64 / nf;
    let leaky = leak_rate > LEAKAGE_VOID_RATE;
    println!(
        "\nLEAKAGE GATE: baseline emits the decoy on {leak_base}/{n} items = {:.4} \
         (void threshold {LEAKAGE_VOID_RATE}) => leaky={leaky}",
        leak_rate
    );

    // ---- Decoy-emission tallies (the primary endpoint's scale) ------------
    let d_steer = count(&|q| q.steer.emits_decoy);
    let d_restore = count(&|q| q.restore.emits_decoy);
    let d_zero = count(&|q| q.zero.emits_decoy);
    let d_rand = count(&|q| q.rand.emits_decoy);

    // ---- Accuracy (the PC1b endpoint), reported as secondary --------------
    let a_steer = count(&|q| q.steer.correct);
    let a_restore = count(&|q| q.restore.correct);
    let a_base = count(&|q| q.base.correct);
    let a_zero = count(&|q| q.zero.correct);
    let a_rand = count(&|q| q.rand.correct);
    let zerovec_pass = 2 * a_zero >= a_base;

    // ---- NLL diagnostics against EVERY control ---------------------------
    let mean = |f: &dyn Fn(&Five) -> f32| five.iter().map(f).sum::<f32>() / n.max(1) as f32;
    let sign = |a: &dyn Fn(&Five) -> f32, b: &dyn Fn(&Five) -> f32| {
        let w = five.iter().filter(|q| a(q) < b(q)).count();
        let l = five.iter().filter(|q| a(q) > b(q)).count();
        serde_json::json!({"wins": w, "losses": l,
            "p_one_sided": common::sign_test_one_sided(w, l),
            "mid_p_one_sided": common::mid_p_one_sided(w, l)})
    };
    // Steer's DECOY nll against every control — the likelihood analogue of the
    // primary endpoint.
    let nlld_sr = sign(&|q| q.steer.nll_decoy, &|q| q.rand.nll_decoy);
    let nlld_sb = sign(&|q| q.steer.nll_decoy, &|q| q.base.nll_decoy);
    let nlld_sz = sign(&|q| q.steer.nll_decoy, &|q| q.zero.nll_decoy);
    let nlld_st = sign(&|q| q.steer.nll_decoy, &|q| q.restore.nll_decoy);
    // Steer's GOLD nll against every control — PC1b's endpoint, for continuity.
    let nllg_sr = sign(&|q| q.steer.nll_gold, &|q| q.rand.nll_gold);
    let nllg_sb = sign(&|q| q.steer.nll_gold, &|q| q.base.nll_gold);
    let nllg_sz = sign(&|q| q.steer.nll_gold, &|q| q.zero.nll_gold);
    let nllg_st = sign(&|q| q.steer.nll_gold, &|q| q.restore.nll_gold);
    // Restore's gold nll vs random — PC1b's own validated likelihood result,
    // replayed here as a within-rung positive control on the LIKELIHOOD arm.
    let nllg_rr = sign(&|q| q.restore.nll_gold, &|q| q.rand.nll_gold);

    let mean_steer_nll_gold = mean(&|q| q.steer.nll_gold);
    let mean_steer_nll_decoy = mean(&|q| q.steer.nll_decoy);

    // ---- zerovec bit-identity (h += 0 must BE baseline) -------------------
    let noop_acc_disagreements = count(&|q| q.zero.correct != q.base.correct);
    let noop_decoy_disagreements = count(&|q| q.zero.emits_decoy != q.base.emits_decoy);
    let noop_text_identical = count(&|q| q.zero.text == q.base.text);
    let noop_max_abs_nll_delta = five
        .iter()
        .map(|q| (q.zero.nll_gold - q.base.nll_gold).abs())
        .fold(0f32, f32::max);
    let noop_max_abs_nll_decoy_delta = five
        .iter()
        .map(|q| (q.zero.nll_decoy - q.base.nll_decoy).abs())
        .fold(0f32, f32::max);
    let noop_exact_items = five
        .iter()
        .filter(|q| q.zero.nll_gold == q.base.nll_gold)
        .count();
    let noop_pass = noop_acc_disagreements == 0
        && noop_decoy_disagreements == 0
        && noop_max_abs_nll_delta <= FUSE_NOOP_TOL
        && noop_max_abs_nll_decoy_delta <= FUSE_NOOP_TOL;

    // ---- Gaming guard (docs/research/045 §3), thresholds UNCHANGED --------
    let ratios: Vec<f64> = five
        .iter()
        .map(|q| {
            if q.base.chars > 0 {
                q.steer.chars as f64 / q.base.chars as f64
            } else {
                1.0
            }
        })
        .collect();
    let degenerate_short_items = ratios.iter().filter(|&&x| x < DEGENERATE_LEN_RATIO).count();
    let mean_len_ratio = ratios.iter().sum::<f64>() / ratios.len().max(1) as f64;
    let degenerate_signature =
        (degenerate_short_items as f64) >= DEGENERATE_ITEM_FRACTION * n as f64;
    let nll_collapse_signature = mean_steer_nll_gold < NLL_COLLAPSE_NATS;
    let gaming_signature = degenerate_signature || nll_collapse_signature;
    // PC2-specific companion, reported alongside and NOT folded into the
    // gating signature so PC1b's guard stays byte-comparable.
    let decoy_nll_collapse_flag = mean_steer_nll_decoy < NLL_COLLAPSE_NATS;

    // ---- Registered verdict, ADR-039's branches, in ADR-039's order ------
    let verdict = if leaky {
        "VOID — LEAKY DECOY CONSTRUCTION. The baseline condition already emits the committed decoy above the 2% pre-registered ceiling, so a decoy emission under `steer` cannot be attributed to the injection. ADR-039 declares the rung void in this case. NO PASS/FAIL READING IS ISSUED and the pre-registered interpretation branches are NOT applied."
    } else if e_pass && !gaming_signature {
        "PASS — THE APPARATUS CAN STEER DECISIONS. `steer` moves the receiver's emitted answer toward the pre-committed decoy against a norm-matched random control at the same site, same operator, same 8 slots. Under ADR-039's registered interpretation PC1b's failure is re-read as restoration being the wrong probe, not as a dead pathway; M5X (ADR-037) and M4b (ADR-035) UNBLOCK. FIREWALL: this is a statement about the APPARATUS ONLY and may NEVER be cited as transfer evidence."
    } else if e_pass && gaming_signature {
        "GAMED PASS — the e-process crossed, but a pre-committed degenerate-output / NLL-collapse signature is present; docs/research/045 §3 requires this be reported as gamed, not clean."
    } else if n_disc == 0 {
        "UNINFORMATIVE — ZERO DISCORDANT PAIRS. The e-process never updated, so the wealth process carries no information about the endpoint. Per ADR-039 this is reported as uninformative and is NOT spun as either outcome; the power floor is stated on its own scale (ADR-036 Decision 3)."
    } else {
        "FAIL — THE APPARATUS CANNOT MOVE A DECISION BY ANY MEANS, not even toward an answer it was explicitly handed. Combined with PC1b this is decisive about the METHOD: this injection paradigm is DECISION-INERT while remaining LIKELIHOOD-LIVE. Consequences accepted in advance (ADR-039): M5X and M4b stay blocked permanently under this apparatus; the ladder closes; every cross-model null is explained by the method and none of them is evidence about latent transferability. Reported without softening (ADR-032). NOTE: this FAIL may NOT be cited as transfer evidence either (ADR-039's symmetric firewall rule), and PC1b's inverted FAIL wording is NOT inherited."
    };

    let receipt = serde_json::json!({
        "stage": "run2-PC2-steering-control-probe",
        "design": "docs/adr/039-pc2-steering-control-pre-registration.md. Payload = PC1b's derivation with the final numeric answer replaced by a pre-committed decoy; SITE, OPERATOR, SLOTS, STREAM = PC1b's, unchanged; statistic = ADR-036's e-process, primary endpoint DECOY-EMISSION RATE (steer vs random, paired per item).",
        "env": common::env_info(&nvcc),
        "pre_committed": true,
        "variant": "pc2-identity-L19lasttoken-decoytf-fuse-questiontail-slots8-eprocess",

        "FIREWALL_apparatus_not_transfer": {
            "verbatim_adr_039": "PC2 is same-model, same-item, identity-transform with gold-adjacent content. It tests the apparatus, never transfer. A PASS proves the pathway can steer a decision; it says NOTHING about whether a cross-model, learned-alignment payload can carry reasoning. A FAIL may not be cited as transfer evidence either — the symmetric rule added after PC1b.",
            "rule": "NEITHER branch of this rung is evidence about transfer.",
        },

        "DO_NOT_INHERIT_PC1B_FAIL_WORDING": {
            "quarantined_text": "PC1b's registered FAIL branch said a powered failure would make the ladder nulls 'evidence about TRANSFER rather than about plumbing.'",
            "why_it_is_wrong": "That is logically inverted (quarantined at the head of ADR-024). A failed positive control makes the METHOD the leading explanation for every null; ruling plumbing out requires a PASS on the endpoint being measured.",
            "this_receipt": "uses ADR-039's branches, which are written to avoid re-inheriting it.",
        },

        "pre_registered_interpretation_recorded_before_the_draw": {
            "pass": "the apparatus CAN steer decisions. PC1b's failure is re-read as restoration being the wrong probe, not as a dead pathway. M5X (ADR-037) and M4b (ADR-035) UNBLOCK, and the ladder's cross-model nulls become evidence about TRANSFER — because the method has finally been shown capable on the endpoint every verdict uses.",
            "fail_with_real_power": "the apparatus cannot move a decision by any means. Combined with PC1b that is decisive about the METHOD: decision-inert while likelihood-live. M5X and M4b stay blocked permanently; the ladder closes; this becomes the publishable result, reported without softening (ADR-032).",
            "underpowered": "reported as uninformative, on its own scale, NOT spun as either outcome (ADR-036 Decision 3).",
            "void": "if baseline already emits the decoy above 2%, the decoy construction is leaky and the rung is VOID — no interpretation branch is applied.",
        },

        "why_steering_and_not_restoration": {
            "pc1b_dissociation": "PC1b established a DISSOCIATION, not an absence. LIKELIHOOD endpoint VALIDATED: the payload beat a norm-matched Gaussian through the same operator at the same 8 positions by 0.237 nats on 198/300 items (p ~ 2e-8). ACCURACY endpoint UNVALIDATED: accuracy went 140 -> 127, and no control in this repository has ever moved accuracy.",
            "what_restoration_cannot_separate": "'the pathway cannot steer decisions' from 'the receiver's errors are not of a kind that the gold state repairs'.",
            "why_steering_is_EASIER": [
                "it removes the confound restoration cannot: a decoy hit is not something the receiver would produce for independent reasons, and `random` hits d at chance by construction, giving a clean floor",
                "its effect ceiling is far higher: restoration is bounded by the ~160/300 items the receiver already gets wrong; steering can move ANY item",
                "it varies exactly one thing from PC1b — the answer value",
            ],
        },

        "the_one_changed_factor_vs_pc1b": {
            "changed": "the final numeric answer of the teacher-forced solution the payload is tapped from: '#### g' becomes '#### d'.",
            "held_identical": {
                "site": SITE.tag(),
                "operator": INJECT_MODE.tag(),
                "operator_equation": INJECT_MODE.equation(),
                "slots": N_SLOTS,
                "receiver_inject_block": RECEIVER_BLOCK,
                "rescale": "to the receiver's natural inject-block median L2, per item",
                "transform": "IDENTITY — no adapter artifact is constructed or applied anywhere in this binary",
                "stream": "adaptation-512 fixed order, 13-item exclusion, N_max 300; HARD-GATED equal to M4i's, PC1b's, the PC2 capture's and the decoy commitment's",
                "decoding": "greedy, batch=1, max_new_tokens=400",
            },
            "proved_by_the_capture": "run2_pc2_capture re-derives the GOLD payload for all 300 items through the same code path and gates on BIT-IDENTITY against PC1b's committed vectors. See run2-pc2-capture-receipt.json § the_one_changed_factor_vs_pc1b.gold_arm_bit_identity.",
            "and_by_this_probe": {
                "check": "the `restore` condition replays PC1b's own payload FILE byte-for-byte through this binary's operator at this binary's site, so its accuracy should reproduce PC1b's `aligned_real` item for item.",
                "items_comparable": restore_comparable,
                "items_agreeing": restore_matches_pc1b,
                "agreement_rate": restore_matches_pc1b as f64 / restore_comparable.max(1) as f64,
                "gating": "NONE — diagnostic. Greedy decoding is deterministic, so a high agreement rate is empirical evidence that PC2's mechanics are PC1b's mechanics; residual disagreement would indicate GPU-level nondeterminism, not a redefinition.",
            },
            "code_path_note": "the five-condition block is written in this binary rather than in common::m3::four_conditions_at, because that shared function returns neither the emitted answer (needed for the decoy endpoint) nor a fifth condition. It reuses the SAME primitives — common::m3::build_site_prompt for the site, InjectionSpec/InjectionMode::Fuse for the operator, forward_capture + norms::stats for the rescale target, common::gaussian_vec with RANDVEC_SEED_BASE for the random control, Sampler::Greedy and teacher_forced_nll for the outcomes — and the `restore` agreement check above is the empirical proof that the assembled block behaves as PC1b's did. Recorded as a real deviation rather than glossed.",
        },

        "LEAKAGE_GATE": {
            "adr_039_rule": "A separate check records the rate at which `baseline` already emits d (expected ~ 0); if that exceeds 2%, the decoy construction is declared leaky and the rung is void.",
            "threshold": LEAKAGE_VOID_RATE,
            "baseline_decoy_emissions": leak_base,
            "n_items": n,
            "rate": leak_rate,
            "leaky": leaky,
            "evaluated_before_interpretation": true,
            "also_for_reference": {
                "zerovec_decoy_emissions": d_zero,
                "random_decoy_emissions": d_rand,
                "note": "zerovec is h += 0 and must equal baseline; random is the registered chance floor for the primary endpoint.",
            },
        },

        "decoy_construction": {
            "commitment_receipt": COMMITMENT_RECEIPT,
            "committed_before_any_forward_pass": true,
            "seed_hex": commit["implementation"]["seed_hex"].clone(),
            "menu": commit["implementation"]["menu"].clone(),
            "menu_histogram": commit["implementation"]["menu_histogram"].clone(),
            "total_redraws": commit["implementation"]["total_redraws"].clone(),
            "per_item_triples": "run2-pc2-decoy-commitment-receipt.json § commitments — (stream_order, item, gold, decoy, perturbation, redraws) for all 300, written before any payload capture",
            "re_verified_in_this_probe": "every commitment row's gold is asserted equal to the dataset's gold for that item, and every decoy is asserted numerically distinct from it, before the item is drawn",
        },

        "e_process": {
            "protocol": "docs/adr/036-successor-rung-evaluation-protocol.md Decision 1 (ADR-030 §3.2's betting rule, adopted verbatim)",
            "PRIMARY_ENDPOINT": "DECOY-EMISSION RATE — whether the receiver's extracted answer equals the item's pre-committed decoy d. NOT accuracy. ADR-039 registers this endpoint.",
            "wealth_rule": "W_0 = 1; on each DISCORDANT pair W <- W * (1 + lambda * (x - 0.5)) where x = 1 if `steer` emits the decoy and `random` does not, x = 0 if the reverse. Concordant items produce no update.",
            "lambda": LAMBDA,
            "lambda_note": "lambda = 2*theta - 1 tuned to theta = 0.65. Fixed in advance; never re-parametrised after seeing W_t.",
            "alpha": E_ALPHA,
            "wealth_threshold": w_threshold,
            "n_max": N_MAX,
            "primary_comparison": "steer vs random, paired per item",
            "stopping_rule": "stop and PASS at the first item where W >= 1/alpha; otherwise consume the full N_max and report the final wealth.",
            "never_restarted": "ONE registered draw over the fixed stream, run once. No item re-drawn, no parameter re-tuned, no restart (ADR-032 honest-fail).",
            "items_drawn": n,
            "n_discordant": n_disc,
            "wins": wins,
            "losses": losses,
            "crossed_at_item": crossed_at,
            "final_wealth": wealth,
            "max_wealth_reached": max_wealth,
            "pass": e_pass,
            "no_p_value_translation": "The e-process outcome is reported on its OWN scale (crossing item count, or final wealth at N_max) and is NOT translated into an equivalent p-value (ADR-036 Decision 3). The NLL sign tests below are separate secondary diagnostics on a different endpoint.",
            "power_context": {
                "pc1b_n_disc_on_accuracy": 64,
                "m4i_n_disc_on_accuracy": 66,
                "note": "PC2's discordance is on a DIFFERENT endpoint (decoy emission), so neither figure predicts it; both are quoted only as this stream's precedent for discordance being available at all at this site.",
            },
            "full_trajectory": trajectory.iter().map(|s| serde_json::json!({
                "order": s.order, "item": s.item,
                "steer_emits_decoy": s.steer_emits_decoy,
                "random_emits_decoy": s.random_emits_decoy,
                "discordant": s.discordant, "x": s.x, "wealth": s.wealth,
            })).collect::<Vec<_>>(),
        },

        "site_provenance": {
            "resolver": "common::m3::build_site_prompt, reused verbatim",
            "site_tag": SITE.tag(),
            "site_description": SITE.description(),
            "m4i_receipt": M4I_RECEIPT,
            "m4i_stream_key_read": "items[*].item",
            "m4i_stream_key_note": "PC1b's probe read m4i['dataset']['indices'], a key M4i's receipt does NOT carry, and so recorded item_stream_identical_to_m4i: false for an identical stream. This probe reads the key that exists and HARD-GATES on equality — the binary aborts on mismatch.",
            "item_stream_identical_to_m4i": true,
            "samples": site_samples,
            "tokenization_excluded": tokenization_excluded,
        },

        "payload_provenance": {
            "steer": {"capture_receipt": CAPTURE_RECEIPT,
                       "file": steer_path.display().to_string(), "sha256": steer_sha,
                       "content": "receiver's OWN block-19 state, teacher-forced over its own solution WITH THE FINAL ANSWER REPLACED BY THE COMMITTED DECOY, last token of the span"},
            "restore": {"capture_receipt": PC1B_CAPTURE_RECEIPT,
                         "file": restore_path.display().to_string(), "sha256": restore_sha,
                         "content": "PC1b's OWN committed payload file, replayed byte-for-byte — not a re-derivation"},
        },

        "manifold_precheck": {
            "status": "NOT REGISTERED for this rung. ADR-039 registers no manifold pre-check, and the payload is a captured receiver state by construction (identity transform, no adapter), so the on-manifold classification PC1b's pre-check existed to establish holds trivially and by the same argument.",
            "gating": "none",
        },

        "item_supply": {
            "source": ADAPTATION_512,
            "source_split": "adaptation-512",
            "consumption_order": "the file's own fixed ascending index order (ADR-036 Decision 2(1))",
            "leakage_exclusions": LEAKAGE_EXCLUSIONS,
            "leakage_exclusions_present_in_split": excluded_present,
            "eligible_pool": eligible.len(),
            "must_travel_with_every_headline_number": "Per ADR-036 Decision 3, any write-up quoting a number from this receipt must state in the same sentence that it was produced under the e-process protocol on the adaptation-512 stream, on the DECOY-EMISSION endpoint. PC1's and every completed rung's numbers were produced under the frozen 40-item protocol on the ACCURACY endpoint and are NOT directly comparable.",
        },

        "dataset": {"source": common::GSM8K_TRAIN_URL, "sha256": train_sha, "indices": stream},
        "decoys": decoys,
        "items": rows,

        "summary": {
            "n_evaluated": n,
            "PRIMARY_decoy_emission_counts": {
                "steer": d_steer, "restore": d_restore, "baseline_uninjected": leak_base,
                "zerovec_injected": d_zero, "random": d_rand, "n": n,
            },
            "PRIMARY_decoy_emission_rates": {
                "steer": d_steer as f64 / nf, "restore": d_restore as f64 / nf,
                "baseline_uninjected": leak_rate, "zerovec_injected": d_zero as f64 / nf,
                "random": d_rand as f64 / nf,
            },
            "primary_e_process_steer_vs_random_on_decoy_emission": {
                "wins": wins, "losses": losses, "n_discordant": n_disc,
                "final_wealth": wealth, "max_wealth_reached": max_wealth,
                "threshold": w_threshold, "crossed_at_item": crossed_at, "pass": e_pass},
            "accuracy_secondary": {
                "steer": a_steer, "restore": a_restore, "baseline_uninjected": a_base,
                "zerovec_injected": a_zero, "random": a_rand,
                "note": "PC1b's endpoint, reported for continuity. It is NOT this rung's primary."},
            "steer_vs_restore_decoy_emission": {"both": count(&|q| q.steer.emits_decoy && q.restore.emits_decoy),
                "steer_only": count(&|q| q.steer.emits_decoy && !q.restore.emits_decoy),
                "restore_only": count(&|q| !q.steer.emits_decoy && q.restore.emits_decoy)},
            "nll_mean": {
                "gold_target": {"steer": mean_steer_nll_gold, "restore": mean(&|q| q.restore.nll_gold),
                    "baseline_uninjected": mean(&|q| q.base.nll_gold),
                    "zerovec_injected": mean(&|q| q.zero.nll_gold), "random": mean(&|q| q.rand.nll_gold)},
                "decoy_target": {"steer": mean_steer_nll_decoy, "restore": mean(&|q| q.restore.nll_decoy),
                    "baseline_uninjected": mean(&|q| q.base.nll_decoy),
                    "zerovec_injected": mean(&|q| q.zero.nll_decoy), "random": mean(&|q| q.rand.nll_decoy)},
            },
            "nll_sign_tests_secondary": {
                "note": "SECONDARY diagnostics on the same collected pairs, on the LIKELIHOOD endpoint. They are not the registered primary and carry no e-process interpretation.",
                "steer_decoy_nll_vs": {"random": nlld_sr, "baseline": nlld_sb,
                                        "zerovec": nlld_sz, "restore": nlld_st},
                "steer_gold_nll_vs": {"random": nllg_sr, "baseline": nllg_sb,
                                       "zerovec": nllg_sz, "restore": nllg_st},
                "restore_gold_nll_vs_random": nllg_rr,
                "restore_gold_nll_vs_random_note": "PC1b's own VALIDATED likelihood result (198W/102L, 0.237 nats), replayed here as a within-rung check that the likelihood arm is still live under this binary's mechanics.",
            },
            "zerovec_is_bit_identical_to_baseline": {
                "expected": "EXACT identity — under fuse the zerovec condition is h += 0",
                "accuracy_disagreements": noop_acc_disagreements,
                "decoy_emission_disagreements": noop_decoy_disagreements,
                "generated_text_identical_items": noop_text_identical,
                "nll_gold_bit_identical_items": noop_exact_items,
                "max_abs_nll_gold_delta": noop_max_abs_nll_delta,
                "max_abs_nll_decoy_delta": noop_max_abs_nll_decoy_delta,
                "tolerance": FUSE_NOOP_TOL,
                "n_items": n,
                "pass": noop_pass,
                "gating": "NONE — operator-correctness diagnostic, reported exactly as PC1b reported it"},
            "zerovec_not_catastrophic": {"criterion": "2 x zerovec_correct >= baseline_correct",
                "pass": zerovec_pass,
                "note": "trivially satisfied because zerovec == baseline under fuse; no evidential weight"},
            "gaming_guard_research_045_section_3": {
                "why": "ITI (arXiv:2306.03341) documents an intervention that appears to restore the answer while merely collapsing the model into a degenerate response. docs/research/045 §3 makes this check mandatory on EVERY positive-control draw.",
                "thresholds_pre_committed": {
                    "degenerate_len_ratio": DEGENERATE_LEN_RATIO,
                    "degenerate_item_fraction": DEGENERATE_ITEM_FRACTION,
                    "nll_collapse_nats": NLL_COLLAPSE_NATS,
                    "note": "PC1b's values, UNCHANGED, so the two rungs' guards are directly comparable"},
                "mean_steer_over_baseline_generated_char_ratio": mean_len_ratio,
                "degenerate_short_items": degenerate_short_items,
                "degenerate_output_signature": degenerate_signature,
                "mean_steer_nll_gold": mean_steer_nll_gold,
                "nll_collapse_signature": nll_collapse_signature,
                "gaming_signature": gaming_signature,
                "pc2_companion_flag_not_folded_into_the_gate": {
                    "mean_steer_nll_decoy": mean_steer_nll_decoy,
                    "decoy_nll_collapse_flag": decoy_nll_collapse_flag,
                    "why_separate": "PC2's payload encodes the DECOY, so a low decoy-NLL is the expected direction rather than a degeneracy signal. It is reported but deliberately NOT folded into the gating signature, so PC1b's guard stays byte-comparable."},
            },
        },

        "comparison_pc1b_vs_pc2": {
            "pc1b_receipt": PC1B_PROBE_RECEIPT,
            "note": "Both rungs ran the e-process on the same 300-item adaptation-512 stream at the same site with the same operator and slots. They differ in the PAYLOAD'S ANSWER VALUE and in the PRIMARY ENDPOINT (PC1b: accuracy; PC2: decoy emission). Accuracy columns are directly comparable; the primary statistics are NOT the same statistic and are not translated into each other.",
            "pc1b_summary": pc1b_probe["summary"].clone(),
            "pc1b_e_process": pc1b_probe["e_process"].clone(),
        },

        "gates": {
            "steer_payload_sha256_matches_capture_receipt": {"pass": true, "sha256": steer_sha},
            "restore_payload_sha256_matches_pc1b_capture_receipt": {"pass": true, "sha256": restore_sha},
            "capture_gold_arm_bit_identical_to_pc1b": {
                "pass": cap["gates"]["gold_arm_bit_identical_to_pc1b_all_items"]["pass"].as_bool() == Some(true),
                "note": "the one-changed-factor gate, enforced in the capture binary over all 300 items"},
            "decoys_committed_before_any_forward_pass": {"pass": true, "receipt": COMMITMENT_RECEIPT},
            "stream_hard_gated_identical_to_m4i": {"pass": true, "n": N_MAX, "read_from": "items[*].item"},
            "stream_identical_to_pc1b_and_to_the_capture_and_commitment": {"pass": true, "n": N_MAX},
            "identity_transform_no_adapter_weights_loaded": {"pass": true},
            "injection_mode_recorded": {"pass": true, "mode": INJECT_MODE.tag(),
                "equation": INJECT_MODE.equation()},
            "slot_count_unchanged": {"pass": true, "slots": N_SLOTS},
            "eprocess_never_restarted": {"pass": true},
            "LEAKAGE_decoy_not_already_emitted_by_baseline": {
                "pass": !leaky, "rate": leak_rate, "threshold": LEAKAGE_VOID_RATE,
                "note": "ADR-039: above the threshold the rung is VOID and no interpretation branch applies"},
            "PC2_steer_vs_random_decoy_emission_eprocess": {"pass": e_pass,
                "final_wealth": wealth, "threshold": w_threshold,
                "n_discordant": n_disc, "crossed_at_item": crossed_at},
            "zerovec_bit_identical_to_baseline": {"pass": noop_pass, "gating": "none — diagnostic"},
            "zerovec_not_catastrophic": {"pass": zerovec_pass, "degenerate_under_fuse": true},
            "no_gaming_signature": {"pass": !gaming_signature,
                "degenerate_output_signature": degenerate_signature,
                "nll_collapse_signature": nll_collapse_signature},
        },
        "gate_pass": !leaky && e_pass && zerovec_pass && !gaming_signature,
        "verdict": verdict,
        "honest_fail_contract": "ADR-032: ONE registered draw, no retry, full numbers reported either way. A FAILING positive control WITH REAL POWER is the MORE consequential outcome and is stated plainly here, not softened. An UNDERPOWERED outcome is reported as uninformative and is not spun as either branch.",
        "wall_clock_s": t0.elapsed().as_secs_f32(),
    });
    common::write_receipt(
        &crate_path("receipts"),
        "run2-pc2-receipt-identity-L19lasttoken-decoytf-fuse-questiontail-slots8-eprocess.json",
        &receipt,
    )?;

    println!("\n=== PC2 (steering control — APPARATUS ONLY, NEVER TRANSFER) ===");
    println!(
        "DECOY EMISSION  steer {d_steer}/{n}  restore {d_restore}/{n}  baseline {leak_base}/{n}  \
         zerovec {d_zero}/{n}  random {d_rand}/{n}"
    );
    println!(
        "LEAKAGE GATE: baseline rate {leak_rate:.4} vs threshold {LEAKAGE_VOID_RATE} => \
         leaky={leaky} (VOID if true)"
    );
    println!(
        "accuracy (secondary) steer {a_steer}/{n} restore {a_restore}/{n} baseline {a_base}/{n} \
         zerovec {a_zero}/{n} random {a_rand}/{n}"
    );
    println!(
        "e-process on decoy emission: wins {wins}, losses {losses}, n_discordant {n_disc}; \
         final W {wealth:.4} (max ever {max_wealth:.4}) vs threshold {w_threshold} => \
         pass={e_pass} (crossed_at={crossed_at:?})"
    );
    println!(
        "NLL means (decoy target): steer {mean_steer_nll_decoy:.4} restore {:.4} baseline {:.4} \
         zerovec {:.4} random {:.4}",
        mean(&|q| q.restore.nll_decoy),
        mean(&|q| q.base.nll_decoy),
        mean(&|q| q.zero.nll_decoy),
        mean(&|q| q.rand.nll_decoy)
    );
    println!(
        "NLL means (gold target):  steer {mean_steer_nll_gold:.4} restore {:.4} baseline {:.4} \
         zerovec {:.4} random {:.4}",
        mean(&|q| q.restore.nll_gold),
        mean(&|q| q.base.nll_gold),
        mean(&|q| q.zero.nll_gold),
        mean(&|q| q.rand.nll_gold)
    );
    println!(
        "restore reproduces PC1b aligned_real on {restore_matches_pc1b}/{restore_comparable} \
         comparable items"
    );
    println!(
        "zerovec == baseline: acc disagreements {noop_acc_disagreements}, decoy disagreements \
         {noop_decoy_disagreements}, identical texts {noop_text_identical}/{n}, max|dNLL| \
         {noop_max_abs_nll_delta:.3e} => pass={noop_pass}"
    );
    println!(
        "gaming guard: mean len ratio {mean_len_ratio:.3}, degenerate-short items \
         {degenerate_short_items}/{n} => degenerate={degenerate_signature}; \
         nll_collapse={nll_collapse_signature} => gaming_signature={gaming_signature}"
    );
    println!("VERDICT: {verdict}");
    Ok(())
}
