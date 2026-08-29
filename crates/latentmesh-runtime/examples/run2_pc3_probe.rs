//! Run-2 **PC3** — the out-of-sample decision-change confirmation.
//!
//! Registered by [`docs/adr/040-pc3-decision-change-endpoint-pre-registration.md`].
//!
//! # Why PC3 exists
//!
//! PC2 registered a **rare-event** primary — decoy emission, ~2% — which
//! yielded `n_disc = 3` and a minimum attainable one-sided p of **0.125**. It
//! could not have reached α = 0.05 on a perfect run, and it *was* perfect
//! (3 wins, 0 losses). **The registered primary was UNINFORMATIVE.**
//!
//! A post-hoc pass over PC2's own receipt then answered the question at full
//! power: asking *"did the injection change the answer **at all**"* instead of
//! *"did it emit the specific decoy"* gives `n_disc ≈ 77` on the same 300
//! items. **That analysis is post-hoc and may not stand as a registered
//! result.** PC3 pre-registers it and confirms it **out of sample**.
//!
//! # The registered primary endpoint
//!
//! Whether the receiver's extracted answer **differs from its own uninjected
//! baseline answer** — `steer` vs `random`, paired per item, under ADR-036's
//! e-process (λ = 0.30, α = 0.05, wealth threshold 20.0).
//!
//! **Secondary, retained for continuity**: decoy-emission rate, decoy-NLL,
//! gold-NLL.
//!
//! # ⚠️ Two deviations, inherited from the capture and restated here
//!
//! **(a) `N_max` is the remaining pool, not the registered 300.**
//! `adaptation-512` holds 512 eligible indices (**none** of ADR-024's 13
//! exclusions fall inside it); PC2/M4i consumed the first 300; **212 remain.**
//! ADR-040's "next 300 eligible items NOT used by PC2/M4i" does not exist.
//! This rung draws the entire remaining out-of-sample pool — the maximal draw
//! ADR-036 Decision 2 permits. At PC2's measured ~26% discordance the expected
//! `n_disc` still clears ADR-040's own 30-pair floor.
//!
//! **(b) `restore` is derived, not replayed.** No PC1b/PC2 artifact covers
//! these items, so PC2's byte-replay and its cross-run bit-identity gate are
//! unavailable. See the capture receipt's `RECORDED_DEVIATIONS` block.
//!
//! # PRE-REGISTERED INTERPRETATION (ADR-040), recorded before the draw
//!
//! - **PASS** — the apparatus moves decisions **semantically**. PC2's post-hoc
//!   null was a false negative; **M5X (ADR-037) and M4b (ADR-035) UNBLOCK.**
//! - **FAIL with real power** — confirms **out of sample** that decision-level
//!   change is **non-semantic**: injection perturbs ~half of all answers, and
//!   payload content contributes nothing a norm-matched Gaussian does not.
//!   Combined with the likelihood arm this establishes the dissociation as a
//!   registered, out-of-sample result: *the channel is semantic at the
//!   likelihood level and merely perturbative at the decision level.* M5X and
//!   M4b stay blocked permanently, the ladder closes, and this becomes the
//!   publishable finding (ADR-032), reported without softening.
//! - **Underpowered** (`n_disc < 30`) — the rung is reported as uninformative
//!   **and the power model itself is recorded as wrong**, which is a finding
//!   about our estimation, not about the apparatus.
//!
//! # ⛔ Inherited quarantines
//!
//! - PC1b's FAIL wording (*"nulls stand as evidence about TRANSFER"*) is
//!   **logically inverted**, quarantined at the head of ADR-024, and is not
//!   used anywhere in this binary.
//! - **The auto-generated verdict string below is NOT the verdict.** PC2's
//!   read *"the apparatus cannot move a decision by any means"* while its own
//!   data showed 3 wins / 0 losses and ~47% answer changes. **The coordinator
//!   adjudicates the verdict against the receipt**; this string is a
//!   mechanical branch label and says so in the receipt.
//!
//! # FIREWALL (ADR-040 § Firewall)
//!
//! PC3 is same-model, same-item, identity-transform. It tests **the
//! apparatus, never transfer.** **Neither a PASS nor a FAIL may be cited as
//! evidence for or against cross-model transfer.**
//!
//! Run: PATH=/usr/local/cuda-12.8/bin:$PATH \
//!      cargo run --release --features cuda --example run2_pc3_probe

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
const CAPTURE_RECEIPT: &str = "receipts/run2-pc3-capture-receipt.json";
const COMMITMENT_RECEIPT: &str = "receipts/run2-pc3-decoy-commitment-receipt.json";
/// PC2's probe receipt — the exclusion set PC3 must not touch, and the source
/// of the side-by-side comparison block.
const PC2_PROBE_RECEIPT: &str =
    "receipts/run2-pc2-receipt-identity-L19lasttoken-decoytf-fuse-questiontail-slots8-eprocess.json";
const M4I_RECEIPT: &str =
    "receipts/run2-m4i-receipt-cellL18toL14-mlp-pertokenlast-fuse-questiontail-slots8-eprocess.json";

const D_RECEIVER: usize = 1536;
/// `[L19_last_decoy (STEER) | L19_last_gold (RESTORE) | L14_pooled | L14_last]`.
const VECS_PER_ITEM: usize = 4;

/// Mechanics, unchanged from PC2.
const INJECT_MODE: InjectionMode = InjectionMode::Fuse;
const SITE: Site = Site::QuestionTail;

/// ADR-024's 13 probe-overlap items, verbatim from PC2's probe.
const LEAKAGE_EXCLUSIONS: [usize; 13] = [
    150, 1309, 1573, 2365, 2418, 2540, 2958, 2973, 3084, 3825, 3844, 3877, 4746,
];

// ---- ADR-036 Decision 1 / ADR-030 §3.2 e-process parameters, frozen -------
const LAMBDA: f64 = 0.30;
const E_ALPHA: f64 = 0.05;
/// ADR-040's registered ceiling. The out-of-sample pool cannot supply it.
const N_MAX_REGISTERED: usize = 300;
/// ADR-040: below this many discordant pairs the rung is UNINFORMATIVE **and
/// the power model is recorded as wrong**.
const N_DISC_FLOOR: usize = 30;
/// Tolerance for the fuse zero-payload no-op diagnostic (nats).
const FUSE_NOOP_TOL: f32 = 1e-6;

/// Baseline decoy-emission ceiling, retained from PC2 as a SECONDARY-endpoint
/// diagnostic. **It no longer voids the rung**: ADR-040 moves decoy emission
/// off the primary, so a leaky decoy construction can no longer contaminate
/// the registered endpoint. Recorded, not gating.
const LEAKAGE_REFERENCE_RATE: f64 = 0.02;

// --- Pre-committed gaming-guard thresholds (docs/research/045 §3) ----------
// PC1b's and PC2's values, UNCHANGED, so all three rungs' guards compare.
const DEGENERATE_LEN_RATIO: f64 = 0.25;
const DEGENERATE_ITEM_FRACTION: f64 = 0.25;
const NLL_COLLAPSE_NATS: f32 = 0.10;

fn crate_path(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

/// **THE REGISTERED PRIMARY COMPARISON**, pre-committed here so its exact
/// semantics are auditable rather than implicit in a `!=`.
///
/// "The receiver's extracted answer differs from its own uninjected baseline
/// answer" (ADR-040 § Decision). Numeric normalisation is applied — `"2.0"`
/// and `"2"` are the *same* answer and must not count as a change — and a
/// missing extraction on exactly one side **is** a change.
fn answer_differs(a: Option<&str>, b: Option<&str>) -> bool {
    match (a, b) {
        (None, None) => false,
        (Some(x), Some(y)) => !common::answers_equal(x, y),
        _ => true,
    }
}

/// One condition's measured outcome.
#[derive(Clone)]
struct Cond {
    /// The receiver emitted the item's GOLD answer.
    correct: bool,
    /// The receiver emitted the item's committed DECOY — PC2's primary,
    /// retained here as a SECONDARY endpoint.
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

impl Five {
    /// Whether condition `c`'s answer differs from **this item's own**
    /// uninjected baseline answer — the registered primary quantity.
    fn changed(&self, c: &Cond) -> bool {
        answer_differs(c.answer.as_deref(), self.base.answer.as_deref())
    }
}

/// One step of the registered wealth process (primary = answer change).
struct EStep {
    order: usize,
    item: usize,
    steer_changed_answer: bool,
    random_changed_answer: bool,
    discordant: bool,
    x: Option<u8>,
    wealth: f64,
}

/// The two injectable arms carried by PC3's own payload file, per item:
/// `(steer, restore)`. PC2 read only one arm from its file and took `restore`
/// from PC1b's — see deviation (b).
type PayloadArms = (Vec<Vec<f32>>, Vec<Vec<f32>>);

fn read_payloads(path: &Path, want_sha: &str, n: usize) -> anyhow::Result<PayloadArms> {
    let bytes = std::fs::read(path)?;
    let sha = common::sha256_hex(&bytes);
    anyhow::ensure!(
        sha == want_sha,
        "payload sha256 {sha} != receipt-pinned {want_sha} for {}",
        path.display()
    );
    anyhow::ensure!(
        bytes.len() == n * VECS_PER_ITEM * D_RECEIVER * 4,
        "payload file size mismatch for {}",
        path.display()
    );
    let flat: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let mut steer = Vec::with_capacity(n);
    let mut restore = Vec::with_capacity(n);
    for i in 0..n {
        let b = i * VECS_PER_ITEM * D_RECEIVER;
        steer.push(flat[b..b + D_RECEIVER].to_vec());
        restore.push(flat[b + D_RECEIVER..b + 2 * D_RECEIVER].to_vec());
    }
    Ok((steer, restore))
}

/// Read a committed stream out of a receipt, trying `dataset.indices` first
/// and falling back to `items[*].item` — see the capture binary's note on the
/// false negative PC1b recorded by reading only the first key.
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
    let w_threshold = 1.0 / E_ALPHA;

    // ---- Gate 1: PC3's capture receipt and its sha-pinned payload ---------
    let cap: serde_json::Value =
        serde_json::from_slice(&std::fs::read(crate_path(CAPTURE_RECEIPT)).map_err(|err| {
            anyhow::anyhow!(
                "capture receipt {CAPTURE_RECEIPT} unreadable ({err}) — run run2_pc3_capture first"
            )
        })?)?;
    for gate in [
        "OUT_OF_SAMPLE_intersection_with_pc2_m4i_is_empty",
        "pc2_and_m4i_streams_agree",
        "decoys_committed_before_any_forward_pass",
        "no_decoy_equals_its_gold",
        "all_decoys_positive",
        "decoy_edit_confined_to_the_final_answer_token",
        "identity_transform_no_adapter_weights_loaded",
        "all_captured_states_finite",
    ] {
        anyhow::ensure!(
            cap["gates"][gate]["pass"].as_bool() == Some(true),
            "capture receipt gate {gate} did not pass"
        );
    }
    let n_captured = cap["payload_file"]["n_items"]
        .as_u64()
        .ok_or_else(|| anyhow::anyhow!("capture receipt does not record n_items"))?
        as usize;
    let payload_path = PathBuf::from(
        cap["payload_file"]["path"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("capture receipt does not name the payload file"))?,
    );
    let payload_sha = cap["payload_file"]["sha256"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("capture receipt does not pin the payload sha256"))?
        .to_string();
    let (steer_payloads, restore_payloads) =
        read_payloads(&payload_path, &payload_sha, n_captured)?;
    println!(
        "PAYLOAD VERIFIED: sha {payload_sha} ({n_captured} x {VECS_PER_ITEM} x {D_RECEIVER} f32, \
         identity transform); steer and restore both carried in-file"
    );

    // ---- Gate 2: the decoy commitment, read back independently ------------
    let commit: serde_json::Value = serde_json::from_slice(
        &std::fs::read(crate_path(COMMITMENT_RECEIPT)).map_err(|err| {
            anyhow::anyhow!("decoy commitment receipt {COMMITMENT_RECEIPT} unreadable ({err})")
        })?,
    )?;
    let commitments = commit["commitments"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("commitment receipt has no commitments array"))?;
    anyhow::ensure!(
        commitments.len() == n_captured,
        "commitment receipt covers {} items, expected {n_captured}",
        commitments.len()
    );

    // ---- Item supply: re-derived independently of the capture -------------
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

    // ---- THE OUT-OF-SAMPLE GATE, re-evaluated here, not trusted -----------
    let pc2: serde_json::Value =
        serde_json::from_slice(&std::fs::read(crate_path(PC2_PROBE_RECEIPT))?)?;
    let pc2_stream = committed_stream(&pc2);
    anyhow::ensure!(
        pc2_stream.len() == N_MAX_REGISTERED,
        "PC2's committed stream has {} items, expected {N_MAX_REGISTERED}",
        pc2_stream.len()
    );
    let m4i: serde_json::Value = serde_json::from_slice(&std::fs::read(crate_path(M4I_RECEIPT))?)?;
    let m4i_stream = committed_stream(&m4i);
    anyhow::ensure!(
        m4i_stream == pc2_stream,
        "M4i's and PC2's committed streams differ — PC3's exclusion set is ambiguous"
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

    // ---- Model: receiver ONLY (PC3 is a same-model self-pair) -------------
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
    for &idx in &remaining {
        if stream.len() == n_captured {
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
        stream.len() == n_captured,
        "pre-flight resolved {} items but the capture covers {n_captured}",
        stream.len()
    );

    // HARD GATE: not one item of PC3's stream was used by PC2 or M4i.
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

    // The capture and the commitment must have walked EXACTLY this stream, or
    // row i is not item i in one of the artifacts.
    let cap_stream = committed_stream(&cap);
    anyhow::ensure!(
        stream == cap_stream,
        "the PC3 capture receipt's stream differs from this probe's independently-derived one"
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
    println!(
        "site pre-flight: {n_captured} items at {N_SLOTS} question-tail positions, {} excluded on \
         tokenisation grounds; OUT-OF-SAMPLE GATE PASSED — intersection with PC2/M4i's committed \
         300-item stream is EMPTY; stream gated identical to the capture's and the commitment's",
        tokenization_excluded.len()
    );
    for s in &site_samples {
        println!(
            "  item {} -> positions {} decode to {}",
            s["item"], s["positions"], s["positions_decoded"]
        );
    }
    if n_captured < N_MAX_REGISTERED {
        println!(
            "⚠️  DEVIATION (a): ADR-040 registers N_max = {N_MAX_REGISTERED}; the out-of-sample \
             pool holds {}. Drawing the entire remaining pool. See the receipt's \
             RECORDED_DEVIATIONS block.",
            remaining.len()
        );
    }

    // ---- THE DRAW: ADR-036 e-process on ANSWER CHANGE ---------------------
    let mut wealth = 1.0f64;
    let mut max_wealth = 1.0f64;
    let mut min_wealth = 1.0f64;
    let mut trajectory: Vec<EStep> = Vec::new();
    let mut rows: Vec<serde_json::Value> = Vec::new();
    let mut five: Vec<Five> = Vec::new();
    let mut decoys: Vec<i64> = Vec::new();
    let mut wins = 0usize;
    let mut losses = 0usize;
    let mut crossed_at: Option<usize> = None;

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

        // --- site prompt + rescale target (PC2's statements, unchanged) ----
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
        // `random` is norm-matched to the EFFECTIVE STEER vector — PC2's rule.
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

        // --- the five paired conditions -----------------------------------
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

        let q = Five {
            steer: c_steer,
            restore: c_restore,
            base: c_base,
            zero: c_zero,
            rand: c_rand,
        };

        // --- THE REGISTERED WEALTH UPDATE: ANSWER CHANGE, steer vs random --
        let steer_changed = q.changed(&q.steer);
        let random_changed = q.changed(&q.rand);
        let discordant = steer_changed != random_changed;
        let x = if discordant {
            Some(u8::from(steer_changed))
        } else {
            None
        };
        if discordant {
            if steer_changed {
                wins += 1;
            } else {
                losses += 1;
            }
            let xv = f64::from(x.unwrap());
            wealth *= 1.0 + LAMBDA * (xv - 0.5);
            max_wealth = max_wealth.max(wealth);
            min_wealth = min_wealth.min(wealth);
        }

        println!(
            "[{}/{n_captured}] item {idx} gold {} decoy {decoy}: CHANGED steer={steer_changed} \
             restore={} zero={} rand={random_changed} | decoy-emit s={} b={} r={} | correct \
             s={} b={} r={} | disc={discordant} W={wealth:.4} {:.0}s",
            order + 1,
            item.gold,
            q.changed(&q.restore),
            q.changed(&q.zero),
            q.steer.emits_decoy,
            q.base.emits_decoy,
            q.rand.emits_decoy,
            q.steer.correct,
            q.base.correct,
            q.rand.correct,
            t0.elapsed().as_secs_f32()
        );

        let cond_json = |c: &Cond| {
            serde_json::json!({
                "correct": c.correct, "emits_decoy": c.emits_decoy,
                "extracted_answer": c.answer,
                "answer_differs_from_own_baseline": answer_differs(
                    c.answer.as_deref(), q.base.answer.as_deref()),
                "nll_gold": c.nll_gold,
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
                "steer": cond_json(&q.steer),
                "restore": cond_json(&q.restore),
                "baseline_uninjected": cond_json(&q.base),
                "zerovec_injected": cond_json(&q.zero),
                "random": cond_json(&q.rand),
            },
            "PRIMARY_answer_change_vs_own_baseline": {
                "steer": steer_changed, "random": random_changed,
                "restore": q.changed(&q.restore), "zerovec": q.changed(&q.zero),
                "discordant": discordant, "x": x,
            },
            "steer_answer_tail": q.steer.text.chars().rev().take(60).collect::<String>()
                .chars().rev().collect::<String>(),
            "generated_chars": {
                "steer": q.steer.chars, "restore": q.restore.chars,
                "baseline_uninjected": q.base.chars, "zerovec_injected": q.zero.chars,
                "random": q.rand.chars,
            },
        }));

        trajectory.push(EStep {
            order: order + 1,
            item: idx,
            steer_changed_answer: steer_changed,
            random_changed_answer: random_changed,
            discordant,
            x,
            wealth,
        });
        five.push(q);

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
    let underpowered = n_disc < N_DISC_FLOOR;

    let count = |f: &dyn Fn(&Five) -> bool| five.iter().filter(|q| f(q)).count();

    // ---- PRIMARY: answer-change counts, every condition -------------------
    let ch_steer = count(&|q| q.changed(&q.steer));
    let ch_restore = count(&|q| q.changed(&q.restore));
    let ch_zero = count(&|q| q.changed(&q.zero));
    let ch_rand = count(&|q| q.changed(&q.rand));

    // ---- SECONDARY: decoy emission (PC2's primary), retained --------------
    let leak_base = count(&|q| q.base.emits_decoy);
    let leak_rate = leak_base as f64 / nf;
    let d_steer = count(&|q| q.steer.emits_decoy);
    let d_restore = count(&|q| q.restore.emits_decoy);
    let d_zero = count(&|q| q.zero.emits_decoy);
    let d_rand = count(&|q| q.rand.emits_decoy);
    // PC2's e-process, recomputed on this stream purely for continuity. It is
    // NOT this rung's registered statistic and gates nothing.
    let (mut sec_w, mut sec_wins, mut sec_losses) = (1.0f64, 0usize, 0usize);
    for q in &five {
        if q.steer.emits_decoy != q.rand.emits_decoy {
            if q.steer.emits_decoy {
                sec_wins += 1;
            } else {
                sec_losses += 1;
            }
            sec_w *= 1.0 + LAMBDA * (f64::from(u8::from(q.steer.emits_decoy)) - 0.5);
        }
    }

    // ---- Accuracy (PC1b's endpoint), reported as secondary ----------------
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
            "mean_delta_nats": mean(a) - mean(b),
            "p_one_sided": common::sign_test_one_sided(w, l),
            "mid_p_one_sided": common::mid_p_one_sided(w, l)})
    };
    // Steer's DECOY nll against every control.
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
    // replayed as a within-rung positive control on the LIKELIHOOD arm.
    let nllg_rr = sign(&|q| q.restore.nll_gold, &|q| q.rand.nll_gold);

    let mean_steer_nll_gold = mean(&|q| q.steer.nll_gold);
    let mean_steer_nll_decoy = mean(&|q| q.steer.nll_decoy);

    // ---- zerovec bit-identity (h += 0 must BE baseline) -------------------
    let noop_acc_disagreements = count(&|q| q.zero.correct != q.base.correct);
    let noop_decoy_disagreements = count(&|q| q.zero.emits_decoy != q.base.emits_decoy);
    let noop_answer_changes = ch_zero;
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
        && noop_answer_changes == 0
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
    let decoy_nll_collapse_flag = mean_steer_nll_decoy < NLL_COLLAPSE_NATS;

    // ---- MECHANICAL branch label — NOT the adjudicated verdict ------------
    // ADR-040 § Inherited quarantines: "Do NOT let a probe binary's
    // auto-generated verdict string stand as the verdict." PC2's said "the
    // apparatus cannot move a decision by any means" while its own data showed
    // 3 wins / 0 losses. This string is a branch label; the coordinator
    // adjudicates against the numbers.
    let branch = if e_pass && !gaming_signature {
        "PASS_BRANCH"
    } else if e_pass && gaming_signature {
        "GAMED_PASS_BRANCH"
    } else if underpowered {
        "UNDERPOWERED_BRANCH"
    } else {
        "FAIL_WITH_REAL_POWER_BRANCH"
    };
    let branch_text = match branch {
        "PASS_BRANCH" => "PASS BRANCH — the e-process crossed the wealth boundary on the registered primary (answer differs from own uninjected baseline), steer vs a norm-matched random control at the same site, same operator, same 8 slots, on items PC2 never saw. Under ADR-040 this reads as: the apparatus moves decisions SEMANTICALLY; PC2's post-hoc null was a false negative; M5X (ADR-037) and M4b (ADR-035) UNBLOCK. FIREWALL: apparatus only, NEVER transfer, in either direction.",
        "GAMED_PASS_BRANCH" => "GAMED PASS BRANCH — the e-process crossed, but a pre-committed degenerate-output / NLL-collapse signature is present; docs/research/045 §3 requires this be reported as gamed, not clean.",
        "UNDERPOWERED_BRANCH" => "UNDERPOWERED BRANCH — n_disc landed below ADR-040's pre-registered floor of 30. ADR-040 states this 'should be impossible' given its own power calculation, and registers in advance that if it happens the rung is reported as uninformative AND THE POWER MODEL ITSELF IS RECORDED AS WRONG — a finding about our estimation, not about the apparatus. No PASS/FAIL reading is issued.",
        _ => "FAIL-WITH-REAL-POWER BRANCH — the e-process did not cross on the registered primary, at a discordant-pair count at or above ADR-040's pre-registered floor. Under ADR-040 this CONFIRMS OUT OF SAMPLE that decision-level change is NON-SEMANTIC: injection perturbs a large fraction of answers, and the payload's CONTENT contributes nothing a norm-matched Gaussian does not. Combined with the likelihood arm, the registered reading is: THE CHANNEL IS SEMANTIC AT THE LIKELIHOOD LEVEL AND MERELY PERTURBATIVE AT THE DECISION LEVEL. Consequences accepted in advance: M5X and M4b stay blocked permanently under this apparatus; the ladder closes; reported without softening (ADR-032). FIREWALL: this FAIL is NOT transfer evidence, and PC1b's inverted FAIL wording is NOT inherited.",
    };

    let receipt = serde_json::json!({
        "stage": "run2-PC3-decision-change-out-of-sample-probe",
        "design": "docs/adr/040-pc3-decision-change-endpoint-pre-registration.md. PC2's apparatus, two things changed: (1) the item stream is the OUT-OF-SAMPLE tail of adaptation-512, gated to share not one item with PC2/M4i; (2) the registered primary endpoint is ANSWER-DIFFERS-FROM-OWN-UNINJECTED-BASELINE, steer vs random, paired per item, under ADR-036's e-process.",
        "env": common::env_info(&nvcc),
        "pre_committed": true,
        "variant": "pc3-identity-L19lasttoken-decoytf-fuse-questiontail-slots8-eprocess-answerchange-oos",

        "THE_VERDICT_STRING_BELOW_IS_NOT_THE_VERDICT": {
            "adr_040_quarantine_verbatim": "Do NOT let a probe binary's auto-generated verdict string stand as the verdict. PC2's said 'the apparatus cannot move a decision by any means' while its own data showed 3 wins / 0 losses and ~47% answer changes. The coordinator adjudicates the verdict against the receipt.",
            "what_this_field_is": "a MECHANICAL BRANCH LABEL selected by the registered decision rule from (e_pass, gaming_signature, n_disc >= floor). It is not an interpretation and must not be quoted as one.",
            "branch_selected": branch,
            "adjudicate_against": ["summary.PRIMARY_answer_change_counts", "summary.primary_e_process_steer_vs_random_on_answer_change", "e_process.full_trajectory", "summary.zerovec_is_bit_identical_to_baseline"],
        },

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
                "what_was_done_instead": "the ENTIRE remaining out-of-sample pool was drawn — the maximal draw ADR-036 Decision 2's adaptation-512-only rule permits. Its stratified-resample escalation into a further tranche is explicitly 'named as an available secondary option, not built, not scheduled'.",
                "power_consequence_REALISED_not_predicted": {
                    "n_disc_floor_adr_040": N_DISC_FLOOR,
                    "n_disc_realised": n_disc,
                    "underpowered": underpowered,
                },
                "not_a_protocol_shop": "the endpoint, lambda, alpha, wealth threshold, direction, site, operator, slots and decoy construction are exactly as ADR-040 froze them. Only the attainable N changed, and it changed because of arithmetic fixed long before this rung existed.",
            },
            "b_restore_is_derived_not_replayed": {
                "pc2_did": "replayed PC1b's committed payload FILE byte-for-byte as the `restore` arm, and gated its own gold arm bit-identical against that file over all 300 items.",
                "pc3_does": "derives the gold payload through the SAME tap() call as the decoy payload, with the continuation text as the only differing argument.",
                "why": "no PC1b or PC2 artifact covers these out-of-sample items.",
                "what_is_lost": "the cross-run bit-identity check. PC3 cannot prove its derivation has not drifted from PC1b's, because there is nothing to compare against on these items. Stated, not gated away.",
                "what_is_retained": "render_decoy() asserts per item that the two continuation texts differ ONLY in the whitespace-delimited token after the final '####', and both arms run the same tap at the same blocks.",
            },
        },

        "pre_registered_interpretation_recorded_before_the_draw": {
            "pass": "the apparatus moves decisions SEMANTICALLY. PC2's post-hoc null was a false negative; M5X (ADR-037) and M4b (ADR-035) UNBLOCK.",
            "fail_with_real_power": "CONFIRMS OUT OF SAMPLE that decision-level change is non-semantic: injection perturbs ~half of all answers, and payload content contributes nothing a norm-matched Gaussian does not. Combined with the likelihood arm's p = 2.6e-44 this establishes the dissociation as a registered, out-of-sample result: THE CHANNEL IS SEMANTIC AT THE LIKELIHOOD LEVEL AND MERELY PERTURBATIVE AT THE DECISION LEVEL. M5X and M4b stay blocked permanently; the ladder closes; reported without softening (ADR-032).",
            "underpowered": "given ADR-040's own power calculation this should be impossible; if n_disc lands below 30 the rung is reported as uninformative AND THE POWER MODEL ITSELF IS RECORDED AS WRONG — a finding about our estimation, not about the apparatus.",
        },

        "why_this_endpoint_replaces_pc2s": {
            "pc2_primary_was_a_rare_event": "decoy emission ran at ~2%, giving n_disc = 3 and a minimum attainable one-sided p of 0.125. It could not have reached alpha = 0.05 on a perfect run — and it WAS perfect (3 wins, 0 losses). The registered primary was UNINFORMATIVE.",
            "adr_040_mandatory_power_calculation": {
                "note": "ADR-040 makes a power calculation a standing requirement for every future rung, fixing coordinator error #14 — registering an endpoint without computing whether it could ever detect anything.",
                "steer_changes_answer_pc2": "46.7% (140/300 per ADR-040's table)",
                "random_changes_answer_pc2": "50.3% (151/300 per ADR-040's table)",
                "observed_discordance_rate_pc2": "25.7% (77/300 paired)",
                "expected_n_disc_at_300": 77,
                "minimum_attainable_p_at_300": "6.6e-24 (2^-77)",
                "n_for_n_disc_30_floor": 117,
                "endpoint_this_replaces": "decoy emission — expected n_disc ~4.5 and a minimum attainable p of 0.043, needing ~4,224 items for PC1b-level power.",
            },
            "out_of_sample_is_the_point": "ADR-040 §Context: the post-hoc re-analysis of PC2's receipt 'may not stand as a registered result. PC3 exists to pre-register it and confirm it out of sample.' Reusing PC2's items would have defeated the entire purpose of the rung, which is why the empty-intersection gate is HARD.",
        },

        "why_decoys_stay_in_natural_slip_space": {
            "adr_040_REVERSAL": "ADR-039's closing note said a successor must draw decoys AWAY from natural-slip space to recover a genuine chance floor. ADR-040 reverses that: (1) the clean-floor argument applied to the decoy-emission endpoint, which is no longer primary and is moot here; (2) moving decoys away from plausible answers would lower the baseline rate AND lower steer with it — a model that never spontaneously emits an unrelated integer will not emit one when nudged either — making power WORSE; (3) natural-slip decoys are the STRONGER test of steering, being answers the model would plausibly accept, so failing to steer toward them is more damning.",
            "construction_unchanged_from_pc2": "ChaCha8 seed 0x5732 over {g+1, g-1, g+10, 2g}, committed before the draw.",
        },

        "e_process": {
            "protocol": "docs/adr/036-successor-rung-evaluation-protocol.md Decision 1 (ADR-030 §3.2's betting rule, adopted verbatim)",
            "PRIMARY_ENDPOINT": "ANSWER CHANGE — whether the receiver's extracted answer DIFFERS FROM ITS OWN UNINJECTED BASELINE ANSWER. NOT decoy emission (PC2's, now secondary), NOT accuracy (PC1b's, now secondary). ADR-040 registers this endpoint.",
            "endpoint_comparison_semantics_pre_committed": {
                "function": "answer_differs(a, b)",
                "rule": "(None, None) => false; (Some, None) or (None, Some) => TRUE (a missing extraction on exactly one side IS a change); (Some(x), Some(y)) => !answers_equal(x, y), i.e. NUMERIC normalisation is applied so '2.0' and '2' are the SAME answer and do not count as a change.",
                "why_recorded": "the endpoint is a comparison, not a measurement, so its exact semantics are part of the pre-registration and are stated rather than left implicit in a `!=`.",
            },
            "wealth_rule": "W_0 = 1; on each DISCORDANT pair W <- W * (1 + lambda * (x - 0.5)) where x = 1 if `steer` changed the answer and `random` did not, x = 0 if the reverse. Concordant items produce no update.",
            "lambda": LAMBDA,
            "lambda_note": "lambda = 2*theta - 1 tuned to theta = 0.65. Fixed in advance; never re-parametrised after seeing W_t.",
            "alpha": E_ALPHA,
            "wealth_threshold": w_threshold,
            "n_max_registered": N_MAX_REGISTERED,
            "n_max_attainable": remaining.len(),
            "primary_comparison": "steer vs random, paired per item",
            "stopping_rule": "stop and PASS at the first item where W >= 1/alpha; otherwise consume the full attainable pool and report the final wealth.",
            "never_restarted": "ONE registered draw over the fixed out-of-sample stream, run once. No item re-drawn, no parameter re-tuned, no restart (ADR-032 honest-fail).",
            "items_drawn": n,
            "n_discordant": n_disc,
            "n_discordant_floor": N_DISC_FLOOR,
            "underpowered": underpowered,
            "wins": wins,
            "losses": losses,
            "crossed_at_item": crossed_at,
            "final_wealth": wealth,
            "max_wealth_reached": max_wealth,
            "min_wealth_reached": min_wealth,
            "pass": e_pass,
            "no_p_value_translation": "The e-process outcome is reported on its OWN scale (crossing item count, or final wealth) and is NOT translated into an equivalent p-value (ADR-036 Decision 3). The NLL sign tests below are separate secondary diagnostics on a different endpoint.",
            "full_trajectory": trajectory.iter().map(|s| serde_json::json!({
                "order": s.order, "item": s.item,
                "steer_changed_answer": s.steer_changed_answer,
                "random_changed_answer": s.random_changed_answer,
                "discordant": s.discordant, "x": s.x, "wealth": s.wealth,
            })).collect::<Vec<_>>(),
        },

        "site_provenance": {
            "resolver": "common::m3::build_site_prompt, reused verbatim",
            "site_tag": SITE.tag(),
            "site_description": SITE.description(),
            "samples": site_samples,
            "tokenization_excluded": tokenization_excluded,
            "note": "the site is UNCHANGED from PC2. Only the items differ.",
        },

        "payload_provenance": {
            "capture_receipt": CAPTURE_RECEIPT,
            "file": payload_path.display().to_string(),
            "sha256": payload_sha,
            "steer": "the receiver's OWN block-19 state, teacher-forced over its own solution WITH THE FINAL ANSWER REPLACED BY THE COMMITTED DECOY, last token of the span",
            "restore": "the same tap over the GOLD solution — DERIVED IN PC3'S OWN CAPTURE, not replayed from PC1b (deviation (b)); no prior artifact covers these items",
        },

        "item_supply": {
            "source": ADAPTATION_512,
            "source_split": "adaptation-512",
            "consumption_order": "the file's own fixed ascending index order (ADR-036 Decision 2(1)), restricted to the OUT-OF-SAMPLE tail",
            "leakage_exclusions": LEAKAGE_EXCLUSIONS,
            "leakage_exclusions_present_in_split": excluded_present,
            "eligible_pool": eligible.len(),
            "consumed_by_pc2_m4i": used.len(),
            "remaining_out_of_sample": remaining.len(),
            "n_drawn": n,
            "first_item": stream.first(),
            "last_item": stream.last(),
            "intersection_with_pc2_m4i_stream": overlap,
            "must_travel_with_every_headline_number": "Per ADR-036 Decision 3, any write-up quoting a number from this receipt must state in the same sentence that it was produced under the e-process protocol on the OUT-OF-SAMPLE adaptation-512 tail, on the ANSWER-CHANGE endpoint, at N = the attainable pool rather than the registered 300. PC2's numbers were produced on a DISJOINT 300-item stream on the DECOY-EMISSION endpoint; every completed rung's numbers were produced under the frozen 40-item protocol on the ACCURACY endpoint. None of the three are directly comparable.",
        },

        "dataset": {"source": common::GSM8K_TRAIN_URL, "sha256": train_sha, "indices": stream},
        "decoys": decoys,
        "items": rows,

        "summary": {
            "n_evaluated": n,
            "PRIMARY_answer_change_counts": {
                "steer": ch_steer, "restore": ch_restore, "zerovec_injected": ch_zero,
                "random": ch_rand, "n": n,
                "baseline_uninjected": 0,
                "baseline_note": "baseline is the REFERENCE for this endpoint, so its change count is 0 by construction, not by measurement.",
            },
            "PRIMARY_answer_change_rates": {
                "steer": ch_steer as f64 / nf, "restore": ch_restore as f64 / nf,
                "zerovec_injected": ch_zero as f64 / nf, "random": ch_rand as f64 / nf,
            },
            "primary_e_process_steer_vs_random_on_answer_change": {
                "wins": wins, "losses": losses, "n_discordant": n_disc,
                "n_discordant_floor": N_DISC_FLOOR, "underpowered": underpowered,
                "final_wealth": wealth, "max_wealth_reached": max_wealth,
                "min_wealth_reached": min_wealth,
                "threshold": w_threshold, "crossed_at_item": crossed_at, "pass": e_pass},
            "SECONDARY_decoy_emission_counts": {
                "steer": d_steer, "restore": d_restore, "baseline_uninjected": leak_base,
                "zerovec_injected": d_zero, "random": d_rand, "n": n,
                "note": "PC2's PRIMARY endpoint, retained here as a SECONDARY for continuity. ADR-040 moved it off the primary because it is a rare event with no attainable power.",
            },
            "SECONDARY_decoy_emission_eprocess_for_continuity": {
                "wins": sec_wins, "losses": sec_losses, "n_discordant": sec_wins + sec_losses,
                "final_wealth": sec_w,
                "gating": "NONE — this is PC2's statistic recomputed on PC3's stream purely so the two rungs can be read side by side. It is NOT this rung's registered primary and no interpretation branch keys off it.",
            },
            "SECONDARY_baseline_decoy_leakage": {
                "baseline_decoy_emissions": leak_base, "rate": leak_rate,
                "pc2_void_threshold_for_reference": LEAKAGE_REFERENCE_RATE,
                "above_reference": leak_rate > LEAKAGE_REFERENCE_RATE,
                "gating": "NONE — PC2 VOIDED the rung above 2% because decoy emission was its PRIMARY. ADR-040 moves that endpoint to secondary, so baseline leakage can no longer contaminate the registered primary (which is measured against each item's OWN baseline answer). Recorded for continuity, not gating.",
            },
            "accuracy_secondary": {
                "steer": a_steer, "restore": a_restore, "baseline_uninjected": a_base,
                "zerovec_injected": a_zero, "random": a_rand,
                "note": "PC1b's endpoint, reported for continuity. It is NOT this rung's primary."},
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
                "restore_gold_nll_vs_random_note": "PC1b's own VALIDATED likelihood result (198W/102L, 0.237 nats on ITS stream), replayed here on PC3's DISJOINT stream as a within-rung check that the likelihood arm is still live under this binary's mechanics.",
            },
            "zerovec_is_bit_identical_to_baseline": {
                "expected": "EXACT identity — under fuse the zerovec condition is h += 0. ADR-040 requires 0 answer changes and max|dNLL| = 0.0; anything else means the OPERATOR is wrong and must be reported as such.",
                "answer_changes_vs_baseline": noop_answer_changes,
                "accuracy_disagreements": noop_acc_disagreements,
                "decoy_emission_disagreements": noop_decoy_disagreements,
                "generated_text_identical_items": noop_text_identical,
                "nll_gold_bit_identical_items": noop_exact_items,
                "max_abs_nll_gold_delta": noop_max_abs_nll_delta,
                "max_abs_nll_decoy_delta": noop_max_abs_nll_decoy_delta,
                "tolerance": FUSE_NOOP_TOL,
                "n_items": n,
                "pass": noop_pass,
                "gating": "NONE — operator-correctness diagnostic, reported exactly as PC1b and PC2 reported it, with the answer-change column added because it is this rung's primary endpoint"},
            "zerovec_not_catastrophic": {"criterion": "2 x zerovec_correct >= baseline_correct",
                "pass": zerovec_pass,
                "note": "trivially satisfied because zerovec == baseline under fuse; no evidential weight"},
            "gaming_guard_research_045_section_3": {
                "why": "ITI (arXiv:2306.03341) documents an intervention that appears to restore the answer while merely collapsing the model into a degenerate response. docs/research/045 §3 makes this check mandatory on EVERY positive-control draw.",
                "thresholds_pre_committed": {
                    "degenerate_len_ratio": DEGENERATE_LEN_RATIO,
                    "degenerate_item_fraction": DEGENERATE_ITEM_FRACTION,
                    "nll_collapse_nats": NLL_COLLAPSE_NATS,
                    "note": "PC1b's and PC2's values, UNCHANGED, so all three rungs' guards are directly comparable"},
                "mean_steer_over_baseline_generated_char_ratio": mean_len_ratio,
                "degenerate_short_items": degenerate_short_items,
                "degenerate_output_signature": degenerate_signature,
                "mean_steer_nll_gold": mean_steer_nll_gold,
                "nll_collapse_signature": nll_collapse_signature,
                "gaming_signature": gaming_signature,
                "companion_flag_not_folded_into_the_gate": {
                    "mean_steer_nll_decoy": mean_steer_nll_decoy,
                    "decoy_nll_collapse_flag": decoy_nll_collapse_flag,
                    "why_separate": "the payload encodes the DECOY, so a low decoy-NLL is the expected direction rather than a degeneracy signal. Reported but deliberately NOT folded into the gating signature, so the guard stays byte-comparable with PC1b's and PC2's."},
            },
        },

        "comparison_pc2_vs_pc3": {
            "pc2_receipt": PC2_PROBE_RECEIPT,
            "streams_are_disjoint": true,
            "note": "PC2 and PC3 ran the same apparatus at the same site with the same operator, slots and decoy construction, on DISJOINT 300- and out-of-sample-tail item streams, with DIFFERENT registered primaries (PC2: decoy emission; PC3: answer change). The decoy-emission and accuracy columns are directly comparable BETWEEN the two receipts as descriptive rates; the registered primary statistics are not the same statistic and are not translated into each other (ADR-036 Decision 3).",
            "pc2_summary": pc2["summary"].clone(),
            "pc2_e_process": pc2["e_process"].clone(),
        },

        "gates": {
            "OUT_OF_SAMPLE_intersection_with_pc2_m4i_is_empty": {"pass": true, "overlap": 0,
                "pc2_stream_items": pc2_stream.len(), "pc3_stream_items": n,
                "note": "HARD GATE, re-evaluated in this binary rather than trusted from the capture receipt. This is the defining property of PC3 and the binary aborts on any overlap."},
            "payload_sha256_matches_capture_receipt": {"pass": true, "sha256": payload_sha},
            "decoys_committed_before_any_forward_pass": {"pass": true, "receipt": COMMITMENT_RECEIPT},
            "stream_identical_to_the_capture_and_commitment": {"pass": true, "n": n},
            "identity_transform_no_adapter_weights_loaded": {"pass": true},
            "injection_mode_recorded": {"pass": true, "mode": INJECT_MODE.tag(),
                "equation": INJECT_MODE.equation()},
            "slot_count_unchanged": {"pass": true, "slots": N_SLOTS},
            "eprocess_never_restarted": {"pass": true},
            "PC3_steer_vs_random_answer_change_eprocess": {"pass": e_pass,
                "final_wealth": wealth, "threshold": w_threshold,
                "n_discordant": n_disc, "crossed_at_item": crossed_at},
            "n_disc_at_or_above_the_registered_floor": {"pass": !underpowered,
                "n_discordant": n_disc, "floor": N_DISC_FLOOR,
                "note": "ADR-040: if this FAILS the rung is uninformative AND the power model is recorded as wrong."},
            "zerovec_bit_identical_to_baseline": {"pass": noop_pass, "gating": "none — diagnostic",
                "answer_changes": noop_answer_changes, "max_abs_nll_delta": noop_max_abs_nll_delta},
            "zerovec_not_catastrophic": {"pass": zerovec_pass, "degenerate_under_fuse": true},
            "no_gaming_signature": {"pass": !gaming_signature,
                "degenerate_output_signature": degenerate_signature,
                "nll_collapse_signature": nll_collapse_signature},
            "n_max_reduced_from_300_pool_exhausted": {
                "pass": true, "is_a_deviation": n < N_MAX_REGISTERED,
                "registered": N_MAX_REGISTERED, "drawn": n,
                "note": "NOT a silent pass — see RECORDED_DEVIATIONS.a."},
            "restore_derived_not_replayed": {
                "pass": true, "is_a_deviation": true,
                "note": "NOT a silent pass — see RECORDED_DEVIATIONS.b."},
        },
        "gate_pass": e_pass && zerovec_pass && !gaming_signature && !underpowered,
        "branch": branch,
        "branch_text_NOT_THE_ADJUDICATED_VERDICT": branch_text,
        "honest_fail_contract": "ADR-032: ONE registered draw, no retry, full numbers reported either way. A FAILING positive control WITH REAL POWER is the MORE consequential outcome and is stated plainly here, not softened. An UNDERPOWERED outcome is reported as uninformative, is not spun as either branch, and additionally records the power model as wrong.",
        "wall_clock_s": t0.elapsed().as_secs_f32(),
    });
    common::write_receipt(
        &crate_path("receipts"),
        "run2-pc3-receipt-identity-L19lasttoken-decoytf-fuse-questiontail-slots8-eprocess-answerchange-oos.json",
        &receipt,
    )?;

    println!("\n=== PC3 (out-of-sample decision-change — APPARATUS ONLY, NEVER TRANSFER) ===");
    println!(
        "PRIMARY — answer differs from own uninjected baseline:  steer {ch_steer}/{n} \
         ({:.1}%)  random {ch_rand}/{n} ({:.1}%)  restore {ch_restore}/{n}  zerovec {ch_zero}/{n}",
        100.0 * ch_steer as f64 / nf,
        100.0 * ch_rand as f64 / nf,
    );
    println!(
        "e-process on ANSWER CHANGE: wins {wins}, losses {losses}, n_discordant {n_disc} \
         (floor {N_DISC_FLOOR}, underpowered={underpowered}); final W {wealth:.4} \
         (max {max_wealth:.4}, min {min_wealth:.4}) vs threshold {w_threshold} => pass={e_pass} \
         (crossed_at={crossed_at:?})"
    );
    println!(
        "SECONDARY decoy emission: steer {d_steer}/{n} restore {d_restore}/{n} baseline \
         {leak_base}/{n} zerovec {d_zero}/{n} random {d_rand}/{n} (baseline rate {leak_rate:.4}); \
         PC2-style e-process on this stream: {sec_wins}W/{sec_losses}L, W {sec_w:.4}"
    );
    println!(
        "accuracy (secondary) steer {a_steer}/{n} restore {a_restore}/{n} baseline {a_base}/{n} \
         zerovec {a_zero}/{n} random {a_rand}/{n}"
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
        "zerovec == baseline: answer changes {noop_answer_changes}, acc disagreements \
         {noop_acc_disagreements}, decoy disagreements {noop_decoy_disagreements}, identical \
         texts {noop_text_identical}/{n}, max|dNLL| {noop_max_abs_nll_delta:.3e} => pass={noop_pass}"
    );
    println!(
        "gaming guard: mean len ratio {mean_len_ratio:.3}, degenerate-short items \
         {degenerate_short_items}/{n} => degenerate={degenerate_signature}; \
         nll_collapse={nll_collapse_signature} => gaming_signature={gaming_signature}"
    );
    println!("BRANCH (mechanical label, NOT the adjudicated verdict): {branch}");
    println!("{branch_text}");
    Ok(())
}
