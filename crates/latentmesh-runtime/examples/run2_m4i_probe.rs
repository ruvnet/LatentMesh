//! Run-2 **M4i** — inject at ORDINARY tokens, not `<|fim_pad|>`
//! (ADR-024 § "M4i PRE-REGISTRATION (2026-08-29, before any run)"), evaluated
//! under **ADR-036's successor-rung e-process**, which this rung is the first
//! to use. `run2_m4h_s1_probe` lineage.
//!
//! **Why this rung exists.** `docs/research/043` identifies a third structural
//! difference from every surveyed working cross-model method, and it is the
//! only one that predicts **inertness** rather than degraded or harmful
//! transfer — precisely the signature ADR-024's MAJOR CORRECTION isolated for
//! the on-manifold family (five configurations within 0.004 nats of baseline).
//! The receiver is Qwen2.5-1.5B-**Instruct**, non-Coder; the base Qwen2.5
//! technical report never mentions FIM, and FIM training is documented only in
//! the Qwen2.5-**Coder** report. `<|fim_pad|>` is therefore plausibly
//! **experientially near-vacant** for this checkpoint — an embedding with no
//! circuitry rather than one actively suppressed. No surveyed method (C2C,
//! LatentMAS, Bicameral) injects at a placeholder; all three inject onto the
//! receiver's own real token positions.
//!
//! **THE SINGLE CHANGED FACTOR, and its comparator.** This rung changes the
//! injection **SITE** and nothing else, measured against **M4h Stage 1** —
//! `receipts/run2-m4h-s1-receipt-cellL18toL14-mlp-pertokenlast-fuse-slots8-nopool-rescaletrue-n40.json`,
//! the most recent comparator. Held fixed against that receipt:
//!   * the **artifact** — M3's already-trained on-manifold reconstruction MLP,
//!     byte-for-byte, hash-asserted against M3's training receipt (frozen
//!     2026-08-28, long before either rung existed) plus its golden pairs;
//!   * the **payload derivation** — `Variant::PerTokenLast` (`apply_last_row`,
//!     per-token translate then the LAST token, de-pooled), the derivation
//!     M4h Stage 1 used, unchanged;
//!   * the **operator** — `InjectionMode::Fuse`, `h[pos] += c*v`;
//!   * the **slot count** — 8, ADR-028-protected, asserted at run time;
//!   * the **depth** — receiver block 14 (the S2 winner cell L18→L14);
//!   * the **rescale rule** — to the per-item natural median at the inject
//!     block; the **decoding** — greedy, batch=1, max 400 new tokens; the
//!     four **conditions** and M4g's frozen control semantics.
//!
//! Changed: `Site::FimPadSlots` → `Site::QuestionTail`.
//!
//! **What "the site" concretely is.** `docs/research/043` §4 offers two
//! ordinary-token variants and this rung takes its **primary recommendation**:
//! the **last 8 tokens of the item's own question**. The alternative (the
//! fixed ANSWER_FORMAT instruction tokens) was considered and rejected before
//! the draw, for a reason recorded in the receipt: those tokens are
//! item-INVARIANT boilerplate, so fusing an item-varying payload onto them
//! would put real content at positions whose own natural state carries no
//! information about the item — structurally closer to the generic-slot
//! failure mode this rung exists to escape, and it would partially reproduce
//! the confound instead of removing it. The question tail is also the closest
//! available analogue to C2C's residual add onto the receiver's own prompt
//! positions, and it is the span the payload is *about*.
//!
//! **The slot sentence is removed entirely**, per `docs/research/043` §4:
//! keeping "stored in these slots: [...]" once the injected positions are
//! ordinary question tokens would reintroduce a textual placeholder cue even
//! though the token identity changed. The receiver prompt is therefore exactly
//! `{question}\n\n{ANSWER_FORMAT}` under the same chat template. This is
//! **constitutive of the site change, not a second factor** — but it does mean
//! this rung's baseline prompt differs from M4h Stage 1's, so cross-rung
//! ACCURACY LEVELS are not comparable. The primary is paired **within** this
//! rung, where all four conditions share one prompt, so the comparison the
//! statistic makes is unaffected.
//!
//! **`random` means something stronger here, declared before the draw.**
//! Under `QuestionTail` + `Fuse` the norm-matched Gaussian perturbs the
//! receiver's own genuine question content rather than an inert placeholder
//! row. `docs/research/043` §4 flags this explicitly: if `aligned` still ties
//! or loses to `random` here, that is MORE informative than at the placeholder
//! site, because `random` is no longer "perturbing nothing". `zerovec` remains
//! a true no-op under fuse (`h += 0`), so it collapses onto
//! `baseline_uninjected` exactly as M4g registered and measured.
//!
//! **STATISTICS — ADR-036's e-process, not the frozen 40-item sign test.**
//! ADR-024's own M4h Stage 1 outcome records that the frozen protocol went
//! blind precisely because payloads improved (n_disc 7 → 3 → 2); ADR-036
//! Decision 4 names M4i as governed by the successor protocol and explicitly
//! supersedes M4i's own "mid-p McNemar primary" pre-registration paragraph.
//! Registered mechanics, verbatim from ADR-030 §3.2 via ADR-036 Decision 1:
//! `W_0 = 1`; concordant item → `W_i = W_{i-1}`; discordant item →
//! `W_i = W_{i-1}·(1 + λ(X_i − 0.5))` with `X_i = 1` iff aligned wins the
//! pair; `λ = 0.30`; **PASS** the instant `W_i ≥ 1/α = 20`; **FAIL** if it has
//! not crossed by `N_max = 300` items. The full `W_t` trajectory is committed
//! either way, and the process is never restarted, re-parametrised or re-run.
//!
//! **Item supply — ADR-036 Decision 2.** Items are drawn from
//! `adaptation-512` in the fixed index order the committed file already
//! carries, never `eval-200`/`holdout-100` (still mechanically locked), and
//! never from the frozen 40-item probe set — ADR-036 corrected ADR-030's
//! "subset by construction" claim and this rung follows the correction. The
//! 13-item leakage-exclusion list is applied; its intersection with
//! `adaptation-512` is computed here rather than assumed.
//!
//! **Comparability — ADR-036 Decision 3.** This receipt reports its e-process
//! outcome on its own scale (crossing item count, or final wealth at
//! `N_max`). It does **not** translate that outcome into a p-value, and it
//! does not recompute an accuracy sign test that could be mistaken for one:
//! the completed-rung era's exact-sign/mid-p numbers stand as recorded and are
//! not restated here as if commensurable. Protocol identity travels with every
//! headline number in the receipt.
//!
//! ONE draw. No retry (ADR-032 honest-fail).
//!
//! Run: PATH=/usr/local/cuda-12.8/bin:$PATH \
//!      cargo run --release --features cuda --example run2_m4i_probe

// The receipt is one large `serde_json::json!` literal (the frozen-receipt
// house style: one object, auditable top to bottom). It expands past rustc's
// default 128-deep macro recursion limit.
#![recursion_limit = "512"]

#[path = "common/mod.rs"]
#[allow(dead_code)]
mod common;

use common::m3::{
    build_site_prompt, run_item_at, Quad, Site, Variant, GOLDEN_REL_TOL, N_SLOTS,
    RANDVEC_SEED_BASE, RECEIVER, RECEIVER_BLOCK, SENDER, SENDER_BLOCK,
};
use common::mlp::MlpTransform;
use latentmesh_runtime::inject::InjectionMode;
use latentmesh_runtime::QwenRuntime;
use std::path::{Path, PathBuf};

const GSM8K_TRAIN_SHA256: &str = "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465";
/// M3's training receipt — the artifact's freeze point, written 2026-08-28.
const TRAINING_RECEIPT: &str = "receipts/run2-m3-training-receipt-cellL18toL14.json";
/// M4g's training receipt, read ONLY for its frozen control-semantics block.
const M4G_TRAINING_RECEIPT: &str = "receipts/run2-m4g-training-receipt-cellL18toL14.json";
/// The comparator this rung isolates its single changed factor against.
const COMPARATOR_RECEIPT: &str =
    "receipts/run2-m4h-s1-receipt-cellL18toL14-mlp-pertokenlast-fuse-slots8-nopool-rescaletrue-n40.json";
/// The manifold pre-check receipt this rung's payload must already appear in.
const PRECHECK_RECEIPT: &str = "receipts/run2-m4i-manifold-precheck-receipt.json";
/// The pre-check candidate label for this rung's payload derivation. It is
/// M4h Stage 1's label because the payload derivation is BYTE-IDENTICAL —
/// same artifact hash, same `apply_last_row`. The pre-check is payload-side
/// only (it measures the emitted vectors against the receiver's own L14
/// states from the committed S2c dumps) and is therefore independent of the
/// injection site this rung changes; re-running it under a second label would
/// imply a second payload that does not exist.
const PRECHECK_LABEL: &str = "m4h-s1-m3-mlp-lasttoken-depooled";
const ARTIFACT: &str = "receipts/run2-m3-mlp-cellL18toL14.f32bin";
const GOLDEN: &str = "receipts/run2-m3-golden-mlp-cellL18toL14.json";
const ADAPTATION_512: &str = "../../harness/latentmesh-live/data/adaptation-512.json";

/// UNCHANGED from M4h Stage 1 — the payload derivation is not this rung's
/// variable.
const VARIANT: Variant = Variant::PerTokenLast;
/// UNCHANGED from M4h Stage 1 — the operator is not this rung's variable.
const INJECT_MODE: InjectionMode = InjectionMode::Fuse;
/// **THE SINGLE CHANGED FACTOR** — ADR-024 M4i, `docs/research/043` §4.
const SITE: Site = Site::QuestionTail;

/// ADR-024's 13 probe-overlap items, verbatim from the ADR's own
/// "Leakage discipline" section and from every M3+ training receipt's
/// `split.excluded_probe_overlap_rows`. ADR-036 Decision 2(2) keeps this
/// exclusion in force for the successor protocol's item stream.
const LEAKAGE_EXCLUSIONS: [usize; 13] = [
    150, 1309, 1573, 2365, 2418, 2540, 2958, 2973, 3084, 3825, 3844, 3877, 4746,
];

// ---- ADR-036 Decision 1 / ADR-030 §3.2 e-process parameters, frozen -------
/// Betting fraction, `λ = 2θ−1` tuned to the smallest interesting effect
/// θ=0.65. Fixed in advance; never re-parametrised after seeing `W_t`.
const LAMBDA: f64 = 0.30;
/// α for the wealth boundary. PASS at `W_i ≥ 1/α`.
const E_ALPHA: f64 = 0.05;
/// The registered budget. 300 items × the ladder's ~10% discordance ≈ 30
/// discordant pairs, the count `docs/research/031` §2.2 identifies as needed
/// for real power at a plausible effect size.
const N_MAX: usize = 300;
/// Tolerance for the fuse zero-payload no-op diagnostic (nats). Reported,
/// never gating.
const FUSE_NOOP_TOL: f32 = 1e-6;

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

fn main() -> anyhow::Result<()> {
    let t0 = std::time::Instant::now();
    let nvcc = latentmesh_runtime::assert_cuda_build_env().map_err(anyhow::Error::msg)?;
    println!("build-env guard: {nvcc}");
    let w_threshold = 1.0 / E_ALPHA;

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

    // ---- Gate 2: the comparator, and the single changed factor ------------
    // The rung is only meaningful if the receipt it isolates its factor
    // against actually ran the SAME artifact under the SAME derivation and
    // the SAME operator, at the placeholder site. Asserted, not asserted-by-
    // prose.
    let comparator: serde_json::Value =
        serde_json::from_slice(&std::fs::read(crate_path(COMPARATOR_RECEIPT)).map_err(|e| {
            anyhow::anyhow!(
                "comparator receipt {COMPARATOR_RECEIPT} unreadable ({e}) — M4i isolates the SITE \
                 change against M4h Stage 1 and cannot state that claim without it"
            )
        })?)?;
    anyhow::ensure!(
        comparator["config"]["transform"]["content_hash"].as_str() == Some(expected_hash.as_str()),
        "the comparator receipt used a different artifact"
    );
    anyhow::ensure!(
        comparator["variant"].as_str() == Some("pertokenlast-fuse"),
        "the comparator receipt used a different payload derivation / operator"
    );
    anyhow::ensure!(
        comparator["config"]["placeholder_token"].as_str() == Some("<|fim_pad|>"),
        "the comparator receipt was not drawn at the placeholder site"
    );
    anyhow::ensure!(
        comparator["config"]["injection_operator"]["mode"].as_str() == Some(INJECT_MODE.tag()),
        "the comparator receipt used a different injection operator"
    );
    println!(
        "comparator: {COMPARATOR_RECEIPT} (same artifact, same {} derivation, same {} operator, \
         placeholder site) — the SINGLE changed factor is the SITE: {} -> {}",
        VARIANT.tag(),
        INJECT_MODE.tag(),
        Site::FimPadSlots.tag(),
        SITE.tag()
    );

    // ---- Gate 3: M4g's frozen control semantics, reused verbatim ----------
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
    println!("control semantics: inherited VERBATIM from {M4G_TRAINING_RECEIPT}");

    // ---- Gate 4 (ORDERING, not evidential): the manifold pre-check --------
    let precheck: serde_json::Value =
        serde_json::from_slice(&std::fs::read(crate_path(PRECHECK_RECEIPT)).map_err(|e| {
            anyhow::anyhow!(
                "manifold pre-check receipt {PRECHECK_RECEIPT} unreadable ({e}) — the pre-check is \
                 ordered BEFORE this draw; run `cargo run --release --example \
                 run2_manifold_precheck run2-m4i-manifold-precheck-receipt.json` first"
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

    // ---- Item supply: ADR-036 Decision 2 ----------------------------------
    let dir = common::run_dir("run2-m4i");
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

    // Leakage exclusions applied, and their effect MEASURED rather than
    // assumed: ADR-036 Decision 2(2) keeps ADR-024's 13-item rule in force.
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
    println!(
        "item supply: adaptation-512 in fixed index order; 13-item leakage exclusion applied, \
         {} of them present in this split; {} eligible items, N_max = {N_MAX}",
        excluded_present.len(),
        eligible.len()
    );
    anyhow::ensure!(
        eligible.len() >= N_MAX,
        "eligible pool ({}) is smaller than N_max ({N_MAX})",
        eligible.len()
    );

    let device = latentmesh_runtime::device().map_err(anyhow::Error::msg)?;
    println!("loading {SENDER} + {RECEIVER} (BF16, concurrent-resident)...");
    let mut sender =
        QwenRuntime::load(SENDER, &device, candle_core::DType::BF16).map_err(anyhow::Error::msg)?;
    let mut receiver = QwenRuntime::load(RECEIVER, &device, candle_core::DType::BF16)
        .map_err(anyhow::Error::msg)?;
    // Resolved but UNUSED under `Site::QuestionTail`: there is no placeholder
    // token anywhere in this rung's prompt. Kept so the shared code path is
    // literally the same function every prior rung called.
    let pad_id = receiver
        .tokenizer
        .token_to_id("<|fim_pad|>")
        .ok_or_else(|| anyhow::anyhow!("<|fim_pad|> not in tokenizer"))?;
    println!("models loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // ---- Tokenisation pre-flight, BEFORE any generation -------------------
    // The site gate is a mechanical property of the prompt (does the
    // question's token span end on a byte-pair boundary, and do the eight
    // chosen tokens decode back to a suffix of the question?). Resolving it
    // for the whole stream up front means no item can be dropped mid-draw for
    // a reason that could correlate with its outcome. Any failures are
    // recorded and excluded here, before the first forward pass.
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
                        "question_tail_check": "decoded text is whitespace-insensitively contained in the item's own question; the window is contiguous and ends at the last token wholly inside it",
                    }));
                }
                stream.push(idx);
            }
            Err(err) => {
                if tokenization_excluded.len() < 3 {
                    println!("  site pre-flight rejected item {idx}: {err}");
                }
                tokenization_excluded.push(serde_json::json!({
                    "item": idx, "reason": err.to_string(),
                }));
            }
        }
    }
    anyhow::ensure!(
        stream.len() == N_MAX,
        "pre-flight resolved only {} of the required {N_MAX} items",
        stream.len()
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
    let mut items_drawn = 0usize;
    let mut degenerate = 0usize;

    for (order, &idx) in stream.iter().enumerate() {
        let item = &all_items[idx];
        items_drawn = order + 1;
        match run_item_at(
            &mut sender,
            &mut receiver,
            &transform,
            item,
            pad_id,
            VARIANT,
            INJECT_MODE,
            SITE,
            &device,
        )? {
            Some((row, q)) => {
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
            }
            None => {
                // A degenerate sender capture yields no pair, so it produces
                // no wealth update — exactly like a concordant item — but it
                // still CONSUMES one of the N_max budget items. Registered
                // that way here rather than decided after seeing the data.
                degenerate += 1;
                println!(
                    "[{}/{N_MAX}] item {idx}: degenerate sender capture pass, no pair (W unchanged \
                     at {wealth:.4}); budget item consumed",
                    order + 1
                );
                trajectory.push(EStep {
                    order: order + 1,
                    item: idx,
                    aligned_correct: false,
                    random_correct: false,
                    discordant: false,
                    x: None,
                    wealth,
                });
                rows.push(
                    serde_json::json!({"item": idx, "skipped": "degenerate sender capture pass"}),
                );
            }
        }
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

    let noop_acc_disagreements = count(&|q| q.zero.0 != q.base.0);
    let noop_max_abs_nll_delta = paired
        .iter()
        .map(|q| (q.zero.1 - q.base.1).abs())
        .fold(0f32, f32::max);
    let noop_exact_items = paired.iter().filter(|q| q.zero.1 == q.base.1).count();
    let noop_pass = noop_acc_disagreements == 0 && noop_max_abs_nll_delta <= FUSE_NOOP_TOL;

    let inversion_vs_baseline_items = paired.iter().filter(|q| q.real.1 > q.base.1).count();
    let inversion_vs_zerovec_items = paired.iter().filter(|q| q.real.1 > q.zero.1).count();
    let inversion_vs_random_items = paired.iter().filter(|q| q.real.1 > q.rand.1).count();

    let trajectory_json: Vec<serde_json::Value> = trajectory
        .iter()
        .map(|s| {
            serde_json::json!({
                "order": s.order, "item": s.item,
                "aligned_correct": s.aligned_correct, "random_correct": s.random_correct,
                "discordant": s.discordant, "x": s.x, "wealth": s.wealth,
            })
        })
        .collect();

    let receipt = serde_json::json!({
        "stage": "run2-M4i-ordinary-token-site-probe",
        "design": "docs/adr/024-run2-trained-thought-adapter-ladder.md § 'M4i PRE-REGISTRATION (2026-08-29, before any run) — inject at ORDINARY tokens, not <|fim_pad|>'; evaluation protocol = docs/adr/036-successor-rung-evaluation-protocol.md (e-process, adaptation-512 item stream), which ADR-036 Decision 4 names as SUPERSEDING M4i's own 'mid-p McNemar primary' pre-registration paragraph. Evidence basis for the site change: docs/research/043-placeholder-token-choice.md.",
        "env": common::env_info(&nvcc),
        "pre_committed": true,
        "variant": "pertokenlast-fuse-questiontail",
        "protocol_identity": {
            "statistics": "ADR-036 anytime-valid Bernoulli e-process (ADR-030 §3.2 mechanics verbatim)",
            "item_supply": "adaptation-512, fixed index order, ADR-024's 13-item leakage exclusion applied",
            "era": "SUCCESSOR-RUNG ERA. Not the frozen 40-item sign-test protocol that governed M3, M4 x3, M4c, M4d, M4g and M4h Stage 1.",
            "must_travel_with_every_headline_number": "Per ADR-036 Decision 3, any write-up quoting a number from this receipt must state in the same sentence that it was produced under the e-process protocol on the adaptation-512 stream, not under the frozen 40-item protocol.",
        },
        "the_single_changed_factor": {
            "factor": "INJECTION SITE",
            "from": Site::FimPadSlots.tag(),
            "to": SITE.tag(),
            "site_description": SITE.description(),
            "comparator_receipt": COMPARATOR_RECEIPT,
            "held_fixed_against_the_comparator": [
                "artifact: M3's already-trained on-manifold reconstruction MLP, byte-identical (hash asserted against M3's training receipt AND against the comparator receipt's own recorded hash)",
                "payload derivation: Variant::PerTokenLast (apply_last_row, per-token translate then LAST token, de-pooled) — the derivation M4h Stage 1 used, explicitly the same one so this rung isolates the SITE",
                "injection operator: InjectionMode::Fuse, h[pos] += c*v",
                "slot count: 8 (ADR-028-protected; asserted at run time)",
                "depth: receiver block 14, cell L18->L14",
                "rescale: to the per-item natural median at the inject block",
                "decoding: greedy, batch=1, max_new_tokens=400",
                "conditions and control semantics: M4g's, verbatim",
            ],
            "prompt_change_is_constitutive_not_a_second_factor": "docs/research/043 §4 specifies removing the 'stored in these slots: [...]' sentence entirely along with the placeholder tokens, because keeping the bracket would reintroduce a textual placeholder cue even once the token identity changed. The receiver prompt is therefore exactly '{question}\\n\\n{ANSWER_FORMAT}' under the same chat template. DISCLOSED CONSEQUENCE: this rung's baseline prompt differs from the comparator's, so cross-rung ACCURACY LEVELS are not comparable. The primary comparison is paired WITHIN this rung, where all four conditions share one prompt, and is unaffected.",
        },
        "site_choice_justification": {
            "chosen": "last 8 tokens of the item's own question",
            "why": "docs/research/043 §4's own primary recommendation. It is the closest available analogue to C2C's residual add onto the receiver's own real prompt positions, it needs no added text, and it is the span the payload is ABOUT — the aligned vector is derived from the sender solving this question.",
            "rejected_alternative": "the fixed ANSWER_FORMAT instruction tokens (043 §4's registered secondary variant)",
            "why_rejected": "those tokens are item-INVARIANT boilerplate. Fusing an item-varying payload onto positions whose own natural state carries no information about the item is structurally closer to the generic-slot failure mode this rung exists to escape, and would partially reproduce the confound instead of removing it. Recorded before the draw; the variant remains available to a future rung.",
            "position_resolution": "the 8 positions are the LAST EIGHT tokens whose byte span lies WHOLLY INSIDE item.question, read off the canonical full-prompt tokenisation's own offset map. Gates: at least 8 such tokens exist; they are contiguous; the window ends at the last token wholly inside the question; and the 8 token ids decode back to text contained in the question. All four are mechanical prompt properties, resolved for the entire stream in a pre-flight BEFORE the first forward pass.",
            "why_offsets_and_not_a_prefix_re_encode": "MEASURED, not assumed: re-encoding the prefix 'header + question' and requiring it to be a prefix of the canonical tokenisation was tried first and REJECTED EVERY ITEM. Qwen2.5's pre-tokeniser groups trailing punctuation with the newlines that follow it, so for a GSM8K question ending in '?' the final token spans '?\\n\\n' and STRADDLES the question/answer-format boundary. That token is not part of the question and is excluded, which is why the decode gate checks containment rather than a strict suffix: the tail window ends at the last token wholly inside the question, leaving the boundary character uncovered. The 8 injected positions are therefore all ordinary question-content tokens, none of them straddling into the instruction text.",
            "exact_positions_recorded_per_item": "every per-item row carries injection_site.positions, .position_token_ids and .positions_decoded",
            "sample_positions": site_samples,
        },
        "why_this_rung": {
            "hypothesis": "docs/research/043: <|fim_pad|> (id 151662, \"special\": false in the receiver's own tokenizer_config) is plausibly EXPERIENTIALLY NEAR-VACANT for Qwen2.5-1.5B-Instruct — the base Qwen2.5 technical report (2412.15115) never mentions FIM; FIM training is documented only in the Qwen2.5-CODER report (2409.12186). An embedding with no circuitry, not one actively suppressed.",
            "why_it_is_the_right_hypothesis_for_THIS_signature": "ADR-024's MAJOR CORRECTION separates two null families: off-manifold payloads are actively HARMFUL (M4c/M4d/M4g, 0W/40L, NLL 2.3-2.5x baseline), on-manifold payloads are harmless and INERT (M3 both variants, M4 all three ranks, run-1 affine — all within ~0.004 nats of baseline). Pooling predicts degraded transfer; one-shot delivery predicts fading transfer; only an injection site the receiver has no circuitry for predicts INERTNESS regardless of payload content.",
            "honest_limits_carried_from_the_pre_registration": "the FIM-exposure claim is inferred from ABSENCE OF MENTION in the base report, not from a stated negative; and NLL alone cannot distinguish 'vacant embedding' from 'mildly trained but unremarkable'. A NULL here demotes the site to a real-but-non-load-bearing structural difference and points back to pooling and receiver scale.",
        },
        "no_training_performed": {
            "trained_this_rung": false,
            "captured_this_rung": false,
            "artifact_origin": "M3's committed adapter, unchanged; its training receipt was frozen 2026-08-28, before this rung was conceived",
            "no_transfer_check_and_why": "M4c/M4d/M4g each carried a transfer check because they were TRAINED through a composed path needing reconciliation with the deployed fused BF16 path. This rung trains nothing and captures nothing new, so there is no composed-vs-deployed gap to reconcile; M3's own probe and M4h Stage 1 likewise had no transfer gate. Asserting one would be theatre.",
        },
        "slot_count_protected": {
            "slots": N_SLOTS,
            "changed": false,
            "note": "The SITE of the 8 delivery positions changes; the NUMBER does not. ADR-024 records that ADR-028 lists 'slot count' on BOTH sides of its evolvable/protected boundary and that the contradiction is FLAGGED AND UNADJUDICATED. This rung stays clear of it by construction, and build_site_prompt asserts positions.len() == 8 for every item.",
        },
        "manifold_precheck_run_before_this_draw": {
            "receipt": PRECHECK_RECEIPT,
            "candidate_label": PRECHECK_LABEL,
            "artifact_hash_matches_this_probe": true,
            "classification": precheck_class,
            "mean_cosine_to_same_item_natural_receiver_L14_pooled": precheck_manifold_cos,
            "item_invariance_mean_pairwise_cosine": precheck_invariance,
            "entropy_nats_rmsnorm_lens": precheck_entropy,
            "gating": "NONE — ORDERING ONLY. Per ADR-024's M4f framing the pre-check is diagnostic; it must EXIST and have measured this exact (artifact hash, derivation) pair before the draw, and it decides nothing about the verdict.",
            "why_the_label_is_m4h_s1s": "the payload derivation is BYTE-IDENTICAL to M4h Stage 1's (same artifact hash, same apply_last_row). The pre-check is payload-side only — it measures emitted vectors against the receiver's own L14 states from the committed S2c dumps — and is therefore independent of the injection site this rung changes. Re-running it under a second label would imply a second payload that does not exist; it was re-run into this rung's own receipt file instead.",
            "scoping_caveat": "cosine-to-natural is measured against the POOLED natural receiver state, which ADR-024 notes is itself ~0.667 cosine from a real un-pooled receiver state. The pre-check's own un-pooled reference row (reference-receiver-L14-single-row) is the like-for-like comparator for a de-pooled payload.",
        },
        "control_semantics_under_fuse_inherited_verbatim_from_m4g": {
            "source_receipt": M4G_TRAINING_RECEIPT,
            "reused_not_restated": "This rung constructs its controls through the SAME shared common::m3 code path M4g and M4h Stage 1 ran, under the same InjectionMode::Fuse. M4g's frozen block is echoed below rather than re-authored, so the definitions cannot drift between rungs.",
            "what_the_site_change_does_to_the_MEANING_of_random": "DECLARED BEFORE THE DRAW (docs/research/043 §4). zerovec is unchanged: h += 0 is an exact no-op under fuse at any site, so it still collapses onto baseline_uninjected. random IS changed in meaning: a norm-matched Gaussian fused onto real question-token activations is a perturbation of GENUINE CONTENT, not of an inert placeholder row. This makes it a STRONGER comparator for the primary than it was at the placeholder site — if aligned ties or loses to random here, that is more informative than the placeholder-site result, because random is no longer perturbing nothing.",
            "frozen_block": control_semantics,
        },
        "config": {
            "sender": SENDER, "receiver": RECEIVER,
            "sender_capture_block": SENDER_BLOCK, "receiver_inject_block": RECEIVER_BLOCK,
            "cell": "L18->L14 (S2 winner; M4i registers on this cell only)",
            "slots": N_SLOTS,
            "injection_site": SITE.tag(),
            "placeholder_token": serde_json::Value::Null,
            "placeholder_token_note": "NONE. This rung's prompt contains no <|fim_pad|> and no slot sentence. The receiver's pad id was resolved only so the shared code path is literally the same function every prior rung called; it is unused under this site.",
            "resolved_pad_id_unused": pad_id,
            "pool_span": "NONE — de-pooled. The payload is the last translated token of the generated span; the 8-position broadcast of that single vector is unchanged.",
            "rescale_to_natural_median": true,
            "natural_norm_source": "receiver forward_capture over this rung's injection prompt at the inject block, per item",
            "decoding": "greedy, batch=1, max_new_tokens=400",
            "randvec_seed_base": RANDVEC_SEED_BASE,
            "injection_operator": {
                "mode": INJECT_MODE.tag(),
                "equation": INJECT_MODE.equation(),
                "asserted_equal_to_m4g_receipt": true,
                "why_fuse_is_load_bearing_here": "overwriting real question tokens would destroy content the receiver needs, confounding 'the position is inert' with 'we deleted the question'. Fuse preserves the receiver's own state at those positions and adds the payload on top (docs/research/043 §4).",
            },
            "transform": {
                "kind": "M3's ALREADY-TRAINED reconstruction MLP 2048->512->1536 ReLU — byte-identical weights, no retraining",
                "file": artifact.display().to_string(),
                "content_hash": transform.content_hash,
                "training_receipt": TRAINING_RECEIPT,
                "payload_derivation": "apply_last_row: relu(x@W1+b1)@W2+b2 on the LAST generated-span token state only — IDENTICAL to M4h Stage 1",
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
                "aligned_real": "sender per-token capture -> M3 MLP per token -> LAST token -> delivered at the 8 question-tail positions, rescaled to natural median, by residual ADD.",
                "random": "per-item seeded gaussian, norm-matched to the effective aligned_real vector, same 8 positions, same operator. Under this site a norm-matched perturbation of the receiver's own GENUINE question content.",
                "zerovec_injected": "TRUE ZERO VECTOR through the same 8-position path (scale: None). Under fuse this is h += 0, an exact NO-OP, so it collapses onto baseline_uninjected.",
                "baseline_uninjected": "no injection (spec=None), same prompt.",
            },
        },
        "e_process": {
            "registered_source": "ADR-036 Decision 1, quoting ADR-030 §3.2 verbatim",
            "rule": "W_0 = 1. Concordant item: W_i = W_{i-1}. Discordant item: W_i = W_{i-1} * (1 + lambda*(X_i - 0.5)), X_i = 1 iff the aligned condition wins the pair. PASS the instant W_i >= 1/alpha.",
            "lambda": LAMBDA,
            "alpha": E_ALPHA,
            "wealth_threshold": w_threshold,
            "n_max": N_MAX,
            "comparison": "aligned_real vs random on paired per-item accuracy — the ladder's standing primary comparison, unchanged; only the STATISTIC applied to it changes",
            "degenerate_item_rule": "an item whose sender capture pass is degenerate yields no pair and therefore no wealth update (identical in effect to a concordant item), but it still CONSUMES one of the N_max budget items. Registered here as the rule, not chosen after seeing the data.",
            "stopping": "the draw stops at the first item where W_i >= the threshold, per the registered rule; otherwise it runs the full N_max.",
            "never_restarted": "ADR-036/ADR-030: the e-process is never restarted, re-parametrised, or re-run against this rung after seeing W_t. This is the one and only draw.",
            "outcome": if e_pass { "PASS" } else { "FAIL" },
            "crossed": e_pass,
            "crossed_at_item_count": crossed_at,
            "items_drawn": items_drawn,
            "final_wealth": wealth,
            "max_wealth_reached": max_wealth,
            "n_discordant": n_disc,
            "discordant_wins_aligned": wins,
            "discordant_losses_aligned": losses,
            "trajectory": trajectory_json,
            "trajectory_is_complete": "the full W_t path is committed regardless of outcome, per ADR-036's honest-fail path — a reader can see whether the process trended toward the boundary and ran out of budget, or stayed flat.",
        },
        "comparability_discipline": {
            "adr": "ADR-036 Decision 3",
            "no_p_value_translation": "This receipt deliberately reports NO exact-sign or mid-p McNemar p-value for the primary accuracy comparison. The e-process outcome is reported on its own scale (crossing item count, or final wealth at N_max) and is NOT translated into an equivalent p-value: a fixed-sample test's p and a sequential test's stopping wealth answer structurally different questions, and a false equivalence would misrepresent both.",
            "no_completed_rung_redrawn": "No completed rung was re-drawn. M3, M4 x3, M4c, M4d, M4g and M4h Stage 1 stand exactly as originally recorded under the frozen 40-item protocol.",
            "if_this_rung_passes_where_a_similar_one_nulled": "Per ADR-036 Decision 3, a PASS here would NOT be evidence that M4h Stage 1's null was wrong — it would be evidence that a higher-powered instrument, applied to a comparable but not identical configuration, detected an effect the old instrument could not have detected regardless of whether it was present.",
            "nll_statistics_below_are_secondary": "The NLL sign tests reported in the summary are the ladder's standing SECONDARY diagnostic (ADR-030 explicitly rejected a continuous NLL statistic as co-primary: on the one draw with a real signal, Wilcoxon-on-NLL gave p=0.6834 against exact-sign p=0.0312). They are computed on this rung's own item population and are not comparable to any prior rung's 40-item NLL numbers.",
        },
        "item_supply": {
            "adr": "ADR-036 Decision 2",
            "source": "harness/latentmesh-live/data/adaptation-512.json",
            "source_split": "adaptation-512",
            "source_sha256_of_train_jsonl": train_sha,
            "consumption_order": "the file's own fixed ascending index order, sequential, never shuffled and never re-seeded",
            "eval_holdout_lock": "eval-200 / holdout-100 untouched and still mechanically locked (no genome-frozen receipt exists)",
            "frozen_40_item_probe_not_used": "ADR-036 Decision 2(1): the e-process item stream is drawn ENTIRELY from adaptation-512, not from the frozen 40-item probe set, because ADR-030's 'subset by construction' premise was found false (the probe intersects adaptation-512 in exactly one item).",
            "leakage_exclusion": {
                "rule": "ADR-024's 13 probe-overlap items, kept in force by ADR-036 Decision 2(2)",
                "excluded_item_indices": LEAKAGE_EXCLUSIONS,
                "of_those_present_in_adaptation_512": excluded_present,
                "measured_effect_on_this_stream": "computed here rather than assumed; see of_those_present_in_adaptation_512",
            },
            "disclosed_overlap_with_the_historical_probe_set": {
                "items": [1153],
                "note": "index 1153 is the ONE item adaptation-512 shares with the frozen 40-item probe set (ADR-036's own verified figure, re-derived here). It is NOT training leakage: adaptation-512 is fully disjoint from the S2c training pool. ADR-036's registered exclusion rule names only the 13-item list, so 1153 stays in the stream — removing it would be an unregistered protocol change. Disclosed rather than silently kept.",
            },
            "eligible_pool_size": eligible.len(),
            "n_max": N_MAX,
            "tokenization_preflight_exclusions": tokenization_excluded,
            "split_discipline": "item-level, never token-level: each drawn item is a whole GSM8K problem.",
        },
        "items": rows,
        "summary": {
            "headline": if e_pass {
                "E-PROCESS PASS under ADR-036's successor protocol (adaptation-512 stream) — NOT comparable to any frozen-40-item-protocol result"
            } else {
                "E-PROCESS FAIL under ADR-036's successor protocol (adaptation-512 stream) — the wealth boundary was not crossed within N_max; NOT comparable to any frozen-40-item-protocol result"
            },
            "n_evaluated": n,
            "n_degenerate_capture": degenerate,
            "e_process": {
                "outcome": if e_pass { "PASS" } else { "FAIL" },
                "crossed_at_item_count": crossed_at,
                "items_drawn": items_drawn,
                "final_wealth": wealth,
                "max_wealth_reached": max_wealth,
                "wealth_threshold": w_threshold,
                "n_discordant": n_disc,
                "wins": wins, "losses": losses,
            },
            "accuracy": {"aligned_real": real_c, "baseline_uninjected": base_c,
                          "zerovec_injected": zero_c, "random": rand_c,
                          "note": "raw counts over the items actually drawn; levels are NOT comparable to any prior rung (different item population AND a different prompt)"},
            "nll_harm_accounting": {
                "question": "Does delivering an ON-MANIFOLD, de-pooled payload at ORDINARY question tokens move teacher-forced NLL at all — in either direction?",
                "framing": "ADR-024 § MAJOR CORRECTION: the unanimous 0/40 NLL inversion belongs to the OFF-MANIFOLD task-loss rungs only (M4c/M4d/M4g). Every on-manifold rung is inert, within ~0.004 nats of baseline. This rung runs M3's on-manifold weights, so its question is whether the SITE change breaks inertness — not whether it breaks an inversion.",
                "items_where_aligned_nll_is_WORSE_than_baseline": inversion_vs_baseline_items,
                "items_where_aligned_nll_is_WORSE_than_zerovec": inversion_vs_zerovec_items,
                "items_where_aligned_nll_is_WORSE_than_random": inversion_vs_random_items,
                "n_items": n,
                "unanimously_worse_than_baseline": inversion_vs_baseline_items == n && n > 0,
            },
            "fuse_zero_is_noop_vs_baseline": {
                "expected": "EXACT identity — under fuse the zerovec condition is h += 0, at any site",
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
                "criterion": "2 x zerovec_correct >= baseline_correct",
                "pass": zerovec_pass,
                "note": "trivially satisfied because zerovec == baseline under fuse; retained for cross-rung continuity, carries NO evidential weight"},
            "aligned_vs_zerovec_accuracy_counts": {"wins": wins_rz, "losses": loss_rz,
                "note": "raw discordant counts only; no p-value, per the comparability discipline above"},
            "aligned_vs_baseline_accuracy_counts": {"wins": wins_rb, "losses": loss_rb,
                "note": "raw discordant counts only; no p-value, per the comparability discipline above"},
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
            "comparator_receipt_matches_on_every_held_fixed_factor": {"pass": true, "receipt": COMPARATOR_RECEIPT},
            "fuse_mode_recorded": {"pass": true, "mode": INJECT_MODE.tag(), "equation": INJECT_MODE.equation()},
            "site_change_recorded_with_exact_token_positions": {"pass": true,
                "site": SITE.tag(),
                "per_item_positions_recorded": true,
                "note": "every per-item row carries injection_site.positions / .position_token_ids / .positions_decoded; the 8 positions were gated to be contiguous, to end at the last token wholly inside the item's own question, and to decode back to text contained in it (see site_choice_justification.position_resolution for why containment, not a strict suffix)"},
            "manifold_precheck_ran_before_this_draw": {"pass": true, "receipt": PRECHECK_RECEIPT,
                "classification": precheck_class, "gating": "ordering only"},
            "m4g_control_semantics_inherited": {"pass": true, "source": M4G_TRAINING_RECEIPT},
            "slot_count_unchanged": {"pass": true, "slots": N_SLOTS},
            "item_supply_matches_adr_036": {"pass": true,
                "source": "adaptation-512", "order": "fixed index order",
                "leakage_exclusions_applied": LEAKAGE_EXCLUSIONS.len()},
            "M4i_e_process": {"pass": e_pass,
                "crossed_at_item_count": crossed_at,
                "final_wealth": wealth,
                "wealth_threshold": w_threshold,
                "items_drawn": items_drawn,
                "n_discordant": n_disc},
            "zerovec_not_catastrophic": {"pass": zerovec_pass,
                "zerovec_correct": zero_c, "baseline_correct": base_c,
                "degenerate_under_fuse": true,
                "note": "trivially satisfied because zerovec == baseline under fuse; no evidential weight"},
            "fuse_zero_payload_is_a_noop": {"pass": noop_pass,
                "accuracy_disagreements": noop_acc_disagreements,
                "max_abs_nll_delta": noop_max_abs_nll_delta,
                "gating": "none — diagnostic"},
        },
        "gate_pass": e_pass && zerovec_pass,
        "honest_fail_contract": "ADR-032 + ADR-036: ONE registered draw, no retry, no restart, no re-parametrisation. The complete W_t trajectory is committed above whatever the outcome. This receipt is written before any interpretation is added to the ADR.",
        "wall_clock_s": t0.elapsed().as_secs_f32(),
    });
    common::write_receipt(
        &crate_path("receipts"),
        "run2-m4i-receipt-cellL18toL14-mlp-pertokenlast-fuse-questiontail-slots8-eprocess.json",
        &receipt,
    )?;

    println!(
        "\nM4i[question-tail/fuse/de-pooled]: e-process {} — items drawn {items_drawn}, \
         W_final {wealth:.4} (max {max_wealth:.4}, threshold {w_threshold}), n_disc {n_disc} \
         ({wins}W/{losses}L)",
        if e_pass { "PASS" } else { "FAIL" }
    );
    match crossed_at {
        Some(k) => println!("wealth boundary crossed at item {k} of the stream"),
        None => println!(
            "wealth boundary NOT crossed within N_max = {N_MAX}; full trajectory committed"
        ),
    }
    println!("accuracy: aligned {real_c}/{n} baseline {base_c}/{n} zerovec {zero_c}/{n} random {rand_c}/{n}");
    println!(
        "NLL means: aligned {:.4} baseline {:.4} zerovec {:.4} random {:.4}",
        mean(&|q| q.real.1),
        mean(&|q| q.base.1),
        mean(&|q| q.zero.1),
        mean(&|q| q.rand.1)
    );
    println!(
        "NLL sign tests (SECONDARY): aligned-vs-baseline {}W/{}L, aligned-vs-zerovec {}W/{}L, \
         aligned-vs-random {}W/{}L",
        nll_rb.0, nll_rb.1, nll_rz.0, nll_rz.1, nll_rr.0, nll_rr.1
    );
    println!(
        "fuse zero-payload no-op diagnostic: {noop_acc_disagreements} accuracy disagreements, \
         {noop_exact_items}/{n} bit-identical NLLs, max |dNLL| {noop_max_abs_nll_delta:.3e} \
         => {noop_pass} (reported, not gating)"
    );
    println!(
        "PROTOCOL IDENTITY: ADR-036 e-process on the adaptation-512 stream. NOT comparable to any \
         frozen-40-item-protocol result, and no p-value translation is offered."
    );
    Ok(())
}
