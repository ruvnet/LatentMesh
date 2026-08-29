//! **Instrument diagnostic, not a rung.** Is the manifold pre-check's
//! `classify()` verdict invariant to the NUMBER of items measured?
//!
//! # Why this exists
//!
//! PC1's payload classified `item-invariant-but-on-manifold` (surprise=TRUE,
//! flagged as needing resolution). PC1b's payload — the SAME derivation, the
//! same identity transform, the same block-19 gold-teacher-forced receiver
//! state — classified `on-manifold-item-varying` (surprise=FALSE). It is
//! tempting to read that flip as the anomaly resolving. **This binary exists
//! to check whether the flip is real or an artifact of N**, before anyone
//! writes the first reading into an ADR.
//!
//! Two of `classify()`'s three inputs are already near-identical between the
//! two payload sets:
//!
//! | input | PC1 (n=40) | PC1b (n=300) |
//! |---|---|---|
//! | item-invariance mean pairwise cosine | 0.8565 | 0.8696 |
//! | manifold cosine to same-item natural L14-pooled | 0.6869 | 0.6905 |
//! | **top-10 token union** | **98** | **219** |
//!
//! Only the third differs materially, and it is judged against
//! `COLLAPSE_TOKEN_UNION = 120` — an **absolute** count. But a union over
//! `n` items can never exceed `n × TOPK`, so its ceiling is 400 at n=40 and
//! 3000 at n=300. A threshold fixed in absolute terms therefore means
//! something different at every N.
//!
//! # The decisive test
//!
//! Recompute PC1b's token union over the **first 40 items of its own
//! stream** — matched to PC1's N — and compare to PC1's 98. If PC1b's
//! 40-item union also lands at or below the 120 threshold, the flip was
//! **purely an N artifact** and the two payloads are geometrically the same
//! family. If it lands well above, the payloads genuinely differ and the
//! anomaly really is resolved.
//!
//! The full union-vs-n curve is reported alongside, so the saturation shape
//! is visible rather than inferred from two points.
//!
//! **This changes no verdict and gates nothing.** It does not touch, rewrite
//! or invalidate the pre-draw pre-check receipt — it writes its own file, and
//! it is deliberately run AFTER the PC1b draw so the "pre-check ran before
//! the draw" ordering stays clean.
//!
//! CPU-ONLY (ADR-034 lane rule).
//!
//! Run: cargo run --release --example run2_pc1b_precheck_n_invariance

#![recursion_limit = "512"]

#[path = "common/mod.rs"]
#[allow(dead_code)]
mod common;

use candle_core::{DType, Device};
use common::lens::{
    classify, cosine, dominant_token_share, mean, mean_pairwise_cosine, project_batch, rms_norm,
    top_k, COLLAPSE_COSINE, COLLAPSE_TOKEN_UNION, OFF_MANIFOLD_COSINE,
};
use common::m3::RECEIVER;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

const PC1B_CAPTURE_RECEIPT: &str = "receipts/run2-pc1b-capture-receipt.json";
const PC1_PRECHECK_RECEIPT: &str = "receipts/run2-pc1-manifold-precheck-receipt.json";
const D_RECEIVER: usize = 1536;
const VECS_PER_ITEM: usize = 3;
const TOPK: usize = 10;
/// PC1's N — the matched subsample size that makes the comparison like-for-like.
const PC1_N: usize = 40;

fn crate_path(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

fn main() -> anyhow::Result<()> {
    let t0 = std::time::Instant::now();
    anyhow::ensure!(
        !cfg!(feature = "cuda"),
        "ADR-034 lane rule: build this diagnostic WITHOUT --features cuda"
    );
    let device = Device::Cpu;
    println!("lane gate: CPU-only (cuda feature off, Device::Cpu)");

    // ---- PC1b's payload, via its capture receipt --------------------------
    let cap: serde_json::Value =
        serde_json::from_slice(&std::fs::read(crate_path(PC1B_CAPTURE_RECEIPT))?)?;
    let payload_path = PathBuf::from(
        cap["payload_file"]["path"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("capture receipt does not name the payload file"))?,
    );
    let want_sha = cap["payload_file"]["sha256"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("capture receipt does not pin a payload sha256"))?;
    let n_items: usize = cap["payload_file"]["n_items"].as_u64().unwrap_or(0) as usize;
    let bytes = std::fs::read(&payload_path)?;
    let got_sha = common::sha256_hex(&bytes);
    anyhow::ensure!(got_sha == want_sha, "payload sha256 mismatch");
    let flat: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let payloads: Vec<Vec<f32>> = (0..n_items)
        .map(|i| {
            let b = i * VECS_PER_ITEM * D_RECEIVER;
            flat[b..b + D_RECEIVER].to_vec()
        })
        .collect();
    let l14_pooled: Vec<Vec<f32>> = (0..n_items)
        .map(|i| {
            let b = i * VECS_PER_ITEM * D_RECEIVER + D_RECEIVER;
            flat[b..b + D_RECEIVER].to_vec()
        })
        .collect();
    println!("PC1b payload VERIFIED: sha {got_sha} ({n_items} items)");

    // ---- Receiver unembedding (CPU), same as the pre-check ----------------
    let api = hf_hub::api::sync::Api::new()?;
    let repo = api.model(RECEIVER.to_string());
    let weights = repo.get("model.safetensors")?;
    let cfg: serde_json::Value = serde_json::from_slice(&std::fs::read(repo.get("config.json")?)?)?;
    let rms_eps = cfg["rms_norm_eps"].as_f64().unwrap_or(1e-6);
    let st = unsafe { candle_core::safetensors::MmapedSafetensors::new(&weights)? };
    let unembed = st
        .load("model.embed_tokens.weight", &device)?
        .to_dtype(DType::F32)?;
    let final_gain: Vec<f32> = st
        .load("model.norm.weight", &device)?
        .to_dtype(DType::F32)?
        .to_vec1()?;

    // Per-item top-10 sets under the RMSNorm lens — the pre-check's own metric.
    let mut top10: Vec<Vec<u32>> = Vec::with_capacity(n_items);
    for p in &payloads {
        let batch = vec![rms_norm(p, &final_gain, rms_eps)];
        let logits = project_batch(&unembed, &batch, &device)?;
        top10.push(top_k(&logits[0], TOPK));
    }

    // ---- The union-vs-n curve ---------------------------------------------
    let union_at = |n: usize| -> usize {
        let mut s: BTreeSet<u32> = BTreeSet::new();
        for row in top10.iter().take(n) {
            s.extend(row.iter().copied());
        }
        s.len()
    };
    let sizes = [10usize, 20, 40, 60, 100, 150, 200, 250, 300];
    let curve: Vec<serde_json::Value> = sizes
        .iter()
        .filter(|&&n| n <= n_items)
        .map(|&n| {
            let u = union_at(n);
            serde_json::json!({
                "n_items": n,
                "top10_token_union": u,
                "max_possible": n * TOPK,
                "fraction_of_max": u as f64 / (n * TOPK) as f64,
                "trips_collapse_rule_at_absolute_threshold": u <= COLLAPSE_TOKEN_UNION,
            })
        })
        .collect();

    // N-invariant support statistic, evaluated over the first `n` items.
    // `union_at` is retained ABOVE only to document the retired rule's
    // saturation curve — it no longer feeds any verdict.
    let dominant_at = |n: usize| -> f64 { dominant_token_share(&top10[..n.min(top10.len())]) };

    // ---- The matched-N verdict: PC1b restricted to its first 40 items -----
    let u40 = union_at(PC1_N);
    let (inv40, _, _) = mean_pairwise_cosine(&payloads[..PC1_N]);
    let man40 = mean(
        &(0..PC1_N)
            .map(|i| cosine(&payloads[i], &l14_pooled[i]))
            .collect::<Vec<f64>>(),
    );
    let class40 = classify(inv40, dominant_at(PC1_N), man40);

    let (inv_all, _, _) = mean_pairwise_cosine(&payloads);
    let man_all = mean(
        &(0..n_items)
            .map(|i| cosine(&payloads[i], &l14_pooled[i]))
            .collect::<Vec<f64>>(),
    );
    let class_all = classify(inv_all, dominant_at(n_items), man_all);

    // ---- PC1's committed numbers, read not retyped ------------------------
    let pc1: serde_json::Value =
        serde_json::from_slice(&std::fs::read(crate_path(PC1_PRECHECK_RECEIPT))?)?;
    let pc1_row = pc1["candidates"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("PC1 pre-check has no candidates"))?
        .iter()
        .find(|c| {
            c["label"].as_str() == Some("pc1-payload-receiver-L19-lasttoken-goldteacherforced")
        })
        .ok_or_else(|| anyhow::anyhow!("PC1 pre-check has no payload row"))?;
    let pc1_union = pc1_row["output_token_support_rmsnorm_lens"]
        ["distinct_tokens_in_union_of_40_top10_sets"]
        .as_u64()
        .unwrap_or(0) as usize;
    let pc1_class = pc1_row["classification"].as_str().unwrap_or("<none>");

    let flip_is_an_n_artifact = class40 != class_all && class40 == pc1_class;

    println!("\n=== token union vs n (PC1b payload, same metric as the pre-check) ===");
    for row in &curve {
        println!(
            "  n={:>3}  union={:>4} / {:>4} max  ({:.3} of max)  trips-collapse={}",
            row["n_items"],
            row["top10_token_union"],
            row["max_possible"],
            row["fraction_of_max"].as_f64().unwrap_or(0.0),
            row["trips_collapse_rule_at_absolute_threshold"]
        );
    }
    println!(
        "\nPC1  (n=40) : union {pc1_union:>4}  => {pc1_class}\n\
         PC1b (n=40) : union {u40:>4}  => {class40}   [matched-N, the like-for-like comparison]\n\
         PC1b (n={n_items}): union {:>4}  => {class_all}",
        union_at(n_items)
    );
    println!(
        "\nflip_is_an_n_artifact = {flip_is_an_n_artifact} \
         (true means PC1b classifies exactly as PC1 did once N is matched)"
    );

    let receipt = serde_json::json!({
        "stage": "run2-PC1b-precheck-N-invariance-diagnostic",
        "what_this_is": "An INSTRUMENT diagnostic, not a rung. It changes no verdict, gates nothing, and does not touch the pre-draw pre-check receipt. Run AFTER the PC1b draw so the pre-check ordering discipline stays clean.",
        "question": "PC1's payload classified item-invariant-but-on-manifold (surprise=TRUE); PC1b's SAME-derivation payload classified on-manifold-item-varying (surprise=FALSE). Is that flip a real geometric difference, or an artifact of measuring 300 items instead of 40?",
        "why_it_matters": "If it is an artifact, then (a) PC1's surprise flag is NOT resolved by PC1b, and (b) classify()'s verdict is not comparable across any two draws of different N — which affects every cross-rung use of that label in ADR-024.",
        "method": "Recompute the pre-check's own top-10-token-union metric over the FIRST 40 items of PC1b's stream — matched to PC1's N — and re-run the same classify() on the matched-N triple. The full union-vs-n curve is reported so the saturation shape is visible rather than inferred.",
        "the_mechanism_under_test": {
            "threshold": "COLLAPSE_TOKEN_UNION",
            "value": COLLAPSE_TOKEN_UNION,
            "form": "ABSOLUTE count, compared against a union whose ceiling is n * TOPK",
            "ceiling_at_n40": PC1_N * TOPK,
            "ceiling_at_n300": n_items * TOPK,
            "consequence": "a fixed absolute threshold encodes a different strictness at every N; it can only be tripped by a small-N draw",
        },
        "matched_n_comparison": {
            "pc1_n": PC1_N,
            "pc1_token_union": pc1_union,
            "pc1_classification": pc1_class,
            "pc1b_token_union_first_40_items": u40,
            "pc1b_item_invariance_first_40": inv40,
            "pc1b_manifold_cosine_first_40": man40,
            "pc1b_classification_at_matched_n": class40,
            "pc1b_classification_at_full_n": class_all,
            "pc1b_token_union_full_n": union_at(n_items),
            "flip_is_an_n_artifact": flip_is_an_n_artifact,
            "reading": if flip_is_an_n_artifact {
                "ARTIFACT: at matched N, PC1b's payload classifies exactly as PC1's did. The two payloads are the same geometric family and the classification flip carries no information about payload quality. PC1's surprise flag is NOT resolved by PC1b, and classify()'s label must not be compared across draws of different N."
            } else {
                "NOT a pure N artifact: at matched N the classification still differs from PC1's, so something about the payload sets genuinely differs and the flip deserves a substantive explanation."
            },
        },
        "token_union_vs_n_curve": curve,
        "thresholds_registered": {
            "collapse_mean_pairwise_cosine_at_or_above": COLLAPSE_COSINE,
            "collapse_top10_token_union_at_or_below": COLLAPSE_TOKEN_UNION,
            "off_manifold_mean_cosine_to_natural_below": OFF_MANIFOLD_COSINE,
            "source": "common/lens.rs — unmodified; this diagnostic reuses classify() rather than reimplementing it",
        },
        "payload_source": {
            "capture_receipt": PC1B_CAPTURE_RECEIPT,
            "payload_file": payload_path.display().to_string(),
            "sha256": got_sha,
        },
        "gating": "NONE. This receipt annotates the instrument; it decides nothing about PC1, PC1b, or any rung's verdict.",
        "wall_clock_s": t0.elapsed().as_secs_f32(),
    });
    common::write_receipt(
        &crate_path("receipts"),
        "run2-pc1b-precheck-n-invariance-receipt.json",
        &receipt,
    )?;
    Ok(())
}
