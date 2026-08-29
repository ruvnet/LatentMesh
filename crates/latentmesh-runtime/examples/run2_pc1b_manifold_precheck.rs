//! Run-2 **PC1b manifold pre-check** — the geometry of the positive
//! control's payload over the e-process stream, on the record BEFORE the draw.
//!
//! ADR-024 § "CORRECTION + a critical interaction between M4i and PC1
//! (2026-08-29)" ¶3 (which registers PC1b); `docs/research/045` §5's mandatory
//! gates ("item-invariance/entropy check (`docs/research/036`'s triple)"), run
//! on the payload `run2_pc1b_capture` wrote.
//!
//! **This is PC1's pre-check, unchanged in method** — same `common/lens.rs`
//! metric kit, same registered thresholds, same `classify()`. It reads a
//! different capture receipt and a payload covering 300 items instead of 40;
//! nothing about the measurement changes.
//!
//! **The expected result is trivial, and that is the point.** PC1b's payload
//! is not an adapter's output — it *is* a real receiver residual-stream state
//! (block 19, gold-teacher-forced, captured from the receiver itself). It must
//! therefore land squarely on-manifold and item-varying. **If this pre-check
//! says otherwise, that is itself a finding about the pipeline** — either the
//! metric's reference is mis-specified, or the pre-check has been mis-reading
//! every adapter it has classified — and it is reported as such rather than
//! waved through.
//!
//! **One caveat is registered up front, before the numbers exist.** The
//! `manifold` metric compares the emitted vector to the receiver's own
//! **pooled L14** state for the same item. PC1b's payload is an **un-pooled
//! L19** state. Two mismatches are therefore baked into that number — depth
//! (19 vs 14) and pooling (single row vs span mean) — and ADR-024 already
//! records that a pooled natural state sits only ~0.667 cosine from a real
//! un-pooled one. So the like-for-like comparators are reported *alongside*
//! it, from the same capture: the item's own L14 pooled AND L14 last-row
//! references travel in the payload file for exactly this reason.
//!
//! **CPU-ONLY, ANNOTATES ONLY** (ADR-034 lane rule): the only model-derived
//! object touched is the receiver's tied unembedding plus the final RMSNorm
//! gain, on CPU. It can therefore run while a CUDA lane is busy.
//!
//! **FIREWALL (`docs/research/045` §3, verbatim in the receipt):** a PASS on
//! PC1b proves **liveness only, never transfer**.
//!
//! Run (CPU, default features):
//!   cargo run --release --example run2_pc1b_manifold_precheck

#![recursion_limit = "512"]

#[path = "common/mod.rs"]
#[allow(dead_code)]
mod common;

use candle_core::{DType, Device};
use common::lens::{
    classify, cosine, entropy_nats, logsumexp, mean, mean_pairwise_cosine, mean_resultant_length,
    minmax, project_batch, rms_norm, token_set_stats, top_k, COLLAPSE_COSINE, COLLAPSE_TOKEN_UNION,
    OFF_MANIFOLD_COSINE,
};
use common::m3::RECEIVER;
use latentmesh_runtime::norms;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

const CAPTURE_RECEIPT: &str = "receipts/run2-pc1b-capture-receipt.json";
const D_RECEIVER: usize = 1536;
const VECS_PER_ITEM: usize = 3;
const TOPK: usize = 10;
const N_DOMINANT: usize = 8;

fn crate_path(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

/// The three vectors the capture stored per item.
struct Vecs {
    l19_last: Vec<f32>,
    l14_pooled: Vec<f32>,
    l14_last: Vec<f32>,
}

/// Per-(candidate, item) measurements, mirroring `run2_manifold_precheck`.
#[derive(Default, Clone)]
struct ItemRow {
    l2: f64,
    cos_to_natural_same_item: f64,
    top10: Vec<u32>,
    entropy_rmsnorm: f64,
    entropy_plain: f64,
    gold_rank_rmsnorm: f64,
    gold_best_rank_rmsnorm: usize,
}

#[allow(clippy::too_many_lines)]
fn main() -> anyhow::Result<()> {
    let t0 = std::time::Instant::now();

    // ---- Lane gate: CPU-only, no CUDA workload (ADR-034) ------------------
    anyhow::ensure!(
        !cfg!(feature = "cuda"),
        "ADR-034 lane rule: build this pre-check WITHOUT --features cuda"
    );
    let device = Device::Cpu;
    anyhow::ensure!(device.is_cpu(), "device must be CPU");
    println!("lane gate: CPU-only (cuda feature off, Device::Cpu)");

    // ---- Gate: the capture receipt, and the payload file it pins ----------
    let cap: serde_json::Value =
        serde_json::from_slice(&std::fs::read(crate_path(CAPTURE_RECEIPT)).map_err(|e| {
            anyhow::anyhow!(
                "capture receipt {CAPTURE_RECEIPT} unreadable ({e}) — run run2_pc1b_capture first"
            )
        })?)?;
    anyhow::ensure!(
        cap["gates"]["pc1_overlap_item_bit_identical"]["pass"].as_bool() == Some(true),
        "the capture receipt does not record the PC1 bit-identity gate"
    );
    anyhow::ensure!(
        cap["gates"]["stream_identical_to_m4i"]["pass"].as_bool() == Some(true),
        "the capture receipt does not record that the stream is M4i's"
    );
    anyhow::ensure!(
        cap["gates"]["identity_transform_no_adapter_weights_loaded"]["pass"].as_bool()
            == Some(true),
        "the capture receipt does not record the identity-transform gate"
    );
    let payload_path = PathBuf::from(
        cap["payload_file"]["path"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("capture receipt does not name the payload file"))?,
    );
    let want_sha = cap["payload_file"]["sha256"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("capture receipt does not pin the payload sha256"))?;
    let n_items: usize = cap["payload_file"]["n_items"]
        .as_u64()
        .ok_or_else(|| anyhow::anyhow!("capture receipt does not record n_items"))?
        as usize;
    let bytes = std::fs::read(&payload_path)?;
    let got_sha = common::sha256_hex(&bytes);
    anyhow::ensure!(
        got_sha == want_sha,
        "payload file sha256 {got_sha} != capture-receipt-pinned {want_sha}"
    );
    anyhow::ensure!(
        bytes.len() == n_items * VECS_PER_ITEM * D_RECEIVER * 4,
        "payload file has {} bytes, expected {}",
        bytes.len(),
        n_items * VECS_PER_ITEM * D_RECEIVER * 4
    );
    let flat: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let vecs: Vec<Vecs> = (0..n_items)
        .map(|i| {
            let base = i * VECS_PER_ITEM * D_RECEIVER;
            let g = |k: usize| flat[base + k * D_RECEIVER..base + (k + 1) * D_RECEIVER].to_vec();
            Vecs {
                l19_last: g(0),
                l14_pooled: g(1),
                l14_last: g(2),
            }
        })
        .collect();
    let items: Vec<usize> = cap["dataset"]["indices"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("capture receipt has no dataset.indices"))?
        .iter()
        .filter_map(|v| v.as_u64())
        .map(|v| v as usize)
        .collect();
    anyhow::ensure!(items.len() == n_items, "capture receipt indices count");
    let golds: Vec<String> = cap["items"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("capture receipt has no items array"))?
        .iter()
        .map(|r| r["gold"].as_str().unwrap_or_default().to_string())
        .collect();
    anyhow::ensure!(golds.len() == n_items, "capture receipt gold count");
    println!("payload VERIFIED against {CAPTURE_RECEIPT}: sha {got_sha}");

    // ---- Receiver unembedding (CPU) ---------------------------------------
    let api = hf_hub::api::sync::Api::new()?;
    let repo = api.model(RECEIVER.to_string());
    let weights = repo.get("model.safetensors")?;
    let cfg: serde_json::Value = serde_json::from_slice(&std::fs::read(repo.get("config.json")?)?)?;
    anyhow::ensure!(
        cfg["tie_word_embeddings"].as_bool() == Some(true),
        "receiver is not tied-embedding"
    );
    let rms_eps = cfg["rms_norm_eps"].as_f64().unwrap_or(1e-6);
    let tokenizer = tokenizers::Tokenizer::from_file(repo.get("tokenizer.json")?)
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    let st = unsafe { candle_core::safetensors::MmapedSafetensors::new(&weights)? };
    let unembed = st
        .load("model.embed_tokens.weight", &device)?
        .to_dtype(DType::F32)?;
    let final_gain: Vec<f32> = st
        .load("model.norm.weight", &device)?
        .to_dtype(DType::F32)?
        .to_vec1()?;
    let (vocab, hidden) = unembed.dims2()?;
    anyhow::ensure!(hidden == D_RECEIVER, "unembedding hidden {hidden}");
    println!("unembedding: [{vocab} x {hidden}] f32 on CPU, tied; rms_norm_eps {rms_eps}");

    // ---- Candidates: the payload, plus the two natural references ---------
    // Same shape as run2_manifold_precheck's candidate table, so the rows are
    // directly comparable to every adapter it has classified.
    let labels = [
        "pc1b-payload-receiver-L19-lasttoken-goldteacherforced",
        "reference-receiver-L14-pooled",
        "reference-receiver-L14-single-row",
    ];
    let emitted: Vec<Vec<Vec<f32>>> = vec![
        vecs.iter().map(|v| v.l19_last.clone()).collect(),
        vecs.iter().map(|v| v.l14_pooled.clone()).collect(),
        vecs.iter().map(|v| v.l14_last.clone()).collect(),
    ];
    let n_cand = labels.len();

    let mut per_item: Vec<Vec<ItemRow>> = vec![Vec::with_capacity(n_items); n_cand];
    for (i, v) in vecs.iter().enumerate() {
        let vs: Vec<Vec<f32>> = (0..n_cand).map(|k| emitted[k][i].clone()).collect();
        let mut batch: Vec<Vec<f32>> = vs.clone();
        batch.extend(vs.iter().map(|x| rms_norm(x, &final_gain, rms_eps)));
        let logits = project_batch(&unembed, &batch, &device)?;
        let gold_toks: Vec<u32> = tokenizer
            .encode(format!("#### {}", golds[i]), false)
            .map_err(|e| anyhow::anyhow!("encode: {e}"))?
            .get_ids()
            .to_vec();
        for (k, x) in vs.iter().enumerate() {
            let z_plain = &logits[k];
            let z_norm = &logits[k + n_cand];
            let lse_n = logsumexp(z_norm);
            let g_n = token_set_stats(z_norm, &gold_toks, lse_n);
            per_item[k].push(ItemRow {
                l2: f64::from(norms::l2(x)),
                // The registered `manifold` metric: cosine to the receiver's
                // OWN pooled L14 state for the same item.
                cos_to_natural_same_item: cosine(x, &v.l14_pooled),
                top10: top_k(z_norm, TOPK),
                entropy_rmsnorm: entropy_nats(z_norm),
                entropy_plain: entropy_nats(z_plain),
                gold_rank_rmsnorm: g_n.mean_rank,
                gold_best_rank_rmsnorm: g_n.best_rank,
            });
        }
    }

    // Like-for-like comparators for the payload, which the registered metric
    // cannot supply (depth AND pooling both differ from the L14-pooled ref).
    let cos_payload_to_l14_last: Vec<f64> = vecs
        .iter()
        .map(|v| cosine(&v.l19_last, &v.l14_last))
        .collect();
    let cos_l14_last_to_l14_pooled: Vec<f64> = vecs
        .iter()
        .map(|v| cosine(&v.l14_last, &v.l14_pooled))
        .collect();

    let mut table = Vec::new();
    let mut payload_verdict = String::new();
    for (k, label) in labels.iter().enumerate() {
        let rows = &per_item[k];
        let col = |f: fn(&ItemRow) -> f64| -> Vec<f64> { rows.iter().map(f).collect() };
        let (inv_mean, inv_min, inv_max) = mean_pairwise_cosine(&emitted[k]);
        let resultant = mean_resultant_length(&emitted[k]);
        let mut counts: BTreeMap<u32, usize> = BTreeMap::new();
        for r in rows {
            for &t in &r.top10 {
                *counts.entry(t).or_default() += 1;
            }
        }
        let token_union = counts.len();
        let mut ordered: Vec<(u32, usize)> = counts.into_iter().collect();
        ordered.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        let dominant: Vec<serde_json::Value> = ordered
            .iter()
            .take(N_DOMINANT)
            .map(|(t, n)| {
                serde_json::json!({
                    "token_id": t,
                    "decoded": tokenizer.decode(&[*t], false).unwrap_or_else(|_| format!("<{t}>")),
                    "in_top10_of_n_items": n,
                })
            })
            .collect();
        let manifold_cos = mean(&col(|r| r.cos_to_natural_same_item));
        let gold_n = mean(&col(|r| r.gold_rank_rmsnorm));
        let verdict = classify(inv_mean, token_union, manifold_cos);
        if k == 0 {
            payload_verdict = verdict.to_string();
        }
        table.push(serde_json::json!({
            "label": label,
            "family": if k == 0 { "pc1b-identity-no-adapter" } else { "reference-natural-receiver-state" },
            "training": "NONE — identity; this vector IS a captured receiver state",
            "item_invariance": {
                "mean_pairwise_cosine_between_emitted_vectors": inv_mean,
                "min": inv_min, "max": inv_max,
                "mean_resultant_length_of_unit_vectors": resultant,
            },
            "output_token_support_rmsnorm_lens": {
                "distinct_tokens_in_union_of_all_item_top10_sets": token_union,
                "max_possible": n_items * TOPK,
                "dominant_tokens": dominant,
            },
            "gold_answer_tokens": {
                "mean_rank_rmsnorm": gold_n,
                "vocab_percentile_rmsnorm": gold_n / vocab as f64,
                "mean_best_rank_rmsnorm": mean(&rows.iter().map(|r| r.gold_best_rank_rmsnorm as f64).collect::<Vec<_>>()),
            },
            "entropy_nats": {
                "rmsnorm_lens_mean": mean(&col(|r| r.entropy_rmsnorm)),
                "plain_lens_mean": mean(&col(|r| r.entropy_plain)),
                "uniform": (vocab as f64).ln(),
            },
            "manifold": {
                "mean_cosine_to_same_item_natural_receiver_L14_pooled": manifold_cos,
                "emitted_l2": {"mean": mean(&col(|r| r.l2)),
                                "min": minmax(&col(|r| r.l2)).0, "max": minmax(&col(|r| r.l2)).1},
            },
            "classification": verdict,
        }));
        println!(
            "{label:56} inv-cos {inv_mean:7.4}  tokens {token_union:>3}/{}  manifold-cos \
             {manifold_cos:7.4}  H {:5.2}  => {verdict}",
            n_items * TOPK,
            mean(&col(|r| r.entropy_rmsnorm)),
        );
    }

    // The registered expectation, stated before the numbers were produced.
    let payload_on_manifold_item_varying = payload_verdict == "on-manifold-item-varying";
    let surprise = !payload_on_manifold_item_varying;

    let receipt = serde_json::json!({
        "stage": "run2-PC1b-manifold-precheck",
        "design": "docs/adr/024-run2-trained-thought-adapter-ladder.md § 'PC1 PRE-REGISTRATION (2026-08-29)'; docs/research/045 §5's mandatory item-invariance/entropy gate (docs/research/036's triple)",
        "env": {
            "evidence_label": "deterministic CPU analysis over the sha-pinned PC1b payload — no probe draw, annotates only",
            "unembedding": format!("{RECEIVER} model.embed_tokens.weight (tied), f32 on CPU"),
            "vocab": vocab, "hidden": hidden, "rms_norm_eps": rms_eps,
        },
        "firewall_verbatim_research_045_section_3": "This control used a same-model, same-item, identity-transform injection with gold-adjacent content. It tests whether this repository's injection mechanics (delivery operator, payload shape, injection site, norm-rescale) are capable of carrying signal at all. It does not test, and must not be cited as evidence for or against, whether a cross-model-derived, learned-alignment payload can transfer reasoning content — that is exactly the separate question M3 through M5X exist to answer.",
        "liveness_not_transfer": "A PASS on PC1b proves LIVENESS ONLY, NEVER TRANSFER, and may never be cited as evidence for the transfer claim.",
        "payload_source": {
            "capture_receipt": CAPTURE_RECEIPT,
            "payload_file": payload_path.display().to_string(),
            "sha256": got_sha,
            "derivation": "receiver's OWN block-19 state, gold-teacher-forced, LAST token of the span — identity transform, no adapter",
        },
        "registered_expectation_stated_before_the_numbers": {
            "expected": "on-manifold-item-varying — trivially, because the payload IS a real receiver residual-stream state rather than an adapter's output",
            "if_it_is_not": "a finding about the PIPELINE, not about PC1b: either the registered manifold metric's reference is mis-specified for un-pooled/other-depth states, or the pre-check has been mis-reading the adapters it has classified. Reported, not waved through.",
            "observed": payload_verdict,
            "matches_expectation": payload_on_manifold_item_varying,
            "surprise_flag": surprise,
        },
        "registered_metric_caveat": {
            "metric": "mean_cosine_to_same_item_natural_receiver_L14_pooled",
            "why_it_understates_here": "PC1b's payload is an UN-POOLED L19 state; the metric's reference is a POOLED L14 state. BOTH depth (19 vs 14) and pooling (single row vs span mean) differ. ADR-024 already records that a pooled natural state sits only ~0.667 cosine from a real un-pooled one, so this number is not a clean on/off-manifold reading for this payload.",
            "like_for_like_comparators_from_the_same_capture": {
                "mean_cosine_payload_L19_last_to_same_item_L14_last": mean(&cos_payload_to_l14_last),
                "mean_cosine_L14_last_to_same_item_L14_pooled": mean(&cos_l14_last_to_l14_pooled),
                "reading": "the second number isolates the POOLING half of the mismatch using two states the receiver genuinely produced, so the first can be read against it rather than against 1.0",
            },
        },
        "thresholds_registered": {
            "collapse_mean_pairwise_cosine_at_or_above": COLLAPSE_COSINE,
            "collapse_top10_token_union_at_or_below": COLLAPSE_TOKEN_UNION,
            "off_manifold_mean_cosine_to_natural_below": OFF_MANIFOLD_COSINE,
            "source": "common/lens.rs — the same constants and the same classify() every prior pre-check used, reused rather than copied",
        },
        "gating": "NONE — ORDERING ONLY, per ADR-024's M4f framing. This receipt must EXIST and must have measured this exact payload sha before the PC1 draw; its classification decides nothing about the verdict.",
        "candidates": table,
        "dataset": {"indices": items},
        "wall_clock_s": t0.elapsed().as_secs_f32(),
    });
    common::write_receipt(
        &crate_path("receipts"),
        "run2-pc1b-manifold-precheck-receipt.json",
        &receipt,
    )?;
    println!(
        "PC1b pre-check: payload classified {payload_verdict} (expected on-manifold-item-varying; \
         surprise={surprise}) in {:.1}s",
        t0.elapsed().as_secs_f32()
    );
    Ok(())
}
