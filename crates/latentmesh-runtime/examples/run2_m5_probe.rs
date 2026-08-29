//! Run-2 **M5** — receiver-side adaptation, the draw
//! (`docs/adr/045-m5-receiver-side-adaptation-pre-registration.md`), evaluated
//! under ADR-036's successor-rung e-process. `run2_m4i_probe` lineage.
//!
//! **Why this rung exists.** The activation-injection ladder is closed with a
//! scoped negative: direct injection into a **frozen** receiver is
//! decision-inert at one and two layers, on a powered test. Every rung to date
//! trained the *payload* to suit a frozen receiver; **none trained the
//! receiver to accept the payload**. That asymmetry is the last untested axis,
//! and arXiv:2606.05711's Table 5 finds all 18 surveyed latent-communication
//! methods leave the receiver's own weights frozen.
//!
//! **THE SINGLE CHANGED FACTOR, and its comparator.** This rung changes the
//! **receiver** and nothing else, measured against **M4i**. Held fixed against
//! that receipt, and asserted at run time rather than claimed in prose: the
//! payload **artifact** (M3's trained on-manifold reconstruction MLP,
//! byte-for-byte); the payload **derivation** (`Variant::PerTokenLast`); the
//! **operator** (`InjectionMode::Fuse`); the **site** (`Site::QuestionTail`);
//! the **slot count** (8); the **depth** (block 14, cell L18→L14); the
//! **rescale rule**; the **decoding** (greedy, batch = 1, ≤ 400 new tokens);
//! the four **conditions**; the **item stream** (`adaptation-512`, fixed index
//! order); and the **statistic** (λ = 0.30, PASS at W ≥ 20, N_max = 300).
//!
//! Changed: the receiver carries a trained rank-`r` additive LoRA on its
//! residual stream after block 14.
//!
//! **The confound, and why the primary is immune.** ADR-045 flags it: a
//! task-loss-adapted receiver could simply become a better GSM8K solver,
//! improving regardless of injected content. Therefore **`baseline` is
//! re-measured on this adapted receiver** — no frozen-receiver rung's baseline
//! is reused anywhere in this receipt. But the registered primary is
//! **`aligned` vs `random`**, and both arms run on the same adapted receiver,
//! so a general fine-tuning effect raises both and cancels. The primary is
//! immune; `aligned` vs `baseline` is secondary and is not.
//!
//! **The inherited hazard, declared before the draw.** M4c/M4d/M4g trained the
//! payload through this same task loss and produced a reproducible 0W/40L NLL
//! inversion, diagnosed as OFF-MANIFOLD rather than destructive. M5 trains the
//! receiver instead, so it is a different experiment — but if `aligned`
//! inverts the same way here, ADR-045 registers the first hypothesis as
//! off-manifold input, **not** a channel effect, and the receipt says so.
//!
//! **ΔV** is not computed here and never was a training signal
//! (`docs/research/034` §3 prices one powered `verify_edge` draw at ~3 GPU-h).
//!
//! The e-process state, the per-condition accounting and the receipt literal
//! live in `common/m5_draw.rs` (file-size discipline); what stays here is the
//! gate sequence and the draw loop, which is what a reader audits against
//! ADR-045.
//!
//! ONE draw per rank. No retry (ADR-032 honest-fail).
//!
//! Run: PATH=/usr/local/cuda-12.8/bin:$PATH \
//!      cargo run --release --features cuda --example run2_m5_probe -- [rank]

// The receipt in `common/m5_draw.rs` is one large `serde_json::json!` literal
// (the frozen-receipt house style: one object, auditable top to bottom). It
// expands past rustc's default 128-deep macro recursion limit.
#![recursion_limit = "1024"]

#[path = "common/mod.rs"]
#[allow(dead_code)]
mod common;

// Declared HERE rather than inside `common/mod.rs`: that module is
// `#[path]`-included into every example in the crate, and this one carries the
// receipt literal, whose `json!` expansion exceeds the default macro recursion
// limit. Keeping it out of `common` means no other example pays for it.
#[path = "common/m5_draw.rs"]
mod m5_draw;
#[path = "common/m5_receipt.rs"]
mod m5_receipt;

use common::m3::{
    build_site_prompt, run_item_at, Site, Variant, GOLDEN_REL_TOL, RANDVEC_SEED_BASE, RECEIVER,
    RECEIVER_BLOCK, SENDER,
};
use common::mlp::MlpTransform;
use latentmesh_runtime::inject::InjectionMode;
use latentmesh_runtime::QwenRuntime;
use m5_draw::DrawOutcome;
use m5_draw::{E_ALPHA, LAMBDA, N_MAX};
use m5_receipt::{receipt, ReceiptCtx};
use std::path::{Path, PathBuf};

const GSM8K_TRAIN_SHA256: &str = "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465";
/// M3's training receipt — the payload artifact's freeze point (2026-08-28).
const TRAINING_RECEIPT: &str = "receipts/run2-m3-training-receipt-cellL18toL14.json";
/// The comparator this rung isolates its single changed factor against.
const COMPARATOR_RECEIPT: &str =
    "receipts/run2-m4i-receipt-cellL18toL14-mlp-pertokenlast-fuse-questiontail-slots8-eprocess.json";
const ARTIFACT: &str = "receipts/run2-m3-mlp-cellL18toL14.f32bin";
const GOLDEN: &str = "receipts/run2-m3-golden-mlp-cellL18toL14.json";
const ADAPTATION_512: &str = "../../harness/latentmesh-live/data/adaptation-512.json";

/// UNCHANGED from M4i — none of these is this rung's variable.
const VARIANT: Variant = Variant::PerTokenLast;
const INJECT_MODE: InjectionMode = InjectionMode::Fuse;
const SITE: Site = Site::QuestionTail;

/// ADR-024's 13 probe-overlap items, kept in force by ADR-036 Decision 2(2).
const LEAKAGE_EXCLUSIONS: [usize; 13] = [
    150, 1309, 1573, 2365, 2418, 2540, 2958, 2973, 3084, 3825, 3844, 3877, 4746,
];

fn crate_path(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

fn main() -> anyhow::Result<()> {
    let t0 = std::time::Instant::now();
    let nvcc = latentmesh_runtime::assert_cuda_build_env().map_err(anyhow::Error::msg)?;
    println!("build-env guard: {nvcc}");
    let w_threshold = 1.0 / E_ALPHA;
    let rank: usize = match std::env::args().nth(1) {
        Some(a) => a.parse()?,
        None => 1,
    };
    // SMOKE MODE (see the trainer): `LM_M5_SMOKE=n` reads the M5 artifacts
    // from the gitignored target/ tree, caps the stream at n items, and writes
    // a receipt whose name and contents mark it as NOT the registered draw.
    // The registered draw is the invocation with the variable unset. A smoke
    // run is not a peek at it either: it runs against a DIFFERENT, smoke-
    // trained adapter, so its outcomes carry no information about the real one.
    let smoke: Option<usize> = std::env::var("LM_M5_SMOKE")
        .ok()
        .map(|v| v.parse::<usize>())
        .transpose()?;
    let n_max = smoke.unwrap_or(N_MAX);
    let m5_dir = match smoke {
        None => crate_path("receipts"),
        Some(n) => {
            let d = crate_path("target/latentmesh-runs/run2-m5-smoke");
            std::fs::create_dir_all(&d)?;
            println!("SMOKE MODE: {n} items, M5 artifacts from {}", d.display());
            d
        }
    };

    // ---- Gate 1: payload artifact vs M3's FROZEN training receipt ---------
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
        "payload artifact hash {} != M3 training receipt's {expected_hash}",
        transform.content_hash
    );
    let golden = crate_path(GOLDEN);
    let (golden_n, golden_max_rel, golden_seed) =
        common::mlp::verify_against_golden(&transform, &golden, GOLDEN_REL_TOL)?;
    println!(
        "payload adapter ({}): verified against {golden_n} golden pairs, max rel L2 \
         {golden_max_rel:.3e}",
        transform.content_hash
    );

    // ---- Gate 2: the comparator, and the single changed factor ------------
    let comparator: serde_json::Value =
        serde_json::from_slice(&std::fs::read(crate_path(COMPARATOR_RECEIPT)).map_err(|e| {
            anyhow::anyhow!(
                "comparator receipt {COMPARATOR_RECEIPT} unreadable ({e}) — M5 isolates the \
                 RECEIVER change against M4i and cannot state that claim without it"
            )
        })?)?;
    anyhow::ensure!(
        comparator["config"]["transform"]["content_hash"].as_str() == Some(expected_hash.as_str()),
        "the comparator receipt used a different payload artifact"
    );
    anyhow::ensure!(
        comparator["variant"].as_str() == Some("pertokenlast-fuse-questiontail"),
        "the comparator receipt used a different derivation / operator / site"
    );
    anyhow::ensure!(
        comparator["config"]["injection_operator"]["mode"].as_str() == Some(INJECT_MODE.tag())
            && comparator["config"]["injection_site"].as_str() == Some(SITE.tag()),
        "the comparator receipt used a different operator or site"
    );
    anyhow::ensure!(
        comparator["e_process"]["lambda"].as_f64() == Some(LAMBDA)
            && comparator["e_process"]["n_max"].as_u64() == Some(N_MAX as u64),
        "the comparator receipt ran different e-process parameters"
    );
    println!(
        "comparator: {COMPARATOR_RECEIPT} (same payload, derivation, operator, site, stream and \
         statistic) — the SINGLE changed factor is THE RECEIVER"
    );

    // ---- Gate 3: the M5 training receipt and its transfer check -----------
    let m5_training: serde_json::Value = serde_json::from_slice(
        &std::fs::read(m5_dir.join(format!(
            "run2-m5-training-receipt-cellL18toL14-r{rank}.json"
        )))
        .map_err(|e| anyhow::anyhow!("M5 training receipt for rank {rank} unreadable ({e})"))?,
    )?;
    let transfer: serde_json::Value = serde_json::from_slice(
        &std::fs::read(m5_dir.join(format!(
            "run2-m5-transfer-receipt-cellL18toL14-r{rank}.json"
        )))
        .map_err(|e| {
            anyhow::anyhow!(
                "M5 transfer-check receipt for rank {rank} unreadable ({e}) — the check is ordered \
                 BEFORE this draw and GATES it; run run2_m5_transfer_check first"
            )
        })?,
    )?;
    anyhow::ensure!(
        transfer["gate_pass"].as_bool() == Some(true),
        "the M5 transfer check did NOT pass — the draw must not be invoked (a null would be \
         confounded by the composed->fused BF16 gap). That receipt is the honest outcome."
    );
    anyhow::ensure!(
        transfer["config"]["adapter"]["content_hash"].as_str()
            == m5_training["artifact"]["content_hash_sha256"].as_str(),
        "the transfer check measured a different adapter than the training receipt names"
    );
    println!(
        "transfer check PASSED: fused gold-continuation NLL {} (on) vs {} (off) over {} holdout \
         items",
        transfer["summary"]["mean_fused_nll_adapter_on"],
        transfer["summary"]["mean_fused_nll_adapter_off"],
        transfer["summary"]["n_evaluated"]
    );

    // ---- Item supply: ADR-036 Decision 2 ----------------------------------
    let dir = common::run_dir("run2-m5");
    let data = dir.join("gsm8k-train.jsonl");
    let train_sha = common::fetch(common::GSM8K_TRAIN_URL, &data)?;
    anyhow::ensure!(train_sha == GSM8K_TRAIN_SHA256);
    let all_items = common::load_gsm8k(&data)?;

    let adaptation: serde_json::Value =
        serde_json::from_slice(&std::fs::read(crate_path(ADAPTATION_512))?)?;
    anyhow::ensure!(
        adaptation["split"].as_str() == Some("adaptation-512")
            && adaptation["source_sha256"].as_str() == Some(GSM8K_TRAIN_SHA256),
        "the item-supply file is not adaptation-512 over this train.jsonl"
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
        eligible.len() >= n_max,
        "eligible pool ({}) is smaller than N_max ({n_max})",
        eligible.len()
    );

    // ---- Models, then the adapter installed on the receiver ---------------
    let device = latentmesh_runtime::device().map_err(anyhow::Error::msg)?;
    println!("loading {SENDER} + {RECEIVER} (BF16, concurrent-resident)...");
    let mut sender =
        QwenRuntime::load(SENDER, &device, candle_core::DType::BF16).map_err(anyhow::Error::msg)?;
    let mut receiver = QwenRuntime::load(RECEIVER, &device, candle_core::DType::BF16)
        .map_err(anyhow::Error::msg)?;
    // Resolved but UNUSED under `Site::QuestionTail` (no placeholder exists in
    // this rung's prompt). Kept so the shared code path is literally the same
    // function every prior rung called.
    let pad_id = receiver
        .tokenizer
        .token_to_id("<|fim_pad|>")
        .ok_or_else(|| anyhow::anyhow!("<|fim_pad|> not in tokenizer"))?;
    let adapter = common::m5::load_adapter(&m5_dir, rank, &m5_training, &device)?;
    receiver.model.set_residual_lora(Some(adapter.lora.clone()));
    anyhow::ensure!(
        receiver.model.residual_lora().map(|l| l.after_block) == Some(RECEIVER_BLOCK),
        "the adapter is not installed at the injection block"
    );
    println!(
        "RECEIVER ADAPTED: rank {} LoRA {} ({} params, scaling {}) installed after block {} — ALL \
         FOUR conditions below, baseline included, run through THIS receiver",
        adapter.lora.rank,
        adapter.lora.content_hash,
        adapter.param_count,
        adapter.lora.scaling(),
        RECEIVER_BLOCK
    );
    println!("models loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // ---- Tokenisation pre-flight, BEFORE any generation -------------------
    // Resolving the site for the whole stream up front means no item can be
    // dropped mid-draw for a reason that could correlate with its outcome.
    let mut stream: Vec<usize> = Vec::new();
    let mut tokenization_excluded: Vec<serde_json::Value> = Vec::new();
    let mut site_samples: Vec<serde_json::Value> = Vec::new();
    for &idx in &eligible {
        if stream.len() == n_max {
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
        stream.len() == n_max,
        "pre-flight resolved only {} of the required {n_max} items",
        stream.len()
    );
    println!(
        "site pre-flight: {} items resolved, {} excluded on tokenisation grounds (before any \
         forward pass)",
        stream.len(),
        tokenization_excluded.len()
    );

    // ---- THE DRAW: ADR-036 e-process over the fixed stream ----------------
    let mut o = DrawOutcome::default();
    for (order, &idx) in stream.iter().enumerate() {
        let item = &all_items[idx];
        o.items_drawn = order + 1;
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
                let (ra, rb, rz, rr) = (q.real, q.base, q.zero, q.rand);
                let discordant = o.push_pair(order + 1, idx, row, q);
                println!(
                    "[{}/{n_max}] item {idx}: aligned={} baseline={} zerovec={} random={} \
                     (nll {:.3}/{:.3}/{:.3}/{:.3}) | disc={discordant} W={:.4} {:.0}s",
                    order + 1,
                    ra.0,
                    rb.0,
                    rz.0,
                    rr.0,
                    ra.1,
                    rb.1,
                    rz.1,
                    rr.1,
                    o.wealth,
                    t0.elapsed().as_secs_f32()
                );
            }
            None => {
                o.push_degenerate(order + 1, idx);
                println!(
                    "[{}/{n_max}] item {idx}: degenerate sender capture pass, no pair (W unchanged \
                     at {:.4}); budget item consumed",
                    order + 1,
                    o.wealth
                );
            }
        }
        if o.wealth >= w_threshold {
            o.crossed_at = Some(order + 1);
            println!(
                "e-process CROSSED: W = {:.4} >= {w_threshold} at item {} — stopping, per the \
                 registered rule",
                o.wealth, o.items_drawn
            );
            break;
        }
    }

    // ---- Receipt ----------------------------------------------------------
    let ctx = ReceiptCtx {
        rank,
        env: common::env_info(&nvcc),
        comparator_receipt: COMPARATOR_RECEIPT,
        payload_training_receipt: TRAINING_RECEIPT,
        payload_artifact: artifact.display().to_string(),
        payload_hash: &transform.content_hash,
        payload_golden_file: golden.display().to_string(),
        payload_golden_pairs: golden_n,
        payload_golden_seed: golden_seed,
        payload_golden_max_rel: golden_max_rel,
        payload_golden_tol: GOLDEN_REL_TOL,
        adapter: &adapter,
        m5_training: &m5_training,
        transfer: &transfer,
        site_tag: SITE.tag(),
        site_description: SITE.description(),
        inject_mode: INJECT_MODE,
        randvec_seed_base: RANDVEC_SEED_BASE,
        pad_id,
        train_sha,
        leakage_exclusions: LEAKAGE_EXCLUSIONS.to_vec(),
        excluded_present,
        eligible: eligible.len(),
        tokenization_excluded,
        site_samples,
    };
    let mut r = receipt(&ctx, &o);
    r["wall_clock_s"] = serde_json::json!(t0.elapsed().as_secs_f32());
    let name = match smoke {
        None => format!(
            "run2-m5-receipt-cellL18toL14-loraR{rank}-pertokenlast-fuse-questiontail-slots8-eprocess.json"
        ),
        Some(n) => {
            r["SMOKE_RUN"] = serde_json::json!({
                "items": n,
                "warning": "SMOKE RUN — a pipeline proof, NOT the registered M5 draw. The stream is truncated and the adapter is a smoke-trained one, so no number here is evidence about anything. ADR-045's one-draw rule is unaffected: this is not a draw.",
            });
            r["gate_pass"] = serde_json::json!(false);
            format!("SMOKE-run2-m5-receipt-loraR{rank}-{n}items.json")
        }
    };
    common::write_receipt(&m5_dir, &name, &r)?;

    m5_draw::print_summary(rank, n_max, &o, &r);
    Ok(())
}
