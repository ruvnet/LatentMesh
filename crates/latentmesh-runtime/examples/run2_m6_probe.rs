//! Run-2 **M6** — the content axis
//! (`docs/adr/047-m6-manifold-content-factorial.md`), evaluated under ADR-036's
//! successor-rung e-process. `run2_m5_probe` lineage.
//!
//! **Why this rung exists.** M5's rank-2 primary hit 46W/23L — `aligned`
//! beating `random` on the decision endpoint, trending hard at the boundary.
//! That is not interpretable, and M5's own accuracy field says why: at every
//! rank accuracy ordered **`baseline` > `aligned` > `random`**. Every injection
//! hurts; the on-manifold payload hurts least. So "aligned beats random" is
//! equally explained by `aligned` being a *gentler perturbation* — no content
//! need be transmitted for it to appear. The defect is in the control:
//! `random` is a norm-matched Gaussian, wrong on **two axes at once**
//! (content-free AND off-manifold), so beating it identifies neither.
//!
//! **THE SINGLE CHANGED FACTOR.** The control set. `mismatched` — the previous
//! drawn item's genuine payload — is on-manifold, produced by the identical
//! computation, and rescaled by the identical rule, so it is norm-identical to
//! `aligned` and differs **only** in which episode's content it encodes.
//! Beating it cannot be explained by disruption magnitude. The registered
//! primary moves to `aligned` vs `mismatched`; everything else — payload
//! artifact, derivation, operator, site, depth, rescale rule, decoding, stream,
//! statistic and the receiver including its adapter — is held fixed against
//! M5's receipt and asserted at run time.
//!
//! **Five conditions, not the registered six.** ADR-047's `aligned_displaced`
//! cell and its MANIFOLD primary were withdrawn after the §5 manipulation check
//! (coordinator error #24, recorded in the ADR's outcomes). The doses PASSED
//! the registered gate 4/4; the cell was withdrawn on evidence beyond it,
//! because a displaced payload measured as a point on the `aligned`→`random`
//! segment under every instrument in the lens kit rather than as a distinct
//! factorial cell. The CONTENT axis is untouched by that and stands as
//! registered.
//!
//! **`mismatched` is not ours.** arXiv:2607.26773's "other-example message".
//! Cited, not re-derived.
//!
//! The e-process state lives in `common/m6_draw.rs` and the receipt literal in
//! `common/m6_receipt.rs` (file-size discipline); what stays here is the gate
//! sequence and the draw loop, which is what a reader audits against ADR-047.
//!
//! ONE draw. No retry (ADR-032 honest-fail).
//!
//! Run: PATH=/usr/local/cuda-12.8/bin:$PATH \
//!      cargo run --release --features cuda --example run2_m6_probe -- <rank>

#![recursion_limit = "1024"]

#[path = "common/mod.rs"]
#[allow(dead_code)]
mod common;

// Declared HERE rather than in `common/mod.rs`, which is `#[path]`-included
// into every example: this module carries the receipt literal, whose `json!`
// expansion exceeds the default macro recursion limit. Keeping it out of
// `common` means no other example pays for it.
#[path = "common/m6_draw.rs"]
mod m6_draw;
#[path = "common/m6_receipt.rs"]
mod m6_receipt;

use common::m3::{
    Site, Variant, GOLDEN_REL_TOL, RANDVEC_SEED_BASE, RECEIVER, RECEIVER_BLOCK, SENDER,
};
use common::m6::five_conditions_at;
use common::mlp::MlpTransform;
use latentmesh_runtime::inject::InjectionMode;
use latentmesh_runtime::QwenRuntime;
use m6_draw::{DrawOutcome, E_ALPHA, LAMBDA, N_MAX};
use m6_receipt::{receipt, ReceiptCtx};
use std::path::{Path, PathBuf};

const GSM8K_TRAIN_SHA256: &str = "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465";
/// M3's training receipt — the payload artifact's freeze point.
const TRAINING_RECEIPT: &str = "receipts/run2-m3-training-receipt-cellL18toL14.json";
const ARTIFACT: &str = "receipts/run2-m3-mlp-cellL18toL14.f32bin";
const GOLDEN: &str = "receipts/run2-m3-golden-mlp-cellL18toL14.json";
const ADAPTATION_512: &str = "../../harness/latentmesh-live/data/adaptation-512.json";
/// The §5 manipulation check that gates this rung, and whose outcome withdrew
/// the `aligned_displaced` cell.
const PHASE1_RECEIPT: &str = "receipts/run2-manifold-precheck-m6-phase1.json";

/// UNCHANGED from M5 — none of these is this rung's variable.
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

fn comparator_receipt(rank: usize) -> String {
    format!(
        "receipts/run2-m5-receipt-cellL18toL14-loraR{rank}-pertokenlast-fuse-questiontail-slots8-eprocess.json"
    )
}

fn main() -> anyhow::Result<()> {
    let t0 = std::time::Instant::now();
    let nvcc = latentmesh_runtime::assert_cuda_build_env().map_err(anyhow::Error::msg)?;
    println!("build-env guard: {nvcc}");
    let w_threshold = 1.0 / E_ALPHA;

    // The rank is REQUIRED, with no default. ADR-047 anchors its power model
    // on M5's measured battery and compares its wall clock to M5's, so it runs
    // on an M5-adapted receiver — but M5 produced THREE adapted receivers and
    // the frozen text never names which. That gap is a registration question
    // for the coordinator, not a default this probe may quietly pick: a silent
    // default would make an unregistered choice look like a registered one.
    let rank: usize = match std::env::args().nth(1) {
        Some(a) => a.parse()?,
        None => anyhow::bail!(
            "M6 needs the M5 adapter rank named explicitly: `-- <rank>`. ADR-047 does not name \
             one (see the probe header), so there is no default to fall back on."
        ),
    };
    anyhow::ensure!(
        matches!(rank, 1 | 2 | 4),
        "rank {rank} is not one of M5's drawn ranks (1, 2, 4)"
    );
    let smoke: Option<usize> = std::env::var("LM_M6_SMOKE")
        .ok()
        .map(|v| v.parse::<usize>())
        .transpose()?;
    let n_max = smoke.unwrap_or(N_MAX);
    if let Some(k) = smoke {
        println!(
            "SMOKE MODE: {k} items. This is NOT the registered draw and its receipt is marked \
             pre_committed: false."
        );
    }
    let receipts = crate_path("receipts");

    // ---- Gate 1: the phase-1 manipulation check --------------------------
    // ADR-047 §5 makes it gate the draw. It ran, and its outcome is why this
    // probe has five conditions rather than six.
    let phase1: serde_json::Value =
        serde_json::from_slice(&std::fs::read(crate_path(PHASE1_RECEIPT)).map_err(|e| {
            anyhow::anyhow!(
                "the ADR-047 §5 manipulation check receipt is unreadable ({e}) — it GATES \
                 this draw"
            )
        })?)?;
    let mc = &phase1["m6_manipulation_check"];
    let admitted = mc["n_admitted"]
        .as_u64()
        .ok_or_else(|| anyhow::anyhow!("the phase-1 receipt carries no §5 gate outcome"))?;
    let dropped = mc["n_dropped"].as_u64().unwrap_or_default();
    println!(
        "phase-1 manipulation check: {admitted} doses admitted, {dropped} dropped; the \
         aligned_displaced cell was nevertheless WITHDRAWN, on evidence BEYOND the gate \
         (coordinator error #24), so this draw runs FIVE conditions and ONE primary"
    );

    // ---- Gate 2: payload artifact vs M3's FROZEN training receipt ---------
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

    // ---- Gate 3: the comparator, and the single changed factor ------------
    let comparator_rel = comparator_receipt(rank);
    let comparator: serde_json::Value =
        serde_json::from_slice(&std::fs::read(crate_path(&comparator_rel)).map_err(|e| {
            anyhow::anyhow!(
                "comparator receipt {comparator_rel} unreadable ({e}) — M6 isolates its CONTROL \
                 SET change against M5 at the same rank and cannot state that claim without it"
            )
        })?)?;
    anyhow::ensure!(
        comparator["config"]["transform"]["content_hash"].as_str() == Some(expected_hash.as_str()),
        "the comparator receipt used a different payload artifact"
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
        "comparator: {comparator_rel} (same payload, derivation, operator, site, stream, \
         statistic and receiver) — the SINGLE changed factor is THE CONTROL SET"
    );

    // ---- Gate 4: the M5 adapter, its training and transfer receipts -------
    // The transfer check is a property of the ADAPTER, and M6 changes no
    // adapter — so M5's committed receipt is read and gated on, never
    // regenerated. Re-running it would reproduce the same numbers under a name
    // that would overwrite a frozen M5 receipt.
    let m5_training: serde_json::Value = serde_json::from_slice(
        &std::fs::read(receipts.join(format!(
            "run2-m5-training-receipt-cellL18toL14-r{rank}.json"
        )))
        .map_err(|e| anyhow::anyhow!("M5 training receipt for rank {rank} unreadable ({e})"))?,
    )?;
    let transfer: serde_json::Value = serde_json::from_slice(
        &std::fs::read(receipts.join(format!(
            "run2-m5-transfer-receipt-cellL18toL14-r{rank}.json"
        )))
        .map_err(|e| {
            anyhow::anyhow!("M5 transfer-check receipt for rank {rank} unreadable ({e})")
        })?,
    )?;
    anyhow::ensure!(
        transfer["gate_pass"].as_bool() == Some(true),
        "the M5 transfer check for rank {rank} did NOT pass — the draw must not be invoked (a \
         null would be confounded by the composed->fused BF16 gap)"
    );
    anyhow::ensure!(
        transfer["config"]["adapter"]["content_hash"].as_str()
            == m5_training["artifact"]["content_hash_sha256"].as_str(),
        "the transfer check measured a different adapter than the training receipt names"
    );
    let gen_diag = &transfer["summary"]["generation_diagnostic"];
    anyhow::ensure!(
        gen_diag["accuracy_adapter_off"].is_number() && gen_diag["accuracy_adapter_on"].is_number(),
        "the transfer receipt carries no generation diagnostic — mandatory since ADR-045 error #22"
    );
    println!(
        "transfer check (M5's committed receipt for rank {rank}, not re-run) PASSED: fused CE {} \
         (on) vs {} (off) over {} holdout items",
        transfer["summary"]["mean_fused_train_target_ce_adapter_on"],
        transfer["summary"]["mean_fused_train_target_ce_adapter_off"],
        transfer["summary"]["n_evaluated"]
    );
    println!(
        "generation diagnostic (NON-gating, carried into this receipt): baseline accuracy \
         adapter-on {} vs off {} of {}; mean generated chars {} vs {}",
        gen_diag["accuracy_adapter_on"],
        gen_diag["accuracy_adapter_off"],
        gen_diag["n_items"],
        gen_diag["mean_generated_chars_adapter_on"],
        gen_diag["mean_generated_chars_adapter_off"]
    );

    // ---- Item supply: ADR-036 Decision 2 ----------------------------------
    let dir = common::run_dir("run2-m6");
    let supply = common::m6_supply::item_supply(
        &dir,
        &crate_path(ADAPTATION_512),
        GSM8K_TRAIN_SHA256,
        &LEAKAGE_EXCLUSIONS,
        n_max,
    )?;

    // ---- Models, then the adapter installed on the receiver ---------------
    let device = latentmesh_runtime::device().map_err(anyhow::Error::msg)?;
    println!("loading {SENDER} + {RECEIVER} (BF16, concurrent-resident)...");
    let mut sender =
        QwenRuntime::load(SENDER, &device, candle_core::DType::BF16).map_err(anyhow::Error::msg)?;
    let mut receiver = QwenRuntime::load(RECEIVER, &device, candle_core::DType::BF16)
        .map_err(anyhow::Error::msg)?;
    // Resolved but UNUSED under `Site::QuestionTail`; kept so the shared code
    // path is literally the same function every prior rung called.
    let pad_id = receiver
        .tokenizer
        .token_to_id("<|fim_pad|>")
        .ok_or_else(|| anyhow::anyhow!("<|fim_pad|> not in tokenizer"))?;
    let adapter = common::m5::load_adapter(&receipts, rank, &m5_training, &device)?;
    receiver.model.set_residual_lora(Some(adapter.lora.clone()));
    anyhow::ensure!(
        receiver.model.residual_lora().map(|l| l.after_block) == Some(RECEIVER_BLOCK),
        "the adapter is not installed at the injection block"
    );
    println!(
        "RECEIVER ADAPTED: rank {} LoRA {} ({} params) installed after block {RECEIVER_BLOCK} — \
         ALL FIVE conditions below, baseline included, run through THIS receiver",
        adapter.lora.rank, adapter.lora.content_hash, adapter.param_count
    );
    println!("models loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // ---- Tokenisation pre-flight, BEFORE any generation -------------------
    let pf = common::m6_supply::preflight(&receiver, &supply, pad_id, SITE, n_max)?;
    println!(
        "site pre-flight: {} items resolved, {} excluded on tokenisation grounds (before any \
         forward pass)",
        pf.stream.len(),
        pf.tokenization_excluded.len()
    );

    // ---- The mismatched control's priming payload -------------------------
    // Stream order 0 has no predecessor. The registered priming item is the
    // LAST eligible index, which the assertion below shows is outside the
    // drawn stream — so no lookahead is introduced and early stopping stays
    // honest. Ported from `run3_gated_text_probe.rs:313-317, 361-363`.
    let priming_item = *supply.eligible.last().unwrap();
    anyhow::ensure!(
        !pf.stream.contains(&priming_item),
        "priming item {priming_item} is inside the drawn stream"
    );
    let (mut prev_payload, _, _) = common::m6::capture_payload(
        &mut sender,
        &transform,
        &supply.all_items[priming_item],
        VARIANT,
        &device,
    )?
    .ok_or_else(|| {
        anyhow::anyhow!("priming item {priming_item} produced a degenerate sender pass")
    })?;
    let mut prev_payload_from = priming_item;
    println!(
        "mismatched control primed from item {priming_item} (outside the stream); it is then the \
         PREVIOUS drawn item's payload, carried forward with no lookahead"
    );

    // ---- THE DRAW: ADR-036 e-process over the fixed stream ----------------
    let mut o = DrawOutcome::default();
    for (order, &idx) in pf.stream.iter().enumerate() {
        let item = &supply.all_items[idx];
        o.items_drawn = order + 1;
        match common::m6::capture_payload(&mut sender, &transform, item, VARIANT, &device)? {
            Some((aligned, sender_pass, meta)) => {
                let (row, q) = five_conditions_at(
                    &mut receiver,
                    item,
                    pad_id,
                    &aligned,
                    &prev_payload,
                    prev_payload_from,
                    &sender_pass,
                    &meta,
                    INJECT_MODE,
                    SITE,
                    &device,
                )?;
                let discordant = o.push_pair(order + 1, idx, row, q);
                println!(
                    "[{}/{n_max}] item {idx}: aligned={} mismatched={} (from {prev_payload_from}) \
                     baseline={} zerovec={} random={} (nll {:.3}/{:.3}/{:.3}/{:.3}/{:.3}) | \
                     disc={discordant} W={:.4} {:.0}s",
                    order + 1,
                    q.real.0,
                    q.mism.0,
                    q.base.0,
                    q.zero.0,
                    q.rand.0,
                    q.real.1,
                    q.mism.1,
                    q.base.1,
                    q.zero.1,
                    q.rand.1,
                    o.wealth,
                    t0.elapsed().as_secs_f32()
                );
                // Carry forward AFTER the item is evaluated — the ordering is
                // what makes "no lookahead" true rather than merely claimed.
                prev_payload = aligned;
                prev_payload_from = idx;
            }
            None => {
                o.push_degenerate(order + 1, idx);
                println!(
                    "[{}/{n_max}] item {idx}: degenerate sender capture pass, no pair (W unchanged \
                     at {:.4}); budget item consumed, carried-forward control left at item {}",
                    order + 1,
                    o.wealth,
                    prev_payload_from
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
        comparator_receipt: &comparator_rel,
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
        train_sha: supply.train_sha,
        leakage_exclusions: LEAKAGE_EXCLUSIONS.to_vec(),
        excluded_present: supply.excluded_present,
        eligible: supply.eligible.len(),
        priming_item,
        tokenization_excluded: pf.tokenization_excluded,
        site_samples: pf.site_samples,
        phase1_receipt: PHASE1_RECEIPT,
    };
    let mut r = receipt(&ctx, &o);
    if smoke.is_some() {
        r["pre_committed"] = serde_json::json!(false);
        r["smoke"] = serde_json::json!(true);
    }
    m6_draw::print_summary(n_max, &o, &r);

    let name = match smoke {
        None => format!(
            "run2-m6-receipt-cellL18toL14-loraR{rank}-contentaxis-questiontail-eprocess.json"
        ),
        Some(_) => format!("run2-m6-SMOKE-receipt-loraR{rank}.json"),
    };
    let out = match smoke {
        None => receipts.join(&name),
        Some(_) => dir.join(&name),
    };
    std::fs::write(&out, serde_json::to_string_pretty(&r)?)?;
    println!("receipt written: {}", out.display());
    Ok(())
}
