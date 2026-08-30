//! M6 item supply and tokenisation pre-flight (ADR-036 Decision 2), split out
//! of `run2_m6_probe.rs` for file-size discipline.
//!
//! Mechanical and unchanged from M5 — the stream, its order and its exclusions
//! are held-fixed factors, not this rung's variable. Keeping them here leaves
//! the probe holding only the four gates and the draw loop, which is what a
//! reader audits against ADR-047.
//!
//! One M6-specific assertion lives here: the eligible pool must be **strictly
//! larger** than `n_max`, because the `mismatched` control needs a priming item
//! from outside the drawn stream. M5 required only `>=`.

use super::m3::{build_site_prompt, Site};
use super::Gsm8kItem;
use latentmesh_runtime::QwenRuntime;
use std::path::Path;

pub struct Supply {
    pub all_items: Vec<Gsm8kItem>,
    pub eligible: Vec<usize>,
    pub excluded_present: Vec<usize>,
    pub train_sha: String,
}

/// `adaptation-512` in fixed index order with ADR-024's leakage exclusions
/// applied, over a sha256-pinned GSM8K train split.
pub fn item_supply(
    run_dir: &Path,
    adaptation_file: &Path,
    train_sha256: &str,
    leakage_exclusions: &[usize],
    n_max: usize,
) -> anyhow::Result<Supply> {
    let data = run_dir.join("gsm8k-train.jsonl");
    let train_sha = super::fetch(super::GSM8K_TRAIN_URL, &data)?;
    anyhow::ensure!(
        train_sha == train_sha256,
        "GSM8K train split sha256 {train_sha} != the pinned {train_sha256}"
    );
    let all_items = super::load_gsm8k(&data)?;

    let adaptation: serde_json::Value = serde_json::from_slice(&std::fs::read(adaptation_file)?)?;
    anyhow::ensure!(
        adaptation["split"].as_str() == Some("adaptation-512")
            && adaptation["source_sha256"].as_str() == Some(train_sha256),
        "the item-supply file is not adaptation-512 over this train.jsonl"
    );
    let indices: Vec<usize> = adaptation["indices"]
        .as_array()
        .expect("adaptation-512 indices")
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    anyhow::ensure!(indices.len() == 512);
    anyhow::ensure!(
        indices.windows(2).all(|w| w[0] < w[1]),
        "adaptation-512 is not in ascending index order"
    );
    let excluded_present: Vec<usize> = leakage_exclusions
        .iter()
        .copied()
        .filter(|i| indices.contains(i))
        .collect();
    let eligible: Vec<usize> = indices
        .iter()
        .copied()
        .filter(|i| !leakage_exclusions.contains(i))
        .collect();
    // STRICTLY greater: the last eligible index primes the `mismatched`
    // control and must lie outside the drawn stream.
    anyhow::ensure!(
        eligible.len() > n_max,
        "eligible pool ({}) leaves no item outside the {n_max}-item stream to prime the \
         mismatched control",
        eligible.len()
    );
    Ok(Supply {
        all_items,
        eligible,
        excluded_present,
        train_sha,
    })
}

pub struct PreFlight {
    pub stream: Vec<usize>,
    pub tokenization_excluded: Vec<serde_json::Value>,
    pub site_samples: Vec<serde_json::Value>,
}

/// Resolve the delivery site for the whole stream **before any generation**,
/// so no item can be dropped mid-draw for a reason that could correlate with
/// its outcome.
pub fn preflight(
    receiver: &QwenRuntime,
    supply: &Supply,
    pad_id: u32,
    site: Site,
    n_max: usize,
) -> anyhow::Result<PreFlight> {
    let mut stream: Vec<usize> = Vec::new();
    let mut tokenization_excluded: Vec<serde_json::Value> = Vec::new();
    let mut site_samples: Vec<serde_json::Value> = Vec::new();
    for &idx in &supply.eligible {
        if stream.len() == n_max {
            break;
        }
        match build_site_prompt(receiver, &supply.all_items[idx], pad_id, site) {
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
    Ok(PreFlight {
        stream,
        tokenization_excluded,
        site_samples,
    })
}
