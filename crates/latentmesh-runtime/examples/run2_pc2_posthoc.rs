//! Run-2 **PC2 post-hoc** — the void test registered by ADR-039's mid-draw
//! amendment, computed from the committed draw receipt.
//!
//! # Why this binary exists
//!
//! ADR-039's original text asserted that *"`random` hits the decoy at chance by
//! construction, giving a clean floor"* and voided the rung above a **2%**
//! baseline decoy-emission rate. The **AMENDMENT** appended to ADR-039 (commit
//! `b7dbe0e`, written **mid-draw at item 9 of 300 with the primary
//! undecided**) records that this premise is **false**: the decoy menu
//! `{g+1, g-1, g+10, 2g}` is exactly the space of **natural arithmetic slips**,
//! so a model that errs on GSM8K lands on those disproportionately and the
//! baseline rate is **structurally above chance**.
//!
//! The amendment does **not** relax the gate. The 2% gate is reported exactly
//! as it falls by `run2_pc2_probe`. What the amendment supplies is a
//! **pre-committed test for what a trip means**:
//!
//! > *the rung is void **only if `baseline` and `steer` are statistically
//! > indistinguishable on decoy-emission*** — i.e. the payload adds nothing
//! > over the model's own error propensity. If `steer` separates from `random`
//! > while baseline is merely elevated, the rung is **valid with reduced
//! > power**.
//!
//! # What this binary does and does not do
//!
//! - It **reads the committed draw receipt only**. No model is loaded, no item
//!   is re-drawn, no generation is repeated, and the e-process is neither
//!   restarted nor re-parametrised (ADR-032). Every number here is a
//!   deterministic function of data already committed to disk.
//! - The amendment names the test **in words** but not as a statistic. The
//!   statistic used here is stated rather than assumed: the **paired exact sign
//!   test on discordant pairs**, the same family used for this ladder's NLL
//!   secondaries and for every pre-ADR-036 rung, one-sided in the registered
//!   direction (`steer` > comparator) with the **two-sided** p reported
//!   alongside so a reader can apply either reading.
//! - It additionally breaks baseline leakage down **per perturbation**
//!   (`g+1` / `g-1` / `g+10` / `2g`), which is direct evidence for ADR-039's
//!   PC3 design note that decoys should be drawn away from natural-slip space.
//!
//! Run: cargo run --release --example run2_pc2_posthoc

#[path = "common/mod.rs"]
#[allow(dead_code)]
mod common;

use std::path::{Path, PathBuf};

const DRAW_RECEIPT: &str =
    "receipts/run2-pc2-receipt-identity-L19lasttoken-decoytf-fuse-questiontail-slots8-eprocess.json";
const ADR_AMENDMENT_COMMIT: &str = "b7dbe0e";
const ALPHA: f64 = 0.05;

fn crate_path(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

/// A paired comparison on a binary outcome: discordant counts, exact one-sided
/// sign-test p in the registered direction, its mid-p, and the two-sided p.
fn paired(label: &str, a: &[bool], b: &[bool]) -> serde_json::Value {
    let wins = a.iter().zip(b).filter(|(x, y)| **x && !**y).count();
    let losses = a.iter().zip(b).filter(|(x, y)| !**x && **y).count();
    let n_disc = wins + losses;
    let p_one = common::sign_test_one_sided(wins, losses);
    let mid_p = common::mid_p_one_sided(wins, losses);
    // Two-sided exact sign test: 2 x the smaller tail, capped at 1.
    let p_two = (2.0 * p_one.min(common::sign_test_one_sided(losses, wins))).min(1.0);
    // Minimum attainable one-sided p at this discordance — the power floor,
    // reported on its own scale per ADR-036 Decision 3.
    let floor = if n_disc == 0 {
        1.0
    } else {
        common::sign_test_one_sided(n_disc, 0)
    };
    serde_json::json!({
        "comparison": label,
        "wins_a_only": wins,
        "losses_b_only": losses,
        "n_discordant": n_disc,
        "p_one_sided_exact": p_one,
        "mid_p_one_sided": mid_p,
        "p_two_sided_exact": p_two,
        "min_attainable_one_sided_p_at_this_n_disc": floor,
        "power_limited": floor > ALPHA,
        "separates_one_sided_at_alpha_0_05": p_one < ALPHA,
        "separates_two_sided_at_alpha_0_05": p_two < ALPHA,
    })
}

fn main() -> anyhow::Result<()> {
    let draw: serde_json::Value =
        serde_json::from_slice(&std::fs::read(crate_path(DRAW_RECEIPT)).map_err(|err| {
            anyhow::anyhow!("draw receipt {DRAW_RECEIPT} unreadable ({err}) — run the probe first")
        })?)?;
    let rows = draw["items"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("draw receipt has no items array"))?;
    anyhow::ensure!(!rows.is_empty(), "draw receipt has zero items");

    let flag = |r: &serde_json::Value, cond: &str| -> anyhow::Result<bool> {
        r["conditions"][cond]["emits_decoy"]
            .as_bool()
            .ok_or_else(|| anyhow::anyhow!("row for item {} has no {cond}.emits_decoy", r["item"]))
    };
    let col =
        |cond: &str| -> anyhow::Result<Vec<bool>> { rows.iter().map(|r| flag(r, cond)).collect() };
    let steer = col("steer")?;
    let restore = col("restore")?;
    let base = col("baseline_uninjected")?;
    let zero = col("zerovec_injected")?;
    let rand = col("random")?;
    let n = steer.len();
    let nf = n as f64;

    let rate = |v: &[bool]| v.iter().filter(|x| **x).count() as f64 / nf;
    let cnt = |v: &[bool]| v.iter().filter(|x| **x).count();

    // ---- THE VOID TEST (ADR-039 amendment) --------------------------------
    let steer_vs_base = paired("steer vs baseline (THE VOID TEST)", &steer, &base);
    // Reported alongside so the void decision can be read in context.
    let steer_vs_rand = paired(
        "steer vs random (the registered primary's pair)",
        &steer,
        &rand,
    );
    let steer_vs_restore = paired("steer vs restore", &steer, &restore);
    let restore_vs_base = paired("restore vs baseline", &restore, &base);
    let rand_vs_base = paired("random vs baseline", &rand, &base);
    let zero_vs_base = paired("zerovec vs baseline (must be 0/0)", &zero, &base);

    let void = !(steer_vs_base["separates_two_sided_at_alpha_0_05"]
        .as_bool()
        .unwrap_or(false));

    // ---- Per-perturbation leakage (evidence for the PC3 design note) ------
    let mut per_pert: std::collections::BTreeMap<String, (usize, usize, usize)> =
        std::collections::BTreeMap::new();
    for (i, r) in rows.iter().enumerate() {
        let p = r["perturbation"].as_str().unwrap_or("<none>").to_string();
        let e = per_pert.entry(p).or_insert((0, 0, 0));
        e.0 += 1;
        if base[i] {
            e.1 += 1;
        }
        if steer[i] {
            e.2 += 1;
        }
    }
    let per_pert_json: Vec<serde_json::Value> = per_pert
        .iter()
        .map(|(k, (n_items, b, s))| {
            serde_json::json!({
                "perturbation": k, "items": n_items,
                "baseline_decoy_emissions": b,
                "baseline_rate": *b as f64 / *n_items as f64,
                "steer_decoy_emissions": s,
                "steer_rate": *s as f64 / *n_items as f64,
            })
        })
        .collect();

    let ep = &draw["e_process"];
    let receipt = serde_json::json!({
        "stage": "run2-PC2-posthoc-void-test",
        "design": "docs/adr/039-pc2-steering-control-pre-registration.md § AMENDMENT (commit b7dbe0e, written mid-draw at item 9 of 300 with the primary undecided). The amendment supplies the pre-committed test for what a 2% leakage-gate trip MEANS; it does not relax the gate.",
        "source_receipt": DRAW_RECEIPT,
        "computed_from": "the committed per-item rows of the draw receipt ONLY. No model was loaded, no item re-drawn, no generation repeated, and the e-process was neither restarted nor re-parametrised (ADR-032).",
        "adr_amendment_commit": ADR_AMENDMENT_COMMIT,

        "statistic_choice_stated_not_assumed": {
            "what_the_amendment_says": "the rung is void only if `baseline` and `steer` are statistically indistinguishable on decoy-emission",
            "what_it_does_not_say": "it names the test in words but does not name a statistic or a tail",
            "statistic_used_here": "paired exact sign test on discordant pairs — the same family used for this ladder's NLL secondaries and every pre-ADR-036 rung",
            "tail_used_for_the_VOID_decision": "TWO-SIDED at alpha = 0.05, because 'indistinguishable' is a symmetric claim; the one-sided p in the registered direction (steer > baseline) is reported alongside so a reader can apply either reading",
            "power_floor_reported": "min attainable one-sided p at the observed discordance is reported for every comparison, per ADR-036 Decision 3, so an underpowered comparison cannot be read as a null",
        },

        "decoy_emission_rates": {
            "n_items": n,
            "steer": {"count": cnt(&steer), "rate": rate(&steer)},
            "restore": {"count": cnt(&restore), "rate": rate(&restore)},
            "baseline_uninjected": {"count": cnt(&base), "rate": rate(&base)},
            "zerovec_injected": {"count": cnt(&zero), "rate": rate(&zero)},
            "random": {"count": cnt(&rand), "rate": rate(&rand)},
        },

        "THE_VOID_TEST": {
            "rule": "void only if baseline and steer are statistically indistinguishable on decoy-emission",
            "result": steer_vs_base.clone(),
            "void": void,
            "reading": if void {
                "VOID — `steer` does not separate from `baseline` on decoy-emission, so the payload adds nothing over the model's own natural-slip propensity."
            } else {
                "NOT VOID — `steer` separates from `baseline` on decoy-emission, so the elevated floor costs POWER, not validity. The rung is valid with reduced power."
            },
        },

        "paired_comparisons_on_decoy_emission": [
            steer_vs_base, steer_vs_rand, steer_vs_restore,
            restore_vs_base, rand_vs_base, zero_vs_base,
        ],

        "registered_primary_unchanged": {
            "note": "The e-process on `steer` vs `random` is the registered primary and is NOT recomputed here. Its committed values are echoed for context only.",
            "n_discordant": ep["n_discordant"].clone(),
            "wins": ep["wins"].clone(),
            "losses": ep["losses"].clone(),
            "final_wealth": ep["final_wealth"].clone(),
            "max_wealth_reached": ep["max_wealth_reached"].clone(),
            "crossed_at_item": ep["crossed_at_item"].clone(),
            "pass": ep["pass"].clone(),
        },

        "leakage_by_perturbation_evidence_for_the_PC3_design_note": {
            "why": "ADR-039's amendment registers, as a design note for a successor rung, that decoys should be drawn AWAY from natural-slip space (a random integer of similar magnitude) to recover a genuine chance floor. Per-perturbation baseline rates show which slips the model actually makes and are the direct evidence for that note.",
            "rows": per_pert_json,
        },

        "the_2pct_gate_as_it_fell": {
            "threshold": 0.02,
            "measured_baseline_rate": rate(&base),
            "tripped": rate(&base) > 0.02,
            "reported_without_softening": "The registered gate is reported exactly as it falls. The amendment changes only the INTERPRETATION of a trip — from 'the decoy construction is leaky and the rung is void' to 'the registered threshold was derived from a false premise about the floor' — and defers the decision to the void test above.",
        },
    });
    common::write_receipt(
        &crate_path("receipts"),
        "run2-pc2-posthoc-void-test-receipt.json",
        &receipt,
    )?;

    println!("=== PC2 post-hoc void test (ADR-039 amendment, commit {ADR_AMENDMENT_COMMIT}) ===");
    println!(
        "decoy-emission rates over {n} items: steer {:.4} restore {:.4} baseline {:.4} zerovec \
         {:.4} random {:.4}",
        rate(&steer),
        rate(&restore),
        rate(&base),
        rate(&zero),
        rate(&rand)
    );
    println!(
        "2% gate: baseline {:.4} vs 0.02 => tripped={}",
        rate(&base),
        rate(&base) > 0.02
    );
    for c in receipt["paired_comparisons_on_decoy_emission"]
        .as_array()
        .unwrap()
    {
        println!(
            "  {:<48} {}W/{}L n_disc {:<4} p1 {:.5} mid-p {:.5} p2 {:.5} floor {:.5}{}",
            c["comparison"].as_str().unwrap_or(""),
            c["wins_a_only"],
            c["losses_b_only"],
            c["n_discordant"],
            c["p_one_sided_exact"].as_f64().unwrap_or(1.0),
            c["mid_p_one_sided"].as_f64().unwrap_or(1.0),
            c["p_two_sided_exact"].as_f64().unwrap_or(1.0),
            c["min_attainable_one_sided_p_at_this_n_disc"]
                .as_f64()
                .unwrap_or(1.0),
            if c["power_limited"].as_bool() == Some(true) {
                "  [POWER-LIMITED]"
            } else {
                ""
            },
        );
    }
    println!("VOID={void} — {}", receipt["THE_VOID_TEST"]["reading"]);
    println!("leakage by perturbation:");
    for r in receipt["leakage_by_perturbation_evidence_for_the_PC3_design_note"]["rows"]
        .as_array()
        .unwrap()
    {
        println!(
            "  {:<5} items {:<4} baseline {:<4} ({:.4})  steer {:<4} ({:.4})",
            r["perturbation"].as_str().unwrap_or(""),
            r["items"],
            r["baseline_decoy_emissions"],
            r["baseline_rate"].as_f64().unwrap_or(0.0),
            r["steer_decoy_emissions"],
            r["steer_rate"].as_f64().unwrap_or(0.0),
        );
    }
    Ok(())
}
