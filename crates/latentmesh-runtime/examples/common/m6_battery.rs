//! M6's control-vs-control battery (ADR-047 §8) — every ordered pair among
//! the five registered conditions, on both endpoints, as **stored receipt
//! fields**.
//!
//! Split out of [`super::m6`] purely for file-size discipline. The reason it
//! exists at all is coordinator error #21: a receipt that only ever tested one
//! condition against one control let an unsupported rank ordering stand as a
//! claim for weeks, because reproducing the control-vs-control comparisons
//! required running a paired sign test rather than reading a value
//! (`docs/research/052` §"One honest qualification"). The pairs a receipt like
//! that silently omits are exactly the control-vs-control ones, so they are
//! computed here whether or not anyone asks for them.

use super::m6::{Cond, Quint, CONDITIONS};

/// Every ordered pair among the five registered conditions, on both endpoints
/// — 20 pairs. Same contract as M5's four-condition battery.
pub fn condition_battery(paired: &[Quint]) -> serde_json::Value {
    let mut out = serde_json::Map::new();
    for &a in &Cond::ALL {
        for &b in &Cond::ALL {
            if a == b {
                continue;
            }
            let (name_a, name_b) = (a.label(), b.label());
            let acc_w = paired
                .iter()
                .filter(|q| q.correct(a) && !q.correct(b))
                .count();
            let acc_l = paired
                .iter()
                .filter(|q| !q.correct(a) && q.correct(b))
                .count();
            let nll_w = paired.iter().filter(|q| q.nll(a) < q.nll(b)).count();
            let nll_l = paired.iter().filter(|q| q.nll(a) > q.nll(b)).count();
            out.insert(
                format!("{name_a}_vs_{name_b}"),
                serde_json::json!({
                    "control_vs_control": a.is_control() && b.is_control(),
                    "registered_primary": a == Cond::AlignedReal && b == Cond::Mismatched,
                    "accuracy": {
                        "wins": acc_w, "losses": acc_l, "n_discordant": acc_w + acc_l,
                        "p_one_sided": super::sign_test_one_sided(acc_w, acc_l),
                        "mid_p_one_sided": super::mid_p_one_sided(acc_w, acc_l),
                    },
                    "nll_lower_is_better": {
                        "wins": nll_w, "losses": nll_l, "n_discordant": nll_w + nll_l,
                        "p_one_sided": super::sign_test_one_sided(nll_w, nll_l),
                        "mid_p_one_sided": super::mid_p_one_sided(nll_w, nll_l),
                    },
                }),
            );
        }
    }
    serde_json::json!({
        "why": "ADR-047 §8 mandatory co-report, carried from ADR-045. Coordinator error #21: a receipt that only tests one condition against one control lets an unsupported rank ordering stand as a claim. Every ordered pair among the registered conditions is computed here and stored, whether or not anyone asks for it.",
        "registered_conditions": CONDITIONS,
        "five_not_six": "ADR-047 §8 says thirty pairs at six conditions. There are twenty at five, because the `aligned_displaced` cell was withdrawn after the §5 manipulation check (coordinator error #24, ADR-047 outcomes): typicality measured as an affine restatement of the dose, and every property of a displaced payload moved monotonically toward `random`, so the cell was a point on the aligned-to-random segment rather than a distinct factorial cell. The MANIFOLD primary went with it; the CONTENT primary is unchanged.",
        "registered_primary": "aligned_real_vs_mismatched — the CONTENT axis, both arms on-manifold and norm-identical, differing only in which episode's content the payload encodes. It is flagged per-pair above so a reader does not have to know which comparison was registered.",
        "zerovec_is_degenerate_under_fuse": "under InjectionMode::Fuse the zerovec condition is h += 0, an exact no-op, so zerovec == baseline by construction. Every pair involving zerovec is therefore an operator-correctness check, not an independent control comparison.",
        "statistics_are_secondary": "these p-values are the ladder's standing SECONDARY diagnostics. The rung's verdict is the e-process outcome on the registered primary, and no number here is translated into or out of that scale.",
        "pairs": out,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quint(
        r: (bool, f32),
        m: (bool, f32),
        b: (bool, f32),
        z: (bool, f32),
        n: (bool, f32),
    ) -> Quint {
        Quint {
            real: r,
            mism: m,
            base: b,
            zero: z,
            rand: n,
        }
    }

    #[test]
    fn battery_covers_every_ordered_pair_and_flags_the_primary() {
        let paired = vec![
            quint(
                (true, 1.0),
                (false, 2.5),
                (false, 2.0),
                (false, 2.0),
                (false, 3.0),
            ),
            quint(
                (false, 2.0),
                (false, 2.2),
                (true, 1.0),
                (true, 1.0),
                (true, 1.5),
            ),
        ];
        let v = condition_battery(&paired);
        let pairs = v["pairs"].as_object().unwrap();
        assert_eq!(pairs.len(), 5 * 4);

        // The registered primary is present, flagged, and is not a
        // control-vs-control pair.
        let am = &pairs["aligned_real_vs_mismatched"];
        assert_eq!(am["registered_primary"], serde_json::json!(true));
        assert_eq!(am["control_vs_control"], serde_json::json!(false));
        assert_eq!(am["accuracy"]["wins"], serde_json::json!(1));
        assert_eq!(am["accuracy"]["losses"], serde_json::json!(0));
        assert_eq!(am["nll_lower_is_better"]["wins"], serde_json::json!(2));

        // Its converse is NOT the primary — the flag is directional.
        assert_eq!(
            pairs["mismatched_vs_aligned_real"]["registered_primary"],
            serde_json::json!(false)
        );

        // mismatched is a control, so mismatched-vs-random is flagged.
        assert_eq!(
            pairs["mismatched_vs_random"]["control_vs_control"],
            serde_json::json!(true)
        );

        // Under fuse zerovec == baseline; the fixture reproduces that.
        let zb = &pairs["zerovec_injected_vs_baseline_uninjected"];
        assert_eq!(zb["accuracy"]["n_discordant"], serde_json::json!(0));
        assert_eq!(
            zb["nll_lower_is_better"]["n_discordant"],
            serde_json::json!(0)
        );
    }
}
