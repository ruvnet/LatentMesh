//! M5 probe-side helpers (ADR-045): loading and verifying the trained
//! receiver adapter, and the **complete pairwise comparison battery** the ADR
//! makes mandatory.
//!
//! On the battery, and why it is here rather than left to a reader: coordinator
//! error #21 happened because a receipt only ever tested one condition against
//! one control, which let an unsupported rank ordering stand as a claim for
//! weeks. [`condition_battery`] computes **every ordered pair** among the four
//! registered conditions — including the control-vs-control pairs — on both
//! endpoints, and stores them as receipt fields.
//!
//! Disclosed scope limit, stated rather than papered over: ADR-045 registers
//! exactly four conditions (`aligned`, `baseline`, `zerovec`, `random`).
//! ADR-003's `mismatched` and `self_generated` controls are **not** among
//! them, so no comparison involving them can be computed here; adding a fifth
//! condition would be an unregistered change to a frozen design. The battery
//! is therefore complete **over the registered condition set**, and the
//! receipt says so in those words.

use super::m3::Quad;
use super::mlp::D_OUT;
use latentmesh_runtime::lora::{ResidualLora, LORA_ALPHA};
use std::path::Path;

/// A loaded, hash- and golden-verified M5 adapter.
pub struct LoadedAdapter {
    pub lora: ResidualLora,
    pub golden_pairs: usize,
    pub golden_max_rel: f32,
    pub param_count: usize,
    /// The seeded-init artifact's hash, carried through for the receipt.
    pub init_content_hash: String,
}

#[derive(serde::Deserialize)]
struct LoraGoldenFile {
    artifact_file_sha256: String,
    input_seed_chacha8: u64,
    n_pairs: usize,
    inputs: Vec<Vec<f32>>,
    /// The adapter's DELTA for a single residual row (not `h + delta`).
    outputs: Vec<Vec<f32>>,
}

/// Load the trained adapter named by an M5 training receipt, asserting its
/// content hash against that receipt and verifying it against the golden
/// pairs the trainer produced **through this same type** on CPU.
pub fn load_adapter(
    receipts: &Path,
    rank: usize,
    training_receipt: &serde_json::Value,
    device: &candle_core::Device,
) -> anyhow::Result<LoadedAdapter> {
    let art = &training_receipt["artifact"];
    let file = art["file"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("training receipt has no artifact.file"))?;
    let expected = art["content_hash_sha256"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("training receipt has no artifact.content_hash_sha256"))?;
    let lora = ResidualLora::load_artifact(
        &receipts.join(file),
        D_OUT,
        super::m3::RECEIVER_BLOCK,
        LORA_ALPHA,
        device,
    )?;
    anyhow::ensure!(
        lora.content_hash == expected,
        "adapter {file} content_hash {} != the training receipt's {expected}",
        lora.content_hash
    );
    anyhow::ensure!(
        lora.rank == rank,
        "adapter file carries rank {} but rank {rank} was requested",
        lora.rank
    );

    let golden_file = art["golden_file"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("training receipt has no artifact.golden_file"))?;
    let g: LoraGoldenFile = serde_json::from_slice(&std::fs::read(receipts.join(golden_file))?)?;
    anyhow::ensure!(
        g.artifact_file_sha256 == lora.content_hash,
        "golden file was produced for a different artifact"
    );
    anyhow::ensure!(
        g.n_pairs >= 8 && g.inputs.len() == g.n_pairs && g.outputs.len() == g.n_pairs,
        "golden file must carry >= 8 input/output pairs"
    );
    let mut max_rel = 0f32;
    let mut any_nonzero = false;
    for (x, y_gold) in g.inputs.iter().zip(&g.outputs) {
        let xt = candle_core::Tensor::from_vec(x.clone(), (1, 1, D_OUT), device)?;
        let y = lora.delta(&xt)?.flatten_all()?.to_vec1::<f32>()?;
        let diff: f32 = y
            .iter()
            .zip(y_gold)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            .sqrt();
        let norm: f32 = y_gold.iter().map(|v| v * v).sum::<f32>().sqrt();
        any_nonzero |= norm > 0.0;
        // A zero-norm golden (the identity adapter) admits no relative error;
        // require exactness instead of dividing by zero.
        let rel = if norm > 0.0 { diff / norm } else { diff };
        max_rel = max_rel.max(rel);
        anyhow::ensure!(
            rel <= super::m3::GOLDEN_REL_TOL,
            "the loaded adapter diverges from its golden pairs: relative L2 error {rel}"
        );
    }
    anyhow::ensure!(
        any_nonzero,
        "every golden output is zero — this artifact is the IDENTITY adapter (B still zero), so \
         nothing was trained; refusing to treat it as a trained adapter"
    );
    let _ = g.input_seed_chacha8;
    Ok(LoadedAdapter {
        param_count: lora.param_count(),
        golden_pairs: g.n_pairs,
        golden_max_rel: max_rel,
        init_content_hash: art["init_content_hash_sha256"]
            .as_str()
            .unwrap_or_default()
            .to_string(),
        lora,
    })
}

/// The four registered conditions, in receipt order.
pub const CONDITIONS: [&str; 4] = [
    "aligned_real",
    "baseline_uninjected",
    "zerovec_injected",
    "random",
];

/// The registered conditions as a **type**, so the endpoint accessors below
/// cannot silently mis-route.
///
/// This replaces a `usize` index whose match ended in `_ => q.rand`. Under
/// that arm a fifth condition would have been read as `random` on **both**
/// endpoints — a wrong number written into a stored receipt field, with no
/// compile error, no panic, and no signal in the output that anything had
/// gone wrong. M6 adds exactly such a condition, which is what made this
/// worth fixing before rather than after.
///
/// The enum stays four-armed because ADR-045 registers four. M6's five-arm
/// set lives on its own type in [`super::m6`]; widening *this* one now fails
/// to compile at both accessors instead of quietly changing M5's frozen
/// numbers.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Cond {
    AlignedReal,
    BaselineUninjected,
    ZerovecInjected,
    Random,
}

impl Cond {
    /// In receipt order — the same order, and the same labels, as
    /// [`CONDITIONS`]. `labels_match_the_registered_order` pins that.
    const ALL: [Cond; 4] = [
        Cond::AlignedReal,
        Cond::BaselineUninjected,
        Cond::ZerovecInjected,
        Cond::Random,
    ];

    fn label(self) -> &'static str {
        match self {
            Cond::AlignedReal => "aligned_real",
            Cond::BaselineUninjected => "baseline_uninjected",
            Cond::ZerovecInjected => "zerovec_injected",
            Cond::Random => "random",
        }
    }

    /// Everything except the treatment arm. Drives the battery's
    /// `control_vs_control` flag, which previously read `a != 0 && b != 0`.
    fn is_control(self) -> bool {
        self != Cond::AlignedReal
    }
}

fn endpoints(q: &Quad, which: Cond) -> (bool, f32) {
    match which {
        Cond::AlignedReal => q.real,
        Cond::BaselineUninjected => q.base,
        Cond::ZerovecInjected => q.zero,
        Cond::Random => q.rand,
    }
}

fn correct(q: &Quad, which: Cond) -> bool {
    endpoints(q, which).0
}

fn nll(q: &Quad, which: Cond) -> f32 {
    endpoints(q, which).1
}

/// Every ordered pair among the four registered conditions, on both endpoints.
///
/// For each pair `(a, b)`: the accuracy discordant counts (`a` right and `b`
/// wrong, and the converse) with the one-sided exact sign test and its mid-p;
/// and the NLL sign test (`a` lower than `b`, and the converse) with the same
/// two statistics. Control-vs-control pairs are included and flagged, because
/// they are the ones a receipt that only tests the treatment arm silently
/// omits.
pub fn condition_battery(paired: &[Quad]) -> serde_json::Value {
    let mut out = serde_json::Map::new();
    for &a in &Cond::ALL {
        for &b in &Cond::ALL {
            if a == b {
                continue;
            }
            let (name_a, name_b) = (a.label(), b.label());
            let acc_w = paired
                .iter()
                .filter(|q| correct(q, a) && !correct(q, b))
                .count();
            let acc_l = paired
                .iter()
                .filter(|q| !correct(q, a) && correct(q, b))
                .count();
            let nll_w = paired.iter().filter(|q| nll(q, a) < nll(q, b)).count();
            let nll_l = paired.iter().filter(|q| nll(q, a) > nll(q, b)).count();
            out.insert(
                format!("{name_a}_vs_{name_b}"),
                serde_json::json!({
                    "control_vs_control": a.is_control() && b.is_control(),
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
        "why": "ADR-045 mandatory co-report. Coordinator error #21: a receipt that only tests one condition against one control lets an unsupported rank ordering stand as a claim. Every ordered pair among the registered conditions is computed here and stored, whether or not anyone asks for it.",
        "registered_conditions": CONDITIONS,
        "scope_limit_disclosed": "ADR-045 registers exactly these four conditions. ADR-003's `mismatched` and `self_generated` controls are NOT registered for this rung, so no comparison involving them appears here — adding a fifth condition would be an unregistered change to a frozen design. This battery is complete OVER THE REGISTERED SET, not over ADR-003's five-control set.",
        "zerovec_is_degenerate_under_fuse": "under InjectionMode::Fuse the zerovec condition is h += 0, an exact no-op, so zerovec == baseline by construction. Every pair involving zerovec is therefore an operator-correctness check, not an independent control comparison, and the substantive control-vs-control pair is random vs baseline.",
        "statistics_are_secondary": "these p-values are the ladder's standing SECONDARY diagnostics. The rung's verdict is the e-process outcome on aligned vs random, and no number here is translated into or out of that scale.",
        "pairs": out,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quad(r: (bool, f32), b: (bool, f32), z: (bool, f32), n: (bool, f32)) -> Quad {
        Quad {
            real: r,
            base: b,
            zero: z,
            rand: n,
        }
    }

    /// The refactor from a `usize` index to [`Cond`] must not have moved a
    /// single receipt key. M5 is closed and its three drawn receipts are
    /// committed, so a renamed or reordered pair key would silently
    /// desynchronise the stored battery from the code that claims to produce
    /// it.
    #[test]
    fn labels_match_the_registered_order() {
        assert_eq!(Cond::ALL.len(), CONDITIONS.len());
        for (c, name) in Cond::ALL.iter().zip(CONDITIONS.iter()) {
            assert_eq!(c.label(), *name);
        }
        // The treatment arm is index 0, which is what the old
        // `a != 0 && b != 0` flag encoded.
        assert!(!Cond::ALL[0].is_control());
        assert!(Cond::ALL[1..].iter().all(|c| c.is_control()));
    }

    #[test]
    fn battery_covers_every_ordered_pair_and_flags_control_vs_control() {
        let paired = vec![
            quad((true, 1.0), (false, 2.0), (false, 2.0), (false, 3.0)),
            quad((false, 2.0), (true, 1.0), (true, 1.0), (true, 1.5)),
        ];
        let v = condition_battery(&paired);
        let pairs = v["pairs"].as_object().unwrap();
        assert_eq!(pairs.len(), 4 * 3);
        // Treatment arm.
        let ar = &pairs["aligned_real_vs_random"];
        assert_eq!(ar["control_vs_control"], serde_json::json!(false));
        assert_eq!(ar["accuracy"]["wins"], serde_json::json!(1));
        assert_eq!(ar["accuracy"]["losses"], serde_json::json!(1));
        assert_eq!(ar["nll_lower_is_better"]["wins"], serde_json::json!(1));
        // Control vs control is present, flagged, and computed.
        let rb = &pairs["random_vs_baseline_uninjected"];
        assert_eq!(rb["control_vs_control"], serde_json::json!(true));
        assert_eq!(rb["nll_lower_is_better"]["losses"], serde_json::json!(2));
        // Under fuse zerovec == baseline; the fixture reproduces that, and the
        // pair must come out with no discordance at all.
        let zb = &pairs["zerovec_injected_vs_baseline_uninjected"];
        assert_eq!(zb["accuracy"]["n_discordant"], serde_json::json!(0));
        assert_eq!(
            zb["nll_lower_is_better"]["n_discordant"],
            serde_json::json!(0)
        );
    }
}
