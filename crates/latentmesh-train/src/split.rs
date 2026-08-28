//! ADR-024 leakage-safe split machinery.
//!
//! `fit_holdout_split` is a byte-exact replica of
//! `harness/latentmesh-live/src/calibrate.rs:115` (same rand 0.8
//! `SliceRandom::shuffle` over a rand_chacha 0.3 `ChaCha8Rng`), kept in this
//! crate because the harness cannot be a path dependency (its
//! latentmesh-core dep pins `half = "=2.4.1"`, un-unifiable with candle's
//! `half >=2.5`). Equivalence with the harness's own output at
//! `(n=2560, seed=FIT_SPLIT_SEED)` was verified against a ground-truth run
//! of the harness function this session; the pinned digests below are that
//! ground truth, asserted by unit test so any drift (rand version bump,
//! shuffle change) fails loudly instead of silently corrupting the leakage
//! discipline.

use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use sha2::{Digest, Sha256};

/// Frozen 80/20 split seed, verbatim from
/// `harness/latentmesh-live/src/calibrate.rs` (`FIT_SPLIT_SEED`), reused per
/// ADR-024's leakage rule step 1 to preserve frozen-seed comparability with
/// the S2/S2c calibration work.
pub const FIT_SPLIT_SEED: u64 = 0x24C0_DE03;

/// The 13 S2c dataset rows whose items are in ADR-023's frozen 40-item
/// probe set — ADR-024 § Leakage discipline, computed by exact set
/// intersection (M0 data-shape scout, graded primary). Dropped from
/// whichever split side they land in, AFTER the frozen 80/20 split.
pub const PROBE_OVERLAP_ROWS: [usize; 13] = [
    69, 683, 824, 1258, 1281, 1346, 1577, 1586, 1647, 2052, 2058, 2078, 2529,
];

/// The (row → GSM8K item) pairs for [`PROBE_OVERLAP_ROWS`], from ADR-024.
/// Asserted against the dump index's `item_indices` before any training.
pub const PROBE_OVERLAP_ROW_ITEMS: [(usize, usize); 13] = [
    (69, 150),
    (683, 1309),
    (824, 1573),
    (1258, 2365),
    (1281, 2418),
    (1346, 2540),
    (1577, 2958),
    (1586, 2973),
    (1647, 3084),
    (2052, 3825),
    (2058, 3844),
    (2078, 3877),
    (2529, 4746),
];

/// Deterministic 80/20 row split: ChaCha8-seeded shuffle of `0..n`, first
/// 80% fit, rest held out. Replica of the harness function (see module doc).
pub fn fit_holdout_split(n: usize, seed: u64) -> (Vec<usize>, Vec<usize>) {
    let mut order: Vec<usize> = (0..n).collect();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    order.shuffle(&mut rng);
    let n_fit = n * 8 / 10;
    (order[..n_fit].to_vec(), order[n_fit..].to_vec())
}

/// One excluded row's audit record for the training receipt.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ExcludedRow {
    pub row: usize,
    pub item: usize,
    /// Which side of the frozen 80/20 split the row landed in ("fit" or
    /// "holdout") before exclusion.
    pub side: &'static str,
}

/// The ADR-024 leakage-safe split: frozen 80/20 split FIRST, then the 13
/// probe-overlap rows dropped from whichever side they landed in.
#[derive(Debug, Clone)]
pub struct LeakageSafeSplit {
    /// Fit-side rows after exclusion, in shuffle order.
    pub fit: Vec<usize>,
    /// Holdout-side rows after exclusion, in shuffle order.
    pub holdout: Vec<usize>,
    /// Per-row exclusion audit list (all 13, with the side each came from).
    pub excluded: Vec<ExcludedRow>,
}

/// Build the leakage-safe split per ADR-024's frozen rule (steps 1–3:
/// seeded split by ITEM first, then drop the probe-overlap rows).
pub fn leakage_safe_split(n: usize) -> LeakageSafeSplit {
    let (fit_all, holdout_all) = fit_holdout_split(n, FIT_SPLIT_SEED);
    let dropped: std::collections::HashSet<usize> = PROBE_OVERLAP_ROWS.into_iter().collect();
    let side_of =
        |rows: &[usize]| -> std::collections::HashSet<usize> { rows.iter().copied().collect() };
    let fit_set = side_of(&fit_all);
    let mut excluded = Vec::with_capacity(PROBE_OVERLAP_ROWS.len());
    for (row, item) in PROBE_OVERLAP_ROW_ITEMS {
        let side = if fit_set.contains(&row) {
            "fit"
        } else {
            "holdout"
        };
        excluded.push(ExcludedRow { row, item, side });
    }
    let fit = fit_all
        .into_iter()
        .filter(|r| !dropped.contains(r))
        .collect();
    let holdout = holdout_all
        .into_iter()
        .filter(|r| !dropped.contains(r))
        .collect();
    LeakageSafeSplit {
        fit,
        holdout,
        excluded,
    }
}

/// sha256 over the comma-joined decimal row list — the digest form pinned in
/// receipts (and in the ground-truth test below).
pub fn rows_sha256(rows: &[usize]) -> String {
    let joined = rows
        .iter()
        .map(|r| r.to_string())
        .collect::<Vec<_>>()
        .join(",");
    format!("{:x}", Sha256::digest(joined.as_bytes()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ground truth measured this session by running the HARNESS's own
    /// `fit_holdout_split(2560, FIT_SPLIT_SEED)` (scratch crate path-depping
    /// harness/latentmesh-live) and hashing the comma-joined index lists.
    /// This pins replica≡harness equivalence.
    const HARNESS_FIT_SHA256: &str =
        "f1a35fbec5cd4e082b8e16c136124d62fcbc98730edbb2bcd8fbca5910c99f9b";
    const HARNESS_HOLDOUT_SHA256: &str =
        "c0f70e762e689e66b01ed62a61b6c106264300a822c4cbccb5e1c8bf84464fde";

    #[test]
    fn replica_matches_harness_ground_truth() {
        let (fit, hold) = fit_holdout_split(2560, FIT_SPLIT_SEED);
        assert_eq!(fit.len(), 2048);
        assert_eq!(hold.len(), 512);
        assert_eq!(rows_sha256(&fit), HARNESS_FIT_SHA256);
        assert_eq!(rows_sha256(&hold), HARNESS_HOLDOUT_SHA256);
    }

    #[test]
    fn split_is_deterministic_and_partitions() {
        let (f1, h1) = fit_holdout_split(2560, FIT_SPLIT_SEED);
        let (f2, h2) = fit_holdout_split(2560, FIT_SPLIT_SEED);
        assert_eq!(f1, f2);
        assert_eq!(h1, h2);
        let mut all: Vec<usize> = f1.iter().chain(h1.iter()).copied().collect();
        all.sort_unstable();
        assert_eq!(all, (0..2560).collect::<Vec<_>>());
    }

    #[test]
    fn leakage_exclusion_arithmetic() {
        let s = leakage_safe_split(2560);
        // Measured this session against the harness ground truth: 11 of the
        // 13 probe rows land fit-side, 2 (rows 1346, 1586) holdout-side.
        assert_eq!(s.fit.len(), 2037);
        assert_eq!(s.holdout.len(), 510);
        assert_eq!(s.excluded.len(), 13);
        let holdout_side: Vec<usize> = s
            .excluded
            .iter()
            .filter(|e| e.side == "holdout")
            .map(|e| e.row)
            .collect();
        assert_eq!(holdout_side, vec![1346, 1586]);
        for e in &s.excluded {
            assert!(!s.fit.contains(&e.row));
            assert!(!s.holdout.contains(&e.row));
        }
        // n_fit within ADR-024's disclosed expected range 2,037–2,048.
        assert!((2035..=2048).contains(&s.fit.len()));
    }
}
