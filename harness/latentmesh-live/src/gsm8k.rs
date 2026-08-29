//! GSM8K raw-JSONL loading and the seeded, committed dataset splits
//! (design doc 024 §3 `gsm8k.rs`, §6 split discipline).
//!
//! Splits (ChaCha8-seeded, index lists committed under `data/`):
//! - `calibration-4000` (train) — S2 teacher-forced calibration pairs
//! - `adaptation-512`  (train) — Darwin fitness + per-generation audits only
//! - `eval-200`        (test)  — frozen-genome evaluation
//! - `holdout-100`     (test)  — sealed
//!
//! Calibration and adaptation are drawn in ONE without-replacement sample of
//! 4,512 train indices (first 4,000 / next 512), so they are disjoint by
//! construction. Eval and holdout are one 300-index sample of test (first
//! 200 / last 100), likewise disjoint.
//!
//! §6 mechanical refusal: [`eval_items`] and [`holdout_items`] return an
//! error unless a genome-frozen receipt exists at
//! [`GENOME_FROZEN_RECEIPT`]. No such receipt exists at S2 — nothing in this
//! stage can touch an eval or holdout item.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::{Path, PathBuf};

pub const GSM8K_TRAIN_URL: &str =
    "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl";
pub const GSM8K_TEST_URL: &str =
    "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl";
/// Design doc 024 §6 pins the test split at this sha256.
pub const GSM8K_TEST_SHA256: &str =
    "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14";
/// train.jsonl has no design pin; this is the sha256 measured 2026-08-28 and
/// cross-checked against the committed S1a receipt
/// (`crates/latentmesh-runtime/receipts/s1a-receipt-*.json` `dataset.sha256`)
/// for evidence continuity across stages.
pub const GSM8K_TRAIN_SHA256_S1A: &str =
    "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465";

/// Split seeds — arbitrary constants committed BEFORE any data was drawn
/// (seed discipline, design §6). One seed per source file.
pub const TRAIN_SPLIT_SEED: u64 = 0x24C0_DE01;
pub const TEST_SPLIT_SEED: u64 = 0x24C0_DE02;

/// Path (relative to this crate) of the genome-frozen receipt that S5 writes
/// after Darwin freeze. Until it exists, eval/holdout item access refuses.
pub const GENOME_FROZEN_RECEIPT: &str = "receipts/genome-frozen.json";

pub fn crate_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

pub fn data_dir() -> PathBuf {
    crate_dir().join("data")
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Download `url` to `dest` unless present; return the file's sha256.
pub fn fetch(url: &str, dest: &Path) -> anyhow::Result<String> {
    if !dest.exists() {
        let resp = ureq::get(url).call()?;
        let mut bytes = Vec::new();
        resp.into_reader().read_to_end(&mut bytes)?;
        std::fs::write(dest, &bytes)?;
    }
    Ok(sha256_hex(&std::fs::read(dest)?))
}

#[derive(Debug, Clone)]
pub struct Gsm8kItem {
    pub index: usize,
    pub question: String,
    pub answer_text: String,
    pub gold: String,
}

/// Parse GSM8K raw JSONL; `gold` is the normalized text after `#### `.
pub fn load_gsm8k(path: &Path) -> anyhow::Result<Vec<Gsm8kItem>> {
    let mut items = Vec::new();
    for (index, line) in std::fs::read_to_string(path)?.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(line)?;
        let question = v["question"].as_str().unwrap_or_default().to_string();
        let answer_text = v["answer"].as_str().unwrap_or_default().to_string();
        let gold = extract_final_answer(&answer_text)
            .ok_or_else(|| anyhow::anyhow!("item {index}: no '#### n' in gold answer"))?;
        items.push(Gsm8kItem {
            index,
            question,
            answer_text,
            gold,
        });
    }
    Ok(items)
}

/// Normalized final answer: text after the last `####`, commas/periods
/// stripped, whitespace trimmed.
pub fn extract_final_answer(text: &str) -> Option<String> {
    let after = text.rsplit_once("####")?.1;
    let token = after.split_whitespace().next()?;
    let cleaned: String = token
        .chars()
        .filter(|c| c.is_ascii_digit() || *c == '-' || *c == '.')
        .collect();
    let cleaned = cleaned.trim_end_matches('.').to_string();
    (!cleaned.is_empty()).then_some(cleaned)
}

/// One committed index list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitFile {
    pub split: String,
    pub source: String,
    pub source_sha256: String,
    pub seed: u64,
    pub indices: Vec<usize>,
}

impl SplitFile {
    pub fn load(name: &str) -> anyhow::Result<Self> {
        let path = data_dir().join(format!("{name}.json"));
        let text = std::fs::read_to_string(&path)
            .map_err(|e| anyhow::anyhow!("read {}: {e}", path.display()))?;
        Ok(serde_json::from_str(&text)?)
    }
}

/// §6 mechanical refusal: eval items are inaccessible until a genome-frozen
/// receipt exists.
pub fn eval_items() -> anyhow::Result<Vec<Gsm8kItem>> {
    frozen_gate_then_load("eval-200")
}

/// §6 mechanical refusal: holdout items are sealed until genome freeze.
pub fn holdout_items() -> anyhow::Result<Vec<Gsm8kItem>> {
    frozen_gate_then_load("holdout-100")
}

fn frozen_gate_then_load(split: &str) -> anyhow::Result<Vec<Gsm8kItem>> {
    let receipt = crate_dir().join(GENOME_FROZEN_RECEIPT);
    anyhow::ensure!(
        receipt.exists(),
        "REFUSED: split '{split}' is locked until a genome-frozen receipt exists at {} \
         (design 024 §6: the harness mechanically refuses eval and holdout indices \
         before Darwin genome freeze)",
        receipt.display()
    );
    let spec = SplitFile::load(split)?;
    let path = data_dir().join(&spec.source);
    let all = load_gsm8k(&path)?;
    Ok(spec.indices.iter().map(|&i| all[i].clone()).collect())
}

/// Generate and write all four committed split files + the splits receipt.
/// Idempotent given the same seeds/sources (ChaCha8 is deterministic).
pub fn make_splits() -> anyhow::Result<serde_json::Value> {
    let dir = data_dir();
    std::fs::create_dir_all(&dir)?;
    let train_path = dir.join("gsm8k-train.jsonl");
    let test_path = dir.join("gsm8k-test.jsonl");
    let train_sha = fetch(GSM8K_TRAIN_URL, &train_path)?;
    let test_sha = fetch(GSM8K_TEST_URL, &test_path)?;
    anyhow::ensure!(
        test_sha == GSM8K_TEST_SHA256,
        "test.jsonl sha256 {test_sha} != design pin {GSM8K_TEST_SHA256}"
    );
    anyhow::ensure!(
        train_sha == GSM8K_TRAIN_SHA256_S1A,
        "train.jsonl sha256 {train_sha} != S1a-receipt continuity value {GSM8K_TRAIN_SHA256_S1A}"
    );
    let n_train = load_gsm8k(&train_path)?.len();
    let n_test = load_gsm8k(&test_path)?.len();

    // ONE without-replacement sample per source file; contiguous slices of it
    // form the splits, so disjointness holds by construction (and is verified
    // below anyway).
    let mut rng = ChaCha8Rng::seed_from_u64(TRAIN_SPLIT_SEED);
    let train_sample = rand::seq::index::sample(&mut rng, n_train, 4512).into_vec();
    let mut rng = ChaCha8Rng::seed_from_u64(TEST_SPLIT_SEED);
    let test_sample = rand::seq::index::sample(&mut rng, n_test, 300).into_vec();

    let write = |name: &str, source: &str, sha: &str, seed: u64, mut idx: Vec<usize>| {
        idx.sort_unstable();
        let f = SplitFile {
            split: name.to_string(),
            source: source.to_string(),
            source_sha256: sha.to_string(),
            seed,
            indices: idx,
        };
        let path = dir.join(format!("{name}.json"));
        std::fs::write(&path, serde_json::to_string(&f).unwrap())?;
        anyhow::Ok((
            name.to_string(),
            f.indices.len(),
            f.indices[0],
            *f.indices.last().unwrap(),
        ))
    };
    let cal = write(
        "calibration-4000",
        "gsm8k-train.jsonl",
        &train_sha,
        TRAIN_SPLIT_SEED,
        train_sample[..4000].to_vec(),
    )?;
    let ada = write(
        "adaptation-512",
        "gsm8k-train.jsonl",
        &train_sha,
        TRAIN_SPLIT_SEED,
        train_sample[4000..].to_vec(),
    )?;
    let eva = write(
        "eval-200",
        "gsm8k-test.jsonl",
        &test_sha,
        TEST_SPLIT_SEED,
        test_sample[..200].to_vec(),
    )?;
    let hold = write(
        "holdout-100",
        "gsm8k-test.jsonl",
        &test_sha,
        TEST_SPLIT_SEED,
        test_sample[200..].to_vec(),
    )?;

    // Verify disjointness explicitly (not just by construction).
    let cal_set: std::collections::BTreeSet<usize> = train_sample[..4000].iter().copied().collect();
    let overlap = train_sample[4000..]
        .iter()
        .filter(|i| cal_set.contains(i))
        .count();
    anyhow::ensure!(overlap == 0, "calibration/adaptation overlap: {overlap}");
    let eval_set: std::collections::BTreeSet<usize> = test_sample[..200].iter().copied().collect();
    let overlap_t = test_sample[200..]
        .iter()
        .filter(|i| eval_set.contains(i))
        .count();
    anyhow::ensure!(overlap_t == 0, "eval/holdout overlap: {overlap_t}");

    let stat = |s: (String, usize, usize, usize)| serde_json::json!({"split": s.0, "n": s.1, "min_index": s.2, "max_index": s.3});
    Ok(serde_json::json!({
        "stage": "S2-splits",
        "design": "docs/research/024-live-latent-experiment-design.md sections 3, 6",
        "evidence_label": "deterministic seeded split generation; no model, no simulation",
        "sources": {
            "train": {"file": "gsm8k-train.jsonl", "url": GSM8K_TRAIN_URL, "sha256": train_sha, "n_items": n_train},
            "test":  {"file": "gsm8k-test.jsonl", "url": GSM8K_TEST_URL, "sha256": test_sha, "n_items": n_test,
                       "design_pin_verified": true},
        },
        "seeds": {"train_split_chacha8": TRAIN_SPLIT_SEED, "test_split_chacha8": TEST_SPLIT_SEED,
                   "discipline": "one without-replacement sample per source; contiguous slices form the splits"},
        "splits": [stat(cal), stat(ada), stat(eva), stat(hold)],
        "disjointness": {"calibration_vs_adaptation_overlap": 0, "eval_vs_holdout_overlap": 0,
                          "train_vs_test": "different source files by construction"},
        "eval_holdout_lock": format!("{GENOME_FROZEN_RECEIPT} absent; eval_items()/holdout_items() refuse"),
        "unix_time": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn answer_normalization() {
        assert_eq!(
            extract_final_answer("x #### 1,000").as_deref(),
            Some("1000")
        );
        assert_eq!(extract_final_answer("#### 18.").as_deref(), Some("18"));
        assert_eq!(extract_final_answer("no marker"), None);
    }

    #[test]
    fn eval_and_holdout_refuse_without_frozen_receipt() {
        // No genome-frozen receipt exists at S2; the mechanical refusal must
        // hold regardless of whether the split files themselves exist.
        assert!(!crate_dir().join(GENOME_FROZEN_RECEIPT).exists());
        assert!(eval_items().is_err());
        assert!(holdout_items().is_err());
    }

    #[test]
    fn seeded_sampling_is_deterministic() {
        let mut a = ChaCha8Rng::seed_from_u64(TRAIN_SPLIT_SEED);
        let mut b = ChaCha8Rng::seed_from_u64(TRAIN_SPLIT_SEED);
        let s1 = rand::seq::index::sample(&mut a, 7473, 4512).into_vec();
        let s2 = rand::seq::index::sample(&mut b, 7473, 4512).into_vec();
        assert_eq!(s1, s2);
    }
}
