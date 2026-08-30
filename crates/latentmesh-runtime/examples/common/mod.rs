//! Shared helpers for the S0/S1a probe examples (not an example target —
//! included via `#[path]` from each probe). GSM8K raw-JSONL loading with
//! sha256 receipts, answer normalization, the exact sign test, and receipt
//! environment capture.

pub mod affine;
pub mod displace;
pub mod fastgrnn;
pub mod lens;
pub mod m3;
pub mod m5;
pub mod m6;
pub mod m6_battery;
pub mod m6_supply;
pub mod mlp;

use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::{Path, PathBuf};

/// GSM8K raw JSONL sources (openai/grade-school-math, master branch).
pub const GSM8K_TEST_URL: &str =
    "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl";
pub const GSM8K_TRAIN_URL: &str =
    "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl";
/// Design doc 024 §6 pins the test split at this sha256.
pub const GSM8K_TEST_SHA256: &str =
    "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14";

/// Answer-format instruction appended to every GSM8K question. Revision 2:
/// the run-1 phrasing ("give the final answer on the last line as
/// '#### <number>'") was ignored by Qwen2.5-1.5B-Instruct in 22/40 greedy
/// generations (measured, s1a run 1); this example-anchored phrasing produced
/// compliant '#### n' output in the gen_debug check.
pub const ANSWER_FORMAT: &str = "Solve the problem step by step. Then write the final line in exactly this format:\n#### <numeric answer>\nFor example, if the answer were 5, the last line must be: #### 5";

/// Probe run directory (inside the crate's gitignored `target/`, by design:
/// run artifacts never enter the source tree).
pub fn run_dir(stage: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target/latentmesh-runs")
        .join(stage);
    std::fs::create_dir_all(&dir).expect("create run dir");
    dir
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Download `url` to `dest` unless it already exists; return (path, sha256).
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

/// **Pre-committed gold rendering rule** (PC1's, promoted here verbatim so
/// PC1b derives its payload through the SAME text as PC1 rather than through
/// a second, independently-typed copy of the rule).
///
/// GSM8K's `answer` field is the human reference solution; it already ends
/// with the `#### <n>` line the receiver's own `ANSWER_FORMAT` instruction
/// asks for. The only edit is removal of the dataset's `<<a+b=c>>` calculator
/// annotations, which are an artifact of the GSM8K authoring tool and appear
/// in no model's output distribution. Nothing else is added — in particular
/// no EOS token — so the captured span's LAST row is the state at the final
/// answer token.
///
/// Behavioural identity with `run2_pc1_capture`'s private copy is not merely
/// asserted by inspection: PC1b's capture re-derives the payload for train
/// index 1153 (the one item `adaptation-512` shares with S1a's frozen 40) and
/// gates on it being **bit-identical** to PC1's committed vector for that
/// item, which exercises this function end-to-end.
pub fn render_gold(answer_text: &str) -> String {
    let mut out = String::with_capacity(answer_text.len());
    let mut rest = answer_text;
    while let Some(i) = rest.find("<<") {
        out.push_str(&rest[..i]);
        match rest[i..].find(">>") {
            Some(j) => rest = &rest[i + j + 2..],
            None => {
                rest = "";
                break;
            }
        }
    }
    out.push_str(rest);
    out
}

/// Model-answer extraction: the `####`-marked answer when present, else the
/// last number in the text (the standard "flexible extract" fallback —
/// Qwen2.5-1.5B frequently answers correctly without emitting the marker;
/// scoring those as wrong measured format adherence, not the latent channel).
pub fn extract_answer(text: &str) -> Option<String> {
    extract_final_answer(text).or_else(|| last_number(text))
}

/// Numeric-equivalence comparison ("2.0" == "2"); falls back to string
/// equality when either side is not parseable as f64.
pub fn answers_equal(a: &str, b: &str) -> bool {
    match (a.parse::<f64>(), b.parse::<f64>()) {
        (Ok(x), Ok(y)) => x == y,
        _ => a == b,
    }
}

/// Last number appearing in the text, normalized like [`extract_final_answer`].
pub fn last_number(text: &str) -> Option<String> {
    text.split_whitespace()
        .filter_map(|token| {
            let cleaned: String = token
                .chars()
                .filter(|c| c.is_ascii_digit() || *c == '-' || *c == '.')
                .collect();
            let cleaned = cleaned.trim_end_matches('.').trim_end_matches('-');
            (cleaned.chars().any(|c| c.is_ascii_digit())).then(|| cleaned.to_string())
        })
        .next_back()
}

/// Normalized final answer: text after the last `####`, commas stripped,
/// trailing period stripped, whitespace trimmed.
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

/// One-sided exact sign test on discordant pairs: p = P(X >= wins) for
/// X ~ Binomial(wins + losses, 0.5). Returns (wins, losses, p). With zero
/// discordant pairs the test is undefined; p = 1.0 (starved).
pub fn sign_test_one_sided(wins: usize, losses: usize) -> f64 {
    let n = wins + losses;
    if n == 0 {
        return 1.0;
    }
    let mut p = 0f64;
    for k in wins..=n {
        p += binom_pmf(n, k);
    }
    p.min(1.0)
}

/// One-sided **mid-p McNemar** statistic on the same discordant pairs:
/// `exact_p − 0.5·P(X = wins)` (Fagerland, Lydersen & Laake 2013, *BMC Med
/// Res Methodol* 13:91; `docs/research/031` §2.4, adopted as run 3's primary
/// statistic by ADR-030).
///
/// **Reporting-only in run 2.** Run-2 receipts record their verdict on
/// [`sign_test_one_sided`]; this value is disclosed alongside it and gates
/// nothing — the frozen protocol's statistic is ADR-028-protected and is not
/// retroactively changed by adding a second number to a receipt.
pub fn mid_p_one_sided(wins: usize, losses: usize) -> f64 {
    let n = wins + losses;
    if n == 0 {
        return 1.0;
    }
    (sign_test_one_sided(wins, losses) - 0.5 * binom_pmf(n, wins)).clamp(0.0, 1.0)
}

fn binom_pmf(n: usize, k: usize) -> f64 {
    (ln_choose(n, k) - (n as f64) * std::f64::consts::LN_2).exp()
}

fn ln_choose(n: usize, k: usize) -> f64 {
    ln_fact(n) - ln_fact(k) - ln_fact(n - k)
}

fn ln_fact(n: usize) -> f64 {
    (2..=n).map(|i| (i as f64).ln()).sum()
}

/// Environment block for receipts (risk #4: toolkit recorded in every
/// receipt) — evidence grade "live-model, single-host, simulation-free".
pub fn env_info(nvcc_runtime: &str) -> serde_json::Value {
    let gpu = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|e| format!("nvidia-smi unavailable: {e}"));
    let git = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    serde_json::json!({
        "evidence_label": "live-model, single-host, simulation-free",
        "gpu": gpu,
        "nvcc": nvcc_runtime,
        "nvcc_at_build": latentmesh_runtime::NVCC_RELEASE_AT_BUILD,
        "git_commit": git,
        "crate": "latentmesh-runtime 0.1.0 (vendored candle-transformers 0.9.2 qwen2)",
        "unix_time": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
    })
}

/// Standard-normal vector via Box–Muller over a seeded ChaCha8 stream —
/// the exact S1a/S2b random-control generator (those examples carry their
/// own private copies, predating this shared one; run-2 probes use this).
pub fn gaussian_vec(rng: &mut rand_chacha::ChaCha8Rng, n: usize) -> Vec<f32> {
    use rand::Rng;
    let mut v = Vec::with_capacity(n);
    while v.len() < n {
        let u1: f64 = rng.gen_range(f64::MIN_POSITIVE..1.0);
        let u2: f64 = rng.gen::<f64>();
        let r = (-2.0 * u1.ln()).sqrt();
        let t = 2.0 * std::f64::consts::PI * u2;
        v.push((r * t.cos()) as f32);
        if v.len() < n {
            v.push((r * t.sin()) as f32);
        }
    }
    v
}

/// Write a JSON receipt and echo its path.
pub fn write_receipt(
    dir: &Path,
    name: &str,
    receipt: &serde_json::Value,
) -> anyhow::Result<PathBuf> {
    let path = dir.join(name);
    std::fs::write(&path, serde_json::to_string_pretty(receipt)?)?;
    println!("receipt: {}", path.display());
    Ok(path)
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
        assert_eq!(extract_final_answer("#### -3"), Some("-3".to_string()));
        assert_eq!(extract_final_answer("no marker"), None);
        assert_eq!(
            extract_answer("the answer is 42, sure.").as_deref(),
            Some("42")
        );
        assert_eq!(extract_answer("costs $2.00 total").as_deref(), Some("2.00"));
        assert!(answers_equal("2.0", "2"));
        assert!(answers_equal("1000", "1000"));
        assert!(!answers_equal("2.5", "2"));
    }

    #[test]
    fn sign_test_values() {
        // 5 wins, 0 losses => p = 2^-5 = 0.03125
        assert!((sign_test_one_sided(5, 0) - 0.03125).abs() < 1e-12);
        assert!((sign_test_one_sided(0, 0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn mid_p_matches_research_031_table() {
        // docs/research/031 §1: draw #1 (5W/0L) 0.0312 -> 0.0156 mid-p;
        // draw #10 (4W/1L) 0.1875 -> 0.1094; draw #6 (2W/2L) 0.6875 -> 0.5000.
        assert!((mid_p_one_sided(5, 0) - 0.015625).abs() < 1e-12);
        assert!((mid_p_one_sided(4, 1) - 0.109375).abs() < 1e-12);
        assert!((mid_p_one_sided(2, 2) - 0.5).abs() < 1e-12);
        // M4c's own primary draw (4W/2L): exact 0.34375 -> mid-p 0.2265625.
        assert!((sign_test_one_sided(4, 2) - 0.34375).abs() < 1e-12);
        assert!((mid_p_one_sided(4, 2) - 0.2265625).abs() < 1e-12);
        assert!((mid_p_one_sided(0, 0) - 1.0).abs() < 1e-12);
    }
}
