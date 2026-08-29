//! Task-loss training data assembly (ADR-024 M4c): the saved S2c token
//! streams (each item's sender-generated span — the C2C-style CE target),
//! GSM8K question text (for the receiver's slotted injection prompt), and
//! the per-item sequence/span arithmetic under the measured seq-len cap.
//!
//! PROMPT-PARITY GATE: the probe-side prompt constants (`SYSTEM`,
//! `ANSWER_FORMAT`, the slot sentence) live in the runtime's
//! `examples/common/` tree, which this crate cannot import (the dependency
//! runs the other way). They are duplicated here VERBATIM, and the
//! duplication is pinned measurably: for every item the reconstructed
//! sender capture prompt is re-encoded and asserted token-identical to the
//! stream's stored `prompt_tokens` (which the S2c dump produced with the
//! probe's own constants + tokenizer) — one gate covering SYSTEM,
//! ANSWER_FORMAT, the chat template, and tokenizer parity at once. The slot
//! sentence itself is covered by the 8-placeholder-position assert.

use latentmesh_runtime::QwenRuntime;
use sha2::Digest;
use std::path::Path;

/// Verbatim copy of `examples/common/m3.rs::SYSTEM`.
pub const SYSTEM: &str = "You are a careful math tutor.";
/// Verbatim copy of `examples/common/mod.rs::ANSWER_FORMAT` (revision 2).
pub const ANSWER_FORMAT: &str = "Solve the problem step by step. Then write the final line in exactly this format:\n#### <numeric answer>\nFor example, if the answer were 5, the last line must be: #### 5";
/// Committed durable copy of the S2c token streams + its pinned sha256
/// (same pins as `examples/run2_pertoken_dump.rs`).
pub const STREAMS_REL: &str = "../../harness/latentmesh-live/data/s2c-token-streams.jsonl";
pub const STREAMS_SHA256: &str = "ad5713116cb5acf663fe9c33ac58aaedb1fd64fcf629dd78b21439d244958539";
/// Committed GSM8K train split + the pin from `examples/run2_m3_probe.rs`.
pub const GSM8K_TRAIN_REL: &str = "../../harness/latentmesh-live/data/gsm8k-train.jsonl";
pub const GSM8K_TRAIN_SHA256: &str =
    "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465";

/// One saved S2c stream row (schema of `run2_pertoken_dump.rs::StreamRow`).
#[derive(serde::Deserialize)]
pub struct StreamRow {
    pub row: usize,
    pub item: usize,
    pub prompt_tokens: Vec<u32>,
    pub gen_tokens: Vec<u32>,
    #[allow(dead_code)]
    pub sender_first_pass_correct: bool,
}

/// sha256-gated file read.
pub fn read_pinned(path: &Path, expected_sha256: &str) -> anyhow::Result<Vec<u8>> {
    let bytes = std::fs::read(path)?;
    let sha = format!("{:x}", sha2::Sha256::digest(&bytes));
    anyhow::ensure!(
        sha == expected_sha256,
        "{}: sha256 {sha} != pinned {expected_sha256}",
        path.display()
    );
    Ok(bytes)
}

/// Load the token streams, sha-gated, asserting row order 0..n.
pub fn load_streams(path: &Path) -> anyhow::Result<Vec<StreamRow>> {
    let bytes = read_pinned(path, STREAMS_SHA256)?;
    let mut rows = Vec::new();
    for line in String::from_utf8(bytes)?.lines() {
        if line.trim().is_empty() {
            continue;
        }
        rows.push(serde_json::from_str::<StreamRow>(line)?);
    }
    for (i, r) in rows.iter().enumerate() {
        anyhow::ensure!(r.row == i, "stream row {} out of order (row={})", i, r.row);
    }
    Ok(rows)
}

/// Load GSM8K question text by item index, sha-gated.
pub fn load_gsm8k_questions(path: &Path) -> anyhow::Result<Vec<String>> {
    let bytes = read_pinned(path, GSM8K_TRAIN_SHA256)?;
    let mut questions = Vec::new();
    for line in String::from_utf8(bytes)?.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(line)?;
        questions.push(v["question"].as_str().unwrap_or_default().to_string());
    }
    Ok(questions)
}

/// Fetch the receiver's tokenizer from the hf-hub cache (tokenizer only —
/// no model weights; encode convention matches `QwenRuntime::encode`, i.e.
/// `add_special_tokens = false`).
pub fn load_tokenizer(model_id: &str) -> anyhow::Result<tokenizers::Tokenizer> {
    let api = hf_hub::api::sync::Api::new()?;
    let repo = api.model(model_id.to_string());
    tokenizers::Tokenizer::from_file(repo.get("tokenizer.json")?)
        .map_err(|e| anyhow::anyhow!("tokenizer load: {e}"))
}

pub fn encode(tok: &tokenizers::Tokenizer, text: &str) -> anyhow::Result<Vec<u32>> {
    Ok(tok
        .encode(text, false)
        .map_err(|e| anyhow::anyhow!("encode: {e}"))?
        .get_ids()
        .to_vec())
}

/// The sender capture prompt (the S2c convention the streams were built
/// with) — reconstructed for the prompt-parity gate.
pub fn capture_prompt(question: &str) -> String {
    QwenRuntime::chat_prompt(SYSTEM, &format!("{question}\n\n{ANSWER_FORMAT}"))
}

/// The receiver's slotted injection prompt — VERBATIM the probe's wording
/// (`examples/common/m3.rs::four_conditions`, step 3).
pub fn injection_prompt(question: &str, n_slots: usize) -> String {
    let slots = "<|fim_pad|>".repeat(n_slots);
    QwenRuntime::chat_prompt(
        SYSTEM,
        &format!(
            "{question}\n\nA compressed latent hint from a previous solver is stored in these slots: [{slots}]\n\n{ANSWER_FORMAT}"
        ),
    )
}

/// One trainable item: the receiver-side teacher-forced sequence, its loss
/// span, and the dump coordinates of the sender's per-token rows.
pub struct TaskItem {
    /// Dataset row (0..2560) — the split/leakage unit.
    pub row: usize,
    /// GSM8K train item index.
    pub item: usize,
    /// Injection-prompt token ids (with the 8 placeholder slots).
    pub inj_len: usize,
    /// Positions of the placeholder slots within the injection prompt.
    pub slot_positions: Vec<usize>,
    /// `inj_tokens ‖ gen_tokens[..target_len]` — the training sequence.
    pub full_tokens: Vec<u32>,
    /// CE targets: `gen_tokens[..target_len]`.
    pub target_tokens: Vec<u32>,
    /// Logits-row start: `inj_len - 1` (row p predicts token p+1).
    pub span_start: usize,
    /// Global token offset of this item's block in the dump layer files.
    pub tok0: usize,
    /// Full generated-span row count (adapter input; NOT capped — pooling
    /// covers the full span exactly as the probe pools).
    pub n_rows: usize,
}

/// An item excluded by the sequence-cap rule, for the receipt.
#[derive(serde::Serialize)]
pub struct SkippedItem {
    pub row: usize,
    pub item: usize,
    pub inj_len: usize,
    pub target_fit: usize,
    pub reason: &'static str,
}

/// Build task items for a set of dataset rows under the frozen cap rule:
/// `target_len = min(gen_len, seq_cap - inj_len)`, skip if `< min_target`.
/// Every item passes the prompt-parity gate (see module docs) and the
/// slot-count assert before inclusion.
#[allow(clippy::too_many_arguments)]
pub fn build_task_items(
    rows: &[usize],
    streams: &[StreamRow],
    questions: &[String],
    tok: &tokenizers::Tokenizer,
    token_offsets: &[u64],
    pad_id: u32,
    n_slots: usize,
    seq_cap: usize,
    min_target: usize,
) -> anyhow::Result<(Vec<TaskItem>, Vec<SkippedItem>)> {
    let mut items = Vec::with_capacity(rows.len());
    let mut skipped = Vec::new();
    for &row in rows {
        let sr = &streams[row];
        let q = &questions[sr.item];
        // Prompt-parity gate: reconstructed capture prompt must re-encode to
        // the stored stream tokens exactly.
        let cap_ids = encode(tok, &capture_prompt(q))?;
        anyhow::ensure!(
            cap_ids == sr.prompt_tokens,
            "row {row} (item {}): prompt-parity gate failed — reconstructed capture prompt \
             re-encodes to {} tokens != stored {}",
            sr.item,
            cap_ids.len(),
            sr.prompt_tokens.len()
        );
        let inj_ids = encode(tok, &injection_prompt(q, n_slots))?;
        let slot_positions: Vec<usize> = inj_ids
            .iter()
            .enumerate()
            .filter_map(|(i, &t)| (t == pad_id).then_some(i))
            .collect();
        anyhow::ensure!(
            slot_positions.len() == n_slots,
            "row {row}: {} placeholder slots != {n_slots}",
            slot_positions.len()
        );
        let inj_len = inj_ids.len();
        let target_fit = seq_cap.saturating_sub(inj_len).min(sr.gen_tokens.len());
        if target_fit < min_target {
            skipped.push(SkippedItem {
                row,
                item: sr.item,
                inj_len,
                target_fit,
                reason: "target span under min_target after seq cap",
            });
            continue;
        }
        let target_tokens = sr.gen_tokens[..target_fit].to_vec();
        let mut full_tokens = inj_ids;
        full_tokens.extend_from_slice(&target_tokens);
        items.push(TaskItem {
            row,
            item: sr.item,
            inj_len,
            slot_positions,
            span_start: inj_len - 1,
            full_tokens,
            target_tokens,
            tok0: token_offsets[row] as usize,
            n_rows: sr.gen_tokens.len(),
        });
    }
    Ok((items, skipped))
}

/// Current per-pid VRAM (MiB) via nvidia-smi; `None` when unavailable.
pub fn process_vram_mib() -> Option<u64> {
    let pid = std::process::id();
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    String::from_utf8_lossy(&out.stdout).lines().find_map(|l| {
        let mut it = l.split(',').map(str::trim);
        match (it.next(), it.next()) {
            (Some(p), Some(m)) if p.parse::<u32>().ok() == Some(pid) => m.parse::<u64>().ok(),
            _ => None,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_shapes() {
        let cap = capture_prompt("Q?");
        assert!(cap.starts_with("<|im_start|>system\nYou are a careful math tutor.<|im_end|>"));
        assert!(cap.contains("Q?\n\nSolve the problem step by step."));
        assert!(cap.ends_with("<|im_start|>assistant\n"));
        let inj = injection_prompt("Q?", 8);
        assert_eq!(inj.matches("<|fim_pad|>").count(), 8);
        assert!(inj.contains(
            "A compressed latent hint from a previous solver is stored in these slots: ["
        ));
        // The injection prompt differs from the capture prompt ONLY by the
        // inserted slot sentence.
        let without: String = inj.replacen(
            "\n\nA compressed latent hint from a previous solver is stored in these slots: [<|fim_pad|><|fim_pad|><|fim_pad|><|fim_pad|><|fim_pad|><|fim_pad|><|fim_pad|><|fim_pad|>]",
            "",
            1,
        );
        assert_eq!(without, cap);
    }

    #[test]
    fn answer_format_matches_probe_revision2() {
        // Pin the duplicated constant's content shape (full parity with the
        // probe side is asserted at run time by the prompt-parity gate).
        assert!(ANSWER_FORMAT.starts_with("Solve the problem step by step."));
        assert!(ANSWER_FORMAT.contains("#### <numeric answer>"));
        assert!(ANSWER_FORMAT.ends_with("the last line must be: #### 5"));
    }
}
