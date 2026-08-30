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

// ---------------------------------------------------------------------------
// ADR-045 M5: question-tail site + gold-answer continuation target.
//
// M5's receiver prompt has NO slot sentence and NO placeholder token: it is
// exactly `chat_prompt(SYSTEM, "{question}\n\n{ANSWER_FORMAT}")`, which is
// BYTE-IDENTICAL to the sender's capture prompt. That is load-bearing rather
// than incidental — it means the existing prompt-parity gate (re-encode ==
// the S2c stream's stored `prompt_tokens`) pins M5's own injected prompt
// bit-for-bit, with no second copy of the wording to drift.
//
// The CE target is the probe's OWN likelihood target, `"#### {gold}"`
// (`examples/common/m3.rs::four_conditions_at`, step 5), not the sender's
// generated span — `docs/research/034` §5.2 names the sender-span target as
// M4c's diagnosed mismatch, and ADR-045 registers the gold continuation.
// ---------------------------------------------------------------------------

/// Normalised final answer: text after the last `####`, non-numeric characters
/// stripped, trailing period removed. Verbatim behaviour of the probe's
/// `examples/common/mod.rs::extract_final_answer` (which this crate cannot
/// import — the dependency runs the other way). Pinned by unit test below and
/// by the `#### {gold}` token comparison the M5 trainer records.
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

/// **The M5 training target after coordinator error #22.** Verbatim replica of
/// `examples/common/mod.rs::render_gold` (which this crate cannot import — the
/// dependency runs the other way), pinned by unit test below.
///
/// GSM8K's `answer` field is the human reference *solution*: the reasoning
/// followed by the `#### <n>` line the receiver's own `ANSWER_FORMAT` asks for.
/// The only edit is removal of the dataset's `<<a+b=c>>` calculator
/// annotations, an artifact of the GSM8K authoring tool that appears in no
/// model's output distribution. Nothing is added, in particular no EOS.
///
/// **Why this and not `"#### {gold}"`.** ADR-045 registered the answer line
/// alone. Trained on that, a rank-1 adapter cut holdout CE 2.3172 → 0.6643
/// (508W/2L) while GSM8K accuracy collapsed 31/64 → 5/64: raising the
/// likelihood of the answer line does not require solving the problem, and the
/// cheapest descent direction is to stop reasoning and emit it immediately.
/// The amended target CONTAINS the reasoning whose absence was the pathology.
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

/// GSM8K `(question, rendered gold solution)` by item index, sha-gated. The
/// second field is [`render_gold`]'s output — the amended training target —
/// not the normalised answer token.
pub fn load_gsm8k_items(path: &Path) -> anyhow::Result<Vec<(String, String)>> {
    let bytes = read_pinned(path, GSM8K_TRAIN_SHA256)?;
    let mut items = Vec::new();
    for (index, line) in String::from_utf8(bytes)?.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(line)?;
        let question = v["question"].as_str().unwrap_or_default().to_string();
        let answer = v["answer"].as_str().unwrap_or_default();
        // Every item must still carry a parsable answer line; the probe's
        // endpoint depends on it, so a solution without one is a data fault.
        extract_final_answer(answer)
            .ok_or_else(|| anyhow::anyhow!("item {index}: no '#### n' in gold answer"))?;
        items.push((question, render_gold(answer)));
    }
    Ok(items)
}

/// Whitespace-insensitive containment helper for the question-tail gate.
fn squeeze(s: &str) -> String {
    s.chars().filter(|c| !c.is_whitespace()).collect()
}

/// Resolve the `n_slots` question-tail delivery positions for one item —
/// the training-side replica of `examples/common/m3.rs::build_site_prompt`
/// under `Site::QuestionTail`, algorithm for algorithm.
///
/// Positions are read off the CANONICAL tokenisation's own offset map: the
/// last `n_slots` tokens whose byte span lies wholly inside `question`.
/// Re-encoding a prefix instead was measured to reject every item on the probe
/// side (Qwen2.5's pre-tokeniser groups a trailing `?` with the newlines that
/// follow it, so the boundary token straddles question/answer-format), which
/// is why the decode gate checks containment rather than a strict suffix.
///
/// Returns `(prompt_tokens, positions, position_token_ids)`.
pub fn question_tail_positions(
    tok: &tokenizers::Tokenizer,
    question: &str,
    n_slots: usize,
) -> anyhow::Result<(Vec<u32>, Vec<usize>, Vec<u32>)> {
    let user = format!("{question}\n\n{ANSWER_FORMAT}");
    let full = QwenRuntime::chat_prompt(SYSTEM, &user);
    let q_start = full
        .find(&user)
        .ok_or_else(|| anyhow::anyhow!("chat template did not contain the user turn"))?;
    let q_end = q_start + question.len();
    let enc = tok
        .encode(full.as_str(), false)
        .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
    let tokens = enc.get_ids().to_vec();
    let inside: Vec<usize> = enc
        .get_offsets()
        .iter()
        .enumerate()
        .filter(|(_, &(s, t))| t > s && s >= q_start && t <= q_end)
        .map(|(i, _)| i)
        .collect();
    anyhow::ensure!(
        inside.len() >= n_slots,
        "only {} tokens lie wholly inside the question, fewer than the {n_slots} required",
        inside.len()
    );
    let positions: Vec<usize> = inside[inside.len() - n_slots..].to_vec();
    anyhow::ensure!(
        positions.windows(2).all(|w| w[1] == w[0] + 1),
        "the resolved question-tail positions are not contiguous ({positions:?})"
    );
    anyhow::ensure!(
        positions[n_slots - 1] == inside[inside.len() - 1],
        "the tail window does not end at the last token wholly inside the question"
    );
    let position_token_ids: Vec<u32> = positions.iter().map(|&p| tokens[p]).collect();
    let decoded = tok
        .decode(&position_token_ids, true)
        .map_err(|e| anyhow::anyhow!("decode: {e}"))?;
    anyhow::ensure!(
        squeeze(question).contains(&squeeze(&decoded)),
        "the {n_slots} injected positions decode to {decoded:?}, which is not part of the question"
    );
    Ok((tokens, positions, position_token_ids))
}

/// One M5 trainable item.
pub struct M5Item {
    /// Dataset row (0..2560) — the split/leakage unit.
    pub row: usize,
    /// GSM8K train item index.
    pub item: usize,
    /// Receiver prompt length (== the S2c capture prompt's length).
    pub prompt_len: usize,
    /// The 8 question-tail delivery positions.
    pub positions: Vec<usize>,
    pub position_token_ids: Vec<u32>,
    /// `prompt ‖ render_gold(answer)[..target_len]` — the teacher-forced
    /// training sequence.
    pub full_tokens: Vec<u32>,
    /// CE targets: the rendered gold **solution** tokens, capped.
    pub target_tokens: Vec<u32>,
    /// Full rendered-solution token count before the cap — so the receipt can
    /// report how much of the reasoning each item's CE actually covered.
    pub target_tokens_total: usize,
    /// Logits-row start: `prompt_len - 1` (row p predicts token p+1).
    pub span_start: usize,
    /// Global token index of the item's LAST generated-span row in the dump —
    /// the only row `apply_last_row` reads.
    pub last_row_tok: usize,
}

impl M5Item {
    /// Fraction of the rendered gold solution the CE span covers.
    pub fn covered_fraction(&self) -> f64 {
        self.target_tokens.len() as f64 / self.target_tokens_total.max(1) as f64
    }
}

/// Build M5 items for a set of dataset rows. Every item passes the
/// prompt-parity gate, the question-tail site gate, and the sequence cap.
///
/// **Cap rule, M4c's verbatim** (`target_len = min(len, seq_cap - prompt_len)`,
/// skip only if `< min_target`): the rendered gold solution is TRUNCATED
/// rather than the item dropped. Dropping instead would silently keep only
/// short problems, which correlates with difficulty and would bias the fit set;
/// truncating keeps every item and costs the tail of the reasoning on the long
/// ones. The covered fraction is recorded per item and summarised in the
/// receipt rather than left implicit.
#[allow(clippy::too_many_arguments)]
pub fn build_m5_items(
    rows: &[usize],
    streams: &[StreamRow],
    items: &[(String, String)],
    tok: &tokenizers::Tokenizer,
    token_offsets: &[u64],
    n_slots: usize,
    seq_cap: usize,
    min_target: usize,
) -> anyhow::Result<(Vec<M5Item>, Vec<SkippedItem>)> {
    let mut out = Vec::with_capacity(rows.len());
    let mut skipped = Vec::new();
    for &row in rows {
        let sr = &streams[row];
        let (question, gold) = &items[sr.item];
        let (tokens, positions, position_token_ids) =
            question_tail_positions(tok, question, n_slots)
                .map_err(|e| anyhow::anyhow!("row {row} (item {}): {e}", sr.item))?;
        // Prompt-parity gate: M5's receiver prompt IS the capture prompt, so
        // this pins the injected prompt bit-for-bit against the S2c dump.
        anyhow::ensure!(
            tokens == sr.prompt_tokens,
            "row {row} (item {}): prompt-parity gate failed — the question-tail prompt \
             re-encodes to {} tokens != the stream's stored {}",
            sr.item,
            tokens.len(),
            sr.prompt_tokens.len()
        );
        // `gold` is render_gold's output: the full solution, `<<..>>` stripped.
        let full_target = encode(tok, gold)?;
        let prompt_len = tokens.len();
        let target_fit = seq_cap.saturating_sub(prompt_len).min(full_target.len());
        if target_fit < min_target {
            skipped.push(SkippedItem {
                row,
                item: sr.item,
                inj_len: prompt_len,
                target_fit,
                reason: "gold-solution span under min_target after the seq cap",
            });
            continue;
        }
        let target_tokens = full_target[..target_fit].to_vec();
        let mut full_tokens = tokens;
        full_tokens.extend_from_slice(&target_tokens);
        out.push(M5Item {
            row,
            item: sr.item,
            prompt_len,
            positions,
            position_token_ids,
            full_tokens,
            target_tokens,
            target_tokens_total: full_target.len(),
            span_start: prompt_len - 1,
            last_row_tok: token_offsets[row] as usize + sr.gen_tokens.len() - 1,
        });
    }
    Ok((out, skipped))
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

    /// The gold-rendering rule must behave exactly like the probe's
    /// `extract_final_answer` on the cases its own test pins.
    #[test]
    fn gold_extraction_matches_the_probe_rule() {
        assert_eq!(
            extract_final_answer("x #### 1,000").as_deref(),
            Some("1000")
        );
        assert_eq!(extract_final_answer("#### 18.").as_deref(), Some("18"));
        assert_eq!(extract_final_answer("#### -3").as_deref(), Some("-3"));
        assert_eq!(extract_final_answer("no marker"), None);
    }

    /// The amended M5 target must be the probe's own `render_gold` behaviour:
    /// calculator annotations stripped, everything else — the reasoning AND
    /// the `#### n` line — kept verbatim.
    #[test]
    fn render_gold_strips_only_calculator_annotations() {
        assert_eq!(render_gold("a <<1+1=2>>b"), "a b");
        assert_eq!(render_gold("no annotations"), "no annotations");
        assert_eq!(render_gold("x <<unclosed"), "x ");
        let answer = "Janet has 3 <<2+1=3>>eggs.\nShe sells 2.\n#### 1";
        let rendered = render_gold(answer);
        assert!(!rendered.contains("<<"));
        // The reasoning survives — that is the entire point of the amendment.
        assert!(rendered.contains("She sells 2."));
        assert!(rendered.ends_with("#### 1"));
        assert_eq!(extract_final_answer(&rendered).as_deref(), Some("1"));
    }

    #[test]
    fn m5_prompt_is_the_capture_prompt() {
        // ADR-045 M5 removes the slot sentence entirely, so the receiver
        // prompt collapses onto the capture prompt. `question_tail_positions`
        // tokenises exactly this string.
        let q = "Q?";
        let user = format!("{q}\n\n{ANSWER_FORMAT}");
        assert_eq!(QwenRuntime::chat_prompt(SYSTEM, &user), capture_prompt(q));
        assert!(!capture_prompt(q).contains("<|fim_pad|>"));
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
