//! S2 — calibration dump (design doc 024 §5.1–§5.2, §7 S2, runtime half).
//!
//! Teacher-forces the SAME text (chat-formatted question + gold solution)
//! through BOTH models (Qwen2.5-3B sender, Qwen2.5-1.5B receiver) as
//! prefill-only passes over the committed calibration-4000 GSM8K-train index
//! list, mean-pools the residual over the gold-solution token span at each of
//! the 3-per-model sweep depths ({50%, 66%, 80%} relative depth,
//! `layer = ceil(frac × n_layers)`: sender {18, 24, 29}/36, receiver
//! {14, 19, 23}/28), and dumps one raw little-endian f32 matrix per
//! (model, depth) plus `manifest.json` to
//! `target/latentmesh-runs/s2/`. The workspace-side fitter
//! (`harness/latentmesh-live calibrate`) consumes the dump — the two crates
//! cannot share a cargo dependency graph, so the boundary is files only.
//!
//! Choices recorded for auditability:
//! - Gold text normalization: GSM8K calculator annotations `<<...>>` are
//!   stripped (generated text never contains them; reduces the §5.4
//!   registered gold-vs-generated distribution shift). The `#### n` line is
//!   kept — live pooling covers the full generated span, which includes it.
//! - Prompt/solution are tokenized separately and concatenated, mirroring
//!   the generation-time prompt/continuation token boundary.
//! - Models run sequentially (sender fully, then receiver): calibration has
//!   no cross-model step per item, and §4's concurrent-residency requirement
//!   applies to the four conditions, not calibration.
//! - Any per-item failure ABORTS the run: the design's n>d fix needs all
//!   4,000 rows (3,200 fit rows vs sender dim 2,048 has zero slack).
//!
//! Run: PATH=/usr/local/cuda-12.8/bin:$PATH \
//!      cargo run --release --features cuda --example s2_calibrate_dump

#[path = "common/mod.rs"]
#[allow(dead_code)]
mod common;

use latentmesh_runtime::{
    capture::{forward_capture_multi, forward_unpatched, logits_bit_identical},
    layer_at_relative_depth, QwenRuntime,
};
use std::io::Write as _;
use std::path::{Path, PathBuf};

const SENDER: &str = "Qwen/Qwen2.5-3B-Instruct";
const RECEIVER: &str = "Qwen/Qwen2.5-1.5B-Instruct";
const SYSTEM: &str = "You are a careful math tutor.";
const DEPTH_FRACS: [f64; 3] = [0.50, 0.66, 0.80];
const SPLIT_NAME: &str = "calibration-4000";

fn harness_data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../harness/latentmesh-live/data")
}

/// Strip GSM8K calculator annotations `<<...>>` from the gold solution.
fn normalize_gold(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find("<<") {
        out.push_str(&rest[..start]);
        match rest[start..].find(">>") {
            Some(end) => rest = &rest[start + end + 2..],
            None => {
                rest = &rest[start + 2..];
            }
        }
    }
    out.push_str(rest);
    out
}

#[derive(serde::Deserialize)]
struct SplitFile {
    split: String,
    source: String,
    source_sha256: String,
    seed: u64,
    indices: Vec<usize>,
}

struct PhaseOut {
    /// Flat row-major pooled rows per swept layer, `layers[i] -> n_rows × dim`.
    buffers: Vec<Vec<f32>>,
    layers: Vec<usize>,
    hidden_size: usize,
    n_layers: usize,
    parity_bit_identical: bool,
    wall_s: f64,
}

/// One model's full teacher-forced pass over all items.
fn run_phase(
    model_id: &str,
    token_streams: &mut Vec<(Vec<u32>, usize)>,
    check_tokens_against_existing: bool,
    items: &[(usize, String, String)],
    t0: &std::time::Instant,
) -> anyhow::Result<PhaseOut> {
    let e = anyhow::Error::msg;
    let p0 = std::time::Instant::now();
    let device = latentmesh_runtime::device().map_err(e)?;
    let mut rt = QwenRuntime::load(model_id, &device, candle_core::DType::BF16).map_err(e)?;
    let n_layers = rt.config.num_hidden_layers;
    let hidden = rt.config.hidden_size;
    let layers: Vec<usize> = DEPTH_FRACS
        .iter()
        .map(|&f| layer_at_relative_depth(f, n_layers))
        .collect();
    println!("{model_id}: {n_layers} layers, hidden {hidden}, sweep taps after blocks {layers:?}");

    let mut buffers: Vec<Vec<f32>> = vec![Vec::with_capacity(items.len() * hidden); layers.len()];
    let mut parity = false;
    for (row, (idx, prompt, solution)) in items.iter().enumerate() {
        // Tokenize prompt and solution separately (generation-boundary
        // mirror); token ids must be IDENTICAL across the two models (§2:
        // same tokenizer — trivial pairing). Asserted, not assumed.
        let p_toks = rt.encode(prompt).map_err(e)?;
        let s_toks = rt.encode(solution).map_err(e)?;
        anyhow::ensure!(!s_toks.is_empty(), "item {idx}: empty solution tokens");
        let full: Vec<u32> = p_toks.iter().chain(s_toks.iter()).copied().collect();
        if check_tokens_against_existing {
            anyhow::ensure!(
                token_streams[row] == (full.clone(), p_toks.len()),
                "item {idx}: token stream differs between models — same-tokenizer pairing assumption broken"
            );
        } else {
            token_streams.push((full.clone(), p_toks.len()));
        }
        let span = p_toks.len()..full.len();
        let (logits, caps) = forward_capture_multi(&mut rt.model, &full, &layers, span, &device)
            .map_err(|err| anyhow::anyhow!("item {idx} (row {row}): {err}"))?;
        if row == 0 {
            // Multi-tap parity gate: the S0 evidence covers single-tap
            // Capture only; re-measure for CaptureMany rather than trusting
            // the clone-only argument.
            let ref_logits = forward_unpatched(&mut rt.model, &full, &device).map_err(e)?;
            parity = logits_bit_identical(&logits, &ref_logits).map_err(e)?;
            anyhow::ensure!(parity, "multi-tap logits parity FAILED for {model_id}");
        }
        for (b, cap) in buffers.iter_mut().zip(&caps) {
            anyhow::ensure!(
                cap.hidden_size == hidden,
                "item {idx}: capture dim {} != {hidden}",
                cap.hidden_size
            );
            b.extend_from_slice(&cap.pooled);
        }
        if (row + 1) % 200 == 0 {
            println!(
                "  [{}/{}] {:.0}s elapsed (total {:.0}s)",
                row + 1,
                items.len(),
                p0.elapsed().as_secs_f32(),
                t0.elapsed().as_secs_f32()
            );
        }
    }
    Ok(PhaseOut {
        buffers,
        layers,
        hidden_size: hidden,
        n_layers,
        parity_bit_identical: parity,
        wall_s: p0.elapsed().as_secs_f64(),
    })
}

/// Write one raw little-endian f32 bin; return (file_name, sha256).
fn write_bin(dir: &Path, name: &str, data: &[f32]) -> anyhow::Result<(String, String)> {
    let path = dir.join(name);
    let mut f = std::io::BufWriter::new(std::fs::File::create(&path)?);
    for v in data {
        f.write_all(&v.to_le_bytes())?;
    }
    f.flush()?;
    drop(f);
    let sha = common::sha256_hex(&std::fs::read(&path)?);
    Ok((name.to_string(), sha))
}

fn side_json(
    tag: &str,
    model: &str,
    out: &PhaseOut,
    files: &[(String, String)],
) -> serde_json::Value {
    serde_json::json!({
        "model": model,
        "n_layers": out.n_layers,
        "hidden_size": out.hidden_size,
        "layers": out.layers,
        "files": out.layers.iter().zip(files).map(|(l, (f, s))| {
            (l.to_string(), serde_json::json!({"file": f, "sha256": s}))
        }).collect::<serde_json::Map<_, _>>(),
        "tag": tag,
    })
}

fn main() -> anyhow::Result<()> {
    let t0 = std::time::Instant::now();
    let nvcc = latentmesh_runtime::assert_cuda_build_env().map_err(anyhow::Error::msg)?;
    println!("build-env guard: {nvcc}");

    // Committed calibration index list + its source file, sha-verified.
    let split_path = harness_data_dir().join(format!("{SPLIT_NAME}.json"));
    let split_bytes = std::fs::read(&split_path).map_err(|e| {
        anyhow::anyhow!(
            "read {} (run `latentmesh-live make-splits` first): {e}",
            split_path.display()
        )
    })?;
    let split_sha = common::sha256_hex(&split_bytes);
    let split: SplitFile = serde_json::from_slice(&split_bytes)?;
    anyhow::ensure!(
        split.split == SPLIT_NAME,
        "split file mislabeled: {}",
        split.split
    );
    let data_path = harness_data_dir().join(&split.source);
    let source_sha = common::sha256_hex(&std::fs::read(&data_path)?);
    anyhow::ensure!(
        source_sha == split.source_sha256,
        "{}: sha256 {source_sha} != split file's {}",
        data_path.display(),
        split.source_sha256
    );
    let all_items = common::load_gsm8k(&data_path)?;
    println!(
        "{}: {} items (sha256 {source_sha}); calibration list: {} indices, seed {:#x}",
        split.source,
        all_items.len(),
        split.indices.len(),
        split.seed
    );

    // Teacher-forced texts, in index-list order (row i <-> indices[i]).
    let items: Vec<(usize, String, String)> = split
        .indices
        .iter()
        .map(|&idx| {
            let it = &all_items[idx];
            let prompt = QwenRuntime::chat_prompt(
                SYSTEM,
                &format!("{}\n\n{}", it.question, common::ANSWER_FORMAT),
            );
            (idx, prompt, normalize_gold(&it.answer_text))
        })
        .collect();

    let dir = common::run_dir("s2");
    let mut token_streams: Vec<(Vec<u32>, usize)> = Vec::with_capacity(items.len());

    // Sequential phases: sender fully, then receiver (see module doc).
    let sender_out = run_phase(SENDER, &mut token_streams, false, &items, &t0)?;
    let receiver_out = run_phase(RECEIVER, &mut token_streams, true, &items, &t0)?;

    let write_side = |tag: &str, out: &PhaseOut| -> anyhow::Result<Vec<(String, String)>> {
        out.layers
            .iter()
            .zip(&out.buffers)
            .map(|(l, buf)| {
                anyhow::ensure!(
                    buf.len() == items.len() * out.hidden_size,
                    "{tag} L{l}: {} values != {} rows x {}",
                    buf.len(),
                    items.len(),
                    out.hidden_size
                );
                write_bin(&dir, &format!("{tag}_L{l}.f32bin"), buf)
            })
            .collect()
    };
    let sender_files = write_side("sender", &sender_out)?;
    let receiver_files = write_side("receiver", &receiver_out)?;

    let token_totals: usize = token_streams.iter().map(|(t, _)| t.len()).sum();
    let solution_tokens: usize = token_streams.iter().map(|(t, p)| t.len() - p).sum();
    let manifest = serde_json::json!({
        "n_rows": items.len(),
        "item_indices": split.indices,
        "sender": side_json("sender", SENDER, &sender_out, &sender_files),
        "receiver": side_json("receiver", RECEIVER, &receiver_out, &receiver_files),
        "format": "raw little-endian f32, row-major n_rows x hidden_size; row i pairs with item_indices[i]",
        "pooling": "mean over the gold-solution token span (prompt tokens excluded) of the residual after block L",
        "depth_rule": "layer = ceil(frac x n_layers), fracs [0.50, 0.66, 0.80] (reproduces design anchors 24/36 and 19/28)",
        "gold_text_normalization": "calculator annotations <<...>> removed; '#### n' line kept",
        "prompt": {"system": SYSTEM, "answer_format_appended": common::ANSWER_FORMAT,
                    "tokenization": "prompt and solution encoded separately, concatenated (generation-boundary mirror)"},
        "dataset": {"source": split.source, "source_sha256": source_sha,
                     "split_file": format!("{SPLIT_NAME}.json"), "split_file_sha256": split_sha,
                     "split_seed_chacha8": split.seed,
                     "indices_min": split.indices.iter().min(), "indices_max": split.indices.iter().max()},
    });
    let manifest_path = common::write_receipt(&dir, "manifest.json", &manifest)?;
    let manifest_sha = common::sha256_hex(&std::fs::read(&manifest_path)?);

    let receipt = serde_json::json!({
        "stage": "S2-dump",
        "design": "docs/research/024-live-latent-experiment-design.md sections 5.1-5.2, 7 S2",
        "env": common::env_info(&nvcc),
        "decoding": "none (teacher-forced prefill only; deterministic, no sampling)",
        "manifest_sha256": manifest_sha,
        "dataset": manifest["dataset"].clone(),
        "train_indices_only": {
            "source_file": split.source,
            "note": "all rows drawn from the committed calibration-4000 GSM8K-TRAIN index list; eval/holdout live in gsm8k-test.jsonl and are mechanically refused by the harness until genome freeze",
            "n_indices": split.indices.len(),
        },
        "gates": {
            "all_4000_rows_present": {"n_rows": items.len(), "pass": items.len() == 4000,
                                        "note": "abort-on-any-failure policy: a missing row would silently break the n>d fit gate"},
            "sender_capture_dim": {"measured": sender_out.hidden_size, "expect": 2048, "pass": sender_out.hidden_size == 2048},
            "receiver_capture_dim": {"measured": receiver_out.hidden_size, "expect": 1536, "pass": receiver_out.hidden_size == 1536},
            "sender_multi_tap_logits_parity_bit_identical": {"pass": sender_out.parity_bit_identical},
            "receiver_multi_tap_logits_parity_bit_identical": {"pass": receiver_out.parity_bit_identical},
            "token_streams_identical_across_models": {"pass": true, "note": "asserted per item during the receiver phase"},
        },
        "tokens": {"total_teacher_forced": token_totals, "solution_span_total": solution_tokens,
                    "mean_per_item": token_totals as f64 / items.len() as f64},
        "wall_clock_s": {"sender_phase": sender_out.wall_s, "receiver_phase": receiver_out.wall_s,
                          "total": t0.elapsed().as_secs_f64()},
    });
    common::write_receipt(&dir, "s2-dump-receipt.json", &receipt)?;
    println!(
        "S2 dump complete: {} rows x (3 sender + 3 receiver depths), {:.0}s total",
        items.len(),
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}
