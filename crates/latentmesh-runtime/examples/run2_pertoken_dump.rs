//! Run 2 — per-token paired capture over the saved S2c token streams
//! (M2 of the run-2 thought-adapter plan; data-shape scout spec 2026-08-28).
//!
//! Run 1 (ADR-023) proved pooled linear maps carry no functional signal; run
//! 2 trains sequence adapters and therefore needs the PRE-POOLING per-token
//! rows the pooled S2c asset averaged away. This example re-derives them
//! with NO new generation: every item's full prompt+generated token stream
//! was saved during the S2c dump (now committed at
//! `harness/latentmesh-live/data/s2c-token-streams.jsonl`, sha-pinned
//! below), so the whole capture is teacher-forced prefill through both
//! models — the exact convention S2c phase 2 already used — at minutes of
//! GPU time instead of the ~2.5 GPU-h the generations originally cost.
//!
//! PRE-COMMITTED CHOICES (fixed in this source before the run):
//!
//! 1. **Spans: GENERATED tokens only** — matches the S2c pooling convention
//!    ("mean over the sender-GENERATED token span") and the live transmit
//!    distribution. 719,115 tokens total (measured from the streams file;
//!    gated exactly below).
//! 2. **Cells: the two ADR-023-registered cells only** — sender {L18, L24},
//!    receiver {L14, L19}. Same as S2c.
//! 3. **Format: ragged concatenated blocks.** Per layer one file
//!    `sender_L18.tok.f32bin` etc. = concatenation of per-item
//!    `[T_i x dim]` row-major little-endian f32 blocks in streams-row
//!    order, plus ONE shared index (`run2-pertoken-index.json`): cumulative
//!    token offsets (valid for all four files — T_i is identical across
//!    models per item by the tokenizer-parity gate), item indices,
//!    prompt/gen lengths, per-file sha256s. A training loop mmaps a layer
//!    file and reinterprets `&[f32]`; item i's block starts at byte
//!    `token_offsets[i] * dim * 4`.
//! 4. **Writes are block-buffered, not row-flushed** — the S2c per-row
//!    flush pattern would make a ~21 GB dump I/O-bound; this run is minutes
//!    long, so full restart (not resume) is the crash-recovery path.
//! 5. **Verification gates** (all measured, recorded in the receipt):
//!    streams sha256; per-item prompt re-encoding asserted equal to the
//!    stored stream in BOTH model phases (token streams identical across
//!    models); per-item `rows.len() == T_i x dim`; multi-tap logits parity
//!    (bit-identical) per model; total generated tokens == 719,115 exactly;
//!    per-file byte sizes == tokens x dim x 4 exactly; index round-trip
//!    (reload, seek 5 seeded-random items, verify block shapes); pooled-
//!    mean recomputation on 8 seeded-random items — the f64 mean of the
//!    dumped per-token rows must reproduce the committed S2c pooled vector
//!    (bins sha-verified against `receipts/s2c-manifest.json`) within
//!    `|diff| <= 1e-2 + 1e-3 x |pooled|` per element, proving capture-path
//!    consistency with run 1's data.
//!
//! Run: PATH=/usr/local/cuda-12.8/bin:$PATH \
//!      cargo run --release --features cuda --example run2_pertoken_dump
//! Smoke (non-evidence, separate dir, receipt stays in the run dir):
//!      append `-- --smoke 2`.

#[path = "common/mod.rs"]
#[allow(dead_code)]
mod common;

use latentmesh_runtime::{
    capture::{forward_capture_multi_with_rows, forward_unpatched, logits_bit_identical},
    QwenRuntime,
};
use rand::seq::SliceRandom as _;
use rand::SeedableRng as _;
use sha2::Digest as _;
use std::io::{Read as _, Seek as _, Write as _};
use std::path::{Path, PathBuf};

const SENDER: &str = "Qwen/Qwen2.5-3B-Instruct";
const RECEIVER: &str = "Qwen/Qwen2.5-1.5B-Instruct";
const SYSTEM: &str = "You are a careful math tutor.";
const SENDER_LAYERS: [usize; 2] = [18, 24];
const RECEIVER_LAYERS: [usize; 2] = [14, 19];
const SENDER_DIM: usize = 2048;
const RECEIVER_DIM: usize = 1536;
/// Committed durable copy of the S2c token streams (sole encoding of the
/// ~2.5 GPU-h of greedy generations; commit 540c34e).
const STREAMS_FILE: &str = "s2c-token-streams.jsonl";
const STREAMS_SHA256: &str = "ad5713116cb5acf663fe9c33ac58aaedb1fd64fcf629dd78b21439d244958539";
const EXPECT_ROWS: usize = 2560;
/// Total generated tokens across all 2560 streams (measured by the
/// data-shape scout; cross-checked against the S2c receipt's
/// rows.generated_tokens_total). Gated exactly on full runs.
const EXPECT_TOTAL_GEN_TOKENS: u64 = 719_115;
/// ChaCha8 seed for the post-write verification draws ("RUN2" in ASCII).
const CHECK_SEED: u64 = 0x5255_4E32;
const INDEX_ROUND_TRIP_ITEMS: usize = 5;
const POOLED_CHECK_ITEMS: usize = 8;

fn harness_data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../harness/latentmesh-live/data")
}

fn receipts_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("receipts")
}

/// One saved S2c item: the sender-generated token stream both models
/// teacher-force (same schema the S2c dump wrote).
#[derive(serde::Deserialize)]
struct StreamRow {
    row: usize,
    item: usize,
    prompt_tokens: Vec<u32>,
    gen_tokens: Vec<u32>,
    #[allow(dead_code)]
    sender_first_pass_correct: bool,
}

/// Block-buffered ragged f32 appender with incremental sha256 (choice 4:
/// one `write_all` per item block; flush only at finish).
struct TokBinWriter {
    file: std::io::BufWriter<std::fs::File>,
    hasher: sha2::Sha256,
    path: PathBuf,
    dim: usize,
    tokens: u64,
    buf: Vec<u8>,
}

/// A finished ragged layer file: name, byte size, sha256, token count.
struct TokBinFile {
    name: String,
    path: PathBuf,
    dim: usize,
    tokens: u64,
    bytes: u64,
    sha256: String,
}

impl TokBinWriter {
    fn create(dir: &Path, name: &str, dim: usize) -> anyhow::Result<Self> {
        let path = dir.join(name);
        Ok(Self {
            file: std::io::BufWriter::with_capacity(16 << 20, std::fs::File::create(&path)?),
            hasher: sha2::Sha256::new(),
            path,
            dim,
            tokens: 0,
            buf: Vec::new(),
        })
    }
    /// Append one item's `[t x dim]` row-major block.
    fn append_block(&mut self, rows: &[f32], t: usize) -> anyhow::Result<()> {
        anyhow::ensure!(
            rows.len() == t * self.dim,
            "{}: block len {} != {t} x {}",
            self.path.display(),
            rows.len(),
            self.dim
        );
        self.buf.clear();
        self.buf.reserve(rows.len() * 4);
        for v in rows {
            self.buf.extend_from_slice(&v.to_le_bytes());
        }
        self.hasher.update(&self.buf);
        self.file.write_all(&self.buf)?;
        self.tokens += t as u64;
        Ok(())
    }
    fn finish(self) -> anyhow::Result<TokBinFile> {
        let Self {
            mut file,
            hasher,
            path,
            dim,
            tokens,
            ..
        } = self;
        file.flush()?;
        drop(file);
        let bytes = std::fs::metadata(&path)?.len();
        Ok(TokBinFile {
            name: path.file_name().unwrap().to_string_lossy().to_string(),
            path,
            dim,
            tokens,
            bytes,
            sha256: format!("{:x}", hasher.finalize()),
        })
    }
}

/// Read item `i`'s `[gen_len[i] x dim]` block from a ragged layer file via
/// the shared index (seek + exact read — the round-trip the index exists
/// to serve).
fn read_block(
    path: &Path,
    dim: usize,
    token_offsets: &[u64],
    gen_len: &[u64],
    i: usize,
) -> anyhow::Result<Vec<f32>> {
    let mut f = std::fs::File::open(path)?;
    f.seek(std::io::SeekFrom::Start(token_offsets[i] * dim as u64 * 4))?;
    let n = gen_len[i] as usize * dim;
    let mut bytes = vec![0u8; n * 4];
    f.read_exact(&mut bytes)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}

fn parse_smoke() -> Option<usize> {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == "--smoke")
        .map(|i| args[i + 1].parse().expect("--smoke N"))
}

#[allow(clippy::too_many_lines)]
fn main() -> anyhow::Result<()> {
    let t0 = std::time::Instant::now();
    let e = anyhow::Error::msg;
    let smoke = parse_smoke();
    let nvcc = latentmesh_runtime::assert_cuda_build_env().map_err(e)?;
    println!("build-env guard: {nvcc}  smoke={smoke:?}");

    // ---- Load + verify the committed token streams -----------------------
    let streams_path = harness_data_dir().join(STREAMS_FILE);
    let streams_bytes = std::fs::read(&streams_path)?;
    let streams_sha = common::sha256_hex(&streams_bytes);
    anyhow::ensure!(
        streams_sha == STREAMS_SHA256,
        "{}: sha256 {streams_sha} != pinned {STREAMS_SHA256}",
        streams_path.display()
    );
    let rows: Vec<StreamRow> = String::from_utf8(streams_bytes)?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).map_err(anyhow::Error::from))
        .collect::<anyhow::Result<_>>()?;
    anyhow::ensure!(
        rows.len() == EXPECT_ROWS,
        "streams: {} rows != {EXPECT_ROWS}",
        rows.len()
    );
    for (i, r) in rows.iter().enumerate() {
        anyhow::ensure!(r.row == i, "streams row {i} carries row id {}", r.row);
        anyhow::ensure!(!r.gen_tokens.is_empty(), "streams row {i}: empty gen span");
    }

    // Committed S2c manifest: row order + pooled-bin sha256s (the pooled
    // vectors the recomputation gate reproduces).
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(receipts_dir().join("s2c-manifest.json"))?)?;
    let manifest_items: Vec<usize> = manifest["item_indices"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    anyhow::ensure!(
        manifest_items.len() == rows.len()
            && manifest_items.iter().zip(&rows).all(|(m, r)| *m == r.item),
        "streams item order != committed s2c-manifest item_indices"
    );

    // Dataset (for rebuilding each item's prompt text): same loading
    // discipline as the S2c dump, additionally pinned to the shas the
    // committed manifest recorded.
    let split_path = harness_data_dir().join("calibration-4000.json");
    let split_bytes = std::fs::read(&split_path)?;
    anyhow::ensure!(
        common::sha256_hex(&split_bytes) == manifest["dataset"]["split_file_sha256"],
        "calibration-4000.json sha drifted from committed s2c manifest"
    );
    let split: serde_json::Value = serde_json::from_slice(&split_bytes)?;
    let data_path = harness_data_dir().join(split["source"].as_str().unwrap());
    let source_sha = common::sha256_hex(&std::fs::read(&data_path)?);
    anyhow::ensure!(
        source_sha == manifest["dataset"]["source_sha256"],
        "{}: sha drifted from committed s2c manifest",
        data_path.display()
    );
    let all_items = common::load_gsm8k(&data_path)?;
    println!(
        "streams verified: {} rows, sha {streams_sha}; dataset {} items",
        rows.len(),
        all_items.len()
    );

    let dir = common::run_dir(if smoke.is_some() {
        "run2-smoke"
    } else {
        "run2"
    });
    let rows: &[StreamRow] = match smoke {
        Some(n) => &rows[..n.min(rows.len())],
        None => &rows,
    };
    let n_items = rows.len();
    let prompt_for = |row: &StreamRow| {
        QwenRuntime::chat_prompt(
            SYSTEM,
            &format!(
                "{}\n\n{}",
                all_items[row.item].question,
                common::ANSWER_FORMAT
            ),
        )
    };

    let device = latentmesh_runtime::device().map_err(e)?;

    // ---- Phase per model: teacher-forced prefill, dump span rows ---------
    // Returns (files, parity_pass, phase_seconds).
    let run_phase = |model_id: &str,
                     dim: usize,
                     layers: &[usize],
                     prefix: &str|
     -> anyhow::Result<(Vec<TokBinFile>, bool, f64)> {
        println!("loading {model_id}...");
        let p = std::time::Instant::now();
        let mut rt = QwenRuntime::load(model_id, &device, candle_core::DType::BF16).map_err(e)?;
        anyhow::ensure!(rt.config.hidden_size == dim);
        println!("loaded in {:.1}s", p.elapsed().as_secs_f64());
        let mut bins: Vec<TokBinWriter> = layers
            .iter()
            .map(|l| TokBinWriter::create(&dir, &format!("{prefix}_L{l}.tok.f32bin"), dim))
            .collect::<anyhow::Result<_>>()?;
        let mut parity = false;
        for (i, row) in rows.iter().enumerate() {
            // Token-stream parity across models: THIS model's encoding of
            // the prompt must reproduce the stored stream (asserted in both
            // phases; the generated span is shared as raw ids).
            let enc = rt.encode(&prompt_for(row)).map_err(e)?;
            anyhow::ensure!(
                enc == row.prompt_tokens,
                "item {}: {model_id} prompt encoding differs from stored stream",
                row.item
            );
            let full: Vec<u32> = row
                .prompt_tokens
                .iter()
                .chain(row.gen_tokens.iter())
                .copied()
                .collect();
            let span = row.prompt_tokens.len()..full.len();
            let t_i = row.gen_tokens.len();
            let (logits, caps) =
                forward_capture_multi_with_rows(&mut rt.model, &full, layers, span, &device)
                    .map_err(|err| anyhow::anyhow!("item {}: {err}", row.item))?;
            if i == 0 {
                let ref_logits = forward_unpatched(&mut rt.model, &full, &device).map_err(e)?;
                parity = logits_bit_identical(&logits, &ref_logits).map_err(e)?;
                anyhow::ensure!(parity, "{model_id} multi-tap logits parity FAILED");
            }
            for (b, cap) in bins.iter_mut().zip(&caps) {
                anyhow::ensure!(cap.capture.hidden_size == dim);
                b.append_block(&cap.rows, t_i)?;
            }
            if (i + 1) % 256 == 0 {
                println!("  [{}/{n_items}] {:.0}s", i + 1, p.elapsed().as_secs_f32());
            }
        }
        let files = bins
            .into_iter()
            .map(TokBinWriter::finish)
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok((files, parity, p.elapsed().as_secs_f64()))
    };

    let (sender_files, sender_parity, sender_phase_s) =
        run_phase(SENDER, SENDER_DIM, &SENDER_LAYERS, "sender")?;
    let (receiver_files, receiver_parity, receiver_phase_s) =
        run_phase(RECEIVER, RECEIVER_DIM, &RECEIVER_LAYERS, "receiver")?;
    let files: Vec<&TokBinFile> = sender_files.iter().chain(&receiver_files).collect();

    // ---- Shared index -----------------------------------------------------
    let gen_len: Vec<u64> = rows.iter().map(|r| r.gen_tokens.len() as u64).collect();
    let mut token_offsets: Vec<u64> = Vec::with_capacity(n_items + 1);
    token_offsets.push(0);
    for g in &gen_len {
        token_offsets.push(token_offsets.last().unwrap() + g);
    }
    let total_tokens = *token_offsets.last().unwrap();
    if smoke.is_none() {
        anyhow::ensure!(
            total_tokens == EXPECT_TOTAL_GEN_TOKENS,
            "total generated tokens {total_tokens} != expected {EXPECT_TOTAL_GEN_TOKENS}"
        );
    }
    let index = serde_json::json!({
        "format": "per layer file: concatenation of per-item [T_i x dim] row-major little-endian f32 blocks in streams-row order; item i's block starts at byte token_offsets[i] * dim * 4 and holds gen_len[i] rows; token_offsets is shared by all four files (T_i identical across models per item, tokenizer-parity gate)",
        "spans": "sender-GENERATED tokens only (prompt excluded) — the S2c pooling convention",
        "streams": {"file": STREAMS_FILE, "sha256": streams_sha},
        "n_items": n_items,
        "total_tokens": total_tokens,
        "files": files.iter().map(|f| (f.name.clone(), serde_json::json!({
            "dim": f.dim, "tokens": f.tokens, "bytes": f.bytes, "sha256": f.sha256,
        }))).collect::<serde_json::Map<_, _>>(),
        "item_indices": rows.iter().map(|r| r.item).collect::<Vec<_>>(),
        "prompt_len": rows.iter().map(|r| r.prompt_tokens.len()).collect::<Vec<_>>(),
        "gen_len": gen_len,
        "token_offsets": token_offsets,
    });
    let index_path = common::write_receipt(&dir, "run2-pertoken-index.json", &index)?;
    let index_sha = common::sha256_hex(&std::fs::read(&index_path)?);

    // ---- Post-write verification gates ------------------------------------
    let v0 = std::time::Instant::now();

    // Gate: per-file byte sizes == tokens x dim x 4 exactly.
    let mut size_gate = Vec::new();
    let mut sizes_pass = true;
    for f in &files {
        let expect = f.tokens * f.dim as u64 * 4;
        let pass = f.bytes == expect && f.tokens == total_tokens;
        sizes_pass &= pass;
        size_gate.push(serde_json::json!({
            "file": f.name, "tokens": f.tokens, "bytes": f.bytes,
            "expect_bytes": expect, "pass": pass,
        }));
    }
    anyhow::ensure!(sizes_pass, "file byte-size gate failed: {size_gate:?}");

    // Gate: index round-trips — reload from disk, seek seeded-random items
    // in every file, verify block shapes.
    let reloaded: serde_json::Value = serde_json::from_slice(&std::fs::read(&index_path)?)?;
    let r_offsets: Vec<u64> = reloaded["token_offsets"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap())
        .collect();
    let r_gen_len: Vec<u64> = reloaded["gen_len"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap())
        .collect();
    anyhow::ensure!(
        r_offsets.len() == n_items + 1
            && r_offsets[0] == 0
            && r_gen_len.len() == n_items
            && (0..n_items).all(|i| r_offsets[i + 1] - r_offsets[i] == r_gen_len[i])
            && r_offsets[n_items] == total_tokens,
        "reloaded index offsets inconsistent"
    );
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(CHECK_SEED);
    let mut order: Vec<usize> = (0..n_items).collect();
    order.shuffle(&mut rng);
    let rt_items: Vec<usize> = order
        .iter()
        .copied()
        .take(INDEX_ROUND_TRIP_ITEMS.min(n_items))
        .collect();
    for &i in &rt_items {
        for f in &files {
            let block = read_block(&f.path, f.dim, &r_offsets, &r_gen_len, i)?;
            anyhow::ensure!(
                block.len() == r_gen_len[i] as usize * f.dim,
                "round-trip: {} item {i} block shape mismatch",
                f.name
            );
        }
    }
    println!("index round-trip OK on items {rt_items:?}");

    // Gate: pooled-mean recomputation — the f64 mean of the dumped per-token
    // rows must reproduce the committed S2c pooled vector (capture-path
    // consistency with run 1's data). S2c bins sha-verified first.
    order.shuffle(&mut rng);
    let pooled_items: Vec<usize> = order
        .iter()
        .copied()
        .take(POOLED_CHECK_ITEMS.min(n_items))
        .collect();
    let s2c_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/latentmesh-runs/s2c");
    let mut max_abs_diff = 0f64;
    let mut max_rel_diff = 0f64;
    let sides = [
        ("sender", &SENDER_LAYERS, SENDER_DIM, &sender_files),
        ("receiver", &RECEIVER_LAYERS, RECEIVER_DIM, &receiver_files),
    ];
    for (side, layers, dim, side_files) in sides {
        for (l, f) in layers.iter().zip(side_files) {
            let entry = &manifest[side]["files"][l.to_string()];
            let pooled_path = s2c_dir.join(entry["file"].as_str().unwrap());
            let pooled_bytes = std::fs::read(&pooled_path)?;
            anyhow::ensure!(
                common::sha256_hex(&pooled_bytes) == entry["sha256"],
                "{}: sha256 differs from committed s2c-manifest — refusing pooled check",
                pooled_path.display()
            );
            for &i in &pooled_items {
                let block = read_block(&f.path, dim, &r_offsets, &r_gen_len, i)?;
                let t = r_gen_len[i] as usize;
                let mut mean = vec![0f64; dim];
                for row in block.chunks_exact(dim) {
                    for (m, v) in mean.iter_mut().zip(row) {
                        *m += f64::from(*v);
                    }
                }
                for m in &mut mean {
                    *m /= t as f64;
                }
                let start = i * dim * 4;
                for (d, m) in mean.iter().enumerate() {
                    let b = &pooled_bytes[start + d * 4..start + d * 4 + 4];
                    let p = f64::from(f32::from_le_bytes(b.try_into().unwrap()));
                    let diff = (m - p).abs();
                    max_abs_diff = max_abs_diff.max(diff);
                    max_rel_diff = max_rel_diff.max(diff / p.abs().max(1e-12));
                    anyhow::ensure!(
                        diff <= 1e-2 + 1e-3 * p.abs(),
                        "pooled recompute FAILED: {} item {i} dim {d}: mean {m} vs pooled {p}",
                        f.name
                    );
                }
            }
        }
    }
    let verify_s = v0.elapsed().as_secs_f64();
    println!(
        "pooled recompute OK on items {pooled_items:?}: max_abs_diff {max_abs_diff:.3e}, max_rel_diff {max_rel_diff:.3e}"
    );

    // ---- Receipt ----------------------------------------------------------
    let receipt = serde_json::json!({
        "stage": "run2-pertoken-dump",
        "design": "run-2 M2 per-token paired capture over saved S2c token streams (teacher-forced prefill only, no generation; data-shape scout spec 2026-08-28; S2c conventions: docs/adr/023, docs/research/024 sections 5.2/5.4)",
        "env": common::env_info(&nvcc),
        "smoke_non_evidence": smoke.is_some(),
        "streams": {"file": STREAMS_FILE, "sha256_measured": streams_sha,
                     "sha256_pinned": STREAMS_SHA256, "n_rows": n_items},
        "dataset": manifest["dataset"].clone(),
        "spans": "sender-GENERATED tokens only (prompt excluded) — matches S2c pooling convention and the live transmit distribution",
        "run_dir": dir.display().to_string(),
        "files": files.iter().map(|f| (f.name.clone(), serde_json::json!({
            "path": f.path.display().to_string(), "dim": f.dim, "tokens": f.tokens,
            "bytes": f.bytes, "sha256": f.sha256,
        }))).collect::<serde_json::Map<_, _>>(),
        "index": {"file": "run2-pertoken-index.json", "sha256": index_sha},
        "seeds": {"verification_draws_chacha8": CHECK_SEED,
                   "note": "capture itself is deterministic (teacher-forced prefill, greedy streams fixed); the seed only drives the post-write verification item draws"},
        "gates": {
            "streams_sha256": {"measured": streams_sha, "expect": STREAMS_SHA256, "pass": true},
            "token_streams_identical_across_models": {"pass": true,
                "note": "per item, BOTH models' prompt re-encoding asserted equal to the stored stream; generated span shared as raw token ids (same tokenizer)"},
            "per_item_token_count": {"pass": true,
                "note": "per item and layer, captured rows.len() asserted == gen_len x dim; T_i == gen_len from the streams file by construction of the capture span"},
            "sender_multi_tap_logits_parity_bit_identical": {"pass": sender_parity},
            "receiver_multi_tap_logits_parity_bit_identical": {"pass": receiver_parity},
            "total_generated_tokens": {"measured": total_tokens,
                "expect": (smoke.is_none()).then_some(EXPECT_TOTAL_GEN_TOKENS),
                "pass": smoke.is_some() || total_tokens == EXPECT_TOTAL_GEN_TOKENS},
            "file_bytes_exact": {"per_file": size_gate, "pass": sizes_pass},
            "index_round_trip": {"items_checked": rt_items, "pass": true},
            "pooled_mean_recompute": {"items_checked": pooled_items,
                "tolerance": "abs(mean_f64 - pooled_f32) <= 1e-2 + 1e-3 * abs(pooled)",
                "max_abs_diff": max_abs_diff, "max_rel_diff": max_rel_diff,
                "s2c_bins_sha_verified_against_committed_manifest": true, "pass": true},
        },
        "wall_clock_s": {"sender_phase": sender_phase_s, "receiver_phase": receiver_phase_s,
                          "verify_phase": verify_s, "total": t0.elapsed().as_secs_f64()},
    });
    let receipt_dir = if smoke.is_some() {
        dir.clone()
    } else {
        receipts_dir()
    };
    common::write_receipt(&receipt_dir, "run2-pertoken-dump-receipt.json", &receipt)?;
    println!(
        "run2 per-token dump complete: {n_items} items, {total_tokens} tokens x 4 layer files, {:.0}s total",
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}
