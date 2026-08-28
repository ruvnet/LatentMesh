//! S2c — GENERATED-pairs calibration dump (ADR-023 Deviation 7 contingency;
//! design doc 024 §5.4 / §8 risk 6, runtime half).
//!
//! The S2b bridge probe failed at both registered cells on the gold-teacher-
//! forced transform (ADR-023 stage table). Design §5.4 pre-names the
//! gold-vs-generated calibration distribution shift as the prime suspect and
//! pre-costs this contingency: "spend one sender-generation pass per
//! calibration item to regenerate pairs". This example executes exactly that
//! recipe: per calibration item the SENDER (Qwen2.5-3B) solves the problem
//! itself (greedy), and the resulting generated text — not the gold solution
//! — becomes the paired teacher-forced text for BOTH models, pooled over the
//! generated span with the S2 conventions unchanged.
//!
//! PRE-COMMITTED CHOICES (fixed in this source BEFORE any run; the receipt
//! echoes them — every choice design §5.4 leaves open is resolved here, not
//! after looking at outcomes):
//!
//! 1. **Pairing semantics — one sender-generation pass, same text through
//!    both models (§5.4 verbatim + §5.2's pairing definition).** §5.4 says
//!    "one sender-generation pass per calibration item"; §5.2 defines
//!    pairing as "teacher-force the SAME text through both models". So:
//!    sender generates; sender states are pooled over its own generated span
//!    (byte-for-byte the live S2b capture recipe — calibration X now matches
//!    the live transmit distribution, which is the entire point); receiver
//!    teacher-forces the SAME token stream and pools over the same span.
//!    REJECTED alternative ("each model generates its own solution"): that
//!    would pair states over DIFFERENT texts, breaking §5.2's same-text
//!    pairing and making the regression target incoherent; the design never
//!    asks for a receiver-generation pass.
//! 2. **Generation cap 400, greedy, batch=1** — §5.4 is silent on the cap;
//!    400 is the S1a/S2b live capture budget (`s2b_bridge_probe`
//!    MAX_NEW_TOKENS), i.e. the exact distribution the transform must serve.
//! 3. **Cells: the two ADR-023-registered cells only** — sender {L18, L24},
//!    receiver {L14, L19}, fitted as L18→L14 (S2 winner) and L24→L19
//!    (Deviation 6 anchor). No 3×3 sweep re-opening.
//! 4. **Item subset: prefix of the committed calibration-4000 index list.**
//!    The committed list is stored SORTED (ascending train-file index), so a
//!    prefix is the lowest-file-position portion of the seeded 4000-item
//!    subset — GSM8K train-file order carries no known difficulty/topic
//!    ordering, and the 80/20 fit/holdout split over collected rows remains
//!    the seeded shuffle (harness FIT_SPLIT_SEED). Stated truthfully: this
//!    is NOT a fresh random subsample of the 4000.
//! 5. **Budget ladder (n chosen by formula, not by outcome).** The design's
//!    "~1 GPU-h" contingency estimate is already known to be ~4× optimistic
//!    for its own recipe (S2b measured ~3.5 s per 3B greedy solve; 4000
//!    solves ≈ 4 GPU-h). The binding scientific constraint is fit
//!    identifiability: n_fit ≥ 2048 (= d_sender, the design §1 n>d gate) —
//!    and below n_fit = 1536 (= d_receiver) the Procrustes polar factor
//!    itself is non-unique, so a probe null would be unattributable (fit
//!    artifact vs dead channel), the exact confound S1a's kill-switch
//!    language exists to prevent. Pre-committed ladder, evaluated once at
//!    MEASURE_ROWS successful rows with the measured rate r (s/row):
//!    T(n) = elapsed_load + n·r·RECEIVER_ALLOWANCE. Quota = 2560 rows
//!    (n_fit = 2048) if T(2560) ≤ CEILING (3.5 h); else 1920 rows (n_fit =
//!    1536, polar-uniqueness floor, MᵀM full-rank equivalence per ADR-002
//!    amendment); else max n with T(n) ≤ CEILING (loud sub-uniqueness
//!    disclosure).
//!    "Successful rows": items whose sender generation is non-empty; skipped
//!    items are recorded and replaced by continuing down the list (quota
//!    counts successes, so one skip cannot starve the n>d gate).
//! 6. **Skip policy**: empty sender generation ⇒ item skipped on BOTH sides
//!    (rows must pair); skip indices recorded in the receipt. S2's
//!    abort-on-any-failure policy existed to protect an exact 4000-row
//!    count; here the quota mechanism provides that protection.
//! 7. **Prompt** = the live S2b sender capture prompt verbatim (same SYSTEM,
//!    same ANSWER_FORMAT). Sender first-pass accuracy recorded as a health
//!    diagnostic (expected ~70-80% for 3B on GSM8K), not a gate.
//!
//! Incremental persistence: pooled rows are appended to the per-layer
//! .f32bin files and token streams to a JSONL as each item completes, so a
//! crash mid-run preserves the GPU work already spent.
//!
//! Run: PATH=/usr/local/cuda-12.8/bin:$PATH \
//!      cargo run --release --features cuda --example s2c_generated_dump
//! Smoke (non-evidence, separate dir): append `-- --smoke 2`.

#[path = "common/mod.rs"]
#[allow(dead_code)]
mod common;

use latentmesh_runtime::{
    capture::{forward_capture_multi, forward_unpatched, logits_bit_identical},
    sampler::{Sampler, Sampling},
    QwenRuntime,
};
use std::io::Write as _;
use std::path::{Path, PathBuf};

const SENDER: &str = "Qwen/Qwen2.5-3B-Instruct";
const RECEIVER: &str = "Qwen/Qwen2.5-1.5B-Instruct";
const SYSTEM: &str = "You are a careful math tutor.";
const SENDER_LAYERS: [usize; 2] = [18, 24];
const RECEIVER_LAYERS: [usize; 2] = [14, 19];
const MAX_NEW_TOKENS: usize = 400;
const SPLIT_NAME: &str = "calibration-4000";
/// Budget-ladder constants (pre-committed choice 5).
const MEASURE_ROWS: usize = 25;
const CEILING_S: f64 = 3.5 * 3600.0;
const RECEIVER_ALLOWANCE: f64 = 1.03;
const QUOTA_N_OVER_D: usize = 2560; // 80% = 2048 = d_sender
const QUOTA_POLAR_FLOOR: usize = 1920; // 80% = 1536 = d_receiver

fn harness_data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../harness/latentmesh-live/data")
}

#[derive(serde::Deserialize)]
struct SplitFile {
    split: String,
    source: String,
    source_sha256: String,
    seed: u64,
    indices: Vec<usize>,
}

/// One collected item: the sender-generated token stream both models pool over.
#[derive(serde::Serialize, serde::Deserialize)]
struct StreamRow {
    row: usize,
    item: usize,
    prompt_tokens: Vec<u32>,
    gen_tokens: Vec<u32>,
    sender_first_pass_correct: bool,
}

/// Incremental f32 bin appender (crash-safe persistence).
struct BinWriter {
    file: std::io::BufWriter<std::fs::File>,
    path: PathBuf,
    rows: usize,
    dim: usize,
}

impl BinWriter {
    fn create(dir: &Path, name: &str, dim: usize) -> anyhow::Result<Self> {
        let path = dir.join(name);
        Ok(Self {
            file: std::io::BufWriter::new(std::fs::File::create(&path)?),
            path,
            rows: 0,
            dim,
        })
    }
    /// Resume: verify the file holds at least `rows` complete rows, truncate
    /// to exactly that (bin rows are written BEFORE the jsonl line, so the
    /// bin can hold at most one extra complete/partial row), open append.
    fn open_resume(dir: &Path, name: &str, dim: usize, rows: usize) -> anyhow::Result<Self> {
        let path = dir.join(name);
        let want = (rows * dim * 4) as u64;
        let have = std::fs::metadata(&path)?.len();
        anyhow::ensure!(
            have >= want,
            "{}: {have} bytes < {want} needed for {rows} persisted rows",
            path.display()
        );
        let file = std::fs::OpenOptions::new().write(true).open(&path)?;
        file.set_len(want)?;
        drop(file);
        let file = std::fs::OpenOptions::new().append(true).open(&path)?;
        Ok(Self {
            file: std::io::BufWriter::new(file),
            path,
            rows,
            dim,
        })
    }
    fn append(&mut self, row: &[f32]) -> anyhow::Result<()> {
        anyhow::ensure!(
            row.len() == self.dim,
            "row dim {} != {}",
            row.len(),
            self.dim
        );
        for v in row {
            self.file.write_all(&v.to_le_bytes())?;
        }
        self.rows += 1;
        self.file.flush()?;
        Ok(())
    }
    fn finish(self) -> anyhow::Result<(String, String, usize)> {
        drop(self.file);
        let sha = common::sha256_hex(&std::fs::read(&self.path)?);
        let name = self.path.file_name().unwrap().to_string_lossy().to_string();
        Ok((name, sha, self.rows))
    }
}

/// Realized sender rate (s/row) of the killed first attempt, read from its
/// log (`ladder: rate 3.51 s/row`, quota 2560): used only to charge the
/// resumed process for the GPU time the first attempt already spent, so the
/// pre-committed 3.5 h ceiling covers the SUM of attempts, not each one.
const RESUME_PRIOR_RATE_S: f64 = 3.51;

fn parse_flags() -> (Option<usize>, bool) {
    let args: Vec<String> = std::env::args().collect();
    let smoke = args
        .iter()
        .position(|a| a == "--smoke")
        .map(|i| args[i + 1].parse().expect("--smoke N"));
    let resume = args.iter().any(|a| a == "--resume");
    (smoke, resume)
}

fn main() -> anyhow::Result<()> {
    let t0 = std::time::Instant::now();
    let e = anyhow::Error::msg;
    let (smoke, resume) = parse_flags();
    anyhow::ensure!(
        !(smoke.is_some() && resume),
        "--smoke and --resume are mutually exclusive"
    );
    let nvcc = latentmesh_runtime::assert_cuda_build_env().map_err(e)?;
    println!("build-env guard: {nvcc}  smoke={smoke:?} resume={resume}");

    // Committed calibration index list + its source file, sha-verified
    // (identical loading discipline to s2_calibrate_dump).
    let split_path = harness_data_dir().join(format!("{SPLIT_NAME}.json"));
    let split_bytes = std::fs::read(&split_path)?;
    let split_sha = common::sha256_hex(&split_bytes);
    let split: SplitFile = serde_json::from_slice(&split_bytes)?;
    anyhow::ensure!(
        split.split == SPLIT_NAME,
        "split mislabeled: {}",
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
        "{}: {} items; calibration list {} indices (STORED SORTED; prefix rule, choice 4)",
        split.source,
        all_items.len(),
        split.indices.len()
    );

    let dir = common::run_dir(if smoke.is_some() { "s2c-smoke" } else { "s2c" });

    // Pre-plan receipt: written BEFORE any model loads (choices frozen
    // pre-run; the module doc above is the authoritative text).
    let preplan = serde_json::json!({
        "stage": "S2c-preplan",
        "written_before_model_load": true,
        "pairing": "one sender-generation pass per item (design 024 section 5.4 verbatim); SAME generated text teacher-forced through both models (section 5.2 pairing); sender pooled over its own generated span (the live S2b capture recipe), receiver pooled over the same span. REJECTED: each-model-generates-its-own (pairs states over different texts).",
        "decoding": "greedy, batch=1, max_new_tokens=400 (the S1a/S2b live capture budget; section 5.4 silent)",
        "cells": {"sender_layers": SENDER_LAYERS, "receiver_layers": RECEIVER_LAYERS,
                   "registered_pairs_only": ["L18->L14 (S2 winner)", "L24->L19 (Deviation 6 anchor)"]},
        "item_subset": "prefix of the committed calibration-4000 index list (stored sorted ascending; NOT a fresh random subsample; train-file order carries no known difficulty ordering)",
        "budget_ladder": {"measure_rows": MEASURE_ROWS, "ceiling_s": CEILING_S,
            "receiver_allowance_factor": RECEIVER_ALLOWANCE,
            "quota_rule": "2560 (n_fit=2048=d_sender) if projected <= ceiling; else 1920 (n_fit=1536=d_receiver polar-uniqueness floor, MtM full-rank equivalence per ADR-002 amendment); else max n within ceiling (loud sub-uniqueness disclosure)",
            "design_estimate_note": "design's ~1 GPU-h contingency estimate is ~4x optimistic for its own recipe (S2b measured ~3.5 s per 3B greedy solve x 4000 items); identifiability (n>d) outranks the cost estimate — a probe against an underdetermined transform would make a null unattributable, the exact confound the S1a kill-switch language forbids"},
        "skip_policy": "empty sender generation => item skipped on both sides, recorded; quota counts successes",
        "smoke": smoke,
        "resume": resume,
    });
    common::write_receipt(&dir, "s2c-preplan.json", &preplan)?;

    // Resume state: rows already persisted by a killed earlier attempt
    // (incremental persistence is exactly for this). The quota decided by
    // that attempt's ladder (2560; rate 3.51 s/row, T(2560)=9261s <= 12600s
    // ceiling, from its log) carries forward unchanged — the ladder is a
    // pre-committed formula, not a per-attempt re-negotiation.
    let streams_path = dir.join("token-streams.jsonl");
    let prior: Vec<StreamRow> = if resume {
        std::fs::read_to_string(&streams_path)?
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).map_err(anyhow::Error::from))
            .collect::<anyhow::Result<_>>()?
    } else {
        Vec::new()
    };
    let prior_n = prior.len();
    if resume {
        println!("resume: {prior_n} rows already persisted; quota carried forward");
    }

    let device = latentmesh_runtime::device().map_err(e)?;

    // ---- Phase 1: sender generates + captures L18/L24 over generated span --
    println!("phase 1: loading {SENDER}...");
    let p1 = std::time::Instant::now();
    let mut sender = QwenRuntime::load(SENDER, &device, candle_core::DType::BF16).map_err(e)?;
    anyhow::ensure!(sender.config.hidden_size == 2048);
    let load_s = p1.elapsed().as_secs_f64();
    println!("sender loaded in {load_s:.1}s");

    let mut s_bins: Vec<BinWriter> = SENDER_LAYERS
        .iter()
        .map(|l| {
            if resume {
                BinWriter::open_resume(&dir, &format!("sender_L{l}.f32bin"), 2048, prior_n)
            } else {
                BinWriter::create(&dir, &format!("sender_L{l}.f32bin"), 2048)
            }
        })
        .collect::<anyhow::Result<_>>()?;
    let mut streams_file = std::io::BufWriter::new(if resume {
        std::fs::OpenOptions::new()
            .append(true)
            .open(&streams_path)?
    } else {
        std::fs::File::create(&streams_path)?
    });

    // Smoke runs fix the quota up front; resumed runs carry forward the
    // killed attempt's ladder decision (see resume block above).
    let mut quota: Option<usize> = if resume { Some(QUOTA_N_OVER_D) } else { smoke };
    let mut ladder_report = if resume {
        serde_json::json!({
            "carried_forward_from_killed_attempt": true,
            "rate_s_per_row": RESUME_PRIOR_RATE_S,
            "projected_s_2560": 9261.0,
            "ceiling_s": CEILING_S,
            "chosen_quota": QUOTA_N_OVER_D,
            "n_fit_at_quota_80pct": QUOTA_N_OVER_D * 8 / 10,
            "note": "ladder decision realized by the first attempt before it was killed at ~1053 rows (its log: 'ladder: rate 3.51 s/row; T(2560)=9261s T(1920)=6946s ceiling=12600s => quota 2560'); formula pre-committed, not re-evaluated per attempt",
        })
    } else {
        serde_json::json!(null)
    };
    let mut collected: Vec<StreamRow> = prior;
    let already: std::collections::HashSet<usize> = collected.iter().map(|r| r.item).collect();
    let mut skipped: Vec<usize> = Vec::new();
    let mut sender_parity = false;
    let mut first_pass_correct = collected
        .iter()
        .filter(|r| r.sender_first_pass_correct)
        .count();
    // Charge the resumed process for the killed attempt's GPU time so the
    // pre-committed ceiling bounds the SUM across attempts.
    let ceiling_remaining = CEILING_S - prior_n as f64 * RESUME_PRIOR_RATE_S;
    let gen_start = std::time::Instant::now();

    for &idx in &split.indices {
        if let Some(q) = quota {
            if collected.len() >= q {
                break;
            }
        }
        if already.contains(&idx) {
            continue;
        }
        if gen_start.elapsed().as_secs_f64() > ceiling_remaining {
            println!(
                "budget ceiling reached at {} rows — stopping collection",
                collected.len()
            );
            break;
        }
        let item = &all_items[idx];
        let prompt = QwenRuntime::chat_prompt(
            SYSTEM,
            &format!("{}\n\n{}", item.question, common::ANSWER_FORMAT),
        );
        let p_toks = sender.encode(&prompt).map_err(e)?;
        let mut greedy = Sampler::new(Sampling::Greedy, 0);
        let gen = sender
            .generate(&p_toks, None, &mut greedy, MAX_NEW_TOKENS, false)
            .map_err(e)?;
        if gen.tokens.is_empty() {
            println!("item {idx}: empty generation, skipped (choice 6)");
            skipped.push(idx);
            continue;
        }
        let full: Vec<u32> = p_toks.iter().chain(gen.tokens.iter()).copied().collect();
        let span = p_toks.len()..full.len();
        let (logits, caps) =
            forward_capture_multi(&mut sender.model, &full, &SENDER_LAYERS, span, &device)
                .map_err(|err| anyhow::anyhow!("item {idx}: {err}"))?;
        if collected.len() == prior_n {
            // First row THIS process captures (row 0 on a fresh run): the
            // multi-tap parity gate is re-measured per process.
            let ref_logits = forward_unpatched(&mut sender.model, &full, &device).map_err(e)?;
            sender_parity = logits_bit_identical(&logits, &ref_logits).map_err(e)?;
            anyhow::ensure!(sender_parity, "sender multi-tap logits parity FAILED");
        }
        for (b, cap) in s_bins.iter_mut().zip(&caps) {
            b.append(&cap.pooled)?;
        }
        let correct = common::extract_answer(&gen.text)
            .is_some_and(|a| common::answers_equal(&a, &item.gold));
        first_pass_correct += usize::from(correct);
        let row = StreamRow {
            row: collected.len(),
            item: idx,
            prompt_tokens: p_toks,
            gen_tokens: gen.tokens,
            sender_first_pass_correct: correct,
        };
        serde_json::to_writer(&mut streams_file, &row)?;
        streams_file.write_all(b"\n")?;
        streams_file.flush()?;
        collected.push(row);

        // Budget-ladder evaluation, exactly once (choice 5).
        if collected.len() == MEASURE_ROWS && quota.is_none() {
            let elapsed = gen_start.elapsed().as_secs_f64();
            let rate = elapsed / MEASURE_ROWS as f64;
            let project = |n: usize| load_s + n as f64 * rate * RECEIVER_ALLOWANCE;
            let q = if project(QUOTA_N_OVER_D) <= CEILING_S {
                QUOTA_N_OVER_D
            } else if project(QUOTA_POLAR_FLOOR) <= CEILING_S {
                QUOTA_POLAR_FLOOR
            } else {
                ((CEILING_S - load_s) / (rate * RECEIVER_ALLOWANCE)) as usize
            };
            ladder_report = serde_json::json!({
                "measured_at_rows": MEASURE_ROWS,
                "elapsed_s": elapsed,
                "rate_s_per_row": rate,
                "projected_s_2560": project(QUOTA_N_OVER_D),
                "projected_s_1920": project(QUOTA_POLAR_FLOOR),
                "ceiling_s": CEILING_S,
                "chosen_quota": q,
                "n_fit_at_quota_80pct": q * 8 / 10,
            });
            println!(
                "ladder: rate {rate:.2} s/row; T(2560)={:.0}s T(1920)={:.0}s ceiling={CEILING_S:.0}s => quota {q}",
                project(QUOTA_N_OVER_D),
                project(QUOTA_POLAR_FLOOR)
            );
            quota = Some(q);
        }
        if collected.len() % 100 == 0 {
            println!(
                "  [{} rows, {} skipped] {:.0}s elapsed",
                collected.len(),
                skipped.len(),
                t0.elapsed().as_secs_f32()
            );
        }
    }
    drop(streams_file);
    let sender_phase_s = p1.elapsed().as_secs_f64();
    let n_rows = collected.len();
    anyhow::ensure!(n_rows >= 2, "fewer than 2 rows collected");
    println!(
        "phase 1 done: {n_rows} rows, {} skipped, sender first-pass accuracy {first_pass_correct}/{n_rows}, {sender_phase_s:.0}s",
        skipped.len()
    );
    drop(sender);
    let sender_files: Vec<(String, String, usize)> = s_bins
        .into_iter()
        .map(BinWriter::finish)
        .collect::<anyhow::Result<_>>()?;

    // ---- Phase 2: receiver teacher-forces the SAME streams, L14/L19 -------
    println!("phase 2: loading {RECEIVER}...");
    let p2 = std::time::Instant::now();
    let mut receiver = QwenRuntime::load(RECEIVER, &device, candle_core::DType::BF16).map_err(e)?;
    anyhow::ensure!(receiver.config.hidden_size == 1536);
    let mut r_bins: Vec<BinWriter> = RECEIVER_LAYERS
        .iter()
        .map(|l| BinWriter::create(&dir, &format!("receiver_L{l}.f32bin"), 1536))
        .collect::<anyhow::Result<_>>()?;
    let mut receiver_parity = false;
    for (i, row) in collected.iter().enumerate() {
        // Same-tokenizer pairing assertion (as S2): the receiver's encoding
        // of the prompt must reproduce the sender's token ids; the generated
        // continuation is shared as raw ids.
        let item = &all_items[row.item];
        let prompt = QwenRuntime::chat_prompt(
            SYSTEM,
            &format!("{}\n\n{}", item.question, common::ANSWER_FORMAT),
        );
        let rp = receiver.encode(&prompt).map_err(e)?;
        anyhow::ensure!(
            rp == row.prompt_tokens,
            "item {}: prompt token stream differs between models — same-tokenizer pairing broken",
            row.item
        );
        let full: Vec<u32> = row
            .prompt_tokens
            .iter()
            .chain(row.gen_tokens.iter())
            .copied()
            .collect();
        let span = row.prompt_tokens.len()..full.len();
        let (logits, caps) =
            forward_capture_multi(&mut receiver.model, &full, &RECEIVER_LAYERS, span, &device)
                .map_err(|err| anyhow::anyhow!("item {}: {err}", row.item))?;
        if i == 0 {
            let ref_logits = forward_unpatched(&mut receiver.model, &full, &device).map_err(e)?;
            receiver_parity = logits_bit_identical(&logits, &ref_logits).map_err(e)?;
            anyhow::ensure!(receiver_parity, "receiver multi-tap logits parity FAILED");
        }
        for (b, cap) in r_bins.iter_mut().zip(&caps) {
            b.append(&cap.pooled)?;
        }
        if (i + 1) % 500 == 0 {
            println!("  [{}/{n_rows}] {:.0}s", i + 1, p2.elapsed().as_secs_f32());
        }
    }
    let receiver_phase_s = p2.elapsed().as_secs_f64();
    let receiver_files: Vec<(String, String, usize)> = r_bins
        .into_iter()
        .map(BinWriter::finish)
        .collect::<anyhow::Result<_>>()?;
    println!("phase 2 done in {receiver_phase_s:.0}s");

    // ---- Manifest + receipt ----------------------------------------------
    let side_json = |model: &str,
                     n_layers: usize,
                     hidden: usize,
                     layers: &[usize],
                     files: &[(String, String, usize)]| {
        serde_json::json!({
            "model": model, "n_layers": n_layers, "hidden_size": hidden, "layers": layers,
            "files": layers.iter().zip(files).map(|(l, (f, s, _))| {
                (l.to_string(), serde_json::json!({"file": f, "sha256": s}))
            }).collect::<serde_json::Map<_, _>>(),
        })
    };
    let gen_tokens_total: usize = collected.iter().map(|r| r.gen_tokens.len()).sum();
    let manifest = serde_json::json!({
        "n_rows": n_rows,
        "item_indices": collected.iter().map(|r| r.item).collect::<Vec<_>>(),
        "sender": side_json(SENDER, 36, 2048, &SENDER_LAYERS, &sender_files),
        "receiver": side_json(RECEIVER, 28, 1536, &RECEIVER_LAYERS, &receiver_files),
        "format": "raw little-endian f32, row-major n_rows x hidden_size; row i pairs with item_indices[i]",
        "pooling": "mean over the sender-GENERATED token span (prompt tokens excluded) of the residual after block L — same text through both models",
        "pairing": preplan["pairing"],
        "decoding": preplan["decoding"],
        "dataset": {"source": split.source, "source_sha256": source_sha,
                     "split_file": format!("{SPLIT_NAME}.json"), "split_file_sha256": split_sha,
                     "split_seed_chacha8": split.seed,
                     "subset_rule": preplan["item_subset"]},
    });
    let manifest_path = common::write_receipt(&dir, "manifest.json", &manifest)?;
    let manifest_sha = common::sha256_hex(&std::fs::read(&manifest_path)?);

    let receipt = serde_json::json!({
        "stage": "S2c-generated-dump",
        "design": "docs/adr/023 Deviation 7; docs/research/024 sections 5.4 (contingency recipe), 5.2 (pairing), 8 risk 6",
        "env": common::env_info(&nvcc),
        "smoke_non_evidence": smoke.is_some(),
        "resumed": resume.then(|| serde_json::json!({
            "prior_rows_from_killed_attempt": prior_n,
            "reason": "first attempt's background task was killed by the session harness at ~1053 rows; incremental persistence preserved every completed row (bins + token-streams.jsonl verified consistent: 1053 rows, 0 bytes partial)",
            "prior_gpu_s_charged": prior_n as f64 * RESUME_PRIOR_RATE_S,
            "wall_clock_note": "wall_clock_s below covers THIS process only; total GPU spend = prior_gpu_s_charged + this process's phases",
        })),
        "preplan": preplan,
        "manifest_sha256": manifest_sha,
        "dataset": manifest["dataset"].clone(),
        "budget_ladder_realized": ladder_report,
        "gates": {
            "sender_capture_dim": {"measured": 2048, "expect": 2048, "pass": true},
            "receiver_capture_dim": {"measured": 1536, "expect": 1536, "pass": true},
            "sender_multi_tap_logits_parity_bit_identical": {"pass": sender_parity},
            "receiver_multi_tap_logits_parity_bit_identical": {"pass": receiver_parity},
            "token_streams_identical_across_models": {"pass": true,
                "note": "receiver prompt encoding asserted equal to sender's per item; generated span shared as raw token ids (same tokenizer)"},
            "n_over_d": {"n_rows": n_rows, "n_fit_80pct": n_rows * 8 / 10,
                "d_sender": 2048, "pass": n_rows * 8 / 10 >= 2048,
                "polar_uniqueness_floor_d_receiver": 1536,
                "polar_unique": n_rows * 8 / 10 >= 1536},
        },
        "rows": {"collected": n_rows, "skipped_items": skipped,
                  "sender_first_pass_correct": first_pass_correct,
                  "generated_tokens_total": gen_tokens_total,
                  "generated_tokens_mean": gen_tokens_total as f64 / n_rows as f64},
        "wall_clock_s": {"sender_phase": sender_phase_s, "receiver_phase": receiver_phase_s,
                          "total": t0.elapsed().as_secs_f64()},
    });
    common::write_receipt(&dir, "s2c-generated-dump-receipt.json", &receipt)?;
    println!(
        "S2c dump complete: {n_rows} rows x (2 sender + 2 receiver layers), {:.0}s total",
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}
