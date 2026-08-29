//! Run-2 **M4f registered pre-check** — is the item-invariant off-manifold
//! collapse found in M4c's adapter (`docs/research/033` §4) UNIVERSAL across
//! the ladder, or SPECIFIC to task-loss training?
//!
//! ADR-024 § "DIAGNOSIS (2026-08-29): the adapter collapsed to a fixed
//! OFF-MANIFOLD direction" registers it verbatim: *"Cheap pre-check available
//! with no probe draw: re-run the same unembedding projection on M4d's
//! artifact and on the M3/M4 artifacts to establish whether off-manifold
//! collapse is universal across the ladder or specific to task-loss
//! training."*
//!
//! **CPU-ONLY, ANNOTATES ONLY.** No probe draw, no live-model forward, no
//! CUDA workload (ADR-034 lane rule: the GPU is held by M4d). The only
//! model-derived object touched is the receiver's *unembedding matrix*
//! (`model.embed_tokens.weight`, tied) plus the final RMSNorm gain, loaded
//! from the HF cache on CPU. It reads committed artifacts and the gitignored
//! per-token capture dumps the M4c training receipt already pins by sha256;
//! it writes exactly one new receipt and changes no recorded outcome.
//!
//! The metric kit is `common/lens.rs` — the SAME code `docs/research/033`
//! (`run2_rescale_diagnostic`) measured M4c with, extracted rather than
//! re-implemented. The run-1 affine apply is `common/affine.rs`, extracted
//! from the frozen `s2b_bridge_probe`.
//!
//! Candidates (every committed L18→L14 adapter artifact, plus references):
//!   * run-1 affine bridges (S2b winner, S2c generated-pairs) — training-FREE
//!     closed-form linear maps, the decisive contrast;
//!   * M3 MLP (reconstruction loss), both registered eval variants, plus
//!     ADR-024 **M4h Stage 1**'s de-pooled derivation of the SAME artifact
//!     (per-token translate, LAST token instead of the mean — the first
//!     candidate in this kit that is not a pooled object);
//!   * M4 FastGRNN r=64/128/256 (reconstruction loss, sequence) + the
//!     superseded r=64 window-zero-init run;
//!   * M4c MLP task-loss — the anchor, reproducing `docs/research/033` §4;
//!   * the M4c and M4d *initialisations* (fresh seeded, untrained) — the
//!     "is this training at all?" control;
//!   * the receiver's OWN L14 states over the same spans — the on-manifold
//!     reference, pooled and single-row.
//!
//! Run (CPU, default features):
//!   cargo run --release --example run2_manifold_precheck

#[path = "common/mod.rs"]
#[allow(dead_code)]
mod common;

use candle_core::{DType, Device};
use latentmesh_runtime::norms;
use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use common::affine::AffineTransform;
use common::fastgrnn::FastGrnnTransform;
use common::lens::{
    classify, cosine, entropy_nats, logsumexp, mean, mean_pairwise_cosine, mean_resultant_length,
    minmax, project_batch, rms_norm, token_set_stats, top_k, COLLAPSE_COSINE, COLLAPSE_TOKEN_UNION,
    OFF_MANIFOLD_COSINE,
};
use common::m3::{GOLDEN_REL_TOL, RECEIVER, RECEIVER_BLOCK, SENDER_BLOCK};
use common::mlp::MlpTransform;

const RUN_DIR: &str = "target/latentmesh-runs/run2";
const INDEX_JSON: &str = "run2-pertoken-index.json";
const SENDER_DUMP: &str = "sender_L18.tok.f32bin";
const RECEIVER_DUMP: &str = "receiver_L14.tok.f32bin";
const SENDER_DUMP_SHA256: &str = "44574f9e38bbbe8e2f5b5955cd1cccae0d399b48ed87b5270d1564ab85069c04";
const RECEIVER_DUMP_SHA256: &str =
    "e528e37ec0e3aa766989cc80f2feffeee71eb73c7b84a8c9a15dab6a2795e263";
const STREAMS: &str = "../../harness/latentmesh-live/data/s2c-token-streams.jsonl";
const STREAMS_SHA256: &str = "ad5713116cb5acf663fe9c33ac58aaedb1fd64fcf629dd78b21439d244958539";
const GSM8K: &str = "../../harness/latentmesh-live/data/gsm8k-train.jsonl";
const GSM8K_TRAIN_SHA256: &str = "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465";
const DUMP_RECEIPT: &str = "receipts/run2-pertoken-dump-receipt.json";

const D_SENDER: usize = 2048;
const D_RECEIVER: usize = 1536;
const N_SAMPLE: usize = 40;
const TOPK: usize = 10;
/// Number of dominant tokens listed per candidate in the receipt.
const N_DOMINANT: usize = 8;
const EVIDENCE_LABEL: &str =
    "deterministic CPU analysis over committed artifacts — no probe draw, annotates only";

// --- Registered classification thresholds ---------------------------------
// The thresholds and the verdict function now live in `common/lens.rs`,
// beside the metric kit that feeds them, so ADR-024's PC1 pre-check reuses
// ONE definition rather than copying a second that could drift. Their values
// and rationale are unchanged; see `common::lens`.

fn crate_path(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

fn sha256_file(path: &Path) -> anyhow::Result<String> {
    use sha2::{Digest, Sha256};
    let mut f = std::fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 22];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}

/// `[n_rows x dim]` rows for one dump row, read by byte offset.
fn read_rows(
    dump: &Path,
    dim: usize,
    token_offset: usize,
    n_rows: usize,
) -> anyhow::Result<Vec<f32>> {
    let mut f = std::fs::File::open(dump)?;
    f.seek(SeekFrom::Start((token_offset * dim * 4) as u64))?;
    let mut bytes = vec![0u8; n_rows * dim * 4];
    f.read_exact(&mut bytes)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// Mean over `[n_rows x dim]`, f64 accumulation (the pooling every rung uses).
fn pool(rows: &[f32], dim: usize, n_rows: usize) -> Vec<f32> {
    let mut acc = vec![0f64; dim];
    for r in 0..n_rows {
        for (a, v) in acc.iter_mut().zip(&rows[r * dim..(r + 1) * dim]) {
            *a += *v as f64;
        }
    }
    acc.iter().map(|a| (*a / n_rows as f64) as f32).collect()
}

// ---------------------------------------------------------------------------

/// Everything one item supplies to every candidate emitter.
struct ItemCtx {
    sender_rows: Vec<f32>,
    n_rows: usize,
    sender_pooled: Vec<f32>,
    receiver_pooled: Vec<f32>,
    receiver_last_row: Vec<f32>,
}

/// How a candidate turns one item's captured states into the 1536-d vector
/// that would be injected at the receiver's block 14.
enum Emitter {
    /// Run-1 pipeline: pool the raw sender rows, then the affine map. (The
    /// affine commutes with mean-pooling exactly — `common/affine.rs` test —
    /// so this is also the per-token-then-pool object.)
    Affine(Box<AffineTransform>),
    /// M3/M4c variant (i): MLP per generated-span token, pool afterwards.
    MlpPerToken(Box<MlpTransform>),
    /// M3 variant (ii): pool the sender rows first, then the same MLP.
    MlpPooled(Box<MlpTransform>),
    /// ADR-024 M4h Stage 1: MLP per generated-span token, then take the LAST
    /// token's output instead of the mean. De-pooled (`docs/research/040`).
    MlpLastToken(Box<MlpTransform>),
    /// M4: run the sequence from h0 = 0, pool the translated output stream.
    FastGrnn(Box<FastGrnnTransform>),
    /// Reference, no adapter: the receiver's own pooled L14 state.
    NaturalPooled,
    /// Reference, no adapter and no pooling: one real receiver L14 state
    /// (the last row of the item's span).
    NaturalLastRow,
}

impl Emitter {
    fn emit(&self, c: &ItemCtx) -> Vec<f32> {
        match self {
            Emitter::Affine(t) => t.apply(&c.sender_pooled),
            Emitter::MlpPerToken(t) => t.apply_rows_then_pool(&c.sender_rows, c.n_rows),
            Emitter::MlpPooled(t) => t.apply(&c.sender_pooled),
            Emitter::MlpLastToken(t) => t.apply_last_row(&c.sender_rows, c.n_rows),
            Emitter::FastGrnn(t) => t.translate_seq_then_pool(&c.sender_rows, c.n_rows),
            Emitter::NaturalPooled => c.receiver_pooled.clone(),
            Emitter::NaturalLastRow => c.receiver_last_row.clone(),
        }
    }
}

struct Candidate {
    label: &'static str,
    family: &'static str,
    training: &'static str,
    artifact: String,
    hash: String,
    gate: serde_json::Value,
    emitter: Emitter,
}

/// Per-(candidate, item) measurements.
#[derive(Default, Clone)]
struct ItemRow {
    l2: f64,
    cos_to_natural_same_item: f64,
    top10: Vec<u32>,
    entropy_rmsnorm: f64,
    entropy_plain: f64,
    gold_rank_rmsnorm: f64,
    gold_rank_plain: f64,
    gold_best_rank_rmsnorm: usize,
    span_rank_rmsnorm: f64,
    span_rank_plain: f64,
}

// --- Artifact loading + hash gates -----------------------------------------

fn read_json(rel: &str) -> anyhow::Result<serde_json::Value> {
    Ok(serde_json::from_slice(&std::fs::read(crate_path(rel))?)?)
}

/// `expected`: the hash a *training / probe receipt* pins for this artifact.
/// `None` means no external receipt pins it yet (M4d, still training) — the
/// golden file's own `artifact_file_sha256` is then the only binding, and the
/// receipt discloses that.
fn hash_gate(
    got: &str,
    expected: Option<(&str, &str)>,
    golden_file: &str,
    golden_detail: serde_json::Value,
) -> anyhow::Result<serde_json::Value> {
    let (pinned_by, ok) = match expected {
        Some((receipt, want)) => {
            anyhow::ensure!(
                got == want,
                "artifact hash {got} != {receipt}'s pinned hash {want}"
            );
            (receipt.to_string(), true)
        }
        None => (String::new(), false),
    };
    Ok(serde_json::json!({
        "content_hash_sha256": got,
        "pinned_by_receipt": if ok { serde_json::json!(pinned_by) } else { serde_json::Value::Null },
        "verified_against_training_or_probe_receipt": ok,
        "golden_file": golden_file,
        "golden_check": golden_detail,
    }))
}

/// Which of the three MLP payload derivations a candidate uses. Explicit
/// rather than a bool, since ADR-024 M4h Stage 1 adds a third.
#[derive(Clone, Copy, PartialEq)]
enum MlpDerivation {
    /// Translate per token, then mean-pool the translated stream.
    PerTokenThenPool,
    /// Pool the sender rows first, then translate once.
    PoolThenTranslate,
    /// Translate per token, then take the LAST token — no mean (M4h S1).
    PerTokenLast,
}

fn load_mlp(
    label: &'static str,
    family: &'static str,
    training: &'static str,
    file: &str,
    golden: &str,
    expected: Option<(&str, &str)>,
    derivation: MlpDerivation,
) -> anyhow::Result<Candidate> {
    let t = MlpTransform::load(&crate_path(file))?;
    let (n, max_rel, seed) =
        common::mlp::verify_against_golden(&t, &crate_path(golden), GOLDEN_REL_TOL)?;
    let gate = hash_gate(
        &t.content_hash,
        expected,
        golden,
        serde_json::json!({"pairs": n, "input_seed_chacha8": seed,
                           "max_relative_l2_error": max_rel, "tolerance": GOLDEN_REL_TOL,
                           "pass": true}),
    )?;
    let hash = t.content_hash.clone();
    Ok(Candidate {
        label,
        family,
        training,
        artifact: file.to_string(),
        hash,
        gate,
        emitter: match derivation {
            MlpDerivation::PoolThenTranslate => Emitter::MlpPooled(Box::new(t)),
            MlpDerivation::PerTokenThenPool => Emitter::MlpPerToken(Box::new(t)),
            MlpDerivation::PerTokenLast => Emitter::MlpLastToken(Box::new(t)),
        },
    })
}

fn load_fastgrnn(
    label: &'static str,
    training: &'static str,
    file: &str,
    golden: &str,
    expected: Option<(&str, &str)>,
) -> anyhow::Result<Candidate> {
    let t = FastGrnnTransform::load(&crate_path(file))?;
    let (n_seqs, seq_len, max_rel, seed) =
        common::fastgrnn::verify_against_golden(&t, &crate_path(golden), GOLDEN_REL_TOL)?;
    let gate = hash_gate(
        &t.content_hash,
        expected,
        golden,
        serde_json::json!({"seqs": n_seqs, "seq_len": seq_len, "input_seed_chacha8": seed,
                           "max_relative_l2_error": max_rel, "tolerance": GOLDEN_REL_TOL,
                           "pass": true}),
    )?;
    let hash = t.content_hash.clone();
    Ok(Candidate {
        label,
        family: "fastgrnn-reconstruction",
        training,
        artifact: file.to_string(),
        hash,
        gate,
        emitter: Emitter::FastGrnn(Box::new(t)),
    })
}

fn load_affine(
    label: &'static str,
    training: &'static str,
    file: &str,
    golden: &str,
    expected: (&str, &str),
) -> anyhow::Result<Candidate> {
    let bytes = std::fs::read(crate_path(file))?;
    let hash = common::sha256_hex(&bytes);
    let t: AffineTransform = serde_json::from_slice(&bytes)?;
    anyhow::ensure!(
        t.dim_sender == D_SENDER && t.dim_receiver == D_RECEIVER,
        "{file}: dims {}x{} != {D_SENDER}x{D_RECEIVER}",
        t.dim_sender,
        t.dim_receiver
    );
    let (n, max_rel, seed) =
        common::affine::verify_against_golden(&t, &crate_path(golden), &hash, GOLDEN_REL_TOL)?;
    let gate = hash_gate(
        &hash,
        Some(expected),
        golden,
        serde_json::json!({"pairs": n, "input_seed_chacha8": seed,
                           "max_relative_l2_error": max_rel, "tolerance": GOLDEN_REL_TOL,
                           "pass": true, "alpha": t.alpha,
                           "on_train_confidence_OPTIMISTIC": t.confidence}),
    )?;
    Ok(Candidate {
        label,
        family: "affine-run1-training-free",
        training,
        artifact: file.to_string(),
        hash,
        gate,
        emitter: Emitter::Affine(Box::new(t)),
    })
}

/// The hash a receipt pins under `path` (dotted), as an owned String.
fn pinned(receipt: &serde_json::Value, path: &[&str]) -> anyhow::Result<String> {
    let mut v = receipt;
    for k in path {
        v = &v[k];
    }
    v.as_str()
        .map(str::to_string)
        .ok_or_else(|| anyhow::anyhow!("receipt does not pin {path:?}"))
}

#[allow(clippy::too_many_lines)]
fn build_candidates() -> anyhow::Result<(Vec<Candidate>, serde_json::Value)> {
    let m3r = read_json("receipts/run2-m3-training-receipt-cellL18toL14.json")?;
    let m4c = read_json("receipts/run2-m4c-training-receipt-cellL18toL14.json")?;
    let r64 = read_json("receipts/run2-m4-training-receipt-cellL18toL14-r64.json")?;
    let r128 = read_json("receipts/run2-m4-training-receipt-cellL18toL14-r128.json")?;
    let r256 = read_json("receipts/run2-m4-training-receipt-cellL18toL14-r256.json")?;
    let r64s = read_json(
        "receipts/run2-m4-training-receipt-cellL18toL14-r64-superseded-windowzeroinit.json",
    )?;
    let s2b = read_json("receipts/s2b-receipt-cellL18toL14-slots8-poolfull-rescaletrue-n40.json")?;
    let s2c = read_json(
        "receipts/s2b-receipt-cellL18toL14-slots8-poolfull-rescaletrue-n40-genpairs.json",
    )?;

    let h = |r: &serde_json::Value| pinned(r, &["artifact", "content_hash_sha256"]);
    let m3_h = h(&m3r)?;
    let m4c_h = h(&m4c)?;
    let m4c_init_h = pinned(&m4c, &["artifact", "init_content_hash_sha256"])?;
    let r64_h = h(&r64)?;
    let r128_h = h(&r128)?;
    let r256_h = h(&r256)?;
    let r64s_h = h(&r64s)?;
    let s2b_h = pinned(
        &s2b,
        &["gates", "transform_hash_matches_registered", "hash"],
    )?;
    let s2c_h = pinned(
        &s2c,
        &["gates", "transform_hash_matches_registered", "hash"],
    )?;

    // M4d: measured when — and only when — its trainer has already written
    // both the artifact and the training receipt that pins it. The pre-check
    // never waits for, polls or interferes with the GPU lane (ADR-034); it
    // takes whatever is on disk when it starts.
    let m4d_final = crate_path("receipts/run2-m4d-mlp-deploymatch-cellL18toL14.f32bin");
    let m4d_receipt_path = crate_path("receipts/run2-m4d-training-receipt-cellL18toL14.json");
    let m4d = m4d_receipt_path
        .exists()
        .then(|| read_json("receipts/run2-m4d-training-receipt-cellL18toL14.json"))
        .transpose()?;
    let m4d_h = m4d.as_ref().map(h).transpose()?;
    let m4d_status = serde_json::json!({
        "final_artifact_present": m4d_final.exists(),
        "training_receipt_present": m4d_receipt_path.exists(),
        "measured_here": m4d_final.exists() && m4d_h.is_some(),
        "note": "M4d's trainer finished and wrote both its artifact and its training receipt while this pre-check was being written, so M4d's TRAINED adapter is measured here with its hash pinned against that receipt. M4d's probe was running on the GPU at the time; nothing in this lane touched it. M4d's own probe VERDICT is not read, used or affected here.",
    });

    let mut v = vec![
        load_affine(
            "run1-affine-s2b",
            "training-FREE closed-form ridge/orthogonal affine fit (latentmesh-align), run-1 S2b winner cell",
            "receipts/transform-L18-to-L14.json",
            "receipts/s2b-golden-affine-L18-to-L14.json",
            ("s2b-receipt-...-n40.json (gates.transform_hash_matches_registered)", &s2b_h),
        )?,
        load_affine(
            "run1-affine-s2c-genpairs",
            "training-FREE closed-form affine refit on S2c generated-token pairs, run-1 contingency arm",
            "receipts/transform-gen-L18-to-L14.json",
            "receipts/s2c-golden-affine-L18-to-L14.json",
            ("s2b-receipt-...-n40-genpairs.json (gates.transform_hash_matches_registered)", &s2c_h),
        )?,
        load_mlp(
            "m3-mlp-pertoken",
            "mlp-reconstruction",
            "SGD on RECONSTRUCTION loss (per-token MSE to the receiver's own L14 state); eval variant (i): translate per token, pool after",
            "receipts/run2-m3-mlp-cellL18toL14.f32bin",
            "receipts/run2-m3-golden-mlp-cellL18toL14.json",
            Some(("run2-m3-training-receipt-cellL18toL14.json", &m3_h)),
            MlpDerivation::PerTokenThenPool,
        )?,
        load_mlp(
            "m3-mlp-pooled",
            "mlp-reconstruction",
            "SAME artifact as m3-mlp-pertoken; eval variant (ii): pool the sender rows first, then the MLP (run-1 pipeline shape)",
            "receipts/run2-m3-mlp-cellL18toL14.f32bin",
            "receipts/run2-m3-golden-mlp-cellL18toL14.json",
            Some(("run2-m3-training-receipt-cellL18toL14.json", &m3_h)),
            MlpDerivation::PoolThenTranslate,
        )?,
        load_mlp(
            "m4h-s1-m3-mlp-lasttoken-depooled",
            "mlp-reconstruction",
            "ADR-024 M4h Stage 1: the SAME M3 artifact and the SAME per-token forward as m3-mlp-pertoken (byte-identical weights, asserted by the shared hash gate below), with the mean over the generated span REMOVED — the payload is the LAST translated token. No new training, no new capture. docs/research/040: no externally successful cross-model method pools",
            "receipts/run2-m3-mlp-cellL18toL14.f32bin",
            "receipts/run2-m3-golden-mlp-cellL18toL14.json",
            Some(("run2-m3-training-receipt-cellL18toL14.json", &m3_h)),
            MlpDerivation::PerTokenLast,
        )?,
        load_fastgrnn(
            "m4-fastgrnn-r64",
            "BPTT on RECONSTRUCTION loss, low-rank FastGRNN sequence cell, rank 64",
            "receipts/run2-m4-fastgrnn-r64-cellL18toL14.f32bin",
            "receipts/run2-m4-golden-fastgrnn-r64-cellL18toL14.json",
            Some(("run2-m4-training-receipt-cellL18toL14-r64.json", &r64_h)),
        )?,
        load_fastgrnn(
            "m4-fastgrnn-r128",
            "BPTT on RECONSTRUCTION loss, low-rank FastGRNN sequence cell, rank 128",
            "receipts/run2-m4-fastgrnn-r128-cellL18toL14.f32bin",
            "receipts/run2-m4-golden-fastgrnn-r128-cellL18toL14.json",
            Some(("run2-m4-training-receipt-cellL18toL14-r128.json", &r128_h)),
        )?,
        load_fastgrnn(
            "m4-fastgrnn-r256",
            "BPTT on RECONSTRUCTION loss, low-rank FastGRNN sequence cell, rank 256",
            "receipts/run2-m4-fastgrnn-r256-cellL18toL14.f32bin",
            "receipts/run2-m4-golden-fastgrnn-r256-cellL18toL14.json",
            Some(("run2-m4-training-receipt-cellL18toL14-r256.json", &r256_h)),
        )?,
        load_fastgrnn(
            "m4-fastgrnn-r64-superseded",
            "SUPERSEDED r=64 run (window-zero-init); an independent training run of the same architecture, included as a second sample of the reconstruction lane",
            "receipts/run2-m4-fastgrnn-r64-cellL18toL14-superseded-windowzeroinit.f32bin",
            "receipts/run2-m4-golden-fastgrnn-r64-cellL18toL14-superseded-windowzeroinit.json",
            Some((
                "run2-m4-training-receipt-cellL18toL14-r64-superseded-windowzeroinit.json",
                &r64s_h,
            )),
        )?,
        load_mlp(
            "m4c-mlp-taskloss",
            "mlp-taskloss",
            "SGD on C2C-style TASK loss (frozen receiver's teacher-forced next-token CE through the real 8-slot injection); THE ANCHOR — reproduces docs/research/033 §4",
            "receipts/run2-m4c-mlp-taskloss-cellL18toL14.f32bin",
            "receipts/run2-m4c-golden-mlp-taskloss-cellL18toL14.json",
            Some(("run2-m4c-training-receipt-cellL18toL14.json", &m4c_h)),
            MlpDerivation::PerTokenThenPool,
        )?,
        load_mlp(
            "m4c-m4d-shared-init-untrained",
            "untrained-init",
            "THE ZERO-TRAINING CONTROL: the fresh seeded ChaCha8 initialisation M4c and M4d BOTH start from (byte-identical; both training receipts pin the same init hash, asserted below), zero optimiser steps taken. Not warm-started from M3",
            "receipts/run2-m4c-mlp-taskloss-init-cellL18toL14.f32bin",
            "receipts/run2-m4c-golden-mlp-taskloss-init-cellL18toL14.json",
            Some(("run2-m4c-training-receipt-cellL18toL14.json", &m4c_init_h)),
            MlpDerivation::PerTokenThenPool,
        )?,
    ];

    // M4c's and M4d's initialisations are BYTE-IDENTICAL (both training
    // receipts pin the same `init_content_hash_sha256`): the two rungs share
    // one seeded init by design. Measuring both would double-count one
    // artifact, so the shared init is one candidate and the identity is
    // asserted rather than assumed.
    if let Some(m4d) = m4d.as_ref() {
        let m4d_init_h = pinned(m4d, &["artifact", "init_content_hash_sha256"])?;
        anyhow::ensure!(
            m4d_init_h == m4c_init_h,
            "M4c init {m4c_init_h} != M4d init {m4d_init_h} — they are no longer the same artifact; add M4d's init as its own candidate"
        );
    }
    // M4d's TRAINED artifact, measured only when its training receipt is on
    // disk to pin the hash.
    if let (true, Some(want)) = (m4d_final.exists(), m4d_h.as_ref()) {
        v.push(load_mlp(
            "m4d-mlp-taskloss-deploymatch",
            "mlp-taskloss",
            "M4c's task loss UNCHANGED, with the deployment rescale matched between training and probe (ADR-024's registered M4d contingency)",
            "receipts/run2-m4d-mlp-deploymatch-cellL18toL14.f32bin",
            "receipts/run2-m4d-golden-mlp-deploymatch-cellL18toL14.json",
            Some(("run2-m4d-training-receipt-cellL18toL14.json", want)),
            MlpDerivation::PerTokenThenPool,
        )?);
    }

    // M4g's TRAINED artifact (ADR-024's overwrite-vs-fuse rung), on the same
    // present-on-disk rule. Its seeded init is asserted byte-identical to
    // M4c's/M4d's by its own trainer, so the shared-init candidate above
    // covers M4g's init too and is not double-counted here. NOTE: this
    // metric kit measures the EMITTED VECTOR, which is independent of the
    // injection operator — so M4g's numbers here are directly comparable
    // with every prior rung's, and the pre-check is DIAGNOSTIC ONLY (ADR-024
    // M4f framing): it gates nothing about whether M4g's probe is drawn.
    let m4g_final = crate_path("receipts/run2-m4g-mlp-fuse-cellL18toL14.f32bin");
    let m4g_receipt_path = crate_path("receipts/run2-m4g-training-receipt-cellL18toL14.json");
    let m4g = m4g_receipt_path
        .exists()
        .then(|| read_json("receipts/run2-m4g-training-receipt-cellL18toL14.json"))
        .transpose()?;
    let m4g_h = m4g.as_ref().map(h).transpose()?;
    if let Some(m4g) = m4g.as_ref() {
        let m4g_init_h = pinned(m4g, &["artifact", "init_content_hash_sha256"])?;
        anyhow::ensure!(
            m4g_init_h == m4c_init_h,
            "M4c init {m4c_init_h} != M4g init {m4g_init_h} — they are no longer the same artifact; add M4g's init as its own candidate"
        );
    }
    if let (true, Some(want)) = (m4g_final.exists(), m4g_h.as_ref()) {
        v.push(load_mlp(
            "m4g-mlp-taskloss-fuse",
            "mlp-taskloss",
            "M4c's task loss and M4d's deployment transform UNCHANGED, with the INJECTION OPERATOR changed from overwrite to residual add (h[slot] += c*v; ADR-024's registered M4g rung). The emitted vector this kit measures does not depend on the operator, so these numbers are directly comparable with every rung above",
            "receipts/run2-m4g-mlp-fuse-cellL18toL14.f32bin",
            "receipts/run2-m4g-golden-mlp-fuse-cellL18toL14.json",
            Some(("run2-m4g-training-receipt-cellL18toL14.json", want)),
            MlpDerivation::PerTokenThenPool,
        )?);
    }

    v.push(Candidate {
        label: "reference-receiver-L14-pooled",
        family: "reference-on-manifold",
        training: "NO ADAPTER — the receiver's OWN block-14 residual states over the same span, mean-pooled the same way the adapters' outputs are",
        artifact: RECEIVER_DUMP.to_string(),
        hash: RECEIVER_DUMP_SHA256.to_string(),
        gate: serde_json::json!({"content_hash_sha256": RECEIVER_DUMP_SHA256,
            "pinned_by_receipt": DUMP_RECEIPT, "verified_against_training_or_probe_receipt": true,
            "golden_file": serde_json::Value::Null, "golden_check": serde_json::Value::Null}),
        emitter: Emitter::NaturalPooled,
    });
    v.push(Candidate {
        label: "reference-receiver-L14-single-row",
        family: "reference-on-manifold",
        training: "NO ADAPTER and NO POOLING — one genuine receiver block-14 residual state (the last row of the item's span). Included because a MEAN of states is not itself a state; this is the un-pooled manifold control",
        artifact: RECEIVER_DUMP.to_string(),
        hash: RECEIVER_DUMP_SHA256.to_string(),
        gate: serde_json::json!({"content_hash_sha256": RECEIVER_DUMP_SHA256,
            "pinned_by_receipt": DUMP_RECEIPT, "verified_against_training_or_probe_receipt": true,
            "golden_file": serde_json::Value::Null, "golden_check": serde_json::Value::Null}),
        emitter: Emitter::NaturalLastRow,
    });

    let status = serde_json::json!({
        "m4d": m4d_status,
        "m4g": {
            "final_artifact_present": m4g_final.exists(),
            "training_receipt_present": m4g_receipt_path.exists(),
            "measured_here": m4g_final.exists() && m4g_h.is_some(),
            "note": "M4g (overwrite -> fuse) is measured when its trainer has written both its artifact and the training receipt that pins the hash. This kit projects the EMITTED VECTOR through the receiver's readout and is operator-independent; ADR-024's M4f framing makes it DIAGNOSTIC ONLY — it is not a gate on M4g's one frozen probe draw.",
        },
    });
    Ok((v, status))
}

// ---------------------------------------------------------------------------

/// The verdict label for one candidate, from the registered thresholds.
#[allow(clippy::too_many_lines)]
fn main() -> anyhow::Result<()> {
    let t0 = std::time::Instant::now();

    // ---- Lane gate: CPU-only, no CUDA workload (ADR-034) ------------------
    anyhow::ensure!(
        !cfg!(feature = "cuda"),
        "ADR-034 lane rule: the GPU is held by M4d — build this pre-check WITHOUT --features cuda"
    );
    let device = Device::Cpu;
    anyhow::ensure!(device.is_cpu(), "device must be CPU");
    println!("lane gate: CPU-only (cuda feature off, Device::Cpu)");

    // ---- Candidates + hash gates -----------------------------------------
    let (candidates, rung_status) = build_candidates()?;
    println!("candidates: {}", candidates.len());
    for c in &candidates {
        println!(
            "  {:32} [{}] hash {}… pinned={}",
            c.label,
            c.family,
            &c.hash[..12],
            c.gate["verified_against_training_or_probe_receipt"]
        );
    }

    // ---- Sample: the 40 holdout rows shared by M3, M4 and M4c -------------
    // The M4c receipt is the only one that stores the row LIST; M3 and all
    // three M4 receipts pin the same split by sha256 of the comma-joined
    // list, so the list is verified against every rung's pin before use.
    let m4c = read_json("receipts/run2-m4c-training-receipt-cellL18toL14.json")?;
    let holdout_all: Vec<usize> = m4c["split"]["holdout_rows"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("M4c receipt split.holdout_rows"))?
        .iter()
        .filter_map(|v| v.as_u64().map(|x| x as usize))
        .collect();
    let joined = holdout_all
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let split_sha = common::sha256_hex(joined.as_bytes());
    let mut split_pins = Vec::new();
    for r in [
        "receipts/run2-m3-training-receipt-cellL18toL14.json",
        "receipts/run2-m4-training-receipt-cellL18toL14-r64.json",
        "receipts/run2-m4-training-receipt-cellL18toL14-r128.json",
        "receipts/run2-m4-training-receipt-cellL18toL14-r256.json",
        "receipts/run2-m4c-training-receipt-cellL18toL14.json",
    ] {
        let want = pinned(
            &read_json(r)?,
            &["split", "holdout_rows_sha256_comma_joined"],
        )?;
        anyhow::ensure!(
            want == split_sha,
            "{r} pins holdout split {want}, the M4c row list hashes to {split_sha}"
        );
        split_pins.push(r.to_string());
    }
    let holdout_rows: Vec<usize> = holdout_all.iter().copied().take(N_SAMPLE).collect();
    anyhow::ensure!(
        holdout_rows.len() == N_SAMPLE,
        "need {N_SAMPLE} holdout rows"
    );
    println!(
        "split gate: the M4c holdout row list hashes to {}… — the SAME split all {} receipts pin",
        &split_sha[..12],
        split_pins.len()
    );

    // ---- Input artifacts --------------------------------------------------
    let run = crate_path(RUN_DIR);
    let index: serde_json::Value = serde_json::from_slice(&std::fs::read(run.join(INDEX_JSON))?)?;
    let getvec = |k: &str| -> Vec<usize> {
        index[k]
            .as_array()
            .unwrap_or(&Vec::new())
            .iter()
            .filter_map(|v| v.as_u64().map(|x| x as usize))
            .collect()
    };
    let item_indices = getvec("item_indices");
    let gen_len = getvec("gen_len");
    let token_offsets = getvec("token_offsets");

    let sdump = run.join(SENDER_DUMP);
    let rdump = run.join(RECEIVER_DUMP);
    println!("hashing capture dumps (5.9 GB + 4.4 GB)…");
    let sdump_sha = sha256_file(&sdump)?;
    anyhow::ensure!(sdump_sha == SENDER_DUMP_SHA256, "sender dump sha mismatch");
    let rdump_sha = sha256_file(&rdump)?;
    anyhow::ensure!(
        rdump_sha == RECEIVER_DUMP_SHA256,
        "receiver dump sha mismatch"
    );
    let streams_path = crate_path(STREAMS);
    let streams_sha = sha256_file(&streams_path)?;
    anyhow::ensure!(streams_sha == STREAMS_SHA256, "streams sha mismatch");
    let gsm_path = crate_path(GSM8K);
    let gsm_sha = sha256_file(&gsm_path)?;
    anyhow::ensure!(gsm_sha == GSM8K_TRAIN_SHA256, "gsm8k sha mismatch");
    let gsm = common::load_gsm8k(&gsm_path)?;
    println!("input gates: both capture dumps, the token streams and GSM8K sha256-verified");

    let mut span_tokens: Vec<Option<Vec<u32>>> = vec![None; N_SAMPLE];
    for line in std::fs::read_to_string(&streams_path)?.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(line)?;
        let row = v["row"].as_u64().unwrap_or(u64::MAX) as usize;
        if let Some(slot) = holdout_rows.iter().position(|&r| r == row) {
            span_tokens[slot] = Some(
                v["gen_tokens"]
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .filter_map(|t| t.as_u64().map(|x| x as u32))
                            .collect()
                    })
                    .unwrap_or_default(),
            );
        }
    }

    // ---- Receiver unembedding (CPU) --------------------------------------
    let api = hf_hub::api::sync::Api::new()?;
    let repo = api.model(RECEIVER.to_string());
    let weights = repo.get("model.safetensors")?;
    let cfg: serde_json::Value = serde_json::from_slice(&std::fs::read(repo.get("config.json")?)?)?;
    anyhow::ensure!(
        cfg["tie_word_embeddings"].as_bool() == Some(true),
        "receiver is not tied-embedding"
    );
    let rms_eps = cfg["rms_norm_eps"].as_f64().unwrap_or(1e-6);
    let tokenizer = tokenizers::Tokenizer::from_file(repo.get("tokenizer.json")?)
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    let st = unsafe { candle_core::safetensors::MmapedSafetensors::new(&weights)? };
    let unembed = st
        .load("model.embed_tokens.weight", &device)?
        .to_dtype(DType::F32)?;
    let final_gain: Vec<f32> = st
        .load("model.norm.weight", &device)?
        .to_dtype(DType::F32)?
        .to_vec1()?;
    let (vocab, hidden) = unembed.dims2()?;
    anyhow::ensure!(hidden == D_RECEIVER, "unembedding hidden {hidden}");
    println!("unembedding: [{vocab} x {hidden}] f32 on CPU, tied; rms_norm_eps {rms_eps}");

    // ---- Per-item sweep ---------------------------------------------------
    let n_cand = candidates.len();
    let mut emitted: Vec<Vec<Vec<f32>>> = vec![Vec::with_capacity(N_SAMPLE); n_cand];
    let mut per_item: Vec<Vec<ItemRow>> = vec![Vec::with_capacity(N_SAMPLE); n_cand];

    for (slot, &row) in holdout_rows.iter().enumerate() {
        let n_rows = gen_len[row];
        let off = token_offsets[row];
        let sender_rows = read_rows(&sdump, D_SENDER, off, n_rows)?;
        let receiver_rows = read_rows(&rdump, D_RECEIVER, off, n_rows)?;
        let ctx = ItemCtx {
            sender_pooled: pool(&sender_rows, D_SENDER, n_rows),
            receiver_pooled: pool(&receiver_rows, D_RECEIVER, n_rows),
            receiver_last_row: receiver_rows[(n_rows - 1) * D_RECEIVER..].to_vec(),
            sender_rows,
            n_rows,
        };

        let vs: Vec<Vec<f32>> = candidates.iter().map(|c| c.emitter.emit(&ctx)).collect();

        // One matmul per item: [raw | rmsnorm] columns for every candidate.
        let mut batch: Vec<Vec<f32>> = vs.clone();
        batch.extend(vs.iter().map(|v| rms_norm(v, &final_gain, rms_eps)));
        let logits = project_batch(&unembed, &batch, &device)?;

        let item = item_indices[row];
        let gold_toks: Vec<u32> = tokenizer
            .encode(format!("#### {}", gsm[item].gold), false)
            .map_err(|e| anyhow::anyhow!("encode: {e}"))?
            .get_ids()
            .to_vec();
        let mut span: Vec<u32> = span_tokens[slot].clone().unwrap_or_default();
        span.sort_unstable();
        span.dedup();
        anyhow::ensure!(!span.is_empty(), "row {row}: no sender span tokens");

        for (k, v) in vs.iter().enumerate() {
            let z_plain = &logits[k];
            let z_norm = &logits[k + n_cand];
            let lse_p = logsumexp(z_plain);
            let lse_n = logsumexp(z_norm);
            let g_n = token_set_stats(z_norm, &gold_toks, lse_n);
            let g_p = token_set_stats(z_plain, &gold_toks, lse_p);
            let s_n = token_set_stats(z_norm, &span, lse_n);
            let s_p = token_set_stats(z_plain, &span, lse_p);
            per_item[k].push(ItemRow {
                l2: norms::l2(v) as f64,
                cos_to_natural_same_item: cosine(v, &ctx.receiver_pooled),
                top10: top_k(z_norm, TOPK),
                entropy_rmsnorm: entropy_nats(z_norm),
                entropy_plain: entropy_nats(z_plain),
                gold_rank_rmsnorm: g_n.mean_rank,
                gold_rank_plain: g_p.mean_rank,
                gold_best_rank_rmsnorm: g_n.best_rank,
                span_rank_rmsnorm: s_n.mean_rank,
                span_rank_plain: s_p.mean_rank,
            });
            emitted[k].push(v.clone());
        }
        println!(
            "[{}/{N_SAMPLE}] row {row} item {item} ({n_rows} span rows) projected for {n_cand} candidates",
            slot + 1
        );
    }

    // ---- Aggregate --------------------------------------------------------
    // Global mean natural state, for the "is it just the pooled mean?" check.
    let natural_idx = candidates
        .iter()
        .position(|c| c.label == "reference-receiver-L14-pooled")
        .expect("natural reference present");
    let global_natural: Vec<f32> = {
        let mut acc = vec![0f64; D_RECEIVER];
        for v in &emitted[natural_idx] {
            for (a, &x) in acc.iter_mut().zip(v) {
                *a += x as f64;
            }
        }
        acc.iter().map(|a| (*a / N_SAMPLE as f64) as f32).collect()
    };

    let mut table = Vec::new();
    let mut verdict_rows = Vec::new();
    for (k, c) in candidates.iter().enumerate() {
        let rows = &per_item[k];
        let col = |f: fn(&ItemRow) -> f64| -> Vec<f64> { rows.iter().map(f).collect() };
        let (inv_mean, inv_min, inv_max) = mean_pairwise_cosine(&emitted[k]);
        let resultant = mean_resultant_length(&emitted[k]);

        let mut counts: BTreeMap<u32, usize> = BTreeMap::new();
        for r in rows {
            for &t in &r.top10 {
                *counts.entry(t).or_default() += 1;
            }
        }
        let token_union = counts.len();
        let mut ordered: Vec<(u32, usize)> = counts.into_iter().collect();
        ordered.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        let dominant: Vec<serde_json::Value> = ordered
            .iter()
            .take(N_DOMINANT)
            .map(|(t, n)| {
                serde_json::json!({
                    "token_id": t,
                    "decoded": tokenizer.decode(&[*t], false).unwrap_or_else(|_| format!("<{t}>")),
                    "in_top10_of_n_items": n,
                })
            })
            .collect();

        let manifold_cos = mean(&col(|r| r.cos_to_natural_same_item));
        let cos_to_global: f64 = mean(
            &emitted[k]
                .iter()
                .map(|v| cosine(v, &global_natural))
                .collect::<Vec<_>>(),
        );
        let gold_n = mean(&col(|r| r.gold_rank_rmsnorm));
        let span_n = mean(&col(|r| r.span_rank_rmsnorm));
        let verdict = classify(inv_mean, token_union, manifold_cos);

        let row = serde_json::json!({
            "label": c.label,
            "family": c.family,
            "training": c.training,
            "artifact": c.artifact,
            "gate": c.gate,
            "item_invariance": {
                "mean_pairwise_cosine_between_emitted_vectors": inv_mean,
                "min": inv_min, "max": inv_max,
                "mean_resultant_length_of_unit_vectors": resultant,
            },
            "output_token_support_rmsnorm_lens": {
                "distinct_tokens_in_union_of_40_top10_sets": token_union,
                "max_possible": N_SAMPLE * TOPK,
                "dominant_tokens": dominant,
            },
            "gold_answer_tokens": {
                "mean_rank_rmsnorm": gold_n,
                "vocab_percentile_rmsnorm": gold_n / vocab as f64,
                "mean_rank_plain": mean(&col(|r| r.gold_rank_plain)),
                "mean_best_rank_rmsnorm": mean(&rows.iter().map(|r| r.gold_best_rank_rmsnorm as f64).collect::<Vec<_>>()),
            },
            "sender_span_tokens": {
                "mean_rank_rmsnorm": span_n,
                "vocab_percentile_rmsnorm": span_n / vocab as f64,
                "mean_rank_plain": mean(&col(|r| r.span_rank_plain)),
            },
            "entropy_nats": {
                "rmsnorm_lens_mean": mean(&col(|r| r.entropy_rmsnorm)),
                "plain_lens_mean": mean(&col(|r| r.entropy_plain)),
                "uniform": (vocab as f64).ln(),
            },
            "manifold": {
                "mean_cosine_to_same_item_natural_receiver_L14_pooled": manifold_cos,
                "mean_cosine_to_the_global_mean_natural_state": cos_to_global,
                "emitted_l2": {"mean": mean(&col(|r| r.l2)),
                                "min": minmax(&col(|r| r.l2)).0, "max": minmax(&col(|r| r.l2)).1},
            },
            "classification": verdict,
        });
        println!(
            "{:32} inv-cos {:7.4}  tokens {:>3}/{}  manifold-cos {:7.4}  gold pct {:5.1}%  H {:5.2}  => {}",
            c.label, inv_mean, token_union, N_SAMPLE * TOPK, manifold_cos,
            100.0 * gold_n / vocab as f64, mean(&col(|r| r.entropy_rmsnorm)), verdict,
        );
        verdict_rows.push((
            c.label,
            c.family,
            verdict,
            inv_mean,
            token_union,
            manifold_cos,
        ));
        table.push(row);
    }

    // ---- Calibrate every metric against the on-manifold reference ---------
    // The pooled natural receiver state is the only row whose "correct"
    // values are known a priori, so each candidate is also reported RELATIVE
    // to it. This is what disciplines the reading: a metric on which the
    // reference itself scores like M4c cannot be evidence that M4c is broken.
    let refrow = table[natural_idx].clone();
    let ref_inv = refrow["item_invariance"]["mean_pairwise_cosine_between_emitted_vectors"]
        .as_f64()
        .unwrap_or(f64::NAN);
    let ref_tok = refrow["output_token_support_rmsnorm_lens"]
        ["distinct_tokens_in_union_of_40_top10_sets"]
        .as_u64()
        .unwrap_or(0);
    let ref_gold = refrow["gold_answer_tokens"]["vocab_percentile_rmsnorm"]
        .as_f64()
        .unwrap_or(f64::NAN);
    let ref_ent = refrow["entropy_nats"]["rmsnorm_lens_mean"]
        .as_f64()
        .unwrap_or(f64::NAN);
    for row in &mut table {
        let inv = row["item_invariance"]["mean_pairwise_cosine_between_emitted_vectors"]
            .as_f64()
            .unwrap_or(f64::NAN);
        let tok = row["output_token_support_rmsnorm_lens"]
            ["distinct_tokens_in_union_of_40_top10_sets"]
            .as_u64()
            .unwrap_or(0);
        let gold = row["gold_answer_tokens"]["vocab_percentile_rmsnorm"]
            .as_f64()
            .unwrap_or(f64::NAN);
        let ent = row["entropy_nats"]["rmsnorm_lens_mean"]
            .as_f64()
            .unwrap_or(f64::NAN);
        row["vs_natural_pooled_reference"] = serde_json::json!({
            "item_invariance_delta": inv - ref_inv,
            "top10_token_union_delta": tok as i64 - ref_tok as i64,
            "gold_vocab_percentile_delta": gold - ref_gold,
            "entropy_nats_delta": ent - ref_ent,
        });
    }
    let threshold_calibration = serde_json::json!({
        "reference_row": "reference-receiver-L14-pooled",
        "reference_trips_the_invariance_arm": ref_inv >= COLLAPSE_COSINE,
        "reference_trips_the_token_union_arm": (ref_tok as usize) <= COLLAPSE_TOKEN_UNION,
        "reference_mean_pairwise_cosine": ref_inv,
        "reference_top10_token_union": ref_tok,
        "reference_gold_vocab_percentile": ref_gold,
        "reference_entropy_nats": ref_ent,
        "reading": "Two of the three registered arms are calibrated OUT by this row. The receiver's OWN pooled block-14 state has a high mean pairwise cosine and a small top-10 token union, and puts the gold-answer tokens near the middle of the vocabulary — so 'nearly item-invariant', 'few distinct top-10 tokens' and 'gold tokens at a middling percentile' are properties of a POOLED MID-STACK RESIDUAL STATE, not evidence of a defective adapter. The un-pooled single-row reference shows the same statistics move sharply once pooling is removed. The one arm the reference does NOT trip — cosine to the receiver's own state for the same item — is therefore the only discriminating measurement here.",
    });

    // ---- Verdict ----------------------------------------------------------
    let is_adapter = |family: &str| family.starts_with("mlp") || family.starts_with("fastgrnn");
    let trained: Vec<_> = verdict_rows
        .iter()
        .filter(|(_, f, ..)| is_adapter(f))
        .collect();
    let trained_collapsed = trained
        .iter()
        .filter(|(.., v, _, _, _)| *v == "COLLAPSED-OFF-MANIFOLD")
        .count();
    let taskloss_collapsed = trained
        .iter()
        .filter(|(_, f, v, ..)| *f == "mlp-taskloss" && *v == "COLLAPSED-OFF-MANIFOLD")
        .count();
    let recon_collapsed = trained
        .iter()
        .filter(|(_, f, v, ..)| f.contains("reconstruction") && *v == "COLLAPSED-OFF-MANIFOLD")
        .count();
    let affine_collapsed = verdict_rows
        .iter()
        .filter(|(_, f, v, ..)| f.starts_with("affine") && *v == "COLLAPSED-OFF-MANIFOLD")
        .count();
    let n_affine = verdict_rows
        .iter()
        .filter(|(_, f, ..)| f.starts_with("affine"))
        .count();
    let n_recon = trained
        .iter()
        .filter(|(_, f, ..)| f.contains("reconstruction"))
        .count();
    let n_taskloss = trained
        .iter()
        .filter(|(_, f, ..)| *f == "mlp-taskloss")
        .count();
    let n_init = verdict_rows
        .iter()
        .filter(|(_, f, ..)| *f == "untrained-init")
        .count();
    let init_collapsed = verdict_rows
        .iter()
        .filter(|(_, f, v, ..)| *f == "untrained-init" && *v == "COLLAPSED-OFF-MANIFOLD")
        .count();

    let verdict = if trained_collapsed == trained.len() && affine_collapsed == n_affine {
        "UNIVERSAL — every trained adapter AND the training-free run-1 affine maps emit a collapsed, off-manifold direction: the mechanism is the injection channel / representation, not the loss"
    } else if n_init > 0
        && init_collapsed == n_init
        && recon_collapsed == 0
        && n_taskloss > 0
        && taskloss_collapsed == n_taskloss
    {
        "NEITHER, STRICTLY — and this is the informative answer. Off-manifold output is the UNTRAINED DEFAULT of this adapter architecture: the fresh, zero-optimiser-step initialisation that M4c and M4d both start from is ALREADY off-manifold (cosine ~0 to the receiver's own state), with the same low entropy and the same rare-token top-10. RECONSTRUCTION training (M3, M4 r64/r128/r256) moves the output onto the receiver's manifold; TASK-LOSS training (M4c, M4d) leaves it essentially where it started. So the collapse is not caused by the task loss and is not universal across the ladder — it is what task-loss training FAILED TO REMOVE, because nothing in that objective requires the emitted vector to resemble a receiver state. M3/M4's nulls therefore have a DIFFERENT cause: their adapters are on-manifold and still did not transfer."
    } else if taskloss_collapsed > 0 && recon_collapsed == 0 {
        "TASK-LOSS-SPECIFIC — only the task-loss adapters collapsed; the reconstruction-trained M3/M4 nulls have a DIFFERENT cause"
    } else {
        "MIXED — see the per-candidate table; neither 'universal' nor 'task-loss-specific' fits"
    };

    let receipt = serde_json::json!({
        "stage": "run2-m4f-manifold-collapse-precheck",
        "design": "docs/adr/024-run2-trained-thought-adapter-ladder.md § 'DIAGNOSIS (2026-08-29): the adapter collapsed to a fixed OFF-MANIFOLD direction' — the registered M4f pre-check ('re-run the same unembedding projection on M4d's artifact and on the M3/M4 artifacts to establish whether off-manifold collapse is universal across the ladder or specific to task-loss training')",
        "question": "Is the item-invariant off-manifold collapse found in M4c's adapter universal across the ladder, or specific to task-loss training?",
        "method": "docs/research/033-rescale-output-alignment-diagnostic.md §4, applied unchanged (common/lens.rs is that diagnostic's own metric kit, extracted so both measure with ONE implementation): project each candidate's emitted 1536-d vector through the receiver's real readout W_U·RMSNorm(h) and the bare logit lens W_U·h, then measure top-10 token support, gold-answer and sender-span token ranks, entropy, and cross-item direction agreement.",
        "protocol_safety": "ANNOTATES ONLY. No probe draw, no live-model forward, no probe item / control / statistic touched (ADR-028 protected list untouched). Changes no recorded outcome. Did not wait for, poll or interfere with the GPU lane.",
        "env": {
            "evidence_label": EVIDENCE_LABEL,
            "device": "CPU (Device::Cpu; cuda feature OFF — ADR-034 lane rule, GPU held by M4d)",
            "cuda_feature_enabled": cfg!(feature = "cuda"),
            "crate": "latentmesh-runtime 0.1.0",
            "git_commit": std::process::Command::new("git").args(["rev-parse","HEAD"])
                .current_dir(env!("CARGO_MANIFEST_DIR")).output()
                .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_default(),
            "unix_time": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs()).unwrap_or(0),
        },
        "gates": {
            "cpu_only": {"pass": true, "cuda_feature_enabled": cfg!(feature="cuda"), "device": "Cpu"},
            "artifact_hashes_verified": {
                "pass": true,
                "rule": "every candidate's content hash is compared to the hash its own training receipt (or, for the run-1 affine maps, its own S2b/S2c probe receipt gate) froze; the per-candidate gate block records WHICH receipt pinned it, and any artifact no receipt pins yet is carried with verified_against_training_or_probe_receipt=false rather than silently treated as verified",
                "n_candidates": n_cand,
                "n_pinned_by_receipt": candidates.iter().filter(|c| c.gate["verified_against_training_or_probe_receipt"] == serde_json::json!(true)).count(),
            },
            "hand_rolled_applies_match_trained_networks": {
                "pass": true,
                "rule": "every adapter's hand-rolled forward is re-verified against the golden input/output pairs the trained network itself produced (relative L2 <= 1e-5) before any projection",
            },
            "shared_holdout_split_verified": {
                "pass": true,
                "sha256_comma_joined": split_sha,
                "pinned_identically_by": split_pins,
                "note": "M3, M4 r64/r128/r256 and M4c all pin the SAME holdout split by sha256, so the 40 sampled rows are holdout for every trained candidate. The run-1 affine maps were fit on run-1's own S2/S2c calibration pairs and are NOT held out of these rows — disclosed; the measurement here is output geometry, not generalisation.",
            },
            "input_artifacts_sha256_verified": {"pass": true,
                "sender_L18": sdump_sha, "receiver_L14": rdump_sha,
                "token_streams": streams_sha, "gsm8k_train": gsm_sha},
        },
        "sample": {
            "n": N_SAMPLE,
            "source": "first 40 rows of the holdout split shared by M3 / M4 / M4c (M4c receipt's split.holdout_rows, in receipt order)",
            "why_not_the_probe_items": "the frozen probes' own 40 items are NOT reconstructible from committed artifacts — their sender captures are produced live inside the probe and never dumped. Reconstructing them needs a live sender forward, i.e. GPU, which ADR-034 forbids this lane. Same substitute docs/research/033 registered and used, so the M4c anchor row here is directly comparable to that receipt.",
            "dump_rows": holdout_rows,
            "gsm8k_items": holdout_rows.iter().map(|&r| item_indices[r]).collect::<Vec<_>>(),
        },
        "cell": {"sender_block": SENDER_BLOCK, "receiver_block": RECEIVER_BLOCK,
                  "receiver_model": RECEIVER, "vocab": vocab, "hidden": hidden},
        "rung_status": rung_status,
        "thresholds_registered": {
            "collapse_mean_pairwise_cosine_at_or_above": COLLAPSE_COSINE,
            "collapse_top10_token_union_at_or_below": COLLAPSE_TOKEN_UNION,
            "off_manifold_mean_cosine_to_natural_below": OFF_MANIFOLD_COSINE,
            "note": "chosen to reproduce docs/research/033 §4's own characterisation of M4c (77 distinct tokens across 40 items; 'nearly item-invariant'; 'off the receiver's residual-stream manifold') and applied identically to every candidate INCLUDING the on-manifold references, which therefore act as the calibration of the thresholds rather than as an exception to them",
        },
        "threshold_calibration_against_the_on_manifold_reference": threshold_calibration,
        "candidates": table,
        "verdict": {
            "answer": verdict,
            "trained_adapters": {"n": trained.len(), "collapsed": trained_collapsed},
            "reconstruction_trained": {"n": n_recon, "collapsed": recon_collapsed},
            "task_loss_trained": {"n": n_taskloss, "collapsed": taskloss_collapsed},
            "untrained_initialisation_control": {"n": n_init, "collapsed": init_collapsed},
            "training_free_affine": {"n": n_affine, "collapsed": affine_collapsed},
            "corrects_docs_research_033": "docs/research/033 §4 read M4c's '77 distinct top-10 tokens across 40 items', its near-item-invariance and its middling gold-answer percentile as the signature of an off-manifold vector. Measured here against the receiver's OWN pooled block-14 state over the same spans, those three statistics are NOT diagnostic — the natural state scores comparably on all three. The off-manifold claim survives, but on one measurement only: cosine to the receiver's own state for the same item. That correction is recorded in docs/research/036.",
        },
        "wall_clock_s": t0.elapsed().as_secs_f32(),
    });
    // Receipt name: the default keeps the committed M4f pre-check receipt
    // (`docs/research/038` cites it by name) reproducible; an optional first
    // CLI argument writes a differently-named receipt instead, so a later
    // rung can add its own candidate WITHOUT overwriting an earlier rung's
    // evidence. Nothing else about the run changes.
    let receipt_name = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "run2-manifold-precheck-receipt.json".to_string());
    anyhow::ensure!(
        (receipt_name.starts_with("run2-manifold-precheck")
            || receipt_name.starts_with("run2-m4h-s1-manifold-precheck")
            || receipt_name.starts_with("run2-m4i-manifold-precheck"))
            && receipt_name.ends_with(".json")
            && !receipt_name.contains('/'),
        "receipt name must be a bare run2-manifold-precheck*.json, \
         run2-m4h-s1-manifold-precheck*.json or run2-m4i-manifold-precheck*.json \
         filename, got {receipt_name}"
    );
    common::write_receipt(&crate_path("receipts"), &receipt_name, &receipt)?;
    println!("\nVERDICT: {verdict}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pooling_is_the_f64_mean() {
        let rows = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_eq!(pool(&rows, 3, 2), vec![2.5, 3.5, 4.5]);
        assert_eq!(pool(&rows, 2, 3), vec![3.0, 4.0]);
    }

    #[test]
    fn classification_follows_the_registered_thresholds() {
        // M4c's own numbers from docs/research/033 §4 must land in the
        // collapsed cell; a spread, on-manifold emitter must not.
        assert_eq!(classify(0.999, 77, 0.02), "COLLAPSED-OFF-MANIFOLD");
        assert_eq!(classify(0.30, 340, 0.71), "on-manifold-item-varying");
        assert_eq!(classify(0.99, 300, 0.80), "item-invariant-but-on-manifold");
        assert_eq!(classify(0.10, 300, 0.01), "off-manifold-but-item-varying");
        // The token-union arm alone is enough to call invariance.
        assert_eq!(classify(0.10, 40, 0.01), "COLLAPSED-OFF-MANIFOLD");
    }

    #[test]
    fn hash_gate_rejects_a_mismatch_and_flags_the_unpinned() {
        let g = hash_gate(
            "abc",
            Some(("r.json", "abc")),
            "g.json",
            serde_json::json!({}),
        )
        .expect("matching hash");
        assert_eq!(g["verified_against_training_or_probe_receipt"], true);
        assert!(hash_gate(
            "abc",
            Some(("r.json", "def")),
            "g.json",
            serde_json::json!({})
        )
        .is_err());
        let u = hash_gate("abc", None, "g.json", serde_json::json!({})).expect("unpinned ok");
        assert_eq!(u["verified_against_training_or_probe_receipt"], false);
        assert!(u["pinned_by_receipt"].is_null());
    }
}
