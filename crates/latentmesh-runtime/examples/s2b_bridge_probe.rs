//! S2b — mandatory S2→S3 bridge probe (ADR-023 A7 coordinator resolution +
//! Deviation 6 winner-cell ruling).
//!
//! Re-runs the S1a protocol with the S2-fitted CROSS-MODEL transform:
//! sender Qwen2.5-3B solves each item (greedy); its pooled residual after
//! the sender sweep block over the generated reasoning is pushed through the
//! frozen affine transform `y = μ_r + α·(z − μ_s)·R` and injected into
//! receiver Qwen2.5-1.5B placeholder slots. Default cell = the S2 winner
//! L18→L14 (transform content_hash eb3f42ed…); the ADR-023 Deviation 6
//! registered fallback is the anchor cell L24→L19 (its re-emitted artifact
//! passed via `--transform`/`--golden`/`--sender-block`/`--receiver-block`/
//! `--expected-hash`).
//!
//! PRE-COMMITTED ANALYSIS (fixed in this source before any run; the receipt
//! echoes it — identical to S1a's frozen protocol wherever it applies):
//!   - primary gate (A7(b)): one-sided exact sign test on paired ACCURACY,
//!     aligned_real > random, over discordant pairs, alpha = 0.05
//!   - A7(c) gate: `zerovec_injected` — a TRUE ZERO VECTOR written through
//!     the real 8-slot injection path (positions occupied; NOT S0's G5
//!     empty-position no-op branch, and NOT S1a's "zero" condition, which
//!     was uninjected) — reported against `baseline_uninjected`.
//!     Pre-committed pass criterion ("not catastrophically below baseline"):
//!     2 × zerovec accuracy ≥ baseline accuracy. Accuracies and paired sign
//!     tests are reported either way; the gate never gates the reporting.
//!   - secondary diagnostic (not a gate): paired teacher-forced mean NLL of
//!     "#### <gold>", same sign tests
//!   - decoding: greedy, batch=1, max 400 new tokens (S1a probe budget)
//!   - 8 slots, pool over the full generated span, rescale ON to the
//!     receiver's natural inject-block MEDIAN per-position L2 norm, measured
//!     per item over the slotted injection prompt (the S0 cross-model
//!     precedent, `s0-receipt.json config.injection_vector`). The zero
//!     vector carries `scale: None`: rescale-to-median is undefined at norm
//!     0, and zero-stays-zero is the only reading.
//!   - items: the SAME 40 GSM8K-train items as S1a (ChaCha8 seed 0x51A1);
//!     the derived indices are asserted equal to the committed S1a receipt's
//!     list. Random vectors: ChaCha8 per-item seed 0x51A2_0000 + index,
//!     norm-matched to the effective aligned_real vector (as S1a).
//!
//! Transform loading: latentmesh-align cannot be a path dependency here
//! (align → latentmesh-core pins `half = "=2.4.1"`; candle needs `half
//! ^2.5` — unresolvable in one lockfile), so the affine apply is
//! HAND-ROLLED and verified at startup against golden input/output pairs
//! produced by latentmesh-align's own `apply` (≥8 vectors, per-vector
//! relative L2 error ≤ 1e-5, recorded in the receipt). The artifact's
//! content_hash equals sha256(file bytes) by construction (calibrate writes
//! `serde_json::to_vec(&t)`, exactly the bytes `content_hash` hashes).
//!
//! Receipts: `crates/latentmesh-runtime/target/latentmesh-runs/s2b/`.
//! Run: PATH=/usr/local/cuda-12.8/bin:$PATH \
//!      cargo run --release --features cuda --example s2b_bridge_probe

#[path = "common/mod.rs"]
#[allow(dead_code)]
mod common;

use latentmesh_runtime::{
    capture::forward_capture,
    inject::{teacher_forced_nll, InjectionSpec},
    norms,
    sampler::{Sampler, Sampling},
    QwenRuntime,
};
// The run-1 affine bridge apply lives in `common/affine.rs` so the frozen
// probe and any later re-projection diagnostic share ONE implementation
// (golden-verified against latentmesh-align's own apply at startup).
use common::affine::{verify_against_golden, AffineTransform};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::path::{Path, PathBuf};

const SENDER: &str = "Qwen/Qwen2.5-3B-Instruct";
const RECEIVER: &str = "Qwen/Qwen2.5-1.5B-Instruct";
const ITEM_SEED: u64 = 0x51A1;
const RANDVEC_SEED_BASE: u64 = 0x51A2_0000;
const MAX_NEW_TOKENS: usize = 400;
const N_SLOTS: usize = 8;
const ALPHA: f64 = 0.05;
const SYSTEM: &str = "You are a careful math tutor.";
const GOLDEN_REL_TOL: f32 = 1e-5;
/// GSM8K train.jsonl pin (ADR-023 dataset table, verbatim from the S1a receipt).
const GSM8K_TRAIN_SHA256: &str = "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465";
/// S2 winner cell L18→L14 (ADR-023 stage table / Deviation 6).
const WINNER_HASH: &str = "eb3f42edde853824642a2b811577e2c767f73c2c179fe03a05ac8dac23704457";
/// Registered fallback anchor cell L24→L19 (`s2-calibration-receipt.json` sweep_cells).
const ANCHOR_HASH: &str = "e892cfb7034012c3fb3f9fa1979e5a7fedfbb2adfd3e11fe575c40c2835cc801";
/// Committed S1a receipt whose item indices this probe must reproduce.
const S1A_RECEIPT: &str = "receipts/s1a-receipt-slots8-block19-poolfull-rescaletrue-n40.json";

#[derive(Debug, Clone)]
struct ProbeConfig {
    transform: PathBuf,
    golden: PathBuf,
    sender_block: usize,
    receiver_block: usize,
    expected_hash: String,
    n_items: usize,
    /// ADR-023 Deviation 7 generated-pairs contingency run: the transform
    /// hash is the NEW generated-pairs fit (validated against
    /// `--expected-hash` exactly as always), while every protocol knob —
    /// items, sign test, alpha, slots, pooling, rescale, decoding — stays
    /// this file's frozen S1a protocol, untouched by the flag.
    dev7_contingency: bool,
}

impl ProbeConfig {
    /// Pre-committed iff the frozen probe shape holds AND the cell is one of
    /// the two ADR-023-registered cells: the S2 winner L18→L14 or the
    /// Deviation 6 fallback anchor L24→L19 (the fallback is itself
    /// registered, so a fallback run stays pre-committed). Under
    /// `--dev7-contingency` (the ADR-023 Deviation 7 pre-registered
    /// generated-pairs contingency) the same two cell shapes are
    /// pre-committed with the generated-pairs transform hash — the
    /// contingency, its outcome rule, and its one-probe-per-cell fallback
    /// chain are themselves frozen in ADR-023 before this run.
    fn pre_committed(&self) -> bool {
        let winner_shape = self.sender_block == 18 && self.receiver_block == 14;
        let anchor_shape = self.sender_block == 24 && self.receiver_block == 19;
        let winner_cell = winner_shape && self.expected_hash == WINNER_HASH;
        let anchor_cell = anchor_shape && self.expected_hash == ANCHOR_HASH;
        let dev7_cell = self.dev7_contingency && (winner_shape || anchor_shape);
        self.n_items == 40 && (winner_cell || anchor_cell || dev7_cell)
    }
    fn tag(&self) -> String {
        format!(
            "cellL{}toL{}-slots8-poolfull-rescaletrue-n{}{}",
            self.sender_block,
            self.receiver_block,
            self.n_items,
            if self.dev7_contingency {
                "-genpairs"
            } else {
                ""
            }
        )
    }
}

fn crate_path(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

fn parse_args() -> ProbeConfig {
    let mut cfg = ProbeConfig {
        transform: crate_path("receipts/transform-L18-to-L14.json"),
        golden: crate_path("receipts/s2b-golden-affine-L18-to-L14.json"),
        sender_block: 18,
        receiver_block: 14,
        expected_hash: WINNER_HASH.to_string(),
        n_items: 40,
        dev7_contingency: false,
    };
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        let next = |i: usize| args.get(i + 1).cloned().expect("missing arg value");
        match args[i].as_str() {
            "--dev7-contingency" => {
                cfg.dev7_contingency = true;
                i += 1;
                continue;
            }
            "--transform" => cfg.transform = PathBuf::from(next(i)),
            "--golden" => cfg.golden = PathBuf::from(next(i)),
            "--sender-block" => cfg.sender_block = next(i).parse().expect("--sender-block N"),
            "--receiver-block" => cfg.receiver_block = next(i).parse().expect("--receiver-block N"),
            "--expected-hash" => cfg.expected_hash = next(i),
            "--items" => cfg.n_items = next(i).parse().expect("--items N"),
            other => panic!("unknown arg {other}"),
        }
        i += 2;
    }
    cfg
}

fn main() -> anyhow::Result<()> {
    let t0 = std::time::Instant::now();
    let cfg = parse_args();
    let nvcc = latentmesh_runtime::assert_cuda_build_env().map_err(anyhow::Error::msg)?;
    println!(
        "build-env guard: {nvcc}\nconfig: {cfg:?} (pre_committed={})",
        cfg.pre_committed()
    );

    // ---- Transform: hash gate + hand-rolled-apply golden verification -----
    let t_bytes = std::fs::read(&cfg.transform)?;
    let t_sha = common::sha256_hex(&t_bytes);
    anyhow::ensure!(
        t_sha == cfg.expected_hash,
        "transform {} sha256/content_hash {t_sha} != expected {}",
        cfg.transform.display(),
        cfg.expected_hash
    );
    let transform: AffineTransform = serde_json::from_slice(&t_bytes)?;
    anyhow::ensure!(
        transform.dim_sender == 2048 && transform.dim_receiver == 1536,
        "transform dims {}x{} != 2048x1536",
        transform.dim_sender,
        transform.dim_receiver
    );
    let (golden_n, golden_max_rel, golden_seed) =
        verify_against_golden(&transform, &cfg.golden, &t_sha, GOLDEN_REL_TOL)?;
    println!(
        "transform {} (content_hash {t_sha}): hand-rolled affine apply verified against \
         {golden_n} latentmesh-align golden pairs, max relative L2 error {golden_max_rel:.3e} <= {GOLDEN_REL_TOL:.0e}",
        cfg.transform.display()
    );

    // ---- Dataset: pinned sha + the exact S1a item set ---------------------
    let dir = common::run_dir("s2b");
    let data = dir.join("gsm8k-train.jsonl");
    let train_sha = common::fetch(common::GSM8K_TRAIN_URL, &data)?;
    anyhow::ensure!(
        train_sha == GSM8K_TRAIN_SHA256,
        "gsm8k train.jsonl sha256 {train_sha} != pinned {GSM8K_TRAIN_SHA256}"
    );
    let all_items = common::load_gsm8k(&data)?;
    let mut rng = ChaCha8Rng::seed_from_u64(ITEM_SEED);
    let mut indices = rand::seq::index::sample(&mut rng, all_items.len(), cfg.n_items).into_vec();
    indices.sort_unstable();
    let mut s1a_indices_match = false;
    if cfg.n_items == 40 {
        let s1a: serde_json::Value =
            serde_json::from_slice(&std::fs::read(crate_path(S1A_RECEIPT))?)?;
        let s1a_idx: Vec<usize> = s1a["dataset"]["indices"]
            .as_array()
            .expect("S1a receipt indices")
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        anyhow::ensure!(
            indices == s1a_idx,
            "derived indices differ from the committed S1a receipt's item set"
        );
        s1a_indices_match = true;
        println!("item set: 40 indices identical to the committed S1a receipt");
    }

    let device = latentmesh_runtime::device().map_err(anyhow::Error::msg)?;
    println!("loading {SENDER} + {RECEIVER} (BF16, concurrent-resident)...");
    let mut sender =
        QwenRuntime::load(SENDER, &device, candle_core::DType::BF16).map_err(anyhow::Error::msg)?;
    let mut receiver = QwenRuntime::load(RECEIVER, &device, candle_core::DType::BF16)
        .map_err(anyhow::Error::msg)?;
    let pad_id = receiver
        .tokenizer
        .token_to_id("<|fim_pad|>")
        .ok_or_else(|| anyhow::anyhow!("<|fim_pad|> not in tokenizer"))?;
    println!("models loaded in {:.1}s", t0.elapsed().as_secs_f32());

    let mut rows = Vec::new();
    let mut paired: Vec<Quad> = Vec::new();
    for (done, &idx) in indices.iter().enumerate() {
        let item = &all_items[idx];
        match run_item(
            &mut sender,
            &mut receiver,
            &transform,
            item,
            pad_id,
            &cfg,
            &device,
        )? {
            Some((row, q)) => {
                println!(
                    "[{}/{}] item {idx}: aligned={} baseline={} zerovec={} random={} (nll {:.3}/{:.3}/{:.3}/{:.3}) {:.0}s",
                    done + 1,
                    indices.len(),
                    q.real.0,
                    q.base.0,
                    q.zero.0,
                    q.rand.0,
                    q.real.1,
                    q.base.1,
                    q.zero.1,
                    q.rand.1,
                    t0.elapsed().as_secs_f32()
                );
                rows.push(row);
                paired.push(q);
            }
            None => {
                println!(
                    "[{}/{}] item {idx}: degenerate sender capture pass, skipped",
                    done + 1,
                    indices.len()
                );
                rows.push(
                    serde_json::json!({"item": idx, "skipped": "degenerate sender capture pass"}),
                );
            }
        }
    }

    // ---- Pre-committed analysis ------------------------------------------
    let n = paired.len();
    let count = |f: &dyn Fn(&Quad) -> bool| paired.iter().filter(|q| f(q)).count();
    let real_c = count(&|q| q.real.0);
    let base_c = count(&|q| q.base.0);
    let zero_c = count(&|q| q.zero.0);
    let rand_c = count(&|q| q.rand.0);
    // Primary gate A7(b): aligned_real vs random.
    let wins_rr = count(&|q| q.real.0 && !q.rand.0);
    let loss_rr = count(&|q| !q.real.0 && q.rand.0);
    let p_primary = common::sign_test_one_sided(wins_rr, loss_rr);
    let primary_pass = p_primary < ALPHA;
    // A7(c): zerovec-through-real-path vs uninjected baseline.
    let zb_base_wins = count(&|q| q.base.0 && !q.zero.0);
    let zb_zero_wins = count(&|q| q.zero.0 && !q.base.0);
    let p_base_gt_zero = common::sign_test_one_sided(zb_base_wins, zb_zero_wins);
    let p_zero_gt_base = common::sign_test_one_sided(zb_zero_wins, zb_base_wins);
    let zerovec_pass = 2 * zero_c >= base_c;
    // Reported alongside (as S1a's real_vs_zero was): aligned vs the others.
    let wins_rz = count(&|q| q.real.0 && !q.zero.0);
    let loss_rz = count(&|q| !q.real.0 && q.zero.0);
    let wins_rb = count(&|q| q.real.0 && !q.base.0);
    let loss_rb = count(&|q| !q.real.0 && q.base.0);
    // Secondary NLL diagnostics (lower is better).
    let nll_sign = |a: &dyn Fn(&Quad) -> f32, b: &dyn Fn(&Quad) -> f32| {
        let w = paired.iter().filter(|q| a(q) < b(q)).count();
        let l = paired.iter().filter(|q| a(q) > b(q)).count();
        (w, l, common::sign_test_one_sided(w, l))
    };
    let nll_rr = nll_sign(&|q| q.real.1, &|q| q.rand.1);
    let nll_rz = nll_sign(&|q| q.real.1, &|q| q.zero.1);
    let nll_zb = nll_sign(&|q| q.zero.1, &|q| q.base.1);
    let mean = |f: &dyn Fn(&Quad) -> f32| paired.iter().map(f).sum::<f32>() / n.max(1) as f32;

    let receipt = serde_json::json!({
        "stage": "S2b-bridge-probe",
        "design": "docs/adr/023-live-four-condition-run1-pre-registration.md (A7 coordinator resolution, Deviation 6); protocol = S1a's frozen protocol (docs/research/024 section 7 S1a) with the S2-fitted cross-model transform",
        "env": common::env_info(&nvcc),
        "pre_committed": cfg.pre_committed(),
        "contingency": cfg.dev7_contingency.then_some(
            "ADR-023 Deviation 7: pre-registered generated-pairs recalibration re-probe; \
             transform fitted on sender-GENERATED reasoning pairs (S2c), protocol knobs frozen \
             (same 40 items, same sign test, same alpha, same slots/pooling/rescale/decoding)"),
        "config": {
            "sender": SENDER, "receiver": RECEIVER,
            "sender_capture_block": cfg.sender_block, "receiver_inject_block": cfg.receiver_block,
            "slots": N_SLOTS, "placeholder_token": "<|fim_pad|>", "placeholder_id": pad_id,
            "pool_span": "full generated span",
            "rescale_to_natural_median": true,
            "natural_norm_source": "receiver forward_capture over the slotted injection prompt at the inject block, per item (S0 cross-model precedent, s0-receipt.json config.injection_vector)",
            "decoding": "greedy, batch=1, max_new_tokens=400",
            "item_seed_chacha8": ITEM_SEED, "randvec_seed_base": RANDVEC_SEED_BASE,
            "transform": {
                "file": cfg.transform.display().to_string(),
                "content_hash": t_sha,
                "expected_hash": cfg.expected_hash,
                "dims": [transform.dim_sender, transform.dim_receiver],
                "alpha": transform.alpha,
                "on_train_confidence_OPTIMISTIC_not_the_quality_number": transform.confidence,
                "apply": "hand-rolled affine y = mu_r + alpha*(z - mu_s)*R (latentmesh-align cannot be a path dep: align -> latentmesh-core pins half =2.4.1 vs candle's half ^2.5)",
                "hand_rolled_apply_verification": {
                    "golden_file": cfg.golden.display().to_string(),
                    "golden_pairs": golden_n,
                    "golden_input_seed_chacha8": golden_seed,
                    "max_relative_l2_error": golden_max_rel,
                    "tolerance": GOLDEN_REL_TOL,
                    "pass": true,
                    "note": "golden outputs produced by latentmesh-align::AlignmentTransform::apply on the same artifact; asserted before any model ran",
                },
            },
            "conditions": {
                "aligned_real": "sender pooled capture -> frozen affine transform -> 8-slot injection, rescaled to natural median",
                "random": "per-item seeded gaussian, norm-matched to the effective aligned_real vector, real path",
                "zerovec_injected": "TRUE ZERO VECTOR through the real 8-slot injection path (scale: None — rescale undefined at norm 0); closes ADR-023 A7(c): distinct from S0 G5's empty-position branch and from S1a's uninjected 'zero'",
                "baseline_uninjected": "no injection (spec=None), same prompt",
            },
            "primary_test": "one-sided exact sign test, paired accuracy, aligned_real > random, alpha 0.05 (A7(b))",
            "zerovec_gate": "pre-committed: pass iff 2 x zerovec accuracy >= baseline accuracy ('not catastrophically below baseline'); numbers reported either way (A7(c))",
            "secondary_diagnostic": "one-sided sign tests on paired teacher-forced NLL of '#### <gold>'",
        },
        "dataset": {"source": common::GSM8K_TRAIN_URL, "sha256": train_sha, "indices": indices,
                     "s1a_indices_match": s1a_indices_match},
        "items": rows,
        "summary": {
            "n_evaluated": n,
            "accuracy": {"aligned_real": real_c, "baseline_uninjected": base_c,
                          "zerovec_injected": zero_c, "random": rand_c},
            "primary_aligned_vs_random": {"wins": wins_rr, "losses": loss_rr,
                "p_one_sided": p_primary, "alpha": ALPHA, "pass": primary_pass},
            "zerovec_vs_baseline": {
                "baseline_wins": zb_base_wins, "zerovec_wins": zb_zero_wins,
                "p_baseline_gt_zerovec": p_base_gt_zero, "p_zerovec_gt_baseline": p_zero_gt_base,
                "criterion": "2 x zerovec_correct >= baseline_correct",
                "pass": zerovec_pass},
            "aligned_vs_zerovec": {"wins": wins_rz, "losses": loss_rz,
                "p_one_sided": common::sign_test_one_sided(wins_rz, loss_rz)},
            "aligned_vs_baseline": {"wins": wins_rb, "losses": loss_rb,
                "p_one_sided": common::sign_test_one_sided(wins_rb, loss_rb)},
            "nll_mean": {"aligned_real": mean(&|q| q.real.1), "baseline_uninjected": mean(&|q| q.base.1),
                          "zerovec_injected": mean(&|q| q.zero.1), "random": mean(&|q| q.rand.1)},
            "nll_aligned_vs_random": {"wins": nll_rr.0, "losses": nll_rr.1, "p_one_sided": nll_rr.2},
            "nll_aligned_vs_zerovec": {"wins": nll_rz.0, "losses": nll_rz.1, "p_one_sided": nll_rz.2},
            "nll_zerovec_vs_baseline": {"wins": nll_zb.0, "losses": nll_zb.1, "p_one_sided": nll_zb.2},
        },
        "gates": {
            "transform_hash_matches_registered": {"pass": true, "hash": t_sha},
            "hand_rolled_apply_matches_align_crate": {"pass": true, "max_relative_l2_error": golden_max_rel},
            "s1a_item_set_reproduced": {"pass": s1a_indices_match},
            "A7b_aligned_real_vs_random": {"pass": primary_pass, "p": p_primary},
            "A7c_zerovec_not_catastrophic": {"pass": zerovec_pass,
                "zerovec_correct": zero_c, "baseline_correct": base_c},
        },
        "gate_pass": primary_pass && zerovec_pass,
        "wall_clock_s": t0.elapsed().as_secs_f32(),
    });
    let name = format!("s2b-receipt-{}.json", cfg.tag());
    common::write_receipt(&dir, &name, &receipt)?;
    println!(
        "S2b: acc aligned {real_c}/{n} baseline {base_c}/{n} zerovec {zero_c}/{n} random {rand_c}/{n}"
    );
    println!(
        "A7(b) primary p={p_primary:.4} (wins {wins_rr}, losses {loss_rr}) => pass={primary_pass}; \
         A7(c) zerovec {zero_c} vs baseline {base_c} => pass={zerovec_pass}"
    );
    println!(
        "secondary NLL aligned-vs-random p={:.4} ({}W/{}L)",
        nll_rr.2, nll_rr.0, nll_rr.1
    );
    Ok(())
}

/// Per-condition (correct, nll_gold) for aligned_real / baseline_uninjected /
/// zerovec_injected / random.
struct Quad {
    real: (bool, f32),
    base: (bool, f32),
    zero: (bool, f32),
    rand: (bool, f32),
}

type QuadRow = (serde_json::Value, Quad);

#[allow(clippy::too_many_arguments)]
fn run_item(
    sender: &mut QwenRuntime,
    receiver: &mut QwenRuntime,
    transform: &AffineTransform,
    item: &common::Gsm8kItem,
    pad_id: u32,
    cfg: &ProbeConfig,
    device: &candle_core::Device,
) -> anyhow::Result<Option<QuadRow>> {
    let e = anyhow::Error::msg;
    let fmt = common::ANSWER_FORMAT;
    let mut greedy = Sampler::new(Sampling::Greedy, 0);

    // 1) Sender capture pass: solve, then teacher-forced re-prefill with the
    //    tap after the sender sweep block, pooled over the generated span.
    let cap_prompt = QwenRuntime::chat_prompt(SYSTEM, &format!("{}\n\n{fmt}", item.question));
    let cap_tokens = sender.encode(&cap_prompt).map_err(e)?;
    let gen = sender
        .generate(&cap_tokens, None, &mut greedy, MAX_NEW_TOKENS, false)
        .map_err(e)?;
    if gen.tokens.is_empty() {
        return Ok(None);
    }
    let full: Vec<u32> = cap_tokens
        .iter()
        .chain(gen.tokens.iter())
        .copied()
        .collect();
    let (_, cap) = forward_capture(
        &mut sender.model,
        &full,
        cfg.sender_block,
        cap_tokens.len()..full.len(),
        device,
    )
    .map_err(e)?;
    anyhow::ensure!(
        cap.hidden_size == transform.dim_sender,
        "sender capture dim {} != transform dim_sender {}",
        cap.hidden_size,
        transform.dim_sender
    );
    let first_pass_answer = common::extract_answer(&gen.text);
    let first_pass_correct = first_pass_answer
        .as_deref()
        .is_some_and(|a| common::answers_equal(a, &item.gold));

    // 2) Frozen affine transform: sender 2048 -> receiver 1536.
    let aligned = transform.apply(&cap.pooled);

    // 3) Receiver injection prompt with placeholder slots (S1a wording).
    let slots = "<|fim_pad|>".repeat(N_SLOTS);
    let inj_prompt = QwenRuntime::chat_prompt(
        SYSTEM,
        &format!(
            "{}\n\nA compressed latent hint from a previous solver is stored in these slots: [{slots}]\n\n{fmt}",
            item.question
        ),
    );
    let inj_tokens = receiver.encode(&inj_prompt).map_err(e)?;
    let positions = QwenRuntime::placeholder_positions(&inj_tokens, pad_id);
    anyhow::ensure!(positions.len() == N_SLOTS, "slot count mismatch");

    // 4) Natural inject-block norms of the receiver over the slotted prompt
    //    (S0 cross-model precedent) — the rescale target.
    let (_, nat_cap) = forward_capture(
        &mut receiver.model,
        &inj_tokens,
        cfg.receiver_block,
        0..inj_tokens.len(),
        device,
    )
    .map_err(e)?;
    let natural = norms::stats(nat_cap.per_position_l2.clone());

    let aligned_l2 = norms::l2(&aligned);
    let real = InjectionSpec {
        after_block: cfg.receiver_block,
        positions: positions.clone(),
        vector: aligned.clone(),
        scale: Some(natural.median / aligned_l2),
    };
    let target_l2 = norms::l2(&real.effective_vector());
    // Random control: per-item seeded gaussian, norm-matched (as S1a).
    let mut vrng = ChaCha8Rng::seed_from_u64(RANDVEC_SEED_BASE + item.index as u64);
    let gauss = gaussian_vec(&mut vrng, aligned.len());
    let random = InjectionSpec {
        after_block: cfg.receiver_block,
        positions: positions.clone(),
        vector: gauss.clone(),
        scale: Some(target_l2 / norms::l2(&gauss)),
    };
    // A7(c) control: TRUE zero vector through the real 8-slot path.
    let zerovec = InjectionSpec {
        after_block: cfg.receiver_block,
        positions,
        vector: vec![0f32; aligned.len()],
        scale: None,
    };

    // 5) Paired conditions.
    let mut outcome = |spec: Option<&InjectionSpec>| -> anyhow::Result<(bool, f32, String)> {
        let mut s = Sampler::new(Sampling::Greedy, 0);
        let out = receiver
            .generate(&inj_tokens, spec, &mut s, MAX_NEW_TOKENS, false)
            .map_err(e)?;
        let correct = common::extract_answer(&out.text)
            .is_some_and(|a| common::answers_equal(&a, &item.gold));
        let answer_toks = receiver.encode(&format!("#### {}", item.gold)).map_err(e)?;
        let nll_tokens: Vec<u32> = inj_tokens
            .iter()
            .chain(answer_toks.iter())
            .copied()
            .collect();
        let nll = teacher_forced_nll(
            &mut receiver.model,
            &nll_tokens,
            inj_tokens.len()..nll_tokens.len(),
            spec,
            device,
        )
        .map_err(e)?;
        Ok((correct, nll, out.text))
    };
    let (real_ok, real_nll, real_text) = outcome(Some(&real))?;
    let (base_ok, base_nll, _) = outcome(None)?;
    let (zero_ok, zero_nll, _) = outcome(Some(&zerovec))?;
    let (rand_ok, rand_nll, _) = outcome(Some(&random))?;

    let row = serde_json::json!({
        "item": item.index,
        "gold": item.gold,
        "sender_first_pass": {"correct": first_pass_correct, "answer": first_pass_answer,
                               "generated_tokens": gen.tokens.len()},
        "capture": {
            "hidden_size": cap.hidden_size,
            "pooled_l2_raw": norms::l2(&cap.pooled),
            "aligned_l2_raw": aligned_l2,
            "injected_l2": target_l2,
            "natural_inject_block_norms": natural,
            "span": [cap.span.start, cap.span.end],
        },
        "conditions": {
            "aligned_real": {"correct": real_ok, "nll_gold": real_nll},
            "baseline_uninjected": {"correct": base_ok, "nll_gold": base_nll},
            "zerovec_injected": {"correct": zero_ok, "nll_gold": zero_nll},
            "random": {"correct": rand_ok, "nll_gold": rand_nll},
        },
        "aligned_answer_tail": real_text.chars().rev().take(60).collect::<String>().chars().rev().collect::<String>(),
    });
    Ok(Some((
        row,
        Quad {
            real: (real_ok, real_nll),
            base: (base_ok, base_nll),
            zero: (zero_ok, zero_nll),
            rand: (rand_ok, rand_nll),
        },
    )))
}

/// Standard-normal vector via Box-Muller over a seeded ChaCha8 stream
/// (identical to S1a's generator).
fn gaussian_vec(rng: &mut ChaCha8Rng, n: usize) -> Vec<f32> {
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
