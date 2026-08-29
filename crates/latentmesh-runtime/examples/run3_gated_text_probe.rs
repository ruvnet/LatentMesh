//! Run 3 — causally-gated TEXT communication (ADR-030, stage A).
//!
//! ADR-030 tests whether ADR-003's five-control causal gate has standalone
//! value on the channel agents actually use today — text. This example is the
//! `StaticText` / `StaticText+Gate` stage: the sender solves each item, its
//! generated solution is handed to the receiver **as text, in-context** (no
//! adapter, no transform, no injection anywhere in this file), and the
//! receiver's accuracy under that message is compared against ADR-003's four
//! applicable controls.
//!
//! ADR-030's own acceptance section states the primary test is scored against
//! "`CausalDynamicText`'s frozen-genome evaluation, **or** `StaticText+Gate`'s
//! fixed-pipeline evaluation — both are scored against this same test". This
//! stage is the second of those two paths, so it reaches the frozen primary
//! endpoint without a Darwin loop. The `CausalDynamicText` Darwin loop, its
//! genome freeze, and the `eval-200` unlock that freeze would trigger are
//! deliberately NOT in this file (see `not_run_in_this_stage` in the receipt).
//!
//! ## Controls (ADR-003's five, minus the one ADR-030 drops with reasons)
//!
//! `text_equivalent` is degenerate when the channel under test is already text
//! and ADR-030 drops it explicitly. The four retained, verbatim from ADR-030:
//! - `zero` — no message reaches the receiver (bare question prompt)
//! - `random` — a random token sequence, length-matched, not the sender's real
//!   content
//! - `mismatched` — another episode's real message, content-shaped but wrong
//!   task
//! - `self_generated` — the receiver's own prior output fed back to itself
//!
//! ## PRE-COMMITTED ANALYSIS (frozen in this source before any item is drawn)
//!
//! Primary: ADR-030 §3.2's one-sided Bernoulli e-process, `W_0 = 1`,
//! `W_i = W_{i-1}·(1 + λ(X_i − 0.5))` on discordant items only, `λ = 0.30`,
//! PASS at `W ≥ 1/α = 20`, budget `N_max = 300`. ADR-030 requires the
//! comparison be against "the best (most favorable to the null) of the four
//! controls". A sequential test cannot know post-hoc which control is best, so
//! that requirement is implemented as an **intersection–union test**: FOUR
//! wealth processes run in parallel, one per control, and the registered PASS
//! is `min_k W_k ≥ 20`. That is exactly "beats the best control", it is valid
//! at α without correction (a PASS requires every component null to be
//! rejected), and it is conservative in the same direction ADR-030 §5 already
//! discloses its post-hoc best-control selection to be. The control whose
//! wealth ends lowest is reported as the realised "best control".
//!
//! Reported alongside, never in place of (ADR-030): one-sided mid-p McNemar
//! and the classic exact sign test on the accumulated pairs, per control; the
//! raw accuracy delta with its exact binomial interval; GPU-seconds per
//! condition. Gold-token NLL is NOT computed: ADR-030 explicitly rejects it as
//! a co-primary or replacement statistic, so spending decode on it would be
//! spending it on a statistic the pre-registration already ruled out.
//!
//! ## Item supply — ADR-030's stated rule is superseded by ADR-036's correction
//!
//! ADR-030 §"Sampler, dataset and split discipline" says the frozen 40-item
//! probe is "a subset of `adaptation-512` by construction" and should be the
//! e-process's starting population. **That premise is false and this repo has
//! already corrected it**: ADR-036 Decision 2 verified by direct set
//! intersection that the probe meets `adaptation-512` in exactly ONE item
//! (index 1153), and ruled that a successor e-process draws its stream
//! entirely from `adaptation-512` in the file's own ascending index order.
//! This stage follows ADR-036's corrected rule, not ADR-030's erroneous one —
//! disclosed here and in the receipt rather than silently resolved. It has the
//! side benefit that this stage's `zero` control runs on the *identical* item
//! population as `run2-m4i`'s `baseline_uninjected` (140/300), making that a
//! like-for-like comparator rather than an approximate one.
//!
//! The 13-item leakage-exclusion list is applied for protocol identity, and
//! its effect is MEASURED here rather than assumed (it is empty on this split).
//! Nothing in this stage is trained, so the exclusion has no mechanism to bite;
//! it is kept because ADR-036 keeps it, not because it does work here.
//!
//! Receipts: `crates/latentmesh-runtime/target/latentmesh-runs/run3/`.
//! Run: PATH=/usr/local/cuda-12.8/bin:$PATH \
//!      cargo run --release --features cuda --example run3_gated_text_probe

#[path = "common/mod.rs"]
#[allow(dead_code)]
mod common;

use latentmesh_runtime::{
    sampler::{Sampler, Sampling},
    QwenRuntime,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::path::{Path, PathBuf};

const SENDER: &str = "Qwen/Qwen2.5-3B-Instruct";
const RECEIVER: &str = "Qwen/Qwen2.5-1.5B-Instruct";
const SYSTEM: &str = "You are a careful math tutor.";
/// The S1a probe budget, unchanged through every rung of the ladder.
const MAX_NEW_TOKENS: usize = 400;
/// GSM8K train.jsonl pin (ADR-023 dataset table, verbatim from the S1a receipt).
const GSM8K_TRAIN_SHA256: &str = "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465";
const ADAPTATION_512: &str = "../../harness/latentmesh-live/data/adaptation-512.json";
/// The m4i receipt whose `baseline_uninjected` count is this stage's
/// like-for-like `zero`-control comparator (same population, same order).
const M4I_RECEIPT: &str =
    "receipts/run2-m4i-receipt-cellL18toL14-mlp-pertokenlast-fuse-questiontail-slots8-eprocess.json";

// ---- ADR-030 §3.2 e-process parameters, frozen ---------------------------
/// Betting fraction, `λ = 2θ−1` tuned to the smallest interesting effect
/// θ=0.65. Fixed in advance; never re-parametrised after seeing `W_t`.
const LAMBDA: f64 = 0.30;
/// α for the wealth boundary. PASS at `W_i ≥ 1/α`.
const E_ALPHA: f64 = 0.05;
/// The registered budget.
const N_MAX: usize = 300;

/// ADR-024's 13 probe-overlap rows, kept in force by ADR-036 Decision 2(2).
const LEAKAGE_EXCLUSIONS: [usize; 13] = [
    150, 1309, 1573, 2365, 2418, 2540, 2958, 2973, 3084, 3825, 3844, 3877, 4746,
];

/// Per-item seed base for the `random` control's token draw.
const RANDTOK_SEED_BASE: u64 = 0x2030_0000;

/// ADR-030's registered Darwin-loop seed, `latentmesh-run3-audit-run-seed`.
/// Recorded in this stage's receipt for provenance and **not used here** —
/// this stage runs no Darwin loop.
const DARWIN_SEED_UNUSED_IN_THIS_STAGE: u64 = 11_796_393_239_420_137_246;

/// The four controls, in their registered order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Control {
    Zero,
    Random,
    Mismatched,
    SelfGenerated,
}

impl Control {
    const ALL: [Control; 4] = [
        Control::Zero,
        Control::Random,
        Control::Mismatched,
        Control::SelfGenerated,
    ];
    fn tag(self) -> &'static str {
        match self {
            Control::Zero => "zero",
            Control::Random => "random",
            Control::Mismatched => "mismatched",
            Control::SelfGenerated => "self_generated",
        }
    }
}

/// One item's outcome: the gated-text condition plus the four controls.
struct Row {
    text_correct: bool,
    control_correct: [bool; 4],
}

/// One step of a single control's registered wealth process.
struct EStep {
    order: usize,
    item: usize,
    text_correct: bool,
    control_correct: bool,
    discordant: bool,
    x: Option<u8>,
    wealth: f64,
}

fn crate_path(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

/// The receiver prompt for a given message. `None` is the `zero` control:
/// ADR-003 defines it as "no message reaches the receiver", so it is the bare
/// question prompt and is structurally different from the message-bearing
/// conditions by that definition. Disclosed, not smoothed over.
fn receiver_prompt(question: &str, message: Option<&str>) -> String {
    let fmt = common::ANSWER_FORMAT;
    match message {
        None => QwenRuntime::chat_prompt(SYSTEM, &format!("{question}\n\n{fmt}")),
        Some(m) => QwenRuntime::chat_prompt(
            SYSTEM,
            &format!("{question}\n\nA message from a previous solver:\n{m}\n\n{fmt}"),
        ),
    }
}

/// Exact binomial (Clopper–Pearson) interval for `k/n`, via the incomplete
/// beta relation evaluated by bisection on the binomial tail.
fn clopper_pearson(k: usize, n: usize, alpha: f64) -> (f64, f64) {
    if n == 0 {
        return (0.0, 1.0);
    }
    // P(X >= k | p) increases in p; P(X <= k | p) decreases in p.
    let tail_ge = |p: f64| -> f64 { (k..=n).map(|i| binom_pmf_p(n, i, p)).sum() };
    let tail_le = |p: f64| -> f64 { (0..=k).map(|i| binom_pmf_p(n, i, p)).sum() };
    let bisect = |f: &dyn Fn(f64) -> f64, target: f64, increasing: bool| -> f64 {
        let (mut lo, mut hi) = (0.0f64, 1.0f64);
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if (f(mid) < target) == increasing {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    };
    let lower = if k == 0 {
        0.0
    } else {
        bisect(&tail_ge, alpha / 2.0, true)
    };
    let upper = if k == n {
        1.0
    } else {
        bisect(&tail_le, alpha / 2.0, false)
    };
    (lower, upper)
}

fn binom_pmf_p(n: usize, k: usize, p: f64) -> f64 {
    if p <= 0.0 {
        return if k == 0 { 1.0 } else { 0.0 };
    }
    if p >= 1.0 {
        return if k == n { 1.0 } else { 0.0 };
    }
    let ln_c: f64 = (1..=k).map(|i| ((n - k + i) as f64 / i as f64).ln()).sum();
    (ln_c + k as f64 * p.ln() + (n - k) as f64 * (1.0 - p).ln()).exp()
}

/// `--smoke N` runs a mechanics check on N items taken from **outside** the
/// registered e-process stream, so it cannot peek at the one and only draw.
/// It writes a separately-named receipt marked `pre_committed: false`.
fn parse_smoke() -> Option<usize> {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == "--smoke")
        .map(|i| args[i + 1].parse().expect("--smoke N"))
}

fn main() -> anyhow::Result<()> {
    let t0 = std::time::Instant::now();
    let smoke = parse_smoke();
    let nvcc = latentmesh_runtime::assert_cuda_build_env().map_err(anyhow::Error::msg)?;
    println!("build-env guard: {nvcc}");
    let w_threshold = 1.0 / E_ALPHA;

    // ---- Dataset: pinned sha, ADR-036-corrected item supply ---------------
    let dir = common::run_dir("run3");
    let data = dir.join("gsm8k-train.jsonl");
    let train_sha = common::fetch(common::GSM8K_TRAIN_URL, &data)?;
    anyhow::ensure!(
        train_sha == GSM8K_TRAIN_SHA256,
        "gsm8k train.jsonl sha256 {train_sha} != pinned {GSM8K_TRAIN_SHA256}"
    );
    let all_items = common::load_gsm8k(&data)?;

    let split: serde_json::Value =
        serde_json::from_slice(&std::fs::read(crate_path(ADAPTATION_512))?)?;
    let adaptation_indices: Vec<usize> = split["indices"]
        .as_array()
        .expect("adaptation-512 indices")
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    anyhow::ensure!(adaptation_indices.len() == 512);
    anyhow::ensure!(
        adaptation_indices.windows(2).all(|w| w[0] < w[1]),
        "adaptation-512 is not in ascending index order — the registered consumption order is the \
         file's own fixed index order"
    );
    // Measured, not assumed: ADR-036 keeps the 13-item rule in force, and its
    // effect on THIS split is computed here.
    let excluded_present: Vec<usize> = LEAKAGE_EXCLUSIONS
        .iter()
        .copied()
        .filter(|i| adaptation_indices.contains(i))
        .collect();
    let eligible: Vec<usize> = adaptation_indices
        .iter()
        .copied()
        .filter(|i| !LEAKAGE_EXCLUSIONS.contains(i))
        .collect();
    anyhow::ensure!(
        eligible.len() > N_MAX,
        "eligible pool ({}) must exceed N_max ({N_MAX}) so the mismatched-control priming item \
         sits outside the drawn stream",
        eligible.len()
    );
    // The registered stream is the first N_max eligible items. A `--smoke`
    // mechanics check draws instead from the tail of the eligible pool, which
    // is disjoint from that stream by construction — so no smoke run can
    // observe an outcome on an item the real draw will later consume.
    let stream: Vec<usize> = match smoke {
        None => eligible[..N_MAX].to_vec(),
        Some(k) => {
            let tail = &eligible[N_MAX..eligible.len() - 1];
            anyhow::ensure!(
                k <= tail.len(),
                "--smoke {k} exceeds the {} items available outside the registered stream",
                tail.len()
            );
            println!(
                "SMOKE MECHANICS CHECK: {k} items from OUTSIDE the registered e-process stream. \
                 This is not a draw and its receipt is marked pre_committed: false."
            );
            tail[..k].to_vec()
        }
    };
    // The `mismatched` control at stream order 0 needs a real message from
    // another episode with NO lookahead into the stream (a lookahead would
    // make early stopping dishonest). The registered priming item is the LAST
    // eligible index, which is outside the stream by the assertion above.
    let priming_item = *eligible.last().unwrap();
    anyhow::ensure!(
        !stream.contains(&priming_item),
        "priming item {priming_item} is inside the drawn stream"
    );
    println!(
        "item supply (ADR-036 Decision 2, correcting ADR-030's false 'subset by construction' \
         premise): adaptation-512 in fixed index order; leakage exclusion present on this split: \
         {excluded_present:?}; {} eligible, stream = first {N_MAX}; mismatched-priming item = \
         {priming_item} (outside the stream)",
        eligible.len()
    );

    // The like-for-like `zero` comparator, read from the committed receipt.
    let m4i: serde_json::Value = serde_json::from_slice(&std::fs::read(crate_path(M4I_RECEIPT))?)?;
    let m4i_baseline = m4i["summary"]["accuracy"]["baseline_uninjected"]
        .as_u64()
        .expect("m4i summary.accuracy.baseline_uninjected");
    let m4i_n = m4i["summary"]["n_evaluated"]
        .as_u64()
        .expect("m4i summary.n_evaluated");
    println!(
        "zero-control comparator: run2-m4i baseline_uninjected {m4i_baseline}/{m4i_n} on this \
         identical population"
    );

    let device = latentmesh_runtime::device().map_err(anyhow::Error::msg)?;
    println!("loading {SENDER} + {RECEIVER} (BF16, concurrent-resident)...");
    let mut sender =
        QwenRuntime::load(SENDER, &device, candle_core::DType::BF16).map_err(anyhow::Error::msg)?;
    let mut receiver = QwenRuntime::load(RECEIVER, &device, candle_core::DType::BF16)
        .map_err(anyhow::Error::msg)?;
    println!("models loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // The `random` control draws uniformly from the receiver's ORDINARY token
    // ids — every id below the first added/special token. Resolved from the
    // tokenizer rather than hard-coded, and recorded.
    let first_special = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|fim_pad|>"]
        .iter()
        .filter_map(|t| receiver.tokenizer.token_to_id(t))
        .min()
        .ok_or_else(|| anyhow::anyhow!("no known special token in the receiver tokenizer"))?;
    println!(
        "random-control vocabulary: ordinary ids [0, {first_special}) (first special id is \
         {first_special})"
    );

    // ---- The mismatched control's priming message -------------------------
    let mut prev_message = sender_message(&mut sender, &all_items[priming_item])?
        .ok_or_else(|| anyhow::anyhow!("priming item {priming_item} produced no sender message"))?;
    let mut prev_message_from = priming_item;

    // ---- THE DRAW ---------------------------------------------------------
    let mut wealth = [1.0f64; 4];
    let mut max_wealth = [1.0f64; 4];
    let mut wins = [0usize; 4];
    let mut losses = [0usize; 4];
    let mut trajectory: [Vec<EStep>; 4] = std::array::from_fn(|_| Vec::new());
    let mut rows: Vec<serde_json::Value> = Vec::new();
    let mut paired: Vec<Row> = Vec::new();
    let mut crossed_at: Option<usize> = None;
    let mut items_drawn = 0usize;
    let mut degenerate = 0usize;
    let mut sender_first_pass_correct = 0usize;
    let mut cond_seconds = [0f64; 5]; // [text, zero, random, mismatched, self_generated]
    let mut sender_seconds = 0f64;

    for (order, &idx) in stream.iter().enumerate() {
        let item = &all_items[idx];
        items_drawn = order + 1;

        // 1) Sender pass. The message IS the sender's own generated solution
        //    text — ADR-023 defines StaticText as a "fixed hand-designed
        //    2-agent pipeline, text channel", and this is that pipeline's
        //    message with nothing added between the two agents.
        let ts = std::time::Instant::now();
        let message = sender_message(&mut sender, item)?;
        sender_seconds += ts.elapsed().as_secs_f64();
        let Some(message) = message else {
            // Registered identically to every prior rung: a degenerate sender
            // pass yields no pair and therefore no wealth update, but it still
            // CONSUMES one of the N_max budget items.
            degenerate += 1;
            println!(
                "[{}/{N_MAX}] item {idx}: degenerate sender pass, no pair (W unchanged); budget \
                 item consumed",
                order + 1
            );
            for k in 0..Control::ALL.len() {
                trajectory[k].push(EStep {
                    order: order + 1,
                    item: idx,
                    text_correct: false,
                    control_correct: false,
                    discordant: false,
                    x: None,
                    wealth: wealth[k],
                });
            }
            rows.push(serde_json::json!({"item": idx, "skipped": "degenerate sender pass"}));
            continue;
        };
        if common::extract_answer(&message).is_some_and(|a| common::answers_equal(&a, &item.gold)) {
            sender_first_pass_correct += 1;
        }
        let message_tokens = receiver.encode(&message).map_err(anyhow::Error::msg)?.len();

        // 2) Receiver, five conditions. Greedy throughout: ADR-030 grants the
        //    deterministic exception for `self_generated` (which needs the
        //    receiver's own prior deterministic output to be well-defined),
        //    and greedy is the witness arm every prior probe drew under.
        let generate = |receiver: &mut QwenRuntime,
                        msg: Option<&str>|
         -> anyhow::Result<(bool, String, f64)> {
            let t = std::time::Instant::now();
            let prompt = receiver_prompt(&item.question, msg);
            let toks = receiver.encode(&prompt).map_err(anyhow::Error::msg)?;
            let mut s = Sampler::new(Sampling::Greedy, 0);
            let out = receiver
                .generate(&toks, None, &mut s, MAX_NEW_TOKENS, false)
                .map_err(anyhow::Error::msg)?;
            let correct = common::extract_answer(&out.text)
                .is_some_and(|a| common::answers_equal(&a, &item.gold));
            Ok((correct, out.text, t.elapsed().as_secs_f64()))
        };

        let (text_ok, _text_out, dt) = generate(&mut receiver, Some(&message))?;
        cond_seconds[0] += dt;
        // `zero`: no message. Its output is also the `self_generated` payload,
        // so that control costs one extra generation, not two.
        let (zero_ok, zero_out, dt) = generate(&mut receiver, None)?;
        cond_seconds[1] += dt;
        // `random`: length-matched (in the receiver's own tokens) uniform draw
        // from the ordinary vocabulary, seeded per item.
        let mut rng = ChaCha8Rng::seed_from_u64(RANDTOK_SEED_BASE + idx as u64);
        let rand_ids: Vec<u32> = (0..message_tokens)
            .map(|_| rng.gen_range(0..first_special))
            .collect();
        let rand_msg = receiver.decode(&rand_ids).map_err(anyhow::Error::msg)?;
        let (rand_ok, _, dt) = generate(&mut receiver, Some(&rand_msg))?;
        cond_seconds[2] += dt;
        // `mismatched`: the previous stream item's real message.
        let (mis_ok, _, dt) = generate(&mut receiver, Some(&prev_message))?;
        cond_seconds[3] += dt;
        // `self_generated`: the receiver's own prior (zero-condition) output.
        let (self_ok, _, dt) = generate(&mut receiver, Some(&zero_out))?;
        cond_seconds[4] += dt;

        let control_correct = [zero_ok, rand_ok, mis_ok, self_ok];

        // 3) Four parallel wealth processes, one per control.
        for k in 0..4 {
            let discordant = text_ok != control_correct[k];
            let x = discordant.then(|| u8::from(text_ok));
            if discordant {
                if text_ok {
                    wins[k] += 1;
                } else {
                    losses[k] += 1;
                }
                wealth[k] *= 1.0 + LAMBDA * (f64::from(x.unwrap()) - 0.5);
                max_wealth[k] = max_wealth[k].max(wealth[k]);
            }
            trajectory[k].push(EStep {
                order: order + 1,
                item: idx,
                text_correct: text_ok,
                control_correct: control_correct[k],
                discordant,
                x,
                wealth: wealth[k],
            });
        }
        let w_min = wealth.iter().copied().fold(f64::INFINITY, f64::min);
        println!(
            "[{}/{N_MAX}] item {idx}: text={text_ok} zero={zero_ok} random={rand_ok} \
             mismatched={mis_ok} self={self_ok} | W=[{:.3} {:.3} {:.3} {:.3}] min={w_min:.3} {:.0}s",
            order + 1,
            wealth[0],
            wealth[1],
            wealth[2],
            wealth[3],
            t0.elapsed().as_secs_f32()
        );

        rows.push(serde_json::json!({
            "item": idx,
            "gold": item.gold,
            "sender_message_tokens": message_tokens,
            "mismatched_message_from_item": prev_message_from,
            "conditions": {
                "gated_text": text_ok,
                "zero": zero_ok,
                "random": rand_ok,
                "mismatched": mis_ok,
                "self_generated": self_ok,
            },
        }));
        paired.push(Row {
            text_correct: text_ok,
            control_correct,
        });

        prev_message = message;
        prev_message_from = idx;

        if w_min >= w_threshold {
            crossed_at = Some(order + 1);
            println!(
                "e-process CROSSED: min_k W_k = {w_min:.4} >= {w_threshold} at item {} of the \
                 stream — stopping, per the registered rule",
                order + 1
            );
            break;
        }
    }

    // ---- Registered outcome + the statistics reported alongside -----------
    let n = paired.len();
    let text_c = paired.iter().filter(|r| r.text_correct).count();
    let mut per_control = Vec::new();
    for (k, c) in Control::ALL.iter().enumerate() {
        let acc = paired.iter().filter(|r| r.control_correct[k]).count();
        let n_disc = wins[k] + losses[k];
        let exact = common::sign_test_one_sided(wins[k], losses[k]);
        let midp = common::mid_p_one_sided(wins[k], losses[k]);
        // Min attainable values AT THIS observed discordant count — the
        // structural-capability fact ADR-030 requires be stated, computed from
        // this draw's own n_disc rather than quoted from a prior rung.
        let min_exact = common::sign_test_one_sided(n_disc, 0);
        let min_midp = common::mid_p_one_sided(n_disc, 0);
        per_control.push(serde_json::json!({
            "control": c.tag(),
            "correct": acc,
            "accuracy": acc as f64 / n.max(1) as f64,
            "e_process": {
                "final_wealth": wealth[k],
                "max_wealth_reached": max_wealth[k],
                "crossed": wealth[k] >= w_threshold,
                "n_discordant": n_disc,
                "wins_text": wins[k],
                "losses_text": losses[k],
            },
            "mid_p_mcnemar_one_sided": midp,
            "exact_sign_one_sided": exact,
            "min_attainable_mid_p_at_this_n_disc": min_midp,
            "min_attainable_exact_p_at_this_n_disc": min_exact,
            "structurally_capable_at_alpha_0_05": min_midp < E_ALPHA,
        }));
    }
    // The realised "best control" = the one most favorable to the null, i.e.
    // the lowest wealth (equivalently the hardest bar the text condition had
    // to clear). Reported, not chosen before the draw.
    let best_k = (0..4)
        .min_by(|&a, &b| wealth[a].partial_cmp(&wealth[b]).unwrap())
        .unwrap();
    let w_min_final = wealth[best_k];
    let e_pass = crossed_at.is_some();

    // Uplift vs the best control, with exact binomial intervals on both arms.
    let best_c = paired.iter().filter(|r| r.control_correct[best_k]).count();
    let (t_lo, t_hi) = clopper_pearson(text_c, n, 0.05);
    let (c_lo, c_hi) = clopper_pearson(best_c, n, 0.05);

    let receipt = serde_json::json!({
        "stage": if smoke.is_some() { "run3-stageA-SMOKE-mechanics-check" } else { "run3-stageA-static-text-gate" },
        "adr": "docs/adr/030-run3-causally-gated-text-pre-registration.md",
        "design": "ADR-030 StaticText / StaticText+Gate. ADR-003's four applicable controls on a TEXT channel; text_equivalent dropped per ADR-030's stated degeneracy argument.",
        "env": common::env_info(&nvcc),
        "pre_committed": smoke.is_none(),
        "smoke_mechanics_check": smoke.map(|k| serde_json::json!({
            "items": k,
            "population": "drawn from the TAIL of the eligible pool, disjoint from the registered N_max stream by construction, so this check cannot observe an outcome on an item the real draw will later consume",
            "not_a_draw": "this receipt is a mechanics check only; it carries no statistical claim and does not consume the one-shot e-process draw.",
        })),
        "no_latent_path_involved": "no adapter, no alignment transform, no injection, no capture. The channel is text consumed in the receiver's own prompt.",
        "deleted_artifacts_ledger": [{
            "filename": "target/latentmesh-runs/run3/run3-stageA-receipt-statictext-gate-eprocess.json",
            "written_at": "the first --smoke 4 mechanics check, 2026-08-29, before the registered draw",
            "why_it_was_written": "a bug in the smoke path: the receipt FILENAME was not switched for smoke runs, so a 4-item mechanics check wrote itself to the real draw's receipt path. Its stage / pre_committed / smoke_mechanics_check fields were correct and identified it as a smoke run; only the filename was wrong.",
            "what_it_contained": "4 items drawn from the TAIL of the eligible pool (indices 4485, 4491, 4496, 4515), all OUTSIDE the registered N_max stream. No registered-stream data of any kind. No statistical claim.",
            "disposition": "DELETED before the registered draw was launched, and the filename bug fixed so smoke runs now write run3-stageA-SMOKE-mechanics-check.json.",
            "why_this_is_recorded": "a deleted artifact should never have to be reconstructed from memory, even a harmless one. Recorded so no future reader has to diff against HEAD to establish that nothing scientific was lost.",
        }],
        "not_run_in_this_stage": {
            "condition": "CausalDynamicText",
            "reason": "ADR-030's primary test is explicitly scored against EITHER CausalDynamicText's frozen-genome evaluation OR StaticText+Gate's fixed-pipeline evaluation. This stage is the latter path and reaches the frozen primary endpoint without a Darwin loop.",
            "consequences_deliberately_not_taken": "no Darwin loop, no genome freeze, no receipts/genome-frozen.json written, and therefore NO unlock of eval-200/holdout-100. The mechanical lock in harness/latentmesh-live/src/gsm8k.rs remains fully engaged after this stage.",
            "darwin_seed_registered_by_adr030_but_unused_here": DARWIN_SEED_UNUSED_IN_THIS_STAGE,
        },
        "config": {
            "sender": SENDER, "receiver": RECEIVER,
            "message_content": "the sender's own greedy generated solution text, verbatim, with nothing added between the two agents (ADR-023's StaticText = 'fixed hand-designed 2-agent pipeline, text channel')",
            "decoding": "greedy, batch=1, max_new_tokens=400, both models, all five conditions",
            "greedy_justification": "ADR-030 grants the deterministic exception for self_generated, which requires the receiver's own prior deterministic output to be well-defined; greedy is also the witness arm every prior rung drew under, and deterministic per-item outcomes maximise paired-test power.",
            "receiver_prompt_template": "chat_prompt(system, '{question}\\n\\nA message from a previous solver:\\n{message}\\n\\n{ANSWER_FORMAT}'); the zero control uses '{question}\\n\\n{ANSWER_FORMAT}'",
            "controls": {
                "text_equivalent_dropped": "ADR-030: degenerate when the channel under test is already text — it would compare text against itself.",
                "zero": "no message reaches the receiver: the bare question prompt. STRUCTURALLY DIFFERENT from the message-bearing conditions (it has no message block) — that is ADR-003's own definition of the control, disclosed rather than smoothed over.",
                "random": "length-matched in the receiver's OWN tokens to the real message, drawn uniformly from ordinary ids [0, first_special) with per-item ChaCha8 seed RANDTOK_SEED_BASE + item index, then decoded to text and placed in the identical message block.",
                "mismatched": "the real sender message from the PREVIOUS item in the stream — content-shaped, wrong task. No lookahead is used anywhere, so early stopping stays honest; stream order 0 is primed from a designated eligible item outside the drawn stream.",
                "self_generated": "the receiver's own zero-condition greedy output, fed back to itself as the message. Reuses that generation rather than recomputing it.",
            },
            "random_control_vocabulary": {"ordinary_ids_below": first_special},
            "randtok_seed_base": RANDTOK_SEED_BASE,
            "mismatched_priming_item": priming_item,
            "nll_not_computed": "ADR-030 explicitly rejects gold-token NLL as a co-primary or replacement statistic (docs/research/031 §2.4: it is blind to the one real effect the ladder produced). Spending decode on it would spend it on a statistic the pre-registration already ruled out.",
        },
        "item_supply": {
            "source_split": "adaptation-512",
            "source": ADAPTATION_512,
            "source_sha256_of_train_jsonl": train_sha,
            "consumption_order": "the file's own fixed ascending index order, sequential, never shuffled and never re-seeded",
            "adr030_rule_superseded": "ADR-030 §'Sampler, dataset and split discipline' names the frozen 40-item probe as the e-process's starting population on the premise that it is 'a subset of adaptation-512 by construction'. ADR-036 Decision 2 verified that premise FALSE by direct set intersection (the probe meets adaptation-512 in exactly one item, index 1153) and ruled that the stream is drawn entirely from adaptation-512 in index order. This stage follows ADR-036's correction, disclosed here rather than silently resolved.",
            "side_benefit": "this stage's zero control therefore runs on the IDENTICAL item population as run2-m4i's baseline_uninjected, making that a like-for-like comparator.",
            "leakage_exclusion": {
                "rule": "ADR-024's 13 probe-overlap items, kept in force by ADR-036 Decision 2(2)",
                "excluded_item_indices": LEAKAGE_EXCLUSIONS,
                "of_those_present_in_adaptation_512": excluded_present,
                "measured_not_assumed": true,
                "no_mechanism_in_this_stage": "nothing here is trained, so the exclusion has no mechanism to bite; it is kept for protocol identity, not because it does work.",
            },
            "eligible_pool_size": eligible.len(),
            "n_max": N_MAX,
            "eval_holdout_lock": "eval-200 / holdout-100 untouched and still mechanically locked (no genome-frozen receipt exists, and this stage does not write one)",
        },
        "power_statement_registered_before_the_draw": {
            "requirement": "ADR-040's standing rule: expected discordant-pair count and minimum attainable p stated from MEASURED rates before any item is consumed.",
            "measured_inputs": {
                "receiver_no_message_on_this_exact_population": format!("{m4i_baseline}/{m4i_n}"),
                "receiver_no_message_source_field": "run2-m4i receipt summary.accuracy.baseline_uninjected / summary.n_evaluated",
                "sender_greedy_first_pass_on_the_frozen_40": "37/40",
                "sender_source_field": "s2b-receipt-cellL18toL14-slots8-poolfull-rescaletrue-n40.json items[].sender_first_pass.correct",
                "noise_floor_discordance_two_near_equal_conditions": "66/300",
                "noise_floor_source_field": "run2-m4i receipt summary.e_process.n_discordant",
            },
            "drift_identity": "wins - losses = n*(p_text - p_control) pins the log-wealth drift exactly: ln(W_n)/n = 0.15114*Delta - 0.01138*d, with Delta the accuracy gap and d the discordance rate.",
            "structural_capability_threshold": "crossing ln(20)=2.9957 within N_max=300 needs drift >= 0.0099857, i.e. Delta >= ~0.089 taking d ~ Delta + 0.22 (the measured noise floor). The measured sender-vs-receiver capability gap is 0.925 - 0.467 = 0.458, so the text channel would have to carry under a fifth of that gap for this endpoint to be structurally incapable.",
            "expected_crossing": "Delta=0.32, d=0.42 gives drift 0.0436/item and a crossing near item 69; Delta=0.15, d=0.35 gives item 160; Delta=0.10, d=0.30 gives item 256.",
            "min_attainable_p_floors": "mid-p McNemar can reject at alpha=0.05 only from n_disc >= 4 (min 0.5*2^-n_disc = 0.03125); the exact sign test needs n_disc >= 5. Both floors are reported against this draw's OWN observed n_disc per control, above.",
            "contrast_with_the_ladder": "the 9 cross-model latent draws (S2b x4, M3 x2, M4 x3) had n_disc = 3,3,4,3,4,3,4,4,5; EIGHT of the nine sat in the dead zone where the exact sign test could not reject at any true effect size — only M4 r=256 (n_disc=5) cleared that floor. That regime arose because Delta was ~0 by construction (m4i: aligned 128 vs random 132). Run 3's Delta is bounded below by a measured capability gap instead.",
            "corrected_family_figure_provenance": "docs/research/047-authoritative-power-table.md section 3.1 is the authoritative recount and was verified directly against that document for this receipt, not quoted from a summary. The widely-circulated '6 of 9' figure is WRONG, and 047 section 3.4 names docs/adr/030 lines 143-144 — this run's OWN pre-registration — as one of the documents propagating it, alongside the corrupted multiset {3,3,3,4,4,4,4,5,5}. That corrupted list is not even self-consistent with '6 of 9': it contains SEVEN values at n_disc in {3,4}. ADR-030 was corrected at commit 212c6c2. Recorded here because this run inherited the error from its own spec and must not propagate it further.",
            "family_figure_is_statistic_dependent": "docs/research/047 section 2: across all 14 draws, 10 of 14 are incapable under the EXACT sign test but only 6 of 14 under MID-P McNemar, because the mid-p floor is 2^-(n_disc+1) rather than 2^-n_disc, which makes n_disc=4 capable (0.03125 < alpha) while n_disc<=3 stays incapable. ADR-030 registers mid-p as primary, so the mid-p floor of n_disc >= 4 is the one that governs run 3 — the same floor derived independently above.",
        },
        "e_process": {
            "registered_source": "ADR-030 section 'Confirmatory scale', quoted verbatim in the ADR",
            "rule": "W_0 = 1. Concordant item: W_i = W_{i-1}. Discordant item: W_i = W_{i-1} * (1 + lambda*(X_i - 0.5)), X_i = 1 iff the gated-text condition wins the pair.",
            "best_control_requirement_implemented_as": "an INTERSECTION-UNION test: four wealth processes run in parallel, one per control, and PASS requires min_k W_k >= 1/alpha. That is exactly ADR-030's 'greater than the best (most favorable to the null) of the four controls' in sequential form; and it is conservative in the same direction ADR-030 section 5 already discloses.",
            "iut_validity_argument": "Let k* be the TRUE best control, i.e. the one under which the composite null 'gated text is no better than the best control' is realised. W_{k*,t} is a nonnegative martingale under that null, so Ville's inequality gives P(sup_t W_{k*,t} >= 20) <= 1/20 = alpha. The registered PASS event {min_k W_k crosses} is a SUBSET of {W_{k*} crosses}, so P(PASS | null) <= alpha. NO multiplicity correction is needed, and none is applied. The IUT is also a more faithful reading of 'beats the best control' than a post-hoc pick would be, because the selection never sees the outcome.",
            "fixed_n_posthoc_reading_reported_alongside": "ADR-030 section 5's literal post-hoc reading is ALSO reported, from these same committed per-control pairs, at zero additional cost and with no new items: see summary.realised_best_control_most_favorable_to_null and the per_control block's mid-p / exact values. Both readings are published so that neither can be said to have been chosen to suit the result.",
            "lambda": LAMBDA,
            "alpha": E_ALPHA,
            "wealth_threshold": w_threshold,
            "n_max": N_MAX,
            "stopping": "the draw stops at the first item where min_k W_k >= the threshold; otherwise it runs the full N_max.",
            "never_restarted": "the e-process is never restarted, re-parametrised, or re-run against this stage after seeing W_t. This is the one and only draw.",
            "degenerate_item_rule": "an item whose sender pass produces no tokens yields no pair and therefore no wealth update (identical in effect to a concordant item), but it still CONSUMES one of the N_max budget items. Registered here, not chosen after seeing the data.",
            "trajectory_is_complete": "the full W_t path for all four controls is committed regardless of outcome.",
            "no_p_value_translation": "no exact-sign or mid-p value is offered as the e-process's own outcome. The sequential result is reported on its own scale (crossing item count, or final wealth at N_max); the fixed-sample statistics below are reported on the accumulated pairs, as ADR-030 requires, and answer a different question.",
        },
        "summary": {
            "n_evaluated": n,
            "items_drawn": items_drawn,
            "n_degenerate_sender_pass": degenerate,
            "sender_greedy_first_pass_correct": sender_first_pass_correct,
            "gated_text_correct": text_c,
            "gated_text_accuracy": text_c as f64 / n.max(1) as f64,
            "per_control": per_control,
            "READ_THIS_BEFORE_THE_MAGNITUDE": {
                "caveat": "The message is the sender's own solution text and therefore CONTAINS THE FINAL ANSWER. A large gated-text vs `zero` effect is close to trivial and must not stand as the headline: `zero` asks only 'does ANY message help?', which is nearly a tautology when the message carries the answer.",
                "headline_comparison_is_text_vs_mismatched": "`mismatched` is the load-bearing control. It supplies another episode's REAL message — content-shaped, same register, and carrying a number of its own — so it asks the causal question: does the RIGHT message help, over a wrong-but-equally-message-shaped one? Read text-vs-mismatched first; read text-vs-zero as context only.",
                "scope_of_any_positive_result": "a pass demonstrates that ADR-003's gate has discriminating power on a text channel. It does NOT establish that latent transfer works, that the gate is novel in the abstract, or that the effect would survive a message that withheld the final answer.",
                "binding_control_check": "if `mismatched` is NOT the realised binding control below, that is itself a finding and is flagged rather than smoothed over.",
            },
            "headline_text_vs_mismatched": {
                "control": Control::ALL[2].tag(),
                "gated_text_correct": text_c,
                "mismatched_correct": paired.iter().filter(|r| r.control_correct[2]).count(),
                "n": n,
                "wins_text": wins[2],
                "losses_text": losses[2],
                "n_discordant": wins[2] + losses[2],
                "final_wealth": wealth[2],
                "mid_p_mcnemar_one_sided": common::mid_p_one_sided(wins[2], losses[2]),
                "exact_sign_one_sided": common::sign_test_one_sided(wins[2], losses[2]),
                "why_this_is_the_headline": "content-matched decoy substitution is the whole causal claim; text-vs-zero is message-vs-silence and is not.",
            },
            "realised_best_control_most_favorable_to_null": Control::ALL[best_k].tag(),
            "mismatched_is_the_binding_control": best_k == 2,
            "binding_control_finding": if best_k == 2 {
                "as expected: `mismatched` is the control most favorable to the null, so the intersection-union PASS is governed by the content-matched decoy comparison — the causal question."
            } else {
                "FLAGGED, NOT SMOOTHED OVER: `mismatched` is NOT the binding control on this draw. Some other control was harder to beat, which means the intersection-union outcome is governed by a comparison other than the content-matched decoy one. Read the per_control block and the headline_text_vs_mismatched block together before drawing any causal conclusion."
            },
            "best_control_selection_timing": "post-hoc from this draw, as ADR-030 section 5 registers; the control with the LOWEST final wealth is the hardest bar the text condition had to clear.",
            "min_wealth_across_controls": w_min_final,
            "e_process_crossed": e_pass,
            "e_process_crossed_at_item": crossed_at,
            "uplift_vs_best_control": {
                "gated_text": {"correct": text_c, "n": n, "exact_95_ci": [t_lo, t_hi]},
                "best_control": {"control": Control::ALL[best_k].tag(), "correct": best_c, "n": n, "exact_95_ci": [c_lo, c_hi]},
                "raw_delta": (text_c as f64 - best_c as f64) / n.max(1) as f64,
                "note": "ADR-030's secondary criterion: reported regardless of whether the primary gate passes.",
            },
            "zero_control_vs_m4i_like_for_like": {
                "this_stage_zero_correct": paired.iter().filter(|r| r.control_correct[0]).count(),
                "this_stage_n": n,
                "m4i_baseline_uninjected": m4i_baseline,
                "m4i_n": m4i_n,
                "comparability": "identical population and consumption order; prompts differ only in that this stage's zero condition carries no placeholder slots (m4i's baseline was also uninjected but drawn under the question-tail prompt).",
            },
        },
        "compute_cost_seconds": {
            "note": "ADR-030's secondary compute criterion: GPU-seconds and wall clock per condition, measured here rather than re-derived.",
            "sender_generation": sender_seconds,
            "receiver_gated_text": cond_seconds[0],
            "receiver_zero": cond_seconds[1],
            "receiver_random": cond_seconds[2],
            "receiver_mismatched": cond_seconds[3],
            "receiver_self_generated": cond_seconds[4],
            "wall_clock_s": t0.elapsed().as_secs_f32(),
        },
        "multiplicity": {
            "family": "ADR-030's multiplicity section: this is draw #11 in the family of architecture/channel tests the frozen protocol lineage has produced (1 same-model mechanics check that passed, 9 cross-model transfer tests that all failed, then this one).",
            "between_rung_correction": "none needed — the ladder's ordering is a fixed-sequence gatekeeping procedure (docs/research/031 section 4.3); this stage was authored and run after M3/M4's outcomes were on record.",
            "within_rung_correction": "none needed — this stage has no parallel variants competing for the same draw. The four control comparisons are an intersection-union test of ONE hypothesis, not four separate hypotheses, so they take no Holm-Bonferroni correction.",
            "disclosure_required_whatever_the_outcome": "this result must be reported as draw #11 in a family of N channel/architecture tests, never as if it were the only test ever run.",
        },
        "trajectories": Control::ALL.iter().enumerate().map(|(k, c)| serde_json::json!({
            "control": c.tag(),
            "steps": trajectory[k].iter().map(|s| serde_json::json!({
                "order": s.order, "item": s.item, "text_correct": s.text_correct,
                "control_correct": s.control_correct, "discordant": s.discordant,
                "x": s.x, "wealth": s.wealth,
            })).collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
        "items": rows,
        "registered_outcome_label_not_an_adjudication": if e_pass {
            "E-PROCESS CROSSED: min_k W_k reached 1/alpha within N_max under the registered intersection-union rule."
        } else {
            "E-PROCESS DID NOT CROSS within N_max under the registered intersection-union rule."
        },
        "adjudication_note": "this label is the mechanical outcome of the registered rule. It is NOT the verdict; the coordinator adjudicates against the numbers above.",
    });
    let receipt_name = if smoke.is_some() {
        "run3-stageA-SMOKE-mechanics-check.json"
    } else {
        "run3-stageA-receipt-statictext-gate-eprocess.json"
    };
    common::write_receipt(&dir, receipt_name, &receipt)?;

    println!("\n--- run 3 stage A ---");
    println!("n = {n} evaluated ({items_drawn} drawn, {degenerate} degenerate)");
    println!("gated_text {text_c}/{n}");
    for (k, c) in Control::ALL.iter().enumerate() {
        let acc = paired.iter().filter(|r| r.control_correct[k]).count();
        println!(
            "  {:<15} {acc}/{n}  W={:.4}  n_disc={}  wins={} losses={}  mid-p={:.5}  exact={:.5}",
            c.tag(),
            wealth[k],
            wins[k] + losses[k],
            wins[k],
            losses[k],
            common::mid_p_one_sided(wins[k], losses[k]),
            common::sign_test_one_sided(wins[k], losses[k]),
        );
    }
    println!(
        "best control (most favorable to null) = {}; min_k W_k = {w_min_final:.4}; crossed = \
         {e_pass} at item {crossed_at:?}",
        Control::ALL[best_k].tag()
    );
    println!(
        "\nCAVEAT BEFORE THE MAGNITUDE: the message contains the final answer, so text-vs-zero is \
         near-tautological. The HEADLINE is text vs mismatched (content-matched decoy): {}W/{}L, \
         mid-p={:.5}. mismatched is the binding control: {}",
        wins[2],
        losses[2],
        common::mid_p_one_sided(wins[2], losses[2]),
        best_k == 2
    );
    Ok(())
}

/// The sender's message for one item: its own greedy solution text. `None`
/// when the sender emits no tokens (the registered degenerate case).
fn sender_message(
    sender: &mut QwenRuntime,
    item: &common::Gsm8kItem,
) -> anyhow::Result<Option<String>> {
    let prompt = QwenRuntime::chat_prompt(
        SYSTEM,
        &format!("{}\n\n{}", item.question, common::ANSWER_FORMAT),
    );
    let toks = sender.encode(&prompt).map_err(anyhow::Error::msg)?;
    let mut greedy = Sampler::new(Sampling::Greedy, 0);
    let out = sender
        .generate(&toks, None, &mut greedy, MAX_NEW_TOKENS, false)
        .map_err(anyhow::Error::msg)?;
    Ok((!out.tokens.is_empty()).then_some(out.text))
}
