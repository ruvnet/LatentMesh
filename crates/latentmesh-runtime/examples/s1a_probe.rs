//! S1a — self-pair injection-mechanics probe (design doc 024 §7 S1a).
//!
//! Qwen2.5-1.5B-Instruct -> itself, transform = identity. Per item: the model
//! solves a GSM8K-train problem (greedy); its pooled L19 residual over the
//! generated reasoning is re-injected into placeholder slots of a fresh pass
//! on the same question; paired conditions real / zero / random.
//!
//! PRE-COMMITTED ANALYSIS (fixed in this source before any run; the receipt
//! echoes it):
//!   - primary gate: one-sided exact sign test on paired ACCURACY,
//!     real > random, over discordant pairs, alpha = 0.05
//!   - secondary diagnostic (not the gate): paired teacher-forced mean NLL of
//!     "#### <gold>" as the immediate assistant continuation, real vs random,
//!     same sign test — used to attribute a starved primary (mechanism dead
//!     vs underpowered) per the design's kill-switch note
//!   - real-vs-zero reported alongside, same tests
//!   - decoding: greedy (deterministic; maximizes paired power; the T=0.6
//!     arm belongs to later stages), max 400 new tokens
//!   - default config: 8 slots, capture block = inject block = 19/28,
//!     pool over the full generated span, rescale ON (to the natural L19
//!     median per-position norm of the same pass)
//!   - items: 40 from GSM8K train.jsonl, ChaCha8 seed 0x51A1, indices in
//!     the receipt; random vectors ChaCha8 per-item seed 0x51A2_0000 + index,
//!     norm-matched to the effective real vector
//!
//! Non-default configs (kill-switch iteration levers: `--slots`, `--block`,
//! `--pool-last`, `--no-rescale`, `--items`) mark the receipt
//! pre_committed=false.
//!
//! DECLARED HARNESS FIXES after the first full run (its receipt preserved as
//! `*.run1-pre-fixes.json`): (a) the vendored BF16 RoPE-table aliasing
//! degraded ALL generations past position ~256 (qwen2_a.rs deviation 4) — a
//! runtime bug, not an analysis change; (b) answer scoring measured `#### n`
//! format adherence, not correctness — extraction now falls back to the last
//! number and compares numerically ("2.0" == "2"). The statistical procedure
//! (paired conditions, one-sided sign test, alpha) is unchanged.
//!
//! Receipts: `crates/latentmesh-runtime/target/latentmesh-runs/s1a/`.
//! Run: PATH=/usr/local/cuda-12.8/bin:$PATH \
//!      cargo run --release --features cuda --example s1a_probe

#[path = "common/mod.rs"]
#[allow(dead_code)]
mod common;

use latentmesh_runtime::{
    capture::forward_capture,
    inject::{teacher_forced_nll, InjectionMode, InjectionSpec},
    norms,
    sampler::{Sampler, Sampling},
    QwenRuntime,
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

const MODEL: &str = "Qwen/Qwen2.5-1.5B-Instruct";
const ITEM_SEED: u64 = 0x51A1;
const RANDVEC_SEED_BASE: u64 = 0x51A2_0000;
const MAX_NEW_TOKENS: usize = 400;
const ALPHA: f64 = 0.05;
const SYSTEM: &str = "You are a careful math tutor.";

#[derive(Debug, Clone)]
struct ProbeConfig {
    slots: usize,
    block: usize,
    pool_last: Option<usize>,
    rescale: bool,
    n_items: usize,
}

impl ProbeConfig {
    fn pre_committed(&self) -> bool {
        self.slots == 8
            && self.block == 19
            && self.pool_last.is_none()
            && self.rescale
            && self.n_items == 40
    }
    fn tag(&self) -> String {
        format!(
            "slots{}-block{}-pool{}-rescale{}-n{}",
            self.slots,
            self.block,
            self.pool_last.map_or("full".into(), |k| k.to_string()),
            self.rescale,
            self.n_items
        )
    }
}

fn parse_args() -> ProbeConfig {
    let mut cfg = ProbeConfig {
        slots: 8,
        block: 19,
        pool_last: None,
        rescale: true,
        n_items: 40,
    };
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        let next = |i: usize| args.get(i + 1).and_then(|v| v.parse().ok());
        match args[i].as_str() {
            "--slots" => cfg.slots = next(i).expect("--slots N"),
            "--block" => cfg.block = next(i).expect("--block N"),
            "--pool-last" => cfg.pool_last = Some(next(i).expect("--pool-last N")),
            "--items" => cfg.n_items = next(i).expect("--items N"),
            "--no-rescale" => {
                cfg.rescale = false;
                i += 1;
                continue;
            }
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

    let dir = common::run_dir("s1a");
    let data = dir.join("gsm8k-train.jsonl");
    let train_sha = common::fetch(common::GSM8K_TRAIN_URL, &data)?;
    let all_items = common::load_gsm8k(&data)?;
    println!(
        "gsm8k train.jsonl: {} items, sha256 {train_sha}",
        all_items.len()
    );

    // Seeded held-out selection (ChaCha8, indices recorded).
    let mut rng = ChaCha8Rng::seed_from_u64(ITEM_SEED);
    let mut indices = rand::seq::index::sample(&mut rng, all_items.len(), cfg.n_items).into_vec();
    indices.sort_unstable();

    let device = latentmesh_runtime::device().map_err(anyhow::Error::msg)?;
    let mut rt =
        QwenRuntime::load(MODEL, &device, candle_core::DType::BF16).map_err(anyhow::Error::msg)?;
    let pad_id = rt
        .tokenizer
        .token_to_id("<|fim_pad|>")
        .ok_or_else(|| anyhow::anyhow!("<|fim_pad|> not in tokenizer"))?;

    let mut rows = Vec::new();
    let mut paired: Vec<Pair> = Vec::new();
    for (done, &idx) in indices.iter().enumerate() {
        let item = &all_items[idx];
        match run_item(&mut rt, item, pad_id, &cfg, &device)? {
            Some((row, pair)) => {
                println!(
                    "[{}/{}] item {idx}: real={} zero={} random={} (nll {:.3}/{:.3}/{:.3}) {:.0}s",
                    done + 1,
                    indices.len(),
                    pair.0,
                    pair.1,
                    pair.2,
                    pair.3,
                    pair.4,
                    pair.5,
                    t0.elapsed().as_secs_f32()
                );
                rows.push(row);
                paired.push(pair);
            }
            None => {
                println!(
                    "[{}/{}] item {idx}: degenerate capture pass, skipped",
                    done + 1,
                    indices.len()
                );
                rows.push(serde_json::json!({"item": idx, "skipped": "degenerate capture pass"}));
            }
        }
    }

    // ---- Pre-committed analysis ------------------------------------------
    let n = paired.len();
    let acc = |f: &dyn Fn(&Pair) -> bool| paired.iter().filter(|p| f(p)).count();
    let real_c = acc(&|p| p.0);
    let zero_c = acc(&|p| p.1);
    let rand_c = acc(&|p| p.2);
    let wins_rr = paired.iter().filter(|p| p.0 && !p.2).count();
    let loss_rr = paired.iter().filter(|p| !p.0 && p.2).count();
    let p_primary = common::sign_test_one_sided(wins_rr, loss_rr);
    let wins_rz = paired.iter().filter(|p| p.0 && !p.1).count();
    let loss_rz = paired.iter().filter(|p| !p.0 && p.1).count();
    let p_real_zero = common::sign_test_one_sided(wins_rz, loss_rz);
    // Secondary: NLL sign tests (lower is better).
    let nwin_rr = paired.iter().filter(|p| p.3 < p.5).count();
    let nloss_rr = paired.iter().filter(|p| p.3 > p.5).count();
    let p_nll_rr = common::sign_test_one_sided(nwin_rr, nloss_rr);
    let nwin_rz = paired.iter().filter(|p| p.3 < p.4).count();
    let nloss_rz = paired.iter().filter(|p| p.3 > p.4).count();
    let p_nll_rz = common::sign_test_one_sided(nwin_rz, nloss_rz);
    let mean = |f: &dyn Fn(&Pair) -> f32| paired.iter().map(f).sum::<f32>() / n.max(1) as f32;

    let gate_pass = p_primary < ALPHA;
    let receipt = serde_json::json!({
        "stage": "S1a",
        "design": "docs/research/024-live-latent-experiment-design.md section 7 S1a",
        "env": common::env_info(&nvcc),
        "pre_committed": cfg.pre_committed(),
        "config": {
            "model": MODEL, "self_pair": true, "transform": "identity",
            "slots": cfg.slots, "capture_block": cfg.block, "inject_block": cfg.block,
            "pool_span": cfg.pool_last.map_or("full generated span".to_string(), |k| format!("last {k} generated tokens")),
            "rescale_to_natural_median": cfg.rescale,
            "decoding": "greedy, batch=1, max_new_tokens=400",
            "item_seed_chacha8": ITEM_SEED, "randvec_seed_base": RANDVEC_SEED_BASE,
            "primary_test": "one-sided exact sign test, paired accuracy, real > random, alpha 0.05",
            "secondary_diagnostic": "one-sided sign test on paired teacher-forced NLL of '#### <gold>', real < random",
        },
        "dataset": {"source": common::GSM8K_TRAIN_URL, "sha256": train_sha, "indices": indices},
        "items": rows,
        "summary": {
            "n_evaluated": n,
            "accuracy": {"real": real_c, "zero": zero_c, "random": rand_c},
            "primary_real_vs_random": {"wins": wins_rr, "losses": loss_rr, "p_one_sided": p_primary, "alpha": ALPHA, "pass": gate_pass},
            "real_vs_zero": {"wins": wins_rz, "losses": loss_rz, "p_one_sided": p_real_zero},
            "nll_mean": {"real": mean(&|p| p.3), "zero": mean(&|p| p.4), "random": mean(&|p| p.5)},
            "nll_real_vs_random": {"wins": nwin_rr, "losses": nloss_rr, "p_one_sided": p_nll_rr},
            "nll_real_vs_zero": {"wins": nwin_rz, "losses": nloss_rz, "p_one_sided": p_nll_rz},
        },
        "gate_pass": gate_pass,
        "wall_clock_s": t0.elapsed().as_secs_f32(),
    });
    let name = format!("s1a-receipt-{}.json", cfg.tag());
    common::write_receipt(&dir, &name, &receipt)?;
    println!(
        "S1a: acc real {real_c}/{n} zero {zero_c}/{n} random {rand_c}/{n}; primary p={p_primary:.4} (wins {wins_rr}, losses {loss_rr}) => gate_pass={gate_pass}"
    );
    println!("secondary NLL real-vs-random p={p_nll_rr:.4} ({nwin_rr}W/{nloss_rr}L)");
    Ok(())
}

/// (real_correct, zero_correct, random_correct, nll_real, nll_zero, nll_random)
type Pair = (bool, bool, bool, f32, f32, f32);
type PairRow = (serde_json::Value, Pair);

fn run_item(
    rt: &mut QwenRuntime,
    item: &common::Gsm8kItem,
    pad_id: u32,
    cfg: &ProbeConfig,
    device: &candle_core::Device,
) -> anyhow::Result<Option<PairRow>> {
    let e = anyhow::Error::msg;
    let fmt = common::ANSWER_FORMAT;
    let mut greedy = Sampler::new(Sampling::Greedy, 0);

    // 1) Capture pass: solve, then teacher-forced re-prefill with the tap.
    let cap_prompt = QwenRuntime::chat_prompt(SYSTEM, &format!("{}\n\n{fmt}", item.question));
    let cap_tokens = rt.encode(&cap_prompt).map_err(e)?;
    let gen = rt
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
    let span_start = match cfg.pool_last {
        Some(k) => full.len().saturating_sub(k).max(cap_tokens.len()),
        None => cap_tokens.len(),
    };
    let (_, cap) = forward_capture(
        &mut rt.model,
        &full,
        cfg.block,
        span_start..full.len(),
        device,
    )
    .map_err(e)?;
    let natural = norms::stats(cap.per_position_l2.clone());
    if std::env::var_os("LATENTMESH_DEBUG").is_some() {
        println!(
            "--- item {} (gold {}) capture-pass generation ---\n{}\n---",
            item.index, item.gold, gen.text
        );
    }
    let first_pass_answer = common::extract_answer(&gen.text);
    let first_pass_correct = first_pass_answer
        .as_deref()
        .is_some_and(|a| common::answers_equal(a, &item.gold));

    // 2) Injection prompt with placeholder slots.
    let slots = "<|fim_pad|>".repeat(cfg.slots);
    let inj_prompt = QwenRuntime::chat_prompt(
        SYSTEM,
        &format!(
            "{}\n\nA compressed latent hint from a previous solver is stored in these slots: [{slots}]\n\n{fmt}",
            item.question
        ),
    );
    let inj_tokens = rt.encode(&inj_prompt).map_err(e)?;
    let positions = QwenRuntime::placeholder_positions(&inj_tokens, pad_id);
    anyhow::ensure!(positions.len() == cfg.slots, "slot count mismatch");

    let pooled_l2 = norms::l2(&cap.pooled);
    let scale = cfg.rescale.then(|| natural.median / pooled_l2);
    let real = InjectionSpec {
        after_block: cfg.block,
        positions: positions.clone(),
        vector: cap.pooled.clone(),
        scale,
        mode: InjectionMode::Overwrite,
    };
    let target_l2 = norms::l2(&real.effective_vector());
    // Random control: per-item seeded gaussian, norm-matched to the real vector.
    let mut vrng = ChaCha8Rng::seed_from_u64(RANDVEC_SEED_BASE + item.index as u64);
    let gauss = gaussian_vec(&mut vrng, cap.pooled.len());
    let random = InjectionSpec {
        after_block: cfg.block,
        positions,
        vector: gauss.clone(),
        scale: Some(target_l2 / norms::l2(&gauss)),
        mode: InjectionMode::Overwrite,
    };

    // 3) Paired conditions: real / zero (slots unreplaced) / random.
    let mut outcome = |spec: Option<&InjectionSpec>| -> anyhow::Result<(bool, f32, String)> {
        let mut s = Sampler::new(Sampling::Greedy, 0);
        let out = rt
            .generate(&inj_tokens, spec, &mut s, MAX_NEW_TOKENS, false)
            .map_err(e)?;
        let correct = common::extract_answer(&out.text)
            .is_some_and(|a| common::answers_equal(&a, &item.gold));
        let answer_toks = rt.encode(&format!("#### {}", item.gold)).map_err(e)?;
        let nll_tokens: Vec<u32> = inj_tokens
            .iter()
            .chain(answer_toks.iter())
            .copied()
            .collect();
        let nll = teacher_forced_nll(
            &mut rt.model,
            &nll_tokens,
            inj_tokens.len()..nll_tokens.len(),
            spec,
            device,
        )
        .map_err(e)?;
        Ok((correct, nll, out.text))
    };
    let (real_ok, real_nll, real_text) = outcome(Some(&real))?;
    let (zero_ok, zero_nll, _) = outcome(None)?;
    let (rand_ok, rand_nll, _) = outcome(Some(&random))?;

    let row = serde_json::json!({
        "item": item.index,
        "gold": item.gold,
        "first_pass": {"correct": first_pass_correct, "answer": first_pass_answer, "generated_tokens": gen.tokens.len()},
        "capture": {
            "hidden_size": cap.hidden_size,
            "pooled_l2_raw": pooled_l2,
            "injected_l2": target_l2,
            "natural_l19_norms": natural,
            "span": [span_start, full.len()],
        },
        "conditions": {
            "real": {"correct": real_ok, "nll_gold": real_nll},
            "zero": {"correct": zero_ok, "nll_gold": zero_nll},
            "random": {"correct": rand_ok, "nll_gold": rand_nll},
        },
        "real_answer_tail": real_text.chars().rev().take(60).collect::<String>().chars().rev().collect::<String>(),
    });
    Ok(Some((
        row,
        (real_ok, zero_ok, rand_ok, real_nll, zero_nll, rand_nll),
    )))
}

/// Standard-normal vector via Box-Muller over a seeded ChaCha8 stream.
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
