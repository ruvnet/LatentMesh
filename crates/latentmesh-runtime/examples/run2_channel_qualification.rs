//! Run-2 M3.5: preregistered self-pair delivery-channel qualification.
//!
//! This is an oracle/control, not an adapter evaluation. It repeats the
//! frozen 40-item probe at the exact failed 1.5B receiver depth and at a
//! matched half-depth 3B scale profile. The model's own full-span mean-pooled
//! residual is injected through eight slots at gains 0.25, 0.5, 1, 2, 4,
//! with a per-item seeded, exact-norm matched random direction.
//!
//! The executable refuses to run without `--execute`, the byte-exact frozen
//! registration, frozen S1a artifact, dataset/item-set matches, CUDA, and a
//! registered profile. Downloads require the explicit `--allow-download`
//! flag; otherwise hf-hub is forced offline before model loading.
//!
//! Run (1.5B profile):
//!   cargo run --release --features cuda --example run2_channel_qualification -- \
//!     --execute --profile qwen2.5-1.5b-exact-channel

#[path = "common/channel_qualification.rs"]
mod channel;
#[path = "common/mod.rs"]
#[allow(dead_code)]
mod common;

use candle_core::Device;
use candle_nn::VarBuilder;
use channel::{GainResult, InjectionPolicy, ModelProfile, Registration};
use latentmesh_runtime::{
    capture::forward_capture,
    inject::{teacher_forced_nll, InjectionSpec},
    norms,
    sampler::{Sampler, Sampling},
    Config, ModelForCausalLM, QwenRuntime,
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::path::{Path, PathBuf};

const REGISTRATION: &str = "receipts/run2-m35-channel-preregistration.json";
const REGISTRATION_SHA256: &str =
    "ebe4e76947fdd514d3759c4b02e8c9189696e635cd14a8c03c3bf8488d445915";
const SYSTEM: &str = "You are a careful math tutor.";

#[derive(Debug)]
struct Args {
    profile: String,
    execute: bool,
    allow_download: bool,
}

fn parse_args() -> anyhow::Result<Args> {
    let args: Vec<String> = std::env::args().collect();
    let mut profile = None;
    let mut execute = false;
    let mut allow_download = false;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--profile" => {
                i += 1;
                profile = Some(
                    args.get(i)
                        .cloned()
                        .ok_or_else(|| anyhow::anyhow!("--profile TAG"))?,
                );
            }
            "--execute" => execute = true,
            "--allow-download" => allow_download = true,
            other => anyhow::bail!("unknown argument {other:?}"),
        }
        i += 1;
    }
    Ok(Args {
        profile: profile.ok_or_else(|| anyhow::anyhow!("--profile is required"))?,
        execute,
        allow_download,
    })
}

fn crate_path(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

fn repo_root() -> anyhow::Result<PathBuf> {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .map(Path::to_path_buf)
        .ok_or_else(|| anyhow::anyhow!("cannot resolve repository root"))
}

fn main() -> anyhow::Result<()> {
    let t0 = std::time::Instant::now();
    let args = parse_args()?;
    anyhow::ensure!(
        args.execute,
        "probe execution is locked; pass --execute after reviewing the frozen registration"
    );

    // Registration hash is checked before parsing. Nothing below can change
    // the registered profile, item set, gains, seeds, or statistics.
    let validated = channel::load_and_validate(
        &crate_path(REGISTRATION),
        REGISTRATION_SHA256,
        &args.profile,
    )?;
    let reg = &validated.registration;
    let profile = channel::profile(reg, &args.profile)?.clone();
    validate_source_artifacts(reg)?;
    println!(
        "registration: {} ({})",
        validated.path.display(),
        validated.raw_sha256
    );

    let nvcc = latentmesh_runtime::assert_cuda_build_env().map_err(anyhow::Error::msg)?;
    let run_dir = common::run_dir(&format!("run2-m35/{}", profile.tag));
    let data_path = run_dir.join("gsm8k-train.jsonl");
    let data_sha = if data_path.exists() {
        common::sha256_hex(&std::fs::read(&data_path)?)
    } else if args.allow_download {
        println!(
            "explicit download authorized: {}",
            reg.frozen_source.dataset.source
        );
        common::fetch(&reg.frozen_source.dataset.source, &data_path)?
    } else {
        anyhow::bail!(
            "dataset cache {} is absent; populate it manually or rerun with explicit --allow-download",
            data_path.display()
        );
    };
    anyhow::ensure!(
        data_sha == reg.frozen_source.dataset.sha256,
        "dataset artifact hash mismatch"
    );
    let items = common::load_gsm8k(&data_path)?;
    let indices = derive_indices(items.len(), reg)?;
    anyhow::ensure!(
        indices == reg.frozen_source.dataset.indices,
        "derived frozen item set mismatch"
    );

    if args.allow_download {
        println!(
            "explicit checkpoint download authorized for {} if cache is incomplete",
            profile.model_id
        );
    } else {
        println!("model loading from verified local hf-hub cache only");
    }

    let device = latentmesh_runtime::device().map_err(anyhow::Error::msg)?;
    let mut runtime = if args.allow_download {
        QwenRuntime::load(&profile.model_id, &device, candle_core::DType::BF16)
            .map_err(anyhow::Error::msg)?
    } else {
        load_from_local_cache(&profile, &device)?
    };
    validate_runtime(&runtime, &profile)?;
    let pad_id = runtime
        .tokenizer
        .token_to_id(&reg.protocol.placeholder_token)
        .ok_or_else(|| anyhow::anyhow!("registered placeholder absent from tokenizer"))?;

    let mut rows = Vec::with_capacity(indices.len());
    let mut measured = Vec::with_capacity(indices.len());
    for (done, &index) in indices.iter().enumerate() {
        let (row, item) = run_item(&mut runtime, &items[index], pad_id, reg, &profile, &device)?;
        println!(
            "[{}/{}] item {}: base={} zero={} identity/random={:?} {:.0}s",
            done + 1,
            indices.len(),
            index,
            item.baseline.correct,
            item.zero.correct,
            item.gains
                .iter()
                .map(|g| (g.identity.correct, g.random.correct))
                .collect::<Vec<_>>(),
            t0.elapsed().as_secs_f32()
        );
        rows.push(row);
        measured.push(item);
    }
    anyhow::ensure!(
        measured.len() == reg.protocol.n_items,
        "not all frozen items were evaluated"
    );
    let summary = summarize(reg, &measured)?;
    let receipt = serde_json::json!({
        "stage": "run2-M3.5-channel-qualification",
        "design": "docs/research/028-activation-transfer-channel-qualification.md",
        "env": common::env_info(&nvcc),
        "pre_committed": true,
        "registration": {
            "path": REGISTRATION,
            "sha256": validated.raw_sha256,
            "protocol_id": reg.protocol_id,
            "base_commit": reg.base_commit,
        },
        "profile": profile,
        "protocol": reg.protocol,
        "dataset": {
            "source": reg.frozen_source.dataset.source,
            "sha256": data_sha,
            "indices": indices,
            "s1a_artifact_validated": true,
        },
        "integrity": {
            "registration_hash": true,
            "registered_source_hashes": true,
            "dataset_hash": true,
            "exact_item_set": true,
            "model_identity": true,
            "model_shape": true,
            "all_injections_validated_before_use": true,
        },
        "items": rows,
        "summary": summary,
        "wall_clock_s": t0.elapsed().as_secs_f32(),
    });
    let out = format!("run2-m35-channel-receipt-{}-n40.json", profile.tag);
    common::write_receipt(&run_dir, &out, &receipt)?;
    println!(
        "M3.5 {} profile_pass={}",
        profile.tag, summary["gates"]["profile_pass"]
    );
    Ok(())
}

/// Offline loader used when network access has not been explicitly granted.
/// This avoids `ApiRepo::get`, whose cache-miss behavior is a download.
fn load_from_local_cache(profile: &ModelProfile, device: &Device) -> anyhow::Result<QwenRuntime> {
    let repo = hf_hub::Cache::from_env().model(profile.model_id.clone());
    let get = |name: &str| {
        repo.get(name).ok_or_else(|| {
            anyhow::anyhow!(
                "checkpoint cache miss for {}/{}; prefetch explicitly or pass --allow-download",
                profile.model_id,
                name
            )
        })
    };
    let config_path = get("config.json")?;
    let tokenizer_path = get("tokenizer.json")?;
    let weights = match profile.tag.as_str() {
        "qwen2.5-1.5b-exact-channel" => vec![get("model.safetensors")?],
        "qwen2.5-3b-scale-oracle" => {
            let index_path = get("model.safetensors.index.json")?;
            let index: serde_json::Value = serde_json::from_slice(&std::fs::read(index_path)?)?;
            let map = index["weight_map"]
                .as_object()
                .ok_or_else(|| anyhow::anyhow!("cached weight index has no weight_map"))?;
            let mut shards: Vec<&str> = map.values().filter_map(|v| v.as_str()).collect();
            shards.sort_unstable();
            shards.dedup();
            anyhow::ensure!(!shards.is_empty(), "cached weight index names no shards");
            shards
                .into_iter()
                .map(get)
                .collect::<anyhow::Result<Vec<_>>>()?
        }
        other => anyhow::bail!("offline loader has no registered profile {other:?}"),
    };
    let config: Config = serde_json::from_slice(&std::fs::read(config_path)?)?;
    let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow::anyhow!("tokenizer load: {e}"))?;
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&weights, candle_core::DType::BF16, device)
            .map_err(anyhow::Error::msg)?
    };
    let model = ModelForCausalLM::new(&config, vb).map_err(anyhow::Error::msg)?;
    Ok(QwenRuntime {
        model,
        tokenizer,
        config,
        device: device.clone(),
        model_id: profile.model_id.clone(),
    })
}

fn validate_source_artifacts(reg: &Registration) -> anyhow::Result<()> {
    let root = repo_root()?;
    let adr = root.join(&reg.frozen_source.adr_024.path);
    anyhow::ensure!(
        channel::sha256_hex(&std::fs::read(&adr)?)
            == reg.frozen_source.adr_024.registered_sha256()?,
        "ADR-024 artifact hash mismatch"
    );
    let s1a = root.join(&reg.frozen_source.s1a_receipt.path);
    channel::validate_s1a_bytes(&std::fs::read(s1a)?, reg)?;
    Ok(())
}

fn derive_indices(n_total: usize, reg: &Registration) -> anyhow::Result<Vec<usize>> {
    anyhow::ensure!(
        n_total >= reg.protocol.n_items,
        "dataset is smaller than frozen probe"
    );
    let mut rng = ChaCha8Rng::seed_from_u64(reg.protocol.item_seed_chacha8);
    let mut got = rand::seq::index::sample(&mut rng, n_total, reg.protocol.n_items).into_vec();
    got.sort_unstable();
    Ok(got)
}

fn validate_runtime(runtime: &QwenRuntime, profile: &ModelProfile) -> anyhow::Result<()> {
    anyhow::ensure!(
        runtime.model_id == profile.model_id,
        "loaded model identity mismatch"
    );
    anyhow::ensure!(
        runtime.config.num_hidden_layers == profile.expected_layers,
        "loaded layer count mismatch"
    );
    anyhow::ensure!(
        runtime.config.hidden_size == profile.hidden_size,
        "loaded hidden size mismatch"
    );
    anyhow::ensure!(
        profile.capture_block > 0 && profile.capture_block <= runtime.config.num_hidden_layers
    );
    anyhow::ensure!(
        profile.inject_block > 0 && profile.inject_block <= runtime.config.num_hidden_layers
    );
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct Outcome {
    correct: bool,
    nll: f32,
}

#[derive(Debug)]
struct GainPair {
    identity: Outcome,
    random: Outcome,
}

#[derive(Debug)]
struct ItemMeasurement {
    baseline: Outcome,
    zero: Outcome,
    gains: Vec<GainPair>,
}

fn run_item(
    rt: &mut QwenRuntime,
    item: &common::Gsm8kItem,
    pad_id: u32,
    reg: &Registration,
    profile: &ModelProfile,
    device: &Device,
) -> anyhow::Result<(serde_json::Value, ItemMeasurement)> {
    let e = anyhow::Error::msg;
    let fmt = common::ANSWER_FORMAT;
    let cap_prompt = QwenRuntime::chat_prompt(SYSTEM, &format!("{}\n\n{fmt}", item.question));
    let cap_tokens = rt.encode(&cap_prompt).map_err(e)?;
    let mut greedy = Sampler::new(Sampling::Greedy, 0);
    let generated = rt
        .generate(
            &cap_tokens,
            None,
            &mut greedy,
            reg.protocol.decoding.max_new_tokens,
            false,
        )
        .map_err(e)?;
    anyhow::ensure!(
        !generated.tokens.is_empty(),
        "item {} produced a degenerate empty capture",
        item.index
    );
    let full: Vec<u32> = cap_tokens
        .iter()
        .chain(&generated.tokens)
        .copied()
        .collect();
    let (_, capture) = forward_capture(
        &mut rt.model,
        &full,
        profile.capture_block,
        cap_tokens.len()..full.len(),
        device,
    )
    .map_err(e)?;
    anyhow::ensure!(
        capture.hidden_size == profile.hidden_size,
        "capture hidden-size mismatch"
    );
    anyhow::ensure!(
        capture.pooled.iter().all(|v| v.is_finite()),
        "nonfinite capture vector"
    );
    let pooled_l2 = norms::l2(&capture.pooled);
    anyhow::ensure!(
        pooled_l2.is_finite() && pooled_l2 > 0.0,
        "invalid pooled capture norm"
    );

    let slots = reg.protocol.placeholder_token.repeat(reg.protocol.n_slots);
    let inj_prompt = QwenRuntime::chat_prompt(SYSTEM, &format!(
        "{}\n\nA compressed latent hint from a previous solver is stored in these slots: [{slots}]\n\n{fmt}",
        item.question
    ));
    let inj_tokens = rt.encode(&inj_prompt).map_err(e)?;
    let positions = QwenRuntime::placeholder_positions(&inj_tokens, pad_id);
    anyhow::ensure!(
        positions.len() == reg.protocol.n_slots,
        "placeholder slot count mismatch"
    );
    let (_, natural_capture) = forward_capture(
        &mut rt.model,
        &inj_tokens,
        profile.inject_block,
        0..inj_tokens.len(),
        device,
    )
    .map_err(e)?;
    let natural = norms::stats(natural_capture.per_position_l2);
    anyhow::ensure!(
        natural.median.is_finite() && natural.median > 0.0,
        "invalid natural median norm"
    );

    let answer_tokens = rt.encode(&format!("#### {}", item.gold)).map_err(e)?;
    let eval = |rt: &mut QwenRuntime, spec: Option<&InjectionSpec>| -> anyhow::Result<Outcome> {
        let mut sampler = Sampler::new(Sampling::Greedy, 0);
        let output = rt
            .generate(
                &inj_tokens,
                spec,
                &mut sampler,
                reg.protocol.decoding.max_new_tokens,
                false,
            )
            .map_err(e)?;
        let correct = common::extract_answer(&output.text)
            .is_some_and(|a| common::answers_equal(&a, &item.gold));
        let nll_tokens: Vec<u32> = inj_tokens.iter().chain(&answer_tokens).copied().collect();
        let nll = teacher_forced_nll(
            &mut rt.model,
            &nll_tokens,
            inj_tokens.len()..nll_tokens.len(),
            spec,
            device,
        )
        .map_err(e)?;
        anyhow::ensure!(nll.is_finite(), "nonfinite diagnostic NLL");
        Ok(Outcome { correct, nll })
    };

    // The order below is itself registered and hash-locked.
    let baseline = eval(rt, None)?;
    let zero_spec = InjectionSpec {
        after_block: profile.inject_block,
        positions: positions.clone(),
        vector: vec![0.0; profile.hidden_size],
        scale: None,
    };
    let zero_validation = channel::validate_injection(
        &zero_spec.vector,
        zero_spec.scale,
        &zero_spec.positions,
        None,
        InjectionPolicy {
            hidden_size: profile.hidden_size,
            n_slots: reg.protocol.n_slots,
            registered_gains: &reg.protocol.gains,
            require_nonzero: false,
        },
    )?;
    let zero = eval(rt, Some(&zero_spec))?;

    let mut random_rng =
        ChaCha8Rng::seed_from_u64(reg.protocol.random_vector_seed_base + item.index as u64);
    let random_direction = common::gaussian_vec(&mut random_rng, profile.hidden_size);
    let random_l2 = norms::l2(&random_direction);
    anyhow::ensure!(
        random_l2.is_finite() && random_l2 > 0.0,
        "invalid random control norm"
    );
    let mut gain_pairs = Vec::with_capacity(reg.protocol.gains.len());
    let mut gain_rows = Vec::with_capacity(reg.protocol.gains.len());
    for &gain in &reg.protocol.gains {
        let identity_scale = natural.median / pooled_l2 * gain;
        let identity_spec = InjectionSpec {
            after_block: profile.inject_block,
            positions: positions.clone(),
            vector: capture.pooled.clone(),
            scale: Some(identity_scale),
        };
        let identity_validation = channel::validate_injection(
            &identity_spec.vector,
            identity_spec.scale,
            &identity_spec.positions,
            Some(gain),
            InjectionPolicy {
                hidden_size: profile.hidden_size,
                n_slots: reg.protocol.n_slots,
                registered_gains: &reg.protocol.gains,
                require_nonzero: true,
            },
        )?;
        let identity = eval(rt, Some(&identity_spec))?;
        let random_scale = identity_validation.effective_l2 as f32 / random_l2;
        let random_spec = InjectionSpec {
            after_block: profile.inject_block,
            positions: positions.clone(),
            vector: random_direction.clone(),
            scale: Some(random_scale),
        };
        let random_validation = channel::validate_injection(
            &random_spec.vector,
            random_spec.scale,
            &random_spec.positions,
            Some(gain),
            InjectionPolicy {
                hidden_size: profile.hidden_size,
                n_slots: reg.protocol.n_slots,
                registered_gains: &reg.protocol.gains,
                require_nonzero: true,
            },
        )?;
        anyhow::ensure!(
            relative_difference(
                identity_validation.effective_l2,
                random_validation.effective_l2
            ) <= 1e-6,
            "random control effective norm is not matched"
        );
        let random = eval(rt, Some(&random_spec))?;
        gain_rows.push(serde_json::json!({
            "gain": gain,
            "identity": outcome_json(identity),
            "matched_random": outcome_json(random),
            "validation": {
                "identity": identity_validation,
                "matched_random": random_validation,
                "relative_l2_difference": relative_difference(identity_validation.effective_l2, random_validation.effective_l2),
            }
        }));
        gain_pairs.push(GainPair { identity, random });
    }
    let first_answer = common::extract_answer(&generated.text);
    let row = serde_json::json!({
        "item": item.index,
        "gold": item.gold,
        "capture_pass": {"answer": first_answer, "generated_tokens": generated.tokens.len()},
        "capture": {"hidden_size":capture.hidden_size,"span":[capture.span.start,capture.span.end],"pooled_l2":pooled_l2,"natural_norms":natural},
        "conditions": {"baseline_uninjected":outcome_json(baseline),"zero_vector":outcome_json(zero),"gains":gain_rows},
        "zero_vector_validation": zero_validation,
    });
    Ok((
        row,
        ItemMeasurement {
            baseline,
            zero,
            gains: gain_pairs,
        },
    ))
}

fn outcome_json(o: Outcome) -> serde_json::Value {
    serde_json::json!({"correct":o.correct,"nll_gold":o.nll})
}

fn relative_difference(a: f64, b: f64) -> f64 {
    (a - b).abs() / a.abs().max(b.abs()).max(f64::MIN_POSITIVE)
}

fn summarize(reg: &Registration, rows: &[ItemMeasurement]) -> anyhow::Result<serde_json::Value> {
    let n = rows.len();
    anyhow::ensure!(n == reg.protocol.n_items);
    let baseline_correct = rows.iter().filter(|r| r.baseline.correct).count();
    let zero_correct = rows.iter().filter(|r| r.zero.correct).count();
    let mean = |values: Vec<f32>| values.iter().sum::<f32>() / values.len().max(1) as f32;
    let mut results = Vec::new();
    let mut nll_rows = Vec::new();
    for (i, &gain) in reg.protocol.gains.iter().enumerate() {
        let identity_correct = rows.iter().filter(|r| r.gains[i].identity.correct).count();
        let random_correct = rows.iter().filter(|r| r.gains[i].random.correct).count();
        let wins = rows
            .iter()
            .filter(|r| r.gains[i].identity.correct && !r.gains[i].random.correct)
            .count();
        let losses = rows
            .iter()
            .filter(|r| !r.gains[i].identity.correct && r.gains[i].random.correct)
            .count();
        results.push(GainResult {
            gain,
            identity_correct,
            random_correct,
            wins,
            losses,
        });
        nll_rows.push((
            mean(rows.iter().map(|r| r.gains[i].identity.nll).collect()),
            mean(rows.iter().map(|r| r.gains[i].random.nll).collect()),
        ));
    }
    let decision = channel::decide(reg, &results, baseline_correct, zero_correct)?;
    let gains: Vec<_> = results.iter().zip(&decision.gains).zip(&nll_rows).map(|((r,d),nll)| serde_json::json!({
        "gain":r.gain,"accuracy":{"identity":r.identity_correct,"matched_random":r.random_correct,"delta":d.accuracy_delta},
        "paired_identity_vs_random":{"wins":r.wins,"losses":r.losses,"p_one_sided_raw":d.p_raw,"p_holm_stability_family":d.p_holm_stability},
        "nll_mean":{"identity":nll.0,"matched_random":nll.1},"primary_pass":d.primary_pass,"stability_pass":d.stability_pass,
    })).collect();
    Ok(serde_json::json!({
        "n_evaluated":n,
        "accuracy":{"baseline_uninjected":baseline_correct,"zero_vector":zero_correct},
        "nll_mean":{"baseline_uninjected":mean(rows.iter().map(|r|r.baseline.nll).collect()),"zero_vector":mean(rows.iter().map(|r|r.zero.nll).collect())},
        "gains":gains,
        "gates":{"zero_vector_pass":decision.zero_vector_pass,"primary_gain_pass":decision.primary_pass,"adjacent_stability_pass":decision.adjacent_stability_pass,"profile_pass":decision.profile_pass},
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relative_norm_check_is_exact_at_match() {
        assert_eq!(relative_difference(12.5, 12.5), 0.0);
        assert!(relative_difference(100.0, 100.00001) < 1e-6);
    }
}
