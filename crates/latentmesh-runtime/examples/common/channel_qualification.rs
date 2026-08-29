//! Fail-closed registration and analysis for the M3.5 channel probe.
//!
//! There are intentionally no model or network entry points here. Unit tests
//! exercise the evidence boundary without loading a checkpoint or touching a
//! probe item.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

pub const SCHEMA_VERSION: u32 = 1;
pub const PROTOCOL_ID: &str = "latentmesh.run2.m3.5.channel-qualification.v1";

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Registration {
    pub schema_version: u32,
    pub protocol_id: String,
    pub registered_at_utc: String,
    pub registration_status: String,
    pub base_commit: String,
    pub decision: String,
    pub frozen_source: FrozenSource,
    pub profiles: Vec<ModelProfile>,
    pub protocol: Protocol,
    pub statistics: Statistics,
    pub gates: serde_json::Value,
    pub scope_limits: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct FrozenSource {
    pub adr_024: FrozenArtifact,
    pub s1a_receipt: FrozenArtifact,
    pub dataset: FrozenDataset,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct FrozenArtifact {
    pub path: String,
    #[serde(default)]
    pub sha256: String,
    #[serde(default)]
    pub sha256_at_registration: String,
}

impl FrozenArtifact {
    pub fn registered_sha256(&self) -> anyhow::Result<&str> {
        match (
            self.sha256.is_empty(),
            self.sha256_at_registration.is_empty(),
        ) {
            (false, true) => Ok(&self.sha256),
            (true, false) => Ok(&self.sha256_at_registration),
            _ => anyhow::bail!("artifact must carry exactly one registered sha256 field"),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct FrozenDataset {
    pub source: String,
    pub sha256: String,
    pub indices: Vec<usize>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ModelProfile {
    pub tag: String,
    pub model_id: String,
    pub expected_layers: usize,
    pub hidden_size: usize,
    pub capture_block: usize,
    pub inject_block: usize,
    pub purpose: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Protocol {
    pub n_items: usize,
    pub n_slots: usize,
    pub pool_span: String,
    pub pool_operation: String,
    pub placeholder_token: String,
    pub item_seed_chacha8: u64,
    pub random_vector_seed_base: u64,
    pub decoding: Decoding,
    pub base_normalization: String,
    pub gain_application: String,
    pub gains: Vec<f32>,
    pub execution_order: Vec<String>,
    pub conditions: serde_json::Value,
    pub secondary_diagnostic: String,
    pub forbidden: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Decoding {
    pub sampling: String,
    pub batch_size: usize,
    pub max_new_tokens: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Statistics {
    pub primary_gain: f32,
    pub primary_test: String,
    pub primary_alpha: f64,
    pub primary_min_accuracy_delta: f64,
    pub stability_adjacent_gains: Vec<f32>,
    pub stability_test: String,
    pub stability_family_alpha: f64,
    pub stability_min_accuracy_delta: f64,
    pub zero_vector_gate: String,
    pub nll_role: String,
    pub effect_reporting: String,
}

#[derive(Debug, Clone)]
pub struct ValidatedRegistration {
    pub registration: Registration,
    pub raw_sha256: String,
    pub path: PathBuf,
}

pub fn load_and_validate(
    path: &Path,
    expected_registration_sha256: &str,
    profile_tag: &str,
) -> anyhow::Result<ValidatedRegistration> {
    anyhow::ensure!(
        valid_sha(expected_registration_sha256),
        "compiled registration sha256 is invalid"
    );
    let bytes = std::fs::read(path).map_err(|e| {
        anyhow::anyhow!(
            "required committed preregistration {} is unavailable: {e}",
            path.display()
        )
    })?;
    let actual = sha256_hex(&bytes);
    anyhow::ensure!(
        actual == expected_registration_sha256,
        "preregistration hash mismatch for {}: got {actual}, expected {expected_registration_sha256}",
        path.display()
    );
    let registration: Registration = serde_json::from_slice(&bytes)
        .map_err(|e| anyhow::anyhow!("invalid preregistration {}: {e}", path.display()))?;
    validate_registration(&registration, profile_tag)?;
    Ok(ValidatedRegistration {
        registration,
        raw_sha256: actual,
        path: path.into(),
    })
}

pub fn validate_registration(reg: &Registration, profile_tag: &str) -> anyhow::Result<()> {
    anyhow::ensure!(
        reg.schema_version == SCHEMA_VERSION,
        "schema_version mismatch"
    );
    anyhow::ensure!(reg.protocol_id == PROTOCOL_ID, "protocol_id mismatch");
    anyhow::ensure!(
        reg.registration_status == "frozen-before-probe",
        "registration is not frozen"
    );
    anyhow::ensure!(
        valid_hex(&reg.base_commit, 40),
        "base_commit must be a full git sha"
    );
    anyhow::ensure!(reg.protocol.n_items == 40, "frozen n_items must be 40");
    anyhow::ensure!(
        reg.frozen_source.dataset.indices.len() == 40,
        "frozen item count mismatch"
    );
    anyhow::ensure!(
        reg.frozen_source
            .dataset
            .indices
            .windows(2)
            .all(|w| w[0] < w[1]),
        "frozen indices must be strictly increasing and unique"
    );
    anyhow::ensure!(
        valid_sha(reg.frozen_source.adr_024.registered_sha256()?),
        "ADR hash invalid"
    );
    anyhow::ensure!(
        valid_sha(reg.frozen_source.s1a_receipt.registered_sha256()?),
        "S1a hash invalid"
    );
    anyhow::ensure!(
        valid_sha(&reg.frozen_source.dataset.sha256),
        "dataset hash invalid"
    );
    anyhow::ensure!(reg.protocol.n_slots == 8, "frozen slots must be 8");
    anyhow::ensure!(
        reg.protocol.pool_span == "full generated span",
        "pool span mismatch"
    );
    anyhow::ensure!(
        reg.protocol.placeholder_token == "<|fim_pad|>",
        "placeholder mismatch"
    );
    anyhow::ensure!(
        reg.protocol.item_seed_chacha8 == 0x51A1,
        "item seed mismatch"
    );
    anyhow::ensure!(
        reg.protocol.random_vector_seed_base == 0x51A2_0000,
        "random seed mismatch"
    );
    anyhow::ensure!(
        reg.protocol.decoding.sampling == "greedy",
        "sampling mismatch"
    );
    anyhow::ensure!(reg.protocol.decoding.batch_size == 1, "batch size mismatch");
    anyhow::ensure!(
        reg.protocol.decoding.max_new_tokens == 400,
        "token budget mismatch"
    );
    anyhow::ensure!(
        reg.protocol.gains == [0.25, 0.5, 1.0, 2.0, 4.0],
        "gain ladder mismatch"
    );
    anyhow::ensure!(
        reg.protocol.execution_order == expected_execution_order(),
        "execution order mismatch"
    );
    anyhow::ensure!(reg.statistics.primary_gain == 1.0, "primary gain mismatch");
    anyhow::ensure!(
        reg.statistics.primary_alpha == 0.01,
        "primary alpha mismatch"
    );
    anyhow::ensure!(
        reg.statistics.primary_min_accuracy_delta == 0.15,
        "primary effect mismatch"
    );
    anyhow::ensure!(
        reg.statistics.stability_adjacent_gains == [0.5, 2.0],
        "stability gains mismatch"
    );
    anyhow::ensure!(
        reg.statistics.stability_family_alpha == 0.05,
        "stability alpha mismatch"
    );
    anyhow::ensure!(
        reg.statistics.stability_min_accuracy_delta == 0.1,
        "stability effect mismatch"
    );
    anyhow::ensure!(reg.profiles.len() == 2, "both profiles must be registered");
    for p in &reg.profiles {
        validate_profile(p)?;
    }
    let _ = profile(reg, profile_tag)?;
    Ok(())
}

pub fn profile<'a>(reg: &'a Registration, tag: &str) -> anyhow::Result<&'a ModelProfile> {
    let mut found = reg.profiles.iter().filter(|p| p.tag == tag);
    let p = found
        .next()
        .ok_or_else(|| anyhow::anyhow!("profile {tag:?} is not preregistered"))?;
    anyhow::ensure!(found.next().is_none(), "duplicate profile tag {tag:?}");
    Ok(p)
}

fn validate_profile(p: &ModelProfile) -> anyhow::Result<()> {
    let expected = match p.tag.as_str() {
        "qwen2.5-1.5b-exact-channel" => ("Qwen/Qwen2.5-1.5B-Instruct", 28, 1536, 14),
        "qwen2.5-3b-scale-oracle" => ("Qwen/Qwen2.5-3B-Instruct", 36, 2048, 18),
        other => anyhow::bail!("unrecognized profile {other:?}"),
    };
    anyhow::ensure!(p.model_id == expected.0, "model identity mismatch");
    anyhow::ensure!(p.expected_layers == expected.1, "layer count mismatch");
    anyhow::ensure!(p.hidden_size == expected.2, "hidden size mismatch");
    anyhow::ensure!(p.capture_block == expected.3, "capture block mismatch");
    anyhow::ensure!(p.inject_block == expected.3, "inject block mismatch");
    Ok(())
}

fn expected_execution_order() -> Vec<String> {
    let mut out = vec!["baseline-uninjected".into(), "zero-vector".into()];
    for gain in ["0.25", "0.5", "1.0", "2.0", "4.0"] {
        out.push(format!("identity-gain-{gain}"));
        out.push(format!("matched-random-gain-{gain}"));
    }
    out
}

pub fn validate_s1a_bytes(bytes: &[u8], reg: &Registration) -> anyhow::Result<serde_json::Value> {
    anyhow::ensure!(
        sha256_hex(bytes) == reg.frozen_source.s1a_receipt.registered_sha256()?,
        "S1a receipt artifact hash mismatch"
    );
    let receipt: serde_json::Value = serde_json::from_slice(bytes)?;
    anyhow::ensure!(receipt["stage"] == "S1a", "not an S1a receipt");
    anyhow::ensure!(
        receipt["config"]["self_pair"] == true,
        "S1a is not self-pair"
    );
    anyhow::ensure!(
        receipt["config"]["transform"] == "identity",
        "S1a transform mismatch"
    );
    anyhow::ensure!(
        receipt["config"]["slots"] == reg.protocol.n_slots,
        "S1a slot mismatch"
    );
    anyhow::ensure!(
        receipt["config"]["item_seed_chacha8"] == reg.protocol.item_seed_chacha8,
        "S1a seed mismatch"
    );
    anyhow::ensure!(
        receipt["dataset"]["sha256"] == reg.frozen_source.dataset.sha256,
        "S1a dataset mismatch"
    );
    anyhow::ensure!(
        json_indices(&receipt["dataset"]["indices"])? == reg.frozen_source.dataset.indices,
        "S1a item mismatch"
    );
    Ok(receipt)
}

fn json_indices(v: &serde_json::Value) -> anyhow::Result<Vec<usize>> {
    v.as_array()
        .ok_or_else(|| anyhow::anyhow!("indices is not an array"))?
        .iter()
        .map(|v| {
            v.as_u64()
                .map(|n| n as usize)
                .ok_or_else(|| anyhow::anyhow!("invalid item index"))
        })
        .collect()
}

fn valid_sha(value: &str) -> bool {
    valid_hex(value, 64)
}

fn valid_hex(value: &str, len: usize) -> bool {
    value.len() == len && value.bytes().all(|b| b.is_ascii_hexdigit())
}
pub fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GainResult {
    pub gain: f32,
    pub identity_correct: usize,
    pub random_correct: usize,
    pub wins: usize,
    pub losses: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GainDecision {
    pub gain: f32,
    pub p_raw: f64,
    pub p_holm_stability: Option<f64>,
    pub accuracy_delta: f64,
    pub primary_pass: bool,
    pub stability_pass: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GateDecision {
    pub gains: Vec<GainDecision>,
    pub zero_vector_pass: bool,
    pub primary_pass: bool,
    pub adjacent_stability_pass: bool,
    pub profile_pass: bool,
}

pub fn decide(
    reg: &Registration,
    results: &[GainResult],
    baseline_correct: usize,
    zero_correct: usize,
) -> anyhow::Result<GateDecision> {
    anyhow::ensure!(
        results.len() == reg.protocol.gains.len(),
        "gain result count mismatch"
    );
    for (r, gain) in results.iter().zip(&reg.protocol.gains) {
        anyhow::ensure!(r.gain == *gain, "gain result order mismatch");
        anyhow::ensure!(
            r.identity_correct <= reg.protocol.n_items && r.random_correct <= reg.protocol.n_items
        );
        anyhow::ensure!(
            r.wins + r.losses <= reg.protocol.n_items,
            "discordant count exceeds items"
        );
        anyhow::ensure!(
            r.identity_correct + r.losses == r.random_correct + r.wins,
            "paired counts are internally inconsistent"
        );
        anyhow::ensure!(
            r.wins <= r.identity_correct && r.losses <= r.random_correct,
            "discordant counts exceed condition totals"
        );
    }
    anyhow::ensure!(
        baseline_correct <= reg.protocol.n_items && zero_correct <= reg.protocol.n_items
    );
    let adjacent_indices: Vec<usize> = reg
        .statistics
        .stability_adjacent_gains
        .iter()
        .map(|gain| {
            reg.protocol
                .gains
                .iter()
                .position(|g| g == gain)
                .ok_or_else(|| anyhow::anyhow!("stability gain absent"))
        })
        .collect::<anyhow::Result<_>>()?;
    let adjacent_raw: Vec<f64> = adjacent_indices
        .iter()
        .map(|&i| sign_test_one_sided(results[i].wins, results[i].losses))
        .collect();
    let adjacent_adjusted = holm_adjust(&adjacent_raw);
    let mut gains = Vec::with_capacity(results.len());
    for (i, r) in results.iter().enumerate() {
        let p = sign_test_one_sided(r.wins, r.losses);
        let delta =
            (r.identity_correct as f64 - r.random_correct as f64) / reg.protocol.n_items as f64;
        let primary_pass = r.gain == reg.statistics.primary_gain
            && p < reg.statistics.primary_alpha
            && delta >= reg.statistics.primary_min_accuracy_delta;
        let adjacent_pos = adjacent_indices.iter().position(|&idx| idx == i);
        let p_holm = adjacent_pos.map(|j| adjacent_adjusted[j]);
        let stability_pass = p_holm.is_some_and(|adjusted| {
            adjusted < reg.statistics.stability_family_alpha
                && delta >= reg.statistics.stability_min_accuracy_delta
        });
        gains.push(GainDecision {
            gain: r.gain,
            p_raw: p,
            p_holm_stability: p_holm,
            accuracy_delta: delta,
            primary_pass,
            stability_pass,
        });
    }
    let zero_vector_pass = 2 * zero_correct >= baseline_correct;
    let primary_pass = gains.iter().any(|g| g.primary_pass);
    let adjacent_stability_pass = gains.iter().any(|g| g.stability_pass);
    Ok(GateDecision {
        gains,
        zero_vector_pass,
        primary_pass,
        adjacent_stability_pass,
        profile_pass: zero_vector_pass && primary_pass && adjacent_stability_pass,
    })
}

pub fn holm_adjust(raw: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..raw.len()).collect();
    order.sort_by(|&a, &b| raw[a].total_cmp(&raw[b]).then(a.cmp(&b)));
    let mut out = vec![1.0; raw.len()];
    let mut prior: f64 = 0.0;
    for (rank, &idx) in order.iter().enumerate() {
        prior = prior.max(((raw.len() - rank) as f64 * raw[idx]).min(1.0));
        out[idx] = prior;
    }
    out
}

fn sign_test_one_sided(wins: usize, losses: usize) -> f64 {
    let n = wins + losses;
    if n == 0 {
        return 1.0;
    }
    (wins..=n)
        .map(|k| (ln_choose(n, k) - n as f64 * std::f64::consts::LN_2).exp())
        .sum::<f64>()
        .min(1.0)
}

fn ln_choose(n: usize, k: usize) -> f64 {
    ln_fact(n) - ln_fact(k) - ln_fact(n - k)
}

fn ln_fact(n: usize) -> f64 {
    (2..=n).map(|i| (i as f64).ln()).sum()
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct InjectionValidation {
    pub vector_length_matches: bool,
    pub positions_match_slots: bool,
    pub positions_unique: bool,
    pub vector_finite: bool,
    pub scale_finite: bool,
    pub gain_registered: bool,
    pub effective_l2: f64,
    pub effective_l2_valid: bool,
    pub pass: bool,
}

/// Validate the complete residual-edit boundary before a model sees it.
/// `require_nonzero=false` is reserved for the registered zero-vector arm.
pub fn validate_injection(
    vector: &[f32],
    scale: Option<f32>,
    positions: &[usize],
    hidden_size: usize,
    n_slots: usize,
    gain: Option<f32>,
    registered_gains: &[f32],
    require_nonzero: bool,
) -> anyhow::Result<InjectionValidation> {
    let vector_length_matches = vector.len() == hidden_size;
    let positions_match_slots = positions.len() == n_slots;
    let mut sorted = positions.to_vec();
    sorted.sort_unstable();
    let positions_unique = sorted.windows(2).all(|w| w[0] != w[1]);
    let vector_finite = vector.iter().all(|v| v.is_finite());
    let scale_finite = scale.map_or(true, f32::is_finite);
    let gain_registered = gain.map_or(true, |g| registered_gains.contains(&g));
    let s = scale.unwrap_or(1.0) as f64;
    let effective_l2 = vector
        .iter()
        .map(|v| (*v as f64 * s).powi(2))
        .sum::<f64>()
        .sqrt();
    let effective_l2_valid = effective_l2.is_finite() && (!require_nonzero || effective_l2 > 0.0);
    let pass = vector_length_matches
        && positions_match_slots
        && positions_unique
        && vector_finite
        && scale_finite
        && gain_registered
        && effective_l2_valid;
    let evidence = InjectionValidation {
        vector_length_matches,
        positions_match_slots,
        positions_unique,
        vector_finite,
        scale_finite,
        gain_registered,
        effective_l2,
        effective_l2_valid,
        pass,
    };
    anyhow::ensure!(pass, "injection boundary validation failed: {evidence:?}");
    Ok(evidence)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);

    fn fixture() -> Registration {
        serde_json::from_value(serde_json::json!({
            "schema_version":1,"protocol_id":PROTOCOL_ID,"registered_at_utc":"2026-08-28T00:00:00Z","registration_status":"frozen-before-probe",
            "base_commit":"78adcf01627e0e1d11b5d180239726dea538ca4c","decision":"pause",
            "frozen_source":{"adr_024":{"path":"docs/adr/024.md","sha256_at_registration":"aa".repeat(32)},"s1a_receipt":{"path":"crates/latentmesh-runtime/receipts/s1a.json","sha256":"bb".repeat(32)},"dataset":{"source":"url","sha256":"cc".repeat(32),"indices":(0..40).collect::<Vec<_>>()}},
            "profiles":[{"tag":"qwen2.5-1.5b-exact-channel","model_id":"Qwen/Qwen2.5-1.5B-Instruct","expected_layers":28,"hidden_size":1536,"capture_block":14,"inject_block":14,"purpose":"exact"},{"tag":"qwen2.5-3b-scale-oracle","model_id":"Qwen/Qwen2.5-3B-Instruct","expected_layers":36,"hidden_size":2048,"capture_block":18,"inject_block":18,"purpose":"scale"}],
            "protocol":{"n_items":40,"n_slots":8,"pool_span":"full generated span","pool_operation":"mean","placeholder_token":"<|fim_pad|>","item_seed_chacha8":20897,"random_vector_seed_base":1369571328,"decoding":{"sampling":"greedy","batch_size":1,"max_new_tokens":400},"base_normalization":"natural median","gain_application":"after normalization","gains":[0.25,0.5,1.0,2.0,4.0],"execution_order":expected_execution_order(),"conditions":{},"secondary_diagnostic":"NLL","forbidden":[]},
            "statistics":{"primary_gain":1.0,"primary_test":"sign","primary_alpha":0.01,"primary_min_accuracy_delta":0.15,"stability_adjacent_gains":[0.5,2.0],"stability_test":"holm","stability_family_alpha":0.05,"stability_min_accuracy_delta":0.1,"zero_vector_gate":"half","nll_role":"diagnostic","effect_reporting":"all"},
            "gates":{},"scope_limits":[]
        })).unwrap()
    }

    fn temp(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "m35-{label}-{}-{}.json",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ))
    }

    #[test]
    fn missing_file_fails_closed() {
        assert!(load_and_validate(
            &temp("missing"),
            &"00".repeat(32),
            "qwen2.5-1.5b-exact-channel"
        )
        .unwrap_err()
        .to_string()
        .contains("unavailable"));
    }
    #[test]
    fn hash_mismatch_fails_closed() {
        let p = temp("hash");
        std::fs::write(&p, serde_json::to_vec(&fixture()).unwrap()).unwrap();
        assert!(
            load_and_validate(&p, &"00".repeat(32), "qwen2.5-1.5b-exact-channel")
                .unwrap_err()
                .to_string()
                .contains("hash mismatch")
        );
        let _ = std::fs::remove_file(p);
    }
    #[test]
    fn canonical_passes_without_side_effects() {
        let p = temp("pass");
        let bytes = serde_json::to_vec(&fixture()).unwrap();
        let hash = sha256_hex(&bytes);
        std::fs::write(&p, &bytes).unwrap();
        assert_eq!(
            load_and_validate(&p, &hash, "qwen2.5-3b-scale-oracle")
                .unwrap()
                .raw_sha256,
            hash
        );
        let _ = std::fs::remove_file(p);
    }
    #[test]
    fn item_and_model_mismatches_fail_closed() {
        let mut r = fixture();
        r.frozen_source.dataset.indices[1] = 0;
        assert!(validate_registration(&r, "qwen2.5-1.5b-exact-channel").is_err());
        let mut r = fixture();
        r.profiles[0].model_id = "lookalike".into();
        assert!(validate_registration(&r, "qwen2.5-1.5b-exact-channel")
            .unwrap_err()
            .to_string()
            .contains("model identity"));
    }
    #[test]
    fn source_hash_mismatch_fails_closed() {
        assert!(validate_s1a_bytes(b"{}", &fixture())
            .unwrap_err()
            .to_string()
            .contains("artifact hash"));
    }
    #[test]
    fn gate_uses_primary_and_registered_adjacent_only() {
        let r = fixture();
        let row = |gain, strong| GainResult {
            gain,
            identity_correct: if strong { 30 } else { 20 },
            random_correct: 20,
            wins: if strong { 10 } else { 0 },
            losses: 0,
        };
        let d = decide(
            &r,
            &[
                row(0.25, false),
                row(0.5, true),
                row(1.0, true),
                row(2.0, false),
                row(4.0, false),
            ],
            24,
            12,
        )
        .unwrap();
        assert!(d.profile_pass);
        let d = decide(
            &r,
            &[
                row(0.25, true),
                row(0.5, false),
                row(1.0, true),
                row(2.0, false),
                row(4.0, false),
            ],
            24,
            12,
        )
        .unwrap();
        assert!(!d.profile_pass);
    }
    #[test]
    fn zero_vector_gate_is_exactly_half_baseline() {
        let r = fixture();
        let row = |gain| GainResult {
            gain,
            identity_correct: 20,
            random_correct: 20,
            wins: 0,
            losses: 0,
        };
        let results = [row(0.25), row(0.5), row(1.0), row(2.0), row(4.0)];
        assert!(decide(&r, &results, 24, 12).unwrap().zero_vector_pass);
        assert!(!decide(&r, &results, 23, 11).unwrap().zero_vector_pass);
    }
    #[test]
    fn inconsistent_paired_counts_fail_closed() {
        let r = fixture();
        let mut results = [0.25, 0.5, 1.0, 2.0, 4.0].map(|gain| GainResult {
            gain,
            identity_correct: 20,
            random_correct: 20,
            wins: 0,
            losses: 0,
        });
        results[2].wins = 1;
        assert!(decide(&r, &results, 20, 20)
            .unwrap_err()
            .to_string()
            .contains("internally inconsistent"));
    }
    #[test]
    fn holm_values_are_deterministic() {
        assert_eq!(holm_adjust(&[0.01, 0.04, 0.03]), vec![0.03, 0.06, 0.06]);
    }
    #[test]
    fn injection_boundary_rejects_nonfinite_wrong_shape_and_unknown_gain() {
        let gains = [0.25, 0.5, 1.0, 2.0, 4.0];
        assert!(validate_injection(
            &[1.0, 2.0],
            Some(1.0),
            &[1, 2],
            2,
            2,
            Some(1.0),
            &gains,
            true
        )
        .is_ok());
        assert!(
            validate_injection(&[1.0], Some(1.0), &[1, 2], 2, 2, Some(1.0), &gains, true).is_err()
        );
        assert!(validate_injection(
            &[1.0, f32::NAN],
            Some(1.0),
            &[1, 2],
            2,
            2,
            Some(1.0),
            &gains,
            true
        )
        .is_err());
        assert!(validate_injection(
            &[1.0, 2.0],
            Some(f32::INFINITY),
            &[1, 2],
            2,
            2,
            Some(1.0),
            &gains,
            true
        )
        .is_err());
        assert!(
            validate_injection(&[1.0, 2.0], Some(1.0), &[1], 2, 2, Some(1.0), &gains, true)
                .is_err()
        );
        assert!(validate_injection(
            &[1.0, 2.0],
            Some(1.0),
            &[1, 2],
            2,
            2,
            Some(3.0),
            &gains,
            true
        )
        .is_err());
        assert!(validate_injection(&[0.0, 0.0], None, &[1, 2], 2, 2, None, &gains, true).is_err());
        assert!(validate_injection(&[0.0, 0.0], None, &[1, 2], 2, 2, None, &gains, false).is_ok());
    }
}
