//! `UtilityDensity` and `RepresentationRouter` — the metric and routing
//! surface for choosing HOW (or whether) to communicate between agents.
//!
//! # The thesis this module serves
//!
//! The old thesis was "latent transfer makes the remote agent think better".
//! Our own receipts overturned that as a universal claim:
//! `docs/research/048-run2-final-synthesis.md` measured text moving decisions
//! +0.512 on the same receiver/population where single-layer latent moved
//! ~0 (p = 0.72, fully powered — a null WITH power, not an absence of
//! evidence; see `docs/research/050-sota-verdict-vs-measured-data.md`).
//! A naive system reasons "latent is efficient, therefore send latent".
//! That is evidenced wrong. The correct policy is: **choose the
//! representation that actually causes useful downstream behaviour** — text
//! when text wins, latent when latent wins, a symbolic delta when bandwidth
//! dominates, and NO MESSAGE when communication adds no value at all. This
//! module is the piece that lets a runtime discover which representation
//! earns its cost, instead of assuming one.
//!
//! # Crate-boundary mapping — this is NOT a reimplementation
//!
//! [`ContentGain`] and [`AgentGain`] are typed re-namings of comparisons
//! `latentmesh-gate::causal` already computes and has already run (ADR-003's
//! five-control test: zero / random / mismatched / self_generated /
//! text_equivalent, admission requiring the WORST control to be beaten).
//! Concretely, for a `latentmesh_gate::causal::EdgeTrial`:
//!
//! | This module | Gate comparison | Isolates |
//! |---|---|---|
//! | [`ContentGain`] | `real` vs `mismatched` | value of correct CONTENT |
//! | [`AgentGain`] | `real` vs `self_generated` | value of consulting ANOTHER agent at all |
//!
//! This crate has **zero dependency on `latentmesh-gate`** (crate-level
//! Invariant 1: latent state is non-authoritative, and nothing here may add
//! an authority type). Callers run the causal test elsewhere and hand the
//! resulting deltas in as plain numbers.
//!
//! [`UtilityDensity`] is the genuinely missing piece: a per-resource
//! normalisation that lets gains measured across incompatible physical units
//! (bytes, seconds, joules, an abstract risk score) be compared on one
//! scale. [`RepresentationRouter`] then scores every candidate
//! representation mode — including `None` — and picks the best.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Value attributable to CONTENT correctness, at matched compute: the
/// causal-test outcome for `real` sender state minus the outcome for a
/// `mismatched` one. Maps to `latentmesh_gate::causal::EdgeTrial::real` vs
/// `::mismatched` — see the module doc's mapping table. A caller who has NOT
/// run that test has no basis for this value; see [`GainMeasurement`] for
/// how the router distinguishes "never tested" from "tested and found zero".
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ContentGain(pub f64);

/// Value attributable to consulting ANOTHER agent at all, at matched
/// compute: the causal-test outcome for another agent's state minus the
/// outcome for the receiver's own self-generated substitute. Maps to
/// `latentmesh_gate::causal::EdgeTrial::real` vs `::self_generated`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct AgentGain(pub f64);

/// Raw, per-resource measurements for one representation-mode candidate, in
/// their natural physical units. Not yet normalised — see [`CostScale`].
/// `risk` is unitless on the caller's own scale (e.g. a residual-risk score
/// from `latentmesh-reasoning::budget::RiskClass`'s ceiling, or any other
/// caller-defined measure); this module does not interpret it.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResourceCost {
    pub bytes: f64,
    pub latency_seconds: f64,
    pub energy_joules: f64,
    pub risk: f64,
}

impl ResourceCost {
    /// The zero cost of communicating nothing.
    pub const ZERO: ResourceCost = ResourceCost {
        bytes: 0.0,
        latency_seconds: 0.0,
        energy_joules: 0.0,
        risk: 0.0,
    };

    fn is_finite_and_nonnegative(&self) -> bool {
        [
            self.bytes,
            self.latency_seconds,
            self.energy_joules,
            self.risk,
        ]
        .into_iter()
        .all(|x| x.is_finite() && x >= 0.0)
    }
}

/// A reference scale per resource that a raw [`ResourceCost`] is divided by
/// before weighting, so bytes/seconds/joules/risk land on a comparable,
/// dimensionless range before they are added together.
///
/// **Why not a literal product of physical units.** An earlier framing of
/// this score multiplied raw `bytes * seconds * joules` in the denominator.
/// That is wrong on two counts: (1) the *unit* of the resulting quantity is
/// meaningless — "gain per byte-second-joule" answers no question anyone
/// asked — and its magnitude is an artifact of which unit system was chosen
/// (kilobytes vs bytes changes the score by 1000x with no change in what was
/// actually communicated); (2) a product denominator EXPLODES toward
/// infinity as any single term approaches zero, so a mode that is merely
/// fast (near-zero latency) would score as effectively free regardless of
/// its bandwidth or risk cost, which is not the intended tradeoff. The
/// additive form after normalisation does not have either failure: each
/// term is bounded by how large that resource's use is relative to its own
/// reference scale, and a zero in one dimension does not erase the other
/// three. Do not "simplify" this back to a product.
///
/// `CostScale` is deliberately NOT `Default` — a default would itself be a
/// fabricated "typical" scale (ADR-040's standing rule against unvalidated
/// compute-vs-value constants). The caller supplies scales that are
/// structural facts about their deployment (e.g. the transport's MTU, the
/// caller's own latency budget, a battery capacity, the gate's own risk
/// ceiling), not a claim about what representation modes cost in general.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CostScale {
    pub bytes: f64,
    pub latency_seconds: f64,
    pub energy_joules: f64,
    pub risk: f64,
}

impl CostScale {
    fn is_valid(&self) -> bool {
        [
            self.bytes,
            self.latency_seconds,
            self.energy_joules,
            self.risk,
        ]
        .into_iter()
        .all(|x| x.is_finite() && x > 0.0)
    }
}

/// Weights `lambda_b, lambda_l, lambda_e, lambda_r` applied to the
/// NORMALISED cost terms in [`UtilityDensity`]'s denominator. The
/// [`Default`] below is an **unvalidated starting point** (equal weighting)
/// — not a claim that bandwidth/latency/energy/risk trade off 1:1:1:1 for
/// any measured task. ADR-040's standing rule: any compute-vs-value claim
/// needs a power calculation and a measurement first, and none has been
/// run for these weights. Callers with a calibrated policy should supply
/// their own.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CostWeights {
    pub bandwidth: f64,
    pub latency: f64,
    pub energy: f64,
    pub risk: f64,
}

impl Default for CostWeights {
    fn default() -> Self {
        Self {
            bandwidth: 0.25,
            latency: 0.25,
            energy: 0.25,
            risk: 0.25,
        }
    }
}

/// Why [`utility_density`] could not produce a score.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum UtilityDensityError {
    /// A [`CostScale`] field was non-finite or `<= 0`; every scale must be a
    /// genuine positive reference, or normalisation is undefined.
    InvalidScale,
    /// A [`ResourceCost`] field was negative or non-finite.
    InvalidCost,
    /// All weighted, normalised cost terms were exactly zero — the
    /// denominator is zero. This means the candidate claims a nonzero gain
    /// for literally zero bytes, latency, energy, AND risk, which is not a
    /// physically real communication (that candidate should be represented
    /// as [`RepresentationMode::None`] instead, whose score is defined as
    /// exactly `0.0` without going through this formula).
    ZeroCost,
}

impl fmt::Display for UtilityDensityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UtilityDensityError::InvalidScale => {
                write!(f, "CostScale fields must all be finite and > 0.0")
            }
            UtilityDensityError::InvalidCost => {
                write!(f, "ResourceCost fields must all be finite and >= 0.0")
            }
            UtilityDensityError::ZeroCost => write!(
                f,
                "all weighted normalised cost terms are zero; use RepresentationMode::None instead"
            ),
        }
    }
}

impl std::error::Error for UtilityDensityError {}

/// `Score(m) = delta_V(m) / (lambda_b*B(m) + lambda_l*L(m) + lambda_e*E(m) + lambda_r*R(m))`,
/// each of `B, L, E, R` normalised to `raw / scale` before weighting — a
/// normalised ADDITIVE cost denominator, not a literal product of physical
/// units (see [`CostScale`]'s doc comment for why the product form is
/// wrong). `delta_v` is the caller-supplied causal gain for this candidate
/// (e.g. a [`ContentGain`] or [`AgentGain`], or any other measured
/// delta-value) — this function does not compute it and does not know which
/// of the two it is; it only turns gain-per-nothing into gain-per-cost.
pub fn utility_density(
    delta_v: f64,
    cost: &ResourceCost,
    scale: &CostScale,
    weights: &CostWeights,
) -> Result<f64, UtilityDensityError> {
    if !scale.is_valid() {
        return Err(UtilityDensityError::InvalidScale);
    }
    if !cost.is_finite_and_nonnegative() {
        return Err(UtilityDensityError::InvalidCost);
    }
    let denom = weights.bandwidth * (cost.bytes / scale.bytes)
        + weights.latency * (cost.latency_seconds / scale.latency_seconds)
        + weights.energy * (cost.energy_joules / scale.energy_joules)
        + weights.risk * (cost.risk / scale.risk);
    if denom <= 0.0 {
        return Err(UtilityDensityError::ZeroCost);
    }
    Ok(delta_v / denom)
}

/// Candidate representation modes a [`RepresentationRouter`] chooses among.
/// `None` is a first-class candidate — "do not communicate" is the correct
/// answer whenever no other mode's gain exceeds its cost, and a router that
/// cannot choose silence is not doing its job.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RepresentationMode {
    /// Communicate nothing.
    None,
    Text,
    /// A compact structured diff over shared symbolic state, not raw text
    /// or raw activations.
    SymbolicDelta,
    /// A sparse subset of feature activations.
    SparseFeature,
    /// A short run of hidden-state vectors prepended as soft-prompt context.
    HiddenPrefix,
    /// Activations from more than one layer (ADR-037/M5X's unblocked
    /// configuration — see `docs/research/050-sota-verdict-vs-measured-data.md`
    /// on why single-layer injection is a distinct, already-tested case).
    MultiLayerLatent,
    /// A serialized key/value attention cache.
    KvState,
    /// A full recurrent-model checkpoint (ADR-041's `ContextCheckpoint`-like
    /// state), not just its output.
    RecurrentCheckpoint,
}

/// Whether this candidate's causal gain has ever been measured. Conflating
/// "we never tested this mode" with "this mode measured as zero gain" is
/// exactly the error this project's ladder spent 90 commits learning to
/// avoid (see `docs/research/048-run2-final-synthesis.md`) — an unmeasured
/// mode must not silently lose to a measured-zero one, or worse, silently
/// win by being skipped from comparison entirely. The router keeps every
/// [`GainMeasurement::Unmeasured`] candidate visible in
/// [`RouteDecision::considered`] but excludes it from scoring: an unmeasured
/// mode can never be *selected*, only flagged as worth testing.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum GainMeasurement {
    /// No causal test (e.g. `latentmesh_gate::causal::verify_edge`) has been
    /// run for this mode against this receiver/task population.
    Unmeasured,
    /// A causal test was run and produced this delta_v — which may itself
    /// be zero, negative, or positive.
    Measured(f64),
}

/// One representation-mode candidate offered to a [`RepresentationRouter`].
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModeCandidate {
    pub mode: RepresentationMode,
    pub gain: GainMeasurement,
    pub cost: ResourceCost,
}

/// The outcome of scoring one [`ModeCandidate`] within a [`RouteDecision`].
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum ScoreOutcome {
    /// Excluded from selection — see [`GainMeasurement::Unmeasured`].
    Unmeasured,
    /// [`utility_density`] returned this error for the candidate; also
    /// excluded from selection.
    Invalid,
    /// `Score(m)`, eligible to win.
    Scored(f64),
}

/// Full result of [`RepresentationRouter::route`]: the winning mode, its
/// score, and a per-candidate trace so a caller can audit why every
/// alternative lost (or was never eligible) rather than trusting a bare
/// enum value.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RouteDecision {
    pub selected: RepresentationMode,
    /// `0.0` when `selected` is [`RepresentationMode::None`] — see
    /// [`RepresentationRouter::route`]'s doc for why `None` bypasses
    /// [`utility_density`] entirely rather than risking a `0/0`.
    pub selected_score: f64,
    /// Every candidate the router was given (`None` is NOT included here —
    /// it is implicit, always available, and always scores exactly `0.0`),
    /// paired with how it scored.
    pub considered: Vec<(RepresentationMode, ScoreOutcome)>,
}

/// Scores [`ModeCandidate`]s and picks the best, using a fixed
/// [`CostScale`]/[`CostWeights`] pair. Holds only configuration — no clock,
/// no thread, no model — so [`RepresentationRouter::route`] is a pure
/// function of its arguments: identical inputs produce an identical
/// [`RouteDecision`] every time.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RepresentationRouter {
    scale: CostScale,
    weights: CostWeights,
}

impl RepresentationRouter {
    pub fn new(scale: CostScale, weights: CostWeights) -> Self {
        Self { scale, weights }
    }

    pub fn scale(&self) -> &CostScale {
        &self.scale
    }

    pub fn weights(&self) -> &CostWeights {
        &self.weights
    }

    /// Score every candidate and select the best, where [`RepresentationMode::None`]
    /// is always implicitly available at a fixed score of exactly `0.0` — the
    /// baseline "communicating nothing costs nothing and gains nothing"
    /// reference point. `None` is scored directly rather than through
    /// [`utility_density`] because its cost is [`ResourceCost::ZERO`] in
    /// every dimension, which would make the formula's denominator zero
    /// (`0/0` is undefined, not `0.0`) — bypassing the formula for this one
    /// case is what makes "no message" a real, always-computable answer
    /// instead of an error case the router has to special-case around at
    /// every call site.
    ///
    /// Ties are broken in favor of `None` first (silence is the conservative
    /// default when nothing strictly beats it), then in favor of the
    /// earliest-listed candidate in `candidates` (deterministic, no implicit
    /// preference among the non-`None` modes).
    pub fn route(&self, candidates: &[ModeCandidate]) -> RouteDecision {
        let mut considered = Vec::with_capacity(candidates.len());
        let mut best_mode = RepresentationMode::None;
        let mut best_score = 0.0f64;

        for candidate in candidates {
            let outcome = match candidate.gain {
                GainMeasurement::Unmeasured => ScoreOutcome::Unmeasured,
                GainMeasurement::Measured(delta_v) => {
                    match utility_density(delta_v, &candidate.cost, &self.scale, &self.weights) {
                        Ok(score) => {
                            if score > best_score {
                                best_score = score;
                                best_mode = candidate.mode;
                            }
                            ScoreOutcome::Scored(score)
                        }
                        Err(_) => ScoreOutcome::Invalid,
                    }
                }
            };
            considered.push((candidate.mode, outcome));
        }

        RouteDecision {
            selected: best_mode,
            selected_score: best_score,
            considered,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scale() -> CostScale {
        CostScale {
            bytes: 1000.0,
            latency_seconds: 1.0,
            energy_joules: 1.0,
            risk: 1.0,
        }
    }

    fn cost(bytes: f64, latency: f64, energy: f64, risk: f64) -> ResourceCost {
        ResourceCost {
            bytes,
            latency_seconds: latency,
            energy_joules: energy,
            risk,
        }
    }

    fn router() -> RepresentationRouter {
        RepresentationRouter::new(scale(), CostWeights::default())
    }

    #[test]
    fn utility_density_is_deterministic() {
        let c = cost(200.0, 0.1, 0.05, 0.1);
        let a = utility_density(0.5, &c, &scale(), &CostWeights::default());
        let b = utility_density(0.5, &c, &scale(), &CostWeights::default());
        assert_eq!(a, b);
    }

    #[test]
    fn utility_density_scales_additively_not_multiplicatively() {
        // Two candidates with the same total normalised cost under equal
        // weights but reached via different resource mixes should score
        // the same — a product denominator would NOT have this property
        // (it would blow up whichever term is near zero).
        let s = scale();
        let w = CostWeights::default();
        let heavy_bytes = cost(2000.0, 0.0, 0.0, 0.0); // normalised: 2.0
        let heavy_latency = cost(0.0, 2.0, 0.0, 0.0); // normalised: 2.0
        let a = utility_density(1.0, &heavy_bytes, &s, &w).unwrap();
        let b = utility_density(1.0, &heavy_latency, &s, &w).unwrap();
        assert!((a - b).abs() < 1e-12, "{a} vs {b}");
    }

    #[test]
    fn zero_cost_nonzero_gain_is_rejected_not_infinite() {
        let err = utility_density(1.0, &ResourceCost::ZERO, &scale(), &CostWeights::default())
            .unwrap_err();
        assert_eq!(err, UtilityDensityError::ZeroCost);
    }

    #[test]
    fn invalid_scale_and_cost_are_rejected() {
        let bad_scale = CostScale {
            bytes: 0.0,
            ..scale()
        };
        assert_eq!(
            utility_density(
                1.0,
                &cost(1.0, 1.0, 1.0, 1.0),
                &bad_scale,
                &CostWeights::default()
            )
            .unwrap_err(),
            UtilityDensityError::InvalidScale
        );
        let bad_cost = cost(-1.0, 1.0, 1.0, 1.0);
        assert_eq!(
            utility_density(1.0, &bad_cost, &scale(), &CostWeights::default()).unwrap_err(),
            UtilityDensityError::InvalidCost
        );
    }

    #[test]
    fn router_route_is_pure() {
        let r = router();
        let candidates = vec![
            ModeCandidate {
                mode: RepresentationMode::Text,
                gain: GainMeasurement::Measured(0.5),
                cost: cost(300.0, 0.2, 0.1, 0.1),
            },
            ModeCandidate {
                mode: RepresentationMode::MultiLayerLatent,
                gain: GainMeasurement::Measured(0.05),
                cost: cost(50.0, 0.05, 0.02, 0.05),
            },
        ];
        assert_eq!(r.route(&candidates), r.route(&candidates));
    }

    /// Direct restatement of this repo's own receipt: text beat single-layer
    /// latent on decision movement while latent moved ~0 (p = 0.72). Synthetic
    /// numbers standing in for that shape, not the real study's magnitudes —
    /// this test asserts the ROUTER'S LOGIC picks the winning representation,
    /// not any calibrated real-world figure.
    #[test]
    fn router_picks_text_over_latent_when_text_wins_on_the_receipts_shape() {
        let r = router();
        let candidates = vec![
            ModeCandidate {
                mode: RepresentationMode::Text,
                gain: GainMeasurement::Measured(0.512),
                cost: cost(400.0, 0.3, 0.1, 0.1),
            },
            ModeCandidate {
                mode: RepresentationMode::MultiLayerLatent,
                gain: GainMeasurement::Measured(0.0),
                cost: cost(50.0, 0.05, 0.02, 0.05),
            },
        ];
        let decision = r.route(&candidates);
        assert_eq!(decision.selected, RepresentationMode::Text);
        assert!(decision.selected_score > 0.0);
    }

    #[test]
    fn none_wins_when_every_measured_candidate_has_nonpositive_gain() {
        let r = router();
        let candidates = vec![
            ModeCandidate {
                mode: RepresentationMode::Text,
                gain: GainMeasurement::Measured(-0.1),
                cost: cost(400.0, 0.3, 0.1, 0.1),
            },
            ModeCandidate {
                mode: RepresentationMode::HiddenPrefix,
                gain: GainMeasurement::Measured(0.0),
                cost: cost(50.0, 0.05, 0.02, 0.05),
            },
        ];
        let decision = r.route(&candidates);
        assert_eq!(decision.selected, RepresentationMode::None);
        assert_eq!(decision.selected_score, 0.0);
    }

    #[test]
    fn unmeasured_candidates_cannot_win_and_are_reported_distinctly_from_measured_zero() {
        let r = router();
        let candidates = vec![
            ModeCandidate {
                mode: RepresentationMode::KvState,
                gain: GainMeasurement::Unmeasured,
                cost: cost(100.0, 0.1, 0.05, 0.05),
            },
            ModeCandidate {
                mode: RepresentationMode::SparseFeature,
                gain: GainMeasurement::Measured(0.0),
                cost: cost(80.0, 0.08, 0.03, 0.03),
            },
        ];
        let decision = r.route(&candidates);
        // Neither strictly beats the implicit None (0.0 gain, 0.0 cost) baseline.
        assert_eq!(decision.selected, RepresentationMode::None);
        assert_eq!(
            decision.considered[0].1,
            ScoreOutcome::Unmeasured,
            "an untested mode must not be conflated with a measured-zero one"
        );
        assert!(matches!(decision.considered[1].1, ScoreOutcome::Scored(s) if s == 0.0));
    }

    #[test]
    fn invalid_candidate_cost_is_excluded_not_selected() {
        let r = router();
        let candidates = vec![ModeCandidate {
            mode: RepresentationMode::SymbolicDelta,
            gain: GainMeasurement::Measured(5.0),
            cost: cost(-10.0, 0.1, 0.1, 0.1),
        }];
        let decision = r.route(&candidates);
        assert_eq!(decision.selected, RepresentationMode::None);
        assert_eq!(decision.considered[0].1, ScoreOutcome::Invalid);
    }

    #[test]
    fn higher_normalised_cost_for_the_same_gain_never_scores_higher() {
        let r = router();
        let low_cost = ModeCandidate {
            mode: RepresentationMode::RecurrentCheckpoint,
            gain: GainMeasurement::Measured(1.0),
            cost: cost(10.0, 0.01, 0.01, 0.01),
        };
        let high_cost = ModeCandidate {
            mode: RepresentationMode::RecurrentCheckpoint,
            gain: GainMeasurement::Measured(1.0),
            cost: cost(500.0, 0.5, 0.5, 0.5),
        };
        let low = r.route(std::slice::from_ref(&low_cost));
        let high = r.route(std::slice::from_ref(&high_cost));
        assert!(low.selected_score > high.selected_score);
    }

    #[test]
    fn ties_are_broken_toward_none_and_then_toward_earlier_candidate() {
        let r = router();
        // A candidate scoring exactly 0.0 (zero gain, positive cost) must
        // NOT beat the implicit None baseline, even though the formula
        // would compute score == 0.0 for it too.
        let zero_gain = vec![ModeCandidate {
            mode: RepresentationMode::Text,
            gain: GainMeasurement::Measured(0.0),
            cost: cost(100.0, 0.1, 0.1, 0.1),
        }];
        assert_eq!(r.route(&zero_gain).selected, RepresentationMode::None);

        // Two candidates with identical positive scores: earliest wins.
        let c = cost(100.0, 0.1, 0.05, 0.05);
        let tied = vec![
            ModeCandidate {
                mode: RepresentationMode::Text,
                gain: GainMeasurement::Measured(1.0),
                cost: c,
            },
            ModeCandidate {
                mode: RepresentationMode::SymbolicDelta,
                gain: GainMeasurement::Measured(1.0),
                cost: c,
            },
        ];
        assert_eq!(r.route(&tied).selected, RepresentationMode::Text);
    }

    #[test]
    fn content_gain_and_agent_gain_are_plain_typed_deltas() {
        // These are re-namings of `latentmesh-gate::causal::EdgeTrial`
        // comparisons (real-vs-mismatched, real-vs-self_generated) — no
        // statistics live here, just the type and the field access.
        let content = ContentGain(0.3);
        let agent = AgentGain(-0.05);
        assert_eq!(content.0, 0.3);
        assert_eq!(agent.0, -0.05);
    }

    #[test]
    fn route_decision_serializes_round_trip() {
        let r = router();
        let candidates = vec![ModeCandidate {
            mode: RepresentationMode::Text,
            gain: GainMeasurement::Measured(0.4),
            cost: cost(200.0, 0.1, 0.05, 0.05),
        }];
        let decision = r.route(&candidates);
        let json = serde_json::to_string(&decision).expect("serialize");
        let back: RouteDecision = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(decision, back);
    }
}
