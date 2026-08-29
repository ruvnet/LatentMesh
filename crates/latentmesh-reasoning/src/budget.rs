//! Adaptive reasoning-budget controller (ADR-041 §6 Algorithm 2, §8
//! Algorithm 4). Two pure, deterministic responsibilities:
//!
//! 1. [`ReasoningBudgetController::route`] — difficulty/risk signals →
//!    [`ReasoningBudget`] (Algorithm 4: score `b`, then `R_target`).
//! 2. [`ReasoningBudgetController::evaluate_step`] — per iteration, decide
//!    stop/continue and expose *which* rule fired (Algorithm 2's stop
//!    policy, §6.4: convergence, verifier confidence, risk class, latency
//!    pressure, compute ceiling — never stop on convergence alone, §6.3).
//!
//! Neither method owns a model, a clock, or a thread: callers measure
//! convergence, confidence, and elapsed-latency fraction and pass them in.
//! Same inputs ⇒ same decision (`deterministic_route_and_evaluate_are_pure`).
//!
//! **On `BudgetTier`:** a compute-allocation label, not an accuracy
//! promise. ADR-041 cites BDH-CQ's reported 21%/27%/29.5% ARC pass@2 across
//! LOW/MEDIUM/HIGH effort — one paper's result on a proprietary
//! architecture, not a transferable property of "more iterations" in
//! general. Do not attach an accuracy figure to [`BudgetTier`] or any doc
//! comment here; ADR-040 requires a power calculation and a measurement
//! before any compute-vs-accuracy claim is asserted for this crate's output.

use serde::{Deserialize, Serialize};

fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

/// `lo + (hi-lo)*t`, rounded to the nearest step. `hi <= lo` returns `lo`.
fn scale_u32(lo: u32, hi: u32, t: f32) -> u32 {
    if hi <= lo {
        return lo;
    }
    lo + ((hi - lo) as f32 * clamp01(t)).round() as u32
}

fn scale_f32(lo: f32, hi: f32, t: f32) -> f32 {
    lo + (hi - lo) * clamp01(t)
}

/// Measured difficulty/pressure signals for one reasoning call (§8.2), each
/// expected in `[0,1]`. Out-of-range values are clamped, not rejected, so
/// `route` stays total and panic-free.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct DifficultySignals {
    /// `u` — verifier/model uncertainty about the query.
    pub uncertainty: f32,
    /// `d` — structural difficulty (composition/relation/dependency depth,
    /// §7.2), independent of semantic similarity.
    pub structural_difficulty: f32,
    /// `n` — novelty relative to retrieved demonstrations/prototypes.
    pub novelty: f32,
    /// `f` — historical failure probability for this query family.
    pub historical_failure_rate: f32,
    /// `r` — continuous residual risk, distinct from the discrete
    /// [`RiskClass`] (which sets hard floors/ceilings).
    pub risk: f32,
    /// `l` — latency pressure (proximity to deadline).
    pub latency_pressure: f32,
    /// `e` — energy/compute pressure.
    pub energy_pressure: f32,
}

/// Weights `alpha_*` in §8.3's budget score. Structural defaults only, not
/// asserted optimal for any task family.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BudgetWeights {
    pub uncertainty: f32,
    pub structural_difficulty: f32,
    pub novelty: f32,
    pub historical_failure_rate: f32,
    pub risk: f32,
    /// Subtracted from the score (§8.3: `minus alpha_l times l`).
    pub latency_pressure: f32,
    /// Subtracted from the score (§8.3: `minus alpha_e times e`).
    pub energy_pressure: f32,
}

impl Default for BudgetWeights {
    fn default() -> Self {
        Self {
            uncertainty: 0.25,
            structural_difficulty: 0.25,
            novelty: 0.15,
            historical_failure_rate: 0.20,
            risk: 0.15,
            latency_pressure: 0.15,
            energy_pressure: 0.10,
        }
    }
}

/// Discrete risk classification (`rho`, §6.2/§6.4), distinct from the
/// continuous [`DifficultySignals::risk`]. Governs two hard floors: how
/// much residual risk a candidate may carry before it blocks an early stop
/// (§18.7), and whether latency/energy pressure may reduce verification
/// below a configured floor (§8.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskClass {
    Routine,
    Elevated,
    SafetyCritical,
}

impl RiskClass {
    /// Exempt from latency/energy pressure reducing budget/verification
    /// (§8.3: "may not reduce verification below a configured floor").
    fn protects_verification_floor(self) -> bool {
        matches!(self, RiskClass::SafetyCritical)
    }
}

/// Tunable knobs for [`ReasoningBudgetController`]. All ranges are
/// structural (iteration counts, workspace sizes, confidence thresholds) —
/// see the module doc before attaching an accuracy claim to any of these.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ControllerConfig {
    /// `R_min` — floor below which a call may never stop early (§6.2).
    pub min_iterations: u32,
    /// `R_max` ceiling for tasks routed at the hardest difficulty.
    pub max_iterations: u32,
    pub min_memory_depth: u32,
    pub max_memory_depth: u32,
    pub min_latent_width: u32,
    pub max_latent_width: u32,
    /// `tau` floor/ceiling — confidence bar to stop early, scaled by
    /// routed difficulty between these bounds.
    pub min_verification_threshold: f32,
    pub max_verification_threshold: f32,
    /// Hard floor on iterations for [`RiskClass::SafetyCritical`].
    pub safety_critical_iteration_floor: u32,
    /// Hard floor on `verification_threshold` for
    /// [`RiskClass::SafetyCritical`].
    pub safety_critical_verification_floor: f32,
    /// `epsilon` — convergence threshold on `||H_{r+1} - H_r|| / ||H_r||`.
    pub convergence_epsilon: f32,
    /// Fraction of the latency budget (`elapsed / budget`) at or above
    /// which the controller forces a stop regardless of convergence.
    pub latency_budget_ceiling: f32,
    pub risk_ceiling_routine: f32,
    pub risk_ceiling_elevated: f32,
    pub risk_ceiling_safety_critical: f32,
    pub weights: BudgetWeights,
}

impl Default for ControllerConfig {
    fn default() -> Self {
        Self {
            min_iterations: 1,
            max_iterations: 8,
            min_memory_depth: 1,
            max_memory_depth: 4,
            min_latent_width: 64,
            max_latent_width: 512,
            min_verification_threshold: 0.5,
            max_verification_threshold: 0.9,
            safety_critical_iteration_floor: 4,
            safety_critical_verification_floor: 0.85,
            convergence_epsilon: 0.01,
            latency_budget_ceiling: 1.0,
            risk_ceiling_routine: 0.5,
            risk_ceiling_elevated: 0.3,
            risk_ceiling_safety_critical: 0.1,
            weights: BudgetWeights::default(),
        }
    }
}

/// The routed compute allocation for one reasoning call. Structural only —
/// see the module doc's note on [`BudgetTier`].
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReasoningBudget {
    /// `R_target` (§8.3) — this call's iteration ceiling, used by
    /// [`ReasoningBudgetController::evaluate_step`] as the compute-ceiling
    /// rule.
    pub iterations: u32,
    /// Bounded recurrent-context retention depth (§5.3's gated-update
    /// option) allocated to this task.
    pub memory_depth: u32,
    /// Structural width of the allocated latent workspace `H`. Units are
    /// defined by whichever workspace implementation consumes this value.
    pub latent_width: u32,
    /// `tau` — verifier-confidence bar this task must clear to stop early.
    pub verification_threshold: f32,
}

/// Structural difficulty→compute-tier label for observability. **Not an
/// accuracy promise — see the module doc before adding to this type.**
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BudgetTier {
    Low,
    Medium,
    High,
}

impl BudgetTier {
    fn from_score(score: f32) -> Self {
        let s = clamp01(score);
        if s < 1.0 / 3.0 {
            BudgetTier::Low
        } else if s < 2.0 / 3.0 {
            BudgetTier::Medium
        } else {
            BudgetTier::High
        }
    }
}

/// Output of [`ReasoningBudgetController::route`]: the allocation, its tier
/// label, and the raw `b` score, so a caller can audit or re-bucket without
/// recomputing.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RoutingDecision {
    pub budget: ReasoningBudget,
    pub tier: BudgetTier,
    /// `b ∈ [0,1]` from §8.3, after any [`RiskClass::SafetyCritical`]
    /// floor adjustment.
    pub score: f32,
}

/// One recurrent iteration's measurements, supplied by the caller — this
/// controller computes none of them. `iteration` is 1-indexed, matching
/// §6.2's `for r in 1 to R_max`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct IterationMeasurement {
    pub iteration: u32,
    /// `||H_{r+1} - H_r|| / max(||H_r||, tiny)`.
    pub convergence_delta: f32,
    /// `score.confidence` from the local verifier `V`.
    pub verifier_confidence: f32,
    /// `score.policy_safe` from the local verifier `V`.
    pub policy_safe: bool,
    /// Measured residual risk of the current candidate.
    pub risk_signal: f32,
    /// `elapsed / latency_budget`, measured by the caller.
    pub latency_used_fraction: f32,
}

/// The state of every individual stopping rule at one iteration (§6.4's
/// full list), independent of which one decided the outcome. Kept
/// alongside [`StepDecision::outcome`] so a caller can tell, e.g.,
/// "confidence was already high but risk blocked the stop" from "continued
/// for no interesting reason".
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuleTrace {
    pub min_iterations_met: bool,
    pub convergence_satisfied: bool,
    pub confidence_satisfied: bool,
    pub risk_satisfied: bool,
    pub policy_safe: bool,
    pub latency_satisfied: bool,
    /// `true` while under the task's iteration ceiling.
    pub ceiling_satisfied: bool,
}

/// Why [`StepDecision::outcome`] stopped — the answer to "tell converged
/// from hit the ceiling".
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum StopReason {
    /// Every early-stop condition held together: past `R_min`, latent
    /// convergence, verifier confidence, policy safety, and risk-budget
    /// sufficiency (§6.2's `best = candidate` branch).
    Converged,
    /// The task's iteration ceiling (`ReasoningBudget::iterations`) was
    /// reached without the composite early-stop condition holding (§18.7
    /// runaway mitigation).
    ComputeCeiling,
    /// The measured latency budget was exhausted before convergence or the
    /// compute ceiling (§6.4: "remaining latency budget").
    LatencyBudgetExceeded,
}

/// Continue iterating, or stop for the given reason.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum StopOutcome {
    Continue,
    Stop(StopReason),
}

/// Full result of evaluating one iteration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StepDecision {
    pub iteration: u32,
    pub rules: RuleTrace,
    pub outcome: StopOutcome,
}

/// Routes difficulty signals to a [`ReasoningBudget`], and evaluates
/// per-iteration stop rules against one. Holds only configuration — no
/// model, clock, or thread — so both methods are pure functions of their
/// arguments.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReasoningBudgetController {
    config: ControllerConfig,
}

impl ReasoningBudgetController {
    pub fn new(config: ControllerConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &ControllerConfig {
        &self.config
    }

    fn risk_ceiling(&self, risk_class: RiskClass) -> f32 {
        match risk_class {
            RiskClass::Routine => self.config.risk_ceiling_routine,
            RiskClass::Elevated => self.config.risk_ceiling_elevated,
            RiskClass::SafetyCritical => self.config.risk_ceiling_safety_critical,
        }
    }

    /// §8.3 Algorithm 4: fold [`DifficultySignals`] into score `b`, then map
    /// `b` into a [`ReasoningBudget`] and [`BudgetTier`]. For
    /// [`RiskClass::SafetyCritical`], latency/energy pressure cannot depress
    /// the score, and iterations/verification_threshold are floored (§8.3).
    pub fn route(&self, signals: &DifficultySignals, risk_class: RiskClass) -> RoutingDecision {
        let w = &self.config.weights;
        let s = DifficultySignals {
            uncertainty: clamp01(signals.uncertainty),
            structural_difficulty: clamp01(signals.structural_difficulty),
            novelty: clamp01(signals.novelty),
            historical_failure_rate: clamp01(signals.historical_failure_rate),
            risk: clamp01(signals.risk),
            latency_pressure: clamp01(signals.latency_pressure),
            energy_pressure: clamp01(signals.energy_pressure),
        };

        let (latency_term, energy_term) = if risk_class.protects_verification_floor() {
            (0.0, 0.0)
        } else {
            (
                w.latency_pressure * s.latency_pressure,
                w.energy_pressure * s.energy_pressure,
            )
        };

        let raw = w.uncertainty * s.uncertainty
            + w.structural_difficulty * s.structural_difficulty
            + w.novelty * s.novelty
            + w.risk * s.risk
            + w.historical_failure_rate * s.historical_failure_rate
            - latency_term
            - energy_term;
        let score = clamp01(raw);
        let cfg = &self.config;

        let mut iterations = scale_u32(cfg.min_iterations, cfg.max_iterations, score);
        let memory_depth = scale_u32(cfg.min_memory_depth, cfg.max_memory_depth, score);
        let latent_width = scale_u32(cfg.min_latent_width, cfg.max_latent_width, score);
        let mut verification_threshold = scale_f32(
            cfg.min_verification_threshold,
            cfg.max_verification_threshold,
            score,
        );

        if risk_class.protects_verification_floor() {
            iterations = iterations.max(cfg.safety_critical_iteration_floor);
            verification_threshold =
                verification_threshold.max(cfg.safety_critical_verification_floor);
        }

        RoutingDecision {
            budget: ReasoningBudget {
                iterations,
                memory_depth,
                latent_width,
                verification_threshold,
            },
            tier: BudgetTier::from_score(score),
            score,
        }
    }

    /// §6.2 Algorithm 2's stop policy for one iteration against a
    /// previously routed `budget`: `budget.iterations` is this call's
    /// `R_max`, `budget.verification_threshold` is this call's `tau`.
    /// Never stops on convergence alone (§6.3) — every rule in
    /// [`RuleTrace`] is checked, and [`StepDecision::outcome`] names
    /// whichever one decided the outcome.
    pub fn evaluate_step(
        &self,
        budget: &ReasoningBudget,
        measurement: &IterationMeasurement,
        risk_class: RiskClass,
    ) -> StepDecision {
        let cfg = &self.config;
        let rules = RuleTrace {
            min_iterations_met: measurement.iteration >= cfg.min_iterations,
            convergence_satisfied: measurement.convergence_delta <= cfg.convergence_epsilon,
            confidence_satisfied: measurement.verifier_confidence >= budget.verification_threshold,
            risk_satisfied: measurement.risk_signal <= self.risk_ceiling(risk_class),
            policy_safe: measurement.policy_safe,
            latency_satisfied: measurement.latency_used_fraction < cfg.latency_budget_ceiling,
            ceiling_satisfied: measurement.iteration < budget.iterations,
        };

        let converged = rules.min_iterations_met
            && rules.convergence_satisfied
            && rules.confidence_satisfied
            && rules.policy_safe
            && rules.risk_satisfied;

        let outcome = if converged {
            StopOutcome::Stop(StopReason::Converged)
        } else if !rules.ceiling_satisfied {
            StopOutcome::Stop(StopReason::ComputeCeiling)
        } else if !rules.latency_satisfied {
            StopOutcome::Stop(StopReason::LatencyBudgetExceeded)
        } else {
            StopOutcome::Continue
        };

        StepDecision {
            iteration: measurement.iteration,
            rules,
            outcome,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn signals(v: f32) -> DifficultySignals {
        DifficultySignals {
            uncertainty: v,
            structural_difficulty: v,
            novelty: v,
            historical_failure_rate: v,
            risk: v,
            latency_pressure: 0.0,
            energy_pressure: 0.0,
        }
    }

    fn bud(iterations: u32, tau: f32) -> ReasoningBudget {
        ReasoningBudget {
            iterations,
            memory_depth: 2,
            latent_width: 128,
            verification_threshold: tau,
        }
    }

    fn meas(iter: u32, delta: f32, conf: f32, risk: f32, lat: f32) -> IterationMeasurement {
        IterationMeasurement {
            iteration: iter,
            convergence_delta: delta,
            verifier_confidence: conf,
            policy_safe: true,
            risk_signal: risk,
            latency_used_fraction: lat,
        }
    }

    fn ctl() -> ReasoningBudgetController {
        ReasoningBudgetController::new(ControllerConfig::default())
    }

    #[test]
    fn deterministic_route_and_evaluate_are_pure() {
        let c = ctl();
        let s = signals(0.6);
        let (r1, r2) = (
            c.route(&s, RiskClass::Elevated),
            c.route(&s, RiskClass::Elevated),
        );
        assert_eq!(r1, r2);
        let m = meas(3, 0.02, 0.6, 0.2, 0.4);
        let e = RiskClass::Elevated;
        assert_eq!(
            c.evaluate_step(&r1.budget, &m, e),
            c.evaluate_step(&r1.budget, &m, e)
        );
    }

    #[test]
    fn higher_difficulty_never_yields_fewer_iterations() {
        let c = ctl();
        let easy = c.route(&signals(0.05), RiskClass::Routine);
        let hard = c.route(&signals(0.95), RiskClass::Routine);
        assert!(hard.budget.iterations >= easy.budget.iterations);
        assert!(hard.budget.memory_depth >= easy.budget.memory_depth);
        assert!(hard.budget.latent_width >= easy.budget.latent_width);
        assert!(hard.score >= easy.score);
        assert_eq!(easy.tier, BudgetTier::Low);
        assert_eq!(hard.tier, BudgetTier::High);
    }

    #[test]
    fn latency_pressure_reduces_score_unless_safety_critical() {
        let c = ctl();
        let mut s = signals(0.5);
        let baseline = c.route(&s, RiskClass::Routine);
        s.latency_pressure = 1.0;
        s.energy_pressure = 1.0;
        assert!(c.route(&s, RiskClass::Routine).score < baseline.score);

        let sc_baseline = c.route(&signals(0.5), RiskClass::SafetyCritical);
        assert_eq!(
            c.route(&s, RiskClass::SafetyCritical).score,
            sc_baseline.score
        );
    }

    #[test]
    fn safety_critical_floors_iterations_and_verification_threshold() {
        let c = ctl();
        let routed = c.route(&signals(0.0), RiskClass::SafetyCritical);
        assert!(routed.budget.iterations >= c.config().safety_critical_iteration_floor);
        assert!(
            routed.budget.verification_threshold >= c.config().safety_critical_verification_floor
        );
    }

    #[test]
    fn converges_only_when_every_condition_holds() {
        let d = ctl().evaluate_step(
            &bud(6, 0.7),
            &meas(3, 0.005, 0.8, 0.1, 0.3),
            RiskClass::Routine,
        );
        assert_eq!(d.outcome, StopOutcome::Stop(StopReason::Converged));
        assert!(
            d.rules.convergence_satisfied && d.rules.confidence_satisfied && d.rules.risk_satisfied
        );
    }

    #[test]
    fn convergence_alone_does_not_stop_when_confidence_is_low() {
        // Fully converged numerically (delta=0.001) but not trusted (confidence=0.4).
        let d = ctl().evaluate_step(
            &bud(6, 0.9),
            &meas(3, 0.001, 0.4, 0.1, 0.3),
            RiskClass::Routine,
        );
        assert_eq!(d.outcome, StopOutcome::Continue);
        assert!(d.rules.convergence_satisfied && !d.rules.confidence_satisfied);
    }

    #[test]
    fn risk_class_blocks_early_stop_despite_convergence_and_confidence() {
        // risk_signal=0.5 exceeds the SafetyCritical ceiling (default 0.1).
        let m = meas(3, 0.001, 0.95, 0.5, 0.1);
        let d = ctl().evaluate_step(&bud(6, 0.5), &m, RiskClass::SafetyCritical);
        assert_eq!(d.outcome, StopOutcome::Continue);
        assert!(!d.rules.risk_satisfied);
    }

    #[test]
    fn min_iterations_floor_prevents_premature_stop() {
        let config = ControllerConfig {
            min_iterations: 2,
            ..ControllerConfig::default()
        };
        let c = ReasoningBudgetController::new(config);
        let d = c.evaluate_step(
            &bud(6, 0.5),
            &meas(1, 0.0, 1.0, 0.0, 0.0),
            RiskClass::Routine,
        );
        assert_eq!(d.outcome, StopOutcome::Continue);
        assert!(!d.rules.min_iterations_met);
    }

    #[test]
    fn compute_ceiling_fires_when_budget_exhausted_without_convergence() {
        let m = meas(4, 0.5, 0.2, 0.05, 0.5);
        let d = ctl().evaluate_step(&bud(4, 0.9), &m, RiskClass::Routine);
        assert_eq!(d.outcome, StopOutcome::Stop(StopReason::ComputeCeiling));
        assert!(!d.rules.ceiling_satisfied);
    }

    #[test]
    fn ceiling_iteration_that_also_converges_reports_converged_not_ceiling() {
        let m = meas(4, 0.001, 0.9, 0.05, 0.5);
        let d = ctl().evaluate_step(&bud(4, 0.5), &m, RiskClass::Routine);
        assert_eq!(d.outcome, StopOutcome::Stop(StopReason::Converged));
    }

    #[test]
    fn latency_budget_exceeded_stops_before_ceiling() {
        // Iteration 3 of an 8-iteration ceiling, but the deadline is spent.
        let m = meas(3, 0.5, 0.2, 0.05, 1.0);
        let d = ctl().evaluate_step(&bud(8, 0.9), &m, RiskClass::Routine);
        assert_eq!(
            d.outcome,
            StopOutcome::Stop(StopReason::LatencyBudgetExceeded)
        );
        assert!(d.rules.ceiling_satisfied && !d.rules.latency_satisfied);
    }

    #[test]
    fn budget_serializes_round_trip() {
        let routed = ctl().route(&signals(0.4), RiskClass::Elevated);
        let json = serde_json::to_string(&routed.budget).expect("serialize");
        let back: ReasoningBudget = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(routed.budget, back);
    }

    #[test]
    fn tier_boundaries_are_monotonic() {
        assert_eq!(BudgetTier::from_score(0.0), BudgetTier::Low);
        assert_eq!(BudgetTier::from_score(0.32), BudgetTier::Low);
        assert_eq!(BudgetTier::from_score(0.34), BudgetTier::Medium);
        assert_eq!(BudgetTier::from_score(0.66), BudgetTier::Medium);
        assert_eq!(BudgetTier::from_score(0.68), BudgetTier::High);
        assert_eq!(BudgetTier::from_score(1.0), BudgetTier::High);
    }
}
