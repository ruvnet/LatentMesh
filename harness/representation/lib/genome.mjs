// ADR-046 §3's eighth policy surface — `representationPolicy` — as an
// enumerable genome, plus the rule that turns one genome plus the committed
// evidence into a routed decision and a fitness.
//
// The genes are exactly §3's table: channel, admission, controls, alpha,
// costWeights. What each gene is allowed to range over is constrained below,
// and every constraint is a stated safety property rather than a tuning knob.

import {
  MODE,
  UNMEASURED,
  assertCostParity,
  deriveCostScale,
  measured,
  route,
} from "./routing.mjs";

export const CHANNELS = Object.freeze([MODE.Text, MODE.SemanticDelta, MODE.Latent, MODE.None]);
export const ADMISSIONS = Object.freeze(["causal-gate", "always", "never"]);

/** The e-process of record was drawn at alpha = 0.05 and is never restarted
 *  (`e_process.never_restarted`). Re-thresholding the committed wealth at a
 *  STRICTER alpha is conservative and re-runs nothing; re-thresholding looser
 *  would be a post-hoc weakening of a pre-registered bar, so the lattice does
 *  not contain one. */
export const ALPHA_OF_RECORD = 0.05;
export const ALPHAS = Object.freeze([0.05, 0.01, 0.001]);

export const CONTROL_SETS = Object.freeze([
  Object.freeze(["random"]),
  Object.freeze(["random", "zero"]),
  Object.freeze(["zero", "random", "mismatched", "self_generated"]),
]);

/** Points on the weight simplex. These govern ROUTING inside a genome; they
 *  do not govern how genomes are compared (see `FROZEN_SCORING_WEIGHTS`). */
export const COST_WEIGHT_SETS = Object.freeze([
  Object.freeze({ name: "equal", bandwidth: 0.25, latency: 0.25, energy: 0.25, risk: 0.25 }),
  Object.freeze({ name: "bandwidth-heavy", bandwidth: 0.7, latency: 0.1, energy: 0.1, risk: 0.1 }),
  Object.freeze({ name: "latency-heavy", bandwidth: 0.1, latency: 0.7, energy: 0.1, risk: 0.1 }),
  Object.freeze({ name: "risk-heavy", bandwidth: 0.1, latency: 0.1, energy: 0.1, risk: 0.7 }),
]);

/**
 * Genomes are compared under ONE frozen weight vector, never under their own.
 *
 * Without this, `costWeights` is a reward-hacking surface: with the weights
 * free, a genome maximises its own fitness by moving all weight onto whichever
 * resource it happens to spend least of, and the resulting numbers are not on
 * a common scale, so "champion" would mean "best at choosing its own yardstick".
 * A genome's own weights still decide WHICH channel it routes to; only the
 * cross-genome comparison is frozen.
 */
export const FROZEN_SCORING_WEIGHTS = COST_WEIGHT_SETS[0];

export const ROOT_POLICY = Object.freeze({
  channel: MODE.Text,
  admission: "causal-gate",
  controls: CONTROL_SETS[2],
  alpha: ALPHA_OF_RECORD,
  costWeights: COST_WEIGHT_SETS[0].name,
});

export function enumerateGenomes() {
  const genomes = [];
  for (const channel of CHANNELS) {
    for (const admission of ADMISSIONS) {
      for (const controls of CONTROL_SETS) {
        for (const alpha of ALPHAS) {
          for (const weights of COST_WEIGHT_SETS) {
            genomes.push(Object.freeze({ channel, admission, controls, alpha, costWeights: weights.name }));
          }
        }
      }
    }
  }
  return Object.freeze(genomes);
}

export function weightsOf(genome) {
  const found = COST_WEIGHT_SETS.find((entry) => entry.name === genome.costWeights);
  if (!found) throw new Error(`unknown costWeights ${genome.costWeights}`);
  return found;
}

export function genomeKey(genome) {
  return `${genome.channel}|${genome.admission}|${genome.controls.join("+")}|${genome.alpha}|${genome.costWeights}`;
}

/**
 * Admit (or refuse) one evidence record under one policy, and report the
 * gain that admission leaves.
 *
 * Three outcomes, and the difference between the last two is the whole point:
 *
 * - `unmeasured`  the evidence cannot speak to the policy's required controls
 *                 at all — either no receipt exists for the channel, or the
 *                 receipt has no wealth process for a control the policy
 *                 demands. Ineligible: it has no gain to score.
 * - `refused`     the evidence exists, the test ran, and it did not clear the
 *                 bar. The admitted gain is a MEASURED 0.0 — eligible, scored,
 *                 and beaten by `None` on the tie-break.
 * - `admitted`    the bar was cleared; the admitted gain is the measured
 *                 delta against the control most favourable to the null.
 */
export function admit(evidence, genome) {
  if (evidence.measured === false) {
    return { status: "unmeasured", gain: UNMEASURED, reason: evidence.reason };
  }
  if (genome.admission === "never") {
    return { status: "refused", gain: measured(0), reason: "policy admission is 'never'" };
  }

  const byName = new Map(evidence.controls.map((control) => [control.control, control]));
  const missing = genome.controls.filter((name) => !byName.has(name));
  if (missing.length > 0) {
    return {
      status: "unmeasured",
      gain: UNMEASURED,
      reason: `${evidence.receipt} has no measured control named ${missing.join(", ")}; the policy's required control set was never run against this channel`,
    };
  }
  const required = genome.controls.map((name) => byName.get(name));

  // "Worst of the controls" = the one most favourable to the null = the one
  // that got the most items right. ADR-003's admission shape, applied to the
  // policy's own required subset.
  const worst = required.reduce((a, b) => (b.correct > a.correct ? b : a));
  const rawDelta = (evidence.correct - worst.correct) / evidence.n;

  if (genome.admission === "always") {
    return { status: "admitted", gain: measured(rawDelta), reason: "policy admission is 'always' — no gate applied", worst, rawDelta };
  }

  // causal-gate: ADR-030's intersection-union e-process. PASS requires EVERY
  // required control's wealth to reach 1/alpha.
  const threshold = 1 / genome.alpha;
  const minWealth = Math.min(...required.map((control) => control.finalWealth));
  if (!(minWealth >= threshold)) {
    return {
      status: "refused",
      gain: measured(0),
      reason: `min wealth ${minWealth} across required controls is below 1/alpha = ${threshold}`,
      worst,
      rawDelta,
      minWealth,
      threshold,
    };
  }
  return { status: "admitted", gain: measured(rawDelta), reason: `min wealth ${minWealth} >= 1/alpha = ${threshold}`, worst, rawDelta, minWealth, threshold };
}

/** Candidates a genome is permitted to consider: its `channel` gene is a
 *  scope filter over the corpus. `channel: None` permits nothing, so the
 *  router returns `None` at 0.0. */
export function candidatesFor(corpus, genome) {
  const permitted = corpus.filter((evidence) => evidence.channel === genome.channel);
  return permitted.map((evidence) => {
    const decision = admit(evidence, genome);
    return {
      id: evidence.id,
      mode: evidence.channel,
      gain: decision.gain,
      cost: evidence.cost,
      admission: decision,
    };
  });
}

/**
 * Route one genome over the corpus and score it. `scale` must be the corpus-
 * wide scale from `deriveCostScale`, computed once, so that every genome is
 * measured against the same reference.
 */
export function evaluate(corpus, genome, scale) {
  const candidates = candidatesFor(corpus, genome);
  const scorable = candidates.filter((candidate) => candidate.gain.kind === "Measured");
  const unmeasuredResources = assertCostParity(scorable);

  const routingDecision = route(candidates, scale, weightsOf(genome));
  const fitnessDecision = route(candidates, scale, FROZEN_SCORING_WEIGHTS);

  // The genome's own weights choose the mode; fitness is that mode's score on
  // the frozen yardstick, so no genome can inflate itself by re-weighting.
  const chosen = fitnessDecision.considered.find((entry) => entry.id === routingDecision.selectedId);
  const fitness = routingDecision.selected === MODE.None || !chosen ? 0 : chosen.score;

  return {
    genome,
    selected: routingDecision.selected,
    selectedId: routingDecision.selectedId,
    fitness,
    routedScore: routingDecision.selectedScore,
    considered: routingDecision.considered,
    admissions: candidates.map((candidate) => ({
      id: candidate.id,
      status: candidate.admission.status,
      reason: candidate.admission.reason,
      rawDelta: candidate.admission.rawDelta ?? null,
      worstControl: candidate.admission.worst ? candidate.admission.worst.control : null,
    })),
    unmeasuredResources,
  };
}

export function buildScale(corpus) {
  return deriveCostScale(corpus.filter((evidence) => evidence.measured !== false));
}

/**
 * The FROZEN promotion rule (harness/air's convention: the rule is fixed, the
 * policy is what evolves). A candidate replaces the incumbent only if:
 *
 *  1. it scores strictly higher on the frozen yardstick;
 *  2. it does not WEAKEN the bar — it must require every control its channel's
 *     evidence actually provides. Without this clause the search has a trivial
 *     exploit: "worst of the required controls" is monotone, so dropping the
 *     hardest control raises delta_V for free. Strictness is a safety property
 *     of the gate, not a fitness dimension, so it is floored here rather than
 *     evolved. (The counterfactual is reported, not hidden — see
 *     `optimize.mjs`'s `unconstrained_note`.)
 *  3. it does not loosen alpha past the pre-registered 0.05;
 *  4. it routes to a mode that was actually admitted, or to `None`.
 */
export function promotionRule(candidate, incumbent, corpus) {
  const reasons = [];
  if (!(candidate.fitness > (incumbent ? incumbent.fitness : -Infinity))) {
    reasons.push("fitness does not strictly improve on the incumbent");
  }
  const available = new Set(
    corpus
      .filter((evidence) => evidence.channel === candidate.genome.channel && evidence.measured !== false)
      .flatMap((evidence) => evidence.controls.map((control) => control.control)),
  );
  for (const control of available) {
    if (!candidate.genome.controls.includes(control)) {
      reasons.push(`weakens the bar: control '${control}' is measured for this channel but not required by the policy`);
    }
  }
  if (candidate.genome.alpha > ALPHA_OF_RECORD) {
    reasons.push(`alpha ${candidate.genome.alpha} is looser than the pre-registered ${ALPHA_OF_RECORD}`);
  }
  if (candidate.selected !== MODE.None) {
    const admission = candidate.admissions.find((entry) => entry.id === candidate.selectedId);
    if (!admission || admission.status !== "admitted") {
      reasons.push(`routes to ${candidate.selected} which was not admitted`);
    }
  }
  return { promote: reasons.length === 0, reasons };
}
