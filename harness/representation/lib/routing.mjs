// A JS mirror of crates/latentmesh-reasoning/src/routing.rs — the same
// `utility_density` formula, the same three-state gain, the same first-class
// `None`, the same tie-breaks. Deliberately dependency-free (harness/evolve's
// lib/verify.mjs convention) so the scoring rule can be read end to end.
//
// The one behaviour this file exists to protect:
//
//   `Unmeasured` is not a number. It has no `deltaV` field, `utilityDensity`
//   refuses anything that is not a finite number, and the router only reaches
//   the formula inside the `Measured` branch. An unmeasured candidate is
//   therefore ineligible STRUCTURALLY — not by a policy check someone can
//   forget to write — and `Measured(0)` and `Unmeasured` cannot collapse into
//   each other because one has a field the other does not.

export const MODE = Object.freeze({
  None: "None",
  Text: "text",
  SemanticDelta: "semantic-delta",
  Latent: "latent",
});

export const RESOURCES = Object.freeze(["bytes", "latencySeconds", "energyJoules", "risk"]);

const WEIGHT_OF_RESOURCE = Object.freeze({
  bytes: "bandwidth",
  latencySeconds: "latency",
  energyJoules: "energy",
  risk: "risk",
});

/** The gain of a candidate no causal test has ever been run for. Carries NO
 *  numeric field, by construction. */
export const UNMEASURED = Object.freeze({ kind: "Unmeasured" });

/** A causal test WAS run and produced this delta — which may be zero,
 *  negative, or positive. `Measured(0)` is a real, eligible measurement. */
export function measured(deltaV) {
  if (typeof deltaV !== "number" || !Number.isFinite(deltaV)) {
    throw new TypeError(`measured() needs a finite number, got ${JSON.stringify(deltaV)}`);
  }
  return Object.freeze({ kind: "Measured", deltaV });
}

export function isUnmeasured(gain) {
  return gain === UNMEASURED || (gain && gain.kind === "Unmeasured");
}

export const SCORE_OUTCOME = Object.freeze({
  Unmeasured: "Unmeasured",
  Invalid: "Invalid",
  Scored: "Scored",
});

/**
 * `Score(m) = deltaV / sum over MEASURED resources r of ( w_r * cost_r / scale_r )`.
 *
 * Additive after normalisation, never a product of raw physical units — see
 * routing.rs's `CostScale` doc for why the product form is wrong (unit-
 * dependent magnitude; explodes as any one term approaches zero).
 *
 * A `null` cost means that resource was never measured for this candidate. It
 * is dropped from the sum and the remaining weights are renormalised, so the
 * weight vector stays a simplex and an unmeasured resource cannot silently
 * shrink one candidate's denominator relative to another's. Callers must
 * additionally assert cost parity (see `assertCostParity`).
 */
export function utilityDensity(deltaV, cost, scale, weights) {
  if (typeof deltaV !== "number" || !Number.isFinite(deltaV)) {
    throw new TypeError("utilityDensity needs a finite measured deltaV; an Unmeasured gain has no score");
  }
  const present = [];
  for (const resource of RESOURCES) {
    const raw = cost[resource];
    if (raw === null || raw === undefined) continue;
    if (!Number.isFinite(raw) || raw < 0) return { ok: false, error: "InvalidCost" };
    const reference = scale[resource];
    if (!Number.isFinite(reference) || reference <= 0) return { ok: false, error: "InvalidScale" };
    present.push({ normalised: raw / reference, weight: weights[WEIGHT_OF_RESOURCE[resource]] });
  }
  if (present.length === 0) return { ok: false, error: "NoMeasuredResource" };
  const weightSum = present.reduce((total, term) => total + term.weight, 0);
  if (!(weightSum > 0)) return { ok: false, error: "InvalidWeights" };
  const denominator = present.reduce(
    (total, term) => total + (term.weight / weightSum) * term.normalised,
    0,
  );
  if (!(denominator > 0)) return { ok: false, error: "ZeroCost" };
  return { ok: true, score: deltaV / denominator };
}

/** Every candidate must leave exactly the same resources unmeasured, or their
 *  denominators are not comparable and no ranking between them is meaningful. */
export function assertCostParity(candidates) {
  const signature = (candidate) =>
    RESOURCES.filter((resource) => candidate.cost[resource] === null || candidate.cost[resource] === undefined).join(",");
  const first = candidates.length > 0 ? signature(candidates[0]) : "";
  for (const candidate of candidates) {
    if (signature(candidate) !== first) {
      throw new Error(
        `cost parity violated: ${candidate.mode}/${candidate.id} leaves [${signature(candidate)}] unmeasured, expected [${first}]`,
      );
    }
  }
  return first === "" ? [] : first.split(",");
}

/**
 * Reference scales, one per resource, taken as the maximum measured value
 * across the WHOLE evidence corpus so every normalised cost lands in [0, 1].
 *
 * This is a scale derived from the candidate corpus, NOT a deployment fact
 * (routing.rs deliberately refuses to ship a `Default` CostScale for exactly
 * that reason). It is recorded as such in the receipt. It must be computed
 * once over the full corpus and reused for every genome — a per-genome scale
 * would make each policy the unit of its own measurement and every single-
 * candidate policy would trivially normalise to 1.
 */
export function deriveCostScale(candidates) {
  const scale = {};
  for (const resource of RESOURCES) {
    let max = 0;
    for (const candidate of candidates) {
      const raw = candidate.cost[resource];
      if (typeof raw === "number" && Number.isFinite(raw) && raw > max) max = raw;
    }
    scale[resource] = max > 0 ? max : 1; // a corpus-wide zero normalises to 0/1 = 0
  }
  return Object.freeze(scale);
}

/**
 * Score every candidate and pick the best, with `None` always implicitly
 * available at exactly 0.0 (its cost is zero in every dimension, so the
 * formula's denominator would be zero and 0/0 is undefined, not 0).
 *
 * Ties favour `None` first — silence is the conservative default when nothing
 * strictly beats it — then the earliest-listed candidate.
 */
export function route(candidates, scale, weights) {
  const considered = [];
  let selected = MODE.None;
  let selectedScore = 0;
  let selectedId = null;

  for (const candidate of candidates) {
    if (isUnmeasured(candidate.gain)) {
      considered.push({ id: candidate.id, mode: candidate.mode, outcome: SCORE_OUTCOME.Unmeasured, score: null });
      continue;
    }
    const result = utilityDensity(candidate.gain.deltaV, candidate.cost, scale, weights);
    if (!result.ok) {
      considered.push({
        id: candidate.id,
        mode: candidate.mode,
        outcome: SCORE_OUTCOME.Invalid,
        score: null,
        error: result.error,
      });
      continue;
    }
    if (result.score > selectedScore) {
      selectedScore = result.score;
      selected = candidate.mode;
      selectedId = candidate.id;
    }
    considered.push({ id: candidate.id, mode: candidate.mode, outcome: SCORE_OUTCOME.Scored, score: result.score });
  }

  return { selected, selectedId, selectedScore, considered };
}
