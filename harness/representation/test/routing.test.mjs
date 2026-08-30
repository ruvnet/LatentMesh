import test from "node:test";
import assert from "node:assert/strict";
import {
  MODE,
  SCORE_OUTCOME,
  UNMEASURED,
  assertCostParity,
  deriveCostScale,
  isUnmeasured,
  measured,
  route,
  utilityDensity,
} from "../lib/routing.mjs";

const scale = { bytes: 1000, latencySeconds: 1, energyJoules: 1, risk: 1 };
const weights = { bandwidth: 0.25, latency: 0.25, energy: 0.25, risk: 0.25 };
const cost = (bytes, latencySeconds, energyJoules, risk) => ({ bytes, latencySeconds, energyJoules, risk });

test("utility density is deterministic", () => {
  const c = cost(200, 0.1, 0.05, 0.1);
  assert.deepEqual(utilityDensity(0.5, c, scale, weights), utilityDensity(0.5, c, scale, weights));
});

test("the denominator is additive after normalisation, not a product", () => {
  // Same total normalised cost via different resource mixes must score the
  // same. A product denominator would blow up on whichever term is near zero.
  const heavyBytes = utilityDensity(1, cost(2000, 0, 0, 0), scale, weights);
  const heavyLatency = utilityDensity(1, cost(0, 2, 0, 0), scale, weights);
  assert.ok(Math.abs(heavyBytes.score - heavyLatency.score) < 1e-12);
});

test("invalid scale and invalid cost are refused by name", () => {
  assert.equal(utilityDensity(1, cost(1, 1, 1, 1), { ...scale, bytes: 0 }, weights).error, "InvalidScale");
  assert.equal(utilityDensity(1, cost(-1, 1, 1, 1), scale, weights).error, "InvalidCost");
});

test("a zero cost with nonzero gain is refused, never infinite", () => {
  assert.equal(utilityDensity(1, cost(0, 0, 0, 0), scale, weights).error, "ZeroCost");
});

test("higher normalised cost for the same gain never scores higher", () => {
  const low = utilityDensity(1, cost(10, 0.01, 0.01, 0.01), scale, weights).score;
  const high = utilityDensity(1, cost(500, 0.5, 0.5, 0.5), scale, weights).score;
  assert.ok(low > high);
});

test("an unmeasured gain has no deltaV field and cannot be scored", () => {
  assert.equal("deltaV" in UNMEASURED, false);
  assert.throws(() => utilityDensity(UNMEASURED.deltaV, cost(1, 1, 1, 1), scale, weights), TypeError);
  assert.throws(() => measured(undefined), TypeError);
});

test("Measured(0) and Unmeasured are distinct and do not collapse", () => {
  const zero = measured(0);
  assert.equal(zero.kind, "Measured");
  assert.equal(zero.deltaV, 0);
  assert.equal(isUnmeasured(zero), false);
  assert.equal(isUnmeasured(UNMEASURED), true);
  assert.notDeepEqual(zero, UNMEASURED);
});

test("an unmeasured candidate is reported but can never be selected", () => {
  const decision = route(
    [
      { id: "u", mode: MODE.Latent, gain: UNMEASURED, cost: cost(1, 0.001, 0.001, 0) },
      { id: "z", mode: MODE.Text, gain: measured(0), cost: cost(80, 0.08, 0.03, 0.03) },
    ],
    scale,
    weights,
  );
  assert.equal(decision.selected, MODE.None);
  assert.equal(decision.considered[0].outcome, SCORE_OUTCOME.Unmeasured);
  assert.equal(decision.considered[0].score, null);
  assert.equal(decision.considered[1].outcome, SCORE_OUTCOME.Scored);
  assert.equal(decision.considered[1].score, 0);
});

test("an unmeasured candidate cannot win even when it would dominate on cost", () => {
  // The inverse of UCB1: this candidate is the cheapest by three orders of
  // magnitude, and it still loses to a measured competitor.
  const decision = route(
    [
      { id: "free-but-untested", mode: MODE.SemanticDelta, gain: UNMEASURED, cost: cost(1, 1e-6, 1e-6, 0) },
      { id: "measured", mode: MODE.Text, gain: measured(0.5), cost: cost(5000, 5, 5, 1) },
    ],
    scale,
    weights,
  );
  assert.equal(decision.selected, MODE.Text);
  assert.equal(decision.selectedId, "measured");
});

test("None wins when every measured candidate has non-positive gain", () => {
  const decision = route(
    [
      { id: "a", mode: MODE.Text, gain: measured(-0.1), cost: cost(400, 0.3, 0.1, 0.1) },
      { id: "b", mode: MODE.Latent, gain: measured(0), cost: cost(50, 0.05, 0.02, 0.05) },
    ],
    scale,
    weights,
  );
  assert.equal(decision.selected, MODE.None);
  assert.equal(decision.selectedScore, 0);
});

test("ties favour None first, then the earliest candidate", () => {
  const zeroGain = route([{ id: "t", mode: MODE.Text, gain: measured(0), cost: cost(100, 0.1, 0.1, 0.1) }], scale, weights);
  assert.equal(zeroGain.selected, MODE.None);

  const c = cost(100, 0.1, 0.05, 0.05);
  const tied = route(
    [
      { id: "first", mode: MODE.Text, gain: measured(1), cost: c },
      { id: "second", mode: MODE.SemanticDelta, gain: measured(1), cost: c },
    ],
    scale,
    weights,
  );
  assert.equal(tied.selectedId, "first");
});

test("an unmeasured resource is dropped and the remaining weights renormalise", () => {
  const withEnergy = utilityDensity(1, cost(500, 0.5, 0, 0), scale, weights).score;
  const withoutEnergy = utilityDensity(1, { bytes: 500, latencySeconds: 0.5, energyJoules: null, risk: 0 }, scale, weights).score;
  // Dropping a zero-cost resource and renormalising raises the denominator,
  // so the score falls — the unmeasured resource is not a free discount.
  assert.ok(withoutEnergy < withEnergy);
});

test("cost parity is asserted, not assumed", () => {
  const ok = [
    { id: "a", mode: MODE.Text, cost: cost(1, 1, null, 0) },
    { id: "b", mode: MODE.Latent, cost: cost(2, 2, null, 0) },
  ];
  assert.deepEqual(assertCostParity(ok), ["energyJoules"]);
  assert.throws(
    () => assertCostParity([...ok, { id: "c", mode: MODE.Latent, cost: cost(3, 3, 3, 0) }]),
    /cost parity violated/,
  );
});

test("the cost scale is the corpus-wide maximum per resource", () => {
  const derived = deriveCostScale([
    { cost: cost(100, 4, null, 0) },
    { cost: cost(12288, 2, null, 0) },
  ]);
  assert.equal(derived.bytes, 12288);
  assert.equal(derived.latencySeconds, 4);
  assert.equal(derived.energyJoules, 1); // never measured -> a 1 that normalises 0 to 0
  assert.equal(derived.risk, 1); // measured, but zero corpus-wide
});
