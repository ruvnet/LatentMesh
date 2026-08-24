import assert from "node:assert/strict";
import test from "node:test";
import {
  ACCEPTANCE_TARGETS,
  ROOT_POLICY,
  benchmarkPolicy,
  evaluatePolicy,
  makeSuite,
  promotionRule,
  simulate,
} from "../lib/simulator.mjs";

test("simulation is deterministic for a frozen policy and channel", () => {
  const item = makeSuite("single", [7]).items[0];
  assert.deepEqual(simulate(ROOT_POLICY, item), simulate(ROOT_POLICY, item));
});

test("stronger FEC improves a difficult frame probability", () => {
  const item = makeSuite("difficult", [9], 2).items[0];
  const none = simulate(ROOT_POLICY, item);
  const protectedFrame = simulate({ ...ROOT_POLICY, fecRate: "1/2" }, item);
  assert.ok(protectedFrame.frameSuccess > none.frameSuccess);
});

test("promotion gate rejects a critical agreement regression", () => {
  const verdict = promotionRule({
    baseline: { primary: 100, criticalAgreement: 0.995, fallbackRate: 0.2, costPerWin: 1, regressed: false },
    candidate: { primary: 120, criticalAgreement: 0.994, fallbackRate: 0.2, costPerWin: 0.9, regressed: false },
    anchor: { baseline: 100, candidate: 100 },
  });
  assert.equal(verdict.promote, false);
  assert.match(verdict.reasons.join(" "), /agreement regressed/);
});

test("holdout evaluator exposes the frozen safety fields", async () => {
  const result = await evaluatePolicy(ROOT_POLICY, makeSuite("test", [1, 2, 3]));
  assert.equal(typeof result.minimumAgreement, "number");
  assert.equal(typeof result.fallbackRate, "number");
  assert.equal(typeof result.regressed, "boolean");
});

test("stage benchmark never promotes simulation into an acceptance claim", () => {
  const result = benchmarkPolicy(ROOT_POLICY, makeSuite("stage", [1, 2, 3], 1.2));
  assert.ok(result.semantic.reduction >= ACCEPTANCE_TARGETS.semanticReduction);
  assert.equal(typeof result.neuralPhy.gain, "number");
  assert.equal(result.acceptancePassed, false);
  assert.match(result.acceptanceBlockedBy.join(" "), /hardware in loop/);
});

test("paired classical and neural trials share the exact wire budget", () => {
  const item = makeSuite("paired", [19], 1.5).items[0];
  const classical = simulate(ROOT_POLICY, item, { neuralEnabled: false });
  const neural = simulate(ROOT_POLICY, item, { neuralEnabled: true });
  assert.equal(classical.wireBytes, neural.wireBytes);
  assert.equal(classical.airtime, neural.airtime);
});
