import test from "node:test";
import assert from "node:assert/strict";
import { verifyReceipt, RECEIPT_SCHEMA } from "../lib/verify.mjs";

function goodReceipt() {
  return {
    schema: RECEIPT_SCHEMA,
    evidence: "simulated",
    seed: 1,
    generations: 40,
    population: 12,
    warm_started: false,
    fitness_before: -10.0,
    fitness_after: 1.5,
    quality_after: 4.4,
    compute_before: 2843.2,
    compute_after: 733.0,
    mutations_proposed: 1440,
    mutations_rejected_by_constitution: 46,
    verification_evaluations: 10800,
    acceptance: {
      compute_reduction: 0.742,
      compute_reduction_target: 0.3,
      compute_reduction_met: true,
      task_success_before: 4.55,
      task_success_after: 4.55,
      task_success_maintained: true,
      surviving_edges: 8,
      unverified_nonmandatory_edges: 0,
      all_surviving_edges_verified: true,
      passed: true,
    },
    not_claimed: ["live multi-agent workload evidence is absent"],
  };
}

test("a compliant receipt passes with no reasons", () => {
  const { pass, reasons } = verifyReceipt(goodReceipt());
  assert.deepEqual(reasons, []);
  assert.equal(pass, true);
});

test("an over-claimed evidence label is refused", () => {
  const receipt = goodReceipt();
  receipt.evidence = "over the air";
  const { pass, reasons } = verifyReceipt(receipt);
  assert.equal(pass, false);
  assert.ok(reasons.some((r) => r.includes("evidence label")));
});

test("a compute reduction below the bound is named", () => {
  const receipt = goodReceipt();
  receipt.acceptance.compute_reduction = 0.1;
  const { pass, reasons } = verifyReceipt(receipt);
  assert.equal(pass, false);
  assert.ok(reasons.some((r) => r.includes("compute reduction")));
});

test("a task-success regression is named", () => {
  const receipt = goodReceipt();
  receipt.acceptance.task_success_after = 2.0;
  const { pass, reasons } = verifyReceipt(receipt);
  assert.equal(pass, false);
  assert.ok(reasons.some((r) => r.includes("task success regressed")));
});

test("surviving unverified edges fail the receipt", () => {
  const receipt = goodReceipt();
  receipt.acceptance.unverified_nonmandatory_edges = 2;
  const { pass, reasons } = verifyReceipt(receipt);
  assert.equal(pass, false);
  assert.ok(reasons.some((r) => r.includes("failed causal verification")));
});

test("a receipt that claims nothing is missing must say so", () => {
  const receipt = goodReceipt();
  receipt.not_claimed = [];
  const { pass } = verifyReceipt(receipt);
  assert.equal(pass, false);
});

test("garbage input never throws", () => {
  for (const junk of [null, 42, "x", [], {}, { schema: "wrong" }]) {
    const { pass } = verifyReceipt(junk);
    assert.equal(pass, false);
  }
});
