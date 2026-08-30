import test from "node:test";
import assert from "node:assert/strict";
import { ReceiptLog, canonical, hash, verifyChain } from "../lib/chain.mjs";
import {
  BENCHMARK_SCHEMA,
  CHAMPION_SCHEMA,
  REQUIRED_EVIDENCE,
  verifyBenchmarkReceipt,
  verifyChampionReceipt,
  verifyUnmeasuredIneligibility,
} from "../lib/verify.mjs";

function chainOfThree() {
  const log = new ReceiptLog();
  log.append("root-policy", { genome: "a" });
  log.append("generation", { generation: 1, fitness: 0.5 });
  log.append("champion", { genome: "b" });
  return log.toJSON();
}

test("canonical JSON sorts keys at every depth so hashes are stable", () => {
  assert.equal(canonical({ b: 1, a: { d: 2, c: 3 } }), canonical({ a: { c: 3, d: 2 }, b: 1 }));
  assert.equal(hash({ b: 1, a: 2 }), hash({ a: 2, b: 1 }));
});

test("a well-formed chain replays", () => {
  const verdict = verifyChain(chainOfThree());
  assert.deepEqual(verdict.reasons, []);
  assert.equal(verdict.pass, true);
});

test("mutating any payload breaks the chain at that index", () => {
  const chain = chainOfThree();
  chain.entries[1].payload.fitness = 0.9;
  const verdict = verifyChain(chain);
  assert.equal(verdict.pass, false);
  assert.equal(verdict.brokenAt, 1);
});

test("reordering entries breaks the chain", () => {
  const chain = chainOfThree();
  [chain.entries[0], chain.entries[1]] = [chain.entries[1], chain.entries[0]];
  assert.equal(verifyChain(chain).pass, false);
});

test("a truncated chain is caught by the head", () => {
  const chain = chainOfThree();
  chain.entries.pop();
  const verdict = verifyChain(chain);
  assert.equal(verdict.pass, false);
  assert.ok(verdict.reasons.some((reason) => reason.includes("head")));
});

function goodBenchmark() {
  return {
    schema: BENCHMARK_SCHEMA,
    evidence: REQUIRED_EVIDENCE,
    selected: "text",
    selectedId: "text/x",
    selectedScore: 1.4,
    candidates: [
      { id: "text/x", mode: "text", outcome: "Scored", score: 1.4 },
      { id: "sd/u", mode: "semantic-delta", outcome: "Unmeasured", score: null },
      { id: "latent/y", mode: "latent", outcome: "Scored", score: 0 },
    ],
    unmeasured_resources: ["energyJoules"],
    sources: [{ receipt: "r.json", receiptSha256: "a".repeat(64) }],
    not_claimed: ["does not claim latent transport works"],
  };
}

test("a compliant benchmark receipt passes with no reasons", () => {
  const { pass, reasons } = verifyBenchmarkReceipt(goodBenchmark());
  assert.deepEqual(reasons, []);
  assert.equal(pass, true);
});

test("an over-claimed evidence label is refused", () => {
  for (const label of ["simulated", "live", "deployed"]) {
    const receipt = { ...goodBenchmark(), evidence: label };
    const { pass, reasons } = verifyBenchmarkReceipt(receipt);
    assert.equal(pass, false, `${label} should be refused`);
    assert.ok(reasons.some((reason) => reason.includes("evidence label")));
  }
});

test("a missing not_claimed list fails", () => {
  const receipt = { ...goodBenchmark(), not_claimed: [] };
  assert.ok(verifyBenchmarkReceipt(receipt).reasons.some((reason) => reason.includes("NOT claim")));
});

test("a source without the sha256 of the file that was read fails", () => {
  const receipt = { ...goodBenchmark(), sources: [{ receipt: "r.json" }] };
  assert.ok(verifyBenchmarkReceipt(receipt).reasons.some((reason) => reason.includes("sha256")));
});

test("dropping the unmeasured channel from the report fails", () => {
  const receipt = goodBenchmark();
  receipt.candidates = receipt.candidates.filter((candidate) => candidate.outcome !== "Unmeasured");
  const { pass, reasons } = verifyBenchmarkReceipt(receipt);
  assert.equal(pass, false);
  assert.ok(reasons.some((reason) => reason.includes("Unmeasured")));
});

test("a receipt that selected an unmeasured candidate is refused", () => {
  const receipt = { ...goodBenchmark(), selected: "semantic-delta", selectedId: "sd/u" };
  const { pass, reasons } = verifyBenchmarkReceipt(receipt);
  assert.equal(pass, false);
  assert.ok(reasons.some((reason) => reason.includes("unmeasured candidate sd/u was selected")));
});

test("an unmeasured candidate carrying a score is refused", () => {
  const verdict = verifyUnmeasuredIneligibility([{ id: "u", outcome: "Unmeasured", score: 0 }], null);
  assert.equal(verdict.pass, false);
  assert.ok(verdict.reasons[0].includes("carries a score"));
});

test("a champion receipt must be deterministic, exhaustive-matching and chained", () => {
  const base = {
    schema: CHAMPION_SCHEMA,
    evidence: REQUIRED_EVIDENCE,
    champion: { genome: { channel: "text", alpha: 0.05 }, selectedId: "text/x", considered: [] },
    deterministic_replay: true,
    matches_exhaustive_optimum: true,
    chain: chainOfThree(),
    sources: [{ receipt: "r.json", receiptSha256: "b".repeat(64) }],
    not_claimed: ["does not revive the latent channel"],
  };
  assert.deepEqual(verifyChampionReceipt(base).reasons, []);

  assert.ok(
    verifyChampionReceipt({ ...base, deterministic_replay: false }).reasons.some((reason) => reason.includes("replayed")),
  );
  assert.ok(
    verifyChampionReceipt({ ...base, matches_exhaustive_optimum: false }).reasons.some((reason) =>
      reason.includes("exhaustive"),
    ),
  );
  assert.ok(
    verifyChampionReceipt({ ...base, champion: { ...base.champion, genome: { channel: "text", alpha: 0.5 } } }).reasons.some(
      (reason) => reason.includes("loosens the pre-registered"),
    ),
  );
});
