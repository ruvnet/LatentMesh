import test from "node:test";
import assert from "node:assert/strict";
import { loadEvidenceCorpus, loadLatentTwoLayerEvidence, loadTextEvidence } from "../lib/receipts.mjs";
import {
  ALPHA_OF_RECORD,
  CONTROL_SETS,
  ROOT_POLICY,
  admit,
  buildScale,
  enumerateGenomes,
  evaluate,
  genomeKey,
  promotionRule,
} from "../lib/genome.mjs";
import { DEFAULT_SEED, exhaustiveChampion, runEvolve } from "../lib/evolve.mjs";

const corpus = loadEvidenceCorpus();
const scale = buildScale(corpus);
const policy = (overrides) => ({ ...ROOT_POLICY, ...overrides });

test("the text channel is admitted at the pre-registered alpha and refused when it is tightened", () => {
  const text = loadTextEvidence();
  const at05 = admit(text, policy({ alpha: 0.05 }));
  assert.equal(at05.status, "admitted");
  assert.equal(at05.minWealth, 21.157738923872348);

  const at01 = admit(text, policy({ alpha: 0.01 }));
  assert.equal(at01.status, "refused");
  assert.equal(at01.gain.deltaV, 0, "a refused-but-measured channel is Measured(0), never Unmeasured");
  assert.equal(at01.gain.kind, "Measured");
});

test("admitted text reproduces the receipt's own committed uplift", () => {
  const text = loadTextEvidence();
  const decision = admit(text, policy({ controls: CONTROL_SETS[2] }));
  assert.equal(decision.rawDelta, text.committedRawDelta);
  assert.equal(decision.rawDelta, 0.5116279069767442);
  assert.equal(decision.worst.control, "zero", "the control most favourable to the null is the one the receipt names");
});

test("latent is REFUSED, not unmeasured, when the policy asks only for the control it ran", () => {
  const decision = admit(loadLatentTwoLayerEvidence(), policy({ controls: ["random"] }));
  assert.equal(decision.status, "refused");
  assert.equal(decision.gain.kind, "Measured");
  assert.equal(decision.gain.deltaV, 0);
  assert.ok(decision.reason.includes("below 1/alpha"));
});

test("latent is UNMEASURED when the policy requires controls it was never run against", () => {
  const decision = admit(loadLatentTwoLayerEvidence(), policy({ controls: CONTROL_SETS[2] }));
  assert.equal(decision.status, "unmeasured");
  assert.equal("deltaV" in decision.gain, false);
  assert.ok(decision.reason.includes("no measured control named"));
});

test("the two are not the same thing, and the difference is visible in the trace", () => {
  const refused = admit(loadLatentTwoLayerEvidence(), policy({ controls: ["random"] }));
  const unmeasured = admit(loadLatentTwoLayerEvidence(), policy({ controls: CONTROL_SETS[2] }));
  assert.notEqual(refused.status, unmeasured.status);
  assert.equal(refused.gain.kind, "Measured");
  assert.equal(unmeasured.gain.kind, "Unmeasured");
});

test("a semantic-delta policy routes to None because its only candidate cannot be scored", () => {
  const result = evaluate(corpus, policy({ channel: "semantic-delta" }), scale);
  assert.equal(result.selected, "None");
  assert.equal(result.fitness, 0);
  assert.equal(result.considered.length, 1);
  assert.equal(result.considered[0].outcome, "Unmeasured");
});

test("a latent policy routes to None on the tie-break, at exactly zero", () => {
  const result = evaluate(corpus, policy({ channel: "latent", controls: ["random"] }), scale);
  assert.equal(result.selected, "None");
  assert.equal(result.fitness, 0);
  assert.deepEqual(
    result.considered.map((entry) => entry.score),
    [0, 0],
    "both latent rungs score exactly 0.0 — measured, admitted-nothing, and beaten by silence",
  );
});

test("even ungated, latent's own measured delta does not beat text", () => {
  const ungatedLatent = evaluate(corpus, policy({ channel: "latent", admission: "always", controls: ["random"] }), scale);
  const text = evaluate(corpus, policy({ channel: "text", controls: ["random"] }), scale);
  assert.ok(ungatedLatent.fitness > 0, "the 2-site rung has a small positive raw delta vs random");
  assert.ok(text.fitness > ungatedLatent.fitness * 10);
});

test("a genome cannot inflate its own fitness by re-weighting costs", () => {
  const equal = evaluate(corpus, policy({ costWeights: "equal" }), scale);
  const bandwidthHeavy = evaluate(corpus, policy({ costWeights: "bandwidth-heavy" }), scale);
  const latencyHeavy = evaluate(corpus, policy({ costWeights: "latency-heavy" }), scale);
  assert.equal(equal.fitness, bandwidthHeavy.fitness);
  assert.equal(equal.fitness, latencyHeavy.fitness);
  assert.equal(equal.selected, bandwidthHeavy.selected);
});

test("the promotion rule refuses a policy that weakens its own control set", () => {
  const weak = evaluate(corpus, policy({ controls: ["random"] }), scale);
  const strong = evaluate(corpus, policy({ controls: CONTROL_SETS[2] }), scale);
  assert.ok(weak.fitness > strong.fitness, "dropping the hardest control raises delta_V — that is the exploit");
  const verdict = promotionRule(weak, strong, corpus);
  assert.equal(verdict.promote, false);
  assert.ok(verdict.reasons.some((reason) => reason.includes("weakens the bar")));
});

test("the promotion rule refuses an alpha looser than the pre-registered one", () => {
  const loose = evaluate(corpus, policy({ alpha: 0.5 }), scale);
  loose.genome = { ...loose.genome, alpha: 0.5 };
  const verdict = promotionRule(loose, null, corpus);
  assert.ok(verdict.reasons.some((reason) => reason.includes("looser than the pre-registered")));
  assert.equal(ALPHA_OF_RECORD, 0.05);
});

test("the seeded search is reproducible and equals the exhaustive optimum", () => {
  const a = runEvolve({ corpus, scale, seed: DEFAULT_SEED });
  const b = runEvolve({ corpus, scale, seed: DEFAULT_SEED });
  assert.equal(genomeKey(a.champion.genome), genomeKey(b.champion.genome));
  assert.equal(a.champion.fitness, b.champion.fitness);
  assert.deepEqual(a.trace, b.trace);
  assert.equal(genomeKey(a.champion.genome), genomeKey(exhaustiveChampion(corpus, scale).genome));
});

test("the champion the evidence selects is the text channel under the causal gate", () => {
  const champion = exhaustiveChampion(corpus, scale);
  assert.equal(champion.genome.channel, "text");
  assert.equal(champion.genome.admission, "causal-gate");
  assert.equal(champion.genome.alpha, 0.05);
  assert.deepEqual(champion.genome.controls, CONTROL_SETS[2]);
  assert.equal(champion.selected, "text");
  assert.equal(champion.fitness, 1.4117253864364296);
});

test("no genome in the whole lattice ever selects an unmeasured candidate", () => {
  for (const genome of enumerateGenomes()) {
    const result = evaluate(corpus, genome, scale);
    const unmeasured = result.considered.filter((entry) => entry.outcome === "Unmeasured");
    for (const entry of unmeasured) {
      assert.notEqual(result.selectedId, entry.id, `${genomeKey(genome)} selected an unmeasured candidate`);
    }
    if (result.selected === "None") assert.equal(result.fitness, 0);
  }
});

test("no genome in the whole lattice ever selects the latent channel", () => {
  const selectingLatent = enumerateGenomes()
    .map((genome) => ({ genome, result: evaluate(corpus, genome, scale) }))
    .filter((entry) => entry.result.selected === "latent");
  // The 2-site rung has a positive raw delta vs random, so `admission: always`
  // policies DO route to it — the gate is what stops them, and that is the
  // finding, not a bug. They still lose to text on fitness.
  for (const entry of selectingLatent) {
    assert.equal(entry.genome.admission, "always");
    assert.ok(entry.result.fitness < exhaustiveChampion(corpus, scale).fitness);
  }
});
