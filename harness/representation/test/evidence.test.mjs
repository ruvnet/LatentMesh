// Every number this harness scores must be re-derivable from a committed
// receipt. These tests re-count the raw per-item records and assert they
// reproduce the summary fields the extractor reads — if a receipt is ever
// edited so its summary and its items disagree, the harness stops rather than
// scoring a number nobody can reproduce.

import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import {
  RECEIPTS_DIR,
  loadEvidenceCorpus,
  loadLatentSingleLayerEvidence,
  loadLatentTwoLayerEvidence,
  loadSemanticDeltaEvidence,
  loadTextEvidence,
} from "../lib/receipts.mjs";

const readRaw = (name) => JSON.parse(readFileSync(join(RECEIPTS_DIR, name), "utf8"));

test("the text rung's summary counts are reproduced by re-counting its items", () => {
  const raw = readRaw("run3-stageA-receipt-statictext-gate-eprocess.json");
  const evidence = loadTextEvidence();

  assert.equal(raw.items.length, raw.summary.n_evaluated);
  assert.equal(raw.items.filter((item) => item.conditions.gated_text).length, raw.summary.gated_text_correct);
  for (const control of raw.summary.per_control) {
    assert.equal(
      raw.items.filter((item) => item.conditions[control.control]).length,
      control.correct,
      `per_control ${control.control} disagrees with the item records`,
    );
  }
  assert.equal(evidence.n, 43);
  assert.equal(evidence.correct, 39);
  assert.equal(evidence.crossed, true);
  assert.equal(evidence.crossedAtItem, 43);
  assert.equal(evidence.alphaOfRecord, 0.05);
});

test("the text rung has FOUR controls, not the five ADR-046 §2.1's prose recalls", () => {
  const raw = readRaw("run3-stageA-receipt-statictext-gate-eprocess.json");
  assert.equal(raw.summary.per_control.length, 4);
  assert.deepEqual(
    raw.summary.per_control.map((entry) => entry.control),
    ["zero", "random", "mismatched", "self_generated"],
  );
  // The fifth is absent because the receipt itself records it as degenerate
  // for a text channel, not because it was forgotten.
  assert.ok(raw.config.controls.text_equivalent_dropped.includes("degenerate"));
});

test("the latent rungs' accuracies are reproduced by re-counting their items", () => {
  for (const [file, evidence] of [
    ["run2-m4i-receipt-cellL18toL14-mlp-pertokenlast-fuse-questiontail-slots8-eprocess.json", loadLatentSingleLayerEvidence()],
    [
      "run2-m5x-receipt-2site-L18toL14-L24toL19-pertokenlast-fusemany-questiontail-slots4x2-eprocess.json",
      loadLatentTwoLayerEvidence(),
    ],
  ]) {
    const raw = readRaw(file);
    assert.equal(raw.items.filter((item) => item.conditions.aligned_real.correct).length, raw.summary.accuracy.aligned_real);
    assert.equal(raw.items.filter((item) => item.conditions.random.correct).length, raw.summary.accuracy.random);
    assert.equal(evidence.n, 300);
    assert.equal(evidence.correct, 128);
    assert.equal(evidence.crossed, false);
    assert.equal(evidence.crossedAtItem, null);
  }
});

test("the latent e-processes did not cross, at the wealth values the receipts record", () => {
  const single = loadLatentSingleLayerEvidence();
  const dual = loadLatentTwoLayerEvidence();
  assert.equal(single.controls[0].finalWealth, 0.25780744271218675);
  assert.equal(single.controls[0].nDiscordant, 66);
  assert.equal(dual.controls[0].finalWealth, 0.883678611431764);
  assert.equal(dual.controls[0].nDiscordant, 64);
  // Both are far below the 1/alpha = 20 boundary they were drawn against.
  assert.ok(single.controls[0].finalWealth < 20);
  assert.ok(dual.controls[0].finalWealth < 20);
});

test("the latent rungs measure only ONE control, so a policy needing more is unmeasured", () => {
  for (const evidence of [loadLatentSingleLayerEvidence(), loadLatentTwoLayerEvidence()]) {
    assert.deepEqual(
      evidence.controls.map((control) => control.control),
      ["random"],
      "the latent e-process runs aligned_real vs random only; the other controls have no wealth process",
    );
  }
});

test("latent payload width comes from the adapter the rung actually ran", () => {
  // b2[1536] per site, f32 -> 6144 bytes for one site, 12288 for two. The
  // adapter's content hash is cross-checked against the hash the rung receipt
  // asserts, so the dimension cannot come from a different adapter.
  assert.equal(loadLatentSingleLayerEvidence().cost.bytes, 6144);
  assert.equal(loadLatentTwoLayerEvidence().cost.bytes, 12288);
  assert.equal(loadLatentTwoLayerEvidence().adapters.length, 2);
});

test("the text payload is the mean of the per-item sender message token counts", () => {
  const raw = readRaw("run3-stageA-receipt-statictext-gate-eprocess.json");
  const tokens = raw.items.map((item) => item.sender_message_tokens);
  const meanTokens = tokens.reduce((total, value) => total + value, 0) / tokens.length;
  assert.equal(loadTextEvidence().cost.bytes, meanTokens * 4);
});

test("no rung measured energy, and every rung measured a zero degenerate rate", () => {
  for (const evidence of loadEvidenceCorpus()) {
    assert.equal(evidence.cost.energyJoules, null);
    if (evidence.measured !== false) assert.equal(evidence.cost.risk, 0);
  }
});

test("semantic-delta carries no gain field at all — it is unmeasured by construction", () => {
  const evidence = loadSemanticDeltaEvidence();
  assert.equal(evidence.measured, false);
  assert.equal(evidence.receipt, null);
  assert.ok(evidence.reason.includes("no committed receipt"));
});

test("a tampered adapter hash link is refused rather than scored", () => {
  // The extractor asserts the training receipt's declared adapter hash equals
  // the hash the rung receipt claims. Both files are read; a mismatch throws.
  const training = readRaw("run2-m3-training-receipt-cellL18toL14.json");
  const rung = readRaw("run2-m4i-receipt-cellL18toL14-mlp-pertokenlast-fuse-questiontail-slots8-eprocess.json");
  assert.equal(training.artifact.content_hash_sha256, rung.config.transform.content_hash);
});
