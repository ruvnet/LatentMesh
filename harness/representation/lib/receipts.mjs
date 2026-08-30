// Read the REAL, ALREADY-COMMITTED measurement receipts under
// crates/latentmesh-runtime/receipts/ and turn them into the evidence this
// harness scores. There is no simulator here and there must never be one:
// ADR-046 §4's entire claim is that fitness is computed from receipts this
// repository already holds, so a synthetic evaluator would falsify the one
// property the harness exists to demonstrate.
//
// Every value below is read from a named field path, and the path is carried
// into the emitted receipt (`fieldPaths`) so a reader can re-check it with
// `jq` against the same file. Nothing is remembered, re-derived from prose,
// or copied out of an ADR — where the ADR's prose and the receipt disagree,
// the receipt wins and the disagreement is reported (see README §Discrepancies).

import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
export const RECEIPTS_DIR = join(here, "..", "..", "..", "crates", "latentmesh-runtime", "receipts");

/** Bytes per token id on the wire: Qwen2.5's ordinary-id ceiling exceeds
 *  2^16 (run3 `config.random_control_vocabulary.ordinary_ids_below`), so a
 *  token id does not fit in u16 and a u32 encoding is the smallest honest
 *  one. Stated as an ENCODING ASSUMPTION, not a measurement. */
export const BYTES_PER_TOKEN_ID = 4;
/** Bytes per transported activation scalar: the adapters' own declared
 *  `artifact.layout` says "little-endian f32". */
export const BYTES_PER_F32 = 4;

function readReceipt(name) {
  const path = join(RECEIPTS_DIR, name);
  const raw = readFileSync(path, "utf8");
  return { name, path, sha256: createHash("sha256").update(raw).digest("hex"), json: JSON.parse(raw) };
}

/** Pull `b2[N]` out of an adapter training receipt's `artifact.layout` — the
 *  output width of the reconstruction MLP is the number of f32 scalars one
 *  latent payload actually carries. */
function payloadFloatsFromLayout(layout) {
  const match = /\bb2\[(\d+)\]/.exec(String(layout));
  if (!match) throw new Error(`adapter layout does not declare b2[N]: ${layout}`);
  return Number(match[1]);
}

function mean(values) {
  return values.reduce((total, value) => total + value, 0) / values.length;
}

/**
 * The text channel: run 3 stage A, a static-text 2-agent pipeline gated by
 * ADR-030's e-process against four controls.
 *
 * NOTE ON THE CONTROL COUNT. ADR-046 §2.1 says "the worst of five controls".
 * This receipt has FOUR (`config.controls.text_equivalent_dropped` records
 * that the fifth is degenerate when the channel under test is itself text).
 * The four present are used; the fifth is not invented.
 */
export function loadTextEvidence() {
  const { name, sha256, json } = readReceipt("run3-stageA-receipt-statictext-gate-eprocess.json");
  const n = json.summary.n_evaluated;
  const controls = json.summary.per_control.map((entry) => ({
    control: entry.control,
    correct: entry.correct,
    finalWealth: entry.e_process.final_wealth,
    winsText: entry.e_process.wins_text,
    lossesText: entry.e_process.losses_text,
    nDiscordant: entry.e_process.n_discordant,
    exactSignOneSided: entry.exact_sign_one_sided,
  }));
  const cost = json.compute_cost_seconds;
  return Object.freeze({
    id: "text/run3-stageA-statictext",
    channel: "text",
    receipt: name,
    receiptSha256: sha256,
    n,
    correct: json.summary.gated_text_correct,
    controls,
    alphaOfRecord: json.e_process.alpha,
    crossed: json.summary.e_process_crossed,
    crossedAtItem: json.summary.e_process_crossed_at_item,
    minWealthAcrossControls: json.summary.min_wealth_across_controls,
    committedRawDelta: json.summary.uplift_vs_best_control.raw_delta,
    cost: Object.freeze({
      // One item's payload is the sender's own generated solution text.
      bytes: mean(json.items.map((item) => item.sender_message_tokens)) * BYTES_PER_TOKEN_ID,
      // Per-condition GPU seconds ARE recorded here, so the channel's own two
      // passes can be charged exactly rather than amortised.
      latencySeconds: (cost.sender_generation + cost.receiver_gated_text) / n,
      energyJoules: null, // never measured in any of these rungs
      risk: json.summary.n_degenerate_sender_pass / n,
    }),
    fieldPaths: Object.freeze({
      n: "summary.n_evaluated",
      correct: "summary.gated_text_correct",
      controls: "summary.per_control[].{control,correct,e_process.final_wealth}",
      alphaOfRecord: "e_process.alpha",
      crossed: "summary.e_process_crossed",
      crossedAtItem: "summary.e_process_crossed_at_item",
      committedRawDelta: "summary.uplift_vs_best_control.raw_delta",
      bytes: "mean(items[].sender_message_tokens) * 4",
      latencySeconds:
        "(compute_cost_seconds.sender_generation + compute_cost_seconds.receiver_gated_text) / summary.n_evaluated",
      risk: "summary.n_degenerate_sender_pass / summary.n_evaluated",
    }),
  });
}

/**
 * One latent rung. `sites` is how many capture->inject adapter sites carry a
 * payload; each contributes `b2[N]` f32 scalars, read from that site's own
 * training receipt (whose `artifact.content_hash_sha256` is cross-checked
 * against the hash the rung receipt itself asserts, so the dimension can only
 * come from the adapter the rung actually ran).
 */
function loadLatentRung({ id, file, adapters, expectedHashes }) {
  const { name, sha256, json } = readReceipt(file);
  const n = json.summary.n_evaluated;
  let payloadFloats = 0;
  const adapterFacts = [];
  for (let index = 0; index < adapters.length; index += 1) {
    const training = readReceipt(adapters[index]).json;
    const declaredHash = training.artifact.content_hash_sha256;
    if (declaredHash !== expectedHashes[index]) {
      throw new Error(
        `${adapters[index]} declares adapter hash ${declaredHash}, but ${file} asserts ${expectedHashes[index]}`,
      );
    }
    const floats = payloadFloatsFromLayout(training.artifact.layout);
    payloadFloats += floats;
    adapterFacts.push({ trainingReceipt: adapters[index], contentHashSha256: declaredHash, payloadFloats: floats });
  }
  // Conditions per item, counted from the receipt's own item structure — the
  // divisor used to amortise wall clock, not a guess about the schedule.
  const conditionsPerItem = Object.keys(json.items[0].conditions).length;
  return Object.freeze({
    id,
    channel: "latent",
    receipt: name,
    receiptSha256: sha256,
    n,
    correct: json.summary.accuracy.aligned_real,
    // The e-process here runs ONE comparison: aligned_real vs random. That is
    // the only control with a wealth process, so it is the only control this
    // channel can be gated on. Absence is reported as absence, never filled in.
    controls: [
      {
        control: "random",
        correct: json.summary.accuracy.random,
        finalWealth: json.e_process.final_wealth,
        winsText: json.e_process.discordant_wins_aligned,
        lossesText: json.e_process.discordant_losses_aligned,
        nDiscordant: json.e_process.n_discordant,
        exactSignOneSided: null,
      },
    ],
    alphaOfRecord: json.e_process.alpha,
    crossed: json.e_process.crossed === true || json.summary.e_process.crossed_at_item_count !== null,
    crossedAtItem: json.summary.e_process.crossed_at_item_count,
    minWealthAcrossControls: json.e_process.final_wealth,
    committedRawDelta: null,
    adapters: adapterFacts,
    cost: Object.freeze({
      bytes: payloadFloats * BYTES_PER_F32,
      // No per-condition breakdown was recorded for these rungs, so wall clock
      // is amortised over items x conditions. This UNDERSTATES the aligned
      // condition's own share if conditions differ in cost — i.e. it is
      // charitable to latent, the channel this harness is expected to reject.
      latencySeconds: json.wall_clock_s / (n * conditionsPerItem),
      energyJoules: null,
      risk: json.summary.n_degenerate_capture / n,
    }),
    fieldPaths: Object.freeze({
      n: "summary.n_evaluated",
      correct: "summary.accuracy.aligned_real",
      controlRandomCorrect: "summary.accuracy.random",
      controlRandomWealth: "e_process.final_wealth",
      nDiscordant: "e_process.n_discordant",
      alphaOfRecord: "e_process.alpha",
      crossedAtItem: "summary.e_process.crossed_at_item_count",
      bytes: "sum over sites of b2[N] from the adapter training receipt's artifact.layout, x 4",
      latencySeconds: "wall_clock_s / (summary.n_evaluated * count(items[0].conditions))",
      risk: "summary.n_degenerate_capture / summary.n_evaluated",
    }),
  });
}

export function loadLatentSingleLayerEvidence() {
  return loadLatentRung({
    id: "latent/run2-m4i-1site-L18toL14",
    file: "run2-m4i-receipt-cellL18toL14-mlp-pertokenlast-fuse-questiontail-slots8-eprocess.json",
    adapters: ["run2-m3-training-receipt-cellL18toL14.json"],
    expectedHashes: ["a864e518cdfb09429988862775278bd7734e4342f8c6445fc6081f5ed9d73c91"],
  });
}

export function loadLatentTwoLayerEvidence() {
  return loadLatentRung({
    id: "latent/run2-m5x-2site-L18toL14-L24toL19",
    file: "run2-m5x-receipt-2site-L18toL14-L24toL19-pertokenlast-fusemany-questiontail-slots4x2-eprocess.json",
    adapters: ["run2-m3-training-receipt-cellL18toL14.json", "run2-m5x-training-receipt-cellL24toL19.json"],
    expectedHashes: [
      "a864e518cdfb09429988862775278bd7734e4342f8c6445fc6081f5ed9d73c91",
      "facd5c6f1f94c83c8eff6f948009502af7cf98ac4bf851718e355a5ec6fba638",
    ],
  });
}

/**
 * `semantic-delta` is in ADR-046 §3's lattice and NO receipt measures it.
 * That is not an oversight to paper over with a plausible number: it is the
 * live demonstration of §2.2. This evidence record carries `measured: false`
 * and no gain field at all, so it is structurally impossible to score.
 */
export function loadSemanticDeltaEvidence() {
  return Object.freeze({
    id: "semantic-delta/unmeasured",
    channel: "semantic-delta",
    receipt: null,
    receiptSha256: null,
    measured: false,
    reason:
      "no committed receipt in crates/latentmesh-runtime/receipts/ measures a semantic-delta channel against any control; ADR-046 §2.2 makes an unmeasured channel ineligible rather than optimistically explored",
    cost: Object.freeze({ bytes: null, latencySeconds: null, energyJoules: null, risk: null }),
  });
}

/** Every candidate representation the committed corpus can speak to. */
export function loadEvidenceCorpus() {
  return Object.freeze([
    loadTextEvidence(),
    loadSemanticDeltaEvidence(),
    loadLatentSingleLayerEvidence(),
    loadLatentTwoLayerEvidence(),
  ]);
}

/** Resources no rung in the corpus ever measured. Reported loudly because an
 *  unmeasured COST silently counts as zero, which is the opposite of how an
 *  unmeasured GAIN is treated — see README §The asymmetry. */
export const UNMEASURED_RESOURCES = Object.freeze(["energyJoules"]);
