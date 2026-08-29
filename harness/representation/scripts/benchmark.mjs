// Score every representation channel the committed corpus can speak to,
// side by side, under the root policy — the "what does the evidence actually
// say" pass, before any evolution happens.
//
// Two readings are emitted because the two channels were not run against the
// same controls, and collapsing that into one number would hide it:
//
//   common-control  the only control BOTH channels measured (`random`).
//                   Latent is Measured here, and gate-refused.
//   strict-control  the root policy's full four-control set. Latent has no
//                   wealth process for three of them, so it is Unmeasured —
//                   an evidence gap reported as a gap.
//
// Writes artifacts/representation-benchmark.json.

import { mkdir, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { loadEvidenceCorpus, UNMEASURED_RESOURCES } from "../lib/receipts.mjs";
import { assertCostParity, route } from "../lib/routing.mjs";
import { ROOT_POLICY, admit, buildScale, weightsOf } from "../lib/genome.mjs";
import { verifyBenchmarkReceipt } from "../lib/verify.mjs";

const corpus = loadEvidenceCorpus();
const scale = buildScale(corpus);
const weights = weightsOf(ROOT_POLICY);

function reading(name, controls, description) {
  const policy = { ...ROOT_POLICY, controls };
  const candidates = corpus.map((evidence) => {
    const decision = admit(evidence, policy);
    return { id: evidence.id, mode: evidence.channel, gain: decision.gain, cost: evidence.cost, admission: decision };
  });
  assertCostParity(candidates.filter((candidate) => candidate.gain.kind === "Measured"));
  const decision = route(candidates, scale, weights);
  return {
    name,
    description,
    controls,
    selected: decision.selected,
    selectedId: decision.selectedId,
    selectedScore: decision.selectedScore,
    candidates: decision.considered.map((entry) => {
      const source = candidates.find((candidate) => candidate.id === entry.id);
      return {
        ...entry,
        admission: source.admission.status,
        admissionReason: source.admission.reason,
        rawDelta: source.admission.rawDelta ?? null,
        worstControl: source.admission.worst ? source.admission.worst.control : null,
        cost: source.cost,
      };
    }),
  };
}

const common = reading(
  "common-control",
  ["random"],
  "the only control measured for BOTH channels; the like-for-like comparison",
);
const strict = reading(
  "strict-control",
  ROOT_POLICY.controls,
  "the root policy's four-control set; latent has no wealth process for zero/mismatched/self_generated",
);

const receipt = {
  schema: "latentmesh-representation-benchmark-v1",
  evidence: "measured-committed-receipts",
  adr: "046",
  policy: ROOT_POLICY,
  cost_scale: {
    values: scale,
    provenance:
      "maximum measured value per resource across the whole corpus, so every normalised cost lands in [0,1]. A corpus-derived reference, NOT a deployment fact — routing.rs deliberately ships no default CostScale for exactly this reason.",
  },
  cost_weights: weights,
  unmeasured_resources: UNMEASURED_RESOURCES,
  unmeasured_resource_note:
    "energy was never measured by any rung. An unmeasured COST is dropped from the denominator (and the remaining weights renormalised), which is the OPPOSITE treatment to an unmeasured GAIN, which is ineligible. The asymmetry is safe only because the same resource is unmeasured for every candidate — cost parity is asserted before scoring.",
  readings: [common, strict],
  // The primary reading is the like-for-like one.
  selected: common.selected,
  selectedId: common.selectedId,
  selectedScore: common.selectedScore,
  candidates: common.candidates,
  sources: corpus.map((evidence) => ({
    id: evidence.id,
    channel: evidence.channel,
    receipt: evidence.receipt,
    receiptSha256: evidence.receiptSha256,
    measured: evidence.measured !== false,
    n: evidence.n ?? null,
    correct: evidence.correct ?? null,
    controls: (evidence.controls ?? []).map((control) => ({
      control: control.control,
      correct: control.correct,
      finalWealth: control.finalWealth,
    })),
    crossed: evidence.crossed ?? null,
    crossedAtItem: evidence.crossedAtItem ?? null,
    cost: evidence.cost,
    fieldPaths: evidence.fieldPaths ?? null,
  })),
  not_claimed: [
    "does NOT claim latent state transport works; it claims the committed measurement of it did not clear its own pre-registered bar",
    "does NOT re-run any statistical test — every wealth, count and delta is read from a committed receipt",
    "does NOT measure energy; that resource is absent from every rung and is dropped from the denominator",
    "does NOT compare the two channels' `random` controls to each other — text's random is random TOKENS, latent's is a norm-matched random VECTOR; each control is only ever used against its own channel",
    "does NOT change MetaHarness upstream, Darwin's seven surfaces, or the runtime — this harness is dev-only and removable (ADR-046 §6)",
  ],
};

const verdict = verifyBenchmarkReceipt(receipt);
const here = dirname(fileURLToPath(import.meta.url));
const output = join(here, "..", "artifacts");
await mkdir(output, { recursive: true });
await writeFile(join(output, "representation-benchmark.json"), `${JSON.stringify(receipt, null, 2)}\n`);

console.log(
  JSON.stringify(
    {
      evidence: receipt.evidence,
      readings: receipt.readings.map((entry) => ({
        name: entry.name,
        selected: entry.selected,
        selectedScore: entry.selectedScore,
        candidates: entry.candidates.map((candidate) => ({
          id: candidate.id,
          outcome: candidate.outcome,
          score: candidate.score,
          admission: candidate.admission,
        })),
      })),
      pass: verdict.pass,
      reasons: verdict.reasons,
    },
    null,
    2,
  ),
);

if (!verdict.pass) {
  console.error("benchmark receipt failed its own verification — see reasons above.");
  process.exit(1);
}
