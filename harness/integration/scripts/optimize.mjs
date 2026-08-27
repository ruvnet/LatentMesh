// ADR-022's self-optimizing loop: a Darwin-loop-style search (ADR-018's
// pattern, reused for adapter parameters — MTU packing/fragmentation
// threshold, bridge batching interval) over
// `latentmesh-integration/lib/optimizer.mjs`'s search space, evaluated
// against the meshtastic e2e suite's own measured receipt as the fitness
// signal. Requires `artifacts/meshtastic-receipt.json` to already exist —
// run `npm run meshtastic` (or `npm run validate`, which runs suites first)
// before this script.

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { baselineCandidate, evaluateCandidate, runDarwinSearch, SEARCH_SPACE } from "../lib/optimizer.mjs";
import { OPTIMIZE_RECEIPT_SCHEMA, verifyOptimizeReceipt } from "../lib/verify.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const artifactsDir = join(here, "..", "artifacts");
const meshtasticReceiptPath = join(artifactsDir, "meshtastic-receipt.json");
const receiptPath = join(artifactsDir, "optimize-receipt.json");

if (!existsSync(meshtasticReceiptPath)) {
  console.error(
    `optimize: ${meshtasticReceiptPath} does not exist — run \`npm run meshtastic\` first so this loop has a measured receipt to optimize against.`,
  );
  process.exit(1);
}

const meshtasticReceipt = JSON.parse(readFileSync(meshtasticReceiptPath, "utf8"));
const measurements = {
  usableBytesPerPacket: meshtasticReceipt.driver.usable_bytes_per_packet,
  multiFragmentMessageBytes: meshtasticReceipt.driver.multi_fragment.message_bytes,
};

const SEED = 2026_0827;
const GENERATIONS = 10;
const POPULATION_SIZE = 8;
const ELITE_SIZE = 2;

const search = runDarwinSearch({
  seed: SEED,
  generations: GENERATIONS,
  populationSize: POPULATION_SIZE,
  eliteSize: ELITE_SIZE,
  measurements,
});

const baseline = baselineCandidate(measurements);
const baselineResult = evaluateCandidate(baseline, measurements);
const best = search.bestEver;

const receipt = {
  schema: OPTIMIZE_RECEIPT_SCHEMA,
  evidence: "simulated",
  evidence_detail: "deterministic simulation — no hardware, no live peers, no credentials",
  determinism: `seeded LCG PRNG (seed ${SEED}, same generator style as harness/air's lib/simulator.mjs), no wall-clock or OS-RNG input anywhere in the search`,
  seed: SEED,
  generations: GENERATIONS,
  population: POPULATION_SIZE,
  elite_size: ELITE_SIZE,
  search_space: SEARCH_SPACE,
  measurements_source: "artifacts/meshtastic-receipt.json (driver.usable_bytes_per_packet, driver.multi_fragment.message_bytes)",
  measurements,
  mutations_proposed: search.mutationsProposed,
  mutations_rejected: search.mutationsRejected,
  trajectory: search.trajectory,
  baseline: {
    fragmentation_threshold_bytes: baseline.fragmentationThresholdBytes,
    bridge_batching_interval_ms: baseline.bridgeBatchingIntervalMs,
    score: baselineResult.score,
    description: "today's implementation: fragment_message already packs to the measured Meshtastic packet ceiling, and the bridge crate has no batching scheduler (every decoded delta dispatches immediately)",
  },
  selected: {
    fragmentation_threshold_bytes: best.candidate.fragmentationThresholdBytes,
    bridge_batching_interval_ms: best.candidate.bridgeBatchingIntervalMs,
    score: best.result.score,
    fragments: best.result.fragments,
    total_cost: best.result.totalCost,
  },
  round_trip_correctness_preserved:
    best.candidate.fragmentationThresholdBytes <= measurements.usableBytesPerPacket,
  not_claimed: [
    "simulation-optimal only — the cost model (fragment dispatch units, bridge overhead units, batch-wait weighting) is an illustrative simulated proxy, not a measured real-world airtime, latency, or energy claim",
    "no live-agent, live-radio, or live-credential evidence",
    "does not claim to discover new protocol behavior — it tunes packing/batching constants within the space ADR-022 names",
  ],
};

const verdict = verifyOptimizeReceipt(receipt);
receipt.acceptance = { passed: verdict.pass, reasons: verdict.reasons };

mkdirSync(artifactsDir, { recursive: true });
writeFileSync(receiptPath, `${JSON.stringify(receipt, null, 2)}\n`);
console.log(JSON.stringify(receipt, null, 2));

if (!verdict.pass) {
  console.error("optimize loop FAILED verification:", verdict.reasons.join("; "));
  process.exit(1);
}
console.log(
  `optimize loop passed: selected threshold=${receipt.selected.fragmentation_threshold_bytes}B, batching interval=${receipt.selected.bridge_batching_interval_ms}ms (simulation-optimal only).`,
);
