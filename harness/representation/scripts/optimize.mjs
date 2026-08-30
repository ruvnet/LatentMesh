// Evolve `representationPolicy` — ADR-046 §3's eighth policy surface — over
// the committed receipts, and emit a hash-chained champion receipt.
//
// Three guards against the failure mode that would make this worthless (a
// harness that returns the answer its author wanted):
//
//  1. the seeded search is replayed and must land on the identical champion;
//  2. the champion must equal the EXHAUSTIVE optimum over the whole lattice,
//     so "it converged" cannot stand in for "it is right";
//  3. the counterfactual with the no-weakening clause removed is computed and
//     reported whether or not it agrees, so the constraint's effect is
//     visible instead of assumed.
//
// Writes artifacts/champion-policy.json.

import { mkdir, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { loadEvidenceCorpus, UNMEASURED_RESOURCES } from "../lib/receipts.mjs";
import { ROOT_POLICY, buildScale, evaluate, genomeKey } from "../lib/genome.mjs";
import {
  DEFAULT_SEED,
  exhaustiveChampion,
  exhaustiveChampionWithoutControlFloor,
  runEvolve,
} from "../lib/evolve.mjs";
import { ReceiptLog, verifyChain } from "../lib/chain.mjs";
import { verifyChampionReceipt } from "../lib/verify.mjs";

const corpus = loadEvidenceCorpus();
const scale = buildScale(corpus);

const first = runEvolve({ corpus, scale, seed: DEFAULT_SEED });
const replay = runEvolve({ corpus, scale, seed: DEFAULT_SEED });
const deterministic = genomeKey(first.champion.genome) === genomeKey(replay.champion.genome);

const exhaustive = exhaustiveChampion(corpus, scale);
const matchesExhaustive = genomeKey(first.champion.genome) === genomeKey(exhaustive.genome);
const unconstrained = exhaustiveChampionWithoutControlFloor(corpus, scale);

const log = new ReceiptLog();
log.append("root-policy", { genome: ROOT_POLICY, evaluation: summarise(evaluate(corpus, ROOT_POLICY, scale)) });
for (const generation of first.trace) log.append("generation", generation);
log.append("champion", summarise(first.champion));

function summarise(evaluation) {
  return {
    genome: evaluation.genome,
    selected: evaluation.selected,
    selectedId: evaluation.selectedId,
    fitness: evaluation.fitness,
    considered: evaluation.considered,
    admissions: evaluation.admissions,
  };
}

const chain = log.toJSON();
const chainVerdict = verifyChain(chain);

const receipt = {
  schema: "latentmesh-representation-champion-v1",
  evidence: "measured-committed-receipts",
  adr: "046",
  seed: DEFAULT_SEED,
  generations: first.generations,
  distinct_policies_evaluated: first.evaluations,
  lattice_size: 4 * 3 * 3 * 3 * 4,
  root_policy: ROOT_POLICY,
  champion: summarise(first.champion),
  deterministic_replay: deterministic,
  matches_exhaustive_optimum: matchesExhaustive,
  exhaustive_optimum: summarise(exhaustive),
  unconstrained_note: {
    what:
      "the same exhaustive search with the promotion rule's no-weakening clause removed, so a policy MAY drop the hardest control it was measured against",
    why:
      "'worst of the required controls' is monotone: dropping a control can only raise delta_V. Without the clause the search is rewarded for weakening its own gate. Reported rather than hidden.",
    champion: unconstrained ? summarise(unconstrained) : null,
    differs_from_constrained_champion: unconstrained
      ? genomeKey(unconstrained.genome) !== genomeKey(first.champion.genome)
      : null,
  },
  cost_scale: scale,
  unmeasured_resources: UNMEASURED_RESOURCES,
  chain,
  chain_verified: chainVerdict.pass,
  sources: corpus.map((evidence) => ({
    id: evidence.id,
    channel: evidence.channel,
    receipt: evidence.receipt,
    receiptSha256: evidence.receiptSha256,
    measured: evidence.measured !== false,
  })),
  not_claimed: [
    "does NOT revive the latent channel — it stays in the lattice as a measurable option that currently scores zero (ADR-046 §5)",
    "does NOT re-run any statistical test; admission re-reads committed wealth values and only ever at an alpha at least as strict as the pre-registered 0.05",
    "does NOT assert that the champion genome is deployed — nothing in the runtime reads this file (ADR-046 §6)",
    "does NOT claim the cost scale is a deployment fact; it is derived from this corpus",
  ],
};

const verdict = verifyChampionReceipt(receipt);
const here = dirname(fileURLToPath(import.meta.url));
const output = join(here, "..", "artifacts");
await mkdir(output, { recursive: true });
await writeFile(join(output, "champion-policy.json"), `${JSON.stringify(receipt, null, 2)}\n`);

console.log(
  JSON.stringify(
    {
      evidence: receipt.evidence,
      champion: first.champion.genome,
      selected: first.champion.selected,
      selectedId: first.champion.selectedId,
      fitness: first.champion.fitness,
      admissions: first.champion.admissions,
      deterministic_replay: deterministic,
      matches_exhaustive_optimum: matchesExhaustive,
      unconstrained_champion: unconstrained ? unconstrained.genome : null,
      chain_head: chain.head,
      chain_verified: chainVerdict.pass,
      pass: verdict.pass,
      reasons: verdict.reasons,
    },
    null,
    2,
  ),
);

if (!verdict.pass) {
  console.error("champion receipt failed its own verification — see reasons above.");
  process.exit(1);
}
