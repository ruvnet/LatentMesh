import { mkdir, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import {
  makeSigner,
  runFlywheelGenerations,
  verifyReplayBundle,
} from "@metaharness/flywheel";
import {
  DOMAINS,
  ROOT_POLICY,
  evaluatePolicy,
  makeSuite,
  promotionRule,
} from "../lib/simulator.mjs";

const holdout = makeSuite("air-holdout-v1", Array.from({ length: 16 }, (_, index) => index + 1));
const anchor = makeSuite("air-anchor-v1", Array.from({ length: 16 }, (_, index) => index + 101), 1.15);
const cursors = Object.fromEntries(Object.keys(DOMAINS).map((key) => [key, 0]));

async function proposer(base, target) {
  const domain = DOMAINS[target];
  for (let offset = 0; offset < domain.length; offset += 1) {
    const candidate = domain[(cursors[target] + offset) % domain.length];
    if (candidate !== base.policy[target]) {
      cursors[target] = (cursors[target] + offset + 1) % domain.length;
      return candidate;
    }
  }
  return base.policy[target];
}

const startedAt = new Date().toISOString();
const result = await runFlywheelGenerations({
  rootPolicy: ROOT_POLICY,
  proposer,
  evaluator: evaluatePolicy,
  promotionRule,
  holdout,
  anchor,
  maxGenerations: 12,
  signer: makeSigner(),
  dataSource: "SIMULATED",
  cacheEvaluations: true,
  now: (generation) => `${startedAt}#gen${generation}`,
});

const replayVerdict = verifyReplayBundle(result.replayBundle);
if (!replayVerdict.pass) throw new Error(`MetaHarness replay verification failed: ${JSON.stringify(replayVerdict)}`);

const here = dirname(fileURLToPath(import.meta.url));
const output = join(here, "..", "artifacts");
await mkdir(output, { recursive: true });
await writeFile(join(output, "replay-bundle.json"), `${JSON.stringify(result.replayBundle, null, 2)}\n`);
await writeFile(join(output, "tuned-policy.json"), `${JSON.stringify({
  schema: "latentmesh-air-policy-v1",
  evidence: "simulated",
  root: ROOT_POLICY,
  tuned: result.finalPolicy,
  liftCurve: result.liftCurve,
  replayVerified: true,
  startedAt,
}, null, 2)}\n`);

console.log(JSON.stringify({
  evidence: "simulated",
  generations: result.generationsRun,
  promotions: result.promotions.length,
  tuned: result.finalPolicy,
  replayVerified: true,
}, null, 2));
