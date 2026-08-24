import { mkdir, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import {
  ACCEPTANCE_TARGETS,
  ROOT_POLICY,
  benchmarkPolicy,
  makeSuite,
} from "../lib/simulator.mjs";

const suite = makeSuite(
  "air-stage-benchmark-v1",
  Array.from({ length: 64 }, (_, index) => index + 401),
  1.35,
);
const result = benchmarkPolicy(ROOT_POLICY, suite);
const receipt = Object.freeze({
  schema: "latentmesh-air-stage-benchmark-v1",
  evidence: "simulated",
  targets: ACCEPTANCE_TARGETS,
  policy: ROOT_POLICY,
  suite: Object.freeze({ id: suite.id, cases: suite.items.length }),
  result,
});

const here = dirname(fileURLToPath(import.meta.url));
const output = join(here, "..", "artifacts");
await mkdir(output, { recursive: true });
await writeFile(join(output, "stage-benchmark.json"), `${JSON.stringify(receipt, null, 2)}\n`);
console.log(JSON.stringify(receipt, null, 2));
