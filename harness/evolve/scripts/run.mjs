// Run the ADR-018 Darwin acceptance suite via the latentmesh-evolve binary
// and verify its receipt against ADR-006's bounds. Writes the receipt and the
// verification verdict to harness/evolve/artifacts/, in the same
// evidence-labelled spirit as harness/air's artifacts.

import { execFileSync } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { verifyReceipt } from "../lib/verify.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = join(here, "..", "..", "..");
const artifactsDir = join(here, "..", "artifacts");
const receiptPath = join(artifactsDir, "evolve-receipt.json");

mkdirSync(artifactsDir, { recursive: true });

console.log("building and running latentmesh-evolve (release, frozen seed)...");
execFileSync(
  "cargo",
  ["run", "--release", "--quiet", "-p", "latentmesh-evolve", "--", "--receipt", receiptPath],
  { cwd: repoRoot, stdio: ["ignore", "inherit", "inherit"] },
);

const receipt = JSON.parse(readFileSync(receiptPath, "utf8"));
const verdict = verifyReceipt(receipt);

const summary = {
  schema: "latentmesh-evolve-harness-verdict-v1",
  evidence: receipt.evidence,
  receipt: "evolve-receipt.json",
  bounds: { computeReduction: 0.3, taskSuccessNonDecreasing: true, allEdgesCausallyVerified: true },
  pass: verdict.pass,
  reasons: verdict.reasons,
};
writeFileSync(join(artifactsDir, "harness-verdict.json"), `${JSON.stringify(summary, null, 2)}\n`);

console.log(JSON.stringify(summary, null, 2));
if (!verdict.pass) {
  console.error("ADR-006 acceptance bounds NOT met — see reasons above.");
  process.exit(1);
}
console.log("ADR-006 acceptance bounds met (deterministic simulation — no live-agent claim).");
