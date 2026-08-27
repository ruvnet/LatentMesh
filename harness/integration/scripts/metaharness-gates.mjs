// ADR-022's optional gate layer: score, genome, mcp-scan, threat-model,
// invoked as pinned external CLI tooling (see lib/metaharness-gates.mjs).
// Per ADR-022's Decision section, absence of the external tool degrades
// this receipt's individual gate entries — it never fails this script or
// `npm run validate`; `cargo test --workspace` never depends on this step
// either. Each gate's pass/fail claim is scoped to exactly what it
// measures: a passing threat-model gate is not a live-security-review
// claim, a passing score/genome gate is not a production-readiness claim.

import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { GATE_NAMES, runMetaharnessGates } from "../lib/metaharness-gates.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const artifactsDir = join(here, "..", "artifacts");
const receiptPath = join(artifactsDir, "metaharness-gates-receipt.json");

mkdirSync(artifactsDir, { recursive: true });

// Gates run once, against the integration wave's three crates as a group —
// the harness directory itself is the natural "target" for score/genome/
// mcp-scan/threat-model since ADR-022 scopes this gate layer to the
// integration wave, not to any one crate.
const targetPath = "crates";
const result = runMetaharnessGates(targetPath);

const receipt = {
  schema: "latentmesh-integration-metaharness-gates-receipt-v1",
  evidence: "simulated",
  evidence_detail: "deterministic simulation — no hardware, no live peers, no credentials",
  tool_present: result.toolPresent,
  tool_path: result.toolPath,
  target: targetPath,
  gates: result.gates,
  not_claimed: [
    "a passing threat-model gate is not a live-security-review claim",
    "a passing score/genome gate is not a production-readiness claim",
    "these gates are never a build or runtime dependency of any workspace crate — cargo test --workspace never depends on their presence",
    result.toolPresent
      ? "external MetaHarness tooling was invoked from a locally-resolved path only, never installed or fetched over the network by this harness"
      : "no external MetaHarness tooling (METAHARNESS_CLI) was present in this environment — every gate below is skipped, not failed",
  ],
};

writeFileSync(receiptPath, `${JSON.stringify(receipt, null, 2)}\n`);
console.log(JSON.stringify(receipt, null, 2));

const skippedCount = GATE_NAMES.filter((name) => receipt.gates[name].skipped).length;
if (skippedCount === GATE_NAMES.length) {
  console.log(
    "all MetaHarness gates skipped (METAHARNESS_CLI unset) — this is expected in a fresh checkout and does not fail validate.",
  );
} else {
  console.log(`${GATE_NAMES.length - skippedCount}/${GATE_NAMES.length} MetaHarness gates ran.`);
}
// This step never exits non-zero: an optional, removable augmentation layer
// cannot fail the suite it augments (ADR-022's Decision section).
