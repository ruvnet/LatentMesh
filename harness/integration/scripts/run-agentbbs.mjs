// ADR-022 e2e suite 2: agentbbs bridge round-trip. Runs
// crates/latentmesh-agentbbs-bridge's `e2e_loopback` example (release) and
// wraps its JSON output in an evidence-labelled receipt, verified against
// ADR-020's pinned post_message/ReplicateMessage contract shape. When
// AGENTBBS_MCP_BIN is set in the environment (a built `agentbbs` binary,
// same convention as tests/live_agentbbs_mcp.rs), the driver additionally
// roundtrips it live over stdio; this runner never sets that variable
// itself, so a fresh checkout stays hermetic.

import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { repoRoot, runExampleDriver } from "../lib/cargo-driver.mjs";
import { AGENTBBS_BOUNDS, AGENTBBS_RECEIPT_SCHEMA, verifyAgentbbsReceipt } from "../lib/verify.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const root = repoRoot(here);
const artifactsDir = join(here, "..", "artifacts");
const receiptPath = join(artifactsDir, "agentbbs-receipt.json");

mkdirSync(artifactsDir, { recursive: true });

console.log("building and running latentmesh-agentbbs-bridge e2e_loopback (release)...");
const driver = runExampleDriver({
  cwd: root,
  pkg: "latentmesh-agentbbs-bridge",
  example: "e2e_loopback",
});

const receipt = {
  schema: AGENTBBS_RECEIPT_SCHEMA,
  evidence: "simulated",
  evidence_detail: "deterministic simulation — no hardware, no live peers, no credentials",
  scenario: "decode SemanticDelta -> bridge mapping -> post_message/ReplicateMessage contract shape (ADR-020/ADR-022)",
  determinism: "fixed Ed25519 identity seed [9u8;32] (Identity::from_seed, never Identity::generate) and fixed created_at 2026-08-27T12:00:00Z — matches this crate's own tests/golden_payloads.rs fixtures, no RNG",
  driver,
  not_claimed: [
    "no live agentbbs peer or federation network",
    driver?.mcp?.attempted
      ? "the live agentbbs mcp roundtrip exercised only a local subprocess, not a networked deployment"
      : "no live agentbbs mcp binary was present in this environment (AGENTBBS_MCP_BIN unset) — the MCP roundtrip leg did not run",
  ],
};

const verdict = verifyAgentbbsReceipt(receipt);
receipt.acceptance = { bounds: AGENTBBS_BOUNDS, passed: verdict.pass, reasons: verdict.reasons };

writeFileSync(receiptPath, `${JSON.stringify(receipt, null, 2)}\n`);
console.log(JSON.stringify(receipt, null, 2));

if (!verdict.pass) {
  console.error("agentbbs e2e suite FAILED:", verdict.reasons.join("; "));
  process.exit(1);
}
console.log("agentbbs e2e suite passed (deterministic simulation — no live-peer claim).");
