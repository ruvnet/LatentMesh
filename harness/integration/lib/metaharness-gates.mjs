// Optional, removable MetaHarness gates (ADR-022): score, genome, mcp-scan,
// threat-model, invoked as pinned external CLI tooling (the
// `ruflo-metaharness` surface) exactly the way harness/evolve's
// scripts/run.mjs subprocess-invokes `cargo run` — never a build or runtime
// dependency of any workspace crate, never installed via `npx` (that would
// reach the network and break `npm run validate`'s offline contract).
//
// CLI shape (`metaharness <gate> <target> --json`, e.g. `npx metaharness
// score . --json` / `npx metaharness genome .`) is the one the published
// `metaharness` package exposes; this harness never invokes `npx` itself —
// the pinned binary is located via the `METAHARNESS_CLI` environment
// variable, the same "env var names a locally-resolved binary, absence
// skips rather than fails" convention `crates/latentmesh-agentbbs-bridge`'s
// `AGENTBBS_MCP_BIN` / `tests/live_agentbbs_mcp.rs` already established for
// this repository.

import { execFileSync } from "node:child_process";

export const GATE_NAMES = Object.freeze(["score", "genome", "mcp-scan", "threat-model"]);

/**
 * Runs one gate (`<METAHARNESS_CLI> <gateName> <targetPath> --json`) and
 * reports its outcome without throwing — a failed or absent external tool
 * degrades this one gate's entry, never the calling suite.
 */
function runOneGate(cliPath, gateName, targetPath) {
  try {
    const stdout = execFileSync(cliPath, [gateName, targetPath, "--json"], {
      encoding: "utf8",
      stdio: ["ignore", "pipe", "pipe"],
      timeout: 60_000,
    });
    return { ran: true, skipped: false, reason: null, output: stdout.trim().slice(0, 4000) };
  } catch (err) {
    return {
      ran: false,
      skipped: true,
      reason: `external tool invocation failed: ${err.message}`,
      output: null,
    };
  }
}

/**
 * Runs the four ADR-022 gates against `targetPath` (a repo-relative path
 * the gate operates over — e.g. a crate directory). Returns
 * `{ toolPresent, toolPath, gates: { score, genome, "mcp-scan", "threat-model" } }`.
 * Every gate entry has `{ ran, skipped, reason, output }` — `skipped` is
 * `true` and `reason` explains why whenever the tool is absent or errors,
 * never a thrown exception.
 */
export function runMetaharnessGates(targetPath) {
  const cliPath = process.env.METAHARNESS_CLI;
  const gates = {};
  if (!cliPath) {
    for (const name of GATE_NAMES) {
      gates[name] = {
        ran: false,
        skipped: true,
        reason: "METAHARNESS_CLI is not set — external MetaHarness tooling is not present in this environment (ADR-022: optional, removable augmentation, never a build or runtime dependency)",
        output: null,
      };
    }
    return { toolPresent: false, toolPath: null, gates };
  }
  for (const name of GATE_NAMES) {
    gates[name] = runOneGate(cliPath, name, targetPath);
  }
  return { toolPresent: true, toolPath: cliPath, gates };
}
