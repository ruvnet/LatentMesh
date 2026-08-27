// ADR-022 e2e suite 1: Air-over-Meshtastic-framing. Runs
// crates/latentmesh-meshtastic's `e2e_loopback` example (release, no
// hardware) and wraps its JSON output in an evidence-labelled receipt,
// verified against the bounds ADR-019/ADR-022 name (single-packet
// unsigned/signed deltas, a >217-usable-byte multi-fragment message).

import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { repoRoot, runExampleDriver } from "../lib/cargo-driver.mjs";
import { MESHTASTIC_BOUNDS, MESHTASTIC_RECEIPT_SCHEMA, verifyMeshtasticReceipt } from "../lib/verify.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const root = repoRoot(here);
const artifactsDir = join(here, "..", "artifacts");
const receiptPath = join(artifactsDir, "meshtastic-receipt.json");

mkdirSync(artifactsDir, { recursive: true });

console.log("building and running latentmesh-meshtastic e2e_loopback (release)...");
const driver = runExampleDriver({ cwd: root, pkg: "latentmesh-meshtastic", example: "e2e_loopback" });

const receipt = {
  schema: MESHTASTIC_RECEIPT_SCHEMA,
  evidence: "simulated",
  evidence_detail: "deterministic simulation — no hardware, no live peers, no credentials",
  scenario: "Air frame -> device-API framing -> loopback channel -> reassembly (ADR-019/ADR-022)",
  determinism: "fixed fixtures only, no RNG: a single one-field CriticalState delta (field 1, Bool false->true) with fixed stream_id/sequence/priority/state_tag, and a fixed 300-byte multi-fragment payload built from a deterministic byte sequence (0..300 mod 256)",
  driver,
  not_claimed: [
    "no Meshtastic hardware present on this host",
    "no real serial/TCP connection to a Meshtastic node",
    "no real LoRa RF transmission, multi-hop relay, or ACK behavior",
  ],
};

const verdict = verifyMeshtasticReceipt(receipt);
receipt.acceptance = { bounds: MESHTASTIC_BOUNDS, passed: verdict.pass, reasons: verdict.reasons };

writeFileSync(receiptPath, `${JSON.stringify(receipt, null, 2)}\n`);
console.log(JSON.stringify(receipt, null, 2));

if (!verdict.pass) {
  console.error("meshtastic e2e suite FAILED:", verdict.reasons.join("; "));
  process.exit(1);
}
console.log("meshtastic e2e suite passed (deterministic simulation — no hardware claim).");
