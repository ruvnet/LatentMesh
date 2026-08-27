// ADR-022 e2e suite 3: cognitum signed-request contract tests. Runs
// crates/latentmesh-cognitum-client's `e2e_loopback` example (release,
// `http` feature) and wraps its JSON output in an evidence-labelled
// receipt, verified against ADR-021's canonical-string/signing contract and
// the published request/response schema. The example's mock HTTP server is
// a same-process std::net::TcpListener loopback responder — no network call
// ever leaves this host, and no credential for a real cognitum device is
// used (fixture keypair + FixedClock only).

import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { repoRoot, runExampleDriver } from "../lib/cargo-driver.mjs";
import { COGNITUM_BOUNDS, COGNITUM_RECEIPT_SCHEMA, verifyCognitumReceipt } from "../lib/verify.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const root = repoRoot(here);
const artifactsDir = join(here, "..", "artifacts");
const receiptPath = join(artifactsDir, "cognitum-receipt.json");

mkdirSync(artifactsDir, { recursive: true });

console.log("building and running latentmesh-cognitum-client e2e_loopback (release, --features http)...");
const driver = runExampleDriver({
  cwd: root,
  pkg: "latentmesh-cognitum-client",
  example: "e2e_loopback",
  features: "http",
});

const receipt = {
  schema: COGNITUM_RECEIPT_SCHEMA,
  evidence: "simulated",
  evidence_detail: "deterministic simulation — no hardware, no live peers, no credentials",
  scenario: "canonical string -> Ed25519 signature -> mock-server verification, deterministic injected clock (ADR-021/ADR-022)",
  determinism: "fixed Ed25519 seed [7u8;32] and a FixedClock (never SystemClock) pinned at unix 1777562280 — matches this crate's own src/signing.rs golden tests, no RNG, no wall-clock read",
  driver,
  not_claimed: [
    "no network call was made to api.cognitum.one or any real cognitum host",
    "no provisioned device credential exists or was used — the keypair is an in-repo test fixture",
    "the local V0 appliance surface (/api/v1/v0/*) is out of scope for this crate and this suite",
  ],
};

const verdict = verifyCognitumReceipt(receipt);
receipt.acceptance = { bounds: COGNITUM_BOUNDS, passed: verdict.pass, reasons: verdict.reasons };

writeFileSync(receiptPath, `${JSON.stringify(receipt, null, 2)}\n`);
console.log(JSON.stringify(receipt, null, 2));

if (!verdict.pass) {
  console.error("cognitum e2e suite FAILED:", verdict.reasons.join("; "));
  process.exit(1);
}
console.log("cognitum e2e suite passed (deterministic simulation — no credential claim).");
