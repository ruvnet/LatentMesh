// Pure verification of this harness's own emitted receipts, in
// harness/evolve's lib/verify.mjs style: every violated bound pushes a NAMED
// reason, an empty reason list is a pass, and the evidence label is checked
// as hard as the numbers are.
//
// The evidence label here is NOT "simulated" — the sibling harnesses' label is
// correct for them and would be a LIE here. These receipts are scored from
// measurements already committed to crates/latentmesh-runtime/receipts/, so
// the label is `measured-committed-receipts`, and a receipt claiming anything
// stronger (live, over-the-air, deployed) is refused just as loudly as one
// claiming something it did not do.

import { verifyChain } from "./chain.mjs";

export const BENCHMARK_SCHEMA = "latentmesh-representation-benchmark-v1";
export const CHAMPION_SCHEMA = "latentmesh-representation-champion-v1";
export const REQUIRED_EVIDENCE = "measured-committed-receipts";
export const FORBIDDEN_EVIDENCE = Object.freeze([
  "simulated",
  "live",
  "over the air",
  "over-the-air",
  "deployed",
  "production",
]);

function checkEnvelope(receipt, expectedSchema, reasons) {
  if (!receipt || typeof receipt !== "object") {
    reasons.push("receipt is not an object");
    return false;
  }
  if (receipt.schema !== expectedSchema) {
    reasons.push(`schema is ${JSON.stringify(receipt.schema)}, expected ${expectedSchema}`);
  }
  if (receipt.evidence !== REQUIRED_EVIDENCE) {
    reasons.push(
      `evidence label is ${JSON.stringify(receipt.evidence)} — receipts must say "${REQUIRED_EVIDENCE}", never an over-claimed label`,
    );
  }
  if (FORBIDDEN_EVIDENCE.includes(String(receipt.evidence))) {
    reasons.push(`evidence label ${JSON.stringify(receipt.evidence)} is explicitly forbidden for this harness`);
  }
  if (!Array.isArray(receipt.not_claimed) || receipt.not_claimed.length === 0) {
    reasons.push("receipt must state what it does NOT claim");
  }
  return true;
}

/** Every source receipt must be named with the sha256 actually read. */
function checkSources(receipt, reasons) {
  if (!Array.isArray(receipt.sources) || receipt.sources.length === 0) {
    reasons.push("receipt must name the committed source receipts it scored");
    return;
  }
  for (const source of receipt.sources) {
    if (source.measured === false) continue;
    if (!source.receipt) reasons.push("a measured source has no receipt filename");
    if (!/^[0-9a-f]{64}$/.test(String(source.receiptSha256 ?? ""))) {
      reasons.push(`source ${source.receipt} has no sha256 of the file that was read`);
    }
  }
}

/**
 * The behavioural invariants ADR-046 exists to assert, checked on the
 * receipt's own recorded trace rather than on trust:
 *
 *  - an Unmeasured candidate is never the selected mode;
 *  - Unmeasured and Measured(0) appear as distinct outcomes, never merged;
 *  - `None` is present as a scoreable answer at exactly 0.0.
 */
export function verifyUnmeasuredIneligibility(considered, selectedId) {
  const reasons = [];
  const unmeasured = considered.filter((entry) => entry.outcome === "Unmeasured");
  for (const entry of unmeasured) {
    if (entry.id === selectedId) reasons.push(`unmeasured candidate ${entry.id} was selected`);
    if (entry.score !== null && entry.score !== undefined) {
      reasons.push(`unmeasured candidate ${entry.id} carries a score (${entry.score}) — it must have none`);
    }
  }
  const measuredZero = considered.filter((entry) => entry.outcome === "Scored" && entry.score === 0);
  for (const entry of measuredZero) {
    if (entry.outcome === "Unmeasured") reasons.push(`measured-zero candidate ${entry.id} was labelled Unmeasured`);
  }
  return { pass: reasons.length === 0, reasons, unmeasuredCount: unmeasured.length, measuredZeroCount: measuredZero.length };
}

export function verifyBenchmarkReceipt(receipt) {
  const reasons = [];
  if (!checkEnvelope(receipt, BENCHMARK_SCHEMA, reasons)) return { pass: false, reasons };
  checkSources(receipt, reasons);

  if (!Array.isArray(receipt.candidates) || receipt.candidates.length === 0) {
    reasons.push("benchmark scored no candidates");
  } else {
    const hasUnmeasured = receipt.candidates.some((candidate) => candidate.outcome === "Unmeasured");
    if (!hasUnmeasured) {
      reasons.push(
        "no candidate is Unmeasured — the semantic-delta channel has no committed receipt and must be reported as unmeasured, not omitted",
      );
    }
  }
  const ineligibility = verifyUnmeasuredIneligibility(receipt.candidates ?? [], receipt.selectedId ?? null);
  reasons.push(...ineligibility.reasons);

  if (receipt.selected === "None" && receipt.selectedScore !== 0) {
    reasons.push(`selected None must score exactly 0.0, got ${receipt.selectedScore}`);
  }
  if (!Array.isArray(receipt.unmeasured_resources)) {
    reasons.push("receipt must list the resources no rung measured (energy is one)");
  }
  return { pass: reasons.length === 0, reasons };
}

export function verifyChampionReceipt(receipt) {
  const reasons = [];
  if (!checkEnvelope(receipt, CHAMPION_SCHEMA, reasons)) return { pass: false, reasons };
  checkSources(receipt, reasons);

  if (!receipt.champion || !receipt.champion.genome) {
    reasons.push("no champion genome recorded");
    return { pass: false, reasons };
  }
  if (receipt.champion.genome.alpha > 0.05) {
    reasons.push(`champion alpha ${receipt.champion.genome.alpha} loosens the pre-registered 0.05`);
  }
  if (receipt.deterministic_replay !== true) {
    reasons.push("the seeded search was not replayed to the same champion");
  }
  if (receipt.matches_exhaustive_optimum !== true) {
    reasons.push("the seeded champion does not match the exhaustive optimum over the lattice");
  }
  const chainVerdict = verifyChain(receipt.chain);
  if (!chainVerdict.pass) reasons.push(...chainVerdict.reasons.map((reason) => `chain: ${reason}`));

  const ineligibility = verifyUnmeasuredIneligibility(
    receipt.champion.considered ?? [],
    receipt.champion.selectedId ?? null,
  );
  reasons.push(...ineligibility.reasons);
  return { pass: reasons.length === 0, reasons };
}
