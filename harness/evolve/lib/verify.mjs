// Pure verification of a latentmesh-evolve receipt (ADR-018) against
// ADR-006's acceptance bounds. Mirrors harness/air's frozen-promotion-rule
// style: every failed bound pushes a named reason; an empty reason list is a
// pass. Deliberately dependency-free.

export const RECEIPT_SCHEMA = "latentmesh-evolve-receipt-v1";
export const REQUIRED_EVIDENCE = "simulated";

export const ACCEPTANCE_BOUNDS = Object.freeze({
  computeReduction: 0.3,
  taskSuccessTolerance: 1e-9,
});

/**
 * Verify one parsed receipt object. Returns { pass, reasons } where reasons
 * name every violated bound (never just a boolean, so CI logs say why).
 */
export function verifyReceipt(receipt) {
  const reasons = [];
  if (!receipt || typeof receipt !== "object") {
    return { pass: false, reasons: ["receipt is not an object"] };
  }
  if (receipt.schema !== RECEIPT_SCHEMA) {
    reasons.push(`schema is ${JSON.stringify(receipt.schema)}, expected ${RECEIPT_SCHEMA}`);
  }
  if (receipt.evidence !== REQUIRED_EVIDENCE) {
    reasons.push(
      `evidence label is ${JSON.stringify(receipt.evidence)} — receipts must say "${REQUIRED_EVIDENCE}", never an over-claimed label`,
    );
  }
  const a = receipt.acceptance;
  if (!a || typeof a !== "object") {
    reasons.push("acceptance report is missing");
    return { pass: false, reasons };
  }
  if (!(a.compute_reduction >= ACCEPTANCE_BOUNDS.computeReduction)) {
    reasons.push(
      `compute reduction ${a.compute_reduction} is below the ${ACCEPTANCE_BOUNDS.computeReduction} bound`,
    );
  }
  if (
    !(
      a.task_success_after >=
      a.task_success_before - ACCEPTANCE_BOUNDS.taskSuccessTolerance
    )
  ) {
    reasons.push(
      `task success regressed: ${a.task_success_before} -> ${a.task_success_after}`,
    );
  }
  if (a.unverified_nonmandatory_edges !== 0) {
    reasons.push(
      `${a.unverified_nonmandatory_edges} surviving non-mandatory edges failed causal verification`,
    );
  }
  if (a.passed !== true && reasons.length === 0) {
    reasons.push("receipt's own acceptance.passed is false despite bounds holding");
  }
  if (!Array.isArray(receipt.not_claimed) || receipt.not_claimed.length === 0) {
    reasons.push("receipt must state what it does NOT claim");
  }
  if (!(receipt.verification_evaluations > 0)) {
    reasons.push("no verification evaluations recorded — causal fitness cannot have run");
  }
  return { pass: reasons.length === 0, reasons };
}
