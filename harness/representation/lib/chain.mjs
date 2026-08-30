// Hash-chained receipts — a direct port of upstream MetaHarness's ADR-047
// audit layer (`agent-harness-generator/packages/harness/src/receipts.ts`):
// canonical JSON with recursively sorted keys, SHA-256, a 64-zero genesis,
// `prevHash`/`thisHash` per entry, and a `verify()` that replays every link
// and names the first broken index. The names are upstream's on purpose — this
// is the existing mechanism applied to a new payload, not a second design.
//
// Why not `@metaharness/flywheel`'s signer (which harness/air uses): flywheel
// signs a simulated proposer/evaluator loop's replay bundle. This harness has
// no simulator to sign — its evidence is receipts already committed to this
// repository — so a signature would assert a provenance it cannot back. The
// chain gives tamper-evidence, which is the property actually needed.

import { createHash } from "node:crypto";

const GENESIS = "0".repeat(64);
export { GENESIS };

/** Deterministic JSON: object keys sorted recursively so hashes are stable. */
export function canonical(value) {
  return JSON.stringify(sortKeys(value));
}

function sortKeys(value) {
  if (Array.isArray(value)) return value.map(sortKeys);
  if (value && typeof value === "object") {
    const out = {};
    for (const key of Object.keys(value).sort()) out[key] = sortKeys(value[key]);
    return out;
  }
  return value;
}

/** SHA-256 hex of a value's canonical JSON form. */
export function hash(value) {
  return createHash("sha256").update(canonical(value)).digest("hex");
}

/**
 * Append-only, hash-chained log. Each entry's `thisHash` covers its body
 * INCLUDING `prevHash`, so reordering or mutating any entry breaks replay.
 */
export class ReceiptLog {
  #receipts = [];

  /** Append one step, chaining onto the tail. Returns the stored entry. */
  append(step, payload) {
    const prevHash = this.#receipts.length ? this.#receipts[this.#receipts.length - 1].thisHash : GENESIS;
    const body = { index: this.#receipts.length, step, payloadHash: hash(payload), payload, prevHash };
    const receipt = { ...body, thisHash: hash(body) };
    this.#receipts.push(receipt);
    return receipt;
  }

  entries() {
    return this.#receipts;
  }

  head() {
    return this.#receipts.length ? this.#receipts[this.#receipts.length - 1].thisHash : GENESIS;
  }

  toJSON() {
    return { genesis: GENESIS, head: this.head(), entries: this.#receipts };
  }
}

/**
 * Replay a serialized chain. Returns `{ pass, brokenAt, reasons }` — reasons
 * name what failed, never a bare boolean, matching the sibling harnesses'
 * "every failed bound pushes a named reason" convention.
 */
export function verifyChain(chain) {
  const reasons = [];
  if (!chain || !Array.isArray(chain.entries)) {
    return { pass: false, brokenAt: 0, reasons: ["chain has no entries array"] };
  }
  let prevHash = chain.genesis ?? GENESIS;
  let brokenAt = null;
  for (let index = 0; index < chain.entries.length; index += 1) {
    const receipt = chain.entries[index];
    const { thisHash, ...body } = receipt;
    if (receipt.prevHash !== prevHash) {
      reasons.push(`entry ${index} prevHash does not chain`);
      if (brokenAt === null) brokenAt = index;
    }
    if (body.index !== index) {
      reasons.push(`entry ${index} declares index ${body.index}`);
      if (brokenAt === null) brokenAt = index;
    }
    if (hash(body) !== thisHash) {
      reasons.push(`entry ${index} thisHash does not match body`);
      if (brokenAt === null) brokenAt = index;
    }
    prevHash = thisHash;
  }
  if (chain.head !== prevHash) reasons.push("chain head does not match the last entry's hash");
  return { pass: reasons.length === 0, brokenAt, reasons };
}
