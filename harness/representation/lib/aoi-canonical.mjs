// ADR-049 section 4 canonical JSON: UTF-8, object keys sorted bytewise, no
// whitespace, integers only (no floats: rates are ppm), printable-ASCII
// strings. Rust (latentmesh-bench) and this file must agree byte for byte; the
// shared vector is crates/latentmesh-bench/testdata/canonical_json_v1.json.

import { createHash } from "node:crypto";

const PRINTABLE_ASCII = /^[\x20-\x7e]*$/;

function canonicalString(value) {
  if (!PRINTABLE_ASCII.test(value)) throw new TypeError("canonical JSON: string is not printable ASCII");
  return JSON.stringify(value); // only escapes " and \ for printable ASCII
}

function canonicalObject(value) {
  const keys = Object.keys(value).sort(); // ASCII keys: UTF-16 order == bytewise
  const parts = keys.map((k) => `${canonicalString(k)}:${canonicalJson(value[k])}`);
  return `{${parts.join(",")}}`;
}

export function canonicalJson(value) {
  if (value === null) return "null";
  switch (typeof value) {
    case "boolean":
      return value ? "true" : "false";
    case "number":
      if (!Number.isSafeInteger(value)) throw new TypeError(`canonical JSON: non-safe-integer number ${value}`);
      return String(value);
    case "string":
      return canonicalString(value);
    case "object":
      return Array.isArray(value) ? `[${value.map(canonicalJson).join(",")}]` : canonicalObject(value);
    default:
      throw new TypeError(`canonical JSON: unsupported type ${typeof value}`);
  }
}

export function sha256Hex(text) {
  return createHash("sha256").update(text, "utf8").digest("hex");
}

/** sha256 of the canonical JSON of {config, axes, seeds}. */
export function frozenDigest(receipt) {
  const { config, axes, seeds } = receipt;
  return sha256Hex(canonicalJson({ config, axes, seeds }));
}

/** sha256 of the canonical JSON of the receipt without `receipt_sha256`. */
export function receiptDigest(receipt) {
  const { receipt_sha256: _omitted, ...rest } = receipt;
  return sha256Hex(canonicalJson(rest));
}
