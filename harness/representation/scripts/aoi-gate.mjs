// ADR-049 promotion gate CLI.
//   node scripts/aoi-gate.mjs <receipt.json> [--pinned lib/aoi-frozen.json] [--out artifacts/aoi-promotion.json]
// Exit 0 PROMOTE, 1 REJECT, 2 INVALID (including a missing pinned file).

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { evaluateAoiPromotion } from "../lib/aoi-gate.mjs";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");

function parseArgs(argv) {
  const opts = {
    receipt: null,
    pinned: resolve(root, "lib/aoi-frozen.json"),
    out: resolve(root, "artifacts/aoi-promotion.json"),
  };
  for (let i = 0; i < argv.length; i += 1) {
    if (argv[i] === "--pinned") opts.pinned = resolve(argv[++i]);
    else if (argv[i] === "--out") opts.out = resolve(argv[++i]);
    else if (!opts.receipt) opts.receipt = resolve(argv[i]);
    else throw new Error(`unexpected argument ${argv[i]}`);
  }
  if (!opts.receipt) throw new Error("usage: aoi-gate.mjs <receipt.json> [--pinned P] [--out O]");
  return opts;
}

function run() {
  const opts = parseArgs(process.argv.slice(2));
  const receipt = JSON.parse(readFileSync(opts.receipt, "utf8"));
  const pinned = existsSync(opts.pinned) ? JSON.parse(readFileSync(opts.pinned, "utf8")) : null;
  const result = evaluateAoiPromotion(receipt, pinned);
  mkdirSync(dirname(opts.out), { recursive: true });
  writeFileSync(opts.out, `${JSON.stringify(result, null, 2)}\n`);
  return result;
}

function main() {
  let result;
  try {
    result = run();
  } catch (err) {
    result = { verdict: "INVALID", criteria: { c1: false, c2: false, c3: false }, integrity: null, guards: null, reasons: [err.message] };
  }
  console.log(JSON.stringify(result, null, 2));
  process.exit({ PROMOTE: 0, REJECT: 1, INVALID: 2 }[result.verdict]);
}

main();
