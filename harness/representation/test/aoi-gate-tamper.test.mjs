// Adversarial gate tests (issue #22) run against the REAL committed receipt
// crates/latentmesh-bench/receipts/aoi-bench-v1-47bff02.json, not synthetic ones.
// Tests marked `skip` assert the DESIRED behaviour for a confirmed gate gap.
import test from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { AOI_SCHEMA, evaluateAoiPromotion, frozenDigest, receiptDigest } from "../lib/aoi-gate.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const RECEIPT = resolve(here, "../../../crates/latentmesh-bench/receipts/aoi-bench-v1-47bff02.json");
const CLI = resolve(here, "../scripts/aoi-gate.mjs");
const RAW = readFileSync(RECEIPT, "utf8");
const fresh = () => JSON.parse(RAW);
const pinOf = (r) => ({ schema: AOI_SCHEMA, frozen_digest_sha256: r.frozen_digest_sha256 });
const PIN = pinOf(fresh());
const reseal = (r) => {
  r.frozen_digest_sha256 = frozenDigest(r);
  r.receipt_sha256 = receiptDigest(r);
  return r;
};
/** Re-seal only receipt_sha256; the pinned frozen digest stays honest. */
const resealBody = (r) => {
  r.receipt_sha256 = receiptDigest(r);
  return r;
};
const isP = (c) => c.axes.task_relevance_ppm <= 500000;
const S = (c) => c.baselines.relevance_selected;
const W = (c) => c.baselines.whole_payload;

test("real receipt: both digests recompute in JS exactly as Rust wrote them", () => {
  const r = fresh();
  assert.equal(frozenDigest(r), r.frozen_digest_sha256);
  assert.equal(receiptDigest(r), r.receipt_sha256);
});

test("real receipt + its own pinned digest PROMOTEs with all guards green", () => {
  const out = evaluateAoiPromotion(fresh(), PIN);
  assert.equal(out.verdict, "PROMOTE", JSON.stringify(out.reasons));
  assert.deepEqual(out.guards, { priority_field: true, no_harm: true });
});

test("flipping ONE per-seed metric without resealing is INVALID (receipt_sha256)", () => {
  const r = fresh();
  r.cells[7].baselines.relevance_selected.per_seed[2].utility_micro += 1;
  const out = evaluateAoiPromotion(r, PIN);
  assert.equal(out.verdict, "INVALID");
  assert.match(out.reasons.join(), /receipt_sha256 does not recompute/);
});

test("changing the seed list without resealing is INVALID", () => {
  const r = fresh();
  r.seeds[0] += 1;
  assert.equal(evaluateAoiPromotion(r, PIN).verdict, "INVALID");
});

test("resealing a config change is INVALID against the pinned frozen digest", () => {
  const r = fresh();
  r.config.stale_age_ms += 1;
  reseal(r);
  const out = evaluateAoiPromotion(r, PIN);
  assert.equal(out.verdict, "INVALID");
  assert.match(out.reasons.join(), /pinned/);
});

test("resealing a changed axes list is INVALID against the pinned frozen digest", () => {
  const r = fresh();
  r.axes.airtime_budget_bytes = [1200];
  reseal(r);
  assert.equal(evaluateAoiPromotion(r, PIN).verdict, "INVALID");
});

test("a consistent tamper (utility cut in ONE seed, body resealed) flips PROMOTE to REJECT by the math", () => {
  const r = fresh();
  const seed = 3;
  for (const c of r.cells.filter(isP)) {
    const s = S(c).per_seed[seed];
    const w = W(c).per_seed[seed];
    s.utility_micro = Math.floor(w.utility_micro / 2); // c1 and c2 fail
    s.payload_bytes_tx = w.payload_bytes_tx; // c1 payload fails
    s.bytes_on_air = w.bytes_on_air + 1; // c2 airtime fails
    s.stale_deliveries = w.stale_deliveries; // c3 rate fails
    s.messages_delivered = w.messages_delivered;
  }
  resealBody(r);
  const bad = evaluateAoiPromotion(r, PIN);
  assert.equal(bad.verdict, "REJECT");
  assert.deepEqual(bad.criteria, { c1: false, c2: false, c3: false });
});

test("a single-seed utility dip that keeps c1 alive in that seed does not flip the verdict", () => {
  const r = fresh();
  for (const c of r.cells.filter(isP)) {
    const s = S(c).per_seed[0];
    s.utility_micro = Math.ceil((W(c).per_seed[0].utility_micro * 99) / 100); // still >= 0.98 W
  }
  resealBody(r);
  assert.equal(evaluateAoiPromotion(r, PIN).verdict, "PROMOTE");
});

test("mutation: a utility boost obtained by violating an invariant is REJECTed", () => {
  for (const key of [
    "mandatory_violations",
    "replay_inconsistencies",
    "queue_bound_violations",
    "airtime_window_violations",
  ]) {
    const r = fresh();
    for (const c of r.cells.filter(isP)) S(c).per_seed.forEach((s) => (s.utility_micro *= 3));
    S(r.cells[0]).per_seed[0][key] = 1;
    r.integrity[`${key}_total`] = 1;
    resealBody(r);
    const out = evaluateAoiPromotion(r, PIN);
    assert.equal(out.verdict, "REJECT", key);
    assert.equal(out.integrity.ok, false);
  }
});

test("per-seed violation hidden behind totals of 0 still REJECTs; total>0 with clean rows REJECTs", () => {
  const a = fresh();
  S(a.cells[5]).per_seed[4].mandatory_violations = 2;
  assert.equal(evaluateAoiPromotion(resealBody(a), PIN).verdict, "REJECT");
  const b = fresh();
  b.integrity.replay_inconsistencies_total = 1;
  assert.equal(evaluateAoiPromotion(resealBody(b), PIN).verdict, "REJECT");
});

test("a violation on a baseline other than S (priority_field) also blocks promotion", () => {
  const r = fresh();
  r.cells[0].baselines.priority_field.per_seed[0].queue_bound_violations = 1;
  assert.equal(evaluateAoiPromotion(resealBody(r), PIN).verdict, "REJECT");
});

test("a missing or non-integer integrity total is not treated as zero", () => {
  const a = fresh();
  delete a.integrity.mandatory_violations_total;
  assert.equal(evaluateAoiPromotion(resealBody(a), PIN).verdict, "REJECT");
  const b = fresh();
  b.integrity.mandatory_violations_total = 0.5; // float: not canonicalizable, so never sealed
  assert.equal(evaluateAoiPromotion(b, PIN).verdict, "INVALID");
});

test("identity failures on the real receipt are INVALID, never REJECT/PROMOTE", () => {
  const cases = {
    unfrozen: (r) => (r.frozen = false),
    fourSeeds: (r) => {
      r.seeds.pop();
      for (const c of r.cells) for (const b of Object.values(c.baselines)) b.per_seed.pop();
    },
    dirtyNoSnapshot: (r) => (r.git = { commit: "x", dirty: true, dirty_snapshot_sha256: null }),
    wrongSchema: (r) => (r.schema = "latentmesh.aoi-bench.v0"),
    noCells: (r) => (r.cells = []),
  };
  for (const [name, mutate] of Object.entries(cases)) {
    const r = fresh();
    mutate(r);
    reseal(r);
    assert.equal(evaluateAoiPromotion(r, pinOf(fresh())).verdict, "INVALID", name);
  }
  assert.equal(evaluateAoiPromotion(fresh(), null).verdict, "INVALID");
  assert.equal(evaluateAoiPromotion(fresh(), undefined).verdict, "INVALID");
});

test("a per_seed row missing for one seed in one cell is INVALID", () => {
  const r = fresh();
  r.cells[100].baselines.whole_payload.per_seed.pop();
  assert.equal(evaluateAoiPromotion(resealBody(r), PIN).verdict, "INVALID");
});

test("c2 needs a strict 1.20 ratio: exactly 119/100 of W in one seed fails c2", () => {
  const r = fresh();
  // Force c1 and c3 off so c2 alone decides, then sit just below/above the line.
  for (const c of r.cells.filter(isP)) {
    for (const [i, s] of S(c).per_seed.entries()) {
      const w = W(c).per_seed[i];
      s.payload_bytes_tx = w.payload_bytes_tx;
      s.stale_deliveries = w.stale_deliveries;
      s.useful_fresh_deliveries = 0;
      s.bytes_on_air = w.bytes_on_air;
      s.utility_micro = w.utility_micro * 6; // plenty
    }
  }
  assert.equal(evaluateAoiPromotion(resealBody(structuredClone(r)), PIN).criteria.c2, true);
  // Exactly 1.20 holds, 1.20 minus one micro-unit in a single seed does not.
  for (const c of r.cells.filter(isP)) {
    const s = S(c).per_seed[1];
    s.utility_micro = (W(c).per_seed[1].utility_micro * 12) / 10;
  }
  const sumS = r.cells.filter(isP).reduce((a, c) => a + S(c).per_seed[1].utility_micro, 0);
  const sumW = r.cells.filter(isP).reduce((a, c) => a + W(c).per_seed[1].utility_micro, 0);
  const lineHolds = sumS * 100 >= 120 * sumW;
  assert.equal(evaluateAoiPromotion(resealBody(structuredClone(r)), PIN).criteria.c2, lineHolds);
  S(r.cells.find(isP)).per_seed[1].utility_micro -= Math.ceil(sumS - (sumW * 120) / 100) + 1;
  assert.equal(evaluateAoiPromotion(resealBody(r), PIN).criteria.c2, false);
});

// ---- Confirmed gate gaps: desired behaviour asserted, skipped until fixed ----

test("GAP: dropping cells (resealed) must be INVALID; the gate does not check cells == product(axes)", { skip: "defect: gate never compares cells.length/cell_id to the frozen axes" }, () => {
  const r = fresh();
  r.cells = r.cells.filter((c) => isP(c) || c.cell_id % 2 === 0);
  assert.equal(evaluateAoiPromotion(resealBody(r), PIN).verdict, "INVALID");
});

test("GAP: per_seed/mean disagreement must be INVALID; no-harm guard reads only `mean`", { skip: "defect: mean is never cross-checked against per_seed" }, () => {
  const r = fresh();
  // Per-seed S useful deliveries collapse to 0 in a cell, but the mean still looks healthy.
  for (const s of S(r.cells[0]).per_seed) s.useful_fresh_deliveries = 0;
  assert.notEqual(evaluateAoiPromotion(resealBody(r), PIN).verdict, "PROMOTE");
});

test("GAP: a cell whose axes are not a member of the frozen axes must be INVALID", { skip: "defect: cell.axes values are not validated against receipt.axes" }, () => {
  const r = fresh();
  r.cells[3].axes.airtime_budget_bytes = 99999;
  assert.equal(evaluateAoiPromotion(resealBody(r), PIN).verdict, "INVALID");
});

test("GAP: duplicate cell_id rows must be INVALID", { skip: "defect: cell_id uniqueness is not checked" }, () => {
  const r = fresh();
  r.cells.push(structuredClone(r.cells[0]));
  assert.equal(evaluateAoiPromotion(resealBody(r), PIN).verdict, "INVALID");
});

// ---- CLI ----

function cli(receipt, pinned) {
  const dir = mkdtempSync(join(tmpdir(), "aoi-gate-"));
  const rp = join(dir, "r.json");
  writeFileSync(rp, typeof receipt === "string" ? receipt : JSON.stringify(receipt));
  const args = [CLI, rp, "--out", join(dir, "out.json")];
  if (pinned) {
    const pp = join(dir, "pinned.json");
    writeFileSync(pp, JSON.stringify(pinned));
    args.push("--pinned", pp);
  } else {
    args.push("--pinned", join(dir, "absent.json"));
  }
  return spawnSync(process.execPath, args, { encoding: "utf8" });
}

test("CLI exit codes: PROMOTE 0, REJECT 1, INVALID 2 (missing pin, tamper, bad JSON)", () => {
  assert.equal(cli(RAW, PIN).status, 0);
  const rej = fresh();
  rej.integrity.mandatory_violations_total = 1;
  assert.equal(cli(resealBody(rej), PIN).status, 1);
  assert.equal(cli(RAW, null).status, 2, "unpinned freeze");
  const t = fresh();
  t.cells[0].baselines.whole_payload.per_seed[0].utility_micro += 1;
  assert.equal(cli(t, PIN).status, 2);
  assert.equal(cli("{not json", PIN).status, 2);
});
