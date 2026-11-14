import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  AOI_SCHEMA,
  canonicalJson,
  evaluateAoiPromotion,
  frozenDigest,
  receiptDigest,
  sha256Hex,
} from "../lib/aoi-gate.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const VECTOR = resolve(here, "../../../crates/latentmesh-bench/testdata/canonical_json_v1.json");
const SEEDS = [11, 23, 37, 53, 71];

// Baseline numbers: S == W, so no criterion holds until a test overrides S.
const BASE = {
  payload_bytes_tx: 1000, utility_micro: 1000, bytes_on_air: 1000, stale_deliveries: 100,
  messages_delivered: 400, useful_fresh_deliveries: 300,
  mandatory_violations: 0, replay_inconsistencies: 0, queue_bound_violations: 0, airtime_window_violations: 0,
};

function baseline(overrides = {}) {
  const row = { ...BASE, ...overrides };
  return { per_seed: SEEDS.map((seed) => ({ seed, ...row })), mean: { ...row } };
}

function cell(id, rho, s = {}) {
  return {
    cell_id: id,
    axes: { task_relevance_ppm: rho },
    baselines: { whole_payload: baseline(), priority_field: baseline(), relevance_selected: baseline(s) },
  };
}

/** Recompute both digests so a mutated receipt is internally consistent. */
function seal(receipt) {
  receipt.frozen_digest_sha256 = frozenDigest(receipt);
  receipt.receipt_sha256 = receiptDigest(receipt);
  return receipt;
}

function makeReceipt(sOverrides = {}, mutate = () => {}) {
  const receipt = {
    schema: AOI_SCHEMA,
    schema_version: 1,
    evidence_label: "simulated-deterministic",
    frozen: true,
    git: { commit: "abc1234", dirty: false, dirty_snapshot_sha256: null },
    seeds: [...SEEDS],
    axes: { task_relevance_ppm: [250000, 500000, 1000000] },
    config: { msg_count: 512, primary_cells: "task_relevance_ppm<=500000" },
    cells: [cell(0, 250000, sOverrides), cell(1, 500000, sOverrides), cell(2, 1000000)],
    integrity: {
      mandatory_violations_total: 0, replay_inconsistencies_total: 0,
      queue_bound_violations_total: 0, airtime_window_violations_total: 0,
    },
  };
  mutate(receipt);
  return seal(receipt);
}

const pin = (receipt) => ({ schema: AOI_SCHEMA, frozen_digest_sha256: receipt.frozen_digest_sha256 });
const run = (receipt, pinned = pin(receipt)) => evaluateAoiPromotion(receipt, pinned);

const C1 = { payload_bytes_tx: 700, utility_micro: 990, bytes_on_air: 700 };
const C2 = { utility_micro: 1300 };
const C3 = { stale_deliveries: 50 };

test("canonicalizer reproduces the cross-language vector", () => {
  const v = JSON.parse(readFileSync(VECTOR, "utf8"));
  assert.equal(canonicalJson(v.input), v.canonical);
  assert.equal(sha256Hex(v.canonical), v.sha256);
});

test("canonicalizer rejects floats, unsafe integers, non-ASCII and undefined", () => {
  assert.throws(() => canonicalJson({ a: 1.5 }), /non-safe-integer/);
  assert.throws(() => canonicalJson({ a: 2 ** 60 }), /non-safe-integer/);
  assert.throws(() => canonicalJson({ a: "café" }), /ASCII/);
  assert.throws(() => canonicalJson({ a: undefined }), /unsupported/);
});

test("a float anywhere in a receipt makes it INVALID", () => {
  const r = makeReceipt(C2);
  r.config.bad = 0.5;
  const out = run(r);
  assert.equal(out.verdict, "INVALID");
  assert.match(out.reasons.join(), /not canonicalizable/);
});

test("PROMOTE via c1 alone", () => {
  const out = run(makeReceipt(C1));
  assert.deepEqual(out.criteria, { c1: true, c2: false, c3: false });
  assert.equal(out.verdict, "PROMOTE");
  assert.deepEqual(out.reasons, []);
});

test("PROMOTE via c2 alone", () => {
  const out = run(makeReceipt(C2));
  assert.deepEqual(out.criteria, { c1: false, c2: true, c3: false });
  assert.equal(out.verdict, "PROMOTE");
});

test("PROMOTE via c3 alone", () => {
  const out = run(makeReceipt(C3));
  assert.deepEqual(out.criteria, { c1: false, c2: false, c3: true });
  assert.equal(out.verdict, "PROMOTE");
});

test("REJECT when no criterion holds", () => {
  const out = run(makeReceipt());
  assert.equal(out.verdict, "REJECT");
  assert.deepEqual(out.criteria, { c1: false, c2: false, c3: false });
});

test("c3 cannot hold when whole_payload has zero stale deliveries", () => {
  const r = makeReceipt({}, (x) => {
    for (const c of x.cells) for (const row of c.baselines.whole_payload.per_seed) row.stale_deliveries = 0;
  });
  assert.equal(run(r).criteria.c3, false);
});

for (const key of ["mandatory_violations", "replay_inconsistencies", "queue_bound_violations", "airtime_window_violations"]) {
  test(`nonzero ${key}_total forces REJECT even with a winning selector`, () => {
    const r = makeReceipt(C2, (x) => { x.integrity[`${key}_total`] = 1; });
    const out = run(r);
    assert.equal(out.verdict, "REJECT");
    assert.equal(out.integrity.ok, false);
  });
}

test("a per-seed integrity counter forces REJECT", () => {
  const r = makeReceipt(C2, (x) => { x.cells[0].baselines.relevance_selected.per_seed[2].mandatory_violations = 3; });
  assert.equal(run(r).verdict, "REJECT");
});

test("a criterion holding in only 4 of 5 seeds is not satisfied", () => {
  const r = makeReceipt(C2, (x) => {
    x.cells[0].baselines.relevance_selected.per_seed[4].utility_micro = 1000;
    x.cells[1].baselines.relevance_selected.per_seed[4].utility_micro = 1000;
  });
  const out = run(r);
  assert.equal(out.criteria.c2, false);
  assert.equal(out.verdict, "REJECT");
});

test("a 4-seed receipt is INVALID", () => {
  const r = makeReceipt(C2, (x) => {
    x.seeds = SEEDS.slice(0, 4);
    for (const c of x.cells) for (const b of Object.values(c.baselines)) b.per_seed.pop();
  });
  const out = run(r);
  assert.equal(out.verdict, "INVALID");
  assert.match(out.reasons.join(), /seeds/);
});

test("frozen:false is INVALID", () => {
  assert.equal(run(makeReceipt(C2, (x) => { x.frozen = false; })).verdict, "INVALID");
});

test("tampered receipt_sha256 is INVALID", () => {
  const r = makeReceipt(C2);
  r.receipt_sha256 = "f".repeat(64);
  assert.equal(run(r).verdict, "INVALID");
});

test("tampering with a cell after sealing is INVALID", () => {
  const r = makeReceipt(C2);
  r.cells[0].baselines.relevance_selected.per_seed[0].utility_micro = 9999;
  assert.equal(run(r).verdict, "INVALID");
});

test("tampered frozen digest is INVALID", () => {
  const r = makeReceipt(C2);
  r.frozen_digest_sha256 = "e".repeat(64);
  r.receipt_sha256 = receiptDigest(r);
  assert.equal(run(r, pin(r)).verdict, "INVALID");
});

test("frozen digest mismatch against the pinned digest is INVALID", () => {
  const r = makeReceipt(C2);
  const out = run(r, { schema: AOI_SCHEMA, frozen_digest_sha256: "a".repeat(64) });
  assert.equal(out.verdict, "INVALID");
  assert.match(out.reasons.join(), /pinned/);
});

test("pinned absent is INVALID", () => {
  const r = makeReceipt(C2);
  assert.equal(evaluateAoiPromotion(r, null).verdict, "INVALID");
  assert.equal(evaluateAoiPromotion(r, undefined).verdict, "INVALID");
});

test("dirty without a snapshot hash is INVALID; with one it is accepted", () => {
  const dirty = makeReceipt(C2, (x) => { x.git = { commit: "abc", dirty: true, dirty_snapshot_sha256: null }; });
  assert.equal(run(dirty).verdict, "INVALID");
  const snap = makeReceipt(C2, (x) => { x.git = { commit: "abc", dirty: true, dirty_snapshot_sha256: "b".repeat(64) }; });
  assert.equal(run(snap).verdict, "PROMOTE");
});

test("priority_field guard blocks promotion when it has higher utility per byte", () => {
  const r = makeReceipt(C2, (x) => {
    for (const c of x.cells) for (const row of c.baselines.priority_field.per_seed) row.utility_micro = 2000;
  });
  const out = run(r);
  assert.equal(out.criteria.c2, true);
  assert.equal(out.guards.priority_field, false);
  assert.equal(out.verdict, "REJECT");
});

test("priority_field guard fails if any single seed is worse", () => {
  const r = makeReceipt(C2, (x) => { x.cells[0].baselines.priority_field.per_seed[3].utility_micro = 5000; });
  assert.equal(run(r).guards.priority_field, false);
});

test("no-harm guard blocks promotion when a cell loses useful deliveries", () => {
  const r = makeReceipt(C2, (x) => { x.cells[2].baselines.relevance_selected.mean.useful_fresh_deliveries = 269; });
  const out = run(r);
  assert.equal(out.guards.no_harm, false);
  assert.equal(out.verdict, "REJECT");
});

test("no-harm guard allows exactly 0.90 of whole_payload", () => {
  const r = makeReceipt(C2, (x) => { x.cells[2].baselines.relevance_selected.mean.useful_fresh_deliveries = 270; });
  assert.equal(run(r).guards.no_harm, true);
});

test("non-primary cells do not feed the utility criteria", () => {
  const r = makeReceipt({}, (x) => {
    for (const row of x.cells[2].baselines.relevance_selected.per_seed) row.utility_micro = 100000;
  });
  assert.equal(run(r).criteria.c2, false);
});
