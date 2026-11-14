// ADR-049 section 5 promotion gate for the AoI / task-relevance benchmark.
// Pure: (receipt, pinned) -> verdict. Integer/BigInt arithmetic only; every
// ratio is cross-multiplied, never divided. Thresholds are frozen constants.
//
// Zero-denominator conventions (documented choices):
//   * stale rate = sum(stale_deliveries)/sum(messages_delivered). If either
//     baseline delivered nothing the rate is undefined, so c3 cannot hold.
//     If W's stale count is 0 there is nothing to reduce, so c3 cannot hold.
//   * utility per byte with zero bytes_on_air is 0 (ADR metric definition).

import { frozenDigest, receiptDigest } from "./aoi-canonical.mjs";

export { canonicalJson, frozenDigest, receiptDigest, sha256Hex } from "./aoi-canonical.mjs";

export const AOI_SCHEMA = "latentmesh.aoi-bench.v1";
export const MIN_SEEDS = 5;
export const PRIMARY_MAX_RELEVANCE_PPM = 500000;

/** Each threshold is `num/den`, evaluated by cross-multiplication. */
export const THRESHOLDS = Object.freeze({
  c1_payload_max: Object.freeze({ num: 80n, den: 100n }), // S.payload <= 0.80 W.payload
  c1_utility_min: Object.freeze({ num: 98n, den: 100n }), // S.utility >= 0.98 W.utility
  c2_utility_min: Object.freeze({ num: 120n, den: 100n }), // S.utility >= 1.20 W.utility
  c3_stale_rate_max: Object.freeze({ num: 75n, den: 100n }), // S.rate <= 0.75 W.rate
  c3_useful_min: Object.freeze({ num: 95n, den: 100n }), // S.useful >= 0.95 W.useful
  noharm_useful_min: Object.freeze({ num: 90n, den: 100n }), // per cell, seed-mean
});

const INTEGRITY_KEYS = [
  "mandatory_violations",
  "replay_inconsistencies",
  "queue_bound_violations",
  "airtime_window_violations",
];
/** Frozen axis names, in the order of the bench's Cartesian product. */
const AXIS_KEYS = [
  "age_of_information_ms", "airtime_budget_bytes", "queue_depth", "loss_rate_ppm",
  "reorder_rate_ppm", "link_outage_ms", "task_relevance_ppm",
];
/** Every per-seed and mean metric the bench writes (ADR-049 section 3). */
const METRIC_KEYS = [
  "messages_generated", "messages_delivered", "bytes_on_air", "payload_bytes_tx",
  "utility_micro", "utility_per_byte_micro", "stale_deliveries", "stale_delivery_rate_ppm",
  "useful_fresh_deliveries", "queue_occupancy_mean_milli", "queue_occupancy_max",
  "queue_overflow_drops", "stale_drops", "unschedulable_drops", "evidence_age_p50_ms",
  "evidence_age_p95_ms", "mandatory_violations", "replay_inconsistencies",
  "queue_bound_violations", "airtime_window_violations", "replay_window_rejections",
];
const HEX64 = /^[0-9a-f]{64}$/;
const BASELINES = ["whole_payload", "priority_field", "relevance_selected"];
const SUM_KEYS = ["payload_bytes_tx", "utility_micro", "bytes_on_air", "stale_deliveries",
  "messages_delivered", "useful_fresh_deliveries"];

const big = (v) => BigInt(v);
/** a <= (t.num/t.den) * b  <=>  a*den <= num*b */
const leScaled = (a, t, b) => big(a) * t.den <= t.num * big(b);
/** a >= (t.num/t.den) * b  <=>  a*den >= num*b */
const geScaled = (a, t, b) => big(a) * t.den >= t.num * big(b);

/** Ratio an/ad >= bn/bd, cross-multiplied; a zero denominator means ratio 0. */
function ratioGE(an, ad, bn, bd) {
  if (big(ad) === 0n) return big(bd) === 0n || big(bn) === 0n;
  if (big(bd) === 0n) return true;
  return big(an) * big(bd) >= big(bn) * big(ad);
}

function checkIdentity(receipt, pinned, reasons) {
  if (!pinned) reasons.push("pinned frozen digest is absent: the benchmark is not frozen");
  if (receipt.schema !== AOI_SCHEMA) reasons.push(`schema is not ${AOI_SCHEMA}`);
  if (receipt.frozen !== true) reasons.push("receipt is not frozen (frozen != true)");
  if (!Array.isArray(receipt.seeds) || receipt.seeds.length < MIN_SEEDS) {
    reasons.push(`fewer than ${MIN_SEEDS} seeds`);
  }
  const git = receipt.git ?? {};
  if (git.dirty === true && !HEX64.test(String(git.dirty_snapshot_sha256 ?? ""))) {
    reasons.push("git.dirty is true without a dirty_snapshot_sha256");
  }
}

function checkDigests(receipt, pinned, reasons) {
  let frozen;
  try {
    frozen = frozenDigest(receipt);
    if (receipt.frozen_digest_sha256 !== frozen) reasons.push("frozen_digest_sha256 does not recompute");
    if (receipt.receipt_sha256 !== receiptDigest(receipt)) reasons.push("receipt_sha256 does not recompute");
  } catch (err) {
    reasons.push(`receipt is not canonicalizable: ${err.message}`);
    return;
  }
  if (pinned) {
    if (pinned.schema !== AOI_SCHEMA) reasons.push("pinned schema mismatch");
    if (pinned.frozen_digest_sha256 !== frozen) reasons.push("frozen digest differs from the pinned digest");
  }
}

const isPrimary = (cell) => cell.axes.task_relevance_ppm <= PRIMARY_MAX_RELEVANCE_PPM;

/** Cell binding (D1): cells must be exactly the Cartesian product of the frozen axes. */
function checkGrid(receipt, reasons) {
  const axes = receipt.axes ?? {};
  const lists = AXIS_KEYS.map((k) => axes[k]);
  if (!lists.every((l) => Array.isArray(l) && l.length > 0)) {
    reasons.push("receipt.axes lacks a non-empty list for each of the seven axes");
    return;
  }
  const expected = lists.reduce((a, l) => a * l.length, 1);
  if (receipt.cells.length !== expected) {
    reasons.push(`cells has ${receipt.cells.length} rows, the frozen axes imply ${expected}`);
    return;
  }
  const allowed = lists.map((l) => new Set(l));
  const ids = new Set();
  const tuples = new Set();
  for (const cell of receipt.cells) {
    ids.add(cell?.cell_id);
    const values = AXIS_KEYS.map((k) => cell?.axes?.[k]);
    values.forEach((v, i) => {
      if (!allowed[i].has(v)) reasons.push(`cell ${cell?.cell_id} ${AXIS_KEYS[i]} ${v} is not a frozen axis value`);
    });
    tuples.add(values.join(","));
  }
  if (ids.size !== receipt.cells.length) reasons.push("cell_id values are not unique");
  if (tuples.size !== expected) reasons.push("cells do not cover the full Cartesian product of the frozen axes");
}

/** Mean cross-check (D2): each mean must equal the integer floor mean of per_seed. */
function checkMeans(cell, name, seeds, reasons) {
  const b = cell.baselines[name];
  for (const [i, row] of b.per_seed.entries()) {
    if (row.seed !== seeds[i]) reasons.push(`cell ${cell.cell_id} ${name} row ${i} is seed ${row.seed}, not ${seeds[i]}`);
    for (const key of METRIC_KEYS) {
      if (!Number.isSafeInteger(row[key]) || row[key] < 0) {
        reasons.push(`cell ${cell.cell_id} ${name} seed ${row.seed} ${key} is not a non-negative integer`);
        return;
      }
    }
  }
  for (const key of METRIC_KEYS) {
    if (!Number.isSafeInteger(b.mean[key])) {
      reasons.push(`cell ${cell.cell_id} ${name} mean.${key} is not an integer`);
      continue;
    }
    const sum = b.per_seed.reduce((acc, row) => acc + BigInt(row[key]), 0n);
    const expect = sum / BigInt(b.per_seed.length);
    if (BigInt(b.mean[key]) !== expect) {
      reasons.push(`cell ${cell.cell_id} ${name} mean.${key} is ${b.mean[key]}, per_seed gives ${expect}`);
    }
  }
}

function checkShape(receipt, reasons) {
  const n = receipt.seeds.length;
  if (!Array.isArray(receipt.cells) || receipt.cells.length === 0) {
    reasons.push("receipt has no cells");
    return;
  }
  for (const cell of receipt.cells) {
    for (const name of BASELINES) {
      const b = cell?.baselines?.[name];
      if (!b || !Array.isArray(b.per_seed) || b.per_seed.length !== n || !b.mean) {
        reasons.push(`cell ${cell?.cell_id} baseline ${name} lacks per_seed for every seed or mean`);
        return;
      }
    }
  }
  checkGrid(receipt, reasons);
  if (reasons.length > 0) return;
  for (const cell of receipt.cells) {
    for (const name of BASELINES) checkMeans(cell, name, receipt.seeds, reasons);
  }
  if (reasons.length > 0) return;
  if (!receipt.cells.some(isPrimary)) reasons.push("no primary cells (task_relevance_ppm <= 500000)");
}

function integrityVerdict(receipt) {
  const totals = {};
  const failures = [];
  for (const key of INTEGRITY_KEYS) {
    totals[key] = receipt.integrity?.[`${key}_total`];
    if (totals[key] !== 0) failures.push(`integrity ${key}_total is ${totals[key]}, must be 0`);
  }
  for (const cell of receipt.cells) {
    for (const name of BASELINES) {
      for (const row of cell.baselines[name].per_seed) {
        for (const key of INTEGRITY_KEYS) {
          if (row[key] !== undefined && row[key] !== 0) {
            failures.push(`cell ${cell.cell_id} ${name} seed ${row.seed} ${key} is ${row[key]}`);
          }
        }
      }
    }
  }
  return { ok: failures.length === 0, totals, failures };
}

/** Per-seed sums over the primary set P for one baseline. */
function aggregate(cells, name, seedIndex) {
  const acc = Object.fromEntries(SUM_KEYS.map((k) => [k, 0n]));
  for (const cell of cells) {
    const row = cell.baselines[name].per_seed[seedIndex];
    for (const k of SUM_KEYS) acc[k] += big(row[k]);
  }
  return acc;
}

function c1(S, W) {
  return leScaled(S.payload_bytes_tx, THRESHOLDS.c1_payload_max, W.payload_bytes_tx)
    && geScaled(S.utility_micro, THRESHOLDS.c1_utility_min, W.utility_micro);
}

function c2(S, W) {
  return geScaled(S.utility_micro, THRESHOLDS.c2_utility_min, W.utility_micro)
    && S.bytes_on_air <= W.bytes_on_air;
}

function c3(S, W) {
  if (S.messages_delivered === 0n || W.messages_delivered === 0n || W.stale_deliveries === 0n) return false;
  const t = THRESHOLDS.c3_stale_rate_max;
  // S.stale/S.deliv <= (num/den) * W.stale/W.deliv, cross-multiplied.
  const rateOk = S.stale_deliveries * W.messages_delivered * t.den
    <= t.num * W.stale_deliveries * S.messages_delivered;
  return rateOk && geScaled(S.useful_fresh_deliveries, THRESHOLDS.c3_useful_min, W.useful_fresh_deliveries);
}

function evaluateCriteria(receipt) {
  const P = receipt.cells.filter(isPrimary);
  const criteria = { c1: true, c2: true, c3: true };
  const guards = { priority_field: true, no_harm: true };
  for (let i = 0; i < receipt.seeds.length; i += 1) {
    const S = aggregate(P, "relevance_selected", i);
    const W = aggregate(P, "whole_payload", i);
    const F = aggregate(P, "priority_field", i);
    criteria.c1 &&= c1(S, W);
    criteria.c2 &&= c2(S, W);
    criteria.c3 &&= c3(S, W);
    guards.priority_field &&= ratioGE(S.utility_micro, S.bytes_on_air, F.utility_micro, F.bytes_on_air);
  }
  for (const cell of receipt.cells) {
    const s = cell.baselines.relevance_selected.mean.useful_fresh_deliveries;
    const w = cell.baselines.whole_payload.mean.useful_fresh_deliveries;
    guards.no_harm &&= geScaled(s, THRESHOLDS.noharm_useful_min, w);
  }
  return { criteria, guards };
}

const NONE = { c1: false, c2: false, c3: false };
const NO_GUARDS = { priority_field: false, no_harm: false };
const invalid = (reasons) => ({ verdict: "INVALID", criteria: NONE, integrity: null, guards: NO_GUARDS, reasons });

export function evaluateAoiPromotion(receipt, pinned) {
  const reasons = [];
  if (!receipt || typeof receipt !== "object") return invalid(["receipt is not an object"]);
  checkIdentity(receipt, pinned ?? null, reasons);
  checkDigests(receipt, pinned ?? null, reasons);
  if (reasons.length === 0) checkShape(receipt, reasons);
  if (reasons.length > 0) return invalid(reasons);

  const integrity = integrityVerdict(receipt);
  const { criteria, guards } = evaluateCriteria(receipt);
  if (!integrity.ok) {
    return { verdict: "REJECT", criteria, integrity, guards, reasons: integrity.failures };
  }
  const why = [];
  if (!(criteria.c1 || criteria.c2 || criteria.c3)) why.push("no utility criterion c1-c3 holds in every seed");
  if (!guards.priority_field) why.push("guard: priority_field has higher utility_per_byte than relevance_selected in some seed");
  if (!guards.no_harm) why.push("guard: relevance_selected useful_fresh_deliveries below 0.90 of whole_payload in some cell");
  return { verdict: why.length === 0 ? "PROMOTE" : "REJECT", criteria, integrity, guards, reasons: why };
}
