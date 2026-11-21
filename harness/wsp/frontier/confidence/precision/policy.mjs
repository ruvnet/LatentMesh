import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { fitLinear, modelHash } from '../linear-fit.mjs';
import { margin } from '../receiver.mjs';

const [phase, out] = process.argv.slice(2);
if (!out || !['fit', 'predict'].includes(phase)) throw Error('usage: node policy.mjs fit|predict RUN_DIRECTORY');
const file = name => join(out, name + '.json');
const sha = raw => createHash('sha256').update(raw).digest('hex');
const write = (name, data) => writeFileSync(file(name), JSON.stringify(data) + '\n');
const arms = ['float13', 'wsp48', 'softKD', 'local', 'scoreOnly', 'confidenceOnly'];
const source = (arm, available) => available.indexOf(arm === 'scoreOnly' ? 'float13' : arm);

function points(ds, arm, available) {
  const k = source(arm, available);
  return ds.scores.map((v, i) => [arm === 'confidenceOnly' ? 0 : v[k], arm === 'scoreOnly' ? 0 : ds.confidence[i]]);
}

function better(rank, previous) {
  if (!previous) return true;
  for (let i = 0; i < rank.length; i++) if (rank[i] !== previous[i]) return rank[i] > previous[i];
  return false;
}

function select(w, set, labels, strict) {
  const sorted = set.map((p, i) => ({ z: w[0] * p[0] + w[1] * p[1], y: labels[i] })).sort((a, b) => b.z - a.z);
  const positive = labels.filter(Boolean).length;
  let tp = 0, best, rank;
  for (let i = 0; i < sorted.length;) {
    const current = sorted[i].z;
    do { tp += sorted[i++].y; } while (i < sorted.length && sorted[i].z === current);
    if (i < 20 || (strict && 2 * tp < i)) continue;
    const cut = i < sorted.length ? sorted[i].z : -Infinity;
    const precision = tp / i, recall = tp / positive;
    const r = strict ? [recall, precision, cut] : [5 * tp / (4 * i + positive), precision, recall, cut];
    if (!better(r, rank)) continue;
    rank = r;
    best = { coefficients: Number.isFinite(cut) ? [w[0], w[1], -cut] : [0, 0, 1],
      selection: { flags: i, tp, fp: i - tp } };
  }
  if (!best) best = { coefficients: [0, 0, -1], selection: { flags: 0, tp: 0, fp: 0 } };
  assert.equal(set.filter(p => margin(best.coefficients, ...p) > 0).length, best.selection.flags);
  return best;
}

if (phase === 'fit') {
  const raw = readFileSync(file('fit'));
  const data = JSON.parse(raw);
  const { fit, tune } = data.sets;
  if (fit.truth.length > 768 || fit.truth.length === 0) throw Error('wrong fit budget');
  const policies = {};
  for (const arm of arms) {
    const head = await fitLinear(points(fit, arm, data.arms), fit.truth.map(y => y ? 1 : -1));
    const tuning = points(tune, arm, data.arms);
    policies[arm] = {};
    for (const mode of ['f05', 'majority']) {
      const operating = select(head.weights, tuning, tune.truth, mode === 'majority');
      policies[arm][mode] = { ...operating, modelSha256: await modelHash(operating.coefficients) };
    }
  }
  const sets = Object.fromEntries(Object.entries(data.sets).map(([k,v]) => [k, { ids: v.ids, rows: v.truth.length,
    angry: v.truth.filter(Boolean).length, groupIDs: [...new Set(v.groups)].sort((a, b) => a - b) }]));
  write('lock', { schema: 'latentmesh.precision-policy/1', headsSha256: data.headsSha256, fitSha256: sha(raw), sets, policies });
} else {
  const raw = readFileSync(file('lock')), lock = JSON.parse(raw);
  const data = JSON.parse(readFileSync(file('test')));
  assert.equal(data.lockSha256, sha(raw)); assert.equal(data.headsSha256, lock.headsSha256);
  const ds = data.sets.test, predictions = {};
  for (const arm of arms) {
    const p = points(ds, arm, data.arms);
    predictions[arm] = {};
    for (const mode of ['f05','majority']) {
      predictions[arm][mode] = p.map(v => Number(margin(lock.policies[arm][mode].coefficients, ...v) > 0)).join('');
    }
  }
  const initial = JSON.parse(readFileSync(new URL('../lock.json', import.meta.url)));
  const previous = {};
  for (const g of ['transfer', 'adapted']) {
    const coeff = initial.policies[g].float13.separate.coefficients;
    previous[g] = points(ds, 'float13', data.arms).map(v => Number(margin(coeff, ...v) > 0)).join('');
  }
  write('observations', { lockSha256: sha(raw), ids: ds.ids, groups: ds.groups, truth: ds.truth.join(''),
    native: ds.native.join(''), sentiment: ds.sentiment.join(''), predictions, previous });
}
