import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { wsp } from '../../../dependency.mjs';

const directory = process.argv[2];
if (!directory) throw Error('usage: node fit.mjs NEW_RUN_DIRECTORY');
const data = JSON.parse(readFileSync(join(directory, 'features.json')));
const { fit, tune } = data;
if (fit.rows !== fit.truth.length || tune.rows !== tune.truth.length || fit.rows > 768) throw Error('invalid label budget');

function features(split, name) {
  return split.pc.map((row, i) => name === 'pca' ? row : [...row, split[name][i]]);
}

function preferred(rank, best) {
  if (!best) return true;
  for (let i = 0; i < rank.length; i++) if (rank[i] !== best[i]) return rank[i] > best[i];
  return false;
}

function select(scores, truth) {
  const order = scores.map((score, i) => [score, truth[i]]).sort((a, b) => b[0] - a[0]);
  if (!order.every(([score]) => Number.isFinite(score))) throw Error('nonfinite validation margin');
  const positives = truth.reduce((a, b) => a + b, 0);
  const best = {}, ranks = {};
  let tp = 0;
  for (let i = 0; i < order.length;) {
    const last = order[i][0];
    do { tp += order[i++][1]; } while (i < order.length && order[i][0] === last);
    if (i < 20) continue;
    // Use the midpoint in the unoccupied interval so roundoff in a folded
    // representation cannot flip the boundary validation example.
    const cut = i < order.length ? (last + order[i][0]) / 2 : last - Math.max(1, Math.abs(last));
    const precision = tp / i, recall = tp / positives;
    const candidates = { f05: [5 * tp / (4 * i + positives), precision, recall, cut] };
    if (2 * tp >= i) candidates.majority = [recall, precision, cut];
    for (const [mode, rank] of Object.entries(candidates)) {
      if (!preferred(rank, ranks[mode])) continue;
      ranks[mode] = rank;
      best[mode] = { threshold: cut, flags: i, tp };
    }
  }
  for (const mode of ['majority', 'f05']) {
    best[mode] ??= { threshold: null, flags: 0, tp: 0 };
    const p = best[mode];
    const flags = scores.map(v => p.threshold !== null && v > p.threshold);
    if (flags.filter(Boolean).length !== p.flags || flags.reduce((a, v, i) => a + Number(v) * truth[i], 0) !== p.tp)
      throw Error('threshold fails validation witness');
  }
  return best;
}

const description = split => Object.fromEntries(['rows','positive','dialogues','idsSha256'].map(k => [k, split[k]]));
const record = { data: { fit: description(fit), tune: description(tune), trainScreened: data.trainScreened, screen: data.screen }, methods: {} };
for (const name of ['transfer','local','pca']) {
  const x = features(fit, name), width = x[0].length;
  const base = { schema: 'ruv-fann.dense-layers/1', layers: [
    { inputs: width, outputs: 1, weights: Array(width).fill(0), bias: [0], activation: 'Linear', steepness: 1 }
  ] };
  const batch = { rows: fit.rows, inputs: x.flat() };
  const cfg = { simd: false };
  const trace = await wsp.traceWeightModel(base, batch, cfg);
  const patch = await wsp.proposeReadoutDelta(base, batch, { baseHash: trace.baseHash, inputHash: trace.inputHash,
    output: 0, deltas: fit.truth.map(y => 2*y - 1) }, cfg);
  const model = patch ? await wsp.applyWeightPatch(base, patch) : base;
  const h = model.layers[0];
  const scores = features(tune, name).map(row => row.reduce((z, v, i) => z + v*h.weights[i], h.bias[0]));
  record.methods[name] = { slopes: h.weights, intercept: h.bias[0], modelSha256: await wsp.fingerprintModel(model),
    points: select(scores, tune.truth) };
}
writeFileSync(join(directory, 'policies.json'), JSON.stringify(record) + '\n');
