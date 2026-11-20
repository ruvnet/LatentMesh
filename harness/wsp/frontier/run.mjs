import assert from 'node:assert/strict';
import { execFileSync, spawnSync } from 'node:child_process';
import { generateKeyPairSync } from 'node:crypto';
import { readFileSync, writeFileSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { wsp } from '../dependency.mjs';
import { methods, sha, encode, decode } from './packet.mjs';

const [stage, output] = process.argv.slice(2);
if (!['compile', 'predict'].includes(stage) || !output) throw Error('usage: node run.mjs compile|predict OUTPUT');
const out = resolve(output), path = name => join(out, name + '.json');
const read = name => JSON.parse(readFileSync(path(name), 'utf8'));
const write = (name, x) => writeFileSync(path(name), JSON.stringify(x) + '\n');
const hash = name => sha(readFileSync(path(name)));
const settings = { simd: false };
const model = w => ({ schema: 'ruv-fann.dense-layers/1', layers: [{ inputs: w, outputs: 1, weights: Array(w).fill(0), bias: [0],
  activation: 'Linear', steepness: 1 }] });
const batch = (rows, w) => ({ rows: rows.length, inputs: rows.flatMap(row => row.slice(0, w)) });
const compactMethods = [...methods.filter(m => m !== 'wsp48'), 'local', 'intact'];
const asBits = bool => bool.map(Number).join('');

async function ridge(bank, teacher, data, width) {
  const base = model(width), rows = [...bank.slice(0, teacher.length), ...data.local, ...data.guard];
  const values = [...teacher, ...data.localLabels, ...data.guard.map(() => -1)], inputs = batch(rows, width);
  const trace = await wsp.traceWeightModel(base, inputs, settings);
  const patch = await wsp.proposeReadoutDelta(base, inputs, { baseHash: trace.baseHash, inputHash: trace.inputHash, output: 0, deltas: values }, settings);
  const fitted = await wsp.applyWeightPatch(base, patch);
  return [...fitted.layers[0].weights, fitted.layers[0].bias[0]];
}

async function apply(values) {
  const width = values.length - 1, base = model(width);
  const edits = values.flatMap((value, i) => value === 0 ? [] : [{
    layer: 0, parameter: i === width ? 'bias' : 'weights', index: i === width ? 0 : i, value
  }]);
  if (!edits.length) return { fitted: base, patch: null };
  const patch = { schema: 'ruv-fann.weight-patch/1', baseHash: await wsp.fingerprintModel(base), edits };
  return { fitted: await wsp.applyWeightPatch(base, patch), patch };
}

async function scores(fitted, rows) {
  const w = fitted.layers[0].inputs, result = [];
  for (let i = 0; i < rows.length; i += 128) {
    const trace = await wsp.traceWeightModel(fitted, batch(rows.slice(i, i + 128), w), settings);
    result.push(...trace.layers[0].activations);
  }
  return result;
}

function overrides(v, native, policy) {
  if (policy.cut === 'never') return false;
  return (!policy.negativeOnly || native === 0) && (policy.cut === 'always' || v > policy.cut);
}
function calibrate(skill, retain, data) {
  const cuts = ['never', ...Array.from(new Set([...skill, ...retain, 0])).sort((a, b) => b - a), 'always'];
  let winner, rank;
  for (const negativeOnly of [false, true]) for (const [j, cut] of cuts.entries()) {
    const p = { negativeOnly, cut };
    const sp = skill.map((v, i) => overrides(v, data.nativeSkill[i], p));
    const rp = retain.map((v, i) => overrides(v, data.nativeRetain[i], p));
    const lost = rp.filter((b, i) => b && data.nativeRetain[i] === data.retained[i]).length;
    if (lost * 200 > retain.length) continue;
    const correct = sp.filter((b, i) => b === data.truth[i]).length;
    const total = [...sp, ...rp].filter(Boolean).length;
    const key = [-correct, lost, total, Number(negativeOnly), j];
    if (rank && !key.some((v, k) => v < rank[k] && key.slice(0, k).every((a, n) => a === rank[n]))) continue;
    winner = { ...p, calibrationCorrect: correct, calibrationLost: lost, calibrationOverrides: total }; rank = key;
  }
  return winner;
}

function air(bin, signed) {
  const r = spawnSync(bin, { input: signed });
  assert.equal(r.status, 0, r.stderr?.toString()); assert.deepEqual(r.stdout, signed);
  const [frames, wireBytes] = r.stderr.toString().trim().split(/\s+/).map(Number);
  assert.equal(frames, 1); assert.equal(wireBytes, 218);
  return { payload: r.stdout, bytes: { useful: 52, application: signed.length, frames, wire: wireBytes } };
}

async function compile(bin) {
  const f = read('fit'), pub = read('public');
  assert.deepEqual(f.public, pub);
  const { publicKey, privateKey } = generateKeyPairSync('ed25519');
  const full = await ridge(f.bank, f.teacher, f, 47), narrow = await ridge(f.bank, f.teacher, f, 12);
  const momentError = Math.max(...narrow.map((v, i) => Math.abs(v - f.momentCoef[i])));
  assert.ok(momentError < 1e-4, '13 float moments must reproduce the WSP reference solve');
  const patches = {}, results = {};
  for (const m of [...methods, 'local', 'oracle', 'intact']) {
    let coef, bytes = { useful: 0, application: 0, frames: 0, wire: 0 };
    if (m === 'local') coef = await ridge([], [], f, 47);
    else if (m === 'oracle') coef = full;
    else if (m === 'intact') coef = Array(48).fill(0);
    else {
      const direct = ['wsp48', 'float13', 'softKD'].includes(m);
      const width = m === 'float13' ? 12 : 47;
      const pact = sha(JSON.stringify([pub, m, await wsp.fingerprintModel(model(width))]));
      const n = { hard416: 416, binary4: 104, text4: 78, observed8: 48 }[m];
      const v = m === 'wsp48' ? full : m === 'float13' ? narrow : m === 'softKD' ? f.kd : f.teacher.slice(0, n);
      const wire = air(bin, encode(m, v, pact, privateKey)); bytes = wire.bytes;
      const received = decode(m, wire.payload, pact, publicKey);
      coef = direct ? received : await ridge(f.bank, received, f, width);
    }
    const { fitted, patch } = await apply(coef);
    const cs = await scores(fitted, f.calSkill), cr = await scores(fitted, f.calRetain);
    const policy = m === 'intact' ? { cut: 'never', negativeOnly: false } : calibrate(cs, cr, f);
    patches[m] = { coef, policy };
    results[m] = { bytes, nonzeroEdits: patch?.edits.length ?? 0, patchHash: sha(JSON.stringify(patch)), policy };
  }
  write('patches', patches);
  write('lock', { schema: 'latentmesh.frontier-lock/1', publicSha256: hash('public'), patchesSha256: hash('patches'),
    fitSha256: hash('fit'), compactMethods, fullVsQ8MaxCoefficientError: Math.max(...full.map((x, i) => Math.abs(x - patches.wsp48.coef[i]))),
    momentMaxCoefficientError: momentError, publicKeySpkiBytes: publicKey.export({ type: 'spki', format: 'der' }).length, methods: results });
}

async function predict() {
  const c = read('heldout'), p = read('patches'), lock = read('lock');
  assert.equal(lock.patchesSha256, hash('patches')); assert.equal(lock.publicSha256, hash('public'));
  const result = { schema: 'latentmesh.frontier-observations/1', lockSha256: hash('lock'), compactMethods, domains: {} };
  for (const domain of ['go', 'tweet']) {
    const s = c[domain];
    result.domains[domain] = { primaryIds: s.skill.ids, retentionIds: s.retain.ids, truth: asBits(s.truth),
      labels: s.labels.join(''), native: s.retain.native.join(''), methods: {} };
  }
  result.official = { ids: c.official.ids, truth: c.official.truth.join(''), specialist: c.official.specialist.join(''),
    exactDevelopmentTextOverlaps: c.official.exactDevelopmentTextOverlaps, methods: {} };
  for (const [m, { coef, policy }] of Object.entries(p)) {
    const { fitted, patch } = await apply(coef);
    assert.equal(sha(JSON.stringify(patch)), lock.methods[m].patchHash);
    const route = async data => (await scores(fitted, data.x)).map((v, i) => overrides(v, data.native[i], policy));
    for (const d of ['go', 'tweet']) {
      result.domains[d].methods[m] = { positive: asBits(await route(c[d].skill)), override: asBits(await route(c[d].retain)) };
    }
    if (m === 'wsp48') {
      const sample = read('graph-sample'), w = await scores(fitted, sample.map(row => row.x));
      write('standalone-graph', w.map((value, i) => ({ value,
        route: overrides(value, sample[i].native, policy) ? 3 : sample[i].native })));
    }
    const projected = await route(c.official);
    result.official.methods[m] = projected.map((bit, i) => bit ? 0 : c.official.specialist[i]).join('');
  }
  write('observations', result);
}

if (stage === 'predict') await predict();
else {
  const temp = mkdtempSync(join(tmpdir(), 'latentmesh-frontier-'));
  try {
    const bin = join(temp, 'air'), cwd = fileURLToPath(new URL('../../../', import.meta.url));
    execFileSync(process.env.CC || 'cc', ['-std=c11', '-O2', '-Wall', '-Wextra', '-Werror', '-Ic/include',
      'harness/wsp/air-loopback.c', 'c/src/air.c', 'c/src/frame.c', 'c/src/fec.c', '-lm', '-o', bin], { cwd });
    await compile(bin);
  } finally { rmSync(temp, { recursive: true, force: true }); }
}
