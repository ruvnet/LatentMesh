import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { createHash } from 'node:crypto';
import { fitLinear, modelHash } from './linear-fit.mjs';
import { margin } from './receiver.mjs';

const out = process.argv[2];
if (!out) throw Error('usage: node fit.mjs RUN_DIRECTORY');
const input = readFileSync(join(out, 'fit.json'));
const { headsSha256, arms, sets } = JSON.parse(input);
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
function samples(data, method) {
  const col = arms.indexOf(method === 'scoreOnly' ? 'wsp48' : method);
  return data.native.map((native, i) => ({
    point: [method === 'confidenceOnly' ? 0 : data.scores[i][col], method === 'scoreOnly' ? 0 : data.confidence[i]],
    skill: data.skill[i], correctNative: native === data.sentiment[i]
  }));
}

function better(a, b) {
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return a[i] > b[i];
  return false;
}

function choose(weights, skill, guard, mode) {
  const raw = r => weights[0] * r.point[0] + weights[1] * r.point[1];
  const ss = skill.map(raw), gs = guard.map(raw);
  const boundary = mode === 'separate' ? [...ss] : [...ss, ...gs];
  const cuts = [Infinity, ...new Set(boundary.sort((a, b) => b - a)), -Infinity];
  const positives = skill.filter(r => r.skill === 1).length;
  const negatives = skill.length - positives;
  assert(positives && negatives && guard.length);
  let best;
  for (const cut of cuts) {
    let tp = 0, fp = 0, lost = 0;
    for (let i = 0; i < skill.length; i++) if (ss[i] > cut) {
      if (skill[i].skill === 1) tp++; else fp++;
    }
    for (let i = 0; i < guard.length; i++) if (gs[i] > cut && guard[i].correctNative) lost++;
    if (mode === 'protected' && lost * 200 > guard.length) continue;
    const rank = [tp * negatives + (negatives - fp) * positives, mode === 'separate' ? 0 : -lost, -(tp + fp), cut];
    if (best && !better(rank, best.rank)) continue;
    const w = Number.isFinite(cut) ? [weights[0], weights[1], -cut] : [0, 0, cut === Infinity ? -1 : 1];
    best = { rank, w, tp, fp, lost };
  }
  const { w, tp, fp, lost } = best;
  assert.equal(guard.filter(r => r.correctNative && margin(w, ...r.point) > 0).length, lost);
  return { coefficients: w, selection: { tp, fp, lost } };
}

async function ridge(skillFit, guardFit) {
  const rows = [...skillFit, ...guardFit];
  const delta = [...skillFit.map(r => r.skill === 1 ? 1 : -1), ...guardFit.map(() => -1)];
  return fitLinear(rows.map(r => r.point), delta);
}

async function fit(method, name) {
  const get = key => samples(sets[name + '.' + key], method);
  const skill = get('skillFit'), guards = get('guardFit').filter(r => r.correctNative);
  const head = await ridge(skill, guards), separateHead = await ridge(skill, []);
  const candidates = { fit: [...head.weights, head.bias[0]], separateFit: [...separateHead.weights, separateHead.bias[0]], correctGuardFit: guards.length };
  for (const mode of ['protected', 'tag', 'separate']) {
    const weights = mode === 'separate' ? separateHead.weights : head.weights;
    const selected = choose(weights, get('skillTune'), get('guardTune'), mode);
    candidates[mode] = { ...selected, modelSha256: await modelHash(selected.coefficients) };
  }
  return candidates;
}

const result = { schema: 'latentmesh.receiver-confidence/1', headsSha256, fitSha256: sha(input), arms,
  sets: Object.fromEntries(Object.entries(sets).map(([key, v]) => [key, { ids: v.ids, rows: v.native.length }])), policies: {} };
for (const name of ['transfer', 'adapted']) {
  const methods = {};
  for (const method of [...arms, 'scoreOnly', 'confidenceOnly']) methods[method] = await fit(method, name);
  result.policies[name] = methods;
}
writeFileSync(join(out, 'lock.json'), JSON.stringify(result, null, 2) + '\n');
