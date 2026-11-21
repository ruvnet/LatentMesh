import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { margin } from './receiver.mjs';

const out = process.argv[2];
if (!out) throw Error('usage: node predict.mjs RUN_DIRECTORY');
const lockBytes = readFileSync(join(out, 'lock.json'));
const lock = JSON.parse(lockBytes);
const data = JSON.parse(readFileSync(join(out, 'test.json')));
const lockSha256 = createHash('sha256').update(lockBytes).digest('hex');
assert.equal(data.lockSha256, lockSha256);
assert.equal(data.headsSha256, lock.headsSha256);
assert.deepEqual(data.arms, lock.arms);
const original = JSON.parse(readFileSync(new URL('../lock.json', import.meta.url))).methods.wsp48.policy;
const stricter = JSON.parse(readFileSync(new URL('../target/lock.json', import.meta.url))).policies.wsp48;
const digits = ar => ar.map(v => v < 0 ? 'x' : String(v)).join('');
const output = { schema: 'latentmesh.confidence-observations/1', lockSha256, panels: {} };
for (const [key, name, sources] of [['unseen', 'transfer', ['meld']], ['matched', 'adapted', ['meld']],
                                      ['social', 'transfer', ['carer', 'tweet']]]) {
  const get = key => sources.flatMap(k => data.sets[k][key]);
  const native = get('native'), confidence = get('confidence'), scores = get('scores');
  const panel = { ids: sources.map(k => data.sets[k].ids), groups: get('group'), skill: digits(get('skill')),
    sentiment: digits(get('sentiment')), native: digits(native), methods: {} };
  for (const [m, policy] of Object.entries(lock.policies[name])) {
    const col = data.arms.indexOf(m === 'scoreOnly' ? 'wsp48' : m);
    panel.methods[m] = Object.fromEntries(['protected', 'tag', 'separate'].map(mode => [mode, digits(scores.map((s, i) =>
      Number(margin(policy[mode].coefficients, m === 'confidenceOnly' ? 0 : s[col], m === 'scoreOnly' ? 0 : confidence[i]) > 0)))]));
  }
  const zeros = '0'.repeat(native.length);
  panel.methods.intact = { protected: zeros, tag: zeros, separate: zeros };
  for (const [label, p] of Object.entries({ priorSource: original, priorProtected: stricter })) {
    const decisions = digits(scores.map((s, i) => Number((!p.negativeOnly || native[i] === 0) &&
      (p.cut === 'always' || (p.cut !== 'never' && s[0] > p.cut)))));
    panel.methods[label] = { protected: decisions, tag: decisions, separate: decisions };
  }
  output.panels[key] = panel;
}
writeFileSync(join(out, 'observations.json'), JSON.stringify(output) + '\n');
