import assert from 'node:assert/strict';
import test from 'node:test';
import { readReceiver } from './receiver.mjs';

test('returns an optional anger tag alongside the original three class decision', () => {
  const w = [2, -0.5, 0.1];
  const result = readReceiver(w, 1, new Float32Array([-1, 3, 0]));
  assert.equal(result.sentiment, 1);
  assert.equal(result.anger, true);
  assert.equal(result.route, 3);
  assert.deepEqual(readReceiver(w, 1, [9999, 10003, 10000]), result);
  const protectedResult = readReceiver([0, 0, -1], 4, [-1, 3, 0]);
  assert.equal(protectedResult.route, protectedResult.sentiment);
  assert.equal(protectedResult.anger, false);
});

test('handles tied logits, strict zero thresholds and malformed inputs', () => {
  const r = readReceiver([1, 0, 0], 0, [5, 5, 5]);
  assert.equal(r.confidence, 1/3);
  assert.equal(r.sentiment, 0);
  assert.equal(r.anger, false);
  for (const [w, score, logits] of [[[1, 2], 3, [1, 0, 0]], [[0, 0, NaN], 2, [0, 0, 1]],
    [[1, 2, 3], Infinity, [0, 0, 1]], [[1, 2, 3], 2, [1, 2]], [[1, 2, 3], 2, [1, 2, Infinity]]]) {
    assert.throws(() => readReceiver(w, score, logits), RangeError);
  }
});

test('published WSP fingerprints cover every operating point and exact coefficients', async () => {
  const { readFileSync } = await import('node:fs');
  const { modelHash } = await import('./linear-fit.mjs');
  const lock = JSON.parse(readFileSync(new URL('./lock.json', import.meta.url)));
  for (const group of Object.values(lock.policies)) {
    for (const policy of Object.values(group)) {
      for (const mode of ['protected', 'tag', 'separate']) {
        assert.equal(await modelHash(policy[mode].coefficients), policy[mode].modelSha256);
      }
    }
  }
});

test('all emotion programs match the committed WSP fingerprints', async () => {
  const { readFileSync } = await import('node:fs');
  const { modelHash } = await import('./linear-fit.mjs');
  const lock = JSON.parse(readFileSync(new URL('./precision/lock.json', import.meta.url)));
  for (const [arm, policy] of Object.entries(lock.policies)) {
    for (const point of Object.values(policy)) {
      assert.equal(await modelHash(point.coefficients), point.modelSha256);
      if (arm === 'scoreOnly') assert.equal(point.coefficients[1], 0);
    }
  }
});

test('both CPUs used the same held out rows, labels, dialogues and frozen programs', async () => {
  const { readFileSync } = await import('node:fs');
  const { inflateSync } = await import('node:zlib');
  function archive(host) {
    const file = new URL('./precision/' + host + '/observations.archive.json', import.meta.url);
    const data = JSON.parse(readFileSync(file));
    return JSON.parse(inflateSync(Buffer.from(data.data, 'base64')));
  }
  const first = archive('reference'), second = archive('ruos');
  for (const key of ['ids', 'truth', 'sentiment', 'groups', 'lockSha256']) {
    assert.deepEqual(first[key], second[key], key);
  }
  const read = host => JSON.parse(readFileSync(new URL('./' + host + '/graphs.json', import.meta.url)));
  const selected = graphs => graphs.find(g => g.parity.arm === 'float13' && g.parity.regime === 'transfer' && g.parity.mode === 'separate');
  const [reference, remote] = [read('reference'), read('ruos')].map(selected);
  assert.equal(reference.install.modelSha256, remote.install.modelSha256);
  for (const graph of [reference, remote]) {
    assert.equal(graph.parity.modelSha256, graph.install.modelSha256);
    assert.equal(graph.parity.rows, 96);
    for (const field of ['maxNativeLogitDrift', 'tagDisagreements', 'routeDisagreements']) assert.equal(graph.parity[field], 0);
  }
});
