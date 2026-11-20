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
