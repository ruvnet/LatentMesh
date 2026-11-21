import assert from 'node:assert/strict';
import test from 'node:test';
import { generateKeyPairSync, sign } from 'node:crypto';
import { wsp } from './dependency.mjs';
import { decodeTransfer, encodeHiddenTransfer, encodeTransfer, fingerprintDirection, receiveTransfer } from './transfer.mjs';

const key = generateKeyPairSync('ed25519'), other = generateKeyPairSync('ed25519');
const probes = { rows: 5, inputs: [-2, -1, 0, 1, 2] };
const mk = layers => ({ schema: 'ruv-fann.dense-layers/1', layers });
const layer = (inputs, outputs, weights, bias, activation = 'Linear') =>
  ({ inputs, outputs, weights, bias, activation, steepness: 1 });
const base = mk([layer(1, 2, [0, -1], [0, 0])]);
const edited = mk([layer(1, 2, [2, -1], [1, 0])]);
const receiver = mk([layer(1, 2, [-1, 1], [0, 0], 'ReLU'), layer(2, 2, [1, -1, 0, 0], [0, 0])]);
const { inputHash, baseHash } = await wsp.traceWeightModel(base, probes, { simd: false });
const pact = { schema: 'latentmesh.readout-contract/1', skill: 'test/affine', sourceOutput: 0, targetOutput: 1,
  sourceBaseHash: baseHash, sourceEditedHash: await wsp.fingerprintModel(edited),
  targetBaseHash: await wsp.fingerprintModel(receiver), inputHash };

test('transfers a nonconstant hidden coordinate that the sender output discards entirely', async () => {
  const source = mk([layer(1, 3, [1, -1, 0.5], [0, 0, 0], 'ReLU'), layer(3, 1, [0, 0, 0], [4])]);
  const trace = await wsp.traceWeightModel(source, probes, { simd: false });
  assert.deepEqual(trace.layers.at(-1).activations, Array(5).fill(4));
  const direction = { layer: 0, weights: [1, -2, 0], bias: 0.25 };
  const hidden = { schema: 'latentmesh.hidden-contract/1', skill: 'test/signed-feature',
    sourceHash: trace.baseHash, directionHash: fingerprintDirection(direction), inputHash,
    targetBaseHash: pact.targetBaseHash, targetOutput: 1 };
  const packet = await encodeHiddenTransfer(source, probes, direction, hidden, key.privateKey);
  assert.equal(new Set(decodeTransfer(packet, hidden, key.publicKey, 'hiddenQ8').values).size, 5);
  const patch = await receiveTransfer(packet, receiver, probes, hidden, key.publicKey, 'hiddenQ8');
  const evaluation = await wsp.evaluateWeightPatch(receiver, patch,
    { rows: 3, inputs: [-1.5, 0.5, 3], targets: [1.5, -2.75, -0.5, 0.75, -3, 3.25] },
    { maxMSE: 0.001, maxRegressionMSE: 0 }, { simd: false });
  assert.ok(evaluation.accepted);
  assert.throws(() => decodeTransfer(packet, hidden, key.publicKey, 'q8'), /contract type/);
  await assert.rejects(encodeHiddenTransfer(source, probes, { ...direction, bias: 2 }, hidden, key.privateKey), /direction contract/);
  const final = { layer: 1, weights: [1], bias: 0 };
  await assert.rejects(encodeHiddenTransfer(source, probes, final,
    { ...hidden, directionHash: fingerprintDirection(final) }, key.privateKey), /hidden layer/);
});

test('transfers one output between different architectures, evaluates fresh cases and rolls back', async () => {
  const packet = await encodeTransfer(base, edited, probes, pact, key.privateKey);
  assert.equal(packet.length, 111);
  const { values } = decodeTransfer(packet, pact, key.publicKey);
  assert.ok(values.every((v, i) => Math.abs(v - (2 * probes.inputs[i] + 1)) <= 0.021));
  const patch = await receiveTransfer(packet, receiver, probes, pact, key.publicKey);
  assert.ok(patch.edits.every(e => e.layer === 1 && (e.parameter === 'bias' ? e.index === 1 : e.index >= 2)));
  const ws = await wsp.createWeightWorkspace(receiver, { simd: false });
  const data = { rows: 2, inputs: [-1.5, 3], targets: [1.5, -2, -3, 7] };
  const reject = await ws.propose(patch, data, { maxMSE: 0, maxRegressionMSE: 0 });
  assert.throws(() => ws.promote(reject.id, 1), /not accepted/);
  const okay = await ws.propose(patch, data, { maxMSE: 0.001, maxRegressionMSE: 0 });
  assert.ok(okay.evaluation.accepted);
  assert.equal(ws.snapshot().hash, pact.targetBaseHash);
  const promoted = ws.promote(okay.id, 1);
  await assert.rejects(receiveTransfer(packet, promoted.model, probes, pact, key.publicKey), /hash mismatch/i);
  assert.equal(ws.rollback(1, promoted.revision).hash, pact.targetBaseHash);
});

test('pins both sender revisions, ordered probes, receiver, meaning and trusted signer', async () => {
  const payload = await encodeTransfer(base, edited, probes, pact, key.privateKey);
  const reversed = { ...probes, inputs: probes.inputs.toReversed() };
  await assert.rejects(encodeTransfer(edited, base, probes, pact, key.privateKey), /sender/);
  await assert.rejects(encodeTransfer(base, edited, reversed, pact, key.privateKey), /probes/);
  await assert.rejects(receiveTransfer(payload, receiver, reversed, pact, key.publicKey), /hash mismatch/i);
  assert.throws(() => decodeTransfer(payload, { ...pact, skill: 'different' }, key.publicKey), /contract mismatch/);
  assert.throws(() => decodeTransfer(payload, { ...pact, targetOutput: 0 }, key.publicKey), /contract mismatch/);
  assert.throws(() => decodeTransfer(payload, pact, other.publicKey), /signature/);
  const corrupt = Buffer.from(payload); corrupt[43] ^= 8;
  assert.throws(() => decodeTransfer(corrupt, pact, key.publicKey), /signature/);
  assert.throws(() => decodeTransfer(payload, pact, key.publicKey, 'teacherBit'), /layout/);
  for (const invalid of [payload.subarray(0, -1), Buffer.concat([payload, Buffer.alloc(1)]), Buffer.alloc(2048)])
    assert.throws(() => decodeTransfer(invalid, pact, key.publicKey), /length|layout/);
});

test('handles zero deltas, packed teacher labels and malformed signed payloads', async () => {
  const unchanged = { ...pact, sourceEditedHash: baseHash };
  const zero = await encodeTransfer(base, base, probes, unchanged, key.privateKey);
  assert.equal(await receiveTransfer(zero, receiver, probes, unchanged, key.publicKey), null);
  function alter(packet, mutate) {
    const body = Buffer.from(packet.subarray(0, -64)); mutate(body);
    return Buffer.concat([body, sign(null, body, key.privateKey)]);
  }
  for (const mutate of [b => b.writeFloatBE(NaN, 38), b => b.writeFloatBE(-1, 38),
    b => b.writeInt8(-128, 42), b => b.writeInt8(1, 42)]) {
    assert.throws(() => decodeTransfer(alter(zero, mutate), unchanged, key.publicKey), /scale|coefficient/);
  }
  const bits = await encodeTransfer(base, edited, probes, pact, key.privateKey, 'teacherBit');
  assert.deepEqual(decodeTransfer(bits, pact, key.publicKey, 'teacherBit').values, [0, 0, 1, 1, 1]);
  assert.ok(await receiveTransfer(bits, receiver, probes, pact, key.publicKey, 'teacherBit'));
  assert.throws(() => decodeTransfer(alter(bits, b => { b[38] |= 128; }), pact, key.publicKey, 'teacherBit'), /padding/);
  await assert.rejects(encodeTransfer(base, edited, { rows: 65, inputs: [] }, pact, key.privateKey), /64/);
});
