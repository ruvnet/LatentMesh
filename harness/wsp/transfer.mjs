import { createHash, sign, verify } from 'node:crypto';
import { wsp } from './dependency.mjs';

const magic = Buffer.from('LWS1');
const header = 38; // magic, encoding, count, contract digest
const signatureBytes = 64;
const maxRows = 64;
const options = { simd: false };
const digestPattern = /^[a-f0-9]{64}$/;

// The caller provisions this manifest and the verification key independently.
// Only its digest travels. Each checkpoint and the probe order have fixed roles.
function agreement(contract) {
  const { schema, skill, sourceBaseHash, sourceEditedHash, sourceHash, directionHash,
    targetBaseHash, inputHash, sourceOutput, targetOutput } = contract;
  const hidden = schema === 'latentmesh.hidden-contract/1';
  if ((!hidden && schema !== 'latentmesh.readout-contract/1') || typeof skill !== 'string' ||
      !/^[a-z0-9][a-z0-9._/:-]{0,95}$/.test(skill)) throw new Error('invalid skill contract');
  const hashes = [...(hidden ? [sourceHash, directionHash] : [sourceBaseHash, sourceEditedHash]), targetBaseHash, inputHash];
  if (hashes.some(h => typeof h !== 'string' || !digestPattern.test(h))) throw new Error('invalid contract hash');
  const outputs = hidden ? [targetOutput] : [sourceOutput, targetOutput];
  if (outputs.some(i => !Number.isSafeInteger(i) || i < 0 || i > 65535))
    throw new Error('invalid contract output');
  const digest = createHash('sha256').update(JSON.stringify([schema, skill, ...hashes, ...outputs])).digest();
  return { digest, hidden, sourceHash, directionHash, sourceBaseHash, sourceEditedHash, targetBaseHash, inputHash, sourceOutput, targetOutput };
}

function rowCount(rows) {
  if (!Number.isInteger(rows) || rows < 1 || rows > maxRows) throw new RangeError('require 1 to 64 shared probes');
  return rows;
}

function encodingTag(encoding) {
  if (encoding === 'q8') return 0;
  if (encoding === 'teacherBit') return 1;
  if (encoding === 'hiddenQ8') return 2;
  throw new Error('unsupported transfer encoding');
}

function score(trace, row, output) {
  const head = trace.layers.at(-1);
  if (output >= head.outputs) throw new Error('output outside checkpoint');
  return head.activations[row * head.outputs + output];
}

function winner(trace, row) {
  const head = trace.layers.at(-1);
  let best = 0;
  for (let i = 1; i < head.outputs; i++) if (score(trace, row, i) > score(trace, row, best)) best = i;
  return best;
}

function signedPayload(pact, tag, values, privateKey) {
  if (privateKey?.type !== 'private' || privateKey.asymmetricKeyType !== 'ed25519') throw new Error('require Ed25519 signing key');
  const rows = values.length, isBit = tag === 1;
  const body = Buffer.alloc(header + (isBit ? Math.ceil(rows / 8) : 4 + rows));
  magic.copy(body); body[4] = tag; body[5] = rows; pact.digest.copy(body, 6);
  if (isBit) {
    values.forEach((bit, i) => { if (bit) body[header + (i >> 3)] |= 1 << (i % 8); });
  } else {
    const peak = Math.max(...values.map(Math.abs));
    const scale = peak === 0 ? 0 : Math.fround(peak / 127);
    if (!Number.isFinite(scale) || (peak > 0 && scale === 0)) throw new RangeError('unrepresentable q8 scale');
    body.writeFloatBE(scale, header);
    values.forEach((d, i) => body.writeInt8(scale === 0 ? 0 : Math.max(-127, Math.min(127, Math.round(d / scale))), header + 4 + i));
  }
  return Buffer.concat([body, sign(null, body, privateKey)]);
}

// This is an externally provisioned, local projection with a declared meaning.
// Its coefficients stay at the sender. The receiver sees one coordinate per probe.
function readDirection(direction) {
  const { layer, weights, bias } = direction;
  if (!Number.isSafeInteger(layer) || layer < 0 || layer >= 64 || !Number.isFinite(bias) ||
      !Array.isArray(weights) || weights.length < 1 || weights.length > 128) throw new Error('invalid hidden direction');
  const coefficients = Array.from(weights);
  if (!coefficients.every(Number.isFinite)) throw new Error('invalid hidden coefficient');
  const hash = createHash('sha256').update(JSON.stringify(['latentmesh.linear-direction/1', layer, coefficients, bias])).digest('hex');
  return { layer, coefficients, bias, hash };
}

export function fingerprintDirection(direction) { return readDirection(direction).hash; }

export async function encodeHiddenTransfer(source, probes, direction, contract, privateKey) {
  const pact = agreement(contract), rows = rowCount(probes.rows), axis = readDirection(direction);
  if (!pact.hidden || axis.hash !== pact.directionHash) throw new Error('hidden direction contract mismatch');
  const trace = await wsp.traceWeightModel(source, probes, options);
  if (trace.baseHash !== pact.sourceHash || trace.inputHash !== pact.inputHash) throw new Error('sender checkpoint or probes mismatch');
  const hidden = trace.layers[axis.layer];
  if (axis.layer >= trace.layers.length - 1 || hidden.outputs !== axis.coefficients.length)
    throw new Error('direction must match a hidden layer');
  const values = Array.from({ length: rows }, (_, row) => {
    let value = axis.bias;
    for (let i = 0; i < hidden.outputs; i++) value += axis.coefficients[i] * hidden.activations[row * hidden.outputs + i];
    return value;
  });
  return signedPayload(pact, 2, values, privateKey);
}

export async function encodeTransfer(base, edited, probes, contract, privateKey, encoding = 'q8') {
  const pact = agreement(contract), rows = rowCount(probes.rows), tag = encodingTag(encoding);
  if (pact.hidden || tag === 2) throw new Error('require a readout contract and encoding');
  const [before, after] = await Promise.all([
    wsp.traceWeightModel(base, probes, options), wsp.traceWeightModel(edited, probes, options)
  ]);
  if (before.baseHash !== pact.sourceBaseHash || after.baseHash !== pact.sourceEditedHash ||
      before.inputHash !== pact.inputHash || after.inputHash !== pact.inputHash)
    throw new Error('sender checkpoint or probes mismatch');
  if (pact.sourceOutput >= Math.min(before.layers.at(-1).outputs, after.layers.at(-1).outputs))
    throw new Error('source output outside checkpoint');
  const values = Array.from({ length: rows }, (_, i) => tag === 1 ? Number(winner(after, i) === pact.sourceOutput)
    : score(after, i, pact.sourceOutput) - score(before, i, pact.sourceOutput));
  return signedPayload(pact, tag, values, privateKey);
}

export function decodeTransfer(packet, contract, publicKey, encoding = 'q8') {
  const pact = agreement(contract), tag = encodingTag(encoding);
  if (pact.hidden !== (tag === 2)) throw new Error('encoding does not match contract type');
  if (publicKey?.type !== 'public' || publicKey.asymmetricKeyType !== 'ed25519') throw new Error('require trusted Ed25519 public key');
  if (!(packet instanceof Uint8Array) || packet.length < header + 1 + signatureBytes ||
      packet.length > header + 4 + maxRows + signatureBytes) throw new RangeError('invalid transfer length');
  const bytes = Buffer.from(packet), rows = rowCount(bytes[5]);
  const end = header + (tag === 1 ? Math.ceil(rows / 8) : 4 + rows);
  if (bytes.length !== end + signatureBytes || !bytes.subarray(0, 4).equals(magic) || bytes[4] !== tag)
    throw new Error('invalid transfer layout or encoding');
  if (!verify(null, bytes.subarray(0, end), publicKey, bytes.subarray(end))) throw new Error('invalid transfer signature');
  if (!bytes.subarray(6, header).equals(pact.digest)) throw new Error('transfer contract mismatch');
  const values = [];
  if (tag !== 1) {
    const scale = bytes.readFloatBE(header);
    if (!Number.isFinite(scale) || scale < 0 || Object.is(scale, -0)) throw new Error('invalid q8 scale');
    for (let i = 0; i < rows; i++) {
      const q = bytes.readInt8(header + 4 + i);
      if (q === -128 || (scale === 0 && q !== 0)) throw new Error('invalid q8 coefficient');
      values.push(q * scale);
    }
  } else {
    if (rows % 8 && bytes[end - 1] >> (rows % 8)) throw new Error('invalid teacher bit padding');
    for (let i = 0; i < rows; i++) values.push((bytes[header + (i >> 3)] >> (i % 8)) & 1);
  }
  return { baseHash: pact.targetBaseHash, inputHash: pact.inputHash, output: pact.targetOutput, values };
}

/** Return a standard WSP proposal. Promotion belongs to a local weight workspace. */
export async function receiveTransfer(packet, receiver, probes, contract, publicKey, encoding = 'q8') {
  const { baseHash, inputHash, output, values } = decodeTransfer(packet, contract, publicKey, encoding);
  if (rowCount(probes.rows) !== values.length) throw new Error('probe count mismatch');
  if (encoding === 'teacherBit') {
    // Teacher membership is an absolute target. Convert it to a correction on
    // the very same receiver snapshot used by the patch fitter.
    const trace = await wsp.traceWeightModel(receiver, probes, options);
    if (trace.baseHash !== baseHash || trace.inputHash !== inputHash) throw new Error('receiver checkpoint or probes mismatch');
    for (let i = 0; i < values.length; i++) values[i] -= score(trace, i, output);
  }
  return wsp.proposeReadoutDelta(receiver, probes, { baseHash, inputHash, output, deltas: values }, options);
}
