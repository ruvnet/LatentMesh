// Experimental, non-authoritative control data. No network, tools or promotion.
import { createHash, sign, verify } from 'node:crypto';

export const MODES = Object.freeze(['explore', 'converge', 'challenge']);
const fail = message => { throw new Error(message); };
const finite = x => typeof x === 'number' && Number.isFinite(x);
const unit = x => finite(x) && x >= 0 && x <= 1;
const id = x => typeof x === 'string' && /^[a-zA-Z0-9._:/-]{1,128}$/.test(x);
const digest = x => typeof x === 'string' && /^[a-f0-9]{64}$/.test(x);
const vector = (x, n) => Array.isArray(x) && x.length === n && x.every(finite);
const hash = bytes => createHash('sha256').update(bytes).digest('hex');

/** Local sparse linear probe; coefficients must be fitted on disjoint data.
 * Thresholds are explicit experimental policy, NOT calibrated confidence claims.
 * Unknown/out-of-support inputs abstain. No universal neuron indices are supplied.
 */
export function detectMode(activation, probe) {
  const {indices, weights, bias, mean, scale, maxZ, minProbability, model, revision} = probe;
  if (!id(model) || !id(revision) || !Array.isArray(indices) || indices.length < 1 || indices.length > 64 ||
      new Set(indices).size !== indices.length || !indices.every(x => Number.isInteger(x) && x >= 0) ||
      !Array.isArray(weights) || weights.length !== 3 || !weights.every(w => vector(w, indices.length)) ||
      !vector(bias, 3) || !vector(mean, indices.length) || !vector(scale, indices.length) ||
      !scale.every(x => x > 0) || !finite(maxZ) || maxZ <= 0 || !unit(minProbability)) fail('invalid probe');
  const unknown = reason => ({mode: 'unknown', probability: 0, reason});
  if (!Array.isArray(activation) || indices.some(i => i >= activation.length || !finite(activation[i]))) return unknown('invalid-activation');
  const z = indices.map((i, j) => (activation[i] - mean[j]) / scale[j]);
  if (z.some(x => !finite(x) || Math.abs(x) > maxZ)) return unknown('out-of-support');
  const logits = weights.map((w, i) => w.reduce((s, v, j) => s + v * z[j], bias[i]));
  if (!logits.every(finite)) return unknown('numeric-overflow');
  const max = Math.max(...logits), exp = logits.map(x => Math.exp(x - max));
  const sum = exp.reduce((a, b) => a + b, 0), probabilities = exp.map(x => x / sum);
  const index = probabilities.indexOf(Math.max(...probabilities));
  if (probabilities.filter(x => x === probabilities[index]).length !== 1 || probabilities[index] < minProbability) return unknown('ambiguous');
  return {mode: MODES[index], probability: probabilities[index], reason: 'local-probe'};
}

// Fixed field order and exact keys prevent unsigned extension fields or ambiguous JSON.
const fields = ['version', 'sender', 'session', 'sequence', 'expiresAt', 'targetModel', 'targetRevision', 'adapterHash', 'mode', 'probability', 'evidenceHash'];
function canonical(p) {
  if (!p || Object.keys(p).sort().join() !== [...fields].sort().join() || p.version !== 1 ||
      ![p.sender,p.session,p.targetModel,p.targetRevision].every(id) ||
      !Number.isSafeInteger(p.sequence) || p.sequence < 0 ||
      !Number.isSafeInteger(p.expiresAt) || p.expiresAt < 0 ||
      !digest(p.adapterHash) || !digest(p.evidenceHash) ||
      ![...MODES,'unknown'].includes(p.mode) || !unit(p.probability)) fail('invalid mode message');
  return Buffer.from(JSON.stringify(Object.fromEntries(fields.map(k => [k,p[k]]))));
}

export function signMessage(payload, privateKey) {
  if (privateKey.asymmetricKeyType !== 'ed25519') fail('Ed25519 key required');
  const bytes = canonical(payload);
  return JSON.stringify({payload: JSON.parse(bytes), signature: sign(null, bytes, privateKey).toString('hex')});
}

/** One receiver per trusted sender/session. Retain high-water state across restart.
 * A valid signature authenticates origin, not correctness or action authority.
 */
export class ModeReceiver {
  #last; #policy;
  constructor(policy, lastSequence = -1) {
    if (!id(policy.sender) || !id(policy.session) || !id(policy.targetModel) || !id(policy.targetRevision) ||
        !digest(policy.adapterHash) || !unit(policy.minProbability) ||
        !Number.isSafeInteger(policy.maxTtlMs) || policy.maxTtlMs <= 0 ||
        policy.publicKey?.asymmetricKeyType !== 'ed25519' ||
        !Number.isSafeInteger(lastSequence) || lastSequence < -1) fail('invalid receiver policy');
    this.#policy = Object.freeze({...policy}); this.#last = lastSequence;
  }
  get lastSequence() { return this.#last; }
  receive(wire, now) {
    if (typeof wire !== 'string' || Buffer.byteLength(wire) > 4096 || !Number.isSafeInteger(now) || now < 0) fail('invalid envelope');
    const envelope = JSON.parse(wire), p = envelope.payload, policy = this.#policy;
    if (Object.keys(envelope).sort().join() !== 'payload,signature' ||
        typeof envelope.signature !== 'string' || !/^[a-f0-9]{128}$/.test(envelope.signature)) fail('invalid signature encoding');
    const bytes = canonical(p);
    if (!verify(null, bytes, policy.publicKey, Buffer.from(envelope.signature,'hex'))) fail('bad signature');
    for (const field of ['sender','session','targetModel','targetRevision','adapterHash']) if (p[field] !== policy[field]) fail(`wrong ${field}`);
    if (p.expiresAt <= now || p.expiresAt - now > policy.maxTtlMs) fail('expired or excessive TTL');
    if (p.sequence <= this.#last) fail('replay');
    this.#last = p.sequence;
    return Object.freeze({mode: p.probability >= policy.minProbability ? p.mode : 'unknown',
      messageHash: hash(bytes), evidenceHash: p.evidenceHash, sequence: p.sequence,
      advisoryOnly: true});
  }
}

/** Hash exact serialized artifact bytes before parsing; trust is an external decision. */
export function loadAdapter(bytes, expectedHash) {
  if (typeof bytes !== 'string' || Buffer.byteLength(bytes) > 4_000_000 || !digest(expectedHash) || hash(bytes) !== expectedHash) fail('adapter integrity');
  const a = JSON.parse(bytes);
  if (!id(a.model) || !id(a.revision) || !Number.isInteger(a.width) || a.width < 1 || a.width > 32768 ||
      !Array.isArray(a.layers) || !a.layers.length || a.layers.length > 16 ||
      new Set(a.layers.map(l => l.index)).size !== a.layers.length) fail('invalid adapter');
  for (const l of a.layers) if (!Number.isInteger(l.index) || l.index < 0 || l.index > 1023 ||
      !l.directions || !MODES.every(m => vector(l.directions[m], a.width))) fail('invalid layer');
  // Deep freeze eliminates artifact mutation after content-hash validation.
  for (const l of a.layers) { for (const m of MODES) Object.freeze(l.directions[m]); Object.freeze(l.directions); Object.freeze(l); }
  Object.freeze(a.layers); return Object.freeze({...a, hash: expectedHash});
}

/** Reference additive residual operator for an OFFLINE experiment only.
 * Returns a copy. Never mutates model weights or input activation, or bypasses
 * the existing causal/authority gates. No live Candle runtime hook is installed.
 */
export function steerResidual(residual, {adapter, model, revision, layer, mode, gain, maxDeltaNorm, researchEnabled = false}) {
  if (!Array.isArray(residual) || !residual.every(finite)) fail('invalid residual');
  if (!finite(gain) || gain < 0 || !finite(maxDeltaNorm) || maxDeltaNorm < 0) fail('invalid bounds');
  const base = [...residual];
  if (!researchEnabled || mode === 'unknown' || gain === 0) return {residual: base, applied: false, reason: 'observe-only'};
  if (!MODES.includes(mode) || model !== adapter.model || revision !== adapter.revision || residual.length !== adapter.width) fail('incompatible adapter');
  const target = adapter.layers.find(l => l.index === layer);
  if (!target) return {residual: base, applied: false, reason: 'unsupported-layer'};
  const delta = target.directions[mode].map(x => x * gain);
  const norm = Math.hypot(...delta);
  if (!finite(norm) || norm > maxDeltaNorm) fail('delta norm budget exceeded');
  const output = residual.map((x,i) => x + delta[i]);
  if (!output.every(finite)) fail('residual overflow');
  return {residual: output, applied: true, reason: 'offline-experiment', deltaNorm: norm,
    receipt: {adapterHash: adapter.hash, model, revision, layer, mode, gain, maxDeltaNorm,
      beforeHash: hash(JSON.stringify(base)), afterHash: hash(JSON.stringify(output))}};
}
