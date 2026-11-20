import { createHash, sign, verify } from 'node:crypto';

export const budget = 52;
const header = 38;
const signatureBytes = 64;
const magic = Buffer.from('LWT2');
export const arms = {
  hybrid: { tag: 1, count: 48, codec: 'q8' },
  hard48: { tag: 2, count: 48, codec: 'bit' },
  hard64: { tag: 3, count: 64, codec: 'bit' },
  decimal52: { tag: 4, count: 52, codec: 'decimal' },
  text4: { tag: 5, count: 64, codec: 'text4' },
  phrase: { tag: 6, count: 64, codec: 'phrase' },
  native: { tag: 7, count: 48, codec: 'q8' },
  shuffled: { tag: 8, count: 48, codec: 'q8' }
};
export const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const scaleLevel = (value, levels) => Math.round((value + 1) * (levels - 1) / 2);
const unscale = (level, levels) => 2 * level / (levels - 1) - 1;

function specFor(arm) {
  if (!Object.hasOwn(arms, arm)) throw new Error('unknown arm');
  return arms[arm];
}

function encodeValues({ codec, count }, values) {
  if (codec === 'phrase') {
    if (typeof values !== 'string' || !/^[a-z ]+\|[a-z ]+$/.test(values) || values.length > budget)
      throw new Error('invalid phrase');
    return Buffer.from(values, 'ascii');
  }
  if (!Array.isArray(values) || values.length !== count || !values.every(v => Number.isFinite(v) && Math.abs(v) <= 1))
    throw new Error('invalid scores');
  if (codec === 'q8') {
    const buf = Buffer.alloc(4 + count);
    const peak = Math.max(...values.map(Math.abs));
    const scale = Math.fround(peak / 127);
    if (peak && !scale) throw new Error('q8 scale underflow');
    buf.writeFloatBE(scale);
    values.forEach((v, i) => buf.writeInt8(scale ? Math.max(-127, Math.min(127, Math.round(v / scale))) : 0, 4 + i));
    return buf;
  }
  if (codec === 'bit') {
    const buf = Buffer.alloc(count / 8);
    values.forEach((v, i) => { if (v > 0) buf[i >> 3] |= 1 << (i % 8); });
    return buf;
  }
  if (codec === 'decimal') return Buffer.from(values.map(v => scaleLevel(v, 10)).join(''), 'ascii');
  const packed = Buffer.alloc(count / 2);
  values.forEach((v, i) => { packed[i >> 1] |= scaleLevel(v, 16) << (i % 2 ? 0 : 4); });
  return Buffer.from(packed.toString('base64'), 'ascii');
}

function decodeValues({ codec, count }, payload) {
  if (codec === 'phrase') {
    const nul = payload.indexOf(0);
    const size = nul < 0 ? payload.length : nul;
    const value = payload.subarray(0, size).toString('latin1');
    if (!encodeValues({ codec, count }, value).equals(payload.subarray(0, size))) throw new Error('invalid phrase');
    return { size, value };
  }
  if (codec === 'q8') {
    const scale = payload.readFloatBE();
    if (!Number.isFinite(scale) || scale < 0 || Object.is(scale, -0) || scale > Math.fround(1 / 127))
      throw new Error('invalid q8 scale');
    const value = Array.from({ length: count }, (_, i) => {
      const q = payload.readInt8(4 + i);
      if (q === -128 || (!scale && q)) throw new Error('invalid q8 coefficient');
      return q * scale;
    });
    return { size: 4 + count, value };
  }
  if (codec === 'bit') return { size: count / 8, value: Array.from({ length: count },
    (_, i) => (payload[i >> 3] & (1 << (i % 8))) ? 1 : -1) };
  if (codec === 'decimal') {
    const digits = payload.subarray(0, count).toString('latin1');
    if (!/^\d+$/.test(digits)) throw new Error('invalid decimal encoding');
    return { size: count, value: Array.from(digits, c => unscale(Number(c), 10)) };
  }
  const size = 4 * Math.ceil(count / 6);
  const ascii = payload.subarray(0, size).toString('latin1'), bytes = Buffer.from(ascii, 'base64');
  if (bytes.length !== count / 2 || bytes.toString('base64') !== ascii) throw new Error('invalid base64 encoding');
  return { size, value: Array.from({ length: count }, (_, i) => unscale((bytes[i >> 1] >> (i % 2 ? 0 : 4)) & 15, 16)) };
}

function checkContract(hash) {
  if (typeof hash !== 'string' || !/^[a-f0-9]{64}$/.test(hash)) throw new Error('invalid contract digest');
}

export function pack(arm, values, contractHash, privateKey) {
  const spec = specFor(arm);
  checkContract(contractHash);
  if (privateKey?.type !== 'private' || privateKey.asymmetricKeyType !== 'ed25519') throw new Error('require Ed25519 signer');
  const data = encodeValues(spec, values);
  const body = Buffer.alloc(header + budget);
  magic.copy(body); body[4] = spec.tag; body[5] = spec.count;
  Buffer.from(contractHash, 'hex').copy(body, 6); data.copy(body, header);
  return Buffer.concat([body, sign(null, body, privateKey)]);
}

export function unpack(arm, packet, contractHash, publicKey) {
  const spec = specFor(arm);
  checkContract(contractHash);
  if (publicKey?.type !== 'public' || publicKey.asymmetricKeyType !== 'ed25519') throw new Error('require trusted Ed25519 verifier');
  if (!(packet instanceof Uint8Array) || packet.length !== header + budget + signatureBytes) throw new Error('incorrect packet length');
  const bytes = Buffer.from(packet), body = bytes.subarray(0, header + budget);
  if (!verify(null, body, publicKey, bytes.subarray(body.length))) throw new Error('signature mismatch');
  if (!body.subarray(0, 4).equals(magic) || body[4] !== spec.tag || body[5] !== spec.count ||
      body.subarray(6, header).toString('hex') !== contractHash) throw new Error('contract or arm mismatch');
  const payload = body.subarray(header), { size, value } = decodeValues(spec, payload);
  if (payload.subarray(size).some(v => v !== 0)) throw new Error('nonzero padding');
  return { value, usefulBytes: size };
}
