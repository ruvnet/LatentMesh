import { createHash, sign, verify } from 'node:crypto';
export const sha = bytes => createHash('sha256').update(bytes).digest('hex');
export const methods = ['wsp48', 'softKD', 'float13', 'hard416', 'binary4', 'text4', 'observed8'];
const magic = Buffer.from('LWF3');
const upper = (value, lo, hi) => Math.min(hi, Math.max(lo, value));
const sizes = { wsp48: 48, softKD: 48, float13: 13, hard416: 416, binary4: 104, text4: 78, observed8: 48 };

export function encode(method, numbers, hash, key) {
  if (!Object.hasOwn(sizes, method) || !Array.isArray(numbers) || numbers.length !== sizes[method] || !numbers.every(Number.isFinite))
    throw new Error('invalid method, count or nonfinite number');
  if (!/^[a-f0-9]{64}$/.test(hash) || key?.type !== 'private' || key.asymmetricKeyType !== 'ed25519') throw new Error('invalid signing context');
  if (['hard416', 'binary4', 'text4', 'observed8'].includes(method) && numbers.some(x => Math.abs(x) > 1))
    throw new Error('teacher confidence out of range');
  const b = Buffer.alloc(52);
  if (method === 'float13') {
    numbers.forEach((x, i) => { if (!Number.isFinite(Math.fround(x))) throw new Error('float range'); b.writeFloatBE(x, i * 4); });
  } else if (method === 'hard416') {
    numbers.forEach((x, i) => { if (x > 0) b[i >> 3] |= 1 << (i % 8); });
  } else if (method === 'text4' || method === 'binary4') {
    const nibbles = Buffer.alloc(numbers.length / 2);
    numbers.forEach((x, i) => { nibbles[i >> 1] |= Math.round((x + 1) * 7.5) << (i % 2 ? 0 : 4); });
    (method === 'text4' ? Buffer.from(nibbles.toString('base64'), 'ascii') : nibbles).copy(b);
  } else {
    const peak = Math.max(...numbers.map(Math.abs)), scale = Math.fround(peak / 127);
    if (!Number.isFinite(scale) || (peak && !scale)) throw new Error('invalid scale');
    b.writeFloatBE(scale);
    numbers.forEach((x, i) => b.writeInt8(scale ? upper(Math.round(x / scale), -127, 127) : 0, i + 4));
  }
  const header = Buffer.concat([magic, Buffer.from([methods.indexOf(method), 1]), Buffer.from(hash, 'hex')]);
  const body = Buffer.concat([header, b]);
  return Buffer.concat([body, sign(null, body, key)]);
}

export function decode(method, packet, hash, key) {
  if (!Object.hasOwn(sizes, method) || !/^[a-f0-9]{64}$/.test(hash) || key?.type !== 'public' || key.asymmetricKeyType !== 'ed25519')
    throw new Error('invalid receiver context');
  if (!(packet instanceof Uint8Array) || packet.length !== 154) throw new Error('packet length');
  const bytes = Buffer.from(packet), body = bytes.subarray(0, 90);
  if (!verify(null, body, key, bytes.subarray(90))) throw new Error('signature mismatch');
  if (!body.subarray(0, 4).equals(magic) || body[4] !== methods.indexOf(method) || body[5] !== 1 || body.subarray(6, 38).toString('hex') !== hash)
    throw new Error('wrong public context or method');
  const b = body.subarray(38), count = sizes[method];
  if (method === 'float13') {
    const v = Array.from({ length: count }, (_, i) => b.readFloatBE(i * 4));
    if (!v.every(Number.isFinite)) throw new Error('invalid parameter');
    return v;
  }
  if (method === 'hard416') return Array.from({ length: count }, (_, i) => b[i >> 3] & (1 << (i % 8)) ? 1 : -1);
  if (method === 'text4' || method === 'binary4') {
    const packed = method === 'text4' ? Buffer.from(b.toString('latin1'), 'base64') : b;
    if (method === 'text4' && (packed.length !== 39 || packed.toString('base64') !== b.toString('latin1'))) throw new Error('invalid text');
    return Array.from({ length: count }, (_, i) => (((packed[i >> 1] >> (i % 2 ? 0 : 4)) & 15) / 7.5) - 1);
  }
  const scale = b.readFloatBE();
  if (!Number.isFinite(scale) || scale < 0 || Object.is(scale, -0) || (method === 'observed8' && scale > Math.fround(1 / 127)))
    throw new Error('invalid scale');
  return Array.from({ length: count }, (_, i) => {
    const n = b.readInt8(4 + i);
    if (n === -128 || (!scale && n)) throw new Error('invalid coefficient');
    return n * scale;
  });
}
