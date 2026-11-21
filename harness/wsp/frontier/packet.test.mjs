import test from 'node:test';
import assert from 'node:assert/strict';
import { generateKeyPairSync, sign } from 'node:crypto';
import { methods, sha, encode, decode } from './packet.mjs';
const { publicKey, privateKey } = generateKeyPairSync('ed25519'), hash = sha('registered');
const counts = [48, 48, 13, 416, 104, 78, 48];
test('all approved formats fit exactly 52 payload bytes and preserve their declared precision', () => {
  methods.forEach((name, k) => {
    const x = Array.from({ length: counts[k] }, (_, i) => Math.cos(i));
    const packet = encode(name, x, hash, privateKey);
    assert.equal(packet.length, 154);
    const result = decode(name, packet, hash, publicKey);
    assert.equal(result.length, x.length);
    if (name === 'hard416') assert.deepEqual(result, x.map(n => n > 0 ? 1 : -1));
    else assert.ok(Math.max(...result.map((v, i) => Math.abs(v - x[i]))) <=
      ({ binary4: .067, text4: .067, float13: 1e-7 }[name] ?? .004));
  });
});
test('refuses the wrong arm, signature, manifest, length and signed invalid coefficients', () => {
  const bytes = encode('wsp48', Array(48).fill(0.5), hash, privateKey);
  assert.throws(() => decode('softKD', bytes, hash, publicKey), /context/);
  assert.throws(() => decode('wsp48', bytes, sha('stale'), publicKey), /context/);
  assert.throws(() => decode('wsp48', bytes.slice(1), hash, publicKey), /length/);
  const altered = Buffer.from(bytes); altered.writeInt8(-128, 42);
  assert.throws(() => decode('wsp48', altered, hash, publicKey), /signature/);
  const body = altered.slice(0, 90);
  assert.throws(() => decode('wsp48', Buffer.concat([body, sign(null, body, privateKey)]), hash, publicKey), /coefficient/);
});
