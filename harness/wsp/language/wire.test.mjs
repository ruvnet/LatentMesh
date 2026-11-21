import assert from 'node:assert/strict';
import test from 'node:test';
import { generateKeyPairSync, sign } from 'node:crypto';
import { arms, pack, unpack } from './wire.mjs';

const { privateKey, publicKey } = generateKeyPairSync('ed25519');
const contract = '3a'.repeat(32), phrase = 'angry furious enraged|annoyed afraid disappointed';
function values(arm) {
  return arm === 'phrase' ? phrase : Array.from({ length: arms[arm].count }, (_, i) => Math.cos(i / 7));
}

test('all controls share the signed budget and recover their specified number of observations', () => {
  for (const arm of Object.keys(arms)) {
    const raw = values(arm), packet = pack(arm, raw, contract, privateKey);
    assert.equal(packet.length, 154);
    const { value, usefulBytes } = unpack(arm, packet, contract, publicKey);
    assert.ok(usefulBytes > 0 && usefulBytes <= 52);
    if (arm === 'phrase') { assert.equal(value, phrase); continue; }
    assert.equal(value.length, raw.length);
    for (let i = 0; i < raw.length; i++) {
      if (arm.startsWith('hard')) assert.equal(value[i], raw[i] > 0 ? 1 : -1);
      else assert.ok(Math.abs(value[i] - raw[i]) < ({ decimal52: 1 / 9 + 1e-10, text4: 1 / 15 + 1e-10 }[arm] ?? 0.004));
    }
  }
});

test('rejects a changed frame, verification key, contract, arm and signed noncanonical padding', () => {
  const packet = pack('hard48', values('hard48'), contract, privateKey);
  assert.throws(() => unpack('hard48', packet, 'bb'.repeat(32), publicKey), /contract/);
  assert.throws(() => unpack('hard64', packet, contract, publicKey), /arm/);
  assert.throws(() => unpack('hard48', packet, contract, generateKeyPairSync('ed25519').publicKey), /signature/);
  const body = Buffer.from(packet.subarray(0, -64)); body[89] = 1;
  assert.throws(() => unpack('hard48', Buffer.concat([body, packet.subarray(-64)]), contract, publicKey), /signature/);
  assert.throws(() => unpack('hard48', Buffer.concat([body, sign(null, body, privateKey)]), contract, publicKey), /padding/);
});

test('zero and invalid inputs are handled explicitly', () => {
  const zeros = Array(48).fill(0);
  assert.deepEqual(unpack('hybrid', pack('hybrid', zeros, contract, privateKey), contract, publicKey).value, zeros);
  assert.throws(() => pack('hybrid', [NaN, ...zeros.slice(1)], contract, privateKey), /scores/);
  assert.throws(() => pack('phrase', 'anger only', contract, privateKey), /invalid phrase/);
  assert.throws(() => pack('unknown', zeros, contract, privateKey), /unknown/);
});
