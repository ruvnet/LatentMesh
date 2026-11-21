import assert from 'node:assert/strict';
import { execFileSync, spawnSync } from 'node:child_process';
import { createPublicKey, generateKeyPairSync } from 'node:crypto';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { decode, encode, sha } from '../../packet.mjs';

const [mode, folder] = process.argv.slice(2);
if (!['send', 'verify'].includes(mode) || !folder) throw Error('usage: wire.mjs send|verify DIRECTORY');
const read = name => JSON.parse(readFileSync(join(folder, name)));
const write = (name, obj) => writeFileSync(join(folder, name), JSON.stringify(obj) + '\n');
const contractHash = (contract, arm) => sha(JSON.stringify([contract, arm]));
const names = ['transfer', 'shuffled'];
const dir = mkdtempSync(join(tmpdir(), 'residual-air-'));
try {
  const exe = join(dir, 'air'), cwd = fileURLToPath(new URL('../../../../../', import.meta.url));
  execFileSync(process.env.CC || 'cc', ['-std=c11','-O2','-Wall','-Wextra','-Werror','-Ic/include',
    'harness/wsp/air-loopback.c','c/src/air.c','c/src/frame.c','c/src/fec.c','-lm','-o',exe], { cwd });
  if (mode === 'send') {
    const { contract, source } = read('enrollment.json');
    const { publicKey, privateKey } = generateKeyPairSync('ed25519');
    const packets = {};
    for (const arm of names) {
      const bytes = encode('float13', source[arm], contractHash(contract, arm), privateKey);
      packets[arm] = bytes.toString('base64');
    }
    write('packets.json', { contract, publicKey:publicKey.export({type:'spki',format:'pem'}), packets });
  }
  const raw = readFileSync(join(folder,'packets.json'));
  const { contract, publicKey, packets } = JSON.parse(raw);
  const key = createPublicKey(publicKey);
  const decoded = {};
  for (const arm of names) {
    const signed = Buffer.from(packets[arm], 'base64');
    const result = spawnSync(exe, {input:signed});
    assert.equal(result.status, 0, result.stderr.toString());
    const [frames, wireBytes] = result.stderr.toString().trim().split(/\s+/).map(Number);
    assert.deepEqual([signed.length, frames, wireBytes], [154,1,218]);
    assert.deepEqual(signed, result.stdout);
    decoded[arm] = decode('float13', result.stdout, contractHash(contract, arm), key);
    assert.equal(decoded[arm].length * 4, 52);
    assert.throws(() => decode('float13', result.stdout, contractHash(contract, names.find(n => n !== arm)), key));
  }
  if (mode === 'send') write('received.json', decoded);
  else {
    const lock = read('lock.json');
    assert.equal(sha(raw), lock.packetsSha256);
    assert.equal(contract.protocolSha256, lock.protocolSha256);
    assert.equal(contract.anchorIdsSha256, lock.anchorIdsSha256);
    for (const arm of names) assert.deepEqual(decoded[arm], lock.methods[arm].affine);
  }
  console.log('Verified real and shuffled packets: 52 useful bytes, 154 signed bytes, 218 wire bytes each');
} finally {
  rmSync(dir, {recursive:true, force:true});
}
