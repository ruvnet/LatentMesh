import assert from 'node:assert/strict';
import { execFileSync, spawnSync } from 'node:child_process';
import { generateKeyPairSync } from 'node:crypto';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { wsp } from '../dependency.mjs';
import { arms, budget, digest, pack, unpack } from './wire.mjs';

const [stage, path] = process.argv.slice(2);
if (!['compile', 'predict'].includes(stage) || !path) throw new Error('usage: node run.mjs compile|predict OUTPUT_DIRECTORY');
const out = resolve(path);
const read = file => JSON.parse(readFileSync(join(out, file), 'utf8'));
const write = (file, value) => writeFileSync(join(out, file), JSON.stringify(value, null, 2) + '\n');
const frozen = read('freeze.json'), freezeHash = digest(readFileSync(join(out, 'freeze.json')));
const model = { schema: 'ruv-fann.dense-layers/1', layers: [{ inputs: frozen.receiverWidth,
  outputs: 1, activation: 'Linear', steepness: 1, weights: Array(frozen.receiverWidth).fill(0), bias: [0] }] };
const opt = { simd: false };
const batch = rows => ({ rows: rows.length, inputs: rows.flat() });
const baseHash = await wsp.fingerprintModel(model);

function senderValues(arm, bank) {
  if (arm === 'phrase') return 'angry furious enraged|annoyed afraid disappointed';
  if (arm === 'shuffled') return bank.permutation.map(i => bank.direction[i]);
  const scores = arm === 'native' ? bank.native : bank.direction;
  return scores.slice(0, arms[arm].count);
}

function cLoopback(program, packet) {
  const proc = spawnSync(program, { input: packet });
  assert.equal(proc.status, 0, proc.stderr?.toString());
  assert.deepEqual(proc.stdout, packet);
  const [frames, wireBytes] = proc.stderr.toString().trim().split(/\s+/).map(Number);
  assert.equal(frames, 1);
  assert.ok(wireBytes > packet.length && wireBytes <= 227);
  return { packet: proc.stdout, frames, wireBytes };
}

async function compile(program) {
  const bank = read('anchors.json');
  assert.deepEqual(bank.manifest, frozen);
  const { privateKey, publicKey } = generateKeyPairSync('ed25519');
  const publicKeyBytes = publicKey.export({ type: 'spki', format: 'der' }).length;
  const candidates = { freezeHash, baseHash, publicKeyBytes, arms: {} }, timings = {};
  for (const [arm, { count }] of Object.entries(arms)) {
    const probes = batch(bank.features.slice(0, count));
    const target = await wsp.traceWeightModel(model, probes, opt);
    const manifest = JSON.stringify([frozen, 'latentmesh.language-packet/1', arm, count, target.baseHash, target.inputHash]);
    const contractHash = digest(manifest);
    const packet = pack(arm, senderValues(arm, bank), contractHash, privateKey);
    const { packet: delivered, frames, wireBytes } = cLoopback(program, packet);
    const receive = async () => {
      const { value, usefulBytes } = unpack(arm, delivered, contractHash, publicKey);
      let scores = value;
      if (arm === 'phrase') {
        // Only the recipient has this cache of its own embeddings of the literal message.
        assert.equal(value, bank.phraseCache.text);
        scores = bank.phraseCache.scores;
      }
      const patch = await wsp.proposeReadoutDelta(model, probes,
        { baseHash: target.baseHash, inputHash: target.inputHash, output: 0, deltas: scores }, opt);
      assert.ok(patch && patch.edits.length <= frozen.receiverWidth + 1);
      return { patch, usefulBytes };
    };
    const { patch, usefulBytes } = await receive();
    const samples = [];
    for (let i = 0; i < 32; i++) {
      const start = performance.now();
      await receive();
      samples.push(performance.now() - start);
    }
    samples.sort((a, b) => a - b);
    timings[arm] = { msP50: samples[15], msP95: samples[30], samples: samples.length };
    candidates.arms[arm] = { patch, patchSha256: digest(JSON.stringify(patch)), contractHash, count, usefulBytes,
      paddingBytes: budget - usefulBytes, applicationBytes: packet.length, wireBytes, frames,
      manifestBytes: Buffer.byteLength(manifest) };
  }
  assert.equal(new Set(Object.values(candidates.arms).map(v => v.wireBytes)).size, 1);
  assert.equal(await wsp.fingerprintModel(model), baseHash);
  write('candidates.json', candidates);
  write('latency-compiler.json', timings);
}

async function predict() {
  const cases = read('heldout.json'), candidates = read('candidates.json');
  assert.equal(cases.freezeSha256, freezeHash);
  assert.equal(candidates.freezeHash, freezeHash);
  assert.equal(candidates.baseHash, baseHash);
  assert.equal(cases.features.length, cases.truth.length + cases.retainedLabels.length);
  const traces = {};
  for (const [arm, { patch, patchSha256 }] of Object.entries(candidates.arms)) {
    assert.equal(digest(JSON.stringify(patch)), patchSha256);
    const candidate = await wsp.applyWeightPatch(model, patch);
    const trace = await wsp.traceWeightModel(candidate, batch(cases.features), opt);
    traces[arm] = trace.layers[0].activations.map(v => v > 0 ? '1' : '0').join('');
  }
  const positive = cases.truth.length;
  const correct = {};
  for (const [arm, bits] of Object.entries({ intact: '0'.repeat(cases.features.length), ...traces })) {
    const primary = cases.truth.map((truth, i) => (bits[i] === '1') === truth);
    const retention = cases.retainedLabels.map((label, i) => bits[positive + i] === '0' && cases.nativeLabels[positive + i] === label);
    const bitmap = values => values.map(v => Number(v)).join('');
    correct[arm] = { primary: bitmap(primary), retention: bitmap(retention) };
  }
  const costs = Object.fromEntries(Object.entries(candidates.arms).map(([key, item]) => {
    const { patch, ...rest } = item;
    return [key, { ...rest, edits: patch.edits.length }];
  }));
  write('predictions.json', { schema: 'latentmesh.language-eval/1', freezeHash,
    candidatesSha256: digest(readFileSync(join(out, 'candidates.json'))), source: cases.sourceMetrics,
    splits: { primary: cases.primaryIdsSha256, retention: cases.retentionIdsSha256 },
    publicKeyBytes: candidates.publicKeyBytes, costs, correct });
}

if (stage === 'compile') {
  const temp = mkdtempSync(join(tmpdir(), 'latentmesh-language-'));
  try {
    const repo = fileURLToPath(new URL('../../../', import.meta.url)), program = join(temp, 'air-loopback');
    execFileSync(process.env.CC || 'cc', ['-std=c11', '-O2', '-Wall', '-Wextra', '-Werror', '-Ic/include',
      'harness/wsp/air-loopback.c', 'c/src/air.c', 'c/src/frame.c', 'c/src/fec.c', '-lm', '-o', program], { cwd: repo });
    await compile(program);
  } finally { rmSync(temp, { recursive: true, force: true }); }
} else await predict();
