import assert from 'node:assert/strict';
import { execFileSync, spawnSync } from 'node:child_process';
import { generateKeyPairSync } from 'node:crypto';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { fannRoot, wsp } from './dependency.mjs';
import { encodeTransfer, encodeHiddenTransfer, fingerprintDirection, decodeTransfer, receiveTransfer } from './transfer.mjs';

const repo = fileURLToPath(new URL('../../', import.meta.url));
const helpers = await import(pathToFileURL(join(fannRoot, 'scripts/wsp-benchmark-lib.mjs')));
const { SEEDS, asDataset, digestJSON, loadDatasets, metrics, rng, standardizer, stratifiedSplit } = helpers;
const runtimeUrl = execFileSync(process.execPath, ['--input-type=module', '-e',
  "process.stdout.write(import.meta.resolve('ruv-fann'))"], { cwd: join(fannRoot, 'packages/wsp'), encoding: 'utf8' });
const runtime = await (await import(runtimeUrl)).loadRuntime({ simd: false });
const { privateKey, publicKey } = generateKeyPairSync('ed25519');
const selected = 2;

function train(rows, hidden, seed) {
  const random = rng(seed);
  const widths = [rows[0].features.length, ...hidden, 3];
  const layers = widths.slice(1).map((outputs, i) => {
    const inputs = widths[i], radius = Math.sqrt(6 / (inputs + outputs));
    return { inputs, outputs, weights: Array.from({ length: inputs * outputs }, () => radius * (2 * random() - 1)),
      bias: Array(outputs).fill(0), activation: i < hidden.length ? 'Tanh' : 'Linear', steepness: 1 };
  });
  const data = asDataset(rows);
  return runtime.train({ schema: 'ruv-fann.dense-layers/1', layers }, { inputs: Float64Array.from(data.inputs),
    targets: Float64Array.from(data.targets), rows: data.rows, epochs: 120, optimizer: 'adamw', learningRate: 0.01 }).model;
}

function blankReadout(model) {
  const copy = structuredClone(model), head = copy.layers.at(-1);
  head.weights.fill(0, selected * head.inputs, (selected + 1) * head.inputs);
  head.bias[selected] = 0;
  return copy;
}

const batch = rows => ({ inputs: rows.flatMap(row => row.features), rows: rows.length });
async function trace(model, rows) { return wsp.traceWeightModel(model, batch(rows), { simd: false }); }
const outputs = trace => trace.layers.at(-1).activations;
const selectedScores = trace => outputs(trace).filter((_, index) => index % 3 === selected);

function runAir(program, packet) {
  const result = spawnSync(program, { input: packet, maxBuffer: 16384 });
  assert.equal(result.status, 0, result.stderr?.toString());
  assert.deepEqual(result.stdout, packet, 'Air must deliver the signed bytes');
  const [frames, wireBytes] = result.stderr.toString().trim().split(/\s+/).map(Number);
  assert.ok(Number.isSafeInteger(frames) && Number.isSafeInteger(wireBytes) && frames > 0 && wireBytes > packet.length);
  return { packet: result.stdout, cost: { applicationBytes: packet.length, frames, airBytes: wireBytes } };
}

function splitFeaturesOnce(split) {
  // Iris has repeated measurement vectors under different row IDs. Keep the
  // earliest partition's copy so no exact probe or score feature enters fitting.
  const seen = new Set();
  return Object.fromEntries(['train', 'validation', 'test'].map(name => [name, split[name].filter(row => {
    const key = JSON.stringify(row.features);
    if (seen.has(key)) return false;
    seen.add(key); return true;
  })]));
}

function shuffle(values, random) {
  const shuffled = values.slice();
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled;
}

function evaluate(rows, scores, baseScores) {
  const labels = rows.map(row => row.label);
  const { accuracy, macroF1, mse, confusion } = metrics(labels, scores);
  const oldRows = rows.filter(row => row.label !== selected).length;
  const oldCorrect = confusion.slice(0, selected).reduce((sum, row, i) => sum + row[i], 0);
  const retainedScoreMaxDrift = Math.max(...scores.map((value, i) => i % 3 === selected ? 0 : Math.abs(value - baseScores[i])));
  return { accuracy, macroF1, mse, oldCorrect, oldRows, retainedScoreMaxDrift, confusion };
}

async function runCase(provenance, rawRows, seed, program) {
  const original = stratifiedSplit(rawRows, seed), split = splitFeaturesOnce(original);
  const scaler = standardizer(split.train);
  const trainRows = scaler.transform(split.train), anchorRows = scaler.transform(split.validation);
  const testRows = scaler.transform(split.test), probes = batch(anchorRows);
  const teacher = train(trainRows, [8], seed), teacherBase = blankReadout(teacher);
  const intact = train(trainRows, [7, 5], seed ^ 0x3372), receiver = blankReadout(intact);
  const [before, after, target] = await Promise.all([teacherBase, teacher, receiver].map(m => trace(m, anchorRows)));
  const contract = { schema: 'latentmesh.readout-contract/1', skill: provenance.id + '/class-2-membership',
    sourceBaseHash: before.baseHash, sourceEditedHash: after.baseHash, targetBaseHash: target.baseHash,
    inputHash: target.inputHash, sourceOutput: selected, targetOutput: selected };
  const packets = {}, arms = {}, patches = {};
  for (const encoding of ['q8', 'teacherBit']) {
    const signed = await encodeTransfer(teacherBase, teacher, probes, contract, privateKey, encoding);
    const wire = runAir(program, signed);
    patches[encoding] = await receiveTransfer(wire.packet, receiver, probes, contract, publicKey, encoding);
    packets[encoding] = { ...wire.cost, values: decodeTransfer(wire.packet, contract, publicKey, encoding).values };
  }
  // With a zeroed final row, its restoration equals an affine projection of
  // the sender hidden state. Check the actual hidden-code path against the
  // registered output-delta arm without selecting or fitting a new direction.
  const head = teacher.layers.at(-1);
  const direction = { layer: teacher.layers.length - 2, weights: head.weights.slice(selected * head.inputs, (selected + 1) * head.inputs),
    bias: head.bias[selected] };
  const hiddenContract = { schema: 'latentmesh.hidden-contract/1', skill: contract.skill, sourceHash: before.baseHash,
    directionHash: fingerprintDirection(direction), inputHash: contract.inputHash, targetBaseHash: target.baseHash, targetOutput: selected };
  const hiddenWire = runAir(program, await encodeHiddenTransfer(teacherBase, probes, direction, hiddenContract, privateKey));
  const hiddenPatch = await receiveTransfer(hiddenWire.packet, receiver, probes, hiddenContract, publicKey, 'hiddenQ8');
  assert.deepEqual(hiddenPatch, patches.q8, 'the hidden projection must reconstruct the same target patch');
  assert.deepEqual(hiddenWire.cost, { applicationBytes: packets.q8.applicationBytes, frames: packets.q8.frames, airBytes: packets.q8.airBytes });
  const q8 = packets.q8.values, previous = selectedScores(before);
  const raw = selectedScores(after).map((v, i) => v - previous[i]);
  const random = rng(seed ^ 0xa017), noise = q8.map(() => 2 * random() - 1);
  const normScale = Math.hypot(...q8) / Math.hypot(...noise);
  for (const [name, deltas] of Object.entries({ float: raw, shuffled: shuffle(q8, rng(seed ^ 0x522b)),
    random: noise.map(value => value * normScale), zero: q8.map(() => 0) })) {
    patches[name] = await wsp.proposeReadoutDelta(receiver, probes, { baseHash: target.baseHash,
      inputHash: target.inputHash, output: selected, deltas }, { simd: false });
  }
  assert.equal(patches.zero, null);
  const baseScores = outputs(await trace(receiver, testRows));
  arms.intact = evaluate(testRows, outputs(await trace(intact, testRows)), baseScores);
  for (const [name, patch] of Object.entries(patches)) {
    const model = patch ? await wsp.applyWeightPatch(receiver, patch) : receiver;
    const scores = outputs(await trace(model, testRows));
    arms[name] = evaluate(testRows, scores, baseScores);
    assert.equal(arms[name].retainedScoreMaxDrift, 0);
  }
  assert.equal(target.baseHash, await wsp.fingerprintModel(receiver));
  const costs = Object.fromEntries(Object.entries(packets).map(([name, { values, ...cost }]) => [name, cost]));
  const ids = Object.fromEntries(Object.entries(split).map(([key, rows]) => [key, rows.map(row => row.id)]));
  return { dataset: provenance.id, seed, datasetSha256: provenance.sha256, originalSplitHash: original.hash,
    uniqueSplitHash: digestJSON(ids), rows: { train: trainRows.length, probes: probes.rows, test: testRows.length },
    contract, hiddenDirectionHash: hiddenContract.directionHash, hiddenPatchEqualsReadoutPatch: true,
    edits: patches.q8.edits.length, candidateHash: await wsp.fingerprintModel(await wsp.applyWeightPatch(receiver, patches.q8)),
    maxQuantizationError: Math.max(...q8.map((v, i) => Math.abs(v - raw[i]))), costs, arms };
}

const temp = mkdtempSync(join(tmpdir(), 'latentmesh-wsp-'));
try {
  const program = join(temp, 'air-loopback');
  execFileSync(process.env.CC || 'cc', ['-std=c11', '-O2', '-Wall', '-Wextra', '-Werror', '-Ic/include',
    'harness/wsp/air-loopback.c', 'c/src/air.c', 'c/src/frame.c', 'c/src/fec.c', '-lm', '-o', program], { cwd: repo });
  const { datasets } = await loadDatasets(), runs = [];
  for (const { provenance, rows } of datasets) for (const seed of SEEDS) runs.push(await runCase(provenance, rows, seed, program));
  console.log(JSON.stringify({ schema: 'latentmesh.readout-repair-benchmark/1',
    evidence: 'controlled ablation on UCI data; portable C loopback; scalar WASM',
    fannRevision: execFileSync('git', ['rev-parse', 'HEAD'], { cwd: fannRoot, encoding: 'utf8' }).trim(), runs }, null, 2));
} finally { rmSync(temp, { recursive: true, force: true }); }
