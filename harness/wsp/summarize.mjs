import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const input = process.argv[2] || new URL('./receipt.json', import.meta.url);
const evidence = JSON.parse(readFileSync(input, 'utf8'));
assert.equal(evidence.schema, 'latentmesh.readout-repair-benchmark/1');
const mean = values => values.reduce((sum, value) => sum + value, 0) / values.length;
const datasets = {};
assert.equal(evidence.runs.length, 10);
for (const dataset of ['iris', 'wine']) {
  const runs = evidence.runs.filter(r => r.dataset === dataset);
  assert.deepEqual(runs.map(r => r.seed), [17, 43, 101, 211, 509]);
  const arms = {};
  for (const key of ['zero', 'intact', 'shuffled', 'random', 'float', 'teacherBit', 'q8']) {
    const values = runs.map(r => r.arms[key]);
    for (const [i, a] of values.entries()) {
      const count = a.confusion.flat().reduce((sum, v) => sum + v, 0);
      const correct = a.confusion.reduce((sum, row, label) => sum + row[label], 0);
      assert.equal(count, runs[i].rows.test); assert.equal(a.accuracy, correct / count);
      assert.equal(a.oldCorrect, a.confusion[0][0] + a.confusion[1][1]);
      assert.equal(a.oldRows, a.confusion.slice(0, 2).flat().reduce((sum, v) => sum + v, 0));
    }
    arms[key] = { accuracy: mean(values.map(a => a.accuracy)), retention: mean(values.map(a => a.oldCorrect / a.oldRows)),
      macroF1: mean(values.map(a => a.macroF1)), mse: mean(values.map(a => a.mse)) };
  }
  const q8AdvantagePoints = 100 * (arms.q8.accuracy - arms.teacherBit.accuracy);
  const retentionChangePoints = 100 * (arms.q8.retention - arms.intact.retention);
  const maxSiblingDrift = Math.max(...runs.map(r => r.arms.q8.retainedScoreMaxDrift));
  const hiddenEquivalence = runs.every(r => r.hiddenPatchEqualsReadoutPatch);
  assert.ok(hiddenEquivalence);
  const range = selector => { const values = runs.map(selector); return [Math.min(...values), Math.max(...values)]; };
  datasets[dataset] = { arms, q8AdvantagePoints, retentionChangePoints, maxSiblingDrift, hiddenEquivalence,
    q8AirBytes: range(r => r.costs.q8.airBytes), bitAirBytes: range(r => r.costs.teacherBit.airBytes),
    edits: range(r => r.edits), frames: range(r => r.costs.q8.frames),
    meetsGate: q8AdvantagePoints >= 2 - 1e-10 && retentionChangePoints >= -1 - 1e-10 && maxSiblingDrift === 0 };
}
console.log(JSON.stringify({ datasets, warrantsLargerTrial: Object.values(datasets).every(d => d.meetsGate) }, null, 2));
