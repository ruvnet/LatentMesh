import { wsp } from '../../dependency.mjs';

export const base = { schema: 'ruv-fann.dense-layers/1', layers: [
  { inputs: 2, outputs: 1, weights: [0, 0], bias: [0], activation: 'Linear', steepness: 1 }
] };

export async function fitLinear(points, targets) {
  const batch = { rows: points.length, inputs: points.flat() };
  const cfg = { simd: false };
  const trace = await wsp.traceWeightModel(base, batch, cfg);
  const patch = await wsp.proposeReadoutDelta(base, batch, {
    baseHash: trace.baseHash, inputHash: trace.inputHash, output: 0, deltas: targets
  }, cfg);
  const fitted = patch ? await wsp.applyWeightPatch(base, patch) : base;
  return fitted.layers[0];
}

export async function modelHash(coefficients) {
  const [a, b, c] = coefficients;
  const edits = [{ layer: 0, parameter: 'weights', index: 0, value: a }, { layer: 0, parameter: 'weights', index: 1, value: b },
    { layer: 0, parameter: 'bias', index: 0, value: c }].filter(e => e.value !== 0);
  const baseHash = await wsp.fingerprintModel(base);
  const fitted = edits.length ? await wsp.applyWeightPatch(base, { schema: 'ruv-fann.weight-patch/1', baseHash, edits }) : base;
  return wsp.fingerprintModel(fitted);
}
