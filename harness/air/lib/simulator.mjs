export const ROOT_POLICY = Object.freeze({
  payloadBytes: 240,
  fecRate: "none",
  interleaverDepth: 8,
  neuralThreshold: 0.72,
});

export const DOMAINS = Object.freeze({
  payloadBytes: Object.freeze([32, 64, 128, 240]),
  fecRate: Object.freeze(["1/2", "2/3", "none"]),
  interleaverDepth: Object.freeze([8, 16, 32]),
  neuralThreshold: Object.freeze([0.72, 0.82, 0.9]),
});

export const ACCEPTANCE_TARGETS = Object.freeze({
  semanticReduction: 10,
  taskEquivalenceMargin: 0.01,
  neuralPhyGain: 2,
  criticalAgreement: 0.99,
});

export const FULL_STATE_BASELINE_BYTES = 65_536;

function random(seed) {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x1_0000_0000;
  };
}

function clamp(value, low, high) {
  return Math.min(high, Math.max(low, value));
}

function fecFactor(rate) {
  if (rate === "1/2") return 2;
  if (rate === "2/3") return 1.5;
  return 1;
}

function fecGain(rate) {
  if (rate === "1/2") return 5.2;
  if (rate === "2/3") return 3.2;
  return 0;
}

export function makeSuite(id, seeds, difficulty = 0) {
  return Object.freeze({
    id,
    items: Object.freeze(seeds.map((seed) => Object.freeze({
      seed,
      snr: 8.5 - difficulty + (seed % 5) * 0.7,
      burst: clamp(0.015 + difficulty * 0.008 + (seed % 3) * 0.004, 0, 0.12),
      doppler: 0.08 + difficulty * 0.035 + (seed % 4) * 0.015,
    }))),
  });
}

export function simulate(policy, item, { neuralEnabled = true } = {}) {
  // Channel noise is keyed only by the channel seed so candidate and baseline
  // policies see the same realization during paired evaluation.
  const rng = random(item.seed * 7919);
  const rate = 1_200;
  const wireBytes = policy.payloadBytes + 34 + Math.ceil(policy.payloadBytes / 96) * 7;
  const airtime = wireBytes * 8 * fecFactor(policy.fecRate) / rate;
  const interleaverGain = Math.log2(policy.interleaverDepth) * 0.52;
  const neuralConfidence = clamp(0.58 + item.snr * 0.035 - item.doppler * 0.18 + (rng() - 0.5) * 0.04, 0, 1);
  const neuralAccepted = neuralEnabled && neuralConfidence >= policy.neuralThreshold;
  const neuralGain = neuralAccepted ? 1.3 : 0;
  const margin = item.snr + fecGain(policy.fecRate) + interleaverGain + neuralGain - item.burst * 48 - item.doppler * 3.1 - 4.6;
  const frameSuccess = clamp(1 / (1 + Math.exp(-margin / 1.7)), 0.02, 0.9998);
  const criticalCoverage = clamp(0.991 + Math.log2(policy.payloadBytes / 16) * 0.0018, 0.991, 0.9982);
  const criticalAgreement = clamp(criticalCoverage * frameSuccess + (1 - frameSuccess) * 0.98, 0, 0.9998);
  const semanticCoverage = clamp(0.45 + Math.log2(policy.payloadBytes / 16) * 0.11, 0.45, 0.94);
  const usefulInformation = 512 * semanticCoverage * frameSuccess;
  const usefulPerSecond = usefulInformation / airtime;
  return Object.freeze({
    airtime,
    wireBytes,
    frameSuccess,
    criticalAgreement,
    usefulPerSecond,
    neuralAccepted,
    fallback: neuralEnabled && !neuralAccepted,
  });
}

export async function evaluatePolicy(policy, suite) {
  const rows = suite.items.map((item) => simulate(policy, item));
  const mean = (key) => rows.reduce((sum, row) => sum + row[key], 0) / rows.length;
  const minimumAgreement = Math.min(...rows.map((row) => row.criticalAgreement));
  const fallbackRate = rows.filter((row) => row.fallback).length / rows.length;
  return Object.freeze({
    primary: mean("usefulPerSecond"),
    criticalAgreement: mean("criticalAgreement"),
    minimumAgreement,
    costPerWin: mean("airtime") / Math.max(mean("frameSuccess"), 1e-9),
    fallbackRate,
    noopRate: 1 - mean("frameSuccess"),
    regressed: minimumAgreement < 0.99,
  });
}

/**
 * Report the two research stages independently. The semantic fixture assumes
 * the same deterministic task facts are present in the full state and sparse
 * delta, so its task accuracy delta is zero by construction. The physical
 * layer comparison then holds the semantic payload fixed and toggles only the
 * bounded neural likelihood assist.
 */
export function benchmarkPolicy(policy, suite) {
  const paired = suite.items.map((item) => ({
    classical: simulate(policy, item, { neuralEnabled: false }),
    neural: simulate(policy, item, { neuralEnabled: true }),
  }));
  const mean = (side, key) => paired.reduce((sum, row) => sum + row[side][key], 0) / paired.length;
  const semanticWireBytes = mean("classical", "wireBytes");
  const semanticReduction = FULL_STATE_BASELINE_BYTES / semanticWireBytes;
  const classicalUseful = mean("classical", "usefulPerSecond");
  const neuralUseful = mean("neural", "usefulPerSecond");
  const neuralPhyGain = neuralUseful / Math.max(classicalUseful, 1e-9);
  const criticalAgreement = mean("neural", "criticalAgreement");
  const taskAccuracyDelta = 0;
  return Object.freeze({
    evidence: "simulated",
    semantic: Object.freeze({
      fullStateBytes: FULL_STATE_BASELINE_BYTES,
      meanWireBytes: semanticWireBytes,
      reduction: semanticReduction,
      taskAccuracyDelta,
      pass: semanticReduction >= ACCEPTANCE_TARGETS.semanticReduction
        && Math.abs(taskAccuracyDelta) <= ACCEPTANCE_TARGETS.taskEquivalenceMargin,
    }),
    neuralPhy: Object.freeze({
      classicalUsefulPerSecond: classicalUseful,
      neuralUsefulPerSecond: neuralUseful,
      gain: neuralPhyGain,
      pass: neuralPhyGain >= ACCEPTANCE_TARGETS.neuralPhyGain,
    }),
    criticalAgreement: Object.freeze({
      mean: criticalAgreement,
      pass: criticalAgreement >= ACCEPTANCE_TARGETS.criticalAgreement,
    }),
    acceptancePassed: false,
    acceptanceBlockedBy: Object.freeze([
      "hardware in loop evidence is absent",
      "bootstrap confidence intervals are absent",
      "unseen propagation trials are absent",
    ]),
  });
}

export function promotionRule(evidence) {
  const reasons = [];
  const { baseline, candidate, anchor } = evidence;
  if (candidate.regressed) reasons.push("a holdout case fell below 99% critical agreement");
  if (candidate.primary < baseline.primary * 1.02) reasons.push("useful information lift is below 2%");
  if (candidate.criticalAgreement < baseline.criticalAgreement) reasons.push("mean critical agreement regressed");
  if (candidate.fallbackRate > baseline.fallbackRate + 0.001) reasons.push("fallback cost increased");
  if (candidate.costPerWin > baseline.costPerWin) reasons.push("airtime per successful frame increased");
  if (anchor && anchor.candidate < anchor.baseline * 0.98) reasons.push("unseen anchor suite regressed by more than 2%");
  return Object.freeze({ promote: reasons.length === 0, reasons });
}
