// Coefficients belong to one pinned sender readout, receiver and operating point.
export function margin(coefficients, score, confidence) {
  if (coefficients.length !== 3 || !coefficients.every(Number.isFinite) ||
      !Number.isFinite(score) || !Number.isFinite(confidence) || confidence < 0 || confidence > 1) {
    throw new RangeError('expected three finite coefficients, a finite readout and a probability');
  }
  return coefficients[0] * score + coefficients[1] * confidence + coefficients[2];
}

// Consume the separate anger tag or opt into the exclusive four-way route.
export function readReceiver(coefficients, score, logits) {
  if (logits.length !== 3 || !logits.every(Number.isFinite)) throw new RangeError('expected three finite sentiment logits');
  const maximum = Math.max(...logits);
  const native = logits.indexOf(maximum);
  const confidence = 1 / logits.reduce((sum, value) => sum + Math.exp(value - maximum), 0);
  const evidence = margin(coefficients, score, confidence);
  return { sentiment: native, confidence, evidence, anger: evidence > 0, route: evidence > 0 ? 3 : native };
}
