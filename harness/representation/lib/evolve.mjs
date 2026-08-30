// A deterministic, seeded evolve loop over the ADR-046 §3 genome, in
// harness/evolve's dependency-free spirit and harness/air's
// root-policy/proposer/promotion-rule shape.
//
// Determinism is the point, not a nicety: the champion is evidence, so the
// same seed and the same committed receipts must reproduce the same champion
// on any machine, forever. The PRNG is an explicit 32-bit stream, never
// Math.random.

import {
  ALPHAS,
  ADMISSIONS,
  CHANNELS,
  CONTROL_SETS,
  COST_WEIGHT_SETS,
  ROOT_POLICY,
  enumerateGenomes,
  evaluate,
  genomeKey,
  promotionRule,
} from "./genome.mjs";

export const DEFAULT_SEED = 0x1ea7be57;

/** mulberry32 — small, exactly specified, integer-seeded. */
export function makeRng(seed) {
  let state = seed >>> 0;
  return function next() {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const GENE_DOMAINS = Object.freeze({
  channel: CHANNELS,
  admission: ADMISSIONS,
  controls: CONTROL_SETS,
  alpha: ALPHAS,
  costWeights: COST_WEIGHT_SETS.map((entry) => entry.name),
});
const GENE_NAMES = Object.freeze(Object.keys(GENE_DOMAINS));

/** Mutate exactly one gene to a different value in its domain. */
export function mutate(genome, rng) {
  const gene = GENE_NAMES[Math.floor(rng() * GENE_NAMES.length)];
  const domain = GENE_DOMAINS[gene];
  const currentIndex = domain.findIndex((value) =>
    gene === "controls" ? value.join("+") === genome.controls.join("+") : value === genome[gene],
  );
  const step = 1 + Math.floor(rng() * (domain.length - 1));
  const next = domain[(Math.max(currentIndex, 0) + step) % domain.length];
  return Object.freeze({ ...genome, [gene]: next });
}

/**
 * Hill-climb with restarts, promoting only through the frozen promotion rule.
 * Returns the champion plus a per-generation trace (which becomes the
 * hash-chained receipt's body).
 */
export function runEvolve({ corpus, scale, seed = DEFAULT_SEED, generations = 200, restartEvery = 25 }) {
  const rng = makeRng(seed);
  const cache = new Map();
  const scoreOf = (genome) => {
    const key = genomeKey(genome);
    if (!cache.has(key)) cache.set(key, evaluate(corpus, genome, scale));
    return cache.get(key);
  };

  let incumbent = scoreOf(ROOT_POLICY);
  let champion = incumbent;
  const trace = [];

  for (let generation = 1; generation <= generations; generation += 1) {
    const proposal = mutate(incumbent.genome, rng);
    const evaluated = scoreOf(proposal);
    const verdict = promotionRule(evaluated, incumbent, corpus);
    if (verdict.promote) {
      incumbent = evaluated;
      const championVerdict = promotionRule(evaluated, champion, corpus);
      if (championVerdict.promote) champion = evaluated;
    }
    trace.push({
      generation,
      proposed: genomeKey(proposal),
      selected: evaluated.selected,
      fitness: evaluated.fitness,
      promoted: verdict.promote,
      reasons: verdict.reasons,
      incumbent: genomeKey(incumbent.genome),
      incumbentFitness: incumbent.fitness,
    });
    if (generation % restartEvery === 0) {
      // Restart from a seeded random point so the climb is not trapped by the
      // root policy's neighbourhood. The champion is never reset.
      const restart = Object.freeze({
        channel: CHANNELS[Math.floor(rng() * CHANNELS.length)],
        admission: ADMISSIONS[Math.floor(rng() * ADMISSIONS.length)],
        controls: CONTROL_SETS[Math.floor(rng() * CONTROL_SETS.length)],
        alpha: ALPHAS[Math.floor(rng() * ALPHAS.length)],
        costWeights: COST_WEIGHT_SETS[Math.floor(rng() * COST_WEIGHT_SETS.length)].name,
      });
      incumbent = scoreOf(restart);
    }
  }

  return { champion, trace, evaluations: cache.size, seed, generations };
}

/**
 * Exhaustive ground truth over the whole (small) lattice, under the same
 * frozen promotion rule. The optimizer asserts that the seeded search found
 * this — a search that merely CONVERGED could still be quietly mis-specified;
 * a search that provably matches the exhaustive optimum cannot be.
 *
 * Ties are broken by the canonical enumeration order of `enumerateGenomes`,
 * which is fixed, so the answer is unique.
 */
export function exhaustiveChampion(corpus, scale) {
  let best = null;
  for (const genome of enumerateGenomes()) {
    const evaluated = evaluate(corpus, genome, scale);
    if (promotionRule(evaluated, best, corpus).promote) best = evaluated;
  }
  return best;
}

/**
 * The same exhaustive search with clause 2 of the promotion rule (no
 * weakening of the control set) REMOVED. Reported alongside the champion so
 * the constraint's effect is visible rather than assumed. If this differs
 * from the champion, the difference IS the exploit the clause closes.
 */
export function exhaustiveChampionWithoutControlFloor(corpus, scale) {
  let best = null;
  for (const genome of enumerateGenomes()) {
    const evaluated = evaluate(corpus, genome, scale);
    if (genome.alpha > 0.05) continue;
    if (evaluated.selected !== "None") {
      const admission = evaluated.admissions.find((entry) => entry.id === evaluated.selectedId);
      if (!admission || admission.status !== "admitted") continue;
    }
    if (!best || evaluated.fitness > best.fitness) best = evaluated;
  }
  return best;
}
