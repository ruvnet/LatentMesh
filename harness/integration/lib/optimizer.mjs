// ADR-022's optimization loop: reuses latentmesh-evolve's deterministic,
// seeded-RNG Darwin loop pattern (ADR-018) — population, generations,
// elitist selection, a seeded PRNG (same LCG style as harness/air's own
// lib/simulator.mjs) — but scoped to adapter parameters instead of
// topology: MTU packing (the fragmentation threshold, bounded above by the
// Meshtastic packet ceiling) and the bridge batching interval.
//
// Fitness is evaluated against the e2e suites' own receipts, not against
// re-hardcoded protocol constants: `usableBytesPerPacket` and
// `multiFragmentMessageBytes` below are read out of
// `artifacts/meshtastic-receipt.json` by scripts/optimize.mjs, not
// duplicated here — so if MESHTASTIC_FRAME_MTU or the fragment overhead
// ever changes, this optimizer's constraint tracks the receipt rather than
// silently drifting from it.
//
// Everything below the search space is a documented, illustrative
// simulated cost model (dimensionless "cost units"), not a measured
// real-world timing or energy claim — consistent with this repository's
// evidence-label discipline (ADR-014/ADR-018/ADR-022). The bridge crate has
// no batching scheduler today (every decoded delta dispatches immediately),
// so the batching-interval side of this model is necessarily a proxy, not
// an exercised code path.

export const SEARCH_SPACE = Object.freeze({
  fragmentationThresholdBytes: Object.freeze([140, 160, 180, 200, 217]),
  bridgeBatchingIntervalMs: Object.freeze([0, 100, 250, 500, 1000, 2000]),
});

/** Simulated per-fragment Meshtastic-packet dispatch cost (dimensionless). */
export const FRAGMENT_UNIT_COST = 10;
/**
 * A deliberately tiny tie-breaker: several thresholds can tie on fragment
 * count for a given message size (e.g. 160B and 217B both need 2 fragments
 * for a 300B message), and among ties the larger threshold is strictly
 * preferable (more packing headroom for a larger future delta, at zero
 * extra fragment cost today). Scaled so it can never outweigh a genuine
 * one-fragment difference (`FRAGMENT_UNIT_COST`): the largest possible
 * headroom bonus (threshold 217) is `217 * 0.01 = 2.17`, well under 10.
 */
const HEADROOM_TIEBREAK_WEIGHT = 0.01;
/** Simulated fixed per-bridge-call (MCP post_message/ReplicateMessage) overhead. */
export const BRIDGE_OVERHEAD_UNITS = 25;
/** Simulated assumption: decoded deltas arrive roughly this often. */
export const DELTA_ARRIVAL_INTERVAL_MS = 150;
/** Weight converting simulated average batch-wait milliseconds into the same cost units as the overhead term. */
export const LATENCY_WEIGHT_PER_MS = 0.05;

function random(seed) {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x1_0000_0000;
  };
}

function pick(rng, domain) {
  return domain[Math.floor(rng() * domain.length) % domain.length];
}

function deltasPerBatchWindow(intervalMs) {
  return 1 + Math.floor(intervalMs / DELTA_ARRIVAL_INTERVAL_MS);
}

/**
 * Evaluates one candidate against `measurements` (the meshtastic e2e
 * receipt's own measured `usable_bytes_per_packet` and
 * `multi_fragment.message_bytes`). Returns `{ valid: false }` for a
 * candidate whose fragmentation threshold exceeds the measured Meshtastic
 * packet ceiling — an invalid candidate, rejected before scoring, mirroring
 * ADR-018's "a mutation that would raise authority above its cap is
 * rejected before evaluation."
 */
export function evaluateCandidate(candidate, measurements) {
  const { fragmentationThresholdBytes, bridgeBatchingIntervalMs } = candidate;
  if (fragmentationThresholdBytes > measurements.usableBytesPerPacket) {
    return { valid: false };
  }
  const fragments = Math.ceil(measurements.multiFragmentMessageBytes / fragmentationThresholdBytes);
  const packingCost =
    fragments * FRAGMENT_UNIT_COST - fragmentationThresholdBytes * HEADROOM_TIEBREAK_WEIGHT;
  const deltasPerWindow = deltasPerBatchWindow(bridgeBatchingIntervalMs);
  const bridgeOverheadPerDelta = BRIDGE_OVERHEAD_UNITS / deltasPerWindow;
  const batchWaitMs = bridgeBatchingIntervalMs / 2;
  const latencyCost = batchWaitMs * LATENCY_WEIGHT_PER_MS + bridgeOverheadPerDelta;
  const totalCost = packingCost + latencyCost;
  return {
    valid: true,
    fragments,
    packingCost,
    deltasPerWindow,
    bridgeOverheadPerDelta,
    batchWaitMs,
    latencyCost,
    totalCost,
    score: -totalCost,
  };
}

function mutate(base, rng, space) {
  // Mutate exactly one of the two parameters to a (possibly the same)
  // neighboring domain value — keeps the search local, generation over
  // generation, rather than resampling uniformly at random every time.
  const mutateThreshold = rng() < 0.5;
  if (mutateThreshold) {
    return {
      fragmentationThresholdBytes: pick(rng, space.fragmentationThresholdBytes),
      bridgeBatchingIntervalMs: base.bridgeBatchingIntervalMs,
    };
  }
  return {
    fragmentationThresholdBytes: base.fragmentationThresholdBytes,
    bridgeBatchingIntervalMs: pick(rng, space.bridgeBatchingIntervalMs),
  };
}

/**
 * The baseline this repository's implementation actually uses today:
 * `fragmentationThresholdBytes` at the measured Meshtastic packet ceiling
 * (what `fragment_message` already packs to — `MESHTASTIC_FRAME_MTU -
 * FRAME_MIN_BYTES`) and `bridgeBatchingIntervalMs = 0` (the bridge crate
 * has no batching scheduler — every decoded delta dispatches immediately).
 */
export function baselineCandidate(measurements) {
  return {
    fragmentationThresholdBytes: measurements.usableBytesPerPacket,
    bridgeBatchingIntervalMs: 0,
  };
}

/**
 * Runs a small deterministic, seeded-RNG generational search (ADR-018's
 * pattern, scoped to this 2-parameter adapter search space) and returns
 * `{ trajectory, bestEver, mutationsProposed, mutationsRejected }`.
 * `trajectory` has exactly `generations` entries, each
 * `{ generation, best_score, best_params }`.
 */
export function runDarwinSearch({ seed, generations, populationSize, eliteSize, measurements }) {
  const rng = random(seed);
  const space = SEARCH_SPACE;
  let mutationsProposed = 0;
  let mutationsRejected = 0;

  const randomCandidate = () => ({
    fragmentationThresholdBytes: pick(rng, space.fragmentationThresholdBytes),
    bridgeBatchingIntervalMs: pick(rng, space.bridgeBatchingIntervalMs),
  });

  let population = Array.from({ length: populationSize }, randomCandidate);
  const trajectory = [];
  let bestEver = null;

  for (let generation = 0; generation < generations; generation += 1) {
    const evaluated = population
      .map((candidate) => ({ candidate, result: evaluateCandidate(candidate, measurements) }))
      .filter((entry) => entry.result.valid);
    evaluated.sort((a, b) => b.result.score - a.result.score);
    const elites = evaluated.slice(0, eliteSize);
    const genBest = elites[0];
    if (!bestEver || genBest.result.score > bestEver.result.score) {
      bestEver = genBest;
    }
    trajectory.push({
      generation,
      best_score: genBest.result.score,
      best_params: {
        fragmentation_threshold_bytes: genBest.candidate.fragmentationThresholdBytes,
        bridge_batching_interval_ms: genBest.candidate.bridgeBatchingIntervalMs,
      },
    });

    const next = elites.map((e) => e.candidate);
    let guard = 0;
    while (next.length < populationSize && guard < populationSize * 20) {
      guard += 1;
      const base = elites[Math.floor(rng() * elites.length) % elites.length].candidate;
      const mutated = mutate(base, rng, space);
      mutationsProposed += 1;
      const result = evaluateCandidate(mutated, measurements);
      if (!result.valid) {
        mutationsRejected += 1;
        continue;
      }
      next.push(mutated);
    }
    // Guard against an under-filled population (should not happen with this
    // search space, since the max threshold value is always valid) rather
    // than silently shrinking the population.
    while (next.length < populationSize) next.push(randomCandidate());
    population = next;
  }

  return { trajectory, bestEver, mutationsProposed, mutationsRejected };
}
