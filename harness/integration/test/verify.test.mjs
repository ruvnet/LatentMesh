import test from "node:test";
import assert from "node:assert/strict";
import {
  AGENTBBS_RECEIPT_SCHEMA,
  COGNITUM_RECEIPT_SCHEMA,
  MESHTASTIC_RECEIPT_SCHEMA,
  OPTIMIZE_RECEIPT_SCHEMA,
  verifyAgentbbsReceipt,
  verifyCognitumReceipt,
  verifyMeshtasticReceipt,
  verifyOptimizeReceipt,
} from "../lib/verify.mjs";
import {
  baselineCandidate,
  evaluateCandidate,
  runDarwinSearch,
  SEARCH_SPACE,
} from "../lib/optimizer.mjs";

function goodMeshtasticReceipt() {
  return {
    schema: MESHTASTIC_RECEIPT_SCHEMA,
    evidence: "simulated",
    driver: {
      mtu: 227,
      usable_bytes_per_packet: 211,
      unsigned: { envelope_bytes: 105, air_frame_bytes: 121, packet_count: 1, portnum_ok: true, round_trip_ok: true },
      signed: { envelope_bytes: 169, air_frame_bytes: 185, packet_count: 1, portnum_ok: true, round_trip_ok: true },
      multi_fragment: { message_bytes: 300, packet_count: 2, portnum_ok: true, round_trip_ok: true },
    },
    not_claimed: ["no Meshtastic hardware present on this host"],
  };
}

test("a compliant meshtastic receipt passes with no reasons", () => {
  const { pass, reasons } = verifyMeshtasticReceipt(goodMeshtasticReceipt());
  assert.deepEqual(reasons, []);
  assert.equal(pass, true);
});

test("an over-claimed evidence label is refused", () => {
  const receipt = goodMeshtasticReceipt();
  receipt.evidence = "over the air";
  const { pass, reasons } = verifyMeshtasticReceipt(receipt);
  assert.equal(pass, false);
  assert.ok(reasons.some((r) => r.includes("evidence label")));
});

test("a missing not_claimed list fails", () => {
  const receipt = goodMeshtasticReceipt();
  receipt.not_claimed = [];
  const { pass, reasons } = verifyMeshtasticReceipt(receipt);
  assert.equal(pass, false);
  assert.ok(reasons.some((r) => r.includes("NOT claim")));
});

test("wrong meshtastic mtu is named", () => {
  const receipt = goodMeshtasticReceipt();
  receipt.driver.mtu = 256;
  const { pass, reasons } = verifyMeshtasticReceipt(receipt);
  assert.equal(pass, false);
  assert.ok(reasons.some((r) => r.includes("mtu is")));
});

test("a multi-fragment packet count that is not 2 fails", () => {
  const receipt = goodMeshtasticReceipt();
  receipt.driver.multi_fragment.packet_count = 3;
  const { pass, reasons } = verifyMeshtasticReceipt(receipt);
  assert.equal(pass, false);
  assert.ok(reasons.some((r) => r.includes("multi-fragment packet count")));
});

test("a round-trip mismatch fails", () => {
  const receipt = goodMeshtasticReceipt();
  receipt.driver.signed.round_trip_ok = false;
  const { pass, reasons } = verifyMeshtasticReceipt(receipt);
  assert.equal(pass, false);
  assert.ok(reasons.some((r) => r.includes("signed delta did not round-trip")));
});

test("garbage input never throws for any verifier", () => {
  for (const junk of [null, 42, "x", [], {}, { schema: "wrong" }]) {
    for (const verify of [verifyMeshtasticReceipt, verifyAgentbbsReceipt, verifyCognitumReceipt, verifyOptimizeReceipt]) {
      const { pass } = verify(junk);
      assert.equal(pass, false);
    }
  }
});

function goodAgentbbsReceipt() {
  return {
    schema: AGENTBBS_RECEIPT_SCHEMA,
    evidence: "simulated",
    driver: {
      decode_round_trip_ok: true,
      post_args_shape_ok: true,
      signature_verified: true,
      replicate_shape_ok: true,
      replicate_verified: true,
      mcp: { attempted: false, ok: false, error: null },
    },
    not_claimed: ["no live agentbbs peer or federation network"],
  };
}

test("a compliant agentbbs receipt passes", () => {
  const { pass, reasons } = verifyAgentbbsReceipt(goodAgentbbsReceipt());
  assert.deepEqual(reasons, []);
  assert.equal(pass, true);
});

test("an attempted-but-failed mcp roundtrip fails the agentbbs receipt", () => {
  const receipt = goodAgentbbsReceipt();
  receipt.driver.mcp = { attempted: true, ok: false, error: "connection closed" };
  const { pass, reasons } = verifyAgentbbsReceipt(receipt);
  assert.equal(pass, false);
  assert.ok(reasons.some((r) => r.includes("connection closed")));
});

test("an unverified signature fails the agentbbs receipt", () => {
  const receipt = goodAgentbbsReceipt();
  receipt.driver.signature_verified = false;
  const { pass, reasons } = verifyAgentbbsReceipt(receipt);
  assert.equal(pass, false);
  assert.ok(reasons.some((r) => r.includes("did not verify")));
});

function goodCognitumReceipt() {
  return {
    schema: COGNITUM_RECEIPT_SCHEMA,
    evidence: "simulated",
    driver: {
      register: { shape_ok: true, response_ok: true },
      heartbeat: { shape_ok: true, response_ok: true, device_id_header_ok: true, signature_verified: true },
    },
    not_claimed: ["no network call was made to api.cognitum.one"],
  };
}

test("a compliant cognitum receipt passes", () => {
  const { pass, reasons } = verifyCognitumReceipt(goodCognitumReceipt());
  assert.deepEqual(reasons, []);
  assert.equal(pass, true);
});

test("an unverified cognitum signature fails", () => {
  const receipt = goodCognitumReceipt();
  receipt.driver.heartbeat.signature_verified = false;
  const { pass, reasons } = verifyCognitumReceipt(receipt);
  assert.equal(pass, false);
  assert.ok(reasons.some((r) => r.includes("did not verify")));
});

// --- optimizer.mjs ---

const MEASUREMENTS = Object.freeze({ usableBytesPerPacket: 211, multiFragmentMessageBytes: 300 });

test("a threshold above the measured packet ceiling is invalid", () => {
  const result = evaluateCandidate(
    { fragmentationThresholdBytes: 227, bridgeBatchingIntervalMs: 0 },
    MEASUREMENTS,
  );
  assert.equal(result.valid, false);
});

test("evaluateCandidate is deterministic for the same inputs", () => {
  const candidate = { fragmentationThresholdBytes: 211, bridgeBatchingIntervalMs: 250 };
  const a = evaluateCandidate(candidate, MEASUREMENTS);
  const b = evaluateCandidate(candidate, MEASUREMENTS);
  assert.deepEqual(a, b);
});

test("a smaller fragmentation threshold never produces fewer fragments", () => {
  const wide = evaluateCandidate({ fragmentationThresholdBytes: 211, bridgeBatchingIntervalMs: 0 }, MEASUREMENTS);
  const narrow = evaluateCandidate({ fragmentationThresholdBytes: 140, bridgeBatchingIntervalMs: 0 }, MEASUREMENTS);
  assert.ok(narrow.fragments >= wide.fragments);
});

test("runDarwinSearch is deterministic for a fixed seed", () => {
  const a = runDarwinSearch({ seed: 42, generations: 6, populationSize: 8, eliteSize: 2, measurements: MEASUREMENTS });
  const b = runDarwinSearch({ seed: 42, generations: 6, populationSize: 8, eliteSize: 2, measurements: MEASUREMENTS });
  assert.deepEqual(a.trajectory, b.trajectory);
  assert.deepEqual(a.bestEver.candidate, b.bestEver.candidate);
});

test("runDarwinSearch produces exactly one trajectory entry per generation", () => {
  const { trajectory } = runDarwinSearch({ seed: 7, generations: 10, populationSize: 8, eliteSize: 2, measurements: MEASUREMENTS });
  assert.equal(trajectory.length, 10);
});

test("runDarwinSearch never selects a threshold above the measured ceiling", () => {
  const { bestEver, trajectory } = runDarwinSearch({ seed: 99, generations: 8, populationSize: 8, eliteSize: 2, measurements: MEASUREMENTS });
  assert.ok(bestEver.candidate.fragmentationThresholdBytes <= MEASUREMENTS.usableBytesPerPacket);
  for (const entry of trajectory) {
    assert.ok(entry.best_params.fragmentation_threshold_bytes <= MEASUREMENTS.usableBytesPerPacket);
  }
});

test("the search finds a score at least as good as the baseline", () => {
  const { bestEver } = runDarwinSearch({ seed: 2026, generations: 10, populationSize: 8, eliteSize: 2, measurements: MEASUREMENTS });
  const baseline = evaluateCandidate(baselineCandidate(MEASUREMENTS), MEASUREMENTS);
  assert.ok(bestEver.result.score >= baseline.score);
});

test("SEARCH_SPACE only contains thresholds at or below a plausible Meshtastic budget", () => {
  for (const value of SEARCH_SPACE.fragmentationThresholdBytes) {
    assert.ok(value <= 211);
  }
});
