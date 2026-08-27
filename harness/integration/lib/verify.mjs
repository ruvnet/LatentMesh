// Pure verification of harness/integration receipts (ADR-022) against the
// bounds each e2e suite's own Decision-section scenario names. Mirrors
// harness/evolve's frozen-promotion-rule style: every failed bound pushes a
// named reason; an empty reason list is a pass. Deliberately
// dependency-free.
//
// Evidence-label discipline (ADR-014 + ADR-018, extended by ADR-022):
// receipts must carry evidence: "simulated" (the exact string harness/evolve
// already hard-requires) plus a non-empty `not_claimed` list. The brief's
// phrase "deterministic simulation — no hardware, no live peers, no
// credentials" lives in `evidence_detail`, not in place of the required
// `evidence` label.

export const EVIDENCE_LABEL = "simulated";

export const MESHTASTIC_RECEIPT_SCHEMA = "latentmesh-integration-meshtastic-receipt-v1";
export const AGENTBBS_RECEIPT_SCHEMA = "latentmesh-integration-agentbbs-receipt-v1";
export const COGNITUM_RECEIPT_SCHEMA = "latentmesh-integration-cognitum-receipt-v1";
export const OPTIMIZE_RECEIPT_SCHEMA = "latentmesh-integration-optimize-receipt-v1";

export const MESHTASTIC_BOUNDS = Object.freeze({
  mtu: 233,
  usableBytesPerPacket: 217,
  unsignedPacketCount: 1,
  signedPacketCount: 1,
  multiFragmentMessageBytes: 300,
  multiFragmentPacketCount: 2,
  roundTripByteIdenticalRequired: true,
  portnumWiringRequired: true,
});

export const AGENTBBS_BOUNDS = Object.freeze({
  decodeRoundTripRequired: true,
  postArgsShapeRequired: true,
  signatureVerificationRequired: true,
  replicateShapeRequired: true,
  replicateVerificationRequired: true,
  liveMcpRoundtripMustSucceedIfAttempted: true,
});

export const COGNITUM_BOUNDS = Object.freeze({
  registerShapeRequired: true,
  registerResponseParseRequired: true,
  heartbeatShapeRequired: true,
  heartbeatResponseParseRequired: true,
  deviceIdHeaderMatchRequired: true,
  signatureVerificationRequired: true,
});

/** Common envelope checks shared by every suite's receipt. */
function verifyEnvelope(receipt, expectedSchema) {
  const reasons = [];
  if (!receipt || typeof receipt !== "object") {
    return { reasons: ["receipt is not an object"], fatal: true };
  }
  if (receipt.schema !== expectedSchema) {
    reasons.push(`schema is ${JSON.stringify(receipt.schema)}, expected ${expectedSchema}`);
  }
  if (receipt.evidence !== EVIDENCE_LABEL) {
    reasons.push(
      `evidence label is ${JSON.stringify(receipt.evidence)} — receipts must say "${EVIDENCE_LABEL}", never an over-claimed label`,
    );
  }
  if (!Array.isArray(receipt.not_claimed) || receipt.not_claimed.length === 0) {
    reasons.push("receipt must state what it does NOT claim");
  }
  return { reasons, fatal: false };
}

export function verifyMeshtasticReceipt(receipt) {
  const { reasons, fatal } = verifyEnvelope(receipt, MESHTASTIC_RECEIPT_SCHEMA);
  if (fatal) return { pass: false, reasons };
  const d = receipt.driver;
  if (!d || typeof d !== "object") {
    reasons.push("driver output is missing");
    return { pass: false, reasons };
  }
  if (d.mtu !== MESHTASTIC_BOUNDS.mtu) {
    reasons.push(`mtu is ${d.mtu}, expected ${MESHTASTIC_BOUNDS.mtu}`);
  }
  if (d.usable_bytes_per_packet !== MESHTASTIC_BOUNDS.usableBytesPerPacket) {
    reasons.push(
      `usable_bytes_per_packet is ${d.usable_bytes_per_packet}, expected ${MESHTASTIC_BOUNDS.usableBytesPerPacket}`,
    );
  }
  if (!d.unsigned || d.unsigned.packet_count !== MESHTASTIC_BOUNDS.unsignedPacketCount) {
    reasons.push("unsigned single-field delta did not fit exactly one Meshtastic packet");
  }
  if (!d.unsigned?.round_trip_ok) reasons.push("unsigned delta did not round-trip byte-identical");
  if (!d.unsigned?.portnum_ok) reasons.push("unsigned packet did not carry PRIVATE_APP portnum");
  if (!d.signed || d.signed.packet_count !== MESHTASTIC_BOUNDS.signedPacketCount) {
    reasons.push("signed single-field delta did not fit exactly one Meshtastic packet");
  }
  if (!d.signed?.round_trip_ok) reasons.push("signed delta did not round-trip byte-identical");
  if (!d.multi_fragment || d.multi_fragment.message_bytes !== MESHTASTIC_BOUNDS.multiFragmentMessageBytes) {
    reasons.push("multi-fragment scenario did not use the documented 300-byte message");
  }
  if (d.multi_fragment?.packet_count !== MESHTASTIC_BOUNDS.multiFragmentPacketCount) {
    reasons.push(
      `multi-fragment packet count is ${d.multi_fragment?.packet_count}, expected ${MESHTASTIC_BOUNDS.multiFragmentPacketCount}`,
    );
  }
  if (!d.multi_fragment?.round_trip_ok) reasons.push("multi-fragment message did not round-trip byte-identical");
  return { pass: reasons.length === 0, reasons };
}

export function verifyAgentbbsReceipt(receipt) {
  const { reasons, fatal } = verifyEnvelope(receipt, AGENTBBS_RECEIPT_SCHEMA);
  if (fatal) return { pass: false, reasons };
  const d = receipt.driver;
  if (!d || typeof d !== "object") {
    reasons.push("driver output is missing");
    return { pass: false, reasons };
  }
  if (!d.decode_round_trip_ok) reasons.push("decoded SemanticDelta did not round-trip through the bridge");
  if (!d.post_args_shape_ok) reasons.push("post_message argument shape did not match the pinned contract");
  if (!d.signature_verified) reasons.push("signed republish message did not verify");
  if (!d.replicate_shape_ok) reasons.push("ReplicateMessage JSON shape did not match the pinned contract");
  if (!d.replicate_verified) reasons.push("ReplicateMessage payload's embedded signature did not verify");
  if (d.mcp?.attempted && !d.mcp?.ok) {
    reasons.push(`live agentbbs mcp roundtrip was attempted and failed: ${d.mcp.error}`);
  }
  return { pass: reasons.length === 0, reasons };
}

export function verifyCognitumReceipt(receipt) {
  const { reasons, fatal } = verifyEnvelope(receipt, COGNITUM_RECEIPT_SCHEMA);
  if (fatal) return { pass: false, reasons };
  const d = receipt.driver;
  if (!d || typeof d !== "object") {
    reasons.push("driver output is missing");
    return { pass: false, reasons };
  }
  if (!d.register?.shape_ok) reasons.push("register request did not match the documented unsigned shape");
  if (!d.register?.response_ok) reasons.push("register mock response was not parsed as expected");
  if (!d.heartbeat?.shape_ok) reasons.push("heartbeat request body did not round-trip the documented schema");
  if (!d.heartbeat?.response_ok) reasons.push("heartbeat mock response was not parsed as expected");
  if (!d.heartbeat?.device_id_header_ok) reasons.push("X-Device-Id header did not match the signing identity");
  if (!d.heartbeat?.signature_verified) {
    reasons.push("Ed25519 signature reconstructed from the wire did not verify against the device's public key");
  }
  return { pass: reasons.length === 0, reasons };
}

export function verifyOptimizeReceipt(receipt) {
  const { reasons, fatal } = verifyEnvelope(receipt, OPTIMIZE_RECEIPT_SCHEMA);
  if (fatal) return { pass: false, reasons };
  if (!Array.isArray(receipt.trajectory) || receipt.trajectory.length !== receipt.generations) {
    reasons.push("trajectory length does not match the recorded generation count");
  }
  if (!receipt.selected || typeof receipt.selected !== "object") {
    reasons.push("no selected parameters recorded");
    return { pass: false, reasons };
  }
  const space = receipt.search_space;
  if (!space?.fragmentationThresholdBytes?.includes(receipt.selected.fragmentation_threshold_bytes)) {
    reasons.push("selected fragmentation threshold is outside the declared search space");
  }
  if (!space?.bridgeBatchingIntervalMs?.includes(receipt.selected.bridge_batching_interval_ms)) {
    reasons.push("selected batching interval is outside the declared search space");
  }
  const usable = receipt.measurements?.usable_bytes_per_packet;
  if (typeof usable === "number" && receipt.selected.fragmentation_threshold_bytes > usable) {
    reasons.push("selected fragmentation threshold exceeds the measured Meshtastic packet ceiling");
  }
  if (!(receipt.selected.score >= receipt.baseline?.score)) {
    reasons.push("selected parameters scored worse than the documented baseline");
  }
  if (!receipt.round_trip_correctness_preserved) {
    reasons.push("optimizer selected parameters that would break round-trip correctness");
  }
  return { pass: reasons.length === 0, reasons };
}
