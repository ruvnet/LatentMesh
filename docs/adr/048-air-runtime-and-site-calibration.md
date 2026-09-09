# ADR-048: AIR runtime separation and offline site residual calibration

Status: partially implemented; hardware and OCUDU integration blocked.

## Context and decision

Keep M3/M5X negative evidence intact. This work does not establish latent-channel
utility. AIR remains deterministic transport with optional, separately qualified
assistance and a higher-level cache/orchestrator.

Separate synchronous native execution from asynchronous observations. Implement
the first software slice in the existing portable-C host library rather than
introduce a second modem. A lane owns its runtime and clock. Policy output is
copied to a temporary, generation/range checked, then accepted only on timely
completion. Eight consecutive failures disable that callback. No hot-path
allocation. An SPSC observer queue is bounded and drops when full.

The current receiver callback is a TRUSTED SYNCHRONOUS CPU experimental block
transform, NOT a CUDA stream hook, OCUDU ABI, or production neural receiver.
Late output is discarded before entering the stateful existing CRC/auth/replay
pipeline. A failed CRC is returned without retrying partially mutated state.
Callbacks cannot be safely preempted: deadlines detect lateness after return;
they do not provide a hard wall-time guarantee. Never admit untrusted code.

For channel models, add a four-feature ridge residual with nested whole-TX
holdouts and fold-local normalization. Input is measured path power plus RT
features. This is a SPARC-inspired baseline, not its unreleased implementation.
No fabricated dataset or claim of replication. JSON artifacts avoid pickle.
Cross-site application fails closed; drift qualification remains a separate gate.

## Acceptance and control flow

1. Valid block -> candidate -> deadline/metadata check -> existing decoder ->
   message callback. Queue publication never waits for an observer.
2. Late/bad callback -> conventional block/intent; repeated misses -> breaker.
3. Fit on remaining TX groups -> tune on inner groups -> predict outer TX group
   -> aggregate held-out errors -> export refit model for offline use.
4. Measured paired-arm receipts -> validation -> gates -> campaign candidate.
   Production promotion is intentionally disabled until independent, cross-site,
   five-seed evidence and confidence intervals exist.

## Alternatives and limits

External RPC is appropriate for orchestration, not assumed suitable for same-slot
decisions. Actual external-process versus native benchmarks remain to be built.
Rust integration is deferred; the execution environment had no Rust toolchain.
GPU-resident PUSCH integration requires inspecting and pinning the real OCUDU SDK,
not inventing a compatible-looking ABI. No OCUDU source was compiled here.
No radio transmissions, cloud GPU purchases, production deployment or model
activation are authorized by software fixture success.

## Threat model

Assets: receiver state, timing availability, model provenance, radio authority.
Inputs: hostile frames, offline JSON measurements, trusted native callbacks.
Controls: existing frame validation/auth/replay remains authoritative; bounded
queue; generation-checked policy; nonfinite input rejection; no network listener;
no deserialization of executable models; no automatic qualification/promotion.
Residual risks: callbacks can hang or corrupt process memory; caller-supplied
provenance can lie; unsigned frames are not authenticated; queue supports only
one producer/consumer; calibration extrapolation can be inaccurate. Keep this
research-only and use authenticated envelopes under existing policy.

## Remaining E2E gates (owner: integration maintainer/operator)

- Pin and compile actual OCUDU SDK; implement GPU tensors/stream completion,
  admission profiles, weight lifecycle/rollback and no-allocation checks.
- Implement external-process comparison and connect Class B decisions to actual
  scheduler commits (current policy helper is not a radio scheduler).
- Connect Class C observer to existing state-cache orchestration; current queue
  is only a transport primitive, not a semantic cache or MCP implementation.
- Supply replay-verified PUSCH and two-site path captures; benchmark material
  calibration and boosting, not only raw/constant offset/ridge.
- Run operator-controlled 20 MHz HIL/OTA, shared-GPU contention and wideband tests.
  Measure energy, latency tails, BLER, bandwidth and end-task utility together.
- Verify receipts independently and add cross-site/five-seed uncertainty gates.

## Research references (motivation, not implementation dependencies)

- https://arxiv.org/abs/2609.07843
- https://arxiv.org/abs/2609.07805
- https://arxiv.org/abs/2609.06077
