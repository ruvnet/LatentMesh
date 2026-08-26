# 015. Live MidStream latent streaming

- **Status**: Accepted and implemented (in-process + QUIC transport; live LLM deployment remains external).
- **Date**: 2026-08-26
- **Related**: [004](004-streaming-latent-state.md) (the design contract this ADR implements), [002](002-latent-packet-protocol.md) (`LatentFrame`), [008](008-capability-governed-execution.md) (authority ladder)

## Context

ADR-004 fixed the integration contract — `LatentFrame.sequence` + MidStream's
chunk pipeline + confidence-gated authority escalation — and left the transport
wiring as open work. MidStream's transport layer is now published as
`midstreamer-quic 0.3.0`, which exposes a `QuicTransport` embedding trait
exactly so downstream crates like this one can write generic code against the
QUIC connection surface without binding to `quinn`. That removes the reason to
wait: the live wiring can be implemented against the published trait, tested
against an in-memory implementation of the same trait, and exercised against a
real QUIC connection without any code change.

## Decision

New crate **`latentmesh-stream`**:

- **`FrameTransport`** — the crate's own minimal transport abstraction
  (`send_frame` / `recv_frame` over length-prefixed `LatentFrame` JSON). Two
  implementations ship: `ChannelTransport` (in-process, deterministic, used by
  tests and benchmarks) and, behind the `midstream-quic` feature,
  `QuicFrameTransport<T: midstreamer_quic::QuicTransport>` which runs the same
  framing over a MidStream QUIC bidirectional stream.
- **Wire format** — 4-byte big-endian length prefix + serde-JSON `LatentFrame`,
  with a hard `MAX_FRAME_BYTES` bound (1 MiB) enforced on both send and
  receive. The identical framing is implemented on the MidStream side
  (`midstream-lm-bridge`), and a shared golden fixture checked into both
  repositories keeps the two codecs byte-compatible in CI.
- **`LatentStreamSender` / `LatentStreamReceiver`** — per-stream monotonic
  `sequence` assignment on send; gap and reorder detection on receive within a
  bounded rolling window (ADR-004's "MidStream rolling-window analysis applies
  unchanged"). Out-of-window, duplicate, and regressed sequences are rejected,
  not silently reordered.
- **Confidence-gated authority escalation** — the receiver tracks cumulative
  evidence across the stream and grants each frame an *effective* authority:
  every stream starts at `ObserveOnly` and may escalate one rung at a time as
  aggregate confidence crosses configured thresholds, but never above the
  frame's own declared authority and never above the gate's cap. Escalation
  state is per-stream, resets on gap detection, and de-escalates when
  confidence drops — governance and streaming compose, per ADR-004.

## Consequences

- The pipelining latency claim of ADR-004 is now measurable in-repo:
  `latentmesh-bench` gains a streamed-vs-sequential pipeline benchmark
  (evidence label: host software benchmark, in-process transport — it proves
  the pipelining shape, not a production deployment's latency).
- The QUIC path type-checks against the published `QuicTransport` trait and is
  integration-tested against an in-memory mock of that trait; a live two-peer
  QUIC test requires a network fixture and stays out of unit CI, consistent
  with how `midstreamer-quic` itself tests.
- A malformed or oversized frame on the wire is a rejected frame, never a
  panic: all decode paths return typed errors and are exercised with malformed
  inputs in tests.

## Implementation status

Implemented this session: `latentmesh-stream` (transport abstraction, framing
codec with bounds, sequencing, escalation), the MidStream-side bridge crate,
the shared golden fixture, tests on both sides, and the pipeline benchmark.
Not claimed: production latency numbers over a real network, or wiring into a
specific live LLM serving deployment.
