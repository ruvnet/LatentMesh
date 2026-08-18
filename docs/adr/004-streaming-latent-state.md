# 004. Streaming latent state

- **Status**: Proposed (design; not wired to a live MidStream deployment).
- **Date**: 2026-08-18
- **Related**: [002](002-latent-packet-protocol.md) (`sequence` field), [001](001-latentmesh-architecture-and-prior-art.md) §3 (StreamMA prior art)

## Context

StreamMA (reported) lets downstream agents consume upstream reasoning incrementally instead of waiting for a complete generation, reportedly averaging +7.3pp across tested benchmarks by removing the serialization bottleneck inherent in sequential text pipelines. MidStream already exists in this stack as "treat a token stream as a first-class signal you can analyze/gate/steer per chunk, with nanosecond-scale scheduling and a QUIC transport" — it is the natural substrate for streaming *latent* frames the same way it streams tokens today.

## Decision

Instead of:

```
Agent A ████████████████
              Agent B ████████████████
                            Agent C ███████████████
```

(each agent waits for the previous one to finish, then serializes/tokenizes)

stream `z_1, z_2, …, z_t` continuously as a sequence of `LatentFrame`s (`sequence: u64` in ADR-002's type) over MidStream's chunk-analysis pipeline, so `B` begins consuming `A`'s partial cognitive state as it is produced, and `C` begins consuming `B`'s partial state in turn — a pipeline across independent models, not a request/response chain.

- **`sequence` gives total order per stream**, so a receiver can detect gaps/reordering and MidStream's existing rolling-window analysis (already used to catch attacks split across chunk boundaries in the antibody/MidStream integration) applies unchanged to latent chunks.
- **Confidence and authority are per-frame** (ADR-002), so a receiver can choose to act on a low-confidence early frame at `ObserveOnly` authority and only escalate to `ActionInfluencing` once enough frames have arrived to raise confidence — streaming and governance compose rather than being separate concerns.
- **Backpressure and bandwidth are causal-value-driven** (ADR-006 §bandwidth), not fixed per stream: an edge with high measured `ΔV` (ADR-003) can be allocated more of MidStream's transport budget than a low-value edge carrying the same nominal traffic.

## Consequences

- The latency story is a real, testable engineering claim (does pipelining N frames reduce wall-clock vs. waiting for completion?) independent of whether the *content* transferred is causally useful (ADR-003) — this ADR is about the transport shape, ADR-003 is about whether what's transported matters.
- MidStream's existing chunk-boundary attack detection extends to latent streams for free, which is directly relevant to ADR-008's governance story: a malicious or corrupted latent frame split across the wire is exactly the case MidStream's rolling window was built to catch, just applied to a different payload type.

## Implementation status

Not implemented this session — this ADR records the integration contract (`LatentFrame.sequence` + MidStream's chunk pipeline + confidence-gated authority escalation) for a follow-up that wires `latentmesh-core` frames through an actual MidStream stream. The packet and codec side (`sequence`, `confidence`, `authority` fields) already exist and are tested (ADR-002); the MidStream transport wiring itself is the open work.
