# 002. Latent packet protocol (`LatentFrame`)

- **Status**: Proposed. **The packet type and codecs are implemented** (`crates/latentmesh-core`, `crates/latentmesh-align`); the network transport is not.
- **Date**: 2026-08-18
- **Related**: [001](001-latentmesh-architecture-and-prior-art.md) (why this is scoped narrower than a competing wire format)

## Context

ADR-001 §3 establishes that AVP (Agent Vector Protocol) already specifies a transport-agnostic binary protocol for KV cache, hidden states, model metadata, and cross-model projection — reportedly explicitly for latent communication, not discovery or orchestration. Building a competing from-scratch wire format here would duplicate published, more mature work. But this workspace's own crates (MidStream, RuVector, MetaHarness, Radio, RVF, RVM) need a concrete Rust type to build and test the streaming/memory/topology/governance layers against *today* — hence a reference type, not a protocol claim.

## Decision

**`LatentFrame`** (`latentmesh-core`) is the packet:

```rust
struct LatentFrame {
    id: String,
    sender_model: String,
    receiver_space: String,   // the embedding space this frame is aligned FOR
    transform_hash: String,   // content hash of the R/α used — never a bare, unhashed matrix
    sequence: u64,             // ADR-004 streaming order
    payload: Payload,          // { encoding, dim, bytes, int8_params }
    confidence: f32,           // latentmesh-align's own fit estimate, in [0,1]
    provenance: Provenance,    // sender_model, context_hash, parents[]
    authority: Authority,      // ObserveOnly < ContextInject < LatentPrefix < ActionInfluencing
    timestamp: u64,
}
```

- **Encoding is explicit and measured, not assumed.** `Encoding::{F32, F16, Int8}` each report their real `bytes_per_element`; `Int8` is genuine per-tensor affine quantization (`scale`, `zero_point`), not a placeholder. `Payload::wire_bytes()` is exact, not estimated — see `latentmesh-bench` (ADR-004) for measured sizes at realistic dimensions.
- **A frame is unaligned data plus alignment metadata, not a black box.** `transform_hash` binds a frame to a specific, reproducible alignment (`latentmesh-align`'s `AlignmentTransform`), so a receiver can verify which `R`/`α` produced `receiver_space` before trusting it — this is the hook ADR-008's governance gate checks.
- **Alignment (`latentmesh-align`) is the general, well-defined technique — orthogonal Procrustes via SVD** (`R = argmin_{R orthogonal} ||A R − B||_F`, closed-form as `R = U Vᵀ` from the SVD of `Aᵀ B`), *not* a re-implementation of StateBridge's specific method. This crate makes an honest claim: given calibration pairs `(a_i, b_i)` in the two spaces, it recovers the least-squares-optimal orthogonal alignment, verifiably, on synthetic data. It does **not** claim StateBridge's reported 22/26 best-or-tied-best result — that requires live heterogeneous LLM hidden states, which this repo does not have access to (ADR-001 §8). `confidence` is derived from the residual `||A R̂ − B||_F` on the calibration set, scaled to `[0,1]` — a real, computed number, not an LLM-asserted one.

## Consequences

- Every other crate in this workspace has one, tested, serde-friendly type to build against; swapping in an AVP-compatible wire encoding later is an adapter at the edge, not a rewrite of ADR-004–008.
- The alignment crate's correctness claim is deliberately smaller than the literature's: "recovers a known orthogonal transform from calibration pairs" (verified by synthetic test), not "aligns real heterogeneous LLM concepts" (unverified here).
- Quantization tradeoffs are measurable, not asserted: `latentmesh-bench` reports actual `wire_bytes()` at F32/F16/Int8 for realistic dims, which is the empirical basis for any bandwidth-allocation decision in ADR-006.

## Implementation

- `crates/latentmesh-core/src/lib.rs` — `Encoding`, `Payload::{encode,decode,wire_bytes}`, `Provenance`, `Authority`, `LatentFrame::content_hash`. Tests: F32/F16/Int8 round-trip within their real precision bounds; the 16×4096-dim FP16 ≈ 128 KiB scaling check; content-hash sensitivity; authority ordering.
- `crates/latentmesh-align/src/lib.rs` — `AlignmentTransform::fit(pairs) -> (R, α, confidence)` via SVD-based orthogonal Procrustes; `apply(z) -> aligned z`. Tests: recovers a known random orthogonal `R` from noiseless synthetic pairs; degrades gracefully (lower confidence) under injected noise; `apply` composed with the known inverse recovers the original vector within tolerance.
