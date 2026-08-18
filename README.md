<div align="center">

# LatentMesh

### A causally-verified latent communication fabric for continuously evolving agent collectives

[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT_OR_Apache--2.0-yellow?style=for-the-badge)](#license)
[![rust](https://img.shields.io/badge/rust-1.77%2B-orange?style=for-the-badge)](#workspace)
[![status](https://img.shields.io/badge/status-research%20prototype-e6b45a?style=for-the-badge)](#honest-status)
[![ADRs: 9](https://img.shields.io/badge/ADRs-9_decisions-6366f1?style=for-the-badge)](docs/adr/README.md)

**[ADRs](docs/adr) · [The story](https://ruvnet.github.io/LatentMesh/)**

</div>

---

> Agents should not need to convert everything they know into language before communicating with another agent. LatentMesh is a research prototype exploring what happens when an agent's hidden state — not its serialized text — becomes the network primitive. It is not a claim that this is unclaimed territory: as of 2026-08-18, [StateBridge](docs/adr/001-latentmesh-architecture-and-prior-art.md), LatentMAS, AVP, and AAFLOW+ already occupy most of the wire-format and raw-transfer ground, and [E2 Explainer](docs/adr/009-online-causal-control-loop.md) and MANTA already do causal attribution and dynamic topology respectively. What this repo bets on instead: **an agent-to-agent latent edge only earns execution authority if it survives a counterfactual test proving it transferred real information** — a continuously-running control loop, not a wire format. See [ADR-001](docs/adr/001-latentmesh-architecture-and-prior-art.md) and [ADR-009](docs/adr/009-online-causal-control-loop.md) for the full, corrected positioning — including where the claim narrowed after a second literature pass in the same day it was written.

## The loop

```
execute → transfer latent state → counterfactual audit → measure causal value
  → update edge authority → persist result → change topology → next execution
```

Instead of:

```
Agent A → generate tokens → serialize text → network → tokenize text → Agent B
```

a `LatentFrame` carries an aligned hidden-state slice directly, and — the actual bet — **no edge is trusted by default**. Every candidate edge `A → B` is tested against five controls (zero state, random state, mismatched-task state, self-generated state, and the *text-equivalent* of the same content) before it may influence anything beyond passive observation. Beating `text_equivalent` specifically is what makes a surviving edge a claim about latent communication, not merely about communication-vs-silence.

```rust
struct LatentFrame {
    id: String,
    sender_model: String,
    receiver_space: String,   // the embedding space this frame is aligned FOR
    transform_hash: String,   // content hash of the R/α actually used
    sequence: u64,             // streaming order
    payload: Payload,          // { encoding: F32|F16|Int8, dim, bytes, int8_params }
    confidence: f32,           // measured alignment-fit quality, in [0,1]
    provenance: Provenance,    // sender_model, context_hash, parents[]
    authority: Authority,      // ObserveOnly < ContextInject < LatentPrefix < ActionInfluencing
    timestamp: u64,
}
```

## Structural guarantees (in the types and the tests, not in policy docs)

- **An edge earns authority; it is never granted it by default** — an edge absent from the gate's policy defaults to `ObserveOnly` regardless of what it requests (`latentmesh-gate`, `Gate::admit`, tested).
- **Causally-unverified edges are structurally capped below `ActionInfluencing`** — `ceiling_from_verdict` maps a rejected/untested edge to `ObserveOnly`; only a statistically significant `ΔV` against all five controls raises the ceiling (tested).
- **The admission gate returns the FIRST violated rule**, in the same deterministic, clock-free, independently-re-runnable style as [`cognitum-one/slack`'s AGL module](https://github.com/cognitum-one/slack/blob/main/docs/adr/0009-agl-admission-module.md) — this repo ports that shape from code-mutation governance onto latent-execution governance (ADR-008).
- **The causal test is nonparametric** — a sign-flip permutation test, no assumption that task-value noise is Gaussian or continuous (`latentmesh-gate::causal`, tested for false-positive rate under the null and invariance to positive rescaling).
- **The risk score is an explicit, documented placeholder**, not a trained trajectory-risk model — using it as if it were validated would misrepresent this repo's safety posture, so it's named as a stand-in in the code and in [ADR-008](docs/adr/008-capability-governed-execution.md), not buried in a footnote.

## Workspace

| Crate | What it does | ADR |
|---|---|---|
| [`latentmesh-core`](crates/latentmesh-core) | `LatentFrame`, `Encoding` (F32/F16/Int8 with real per-tensor affine quantization), `Payload::wire_bytes()` (exact, not estimated), `Provenance`, `Authority` | [002](docs/adr/002-latent-packet-protocol.md) |
| [`latentmesh-align`](crates/latentmesh-align) | Training-free orthogonal alignment (Procrustes/SVD); optimized `O(d²n)` fast path (QR + small SVD) for the realistic same-dim, few-calibration-pairs case, verified against the `O(d³)` dense reference | [002](docs/adr/002-latent-packet-protocol.md) |
| [`latentmesh-gate`](crates/latentmesh-gate) | The admission gate (`execute(z) ⟺ signature ∧ authority ∧ provenance ∧ risk<τ`) and causal edge verification (five-control permutation test) | [003](docs/adr/003-causal-edge-verification.md), [008](docs/adr/008-capability-governed-execution.md) |
| [`latentmesh-bench`](crates/latentmesh-bench) | Real, measured numbers only — no live LLM access, so no claimed cross-model task accuracy; measures wire bytes and alignment/verification wall-clock | [002](docs/adr/002-latent-packet-protocol.md) |

**23 tests pass** (`cargo test --workspace`), **clippy clean** (`cargo clippy --workspace --all-targets -- -D warnings`).

## Measured, not asserted

`cargo run --release -p latentmesh-bench` on this machine, before and after the 2026-08-18 optimization ([ADR-002](docs/adr/002-latent-packet-protocol.md)):

| dim | calibration pairs | fit — before | fit — after | speedup |
|---|---|---|---|---|
| 64 | 16 | 0.171 ms | 0.073 ms | 2.3× |
| 256 | 16 | 13.4 ms | 0.37 ms | 36× |
| 1024 | 16 | 2792 ms | 6.5 ms | **430×** |
| 4096 | 16 | 155,996 ms (~2.6 min) | 161.8 ms | **~964×** |

The "before" number is not a strawman — it's what a direct, textbook dense-SVD orthogonal Procrustes solver actually costs at an LLM-realistic hidden dimension, measured on this repo's own code before the fix. The "after" number comes from exploiting that the calibration set (`n` ≈ 16–64 pairs) makes the transform matrix's true rank far below the embedding dimension — an exact reformulation (`O(d²n)` instead of `O(d³)`), not an approximation; `fast_path_matches_dense_reference` and `fast_path_is_identity_on_the_orthogonal_complement` in `latentmesh-align`'s test suite prove the two paths agree on the calibrated subspace and that the (deliberately chosen) null-space policy — identity on directions with no calibration evidence — holds.

Wire bytes at 16×4096-dim vectors: **256 KiB at F32, 128 KiB at F16, 64.1 KiB at Int8** — exactly matching the back-of-envelope scaling intuition this protocol was built to test, now backed by an actual measurement instead of an assumption.

## Honest status

This is a **research prototype**. What's real: the packet/codec types, the alignment algorithm (both the correctness *and* the performance claims, each proven by its own test), the causal-verification statistics, and the admission gate — all typed, tested, deterministic, and runnable offline right now. What's **not** done: no live open-weight model integration (this environment has no hidden-state access to a real heterogeneous LLM pair), no MidStream/RuVector/Radio/MetaHarness/RVF/RVM wiring (ADR-004–009 are integration contracts, not shipped pipelines), and every specific percentage figure attributed to external literature (StateBridge, LatentMAS, AVP, AAFLOW+, StreamMA, DMoA, E2 Explainer, MANTA, BANDMAS, CrystalMem, TTHE, FedWorld, DreamGuard, and the causal-audit papers cited in the ADRs) is exactly that — attributed, not reproduced. See [ADR-001 §8](docs/adr/001-latentmesh-architecture-and-prior-art.md#8-honest-feasibility) and [ADR-009 §6](docs/adr/009-online-causal-control-loop.md#6-honest-bench-numbers-this-repo-this-session--see-root-readme).

```bash
cargo build --workspace
cargo test --workspace                              # 23 tests
cargo clippy --workspace --all-targets -- -D warnings
cargo run --release -p latentmesh-bench              # real measured numbers
```

## Architecture decisions

| ADR | Decides |
|---|---|
| [001](docs/adr/001-latentmesh-architecture-and-prior-art.md) | The north star, and an honest map of prior art |
| [002](docs/adr/002-latent-packet-protocol.md) | The `LatentFrame` packet + alignment algorithm |
| [003](docs/adr/003-causal-edge-verification.md) | The five-control causal test an edge must survive |
| [004](docs/adr/004-streaming-latent-state.md) | Streaming latent frames over MidStream |
| [005](docs/adr/005-persistent-latent-memory.md) | RuVector's raw→compressed→prototype→symbolic memory continuum |
| [006](docs/adr/006-self-evolving-topology.md) | MetaHarness Darwin mutating the topology itself |
| [007](docs/adr/007-federated-world-models.md) | Radio federating compatible transition rules across nodes |
| [008](docs/adr/008-capability-governed-execution.md) | RVF/RVM's capability gate for opaque latent execution |
| [009](docs/adr/009-online-causal-control-loop.md) | The corrected, narrower claim — the closed loop, and one role per existing ruvnet component |

## The acceptance test

Three heterogeneous open models, three machines, a task requiring continuous collaboration. **Pass** if latent streaming beats text communication by ≥25% in wall-clock latency or token cost while keeping task accuracy within 2 points — *and*, per [ADR-009](docs/adr/009-online-causal-control-loop.md#5-the-one-experiment-not-a-dozen-repos), more than 80% of the edges a `CausalDynamicLatent` topology retains individually survive the mismatched-state control, not just the aggregate comparison. Neither half alone is sufficient. Not run this session — no live model access; see Honest status above.

## License

MIT OR Apache-2.0, matching ruvnet norm. See [LICENSE-MIT](LICENSE-MIT) / [LICENSE-APACHE](LICENSE-APACHE).
