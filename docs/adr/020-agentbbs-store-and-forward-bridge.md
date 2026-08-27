# 020. agentbbs store-and-forward bridge

- **Status**: Accepted — implemented this wave (loopback/simulation evidence only). Updated 2026-08-27.
- **Date**: 2026-08-27.
- **Related**: [007](007-federated-world-models.md) (federation contract this bridge extends to a human-facing surface), [017](017-radio-federated-world-models.md) (the transport this bridge sits downstream of), [019](019-meshtastic-transport-adapter.md) (the likely last-mile transport carrying decoded state to a gateway node)
- **Evidence base**: [docs/research/019-meshtastic-agentbbs-cognitum-research.md](../research/019-meshtastic-agentbbs-cognitum-research.md) §2, §4.3

## Context

`ruvnet/agentbbs` (canonical name `AgentBBS`) is confirmed as the backing
repo for ruflo's `federation_bbs_*` MCP tools: its own CI
(`.github/workflows/ruflo-integration-smoke.yml`) states it guards
compatibility with "what ruflo's `federation_bbs_*` MCP tools (per ADR-164 in
`ruvnet/ruflo`) depend on," and pins four MCP tool names
(`list_boards`, `read_board`, `post_message`, `search_memory`,
`agentbbs-mcp/src/server.rs:132,141,154,168`) and four `FederationPayload`
variants (`AnnounceBoard`, `ReplicateMessage`, `PeerHello`, `Ack`, out of
seven total, `envelope.rs`). That confirms agentbbs's own commitment; it does
**not** confirm ruflo's implementation, since `ruvnet/ruflo` itself was not
read in the research pass — this ADR marks that gap explicitly below.

agentbbs's wire model — `FederationEnvelope { node, seq, payload, signature }`
signing length-prefixed **JSON** (not protobuf) over `serde_json`, Ed25519,
with a byte-opaque `Transport` trait and an 8 MiB `TcpTransport` frame
ceiling (`tcp.rs`, `MAX_FRAME`) — is built for internet/LAN-scale federation
between full nodes. That is a different scale than a 300 bps radio link and
LatentMesh Air's 16-256 byte frames; agentbbs must not be treated as a radio
transport candidate.

## Decision

**A bridge/gateway pattern, not a tunnel.** An Air-connected node with
internet access decodes received `SemanticEnvelope`/`SemanticDelta` messages
and republishes their content into agentbbs — it does not carry agentbbs's
JSON wire format over the radio link, and it does not carry LMS1/LMAD bytes
through agentbbs unmodified. The two protocols meet at a decode/re-encode
boundary, one per direction:

- **Publish path**: decoded delta content → either a `post_message` MCP call
  (simplest, human-board-facing) or, for a more federation-native path, a
  `FederationPayload::ReplicateMessage` sent through
  `agentbbs-federation`'s `Transport` trait. Maps ruflo's
  `federation_bbs_publish` verb onto this path.
- **Discovery/registration path**: `AnnounceBoard`/`PeerHello` let a gateway
  advertise "this LatentMesh stream/node exists, here is its current
  state-hash checkpoint" without requiring any human or agent on the other
  end to hold radio hardware. Maps `federation_bbs_register`.
- **No push/subscribe verb exists in the pinned 4-tool contract.**
  `federation_bbs_watch` has no dedicated push mechanism to map onto — the
  4-tool surface (`list_boards`, `read_board`, `post_message`,
  `search_memory`) supports only polling `read_board`. This ADR adopts that
  as the working assumption (a poll loop, not a push subscription) but flags
  it explicitly as **inferred from agentbbs's side only, not verified
  against `ruvnet/ruflo`'s own ADR-164** — the mission's required open
  verification item. Do not build ruflo-side automation against this mapping
  without reading ADR-164 directly first.
- **New crate `latentmesh-agentbbs-bridge`.** Its core is pure mapping
  functions — `SemanticEnvelope`/`SemanticDelta` (decoded, already
  authenticated and CRC-checked by `latentmesh-air-core`) → agentbbs
  `post_message`/`ReplicateMessage` payload shapes — deliberately hermetically
  testable with no live process. **agentbbs is not a Cargo dependency of this
  workspace.** It is a separate Rust workspace (`ruvnet/AgentBBS`) with its
  own crates; vendoring it as a path/git dependency would pull an
  unbounded, independently-versioned tree into `cargo test --workspace` and
  break the hermetic/offline guarantee ADR-021 also relies on. Instead the
  bridge crate speaks agentbbs's contract as a **wire boundary**: JSON-RPC
  2.0 over stdio to the `agentbbs mcp` subprocess binary
  (`initialize` → `tools/list` → `tools/call`, `protocolVersion:
  "2024-11-05"`) for the MCP path, and hand-rolled `FederationPayload` JSON
  matching the four pinned variants for the federation path. Both are
  data-shape contracts, not code dependencies.
- **Fully loopback-testable today.** agentbbs already ships
  `LoopbackTransport` (in-process, tokio mpsc) for `Transport`-trait testing
  and the `agentbbs mcp` binary is invokable as a subprocess for the MCP
  path — this ADR's acceptance test is: spawn the binary (or a fixture
  double when the binary is unavailable in CI), round-trip a decoded
  `SemanticDelta` through the mapping functions, and confirm the resulting
  `post_message`/`ReplicateMessage` JSON matches the pinned schema. No
  internet-scale, no live peer, no radio hardware required.

## Consequences

Decode/re-encode at the boundary means the bridge owns exactly one
translation responsibility and nothing else — it does not need to understand
agentbbs's board/reputation/moderation model beyond the fields
`post_message`/`ReplicateMessage` require, and it does not need agentbbs to
understand LMS1/LMAD at all. The cost is that this ADR's correctness depends
on a contract (ADR-164, the 4-tool/4-variant surface) that is explicitly
described by its own smoke-test workflow as "ruflo's Phase 2
`federation_bbs_*` wire-up" — i.e., possibly still-moving on the ruflo side.
Building against the 4-tool/4-variant surface as documented is the stable
choice; automating the `watch` mapping specifically should wait for direct
ADR-164 confirmation.

## What is simulated / what is hardware-pending

| Claim | Status |
|---|---|
| Decoded `SemanticEnvelope`/`SemanticDelta` → `post_message`/`ReplicateMessage` JSON mapping | Buildable and testable today, hermetic unit tests, no external process |
| MCP roundtrip (`initialize`/`tools/list`/`tools/call`) against the real `agentbbs mcp` stdio binary | Buildable and testable today if the binary is present in the test environment; falls back to a fixture double otherwise |
| `FederationPayload` roundtrip against `LoopbackTransport` | Buildable and testable today, no network, no radio hardware |
| `federation_bbs_watch` → poll-vs-push mapping | **Inferred, not verified** — read `ruvnet/ruflo` ADR-164 directly before relying on this |
| Real (non-loopback) agentbbs federation between two live nodes over TCP | **Not implemented, not claimed** — research pass only exercised loopback |
| Any live radio delivering the decoded content in the first place | **Hardware-pending** — depends on [019](019-meshtastic-transport-adapter.md) or another Air transport actually running |

## Implementation status

Implemented 2026-08-27, same branch. `crates/latentmesh-agentbbs-bridge`
exists with agentbbs kept out of the Cargo dependency graph (wire contract
transcribed field-for-field with source citations), the mapping-function
core, a hand-rolled JSON-RPC 2.0 stdio MCP client, and an in-memory peer
that explicitly disclaims being agentbbs's own `LoopbackTransport`. 25
hermetic tests pass; one `#[ignore]`d opt-in test was additionally run once
against the real `agentbbs mcp` binary built from a shallow clone and passed
(documented reproducibly in the crate). The `federation_bbs_watch`
poll-vs-push mapping remains inferred-not-verified pending a direct read of
`ruvnet/ruflo` ADR-164.
