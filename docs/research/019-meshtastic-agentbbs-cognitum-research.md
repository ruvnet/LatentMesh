# Research: Meshtastic, agentbbs, and cognitum-one API as LatentMesh Air integration targets

* Purpose: primary-source input for ADR-019+ and the Rust implementation that follows.
* Date: 2026-08-27
* Scope: three integration targets (Meshtastic, `ruvnet/agentbbs`, `cognitum-one/api`) plus a
  synthesis mapping LMS1/LMAD onto each.
* Method: protobuf/source files fetched and grepped directly (not summarized from memory),
  target repos cloned shallow into scratch and read, cognitum-one/api's own OpenAPI spec and
  ADRs read directly. Every load-bearing number below is cited to a path or URL; anything not
  independently confirmed is marked **uncertain**.

---

## 1. Meshtastic

### 1.1 Per-packet payload budget — PRIMARY, verified against raw source

`meshtastic/protobufs` `meshtastic/mesh.proto` (fetched raw, not paraphrased):

```proto
// meshtastic/mesh.proto, around line 1953
/*
 * note: this payload length is ONLY the bytes that are sent inside of the Data protobuf
 * (excluding protobuf overhead). The 16 byte header is outside of this envelope
 */
DATA_PAYLOAD_LEN = 233;
```

So the number to design against is **233 bytes**, and it is *only* the `Data.payload` bytes —
the 16-byte LoRa MAC header sits outside it and is not available to an application.

`Data` message fields (`mesh.proto:1211`, full text pulled):

```proto
message Data {
  PortNum portnum = 1;
  bytes payload = 2;
  bool want_response = 3;
  fixed32 dest = 4;
  fixed32 source = 5;
  fixed32 request_id = 6;
  fixed32 reply_id = 7;
  fixed32 emoji = 8;
  optional uint32 bitfield = 9;
  bytes xeddsa_signature = 10;
}
```

`portnum` selects the payload interpretation. `MeshPacket` (the outer envelope that carries
`Data`) additionally carries routing fields (`from`, `to`, `channel`, `hop_limit`, `want_ack`,
`priority`, `id`) — none of those come out of the 233-byte budget; they're outer-envelope bytes
Meshtastic itself owns.

### 1.2 Third-party portnum mechanism — PRIMARY

`meshtastic/protobufs` `meshtastic/portnums.proto:280-287` (raw file):

```proto
/*
 * Private applications should use portnums >= 256.
 * To simplify initial development and testing you can use "PRIVATE_APP"
 * in your code without needing to rebuild protobuf files (via regen-protos.sh)
 */
PRIVATE_APP = 256;
```

Two paths exist for a third party:
- **Registered app (portnums 64–127 by convention / "registered" range)**: send a PR to
  `portnums.proto` to get a named enum value. This is the ecosystem-sanctioned way to ship a
  supported integration (docs: `meshtastic.org/docs/development/firmware/portnum/`).
- **`PRIVATE_APP = 256`**, or any value ≥256: use immediately, no upstream change needed. This
  is the correct starting point for LatentMesh Air — no coordination dependency, and a
  follow-up PR to register a `LATENTMESH_APP` portnum is a cheap later step once the wire format
  is stable enough to publish (mirrors ADR-011's "public codec" requirement for the ham
  profiles, which is a separate legal lane — see §1.5).

### 1.3 Fragmentation — Meshtastic does NOT auto-fragment app payloads (moderate confidence, converging evidence)

No single doc page states this in one sentence, but three independent pieces of evidence
converge on **no**:

1. `Data.payload` is capped at 233 bytes and the comment explicitly frames it as "the bytes
   sent inside of the Data protobuf" — a single-packet ceiling, not a stream limit.
2. `mesh.proto` does contain a chunking mechanism — `ChunkedPayload` / `ChunkedPayloadResponse`
   (`mesh.proto:2997-3049`, full text pulled) — but it is a distinct, purpose-built
   request/response transfer protocol (payload_id, chunk_count, chunk_index, resend_chunks) used
   elsewhere in the admin/local-API surface (it sits next to `RemoteHardwarePin` in the file),
   **not** a generic fragmentation service available to `Data.payload` on arbitrary portnums.
   There is no code path shown that reassembles a `PRIVATE_APP` payload split across multiple
   `MeshPacket`s.
3. Community sources (Meshtastic Discourse, secondary — not independently verified against
   firmware source) describe application payloads over 200-ish bytes as requiring manual
   chunking by the app.

**Conclusion for design purposes: treat Meshtastic as a single-packet, no-fragmentation
transport at the application layer.** This is good news for LatentMesh Air — `fragment_message`
and `Reassembler` in `crates/latentmesh-air-core/src/fragment.rs` already do exactly this job
and need no Meshtastic-side counterpart; each Air fragment maps 1:1 to one `Data.payload`.

Grade: **inferred, high confidence** (three independent primary-source signals agree; no
single source states it as a flat claim).

### 1.4 Integration surfaces — PRIMARY (framing bytes verified as claimed by the summarizer; not independently byte-verified against firmware source, so grade as inferred/secondary for the exact byte values)

- **Serial/TCP local device API** (`meshtastic.org/docs/development/device/client-api/`):
  4-byte frame header — `START1 = 0x94`, `START2 = 0xc3`, then a 2-byte big-endian protobuf
  length, followed by the `ToRadio`/`FromRadio` protobuf bytes. Receiver treats a claimed length
  `>512` as corruption and resyncs on `START1`. This is the integration surface for a host-side
  bridge process (e.g. a Rust `latentmesh-air-radio` adapter shelling out to, or linking, the
  Meshtastic Python/C++/Go client library, or reimplementing this trivial framing directly).
- **MQTT gateway** (`meshtastic.org/docs/software/integrations/mqtt/`): topic root
  `msh/REGION`; per-channel uplink/downlink topics `msh/REGION/2/e/CHANNEL/USERID` (protobuf
  `ServiceEnvelope`-wrapped `MeshPacket`) and `msh/REGION/2/json/CHANNEL/USERID` (JSON, non-nRF52
  platforms only). If channel encryption is enabled, the protobuf topic keeps the `MeshPacket`
  payload encrypted with the channel key; the JSON topic's behavior under encryption was not
  stated on the fetched page — flag as **uncertain**, verify against firmware source before
  relying on it for anything privacy-sensitive.
- **Channel/PSK encryption**: Meshtastic channels use a pre-shared key (AES-256, default
  well-known "AQ==" PSK unless changed) at the `MeshPacket` layer, independent of any
  application-layer LMS1 signature. This is licensed-exempt-band symmetric encryption, not
  amateur-radio-restricted — see §1.5.

### 1.5 Legal lane — Meshtastic is unlicensed ISM, not amateur radio (important, changes the ADR-011 mapping)

Meshtastic's LoRa hardware operates in license-exempt ISM bands (915 MHz US, 868 MHz EU, etc.),
governed by 47 CFR Part 15 (US) / ISED RSS-247 (Canada) — the same equipment-authorization
regime ADR-011 already lists for the WiFi/BLE adapters, **not** Part 97 / RBR 4 (amateur radio).
That means:

- ADR-011's "no encrypted flag," "call sign identification," and "control operator" constraints
  are Part-97/RBR-4-specific and **do not apply** to a Meshtastic adapter.
- Meshtastic's own channel PSK/AES-256 encryption is legal to use as-is; LatentMesh Air's
  `SIGNED_ENVELOPE` flag (authentication, not encryption, per ADR-010 invariant 5) can ride
  underneath or alongside it without conflict.
- Duty cycle / airtime and power limits are still real constraints, owned by Meshtastic's own
  regulatory region tables and firmware enforcement (`RegionInfo`, `RadioInterface.cpp`), not by
  LatentMesh Air. A search for the exact EU868 duty-cycle percentage returned **conflicting
  secondary claims (1% vs. 10%, on different sub-bands, both attributed to Meshtastic
  behavior)** — this needs a direct read of `firmware/src/mesh/RadioInterface.cpp` before any
  number is written into an ADR. Flagged **uncertain — do not cite a percentage without
  re-verifying against firmware source.**
- Amateur "HamPacket" profile in this repo's own `WireProfile` enum (`wire.rs:24`,
  `WireProfile::HamPacket = 8`) is a *different* adapter with different legal obligations; a new
  Meshtastic adapter should not inherit its constraints or its code path.

### 1.6 Ecosystem norms for third-party PRIVATE_APP use

The registered-portnum list in `portnums.proto` (e.g. `SERIAL_APP`, `RANGE_TEST_APP`,
`STORE_FORWARD_APP`, `ATAK_FORWARDER`, `GROUPALARM_APP`) shows the pattern the ecosystem
expects: a short, versioned binary or protobuf body under one portnum, usually well under the
233-byte ceiling per message, with any larger transfer built as an explicit multi-message
protocol at the application layer (exactly what `STORE_FORWARD_APP` and `ATAK_FORWARDER` do).
LatentMesh Air's existing `fragment_message`/`Reassembler` machinery is a good fit for this
norm — it already does per-message multi-packet sequencing.

---

## 2. `ruvnet/agentbbs`

### 2.1 What it is (PRIMARY — read from cloned source and repo metadata)

`gh repo view ruvnet/agentbbs` → canonical name **AgentBBS**
(`github.com/ruvnet/AgentBBS`), Rust, described as "a multiplayer AI community where humans hang
out in a web app and agents connect over SSH or MCP." It is a Rust workspace built additively on
top of `late.sh`, an existing SSH/TUI social platform (`crates/late-*`), with an AgentBBS layer
on top (`crates/agentbbs-*`): `agentbbs-core` (boards, messages, identity, budget, moderation,
reputation, RVF vector memory…), `agentbbs-federation` (zero-trust node-to-node replication),
`agentbbs-mcp` (MCP server), `agentbbs-web`, `agentbbs-tui`, `agentbbs-wasm` (WASM plugin
sandbox), `agentbbs-bridge` (Slack/Teams/Discord), `agentbbs-arena` (benchmark leaderboard).

### 2.2 Confirmed: this is the backing repo for ruflo's `federation_bbs_*` tools

`agentbbs/.github/workflows/ruflo-integration-smoke.yml` (read in full) is a regression guard
whose own header states its purpose precisely:

> "assert that the agentbbs MCP server + federation crate surface stays compatible with what
> ruflo's `federation_bbs_*` MCP tools (per **ADR-164 in ruvnet/ruflo**) depend on."

It pins four contract surfaces, each independently useful for the ADR-019+ design:

1. **MCP tool names** (`crates/agentbbs-mcp/src/server.rs`, confirmed present at lines 132,
   141, 154, 168): `list_boards`, `read_board`, `post_message`, `search_memory`. Called by
   spawning the `agentbbs mcp` binary as a subprocess and speaking JSON-RPC 2.0 over stdio,
   `protocolVersion: "2024-11-05"`.
2. **`FederationPayload` enum variants** ruflo's publish/register/watch flow targets:
   `AnnounceBoard`, `ReplicateMessage`, `PeerHello`, `Ack` (out of the full set of 7 defined in
   `envelope.rs` — `BoardSnapshot` and `BoardDigest`/`PeerExchange` exist too but aren't part of
   the pinned contract).
3. **RufloAdapter CLI contract**: `agentbbs-federation/src/adapter.rs` shells out to
   `npx ruflo federation init|join|status`.
4. **End-to-end MCP roundtrip** test: `initialize` → `tools/list` → `tools/call` against the
   live binary.

Grade: this confirms the *existence and shape* of the ruflo-side contract from the agentbbs
side, which is solid primary evidence of what agentbbs commits to supporting. It does **not**
confirm ruflo's own `federation_bbs_publish/register/watch` implementation details (the
`ruvnet/ruflo` repo itself was not cloned in this pass) — grade the ruflo-side half **inferred,
not independently verified**. The tool naming in this session's own available MCP tool list
(`federation_bbs_human_join`, `federation_bbs_publish`, `federation_bbs_register`,
`federation_bbs_watch`) is consistent with — but not proof of — the ADR-164 contract described
here.

### 2.3 Wire/transport model — PRIMARY

- **`FederationEnvelope`** (`envelope.rs`): `{node: AgentId, seq: u64, payload: FederationPayload,
  signature: SignatureBytes}`. Signed over deterministic canonical bytes: a version tag, the
  node's hex id, the sequence number, and length-prefixed **JSON** of the payload
  (`serde_json`), Ed25519-signed. This is JSON, not protobuf — a materially different
  seriailzation model from LatentMesh Air's fixed-width binary LMS1/LMAD.
- **`Transport` trait** (`transport.rs`): `async fn send(&self, peer: &Peer, bytes: Vec<u8>)` —
  fully byte-opaque and transport-agnostic. The shipped implementations are `LoopbackTransport`
  (in-process, tokio mpsc, used in tests) and `TcpTransport`/`FederationServer`
  (`tcp.rs`, `MAX_FRAME: u32 = 8 * 1024 * 1024` — an 8 MiB frame ceiling). This is the clean
  seam: anything that can produce `Vec<u8>` can be a federation peer.
- **`Message`** (`board.rs:185`): `{id: MessageId (BLAKE3 of signing bytes), body: MessageBody,
  signature: SignatureBytes}` — content-addressed and Ed25519-signed, same shape discipline as
  `FederationEnvelope`.
- **RVF vector memory** (`agentbbs-core/src/rvf.rs`): a clean-room, non-RuVector, cosine-search
  `.rvf`-format store (`agentbbs.rvf.v1`) backing `search_memory`; documented as "swappable for
  the full RuVector engine via the `agentbbs-federation` AgentDB adapter when that engine is
  present" — i.e. agentbbs already anticipates being backed by a real RuVector/AgentDB store,
  which this repo already has (`crates/latentmesh-memory`, ADR-016).

### 2.4 Scale mismatch with LatentMesh Air (important synthesis input)

AgentBBS's wire format (JSON, up to 8 MiB frames, TCP/MCP-stdio transport) is built for
internet/LAN-scale federation between full nodes, not for a 300 bps radio link. It should **not**
be treated as a radio transport candidate itself. See §4 for the recommended bridge shape.

---

## 3. `cognitum-one/api`

### 3.1 Repo scope and current deploy status — PRIMARY (updates CLAUDE.local.md's table)

`cognitum-one/api`'s own README states it owns the canonical hostname `api.cognitum.one`, the
OpenAPI contract, the API-key model, and the gateway — but explicitly **does not** own the
production deployment source for the API/admin or Seed function fleets (those deploy from the
Website and Mesh ownership lanes per that repo's own issues #64/#111). Status line in the
README: **"Phase A live"** (DNS/TLS resolved, landing page served by `apiHealth`); **Phase B**
(path-based routing via the in-repo `apigateway` Cloud Run service) is still "the next
milestone," i.e. not yet the live routing path. This is consistent with, and adds detail to,
this workstation's existing notes on cognitum-one's non-uniform deploy paths — it does not
contradict them.

### 3.2 The `/v1/*` route surface actually defined in this repo (from `openapi/cognitum-api.yaml`, all paths enumerated directly)

Grouped by relevance to LatentMesh:

- **Device/fleet registration and health** (most directly relevant):
  - `POST /v1/seed/register` — `{deviceId: uuid, publicKey: base64 Ed25519, firmware}`, no auth
    (first-contact provisioning), `409` if `deviceId` reused with a different key.
  - `POST /v1/seed/heartbeat` — Ed25519-signed, body `{uptime_secs, free_memory_kb,
    total_vectors, epoch, wifi_ip, version}`.
  - `POST /v1/seed/analytics`, `POST /v1/seed/event` — free-form signed telemetry/event bodies.
  - `GET /v1/seed/check` / `GET /v1/seed/firmware/{version}` — signed OTA update check and
    binary download.
  - Auth model (`docs/seed-integration.md`, read in full): device generates an Ed25519 keypair
    at first boot, registers the public half via `/v1/seed/register`, then every subsequent
    request is signed: `canonical_request = "POST\n/v1/seed/heartbeat\n{ISO8601
    timestamp}\nsha256(body)"`, sent as `X-Device-Id` / `X-Device-Timestamp` /
    `X-Device-Signature` headers. Server rejects `|server_now - timestamp| > 300s` and rejects
    a duplicate signature within a 10-minute replay window.
- **Semantic-state ingress** (`PUT /v1/spaces/{siteId}`, ADR-099, read in full — see §3.3 below,
  this is the most important single finding for the ADR-010 gap list).
- **MCP gateway**: `GET /v1/mcp/tools` (public tool catalog), `GET /v1/mcp/sse`.
- **OAK decision/policy surface**: `POST /v1/oak/promotions/evaluate` (evaluate a
  promotion candidate — never itself changes serving), `POST /v1/oak/policies/compile`
  (compiles a production-status option into a **signed** executable policy envelope). This is a
  live, deployed analog to ADR-008's capability-governed execution / authority gate concept:
  evaluation and promotion are explicitly separate acts, and only a signed artifact is
  executable.
- **Flywheel self-improvement**: `/v1/evolve`, `/v1/learn`, `/v1/evolve/lineage`,
  `/v1/flywheel/status`, `/v1/flywheel/gate`, `/v1/microlora/*` — passthrough to the meta-LLM
  upstream; conceptually adjacent to this repo's own `latentmesh-evolve` crate and ADR-018's
  MetaHarness/Darwin loop, but a different system (self-improvement of an LLM policy, not of a
  radio/world-model link) — flag as **adjacent, not a direct integration point** unless a future
  ADR wants LatentMesh's evolve loop to report into the same flywheel gate.

### 3.3 `/v1/spaces/{siteId}` — a directly relevant worked precedent for ADR-010's open gaps

ADR-099 (`cognitum-one/api/docs/adr/ADR-099-cognitum-spaces-https-semantic-state-ingress.md`,
read in full) designs exactly the class of problem ADR-010 flags as unsolved: "entity
identifiers, units, observation time, confidence policy, provenance references, and authority"
fields required before "a received update can be promoted into an authoritative WorldGraph
fact" (ADR-010 §"Semantic envelope"). ADR-099's envelope invariants table is a live, shipped
answer to the same problem for a structurally similar system:

| Requirement | ADR-099's answer |
|---|---|
| tenant/site identity | `tenantId` = API key owner, `siteId` charset-bounded |
| ordering | monotonic `eventSequence`, enforced in a Firestore transaction; exact retries are idempotent, changed reuse conflicts |
| time | caller-supplied `observedAt` (may not be future), server-owned `expiresAt`/`ttlSeconds` bounded 30s–24h, anchored to observation time so transport delay can't revive stale state |
| privacy | only `P2`/`P3` accepted; raw/`P0`/`P1` rejected, not coerced |
| provenance/confidence/model version | `edgeRuntime.{homecoreVersion, ruviewModelVersion, calibrationId}`, `confidence`, `inference.evidence[]` |
| payload bound | 64 KiB request limit, bounded JSON depth/counts/strings |
| idempotency | document id derived from `(tenant, siteId)`; `messageId` distinguishes exact retry from conflicting reuse |
| self-attestation limits | publisher-supplied `hardwareManifest`/`manifestVerification` are discarded server-side — a device cannot mark its own hardware "verified" |

This is directly reusable as a checklist when the ADR-019+ authors design LMAD's still-missing
upper-layer schema fields. It's also a cautionary tale worth citing: ADR-099 opens by noting
that its predecessor design (ADR-094, MQTT-based) shipped a read path with **no producer at
all** for months ("Nothing writes a space document... `GET /v1/spaces` can only ever return
`[]`") — a concrete argument for building LatentMesh's receiver-side promotion path and a
minimal producer together, not sequentially.

### 3.4 Reconciling this repo against this workstation's existing deployment notes

CLAUDE.local.md (this session's own project memory) describes a **V0 appliance** local API:
bearer-auth `cog-gateway` on port 9000 serving `/api/v1/v0/*`, edge heartbeat at
`POST /api/v1/v0/edge/heartbeat`, UDP discovery on port 5008. **None of these paths exist in
`cognitum-one/api`'s OpenAPI spec** — the only `/api/v1/v0/*` reference found anywhere in this
repo is a passing cross-reference inside ADR-099 itself:

> "the V0 hardware seam is already an HTTP API (`/api/v1/v0/csi/label`, `/system/csi-events`,
> `/bfld/status`), not a driver."

This confirms `/api/v1/v0/*` is a **separate, locally-run V0-appliance HTTP API** (a different
codebase/service than `cognitum-one/api`, which owns `api.cognitum.one` cloud routes under
plain `/v1/*`). **Conclusion: a LatentMesh node integrating with the cloud fleet API should
target `/v1/seed/*` and `/v1/spaces/*` on `api.cognitum.one` (this repo); a LatentMesh node
integrating with a physically co-located V0 appliance should target its separate `/api/v1/v0/*`
local gateway** — these are two different registration surfaces for two different deployment
shapes, not duplicates of each other. This should be stated explicitly in ADR-019+ so future
readers don't conflate them.

---

## 4. Synthesis

### 4.1 The decisive arithmetic — Air frame + LMS1/LMAD sizes vs. the Meshtastic 233-byte ceiling

Numbers pulled directly from source, not estimated:

| Layer | Size | Source |
|---|---:|---|
| Air outer frame overhead (12B header + 4B CRC32C) | **16 bytes**, fixed, every fragment | `crates/latentmesh-air-core/src/wire.rs`: `FRAME_HEADER_BYTES=12`, `FRAME_MIN_BYTES=16` |
| Air frame ceiling (native, e.g. WiFi/BLE) | 256 bytes total → 240 bytes payload | `wire.rs`: `FRAME_MAX_BYTES=256`, `FRAME_MAX_PAYLOAD=240` |
| LMS1 semantic-envelope header | 48 bytes, +64 if `SIGNED_ENVELOPE` | `crates/latentmesh-air-core/src/envelope.rs`: `ENVELOPE_HEADER_BYTES=48`, `ENVELOPE_SIGNATURE_BYTES=64` |
| LMAD delta body, fixed part | 52 bytes minimum (magic+version+flag+ids+base/result hash), before any `SymbolUpdate`/`Residual` entries | `crates/latentmesh-air-core/src/semantic.rs`: `decode`'s `FIXED_LEN=52` |
| Meshtastic per-packet ceiling | **233 bytes**, `Data.payload` only | `mesh.proto:1953` |

Because Meshtastic does not fragment `Data.payload` for you (§1.3), **the Air frame's own
16-byte overhead must come out of the 233-byte Meshtastic budget**, not out of Air's native
240-byte payload ceiling. That leaves:

```
233 (Meshtastic Data.payload budget)
 - 16 (Air frame header + CRC32C)
= 217 bytes usable Air-fragment payload per Meshtastic packet
```

This is the number the Meshtastic adapter's `fragment_message(..., frame_mtu)` call should use:
`frame_mtu = FRAME_MIN_BYTES + 217 = 233`, i.e. exactly Meshtastic's own ceiling, not Air's
native 256-byte ceiling. Using Air's default 256-byte MTU against a Meshtastic transport would
silently overflow `Data.payload` by 23 bytes and get rejected or truncated at the Meshtastic
layer — worth a code comment or a named constant when this adapter is built.

**Does a typical delta fit in one packet?** An unsigned `SemanticEnvelope` wrapping a
single-field `SemanticDelta` with no residuals: `48 (LMS1 header) + 52 (LMAD fixed) + ~4-6
(one SymbolUpdate) ≈ 104-106 bytes` of LMS1+LMAD body, plus 16 bytes of Air frame overhead ≈
**120-122 bytes total** — comfortably under the 217-byte usable budget, so **one Meshtastic
packet, no Air fragmentation needed** for the common case. A signed envelope (+64 bytes) pushes
that to ~186 bytes — still fits in one packet, but with much less headroom for updates/residuals.
Anything with multiple field updates, residual slots, or a signed envelope carrying several
updates will cross 217 bytes and needs Air's existing multi-fragment path — which maps cleanly
1:1, one Air fragment per Meshtastic `Data.payload`, up to the existing 32-fragment /
`MAX_MESSAGE_BYTES` (32 × 240 = 7680 bytes native; effectively 32 × 217 = 6944 bytes when Air is
scoped to the Meshtastic MTU) ceiling.

### 4.2 Recommended adapter architecture

**Meshtastic is a new row in ADR-011's adapter table, not a new PHY.** It needs:
- One new `WireProfile` variant (the enum has room: current values 0–8 of a 4-bit nibble,
  `wire.rs:15-25`) — e.g. `WireProfile::Meshtastic = 9`.
- A byte-transport adapter analogous to the existing WiFi/BLE ones (`transmitter.rs`'s
  `Transmission::Bytes { frame, bytes }` variant already produces exactly "encoded frame bytes,
  hand them to a transport" — no protocol-specific logic needed in `latentmesh-air-core` or
  `latentmesh-air-radio` itself). The adapter's only job: speak the Meshtastic local-device
  serial/TCP framing (`0x94 0xc3` + 2-byte length + `ToRadio` protobuf, §1.4) to submit each Air
  frame as one `Data.payload` under `PRIVATE_APP` (or a later-registered dedicated portnum).
- **No FEC, interleaving, or modem work required** — unlike the HF/VHF/BPSK profiles, Meshtastic
  owns LoRa PHY, FEC, multi-hop relay, and ACK itself. This makes it the *lightest-weight*
  adapter in the table, closer in effort to WiFi/BLE than to the HF/VHF profiles.
- **A different legal lane than HamPacket** (§1.5) — do not reuse the amateur-radio gate
  (`LM_RF_TX_ENABLE`, call-sign ID, no-encryption) for this adapter; Meshtastic falls under the
  same license-exempt-equipment reasoning already used for the WiFi/BLE rows.
- Use `frame_mtu = 233` bytes (not Air's native 256), per §4.1.

### 4.3 agentbbs: not a radio transport, a store-and-forward/discovery layer above it

Given the scale mismatch (§2.4), the clean shape is a **bridge**, not a tunnel:

1. **Store-and-forward bulletin pattern**: an Air-connected node (gateway/base station) that has
   internet access decodes received `SemanticEnvelope`/`SemanticDelta` messages and republishes
   them as agentbbs content — either as a `post_message` MCP call, or, for a more
   federation-native path, as a `FederationPayload::ReplicateMessage` sent through
   `agentbbs-federation`'s `Transport` trait. This matches the task's own "store-and-forward"
   framing directly, and it's fully loopback-testable today using `LoopbackTransport` and the
   `agentbbs mcp` stdio binary — no radio hardware needed to build and test this bridge.
2. **Discovery/registration pattern**: `AnnounceBoard`/`PeerHello` (both in the ruflo-pinned
   4-variant contract, §2.2) let a gateway advertise "this LatentMesh stream/node exists, here is
   its current state-hash checkpoint" without needing any human or agent to have radio hardware —
   good for a demo/observability surface and for the `list_boards`/`read_board` MCP tools to
   present mesh state to a human.
3. **Mapping ruflo's named verbs**: `federation_bbs_publish` → `post_message` (or
   `ReplicateMessage` via the adapter path) · `federation_bbs_register` →
   `AnnounceBoard`/`PeerHello` · `federation_bbs_watch` → **no dedicated push/subscribe tool
   exists in the 4-tool contract** (`list_boards`, `read_board`, `post_message`,
   `search_memory`) — a "watch" verb is most likely implemented ruflo-side as a poll loop over
   `read_board`, not as a push mechanism from agentbbs. This is an **inferred** mapping, not
   confirmed against ruflo's own source (not cloned this pass) — verify against
   `ruvnet/ruflo` ADR-164 directly before building against it.
4. Note per the smoke-test workflow's own comment: this is explicitly **"ruflo's Phase 2
   federation_bbs_* wire-up"** — i.e., described as not-yet-fully-built even on the ruflo side as
   of this workflow's authorship. Treat the 4-tool/4-variant surface as the stable target to
   build against, not the (possibly still-moving) ruflo Phase 2 internals.

### 4.4 cognitum-one/api: two registration surfaces, pick by deployment shape

- **Cloud fleet path** (`api.cognitum.one`, this repo): register a LatentMesh
  gateway/edge-node as a Seed-shaped device — `POST /v1/seed/register` (device UUID + Ed25519
  pubkey, mirrors LMS1's own asymmetric-signature model for `SIGNED_ENVELOPE`) then
  `POST /v1/seed/heartbeat` populated from decoded state (note: both systems independently named
  a field `epoch` — `cognitum-api`'s heartbeat body and LMS1's `SemanticEnvelope.epoch`, worth
  flagging as a naming coincidence to either align deliberately or keep clearly distinct in the
  ADR). For richer semantic content (not just liveness), `PUT /v1/spaces/{siteId}` is the closer
  match and, per §3.3, already solves the exact provenance/confidence/sequence/privacy-class
  problem ADR-010 leaves open — worth reading as a design reference even if the actual endpoint
  isn't called yet.
- **Local appliance path** (V0, separate codebase): target the co-located appliance's own
  `/api/v1/v0/*` gateway (per this workstation's existing notes) — do **not** conflate this with
  `cognitum-one/api`'s `/v1/*` surface; they are different services (§3.4).
- Both paths are HTTP+JSON, easily mockable — fully loopback-testable against the documented
  OpenAPI contract with a local mock server; Ed25519 signing of the canonical-request string is
  offline-testable without any live credential.

### 4.5 What's testable today vs. hardware-pending

**Buildable and testable in loopback/simulation on this host, no new hardware:**
- Meshtastic adapter: fragment Air frames at the 233-byte MTU, push through a mock
  `Data.payload` channel, verify `Reassembler` round-trips — same style as the existing
  frame/fragment unit tests in `wire.rs`/`fragment.rs`, extendable with `harness/air`'s existing
  simulator.
- agentbbs bridge: decode→`post_message`/`ReplicateMessage`, tested against
  `agentbbs-federation`'s `LoopbackTransport` and the `agentbbs mcp` stdio binary.
- cognitum-one/api bridge: heartbeat/spaces payload construction and Ed25519 request signing,
  tested against a mock HTTP server built from the published OpenAPI contract.

**Hardware- or credential-pending, cannot be closed in this workspace:**
- Any real Meshtastic RF validation — no LoRa hardware present, and this matches this repo's own
  ADR-010/011 status ("no board has been flashed" for the existing ESP32 radio profiles).
- Live `cognitum-one/api` seed registration — needs a provisioned device identity and, per
  ADR-099, a `spaces:write`-scoped API key that doesn't exist by default.
- Real (non-loopback) agentbbs federation between two live nodes over a network — this session
  only exercised the loopback path; TCP/production federation is unverified here.

---

## 5. Evidence table

| # | Claim | Grade | Source |
|---|---|---|---|
| 1 | Meshtastic `Data.payload` max = 233 bytes, excludes 16B LoRa header | Primary | `meshtastic/protobufs` `meshtastic/mesh.proto:1953`, fetched raw |
| 2 | `Data` message field list | Primary | `mesh.proto:1211-1266`, fetched raw |
| 3 | `PRIVATE_APP = 256`; registered apps get a PR-assigned enum value | Primary | `meshtastic/protobufs` `meshtastic/portnums.proto:280-287`, fetched raw |
| 4 | Meshtastic does not auto-fragment/reassemble app `Data.payload` across packets | Inferred (high confidence, 3 converging signals) | `mesh.proto` payload-length comment; `ChunkedPayload`/`ChunkedPayloadResponse` shown to be a separate admin-surface mechanism (`mesh.proto:2997-3049`); community secondary sources |
| 5 | Local device API framing: `0x94 0xc3` + 2B length + protobuf | Secondary (fetched via summarizing fetch of official docs page, not raw-grepped) | `meshtastic.org/docs/development/device/client-api/` |
| 6 | MQTT topics `msh/REGION/2/e/CHANNEL/USERID` (protobuf `ServiceEnvelope`) and `.../json/...` | Secondary | `meshtastic.org/docs/software/integrations/mqtt/` |
| 7 | Encrypted channel keeps protobuf-topic `MeshPacket` payload encrypted; JSON-topic behavior under encryption | Uncertain — not stated on fetched page | same as #6 |
| 8 | Meshtastic LoRa = license-exempt ISM (Part 15/RSS-247), not amateur radio | Primary reasoning from this repo's own ADR-011 categories + well-established Meshtastic band usage | `docs/adr/011-radio-adapters-and-legal-boundary.md` (this repo) + Meshtastic region docs |
| 9 | EU868 duty cycle percentage | **Uncertain — conflicting secondary claims (1% vs 10%), not resolved** | WebSearch summary only; needs direct read of `firmware/src/mesh/RadioInterface.cpp` |
| 10 | agentbbs = ruflo's `federation_bbs_*` backing repo (ADR-164 in ruvnet/ruflo) | Primary (agentbbs-side statement) / Inferred (ruflo-side, not independently verified — ruflo repo not cloned) | `agentbbs/.github/workflows/ruflo-integration-smoke.yml`, read in full |
| 11 | agentbbs-mcp 4 tools: `list_boards`, `read_board`, `post_message`, `search_memory` | Primary | `agentbbs/crates/agentbbs-mcp/src/server.rs:132,141,154,168` |
| 12 | `FederationPayload` variants pinned by ruflo contract: `AnnounceBoard`, `ReplicateMessage`, `PeerHello`, `Ack` (7 total variants exist) | Primary | `agentbbs/crates/agentbbs-federation/src/envelope.rs`, read in full |
| 13 | `Transport` trait is byte-opaque; `TcpTransport` `MAX_FRAME = 8 MiB` | Primary | `agentbbs/crates/agentbbs-federation/src/transport.rs`, `tcp.rs` |
| 14 | FederationEnvelope signs length-prefixed **JSON**, Ed25519 | Primary | `agentbbs/crates/agentbbs-federation/src/envelope.rs` |
| 15 | cognitum-one/api: Phase A live, Phase B (path routing) not yet live | Primary | `cognitum-api/README.md`, read in full |
| 16 | `/v1/seed/register`, `/v1/seed/heartbeat` schemas + Ed25519 canonical-request signing, 300s clock skew, 10min replay window | Primary | `cognitum-api/openapi/cognitum-api.yaml:1491-1620`; `cognitum-api/docs/seed-integration.md`, read in full |
| 17 | `PUT /v1/spaces/{siteId}` envelope invariants (tenant/site id, monotonic sequence, ttl-bounded expiry, P2/P3-only privacy, provenance/confidence/model-version, 64KiB bound, idempotency, self-attestation rejected) | Primary | `cognitum-api/docs/adr/ADR-099-cognitum-spaces-https-semantic-state-ingress.md`, read in full |
| 18 | `/api/v1/v0/*` is a separate, local V0-appliance API, not part of `cognitum-one/api`'s `/v1/*` surface | Primary (by absence + explicit cross-reference) | grep of full `cognitum-api` repo for `/api/v1/v0`, `cog-gateway`, `:9000` → zero hits except one cross-reference inside ADR-099 |
| 19 | Air frame overhead 16B (12B header + 4B CRC32C); frame ceiling 256B/240B payload | Primary | `crates/latentmesh-air-core/src/wire.rs` (this repo) |
| 20 | LMS1 envelope header 48B, +64B signature | Primary | `crates/latentmesh-air-core/src/envelope.rs` (this repo) |
| 21 | LMAD delta fixed body length 52B (`decode`'s `FIXED_LEN`) | Primary | `crates/latentmesh-air-core/src/semantic.rs` (this repo) |
| 22 | 217 usable Air-payload bytes per Meshtastic packet (233 − 16) | Derived arithmetic from #1 and #19 | this document, §4.1 |

---

## 6. Open questions / contradictions to resolve before ADR-019+ is finalized

1. **EU868 (and other region) duty-cycle percentages** are not pinned down (evidence #9) —
   needs a direct read of Meshtastic firmware's `RadioInterface.cpp`/region tables, not docs
   pages, before any specific percentage goes into an ADR.
2. **JSON-topic MQTT encryption behavior** (evidence #7) is unconfirmed — matters if any
   MQTT-bridge design is considered for LatentMesh state hashes.
3. **ruflo's own `federation_bbs_*` implementation** (verb-to-tool mapping in §4.3.3) is inferred
   from agentbbs's side of the contract only; the `ruvnet/ruflo` repo itself should be read
   directly (specifically ADR-164) before committing to the publish/register/watch → MCP-tool
   mapping proposed here.
4. **Whether to register a dedicated Meshtastic portnum** (vs. staying on `PRIVATE_APP = 256`)
   is a product decision, not a research gap — flagging it here since ADR-011's "public codec"
   principle for the ham profiles is a reasonable norm to extend voluntarily to this adapter even
   though it isn't legally required outside amateur bands.
