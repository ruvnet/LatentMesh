# 019. Meshtastic transport adapter

- **Status**: Accepted — implemented this wave (loopback/simulation evidence only). Updated 2026-08-27.
- **Date**: 2026-08-27.
- **Related**: [010](010-latentmesh-air-protocol.md) (frame/envelope this adapter carries unmodified), [011](011-radio-adapters-and-legal-boundary.md) (adapter table and legal boundary this ADR extends), [013](013-esp32-firmware.md) (existing WiFi/BLE adapters, the closest precedent), [014](014-benchmark-and-acceptance-method.md) (stage-gate discipline this adapter must enter at)
- **Evidence base**: [docs/research/019-meshtastic-agentbbs-cognitum-research.md](../research/019-meshtastic-agentbbs-cognitum-research.md) §1, §4.1-4.2

## Context

Meshtastic is a deployed, off-the-shelf LoRa mesh (firmware, hardware, and an
existing user base) that already solves PHY, FEC, multi-hop relay, and ACK —
everything ADR-011's HF/VHF/BPSK rows still need an external modem or TNC for.
Riding on top of it gets LatentMesh Air a working radio path without writing
or certifying a PHY. The question this ADR answers is narrow: how does an Air
frame become a Meshtastic `Data.payload`, and how much of that 233-byte
budget is actually usable once Air's own framing comes out of it.

## Decision

**Meshtastic is a new row in ADR-011's adapter table, not a new PHY.** It
carries LMS1/LMAD unmodified; it owns none of the protocol's semantic or
authentication logic. Concretely:

- **New `WireProfile::Meshtastic = 9`** in `latentmesh-air-core::wire`
  (`wire.rs:15-25`, current values 0-8 fill 9 of a 4-bit nibble's 16 slots).
  Because ADR-010 declares the C codec in `c/**` canonical and this enum a
  **cross-language ABI** locked to shared golden vectors, adding the Rust
  variant alone is incomplete — `lm_air_profile_t`
  (`c/include/latentmesh_air/frame.h:11-21`) needs the matching
  `LM_AIR_PROFILE_MESHTASTIC = 9` entry and a new golden-vector case, or C/Rust
  conformance silently diverges on this one profile byte. This ADR requires
  both sides land together; Rust-only is not an acceptable partial merge.
- **New crate `latentmesh-meshtastic`**, not a module inside
  `latentmesh-air-radio`. `air-radio` is deliberately lean (`no_std`-leaning,
  only `libm` beyond `air-core`) and owns modem/PHY work this adapter has none
  of; the Meshtastic side instead needs a protobuf dependency and byte-level
  serial/TCP framing, which belongs in its own crate the way a future
  hardware-specific adapter would get one. The crate's only job, per
  `transmitter.rs`'s existing `Transmission::Bytes { frame, bytes }` variant
  (already "encoded frame bytes, hand them to a transport" — no
  protocol-specific logic needed upstream): speak Meshtastic's local device
  API to submit each Air frame as one `Data.payload`.
- **Fixed `frame_mtu = 233`, not Air's native 256-byte `FRAME_MAX_BYTES`.**
  Meshtastic's `Data.payload` ceiling is 233 bytes and is not a stream
  limit — three independent primary-source signals in the research doc's §1.3
  converge on **Meshtastic does not auto-fragment application payloads**
  (inferred, high confidence; `ChunkedPayload` in `mesh.proto:2997-3049` is a
  distinct admin-surface mechanism, not a generic `Data.payload` fragmenter).
  So Air's own 16-byte frame overhead (`FRAME_HEADER_BYTES=12` +
  4-byte CRC32C) comes out of the 233-byte budget, not out of Air's 240-byte
  native payload ceiling: `233 − 16 = 217` usable Air-fragment payload bytes
  per Meshtastic packet. The adapter must call
  `fragment_message(meta, message, 233)` — a named constant
  (`MESHTASTIC_FRAME_MTU: usize = 233`), never Air's default MTU, or frames
  silently overflow `Data.payload` by up to 23 bytes and get rejected or
  truncated by Meshtastic itself.
  - Verified against source: LMS1 header is 48 bytes
    (`ENVELOPE_HEADER_BYTES`), LMAD's fixed body is 52 bytes
    (`semantic.rs:298`, `FIXED_LEN`). An unsigned single-field delta —
    48 + 52 + ~4-6 bytes for one `SymbolUpdate` ≈ 104-106 bytes of LMS1+LMAD
    body, plus 16 bytes of Air frame overhead ≈ **120-122 bytes total** —
    fits in one Meshtastic packet with room to spare. A signed envelope
    (+64 bytes, `ENVELOPE_SIGNATURE_BYTES`) pushes that to ~186 bytes — still
    one packet, but tight. Anything larger (multiple field updates, residual
    slots, several updates under one signature) crosses 217 bytes and needs
    Air's existing multi-fragment path, which maps 1:1 — one Air fragment per
    Meshtastic `Data.payload` — up to the existing 32-fragment ceiling
    (32 × 217 = 6,944 bytes effective, vs. 32 × 240 = 7,680 bytes native).
- **Portnum: `PRIVATE_APP = 256`** (`portnums.proto:280-287`) as the starting
  value — no upstream coordination dependency. A follow-up PR to register a
  dedicated `LATENTMESH_APP` portnum is a deliberate later step, not a
  blocker, mirroring ADR-011's "public codec" norm for the ham profiles
  voluntarily extended here even though it isn't legally required outside
  amateur bands (open question #4 in the research doc).
- **Device API framing**: `0x94 0xc3` (`START1`/`START2`) + 2-byte
  big-endian protobuf length + `ToRadio`/`FromRadio` protobuf bytes, over
  serial or TCP to a host-side Meshtastic node (`meshtastic.org` client-API
  docs — secondary source, fetched but not independently byte-verified
  against firmware; grade accordingly if this framing is ever load-bearing
  for a security property). This is the adapter's actual implementation
  surface: a thin encoder/decoder for this framing plus the minimal
  `ToRadio.packet.decoded` / `FromRadio.packet.decoded` protobuf fields
  needed to carry a `Data { portnum: PRIVATE_APP, payload }`.
- **A different legal lane than `HamPacket`.** Meshtastic's LoRa hardware
  operates in license-exempt ISM bands (Part 15 US / RSS-247 Canada) — the
  same equipment-authorization regime ADR-011 already applies to the WiFi/BLE
  rows, not Part 97/RBR-4 amateur radio. This adapter therefore does **not**
  route through ADR-011's amateur gate (`LM_RF_TX_ENABLE`, call-sign ID,
  no-encrypted-flag) — it inherits the WiFi/BLE reasoning instead: certified
  module/equipment authorization, not a control-operator model. Meshtastic's
  own channel PSK/AES-256 encryption is legal to use as-is and unrelated to
  LMS1's `SIGNED_ENVELOPE` flag, which is authentication (ADR-010 invariant
  5), not encryption, and can ride underneath or alongside it without
  conflict.
- **Duty-cycle and airtime limits are Meshtastic's own** (region tables,
  `RadioInterface.cpp`), not this adapter's to encode — matching ADR-011's
  existing stance that band/power/antenna values are station- and
  jurisdiction-specific. **No EU868 (or other region) duty-cycle percentage is
  cited here.** The research pass found conflicting secondary claims (1% vs.
  10%, different sub-bands) and could not resolve them without a direct read
  of Meshtastic firmware source; that number is an explicit open parameter,
  not a design input, until re-verified against `RadioInterface.cpp`
  directly.

## Consequences

This is the lightest-weight adapter in ADR-011's table — no FEC,
interleaving, or modem work, closer in effort to WiFi/BLE than to the
HF/VHF/BPSK profiles, because Meshtastic owns PHY, FEC, relay, and ACK
itself. In exchange, LatentMesh Air gives up MTU headroom: 217 usable bytes
per packet is tighter than Air's native 240, so signed multi-update deltas
cross into multi-fragment mode sooner over Meshtastic than over WiFi/BLE. The
canonical-ABI requirement means this adapter cannot ship as Rust-only without
breaking ADR-010's cross-language conformance guarantee — the C-side enum and
golden vector are not optional follow-up, they are part of "done" for the
`WireProfile` addition itself.

## What is simulated / what is hardware-pending

| Claim | Status |
|---|---|
| `frame_mtu = 233` fragmentation and `Reassembler` round-trip against a mock `Data.payload` channel | Buildable and testable today in loopback, no hardware — same style as existing `wire.rs`/`fragment.rs` unit tests, extendable with `harness/air`'s simulator |
| `0x94 0xc3` device-API frame encode/decode against fixture bytes | Buildable and testable today, no hardware |
| `WireProfile::Meshtastic` C/Rust golden-vector parity | Buildable and testable today, no hardware — this ADR's C-side companion change |
| Real Meshtastic node connection over serial/TCP | **Not implemented** — no Meshtastic hardware present on this host |
| Real LoRa RF transmission, multi-hop relay, or ACK behavior | **Not implemented, not claimed** — Meshtastic's own firmware/hardware responsibility, out of scope for this adapter to validate |
| EU868 (or other region) duty-cycle percentage | **Unresolved parameter** — do not cite a number without a direct `RadioInterface.cpp` read first |
| Registered (non-`PRIVATE_APP`) portnum | **Not pursued** — deliberate later step, not required for this ADR |

## Implementation status

Implemented 2026-08-27, same branch. `WireProfile::Meshtastic = 9` landed on
both sides together (`crates/latentmesh-air-core/src/wire.rs` +
`c/include/latentmesh_air/frame.h`) with the shared golden vector
`wire_frame_meshtastic_v1.hex` byte-identical in both testdata trees and
wired into the C test main. `crates/latentmesh-meshtastic` exists with a
single path dependency on `latentmesh-air-core`, a hand-rolled minimal
protobuf codec (no protoc), `frame_mtu = 233` / 217 usable payload bytes as
executable assertions, and 25 loopback tests passing. Entry into ADR-014's
stage-gate table (protocol correctness → simulated link → hardware transport)
follows the same discipline as every other adapter row: simulation results do
not become an over-the-air claim, and the hardware-pending rows in the table
above remain open.
