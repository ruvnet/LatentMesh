//! The minimal `Data`/`MeshPacket`/`ToRadio`/`FromRadio` subset this adapter
//! needs, hand-encoded/decoded with `protobuf.rs` rather than generated from
//! `.proto` sources (ADR-019 leaves the protobuf-layer choice open; this
//! crate takes the "hand-roll only what's used" branch to avoid a
//! `prost`+`protoc` build dependency for four messages).
//!
//! Field numbers below are quoted from `meshtastic/protobufs`
//! `meshtastic/mesh.proto` (`Data`, `MeshPacket`) and the `ToRadio`/
//! `FromRadio` messages in the same file, and from `meshtastic/portnums.proto`
//! for `PRIVATE_APP`. `Data`'s field list was independently pulled raw in
//! the ADR-019 research pass (see
//! `docs/research/019-meshtastic-agentbbs-cognitum-research.md` §1.1); the
//! `MeshPacket`/`ToRadio`/`FromRadio` field numbers were fetched for this
//! implementation and are **not** independently byte-verified against
//! firmware source — grade them the same as the ADR's own framing-byte
//! citation (secondary, high confidence, re-verify before this is ever
//! load-bearing for a security property).
//!
//! ```proto
//! // meshtastic/mesh.proto — the fields this module reads or writes.
//! message Data {
//!   PortNum portnum = 1;
//!   bytes payload = 2;
//!   // fields 3, 4, 5, 6, 7, 8, 9, 10 exist (want_response, dest, source,
//!   // request_id, reply_id, emoji, bitfield, xeddsa_signature) — unused by
//!   // this adapter, left at their protobuf zero value on encode and
//!   // ignored (not decoded) on receive.
//! }
//! message MeshPacket {
//!   fixed32 from = 1;
//!   fixed32 to = 2;
//!   // field 3 (channel) unused, defaults to the primary channel.
//!   oneof payload_variant {
//!     Data decoded = 4;
//!     bytes encrypted = 5;
//!   }
//!   fixed32 id = 6;
//!   // fields 7-22 (rx_time, rx_snr, hop_limit, want_ack, priority, ...)
//!   // unused by this adapter.
//! }
//! message ToRadio {
//!   oneof payload_variant {
//!     MeshPacket packet = 1;
//!     // want_config_id = 3, disconnect = 4, ... unused by this adapter.
//!   }
//! }
//! message FromRadio {
//!   // id = 1 unused by this adapter.
//!   oneof payload_variant {
//!     MeshPacket packet = 2;
//!     // my_info = 3, node_info = 4, config = 5, ... unused; a `FromRadio`
//!     // carrying one of those instead of `packet` decodes to `Ok(None)`
//!     // from `decode_from_radio`, not an error — it's an administrative
//!     // message this adapter has nothing to do with, not a malformed one.
//!   }
//! }
//! ```

use crate::error::{Error, Result};
use crate::protobuf::{
    encode_bytes_field, encode_fixed32_field, encode_varint_field, for_each_field, FieldValue,
};

/// `meshtastic/portnums.proto:280-287`. "Private applications should use
/// portnums >= 256... you can use PRIVATE_APP in your code without needing
/// to rebuild protobuf files." No upstream coordination dependency; a
/// follow-up PR to register a dedicated `LATENTMESH_APP` portnum is a
/// deliberate later step, not required here (ADR-019).
pub const PRIVATE_APP_PORTNUM: u32 = 256;

const DATA_FIELD_PORTNUM: u32 = 1;
const DATA_FIELD_PAYLOAD: u32 = 2;

const MESH_PACKET_FIELD_FROM: u32 = 1;
const MESH_PACKET_FIELD_TO: u32 = 2;
const MESH_PACKET_FIELD_DECODED: u32 = 4;
const MESH_PACKET_FIELD_ENCRYPTED: u32 = 5;

const TO_RADIO_FIELD_PACKET: u32 = 1;
const TO_RADIO_FIELD_WANT_CONFIG_ID: u32 = 3;
const FROM_RADIO_FIELD_PACKET: u32 = 2;
const FROM_RADIO_FIELD_MY_INFO: u32 = 3;
const FROM_RADIO_FIELD_CONFIG_COMPLETE_ID: u32 = 7;
const FROM_RADIO_FIELD_METADATA: u32 = 13;
const MY_NODE_INFO_FIELD_MY_NODE_NUM: u32 = 1;
const METADATA_FIELD_FIRMWARE_VERSION: u32 = 1;

/// `meshtastic/portnums.proto` — the portduino simulated-radio's own
/// housekeeping portnum. Documented here, and handled by
/// [`decode_data`]/`examples/meshtasticd_interop.rs`, because of a
/// simulator-specific quirk this adapter's live interop example had to
/// work around, **not** because this adapter emits or expects it in normal
/// (real-hardware) operation: `meshtasticd`'s `sim` radio module exchanges
/// "on-air" bytes between simulated nodes over a UDP multicast socket. With
/// only one simulated node running (no peer to relay through, as in the
/// interop example's single-container setup), the node's own broadcast
/// transmission loops back through that same multicast socket and is
/// re-delivered to the local API client — but wrapped as a **new**
/// `Data { portnum: SIMULATOR_APP, payload: <the entire originally-sent
/// Data submessage, portnum+payload together, byte-identical to what was
/// transmitted> }`, rather than preserving the original application
/// portnum. Empirically verified against a live `meshtasticd` v2.7.26
/// instance (`examples/meshtasticd_interop.rs`); not documented upstream
/// and not expected on real hardware exchanging with a genuine second peer
/// node — see that example's discrepancy report for the full writeup.
pub const SIMULATOR_APP_PORTNUM: u32 = 69;

/// Meshtastic's broadcast destination (`NODENUM_BROADCAST` in firmware),
/// the correct default `to` for an adapter with no specific peer in mind.
pub const BROADCAST_ADDR: u32 = 0xffff_ffff;

/// Encodes one `Data { portnum, payload }` message body.
fn encode_data(portnum: u32, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(payload.len() + 8);
    encode_varint_field(DATA_FIELD_PORTNUM, u64::from(portnum), &mut out);
    encode_bytes_field(DATA_FIELD_PAYLOAD, payload, &mut out);
    out
}

/// Decodes one `Data { portnum, payload }` message body, the inverse of
/// [`encode_data`]. Public (unlike `encode_data`) because
/// `examples/meshtasticd_interop.rs` needs it twice: once for the normal
/// `MeshPacket.decoded` path ([`decode_mesh_packet`] already calls this
/// internally) and once more to unwrap the [`SIMULATOR_APP_PORTNUM`]
/// self-echo quirk's inner payload, which is itself another `Data`
/// submessage.
pub fn decode_data(bytes: &[u8]) -> Result<(u32, Vec<u8>)> {
    let mut portnum: Option<u32> = None;
    let mut payload: Vec<u8> = Vec::new();
    for_each_field(bytes, |field, value| {
        match (field, value) {
            (DATA_FIELD_PORTNUM, FieldValue::Varint(value)) => {
                portnum = Some(u32::try_from(value).map_err(|_| Error::InvalidLength)?);
            }
            (DATA_FIELD_PAYLOAD, FieldValue::Bytes(inner)) => payload = inner.to_vec(),
            _ => {}
        }
        Ok(())
    })?;
    Ok((portnum.unwrap_or(0), payload))
}

/// Encodes one `ToRadio { packet: MeshPacket { to, decoded: Data { portnum,
/// payload } } }` — a complete outgoing device-API protobuf body carrying
/// one Air fragment as one Meshtastic `Data.payload`.
pub fn encode_to_radio_data(to: u32, portnum: u32, payload: &[u8]) -> Vec<u8> {
    let data = encode_data(portnum, payload);
    let mut packet = Vec::with_capacity(data.len() + 12);
    // `from` (field 1) is left unset on an outgoing packet — the radio
    // fills in its own node number; only `to` (field 2) is meaningful for
    // an adapter that doesn't track its own node identity.
    encode_fixed32_field(MESH_PACKET_FIELD_TO, to, &mut packet);
    encode_bytes_field(MESH_PACKET_FIELD_DECODED, &data, &mut packet);
    let mut to_radio = Vec::with_capacity(packet.len() + 4);
    encode_bytes_field(TO_RADIO_FIELD_PACKET, &packet, &mut to_radio);
    to_radio
}

/// One `Data` payload extracted from a decoded `FromRadio`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReceivedData {
    pub from: u32,
    pub portnum: u32,
    pub payload: Vec<u8>,
}

/// Decodes a `FromRadio` protobuf body (as handed back by
/// `framing::read_frame`/`FrameDecoder::pull`) and extracts the carried
/// `Data`, if the `payload_variant` oneof is a plaintext `packet.decoded`.
/// Returns `Ok(None)` — not an error — for every other legitimate
/// `FromRadio` shape: administrative messages (`my_info`, `config`, ...),
/// and a `packet` whose `payload_variant` is `encrypted` rather than
/// `decoded` (channel-PSK-encrypted traffic this adapter cannot and should
/// not try to decrypt).
pub fn decode_from_radio(bytes: &[u8]) -> Result<Option<ReceivedData>> {
    let mut packet_bytes: Option<&[u8]> = None;
    for_each_field(bytes, |field, value| {
        if field == FROM_RADIO_FIELD_PACKET {
            if let FieldValue::Bytes(inner) = value {
                packet_bytes = Some(inner);
            }
        }
        Ok(())
    })?;
    let Some(packet_bytes) = packet_bytes else {
        return Ok(None);
    };
    decode_mesh_packet(packet_bytes)
}

fn decode_mesh_packet(bytes: &[u8]) -> Result<Option<ReceivedData>> {
    let mut from: u32 = 0;
    let mut decoded_bytes: Option<&[u8]> = None;
    let mut saw_encrypted = false;
    for_each_field(bytes, |field, value| {
        match (field, value) {
            (MESH_PACKET_FIELD_FROM, FieldValue::Fixed32(value)) => from = value,
            (MESH_PACKET_FIELD_DECODED, FieldValue::Bytes(inner)) => decoded_bytes = Some(inner),
            (MESH_PACKET_FIELD_ENCRYPTED, FieldValue::Bytes(_)) => saw_encrypted = true,
            _ => {}
        }
        Ok(())
    })?;
    let Some(decoded_bytes) = decoded_bytes else {
        return if saw_encrypted {
            Err(Error::NoDecodedData)
        } else {
            Ok(None)
        };
    };
    let (portnum, payload) = decode_data(decoded_bytes)?;
    Ok(Some(ReceivedData {
        from,
        portnum,
        payload,
    }))
}

/// Reshapes an unframed `ToRadio` buffer (as produced by
/// [`encode_to_radio_data`]) into an unframed `FromRadio` buffer carrying
/// the same `MeshPacket`. ToRadio and FromRadio disagree only on the
/// `packet` field number (1 vs. 2) and FromRadio's own top-level `id`;
/// `MeshPacket`/`Data` are shared, so this is exactly what a Meshtastic
/// node echoing our own packet back to us over the local device API would
/// produce — a loopback simulation, not a live-node claim (ADR-019: no
/// Meshtastic hardware is present on this host).
///
/// Public (not `#[cfg(test)]`) so both this crate's own tests and the
/// `examples/e2e_loopback.rs` driver used by `harness/integration` (ADR-022)
/// can exercise the same ToRadio -> FromRadio echo shape. It remains a
/// loopback-simulation aid, never a claim about real Meshtastic firmware
/// behavior.
pub fn simulate_node_echo_as_from_radio(to_radio: &[u8]) -> Vec<u8> {
    let mut packet_bytes = Vec::new();
    for_each_field(to_radio, |field, value| {
        if field == TO_RADIO_FIELD_PACKET {
            if let FieldValue::Bytes(inner) = value {
                packet_bytes = inner.to_vec();
            }
        }
        Ok(())
    })
    .unwrap();
    let mut from_radio = Vec::new();
    encode_bytes_field(FROM_RADIO_FIELD_PACKET, &packet_bytes, &mut from_radio);
    from_radio
}

/// Encodes `ToRadio { want_config_id }` (`mesh.proto` `ToRadio`
/// `payload_variant` oneof, field 3, plain `uint32`, not wrapped in a
/// `MeshPacket`) — the device-API handshake request that starts the config
/// stream a real node replies with (`my_info`, `node_info`, `config`,
/// `module_config`, `metadata`, ..., terminated by `config_complete_id`
/// echoing the same value; see [`decode_handshake_update`]). Field number
/// verified empirically against a live `meshtasticd` v2.7.26 instance
/// (`examples/meshtasticd_interop.rs`) — this crate had no handshake path
/// before that; `ToRadio.packet` (field 1, the only `ToRadio` field this
/// crate previously touched) is untouched by this addition.
pub fn encode_want_config_id(id: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(6);
    encode_varint_field(TO_RADIO_FIELD_WANT_CONFIG_ID, u64::from(id), &mut out);
    out
}

/// One piece of information [`decode_handshake_update`] can extract from a
/// `FromRadio` frame received during the `want_config_id` handshake.
/// Every other legitimate `FromRadio` shape sent during the config stream
/// (`node_info`, `config`, `module_config`, `queueStatus`, `fileInfo`, and
/// others not itemized here) decodes to [`HandshakeUpdate::Other`] —
/// administrative traffic this adapter has nothing to do with, the same
/// "unrecognized is not an error" convention [`decode_from_radio`] uses.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HandshakeUpdate {
    /// `FromRadio.my_info.my_node_num` (field 3 -> field 1) — this node's
    /// own node number, needed to address a self-directed `MeshPacket.to`.
    MyNodeNum(u32),
    /// `FromRadio.config_complete_id` (field 7) — the handshake is done
    /// once this echoes the `id` passed to [`encode_want_config_id`].
    ConfigCompleteId(u32),
    /// `FromRadio.metadata.firmware_version` (field 13 -> field 1) — not
    /// needed to drive the handshake, but useful evidence for an interop
    /// report (which real firmware build was actually exercised).
    FirmwareVersion(String),
    /// Anything else: `node_info`, `config`, `module_config`,
    /// `queueStatus`, `fileInfo`, or a field this crate doesn't itemize.
    Other,
}

/// Decodes one `FromRadio` body seen during the handshake, extracting
/// whichever of [`HandshakeUpdate`]'s three named fields (if any) it
/// carries. Field numbers (`my_info`=3, `config_complete_id`=7,
/// `metadata`=13, and the two field-1 fields nested inside `my_info`/
/// `metadata`) were verified empirically against a live `meshtasticd`
/// v2.7.26 instance's local device API, not taken from the ADR-019
/// research pass's secondary-source field list (which only covered the
/// steady-state `Data`/`MeshPacket` path — see `examples/
/// meshtasticd_interop.rs` for how the handshake fields were captured and
/// cross-checked against `docker logs`).
pub fn decode_handshake_update(bytes: &[u8]) -> Result<HandshakeUpdate> {
    let mut result = HandshakeUpdate::Other;
    for_each_field(bytes, |field, value| {
        match (field, value) {
            (FROM_RADIO_FIELD_MY_INFO, FieldValue::Bytes(inner)) => {
                for_each_field(inner, |inner_field, inner_value| {
                    if let (MY_NODE_INFO_FIELD_MY_NODE_NUM, FieldValue::Varint(node_num)) =
                        (inner_field, inner_value)
                    {
                        let node_num = u32::try_from(node_num).map_err(|_| Error::InvalidLength)?;
                        result = HandshakeUpdate::MyNodeNum(node_num);
                    }
                    Ok(())
                })?;
            }
            (FROM_RADIO_FIELD_CONFIG_COMPLETE_ID, FieldValue::Varint(id)) => {
                let id = u32::try_from(id).map_err(|_| Error::InvalidLength)?;
                result = HandshakeUpdate::ConfigCompleteId(id);
            }
            (FROM_RADIO_FIELD_METADATA, FieldValue::Bytes(inner)) => {
                for_each_field(inner, |inner_field, inner_value| {
                    if let (METADATA_FIELD_FIRMWARE_VERSION, FieldValue::Bytes(text)) =
                        (inner_field, inner_value)
                    {
                        if let Ok(text) = std::str::from_utf8(text) {
                            result = HandshakeUpdate::FirmwareVersion(text.to_string());
                        }
                    }
                    Ok(())
                })?;
            }
            _ => {}
        }
        Ok(())
    })?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_radio_round_trips_through_a_matching_from_radio_shape() {
        let to_radio =
            encode_to_radio_data(BROADCAST_ADDR, PRIVATE_APP_PORTNUM, b"air-frame-bytes");
        let from_radio = simulate_node_echo_as_from_radio(&to_radio);

        let decoded = decode_from_radio(&from_radio).unwrap().unwrap();
        assert_eq!(decoded.portnum, PRIVATE_APP_PORTNUM);
        assert_eq!(decoded.payload, b"air-frame-bytes");
    }

    #[test]
    fn administrative_from_radio_decodes_to_none_not_an_error() {
        // FromRadio.my_info (field 3), not .packet (field 2).
        let mut from_radio = Vec::new();
        encode_bytes_field(3, b"unrelated-my-node-info", &mut from_radio);
        assert_eq!(decode_from_radio(&from_radio), Ok(None));
    }

    #[test]
    fn encrypted_mesh_packet_is_reported_not_silently_dropped() {
        let mut packet = Vec::new();
        encode_bytes_field(
            MESH_PACKET_FIELD_ENCRYPTED,
            b"opaque-ciphertext",
            &mut packet,
        );
        let mut from_radio = Vec::new();
        encode_bytes_field(FROM_RADIO_FIELD_PACKET, &packet, &mut from_radio);
        assert_eq!(decode_from_radio(&from_radio), Err(Error::NoDecodedData));
    }

    #[test]
    fn data_field_layout_matches_the_documented_field_numbers() {
        let data = encode_data(PRIVATE_APP_PORTNUM, b"xy");
        // tag(1,varint)=0x08, varint(256)=0x80 0x02, tag(2,len)=0x12, len=2, "xy"
        assert_eq!(data, [0x08, 0x80, 0x02, 0x12, 0x02, b'x', b'y']);
    }

    #[test]
    fn decode_data_recovers_the_simulator_app_unwrap_shape() {
        // The SIMULATOR_APP_PORTNUM quirk's inner payload is itself a
        // complete Data submessage — decode_data must round-trip it the
        // same as any other Data body.
        let inner = encode_data(PRIVATE_APP_PORTNUM, b"wrapped-payload");
        let (portnum, payload) = decode_data(&inner).unwrap();
        assert_eq!(portnum, PRIVATE_APP_PORTNUM);
        assert_eq!(payload, b"wrapped-payload");
    }

    // The handshake fixtures below are raw FromRadio bodies captured
    // byte-for-byte from a live `meshtasticd` v2.7.26 instance's local
    // device API (127.0.0.1:4403, portduino simulated radio) responding to
    // a real `ToRadio{want_config_id}` — not hand-constructed. They pin the
    // field numbers `decode_handshake_update` relies on against actual
    // firmware output, independent of the live interop example/test that
    // captured them.

    #[test]
    fn decode_handshake_update_extracts_my_node_num_from_a_live_fixture() {
        // FromRadio.my_info for node 0xccddee01 ("!ccddee01" / "Meshtastic
        // ee01"), captured against want_config_id=42.
        let body = hex_bytes("1a140881dcf7e60c58f8eb016a066e61746976657801");
        assert_eq!(
            decode_handshake_update(&body).unwrap(),
            HandshakeUpdate::MyNodeNum(0xccddee01)
        );
    }

    #[test]
    fn decode_handshake_update_extracts_config_complete_id_from_a_live_fixture() {
        let body = hex_bytes("382a"); // FromRadio.config_complete_id = 42
        assert_eq!(
            decode_handshake_update(&body).unwrap(),
            HandshakeUpdate::ConfigCompleteId(42)
        );
    }

    #[test]
    fn decode_handshake_update_extracts_firmware_version_from_a_live_fixture() {
        // FromRadio.metadata.firmware_version = "2.7.26.54e0d8d".
        let body = hex_bytes("6a1e0a0e322e372e32362e353465306438641018200140ab064825580160806a");
        assert_eq!(
            decode_handshake_update(&body).unwrap(),
            HandshakeUpdate::FirmwareVersion("2.7.26.54e0d8d".to_string())
        );
    }

    #[test]
    fn decode_handshake_update_treats_node_info_as_other_not_an_error() {
        // FromRadio.node_info (field 4) — administrative, not one of the
        // three named handshake fields.
        let body = hex_bytes(
            "223a0881dcf7e60c122e0a09216363646465653031120f4d65736874617374696320656530311a04656530312206aabbccddee01282548001a005001",
        );
        assert_eq!(
            decode_handshake_update(&body).unwrap(),
            HandshakeUpdate::Other
        );
    }

    #[test]
    fn encode_want_config_id_matches_the_field_3_varint_shape() {
        // tag(3,varint)=0x18, varint(42)=0x2a — the exact bytes this crate
        // sent to the live node to produce the fixtures above.
        assert_eq!(encode_want_config_id(42), [0x18, 0x2a]);
    }

    /// Turns a hex string (as captured via the interop example's own
    /// hex-dump logging) back into bytes for a fixture-based test.
    fn hex_bytes(hex: &str) -> Vec<u8> {
        (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
            .collect()
    }
}
