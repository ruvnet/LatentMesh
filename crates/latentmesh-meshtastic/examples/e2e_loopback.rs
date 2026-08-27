//! ADR-022 e2e loopback driver for `harness/integration`'s Meshtastic
//! framing suite: fragment a `SemanticEnvelope` at the 233-byte MTU
//! (ADR-019), push fragments through a mock `Data.payload` channel with the
//! `0x94 0xc3` device-API framing applied and stripped, confirm
//! `Reassembler` round-trips byte-identical — for both the single-packet
//! (~120B unsigned / ~186B signed) and multi-fragment (>217B) paths named in
//! ADR-022's Decision section.
//!
//! Prints one JSON object to stdout. No hardware, no live Meshtastic node —
//! every byte here comes from [`latentmesh_meshtastic::simulate_node_echo_as_from_radio`],
//! a loopback simulation of what a node echoing our own packet back would
//! produce (see that function's doc comment). The Node harness runner
//! (`harness/integration/scripts/run-meshtastic.mjs`) wraps this JSON in an
//! evidence-labelled receipt; this binary does not itself claim any
//! evidence label.

use latentmesh_air_core::{
    state_hash_tag, CriticalState, FrameFlags, SemanticClass, SemanticDelta, SemanticEnvelope,
    SymbolValue, FRAME_MIN_BYTES,
};
use latentmesh_meshtastic::{
    decode_from_radio, simulate_node_echo_as_from_radio, MeshtasticAdapter, OutgoingMessage,
    StreamingReceiver, MESHTASTIC_FRAME_MTU, PRIVATE_APP_PORTNUM,
};

fn one_field_delta() -> SemanticDelta {
    let mut before = CriticalState::new();
    before.set(1, SymbolValue::Bool(false)).unwrap();
    let mut after = before.clone();
    after.set(1, SymbolValue::Bool(true)).unwrap();
    SemanticDelta::between(1, 1, 1, &before, &after, vec![]).unwrap()
}

fn outgoing(flags: FrameFlags, message: &[u8], state_tag: u16) -> OutgoingMessage<'_> {
    OutgoingMessage {
        flags,
        stream_id: 7,
        sequence: 1,
        class: SemanticClass::StateDelta,
        priority: 15,
        state_tag,
        message,
    }
}

/// Turns one `encode_message` output packet (device-API-framed `ToRadio`)
/// into what would arrive back over the local device API if a Meshtastic
/// node echoed/relayed it (device-API-framed `FromRadio` carrying the same
/// `MeshPacket`) — see [`simulate_node_echo_as_from_radio`]'s doc comment
/// for why this is representative of real node behavior rather than a test
/// artifact.
fn simulate_node_echo(to_radio_framed: &[u8]) -> Vec<u8> {
    let to_radio_body = latentmesh_meshtastic::decode_single_frame(to_radio_framed).unwrap();
    let from_radio_body = simulate_node_echo_as_from_radio(to_radio_body);
    let mut framed = Vec::new();
    latentmesh_meshtastic::write_frame(&mut framed, &from_radio_body).unwrap();
    framed
}

/// Sends `packets` (device-API-framed `ToRadio` buffers) through the
/// loopback echo, feeding the result one byte at a time into a
/// [`StreamingReceiver`] to exercise the strictest possible transport split,
/// and returns the reassembled message bytes plus a portnum/profile wiring
/// check against the first echoed packet.
fn round_trip(packets: Vec<Vec<u8>>) -> (Option<Vec<u8>>, bool) {
    let mut receiver = StreamingReceiver::new(MeshtasticAdapter::new().unwrap());
    let mut portnum_ok = true;
    let mut complete = None;
    for (index, framed) in packets.iter().enumerate() {
        let echoed = simulate_node_echo(framed);
        if index == 0 {
            let body = latentmesh_meshtastic::decode_single_frame(&echoed).unwrap();
            if let Ok(Some(received)) = decode_from_radio(body) {
                portnum_ok = received.portnum == PRIVATE_APP_PORTNUM;
            } else {
                portnum_ok = false;
            }
        }
        for byte in &echoed {
            receiver.push(std::slice::from_ref(byte)).unwrap();
        }
        complete = receiver.next_message().or(complete);
    }
    (complete.map(|m| m.bytes), portnum_ok)
}

fn main() {
    let adapter = MeshtasticAdapter::new().unwrap();

    // --- Scenario 1: unsigned single-field delta, expected to fit one
    // Meshtastic packet (ADR-019: ~120-122 bytes total).
    let delta = one_field_delta();
    let unsigned_envelope = SemanticEnvelope::wrap_delta(&delta, 15, 0, None).unwrap();
    let unsigned_encoded = unsigned_envelope.encode().unwrap();
    let unsigned_air_frame_bytes = FRAME_MIN_BYTES + unsigned_encoded.len();
    let unsigned_packets = adapter
        .encode_message(outgoing(
            FrameFlags::NONE,
            &unsigned_encoded,
            state_hash_tag(&delta.result_hash),
        ))
        .unwrap();
    let unsigned_packet_count = unsigned_packets.len();
    let (unsigned_result, unsigned_portnum_ok) = round_trip(unsigned_packets);
    let unsigned_round_trip_ok = unsigned_result.as_deref() == Some(unsigned_encoded.as_slice());

    // --- Scenario 2: signed single-field delta, still one packet but tight
    // (ADR-019: ~186 bytes total).
    let signed_envelope = SemanticEnvelope::wrap_delta(&delta, 15, 0, Some([0_u8; 64])).unwrap();
    let signed_encoded = signed_envelope.encode().unwrap();
    let signed_air_frame_bytes = FRAME_MIN_BYTES + signed_encoded.len();
    let signed_packets = adapter
        .encode_message(outgoing(
            FrameFlags::SIGNED_ENVELOPE,
            &signed_encoded,
            state_hash_tag(&delta.result_hash),
        ))
        .unwrap();
    let signed_packet_count = signed_packets.len();
    let (signed_result, signed_portnum_ok) = round_trip(signed_packets);
    let signed_round_trip_ok = signed_result.as_deref() == Some(signed_encoded.as_slice());

    // --- Scenario 3: a message well over the 217-usable-byte-per-packet
    // budget, forcing multi-fragment reassembly across several packets.
    let multi_fragment_bytes = 300_usize;
    let multi_message: Vec<u8> = (0..multi_fragment_bytes as u32).map(|v| v as u8).collect();
    let multi_packets = adapter
        .encode_message(outgoing(FrameFlags::NONE, &multi_message, 0xbeef))
        .unwrap();
    let multi_fragment_packet_count = multi_packets.len();
    let (multi_result, multi_portnum_ok) = round_trip(multi_packets);
    let multi_round_trip_ok = multi_result.as_deref() == Some(multi_message.as_slice());

    let usable_bytes_per_packet = MESHTASTIC_FRAME_MTU - FRAME_MIN_BYTES;

    let output = serde_json_lite::object([
        (
            "suite",
            serde_json_lite::string("meshtastic-framing-loopback-v1"),
        ),
        ("mtu", serde_json_lite::number(MESHTASTIC_FRAME_MTU as f64)),
        (
            "usable_bytes_per_packet",
            serde_json_lite::number(usable_bytes_per_packet as f64),
        ),
        (
            "unsigned",
            serde_json_lite::object([
                (
                    "envelope_bytes",
                    serde_json_lite::number(unsigned_encoded.len() as f64),
                ),
                (
                    "air_frame_bytes",
                    serde_json_lite::number(unsigned_air_frame_bytes as f64),
                ),
                (
                    "packet_count",
                    serde_json_lite::number(unsigned_packet_count as f64),
                ),
                ("portnum_ok", serde_json_lite::boolean(unsigned_portnum_ok)),
                (
                    "round_trip_ok",
                    serde_json_lite::boolean(unsigned_round_trip_ok),
                ),
            ]),
        ),
        (
            "signed",
            serde_json_lite::object([
                (
                    "envelope_bytes",
                    serde_json_lite::number(signed_encoded.len() as f64),
                ),
                (
                    "air_frame_bytes",
                    serde_json_lite::number(signed_air_frame_bytes as f64),
                ),
                (
                    "packet_count",
                    serde_json_lite::number(signed_packet_count as f64),
                ),
                ("portnum_ok", serde_json_lite::boolean(signed_portnum_ok)),
                (
                    "round_trip_ok",
                    serde_json_lite::boolean(signed_round_trip_ok),
                ),
            ]),
        ),
        (
            "multi_fragment",
            serde_json_lite::object([
                (
                    "message_bytes",
                    serde_json_lite::number(multi_fragment_bytes as f64),
                ),
                (
                    "packet_count",
                    serde_json_lite::number(multi_fragment_packet_count as f64),
                ),
                ("portnum_ok", serde_json_lite::boolean(multi_portnum_ok)),
                (
                    "round_trip_ok",
                    serde_json_lite::boolean(multi_round_trip_ok),
                ),
            ]),
        ),
    ]);

    println!("{output}");
}

/// A tiny, dependency-free JSON writer. This crate deliberately carries no
/// `serde_json` dependency (see `Cargo.toml`) — the example only needs to
/// emit a small, fixed-shape object to stdout for the Node harness runner to
/// parse, so a minimal hand-rolled writer avoids adding a new dependency
/// edge to a crate whose only production dependency is `latentmesh-air-core`.
mod serde_json_lite {
    pub enum Value {
        String(String),
        Number(f64),
        Bool(bool),
        Object(Vec<(&'static str, Value)>),
    }

    impl std::fmt::Display for Value {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Value::String(s) => {
                    write!(f, "\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
                }
                Value::Number(n) => {
                    if n.fract() == 0.0 && n.abs() < 1e15 {
                        write!(f, "{}", *n as i64)
                    } else {
                        write!(f, "{n}")
                    }
                }
                Value::Bool(b) => write!(f, "{b}"),
                Value::Object(fields) => {
                    write!(f, "{{")?;
                    for (index, (key, value)) in fields.iter().enumerate() {
                        if index > 0 {
                            write!(f, ",")?;
                        }
                        write!(f, "\"{key}\":{value}")?;
                    }
                    write!(f, "}}")
                }
            }
        }
    }

    pub fn string(s: &str) -> Value {
        Value::String(s.to_string())
    }
    pub fn number(n: f64) -> Value {
        Value::Number(n)
    }
    pub fn boolean(b: bool) -> Value {
        Value::Bool(b)
    }
    pub fn object<const N: usize>(fields: [(&'static str, Value); N]) -> Value {
        Value::Object(fields.into_iter().collect())
    }
}
