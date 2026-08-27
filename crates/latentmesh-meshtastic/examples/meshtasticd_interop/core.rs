//! Shared driver for the live `meshtasticd` interop check (ADR-019), used
//! by both `examples/meshtasticd_interop.rs` (prints a JSON receipt) and
//! `tests/live_meshtasticd.rs` (asserts on the same report). Cargo builds
//! examples and integration tests as separate crates with no way to
//! `use` one from the other, so this file has no `main`/`#[test]` of its
//! own and is instead pulled into both callers verbatim via
//! `#[path = "meshtasticd_interop/core.rs"] mod interop_core;` (see either
//! caller) rather than duplicated. It lives in a `meshtasticd_interop/`
//! subdirectory rather than directly under `examples/` so Cargo's
//! example-autodiscovery (which scans `examples/*.rs`) does not also try
//! to build this file as its own example binary requiring a `main`.
//!
//! Connects to a live `meshtasticd` node's local device API over TCP,
//! performs the real `want_config_id` handshake
//! (`latentmesh_meshtastic::encode_want_config_id`/`decode_handshake_update`),
//! builds a real Air frame via `latentmesh-air-core` exactly as
//! `examples/e2e_loopback.rs` does, and sends it as a genuine `ToRadio`
//! `MeshPacket` — self-addressed and broadcast, single-fragment and
//! multi-fragment — decoding whatever comes back through the real
//! firmware's `FromRadio` stream. Every byte-identity assertion here is
//! against bytes the *firmware* produced and echoed, not a loopback
//! simulation (contrast `examples/e2e_loopback.rs`, which is loopback-only
//! and never opens a socket).
//!
//! `#![allow(dead_code)]`: this file is compiled twice, once per caller
//! binary, and each caller only reads a subset of [`InteropReport`]'s
//! fields (the example serializes all of them to JSON; the test asserts a
//! smaller subset) — a field/const unread by one particular caller is not
//! actually dead code overall, just unused from that caller's angle.

#![allow(dead_code)]

use std::io::{Read as _, Write as _};
use std::net::TcpStream;
use std::time::{Duration, Instant};

use latentmesh_air_core::{
    fragment_message, state_hash_tag, CriticalState, FragmentMeta, FrameFlags, SemanticClass,
    SemanticDelta, SemanticEnvelope, SymbolValue, WireProfile,
};
use latentmesh_meshtastic::{
    decode_data, decode_from_radio, decode_handshake_update, encode_to_radio_data,
    encode_want_config_id, write_frame, FrameDecoder, HandshakeUpdate, MeshtasticAdapter,
    BROADCAST_ADDR, MESHTASTIC_FRAME_MTU, PRIVATE_APP_PORTNUM, SIMULATOR_APP_PORTNUM,
};

/// Arbitrary but fixed `want_config_id` — one handshake per connection, so
/// there is no collision risk to guard against; a fixed value just makes
/// captured fixtures (see `src/data.rs`'s handshake unit tests) reproducible.
pub const WANT_CONFIG_ID: u32 = 0x1157;

/// Overall wall-clock budgets. Real firmware answers in milliseconds over
/// a loopback TCP connection; these are generous, not tuned-tight, so a
/// slow CI runner doesn't flake.
///
/// Reading uses [`FrameDecoder`] (a push/pull state machine fed by raw
/// `TcpStream::read` calls under `READ_POLL_TIMEOUT`) rather than
/// `framing::read_frame` — an earlier version of this file used
/// `read_frame` directly and it desynced in practice: `read_frame` blocks
/// until a full frame is available, but a socket read timeout firing
/// partway through one (header read, then the poll timeout elapses before
/// the body arrives) makes it return `Err` having already consumed some of
/// the frame's bytes, with no way to resume — the next call starts
/// hunting for `START1` mid-body and never finds the frame again.
/// `FrameDecoder` doesn't have this failure mode: a `push` that ends
/// mid-frame just leaves its state machine parked mid-parse, ready to
/// continue on the next `push` — a read timeout only ever means "no bytes
/// available this poll," never "discard what's been read so far."
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(15);
const SELF_ADDRESSED_LISTEN_TIMEOUT: Duration = Duration::from_secs(4);
const BROADCAST_LISTEN_TIMEOUT: Duration = Duration::from_secs(8);
const TRIPWIRE_LISTEN_TIMEOUT: Duration = Duration::from_secs(6);
const READ_POLL_TIMEOUT: Duration = Duration::from_millis(500);
/// Minimum gap before writing any packet (including the first, across
/// separate `round_trip`/tripwire calls) — good practice given `docker
/// logs` showed the portduino `sim` radio taking 1.1-1.3 seconds of
/// simulated LoRa airtime per packet at these sizes: the self-echo
/// delivery this adapter detects success from fires as soon as a packet is
/// multicast-looped-back, well *before* the radio's "Completed sending"
/// event actually frees its one-slot TX queue for the next transmission.
const FRAGMENT_SEND_GAP: Duration = Duration::from_millis(3000);

/// A raw (non-Air-framed) `Data.payload` size confirmed, by direct binary
/// search against a live `meshtasticd` v2.7.26 (portduino, simulated
/// radio) instance, to be cleanly rejected: `ServerAPI` logged "Error=7,
/// return NAK and drop packet" — `Error=7` is `Routing.Error.TOO_LARGE`
/// per `meshtastic/protobufs`' `mesh.proto`. That search is also what set
/// [`MESHTASTIC_FRAME_MTU`] to 227 (sizes ≤227 bytes round-tripped
/// consistently; 228-231 bytes were ambiguous — transmitted at the radio
/// layer per `docker logs` but the self-echo delivery back to the API
/// client was unreliable; 232 is the first size that failed cleanly and
/// consistently, so it's the one used here as a regression tripwire rather
/// than the noisier 228-231 boundary). [`mtu_ceiling_tripwire`] resends
/// this exact size on every interop run: if a future firmware version
/// starts accepting it, `routed_back` flips to `true` and that's worth
/// noticing — [`MESHTASTIC_FRAME_MTU`] might then have room to move back up.
const MTU_CEILING_TRIPWIRE_PAYLOAD_BYTES: usize = 232;

/// Reads whatever bytes are available on `stream` right now (bounded by its
/// configured read timeout — a timeout is "nothing available this poll,"
/// not an error) and returns every complete device-API frame `decoder` can
/// now produce, in arrival order. `decoder` carries any partial frame state
/// across calls, so a frame split across multiple polls (or multiple TCP
/// segments) still assembles correctly.
fn read_available_frames(stream: &mut TcpStream, decoder: &mut FrameDecoder) -> Vec<Vec<u8>> {
    let mut buf = [0_u8; 4096];
    match stream.read(&mut buf) {
        Ok(0) | Err(_) => {} // peer closed, or (typically) a poll timeout
        Ok(count) => decoder.push(&buf[..count]),
    }
    std::iter::from_fn(|| decoder.pull()).collect()
}

pub struct HandshakeReport {
    pub frames_seen: usize,
    pub my_node_num: u32,
    pub firmware_version: Option<String>,
    pub config_complete_id: u32,
}

pub struct RoundTripReport {
    pub message_bytes: usize,
    pub packet_count: usize,
    /// Any evidence of our own `PRIVATE_APP` traffic (directly or via the
    /// `SIMULATOR_APP_PORTNUM` unwrap) coming back through `FromRadio`,
    /// even if reassembly never completed.
    pub routed_back: bool,
    pub reassembled_bytes: Option<Vec<u8>>,
    pub byte_identical: bool,
}

/// Result of [`mtu_ceiling_tripwire`]: a raw, single-packet regression
/// check that the [`MTU_CEILING_TRIPWIRE_PAYLOAD_BYTES`]-byte rejection
/// boundary this crate's `MESHTASTIC_FRAME_MTU` was tuned against still
/// holds. `routed_back == false` is the expected, healthy result.
pub struct TripwireReport {
    pub payload_bytes: usize,
    pub routed_back: bool,
}

pub struct InteropReport {
    pub addr: String,
    pub handshake: HandshakeReport,
    pub self_addressed_unicast: RoundTripReport,
    pub broadcast_single_fragment: RoundTripReport,
    /// Multi-fragment broadcast at the production [`MESHTASTIC_FRAME_MTU`]
    /// (227 bytes) — required to round-trip byte-identical. This is the
    /// scenario that used to run at ADR-019's original 233-byte assumption
    /// and failed; 227 was picked specifically because live interop
    /// testing confirmed it round-trips reliably (see
    /// [`MTU_CEILING_TRIPWIRE_PAYLOAD_BYTES`]'s doc comment for the
    /// boundary search that set it).
    pub broadcast_multi_fragment: RoundTripReport,
    /// Regression tripwire for the rejection boundary
    /// [`MESHTASTIC_FRAME_MTU`] was tuned against — see
    /// [`mtu_ceiling_tripwire`].
    pub mtu_ceiling_tripwire: TripwireReport,
    pub discrepancies: Vec<String>,
}

fn one_field_delta() -> SemanticDelta {
    let mut before = CriticalState::new();
    before.set(1, SymbolValue::Bool(false)).unwrap();
    let mut after = before.clone();
    after.set(1, SymbolValue::Bool(true)).unwrap();
    SemanticDelta::between(1, 1, 1, &before, &after, vec![]).unwrap()
}

/// Runs the full handshake -> self-addressed -> broadcast(single) ->
/// broadcast(multi-fragment) -> MTU-ceiling-tripwire sequence against
/// `addr` and returns the complete report. Panics (with a message naming
/// the failing step) on a hard protocol/connection error — this function
/// is only ever called after the caller has confirmed `MESHTASTICD_ADDR`
/// is set, i.e. the caller explicitly asked for a live run and a failure
/// to talk to it at all is a real failure, not a graceful-skip case.
pub fn run_interop(addr: &str) -> InteropReport {
    let mut stream = TcpStream::connect(addr).unwrap_or_else(|e| panic!("connect to {addr}: {e}"));
    stream.set_nodelay(true).ok();

    let handshake = do_handshake(&mut stream);

    let mut discrepancies = Vec::new();

    let self_addressed_message: &[u8] = b"self-addressed-interop-probe";
    let self_addressed_packets = encode_message_with_mtu(
        handshake.my_node_num,
        self_addressed_message,
        0xf00d,
        MESHTASTIC_FRAME_MTU,
    );
    let mut self_adapter = MeshtasticAdapter::new().unwrap();
    let self_addressed_unicast = round_trip(
        &mut stream,
        &mut self_adapter,
        self_addressed_packets,
        self_addressed_message,
        SELF_ADDRESSED_LISTEN_TIMEOUT,
    );
    if !self_addressed_unicast.routed_back {
        discrepancies.push(format!(
            "ADR-019's interop task assumed a ToRadio MeshPacket addressed to the node's own \
             node number (to=0x{:08x}) would be routed back through the local API client's \
             FromRadio stream. Empirically, against meshtasticd {} (portduino simulated radio), \
             it is NOT: no FromRadio.packet carrying our PRIVATE_APP traffic was observed within \
             a {:?} listen window. Only broadcast (to=0xffffffff) routes back — see the broadcast \
             discrepancy below for how, and note that path is itself a simulator-self-echo \
             artifact, not confirmed real-hardware peer-to-peer delivery.",
            handshake.my_node_num,
            handshake.firmware_version.as_deref().unwrap_or("<unknown>"),
            SELF_ADDRESSED_LISTEN_TIMEOUT,
        ));
    }

    let delta = one_field_delta();
    let envelope = SemanticEnvelope::wrap_delta(&delta, 15, 0, None).unwrap();
    let single_fragment_message = envelope.encode().unwrap();
    let single_fragment_packets = encode_message_with_mtu(
        BROADCAST_ADDR,
        &single_fragment_message,
        state_hash_tag(&delta.result_hash),
        MESHTASTIC_FRAME_MTU,
    );
    let mut broadcast_adapter = MeshtasticAdapter::new().unwrap();
    let broadcast_single_fragment = round_trip(
        &mut stream,
        &mut broadcast_adapter,
        single_fragment_packets,
        &single_fragment_message,
        BROADCAST_LISTEN_TIMEOUT,
    );

    let multi_fragment_message: Vec<u8> = (0..300_u32).map(|value| value as u8).collect();
    let multi_fragment_packets = encode_message_with_mtu(
        BROADCAST_ADDR,
        &multi_fragment_message,
        0xbeef,
        MESHTASTIC_FRAME_MTU,
    );
    let mut multi_fragment_adapter = MeshtasticAdapter::new().unwrap();
    let broadcast_multi_fragment = round_trip(
        &mut stream,
        &mut multi_fragment_adapter,
        multi_fragment_packets,
        &multi_fragment_message,
        BROADCAST_LISTEN_TIMEOUT,
    );

    let mtu_ceiling_tripwire = mtu_ceiling_tripwire(&mut stream);

    if broadcast_single_fragment.routed_back {
        discrepancies.push(
            "The broadcast round-trip only succeeds because meshtasticd's portduino `sim` \
             radio module hears its own multicast transmission back (single-simulated-node \
             self-echo) and re-wraps the original Data submessage as the payload of a new \
             Data{portnum: SIMULATOR_APP(69)} rather than preserving portnum PRIVATE_APP(256) \
             directly. See `latentmesh_meshtastic::SIMULATOR_APP_PORTNUM`'s doc comment for the \
             full mechanism. This is a simulator quirk this adapter had to explicitly unwrap \
             (`decode_data` applied a second time); it is not documented upstream and is not \
             expected against real hardware exchanging with a genuine second peer node."
                .to_string(),
        );
    }

    if mtu_ceiling_tripwire.routed_back {
        discrepancies.push(format!(
            "REGRESSION: the {}-byte MTU-ceiling tripwire round-tripped successfully against \
             meshtasticd {}. This size was previously confirmed rejected with \
             Routing.Error.TOO_LARGE (\"Error=7, return NAK and drop packet\") during the binary \
             search that set MESHTASTIC_FRAME_MTU to 227 (see its doc comment) — if this \
             firmware/version genuinely accepts it now, MESHTASTIC_FRAME_MTU may have room to \
             move back up towards ADR-019's original 233-byte assumption and this ceiling should \
             be re-characterized.",
            mtu_ceiling_tripwire.payload_bytes,
            handshake.firmware_version.as_deref().unwrap_or("<unknown>"),
        ));
    }

    InteropReport {
        addr: addr.to_string(),
        handshake,
        self_addressed_unicast,
        broadcast_single_fragment,
        broadcast_multi_fragment,
        mtu_ceiling_tripwire,
        discrepancies,
    }
}

/// Fragments `message` at `mtu` and device-API-frames each resulting Air
/// fragment as one `ToRadio` `MeshPacket` addressed to `to` — the same two
/// steps `MeshtasticAdapter::encode_message` performs internally,
/// reimplemented here so this driver can build the exact same shape
/// `encode_message` would without needing a `MeshtasticAdapter` instance
/// just to hold a destination.
fn encode_message_with_mtu(to: u32, message: &[u8], state_tag: u16, mtu: usize) -> Vec<Vec<u8>> {
    let meta = FragmentMeta {
        profile: WireProfile::Meshtastic,
        flags: FrameFlags::NONE,
        stream_id: 7,
        sequence: 1,
        class: SemanticClass::StateDelta,
        priority: 15,
        state_tag,
    };
    fragment_message(meta, message, mtu)
        .expect("fragment_message")
        .iter()
        .map(|frame| {
            let air_bytes = frame.encode().expect("SparseRadioFrame::encode");
            let to_radio = encode_to_radio_data(to, PRIVATE_APP_PORTNUM, &air_bytes);
            let mut framed = Vec::new();
            write_frame(&mut framed, &to_radio).expect("write_frame");
            framed
        })
        .collect()
}

fn do_handshake(stream: &mut TcpStream) -> HandshakeReport {
    stream.set_read_timeout(Some(READ_POLL_TIMEOUT)).ok();
    write_frame(stream, &encode_want_config_id(WANT_CONFIG_ID))
        .expect("write ToRadio{want_config_id}");

    let mut decoder = FrameDecoder::new();
    let deadline = Instant::now() + HANDSHAKE_TIMEOUT;
    let mut frames_seen = 0;
    let mut my_node_num = None;
    let mut firmware_version = None;
    let mut config_complete_id = None;

    'poll: while Instant::now() < deadline {
        for body in read_available_frames(stream, &mut decoder) {
            frames_seen += 1;
            match decode_handshake_update(&body) {
                Ok(HandshakeUpdate::MyNodeNum(node_num)) => my_node_num = Some(node_num),
                Ok(HandshakeUpdate::FirmwareVersion(version)) => firmware_version = Some(version),
                Ok(HandshakeUpdate::ConfigCompleteId(id)) if id == WANT_CONFIG_ID => {
                    config_complete_id = Some(id);
                    break 'poll;
                }
                Ok(HandshakeUpdate::ConfigCompleteId(_) | HandshakeUpdate::Other) | Err(_) => {}
            }
        }
    }

    HandshakeReport {
        frames_seen,
        my_node_num: my_node_num
            .expect("handshake completed without a FromRadio.my_info.my_node_num"),
        firmware_version,
        config_complete_id: config_complete_id
            .expect("handshake did not see a matching FromRadio.config_complete_id in time"),
    }
}

/// Sends `packets` (already fragmented/device-API-framed by the caller —
/// see [`encode_message_with_mtu`]), then listens for up to `listen` for
/// them to come back through the real firmware's `FromRadio` stream and
/// reassemble via `adapter` into bytes equal to `message` — trying both the
/// direct `PRIVATE_APP_PORTNUM` shape a real peer/hardware node would
/// produce and the `SIMULATOR_APP_PORTNUM`-wrapped shape this specific
/// portduino simulated-radio instance actually produces (see
/// `run_interop`'s discrepancy notes).
fn round_trip(
    stream: &mut TcpStream,
    adapter: &mut MeshtasticAdapter,
    packets: Vec<Vec<u8>>,
    message: &[u8],
    listen: Duration,
) -> RoundTripReport {
    let packet_count = packets.len();
    // See FRAGMENT_SEND_GAP's doc comment: sleeping before *every* write,
    // including the first, covers both pacing this call's own fragments
    // apart from each other and leaving room for whatever a *previous*
    // round_trip/tripwire call sent to actually clear the radio's one-slot
    // TX queue before this call's first packet arrives.
    for packet in &packets {
        std::thread::sleep(FRAGMENT_SEND_GAP);
        stream.write_all(packet).expect("write ToRadio packet");
    }

    let mut decoder = FrameDecoder::new();
    let deadline = Instant::now() + listen;
    let mut routed_back = false;
    let mut reassembled_bytes = None;
    while Instant::now() < deadline && reassembled_bytes.is_none() {
        for body in read_available_frames(stream, &mut decoder) {
            let Ok(Some(received)) = decode_from_radio(&body) else {
                continue;
            };
            let air_payload = if received.portnum == PRIVATE_APP_PORTNUM {
                // The shape real hardware / a genuine peer node would produce.
                Some(received.payload)
            } else if received.portnum == SIMULATOR_APP_PORTNUM {
                // The portduino-simulator self-echo quirk (see
                // `data::SIMULATOR_APP_PORTNUM`): one more Data layer to
                // unwrap before the real Air-fragment payload is reachable.
                match decode_data(&received.payload) {
                    Ok((inner_portnum, inner_payload)) if inner_portnum == PRIVATE_APP_PORTNUM => {
                        Some(inner_payload)
                    }
                    _ => None,
                }
            } else {
                None
            };
            let Some(air_payload) = air_payload else {
                continue;
            };
            routed_back = true;
            if let Ok(Some(msg)) = adapter.ingest_air_payload(&air_payload) {
                reassembled_bytes = Some(msg.bytes);
                break;
            }
        }
    }

    let byte_identical = reassembled_bytes.as_deref() == Some(message);
    RoundTripReport {
        message_bytes: message.len(),
        packet_count,
        routed_back,
        reassembled_bytes,
        byte_identical,
    }
}

/// Sends one raw (not Air-framed — arbitrary bytes) broadcast
/// `Data.payload` of [`MTU_CEILING_TRIPWIRE_PAYLOAD_BYTES`] and checks
/// whether it comes back through the real firmware's self-echo, the same
/// way [`round_trip`] does, but without any fragmentation/reassembly
/// machinery — this is a direct probe of the firmware's own payload-size
/// rejection boundary, independent of anything this crate encodes.
fn mtu_ceiling_tripwire(stream: &mut TcpStream) -> TripwireReport {
    let payload: Vec<u8> = (0..MTU_CEILING_TRIPWIRE_PAYLOAD_BYTES as u32)
        .map(|value| value as u8)
        .collect();

    std::thread::sleep(FRAGMENT_SEND_GAP);
    let to_radio = encode_to_radio_data(BROADCAST_ADDR, PRIVATE_APP_PORTNUM, &payload);
    let mut framed = Vec::new();
    write_frame(&mut framed, &to_radio).expect("write_frame");
    stream.write_all(&framed).expect("write ToRadio packet");

    let mut decoder = FrameDecoder::new();
    let deadline = Instant::now() + TRIPWIRE_LISTEN_TIMEOUT;
    let mut routed_back = false;
    while Instant::now() < deadline && !routed_back {
        for body in read_available_frames(stream, &mut decoder) {
            let Ok(Some(received)) = decode_from_radio(&body) else {
                continue;
            };
            let echoed_payload = if received.portnum == PRIVATE_APP_PORTNUM {
                Some(received.payload)
            } else if received.portnum == SIMULATOR_APP_PORTNUM {
                match decode_data(&received.payload) {
                    Ok((inner_portnum, inner_payload)) if inner_portnum == PRIVATE_APP_PORTNUM => {
                        Some(inner_payload)
                    }
                    _ => None,
                }
            } else {
                None
            };
            if echoed_payload.as_deref() == Some(payload.as_slice()) {
                routed_back = true;
                break;
            }
        }
    }

    TripwireReport {
        payload_bytes: MTU_CEILING_TRIPWIRE_PAYLOAD_BYTES,
        routed_back,
    }
}
