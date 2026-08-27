//! The adapter itself (ADR-019): fragments/reassembles an Air message using
//! `latentmesh-air-core`'s existing `fragment_message`/`Reassembler` —
//! nothing new is invented here — tuned to Meshtastic's payload budget, and
//! wraps each Air fragment as one `PRIVATE_APP` `Data.payload` inside the
//! device-API framing from `framing.rs`.

use std::collections::VecDeque;

use latentmesh_air_core::{
    fragment_message, FragmentMeta, FrameFlags, ReassembledMessage, Reassembler, ReassemblerConfig,
    SemanticClass, SparseRadioFrame, WireProfile,
};

use crate::data::{self, BROADCAST_ADDR, PRIVATE_APP_PORTNUM};
use crate::error::{Error, Result};
use crate::framing;

/// 227, not `DATA_PAYLOAD_LEN` (233, `mesh.proto`) and not
/// `latentmesh_air_core::FRAME_MAX_BYTES` (256). `DATA_PAYLOAD_LEN` bounds
/// the *encoded* `Data` submessage, not the raw payload bytes carried
/// inside it: the `portnum` varint tag+value and the `payload` bytes
/// tag+length consume protobuf field overhead (~6 bytes at this payload's
/// varint width) that a raw-payload MTU must leave headroom for. 227 is
/// also the empirically-reliable ceiling measured live against a real
/// `meshtasticd` v2.7.26 (portduino, simulated radio) instance: a binary
/// search of raw broadcast `Data.payload` sizes found ≤227 bytes round-trip
/// consistently, 228-231 bytes transmit at the radio layer but the
/// self-echo delivery back to the local API client is unreliable, and 232+
/// bytes are explicitly rejected — `ServerAPI` logs "Error=7, return NAK
/// and drop packet", where `Error=7` is `Routing.Error.TOO_LARGE`
/// (`meshtastic/protobufs`' `mesh.proto`). See
/// `examples/meshtasticd_interop.rs` for the live evidence and the
/// regression tripwire that re-checks the 232-byte rejection boundary.
/// Meshtastic does not auto-fragment application payloads (ADR-019 §1.3),
/// so this is the complete outer-frame MTU handed to `fragment_message` —
/// it already includes Air's own 16-byte frame overhead, leaving 211 usable
/// LMS1/LMAD payload bytes per Meshtastic packet.
pub const MESHTASTIC_FRAME_MTU: usize = 227;

/// One message ready to fragment and send. `profile` is deliberately not a
/// field here — this adapter always fragments with `WireProfile::Meshtastic`;
/// a caller cannot accidentally send a WiFi- or HF-profiled frame out over
/// this transport.
#[derive(Clone, Copy, Debug)]
pub struct OutgoingMessage<'a> {
    pub flags: FrameFlags,
    pub stream_id: u16,
    pub sequence: u16,
    pub class: SemanticClass,
    pub priority: u8,
    pub state_tag: u16,
    pub message: &'a [u8],
}

/// Fragments/reassembles Air messages over the Meshtastic device API.
/// Transport-agnostic: it produces and consumes complete device-API-framed
/// byte buffers, so callers decide how those bytes reach a serial port,
/// TCP socket, or (in tests) an in-memory loopback.
pub struct MeshtasticAdapter {
    reassembler: Reassembler,
    destination: u32,
}

impl MeshtasticAdapter {
    pub fn new() -> Result<Self> {
        Self::with_reassembler_config(ReassemblerConfig::default())
    }

    pub fn with_reassembler_config(config: ReassemblerConfig) -> Result<Self> {
        Ok(Self {
            reassembler: Reassembler::new(config)?,
            destination: BROADCAST_ADDR,
        })
    }

    /// Sets the Meshtastic `to` node address for subsequent
    /// `encode_message` calls. Defaults to `BROADCAST_ADDR`.
    pub fn set_destination(&mut self, to: u32) {
        self.destination = to;
    }

    /// Fragments `outgoing.message` at [`MESHTASTIC_FRAME_MTU`], wraps each
    /// resulting Air fragment as one `PRIVATE_APP` `Data.payload`, and
    /// device-API-frames it. Returns one complete, ready-to-write buffer
    /// per Meshtastic packet, in transmission order.
    pub fn encode_message(&self, outgoing: OutgoingMessage<'_>) -> Result<Vec<Vec<u8>>> {
        let meta = FragmentMeta {
            profile: WireProfile::Meshtastic,
            flags: outgoing.flags,
            stream_id: outgoing.stream_id,
            sequence: outgoing.sequence,
            class: outgoing.class,
            priority: outgoing.priority,
            state_tag: outgoing.state_tag,
        };
        let frames = fragment_message(meta, outgoing.message, MESHTASTIC_FRAME_MTU)?;
        frames
            .iter()
            .map(|frame| self.encode_one_packet(frame))
            .collect()
    }

    fn encode_one_packet(&self, frame: &SparseRadioFrame) -> Result<Vec<u8>> {
        let air_bytes = frame.encode()?;
        let to_radio =
            data::encode_to_radio_data(self.destination, PRIVATE_APP_PORTNUM, &air_bytes);
        let mut framed = Vec::new();
        framing::write_frame(&mut framed, &to_radio).map_err(|_| Error::FrameTooLarge)?;
        Ok(framed)
    }

    /// Ingests one `FromRadio` protobuf body (framing already stripped —
    /// see [`framing::read_frame`]/[`framing::FrameDecoder::pull`]).
    /// `Ok(None)` covers every case where there is nothing yet to hand
    /// back to the caller: an administrative `FromRadio`, traffic on a
    /// different portnum (another application sharing the mesh), or a
    /// still-incomplete in-flight reassembly.
    pub fn ingest_from_radio(&mut self, body: &[u8]) -> Result<Option<ReassembledMessage>> {
        let Some(received) = data::decode_from_radio(body)? else {
            return Ok(None);
        };
        if received.portnum != PRIVATE_APP_PORTNUM {
            return Ok(None);
        }
        self.ingest_air_payload(&received.payload)
    }

    /// Decodes `payload` as one [`SparseRadioFrame`] and pushes it into the
    /// in-progress reassembly, returning a completed message once every
    /// fragment of its stream has arrived. Split out from
    /// [`ingest_from_radio`] so a caller that has already extracted an Air
    /// fragment's bytes through some other path — e.g.
    /// `examples/meshtasticd_interop.rs`'s portduino-simulator
    /// `SIMULATOR_APP_PORTNUM` self-echo workaround
    /// (`data::decode_data`'s doc comment), where the real payload arrives
    /// one `Data` layer deeper than [`ingest_from_radio`] looks — can feed
    /// it straight into the same reassembler without needing its own
    /// second [`Reassembler`] instance.
    pub fn ingest_air_payload(&mut self, payload: &[u8]) -> Result<Option<ReassembledMessage>> {
        let frame = SparseRadioFrame::decode(payload)?;
        Ok(self.reassembler.push(frame)?)
    }
}

/// Convenience wrapper pairing a [`MeshtasticAdapter`] with a
/// [`framing::FrameDecoder`] for a caller that only has raw, arbitrarily
/// chunked receive bytes (e.g. a non-blocking serial read) rather than
/// already-framed `FromRadio` bodies.
pub struct StreamingReceiver {
    adapter: MeshtasticAdapter,
    decoder: framing::FrameDecoder,
    ready: VecDeque<ReassembledMessage>,
}

impl StreamingReceiver {
    pub fn new(adapter: MeshtasticAdapter) -> Self {
        Self {
            adapter,
            decoder: framing::FrameDecoder::new(),
            ready: VecDeque::new(),
        }
    }

    /// Feeds a raw chunk of received bytes; queues any messages that
    /// become complete as a result.
    pub fn push(&mut self, chunk: &[u8]) -> Result<()> {
        self.decoder.push(chunk);
        while let Some(body) = self.decoder.pull() {
            if let Some(message) = self.adapter.ingest_from_radio(&body)? {
                self.ready.push_back(message);
            }
        }
        Ok(())
    }

    pub fn next_message(&mut self) -> Option<ReassembledMessage> {
        self.ready.pop_front()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use latentmesh_air_core::{
        state_hash_tag, CriticalState, SemanticDelta, SemanticEnvelope, SymbolValue,
        FRAME_MIN_BYTES,
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

    #[test]
    fn unsigned_single_field_delta_fits_one_meshtastic_packet() {
        let delta = one_field_delta();
        let envelope = SemanticEnvelope::wrap_delta(&delta, 15, 0, None).unwrap();
        let encoded = envelope.encode().unwrap();
        // ADR-019: 48B LMS1 header + 52B LMAD fixed body + ~4-6B for one
        // SymbolUpdate ~= 104-106 bytes of LMS1+LMAD body.
        assert!(
            (104..=106).contains(&encoded.len()),
            "unsigned envelope length {}",
            encoded.len()
        );

        let adapter = MeshtasticAdapter::new().unwrap();
        let packets = adapter
            .encode_message(outgoing(
                FrameFlags::NONE,
                &encoded,
                state_hash_tag(&delta.result_hash),
            ))
            .unwrap();
        assert_eq!(packets.len(), 1, "must fit one Meshtastic packet");

        let air_frame_len = FRAME_MIN_BYTES + encoded.len();
        // ADR-019: "~120-122 bytes total" for this exact shape.
        assert!(
            (120..=122).contains(&air_frame_len),
            "Air frame length {air_frame_len}"
        );
        assert!(air_frame_len <= MESHTASTIC_FRAME_MTU);
    }

    #[test]
    fn signed_envelope_still_fits_one_meshtastic_packet_but_tightly() {
        let delta = one_field_delta();
        let envelope = SemanticEnvelope::wrap_delta(&delta, 15, 0, Some([0_u8; 64])).unwrap();
        let encoded = envelope.encode().unwrap();
        // Unsigned 104-106B + ENVELOPE_SIGNATURE_BYTES (64) = 168-170B.
        assert!(
            (168..=170).contains(&encoded.len()),
            "signed envelope length {}",
            encoded.len()
        );

        let adapter = MeshtasticAdapter::new().unwrap();
        let packets = adapter
            .encode_message(outgoing(
                FrameFlags::SIGNED_ENVELOPE,
                &encoded,
                state_hash_tag(&delta.result_hash),
            ))
            .unwrap();
        assert_eq!(packets.len(), 1, "signed delta must still fit one packet");

        let air_frame_len = FRAME_MIN_BYTES + encoded.len();
        // ADR-019: "~186 bytes — still one packet, but tight."
        assert!(
            (184..=186).contains(&air_frame_len),
            "Air frame length {air_frame_len}"
        );
        assert!(air_frame_len <= MESHTASTIC_FRAME_MTU);
    }

    /// Turns one `encode_message` output packet (device-API-framed
    /// `ToRadio`) into what would arrive back over the local device API if
    /// a Meshtastic node echoed/relayed it (device-API-framed `FromRadio`
    /// carrying the same `MeshPacket`) — see
    /// `data::simulate_node_echo_as_from_radio` for why this is
    /// representative of real node behavior rather than a test artifact.
    fn simulate_node_echo(to_radio_framed: &[u8]) -> Vec<u8> {
        let to_radio_body = framing::decode_single_frame(to_radio_framed).unwrap();
        let from_radio_body = data::simulate_node_echo_as_from_radio(to_radio_body);
        let mut framed = Vec::new();
        framing::write_frame(&mut framed, &from_radio_body).unwrap();
        framed
    }

    #[test]
    fn message_over_211_usable_bytes_fragments_and_reassembles_across_packets() {
        // 300 bytes of body forces 2 Air fragments at the 211-usable-byte
        // budget (227 - 16), which must round-trip through the full
        // encode -> device-API frame -> (simulated node echo) -> decode ->
        // reassemble pipeline.
        let message: Vec<u8> = (0..300_u32).map(|value| value as u8).collect();
        let adapter = MeshtasticAdapter::new().unwrap();
        let packets = adapter
            .encode_message(outgoing(FrameFlags::NONE, &message, 0xbeef))
            .unwrap();
        assert_eq!(packets.len(), 2, "300B / 211 usable bytes needs 2 packets");
        for framed in &packets {
            assert!(framed.len() <= framing::MAX_DEVICE_API_FRAME_LEN);
        }

        let mut receiver = StreamingReceiver::new(MeshtasticAdapter::new().unwrap());
        let mut complete = None;
        for framed in packets {
            let echoed = simulate_node_echo(&framed);
            // Simulate an arbitrary transport split: feed one byte at a
            // time through the streaming decoder rather than one packet
            // per push.
            for byte in &echoed {
                receiver.push(std::slice::from_ref(byte)).unwrap();
            }
            complete = receiver.next_message().or(complete);
        }
        assert_eq!(complete.unwrap().bytes, message);
    }

    #[test]
    fn portnum_and_profile_are_correctly_wired_on_every_outgoing_packet() {
        let adapter = MeshtasticAdapter::new().unwrap();
        let packets = adapter
            .encode_message(outgoing(FrameFlags::NONE, b"wire-check", 0x1234))
            .unwrap();
        assert_eq!(packets.len(), 1);
        let echoed = simulate_node_echo(&packets[0]);
        let body = framing::decode_single_frame(&echoed).unwrap();
        let received = data::decode_from_radio(body).unwrap().unwrap();
        assert_eq!(received.portnum, PRIVATE_APP_PORTNUM);
        let frame = SparseRadioFrame::decode(&received.payload).unwrap();
        assert_eq!(frame.profile, WireProfile::Meshtastic);
        assert_eq!(received.payload, frame.encode().unwrap());
    }

    #[test]
    fn ingest_ignores_traffic_on_a_different_portnum() {
        let to_radio =
            data::encode_to_radio_data(BROADCAST_ADDR, 1 /* TEXT_MESSAGE_APP */, b"not-ours");
        let mut framed = Vec::new();
        framing::write_frame(&mut framed, &to_radio).unwrap();
        let echoed = simulate_node_echo(&framed);
        let from_radio = framing::decode_single_frame(&echoed).unwrap();

        let mut adapter = MeshtasticAdapter::new().unwrap();
        assert_eq!(adapter.ingest_from_radio(from_radio), Ok(None));
    }

    #[test]
    fn meshtastic_profile_matches_the_air_core_golden_vector() {
        // Same golden byte sequence as
        // latentmesh-air-core/testdata/wire_frame_meshtastic_v1.hex and
        // c/tests/testdata/wire_frame_meshtastic_v1.hex (ADR-019
        // cross-language conformance case for WireProfile::Meshtastic).
        const GOLDEN_HEX: &str = "a1191234010200011f04beefdeadbeef91d70571";
        let frame = SparseRadioFrame {
            profile: WireProfile::Meshtastic,
            flags: FrameFlags::ACK_REQUEST,
            stream_id: 0x1234,
            sequence: 0x0102,
            fragment_index: 0,
            fragment_count: 1,
            class: SemanticClass::StateDelta,
            priority: 15,
            state_tag: 0xbeef,
            payload: vec![0xde, 0xad, 0xbe, 0xef],
        };
        let encoded = frame.encode().unwrap();
        let hex: String = encoded.iter().map(|byte| format!("{byte:02x}")).collect();
        assert_eq!(hex, GOLDEN_HEX);
    }
}
