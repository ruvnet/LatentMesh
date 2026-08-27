//! LatentMesh Air's Meshtastic transport adapter (ADR-019).
//!
//! Meshtastic is a new row in ADR-011's adapter table, not a new PHY: this
//! crate carries LMS1/LMAD unmodified and owns none of the protocol's
//! semantic or authentication logic. Its only job is to speak Meshtastic's
//! local device API (serial or TCP) well enough to submit each Air frame
//! as one `Data.payload` under portnum `PRIVATE_APP`.
//!
//! ## What is simulated vs. hardware-pending
//!
//! Everything in this crate's own `#[cfg(test)]` suite — device-API
//! framing, protobuf encode/decode, fragmentation and reassembly at
//! Meshtastic's 227-byte budget — is buildable and testable today in
//! loopback, with no Meshtastic hardware or software involved: every test
//! in this crate exercises loopback byte streams (`std::io::Cursor`,
//! in-memory `Vec<u8>`, or hand-built protobuf fixtures), none of it an
//! over-the-air claim. `examples/meshtasticd_interop.rs` and
//! `tests/live_meshtasticd.rs` are the exception: opt-in (gated on
//! `MESHTASTICD_ADDR`), they connect to a real `meshtasticd` firmware
//! build (portduino, simulated radio — still no RF, no over-the-air
//! claim) and are what pinned the 227-byte budget above empirically. Real
//! LoRa RF transmission, multi-hop relay, and ACK behavior remain
//! **not implemented/not claimed** here; those are Meshtastic's own
//! firmware/hardware responsibility (ADR-019, "What is simulated / what is
//! hardware-pending").
//!
//! ## Layering
//!
//! - [`protobuf`]: minimal varint/length-delimited/fixed32 wire-format
//!   primitives, no `.proto` codegen.
//! - [`data`]: the `Data`/`MeshPacket`/`ToRadio`/`FromRadio` message subset
//!   this adapter actually reads or writes, built on `protobuf`.
//! - [`framing`]: the `0x94 0xc3` device-API stream framing, both a
//!   blocking `Read`/`Write` form and a push/pull [`framing::FrameDecoder`]
//!   for arbitrarily chunked byte streams.
//! - [`adapter`]: [`MeshtasticAdapter`], which reuses
//!   `latentmesh_air_core::{fragment_message, Reassembler}` unchanged,
//!   tuned to [`MESHTASTIC_FRAME_MTU`], and wires the two layers above
//!   together.

mod adapter;
mod data;
mod error;
mod framing;
mod protobuf;

pub use adapter::{MeshtasticAdapter, OutgoingMessage, StreamingReceiver, MESHTASTIC_FRAME_MTU};
pub use data::{
    decode_data, decode_from_radio, decode_handshake_update, encode_to_radio_data,
    encode_want_config_id, simulate_node_echo_as_from_radio, HandshakeUpdate, ReceivedData,
    BROADCAST_ADDR, PRIVATE_APP_PORTNUM, SIMULATOR_APP_PORTNUM,
};
pub use error::{Error, Result};
pub use framing::{
    decode_single_frame, read_frame, write_frame, FrameDecoder, MAX_DEVICE_API_FRAME_LEN,
};
