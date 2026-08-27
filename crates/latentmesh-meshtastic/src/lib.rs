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
//! Everything in this crate — device-API framing, protobuf encode/decode,
//! fragmentation and reassembly at Meshtastic's 233-byte budget — is
//! buildable and testable today in loopback, with **no Meshtastic hardware
//! present on this host**. A real serial/TCP connection to a Meshtastic
//! node, and real LoRa RF transmission/multi-hop relay/ACK behavior, are
//! **not implemented** here; those are Meshtastic's own firmware/hardware
//! responsibility (ADR-019, "What is simulated / what is hardware-pending").
//! Every test in this crate exercises loopback byte streams
//! (`std::io::Cursor`, in-memory `Vec<u8>`, or hand-built protobuf
//! fixtures) — none of it is an over-the-air claim.
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
    decode_from_radio, encode_to_radio_data, simulate_node_echo_as_from_radio, ReceivedData,
    BROADCAST_ADDR, PRIVATE_APP_PORTNUM,
};
pub use error::{Error, Result};
pub use framing::{
    decode_single_frame, read_frame, write_frame, FrameDecoder, MAX_DEVICE_API_FRAME_LEN,
};
