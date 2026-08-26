//! `latentmesh-stream` — live MidStream latent streaming (ADR-015, implementing
//! ADR-004's contract): stream `z_1, z_2, …, z_t` as a sequence of
//! [`LatentFrame`]s so a downstream agent consumes partial cognitive state as
//! it is produced, instead of waiting for completion.
//!
//! Layering, receive side (every stage is fallible and bounded):
//!
//! ```text
//! bytes → codec (length-prefixed, MAX_FRAME_BYTES) → Gate::admit (ADR-008)
//!       → sequence tracking (gap / duplicate / regression)
//!       → confidence-gated authority escalation (never above the frame's own
//!         declared authority, never above the gate ceiling)
//! ```
//!
//! The `midstream-quic` feature adds [`quic::QuicFrameTransport`], which runs
//! the identical framing over a `midstreamer-quic` bidirectional stream. The
//! wire encoding is shared with the MidStream-side bridge crate via a golden
//! fixture (`testdata/latent_frame_golden.hex`) tested byte-for-byte in both
//! repositories.

pub mod codec;
pub mod error;
pub mod escalation;
pub mod receiver;
pub mod sequence;
pub mod transport;

#[cfg(feature = "midstream-quic")]
pub mod quic;

pub use codec::{
    decode_frame, encode_frame, validate_payload_shape, FrameDecoder, MAX_BUFFERED_BYTES,
    MAX_FRAME_BYTES,
};
pub use error::StreamError;
pub use escalation::{AuthorityEscalator, EscalationConfig};
pub use receiver::{AdmittedFrame, LatentStreamReceiver};
pub use sequence::{LatentStreamSender, SequenceEvent, SequenceTracker};
pub use transport::{ChannelTransport, FrameTransport};

pub use latentmesh_core::{Authority, Encoding, LatentFrame, Payload, Provenance};
