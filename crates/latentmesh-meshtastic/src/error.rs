use std::fmt;

use latentmesh_air_core::AirError;

pub type Result<T> = std::result::Result<T, Error>;

/// Errors from either layer this crate stitches together: LatentMesh Air's
/// own framing/fragmentation (delegated, never reimplemented) and this
/// crate's own protobuf and device-API framing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// Propagated unchanged from `latentmesh-air-core`.
    Air(AirError),
    /// A protobuf varint ran off the end of the buffer, or exceeded 64 bits.
    Truncated,
    /// A length-delimited protobuf field claimed a length past the buffer.
    InvalidLength,
    /// A protobuf wire-type tag was outside the 0/1/2/5 range this crate's
    /// minimal decoder understands (3/4 are the deprecated group types).
    InvalidWireType,
    /// A decoded `FromRadio` had no `packet` field in its `payload_variant`
    /// oneof (it was a `my_info`/`config`/... administrative message).
    NoPacket,
    /// A decoded `MeshPacket` carried `encrypted` rather than `decoded` —
    /// this adapter only understands plaintext `Data` on the local device
    /// API; channel PSK decryption is the radio firmware's job, not ours.
    NoDecodedData,
    /// The device-API frame's claimed protobuf length exceeded the
    /// corruption-resync bound (512 bytes; see `framing` module docs).
    FrameTooLarge,
    /// The device-API byte stream did not start with `START1 START2`
    /// (`0x94 0xc3`) where a frame was expected.
    FrameCorrupt,
}

impl From<AirError> for Error {
    fn from(value: AirError) -> Self {
        Self::Air(value)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Air(inner) => write!(f, "latentmesh air: {inner}"),
            Self::Truncated => f.write_str("truncated protobuf varint"),
            Self::InvalidLength => f.write_str("protobuf length-delimited field out of bounds"),
            Self::InvalidWireType => f.write_str("unsupported protobuf wire type"),
            Self::NoPacket => f.write_str("FromRadio had no packet field"),
            Self::NoDecodedData => f.write_str("MeshPacket had no plaintext decoded Data"),
            Self::FrameTooLarge => f.write_str("device-API frame length exceeds 512 bytes"),
            Self::FrameCorrupt => f.write_str("device-API frame did not start with 0x94 0xc3"),
        }
    }
}

impl std::error::Error for Error {}
