//! Typed errors for every fallible streaming path. A malformed or hostile
//! input is a returned error, never a panic — the decode paths are exercised
//! with truncated, oversized, and corrupt inputs in tests.

use latentmesh_gate::AdmissionError;

/// Everything that can go wrong between raw bytes and an admitted frame.
#[derive(Clone, Debug, PartialEq)]
pub enum StreamError {
    /// Declared frame length exceeds [`crate::MAX_FRAME_BYTES`].
    FrameTooLarge { declared: usize, max: usize },
    /// The payload did not parse as a `LatentFrame`.
    Malformed(String),
    /// The frame failed gate admission (ADR-008); carries the gate's reason.
    Rejected(String),
    /// Sequence number already accepted for this stream.
    DuplicateSequence(u64),
    /// Sequence number is behind the accepted watermark.
    RegressedSequence { got: u64, watermark: u64 },
    /// The underlying transport failed (channel closed, QUIC error, EOF
    /// mid-frame).
    Transport(String),
}

impl StreamError {
    pub(crate) fn from_admission(err: &AdmissionError) -> Self {
        StreamError::Rejected(err.reason())
    }
}

impl core::fmt::Display for StreamError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            StreamError::FrameTooLarge { declared, max } => {
                write!(f, "frame of {declared} bytes exceeds the {max} byte bound")
            }
            StreamError::Malformed(why) => write!(f, "malformed frame: {why}"),
            StreamError::Rejected(reason) => write!(f, "gate rejected frame: {reason}"),
            StreamError::DuplicateSequence(seq) => {
                write!(f, "duplicate sequence {seq}")
            }
            StreamError::RegressedSequence { got, watermark } => {
                write!(f, "sequence {got} is behind watermark {watermark}")
            }
            StreamError::Transport(why) => write!(f, "transport failure: {why}"),
        }
    }
}

impl std::error::Error for StreamError {}
