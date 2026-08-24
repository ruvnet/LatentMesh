use core::fmt;

pub type Result<T> = core::result::Result<T, AirError>;

/// Errors are deliberately coarse enough to expose over untrusted links
/// without turning the parser into an oracle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AirError {
    InvalidMarker,
    UnsupportedVersion,
    InvalidProfile,
    InvalidClass,
    InvalidFlags,
    InvalidLength,
    LimitExceeded,
    Truncated,
    TrailingBytes,
    CrcMismatch,
    InvalidFragment,
    FragmentConflict,
    ReassemblyFull,
    Replay,
    TooOld,
    InvalidSemanticValue,
    BaseStateMismatch,
    ResultStateMismatch,
    AuthenticationRequired,
    AuthenticationFailed,
    InvalidFec,
    DecodeFailed,
}

impl fmt::Display for AirError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidMarker => "invalid LatentMesh Air marker",
            Self::UnsupportedVersion => "unsupported LatentMesh Air version",
            Self::InvalidProfile => "invalid radio profile",
            Self::InvalidClass => "invalid semantic class",
            Self::InvalidFlags => "invalid frame flags",
            Self::InvalidLength => "invalid length",
            Self::LimitExceeded => "configured limit exceeded",
            Self::Truncated => "truncated input",
            Self::TrailingBytes => "trailing input bytes",
            Self::CrcMismatch => "CRC32C mismatch",
            Self::InvalidFragment => "invalid fragment metadata",
            Self::FragmentConflict => "conflicting duplicate fragment",
            Self::ReassemblyFull => "reassembly table full",
            Self::Replay => "replayed sequence",
            Self::TooOld => "sequence is outside the replay window",
            Self::InvalidSemanticValue => "invalid semantic value",
            Self::BaseStateMismatch => "semantic delta base state mismatch",
            Self::ResultStateMismatch => "semantic delta result state mismatch",
            Self::AuthenticationRequired => "signed envelope requires authentication",
            Self::AuthenticationFailed => "signed envelope authentication failed",
            Self::InvalidFec => "invalid convolutional codeword",
            Self::DecodeFailed => "decoder could not produce a valid result",
        })
    }
}

#[cfg(feature = "std")]
impl std::error::Error for AirError {}
