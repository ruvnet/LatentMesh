use alloc::vec::Vec;

use crate::{crc32c, AirError, Result};

/// High nibble `0xa` identifies LatentMesh Air; low nibble is version 1.
pub const PROTOCOL_MARKER: u8 = 0xa1;
pub const FRAME_HEADER_BYTES: usize = 12;
pub const FRAME_MIN_BYTES: usize = FRAME_HEADER_BYTES + 4;
pub const FRAME_MAX_BYTES: usize = 256;
pub const FRAME_MAX_PAYLOAD: usize = FRAME_MAX_BYTES - FRAME_MIN_BYTES;

/// The low nibble of header byte 1. These values are a cross-language ABI.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WireProfile {
    Wifi = 0,
    Ble = 1,
    HfBpsk = 2,
    HfAfsk = 3,
    VhfAfsk = 4,
    VhfCpfsk = 5,
    AmAudio = 6,
    FmAudio = 7,
    HamPacket = 8,
}

impl TryFrom<u8> for WireProfile {
    type Error = AirError;

    fn try_from(value: u8) -> Result<Self> {
        Ok(match value {
            0 => Self::Wifi,
            1 => Self::Ble,
            2 => Self::HfBpsk,
            3 => Self::HfAfsk,
            4 => Self::VhfAfsk,
            5 => Self::VhfCpfsk,
            6 => Self::AmAudio,
            7 => Self::FmAudio,
            8 => Self::HamPacket,
            _ => return Err(AirError::InvalidProfile),
        })
    }
}

/// The high nibble of header byte 1.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrameFlags(u8);

impl FrameFlags {
    pub const ACK_REQUEST: Self = Self(0x10);
    pub const FEC: Self = Self(0x20);
    pub const CONTROL: Self = Self(0x40);
    /// Authentication data is carried inside the semantic envelope, never as
    /// an unbounded outer-frame trailer.
    pub const SIGNED_ENVELOPE: Self = Self(0x80);
    pub const NONE: Self = Self(0);

    pub const fn from_bits(bits: u8) -> Result<Self> {
        if bits & 0x0f != 0 {
            return Err(AirError::InvalidFlags);
        }
        Ok(Self(bits))
    }

    pub const fn bits(self) -> u8 {
        self.0
    }

    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SemanticClass {
    Telemetry = 0,
    StateDelta = 1,
    Acknowledgement = 2,
    Control = 3,
    Diagnostic = 4,
}

impl TryFrom<u8> for SemanticClass {
    type Error = AirError;

    fn try_from(value: u8) -> Result<Self> {
        Ok(match value {
            0 => Self::Telemetry,
            1 => Self::StateDelta,
            2 => Self::Acknowledgement,
            3 => Self::Control,
            4 => Self::Diagnostic,
            _ => return Err(AirError::InvalidClass),
        })
    }
}

/// Compact physical frame. The complete encoded length is `16 + payload`, so
/// every valid frame is between 16 and 256 bytes inclusive.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparseRadioFrame {
    pub profile: WireProfile,
    pub flags: FrameFlags,
    pub stream_id: u16,
    /// Message sequence. All fragments of one message share this value.
    pub sequence: u16,
    pub fragment_index: u8,
    pub fragment_count: u8,
    pub class: SemanticClass,
    /// 0 is lowest and 15 is highest.
    pub priority: u8,
    /// First 16 bits of the full critical-state hash carried in the envelope.
    pub state_tag: u16,
    pub payload: Vec<u8>,
}

impl SparseRadioFrame {
    pub fn validate(&self) -> Result<()> {
        if self.priority > 15 || self.payload.len() > FRAME_MAX_PAYLOAD {
            return Err(AirError::InvalidLength);
        }
        if self.fragment_count == 0
            || self.fragment_count > crate::MAX_FRAGMENTS
            || self.fragment_index >= self.fragment_count
        {
            return Err(AirError::InvalidFragment);
        }
        FrameFlags::from_bits(self.flags.bits())?;
        Ok(())
    }

    pub fn encoded_len(&self) -> usize {
        FRAME_MIN_BYTES + self.payload.len()
    }

    pub fn encode(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let mut out = Vec::with_capacity(self.encoded_len());
        out.push(PROTOCOL_MARKER);
        out.push(self.flags.bits() | self.profile as u8);
        out.extend_from_slice(&self.stream_id.to_be_bytes());
        out.extend_from_slice(&self.sequence.to_be_bytes());
        out.push(self.fragment_index);
        out.push(self.fragment_count);
        out.push(((self.class as u8) << 4) | self.priority);
        out.push(self.payload.len() as u8);
        out.extend_from_slice(&self.state_tag.to_be_bytes());
        out.extend_from_slice(&self.payload);
        let check = crc32c(&out);
        out.extend_from_slice(&check.to_be_bytes());
        Ok(out)
    }

    pub fn decode(input: &[u8]) -> Result<Self> {
        // Never trust payload_len before these fixed-size bounds.
        if !(FRAME_MIN_BYTES..=FRAME_MAX_BYTES).contains(&input.len()) {
            return Err(AirError::InvalidLength);
        }
        if input[0] & 0xf0 != 0xa0 {
            return Err(AirError::InvalidMarker);
        }
        if input[0] != PROTOCOL_MARKER {
            return Err(AirError::UnsupportedVersion);
        }
        let payload_len = usize::from(input[9]);
        let expected_len = FRAME_MIN_BYTES
            .checked_add(payload_len)
            .ok_or(AirError::InvalidLength)?;
        if input.len() != expected_len {
            return Err(AirError::InvalidLength);
        }
        let check_at = input.len() - 4;
        let expected = u32::from_be_bytes([
            input[check_at],
            input[check_at + 1],
            input[check_at + 2],
            input[check_at + 3],
        ]);
        if crc32c(&input[..check_at]) != expected {
            return Err(AirError::CrcMismatch);
        }
        let combined = input[1];
        let profile = WireProfile::try_from(combined & 0x0f)?;
        let flags = FrameFlags::from_bits(combined & 0xf0)?;
        let class_priority = input[8];
        let frame = Self {
            profile,
            flags,
            stream_id: u16::from_be_bytes([input[2], input[3]]),
            sequence: u16::from_be_bytes([input[4], input[5]]),
            fragment_index: input[6],
            fragment_count: input[7],
            class: SemanticClass::try_from(class_priority >> 4)?,
            priority: class_priority & 0x0f,
            state_tag: u16::from_be_bytes([input[10], input[11]]),
            payload: input[FRAME_HEADER_BYTES..check_at].to_vec(),
        };
        frame.validate()?;
        Ok(frame)
    }
}

pub const fn state_hash_tag(hash: &[u8; 16]) -> u16 {
    u16::from_be_bytes([hash[0], hash[1]])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> SparseRadioFrame {
        SparseRadioFrame {
            profile: WireProfile::HfBpsk,
            flags: FrameFlags::FEC.union(FrameFlags::ACK_REQUEST),
            stream_id: 0x1234,
            sequence: 0x0102,
            fragment_index: 0,
            fragment_count: 1,
            class: SemanticClass::StateDelta,
            priority: 15,
            state_tag: 0xbeef,
            payload: alloc::vec![0xde, 0xad, 0xbe, 0xef],
        }
    }

    #[test]
    fn frame_is_deterministic_and_bounded() {
        let encoded = sample().encode().unwrap();
        assert_eq!(encoded.len(), 20);
        assert_eq!(SparseRadioFrame::decode(&encoded).unwrap(), sample());
    }

    #[test]
    fn minimum_and_maximum_frame_sizes_are_supported() {
        let mut frame = sample();
        frame.payload.clear();
        assert_eq!(frame.encode().unwrap().len(), 16);
        frame.payload = alloc::vec![7; FRAME_MAX_PAYLOAD];
        assert_eq!(frame.encode().unwrap().len(), 256);
    }

    #[test]
    fn malformed_length_is_rejected_before_payload_copy() {
        let mut encoded = sample().encode().unwrap();
        encoded[9] = 240;
        assert_eq!(
            SparseRadioFrame::decode(&encoded),
            Err(AirError::InvalidLength)
        );
    }

    #[test]
    fn crc_detects_mutation() {
        let mut encoded = sample().encode().unwrap();
        encoded[12] ^= 1;
        assert_eq!(
            SparseRadioFrame::decode(&encoded),
            Err(AirError::CrcMismatch)
        );
    }
}
