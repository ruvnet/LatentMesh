use alloc::vec::Vec;

use latentmesh_air_core::{
    bits_to_bytes_msb, bytes_to_bits_msb, convolutional_encode, deinterleave_soft, interleave_bits,
    soft_viterbi_decode, AirError, Result, MAX_CODED_BITS,
};

pub const PHY_SYNC: u32 = 0xd391_c5a7;
const SYNC_BITS: usize = 32;
const LENGTH_BITS_REPEATED: usize = 16 * 3;
const PHY_PREFIX_BITS: usize = SYNC_BITS + LENGTH_BITS_REPEATED;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BurstCodec {
    interleaver_columns: usize,
}

impl BurstCodec {
    pub fn new(interleaver_columns: usize) -> Result<Self> {
        if interleaver_columns == 0 || interleaver_columns > 256 {
            return Err(AirError::InvalidLength);
        }
        Ok(Self {
            interleaver_columns,
        })
    }

    pub const fn interleaver_columns(&self) -> usize {
        self.interleaver_columns
    }

    /// Sync, triply repeated coded-length header, then rate-1/2 convolutional
    /// coding and block interleaving. Returned bytes are one bit each (0/1).
    pub fn encode(&self, frame_bytes: &[u8]) -> Result<Vec<u8>> {
        let coded = convolutional_encode(&bytes_to_bits_msb(frame_bytes))?;
        let interleaved = interleave_bits(&coded, self.interleaver_columns)?;
        let coded_len = u16::try_from(interleaved.len()).map_err(|_| AirError::LimitExceeded)?;
        let mut burst = Vec::with_capacity(PHY_PREFIX_BITS + interleaved.len());
        for shift in (0..32).rev() {
            burst.push(((PHY_SYNC >> shift) & 1) as u8);
        }
        for shift in (0..16).rev() {
            let bit = ((coded_len >> shift) & 1) as u8;
            burst.extend_from_slice(&[bit, bit, bit]);
        }
        burst.extend_from_slice(&interleaved);
        Ok(burst)
    }

    /// Decode an aligned burst. Up to three sync errors and one error in each
    /// repeated length triplet are tolerated.
    pub fn decode_aligned(&self, llrs: &[i8]) -> Result<Vec<u8>> {
        if llrs.len() < PHY_PREFIX_BITS {
            return Err(AirError::Truncated);
        }
        let mut sync_errors = 0;
        for (index, llr) in llrs[..SYNC_BITS].iter().enumerate() {
            let expected = ((PHY_SYNC >> (31 - index)) & 1) != 0;
            if (*llr > 0) != expected {
                sync_errors += 1;
            }
        }
        if sync_errors > 3 {
            return Err(AirError::InvalidMarker);
        }
        let coded_len = decode_repeated_length(&llrs[SYNC_BITS..PHY_PREFIX_BITS])?;
        if !(12..=MAX_CODED_BITS).contains(&coded_len) || coded_len % 2 != 0 {
            return Err(AirError::InvalidLength);
        }
        if llrs.len() != PHY_PREFIX_BITS + coded_len {
            return Err(AirError::InvalidLength);
        }
        self.decode_coded(&llrs[PHY_PREFIX_BITS..])
    }

    fn decode_coded(&self, llrs: &[i8]) -> Result<Vec<u8>> {
        let ordered = deinterleave_soft(llrs, self.interleaver_columns)?;
        let decoded = soft_viterbi_decode(&ordered)?;
        bits_to_bytes_msb(&decoded.bits)
    }
}

fn decode_repeated_length(llrs: &[i8]) -> Result<usize> {
    if llrs.len() != LENGTH_BITS_REPEATED {
        return Err(AirError::InvalidLength);
    }
    let mut value = 0_u16;
    for triplet in llrs.chunks_exact(3) {
        let votes = triplet.iter().filter(|llr| **llr > 0).count();
        value = (value << 1) | u16::from(votes >= 2);
    }
    Ok(usize::from(value))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SoftBurstState {
    Searching,
    ReadingHeader,
    ReadingPayload { received: usize, expected: usize },
}

/// Streaming bit/LLR burst deframer with bounded buffers.
#[derive(Clone, Debug)]
pub struct SoftBurstReceiver {
    codec: BurstCodec,
    shift: u32,
    seen: usize,
    header: Vec<i8>,
    payload: Vec<i8>,
    expected: Option<usize>,
}

impl SoftBurstReceiver {
    pub fn new(codec: BurstCodec) -> Self {
        Self {
            codec,
            shift: 0,
            seen: 0,
            header: Vec::with_capacity(LENGTH_BITS_REPEATED),
            payload: Vec::new(),
            expected: None,
        }
    }

    pub fn state(&self) -> SoftBurstState {
        match self.expected {
            Some(expected) => SoftBurstState::ReadingPayload {
                received: self.payload.len(),
                expected,
            },
            None if !self.header.is_empty()
                || (self.seen >= SYNC_BITS && self.shift == PHY_SYNC) =>
            {
                SoftBurstState::ReadingHeader
            }
            None => SoftBurstState::Searching,
        }
    }

    pub fn reset(&mut self) {
        self.shift = 0;
        self.seen = 0;
        self.header.clear();
        self.payload.clear();
        self.expected = None;
    }

    pub fn push_llr(&mut self, llr: i8) -> Result<Option<Vec<u8>>> {
        if let Some(expected) = self.expected {
            self.payload.push(llr);
            if self.payload.len() == expected {
                let result = self.codec.decode_coded(&self.payload);
                self.reset();
                return result.map(Some);
            }
            return Ok(None);
        }

        if self.header.is_empty() {
            self.shift = (self.shift << 1) | u32::from(llr > 0);
            self.seen = self.seen.saturating_add(1);
            if self.seen < SYNC_BITS || self.shift != PHY_SYNC {
                return Ok(None);
            }
            // A sentinel distinguishes header collection from sync search.
            self.header.reserve(LENGTH_BITS_REPEATED);
            self.header.push(i8::MIN);
            return Ok(None);
        }

        if self.header[0] == i8::MIN {
            self.header.clear();
        }
        self.header.push(llr);
        if self.header.len() == LENGTH_BITS_REPEATED {
            let expected = decode_repeated_length(&self.header)?;
            if !(12..=MAX_CODED_BITS).contains(&expected) || expected % 2 != 0 {
                self.reset();
                return Err(AirError::InvalidLength);
            }
            self.header.clear();
            self.payload = Vec::with_capacity(expected);
            self.expected = Some(expected);
        }
        Ok(None)
    }

    pub fn push_slice(&mut self, llrs: &[i8]) -> Result<Vec<Vec<u8>>> {
        let mut frames = Vec::new();
        for llr in llrs {
            if let Some(frame) = self.push_llr(*llr)? {
                frames.push(frame);
            }
        }
        Ok(frames)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn llrs(bits: &[u8]) -> Vec<i8> {
        bits.iter()
            .map(|bit| if *bit == 1 { 90 } else { -90 })
            .collect()
    }

    #[test]
    fn aligned_burst_round_trip_with_errors() {
        let codec = BurstCodec::new(17).unwrap();
        let frame = [0xa1, 0x22, 0, 1, 0, 1, 0, 1, 0x1f, 0, 0, 0, 1, 2, 3, 4];
        let bits = codec.encode(&frame).unwrap();
        let mut soft = llrs(&bits);
        // One sync error, one repeated-header error and three coded errors.
        for index in [2, 34, 91, 137, 201] {
            soft[index] = -soft[index];
        }
        assert_eq!(codec.decode_aligned(&soft).unwrap(), frame);
    }

    #[test]
    fn streaming_receiver_finds_burst_after_noise_bits() {
        let codec = BurstCodec::new(11).unwrap();
        let frame = [7_u8; 20];
        let bits = codec.encode(&frame).unwrap();
        let mut soft = alloc::vec![-80, 80, -80, -80, 80];
        soft.extend(llrs(&bits));
        let mut receiver = SoftBurstReceiver::new(codec);
        let frames = receiver.push_slice(&soft).unwrap();
        assert_eq!(frames, alloc::vec![frame.to_vec()]);
        assert_eq!(receiver.state(), SoftBurstState::Searching);
    }
}
