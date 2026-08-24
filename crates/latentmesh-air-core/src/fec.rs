use alloc::vec;
use alloc::vec::Vec;

use crate::{AirError, Result, FRAME_MAX_BYTES};

/// Rate-1/2, constraint-length-7 code with CCSDS/NASA polynomials (171,133)o.
const G0: u8 = 0o171;
const G1: u8 = 0o133;
const MEMORY: usize = 6;
const STATES: usize = 1 << MEMORY;
pub const MAX_CODED_BITS: usize = (FRAME_MAX_BYTES * 8 + MEMORY) * 2;

pub fn bytes_to_bits_msb(bytes: &[u8]) -> Vec<u8> {
    let mut bits = Vec::with_capacity(bytes.len() * 8);
    for &byte in bytes {
        for shift in (0..8).rev() {
            bits.push((byte >> shift) & 1);
        }
    }
    bits
}

pub fn bits_to_bytes_msb(bits: &[u8]) -> Result<Vec<u8>> {
    if bits.len() % 8 != 0 || bits.iter().any(|bit| *bit > 1) {
        return Err(AirError::InvalidFec);
    }
    let mut bytes = Vec::with_capacity(bits.len() / 8);
    for chunk in bits.chunks_exact(8) {
        let mut byte = 0_u8;
        for bit in chunk {
            byte = (byte << 1) | bit;
        }
        bytes.push(byte);
    }
    Ok(bytes)
}

/// Encode bits and append six zero tail bits to terminate the trellis.
pub fn convolutional_encode(bits: &[u8]) -> Result<Vec<u8>> {
    if bits.len() > FRAME_MAX_BYTES * 8 || bits.iter().any(|bit| *bit > 1) {
        return Err(AirError::InvalidFec);
    }
    let mut output = Vec::with_capacity((bits.len() + MEMORY) * 2);
    let mut state = 0_u8;
    for bit in bits
        .iter()
        .copied()
        .chain(core::iter::repeat(0).take(MEMORY))
    {
        let register = (state << 1) | bit;
        output.push(((register & G0).count_ones() & 1) as u8);
        output.push(((register & G1).count_ones() & 1) as u8);
        state = register & (STATES as u8 - 1);
    }
    Ok(output)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodeResult {
    pub bits: Vec<u8>,
    pub path_metric: i32,
    /// Mean absolute input LLR mapped into 0..255. CRC32C remains the final
    /// acceptance gate; this value is diagnostic, not an integrity signal.
    pub confidence: u8,
}

/// Soft-input Viterbi decoder. Positive LLR means bit 1 is more likely.
pub fn soft_viterbi_decode(llrs: &[i8]) -> Result<DecodeResult> {
    if llrs.len() < MEMORY * 2 || llrs.len() > MAX_CODED_BITS || llrs.len() % 2 != 0 {
        return Err(AirError::InvalidFec);
    }
    let steps = llrs.len() / 2;
    let data_bits = steps.checked_sub(MEMORY).ok_or(AirError::InvalidFec)?;
    if data_bits % 8 != 0 {
        return Err(AirError::InvalidFec);
    }

    const NEGATIVE_INFINITY: i32 = i32::MIN / 4;
    let mut scores = [NEGATIVE_INFINITY; STATES];
    scores[0] = 0;
    let mut history: Vec<[u8; STATES]> = Vec::with_capacity(steps);

    for pair in llrs.chunks_exact(2) {
        let mut next_scores = [NEGATIVE_INFINITY; STATES];
        let mut predecessors = [u8::MAX; STATES];
        for (previous, previous_score) in scores.iter().copied().enumerate() {
            if previous_score == NEGATIVE_INFINITY {
                continue;
            }
            for input in 0..=1_u8 {
                let register = ((previous as u8) << 1) | input;
                let next = usize::from(register & (STATES as u8 - 1));
                let expected0 = ((register & G0).count_ones() & 1) as u8;
                let expected1 = ((register & G1).count_ones() & 1) as u8;
                let branch = llr_metric(pair[0], expected0) + llr_metric(pair[1], expected1);
                let candidate = previous_score.saturating_add(branch);
                if candidate > next_scores[next]
                    || (candidate == next_scores[next] && (previous as u8) < predecessors[next])
                {
                    next_scores[next] = candidate;
                    predecessors[next] = previous as u8;
                }
            }
        }
        scores = next_scores;
        history.push(predecessors);
    }

    if scores[0] == NEGATIVE_INFINITY {
        return Err(AirError::DecodeFailed);
    }
    let mut decoded = vec![0_u8; steps];
    let mut state = 0_usize;
    for step in (0..steps).rev() {
        decoded[step] = (state & 1) as u8;
        let previous = history[step][state];
        if previous == u8::MAX {
            return Err(AirError::DecodeFailed);
        }
        state = usize::from(previous);
    }
    if state != 0 || decoded[data_bits..].iter().any(|bit| *bit != 0) {
        return Err(AirError::DecodeFailed);
    }
    decoded.truncate(data_bits);
    let llr_sum: u64 = llrs.iter().map(|llr| u64::from(llr.unsigned_abs())).sum();
    let confidence = ((llr_sum * 2) / llrs.len() as u64).min(255) as u8;
    Ok(DecodeResult {
        bits: decoded,
        path_metric: scores[0],
        confidence,
    })
}

const fn llr_metric(llr: i8, expected: u8) -> i32 {
    if expected == 1 {
        llr as i32
    } else {
        -(llr as i32)
    }
}

pub fn hard_bits_to_llrs(bits: &[u8], magnitude: i8) -> Result<Vec<i8>> {
    if magnitude <= 0 || bits.iter().any(|bit| *bit > 1) {
        return Err(AirError::InvalidFec);
    }
    Ok(bits
        .iter()
        .map(|bit| if *bit == 1 { magnitude } else { -magnitude })
        .collect())
}

pub fn interleave_bits(bits: &[u8], columns: usize) -> Result<Vec<u8>> {
    if bits.iter().any(|bit| *bit > 1) {
        return Err(AirError::InvalidFec);
    }
    interleave_copy(bits, columns)
}

pub fn interleave_soft(llrs: &[i8], columns: usize) -> Result<Vec<i8>> {
    interleave_copy(llrs, columns)
}

pub fn deinterleave_soft(llrs: &[i8], columns: usize) -> Result<Vec<i8>> {
    if columns == 0 || columns > MAX_CODED_BITS {
        return Err(AirError::InvalidFec);
    }
    if llrs.is_empty() {
        return Ok(Vec::new());
    }
    let rows = llrs.len().div_ceil(columns);
    let mut output = vec![0_i8; llrs.len()];
    let mut source = 0;
    for column in 0..columns {
        for row in 0..rows {
            let target = row * columns + column;
            if target < llrs.len() {
                output[target] = llrs[source];
                source += 1;
            }
        }
    }
    Ok(output)
}

fn interleave_copy<T: Copy>(values: &[T], columns: usize) -> Result<Vec<T>> {
    if columns == 0 || columns > MAX_CODED_BITS {
        return Err(AirError::InvalidFec);
    }
    if values.is_empty() {
        return Ok(Vec::new());
    }
    let rows = values.len().div_ceil(columns);
    let mut output = Vec::with_capacity(values.len());
    for column in 0..columns {
        for row in 0..rows {
            let index = row * columns + column;
            if let Some(value) = values.get(index) {
                output.push(*value);
            }
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_zero_byte_codeword() {
        let encoded = convolutional_encode(&[0; 8]).unwrap();
        assert_eq!(encoded, alloc::vec![0; 28]);
    }

    #[test]
    fn soft_viterbi_corrects_corruption() {
        let bytes = [0xde, 0xad, 0xbe, 0xef];
        let encoded = convolutional_encode(&bytes_to_bits_msb(&bytes)).unwrap();
        let mut llrs = hard_bits_to_llrs(&encoded, 80).unwrap();
        for index in [7, 23, 41] {
            llrs[index] = -llrs[index];
        }
        let decoded = soft_viterbi_decode(&llrs).unwrap();
        assert_eq!(bits_to_bytes_msb(&decoded.bits).unwrap(), bytes);
    }

    #[test]
    fn interleaver_round_trips_irregular_length() {
        let input: Vec<i8> = (0..73).map(|value| value - 36).collect();
        let interleaved = interleave_soft(&input, 11).unwrap();
        assert_eq!(deinterleave_soft(&interleaved, 11).unwrap(), input);
    }

    #[test]
    fn decoder_rejects_unbounded_input() {
        assert_eq!(
            soft_viterbi_decode(&alloc::vec![0; MAX_CODED_BITS + 2]),
            Err(AirError::InvalidFec)
        );
    }
}
