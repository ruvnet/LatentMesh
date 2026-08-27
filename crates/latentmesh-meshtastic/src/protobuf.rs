//! A hand-rolled encoder/decoder for the narrow slice of protobuf wire
//! format this crate needs (ADR-019): varints, length-delimited bytes, and
//! `fixed32`. There is no `.proto` compilation step and no `prost`/`protoc`
//! dependency — see `data.rs` for exactly which fields of which Meshtastic
//! messages this covers and why that subset is sufficient.

use crate::error::{Error, Result};

pub const WIRE_VARINT: u8 = 0;
pub const WIRE_FIXED64: u8 = 1;
pub const WIRE_LEN: u8 = 2;
pub const WIRE_FIXED32: u8 = 5;

/// Protobuf's base-128 varint, LSB group first, continuation bit `0x80`.
pub fn encode_varint(mut value: u64, out: &mut Vec<u8>) {
    loop {
        let byte = (value & 0x7f) as u8;
        value >>= 7;
        if value == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

/// Returns `(value, bytes_consumed)`. Rejects a varint longer than 10 bytes
/// (the max for 64 bits) so a corrupt continuation bit can't spin forever.
pub fn decode_varint(input: &[u8]) -> Result<(u64, usize)> {
    let mut value: u64 = 0;
    for (index, &byte) in input.iter().enumerate().take(10) {
        value |= u64::from(byte & 0x7f) << (7 * index);
        if byte & 0x80 == 0 {
            return Ok((value, index + 1));
        }
    }
    Err(Error::Truncated)
}

fn encode_tag(field: u32, wire_type: u8, out: &mut Vec<u8>) {
    encode_varint((u64::from(field) << 3) | u64::from(wire_type), out);
}

/// Encodes a `varint`-typed field (protobuf's representation for `bool`,
/// every `intN`/`uintN` size, and enums).
pub fn encode_varint_field(field: u32, value: u64, out: &mut Vec<u8>) {
    encode_tag(field, WIRE_VARINT, out);
    encode_varint(value, out);
}

/// Encodes a `fixed32`-typed field. Protobuf `fixed32` is little-endian on
/// the wire regardless of host or wire-frame endianness elsewhere in
/// LatentMesh Air (Air's own frames are big-endian; this is Meshtastic's
/// convention, not ours).
pub fn encode_fixed32_field(field: u32, value: u32, out: &mut Vec<u8>) {
    encode_tag(field, WIRE_FIXED32, out);
    out.extend_from_slice(&value.to_le_bytes());
}

/// Encodes a `bytes`/embedded-message field: tag, varint length, raw bytes.
pub fn encode_bytes_field(field: u32, bytes: &[u8], out: &mut Vec<u8>) {
    encode_tag(field, WIRE_LEN, out);
    encode_varint(bytes.len() as u64, out);
    out.extend_from_slice(bytes);
}

/// One decoded field: its number, and its value in whichever wire-type
/// representation it was carried in. Callers project this into their own
/// typed view (see `data.rs`); an unrecognized field number is simply
/// ignored by the caller's `match`, which is protobuf's normal
/// forward-compatible "skip unknown fields" behavior — there is no separate
/// skip step because every wire type is already fully decoded here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldValue<'a> {
    Varint(u64),
    Fixed64(u64),
    Bytes(&'a [u8]),
    Fixed32(u32),
}

/// Reads one `(field_number, value)` pair starting at `input[0]`. Returns
/// the pair plus how many bytes it consumed.
pub fn read_field(input: &[u8]) -> Result<(u32, FieldValue<'_>, usize)> {
    let (tag, mut offset) = decode_varint(input)?;
    let field = (tag >> 3) as u32;
    let wire_type = (tag & 0x7) as u8;
    let value = match wire_type {
        WIRE_VARINT => {
            let (value, len) = decode_varint(&input[offset..])?;
            offset += len;
            FieldValue::Varint(value)
        }
        WIRE_FIXED64 => {
            let bytes: [u8; 8] = input
                .get(offset..offset + 8)
                .ok_or(Error::Truncated)?
                .try_into()
                .expect("slice of len 8");
            offset += 8;
            FieldValue::Fixed64(u64::from_le_bytes(bytes))
        }
        WIRE_LEN => {
            let (len, len_bytes) = decode_varint(&input[offset..])?;
            offset += len_bytes;
            let len = usize::try_from(len).map_err(|_| Error::InvalidLength)?;
            let end = offset.checked_add(len).ok_or(Error::InvalidLength)?;
            let slice = input.get(offset..end).ok_or(Error::InvalidLength)?;
            offset = end;
            FieldValue::Bytes(slice)
        }
        WIRE_FIXED32 => {
            let bytes: [u8; 4] = input
                .get(offset..offset + 4)
                .ok_or(Error::Truncated)?
                .try_into()
                .expect("slice of len 4");
            offset += 4;
            FieldValue::Fixed32(u32::from_le_bytes(bytes))
        }
        _ => return Err(Error::InvalidWireType),
    };
    Ok((field, value, offset))
}

/// Walks every top-level field of `input`, calling `visit(field_number,
/// value)` for each. Stops at the first decode error; a truncated or
/// malformed trailing field is surfaced to the caller rather than silently
/// dropped.
pub fn for_each_field<'a>(
    input: &'a [u8],
    mut visit: impl FnMut(u32, FieldValue<'a>) -> Result<()>,
) -> Result<()> {
    let mut offset = 0;
    while offset < input.len() {
        let (field, value, consumed) = read_field(&input[offset..])?;
        visit(field, value)?;
        offset += consumed;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varint_round_trips_multi_byte_values() {
        for value in [0_u64, 1, 127, 128, 300, 16_384, u64::MAX] {
            let mut out = Vec::new();
            encode_varint(value, &mut out);
            let (decoded, consumed) = decode_varint(&out).unwrap();
            assert_eq!(decoded, value);
            assert_eq!(consumed, out.len());
        }
    }

    #[test]
    fn truncated_varint_is_rejected() {
        assert_eq!(decode_varint(&[0x80, 0x80]), Err(Error::Truncated));
    }

    #[test]
    fn bytes_field_round_trips() {
        let mut out = Vec::new();
        encode_bytes_field(2, b"hello", &mut out);
        let (field, value, consumed) = read_field(&out).unwrap();
        assert_eq!(field, 2);
        assert_eq!(value, FieldValue::Bytes(b"hello"));
        assert_eq!(consumed, out.len());
    }

    #[test]
    fn fixed32_field_is_little_endian_on_the_wire() {
        let mut out = Vec::new();
        encode_fixed32_field(1, 0x01020304, &mut out);
        // tag byte, then little-endian 04 03 02 01.
        assert_eq!(out, [0x0d, 0x04, 0x03, 0x02, 0x01]);
    }

    #[test]
    fn oversized_length_delimited_field_is_rejected_not_panicking() {
        let mut out = Vec::new();
        encode_tag(1, WIRE_LEN, &mut out);
        encode_varint(u64::MAX, &mut out);
        assert_eq!(read_field(&out), Err(Error::InvalidLength));
    }

    #[test]
    fn for_each_field_visits_every_top_level_field_in_order() {
        let mut out = Vec::new();
        encode_varint_field(1, 42, &mut out);
        encode_bytes_field(2, b"ab", &mut out);
        let mut seen = Vec::new();
        for_each_field(&out, |field, _value| {
            seen.push(field);
            Ok(())
        })
        .unwrap();
        assert_eq!(seen, [1, 2]);
    }
}
