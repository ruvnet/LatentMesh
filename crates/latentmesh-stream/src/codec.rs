//! The wire codec shared with the MidStream-side bridge: a 4-byte big-endian
//! length prefix followed by the serde-JSON encoding of a [`LatentFrame`].
//! JSON (not a bespoke binary format) because `LatentFrame` is already the
//! serde vocabulary of ADR-002 and the MidStream side must interoperate
//! without depending on this crate; the hard [`MAX_FRAME_BYTES`] bound is what
//! makes the choice safe on a network path.

use crate::error::StreamError;
use latentmesh_core::LatentFrame;

/// Hard upper bound on one encoded frame (prefix excluded). 1 MiB holds a
/// 4096-dim F32 payload with an order of magnitude to spare; anything larger
/// is rejected before allocation on the receive path.
pub const MAX_FRAME_BYTES: usize = 1024 * 1024;

/// Bytes of the big-endian length prefix.
pub const LENGTH_PREFIX_BYTES: usize = 4;

/// Encode one frame as `[len: u32 BE][json bytes]`. Shape-validates first,
/// so a locally-constructed inconsistent frame is caught at the sender
/// instead of by the peer's decoder.
pub fn encode_frame(frame: &LatentFrame) -> Result<Vec<u8>, StreamError> {
    validate_payload_shape(frame)?;
    let body = serde_json::to_vec(frame).map_err(|e| StreamError::Malformed(e.to_string()))?;
    if body.len() > MAX_FRAME_BYTES {
        return Err(StreamError::FrameTooLarge {
            declared: body.len(),
            max: MAX_FRAME_BYTES,
        });
    }
    let mut out = Vec::with_capacity(LENGTH_PREFIX_BYTES + body.len());
    out.extend_from_slice(&(body.len() as u32).to_be_bytes());
    out.extend_from_slice(&body);
    Ok(out)
}

/// Decode one frame from the front of `input`. Returns the frame and the
/// total bytes consumed (prefix + body). `Ok(None)` means more bytes are
/// needed; a declared length above the bound or a corrupt body is an error.
pub fn decode_frame(input: &[u8]) -> Result<Option<(LatentFrame, usize)>, StreamError> {
    if input.len() < LENGTH_PREFIX_BYTES {
        return Ok(None);
    }
    let declared = u32::from_be_bytes([input[0], input[1], input[2], input[3]]) as usize;
    if declared > MAX_FRAME_BYTES {
        return Err(StreamError::FrameTooLarge {
            declared,
            max: MAX_FRAME_BYTES,
        });
    }
    let total = LENGTH_PREFIX_BYTES + declared;
    if input.len() < total {
        return Ok(None);
    }
    let frame: LatentFrame = serde_json::from_slice(&input[LENGTH_PREFIX_BYTES..total])
        .map_err(|e| StreamError::Malformed(e.to_string()))?;
    validate_payload_shape(&frame)?;
    Ok(Some((frame, total)))
}

/// Reject frames whose payload shape is internally inconsistent — a
/// shape-mismatched frame decodes "successfully" as JSON but poisons any
/// consumer that trusts `dim` (e.g. `AlignmentTransform::apply` asserts on
/// length). Checked at the wire boundary so nothing downstream ever sees it:
/// `bytes.len()` must match `dim × bytes_per_element`, `Int8` must carry its
/// dequantization params, and those params must be finite.
pub fn validate_payload_shape(frame: &LatentFrame) -> Result<(), StreamError> {
    let payload = &frame.payload;
    let expected = payload
        .dim
        .checked_mul(payload.encoding.bytes_per_element())
        .ok_or_else(|| StreamError::Malformed("payload dim overflows".into()))?;
    if payload.bytes.len() != expected {
        return Err(StreamError::Malformed(format!(
            "payload declares dim {} ({expected} bytes at {:?}) but carries {} bytes",
            payload.dim,
            payload.encoding,
            payload.bytes.len()
        )));
    }
    match (payload.encoding, payload.int8_params) {
        (latentmesh_core::Encoding::Int8, None) => Err(StreamError::Malformed(
            "int8 payload without dequantization params".into(),
        )),
        (latentmesh_core::Encoding::Int8, Some((scale, _))) if !scale.is_finite() => {
            Err(StreamError::Malformed("int8 scale is not finite".into()))
        }
        (latentmesh_core::Encoding::F32 | latentmesh_core::Encoding::F16, Some(_)) => Err(
            StreamError::Malformed("non-int8 payload carries int8 params".into()),
        ),
        _ => Ok(()),
    }
}

/// Incremental decoder for a byte stream that arrives in arbitrary chunks
/// (the QUIC receive path). Internal buffering is hard-bounded: `push`
/// rejects growth past [`MAX_BUFFERED_BYTES`] even when the caller never
/// drains, so no usage pattern can be tricked into unbounded buffering.
#[derive(Debug, Default)]
pub struct FrameDecoder {
    buffer: Vec<u8>,
}

/// Hard cap on the decoder's internal buffer: one maximal frame in flight
/// plus one maximal frame of read-ahead.
pub const MAX_BUFFERED_BYTES: usize = 2 * (MAX_FRAME_BYTES + LENGTH_PREFIX_BYTES);

impl FrameDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Bytes currently buffered awaiting a complete frame.
    pub fn buffered(&self) -> usize {
        self.buffer.len()
    }

    /// Append received bytes. Fails fast when the head frame's declared
    /// length exceeds the bound, and refuses to grow the buffer past
    /// [`MAX_BUFFERED_BYTES`] regardless of content — call
    /// [`FrameDecoder::next_frame`] to drain.
    pub fn push(&mut self, chunk: &[u8]) -> Result<(), StreamError> {
        if self.buffer.len().saturating_add(chunk.len()) > MAX_BUFFERED_BYTES {
            self.buffer.clear();
            return Err(StreamError::Transport(
                "decoder buffer bound exceeded (caller must drain frames)".into(),
            ));
        }
        self.buffer.extend_from_slice(chunk);
        if self.buffer.len() >= LENGTH_PREFIX_BYTES {
            let declared = u32::from_be_bytes([
                self.buffer[0],
                self.buffer[1],
                self.buffer[2],
                self.buffer[3],
            ]) as usize;
            if declared > MAX_FRAME_BYTES {
                self.buffer.clear();
                return Err(StreamError::FrameTooLarge {
                    declared,
                    max: MAX_FRAME_BYTES,
                });
            }
        }
        Ok(())
    }

    /// Pop the next complete frame, if one is buffered.
    pub fn next_frame(&mut self) -> Result<Option<LatentFrame>, StreamError> {
        match decode_frame(&self.buffer)? {
            Some((frame, consumed)) => {
                self.buffer.drain(..consumed);
                Ok(Some(frame))
            }
            None => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use latentmesh_core::{Authority, Encoding, Payload, Provenance};

    fn frame(seq: u64) -> LatentFrame {
        LatentFrame {
            id: format!("f{seq}"),
            sender_model: "sender".into(),
            receiver_space: "receiver".into(),
            transform_hash: "t".into(),
            sequence: seq,
            payload: Payload::encode(&[0.25, -1.5, 3.0], Encoding::F32),
            confidence: 0.9,
            provenance: Provenance {
                sender_model: "sender".into(),
                context_hash: "c".into(),
                parents: vec![],
            },
            authority: Authority::ContextInject,
            timestamp: 1,
        }
    }

    #[test]
    fn round_trips_through_the_codec() {
        let f = frame(7);
        let bytes = encode_frame(&f).unwrap();
        let (back, consumed) = decode_frame(&bytes).unwrap().unwrap();
        assert_eq!(consumed, bytes.len());
        assert_eq!(back.sequence, 7);
        assert_eq!(back.payload.decode(), f.payload.decode());
    }

    #[test]
    fn truncated_input_asks_for_more_bytes_not_a_panic() {
        let bytes = encode_frame(&frame(1)).unwrap();
        for cut in 0..bytes.len() {
            assert_eq!(decode_frame(&bytes[..cut]).unwrap().map(|_| ()), None);
        }
    }

    #[test]
    fn oversized_declared_length_is_rejected_before_allocation() {
        let mut bytes = ((MAX_FRAME_BYTES + 1) as u32).to_be_bytes().to_vec();
        bytes.extend_from_slice(&[0u8; 8]);
        assert!(matches!(
            decode_frame(&bytes),
            Err(StreamError::FrameTooLarge { .. })
        ));
    }

    #[test]
    fn corrupt_body_is_a_malformed_error() {
        let mut bytes = encode_frame(&frame(1)).unwrap();
        let last = bytes.len() - 1;
        bytes[last] = b'X';
        assert!(matches!(
            decode_frame(&bytes),
            Err(StreamError::Malformed(_))
        ));
    }

    #[test]
    fn incremental_decoder_reassembles_across_arbitrary_chunk_boundaries() {
        let mut wire = Vec::new();
        for seq in 0..3 {
            wire.extend_from_slice(&encode_frame(&frame(seq)).unwrap());
        }
        let mut decoder = FrameDecoder::new();
        let mut seen = Vec::new();
        for chunk in wire.chunks(5) {
            decoder.push(chunk).unwrap();
            while let Some(f) = decoder.next_frame().unwrap() {
                seen.push(f.sequence);
            }
        }
        assert_eq!(seen, vec![0, 1, 2]);
        assert_eq!(decoder.buffered(), 0);
    }

    /// Craft wire bytes without `encode_frame`'s own validation, to prove
    /// the decoder independently rejects them.
    fn raw_wire(frame: &LatentFrame) -> Vec<u8> {
        let body = serde_json::to_vec(frame).unwrap();
        let mut out = (body.len() as u32).to_be_bytes().to_vec();
        out.extend_from_slice(&body);
        out
    }

    #[test]
    fn shape_mismatched_payloads_are_rejected_by_encoder_and_decoder() {
        // dim disagrees with bytes.len().
        let mut bad = frame(1);
        bad.payload.dim = 999;
        assert!(matches!(encode_frame(&bad), Err(StreamError::Malformed(_))));
        assert!(matches!(
            decode_frame(&raw_wire(&bad)),
            Err(StreamError::Malformed(_))
        ));

        // Int8 without params.
        let mut bad = frame(2);
        bad.payload = Payload::encode(&[1.0, 2.0], Encoding::Int8);
        bad.payload.int8_params = None;
        assert!(matches!(encode_frame(&bad), Err(StreamError::Malformed(_))));
        assert!(matches!(
            decode_frame(&raw_wire(&bad)),
            Err(StreamError::Malformed(_))
        ));

        // Int8 with a non-finite scale.
        let mut bad = frame(3);
        bad.payload = Payload::encode(&[1.0, 2.0], Encoding::Int8);
        bad.payload.int8_params = Some((f32::INFINITY, 0));
        assert!(matches!(encode_frame(&bad), Err(StreamError::Malformed(_))));

        // F32 carrying stray int8 params.
        let mut bad = frame(4);
        bad.payload.int8_params = Some((1.0, 0));
        assert!(matches!(encode_frame(&bad), Err(StreamError::Malformed(_))));
        assert!(matches!(
            decode_frame(&raw_wire(&bad)),
            Err(StreamError::Malformed(_))
        ));
    }

    #[test]
    fn decoder_refuses_unbounded_buffering_when_never_drained() {
        let mut decoder = FrameDecoder::new();
        let frame_bytes = encode_frame(&frame(1)).unwrap();
        let mut pushed = 0usize;
        let overflow = loop {
            match decoder.push(&frame_bytes) {
                Ok(()) => {
                    pushed += frame_bytes.len();
                    assert!(pushed <= MAX_BUFFERED_BYTES);
                }
                Err(e) => break e,
            }
        };
        assert!(matches!(overflow, StreamError::Transport(_)));
        assert_eq!(decoder.buffered(), 0);
    }

    #[test]
    fn incremental_decoder_rejects_oversized_prefix_and_clears_state() {
        let mut decoder = FrameDecoder::new();
        let bad = (u32::MAX).to_be_bytes();
        assert!(matches!(
            decoder.push(&bad),
            Err(StreamError::FrameTooLarge { .. })
        ));
        assert_eq!(decoder.buffered(), 0);
    }
}
