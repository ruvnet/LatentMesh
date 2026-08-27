//! Meshtastic's local device-API stream framing (ADR-019, serial or TCP):
//! `START1 = 0x94`, `START2 = 0xc3`, a 2-byte big-endian protobuf length,
//! then that many bytes of `ToRadio`/`FromRadio` protobuf. Source:
//! `meshtastic.org/docs/development/device/client-api/` — a secondary,
//! summarizer-fetched source (not independently byte-verified against
//! firmware); grade accordingly if this framing is ever load-bearing for a
//! security property (ADR-019's own caveat, carried forward here).
//!
//! Two ways to use it, both loopback-testable without hardware:
//! - [`write_frame`]/[`read_frame`] over any [`std::io::Write`]/[`std::io::Read`]
//!   (a real serial port or TCP socket, or an in-memory `Vec<u8>`/`Cursor`
//!   in tests).
//! - [`FrameDecoder`], a push/pull state machine for callers that receive
//!   bytes in arbitrary chunks (e.g. a non-blocking socket) rather than
//!   through a blocking `Read`.

use std::io::{self, Read, Write};

use crate::error::{Error, Result};

pub const START1: u8 = 0x94;
pub const START2: u8 = 0xc3;

/// Meshtastic's own receiver resyncs on `START1` if the claimed length
/// exceeds this; used as this crate's corruption bound too, both to reject
/// obviously-wrong lengths early and to cap how much a malicious or
/// corrupted peer can make a `FrameDecoder` buffer before resyncing.
pub const MAX_DEVICE_API_FRAME_LEN: usize = 512;

/// Prepends the `START1 START2 <len_be>` header to `protobuf_bytes` and
/// writes the whole frame to `writer` in one call.
pub fn write_frame<W: Write>(writer: &mut W, protobuf_bytes: &[u8]) -> io::Result<()> {
    if protobuf_bytes.len() > MAX_DEVICE_API_FRAME_LEN {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "protobuf body exceeds MAX_DEVICE_API_FRAME_LEN",
        ));
    }
    let len = protobuf_bytes.len() as u16;
    let mut header = [START1, START2, 0, 0];
    header[2..4].copy_from_slice(&len.to_be_bytes());
    writer.write_all(&header)?;
    writer.write_all(protobuf_bytes)?;
    Ok(())
}

/// Blocks until one complete frame is read from `reader`, resyncing past
/// stray bytes that aren't `START1` the way Meshtastic's own receiver does.
/// Returns the protobuf body only (framing header stripped).
pub fn read_frame<R: Read>(reader: &mut R) -> io::Result<Vec<u8>> {
    let mut byte = [0_u8; 1];
    'outer: loop {
        loop {
            reader.read_exact(&mut byte)?;
            if byte[0] == START1 {
                break;
            }
        }
        // Seeking START2: a repeated START1 here is a new candidate start,
        // not garbage to fall all the way back through — matches
        // FrameDecoder's state machine.
        loop {
            reader.read_exact(&mut byte)?;
            if byte[0] == START2 {
                break;
            }
            if byte[0] != START1 {
                continue 'outer;
            }
        }
        let mut len_bytes = [0_u8; 2];
        reader.read_exact(&mut len_bytes)?;
        let len = usize::from(u16::from_be_bytes(len_bytes));
        if len > MAX_DEVICE_API_FRAME_LEN {
            // Corrupt claimed length: resync by resuming the byte-at-a-time
            // START1 search rather than trusting it as a skip count.
            continue 'outer;
        }
        let mut body = vec![0_u8; len];
        reader.read_exact(&mut body)?;
        return Ok(body);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    SeekingStart1,
    SeekingStart2,
    ReadingLenHigh,
    ReadingLenLow { high: u8 },
    ReadingBody { remaining: usize },
}

/// A push/pull framer for byte streams that don't arrive as one blocking
/// `Read` — feed it whatever chunk boundaries the transport hands you
/// (including a stream split mid-header or mid-body across multiple
/// `push` calls) and pull complete protobuf bodies back out in order.
#[derive(Debug)]
pub struct FrameDecoder {
    state: State,
    body: Vec<u8>,
    ready: std::collections::VecDeque<Vec<u8>>,
}

impl Default for FrameDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl FrameDecoder {
    pub fn new() -> Self {
        Self {
            state: State::SeekingStart1,
            body: Vec::new(),
            ready: std::collections::VecDeque::new(),
        }
    }

    /// Feeds an arbitrary chunk of bytes, however it was split by the
    /// transport. Never fails: corrupt bytes are absorbed by resyncing on
    /// the next `START1`, mirroring Meshtastic's own receiver.
    pub fn push(&mut self, chunk: &[u8]) {
        for &byte in chunk {
            self.push_byte(byte);
        }
    }

    fn push_byte(&mut self, byte: u8) {
        self.state = match self.state {
            State::SeekingStart1 => {
                if byte == START1 {
                    State::SeekingStart2
                } else {
                    State::SeekingStart1
                }
            }
            State::SeekingStart2 => {
                if byte == START2 {
                    State::ReadingLenHigh
                } else if byte == START1 {
                    State::SeekingStart2
                } else {
                    State::SeekingStart1
                }
            }
            State::ReadingLenHigh => State::ReadingLenLow { high: byte },
            State::ReadingLenLow { high } => {
                let len = usize::from(u16::from_be_bytes([high, byte]));
                if len == 0 {
                    self.ready.push_back(Vec::new());
                    State::SeekingStart1
                } else if len > MAX_DEVICE_API_FRAME_LEN {
                    // Corrupt claimed length: drop this candidate frame and
                    // resync, exactly like the blocking `read_frame` path.
                    State::SeekingStart1
                } else {
                    self.body.clear();
                    self.body.reserve(len);
                    State::ReadingBody { remaining: len }
                }
            }
            State::ReadingBody { remaining } => {
                self.body.push(byte);
                if remaining == 1 {
                    self.ready.push_back(std::mem::take(&mut self.body));
                    State::SeekingStart1
                } else {
                    State::ReadingBody {
                        remaining: remaining - 1,
                    }
                }
            }
        };
    }

    /// Returns the next complete protobuf body, if one has arrived, in the
    /// order frames were pushed.
    pub fn pull(&mut self) -> Option<Vec<u8>> {
        self.ready.pop_front()
    }
}

/// One-shot decode of a single framed buffer already known to be exactly
/// one frame (header + body, no trailing bytes). Rejects anything else as
/// [`Error::FrameCorrupt`]/[`Error::FrameTooLarge`] rather than silently
/// accepting a truncated or over-long buffer.
pub fn decode_single_frame(framed: &[u8]) -> Result<&[u8]> {
    let [s1, s2, len_hi, len_lo, body @ ..] = framed else {
        return Err(Error::FrameCorrupt);
    };
    if *s1 != START1 || *s2 != START2 {
        return Err(Error::FrameCorrupt);
    }
    let len = usize::from(u16::from_be_bytes([*len_hi, *len_lo]));
    if len > MAX_DEVICE_API_FRAME_LEN || len != body.len() {
        return Err(Error::FrameTooLarge);
    }
    Ok(body)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_then_read_frame_round_trips_over_a_cursor() {
        let mut stream: Vec<u8> = Vec::new();
        write_frame(&mut stream, b"hello-protobuf").unwrap();
        let mut cursor = std::io::Cursor::new(stream);
        let body = read_frame(&mut cursor).unwrap();
        assert_eq!(body, b"hello-protobuf");
    }

    #[test]
    fn read_frame_resyncs_past_leading_garbage() {
        let mut stream: Vec<u8> = vec![0x00, 0xff, START1]; // stray byte, then a bare START1
        write_frame(&mut stream, b"payload").unwrap();
        let mut cursor = std::io::Cursor::new(stream);
        assert_eq!(read_frame(&mut cursor).unwrap(), b"payload");
    }

    #[test]
    fn frame_decoder_handles_a_single_push_of_one_full_frame() {
        let mut framed = Vec::new();
        write_frame(&mut framed, b"one-shot").unwrap();
        let mut decoder = FrameDecoder::new();
        decoder.push(&framed);
        assert_eq!(decoder.pull().unwrap(), b"one-shot");
        assert!(decoder.pull().is_none());
    }

    #[test]
    fn frame_decoder_handles_the_stream_split_at_every_byte_boundary() {
        let mut framed = Vec::new();
        write_frame(&mut framed, b"split-me-anywhere").unwrap();
        // The strictest form of "split reads": one byte per push call,
        // covering header/length/body boundaries all falling mid-chunk.
        let mut decoder = FrameDecoder::new();
        for byte in &framed {
            decoder.push(std::slice::from_ref(byte));
        }
        assert_eq!(decoder.pull().unwrap(), b"split-me-anywhere");
    }

    #[test]
    fn frame_decoder_recovers_multiple_frames_from_one_chunk() {
        let mut framed = Vec::new();
        write_frame(&mut framed, b"first").unwrap();
        write_frame(&mut framed, b"second").unwrap();
        let mut decoder = FrameDecoder::new();
        decoder.push(&framed);
        assert_eq!(decoder.pull().unwrap(), b"first");
        assert_eq!(decoder.pull().unwrap(), b"second");
        assert!(decoder.pull().is_none());
    }

    #[test]
    fn frame_decoder_resyncs_past_a_corrupt_oversized_length() {
        let mut stream = vec![START1, START2, 0xff, 0xff]; // claims 65535 bytes
        write_frame(&mut stream, b"recovered").unwrap();
        let mut decoder = FrameDecoder::new();
        decoder.push(&stream);
        assert_eq!(decoder.pull().unwrap(), b"recovered");
    }

    #[test]
    fn frame_decoder_resyncs_past_a_stray_start1_inside_seeking_start2() {
        // START1, START1, START2, len, body — the first START1 is a false
        // start; the decoder must not get stuck waiting for a START2 that
        // will never directly follow the first byte.
        let mut stream = vec![START1];
        write_frame(&mut stream, b"after-false-start").unwrap();
        let mut decoder = FrameDecoder::new();
        decoder.push(&stream);
        assert_eq!(decoder.pull().unwrap(), b"after-false-start");
    }

    #[test]
    fn decode_single_frame_matches_write_frame_output() {
        let mut framed = Vec::new();
        write_frame(&mut framed, b"xyz").unwrap();
        assert_eq!(decode_single_frame(&framed).unwrap(), b"xyz");
    }

    #[test]
    fn decode_single_frame_rejects_bad_magic() {
        assert_eq!(
            decode_single_frame(&[0x00, 0x00, 0x00, 0x00]),
            Err(Error::FrameCorrupt)
        );
    }
}
