//! The crate's transport seam. [`FrameTransport`] is deliberately synchronous
//! and minimal — the QUIC path (feature `midstream-quic`) exposes async
//! inherent methods instead of forcing an async runtime on the whole
//! workspace. [`ChannelTransport`] runs the full codec over an in-process
//! duplex so tests and benchmarks exercise the identical byte path a network
//! transport would carry.

use crate::codec::{encode_frame, FrameDecoder};
use crate::error::StreamError;
use latentmesh_core::LatentFrame;
use std::sync::mpsc;

/// A bidirectional, frame-oriented transport endpoint.
pub trait FrameTransport {
    /// Encode and send one frame.
    fn send_frame(&mut self, frame: &LatentFrame) -> Result<(), StreamError>;
    /// Receive the next complete frame, if one is available now.
    fn try_recv_frame(&mut self) -> Result<Option<LatentFrame>, StreamError>;
}

/// In-process duplex transport carrying *encoded bytes* (not frames), so the
/// codec path is identical to a network transport's.
pub struct ChannelTransport {
    tx: mpsc::Sender<Vec<u8>>,
    rx: mpsc::Receiver<Vec<u8>>,
    decoder: FrameDecoder,
}

impl ChannelTransport {
    /// A connected pair of endpoints.
    pub fn pair() -> (ChannelTransport, ChannelTransport) {
        let (a_tx, b_rx) = mpsc::channel();
        let (b_tx, a_rx) = mpsc::channel();
        (
            ChannelTransport {
                tx: a_tx,
                rx: a_rx,
                decoder: FrameDecoder::new(),
            },
            ChannelTransport {
                tx: b_tx,
                rx: b_rx,
                decoder: FrameDecoder::new(),
            },
        )
    }
}

impl FrameTransport for ChannelTransport {
    fn send_frame(&mut self, frame: &LatentFrame) -> Result<(), StreamError> {
        let bytes = encode_frame(frame)?;
        self.tx
            .send(bytes)
            .map_err(|_| StreamError::Transport("channel closed".into()))
    }

    fn try_recv_frame(&mut self) -> Result<Option<LatentFrame>, StreamError> {
        loop {
            if let Some(frame) = self.decoder.next_frame()? {
                return Ok(Some(frame));
            }
            match self.rx.try_recv() {
                Ok(chunk) => self.decoder.push(&chunk)?,
                Err(mpsc::TryRecvError::Empty) => return Ok(None),
                Err(mpsc::TryRecvError::Disconnected) => {
                    if self.decoder.buffered() > 0 {
                        return Err(StreamError::Transport("peer closed mid-frame".into()));
                    }
                    return Ok(None);
                }
            }
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
            sender_model: "m".into(),
            receiver_space: "r".into(),
            transform_hash: "t".into(),
            sequence: seq,
            payload: Payload::encode(&[1.0, 2.0], Encoding::F32),
            confidence: 0.8,
            provenance: Provenance {
                sender_model: "m".into(),
                context_hash: "c".into(),
                parents: vec![],
            },
            authority: Authority::ObserveOnly,
            timestamp: 0,
        }
    }

    #[test]
    fn frames_cross_the_duplex_in_order() {
        let (mut a, mut b) = ChannelTransport::pair();
        for seq in 0..5 {
            a.send_frame(&frame(seq)).unwrap();
        }
        let mut seen = Vec::new();
        while let Some(f) = b.try_recv_frame().unwrap() {
            seen.push(f.sequence);
        }
        assert_eq!(seen, vec![0, 1, 2, 3, 4]);
        // And the reverse direction works independently.
        b.send_frame(&frame(99)).unwrap();
        assert_eq!(a.try_recv_frame().unwrap().unwrap().sequence, 99);
    }

    #[test]
    fn empty_channel_returns_none_not_a_block() {
        let (_a, mut b) = ChannelTransport::pair();
        assert!(b.try_recv_frame().unwrap().is_none());
    }
}
