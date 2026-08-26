//! MidStream QUIC transport (feature `midstream-quic`): the identical framing
//! codec over a `midstreamer-quic 0.3` bidirectional stream. The byte-level
//! seam is [`LatentByteStream`] (async fn in trait — MSRV-safe since 1.75) so
//! the framing logic is testable against an in-memory duplex while the
//! production impl delegates to `midstreamer_quic::QuicStream`.

use crate::codec::{encode_frame, FrameDecoder};
use crate::error::StreamError;
use latentmesh_core::LatentFrame;
use midstreamer_quic::{QuicError, QuicStream, QuicTransport};

/// Minimal async byte duplex the framing runs over.
pub trait LatentByteStream {
    /// Send all of `data`.
    fn send_bytes(
        &mut self,
        data: &[u8],
    ) -> impl core::future::Future<Output = Result<(), StreamError>> + Send;
    /// Receive up to `buf.len()` bytes; `0` means end of stream.
    fn recv_bytes(
        &mut self,
        buf: &mut [u8],
    ) -> impl core::future::Future<Output = Result<usize, StreamError>> + Send;
}

fn quic_err(e: QuicError) -> StreamError {
    StreamError::Transport(e.to_string())
}

impl LatentByteStream for QuicStream {
    async fn send_bytes(&mut self, data: &[u8]) -> Result<(), StreamError> {
        QuicStream::send(self, data)
            .await
            .map(|_| ())
            .map_err(quic_err)
    }

    async fn recv_bytes(&mut self, buf: &mut [u8]) -> Result<usize, StreamError> {
        QuicStream::recv(self, buf).await.map_err(quic_err)
    }
}

/// Frame transport over any [`LatentByteStream`].
pub struct QuicFrameTransport<S: LatentByteStream> {
    stream: S,
    decoder: FrameDecoder,
    read_buf: Vec<u8>,
}

impl<S: LatentByteStream> QuicFrameTransport<S> {
    pub fn new(stream: S) -> Self {
        QuicFrameTransport {
            stream,
            decoder: FrameDecoder::new(),
            read_buf: vec![0u8; 16 * 1024],
        }
    }

    /// Encode and send one frame.
    pub async fn send_frame(&mut self, frame: &LatentFrame) -> Result<(), StreamError> {
        let bytes = encode_frame(frame)?;
        self.stream.send_bytes(&bytes).await
    }

    /// Receive the next complete frame. `Ok(None)` means the peer finished
    /// the stream cleanly; a stream that ends mid-frame is a transport error.
    pub async fn recv_frame(&mut self) -> Result<Option<LatentFrame>, StreamError> {
        loop {
            if let Some(frame) = self.decoder.next_frame()? {
                return Ok(Some(frame));
            }
            let n = self.stream.recv_bytes(&mut self.read_buf).await?;
            if n == 0 {
                if self.decoder.buffered() > 0 {
                    return Err(StreamError::Transport("peer closed mid-frame".into()));
                }
                return Ok(None);
            }
            self.decoder.push(&self.read_buf[..n])?;
        }
    }
}

/// Open a latent stream on a live MidStream QUIC connection (any
/// [`QuicTransport`] implementor, per the published embedding trait).
pub async fn open_latent_stream<T: QuicTransport + ?Sized>(
    transport: &T,
) -> Result<QuicFrameTransport<QuicStream>, StreamError> {
    let stream = transport.open_bi_stream().await.map_err(quic_err)?;
    Ok(QuicFrameTransport::new(stream))
}

/// Accept the peer's next latent stream.
pub async fn accept_latent_stream<T: QuicTransport + ?Sized>(
    transport: &T,
) -> Result<QuicFrameTransport<QuicStream>, StreamError> {
    let stream = transport.accept_bi_stream().await.map_err(quic_err)?;
    Ok(QuicFrameTransport::new(stream))
}

#[cfg(test)]
mod tests {
    use super::*;
    use latentmesh_core::{Authority, Encoding, Payload, Provenance};
    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};

    /// In-memory duplex implementing the same byte contract QUIC provides,
    /// including partial reads across frame boundaries.
    #[derive(Clone, Default)]
    struct MemDuplex {
        to_peer: Arc<Mutex<VecDeque<u8>>>,
        from_peer: Arc<Mutex<VecDeque<u8>>>,
    }

    impl MemDuplex {
        fn pair() -> (MemDuplex, MemDuplex) {
            let ab = Arc::new(Mutex::new(VecDeque::new()));
            let ba = Arc::new(Mutex::new(VecDeque::new()));
            (
                MemDuplex {
                    to_peer: ab.clone(),
                    from_peer: ba.clone(),
                },
                MemDuplex {
                    to_peer: ba,
                    from_peer: ab,
                },
            )
        }
    }

    impl LatentByteStream for MemDuplex {
        async fn send_bytes(&mut self, data: &[u8]) -> Result<(), StreamError> {
            self.to_peer
                .lock()
                .expect("test mutex")
                .extend(data.iter().copied());
            Ok(())
        }

        async fn recv_bytes(&mut self, buf: &mut [u8]) -> Result<usize, StreamError> {
            let mut q = self.from_peer.lock().expect("test mutex");
            // Deliberately return tiny partial reads to exercise reassembly.
            let n = buf.len().min(q.len()).min(3);
            for slot in buf.iter_mut().take(n) {
                *slot = q.pop_front().unwrap_or(0);
            }
            Ok(n)
        }
    }

    fn frame(seq: u64) -> LatentFrame {
        LatentFrame {
            id: format!("f{seq}"),
            sender_model: "m".into(),
            receiver_space: "r".into(),
            transform_hash: "t".into(),
            sequence: seq,
            payload: Payload::encode(&[0.5; 32], Encoding::Int8),
            confidence: 0.7,
            provenance: Provenance {
                sender_model: "m".into(),
                context_hash: "c".into(),
                parents: vec![],
            },
            authority: Authority::ObserveOnly,
            timestamp: 0,
        }
    }

    fn block_on<F: core::future::Future>(fut: F) -> F::Output {
        // The futures here never actually pend (the in-memory duplex is
        // always ready), so a minimal noop-waker executor suffices — no
        // runtime dependency needed. `Waker::noop` is fine here: this module
        // only builds under the `midstream-quic` feature, whose dependency
        // already floors the toolchain above the workspace MSRV path.
        use core::task::{Context, Poll, Waker};
        let waker = Waker::noop();
        let mut cx = Context::from_waker(waker);
        let mut fut = core::pin::pin!(fut);
        loop {
            if let Poll::Ready(out) = fut.as_mut().poll(&mut cx) {
                return out;
            }
        }
    }

    #[test]
    fn frames_survive_partial_reads_over_the_duplex() {
        let (a, b) = MemDuplex::pair();
        let mut tx = QuicFrameTransport::new(a);
        let mut rx = QuicFrameTransport::new(b);
        block_on(async {
            for seq in 0..4 {
                tx.send_frame(&frame(seq)).await.unwrap();
            }
            for seq in 0..4 {
                let got = rx.recv_frame().await.unwrap().unwrap();
                assert_eq!(got.sequence, seq);
            }
        });
    }

    #[test]
    fn quic_connection_satisfies_the_transport_bounds() {
        // Compile-time check mirroring midstreamer-quic's own convention: the
        // helpers accept the concrete published connection type.
        fn _accepts<T: QuicTransport>() {
            let _ = open_latent_stream::<T>;
            let _ = accept_latent_stream::<T>;
        }
        let _ = _accepts::<midstreamer_quic::QuicConnection>;
    }
}
