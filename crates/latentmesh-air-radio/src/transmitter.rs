use alloc::collections::VecDeque;
use alloc::vec::Vec;

use latentmesh_air_core::{
    fragment_message, state_hash_tag, AirError, FragmentMeta, Result, SemanticClass, SemanticDelta,
    SemanticEnvelope, SparseRadioFrame, ENVELOPE_SIGNATURE_BYTES,
};

use crate::{BpskModem, BurstCodec, CpfskModem, IqSample, LinkConfig, Modulation};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TransmitterConfig {
    pub link: LinkConfig,
    pub initial_sequence: u16,
    pub initial_logical_sequence: u64,
    pub max_queued_frames: usize,
}

impl TransmitterConfig {
    pub fn for_link(link: LinkConfig) -> Self {
        Self {
            link,
            initial_sequence: 0,
            initial_logical_sequence: 0,
            max_queued_frames: 64,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransmitterState {
    Idle,
    Ready { queued_frames: usize },
}

#[derive(Clone, Debug, PartialEq)]
pub enum Transmission {
    Bytes {
        frame: SparseRadioFrame,
        bytes: Vec<u8>,
    },
    Iq {
        frame: SparseRadioFrame,
        samples: Vec<IqSample>,
    },
    Pcm {
        frame: SparseRadioFrame,
        samples: Vec<i16>,
    },
}

#[derive(Clone, Debug)]
pub struct AirTransmitter {
    config: TransmitterConfig,
    next_sequence: u16,
    next_logical_sequence: u64,
    queue: VecDeque<SparseRadioFrame>,
}

impl AirTransmitter {
    pub fn new(config: TransmitterConfig) -> Result<Self> {
        config.link.validate()?;
        if config.max_queued_frames == 0 || config.max_queued_frames > 1_024 {
            return Err(AirError::InvalidLength);
        }
        Ok(Self {
            next_sequence: config.initial_sequence,
            next_logical_sequence: config.initial_logical_sequence,
            queue: VecDeque::with_capacity(config.max_queued_frames.min(64)),
            config,
        })
    }

    pub fn state(&self) -> TransmitterState {
        if self.queue.is_empty() {
            TransmitterState::Idle
        } else {
            TransmitterState::Ready {
                queued_frames: self.queue.len(),
            }
        }
    }

    pub const fn next_sequence(&self) -> u16 {
        self.next_sequence
    }

    pub const fn next_logical_sequence(&self) -> u64 {
        self.next_logical_sequence
    }

    pub fn enqueue_delta(
        &mut self,
        stream_id: u16,
        delta: &SemanticDelta,
        priority: u8,
    ) -> Result<u16> {
        self.enqueue_delta_with_signature(stream_id, delta, priority, None)
    }

    pub fn enqueue_delta_with_signature(
        &mut self,
        stream_id: u16,
        delta: &SemanticDelta,
        priority: u8,
        signature: Option<[u8; ENVELOPE_SIGNATURE_BYTES]>,
    ) -> Result<u16> {
        let envelope =
            SemanticEnvelope::wrap_delta(delta, priority, self.next_logical_sequence, signature)?;
        let encoded = envelope.encode()?;
        let flags = if envelope.signature.is_some() {
            self.config
                .link
                .flags
                .union(latentmesh_air_core::FrameFlags::SIGNED_ENVELOPE)
        } else {
            self.config.link.flags
        };
        let sequence = self.enqueue_message_with_flags(
            stream_id,
            SemanticClass::StateDelta,
            priority,
            state_hash_tag(&delta.result_hash),
            &encoded,
            flags,
        )?;
        self.next_logical_sequence = self.next_logical_sequence.wrapping_add(1);
        Ok(sequence)
    }

    pub fn enqueue_message(
        &mut self,
        stream_id: u16,
        class: SemanticClass,
        priority: u8,
        state_tag: u16,
        message: &[u8],
    ) -> Result<u16> {
        self.enqueue_message_with_flags(
            stream_id,
            class,
            priority,
            state_tag,
            message,
            self.config.link.flags,
        )
    }

    fn enqueue_message_with_flags(
        &mut self,
        stream_id: u16,
        class: SemanticClass,
        priority: u8,
        state_tag: u16,
        message: &[u8],
        flags: latentmesh_air_core::FrameFlags,
    ) -> Result<u16> {
        let sequence = self.next_sequence;
        let frames = fragment_message(
            FragmentMeta {
                profile: self.config.link.profile,
                flags,
                stream_id,
                sequence,
                class,
                priority,
                state_tag,
            },
            message,
            self.config.link.frame_mtu,
        )?;
        let new_len = self
            .queue
            .len()
            .checked_add(frames.len())
            .ok_or(AirError::LimitExceeded)?;
        if new_len > self.config.max_queued_frames {
            return Err(AirError::LimitExceeded);
        }
        self.queue.extend(frames);
        self.next_sequence = self.next_sequence.wrapping_add(1);
        Ok(sequence)
    }

    pub fn next_frame(&mut self) -> Option<SparseRadioFrame> {
        self.queue.pop_front()
    }

    pub fn next_transmission(&mut self) -> Result<Option<Transmission>> {
        let Some(frame) = self.next_frame() else {
            return Ok(None);
        };
        let frame_bytes = frame.encode()?;
        let transmission = match self.config.link.modulation {
            Modulation::ByteTransport => Transmission::Bytes {
                frame,
                bytes: frame_bytes,
            },
            Modulation::Bpsk(config) => {
                let bits =
                    BurstCodec::new(self.config.link.interleaver_columns)?.encode(&frame_bytes)?;
                let samples = BpskModem::new(config)?.modulate(&bits)?;
                Transmission::Iq { frame, samples }
            }
            Modulation::Cpfsk(config) => {
                let bits =
                    BurstCodec::new(self.config.link.interleaver_columns)?.encode(&frame_bytes)?;
                let samples = CpfskModem::new(config)?.modulate(&bits)?;
                Transmission::Pcm { frame, samples }
            }
        };
        Ok(Some(transmission))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use latentmesh_air_core::{CriticalState, SymbolValue, WireProfile};

    #[test]
    fn state_machine_queues_fragmented_delta() {
        let mut before = CriticalState::new();
        before.set(1, SymbolValue::Bool(false)).unwrap();
        let mut after = before.clone();
        after.set(1, SymbolValue::Bool(true)).unwrap();
        let delta = SemanticDelta::between(1, 1, 1, &before, &after, alloc::vec![]).unwrap();
        let link = LinkConfig::for_profile(WireProfile::HfBpsk);
        let mut tx = AirTransmitter::new(TransmitterConfig::for_link(link)).unwrap();
        tx.enqueue_delta(4, &delta, 15).unwrap();
        assert!(matches!(tx.state(), TransmitterState::Ready { .. }));
        assert!(matches!(
            tx.next_transmission().unwrap(),
            Some(Transmission::Iq { .. })
        ));
    }
}
