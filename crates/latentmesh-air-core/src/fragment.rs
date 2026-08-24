use alloc::vec;
use alloc::vec::Vec;

use crate::{
    AirError, FrameFlags, Result, SemanticClass, SparseRadioFrame, WireProfile, FRAME_MAX_BYTES,
    FRAME_MAX_PAYLOAD, FRAME_MIN_BYTES,
};

pub const MAX_FRAGMENTS: u8 = 32;
pub const MAX_MESSAGE_BYTES: usize = MAX_FRAGMENTS as usize * FRAME_MAX_PAYLOAD;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FragmentMeta {
    pub profile: WireProfile,
    pub flags: FrameFlags,
    pub stream_id: u16,
    pub sequence: u16,
    pub class: SemanticClass,
    pub priority: u8,
    pub state_tag: u16,
}

pub fn fragment_message(
    meta: FragmentMeta,
    message: &[u8],
    frame_mtu: usize,
) -> Result<Vec<SparseRadioFrame>> {
    if !(FRAME_MIN_BYTES..=FRAME_MAX_BYTES).contains(&frame_mtu) {
        return Err(AirError::InvalidLength);
    }
    if message.len() > MAX_MESSAGE_BYTES {
        return Err(AirError::LimitExceeded);
    }
    let payload_capacity = frame_mtu - FRAME_MIN_BYTES;
    if payload_capacity == 0 && !message.is_empty() {
        return Err(AirError::InvalidLength);
    }
    let count = if message.is_empty() {
        1
    } else {
        message.len().div_ceil(payload_capacity)
    };
    if count > usize::from(MAX_FRAGMENTS) {
        return Err(AirError::LimitExceeded);
    }

    let mut frames = Vec::with_capacity(count);
    if message.is_empty() {
        frames.push(SparseRadioFrame {
            profile: meta.profile,
            flags: meta.flags,
            stream_id: meta.stream_id,
            sequence: meta.sequence,
            fragment_index: 0,
            fragment_count: 1,
            class: meta.class,
            priority: meta.priority,
            state_tag: meta.state_tag,
            payload: Vec::new(),
        });
    } else {
        for (index, payload) in message.chunks(payload_capacity).enumerate() {
            frames.push(SparseRadioFrame {
                profile: meta.profile,
                flags: meta.flags,
                stream_id: meta.stream_id,
                sequence: meta.sequence,
                fragment_index: index as u8,
                fragment_count: count as u8,
                class: meta.class,
                priority: meta.priority,
                state_tag: meta.state_tag,
                payload: payload.to_vec(),
            });
        }
    }
    Ok(frames)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReassemblerConfig {
    pub max_contexts: usize,
    pub max_message_bytes: usize,
    pub max_fragments: u8,
}

impl Default for ReassemblerConfig {
    fn default() -> Self {
        Self {
            max_contexts: 4,
            max_message_bytes: MAX_MESSAGE_BYTES,
            max_fragments: MAX_FRAGMENTS,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReassembledMessage {
    pub profile: WireProfile,
    pub flags: FrameFlags,
    pub stream_id: u16,
    pub sequence: u16,
    pub class: SemanticClass,
    pub priority: u8,
    pub state_tag: u16,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug)]
struct Context {
    profile: WireProfile,
    flags: FrameFlags,
    stream_id: u16,
    sequence: u16,
    class: SemanticClass,
    priority: u8,
    state_tag: u16,
    total_bytes: usize,
    parts: Vec<Option<Vec<u8>>>,
}

/// Fixed-count reassembly table. Callers decide eviction policy; this type
/// refuses a new context when full rather than silently discarding a message.
#[derive(Clone, Debug)]
pub struct Reassembler {
    config: ReassemblerConfig,
    contexts: Vec<Context>,
}

impl Reassembler {
    pub fn new(config: ReassemblerConfig) -> Result<Self> {
        if config.max_contexts == 0
            || config.max_message_bytes == 0
            || config.max_message_bytes > MAX_MESSAGE_BYTES
            || config.max_fragments == 0
            || config.max_fragments > MAX_FRAGMENTS
        {
            return Err(AirError::InvalidLength);
        }
        Ok(Self {
            config,
            contexts: Vec::with_capacity(config.max_contexts),
        })
    }

    pub fn has_in_flight(&self, stream_id: u16, sequence: u16) -> bool {
        self.contexts
            .iter()
            .any(|context| context.stream_id == stream_id && context.sequence == sequence)
    }

    pub fn clear(&mut self, stream_id: u16, sequence: u16) {
        if let Some(index) = self
            .contexts
            .iter()
            .position(|item| item.stream_id == stream_id && item.sequence == sequence)
        {
            self.contexts.swap_remove(index);
        }
    }

    pub fn push(&mut self, frame: SparseRadioFrame) -> Result<Option<ReassembledMessage>> {
        frame.validate()?;
        if frame.fragment_count > self.config.max_fragments {
            return Err(AirError::LimitExceeded);
        }
        let context_index = self
            .contexts
            .iter()
            .position(|item| item.stream_id == frame.stream_id && item.sequence == frame.sequence);
        let index = match context_index {
            Some(index) => index,
            None => {
                if self.contexts.len() == self.config.max_contexts {
                    return Err(AirError::ReassemblyFull);
                }
                self.contexts.push(Context {
                    profile: frame.profile,
                    flags: frame.flags,
                    stream_id: frame.stream_id,
                    sequence: frame.sequence,
                    class: frame.class,
                    priority: frame.priority,
                    state_tag: frame.state_tag,
                    total_bytes: 0,
                    parts: vec![None; usize::from(frame.fragment_count)],
                });
                self.contexts.len() - 1
            }
        };

        let context = &mut self.contexts[index];
        if context.profile != frame.profile
            || context.flags != frame.flags
            || context.class != frame.class
            || context.priority != frame.priority
            || context.state_tag != frame.state_tag
            || context.parts.len() != usize::from(frame.fragment_count)
        {
            return Err(AirError::FragmentConflict);
        }
        let part = &mut context.parts[usize::from(frame.fragment_index)];
        if let Some(existing) = part {
            return if *existing == frame.payload {
                Ok(None)
            } else {
                Err(AirError::FragmentConflict)
            };
        }
        let new_total = context
            .total_bytes
            .checked_add(frame.payload.len())
            .ok_or(AirError::LimitExceeded)?;
        if new_total > self.config.max_message_bytes {
            return Err(AirError::LimitExceeded);
        }
        context.total_bytes = new_total;
        *part = Some(frame.payload);
        if context.parts.iter().any(Option::is_none) {
            return Ok(None);
        }

        let context = self.contexts.swap_remove(index);
        let mut bytes = Vec::with_capacity(context.total_bytes);
        for part in context.parts {
            bytes.extend_from_slice(part.as_deref().ok_or(AirError::InvalidFragment)?);
        }
        Ok(Some(ReassembledMessage {
            profile: context.profile,
            flags: context.flags,
            stream_id: context.stream_id,
            sequence: context.sequence,
            class: context.class,
            priority: context.priority,
            state_tag: context.state_tag,
            bytes,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta() -> FragmentMeta {
        FragmentMeta {
            profile: WireProfile::Ble,
            flags: FrameFlags::NONE,
            stream_id: 2,
            sequence: 9,
            class: SemanticClass::StateDelta,
            priority: 8,
            state_tag: 0x1234,
        }
    }

    #[test]
    fn reassembles_out_of_order_and_ignores_exact_duplicate() {
        let message: Vec<u8> = (0..150).map(|value| value as u8).collect();
        let frames = fragment_message(meta(), &message, 64).unwrap();
        assert_eq!(frames.len(), 4);
        let mut rx = Reassembler::new(ReassemblerConfig::default()).unwrap();
        assert!(rx.push(frames[2].clone()).unwrap().is_none());
        assert!(rx.push(frames[2].clone()).unwrap().is_none());
        assert!(rx.push(frames[0].clone()).unwrap().is_none());
        assert!(rx.push(frames[3].clone()).unwrap().is_none());
        let complete = rx.push(frames[1].clone()).unwrap().unwrap();
        assert_eq!(complete.bytes, message);
    }

    #[test]
    fn rejects_conflicting_duplicate() {
        let frames = fragment_message(meta(), &[1; 100], 64).unwrap();
        let mut rx = Reassembler::new(ReassemblerConfig::default()).unwrap();
        rx.push(frames[0].clone()).unwrap();
        let mut conflict = frames[0].clone();
        conflict.payload[0] ^= 1;
        assert_eq!(rx.push(conflict), Err(AirError::FragmentConflict));
    }

    #[test]
    fn bounded_fragment_count() {
        let too_large = alloc::vec![0; MAX_MESSAGE_BYTES + 1];
        assert_eq!(
            fragment_message(meta(), &too_large, FRAME_MAX_BYTES),
            Err(AirError::LimitExceeded)
        );
    }
}
