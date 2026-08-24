use alloc::vec::Vec;

use latentmesh_air_core::{
    state_hash_tag, AirError, CriticalState, FrameFlags, ReassembledMessage, Reassembler,
    ReassemblerConfig, ReplayDecision, ReplayWindow, Result, SemanticClass, SemanticDelta,
    SemanticEnvelope, SparseRadioFrame,
};

use crate::{
    AssistObservation, BpskModem, BurstCodec, CpfskModem, IqSample, LinkConfig, Modulation,
    SoftBurstReceiver, TinyNeuralAssist,
};

/// `no_std` friendly authentication hook. Production implementations should
/// verify an Ed25519 or equivalent signature over `authentication_bytes`.
pub trait EnvelopeVerifier {
    fn verify(&self, authentication_bytes: &[u8], signature: &[u8; 64]) -> bool;
}

impl<F> EnvelopeVerifier for F
where
    F: Fn(&[u8], &[u8; 64]) -> bool,
{
    fn verify(&self, authentication_bytes: &[u8], signature: &[u8; 64]) -> bool {
        self(authentication_bytes, signature)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReceiverConfig {
    pub link: LinkConfig,
    pub reassembly: ReassemblerConfig,
    pub max_streams: usize,
    pub neural_assist: bool,
}

impl ReceiverConfig {
    pub fn for_link(link: LinkConfig) -> Self {
        Self {
            link,
            reassembly: ReassemblerConfig::default(),
            max_streams: 16,
            neural_assist: false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReceiverState {
    Listening,
    Reassembling,
    MessageReady,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReceivedMessage {
    pub profile: latentmesh_air_core::WireProfile,
    pub flags: FrameFlags,
    pub stream_id: u16,
    pub sequence: u16,
    pub class: SemanticClass,
    pub priority: u8,
    pub state_tag: u16,
    pub bytes: Vec<u8>,
}

impl ReceivedMessage {
    pub fn semantic_envelope(&self) -> Result<SemanticEnvelope> {
        let envelope = SemanticEnvelope::decode(&self.bytes)?;
        if envelope.class != self.class
            || envelope.priority != self.priority
            || state_hash_tag(&envelope.state_hash) != self.state_tag
            || self.flags.contains(FrameFlags::SIGNED_ENVELOPE) != envelope.signature.is_some()
        {
            return Err(AirError::ResultStateMismatch);
        }
        Ok(envelope)
    }

    pub fn semantic_delta(&self) -> Result<SemanticDelta> {
        let envelope = self.semantic_envelope()?;
        if envelope.signature.is_some() {
            return Err(AirError::AuthenticationRequired);
        }
        self.delta_from_envelope(&envelope)
    }

    pub fn semantic_delta_verified<V: EnvelopeVerifier + ?Sized>(
        &self,
        verifier: &V,
    ) -> Result<SemanticDelta> {
        let envelope = self.verified_envelope(verifier)?;
        self.delta_from_envelope(&envelope)
    }

    /// Full 128-bit result-hash validation occurs inside `SemanticDelta::apply`.
    pub fn apply_delta(&self, base: &CriticalState) -> Result<CriticalState> {
        self.semantic_delta()?.apply(base)
    }

    /// Verify a signed LMS1 envelope before applying its deterministic delta.
    /// This is the only high-level application path for signed messages.
    pub fn apply_delta_verified<V: EnvelopeVerifier + ?Sized>(
        &self,
        base: &CriticalState,
        verifier: &V,
    ) -> Result<CriticalState> {
        self.semantic_delta_verified(verifier)?.apply(base)
    }

    fn verified_envelope<V: EnvelopeVerifier + ?Sized>(
        &self,
        verifier: &V,
    ) -> Result<SemanticEnvelope> {
        let envelope = self.semantic_envelope()?;
        let signature = envelope
            .signature
            .as_ref()
            .ok_or(AirError::AuthenticationRequired)?;
        let authentication_bytes = envelope.authentication_bytes()?;
        if !verifier.verify(&authentication_bytes, signature) {
            return Err(AirError::AuthenticationFailed);
        }
        Ok(envelope)
    }

    fn delta_from_envelope(&self, envelope: &SemanticEnvelope) -> Result<SemanticDelta> {
        if self.class != SemanticClass::StateDelta {
            return Err(AirError::InvalidClass);
        }
        let delta = envelope.unwrap_delta()?;
        if state_hash_tag(&delta.result_hash) != self.state_tag {
            return Err(AirError::ResultStateMismatch);
        }
        Ok(delta)
    }

    /// Parse and bind semantic content without treating a transported
    /// signature as verified. Used only by the receive admission path.
    fn validate_semantic_binding(&self) -> Result<()> {
        let envelope = self.semantic_envelope()?;
        self.delta_from_envelope(&envelope).map(|_| ())
    }

    fn validate_admission(&self, verifier: Option<&dyn EnvelopeVerifier>) -> Result<()> {
        if self.flags.contains(FrameFlags::SIGNED_ENVELOPE) {
            let verifier = verifier.ok_or(AirError::AuthenticationRequired)?;
            let envelope = self.verified_envelope(verifier)?;
            if self.class == SemanticClass::StateDelta {
                self.delta_from_envelope(&envelope)?;
            }
            return Ok(());
        }
        if self.class == SemanticClass::StateDelta {
            self.validate_semantic_binding()?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct StreamReplay {
    stream_id: u16,
    window: ReplayWindow,
}

#[derive(Clone, Debug)]
pub struct AirReceiver {
    config: ReceiverConfig,
    reassembler: Reassembler,
    replay: Vec<StreamReplay>,
    soft_burst: SoftBurstReceiver,
    assist: Option<TinyNeuralAssist>,
    state: ReceiverState,
}

impl AirReceiver {
    pub fn new(config: ReceiverConfig) -> Result<Self> {
        config.link.validate()?;
        if config.max_streams == 0 || config.max_streams > 256 {
            return Err(AirError::InvalidLength);
        }
        let codec = BurstCodec::new(config.link.interleaver_columns)?;
        Ok(Self {
            reassembler: Reassembler::new(config.reassembly)?,
            replay: Vec::with_capacity(config.max_streams.min(16)),
            soft_burst: SoftBurstReceiver::new(codec),
            assist: config.neural_assist.then(TinyNeuralAssist::default),
            config,
            state: ReceiverState::Listening,
        })
    }

    pub const fn state(&self) -> ReceiverState {
        self.state
    }

    pub fn neural_assist(&self) -> Option<&TinyNeuralAssist> {
        self.assist.as_ref()
    }

    pub fn neural_assist_mut(&mut self) -> Option<&mut TinyNeuralAssist> {
        self.assist.as_mut()
    }

    pub fn ingest_frame_bytes(&mut self, bytes: &[u8]) -> Result<Option<ReceivedMessage>> {
        self.state = ReceiverState::Listening;
        let frame = SparseRadioFrame::decode(bytes)?;
        self.ingest_frame(frame)
    }

    pub fn ingest_frame_bytes_verified<V: EnvelopeVerifier>(
        &mut self,
        bytes: &[u8],
        verifier: &V,
    ) -> Result<Option<ReceivedMessage>> {
        self.state = ReceiverState::Listening;
        let frame = SparseRadioFrame::decode(bytes)?;
        self.ingest_frame_verified(frame, verifier)
    }

    pub fn ingest_frame(&mut self, frame: SparseRadioFrame) -> Result<Option<ReceivedMessage>> {
        self.ingest_frame_inner(frame, None)
    }

    pub fn ingest_frame_verified<V: EnvelopeVerifier>(
        &mut self,
        frame: SparseRadioFrame,
        verifier: &V,
    ) -> Result<Option<ReceivedMessage>> {
        self.ingest_frame_inner(frame, Some(verifier))
    }

    fn ingest_frame_inner(
        &mut self,
        frame: SparseRadioFrame,
        verifier: Option<&dyn EnvelopeVerifier>,
    ) -> Result<Option<ReceivedMessage>> {
        self.state = ReceiverState::Listening;
        if frame.profile != self.config.link.profile {
            return Err(AirError::InvalidProfile);
        }
        if self.config.link.flags.contains(FrameFlags::FEC) != frame.flags.contains(FrameFlags::FEC)
        {
            return Err(AirError::InvalidFlags);
        }
        // An unverified signed fragment never enters the reassembly table, so
        // it cannot retain poisoned state or consume replay-window capacity.
        if frame.flags.contains(FrameFlags::SIGNED_ENVELOPE) && verifier.is_none() {
            self.reassembler.clear(frame.stream_id, frame.sequence);
            return Err(AirError::AuthenticationRequired);
        }
        if let Some(entry) = self
            .replay
            .iter()
            .find(|entry| entry.stream_id == frame.stream_id)
        {
            match entry.window.classify(frame.sequence) {
                ReplayDecision::Accept => {}
                ReplayDecision::Duplicate
                    if self
                        .reassembler
                        .has_in_flight(frame.stream_id, frame.sequence) => {}
                ReplayDecision::Duplicate => return Err(AirError::Replay),
                ReplayDecision::TooOld => return Err(AirError::TooOld),
            }
        }
        let complete = self.reassembler.push(frame)?;
        let Some(complete) = complete else {
            self.state = ReceiverState::Reassembling;
            return Ok(None);
        };
        let message = received(complete);
        // Authentication, outer/body binding and state-hash binding all happen
        // before replay commit or message exposure.
        message.validate_admission(verifier)?;
        self.commit_replay(message.stream_id, message.sequence)?;
        self.state = ReceiverState::MessageReady;
        Ok(Some(message))
    }

    pub fn ingest_iq_burst(&mut self, samples: &[IqSample]) -> Result<Option<ReceivedMessage>> {
        let frame = self.decode_iq_burst(samples)?;
        self.ingest_frame_bytes(&frame)
    }

    pub fn ingest_iq_burst_verified<V: EnvelopeVerifier>(
        &mut self,
        samples: &[IqSample],
        verifier: &V,
    ) -> Result<Option<ReceivedMessage>> {
        let frame = self.decode_iq_burst(samples)?;
        self.ingest_frame_bytes_verified(&frame, verifier)
    }

    fn decode_iq_burst(&self, samples: &[IqSample]) -> Result<Vec<u8>> {
        let Modulation::Bpsk(config) = self.config.link.modulation else {
            return Err(AirError::InvalidProfile);
        };
        let classical = BpskModem::new(config)?.demodulate_soft(samples)?;
        let llrs = self.assist_llrs(&classical)?;
        BurstCodec::new(self.config.link.interleaver_columns)?.decode_aligned(&llrs)
    }

    pub fn ingest_pcm_burst(&mut self, samples: &[i16]) -> Result<Option<ReceivedMessage>> {
        let frame = self.decode_pcm_burst(samples)?;
        self.ingest_frame_bytes(&frame)
    }

    pub fn ingest_pcm_burst_verified<V: EnvelopeVerifier>(
        &mut self,
        samples: &[i16],
        verifier: &V,
    ) -> Result<Option<ReceivedMessage>> {
        let frame = self.decode_pcm_burst(samples)?;
        self.ingest_frame_bytes_verified(&frame, verifier)
    }

    fn decode_pcm_burst(&self, samples: &[i16]) -> Result<Vec<u8>> {
        let Modulation::Cpfsk(config) = self.config.link.modulation else {
            return Err(AirError::InvalidProfile);
        };
        let classical = CpfskModem::new(config)?.demodulate_soft(samples)?;
        let llrs = self.assist_llrs(&classical)?;
        BurstCodec::new(self.config.link.interleaver_columns)?.decode_aligned(&llrs)
    }

    /// Feed already synchronized soft symbols from an SDR/DSP frontend.
    pub fn ingest_soft_stream(&mut self, llrs: &[i8]) -> Result<Vec<ReceivedMessage>> {
        self.ingest_soft_stream_inner(llrs, None)
    }

    pub fn ingest_soft_stream_verified<V: EnvelopeVerifier>(
        &mut self,
        llrs: &[i8],
        verifier: &V,
    ) -> Result<Vec<ReceivedMessage>> {
        self.ingest_soft_stream_inner(llrs, Some(verifier))
    }

    fn ingest_soft_stream_inner(
        &mut self,
        llrs: &[i8],
        verifier: Option<&dyn EnvelopeVerifier>,
    ) -> Result<Vec<ReceivedMessage>> {
        let frame_bytes = self.soft_burst.push_slice(llrs)?;
        let mut messages = Vec::new();
        for frame in frame_bytes {
            let decoded = SparseRadioFrame::decode(&frame)?;
            if let Some(message) = self.ingest_frame_inner(decoded, verifier)? {
                messages.push(message);
            }
        }
        Ok(messages)
    }

    fn assist_llrs(&self, classical: &[i8]) -> Result<Vec<i8>> {
        let Some(assist) = &self.assist else {
            return Ok(classical.to_vec());
        };
        let observations: Vec<AssistObservation> = classical
            .iter()
            .map(|llr| {
                let energy = f32::from(llr.unsigned_abs()) / 127.0;
                AssistObservation {
                    energy,
                    noise: 1.0 - energy,
                    frequency_error: 0.0,
                }
            })
            .collect();
        assist.refine_batch(classical, &observations)
    }

    fn commit_replay(&mut self, stream_id: u16, sequence: u16) -> Result<()> {
        if let Some(entry) = self
            .replay
            .iter_mut()
            .find(|entry| entry.stream_id == stream_id)
        {
            return entry.window.commit(sequence);
        }
        if self.replay.len() == self.config.max_streams {
            return Err(AirError::LimitExceeded);
        }
        let mut window = ReplayWindow::new();
        window.commit(sequence)?;
        self.replay.push(StreamReplay { stream_id, window });
        Ok(())
    }
}

fn received(message: ReassembledMessage) -> ReceivedMessage {
    ReceivedMessage {
        profile: message.profile,
        flags: message.flags,
        stream_id: message.stream_id,
        sequence: message.sequence,
        class: message.class,
        priority: message.priority,
        state_tag: message.state_tag,
        bytes: message.bytes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AirTransmitter, Transmission, TransmitterConfig};
    use latentmesh_air_core::{
        CriticalState, SemanticDelta, SemanticEnvelope, SymbolValue, WireProfile,
    };

    struct TestVerifier;

    impl EnvelopeVerifier for TestVerifier {
        fn verify(&self, authentication_bytes: &[u8], signature: &[u8; 64]) -> bool {
            test_signature(authentication_bytes) == *signature
        }
    }

    fn test_signature(bytes: &[u8]) -> [u8; 64] {
        // Deterministic test authenticator only. Production callers supply a
        // cryptographic EnvelopeVerifier without changing this crate.
        let mut output = [0_u8; 64];
        for (index, byte) in bytes.iter().enumerate() {
            let slot = index % output.len();
            output[slot] = output[slot]
                .rotate_left(1)
                .wrapping_add(*byte)
                .wrapping_add(index as u8);
        }
        output
    }

    fn signed_inputs(
        valid_signature: bool,
    ) -> (CriticalState, CriticalState, LinkConfig, Vec<Vec<u8>>) {
        let link = LinkConfig::for_profile(WireProfile::Ble);
        let mut tx = AirTransmitter::new(TransmitterConfig::for_link(link)).unwrap();
        let mut base = CriticalState::new();
        base.set(1, SymbolValue::Bool(false)).unwrap();
        let mut target = base.clone();
        target.set(1, SymbolValue::Bool(true)).unwrap();
        let delta = SemanticDelta::between(1, 1, 1, &base, &target, alloc::vec![]).unwrap();
        let placeholder = SemanticEnvelope::wrap_delta(&delta, 15, 0, Some([0_u8; 64])).unwrap();
        let signature = if valid_signature {
            test_signature(&placeholder.authentication_bytes().unwrap())
        } else {
            [0_u8; 64]
        };
        tx.enqueue_delta_with_signature(7, &delta, 15, Some(signature))
            .unwrap();
        let mut frames = Vec::new();
        while let Some(transmission) = tx.next_transmission().unwrap() {
            let Transmission::Bytes { bytes, .. } = transmission else {
                panic!("expected byte transport");
            };
            frames.push(bytes);
        }
        (base, target, link, frames)
    }

    fn signed_fixture() -> (CriticalState, CriticalState, ReceivedMessage) {
        let (base, target, link, frames) = signed_inputs(true);
        let mut rx = AirReceiver::new(ReceiverConfig::for_link(link)).unwrap();
        let mut received = None;
        for bytes in frames {
            received = rx
                .ingest_frame_bytes_verified(&bytes, &TestVerifier)
                .unwrap()
                .or(received);
        }
        (base, target, received.unwrap())
    }

    #[test]
    fn byte_transport_reassembles_and_rejects_replay() {
        let link = LinkConfig::for_profile(WireProfile::Ble);
        let mut tx = AirTransmitter::new(TransmitterConfig::for_link(link)).unwrap();
        let mut base = CriticalState::new();
        base.set(1, SymbolValue::Bool(false)).unwrap();
        let mut target = base.clone();
        target.set(1, SymbolValue::Bool(true)).unwrap();
        let delta = SemanticDelta::between(1, 1, 1, &base, &target, alloc::vec![]).unwrap();
        tx.enqueue_delta(7, &delta, 15).unwrap();
        let Transmission::Bytes { bytes, .. } = tx.next_transmission().unwrap().unwrap() else {
            panic!("expected byte transport");
        };
        let mut rx = AirReceiver::new(ReceiverConfig::for_link(link)).unwrap();
        let message = rx.ingest_frame_bytes(&bytes).unwrap().unwrap();
        assert_eq!(message.apply_delta(&base).unwrap(), target);
        assert_eq!(rx.ingest_frame_bytes(&bytes), Err(AirError::Replay));
    }

    #[test]
    fn valid_signed_delta_applies_only_through_verifier() {
        let (base, target, message) = signed_fixture();
        assert_eq!(
            message.apply_delta_verified(&base, &TestVerifier).unwrap(),
            target
        );
    }

    #[test]
    fn invalid_signed_delta_fails_authentication() {
        let (_, _, link, frames) = signed_inputs(false);
        let mut rx = AirReceiver::new(ReceiverConfig::for_link(link)).unwrap();
        let mut result = Ok(None);
        for bytes in frames {
            result = rx.ingest_frame_bytes_verified(&bytes, &TestVerifier);
        }
        assert_eq!(result, Err(AirError::AuthenticationFailed));
    }

    #[test]
    fn signed_delta_cannot_bypass_verifier_via_high_level_apis() {
        let (base, _, message) = signed_fixture();
        assert_eq!(
            message.semantic_delta(),
            Err(AirError::AuthenticationRequired)
        );
        assert_eq!(
            message.apply_delta(&base),
            Err(AirError::AuthenticationRequired)
        );
    }

    #[test]
    fn unverified_ingest_rejects_signed_fragment_without_retaining_state() {
        let (_, target, link, frames) = signed_inputs(true);
        let mut rx = AirReceiver::new(ReceiverConfig::for_link(link)).unwrap();
        assert_eq!(
            rx.ingest_frame_bytes(&frames[0]),
            Err(AirError::AuthenticationRequired)
        );
        let mut received = None;
        for bytes in frames {
            received = rx
                .ingest_frame_bytes_verified(&bytes, &TestVerifier)
                .unwrap()
                .or(received);
        }
        assert_eq!(
            received
                .unwrap()
                .semantic_delta_verified(&TestVerifier)
                .unwrap()
                .result_hash,
            target.critical_hash()
        );
    }

    #[test]
    fn invalid_signature_does_not_consume_replay_sequence() {
        let (base, target, link, invalid_frames) = signed_inputs(false);
        let (_, _, _, corrected_frames) = signed_inputs(true);
        let invalid_sequence = SparseRadioFrame::decode(&invalid_frames[0])
            .unwrap()
            .sequence;
        let corrected_sequence = SparseRadioFrame::decode(&corrected_frames[0])
            .unwrap()
            .sequence;
        assert_eq!(invalid_sequence, corrected_sequence);

        let mut rx = AirReceiver::new(ReceiverConfig::for_link(link)).unwrap();
        let mut invalid_result = Ok(None);
        for bytes in invalid_frames {
            invalid_result = rx.ingest_frame_bytes_verified(&bytes, &TestVerifier);
        }
        assert_eq!(invalid_result, Err(AirError::AuthenticationFailed));

        let mut corrected = None;
        for bytes in corrected_frames {
            corrected = rx
                .ingest_frame_bytes_verified(&bytes, &TestVerifier)
                .unwrap()
                .or(corrected);
        }
        assert_eq!(
            corrected
                .unwrap()
                .apply_delta_verified(&base, &TestVerifier)
                .unwrap(),
            target
        );
    }
}
