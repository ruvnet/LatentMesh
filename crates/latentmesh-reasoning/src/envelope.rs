//! ADR-041 §13 `ReasoningDeltaEnvelope`: the upper-layer transport object
//! that carries a compact latent reasoning delta between LatentMesh nodes.
//!
//! This is deliberately a *separate* object from `latentmesh-air-core`'s
//! [`SemanticEnvelope`](../../latentmesh_air_core/struct.SemanticEnvelope.html):
//! ADR 010 remains the bounded physical/semantic transport, and this
//! envelope is an upper-layer payload that Air fragments or MidStream
//! streams. It follows the same wire conventions as the Air envelope
//! (fixed big-endian header, magic marker, explicit lengths, checked
//! arithmetic, signature carried-but-untrusted-until-verified) but is not
//! wire-compatible with it -- the two travel through different layers.
//!
//! # The byte budget is real
//!
//! `docs/research/049-adjacent-areas-survey.md` measures the effective
//! per-fragment content budget over LoRa/Meshtastic Air at **~106 bytes
//! unsigned / ~42 bytes signed** (a 211-byte usable MTU minus the LMS1/LMAD
//! Air envelope tax). [`FIXED_HEADER_BYTES`] below is 282 bytes -- larger
//! than that *entire* per-fragment budget before a single byte of actual
//! delta payload is carried. See `fixed_header_overhead_exceeds_a_single_air_fragment`
//! for the measured fragment counts. This is consistent with ADR-041 §13's
//! own framing (this envelope is fragmented by Air, not a single-frame
//! object) but the number must be measured, not assumed -- an envelope that
//! cannot fit a realistic delta into a bounded number of mesh frames is a
//! design defect.

use std::fmt;

use crate::compat::CodecId;

/// Magic marker for the `ReasoningDeltaEnvelope` wire format. Distinct from
/// `latentmesh-air-core`'s `LMS1` marker -- these are different objects on
/// different layers and must never be decoded by the wrong parser.
const ENVELOPE_MAGIC: &[u8; 4] = b"LMRD";

/// Only wire version this crate currently emits or accepts.
pub const ENVELOPE_VERSION: u16 = 1;

/// Detached-signature length, matching `latentmesh-air-core::ENVELOPE_SIGNATURE_BYTES`.
pub const ENVELOPE_SIGNATURE_BYTES: usize = 64;

/// Hard cap on the delta payload body (ADR-041 §17.13: "A latent payload
/// must have explicit size, iteration, and memory bounds"). 1 MiB is far
/// above anything a constrained link can carry in one session; it exists to
/// bound allocation on decode, not to describe a realistic mesh payload.
pub const MAX_BODY_BYTES: usize = 1_048_576;

/// Sum of every fixed-width field in the wire header, in field order:
/// magic(4) + version(2) + session_id(16) + edge_id(16) + sender_id(32) +
/// model_fingerprint(32) + checkpoint_digest(32) + latent_schema(2) +
/// adapter_digest(32) + codec(1) + iteration(4) + parent_state_hash(32) +
/// result_state_hash(32) + causal_score_q15(2) + confidence_q15(2) +
/// risk_class(1) + reconstruction_error_q15(2) + provenance_root(32) +
/// body_len(4) + signature_len(1) + reserved(1).
///
/// This is the envelope's fixed metadata overhead: `encoded_len() ==
/// FIXED_HEADER_BYTES + body.len() + (64 if signed else 0)`.
pub const FIXED_HEADER_BYTES: usize =
    4 + 2 + 16 + 16 + 32 + 32 + 32 + 2 + 32 + 1 + 4 + 32 + 32 + 2 + 2 + 1 + 2 + 32 + 4 + 1 + 1;

/// Saturating upper bound for the Q15 fixed-point encoding used by
/// `causal_score_q15`, `confidence_q15`, and `reconstruction_error_q15`.
pub const Q15_MAX: u16 = 32_767;

/// Encode a value clamped to `[0.0, 1.0]` as an unsigned Q15 fraction.
/// Values above `1.0` saturate at [`Q15_MAX`] rather than wrapping --
/// callers that need to distinguish "very bad" from "off scale" should
/// clamp and flag out of band, not rely on wraparound here.
pub fn q15_from_unit(value: f32) -> u16 {
    let clamped = value.clamp(0.0, 1.0);
    (clamped * f32::from(Q15_MAX)).round() as u16
}

/// Inverse of [`q15_from_unit`].
pub fn unit_from_q15(raw: u16) -> f32 {
    f32::from(raw) / f32::from(Q15_MAX)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EnvelopeError {
    InvalidMarker,
    UnsupportedVersion,
    InvalidFlags,
    InvalidCodec(u8),
    LimitExceeded,
    Truncated,
}

impl fmt::Display for EnvelopeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMarker => f.write_str("invalid ReasoningDeltaEnvelope marker"),
            Self::UnsupportedVersion => f.write_str("unsupported ReasoningDeltaEnvelope version"),
            Self::InvalidFlags => f.write_str("invalid ReasoningDeltaEnvelope flags"),
            Self::InvalidCodec(id) => write!(f, "invalid codec id {id}"),
            Self::LimitExceeded => f.write_str("configured limit exceeded"),
            Self::Truncated => f.write_str("truncated input"),
        }
    }
}

impl std::error::Error for EnvelopeError {}

pub type Result<T> = std::result::Result<T, EnvelopeError>;

/// ADR-041 §13 transport object. All identity/provenance fields are
/// cryptographic digests bound to the envelope (ADR-041 §17.3-4): model
/// identity, checkpoint identity, and adapter identity travel with every
/// delta, never inferred from shape alone.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReasoningDeltaEnvelope {
    pub version: u16,
    pub session_id: [u8; 16],
    pub edge_id: [u8; 16],
    pub sender_id: [u8; 32],
    pub model_fingerprint: [u8; 32],
    pub checkpoint_digest: [u8; 32],
    pub latent_schema: u16,
    pub adapter_digest: [u8; 32],
    pub codec: CodecId,
    pub iteration: u32,
    pub parent_state_hash: [u8; 32],
    pub result_state_hash: [u8; 32],
    pub causal_score_q15: u16,
    pub confidence_q15: u16,
    pub risk_class: u8,
    pub reconstruction_error_q15: u16,
    pub provenance_root: [u8; 32],
    pub body: Vec<u8>,
    pub signature: Option<[u8; ENVELOPE_SIGNATURE_BYTES]>,
}

impl ReasoningDeltaEnvelope {
    pub fn causal_score(&self) -> f32 {
        unit_from_q15(self.causal_score_q15)
    }

    pub fn confidence(&self) -> f32 {
        unit_from_q15(self.confidence_q15)
    }

    pub fn reconstruction_error(&self) -> f32 {
        unit_from_q15(self.reconstruction_error_q15)
    }

    pub fn validate(&self) -> Result<()> {
        if self.version != ENVELOPE_VERSION {
            return Err(EnvelopeError::UnsupportedVersion);
        }
        if self.body.len() > MAX_BODY_BYTES || self.body.len() > u32::MAX as usize {
            return Err(EnvelopeError::LimitExceeded);
        }
        Ok(())
    }

    /// Total encoded length including body and optional signature.
    pub fn encoded_len(&self) -> usize {
        self.overhead_bytes() + self.body.len()
    }

    /// Fixed metadata overhead: everything except the payload body. This is
    /// the number that matters for the mesh byte budget -- see the module
    /// doc.
    pub fn overhead_bytes(&self) -> usize {
        FIXED_HEADER_BYTES
            + if self.signature.is_some() {
                ENVELOPE_SIGNATURE_BYTES
            } else {
                0
            }
    }

    pub fn encode(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let mut out = Vec::with_capacity(self.encoded_len());
        out.extend_from_slice(ENVELOPE_MAGIC);
        out.extend_from_slice(&self.version.to_be_bytes());
        out.extend_from_slice(&self.session_id);
        out.extend_from_slice(&self.edge_id);
        out.extend_from_slice(&self.sender_id);
        out.extend_from_slice(&self.model_fingerprint);
        out.extend_from_slice(&self.checkpoint_digest);
        out.extend_from_slice(&self.latent_schema.to_be_bytes());
        out.extend_from_slice(&self.adapter_digest);
        out.push(self.codec.to_u8());
        out.extend_from_slice(&self.iteration.to_be_bytes());
        out.extend_from_slice(&self.parent_state_hash);
        out.extend_from_slice(&self.result_state_hash);
        out.extend_from_slice(&self.causal_score_q15.to_be_bytes());
        out.extend_from_slice(&self.confidence_q15.to_be_bytes());
        out.push(self.risk_class);
        out.extend_from_slice(&self.reconstruction_error_q15.to_be_bytes());
        out.extend_from_slice(&self.provenance_root);
        out.extend_from_slice(&(self.body.len() as u32).to_be_bytes());
        out.push(if self.signature.is_some() {
            ENVELOPE_SIGNATURE_BYTES as u8
        } else {
            0
        });
        out.push(0); // Reserved; canonical encoders MUST write zero.
        out.extend_from_slice(&self.body);
        if let Some(signature) = self.signature {
            out.extend_from_slice(&signature);
        }
        Ok(out)
    }

    pub fn decode(input: &[u8]) -> Result<Self> {
        if input.len() < FIXED_HEADER_BYTES {
            return Err(EnvelopeError::Truncated);
        }
        if &input[0..4] != ENVELOPE_MAGIC {
            return Err(EnvelopeError::InvalidMarker);
        }
        let version = u16::from_be_bytes([input[4], input[5]]);
        if version != ENVELOPE_VERSION {
            return Err(EnvelopeError::UnsupportedVersion);
        }

        let session_id: [u8; 16] = input[6..22]
            .try_into()
            .map_err(|_| EnvelopeError::Truncated)?;
        let edge_id: [u8; 16] = input[22..38]
            .try_into()
            .map_err(|_| EnvelopeError::Truncated)?;
        let sender_id: [u8; 32] = input[38..70]
            .try_into()
            .map_err(|_| EnvelopeError::Truncated)?;
        let model_fingerprint: [u8; 32] = input[70..102]
            .try_into()
            .map_err(|_| EnvelopeError::Truncated)?;
        let checkpoint_digest: [u8; 32] = input[102..134]
            .try_into()
            .map_err(|_| EnvelopeError::Truncated)?;
        let latent_schema = u16::from_be_bytes([input[134], input[135]]);
        let adapter_digest: [u8; 32] = input[136..168]
            .try_into()
            .map_err(|_| EnvelopeError::Truncated)?;
        let codec =
            CodecId::try_from(input[168]).map_err(|_| EnvelopeError::InvalidCodec(input[168]))?;
        let iteration = u32::from_be_bytes(
            input[169..173]
                .try_into()
                .map_err(|_| EnvelopeError::Truncated)?,
        );
        let parent_state_hash: [u8; 32] = input[173..205]
            .try_into()
            .map_err(|_| EnvelopeError::Truncated)?;
        let result_state_hash: [u8; 32] = input[205..237]
            .try_into()
            .map_err(|_| EnvelopeError::Truncated)?;
        let causal_score_q15 = u16::from_be_bytes([input[237], input[238]]);
        let confidence_q15 = u16::from_be_bytes([input[239], input[240]]);
        let risk_class = input[241];
        let reconstruction_error_q15 = u16::from_be_bytes([input[242], input[243]]);
        let provenance_root: [u8; 32] = input[244..276]
            .try_into()
            .map_err(|_| EnvelopeError::Truncated)?;
        let body_len = u32::from_be_bytes(
            input[276..280]
                .try_into()
                .map_err(|_| EnvelopeError::Truncated)?,
        ) as usize;
        let signature_len = usize::from(input[280]);
        if input[281] != 0 {
            return Err(EnvelopeError::InvalidFlags);
        }
        if signature_len != 0 && signature_len != ENVELOPE_SIGNATURE_BYTES {
            return Err(EnvelopeError::InvalidFlags);
        }
        if body_len > MAX_BODY_BYTES {
            return Err(EnvelopeError::LimitExceeded);
        }

        let body_start = FIXED_HEADER_BYTES;
        let body_end = body_start
            .checked_add(body_len)
            .ok_or(EnvelopeError::LimitExceeded)?;
        let expected_total = body_end
            .checked_add(signature_len)
            .ok_or(EnvelopeError::LimitExceeded)?;
        if expected_total != input.len() {
            return Err(EnvelopeError::Truncated);
        }

        let body = input[body_start..body_end].to_vec();
        let signature = if signature_len == ENVELOPE_SIGNATURE_BYTES {
            Some(
                input[body_end..expected_total]
                    .try_into()
                    .map_err(|_| EnvelopeError::Truncated)?,
            )
        } else {
            None
        };

        let envelope = Self {
            version,
            session_id,
            edge_id,
            sender_id,
            model_fingerprint,
            checkpoint_digest,
            latent_schema,
            adapter_digest,
            codec,
            iteration,
            parent_state_hash,
            result_state_hash,
            causal_score_q15,
            confidence_q15,
            risk_class,
            reconstruction_error_q15,
            provenance_root,
            body,
            signature,
        };
        envelope.validate()?;
        Ok(envelope)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compat::DeltaCodec;
    use crate::compat::SparseTopKInt8Codec;

    fn sample() -> ReasoningDeltaEnvelope {
        ReasoningDeltaEnvelope {
            version: ENVELOPE_VERSION,
            session_id: [1; 16],
            edge_id: [2; 16],
            sender_id: [3; 32],
            model_fingerprint: [4; 32],
            checkpoint_digest: [5; 32],
            latent_schema: 1,
            adapter_digest: [0; 32],
            codec: CodecId::SparseResidualInt8,
            iteration: 7,
            parent_state_hash: [6; 32],
            result_state_hash: [7; 32],
            causal_score_q15: q15_from_unit(0.82),
            confidence_q15: q15_from_unit(0.91),
            risk_class: 2,
            reconstruction_error_q15: q15_from_unit(0.03),
            provenance_root: [8; 32],
            body: vec![9, 9, 9, 9],
            signature: None,
        }
    }

    #[test]
    fn encode_is_deterministic() {
        let envelope = sample();
        assert_eq!(
            envelope.encode().unwrap(),
            envelope.clone().encode().unwrap()
        );
    }

    #[test]
    fn round_trips_unsigned() {
        let envelope = sample();
        let encoded = envelope.encode().unwrap();
        assert_eq!(encoded.len(), envelope.encoded_len());
        assert_eq!(ReasoningDeltaEnvelope::decode(&encoded).unwrap(), envelope);
    }

    #[test]
    fn round_trips_signed() {
        let mut envelope = sample();
        envelope.signature = Some([0x5a; ENVELOPE_SIGNATURE_BYTES]);
        let encoded = envelope.encode().unwrap();
        assert_eq!(encoded.len(), envelope.encoded_len());
        assert_eq!(ReasoningDeltaEnvelope::decode(&encoded).unwrap(), envelope);
    }

    #[test]
    fn rejects_unknown_marker() {
        let mut encoded = sample().encode().unwrap();
        encoded[0] = b'X';
        assert_eq!(
            ReasoningDeltaEnvelope::decode(&encoded),
            Err(EnvelopeError::InvalidMarker)
        );
    }

    #[test]
    fn rejects_reserved_byte_set() {
        let mut encoded = sample().encode().unwrap();
        encoded[281] = 1;
        assert_eq!(
            ReasoningDeltaEnvelope::decode(&encoded),
            Err(EnvelopeError::InvalidFlags)
        );
    }

    #[test]
    fn q15_round_trip_saturates_at_bounds() {
        assert_eq!(q15_from_unit(-1.0), 0);
        assert_eq!(q15_from_unit(2.0), Q15_MAX);
        assert!((unit_from_q15(q15_from_unit(0.5)) - 0.5).abs() < 1e-3);
    }

    /// The number this task exists to produce: measure, don't assume, the
    /// envelope's fixed overhead against the mesh byte budget documented in
    /// docs/research/049-adjacent-areas-survey.md (~106B unsigned / ~42B
    /// signed effective per-fragment content budget over LoRa/Meshtastic,
    /// derived from the 211-byte usable MTU minus LMS1/LMAD Air envelope
    /// tax).
    #[test]
    fn fixed_header_overhead_exceeds_a_single_air_fragment() {
        const AIR_UNSIGNED_FRAGMENT_BUDGET_BYTES: usize = 106;
        const AIR_SIGNED_FRAGMENT_BUDGET_BYTES: usize = 42;

        assert_eq!(FIXED_HEADER_BYTES, 282);

        let unsigned_fragments = FIXED_HEADER_BYTES.div_ceil(AIR_UNSIGNED_FRAGMENT_BUDGET_BYTES);
        assert_eq!(
            unsigned_fragments, 3,
            "unsigned header alone needs 3 Air fragments"
        );

        let signed_overhead = FIXED_HEADER_BYTES + ENVELOPE_SIGNATURE_BYTES;
        let signed_fragments = signed_overhead.div_ceil(AIR_SIGNED_FRAGMENT_BUDGET_BYTES);
        assert_eq!(
            signed_fragments, 9,
            "signed header alone needs 9 Air fragments"
        );
    }

    /// A realistic small delta -- an 8-of-1536 sparse int8 residual, the
    /// kind of payload the reference codec produces -- still keeps the
    /// *total* envelope small relative to a naive full-tensor transmission,
    /// even though it cannot fit in a single mesh fragment.
    #[test]
    fn realistic_sparse_delta_stays_bounded_relative_to_raw_tensor() {
        let mut delta = vec![0f32; 1536];
        delta[3] = 1.5;
        delta[400] = -2.0;
        delta[900] = 0.9;
        let codec = SparseTopKInt8Codec::new(8);
        let body = codec.encode(&delta).unwrap();

        let mut envelope = sample();
        envelope.body = body;
        let encoded_len = envelope.encode().unwrap().len();

        let raw_tensor_bytes = delta.len() * std::mem::size_of::<f32>();
        assert!(
            encoded_len < raw_tensor_bytes,
            "sparse-codec envelope ({encoded_len}B) should beat a raw {raw_tensor_bytes}B tensor"
        );
    }
}
