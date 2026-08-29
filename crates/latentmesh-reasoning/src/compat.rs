//! ADR-041 §10 compatibility handshake, plus the delta codec trait family
//! from §9.
//!
//! The single most important rule enforced here: **equal dimensionality
//! does not imply compatibility.** Two latent workspaces can share
//! `hidden_dimension` and still encode completely different geometries if
//! they come from different checkpoints. [`evaluate_handshake`] never
//! consults `hidden_dimension` when deciding whether a remote delta is
//! usable -- only cryptographically bound identity (`model_fingerprint`,
//! `checkpoint_digest`, `latent_schema_version`) or a verified, benchmarked
//! adapter can grant that. See `equal_hidden_dimension_does_not_imply_compatibility`
//! below for the test that pins this behavior.

use std::cmp::Ordering;
use std::fmt;

/// Static identity and shape metadata a node exchanges before it will trust
/// a remote [`crate::envelope::ReasoningDeltaEnvelope`] (ADR-041 §10).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompatibilityDescriptor {
    /// Short, zero-padded ASCII label (e.g. `b"bdh-cq-150m\0\0\0\0\0"`).
    /// Informational only -- never used as a compatibility signal by itself.
    pub model_family: [u8; 16],
    pub model_fingerprint: [u8; 32],
    pub checkpoint_digest: [u8; 32],
    pub latent_schema_version: u16,
    pub layer_or_workspace_id: u32,
    /// Informational / shape-checking only. See the module doc: this field
    /// MUST NOT be used to decide compatibility.
    pub hidden_dimension: u32,
    pub normalization_scheme: u8,
    pub position_encoding_id: u16,
    pub adapter_id: u32,
    pub adapter_digest: [u8; 32],
    pub quantizer_id: u16,
    pub codec_version: u16,
    pub training_domain_tag: Option<u32>,
}

/// A benchmarked, immutably fingerprinted adapter from `remote`'s model
/// space into `local`'s model space (ADR-041 §11.2). Constructed and
/// maintained by whatever adapter registry the deployment uses; this crate
/// only ever *consumes* one as a handshake input, never mints one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VerifiedAdapter {
    pub adapter_digest: [u8; 32],
    pub source_model_fingerprint: [u8; 32],
    pub target_model_fingerprint: [u8; 32],
    /// True only if the adapter has a current, passing benchmark record.
    /// ADR-041 §11.2: "If its benchmark expires ... the receiver falls back."
    pub benchmarked: bool,
}

/// Outcome of [`evaluate_handshake`], mirroring the receiver policy in
/// ADR-041 §10 exactly:
///
/// ```text
/// if exact model fingerprint and schema match -> identity adapter
/// else if verified adapter exists             -> use adapter
/// else if semantic fallback exists             -> request semantic delta
/// else                                          -> reject
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompatibilityDecision {
    /// Sender and receiver share model fingerprint, checkpoint digest, and
    /// latent schema version. No adapter is applied.
    IdentityAdapter,
    /// A benchmarked adapter maps the remote latent space into the local one.
    VerifiedAdapter { adapter_digest: [u8; 32] },
    /// No latent-compatible path exists; the sender should be asked to
    /// re-send a symbolic/semantic delta instead of a latent one.
    SemanticFallback,
    /// No compatible path exists at all. The envelope must be discarded.
    Reject,
}

/// Decide how (or whether) a `remote` reasoning delta may be merged into
/// `local`'s latent workspace. Pure and deterministic: no I/O, no heuristic
/// dimension matching (ADR-041 §10: "No heuristic dimension matching is
/// permitted").
pub fn evaluate_handshake(
    local: &CompatibilityDescriptor,
    remote: &CompatibilityDescriptor,
    verified_adapter: Option<&VerifiedAdapter>,
    semantic_fallback_available: bool,
) -> CompatibilityDecision {
    if local.model_fingerprint == remote.model_fingerprint
        && local.checkpoint_digest == remote.checkpoint_digest
        && local.latent_schema_version == remote.latent_schema_version
    {
        return CompatibilityDecision::IdentityAdapter;
    }

    if let Some(adapter) = verified_adapter {
        if adapter.benchmarked
            && adapter.source_model_fingerprint == remote.model_fingerprint
            && adapter.target_model_fingerprint == local.model_fingerprint
            && adapter.adapter_digest == remote.adapter_digest
        {
            return CompatibilityDecision::VerifiedAdapter {
                adapter_digest: adapter.adapter_digest,
            };
        }
    }

    if semantic_fallback_available {
        return CompatibilityDecision::SemanticFallback;
    }

    CompatibilityDecision::Reject
}

/// Identifies which [`DeltaCodec`] produced a payload body, carried on the
/// wire as a single byte (ADR-041 §13 `codec: CodecId`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum CodecId {
    /// Full/raw tensor payload. Retained only as an upper-bound comparison
    /// baseline (ADR-041 §21.2) -- never the default wire codec.
    Raw = 0,
    /// Top-k sparse residual, int8 quantized (ADR-041 §9.2). The reference
    /// implementation shipped by this crate: [`SparseTopKInt8Codec`].
    SparseResidualInt8 = 1,
    /// Low-rank factor transmission (ADR-041 §9.3). Trait-only in Phase 1.
    LowRank = 2,
    /// Learned bottleneck codec (ADR-041 §9.4). Trait-only in Phase 1.
    LearnedBottleneck = 3,
    /// Not a latent payload at all: the symbolic/semantic fallback path
    /// (ADR-041 §10, "request semantic delta").
    Semantic = 4,
}

impl CodecId {
    pub fn to_u8(self) -> u8 {
        self as u8
    }
}

impl TryFrom<u8> for CodecId {
    type Error = CompatError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Raw),
            1 => Ok(Self::SparseResidualInt8),
            2 => Ok(Self::LowRank),
            3 => Ok(Self::LearnedBottleneck),
            4 => Ok(Self::Semantic),
            other => Err(CompatError::UnknownCodec(other)),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompatError {
    UnknownCodec(u8),
    EmptyInput,
    TooManyComponents,
    TruncatedPayload,
    LengthMismatch,
}

impl fmt::Display for CompatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownCodec(id) => write!(f, "unknown codec id {id}"),
            Self::EmptyInput => f.write_str("codec input is empty"),
            Self::TooManyComponents => f.write_str("codec input exceeds addressable index range"),
            Self::TruncatedPayload => f.write_str("codec payload is truncated"),
            Self::LengthMismatch => f.write_str("codec output length mismatch"),
        }
    }
}

impl std::error::Error for CompatError {}

/// A latent reasoning delta codec (ADR-041 §9). Implementations trade off
/// transmitted bytes, latency, energy, reconstruction error, and privacy
/// risk per the §9.5 selection objective; this crate does not pick a codec
/// on a caller's behalf.
pub trait DeltaCodec {
    fn codec_id(&self) -> CodecId;

    /// Encode a dense workspace delta into a compact wire body.
    fn encode(&self, delta: &[f32]) -> Result<Vec<u8>, CompatError>;

    /// Decode a wire body back into a dense vector of length `output_len`.
    /// `output_len` must match the dimensionality the encoder observed;
    /// implementations MUST reject a mismatch rather than silently
    /// truncating or padding (ADR-041 §17.14: "never silent coercion").
    fn decode(&self, encoded: &[u8], output_len: usize) -> Result<Vec<f32>, CompatError>;
}

/// Reference implementation of ADR-041 §9.2: top-`k` sparse residual,
/// int8-quantized with a single shared scale.
///
/// Wire body layout (all integers big-endian):
///
/// ```text
/// k_used:     u16
/// output_len: u32
/// scale:      f32
/// [index: u32, value: i8] * k_used
/// ```
pub struct SparseTopKInt8Codec {
    pub k: usize,
}

impl SparseTopKInt8Codec {
    pub fn new(k: usize) -> Self {
        Self { k }
    }
}

impl DeltaCodec for SparseTopKInt8Codec {
    fn codec_id(&self) -> CodecId {
        CodecId::SparseResidualInt8
    }

    fn encode(&self, delta: &[f32]) -> Result<Vec<u8>, CompatError> {
        if delta.is_empty() {
            return Err(CompatError::EmptyInput);
        }
        if delta.len() > u32::MAX as usize {
            return Err(CompatError::TooManyComponents);
        }
        let k = self.k.min(delta.len());

        let max_abs = delta.iter().fold(0f32, |acc, v| acc.max(v.abs()));
        let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };

        // Deterministic selection: rank by descending |value|, ties broken
        // by ascending index, then re-sort the chosen set back into
        // ascending index order for a stable wire encoding.
        let mut ranked: Vec<(usize, f32)> = delta.iter().copied().enumerate().collect();
        ranked.sort_by(|a, b| {
            b.1.abs()
                .partial_cmp(&a.1.abs())
                .unwrap_or(Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        ranked.truncate(k);
        ranked.sort_by_key(|(index, _)| *index);

        let mut out = Vec::with_capacity(2 + 4 + 4 + ranked.len() * 5);
        out.extend_from_slice(&(ranked.len() as u16).to_be_bytes());
        out.extend_from_slice(&(delta.len() as u32).to_be_bytes());
        out.extend_from_slice(&scale.to_be_bytes());
        for (index, value) in &ranked {
            out.extend_from_slice(&(*index as u32).to_be_bytes());
            let quantized = (*value / scale).round().clamp(-127.0, 127.0) as i8;
            out.push(quantized as u8);
        }
        Ok(out)
    }

    fn decode(&self, encoded: &[u8], output_len: usize) -> Result<Vec<f32>, CompatError> {
        if encoded.len() < 10 {
            return Err(CompatError::TruncatedPayload);
        }
        let k = u16::from_be_bytes([encoded[0], encoded[1]]) as usize;
        let declared_len =
            u32::from_be_bytes([encoded[2], encoded[3], encoded[4], encoded[5]]) as usize;
        if declared_len != output_len {
            return Err(CompatError::LengthMismatch);
        }
        let scale = f32::from_be_bytes([encoded[6], encoded[7], encoded[8], encoded[9]]);

        let expected_len = 10usize
            .checked_add(k.checked_mul(5).ok_or(CompatError::TruncatedPayload)?)
            .ok_or(CompatError::TruncatedPayload)?;
        if encoded.len() != expected_len {
            return Err(CompatError::TruncatedPayload);
        }

        let mut out = vec![0f32; output_len];
        let mut cursor = 10usize;
        for _ in 0..k {
            let index = u32::from_be_bytes([
                encoded[cursor],
                encoded[cursor + 1],
                encoded[cursor + 2],
                encoded[cursor + 3],
            ]) as usize;
            let raw = encoded[cursor + 4] as i8;
            if index >= output_len {
                return Err(CompatError::LengthMismatch);
            }
            out[index] = f32::from(raw) * scale;
            cursor += 5;
        }
        Ok(out)
    }
}

/// Normalized L2 reconstruction error, matching the convergence-ratio shape
/// used elsewhere in ADR-041 (§6.2): `norm(diff) / max(norm(original), tiny)`.
pub fn reconstruction_error(original: &[f32], reconstructed: &[f32]) -> f32 {
    let sum_sq_diff: f32 = original
        .iter()
        .zip(reconstructed)
        .map(|(a, b)| (a - b).powi(2))
        .sum();
    let sum_sq_orig: f32 = original.iter().map(|v| v.powi(2)).sum();
    sum_sq_diff.sqrt() / sum_sq_orig.sqrt().max(1e-8)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor(
        model_fingerprint: [u8; 32],
        checkpoint_digest: [u8; 32],
    ) -> CompatibilityDescriptor {
        CompatibilityDescriptor {
            model_family: *b"bdh-cq-150m\0\0\0\0\0",
            model_fingerprint,
            checkpoint_digest,
            latent_schema_version: 1,
            layer_or_workspace_id: 0,
            hidden_dimension: 1536,
            normalization_scheme: 1,
            position_encoding_id: 0,
            adapter_id: 0,
            adapter_digest: [0; 32],
            quantizer_id: 0,
            codec_version: 1,
            training_domain_tag: None,
        }
    }

    #[test]
    fn identical_checkpoint_grants_identity_adapter() {
        let local = descriptor([1; 32], [2; 32]);
        let remote = descriptor([1; 32], [2; 32]);
        assert_eq!(
            evaluate_handshake(&local, &remote, None, false),
            CompatibilityDecision::IdentityAdapter
        );
    }

    /// The load-bearing rule from ADR-041 §10: two descriptors with the
    /// *same* `hidden_dimension` (1536) but different checkpoints must
    /// never be treated as compatible without a verified adapter.
    #[test]
    fn equal_hidden_dimension_does_not_imply_compatibility() {
        let local = descriptor([1; 32], [0xAA; 32]);
        let mut remote = descriptor([1; 32], [0xBB; 32]);
        assert_eq!(local.hidden_dimension, remote.hidden_dimension);
        assert_ne!(local.checkpoint_digest, remote.checkpoint_digest);

        assert_eq!(
            evaluate_handshake(&local, &remote, None, false),
            CompatibilityDecision::Reject
        );
        assert_eq!(
            evaluate_handshake(&local, &remote, None, true),
            CompatibilityDecision::SemanticFallback
        );

        // The sender declares which adapter it used; the receiver's
        // VerifiedAdapter record must match that declaration exactly.
        remote.adapter_digest = [7; 32];
        let adapter = VerifiedAdapter {
            adapter_digest: [7; 32],
            source_model_fingerprint: remote.model_fingerprint,
            target_model_fingerprint: local.model_fingerprint,
            benchmarked: true,
        };
        assert_eq!(
            evaluate_handshake(&local, &remote, Some(&adapter), false),
            CompatibilityDecision::VerifiedAdapter {
                adapter_digest: [7; 32]
            }
        );
    }

    #[test]
    fn unbenchmarked_adapter_does_not_count_as_verified() {
        let local = descriptor([1; 32], [0xAA; 32]);
        let mut remote = descriptor([1; 32], [0xBB; 32]);
        remote.adapter_digest = [7; 32];
        let adapter = VerifiedAdapter {
            adapter_digest: [7; 32],
            source_model_fingerprint: remote.model_fingerprint,
            target_model_fingerprint: local.model_fingerprint,
            benchmarked: false,
        };
        assert_eq!(
            evaluate_handshake(&local, &remote, Some(&adapter), false),
            CompatibilityDecision::Reject
        );
    }

    #[test]
    fn codec_round_trip_is_deterministic_and_bounded() {
        let mut delta = vec![0f32; 1536];
        delta[10] = 4.0;
        delta[200] = -3.5;
        delta[900] = 2.0;
        delta[1500] = -1.8;
        for (index, slot) in delta.iter_mut().enumerate() {
            if *slot == 0.0 {
                *slot = 0.001 * ((index % 7) as f32 - 3.0);
            }
        }

        let codec = SparseTopKInt8Codec::new(8);
        let encoded_a = codec.encode(&delta).unwrap();
        let encoded_b = codec.encode(&delta).unwrap();
        assert_eq!(
            encoded_a, encoded_b,
            "same input must encode to identical bytes"
        );

        // 1536 f32 raw = 6144 bytes; the sparse encoding must be dramatically
        // smaller for a delta this concentrated.
        assert!(encoded_a.len() < 6144 / 8);

        let decoded = codec.decode(&encoded_a, delta.len()).unwrap();
        let error = reconstruction_error(&delta, &decoded);
        assert!(error < 0.05, "reconstruction error too high: {error}");
    }

    #[test]
    fn codec_rejects_output_length_mismatch() {
        let codec = SparseTopKInt8Codec::new(4);
        let delta = vec![1.0f32; 16];
        let encoded = codec.encode(&delta).unwrap();
        assert_eq!(codec.decode(&encoded, 8), Err(CompatError::LengthMismatch));
    }
}
