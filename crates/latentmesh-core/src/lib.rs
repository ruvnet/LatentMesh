//! `latentmesh-core` — the wire vocabulary for LatentMesh (ADR-002): a
//! [`LatentFrame`] carries a sender's hidden-state slice into a receiver's
//! embedding space as a first-class network primitive, instead of forcing it
//! through a serialize→tokenize round trip. This crate is zero-I/O: encoding,
//! quantization, content hashing, and the frame type itself. Alignment lives
//! in `latentmesh-align`; admission/execution governance lives in
//! `latentmesh-gate`.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// How the payload floats are packed on the wire. Real, measurable byte costs
/// per element: `F32`=4B, `F16`=2B, `Int8`=1B (+ a shared scale/zero-point).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Encoding {
    F32,
    F16,
    /// Linear (affine) 8-bit quantization: `value = (q - zero_point) * scale`.
    Int8,
}

impl Encoding {
    /// Bytes per scalar element on the wire (excludes the small fixed
    /// per-tensor scale/zero-point header for `Int8`).
    pub fn bytes_per_element(self) -> usize {
        match self {
            Encoding::F32 => 4,
            Encoding::F16 => 2,
            Encoding::Int8 => 1,
        }
    }
}

/// Quantized payload bytes plus what's needed to reconstruct `f32`s from them.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Payload {
    pub encoding: Encoding,
    /// Number of scalar elements (payload.len() derives from this × encoding).
    pub dim: usize,
    /// The packed bytes (little-endian per element for F32/F16; raw i8 bytes
    /// reinterpreted as u8 for Int8).
    pub bytes: Vec<u8>,
    /// Int8 affine dequant params: `(scale, zero_point)`. `None` for F32/F16.
    pub int8_params: Option<(f32, i32)>,
}

impl Payload {
    /// Encode an `f32` vector at the given [`Encoding`]. Int8 uses per-tensor
    /// min/max affine quantization (the standard, real technique — not a
    /// stand-in).
    pub fn encode(values: &[f32], encoding: Encoding) -> Self {
        match encoding {
            Encoding::F32 => Payload {
                encoding,
                dim: values.len(),
                bytes: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
                int8_params: None,
            },
            Encoding::F16 => Payload {
                encoding,
                dim: values.len(),
                bytes: values
                    .iter()
                    .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
                    .collect(),
                int8_params: None,
            },
            Encoding::Int8 => {
                let (min, max) = values
                    .iter()
                    .fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
                let (min, max) = if min == max {
                    (min - 1.0, max + 1.0)
                } else {
                    (min, max)
                };
                let scale = (max - min) / 255.0;
                let zero_point = (-min / scale).round() as i32;
                let bytes = values
                    .iter()
                    .map(|v| {
                        let q = (v / scale).round() as i32 + zero_point;
                        q.clamp(0, 255) as u8
                    })
                    .collect();
                Payload {
                    encoding,
                    dim: values.len(),
                    bytes,
                    int8_params: Some((scale, zero_point)),
                }
            }
        }
    }

    /// Decode back to `f32`. For `Int8` this is lossy affine dequantization;
    /// for `F32`/`F16` it round-trips exactly (`F16` within half precision).
    pub fn decode(&self) -> Vec<f32> {
        match self.encoding {
            Encoding::F32 => self
                .bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            Encoding::F16 => self
                .bytes
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect(),
            Encoding::Int8 => {
                let (scale, zero_point) = self.int8_params.unwrap_or((1.0, 0));
                self.bytes
                    .iter()
                    .map(|&b| (b as i32 - zero_point) as f32 * scale)
                    .collect()
            }
        }
    }

    /// Wire size in bytes: payload bytes + a small fixed Int8 header.
    pub fn wire_bytes(&self) -> usize {
        self.bytes.len() + if self.int8_params.is_some() { 8 } else { 0 }
    }
}

/// Who/what vouches for a frame's contents — the audit trail a latent packet
/// carries because its payload is not human-inspectable (ADR-007).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Provenance {
    /// Identifier of the model/checkpoint that produced the sender state.
    pub sender_model: String,
    /// Content hash of the upstream context that produced this state (e.g. the
    /// hash of the prompt/trajectory prefix) — NOT the raw text, so provenance
    /// is verifiable without re-exposing the content on the wire.
    pub context_hash: String,
    /// Lineage: parent frame ids this frame was derived/streamed from.
    pub parents: Vec<String>,
}

/// Execution authority a frame's receiver may grant it, ordered by risk —
/// mirrors the AGL authority ladder (`cognitum-one/slack` ADR-0008) applied to
/// latent execution instead of code mutation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Authority {
    /// The receiver may inspect/log the frame but not condition generation on it.
    ObserveOnly,
    /// The receiver may use the frame as retrieval context (soft-prompt-like).
    ContextInject,
    /// The receiver may prefix its own decode with the aligned latent state.
    LatentPrefix,
    /// The receiver may let the frame steer tool/action selection.
    ActionInfluencing,
}

/// A latent communication packet (ADR-002) — the network primitive this crate
/// exists to define. `payload` carries the SENDER's hidden state; alignment
/// into the receiver's space (`latentmesh-align`) happens on receipt, keeping
/// a frame reusable across multiple receivers with different transforms.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LatentFrame {
    pub id: String,
    pub sender_model: String,
    /// The embedding-space identifier this frame is ALIGNED FOR, once aligned
    /// (empty/"unaligned" before a receiver applies its transform).
    pub receiver_space: String,
    /// Content hash of the alignment transform used (binds the frame to a
    /// specific, reproducible `R`/`alpha` — never a bare unhashed matrix).
    pub transform_hash: String,
    /// Monotonic sequence number within a stream (ADR-003).
    pub sequence: u64,
    pub payload: Payload,
    /// Alignment confidence in `[0,1]` (`latentmesh-align`'s own estimate of
    /// how well `R`/`alpha` fit the calibration set this frame relies on).
    pub confidence: f32,
    pub provenance: Provenance,
    pub authority: Authority,
    /// Unix seconds.
    pub timestamp: u64,
}

impl LatentFrame {
    /// Deterministic content hash over the frame (excludes nothing — the
    /// payload IS the content). Used for provenance chaining and dedup.
    pub fn content_hash(&self) -> String {
        let bytes = serde_json::to_vec(self).unwrap_or_default();
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        hex::encode(hasher.finalize())
    }
}

/// Minimal local hex encoder (avoids pulling in a whole `hex` crate for one
/// call site — matches the dependency-lean style of the crates it mirrors).
mod hex {
    pub fn encode(bytes: impl AsRef<[u8]>) -> String {
        bytes.as_ref().iter().map(|b| format!("{b:02x}")).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_round_trips_exactly() {
        let v = vec![0.1_f32, -2.5, 3.333, 0.0, 1e-3];
        let p = Payload::encode(&v, Encoding::F32);
        assert_eq!(p.decode(), v);
        assert_eq!(p.wire_bytes(), v.len() * 4);
    }

    #[test]
    fn f16_round_trips_within_half_precision() {
        let v = vec![0.1_f32, -2.5, 3.333, 0.0];
        let p = Payload::encode(&v, Encoding::F16);
        let back = p.decode();
        for (a, b) in v.iter().zip(back.iter()) {
            assert!((a - b).abs() < 5e-3, "f16 round trip diverged: {a} vs {b}");
        }
        assert_eq!(p.wire_bytes(), v.len() * 2);
    }

    #[test]
    fn int8_round_trips_within_quantization_error() {
        let v: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 8.0).collect();
        let p = Payload::encode(&v, Encoding::Int8);
        let back = p.decode();
        for (a, b) in v.iter().zip(back.iter()) {
            assert!((a - b).abs() < 0.05, "int8 round trip diverged: {a} vs {b}");
        }
        // 64 elements * 1 byte + the 8-byte scale/zero-point header.
        assert_eq!(p.wire_bytes(), 64 + 8);
    }

    #[test]
    fn wire_bytes_match_the_scaling_claim_16x4096_fp16_is_128kb() {
        // The scaling intuition this protocol is built to test: 16 vectors of
        // 4096 dims at FP16 should be ~128 KiB — verify it's not off by a
        // sign or a factor from a hand-wave.
        let per_vec = Payload::encode(&vec![0.0_f32; 4096], Encoding::F16).wire_bytes();
        assert_eq!(per_vec * 16, 128 * 1024);
    }

    #[test]
    fn content_hash_changes_when_payload_changes() {
        let mk = |seq: u64| LatentFrame {
            id: "f1".into(),
            sender_model: "m".into(),
            receiver_space: "r".into(),
            transform_hash: "t".into(),
            sequence: seq,
            payload: Payload::encode(&[1.0, 2.0], Encoding::F32),
            confidence: 0.9,
            provenance: Provenance {
                sender_model: "m".into(),
                context_hash: "c".into(),
                parents: vec![],
            },
            authority: Authority::ContextInject,
            timestamp: 0,
        };
        assert_ne!(mk(1).content_hash(), mk(2).content_hash());
    }

    #[test]
    fn authority_is_ordered_observe_only_below_action_influencing() {
        assert!(Authority::ObserveOnly < Authority::ActionInfluencing);
        assert!(Authority::ContextInject < Authority::LatentPrefix);
    }
}
