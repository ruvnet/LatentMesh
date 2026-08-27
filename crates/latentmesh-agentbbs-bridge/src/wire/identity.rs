//! agentbbs's Ed25519 identity types, transcribed from
//! `agentbbs-core/src/identity.rs`. Split out of [`crate::wire`] to keep
//! that module under this repo's file-size norm; see its doc comment for
//! the full transcription rationale.

use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};

/// Errors from decoding or verifying agentbbs wire types.
#[derive(Debug, thiserror::Error)]
pub enum WireError {
    /// A field failed structural validation (malformed hex, wrong length, …).
    #[error("malformed {field}: {reason}")]
    Malformed { field: &'static str, reason: String },
    /// An Ed25519 signature did not verify, or the signer's key did not
    /// match the claimed author/node.
    #[error("bad signature")]
    BadSignature,
    /// JSON (de)serialization failed.
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
}

/// `agentbbs-core/src/identity.rs:20-93` — a public, anonymous identity: an
/// Ed25519 verifying key, serialized as lowercase hex.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct AgentId([u8; 32]);

impl AgentId {
    /// `identity.rs:31-35` — construct from raw public-key bytes, validating
    /// that they form a canonical Ed25519 point.
    pub fn from_bytes(bytes: [u8; 32]) -> Result<Self, WireError> {
        VerifyingKey::from_bytes(&bytes).map_err(|e| WireError::Malformed {
            field: "agent id",
            reason: e.to_string(),
        })?;
        Ok(AgentId(bytes))
    }

    /// `identity.rs:38-44` — parse from a hex string.
    pub fn from_hex(s: &str) -> Result<Self, WireError> {
        let raw = hex::decode(s.trim()).map_err(|e| WireError::Malformed {
            field: "agent id hex",
            reason: e.to_string(),
        })?;
        let arr: [u8; 32] = raw.try_into().map_err(|_| WireError::Malformed {
            field: "agent id",
            reason: "expected 32 bytes".into(),
        })?;
        Self::from_bytes(arr)
    }

    /// `identity.rs:47-49`.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// `identity.rs:52-54`.
    pub fn to_hex(&self) -> String {
        hex::encode(self.0)
    }

    /// `identity.rs:63-67` — verify a detached signature over `msg` made by
    /// this identity.
    pub fn verify(&self, msg: &[u8], sig: &SignatureBytes) -> Result<(), WireError> {
        let vk = VerifyingKey::from_bytes(&self.0).map_err(|e| WireError::Malformed {
            field: "agent id",
            reason: e.to_string(),
        })?;
        let signature = Signature::from_bytes(&sig.0);
        vk.verify(msg, &signature)
            .map_err(|_| WireError::BadSignature)
    }
}

impl std::fmt::Debug for AgentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "AgentId({}…)", &self.to_hex()[..8])
    }
}

impl Serialize for AgentId {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.to_hex())
    }
}

impl<'de> Deserialize<'de> for AgentId {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        AgentId::from_hex(&s).map_err(serde::de::Error::custom)
    }
}

/// `agentbbs-core/src/identity.rs:96-137` — a detached Ed25519 signature (64
/// bytes), serialized as hex.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct SignatureBytes([u8; 64]);

impl SignatureBytes {
    pub fn to_hex(&self) -> String {
        hex::encode(self.0)
    }

    pub fn from_hex(s: &str) -> Result<Self, WireError> {
        let raw = hex::decode(s.trim()).map_err(|e| WireError::Malformed {
            field: "signature hex",
            reason: e.to_string(),
        })?;
        let arr: [u8; 64] = raw.try_into().map_err(|_| WireError::Malformed {
            field: "signature",
            reason: "expected 64 bytes".into(),
        })?;
        Ok(SignatureBytes(arr))
    }
}

impl std::fmt::Debug for SignatureBytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SignatureBytes({}…)", &self.to_hex()[..16])
    }
}

impl Serialize for SignatureBytes {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.to_hex())
    }
}

impl<'de> Deserialize<'de> for SignatureBytes {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        SignatureBytes::from_hex(&s).map_err(serde::de::Error::custom)
    }
}

/// `agentbbs-core/src/identity.rs:144-179` — a local, secret identity: the
/// Ed25519 signing key plus its public id. Never transmitted.
pub struct Identity {
    signing: SigningKey,
    id: AgentId,
}

impl Identity {
    /// Generate a fresh random identity using the OS RNG.
    pub fn generate() -> Self {
        let mut rng = rand_core::OsRng;
        let signing = SigningKey::generate(&mut rng);
        let id = AgentId(signing.verifying_key().to_bytes());
        Identity { signing, id }
    }

    /// Reconstruct an identity from a 32-byte secret seed. Deterministic —
    /// used throughout this crate's tests so golden fixtures are
    /// reproducible without persisting a real key.
    pub fn from_seed(seed: &[u8; 32]) -> Self {
        let signing = SigningKey::from_bytes(seed);
        let id = AgentId(signing.verifying_key().to_bytes());
        Identity { signing, id }
    }

    pub fn id(&self) -> AgentId {
        self.id
    }

    pub fn sign(&self, msg: &[u8]) -> SignatureBytes {
        SignatureBytes(self.signing.sign(msg).to_bytes())
    }
}

impl std::fmt::Debug for Identity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Identity({}…)", &self.id.to_hex()[..8])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sign_verify_roundtrip() {
        let id = Identity::generate();
        let msg = b"post: hello agents";
        let sig = id.sign(msg);
        assert!(id.id().verify(msg, &sig).is_ok());
    }

    #[test]
    fn tampered_message_fails() {
        let id = Identity::generate();
        let sig = id.sign(b"original");
        assert!(matches!(
            id.id().verify(b"tampered", &sig),
            Err(WireError::BadSignature)
        ));
    }

    #[test]
    fn seed_is_deterministic() {
        let seed = [7u8; 32];
        let a = Identity::from_seed(&seed);
        let b = Identity::from_seed(&seed);
        assert_eq!(a.id(), b.id());
    }
}
