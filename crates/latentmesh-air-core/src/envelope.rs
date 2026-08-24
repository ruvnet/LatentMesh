use alloc::vec::Vec;

use crate::{AirError, Result, SemanticClass, SemanticDelta, MAX_MESSAGE_BYTES};

pub const ENVELOPE_HEADER_BYTES: usize = 48;
pub const ENVELOPE_SIGNATURE_BYTES: usize = 64;
const ENVELOPE_MAGIC: &[u8; 4] = b"LMS1";
const ENVELOPE_VERSION: u8 = 1;

/// Cross-language semantic message envelope. A signature is transported but
/// not considered trusted until the application verifies it against the
/// canonical bytes returned by [`SemanticEnvelope::authentication_bytes`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemanticEnvelope {
    pub class: SemanticClass,
    pub priority: u8,
    pub source_id: u32,
    pub epoch: u32,
    pub message_id: u32,
    pub logical_sequence: u64,
    pub state_hash: [u8; 16],
    pub body: Vec<u8>,
    pub signature: Option<[u8; ENVELOPE_SIGNATURE_BYTES]>,
}

impl SemanticEnvelope {
    pub fn wrap_delta(
        delta: &SemanticDelta,
        priority: u8,
        logical_sequence: u64,
        signature: Option<[u8; ENVELOPE_SIGNATURE_BYTES]>,
    ) -> Result<Self> {
        let envelope = Self {
            class: SemanticClass::StateDelta,
            priority,
            source_id: delta.source_id,
            epoch: delta.epoch,
            message_id: delta.message_id,
            logical_sequence,
            state_hash: delta.result_hash,
            body: delta.encode()?,
            signature,
        };
        envelope.validate()?;
        Ok(envelope)
    }

    pub fn validate(&self) -> Result<()> {
        if self.priority > 15 {
            return Err(AirError::InvalidLength);
        }
        let signature_len = if self.signature.is_some() {
            ENVELOPE_SIGNATURE_BYTES
        } else {
            0
        };
        let total = ENVELOPE_HEADER_BYTES
            .checked_add(self.body.len())
            .and_then(|length| length.checked_add(signature_len))
            .ok_or(AirError::LimitExceeded)?;
        if self.body.len() > u16::MAX as usize || total > MAX_MESSAGE_BYTES {
            return Err(AirError::LimitExceeded);
        }
        Ok(())
    }

    pub fn encoded_len(&self) -> usize {
        ENVELOPE_HEADER_BYTES
            + self.body.len()
            + if self.signature.is_some() {
                ENVELOPE_SIGNATURE_BYTES
            } else {
                0
            }
    }

    pub fn encode(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let signature_len = if self.signature.is_some() {
            ENVELOPE_SIGNATURE_BYTES as u8
        } else {
            0
        };
        let mut out = Vec::with_capacity(self.encoded_len());
        out.extend_from_slice(ENVELOPE_MAGIC);
        out.push(ENVELOPE_VERSION);
        out.push(u8::from(self.signature.is_some()));
        out.push(self.class as u8);
        out.push(self.priority);
        out.extend_from_slice(&self.source_id.to_be_bytes());
        out.extend_from_slice(&self.epoch.to_be_bytes());
        out.extend_from_slice(&self.message_id.to_be_bytes());
        out.extend_from_slice(&self.logical_sequence.to_be_bytes());
        out.extend_from_slice(&(self.body.len() as u16).to_be_bytes());
        out.extend_from_slice(&self.state_hash);
        out.push(signature_len);
        out.push(0); // Reserved; canonical encoders MUST write zero.
        out.extend_from_slice(&self.body);
        if let Some(signature) = self.signature {
            out.extend_from_slice(&signature);
        }
        Ok(out)
    }

    pub fn decode(input: &[u8]) -> Result<Self> {
        if input.len() < ENVELOPE_HEADER_BYTES {
            return Err(AirError::Truncated);
        }
        if input.len() > MAX_MESSAGE_BYTES {
            return Err(AirError::LimitExceeded);
        }
        if &input[..4] != ENVELOPE_MAGIC {
            return Err(AirError::InvalidMarker);
        }
        if input[4] != ENVELOPE_VERSION {
            return Err(AirError::UnsupportedVersion);
        }
        let authenticated = match input[5] {
            0 => false,
            1 => true,
            _ => return Err(AirError::InvalidFlags),
        };
        let class = SemanticClass::try_from(input[6])?;
        let priority = input[7];
        if priority > 15 || input[47] != 0 {
            return Err(AirError::InvalidLength);
        }
        let signature_len = usize::from(input[46]);
        if (authenticated && signature_len != ENVELOPE_SIGNATURE_BYTES)
            || (!authenticated && signature_len != 0)
        {
            return Err(AirError::InvalidFlags);
        }
        let body_len = usize::from(u16::from_be_bytes([input[28], input[29]]));
        let expected = ENVELOPE_HEADER_BYTES
            .checked_add(body_len)
            .and_then(|length| length.checked_add(signature_len))
            .ok_or(AirError::LimitExceeded)?;
        if expected != input.len() {
            return Err(AirError::InvalidLength);
        }
        let source_id =
            u32::from_be_bytes(input[8..12].try_into().map_err(|_| AirError::Truncated)?);
        let epoch = u32::from_be_bytes(input[12..16].try_into().map_err(|_| AirError::Truncated)?);
        let message_id =
            u32::from_be_bytes(input[16..20].try_into().map_err(|_| AirError::Truncated)?);
        let logical_sequence =
            u64::from_be_bytes(input[20..28].try_into().map_err(|_| AirError::Truncated)?);
        let state_hash = input[30..46].try_into().map_err(|_| AirError::Truncated)?;
        let body_end = ENVELOPE_HEADER_BYTES + body_len;
        let signature = if authenticated {
            Some(
                input[body_end..expected]
                    .try_into()
                    .map_err(|_| AirError::Truncated)?,
            )
        } else {
            None
        };
        let envelope = Self {
            class,
            priority,
            source_id,
            epoch,
            message_id,
            logical_sequence,
            state_hash,
            body: input[ENVELOPE_HEADER_BYTES..body_end].to_vec(),
            signature,
        };
        envelope.validate()?;
        Ok(envelope)
    }

    /// Canonical material signed by an authenticated sender: the complete
    /// 48-byte header with `signature_len=64`, followed by the body. The actual
    /// signature bytes are excluded.
    pub fn authentication_bytes(&self) -> Result<Vec<u8>> {
        if self.signature.is_none() {
            return Err(AirError::InvalidFlags);
        }
        let encoded = self.encode()?;
        Ok(encoded[..encoded.len() - ENVELOPE_SIGNATURE_BYTES].to_vec())
    }

    /// Decode the deterministic symbolic/residual delta and bind every
    /// duplicated identity/hash field in the outer envelope to the body.
    pub fn unwrap_delta(&self) -> Result<SemanticDelta> {
        if self.class != SemanticClass::StateDelta {
            return Err(AirError::InvalidClass);
        }
        let delta = SemanticDelta::decode(&self.body)?;
        if delta.source_id != self.source_id
            || delta.epoch != self.epoch
            || delta.message_id != self.message_id
            || delta.result_hash != self.state_hash
        {
            return Err(AirError::ResultStateMismatch);
        }
        Ok(delta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CriticalState, SymbolValue};

    #[test]
    fn signed_envelope_round_trips_and_excludes_signature_from_auth_material() {
        let mut before = CriticalState::new();
        before.set(1, SymbolValue::Bool(false)).unwrap();
        let mut after = before.clone();
        after.set(1, SymbolValue::Bool(true)).unwrap();
        let delta = SemanticDelta::between(1, 2, 3, &before, &after, alloc::vec![]).unwrap();
        let envelope = SemanticEnvelope::wrap_delta(&delta, 15, 4, Some([0x5a; 64])).unwrap();
        let encoded = envelope.encode().unwrap();
        assert_eq!(SemanticEnvelope::decode(&encoded).unwrap(), envelope);
        assert_eq!(
            envelope.authentication_bytes().unwrap().len(),
            encoded.len() - 64
        );
        assert_eq!(envelope.unwrap_delta().unwrap(), delta);
    }

    #[test]
    fn reserved_and_identity_mismatch_are_rejected() {
        let envelope = SemanticEnvelope {
            class: SemanticClass::StateDelta,
            priority: 1,
            source_id: 1,
            epoch: 2,
            message_id: 3,
            logical_sequence: 4,
            state_hash: [0; 16],
            body: alloc::vec![1, 2],
            signature: None,
        };
        let mut encoded = envelope.encode().unwrap();
        encoded[47] = 1;
        assert_eq!(
            SemanticEnvelope::decode(&encoded),
            Err(AirError::InvalidLength)
        );
    }
}
