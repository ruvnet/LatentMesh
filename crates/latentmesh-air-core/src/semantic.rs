use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use sha2::{Digest, Sha256};

use crate::{AirError, Result, MAX_MESSAGE_BYTES};

pub const CRITICAL_HASH_BYTES: usize = 16;
pub const MAX_CRITICAL_FIELDS: usize = 64;
pub const MAX_SYMBOL_BYTES: usize = 32;
pub const MAX_RESIDUALS: usize = 16;
pub const MAX_RESIDUAL_VALUES: usize = 64;

const DELTA_MAGIC: &[u8; 4] = b"LMAD";
const DELTA_VERSION: u8 = 1;

/// Deterministic symbolic values. Critical facts never pass through a learned
/// decoder and floating-point values are represented explicitly as fixed point.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SymbolValue {
    Bool(bool),
    I64(i64),
    U64(u64),
    /// Signed Q16.16 fixed-point value.
    Q16_16(i32),
    Bytes(Vec<u8>),
}

impl SymbolValue {
    fn type_code(&self) -> u8 {
        match self {
            Self::Bool(_) => 1,
            Self::I64(_) => 2,
            Self::U64(_) => 3,
            Self::Q16_16(_) => 4,
            Self::Bytes(_) => 5,
        }
    }

    fn encoded_len(&self) -> usize {
        match self {
            Self::Bool(_) => 1,
            Self::I64(_) | Self::U64(_) => 8,
            Self::Q16_16(_) => 4,
            Self::Bytes(bytes) => bytes.len(),
        }
    }

    fn validate(&self) -> Result<()> {
        if matches!(self, Self::Bytes(bytes) if bytes.len() > MAX_SYMBOL_BYTES) {
            return Err(AirError::LimitExceeded);
        }
        Ok(())
    }

    fn append_value(&self, out: &mut Vec<u8>) {
        match self {
            Self::Bool(value) => out.push(u8::from(*value)),
            Self::I64(value) => out.extend_from_slice(&value.to_be_bytes()),
            Self::U64(value) => out.extend_from_slice(&value.to_be_bytes()),
            Self::Q16_16(value) => out.extend_from_slice(&value.to_be_bytes()),
            Self::Bytes(value) => out.extend_from_slice(value),
        }
    }

    fn decode(type_code: u8, value: &[u8]) -> Result<Self> {
        Ok(match type_code {
            1 if value.len() == 1 && value[0] <= 1 => Self::Bool(value[0] != 0),
            2 if value.len() == 8 => Self::I64(i64::from_be_bytes(
                value.try_into().map_err(|_| AirError::InvalidLength)?,
            )),
            3 if value.len() == 8 => Self::U64(u64::from_be_bytes(
                value.try_into().map_err(|_| AirError::InvalidLength)?,
            )),
            4 if value.len() == 4 => Self::Q16_16(i32::from_be_bytes(
                value.try_into().map_err(|_| AirError::InvalidLength)?,
            )),
            5 if value.len() <= MAX_SYMBOL_BYTES => Self::Bytes(value.to_vec()),
            _ => return Err(AirError::InvalidSemanticValue),
        })
    }
}

/// Canonically ordered critical world state keyed by schema-assigned field ID.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CriticalState {
    fields: BTreeMap<u16, SymbolValue>,
}

impl CriticalState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set(&mut self, field_id: u16, value: SymbolValue) -> Result<()> {
        value.validate()?;
        if !self.fields.contains_key(&field_id) && self.fields.len() == MAX_CRITICAL_FIELDS {
            return Err(AirError::LimitExceeded);
        }
        self.fields.insert(field_id, value);
        Ok(())
    }

    pub fn remove(&mut self, field_id: u16) -> Option<SymbolValue> {
        self.fields.remove(&field_id)
    }

    pub fn get(&self, field_id: u16) -> Option<&SymbolValue> {
        self.fields.get(&field_id)
    }

    pub fn len(&self) -> usize {
        self.fields.len()
    }

    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&u16, &SymbolValue)> {
        self.fields.iter()
    }

    /// SHA-256 truncated to 128 bits over a canonical, typed representation.
    /// This is an agreement/integrity hash, not an authentication mechanism.
    pub fn critical_hash(&self) -> [u8; CRITICAL_HASH_BYTES] {
        let mut hasher = Sha256::new();
        hasher.update(b"LatentMesh-Critical-State-v1");
        hasher.update((self.fields.len() as u16).to_be_bytes());
        for (field_id, value) in &self.fields {
            hasher.update(field_id.to_be_bytes());
            hasher.update([value.type_code(), value.encoded_len() as u8]);
            let mut encoded = Vec::with_capacity(value.encoded_len());
            value.append_value(&mut encoded);
            hasher.update(encoded);
        }
        let digest = hasher.finalize();
        let mut out = [0_u8; CRITICAL_HASH_BYTES];
        out.copy_from_slice(&digest[..CRITICAL_HASH_BYTES]);
        out
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SymbolUpdate {
    Set { field_id: u16, value: SymbolValue },
    Delete { field_id: u16 },
}

impl SymbolUpdate {
    fn field_id(&self) -> u16 {
        match self {
            Self::Set { field_id, .. } | Self::Delete { field_id } => *field_id,
        }
    }
}

/// Quantized learned residual. It may improve noncritical reconstruction but
/// can never create or modify a critical symbolic field.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Residual {
    pub slot: u8,
    pub importance: u8,
    /// Residual values represent `q * 2^scale_exp`.
    pub scale_exp: i8,
    pub values: Vec<i8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemanticDelta {
    pub source_id: u32,
    pub epoch: u32,
    pub message_id: u32,
    pub base_hash: [u8; CRITICAL_HASH_BYTES],
    pub result_hash: [u8; CRITICAL_HASH_BYTES],
    pub updates: Vec<SymbolUpdate>,
    pub residuals: Vec<Residual>,
}

impl SemanticDelta {
    pub fn between(
        source_id: u32,
        epoch: u32,
        message_id: u32,
        before: &CriticalState,
        after: &CriticalState,
        residuals: Vec<Residual>,
    ) -> Result<Self> {
        let mut updates = Vec::new();
        for (&field_id, old) in before.iter() {
            match after.get(field_id) {
                None => updates.push(SymbolUpdate::Delete { field_id }),
                Some(new) if new != old => updates.push(SymbolUpdate::Set {
                    field_id,
                    value: new.clone(),
                }),
                _ => {}
            }
        }
        for (&field_id, value) in after.iter() {
            if before.get(field_id).is_none() {
                updates.push(SymbolUpdate::Set {
                    field_id,
                    value: value.clone(),
                });
            }
        }
        updates.sort_by_key(SymbolUpdate::field_id);
        let delta = Self {
            source_id,
            epoch,
            message_id,
            base_hash: before.critical_hash(),
            result_hash: after.critical_hash(),
            updates,
            residuals,
        };
        delta.validate()?;
        Ok(delta)
    }

    pub fn validate(&self) -> Result<()> {
        if self.updates.len() > MAX_CRITICAL_FIELDS || self.residuals.len() > MAX_RESIDUALS {
            return Err(AirError::LimitExceeded);
        }
        let mut update_ids: Vec<u16> = self.updates.iter().map(SymbolUpdate::field_id).collect();
        update_ids.sort_unstable();
        if update_ids.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(AirError::InvalidSemanticValue);
        }
        for update in &self.updates {
            if let SymbolUpdate::Set { value, .. } = update {
                value.validate()?;
            }
        }
        let mut slots: Vec<u8> = self.residuals.iter().map(|item| item.slot).collect();
        slots.sort_unstable();
        if slots.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(AirError::InvalidSemanticValue);
        }
        if self
            .residuals
            .iter()
            .any(|residual| residual.values.len() > MAX_RESIDUAL_VALUES)
        {
            return Err(AirError::LimitExceeded);
        }
        Ok(())
    }

    /// Encode with canonical update and residual ordering. Equivalent deltas
    /// produce byte-identical output regardless of insertion order.
    pub fn encode(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let mut updates: Vec<&SymbolUpdate> = self.updates.iter().collect();
        updates.sort_by_key(|item| item.field_id());
        let mut residuals: Vec<&Residual> = self.residuals.iter().collect();
        residuals.sort_by_key(|item| item.slot);

        let mut out = Vec::new();
        out.extend_from_slice(DELTA_MAGIC);
        out.push(DELTA_VERSION);
        out.push(u8::from(!residuals.is_empty()));
        out.extend_from_slice(&self.source_id.to_be_bytes());
        out.extend_from_slice(&self.epoch.to_be_bytes());
        out.extend_from_slice(&self.message_id.to_be_bytes());
        out.extend_from_slice(&self.base_hash);
        out.extend_from_slice(&self.result_hash);
        out.push(updates.len() as u8);
        out.push(residuals.len() as u8);

        for update in updates {
            out.extend_from_slice(&update.field_id().to_be_bytes());
            match update {
                SymbolUpdate::Delete { .. } => out.extend_from_slice(&[0, 0]),
                SymbolUpdate::Set { value, .. } => {
                    out.push(value.type_code());
                    out.push(value.encoded_len() as u8);
                    value.append_value(&mut out);
                }
            }
        }
        for residual in residuals {
            out.extend_from_slice(&[
                residual.slot,
                residual.importance,
                residual.scale_exp as u8,
                residual.values.len() as u8,
            ]);
            out.extend(residual.values.iter().map(|value| *value as u8));
        }
        if out.len() > MAX_MESSAGE_BYTES {
            return Err(AirError::LimitExceeded);
        }
        Ok(out)
    }

    pub fn decode(input: &[u8]) -> Result<Self> {
        const FIXED_LEN: usize = 52;
        if input.len() < FIXED_LEN {
            return Err(AirError::Truncated);
        }
        if input.len() > MAX_MESSAGE_BYTES {
            return Err(AirError::LimitExceeded);
        }
        if &input[..4] != DELTA_MAGIC {
            return Err(AirError::InvalidMarker);
        }
        if input[4] != DELTA_VERSION {
            return Err(AirError::UnsupportedVersion);
        }
        if input[5] & !1 != 0 {
            return Err(AirError::InvalidFlags);
        }
        let update_count = usize::from(input[50]);
        let residual_count = usize::from(input[51]);
        if update_count > MAX_CRITICAL_FIELDS || residual_count > MAX_RESIDUALS {
            return Err(AirError::LimitExceeded);
        }
        if (input[5] & 1 != 0) != (residual_count != 0) {
            return Err(AirError::InvalidFlags);
        }

        let mut cursor = Reader::new(&input[6..]);
        let source_id = cursor.u32()?;
        let epoch = cursor.u32()?;
        let message_id = cursor.u32()?;
        let base_hash = cursor.array_16()?;
        let result_hash = cursor.array_16()?;
        // Counts were pre-read to validate before allocating; consume them.
        if usize::from(cursor.u8()?) != update_count || usize::from(cursor.u8()?) != residual_count
        {
            return Err(AirError::InvalidLength);
        }

        let mut updates = Vec::with_capacity(update_count);
        let mut previous_field = None;
        for _ in 0..update_count {
            let field_id = cursor.u16()?;
            if previous_field.is_some_and(|previous| field_id <= previous) {
                return Err(AirError::InvalidSemanticValue);
            }
            previous_field = Some(field_id);
            let type_code = cursor.u8()?;
            let value_len = usize::from(cursor.u8()?);
            if value_len > MAX_SYMBOL_BYTES {
                return Err(AirError::LimitExceeded);
            }
            if type_code == 0 {
                if value_len != 0 {
                    return Err(AirError::InvalidSemanticValue);
                }
                updates.push(SymbolUpdate::Delete { field_id });
            } else {
                let value = SymbolValue::decode(type_code, cursor.take(value_len)?)?;
                updates.push(SymbolUpdate::Set { field_id, value });
            }
        }

        let mut residuals = Vec::with_capacity(residual_count);
        let mut previous_slot = None;
        for _ in 0..residual_count {
            let slot = cursor.u8()?;
            if previous_slot.is_some_and(|previous| slot <= previous) {
                return Err(AirError::InvalidSemanticValue);
            }
            previous_slot = Some(slot);
            let importance = cursor.u8()?;
            let scale_exp = cursor.u8()? as i8;
            let value_len = usize::from(cursor.u8()?);
            if value_len > MAX_RESIDUAL_VALUES {
                return Err(AirError::LimitExceeded);
            }
            let values = cursor
                .take(value_len)?
                .iter()
                .map(|value| *value as i8)
                .collect();
            residuals.push(Residual {
                slot,
                importance,
                scale_exp,
                values,
            });
        }
        if !cursor.is_empty() {
            return Err(AirError::TrailingBytes);
        }
        let delta = Self {
            source_id,
            epoch,
            message_id,
            base_hash,
            result_hash,
            updates,
            residuals,
        };
        delta.validate()?;
        Ok(delta)
    }

    /// Apply only deterministic symbolic operations and verify both hashes.
    /// Learned residuals are returned separately to an optional, noncritical
    /// consumer and are never interpreted by this method.
    pub fn apply(&self, base: &CriticalState) -> Result<CriticalState> {
        self.validate()?;
        if base.critical_hash() != self.base_hash {
            return Err(AirError::BaseStateMismatch);
        }
        let mut result = base.clone();
        for update in &self.updates {
            match update {
                SymbolUpdate::Set { field_id, value } => result.set(*field_id, value.clone())?,
                SymbolUpdate::Delete { field_id } => {
                    result.remove(*field_id);
                }
            }
        }
        if result.critical_hash() != self.result_hash {
            return Err(AirError::ResultStateMismatch);
        }
        Ok(result)
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, at: 0 }
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8]> {
        let end = self.at.checked_add(count).ok_or(AirError::InvalidLength)?;
        let value = self.bytes.get(self.at..end).ok_or(AirError::Truncated)?;
        self.at = end;
        Ok(value)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_be_bytes(
            self.take(2)?.try_into().map_err(|_| AirError::Truncated)?,
        ))
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_be_bytes(
            self.take(4)?.try_into().map_err(|_| AirError::Truncated)?,
        ))
    }

    fn array_16(&mut self) -> Result<[u8; 16]> {
        self.take(16)?.try_into().map_err(|_| AirError::Truncated)
    }

    fn is_empty(&self) -> bool {
        self.at == self.bytes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn states() -> (CriticalState, CriticalState) {
        let mut before = CriticalState::new();
        before.set(1, SymbolValue::Bool(true)).unwrap();
        before.set(2, SymbolValue::U64(827)).unwrap();
        before
            .set(9, SymbolValue::Bytes(alloc::vec![1, 2]))
            .unwrap();
        let mut after = before.clone();
        after.set(2, SymbolValue::U64(828)).unwrap();
        after.set(3, SymbolValue::Q16_16(71 << 16)).unwrap();
        after.remove(9);
        (before, after)
    }

    #[test]
    fn deterministic_delta_round_trip_and_exact_hash_agreement() {
        let (before, after) = states();
        let delta = SemanticDelta::between(
            7,
            3,
            99,
            &before,
            &after,
            alloc::vec![Residual {
                slot: 2,
                importance: 80,
                scale_exp: -3,
                values: alloc::vec![-2, 7, 11],
            }],
        )
        .unwrap();
        let encoded = delta.encode().unwrap();
        let decoded = SemanticDelta::decode(&encoded).unwrap();
        assert_eq!(decoded, delta);
        assert_eq!(decoded.apply(&before).unwrap(), after);
        assert_eq!(
            decoded.apply(&before).unwrap().critical_hash(),
            delta.result_hash
        );
    }

    #[test]
    fn learned_residual_cannot_reconstruct_critical_state() {
        let (before, after) = states();
        let mut delta = SemanticDelta::between(1, 1, 1, &before, &after, alloc::vec![]).unwrap();
        delta.residuals.push(Residual {
            slot: 0,
            importance: 255,
            scale_exp: 0,
            values: alloc::vec![127; MAX_RESIDUAL_VALUES],
        });
        assert_eq!(delta.apply(&before).unwrap(), after);
    }

    #[test]
    fn wrong_base_and_tampered_result_are_rejected() {
        let (before, after) = states();
        let mut delta = SemanticDelta::between(1, 1, 1, &before, &after, alloc::vec![]).unwrap();
        let empty = CriticalState::new();
        assert_eq!(delta.apply(&empty), Err(AirError::BaseStateMismatch));
        delta.result_hash[0] ^= 1;
        assert_eq!(delta.apply(&before), Err(AirError::ResultStateMismatch));
    }

    #[test]
    fn canonical_encoding_ignores_input_order() {
        let (before, after) = states();
        let mut first = SemanticDelta::between(1, 1, 1, &before, &after, alloc::vec![]).unwrap();
        let original = first.encode().unwrap();
        first.updates.reverse();
        assert_eq!(first.encode().unwrap(), original);
    }
}
