#![allow(dead_code)]

use latentmesh_air_core::*;

/// splitmix64: tiny deterministic PRNG, no dependencies.
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    pub fn bytes(&mut self, len: usize) -> Vec<u8> {
        (0..len).map(|_| self.next() as u8).collect()
    }
}

pub const TASK: u16 = 7;

pub fn registry(mandatory: Vec<FieldRef>) -> TaskRegistry {
    TaskRegistry::new(vec![TaskDefinition {
        task_id: TASK,
        version: 1,
        mandatory,
    }])
    .unwrap()
}

pub fn encode_field(field: FieldRef) -> (u8, u16) {
    match field {
        FieldRef::Symbol(id) => (0, id),
        FieldRef::Residual(id) => (1, u16::from(id)),
        FieldRef::Region(id) => (2, u16::from(id)),
        FieldRef::Reserved(r) => (3, r as u16),
    }
}

/// Encode entries exactly as given (callers control ordering).
pub fn mask_bytes(task: u16, entries: &[MaskEntry]) -> Vec<u8> {
    let mut out = task.to_le_bytes().to_vec();
    out.push(entries.len() as u8);
    out.push(0);
    for e in entries {
        let (kind, id) = encode_field(e.field);
        out.push(kind);
        out.extend_from_slice(&id.to_le_bytes());
        out.push(((e.class as u8) << 6) | (e.rank & 0x3f));
    }
    out
}

pub fn entry(field: FieldRef, class: FieldClass, rank: u8) -> MaskEntry {
    MaskEntry { field, class, rank }
}

pub fn bind(
    reg: &TaskRegistry,
    entries: &[MaskEntry],
) -> std::result::Result<BoundMask, SelectorError> {
    reg.bind(UnboundMask::parse(&mask_bytes(TASK, entries))?)
}

pub fn random_value(rng: &mut Rng) -> SymbolValue {
    match rng.below(5) {
        0 => SymbolValue::Bool(rng.below(2) == 1),
        1 => SymbolValue::I64(rng.next() as i64),
        2 => SymbolValue::U64(rng.next()),
        3 => SymbolValue::Q16_16(rng.next() as i32),
        _ => {
            let n = rng.below(33) as usize;
            SymbolValue::Bytes(rng.bytes(n))
        }
    }
}

pub fn random_delta(rng: &mut Rng) -> SemanticDelta {
    let mut before = CriticalState::new();
    let mut after = CriticalState::new();
    for id in 0..rng.below(12) as u16 {
        before.set(id, random_value(rng)).unwrap();
        if rng.below(3) != 0 {
            after.set(id, random_value(rng)).unwrap();
        }
    }
    let mut slots: Vec<u8> = (0..16).collect();
    let n = rng.below(17) as usize;
    let mut residuals = Vec::new();
    for _ in 0..n {
        let slot = slots.swap_remove(rng.below(slots.len() as u64) as usize);
        residuals.push(Residual {
            slot,
            importance: rng.next() as u8,
            scale_exp: rng.next() as i8,
            values: (0..rng.below(65)).map(|_| rng.next() as i8).collect(),
        });
    }
    let (s, e, m) = (rng.next() as u32, rng.next() as u32, rng.next() as u32);
    SemanticDelta::between(s, e, m, &before, &after, residuals).unwrap()
}

pub fn wrap(delta: &SemanticDelta, signed: bool) -> SemanticEnvelope {
    let sig = signed.then_some([0xA5; 64]);
    SemanticEnvelope::wrap_delta(delta, 3, 9, sig).unwrap()
}

pub fn opaque(rng: &mut Rng, len: usize, signed: bool) -> SemanticEnvelope {
    SemanticEnvelope {
        class: SemanticClass::Telemetry,
        priority: 1,
        source_id: 1,
        epoch: 1,
        message_id: 1,
        logical_sequence: 1,
        state_hash: [0; 16],
        body: rng.bytes(len),
        signature: signed.then_some([0x5A; 64]),
    }
}

/// Random mask over residual slots plus Required symbol/reserved noise.
pub fn random_residual_mask(rng: &mut Rng) -> Vec<MaskEntry> {
    let mut entries = Vec::new();
    for slot in 0..16u8 {
        if rng.below(3) != 0 {
            let class = match rng.below(3) {
                0 => FieldClass::Required,
                1 => FieldClass::Optional,
                _ => FieldClass::Discardable,
            };
            entries.push(entry(FieldRef::Residual(slot), class, rng.below(64) as u8));
        }
    }
    entries
}

pub const RESERVED: [ReservedField; 9] = [
    ReservedField::Signature,
    ReservedField::ReplaySequence,
    ReservedField::Epoch,
    ReservedField::MessageId,
    ReservedField::SourceId,
    ReservedField::StateHash,
    ReservedField::Provenance,
    ReservedField::CriticalHash,
    ReservedField::SafetyMetadata,
];
/// Canonical (ascending) mask mixing Reserved, Symbol and Residual entries
/// with random classes and ranks, including the adversarial rank 63.
pub fn hostile_mask(rng: &mut Rng) -> Vec<MaskEntry> {
    let mut entries = Vec::new();
    let hostile = rng.below(4) == 0;
    for sym in 0..12u16 {
        if hostile && rng.below(4) == 0 {
            entries.push(entry(
                FieldRef::Symbol(sym),
                random_class(rng),
                rng_rank(rng),
            ));
        }
    }
    for slot in 0..16u8 {
        if rng.below(3) != 0 {
            entries.push(entry(
                FieldRef::Residual(slot),
                random_class(rng),
                rng_rank(rng),
            ));
        }
    }
    for r in RESERVED {
        if hostile && rng.below(5) == 0 {
            entries.push(entry(
                FieldRef::Reserved(r),
                random_class(rng),
                rng_rank(rng),
            ));
        }
    }
    entries
}

pub fn random_class(rng: &mut Rng) -> FieldClass {
    match rng.below(3) {
        0 => FieldClass::Required,
        1 => FieldClass::Optional,
        _ => FieldClass::Discardable,
    }
}

pub fn rng_rank(rng: &mut Rng) -> u8 {
    [0u8, 63, rng.below(64) as u8][rng.below(3) as usize]
}
