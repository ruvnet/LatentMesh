//! Task-relevance selector (ADR-049, phase 1: accounting only, no encoder).
//!
//! Masks are untrusted until [`TaskRegistry::bind`] succeeds; the planners
//! accept only [`BoundMask`]. Unlisted fields default to `Required`, and
//! reserved, symbolic and registry-mandatory fields can never be downgraded.

mod plan;

use alloc::vec::Vec;
use core::fmt;

pub use plan::{plan_delta, plan_envelope, TransmissionPlan};

pub const MAX_MASK_ENTRIES: usize = 80;
pub const MAX_MASK_WIRE_BYTES: usize = 4 + 4 * MAX_MASK_ENTRIES;
pub const MAX_REGIONS: usize = 16;
pub const REGION_BYTES: usize = 16;
pub const MAX_TASKS: usize = 16;
pub const MAX_SELECTOR_STEPS: usize = 512;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldClass {
    Required,
    Optional,
    Discardable,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ReservedField {
    Signature,
    ReplaySequence,
    Epoch,
    MessageId,
    SourceId,
    StateHash,
    Provenance,
    CriticalHash,
    SafetyMetadata,
}

impl ReservedField {
    fn from_wire(id: u16) -> Result<Self, SelectorError> {
        Ok(match id {
            0 => Self::Signature,
            1 => Self::ReplaySequence,
            2 => Self::Epoch,
            3 => Self::MessageId,
            4 => Self::SourceId,
            5 => Self::StateHash,
            6 => Self::Provenance,
            7 => Self::CriticalHash,
            8 => Self::SafetyMetadata,
            _ => return Err(SelectorError::Malformed),
        })
    }
}

/// Ordering matches the wire ordering `(kind, id)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum FieldRef {
    Symbol(u16),
    Residual(u8),
    Region(u8),
    Reserved(ReservedField),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MaskEntry {
    pub field: FieldRef,
    pub class: FieldClass,
    /// 0..=63; lower ranks are kept first among `Optional` entries.
    pub rank: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectorError {
    Malformed,
    TooManyEntries,
    NotCanonical,
    DuplicateEntry,
    BadClass,
    UnknownTask,
    ReservedDowngrade,
    MandatoryDowngrade,
    SymbolNotSelectable,
    WorkLimit,
}

impl fmt::Display for SelectorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Malformed => "malformed selector mask",
            Self::TooManyEntries => "selector mask exceeds entry or size limit",
            Self::NotCanonical => "selector mask entries are not strictly ascending",
            Self::DuplicateEntry => "duplicate selector mask entry",
            Self::BadClass => "invalid selector field class",
            Self::UnknownTask => "task is not registered",
            Self::ReservedDowngrade => "reserved field cannot be downgraded",
            Self::MandatoryDowngrade => "task-mandatory field cannot be downgraded",
            Self::SymbolNotSelectable => "symbolic update cannot be optional or discardable",
            Self::WorkLimit => "selector work limit exceeded",
        })
    }
}

#[cfg(feature = "std")]
impl std::error::Error for SelectorError {}

/// Parsed but unvalidated against any registry. Cannot reach the planner.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnboundMask {
    task_id: u16,
    entries: Vec<MaskEntry>,
}

/// A mask validated by [`TaskRegistry::bind`]. Constructible only there.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BoundMask {
    task_id: u16,
    entries: Vec<MaskEntry>,
}

impl BoundMask {
    pub fn task_id(&self) -> u16 {
        self.task_id
    }

    pub(crate) fn entries(&self) -> &[MaskEntry] {
        &self.entries
    }
}

impl UnboundMask {
    pub fn task_id(&self) -> u16 {
        self.task_id
    }

    pub fn entries(&self) -> &[MaskEntry] {
        &self.entries
    }

    /// Wire form: `[task_id u16 LE][count u8][0x00]` then 4-byte entries
    /// `[kind u8][id u16 LE][class << 6 | rank]`, strictly ascending.
    pub fn parse(bytes: &[u8]) -> Result<Self, SelectorError> {
        if bytes.len() < 4 {
            return Err(SelectorError::Malformed);
        }
        if bytes.len() > MAX_MASK_WIRE_BYTES {
            return Err(SelectorError::TooManyEntries);
        }
        let count = usize::from(bytes[2]);
        if count > MAX_MASK_ENTRIES {
            return Err(SelectorError::TooManyEntries);
        }
        if bytes[3] != 0 || bytes.len() != 4 + 4 * count {
            return Err(SelectorError::Malformed);
        }
        let mut entries: Vec<MaskEntry> = Vec::with_capacity(count);
        for chunk in bytes[4..].chunks_exact(4) {
            let entry = parse_entry(chunk)?;
            if let Some(prev) = entries.last() {
                match prev.field.cmp(&entry.field) {
                    core::cmp::Ordering::Equal => return Err(SelectorError::DuplicateEntry),
                    core::cmp::Ordering::Greater => return Err(SelectorError::NotCanonical),
                    core::cmp::Ordering::Less => {}
                }
            }
            entries.push(entry);
        }
        Ok(Self {
            task_id: u16::from_le_bytes([bytes[0], bytes[1]]),
            entries,
        })
    }
}

fn parse_entry(chunk: &[u8]) -> Result<MaskEntry, SelectorError> {
    let id = u16::from_le_bytes([chunk[1], chunk[2]]);
    let small = u8::try_from(id).map_err(|_| SelectorError::Malformed);
    let field = match chunk[0] {
        0 => FieldRef::Symbol(id),
        1 => FieldRef::Residual(small?),
        2 => {
            let region = small?;
            if usize::from(region) >= MAX_REGIONS {
                return Err(SelectorError::Malformed);
            }
            FieldRef::Region(region)
        }
        3 => FieldRef::Reserved(ReservedField::from_wire(id)?),
        _ => return Err(SelectorError::Malformed),
    };
    let class = match chunk[3] >> 6 {
        0 => FieldClass::Required,
        1 => FieldClass::Optional,
        2 => FieldClass::Discardable,
        _ => return Err(SelectorError::BadClass),
    };
    Ok(MaskEntry {
        field,
        class,
        rank: chunk[3] & 0x3f,
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TaskDefinition {
    pub task_id: u16,
    pub version: u8,
    /// Strictly ascending, at most [`MAX_MASK_ENTRIES`].
    pub mandatory: Vec<FieldRef>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TaskRegistry {
    defs: Vec<TaskDefinition>,
}

impl TaskRegistry {
    pub fn new(defs: Vec<TaskDefinition>) -> Result<Self, SelectorError> {
        if defs.len() > MAX_TASKS {
            return Err(SelectorError::TooManyEntries);
        }
        for (i, def) in defs.iter().enumerate() {
            if defs[..i].iter().any(|prev| prev.task_id == def.task_id) {
                return Err(SelectorError::DuplicateEntry);
            }
            if def.mandatory.len() > MAX_MASK_ENTRIES {
                return Err(SelectorError::TooManyEntries);
            }
            for pair in def.mandatory.windows(2) {
                match pair[0].cmp(&pair[1]) {
                    core::cmp::Ordering::Equal => return Err(SelectorError::DuplicateEntry),
                    core::cmp::Ordering::Greater => return Err(SelectorError::NotCanonical),
                    core::cmp::Ordering::Less => {}
                }
            }
        }
        Ok(Self { defs })
    }

    pub fn bind(&self, m: UnboundMask) -> Result<BoundMask, SelectorError> {
        let def = self
            .defs
            .iter()
            .find(|d| d.task_id == m.task_id)
            .ok_or(SelectorError::UnknownTask)?;
        for entry in &m.entries {
            check_entry(def, entry)?;
        }
        Ok(BoundMask {
            task_id: m.task_id,
            entries: m.entries,
        })
    }
}

fn check_entry(def: &TaskDefinition, entry: &MaskEntry) -> Result<(), SelectorError> {
    if entry.class == FieldClass::Required {
        return Ok(());
    }
    match entry.field {
        FieldRef::Reserved(_) => Err(SelectorError::ReservedDowngrade),
        FieldRef::Symbol(_) => Err(SelectorError::SymbolNotSelectable),
        field if def.mandatory.binary_search(&field).is_ok() => {
            Err(SelectorError::MandatoryDowngrade)
        }
        _ => Ok(()),
    }
}
