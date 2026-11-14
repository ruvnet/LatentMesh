//! Byte accounting for a bound mask. Sizes are computed here and pinned to
//! the real encoders by tests (ADR-049 invariant 8).

use alloc::vec::Vec;

use super::{
    BoundMask, FieldClass, FieldRef, MaskEntry, SelectorError, MAX_REGIONS, MAX_SELECTOR_STEPS,
    REGION_BYTES,
};
use crate::{SemanticClass, SemanticDelta, SemanticEnvelope, SymbolUpdate, SymbolValue};

const DELTA_FIXED_BYTES: usize = 52;
const UPDATE_HEADER_BYTES: usize = 4;
const RESIDUAL_HEADER_BYTES: usize = 4;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransmissionPlan {
    pub task_id: u16,
    pub fixed_bytes: usize,
    pub kept_symbols: Vec<u16>,
    pub kept_residual_slots: Vec<u8>,
    pub kept_regions: Vec<u8>,
    pub dropped_residual_slots: Vec<u8>,
    pub dropped_regions: Vec<u8>,
    pub original_bytes: usize,
    pub planned_bytes: usize,
}

struct Steps(usize);

impl Steps {
    fn tick(&mut self, n: usize) -> Result<(), SelectorError> {
        self.0 = self.0.saturating_add(n);
        if self.0 > MAX_SELECTOR_STEPS {
            return Err(SelectorError::WorkLimit);
        }
        Ok(())
    }
}

struct Item {
    id: u8,
    size: usize,
    field: FieldRef,
}

struct Selection {
    kept: Vec<u8>,
    dropped: Vec<u8>,
    planned: usize,
    selectable: usize,
}

/// Plan a bare delta. `original_bytes` equals `delta.encode()?.len()`.
pub fn plan_delta(
    d: &SemanticDelta,
    m: &BoundMask,
    byte_cap: Option<usize>,
) -> Result<TransmissionPlan, SelectorError> {
    delta_plan(d, m, byte_cap, 0, &mut Steps(0))
}

/// Plan an envelope. `original_bytes` equals `e.encoded_len()`.
pub fn plan_envelope(
    e: &SemanticEnvelope,
    m: &BoundMask,
    byte_cap: Option<usize>,
) -> Result<TransmissionPlan, SelectorError> {
    let mut steps = Steps(0);
    let overhead = e.encoded_len() - e.body.len();
    if e.class == SemanticClass::StateDelta {
        let delta = e.unwrap_delta().map_err(|_| SelectorError::Malformed)?;
        let plan = delta_plan(&delta, m, byte_cap, overhead, &mut steps)?;
        if plan.original_bytes != e.encoded_len() {
            return Err(SelectorError::Malformed);
        }
        return Ok(plan);
    }
    opaque_plan(e, m, byte_cap, overhead, &mut steps)
}

fn lookup(m: &BoundMask, field: FieldRef) -> Option<&MaskEntry> {
    let entries = m.entries();
    entries
        .binary_search_by(|e| e.field.cmp(&field))
        .ok()
        .and_then(|i| entries.get(i))
}

fn check_entries(
    m: &BoundMask,
    is_delta: bool,
    regions: usize,
    steps: &mut Steps,
) -> Result<(), SelectorError> {
    steps.tick(m.entries().len())?;
    for entry in m.entries() {
        let required = entry.class == FieldClass::Required;
        match entry.field {
            FieldRef::Reserved(_) if !required => return Err(SelectorError::ReservedDowngrade),
            FieldRef::Symbol(_) if !required => return Err(SelectorError::SymbolNotSelectable),
            FieldRef::Region(r) if is_delta || usize::from(r) >= regions => {
                return Err(SelectorError::Malformed)
            }
            FieldRef::Residual(_) if !is_delta => return Err(SelectorError::Malformed),
            _ => {}
        }
    }
    Ok(())
}

fn select(
    m: &BoundMask,
    items: &[Item],
    fixed: usize,
    cap: Option<usize>,
    steps: &mut Steps,
) -> Result<Selection, SelectorError> {
    let mut sel = Selection {
        kept: Vec::new(),
        dropped: Vec::new(),
        planned: fixed,
        selectable: 0,
    };
    let mut optional: Vec<(u8, u8, usize)> = Vec::new();
    steps.tick(items.len())?;
    for item in items {
        sel.selectable += item.size;
        let (class, rank) =
            lookup(m, item.field).map_or((FieldClass::Required, 0), |e| (e.class, e.rank));
        match class {
            FieldClass::Required => {
                sel.planned += item.size;
                sel.kept.push(item.id);
            }
            FieldClass::Discardable => sel.dropped.push(item.id),
            FieldClass::Optional => optional.push((rank, item.id, item.size)),
        }
    }
    optional.sort_unstable();
    steps.tick(optional.len())?;
    // Greedy prefix: once one Optional item does not fit, no later (higher
    // rank) item is kept, so rank order is always respected.
    let mut open = true;
    for (_, id, size) in optional {
        open = open && cap.map_or(true, |c| sel.planned.saturating_add(size) <= c);
        if open {
            sel.planned += size;
            sel.kept.push(id);
        } else {
            sel.dropped.push(id);
        }
    }
    sel.kept.sort_unstable();
    sel.dropped.sort_unstable();
    Ok(sel)
}

fn symbol_bytes(u: &SymbolUpdate) -> usize {
    UPDATE_HEADER_BYTES
        + match u {
            SymbolUpdate::Delete { .. } => 0,
            SymbolUpdate::Set { value, .. } => match value {
                SymbolValue::Bool(_) => 1,
                SymbolValue::I64(_) | SymbolValue::U64(_) => 8,
                SymbolValue::Q16_16(_) => 4,
                SymbolValue::Bytes(b) => b.len(),
            },
        }
}

fn delta_plan(
    d: &SemanticDelta,
    m: &BoundMask,
    cap: Option<usize>,
    overhead: usize,
    steps: &mut Steps,
) -> Result<TransmissionPlan, SelectorError> {
    d.validate().map_err(|_| SelectorError::Malformed)?;
    check_entries(m, true, 0, steps)?;
    steps.tick(d.updates.len())?;
    let mut kept_symbols: Vec<u16> = d
        .updates
        .iter()
        .map(|u| match u {
            SymbolUpdate::Set { field_id, .. } | SymbolUpdate::Delete { field_id } => *field_id,
        })
        .collect();
    kept_symbols.sort_unstable();
    let symbols: usize = d.updates.iter().map(symbol_bytes).sum();
    let fixed = overhead + DELTA_FIXED_BYTES + symbols;
    let items: Vec<Item> = d
        .residuals
        .iter()
        .map(|r| Item {
            id: r.slot,
            size: RESIDUAL_HEADER_BYTES + r.values.len(),
            field: FieldRef::Residual(r.slot),
        })
        .collect();
    let sel = select(m, &items, fixed, cap, steps)?;
    Ok(TransmissionPlan {
        task_id: m.task_id(),
        fixed_bytes: fixed,
        kept_symbols,
        kept_residual_slots: sel.kept,
        kept_regions: Vec::new(),
        dropped_residual_slots: sel.dropped,
        dropped_regions: Vec::new(),
        original_bytes: fixed + sel.selectable,
        planned_bytes: sel.planned,
    })
}

fn opaque_plan(
    e: &SemanticEnvelope,
    m: &BoundMask,
    cap: Option<usize>,
    overhead: usize,
    steps: &mut Steps,
) -> Result<TransmissionPlan, SelectorError> {
    let len = e.body.len();
    let regions = len.div_ceil(REGION_BYTES).min(MAX_REGIONS);
    check_entries(m, false, regions, steps)?;
    let covered = (regions * REGION_BYTES).min(len);
    let fixed = overhead + (len - covered);
    let items: Vec<Item> = (0..regions)
        .map(|i| Item {
            id: i as u8,
            size: REGION_BYTES.min(len - i * REGION_BYTES),
            field: FieldRef::Region(i as u8),
        })
        .collect();
    let sel = select(m, &items, fixed, cap, steps)?;
    Ok(TransmissionPlan {
        task_id: m.task_id(),
        fixed_bytes: fixed,
        kept_symbols: Vec::new(),
        kept_residual_slots: Vec::new(),
        kept_regions: sel.kept,
        dropped_residual_slots: Vec::new(),
        dropped_regions: sel.dropped,
        original_bytes: fixed + sel.selectable,
        planned_bytes: sel.planned,
    })
}
