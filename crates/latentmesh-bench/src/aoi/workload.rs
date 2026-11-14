//! Common-random-number workload: every draw is materialized here, before any
//! baseline runs, so the three baselines are compared on identical inputs.

use latentmesh_air_core::{
    fragment_message, plan_envelope, BoundMask, FieldRef, FragmentMeta, FrameFlags, Residual,
    SemanticClass, SemanticDelta, SemanticEnvelope, SymbolUpdate, SymbolValue, TaskDefinition,
    TaskRegistry, UnboundMask, WireProfile,
};
use rand::{Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

use super::grid::Cell;
use super::utility::{max_weight, WEIGHT_RELEVANT_RESIDUAL, WEIGHT_SYMBOL};

pub const MSG_COUNT: usize = 512;
pub const GEN_PERIOD_MS: u64 = 250;
pub const SYMBOLS: usize = 6;
pub const RESIDUAL_SLOTS: usize = 8;
pub const RESIDUAL_VALUES: usize = 32;
pub const LOSS_DRAWS: usize = 8;
pub const TASK_ID: u16 = 1;
pub const FRAME_MTU: usize = 256;
pub const RESIDUAL_ITEM_BYTES: usize = 4 + RESIDUAL_VALUES;
pub const WHOLE: usize = 0;
pub const PRIORITY: usize = 1;
pub const RELEVANCE: usize = 2;
pub const BASELINES: [&str; 3] = ["whole_payload", "priority_field", "relevance_selected"];
const RUN_MS: u64 = MSG_COUNT as u64 * GEN_PERIOD_MS;

/// Per-message random draws (baseline independent).
pub struct MsgRand {
    pub gen_ms: u64,
    pub reorder_draw: u32,
    pub reorder_mag: u64,
    pub loss_draws: [u32; LOSS_DRAWS],
}

/// What one baseline puts on the link for one message.
#[derive(Clone, Copy, Debug)]
pub struct Payload {
    pub payload_bytes: u64,
    pub frame_bytes: u64,
    pub frames: usize,
    /// `W_delivered` if the message arrives intact.
    pub weight: u64,
}

pub struct Materialized {
    pub w_max: u64,
    pub outage_start_ms: u64,
    pub msgs: Vec<MsgRand>,
    pub payloads: [Vec<Payload>; 3],
    pub mandatory_violations: [u64; 3],
}

/// Frame count and total frame bytes of `bytes` through `fragment_message`.
pub fn frame_cost(sequence: u16, bytes: &[u8]) -> Result<(usize, u64), String> {
    let meta = FragmentMeta {
        profile: WireProfile::Meshtastic,
        flags: FrameFlags::default(),
        stream_id: 1,
        sequence,
        class: SemanticClass::StateDelta,
        priority: 0,
        state_tag: 0,
    };
    let frames = fragment_message(meta, bytes, FRAME_MTU).map_err(|e| format!("{e:?}"))?;
    Ok((
        frames.len(),
        frames.iter().map(|f| f.encoded_len() as u64).sum(),
    ))
}

/// Mask wire form (ADR-049 section 2): residual slots only, ascending.
pub fn mask_bytes(task: u16, required: &[bool; RESIDUAL_SLOTS]) -> Vec<u8> {
    let mut out = task.to_le_bytes().to_vec();
    out.push(RESIDUAL_SLOTS as u8);
    out.push(0);
    for (slot, &req) in required.iter().enumerate() {
        let class: u8 = if req { 0 } else { 2 };
        out.push(1); // kind: Residual
        out.extend_from_slice(&(slot as u16).to_le_bytes());
        out.push(class << 6);
    }
    out
}

fn relevant_set(rng: &mut ChaCha8Rng, rho_ppm: u64) -> [bool; RESIDUAL_SLOTS] {
    let mut order: Vec<usize> = (0..RESIDUAL_SLOTS).collect();
    for i in (1..order.len()).rev() {
        order.swap(i, rng.gen_range(0..=i));
    }
    let n = (rho_ppm * RESIDUAL_SLOTS as u64).div_ceil(1_000_000) as usize;
    let mut set = [false; RESIDUAL_SLOTS];
    for &slot in order.iter().take(n) {
        set[slot] = true;
    }
    set
}

fn bind_mask(relevant: &[bool; RESIDUAL_SLOTS]) -> Result<BoundMask, String> {
    let mandatory = (0..RESIDUAL_SLOTS)
        .filter(|&s| relevant[s])
        .map(|s| FieldRef::Residual(s as u8))
        .collect();
    let def = TaskDefinition {
        task_id: TASK_ID,
        version: 1,
        mandatory,
    };
    let reg = TaskRegistry::new(vec![def]).map_err(|e| format!("{e:?}"))?;
    let raw = UnboundMask::parse(&mask_bytes(TASK_ID, relevant)).map_err(|e| format!("{e:?}"))?;
    reg.bind(raw).map_err(|e| format!("{e:?}"))
}

fn draw_delta(rng: &mut ChaCha8Rng, id: u32, importance: &[u8; RESIDUAL_SLOTS]) -> SemanticDelta {
    let mut base = [0u8; 16];
    let mut result = [0u8; 16];
    rng.fill_bytes(&mut base);
    rng.fill_bytes(&mut result);
    let updates = (0..SYMBOLS)
        .map(|k| SymbolUpdate::Set {
            field_id: k as u16,
            value: SymbolValue::U64(rng.next_u64()),
        })
        .collect();
    let residuals = (0..RESIDUAL_SLOTS)
        .map(|slot| Residual {
            slot: slot as u8,
            importance: importance[slot],
            scale_exp: 0,
            values: (0..RESIDUAL_VALUES).map(|_| rng.next_u32() as i8).collect(),
        })
        .collect();
    SemanticDelta {
        source_id: 1,
        epoch: 1,
        message_id: id,
        base_hash: base,
        result_hash: result,
        updates,
        residuals,
    }
}

/// Greedy prefix by descending importance (ties: lower slot), stopping at the
/// first misfit, mirroring the selector's rank-order rule. Returns kept slots.
fn priority_keep(importance: &[u8; RESIDUAL_SLOTS], fixed: usize, cap: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..RESIDUAL_SLOTS).collect();
    order.sort_by(|&a, &b| importance[b].cmp(&importance[a]).then(a.cmp(&b)));
    let (mut planned, mut kept) = (fixed, Vec::new());
    for slot in order {
        if planned + RESIDUAL_ITEM_BYTES > cap {
            break;
        }
        planned += RESIDUAL_ITEM_BYTES;
        kept.push(slot);
    }
    kept
}

struct Ctx<'a> {
    relevant: [bool; RESIDUAL_SLOTS],
    mask: &'a BoundMask,
    cap: usize,
}

fn build_payloads(
    c: &Ctx,
    seq: u16,
    env: &SemanticEnvelope,
    importance: &[u8; RESIDUAL_SLOTS],
    out: &mut [Vec<Payload>; 3],
    viol: &mut [u64; 3],
) -> Result<(), String> {
    let plan = plan_envelope(env, c.mask, Some(c.cap)).map_err(|e| format!("{e:?}"))?;
    let bytes = env.encode().map_err(|e| format!("{e:?}"))?;
    let (frames, frame_bytes) = frame_cost(seq, &bytes)?;
    let n_rel = c.relevant.iter().filter(|&&r| r).count() as u64;
    let w_max = max_weight(SYMBOLS as u64, n_rel);
    out[WHOLE].push(Payload {
        payload_bytes: bytes.len() as u64,
        frame_bytes,
        frames,
        weight: w_max,
    });

    let kept = priority_keep(importance, plan.fixed_bytes, c.cap);
    let rel_kept = kept.iter().filter(|&&s| c.relevant[s]).count() as u64;
    let p_len = plan.fixed_bytes + kept.len() * RESIDUAL_ITEM_BYTES;
    let (pf, pb) = frame_cost(seq, &vec![0u8; p_len])?;
    out[PRIORITY].push(Payload {
        payload_bytes: p_len as u64,
        frame_bytes: pb,
        frames: pf,
        weight: SYMBOLS as u64 * WEIGHT_SYMBOL + rel_kept * WEIGHT_RELEVANT_RESIDUAL,
    });
    viol[PRIORITY] += u64::from(p_len < plan.fixed_bytes);

    let rel_slots: Vec<u8> = (0..RESIDUAL_SLOTS as u8)
        .filter(|&s| c.relevant[usize::from(s)])
        .collect();
    let ok = plan.kept_symbols.len() == SYMBOLS
        && plan.kept_residual_slots == rel_slots
        && plan.planned_bytes >= plan.fixed_bytes;
    viol[RELEVANCE] += u64::from(!ok);
    let (sf, sb) = frame_cost(seq, &vec![0u8; plan.planned_bytes])?;
    let kept_rel = plan.kept_residual_slots.len() as u64;
    out[RELEVANCE].push(Payload {
        payload_bytes: plan.planned_bytes as u64,
        frame_bytes: sb,
        frames: sf,
        weight: SYMBOLS as u64 * WEIGHT_SYMBOL + kept_rel * WEIGHT_RELEVANT_RESIDUAL,
    });
    Ok(())
}

/// Materialize one `(cell, seed)` workload. `seq_base` is the sequence the
/// first message is assigned at generation (tests use it to force wraparound).
pub fn materialize(cell: &Cell, seed: u64, seq_base: u16) -> Result<Materialized, String> {
    let s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ cell.id;
    let mut rng = ChaCha8Rng::seed_from_u64(s);
    let relevant = relevant_set(&mut rng, cell.relevance_ppm);
    let (lo, hi) = (RUN_MS / 10, RUN_MS / 2);
    let outage_start_ms = lo + rng.gen_range(0..(hi - lo));
    let mask = bind_mask(&relevant)?;
    let ctx = Ctx {
        relevant,
        mask: &mask,
        cap: (cell.budget_bytes / 4) as usize,
    };
    let n_rel = relevant.iter().filter(|&&r| r).count() as u64;
    let mut m = Materialized {
        w_max: max_weight(SYMBOLS as u64, n_rel),
        outage_start_ms,
        msgs: Vec::with_capacity(MSG_COUNT),
        payloads: [Vec::new(), Vec::new(), Vec::new()],
        mandatory_violations: [0; 3],
    };
    for i in 0..MSG_COUNT {
        let mut importance = [0u8; RESIDUAL_SLOTS];
        rng.fill_bytes(&mut importance);
        let delta = draw_delta(&mut rng, i as u32, &importance);
        let reorder_draw = rng.gen_range(0..1_000_000u32);
        let reorder_mag = 1 + u64::from(rng.next_u32() % 3);
        let mut loss_draws = [0u32; LOSS_DRAWS];
        for d in loss_draws.iter_mut() {
            *d = rng.gen_range(0..1_000_000u32);
        }
        m.msgs.push(MsgRand {
            gen_ms: i as u64 * GEN_PERIOD_MS,
            reorder_draw,
            reorder_mag,
            loss_draws,
        });
        let seq = seq_base.wrapping_add(i as u16);
        let env = SemanticEnvelope::wrap_delta(&delta, 0, i as u64, Some([0u8; 64]))
            .map_err(|e| format!("{e:?}"))?;
        build_payloads(
            &ctx,
            seq,
            &env,
            &importance,
            &mut m.payloads,
            &mut m.mandatory_violations,
        )?;
    }
    Ok(m)
}
