//! Window-airtime link, bounded drop-oldest queue, outage, reorder, stale-drop,
//! and a receiver that commits every delivery through `ReplayWindow`.

use std::collections::{BTreeMap, VecDeque};

use latentmesh_air_core::{AirError, ReplayWindow};

use super::utility::{covers_useful, decay_ppm, STALE_AGE_MS};
use super::workload::{Materialized, GEN_PERIOD_MS};

pub const LINK_BYTES_PER_S: u64 = 2400;
const WINDOW_MS: u64 = 1000;

#[derive(Clone, Copy, Debug)]
pub struct SimParams {
    pub age_ms: u64,
    pub budget_bytes: u64,
    pub queue_depth: u64,
    pub loss_ppm: u64,
    pub reorder_ppm: u64,
    pub outage_ms: u64,
    pub stale_drop: bool,
    pub seq_base: u16,
}

/// One value per ADR-049 metric; all integers.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Metrics {
    pub messages_generated: u64,
    pub messages_delivered: u64,
    pub bytes_on_air: u64,
    pub payload_bytes_tx: u64,
    pub utility_micro: u64,
    pub utility_per_byte_micro: u64,
    pub stale_deliveries: u64,
    pub stale_delivery_rate_ppm: u64,
    pub useful_fresh_deliveries: u64,
    pub queue_occupancy_mean_milli: u64,
    pub queue_occupancy_max: u64,
    pub queue_overflow_drops: u64,
    pub stale_drops: u64,
    pub unschedulable_drops: u64,
    pub evidence_age_p50_ms: u64,
    pub evidence_age_p95_ms: u64,
    pub mandatory_violations: u64,
    pub replay_inconsistencies: u64,
    pub queue_bound_violations: u64,
    pub airtime_window_violations: u64,
    /// Legitimately generated messages rejected `TooOld` because they were
    /// displaced 64 or more positions behind the window head. A real loss,
    /// reported separately: it is not an integrity violation.
    pub replay_window_rejections: u64,
}

#[cfg_attr(not(test), allow(dead_code))]
pub struct SimOut {
    pub metrics: Metrics,
    /// Message indices committed by the receiver, in arrival order.
    pub delivered: Vec<usize>,
    pub highest: Option<u16>,
}

/// One frame reaching the receiver. `pub(super)` so the adversarial tests can
/// inject delivery orders directly.
pub(super) struct Delivery {
    pub(super) arrival: u64,
    pub(super) idx: usize,
    pub(super) age: u64,
}

struct Sender {
    q: VecDeque<usize>,
    next: usize,
    t: u64,
    window: u64,
    used: u64,
    occ_sum: u64,
    dropped: Vec<bool>,
    /// Every transmission as `(start_ms, frame_bytes)`, for the post-hoc audit.
    log: Vec<(u64, u64)>,
}

/// Independent post-hoc audit of the airtime budget. Re-sums the transmission
/// log per aligned window (a transmission counts against the window it starts
/// in) and counts transmissions whose window running total exceeds `budget`.
/// It shares no state with the scheduler, so a scheduler bug that overfills a
/// window shows up here instead of being masked by the scheduler's own check.
pub(super) fn audit_airtime(log: &[(u64, u64)], budget: u64) -> u64 {
    let mut totals: BTreeMap<u64, u64> = BTreeMap::new();
    let mut violations = 0;
    for &(start_ms, frame_bytes) in log {
        let total = totals.entry(start_ms / WINDOW_MS).or_default();
        *total += frame_bytes;
        violations += u64::from(*total > budget);
    }
    violations
}

fn nearest_rank(sorted: &[u64], pct: u64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let rank = (sorted.len() as u64 * pct).div_ceil(100).max(1);
    sorted[rank as usize - 1]
}

fn enqueue(s: &mut Sender, m: &Materialized, p: &SimParams, mt: &mut Metrics) {
    while s.next < m.msgs.len() && m.msgs[s.next].gen_ms <= s.t {
        if s.q.len() as u64 >= p.queue_depth {
            if let Some(old) = s.q.pop_front() {
                s.dropped[old] = true;
                mt.queue_overflow_drops += 1;
            }
        }
        s.q.push_back(s.next);
        s.next += 1;
        let len = s.q.len() as u64;
        s.occ_sum += len;
        mt.queue_occupancy_max = mt.queue_occupancy_max.max(len);
        mt.queue_bound_violations += u64::from(len > p.queue_depth);
    }
}

fn lost(m: &Materialized, p: &SimParams, idx: usize, frames: usize, start: u64, end: u64) -> bool {
    let out_end = m.outage_start_ms + p.outage_ms;
    if p.outage_ms > 0 && start < out_end && end > m.outage_start_ms {
        return true;
    }
    m.msgs[idx].loss_draws[..frames]
        .iter()
        .any(|&d| u64::from(d) < p.loss_ppm)
}

/// Sender side: returns deliveries still to be reordered/committed.
fn run_sender(
    m: &Materialized,
    b: usize,
    p: &SimParams,
    mt: &mut Metrics,
) -> (Vec<Delivery>, Vec<bool>) {
    let pays = &m.payloads[b];
    let mut s = Sender {
        q: VecDeque::new(),
        next: 0,
        t: 0,
        window: u64::MAX,
        used: 0,
        occ_sum: 0,
        dropped: vec![false; m.msgs.len()],
        log: Vec::new(),
    };
    let mut out = Vec::new();
    loop {
        enqueue(&mut s, m, p, mt);
        let Some(&idx) = s.q.front() else {
            match m.msgs.get(s.next) {
                Some(next) => {
                    s.t = next.gen_ms;
                    continue;
                }
                None => break,
            }
        };
        let gen = m.msgs[idx].gen_ms;
        if p.stale_drop && p.age_ms + (s.t - gen) > STALE_AGE_MS {
            s.q.pop_front();
            s.dropped[idx] = true;
            mt.stale_drops += 1;
            continue;
        }
        let pay = pays[idx];
        if pay.frame_bytes > p.budget_bytes {
            s.q.pop_front();
            s.dropped[idx] = true;
            mt.unschedulable_drops += 1;
            continue;
        }
        if s.t / WINDOW_MS != s.window {
            s.window = s.t / WINDOW_MS;
            s.used = 0;
        }
        if s.used + pay.frame_bytes > p.budget_bytes {
            s.t = (s.window + 1) * WINDOW_MS;
            continue;
        }
        s.q.pop_front();
        s.used += pay.frame_bytes;
        let start = s.t;
        s.log.push((start, pay.frame_bytes));
        s.t += (pay.frame_bytes * 1000).div_ceil(LINK_BYTES_PER_S);
        mt.bytes_on_air += pay.frame_bytes;
        mt.payload_bytes_tx += pay.payload_bytes;
        if !lost(m, p, idx, pay.frames, start, s.t) {
            let r = &m.msgs[idx];
            let extra = if u64::from(r.reorder_draw) < p.reorder_ppm {
                r.reorder_mag * GEN_PERIOD_MS
            } else {
                0
            };
            out.push(Delivery {
                arrival: s.t + extra,
                idx,
                age: p.age_ms + (s.t - gen) + extra,
            });
        }
    }
    mt.queue_occupancy_mean_milli = s.occ_sum * 1000 / m.msgs.len() as u64;
    mt.airtime_window_violations = audit_airtime(&s.log, p.budget_bytes);
    (out, s.dropped)
}

/// Receiver: every delivery, stale or not, goes through `commit`.
pub(super) fn run_receiver(
    m: &Materialized,
    b: usize,
    p: &SimParams,
    mut deliveries: Vec<Delivery>,
    dropped: &[bool],
    mt: &mut Metrics,
) -> SimOut {
    deliveries.sort_by_key(|d| (d.arrival, d.idx));
    let mut win = ReplayWindow::new();
    let mut committed = vec![false; m.msgs.len()];
    let (mut ages, mut delivered) = (Vec::new(), Vec::new());
    for d in &deliveries {
        let seq = p.seq_base.wrapping_add(d.idx as u16);
        match win.commit(seq) {
            Ok(()) => {
                mt.replay_inconsistencies += u64::from(committed[d.idx] || dropped[d.idx]);
                committed[d.idx] = true;
                delivered.push(d.idx);
                ages.push(d.age);
                score(m, b, d, mt);
            }
            Err(err) => {
                // Displacement behind the window head (0 for an exact repeat).
                let behind = win.highest().map_or(0, |h| h.wrapping_sub(seq));
                if matches!(err, AirError::TooOld) && behind >= 64 && !committed[d.idx] {
                    // A legitimately generated message displaced beyond the
                    // 64-wide window: a visible loss, not an integrity fault.
                    mt.replay_window_rejections += 1;
                } else {
                    // Any other rejection is an inconsistency: a `Duplicate`
                    // decision for a message never committed (a sequence
                    // collision inside the window), or a rejected re-delivery
                    // of a message already committed.
                    mt.replay_inconsistencies += 1;
                }
            }
        }
    }
    ages.sort_unstable();
    mt.evidence_age_p50_ms = nearest_rank(&ages, 50);
    mt.evidence_age_p95_ms = nearest_rank(&ages, 95);
    SimOut {
        metrics: std::mem::take(mt),
        delivered,
        highest: win.highest(),
    }
}

fn score(m: &Materialized, b: usize, d: &Delivery, mt: &mut Metrics) {
    let w = m.payloads[b][d.idx].weight;
    mt.messages_delivered += 1;
    mt.utility_micro += decay_ppm(d.age) * w;
    if d.age > STALE_AGE_MS {
        mt.stale_deliveries += 1;
    } else if covers_useful(w, m.w_max) {
        mt.useful_fresh_deliveries += 1;
    }
}

/// Run baseline `b` (index into `workload::BASELINES`) on a materialized cell.
pub fn simulate(m: &Materialized, b: usize, p: &SimParams) -> SimOut {
    let mut mt = Metrics {
        messages_generated: m.msgs.len() as u64,
        mandatory_violations: m.mandatory_violations[b],
        ..Metrics::default()
    };
    let (deliveries, dropped) = run_sender(m, b, p, &mut mt);
    let mut out = run_receiver(m, b, p, deliveries, &dropped, &mut mt);
    let x = &mut out.metrics;
    x.utility_per_byte_micro = x.utility_micro.checked_div(x.bytes_on_air).unwrap_or(0);
    x.stale_delivery_rate_ppm = (x.stale_deliveries * 1_000_000)
        .checked_div(x.messages_delivered)
        .unwrap_or(0);
    out
}
