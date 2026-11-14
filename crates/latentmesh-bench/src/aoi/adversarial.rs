//! Adversarial tests for the AoI bench (issue #22): fairness of the three
//! baselines, counter detectability via hand-built workloads, determinism.

use latentmesh_air_core::{
    fragment_message, FragmentMeta, FrameFlags, Residual, SemanticClass, SemanticDelta,
    SemanticEnvelope, SymbolUpdate, SymbolValue, WireProfile,
};

use super::grid::Cell;
use super::receipt::params_for;
use super::sim::{audit_airtime, run_receiver, simulate, Delivery, Metrics, SimParams};
use super::workload::{
    frame_cost, materialize, priority_mandatory_violation, Materialized, MsgRand, Payload,
    PriorityKept, LOSS_DRAWS, PRIORITY, RELEVANCE, RESIDUAL_ITEM_BYTES, RESIDUAL_SLOTS,
    RESIDUAL_VALUES, SYMBOLS, WHOLE,
};

fn cell(budget: u64, rho: u64, loss: u64, age: u64, queue: u64) -> Cell {
    Cell {
        id: 9,
        age_ms: age,
        budget_bytes: budget,
        queue_depth: queue,
        loss_ppm: loss,
        reorder_ppm: 0,
        outage_ms: 0,
        relevance_ppm: rho,
    }
}

fn run(c: &Cell, seed: u64) -> [Metrics; 3] {
    let m = materialize(c, seed, 0).unwrap();
    let p = params_for(c, true, 0);
    [0, 1, 2].map(|b| simulate(&m, b, &p).metrics)
}

#[test]
fn utility_is_scored_on_one_relevant_set_for_all_baselines() {
    for rho in [250_000u64, 500_000, 1_000_000] {
        let m = materialize(&cell(1200, rho, 0, 0, 16), 11, 0).unwrap();
        for i in 0..m.msgs.len() {
            assert_eq!(
                m.payloads[WHOLE][i].weight, m.w_max,
                "whole delivers every relevant field"
            );
            assert_eq!(
                m.payloads[RELEVANCE][i].weight, m.w_max,
                "selector keeps the whole relevant set"
            );
            assert!(m.payloads[PRIORITY][i].weight <= m.w_max);
        }
    }
}

#[test]
fn relevance_bytes_equal_whole_minus_exactly_the_irrelevant_residuals() {
    // WHOLE comes from a real env.encode(); RELEVANCE from the planner.
    for (rho, n_rel) in [(250_000u64, 2u64), (500_000, 4), (1_000_000, 8)] {
        let m = materialize(&cell(2400, rho, 0, 0, 16), 5, 0).unwrap();
        let dropped = (RESIDUAL_SLOTS as u64 - n_rel) * RESIDUAL_ITEM_BYTES as u64;
        for i in 0..m.msgs.len() {
            assert_eq!(
                m.payloads[WHOLE][i].payload_bytes - m.payloads[RELEVANCE][i].payload_bytes,
                dropped
            );
        }
    }
}

#[test]
fn zero_buffer_airtime_equals_real_content_airtime() {
    // frame_cost fragments zeros; real encoders must cost the same bytes.
    let meta = FragmentMeta {
        profile: WireProfile::Meshtastic,
        flags: FrameFlags::default(),
        stream_id: 1,
        sequence: 0,
        class: SemanticClass::StateDelta,
        priority: 0,
        state_tag: 0,
    };
    for len in [0usize, 1, 100, 240, 241, 255, 256, 480, 481, 1000] {
        let real: Vec<u8> = (0..len).map(|i| (i * 31 + 7) as u8).collect();
        let f = fragment_message(meta, &real, 256).unwrap();
        let (n, bytes) = frame_cost(0, &vec![0u8; len]).unwrap();
        assert_eq!(n, f.len());
        assert_eq!(bytes, f.iter().map(|x| x.encoded_len() as u64).sum::<u64>());
    }
}

#[test]
fn priority_field_importance_is_uncorrelated_with_relevance_strawman_check() {
    // importance is random per message, independent of the relevant set, so
    // at the tightest cap priority_field keeps no relevant residual at all.
    let m = materialize(&cell(600, 250_000, 0, 0, 16), 11, 0).unwrap();
    assert!(
        m.payloads[PRIORITY].iter().all(|p| p.weight == 6),
        "cap 150 < fixed bytes: only symbols survive"
    );
    // And the selector is exempt from that same cap (Required bytes override it).
    assert!(m.payloads[RELEVANCE][0].payload_bytes > 600 / 4);
    assert!(
        m.payloads[PRIORITY][0].payload_bytes > 600 / 4,
        "fixed bytes alone already exceed the cap"
    );
}

#[test]
fn selector_advantage_exists_without_whole_payload_starvation() {
    // Roomy link, deep queue, no loss: W has zero queue/stale/unschedulable
    // drops. The gain must then come from shorter airtime (lower age) only.
    let [w, f, s] = run(&cell(2400, 250_000, 0, 0, 16), 11);
    assert_eq!(
        w.queue_overflow_drops + w.stale_drops + w.unschedulable_drops,
        0
    );
    assert_eq!(w.messages_delivered, 512);
    assert!(
        s.utility_micro > w.utility_micro,
        "S {} W {}",
        s.utility_micro,
        w.utility_micro
    );
    // Per-message weight is identical, so utility per delivery differs only by age.
    assert_eq!(s.messages_delivered, w.messages_delivered);
    assert!(f.utility_micro >= w.utility_micro);
}

#[test]
fn selector_never_loses_utility_to_whole_in_representative_cells() {
    for budget in [600, 1200, 2400] {
        for rho in [250_000, 500_000, 1_000_000] {
            for loss in [0, 300_000] {
                for seed in [11, 23] {
                    let [w, _f, s] = run(&cell(budget, rho, loss, 0, 4), seed);
                    assert!(
                        s.utility_micro >= w.utility_micro,
                        "budget {budget} rho {rho} loss {loss} seed {seed}: S {} < W {}",
                        s.utility_micro,
                        w.utility_micro
                    );
                }
            }
        }
    }
}

#[test]
fn age_2000_slice_is_degenerate_every_baseline_scores_zero() {
    // 1/3 of the grid (and of the primary set) cannot differentiate anything.
    let [w, f, s] = run(&cell(1200, 250_000, 0, 2000, 16), 11);
    for m in [&w, &f, &s] {
        assert_eq!(m.utility_micro, 0);
        assert_eq!(m.useful_fresh_deliveries, 0);
        assert_eq!(m.stale_deliveries, m.messages_delivered);
    }
}

#[test]
fn priority_field_beats_relevance_in_absolute_utility_when_saturated() {
    // Documents the guard-(a) caveat: it is evaluated on aggregates over the
    // primary set, so saturated cells where priority_field wins outright
    // (budget 600, queue 16: it sends twice as many small messages) are
    // averaged away. In the committed receipt 135/648 cells show S < F.
    for rho in [250_000u64, 500_000] {
        let [_w, f, s] = run(&cell(600, rho, 0, 0, 16), 11);
        assert!(
            f.utility_micro > s.utility_micro,
            "rho {rho}: F {} S {}",
            f.utility_micro,
            s.utility_micro
        );
        assert!(f.messages_delivered > s.messages_delivered);
        // Even per byte (the guard's own metric) F wins this cell.
        assert!(f.utility_micro * s.bytes_on_air > s.utility_micro * f.bytes_on_air);
    }
}

fn msg(gen_ms: u64) -> MsgRand {
    MsgRand {
        gen_ms,
        reorder_draw: u32::MAX,
        reorder_mag: 1,
        loss_draws: [u32::MAX; LOSS_DRAWS],
    }
}

fn hand_built(msgs: Vec<MsgRand>, frame_bytes: u64) -> Materialized {
    let pay = Payload {
        payload_bytes: frame_bytes,
        frame_bytes,
        frames: 1,
        weight: 1,
    };
    let n = msgs.len();
    Materialized {
        w_max: 1,
        outage_start_ms: 0,
        msgs,
        payloads: [vec![pay; n], vec![pay; n], vec![pay; n]],
        mandatory_violations: [0; 3],
    }
}

fn params(budget: u64, queue: u64, reorder_ppm: u64) -> SimParams {
    SimParams {
        age_ms: 0,
        budget_bytes: budget,
        queue_depth: queue,
        loss_ppm: 0,
        reorder_ppm,
        outage_ms: 0,
        stale_drop: false,
        seq_base: 0,
    }
}

#[test]
fn injected_mandatory_violation_reaches_the_metric() {
    let mut m = hand_built(vec![msg(0)], 100);
    m.mandatory_violations = [0, 0, 3];
    let out = simulate(&m, RELEVANCE, &params(1000, 4, 0));
    assert_eq!(out.metrics.mandatory_violations, 3);
    assert_eq!(
        simulate(&m, WHOLE, &params(1000, 4, 0))
            .metrics
            .mandatory_violations,
        0
    );
}

#[test]
fn injected_sequence_collision_is_counted_as_replay_inconsistency() {
    // 65_537 messages: idx 0 and idx 65_536 share u16 sequence 0. Delay every
    // message in between so the two collide at the receiver back to back.
    let n = 65_537usize;
    let mut msgs: Vec<MsgRand> = (0..n).map(|_| msg(0)).collect();
    for m in msgs.iter_mut().take(n - 1).skip(1) {
        m.reorder_draw = 0;
        m.reorder_mag = 1_000_000;
    }
    let m = hand_built(msgs, 1);
    let out = simulate(&m, WHOLE, &params(u64::MAX / 4, n as u64 + 1, 1_000_000));
    assert!(
        out.metrics.replay_inconsistencies >= 1,
        "double commit went undetected"
    );
    assert!(!out.delivered.contains(&(n - 1)) || !out.delivered.contains(&0));
}

#[test]
fn queue_bound_violation_counter_fires_when_the_bound_is_zero() {
    let m = hand_built((0..8).map(|i| msg(i * 250)).collect(), 100);
    let out = simulate(&m, WHOLE, &params(1000, 0, 0));
    assert!(out.metrics.queue_bound_violations > 0);
    let ok = simulate(&m, WHOLE, &params(1000, 4, 0));
    assert_eq!(ok.metrics.queue_bound_violations, 0);
    assert!(ok.metrics.queue_occupancy_max <= 4);
}

#[test]
fn very_late_message_beyond_the_window_is_a_counted_loss_not_an_integrity_fault() {
    // Message 0 arrives after 99 later messages: displaced 99 >= 64, so the
    // window answers TooOld. That is a legitimate loss, reported in
    // replay_window_rejections, and it must not raise replay_inconsistencies.
    let mut msgs: Vec<MsgRand> = (0..100).map(|i| msg(i * 250)).collect();
    msgs[0].reorder_draw = 0;
    msgs[0].reorder_mag = 1_000_000;
    let m = hand_built(msgs, 10);
    let out = simulate(&m, WHOLE, &params(10_000, 16, 1_000_000));
    assert!(
        !out.delivered.contains(&0),
        "late message should be rejected as too old"
    );
    assert_eq!(out.metrics.replay_window_rejections, 1);
    assert_eq!(out.metrics.replay_inconsistencies, 0);
    assert_eq!(out.metrics.messages_delivered, 99);
}

#[test]
fn replay_window_rejection_path_counts_too_old_separately_with_injected_order() {
    // Delivery order injected directly: message 0 is delivered last, after 99
    // commits, so its displacement is 99. Path: TooOld, displaced >= 64.
    let m = hand_built((0..100).map(|i| msg(i * 250)).collect(), 10);
    let deliveries: Vec<Delivery> = (0..100)
        .map(|idx| Delivery {
            arrival: if idx == 0 { u64::MAX } else { idx as u64 },
            idx,
            age: 0,
        })
        .collect();
    let mut mt = Metrics::default();
    let out = run_receiver(
        &m,
        WHOLE,
        &params(10_000, 16, 0),
        deliveries,
        &[false; 100],
        &mut mt,
    );
    assert_eq!(out.metrics.replay_window_rejections, 1);
    assert_eq!(out.metrics.replay_inconsistencies, 0);
    assert!(!out.delivered.contains(&0));
}

#[test]
fn replay_inconsistency_path_counts_a_never_committed_duplicate_with_injected_order() {
    // idx 0 and idx 65_536 share u16 sequence 0. Injected order: 0 then 65_536.
    // 65_536 was never committed and is displaced 0 (inside the window), so the
    // Duplicate decision is an inconsistency, not a window loss.
    let n = 65_537usize;
    let m = hand_built((0..n).map(|_| msg(0)).collect(), 1);
    let deliveries = vec![
        Delivery {
            arrival: 0,
            idx: 0,
            age: 0,
        },
        Delivery {
            arrival: 1,
            idx: 65_536,
            age: 0,
        },
    ];
    let mut mt = Metrics::default();
    let out = run_receiver(
        &m,
        WHOLE,
        &params(u64::MAX / 4, n as u64, 0),
        deliveries,
        &vec![false; n],
        &mut mt,
    );
    assert_eq!(out.delivered, vec![0]);
    assert_eq!(out.metrics.replay_inconsistencies, 1);
    assert_eq!(out.metrics.replay_window_rejections, 0);
}

#[test]
fn airtime_audit_counter_fires_on_an_injected_over_budget_log() {
    // Clean: window 0 carries 1200 B (= budget), window 1 carries 600 B.
    let clean = [(0, 600), (500, 600), (1000, 600)];
    assert_eq!(audit_airtime(&clean, 1200), 0);
    // Injected over-budget: a third 600 B transmission starts inside window 0.
    let over = [(0, 600), (500, 600), (900, 600)];
    assert_eq!(audit_airtime(&over, 1200), 1);
    let mt = Metrics {
        airtime_window_violations: audit_airtime(&over, 1200),
        ..Metrics::default()
    };
    assert_eq!(mt.airtime_window_violations, 1);
}

/// A real signed delta (6 symbols, 8 residual slots of 32 values) and the
/// length of its full envelope encoding.
fn real_delta_and_len() -> (SemanticDelta, usize) {
    let updates = (0..SYMBOLS)
        .map(|k| SymbolUpdate::Set {
            field_id: k as u16,
            value: SymbolValue::U64(k as u64),
        })
        .collect();
    let residuals = (0..RESIDUAL_SLOTS)
        .map(|slot| Residual {
            slot: slot as u8,
            importance: slot as u8,
            scale_exp: 0,
            values: vec![0i8; RESIDUAL_VALUES],
        })
        .collect();
    let delta = SemanticDelta {
        source_id: 1,
        epoch: 1,
        message_id: 0,
        base_hash: [0; 16],
        result_hash: [0; 16],
        updates,
        residuals,
    };
    let env = SemanticEnvelope::wrap_delta(&delta, 0, 0, Some([0u8; 64])).unwrap();
    let len = env.encode().unwrap().len();
    (delta, len)
}

#[test]
fn bad_priority_plan_increments_mandatory_violations() {
    let (delta, len) = real_delta_and_len();
    let fixed = len - RESIDUAL_SLOTS * RESIDUAL_ITEM_BYTES;
    let good = PriorityKept {
        fixed_bytes: fixed,
        symbols: SYMBOLS,
        slots: vec![0, 1, 2],
    };
    let mut viol = [0u64; 3];
    viol[PRIORITY] += priority_mandatory_violation(&good, len, &delta);
    assert_eq!(viol[PRIORITY], 0, "a correct priority plan is clean");

    let bad_plans = [
        (
            "symbol update dropped",
            PriorityKept {
                symbols: SYMBOLS - 1,
                ..good_clone(fixed)
            },
        ),
        (
            "fixed component cut short",
            PriorityKept {
                fixed_bytes: fixed - 1,
                ..good_clone(fixed)
            },
        ),
        (
            "fixed component padded",
            PriorityKept {
                fixed_bytes: fixed + 4,
                ..good_clone(fixed)
            },
        ),
        (
            "residual slot kept twice",
            PriorityKept {
                slots: vec![3, 3],
                ..good_clone(fixed)
            },
        ),
        (
            "residual slot out of range",
            PriorityKept {
                slots: vec![RESIDUAL_SLOTS],
                ..good_clone(fixed)
            },
        ),
    ];
    for (name, bad) in &bad_plans {
        let before = viol[PRIORITY];
        viol[PRIORITY] += priority_mandatory_violation(bad, len, &delta);
        assert_eq!(
            viol[PRIORITY] - before,
            1,
            "{name} must count as a mandatory violation"
        );
    }
}

fn good_clone(fixed: usize) -> PriorityKept {
    PriorityKept {
        fixed_bytes: fixed,
        symbols: SYMBOLS,
        slots: vec![0, 1, 2],
    }
}

#[test]
fn airtime_window_accounting_is_aligned_not_rolling() {
    // Budget 1200 B per aligned 1000 ms window, link 2400 B/s. Four 600 B
    // messages generated at t=500 finish inside one rolling second
    // [500, 1500] (2400 B = 2x budget) yet the counter reports 0 violations.
    let m = hand_built((0..4).map(|_| msg(500)).collect(), 600);
    let out = simulate(&m, WHOLE, &params(1200, 16, 0));
    assert_eq!(out.metrics.bytes_on_air, 2400);
    assert_eq!(out.metrics.airtime_window_violations, 0);
    assert_eq!(
        out.metrics.evidence_age_p95_ms, 1000,
        "last message ends at t=1500"
    );
    assert_eq!(out.metrics.evidence_age_p50_ms, 500);
}

#[test]
fn rerun_of_a_cell_is_bit_identical_across_threads() {
    let c = cell(1200, 500_000, 100_000, 500, 4);
    let a = run(&c, 37);
    let b = std::thread::spawn(move || run(&c, 37)).join().unwrap();
    assert_eq!(a, b);
}
