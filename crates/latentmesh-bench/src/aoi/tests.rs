use latentmesh_air_core::{fragment_message, ReplayWindow};
use serde_json::Value;

use super::canonical::{canonical_json, sha256_hex};
use super::grid::{Axes, Cell};
use super::receipt::{params_for, GitInfo};
use super::sim::simulate;
use super::utility::decay_ppm;
use super::workload::{frame_cost, materialize, BASELINES, PRIORITY, RELEVANCE, WHOLE};
use super::{parse_args, receipt_text, Options};

const VECTOR: &str = include_str!("../../testdata/canonical_json_v1.json");

fn fake_git() -> GitInfo {
    GitInfo {
        commit: "0".repeat(40),
        dirty: false,
        dirty_snapshot_sha256: None,
    }
}

fn cell(age: u64, loss: u64, reorder: u64, outage: u64) -> Cell {
    Cell {
        id: 5,
        age_ms: age,
        budget_bytes: 1200,
        queue_depth: 16,
        loss_ppm: loss,
        reorder_ppm: reorder,
        outage_ms: outage,
        relevance_ppm: 500_000,
    }
}

#[test]
fn canonical_matches_shared_vector() {
    let v: Value = serde_json::from_str(VECTOR).unwrap();
    let canon = canonical_json(&v["input"]).unwrap();
    assert_eq!(canon, v["canonical"].as_str().unwrap());
    assert_eq!(sha256_hex(&canon), v["sha256"].as_str().unwrap());
}

#[test]
fn canonical_rejects_floats_and_non_ascii() {
    assert!(canonical_json(&serde_json::json!({"a": 1.5})).is_err());
    assert!(canonical_json(&serde_json::json!({"a": "caf\u{e9}"})).is_err());
}

#[test]
fn decay_boundaries() {
    assert_eq!(decay_ppm(0), 1_000_000);
    assert_eq!(decay_ppm(999), 500_500);
    assert_eq!(decay_ppm(1000), 500_000);
    assert_eq!(decay_ppm(2000), 250_000);
    assert_eq!(decay_ppm(2001), 0);
}

#[test]
fn replay_is_consistent_across_conditions_and_wraparound() {
    for age in [0, 2000] {
        for loss in [0, 300_000] {
            for reorder in [0, 200_000] {
                for outage in [0, 5000] {
                    for seq_base in [0u16, 65_200] {
                        check_replay(&cell(age, loss, reorder, outage), seq_base);
                    }
                }
            }
        }
    }
}

fn check_replay(c: &Cell, seq_base: u16) {
    let m = materialize(c, 11, seq_base).unwrap();
    for b in 0..BASELINES.len() {
        let on = simulate(&m, b, &params_for(c, true, seq_base));
        let off = simulate(&m, b, &params_for(c, false, seq_base));
        assert_eq!(on.metrics.replay_inconsistencies, 0);
        assert_eq!(off.metrics.replay_inconsistencies, 0);
        let mut w = ReplayWindow::new();
        for &i in off.delivered.iter().filter(|i| on.delivered.contains(i)) {
            w.commit(seq_base.wrapping_add(i as u16)).unwrap();
        }
        assert_eq!(
            w.highest(),
            on.highest,
            "{c:?} base {seq_base} baseline {b}"
        );
    }
}

#[test]
fn sequences_wrap_u16_in_a_run() {
    let c = cell(0, 0, 0, 0);
    let m = materialize(&c, 3, 65_300).unwrap();
    let out = simulate(&m, WHOLE, &params_for(&c, true, 65_300));
    assert!(out.delivered.iter().any(|&i| 65_300usize + i > 65_535));
    assert_eq!(out.metrics.replay_inconsistencies, 0);
}

#[test]
fn whole_payload_bytes_on_air_equals_fragment_frame_bytes() {
    // Lossless, roomy link: every message is transmitted exactly once.
    let mut c = cell(0, 0, 0, 0);
    c.budget_bytes = 2400;
    c.queue_depth = 16;
    let m = materialize(&c, 7, 0).unwrap();
    let expect: u64 = m.payloads[WHOLE]
        .iter()
        .enumerate()
        .map(|(i, p)| {
            frame_cost(i as u16, &vec![0; p.payload_bytes as usize])
                .unwrap()
                .1
        })
        .sum();
    let out = simulate(&m, WHOLE, &params_for(&c, true, 0));
    let sent = out.metrics.messages_generated
        - out.metrics.queue_overflow_drops
        - out.metrics.stale_drops
        - out.metrics.unschedulable_drops;
    assert_eq!(sent, 512, "roomy lossless link should send everything");
    assert_eq!(out.metrics.bytes_on_air, expect);
}

#[test]
fn whole_frame_bytes_match_real_envelope_fragmentation() {
    use latentmesh_air_core::{FragmentMeta, FrameFlags, SemanticClass, WireProfile};
    let c = cell(0, 0, 0, 0);
    let m = materialize(&c, 7, 0).unwrap();
    let pay = m.payloads[WHOLE][0];
    // 48 header + 64 signature + delta; independent frame arithmetic.
    let body = vec![0u8; pay.payload_bytes as usize];
    let meta = FragmentMeta {
        profile: WireProfile::Meshtastic,
        flags: FrameFlags::default(),
        stream_id: 1,
        sequence: 0,
        class: SemanticClass::StateDelta,
        priority: 0,
        state_tag: 0,
    };
    let frames = fragment_message(meta, &body, 256).unwrap();
    let sum: u64 = frames.iter().map(|f| f.encoded_len() as u64).sum();
    assert_eq!(sum, pay.frame_bytes);
    assert_eq!(frames.len(), pay.frames);
}

#[test]
fn relevance_never_exceeds_whole_and_keeps_mandatory() {
    let m = materialize(&cell(0, 0, 0, 0), 7, 0).unwrap();
    for i in 0..m.msgs.len() {
        assert!(m.payloads[RELEVANCE][i].payload_bytes <= m.payloads[WHOLE][i].payload_bytes);
        assert!(m.payloads[PRIORITY][i].payload_bytes <= m.payloads[WHOLE][i].payload_bytes);
    }
    assert_eq!(m.mandatory_violations, [0, 0, 0]);
}

fn quick_opts() -> Options {
    Options {
        seeds: vec![11, 23],
        out: None,
        quick: true,
    }
}

#[test]
fn quick_receipt_is_deterministic_and_self_consistent() {
    let a = receipt_text(&quick_opts(), &fake_git()).unwrap();
    let b = receipt_text(&quick_opts(), &fake_git()).unwrap();
    assert_eq!(a, b, "two runs must be byte-identical");
    let mut v: Value = serde_json::from_str(&a).unwrap();
    assert_eq!(v["frozen"], false);
    assert_eq!(
        v["cells"].as_array().unwrap().len(),
        Axes::quick().cells().len()
    );
    let src = serde_json::json!({"config": v["config"], "axes": v["axes"], "seeds": v["seeds"]});
    let frozen = sha256_hex(&canonical_json(&src).unwrap());
    assert_eq!(v["frozen_digest_sha256"], frozen.as_str());
    let claimed = v.as_object_mut().unwrap().remove("receipt_sha256").unwrap();
    assert_eq!(claimed, sha256_hex(&canonical_json(&v).unwrap()).as_str());
    for key in v["integrity"].as_object().unwrap().values() {
        assert_eq!(key, 0);
    }
}

#[test]
fn full_grid_has_648_cells() {
    assert_eq!(Axes::full().cells().len(), 648);
}

#[test]
fn seed_floor_enforced_outside_quick() {
    let args = |s: &[&str]| s.iter().map(|x| x.to_string()).collect::<Vec<_>>();
    assert!(parse_args(&args(&["--seeds", "1,2,3,4"])).is_err());
    assert!(parse_args(&args(&["--seeds", "1,2,3,4,5"])).is_ok());
    assert!(parse_args(&args(&["--quick", "--seeds", "1"])).is_ok());
    assert!(parse_args(&args(&["--bogus"])).is_err());
}
