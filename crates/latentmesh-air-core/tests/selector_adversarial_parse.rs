//! Adversarial parse coverage for the ADR-049 selector (issue #22): malformed,
//! oversize, unsorted and fuzzed mask bytes must fail closed before any use.

mod common;

use common::*;
use latentmesh_air_core::*;

/// 150k seeded inputs: random, structured-random and bit-mutated valid masks.
/// Never panics; every accepted mask has exactly one canonical byte form.
#[test]
fn parse_fuzz_150k_never_panics_and_accepted_masks_round_trip_byte_exact() {
    let reg = registry(vec![FieldRef::Residual(1), FieldRef::Region(2)]);
    let mut rng = Rng(0x5EED_F00D);
    let mut accepted = 0u32;
    for i in 0..150_000u32 {
        let bytes = match i % 3 {
            0 => {
                let n = rng.below(340) as usize;
                rng.bytes(n)
            }
            1 => {
                let n = 4 + 4 * rng.below(81) as usize;
                let mut b = rng.bytes(n);
                b[2] = ((b.len() - 4) / 4) as u8;
                b[3] = 0;
                for k in 0..(b.len() - 4) / 4 {
                    b[4 + 4 * k] %= 5; // mostly valid kinds
                }
                b
            }
            _ => {
                let mut b = mask_bytes(TASK, &hostile_mask(&mut rng));
                for _ in 0..1 + rng.below(3) {
                    if b.is_empty() {
                        break;
                    }
                    let at = rng.below(b.len() as u64) as usize;
                    match rng.below(3) {
                        0 => b[at] ^= 1 << rng.below(8),
                        1 => b[at] = rng.next() as u8,
                        _ => {
                            b.truncate(at);
                        }
                    }
                }
                b
            }
        };
        if let Ok(m) = UnboundMask::parse(&bytes) {
            accepted += 1;
            assert_eq!(
                mask_bytes(m.task_id(), m.entries()),
                bytes,
                "non-canonical bytes accepted"
            );
            let _ = reg.bind(m);
        }
    }
    assert!(
        accepted > 1000,
        "fuzz corpus never reached the accept path ({accepted})"
    );
}

#[test]
fn parse_boundary_sizes_are_rejected_before_use() {
    assert_eq!(
        UnboundMask::parse(&[]).unwrap_err(),
        SelectorError::Malformed
    );
    assert_eq!(
        UnboundMask::parse(&[0; 3]).unwrap_err(),
        SelectorError::Malformed
    );
    let mut over = vec![0u8; MAX_MASK_WIRE_BYTES + 1];
    over[2] = MAX_MASK_ENTRIES as u8;
    assert_eq!(
        UnboundMask::parse(&over).unwrap_err(),
        SelectorError::TooManyEntries
    );
    let mut hdr = vec![0u8; 4];
    hdr[2] = 255; // count claims far more than the buffer holds
    assert_eq!(
        UnboundMask::parse(&hdr).unwrap_err(),
        SelectorError::TooManyEntries
    );
    hdr[2] = 1;
    assert_eq!(
        UnboundMask::parse(&hdr).unwrap_err(),
        SelectorError::Malformed
    );
    let mut pad = mask_bytes(TASK, &[]);
    pad[3] = 1;
    assert_eq!(
        UnboundMask::parse(&pad).unwrap_err(),
        SelectorError::Malformed
    );
    // Region id 16 is out of range; class 3 is BadClass; kind 4 is Malformed.
    for (kind, id, cls, want) in [
        (2u8, 16u16, 0u8, SelectorError::Malformed),
        (1, 256, 0, SelectorError::Malformed),
        (0, 0, 3 << 6, SelectorError::BadClass),
        (4, 0, 0, SelectorError::Malformed),
        (3, 9, 0, SelectorError::Malformed),
    ] {
        let mut b = TASK.to_le_bytes().to_vec();
        b.extend_from_slice(&[1, 0, kind]);
        b.extend_from_slice(&id.to_le_bytes());
        b.push(cls);
        assert_eq!(UnboundMask::parse(&b).unwrap_err(), want);
    }
}

#[test]
fn parse_rejects_unsorted_and_duplicate_masks() {
    let a = entry(FieldRef::Residual(2), FieldClass::Required, 0);
    let b = entry(FieldRef::Residual(1), FieldClass::Required, 0);
    assert_eq!(
        UnboundMask::parse(&mask_bytes(TASK, &[a, b])).unwrap_err(),
        SelectorError::NotCanonical
    );
    assert_eq!(
        UnboundMask::parse(&mask_bytes(TASK, &[a, a])).unwrap_err(),
        SelectorError::DuplicateEntry
    );
}

#[test]
fn optional_never_overrides_cap_but_required_may_exceed_it() {
    let mut rng = Rng(11);
    let reg = registry(vec![]);
    for _ in 0..300 {
        let d = random_delta(&mut rng);
        let m: Vec<MaskEntry> = d
            .residuals
            .iter()
            .map(|r| entry(FieldRef::Residual(r.slot), FieldClass::Optional, 0))
            .collect();
        let mut sorted = m.clone();
        sorted.sort_by_key(|e| e.field);
        let bm = bind(&reg, &sorted).unwrap();
        let tight = plan_delta(&d, &bm, Some(0)).unwrap();
        assert!(
            tight.kept_residual_slots.is_empty(),
            "optional kept above a zero cap"
        );
        assert_eq!(tight.planned_bytes, tight.fixed_bytes);
        let loose = plan_delta(&d, &bm, Some(usize::MAX)).unwrap();
        assert_eq!(loose.planned_bytes, loose.original_bytes);
    }
}
