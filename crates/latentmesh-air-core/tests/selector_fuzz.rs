mod common;

use common::*;
use latentmesh_air_core::*;

fn valid_mask(rng: &mut Rng) -> Vec<u8> {
    let entries = random_residual_mask(rng);
    mask_bytes(TASK, &entries)
}

/// Parse then bind; never panics, always returns a typed result.
fn exercise(reg: &TaskRegistry, bytes: &[u8]) -> std::result::Result<BoundMask, SelectorError> {
    reg.bind(UnboundMask::parse(bytes)?)
}

#[test]
fn random_bytes_never_panic() {
    let reg = registry(vec![FieldRef::Residual(1)]);
    let mut rng = Rng(0xF00D);
    for _ in 0..20_000 {
        let len = rng.below(400) as usize;
        let mut bytes = rng.bytes(len);
        if bytes.len() >= 4 && rng.below(2) == 0 {
            bytes[0..2].copy_from_slice(&TASK.to_le_bytes());
            bytes[2] = rng.below(90) as u8;
            bytes[3] = 0;
        }
        let _ = exercise(&reg, &bytes);
    }
}

#[test]
fn mutated_valid_masks_never_panic_and_survivors_are_canonical() {
    let reg = registry(vec![FieldRef::Residual(1), FieldRef::Region(2)]);
    let mut rng = Rng(0xBEEF);
    for _ in 0..20_000 {
        let mut bytes = valid_mask(&mut rng);
        for _ in 0..rng.below(4) {
            match rng.below(4) {
                0 if !bytes.is_empty() => {
                    let i = rng.below(bytes.len() as u64) as usize;
                    bytes[i] ^= 1 << rng.below(8);
                }
                1 if !bytes.is_empty() => {
                    bytes.pop();
                }
                2 => bytes.push(rng.next() as u8),
                _ if !bytes.is_empty() => {
                    let i = rng.below(bytes.len() as u64) as usize;
                    bytes[i] = rng.next() as u8;
                }
                _ => {}
            }
        }
        if let Ok(m) = UnboundMask::parse(&bytes) {
            let entries = m.entries();
            assert!(entries.len() <= MAX_MASK_ENTRIES);
            assert!(entries.windows(2).all(|w| w[0].field < w[1].field));
            assert!(entries.iter().all(|e| e.rank <= 63));
        }
        let _ = exercise(&reg, &bytes);
    }
}

#[test]
fn rejects_each_structural_defect_with_its_error() {
    let reg = registry(vec![]);
    let ok = [
        entry(FieldRef::Residual(1), FieldClass::Optional, 0),
        entry(FieldRef::Residual(2), FieldClass::Optional, 0),
    ];
    let bytes = mask_bytes(TASK, &ok);
    assert!(exercise(&reg, &bytes).is_ok());

    // Truncated header and truncated body.
    assert_eq!(exercise(&reg, &[]), Err(SelectorError::Malformed));
    assert_eq!(exercise(&reg, &bytes[..3]), Err(SelectorError::Malformed));
    // Count larger than the buffer.
    let mut b = bytes.clone();
    b[2] = 3;
    assert_eq!(exercise(&reg, &b), Err(SelectorError::Malformed));
    // Count smaller than the buffer (trailing bytes).
    let mut b = bytes.clone();
    b[2] = 1;
    assert_eq!(exercise(&reg, &b), Err(SelectorError::Malformed));
    // Nonzero reserved header byte.
    let mut b = bytes.clone();
    b[3] = 1;
    assert_eq!(exercise(&reg, &b), Err(SelectorError::Malformed));
    // Count above the entry limit, with and without a matching buffer.
    let mut b = bytes.clone();
    b[2] = 81;
    assert_eq!(exercise(&reg, &b), Err(SelectorError::TooManyEntries));
    // Oversize buffer.
    let big = vec![0_u8; MAX_MASK_WIRE_BYTES + 1];
    assert_eq!(exercise(&reg, &big), Err(SelectorError::TooManyEntries));
    // Unsorted and duplicate.
    let unsorted = [ok[1], ok[0]];
    assert_eq!(
        exercise(&reg, &mask_bytes(TASK, &unsorted)),
        Err(SelectorError::NotCanonical)
    );
    let dup = [ok[0], ok[0]];
    assert_eq!(
        exercise(&reg, &mask_bytes(TASK, &dup)),
        Err(SelectorError::DuplicateEntry)
    );
    // Cross-kind ordering is by (kind, id): Region before Residual is unsorted.
    let cross = [entry(FieldRef::Region(0), FieldClass::Required, 0), ok[0]];
    assert_eq!(
        exercise(&reg, &mask_bytes(TASK, &cross)),
        Err(SelectorError::NotCanonical)
    );
    // Bad class (3), bad kind, out-of-range ids.
    let mut b = bytes.clone();
    b[4 + 3] = 3 << 6;
    assert_eq!(exercise(&reg, &b), Err(SelectorError::BadClass));
    let mut b = bytes.clone();
    b[4] = 4;
    assert_eq!(exercise(&reg, &b), Err(SelectorError::Malformed));
    let mut b = bytes.clone();
    b[5..7].copy_from_slice(&256_u16.to_le_bytes());
    assert_eq!(exercise(&reg, &b), Err(SelectorError::Malformed));
    let mut b = mask_bytes(TASK, &[]);
    b[2] = 1;
    b.extend_from_slice(&[2, 16, 0, 0]);
    assert_eq!(exercise(&reg, &b), Err(SelectorError::Malformed));
    let mut b = mask_bytes(TASK, &[]);
    b[2] = 1;
    b.extend_from_slice(&[3, 9, 0, 0]);
    assert_eq!(exercise(&reg, &b), Err(SelectorError::Malformed));
    // Unknown task.
    let b = mask_bytes(TASK + 1, &ok);
    assert_eq!(exercise(&reg, &b), Err(SelectorError::UnknownTask));
}

#[test]
fn max_size_mask_is_accepted_and_one_more_is_not() {
    let reg = registry(vec![]);
    let mut entries = Vec::new();
    for i in 0..MAX_MASK_ENTRIES as u16 {
        entries.push(entry(FieldRef::Symbol(i), FieldClass::Required, 0));
    }
    let bytes = mask_bytes(TASK, &entries);
    assert_eq!(bytes.len(), MAX_MASK_WIRE_BYTES);
    assert!(exercise(&reg, &bytes).is_ok());
    let mut over = bytes;
    over.extend_from_slice(&[0, 99, 0, 0]);
    over[2] = 81;
    assert_eq!(exercise(&reg, &over), Err(SelectorError::TooManyEntries));
}

#[test]
fn planning_with_fuzzed_bound_masks_never_panics() {
    let reg = registry(vec![FieldRef::Residual(0)]);
    let mut rng = Rng(0xCAFE);
    let mut planned = 0;
    for _ in 0..20_000 {
        let mut bytes = valid_mask(&mut rng);
        if !bytes.is_empty() && rng.below(2) == 0 {
            let i = rng.below(bytes.len() as u64) as usize;
            bytes[i] = rng.next() as u8;
        }
        let Ok(bound) = exercise(&reg, &bytes) else {
            continue;
        };
        let d = random_delta(&mut rng);
        let cap = (rng.below(2) == 0).then(|| rng.below(800) as usize);
        let _ = plan_delta(&d, &bound, cap);
        let n = rng.below(300) as usize;
        let env = opaque(&mut rng, n, false);
        let _ = plan_envelope(&env, &bound, cap);
        let _ = plan_envelope(&wrap(&d, true), &bound, cap);
        planned += 1;
    }
    assert!(planned > 100, "fuzz should reach the planner sometimes");
}

#[test]
fn malformed_envelope_body_is_an_error_not_a_panic() {
    let reg = registry(vec![]);
    let bound = bind(&reg, &[]).unwrap();
    let mut rng = Rng(3);
    for len in [0usize, 1, 51, 52, 200] {
        let mut env = opaque(&mut rng, len, false);
        env.class = SemanticClass::StateDelta;
        assert_eq!(
            plan_envelope(&env, &bound, None),
            Err(SelectorError::Malformed)
        );
    }
}
