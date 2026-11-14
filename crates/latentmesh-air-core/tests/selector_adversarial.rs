//! Adversarial coverage for the ADR-049 selector (issue #22). These tests try
//! to break the invariants rather than confirm the happy path.

mod common;

use common::*;
use latentmesh_air_core::*;

const RESERVED: [ReservedField; 9] = [
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
const NON_REQUIRED: [FieldClass; 2] = [FieldClass::Optional, FieldClass::Discardable];

fn adversarial_cap(rng: &mut Rng) -> Option<usize> {
    match rng.below(6) {
        0 => None,
        1 => Some(0),
        2 => Some(1),
        3 => Some(usize::MAX),
        4 => Some(usize::MAX - 1),
        _ => Some(rng.below(900) as usize),
    }
}

/// Canonical (ascending) mask mixing Reserved, Symbol and Residual entries
/// with random classes and ranks, including the adversarial rank 63.
fn hostile_mask(rng: &mut Rng) -> Vec<MaskEntry> {
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

fn random_class(rng: &mut Rng) -> FieldClass {
    match rng.below(3) {
        0 => FieldClass::Required,
        1 => FieldClass::Optional,
        _ => FieldClass::Discardable,
    }
}

fn rng_rank(rng: &mut Rng) -> u8 {
    [0u8, 63, rng.below(64) as u8][rng.below(3) as usize]
}

fn slot_size(d: &SemanticDelta, slot: u8) -> usize {
    4 + d
        .residuals
        .iter()
        .find(|r| r.slot == slot)
        .unwrap()
        .values
        .len()
}

#[test]
fn mandatory_and_required_are_never_dropped_under_hostile_masks_and_caps() {
    let mut rng = Rng(0xDEAD_BEEF);
    let (mut bound_ok, mut rejected) = (0u32, 0u32);
    for _ in 0..3000 {
        let d = random_delta(&mut rng);
        let mandatory: Vec<FieldRef> = (0..16u8)
            .filter(|_| rng.below(4) == 0)
            .map(FieldRef::Residual)
            .collect();
        let reg = registry(mandatory.clone());
        let mask = hostile_mask(&mut rng);
        let cap = adversarial_cap(&mut rng);
        let any_downgraded_protected = mask.iter().any(|e| {
            e.class != FieldClass::Required
                && (matches!(e.field, FieldRef::Reserved(_) | FieldRef::Symbol(_))
                    || mandatory.contains(&e.field))
        });
        match bind(&reg, &mask) {
            Err(_) => {
                rejected += 1;
                assert!(any_downgraded_protected, "bind rejected a legal mask");
            }
            Ok(bm) => {
                bound_ok += 1;
                assert!(
                    !any_downgraded_protected,
                    "bind accepted an illegal downgrade"
                );
                let plan = plan_delta(&d, &bm, cap).unwrap();
                for m in &mandatory {
                    if let FieldRef::Residual(s) = m {
                        if d.residuals.iter().any(|r| r.slot == *s) {
                            assert!(
                                plan.kept_residual_slots.contains(s),
                                "mandatory {s} dropped"
                            );
                        }
                    }
                }
                for r in &d.residuals {
                    let class = mask
                        .iter()
                        .find(|e| e.field == FieldRef::Residual(r.slot))
                        .map_or(FieldClass::Required, |e| e.class);
                    if class == FieldClass::Required {
                        assert!(plan.kept_residual_slots.contains(&r.slot));
                    }
                    if class == FieldClass::Discardable {
                        assert!(!plan.kept_residual_slots.contains(&r.slot));
                    }
                }
                assert_eq!(plan.kept_symbols.len(), d.updates.len());
                let kept: usize = plan
                    .kept_residual_slots
                    .iter()
                    .map(|s| slot_size(&d, *s))
                    .sum();
                assert_eq!(plan.planned_bytes, plan.fixed_bytes + kept);
            }
        }
    }
    assert!(
        bound_ok > 200 && rejected > 200,
        "generator is degenerate: {bound_ok}/{rejected}"
    );
}

#[test]
fn every_reserved_field_rejects_every_downgrade_at_bind() {
    let reg = registry(vec![]);
    for r in RESERVED {
        for class in NON_REQUIRED {
            for rank in [0u8, 63] {
                let m = [entry(FieldRef::Reserved(r), class, rank)];
                assert_eq!(
                    bind(&reg, &m).unwrap_err(),
                    SelectorError::ReservedDowngrade
                );
            }
        }
        // Required on a reserved field is a legal no-op.
        assert!(bind(
            &reg,
            &[entry(FieldRef::Reserved(r), FieldClass::Required, 0)]
        )
        .is_ok());
    }
}

#[test]
fn symbol_downgrade_is_rejected_even_for_ids_absent_from_the_delta() {
    let reg = registry(vec![]);
    for class in NON_REQUIRED {
        for id in [0u16, 5, 63, 64, 1000, u16::MAX] {
            let e = [entry(FieldRef::Symbol(id), class, 0)];
            assert_eq!(
                bind(&reg, &e).unwrap_err(),
                SelectorError::SymbolNotSelectable
            );
        }
    }
}

#[test]
fn mandatory_region_and_residual_downgrade_rejected_but_required_accepted() {
    let reg = registry(vec![FieldRef::Residual(3), FieldRef::Region(2)]);
    for class in NON_REQUIRED {
        for f in [FieldRef::Residual(3), FieldRef::Region(2)] {
            assert_eq!(
                bind(&reg, &[entry(f, class, 9)]).unwrap_err(),
                SelectorError::MandatoryDowngrade
            );
        }
        // A non-mandatory neighbour is still selectable.
        assert!(bind(&reg, &[entry(FieldRef::Residual(4), class, 9)]).is_ok());
    }
    assert!(bind(
        &reg,
        &[entry(FieldRef::Residual(3), FieldClass::Required, 0)]
    )
    .is_ok());
}

#[test]
fn mandatory_unlisted_in_mask_defaults_to_required_and_survives_zero_cap() {
    let mut rng = Rng(5);
    let reg = registry(vec![FieldRef::Residual(0)]);
    for _ in 0..200 {
        let d = random_delta(&mut rng);
        // Mask touches only slots other than 0, discarding them all.
        let m: Vec<MaskEntry> = (1..16u8)
            .map(|s| entry(FieldRef::Residual(s), FieldClass::Discardable, 0))
            .collect();
        let plan = plan_delta(&d, &bind(&reg, &m).unwrap(), Some(0)).unwrap();
        assert_eq!(
            plan.kept_residual_slots.contains(&0),
            d.residuals.iter().any(|r| r.slot == 0)
        );
        assert!(plan.kept_residual_slots.iter().all(|s| *s == 0));
    }
}

#[test]
fn plan_bytes_equal_a_real_reduced_encoding_for_delta_and_signed_envelope() {
    // Guards against the bench undercounting airtime: the plan must equal the
    // length of an actually-encoded reduced message, not just arithmetic.
    let mut rng = Rng(0xAB1E);
    let reg = registry(vec![]);
    for _ in 0..500 {
        let d = random_delta(&mut rng);
        let mask = random_residual_mask(&mut rng);
        let cap = adversarial_cap(&mut rng);
        let bm = bind(&reg, &mask).unwrap();
        let plan = plan_delta(&d, &bm, cap).unwrap();
        let mut reduced = d.clone();
        reduced
            .residuals
            .retain(|r| plan.kept_residual_slots.contains(&r.slot));
        assert_eq!(reduced.encode().unwrap().len(), plan.planned_bytes);

        for signed in [false, true] {
            let env = wrap(&d, signed);
            let eplan = plan_envelope(&env, &bm, cap).unwrap();
            // The cap counts envelope overhead, so keep-sets can differ.
            let mut r2 = d.clone();
            r2.residuals
                .retain(|r| eplan.kept_residual_slots.contains(&r.slot));
            let renv = wrap(&r2, signed);
            assert_eq!(renv.encode().unwrap().len(), eplan.planned_bytes);
            assert_eq!(eplan.original_bytes, env.encode().unwrap().len());
        }
    }
}

#[test]
fn reduced_delta_still_applies_and_verifies_hashes() {
    let mut before = CriticalState::new();
    let mut after = CriticalState::new();
    for id in 0..6u16 {
        before.set(id, SymbolValue::U64(u64::from(id))).unwrap();
        after
            .set(id, SymbolValue::U64(u64::from(id) + 100))
            .unwrap();
    }
    let residuals: Vec<Residual> = (0..8u8)
        .map(|slot| Residual {
            slot,
            importance: slot,
            scale_exp: 0,
            values: vec![slot as i8; 32],
        })
        .collect();
    let d = SemanticDelta::between(1, 1, 1, &before, &after, residuals).unwrap();
    let m: Vec<MaskEntry> = (4..8u8)
        .map(|s| entry(FieldRef::Residual(s), FieldClass::Discardable, 0))
        .collect();
    let plan = plan_delta(&d, &bind(&registry(vec![]), &m).unwrap(), None).unwrap();
    assert_eq!(plan.dropped_residual_slots, vec![4, 5, 6, 7]);
    let mut reduced = d.clone();
    reduced
        .residuals
        .retain(|r| plan.kept_residual_slots.contains(&r.slot));
    assert_eq!(
        reduced.apply(&before).unwrap().critical_hash(),
        after.critical_hash()
    );
}

#[test]
fn opaque_regions_respect_mandatory_and_account_tail_bytes() {
    let mut rng = Rng(0x0BA9);
    let reg = registry(vec![FieldRef::Region(0)]);
    for len in [0usize, 1, 15, 16, 17, 100, 255, 256, 257, 300, 1000] {
        let env = opaque(&mut rng, len, len % 2 == 0);
        let regions = len.div_ceil(REGION_BYTES).min(MAX_REGIONS);
        let mut m: Vec<MaskEntry> = (1..regions as u8)
            .map(|r| entry(FieldRef::Region(r), FieldClass::Discardable, 0))
            .collect();
        m.sort_by_key(|e| e.field);
        let plan = plan_envelope(&env, &bind(&reg, &m).unwrap(), Some(0)).unwrap();
        assert_eq!(plan.original_bytes, env.encoded_len(), "len {len}");
        if regions > 0 {
            assert_eq!(
                plan.kept_regions,
                vec![0],
                "mandatory region 0 must stay, len {len}"
            );
        }
        let dropped: usize = plan
            .dropped_regions
            .iter()
            .map(|r| REGION_BYTES.min(len - usize::from(*r) * REGION_BYTES))
            .sum();
        assert_eq!(
            plan.planned_bytes + dropped,
            plan.original_bytes,
            "len {len}"
        );
    }
}

#[test]
fn region_masks_on_wrong_body_kinds_fail_closed() {
    let mut rng = Rng(3);
    let reg = registry(vec![]);
    let d = random_delta(&mut rng);
    let region = bind(
        &reg,
        &[entry(FieldRef::Region(0), FieldClass::Discardable, 0)],
    )
    .unwrap();
    assert_eq!(
        plan_delta(&d, &region, None).unwrap_err(),
        SelectorError::Malformed
    );
    assert_eq!(
        plan_envelope(&wrap(&d, true), &region, None).unwrap_err(),
        SelectorError::Malformed
    );
    let resid = bind(
        &reg,
        &[entry(FieldRef::Residual(0), FieldClass::Discardable, 0)],
    )
    .unwrap();
    assert_eq!(
        plan_envelope(&opaque(&mut rng, 64, true), &resid, None).unwrap_err(),
        SelectorError::Malformed
    );
    // Region index beyond what a short body has.
    let far = bind(
        &reg,
        &[entry(FieldRef::Region(9), FieldClass::Discardable, 0)],
    )
    .unwrap();
    assert_eq!(
        plan_envelope(&opaque(&mut rng, 20, false), &far, None).unwrap_err(),
        SelectorError::Malformed
    );
}

#[test]
fn registry_rejects_duplicates_overflow_and_unsorted_mandatory() {
    let def = |id: u16, mandatory: Vec<FieldRef>| TaskDefinition {
        task_id: id,
        version: 1,
        mandatory,
    };
    assert_eq!(
        TaskRegistry::new(vec![def(1, vec![]), def(1, vec![])]).unwrap_err(),
        SelectorError::DuplicateEntry
    );
    let many: Vec<TaskDefinition> = (0..=MAX_TASKS as u16).map(|i| def(i, vec![])).collect();
    assert_eq!(
        TaskRegistry::new(many).unwrap_err(),
        SelectorError::TooManyEntries
    );
    let unsorted = vec![FieldRef::Residual(2), FieldRef::Residual(1)];
    assert_eq!(
        TaskRegistry::new(vec![def(1, unsorted)]).unwrap_err(),
        SelectorError::NotCanonical
    );
    let dup = vec![FieldRef::Residual(1), FieldRef::Residual(1)];
    assert_eq!(
        TaskRegistry::new(vec![def(1, dup)]).unwrap_err(),
        SelectorError::DuplicateEntry
    );
    let big: Vec<FieldRef> = (0..=MAX_MASK_ENTRIES as u16)
        .map(FieldRef::Symbol)
        .collect();
    assert_eq!(
        TaskRegistry::new(vec![def(1, big)]).unwrap_err(),
        SelectorError::TooManyEntries
    );
}

#[test]
fn unknown_task_is_rejected() {
    let reg = registry(vec![]);
    let raw = mask_bytes(TASK + 1, &[]);
    assert_eq!(
        reg.bind(UnboundMask::parse(&raw).unwrap()).unwrap_err(),
        SelectorError::UnknownTask
    );
}

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
