mod common;

use common::*;
use latentmesh_air_core::*;

const ROUNDS: u64 = 400;

fn class_of(mask: &[MaskEntry], slot: u8) -> FieldClass {
    mask.iter()
        .find(|e| e.field == FieldRef::Residual(slot))
        .map_or(FieldClass::Required, |e| e.class)
}

fn residual_size(d: &SemanticDelta, slot: u8) -> usize {
    let r = d.residuals.iter().find(|r| r.slot == slot).unwrap();
    4 + r.values.len()
}

#[test]
fn plan_properties_over_random_deltas_and_masks() {
    let reg = registry(vec![]);
    let mut rng = Rng(0xA11CE);
    for _ in 0..ROUNDS {
        let d = random_delta(&mut rng);
        let mask = random_residual_mask(&mut rng);
        let cap = (rng.below(2) == 0).then(|| rng.below(700) as usize);
        let plan = plan_delta(&d, &bind(&reg, &mask).unwrap(), cap).unwrap();

        assert!(plan.planned_bytes <= plan.original_bytes);
        assert_eq!(plan.original_bytes, d.encode().unwrap().len());
        assert_eq!(plan.kept_symbols.len(), d.updates.len());
        let mut required_bytes = plan.fixed_bytes;
        for r in &d.residuals {
            let class = class_of(&mask, r.slot);
            let kept = plan.kept_residual_slots.contains(&r.slot);
            let dropped = plan.dropped_residual_slots.contains(&r.slot);
            assert!(kept ^ dropped, "every residual is kept xor dropped");
            match class {
                FieldClass::Required => {
                    assert!(kept);
                    required_bytes += residual_size(&d, r.slot);
                }
                FieldClass::Discardable => assert!(dropped),
                FieldClass::Optional => {}
            }
        }
        let kept_sum: usize = plan.fixed_bytes
            + plan
                .kept_residual_slots
                .iter()
                .map(|s| residual_size(&d, *s))
                .sum::<usize>();
        assert_eq!(kept_sum, plan.planned_bytes);
        if let Some(cap) = cap {
            // Optional bytes only fit under the cap; Required may exceed it.
            assert!(plan.planned_bytes <= cap.max(required_bytes));
        }
    }
}

#[test]
fn optional_kept_in_rank_order_prefix() {
    let reg = registry(vec![]);
    let mut rng = Rng(77);
    for _ in 0..ROUNDS {
        let d = random_delta(&mut rng);
        let mask: Vec<MaskEntry> = d
            .residuals
            .iter()
            .map(|r| {
                entry(
                    FieldRef::Residual(r.slot),
                    FieldClass::Optional,
                    rng.below(64) as u8,
                )
            })
            .collect();
        let mut sorted = mask.clone();
        sorted.sort_by_key(|e| e.field);
        let cap = Some(rng.below(900) as usize);
        let plan = plan_delta(&d, &bind(&reg, &sorted).unwrap(), cap).unwrap();
        let rank = |slot: u8| {
            let m = sorted.iter().find(|e| e.field == FieldRef::Residual(slot));
            (m.unwrap().rank, slot)
        };
        let max_kept = plan.kept_residual_slots.iter().map(|s| rank(*s)).max();
        let min_dropped = plan.dropped_residual_slots.iter().map(|s| rank(*s)).min();
        if let (Some(k), Some(dr)) = (max_kept, min_dropped) {
            assert!(k < dr, "a kept item must outrank every dropped item");
        }
        if let Some(cap) = cap {
            assert!(plan.planned_bytes <= cap.max(plan.fixed_bytes));
        }
    }
}

#[test]
fn discardable_never_kept_and_none_cap_keeps_all_optional() {
    let reg = registry(vec![]);
    let mut rng = Rng(5);
    for _ in 0..ROUNDS {
        let d = random_delta(&mut rng);
        let mask = random_residual_mask(&mut rng);
        let plan = plan_delta(&d, &bind(&reg, &mask).unwrap(), None).unwrap();
        for r in &d.residuals {
            let kept = plan.kept_residual_slots.contains(&r.slot);
            assert_eq!(kept, class_of(&mask, r.slot) != FieldClass::Discardable);
        }
    }
}

#[test]
fn all_required_reproduces_encoded_len_for_delta_and_envelope() {
    let reg = registry(vec![]);
    let mut rng = Rng(99);
    for i in 0..ROUNDS {
        let d = random_delta(&mut rng);
        let explicit: Vec<MaskEntry> = d
            .residuals
            .iter()
            .map(|r| entry(FieldRef::Residual(r.slot), FieldClass::Required, 0))
            .collect();
        let mut explicit = explicit;
        explicit.sort_by_key(|e| e.field);
        for mask in [vec![], explicit] {
            let bound = bind(&reg, &mask).unwrap();
            let plan = plan_delta(&d, &bound, Some(0)).unwrap();
            assert_eq!(plan.planned_bytes, plan.original_bytes);
            assert_eq!(plan.original_bytes, d.encode().unwrap().len());
            let env = wrap(&d, i % 2 == 0);
            let plan = plan_envelope(&env, &bound, Some(0)).unwrap();
            assert_eq!(plan.planned_bytes, env.encoded_len());
            assert_eq!(plan.original_bytes, env.encoded_len());
            assert_eq!(plan.original_bytes, env.encode().unwrap().len());
        }
    }
}

#[test]
fn envelope_plan_matches_delta_plan_plus_overhead() {
    let reg = registry(vec![]);
    let mut rng = Rng(1234);
    for i in 0..ROUNDS {
        let d = random_delta(&mut rng);
        let bound = bind(&reg, &random_residual_mask(&mut rng)).unwrap();
        let env = wrap(&d, i % 2 == 1);
        let overhead = env.encoded_len() - env.body.len();
        let a = plan_delta(&d, &bound, None).unwrap();
        let b = plan_envelope(&env, &bound, None).unwrap();
        assert_eq!(b.planned_bytes, a.planned_bytes + overhead);
        assert_eq!(b.fixed_bytes, a.fixed_bytes + overhead);
        assert_eq!(a.kept_residual_slots, b.kept_residual_slots);
    }
}

#[test]
fn opaque_body_regions_account_exactly() {
    let reg = registry(vec![]);
    let mut rng = Rng(31337);
    for i in 0..ROUNDS {
        let len = rng.below(400) as usize;
        let env = opaque(&mut rng, len, i % 2 == 0);
        let all = plan_envelope(&env, &bind(&reg, &[]).unwrap(), Some(0)).unwrap();
        assert_eq!(all.planned_bytes, env.encoded_len());
        assert_eq!(all.original_bytes, env.encoded_len());
        let regions = len.div_ceil(REGION_BYTES).min(MAX_REGIONS);
        let mask: Vec<MaskEntry> = (0..regions as u8)
            .map(|r| entry(FieldRef::Region(r), FieldClass::Discardable, 0))
            .collect();
        let plan = plan_envelope(&env, &bind(&reg, &mask).unwrap(), None).unwrap();
        assert_eq!(plan.planned_bytes, plan.fixed_bytes);
        assert_eq!(plan.dropped_regions.len(), regions);
        assert!(plan.kept_regions.is_empty());
    }
}

#[test]
fn reserved_downgrade_always_errors() {
    let reg = registry(vec![]);
    let reserved = [
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
    for r in reserved {
        for class in [FieldClass::Optional, FieldClass::Discardable] {
            let m = [entry(FieldRef::Reserved(r), class, 0)];
            assert_eq!(bind(&reg, &m), Err(SelectorError::ReservedDowngrade));
        }
        let ok = [entry(FieldRef::Reserved(r), FieldClass::Required, 0)];
        assert!(bind(&reg, &ok).is_ok());
    }
}

#[test]
fn symbol_downgrade_errors_and_required_symbol_is_fine() {
    let reg = registry(vec![]);
    for class in [FieldClass::Optional, FieldClass::Discardable] {
        let m = [entry(FieldRef::Symbol(3), class, 5)];
        assert_eq!(bind(&reg, &m), Err(SelectorError::SymbolNotSelectable));
    }
    let mut rng = Rng(8);
    let d = random_delta(&mut rng);
    let m = [entry(FieldRef::Symbol(0), FieldClass::Required, 0)];
    let plan = plan_delta(&d, &bind(&reg, &m).unwrap(), Some(0)).unwrap();
    assert_eq!(plan.kept_symbols.len(), d.updates.len());
}

#[test]
fn mandatory_downgrade_errors() {
    let reg = registry(vec![FieldRef::Residual(2), FieldRef::Region(1)]);
    for f in [FieldRef::Residual(2), FieldRef::Region(1)] {
        for class in [FieldClass::Optional, FieldClass::Discardable] {
            assert_eq!(
                bind(&reg, &[entry(f, class, 0)]),
                Err(SelectorError::MandatoryDowngrade)
            );
        }
        assert!(bind(&reg, &[entry(f, FieldClass::Required, 0)]).is_ok());
    }
    assert!(bind(
        &reg,
        &[entry(FieldRef::Residual(3), FieldClass::Discardable, 0)]
    )
    .is_ok());
}

#[test]
fn wrong_shape_entries_are_malformed_at_plan_time() {
    let reg = registry(vec![]);
    let mut rng = Rng(2);
    let d = random_delta(&mut rng);
    let region = bind(
        &reg,
        &[entry(FieldRef::Region(0), FieldClass::Discardable, 0)],
    )
    .unwrap();
    assert_eq!(plan_delta(&d, &region, None), Err(SelectorError::Malformed));
    assert_eq!(
        plan_envelope(&wrap(&d, false), &region, None),
        Err(SelectorError::Malformed)
    );
    let resid = bind(
        &reg,
        &[entry(FieldRef::Residual(0), FieldClass::Discardable, 0)],
    )
    .unwrap();
    let env = opaque(&mut rng, 64, false);
    assert_eq!(
        plan_envelope(&env, &resid, None),
        Err(SelectorError::Malformed)
    );
    // Region index beyond the body's region count.
    let far = bind(
        &reg,
        &[entry(FieldRef::Region(15), FieldClass::Discardable, 0)],
    )
    .unwrap();
    assert_eq!(
        plan_envelope(&env, &far, None),
        Err(SelectorError::Malformed)
    );
}

#[test]
fn unlisted_fields_default_to_required() {
    let reg = registry(vec![]);
    let mut rng = Rng(11);
    let d = random_delta(&mut rng);
    let plan = plan_delta(&d, &bind(&reg, &[]).unwrap(), Some(0)).unwrap();
    assert!(plan.dropped_residual_slots.is_empty());
    assert_eq!(plan.kept_residual_slots.len(), d.residuals.len());
}

#[test]
fn unknown_task_and_registry_validation() {
    let reg = registry(vec![]);
    let bytes = mask_bytes(TASK + 1, &[]);
    assert_eq!(
        reg.bind(UnboundMask::parse(&bytes).unwrap()),
        Err(SelectorError::UnknownTask)
    );
    let def = |task_id, mandatory| TaskDefinition {
        task_id,
        version: 1,
        mandatory,
    };
    assert_eq!(
        TaskRegistry::new(vec![def(1, vec![]), def(1, vec![])]),
        Err(SelectorError::DuplicateEntry)
    );
    let many: Vec<TaskDefinition> = (0..=MAX_TASKS as u16).map(|i| def(i, vec![])).collect();
    assert_eq!(TaskRegistry::new(many), Err(SelectorError::TooManyEntries));
    assert_eq!(
        TaskRegistry::new(vec![def(
            1,
            vec![FieldRef::Region(2), FieldRef::Residual(1)]
        )]),
        Err(SelectorError::NotCanonical)
    );
    assert_eq!(
        TaskRegistry::new(vec![def(1, vec![FieldRef::Region(2), FieldRef::Region(2)])]),
        Err(SelectorError::DuplicateEntry)
    );
}

#[test]
fn mask_round_trips_and_limits_are_fixed() {
    assert_eq!(MAX_MASK_ENTRIES, 80);
    assert_eq!(MAX_MASK_WIRE_BYTES, 324);
    let mask = vec![
        entry(FieldRef::Symbol(9), FieldClass::Required, 0),
        entry(FieldRef::Residual(4), FieldClass::Optional, 63),
        entry(FieldRef::Region(15), FieldClass::Discardable, 1),
        entry(
            FieldRef::Reserved(ReservedField::Epoch),
            FieldClass::Required,
            0,
        ),
    ];
    let parsed = UnboundMask::parse(&mask_bytes(TASK, &mask)).unwrap();
    assert_eq!(parsed.task_id(), TASK);
    assert_eq!(parsed.entries(), mask.as_slice());
}
