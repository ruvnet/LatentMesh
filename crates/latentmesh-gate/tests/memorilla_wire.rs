use latentmesh_core::{Authority, LatentFrame};
use latentmesh_gate::{AdmissionError, Gate, Policy};

#[test]
fn native_memorilla_frame_preserves_existing_gate_boundary() {
    let value: serde_json::Value = serde_json::from_str(include_str!(
        "../../../native/memorilla/fixtures/frame.json"
    ))
    .unwrap();
    let mut frame: LatentFrame = serde_json::from_value(value["frame"].clone()).unwrap();
    assert_eq!(value["schema"], "memorilla.frame.v1");
    assert_eq!(frame.authority, Authority::ObserveOnly);
    assert_eq!(frame.payload.decode().len(), 36);
    assert_eq!(frame.payload.wire_bytes(), 144);
    let mut policy = Policy::new(1.0);
    assert!(matches!(
        Gate::admit(&frame, &policy),
        Err(AdmissionError::UnknownTransform(_))
    ));
    policy.known_transforms.insert(frame.transform_hash.clone());
    assert!(Gate::admit(&frame, &policy).is_ok());
    frame.authority = Authority::LatentPrefix;
    assert!(matches!(
        Gate::admit(&frame, &policy),
        Err(AdmissionError::AuthorityExceedsCeiling { .. })
    ));
}
