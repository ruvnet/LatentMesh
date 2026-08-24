use latentmesh_air_core::{
    bytes_to_bits_msb, convolutional_encode, CriticalState, FrameFlags, SemanticClass,
    SemanticDelta, SemanticEnvelope, SparseRadioFrame, SymbolUpdate, SymbolValue, WireProfile,
};

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(DIGITS[(byte >> 4) as usize] as char);
        out.push(DIGITS[(byte & 0xf) as usize] as char);
    }
    out
}

#[test]
fn outer_frame_matches_cross_language_vector() {
    let frame = SparseRadioFrame {
        profile: WireProfile::HfBpsk,
        flags: FrameFlags::ACK_REQUEST.union(FrameFlags::FEC),
        stream_id: 0x1234,
        sequence: 0x0102,
        fragment_index: 0,
        fragment_count: 1,
        class: SemanticClass::StateDelta,
        priority: 15,
        state_tag: 0xbeef,
        payload: vec![0xde, 0xad, 0xbe, 0xef],
    };
    assert_eq!(
        hex(&frame.encode().unwrap()),
        include_str!("../testdata/wire_frame_v1.hex").trim()
    );
}

#[test]
fn semantic_envelope_matches_golden_vector() {
    let mut before = CriticalState::new();
    before.set(1, SymbolValue::Bool(false)).unwrap();
    let mut after = CriticalState::new();
    after.set(1, SymbolValue::Bool(true)).unwrap();
    let delta = SemanticDelta {
        source_id: 1,
        epoch: 2,
        message_id: 3,
        base_hash: before.critical_hash(),
        result_hash: after.critical_hash(),
        updates: vec![SymbolUpdate::Set {
            field_id: 1,
            value: SymbolValue::Bool(true),
        }],
        residuals: vec![],
    };
    assert_eq!(
        hex(&delta.encode().unwrap()),
        include_str!("../testdata/semantic_delta_v1.hex").trim()
    );
}

#[test]
fn cross_language_lms1_envelope_matches_golden_vector() {
    let envelope = SemanticEnvelope {
        class: SemanticClass::StateDelta,
        priority: 15,
        source_id: 1,
        epoch: 2,
        message_id: 3,
        logical_sequence: 4,
        state_hash: [
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d,
            0x0e, 0x0f,
        ],
        body: vec![0xde, 0xad, 0xbe, 0xef],
        signature: None,
    };
    assert_eq!(
        hex(&envelope.encode().unwrap()),
        include_str!("../testdata/semantic_envelope_v1.hex").trim()
    );
}

#[test]
fn convolutional_encoder_matches_golden_bits() {
    let coded = convolutional_encode(&bytes_to_bits_msb(&[0xa5])).unwrap();
    let rendered: String = coded
        .iter()
        .map(|bit| if *bit == 1 { '1' } else { '0' })
        .collect();
    assert_eq!(rendered, include_str!("../testdata/fec_k7_a5.bits").trim());
}
