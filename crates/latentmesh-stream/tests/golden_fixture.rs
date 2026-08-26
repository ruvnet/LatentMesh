//! Cross-repository golden fixture (ADR-015): the byte-exact wire encoding of
//! one canonical `LatentFrame`. The identical fixture is checked into the
//! MidStream repository's `latentmesh-bridge` crate; both sides must decode
//! it to the same frame and re-encode it to the same bytes, which is what
//! keeps the two independent codecs compatible without a shared dependency.
//!
//! Regenerate (after a deliberate wire-format change only) with:
//! `LM_BLESS_GOLDEN=1 cargo test -p latentmesh-stream --test golden_fixture`

use latentmesh_core::{Authority, Encoding, LatentFrame, Payload, Provenance};
use latentmesh_stream::{decode_frame, encode_frame};
use std::path::PathBuf;

fn golden_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata/latent_frame_golden.hex")
}

fn canonical_frame() -> LatentFrame {
    let values: Vec<f32> = (0..16).map(|i| (i as f32) / 8.0 - 1.0).collect();
    LatentFrame {
        id: "golden-frame-0001".into(),
        sender_model: "sender-model-a".into(),
        receiver_space: "receiver-space-b".into(),
        transform_hash: "golden-transform".into(),
        sequence: 42,
        payload: Payload::encode(&values, Encoding::Int8),
        confidence: 0.875,
        provenance: Provenance {
            sender_model: "sender-model-a".into(),
            context_hash: "context-hash-0001".into(),
            parents: vec!["parent-0000".into()],
        },
        authority: Authority::ContextInject,
        timestamp: 1_756_166_400,
    }
}

fn to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn from_hex(hex: &str) -> Vec<u8> {
    let hex = hex.trim();
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).expect("valid hex fixture"))
        .collect()
}

#[test]
fn golden_fixture_round_trips_byte_for_byte() {
    let frame = canonical_frame();
    let encoded = encode_frame(&frame).expect("canonical frame encodes");

    if std::env::var_os("LM_BLESS_GOLDEN").is_some() {
        std::fs::write(golden_path(), to_hex(&encoded) + "\n").expect("write fixture");
    }

    let fixture_hex = std::fs::read_to_string(golden_path())
        .expect("testdata/latent_frame_golden.hex must exist (run once with LM_BLESS_GOLDEN=1)");
    let fixture = from_hex(&fixture_hex);

    assert_eq!(
        to_hex(&encoded),
        to_hex(&fixture),
        "encoding the canonical frame no longer matches the golden fixture — \
         this is a wire-format break the MidStream bridge will see too"
    );

    let (decoded, consumed) = decode_frame(&fixture)
        .expect("fixture decodes")
        .expect("fixture is complete");
    assert_eq!(consumed, fixture.len());
    assert_eq!(decoded.id, frame.id);
    assert_eq!(decoded.sequence, frame.sequence);
    assert_eq!(decoded.authority, frame.authority);
    assert_eq!(decoded.payload.bytes, frame.payload.bytes);
    assert_eq!(decoded.payload.int8_params, frame.payload.int8_params);
    assert_eq!(decoded.content_hash(), frame.content_hash());
}
