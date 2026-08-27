//! Golden signing test: a fixed device keypair signing a fixed heartbeat
//! body at a fixed instant must produce an exact, reproducible signature —
//! Ed25519 (RFC 8032) has no randomness in the signing algorithm itself, so
//! this locks in the canonical-string construction + signing pipeline
//! against silent regressions (a changed field order, a changed separator,
//! a changed encoding would all change this value).
//!
//! The expected literals below were captured from this exact test on first
//! write and are independently re-verified in-test against the device's
//! own public key via [`ed25519_dalek::VerifyingKey::verify_strict`], so a
//! future maintainer doesn't have to trust the hardcoded literals alone —
//! only Ed25519 math.

use base64::{engine::general_purpose::STANDARD, Engine as _};
use ed25519_dalek::VerifyingKey;
use latentmesh_cognitum_client::{
    canonical_string, DeviceIdentity, FixedClock, SeedHeartbeatRequest, SEED_HEARTBEAT_PATH,
};

const DEVICE_ID: &str = "0193f1e2-7c4a-7000-9b2e-1a2b3c4d5e6f";
// Fixed, non-secret test seed — not a real device key.
const SEED: [u8; 32] = [
    0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10,
    0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f, 0x20,
];
const TIMESTAMP_UNIX: i64 = 1_777_562_280; // "2026-04-30T15:18:00Z"
const TIMESTAMP_ISO8601: &str = "2026-04-30T15:18:00Z";

// Captured once from `DeviceIdentity::from_signing_key_bytes(DEVICE_ID, &SEED)`
// signing `body()` at `TIMESTAMP_UNIX` — deterministic for this fixed seed
// and message (Ed25519 signing has no randomness), independently
// re-verified in `golden_signature_is_exact_and_self_verifies` below.
const GOLDEN_PUBLIC_KEY_BASE64: &str = "ebVWLo/mVPlAeLES6KmLp5AfhTrmlb7X4OORC60ElmQ=";
const GOLDEN_SIGNATURE_BASE64: &str =
    "q+q0m99wxjesUT6zEOTn+pM273jdewB7EkyPf+6YeDE+zb0tqwxbnf30atj2NvKwEYyah1LgII2lIJYAW1iYAw==";

/// docs/seed-integration.md's worked heartbeat body, cognitum-one/api.
fn body() -> Vec<u8> {
    let req = SeedHeartbeatRequest {
        uptime_secs: Some(12345),
        free_memory_kb: Some(89432),
        total_vectors: Some(42705),
        epoch: Some(1_730_345_880),
        wifi_ip: Some("192.168.1.106".to_string()),
        version: Some("v0.10.12".to_string()),
    };
    serde_json::to_vec(&req).expect("serialize heartbeat body")
}

#[test]
fn golden_signature_is_exact_and_self_verifies() {
    let identity = DeviceIdentity::from_signing_key_bytes(DEVICE_ID, &SEED);
    let clock = FixedClock(TIMESTAMP_UNIX);
    let body = body();

    let signed = identity.sign(&clock, "POST", SEED_HEARTBEAT_PATH, &body);

    assert_eq!(signed.method, "POST");
    assert_eq!(signed.path, "/v1/seed/heartbeat");
    assert_eq!(signed.body, body);
    assert_eq!(
        signed.headers,
        vec![
            ("X-Device-Id".to_string(), DEVICE_ID.to_string()),
            (
                "X-Device-Timestamp".to_string(),
                TIMESTAMP_ISO8601.to_string()
            ),
            (
                "X-Device-Signature".to_string(),
                GOLDEN_SIGNATURE_BASE64.to_string()
            ),
            ("Content-Type".to_string(), "application/json".to_string()),
        ]
    );
    assert_eq!(identity.public_key_base64(), GOLDEN_PUBLIC_KEY_BASE64);

    // Independent re-verification: decode the signature and public key
    // exactly as a server would, rebuild the canonical string from the
    // signed request's own fields, and verify — this does not rely on the
    // hardcoded golden literals being correct, only on Ed25519 math.
    let signature_bytes = STANDARD
        .decode(GOLDEN_SIGNATURE_BASE64)
        .expect("valid base64 signature");
    let signature = ed25519_dalek::Signature::from_bytes(
        &signature_bytes.try_into().expect("64-byte signature"),
    );
    let public_key_bytes = STANDARD
        .decode(GOLDEN_PUBLIC_KEY_BASE64)
        .expect("valid base64 public key");
    let verifying_key =
        VerifyingKey::from_bytes(&public_key_bytes.try_into().expect("32-byte key"))
            .expect("valid Ed25519 public key");
    let canonical = canonical_string(
        &signed.method,
        &signed.path,
        TIMESTAMP_ISO8601,
        &signed.body,
    );
    verifying_key
        .verify_strict(canonical.as_bytes(), &signature)
        .expect("golden signature must verify against the golden public key");
}

#[test]
fn sha256_of_body_is_embedded_in_the_canonical_string() {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(body());
    let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
    let canonical = canonical_string("POST", SEED_HEARTBEAT_PATH, TIMESTAMP_ISO8601, &body());
    assert!(canonical.ends_with(&hex));
}
