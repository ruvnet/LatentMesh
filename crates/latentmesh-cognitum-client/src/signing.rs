//! Ed25519 canonical-request signing for `cognitum-one/api`'s `/v1/seed/*`
//! surface (ADR-021).
//!
//! The canonical string and signed-header scheme are transcribed from the
//! canonical, documented contract in the `cognitum-one/api` repository —
//! `openapi/cognitum-api.yaml`'s `components.securitySchemes.ed25519`
//! (`type: apiKey`, header `X-Device-Signature`, description: "Ed25519
//! signature over `METHOD\nPATH\nTIMESTAMP\nsha256(body)`. Requires
//! `X-Device-Id` and `X-Device-Timestamp` headers.") and
//! `docs/seed-integration.md`'s "Authentication" section, which gives the
//! same construction with a worked example. That repo's `README.md`
//! explicitly marks `functions/seed/` — a *different*, non-matching scheme
//! (hex-encoded keys/signatures, no-newline concatenation, `X-Timestamp`/
//! `X-Signature` headers, no replay-window check) — as a "non-canonical
//! reference fork; do not deploy to production", so this crate follows the
//! OpenAPI/docs contract, not that fork. See `src/lib.rs` for the full
//! deviation note.
//!
//! Signing is over the raw canonical *string* bytes with standard Ed25519
//! (RFC 8032) — not pre-hashed ("Ed25519ph") — matching "Ed25519 signature
//! over `METHOD\n...`" read literally: the thing being signed is the
//! canonical string, not a digest of it. `sha256(body)` is only the body's
//! contribution *inside* that string.
//!
//! **Encoding choices the source does not literally pin — confirmed only
//! against this crate's own tests, not a live server:** neither the OpenAPI
//! spec nor `docs/seed-integration.md` states the encoding of `sha256(body)`
//! inside the canonical string, so this module uses lower-case hex (the
//! conventional choice, matching common canonical-request schemes such as
//! AWS SigV4). Likewise `X-Device-Signature` and `publicKey` are encoded as
//! standard (padded) base64 per the OpenAPI `publicKey` field description
//! ("Base64 Ed25519 public key") — not RFC 4648 URL-safe/unpadded base64,
//! which is never mentioned. `path` in the canonical string is the bare
//! request-target path with no query string for `/v1/seed/register` and
//! `/v1/seed/heartbeat` (neither route takes query parameters), so this is
//! untested for a route that does (e.g. `/v1/seed/check?current=...`) — a
//! caller building a [`SignedRequest`] for such a route should confirm
//! against a real server whether the query string is part of `PATH` before
//! relying on it. The first credentialed call against `api.cognitum.one`
//! should verify all of these before this crate's signing is trusted
//! end-to-end.

use std::fmt::Write as _;

use ed25519_dalek::{Signer, SigningKey};
use sha2::{Digest, Sha256};

use crate::clock::Clock;

/// Lower-case hex-encoded SHA-256 digest of `data`.
fn hex_sha256(data: &[u8]) -> String {
    let digest = Sha256::digest(data);
    let mut out = String::with_capacity(64);
    for byte in digest {
        // `write!` to a `String` is infallible.
        let _ = write!(out, "{byte:02x}");
    }
    out
}

/// Build the canonical string a seed device signs:
/// `METHOD\nPATH\nTIMESTAMP\nsha256(body)`.
///
/// `method` is upper-cased; `path` and `timestamp_iso8601` are used
/// verbatim (the caller is responsible for `path` being the exact request
/// path the server will see, e.g. `/v1/seed/heartbeat`, and
/// `timestamp_iso8601` being produced by
/// [`crate::timestamp::unix_to_iso8601_utc`] so it matches the
/// `X-Device-Timestamp` header sent alongside it).
pub fn canonical_string(method: &str, path: &str, timestamp_iso8601: &str, body: &[u8]) -> String {
    format!(
        "{}\n{}\n{}\n{}",
        method.to_ascii_uppercase(),
        path,
        timestamp_iso8601,
        hex_sha256(body)
    )
}

/// A fully-built, signed HTTP request, independent of any transport. A
/// caller can hand this to any HTTP client — `reqwest`, `ureq`, an
/// embedded-`no_std` stack, whatever fits their deployment — by reading off
/// `method`/`path`/`headers`/`body`. The `http` feature's
/// [`crate::http::CognitumHttpClient`] is one such caller, using `ureq`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SignedRequest {
    /// Upper-case HTTP method, e.g. `"POST"`.
    pub method: String,
    /// The exact request-target path, e.g. `"/v1/seed/heartbeat"`. This is
    /// the same string that was signed into the canonical request — it
    /// MUST be sent to the server unchanged (no trailing-slash
    /// normalization, no query-string reordering) or the signature will
    /// not verify server-side.
    pub path: String,
    /// Header name/value pairs to attach to the request, including the
    /// three signed-auth headers (`X-Device-Id`, `X-Device-Timestamp`,
    /// `X-Device-Signature`) and `Content-Type` when there is a body.
    pub headers: Vec<(String, String)>,
    /// The exact bytes that were hashed into the canonical string. Sending
    /// different bytes than these will not verify.
    pub body: Vec<u8>,
}

/// A registered Seed device's Ed25519 identity: a `deviceId` and the
/// keypair used to sign every request after `/v1/seed/register`.
///
/// The private key never leaves this type — there is no accessor for it.
/// Credentials are supplied by the caller at construction time (from an
/// environment variable, a keystore, wherever); this crate never reads an
/// environment variable or embeds a default keypair itself, so a build of
/// this crate cannot accidentally authenticate as any real device.
pub struct DeviceIdentity {
    device_id: String,
    signing_key: SigningKey,
}

impl DeviceIdentity {
    /// Build an identity from an already-loaded Ed25519 signing key.
    pub fn new(device_id: impl Into<String>, signing_key: SigningKey) -> Self {
        Self {
            device_id: device_id.into(),
            signing_key,
        }
    }

    /// Build an identity from a raw 32-byte Ed25519 seed (the private-key
    /// bytes `SigningKey::from_bytes` expects). This is the constructor a
    /// caller loading `/var/lib/cognitum/device-key` (per
    /// `docs/seed-integration.md`'s "Identity" section) would use.
    pub fn from_signing_key_bytes(device_id: impl Into<String>, seed: &[u8; 32]) -> Self {
        Self::new(device_id, SigningKey::from_bytes(seed))
    }

    pub fn device_id(&self) -> &str {
        &self.device_id
    }

    /// The device's public key, base64-encoded — the exact encoding
    /// `POST /v1/seed/register`'s `publicKey` field expects
    /// (`openapi/cognitum-api.yaml`: `publicKey: { type: string,
    /// description: Base64 Ed25519 public key }`).
    pub fn public_key_base64(&self) -> String {
        use base64::{engine::general_purpose::STANDARD, Engine as _};
        STANDARD.encode(self.signing_key.verifying_key().to_bytes())
    }

    /// Sign a request per the canonical scheme documented in this module,
    /// using `clock` for the timestamp (never `SystemTime::now()` directly
    /// — see [`crate::clock`]).
    pub fn sign(&self, clock: &dyn Clock, method: &str, path: &str, body: &[u8]) -> SignedRequest {
        use base64::{engine::general_purpose::STANDARD, Engine as _};

        let timestamp = crate::timestamp::unix_to_iso8601_utc(clock.now_unix());
        let canonical = canonical_string(method, path, &timestamp, body);
        let signature = self.signing_key.sign(canonical.as_bytes());
        let signature_b64 = STANDARD.encode(signature.to_bytes());

        let mut headers = vec![
            ("X-Device-Id".to_string(), self.device_id.clone()),
            ("X-Device-Timestamp".to_string(), timestamp),
            ("X-Device-Signature".to_string(), signature_b64),
        ];
        if !body.is_empty() {
            headers.push(("Content-Type".to_string(), "application/json".to_string()));
        }

        SignedRequest {
            method: method.to_ascii_uppercase(),
            path: path.to_string(),
            headers,
            body: body.to_vec(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::FixedClock;
    use ed25519_dalek::VerifyingKey;

    #[test]
    fn canonical_string_matches_the_documented_shape() {
        let s = canonical_string("post", "/v1/seed/heartbeat", "2026-04-30T15:18:00Z", b"{}");
        assert_eq!(
            s,
            format!(
                "POST\n/v1/seed/heartbeat\n2026-04-30T15:18:00Z\n{}",
                hex_sha256(b"{}")
            )
        );
        // Body hash is the well-known SHA-256("{}") value, independently
        // computable (e.g. `printf '{}' | sha256sum`).
        assert_eq!(
            hex_sha256(b"{}"),
            "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
        );
    }

    #[test]
    fn canonical_string_of_empty_body_is_sha256_of_empty_string() {
        let s = canonical_string("GET", "/v1/seed/check", "1970-01-01T00:00:00Z", b"");
        // The well-known SHA-256 of the empty string (e.g. `printf '' | sha256sum`).
        assert!(s.ends_with("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"));
    }

    #[test]
    fn sign_produces_a_signature_verifiable_against_the_public_key() {
        let seed = [7u8; 32];
        let identity =
            DeviceIdentity::from_signing_key_bytes("0193f1e2-7c4a-7000-9b2e-1a2b3c4d5e6f", &seed);
        let clock = FixedClock(1_777_562_280);
        let signed = identity.sign(
            &clock,
            "post",
            "/v1/seed/heartbeat",
            br#"{"uptime_secs":1}"#,
        );

        assert_eq!(signed.method, "POST");
        assert_eq!(signed.path, "/v1/seed/heartbeat");
        assert_eq!(
            signed.headers[0],
            (
                "X-Device-Id".to_string(),
                "0193f1e2-7c4a-7000-9b2e-1a2b3c4d5e6f".to_string()
            )
        );
        assert_eq!(
            signed.headers[1],
            (
                "X-Device-Timestamp".to_string(),
                "2026-04-30T15:18:00Z".to_string()
            )
        );

        let sig_b64 = &signed.headers[2].1;
        use base64::{engine::general_purpose::STANDARD, Engine as _};
        let sig_bytes = STANDARD.decode(sig_b64).expect("valid base64");
        let signature =
            ed25519_dalek::Signature::from_bytes(&sig_bytes.try_into().expect("64-byte signature"));

        let public_key_bytes = STANDARD
            .decode(identity.public_key_base64())
            .expect("valid base64");
        let verifying_key =
            VerifyingKey::from_bytes(&public_key_bytes.try_into().expect("32-byte key"))
                .expect("valid Ed25519 public key");

        let canonical = canonical_string(
            "POST",
            "/v1/seed/heartbeat",
            "2026-04-30T15:18:00Z",
            br#"{"uptime_secs":1}"#,
        );
        verifying_key
            .verify_strict(canonical.as_bytes(), &signature)
            .expect("signature must verify against the device's own public key");
    }

    #[test]
    fn empty_body_omits_content_type_header() {
        let seed = [1u8; 32];
        let identity = DeviceIdentity::from_signing_key_bytes("device-a", &seed);
        let signed = identity.sign(&FixedClock(0), "GET", "/v1/seed/check", b"");
        assert!(!signed.headers.iter().any(|(k, _)| k == "Content-Type"));
        assert_eq!(signed.headers.len(), 3);
    }

    #[test]
    fn non_empty_body_adds_json_content_type_header() {
        let seed = [1u8; 32];
        let identity = DeviceIdentity::from_signing_key_bytes("device-a", &seed);
        let signed = identity.sign(&FixedClock(0), "POST", "/v1/seed/heartbeat", b"{}");
        assert_eq!(
            signed.headers[3],
            ("Content-Type".to_string(), "application/json".to_string())
        );
    }

    #[test]
    fn signing_is_deterministic_for_the_same_inputs() {
        let seed = [42u8; 32];
        let identity = DeviceIdentity::from_signing_key_bytes("device-b", &seed);
        let clock = FixedClock(1_700_000_000);
        let a = identity.sign(&clock, "POST", "/v1/seed/heartbeat", b"{}");
        let b = identity.sign(&clock, "POST", "/v1/seed/heartbeat", b"{}");
        assert_eq!(a, b, "Ed25519 (RFC 8032) signing has no randomness");
    }
}
