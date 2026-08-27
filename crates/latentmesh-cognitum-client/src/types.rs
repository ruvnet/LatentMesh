//! Request/response payload types for `POST /v1/seed/register` and
//! `POST /v1/seed/heartbeat`, transcribed field-for-field from
//! `cognitum-one/api`'s `openapi/cognitum-api.yaml` (paths `/v1/seed/register`
//! and `/v1/seed/heartbeat`).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// The request-target paths signed into the canonical string and sent on
/// the wire, exactly as declared in `openapi/cognitum-api.yaml` and
/// `src/gateway/upstream.ts`'s route table in `cognitum-one/api`. Kept as
/// named constants so there is exactly one source of truth shared by
/// signing and transport — see [`crate::http::CognitumHttpClient`].
pub const SEED_REGISTER_PATH: &str = "/v1/seed/register";
pub const SEED_HEARTBEAT_PATH: &str = "/v1/seed/heartbeat";

/// Body of `POST /v1/seed/register`.
///
/// `openapi/cognitum-api.yaml`: `required: [deviceId, publicKey]`,
/// `deviceId: { type: string, format: uuid }`,
/// `publicKey: { type: string, description: Base64 Ed25519 public key }`,
/// `firmware: { type: string }` (optional). This route is unauthenticated
/// (`security: []` — "device provisioning uses Ed25519 sig on subsequent
/// calls") so, unlike heartbeat, no [`crate::signing::SignedRequest`] is
/// involved in sending it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SeedRegisterRequest {
    #[serde(rename = "deviceId")]
    pub device_id: String,
    #[serde(rename = "publicKey")]
    pub public_key: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub firmware: Option<String>,
}

/// Body of `POST /v1/seed/heartbeat`.
///
/// `openapi/cognitum-api.yaml`'s schema for this route lists every field as
/// optional (no `required` array) — all-`Option` here matches that
/// documented contract exactly, rather than assuming any field is
/// mandatory beyond what the spec states.
///
/// **Naming note (ADR-021):** `epoch` here is cognitum's fleet-side
/// heartbeat counter. It is unrelated to, and MUST NOT be aliased with,
/// LatentMesh's own `SemanticEnvelope.epoch` field
/// (`latentmesh-air-core::envelope`) — the two systems do not share an
/// epoch domain; the name collision is coincidental.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SeedHeartbeatRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uptime_secs: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub free_memory_kb: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_vectors: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub epoch: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wifi_ip: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
}

/// Response body for `POST /v1/seed/register`.
///
/// `openapi/cognitum-api.yaml`'s responses for this route declare only
/// human-readable descriptions (`200: Registered`, `409: deviceId already
/// registered with different publicKey`) with no `content` schema — the
/// response shape is not part of the published contract. This type is
/// therefore intentionally a permissive, non-normative capture of whatever
/// JSON object comes back: every field lands in `extra` and unknown keys
/// are preserved rather than rejected, so it never fails to parse a
/// legitimate response for having an undocumented shape. Replace this with
/// a strict typed response if/when the org publishes one.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SeedRegisterResponse {
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

/// Response body for `POST /v1/seed/heartbeat`. Same non-normative shape as
/// [`SeedRegisterResponse`], for the same reason: `openapi/cognitum-api.yaml`
/// declares `200: Acknowledged` with no response `content` schema.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SeedHeartbeatResponse {
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn register_request_serializes_with_camel_case_field_names() {
        let req = SeedRegisterRequest {
            device_id: "0193f1e2-7c4a-7000-9b2e-1a2b3c4d5e6f".to_string(),
            public_key: "cHVibGljLWtleS1ieXRlcw==".to_string(),
            firmware: Some("v0.11.0".to_string()),
        };
        let value = serde_json::to_value(&req).unwrap();
        assert_eq!(
            value,
            json!({
                "deviceId": "0193f1e2-7c4a-7000-9b2e-1a2b3c4d5e6f",
                "publicKey": "cHVibGljLWtleS1ieXRlcw==",
                "firmware": "v0.11.0",
            })
        );
        let round_tripped: SeedRegisterRequest = serde_json::from_value(value).unwrap();
        assert_eq!(round_tripped, req);
    }

    #[test]
    fn register_request_omits_absent_optional_firmware() {
        let req = SeedRegisterRequest {
            device_id: "d".to_string(),
            public_key: "p".to_string(),
            firmware: None,
        };
        let value = serde_json::to_value(&req).unwrap();
        assert!(value.get("firmware").is_none());
    }

    #[test]
    fn heartbeat_request_round_trips_with_only_some_fields_set() {
        let req = SeedHeartbeatRequest {
            uptime_secs: Some(12345),
            total_vectors: Some(42705),
            ..Default::default()
        };
        let value = serde_json::to_value(&req).unwrap();
        assert_eq!(value, json!({"uptime_secs": 12345, "total_vectors": 42705}));
        let round_tripped: SeedHeartbeatRequest = serde_json::from_value(value).unwrap();
        assert_eq!(round_tripped, req);
    }

    #[test]
    fn heartbeat_request_matches_the_seed_integration_doc_example() {
        // docs/seed-integration.md's worked heartbeat body, restricted to
        // the fields openapi/cognitum-api.yaml actually documents — the
        // doc example includes a couple of extra agent-local fields that
        // are not in the published schema and are deliberately not
        // modeled by this type.
        let doc_example = json!({
            "uptime_secs": 12345,
            "free_memory_kb": 89432,
            "total_vectors": 42705,
            "epoch": 1730345880,
            "wifi_ip": "192.168.1.106",
            "version": "v0.10.12"
        });
        let parsed: SeedHeartbeatRequest = serde_json::from_value(doc_example.clone()).unwrap();
        assert_eq!(serde_json::to_value(&parsed).unwrap(), doc_example);
    }

    #[test]
    fn register_response_is_permissive_and_preserves_unknown_fields() {
        // Canned/recorded-shape fixture — no live server response exists
        // to record against (see the type's rustdoc); this stands in for
        // "whatever shape the server actually returns".
        let canned = json!({
            "registered": true,
            "device_id": "0193f1e2-7c4a-7000-9b2e-1a2b3c4d5e6f",
            "update_channel": "stable",
            "check_interval_secs": 3600
        });
        let parsed: SeedRegisterResponse = serde_json::from_value(canned.clone()).unwrap();
        assert_eq!(serde_json::to_value(&parsed).unwrap(), canned);
    }

    #[test]
    fn heartbeat_response_accepts_an_empty_object() {
        let parsed: SeedHeartbeatResponse = serde_json::from_value(json!({})).unwrap();
        assert!(parsed.extra.is_empty());
    }

    #[test]
    fn heartbeat_response_accepts_arbitrary_extra_fields() {
        let canned = json!({"ack": true, "next_heartbeat_secs": 3600, "commands": []});
        let parsed: SeedHeartbeatResponse = serde_json::from_value(canned.clone()).unwrap();
        assert_eq!(serde_json::to_value(&parsed).unwrap(), canned);
    }
}
