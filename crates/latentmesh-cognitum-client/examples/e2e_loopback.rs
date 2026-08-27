//! ADR-022 e2e loopback driver for `harness/integration`'s cognitum contract
//! suite: build and sign a canonical request (ADR-021) against a fixture
//! keypair, replay it against a mock HTTP server built from the published
//! OpenAPI contract, confirm signature verification and payload shape both
//! hold.
//!
//! Deterministic: a fixed Ed25519 seed and a [`FixedClock`] (never
//! `SystemClock`/`SystemTime::now()`) so the canonical string, signature,
//! and timestamp are byte-reproducible across runs. The mock server is a
//! same-process `std::net::TcpListener` loopback responder — the same
//! technique this crate's own `http.rs` tests use — never a network call to
//! `api.cognitum.one`. Requires the `http` feature (`CognitumHttpClient`).
//! Prints one JSON object to stdout; this binary does not itself claim an
//! evidence label.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::sync::mpsc;
use std::thread;

use base64::{engine::general_purpose::STANDARD, Engine as _};
use ed25519_dalek::{Signature, VerifyingKey};
use latentmesh_cognitum_client::{
    canonical_string, CognitumHttpClient, DeviceIdentity, FixedClock, SeedHeartbeatRequest,
    SeedRegisterRequest,
};

/// Fixed seed and clock so this driver's canonical string, signature, and
/// timestamp are byte-reproducible — matching `signing.rs`'s own golden
/// tests, which use the same seed/timestamp pair.
const SEED: [u8; 32] = [7_u8; 32];
const DEVICE_ID: &str = "0193f1e2-7c4a-7000-9b2e-1a2b3c4d5e6f";
const CLOCK_UNIX: i64 = 1_777_562_280;

struct RecordedRequest {
    method: String,
    path: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

/// A minimal single-request HTTP/1.1 responder, same technique as
/// `src/http.rs`'s own `spawn_mock_server` test helper (not reusable
/// directly — it is `#[cfg(test)]`-private to that module).
fn spawn_mock_server(response_json: &'static str) -> (String, mpsc::Receiver<RecordedRequest>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local addr");
    let (tx, rx) = mpsc::channel();

    thread::spawn(move || {
        let (stream, _) = listener.accept().expect("accept one connection");
        let mut reader = BufReader::new(stream.try_clone().expect("clone stream"));

        let mut request_line = String::new();
        reader
            .read_line(&mut request_line)
            .expect("read request line");
        let mut parts = request_line.split_whitespace();
        let method = parts.next().unwrap_or_default().to_string();
        let path = parts.next().unwrap_or_default().to_string();

        let mut headers = Vec::new();
        let mut content_length = 0usize;
        loop {
            let mut line = String::new();
            reader.read_line(&mut line).expect("read header line");
            let line = line.trim_end_matches(['\r', '\n']);
            if line.is_empty() {
                break;
            }
            if let Some((name, value)) = line.split_once(':') {
                let name = name.trim().to_string();
                let value = value.trim().to_string();
                if name.eq_ignore_ascii_case("content-length") {
                    content_length = value.parse().unwrap_or(0);
                }
                headers.push((name, value));
            }
        }

        let mut body = vec![0u8; content_length];
        if content_length > 0 {
            reader.read_exact(&mut body).expect("read body");
        }

        let _ = tx.send(RecordedRequest {
            method,
            path,
            headers,
            body,
        });

        let mut stream = stream;
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            response_json.len(),
            response_json
        );
        stream
            .write_all(response.as_bytes())
            .expect("write response");
    });

    (format!("http://{addr}"), rx)
}

fn header<'a>(recorded: &'a RecordedRequest, name: &str) -> Option<&'a str> {
    recorded
        .headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case(name))
        .map(|(_, v)| v.as_str())
}

fn main() {
    let identity = DeviceIdentity::from_signing_key_bytes(DEVICE_ID, &SEED);
    let clock = FixedClock(CLOCK_UNIX);

    // --- Unauthenticated register call: unsigned per the OpenAPI contract
    // (`security: []`), so this leg checks payload shape only.
    let (register_base_url, register_rx) = spawn_mock_server(r#"{"registered":true}"#);
    let register_client = CognitumHttpClient::new(register_base_url);
    let register_request = SeedRegisterRequest {
        device_id: DEVICE_ID.to_string(),
        public_key: identity.public_key_base64(),
        firmware: Some("v0.11.0".to_string()),
    };
    let register_response = register_client.register(&register_request).unwrap();
    let registered_ok = register_response
        .extra
        .get("registered")
        .and_then(|v| v.as_bool())
        == Some(true);
    let register_recorded = register_rx.recv().expect("register request recorded");
    let register_shape_ok = register_recorded.method == "POST"
        && register_recorded.path == "/v1/seed/register"
        && header(&register_recorded, "X-Device-Signature").is_none();

    // --- Signed heartbeat call: exercises the full canonical-request +
    // Ed25519 signing + server-side verification path.
    let (heartbeat_base_url, heartbeat_rx) = spawn_mock_server(r#"{"ack":true}"#);
    let heartbeat_client = CognitumHttpClient::new(heartbeat_base_url);
    let heartbeat_request = SeedHeartbeatRequest {
        uptime_secs: Some(12345),
        free_memory_kb: Some(89432),
        total_vectors: Some(42705),
        epoch: Some(1_730_345_880),
        wifi_ip: Some("192.168.1.106".to_string()),
        version: Some("v0.10.12".to_string()),
    };
    let heartbeat_response = heartbeat_client
        .heartbeat(&identity, &clock, &heartbeat_request)
        .unwrap();
    let acked_ok = heartbeat_response
        .extra
        .get("ack")
        .and_then(|v| v.as_bool())
        == Some(true);
    let heartbeat_recorded = heartbeat_rx.recv().expect("heartbeat request recorded");

    let device_id_header_ok = header(&heartbeat_recorded, "X-Device-Id") == Some(DEVICE_ID);
    let timestamp_header = header(&heartbeat_recorded, "X-Device-Timestamp")
        .unwrap_or_default()
        .to_string();
    let signature_b64 = header(&heartbeat_recorded, "X-Device-Signature")
        .unwrap_or_default()
        .to_string();

    // Reconstruct the canonical string from exactly what the mock server
    // received on the wire (method/path/timestamp/body) — independent of
    // the client's own signing call — and verify server-side, the same way
    // `docs/seed-integration.md`'s server-side check would.
    let canonical = canonical_string(
        &heartbeat_recorded.method,
        &heartbeat_recorded.path,
        &timestamp_header,
        &heartbeat_recorded.body,
    );
    let sig_bytes = STANDARD.decode(&signature_b64).unwrap_or_default();
    let public_key_bytes = STANDARD.decode(identity.public_key_base64()).unwrap();
    let signature_verified = match (
        <[u8; 64]>::try_from(sig_bytes.as_slice()),
        <[u8; 32]>::try_from(public_key_bytes.as_slice()),
    ) {
        (Ok(sig_arr), Ok(key_arr)) => {
            let signature = Signature::from_bytes(&sig_arr);
            VerifyingKey::from_bytes(&key_arr)
                .map(|vk| vk.verify_strict(canonical.as_bytes(), &signature).is_ok())
                .unwrap_or(false)
        }
        _ => false,
    };

    let heartbeat_body_round_trips: SeedHeartbeatRequest =
        serde_json::from_slice(&heartbeat_recorded.body).unwrap();
    let heartbeat_shape_ok = heartbeat_recorded.method == "POST"
        && heartbeat_recorded.path == "/v1/seed/heartbeat"
        && heartbeat_body_round_trips == heartbeat_request;

    let output = serde_json::json!({
        "suite": "cognitum-contract-loopback-v1",
        "device_id": DEVICE_ID,
        "clock_unix": CLOCK_UNIX,
        "timestamp_iso8601": timestamp_header,
        "register": {
            "shape_ok": register_shape_ok,
            "response_ok": registered_ok,
        },
        "heartbeat": {
            "shape_ok": heartbeat_shape_ok,
            "response_ok": acked_ok,
            "device_id_header_ok": device_id_header_ok,
            "signature_verified": signature_verified,
        },
    });

    println!("{output}");
}
