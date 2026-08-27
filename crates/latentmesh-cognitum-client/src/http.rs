//! Blocking HTTP transport (`ureq`) for [`crate::signing::SignedRequest`],
//! behind the `http` feature. This is the only part of the crate that
//! touches the network — signing and payload construction (the rest of the
//! crate) never require it.
//!
//! `CognitumHttpClient::new` takes the base URL as a required argument with
//! no `Default` impl and no built-in fallback to `https://api.cognitum.one`
//! — a build of this crate cannot silently start talking to real
//! infrastructure. Device credentials are likewise supplied by the caller
//! (see [`crate::signing::DeviceIdentity`]), never read from a
//! crate-internal default or embedded secret.

use std::fmt;
use std::io::Read;
use std::time::Duration;

use crate::clock::Clock;
use crate::signing::{DeviceIdentity, SignedRequest};
use crate::types::{
    SeedHeartbeatRequest, SeedHeartbeatResponse, SeedRegisterRequest, SeedRegisterResponse,
    SEED_HEARTBEAT_PATH, SEED_REGISTER_PATH,
};

/// Transport or server-side failure sending a [`SignedRequest`].
#[derive(Debug)]
pub enum TransportError {
    /// Failed to establish or complete the HTTP exchange (DNS, connect,
    /// TLS, timeout, ...).
    Http(String),
    /// The exchange completed but the server returned a non-2xx status.
    Status { status: u16, body: String },
    /// The 2xx response body did not parse as the expected JSON shape.
    Json(serde_json::Error),
}

impl fmt::Display for TransportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TransportError::Http(msg) => write!(f, "HTTP transport error: {msg}"),
            TransportError::Status { status, body } => {
                write!(f, "HTTP {status} response: {body}")
            }
            TransportError::Json(err) => write!(f, "failed to parse JSON response: {err}"),
        }
    }
}

impl std::error::Error for TransportError {}

impl From<serde_json::Error> for TransportError {
    fn from(err: serde_json::Error) -> Self {
        TransportError::Json(err)
    }
}

/// A blocking client that executes [`SignedRequest`]s against a
/// caller-supplied base URL.
///
/// `base_url` must be scheme+host only (e.g. `"https://api.cognitum.one"`,
/// with **no** `/v1` path segment). Every request path used by this crate
/// — [`SEED_REGISTER_PATH`], [`SEED_HEARTBEAT_PATH`] — already carries the
/// full `/v1/...` prefix, and that exact string is both what gets signed
/// into the canonical request and what gets appended to `base_url` to form
/// the request URL. Keeping the versioned path in one place (the path
/// constants, not `base_url`) means there is no way for the signed path
/// and the requested path to silently drift apart.
pub struct CognitumHttpClient {
    base_url: String,
    agent: ureq::Agent,
}

impl CognitumHttpClient {
    pub fn new(base_url: impl Into<String>) -> Self {
        let agent = ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(10))
            .build();
        Self {
            base_url: base_url.into(),
            agent,
        }
    }

    fn full_url(&self, path: &str) -> String {
        format!("{}{}", self.base_url.trim_end_matches('/'), path)
    }

    /// Execute an already-signed request and return the raw
    /// `(status, body)` on any completed HTTP exchange, 2xx or not.
    pub fn send(&self, request: &SignedRequest) -> Result<(u16, Vec<u8>), TransportError> {
        let url = self.full_url(&request.path);
        let mut req = self.agent.request(&request.method, &url);
        for (name, value) in &request.headers {
            req = req.set(name, value);
        }
        let result = if request.body.is_empty() {
            req.call()
        } else {
            req.send_bytes(&request.body)
        };
        match result {
            Ok(response) => {
                let status = response.status();
                let mut body = Vec::new();
                response
                    .into_reader()
                    .read_to_end(&mut body)
                    .map_err(|e| TransportError::Http(e.to_string()))?;
                Ok((status, body))
            }
            Err(ureq::Error::Status(status, response)) => {
                let mut body = Vec::new();
                let _ = response.into_reader().read_to_end(&mut body);
                Err(TransportError::Status {
                    status,
                    body: String::from_utf8_lossy(&body).into_owned(),
                })
            }
            Err(ureq::Error::Transport(t)) => Err(TransportError::Http(t.to_string())),
        }
    }

    fn expect_2xx(status: u16, body: Vec<u8>) -> Result<Vec<u8>, TransportError> {
        if (200..300).contains(&status) {
            Ok(body)
        } else {
            Err(TransportError::Status {
                status,
                body: String::from_utf8_lossy(&body).into_owned(),
            })
        }
    }

    /// `POST /v1/seed/register` — unauthenticated per
    /// `openapi/cognitum-api.yaml` (`security: []`), so no
    /// [`DeviceIdentity`]/[`Clock`] is needed to send it.
    pub fn register(
        &self,
        request: &SeedRegisterRequest,
    ) -> Result<SeedRegisterResponse, TransportError> {
        let body = serde_json::to_vec(request)?;
        let signed = SignedRequest {
            method: "POST".to_string(),
            path: SEED_REGISTER_PATH.to_string(),
            headers: vec![("Content-Type".to_string(), "application/json".to_string())],
            body,
        };
        let (status, body) = self.send(&signed)?;
        let body = Self::expect_2xx(status, body)?;
        Ok(serde_json::from_slice(&body)?)
    }

    /// `POST /v1/seed/heartbeat` — signed per
    /// `openapi/cognitum-api.yaml`'s `ed25519` security scheme.
    pub fn heartbeat(
        &self,
        identity: &DeviceIdentity,
        clock: &dyn Clock,
        request: &SeedHeartbeatRequest,
    ) -> Result<SeedHeartbeatResponse, TransportError> {
        let body = serde_json::to_vec(request)?;
        let signed = identity.sign(clock, "POST", SEED_HEARTBEAT_PATH, &body);
        let (status, body) = self.send(&signed)?;
        let body = Self::expect_2xx(status, body)?;
        Ok(serde_json::from_slice(&body)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::FixedClock;
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpListener;
    use std::sync::mpsc;
    use std::thread;

    /// Minimal single-request HTTP/1.1 responder: accepts one connection,
    /// reads the request line + headers + (if `Content-Length` is present)
    /// body, hands the parsed request back over `report`, and writes a
    /// canned JSON response. No external mock-server dependency — just
    /// `std::net`.
    struct RecordedRequest {
        method: String,
        path: String,
        headers: Vec<(String, String)>,
        body: Vec<u8>,
    }

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
                std::io::Read::read_exact(&mut reader, &mut body).expect("read body");
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

    #[test]
    fn register_sends_unsigned_post_with_expected_body() {
        let (base_url, rx) = spawn_mock_server(r#"{"registered":true}"#);
        let client = CognitumHttpClient::new(base_url);

        let req = SeedRegisterRequest {
            device_id: "0193f1e2-7c4a-7000-9b2e-1a2b3c4d5e6f".to_string(),
            public_key: "cHVibGljLWtleQ==".to_string(),
            firmware: None,
        };
        let response = client.register(&req).expect("register succeeds");
        assert_eq!(
            response.extra.get("registered").and_then(|v| v.as_bool()),
            Some(true)
        );

        let recorded = rx.recv().expect("mock server recorded a request");
        assert_eq!(recorded.method, "POST");
        assert_eq!(recorded.path, "/v1/seed/register");
        assert!(!recorded
            .headers
            .iter()
            .any(|(k, _)| k.eq_ignore_ascii_case("X-Device-Signature")));
        let parsed_body: SeedRegisterRequest = serde_json::from_slice(&recorded.body).unwrap();
        assert_eq!(parsed_body, req);
    }

    #[test]
    fn heartbeat_sends_signed_headers_matching_the_canonical_scheme() {
        let (base_url, rx) = spawn_mock_server(r#"{"ack":true}"#);
        let client = CognitumHttpClient::new(base_url);
        let identity = DeviceIdentity::from_signing_key_bytes(
            "0193f1e2-7c4a-7000-9b2e-1a2b3c4d5e6f",
            &[9u8; 32],
        );
        let clock = FixedClock(1_777_562_280);

        let req = SeedHeartbeatRequest {
            uptime_secs: Some(12345),
            ..Default::default()
        };
        let response = client
            .heartbeat(&identity, &clock, &req)
            .expect("heartbeat succeeds");
        assert_eq!(
            response.extra.get("ack").and_then(|v| v.as_bool()),
            Some(true)
        );

        let recorded = rx.recv().expect("mock server recorded a request");
        assert_eq!(recorded.method, "POST");
        assert_eq!(recorded.path, "/v1/seed/heartbeat");

        let header = |name: &str| {
            recorded
                .headers
                .iter()
                .find(|(k, _)| k.eq_ignore_ascii_case(name))
                .map(|(_, v)| v.clone())
        };
        assert_eq!(
            header("X-Device-Id"),
            Some("0193f1e2-7c4a-7000-9b2e-1a2b3c4d5e6f".to_string())
        );
        assert_eq!(
            header("X-Device-Timestamp"),
            Some("2026-04-30T15:18:00Z".to_string())
        );
        let signature_b64 = header("X-Device-Signature").expect("signature header present");

        // The signature on the wire must verify against the canonical
        // string built from what was actually sent (method/path/timestamp
        // /body), independent of the client's own signing path.
        use base64::{engine::general_purpose::STANDARD, Engine as _};
        let sig_bytes = STANDARD
            .decode(signature_b64)
            .expect("valid base64 signature");
        let signature =
            ed25519_dalek::Signature::from_bytes(&sig_bytes.try_into().expect("64-byte signature"));
        let public_key_bytes = STANDARD
            .decode(identity.public_key_base64())
            .expect("valid base64 public key");
        let verifying_key = ed25519_dalek::VerifyingKey::from_bytes(
            &public_key_bytes.try_into().expect("32-byte key"),
        )
        .expect("valid Ed25519 public key");
        let canonical = crate::signing::canonical_string(
            &recorded.method,
            &recorded.path,
            &header("X-Device-Timestamp").unwrap(),
            &recorded.body,
        );
        verifying_key
            .verify_strict(canonical.as_bytes(), &signature)
            .expect("signature on the wire must verify");

        let parsed_body: SeedHeartbeatRequest = serde_json::from_slice(&recorded.body).unwrap();
        assert_eq!(parsed_body, req);
    }

    #[test]
    fn non_2xx_status_is_reported_as_a_status_error() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
        let addr = listener.local_addr().expect("local addr");
        thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            let mut buf = [0u8; 1024];
            let _ = std::io::Read::read(&mut stream, &mut buf);
            let body = r#"{"error":"device not registered"}"#;
            let response = format!(
                "HTTP/1.1 404 Not Found\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            stream
                .write_all(response.as_bytes())
                .expect("write response");
        });

        let client = CognitumHttpClient::new(format!("http://{addr}"));
        let identity = DeviceIdentity::from_signing_key_bytes("device-x", &[3u8; 32]);
        let err = client
            .heartbeat(&identity, &FixedClock(0), &SeedHeartbeatRequest::default())
            .expect_err("404 must surface as an error");
        match err {
            TransportError::Status { status, body } => {
                assert_eq!(status, 404);
                assert!(body.contains("device not registered"));
            }
            other => panic!("expected TransportError::Status, got {other:?}"),
        }
    }
}
