//! `latentmesh-cognitum-client` — a client for `cognitum-one/api`'s cloud
//! fleet `/v1/seed/*` surface, implementing ADR-021.
//!
//! # Two integration surfaces — this crate is only one of them
//!
//! ADR-021 records **two distinct** cognitum integration surfaces and is
//! explicit that they must never be conflated:
//!
//! | | Cloud fleet API (this crate) | Local V0 appliance |
//! |---|---|---|
//! | Host | `api.cognitum.one` | co-located appliance |
//! | Routes | `/v1/seed/register`, `/v1/seed/heartbeat`, ... | `/api/v1/v0/*` |
//! | Auth | Ed25519 device keypair, canonical-request signing | bearer token, port 9000 |
//! | Discovery | HTTPS DNS | UDP port 5008 |
//!
//! **This crate implements only the cloud fleet surface.** It does not, and
//! will not, implement a client for the local V0 appliance's
//! `/api/v1/v0/*` gateway — that is explicitly out of scope per ADR-021's
//! Decision section and would need its own crate with its own (bearer)
//! auth model if a co-located deployment shape is chosen later.
//!
//! # Feature gating (mirrors ADR-016's pattern)
//!
//! - **Default build (`default = []`)**: no network dependency at all.
//!   Canonical-request construction ([`signing`]), Ed25519 signing, and the
//!   request/response payload types ([`types`]) are all here and fully
//!   offline-testable — `cargo test -p latentmesh-cognitum-client` needs no
//!   network access.
//! - **`http` feature**: adds [`http::CognitumHttpClient`], a thin blocking
//!   `ureq`-based transport that executes a [`signing::SignedRequest`]
//!   against a caller-supplied base URL.
//!
//! # What this crate does NOT claim
//!
//! No credential for a real cognitum device exists on this host. Every test
//! in this crate runs against either a fixed in-test keypair/clock or a
//! same-process mock HTTP server (`std::net::TcpListener`) — nothing here
//! has ever made a network call to `api.cognitum.one`, and nothing in this
//! crate hardcodes or defaults to that hostname (see
//! [`http::CognitumHttpClient::new`]). A live `POST /v1/seed/register`
//! against production remains credential-pending, exactly as ADR-021's
//! "What is simulated / what is hardware-pending" table states.
//!
//! # A source discrepancy worth recording
//!
//! `cognitum-one/api`'s `functions/seed/index.js` (its `verifyDeviceAuth`)
//! implements a *different* signing scheme than the one this crate follows:
//! hex-encoded keys/signatures, headers `X-Timestamp`/`X-Signature` (not
//! `X-Device-Timestamp`/`X-Device-Signature`), a no-newline-separated
//! `${method}${path}${body}${timestamp}` message, and no replay-window
//! check. That repository's own `README.md` labels `functions/seed/` a
//! "non-canonical reference fork; do not deploy to production" — the
//! canonical contract is `openapi/cognitum-api.yaml`'s
//! `components.securitySchemes.ed25519` plus `docs/seed-integration.md`,
//! which this crate implements (see [`signing`] for the full citation).
//! Worth knowing if a future integrator reads that JS file and is
//! surprised it doesn't match.

pub mod clock;
pub mod signing;
pub mod timestamp;
pub mod types;

#[cfg(feature = "http")]
pub mod http;

pub use clock::{Clock, FixedClock, SystemClock, MAX_CLOCK_SKEW_SECS, REPLAY_WINDOW_SECS};
pub use signing::{canonical_string, DeviceIdentity, SignedRequest};
pub use timestamp::unix_to_iso8601_utc;
pub use types::{
    SeedHeartbeatRequest, SeedHeartbeatResponse, SeedRegisterRequest, SeedRegisterResponse,
    SEED_HEARTBEAT_PATH, SEED_REGISTER_PATH,
};

#[cfg(feature = "http")]
pub use http::{CognitumHttpClient, TransportError};
