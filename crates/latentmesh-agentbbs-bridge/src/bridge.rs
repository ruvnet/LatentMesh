//! Decode/re-encode boundary between LatentMesh Air and agentbbs (ADR-020).
//!
//! This module owns exactly one responsibility: turning an already-decoded,
//! already-authenticated `SemanticEnvelope`/`SemanticDelta` (from
//! `latentmesh-air-core`) into agentbbs bulletin content — a `post_message`
//! argument set or a signed `ReplicateMessage` — and turning discovery
//! intent into `AnnounceBoard`/`PeerHello`. It never carries LMS1/LMAD bytes
//! through agentbbs unmodified, and it never needs agentbbs's board,
//! reputation, or moderation model beyond the fields these payloads require.
//!
//! Every function here is pure and synchronous — no network, no subprocess,
//! no clock reads except where a timestamp is an explicit parameter — so the
//! whole module is hermetically unit-testable, matching ADR-020's "fully
//! loopback-testable today" requirement.

use chrono::{DateTime, Utc};
use latentmesh_air_core::{
    SemanticClass, SemanticDelta, SemanticEnvelope, SymbolUpdate, SymbolValue,
};

use crate::wire::{
    AgentId, Board, FederationPayload, Identity, Message, MessageBody, MessageKind,
    PROTOCOL_VERSION,
};
use crate::WireError;

/// Errors mapping Air content into agentbbs bulletin content.
#[derive(Debug, thiserror::Error)]
pub enum BridgeError {
    /// The envelope did not carry a [`SemanticClass::StateDelta`] body, so
    /// there is no delta to decode and republish.
    #[error("unsupported semantic class: {0:?}")]
    UnsupportedClass(SemanticClass),
    /// The Air delta body failed to decode.
    #[error("air delta decode failed: {0}")]
    AirDecode(#[from] latentmesh_air_core::AirError),
    /// Signing/serializing the resulting agentbbs message failed.
    #[error("wire: {0}")]
    Wire(#[from] WireError),
}

/// Render a hex string like agentbbs's own `AgentId`/hash renderings
/// (lowercase, no separator, no `0x` prefix) so bulletin text reads
/// consistently with agentbbs's own conventions.
fn hex_string(bytes: &[u8]) -> String {
    hex::encode(bytes)
}

fn render_symbol_value(value: &SymbolValue) -> String {
    match value {
        SymbolValue::Bool(v) => format!("bool:{v}"),
        SymbolValue::I64(v) => format!("i64:{v}"),
        SymbolValue::U64(v) => format!("u64:{v}"),
        SymbolValue::Q16_16(v) => format!("q16.16:{:.6}", f64::from(*v) / 65536.0),
        SymbolValue::Bytes(v) => format!("bytes:{}", hex_string(v)),
    }
}

/// A bounded, human-readable subject line for a delta's bulletin post.
/// Deterministic given the delta's own fields (no clock/random input).
pub fn delta_subject(delta: &SemanticDelta) -> String {
    format!(
        "air-delta source={} epoch={} msg={}",
        delta.source_id, delta.epoch, delta.message_id
    )
}

/// Render a `SemanticDelta`'s content as plain bulletin text: the
/// provenance fields, one line per symbolic update, and a residual-slot
/// count (residuals themselves are never rendered — they are a learned,
/// noncritical reconstruction aid, not authoritative content; ADR-017 §
/// "Local-scope validation").
pub fn render_delta_text(delta: &SemanticDelta) -> String {
    let mut text = format!(
        "LatentMesh Air state delta\nsource_id={}\nepoch={}\nmessage_id={}\nbase_hash={}\nresult_hash={}\n",
        delta.source_id,
        delta.epoch,
        delta.message_id,
        hex_string(&delta.base_hash),
        hex_string(&delta.result_hash),
    );
    for update in &delta.updates {
        match update {
            SymbolUpdate::Set { field_id, value } => {
                text.push_str(&format!(
                    "  set field={field_id} value={}\n",
                    render_symbol_value(value)
                ));
            }
            SymbolUpdate::Delete { field_id } => {
                text.push_str(&format!("  delete field={field_id}\n"));
            }
        }
    }
    if !delta.residuals.is_empty() {
        text.push_str(&format!(
            "  {} residual slot(s) present (noncritical, not represented in this bulletin)\n",
            delta.residuals.len()
        ));
    }
    text
}

/// Decode a `SemanticEnvelope` into its wrapped `SemanticDelta`, rejecting
/// anything that is not a state delta. `latentmesh-air-core` has already
/// CRC-checked and (if signed) authenticated the envelope before it reaches
/// this bridge — see ADR-020's Decision section.
pub fn decode_delta(envelope: &SemanticEnvelope) -> Result<SemanticDelta, BridgeError> {
    if envelope.class != SemanticClass::StateDelta {
        return Err(BridgeError::UnsupportedClass(envelope.class));
    }
    SemanticDelta::decode(&envelope.body).map_err(BridgeError::AirDecode)
}

/// The `post_message` MCP tool arguments for republishing a decoded delta —
/// `agentbbs-mcp/src/server.rs:154-166`'s `inputSchema` (`board`, `subject`,
/// `text`). This is the "simplest, human-board-facing" publish path ADR-020
/// names.
pub fn delta_to_post_args(board: &str, delta: &SemanticDelta) -> serde_json::Value {
    serde_json::json!({
        "board": board,
        "subject": delta_subject(delta),
        "text": render_delta_text(delta),
    })
}

/// Build the `MessageBody` a `ReplicateMessage`/`post_message` republish of
/// `delta` would carry, signed by `author` under `handle` on `board` at
/// `created_at`. Always `MessageKind::Post` — a republished Air delta is an
/// ordinary bulletin post, not an agent-process milestone/step.
pub fn delta_to_message_body(
    board: &str,
    delta: &SemanticDelta,
    author: AgentId,
    handle: &str,
    created_at: DateTime<Utc>,
) -> MessageBody {
    MessageBody {
        board: board.to_string(),
        parent: None,
        subject: delta_subject(delta),
        body: render_delta_text(delta),
        author,
        handle: handle.to_string(),
        created_at,
        kind: MessageKind::Post,
    }
}

/// Sign a decoded delta into a fully-formed agentbbs `Message`, ready to
/// wrap in `FederationPayload::ReplicateMessage` or to embed in a
/// `post_message` reply for local bookkeeping.
pub fn sign_delta_message(
    identity: &Identity,
    board: &str,
    delta: &SemanticDelta,
    handle: &str,
    created_at: DateTime<Utc>,
) -> Result<Message, BridgeError> {
    let body = delta_to_message_body(board, delta, identity.id(), handle, created_at);
    Message::sign(identity, body).map_err(BridgeError::from)
}

/// The "more federation-native path" ADR-020 names: wrap a signed,
/// republished delta message in `FederationPayload::ReplicateMessage` for
/// transport through a `Transport`-trait-shaped sink (see [`crate::transport`]).
pub fn delta_to_replicate_payload(
    identity: &Identity,
    board: &str,
    delta: &SemanticDelta,
    handle: &str,
    created_at: DateTime<Utc>,
) -> Result<FederationPayload, BridgeError> {
    Ok(FederationPayload::ReplicateMessage(sign_delta_message(
        identity, board, delta, handle, created_at,
    )?))
}

/// Decode `envelope` directly into a `post_message` argument set. The
/// convenience composition of [`decode_delta`] + [`delta_to_post_args`] for
/// the common case: one Air envelope in, one MCP tool call's arguments out.
pub fn envelope_to_post_args(
    board: &str,
    envelope: &SemanticEnvelope,
) -> Result<serde_json::Value, BridgeError> {
    let delta = decode_delta(envelope)?;
    Ok(delta_to_post_args(board, &delta))
}

/// Decode `envelope` directly into a signed `ReplicateMessage` payload. The
/// convenience composition of [`decode_delta`] + [`delta_to_replicate_payload`].
pub fn envelope_to_replicate_payload(
    identity: &Identity,
    board: &str,
    envelope: &SemanticEnvelope,
    handle: &str,
    created_at: DateTime<Utc>,
) -> Result<FederationPayload, BridgeError> {
    let delta = decode_delta(envelope)?;
    delta_to_replicate_payload(identity, board, &delta, handle, created_at)
}

/// Build the `Board` metadata for "this LatentMesh Air stream/node exists,
/// here is its current state-hash checkpoint" (ADR-020's discovery path).
/// `state_hash` is typically a `SemanticEnvelope::state_hash` or a
/// `CriticalState::critical_hash()`.
pub fn air_stream_board(
    slug: &str,
    title: &str,
    founder: AgentId,
    state_hash: &[u8; 16],
    created_at: DateTime<Utc>,
) -> Board {
    Board {
        slug: slug.to_string(),
        title: title.to_string(),
        description: format!(
            "LatentMesh Air stream checkpoint state_hash={}",
            hex_string(state_hash)
        ),
        locked: false,
        founder,
        created_at,
        federated: true,
    }
}

/// `AnnounceBoard` — advertise a LatentMesh Air stream to agentbbs without
/// requiring the peer to hold radio hardware (ADR-020's discovery path,
/// mapping ruflo's `federation_bbs_register` verb).
pub fn announce_board(board: Board) -> FederationPayload {
    FederationPayload::AnnounceBoard(board)
}

/// `PeerHello` — introduce this gateway node on link-up, speaking the
/// federation wire's own protocol version string (not the MCP handshake
/// version, which lives in [`crate::wire::MCP_PROTOCOL_VERSION`]).
pub fn peer_hello(node: AgentId) -> FederationPayload {
    FederationPayload::PeerHello {
        node,
        protocol: PROTOCOL_VERSION.to_string(),
    }
}

/// `Ack` — acknowledge a previously-seen envelope or message id.
pub fn ack(id: impl Into<String>) -> FederationPayload {
    FederationPayload::Ack { id: id.into() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use latentmesh_air_core::CriticalState;

    fn sample_delta() -> SemanticDelta {
        let mut before = CriticalState::new();
        before.set(1, SymbolValue::Bool(true)).unwrap();
        before.set(2, SymbolValue::U64(827)).unwrap();
        let mut after = before.clone();
        after.set(2, SymbolValue::U64(828)).unwrap();
        after.set(3, SymbolValue::Q16_16(2 << 16)).unwrap();
        SemanticDelta::between(7, 3, 99, &before, &after, Vec::new()).unwrap()
    }

    #[test]
    fn subject_is_deterministic_and_bounded() {
        let delta = sample_delta();
        let subject = delta_subject(&delta);
        assert_eq!(subject, delta_subject(&delta));
        assert!(subject.len() < 128);
    }

    #[test]
    fn render_text_lists_every_update_and_no_residual_values() {
        let delta = sample_delta();
        let text = render_delta_text(&delta);
        assert!(text.contains("set field=2 value=u64:828"));
        assert!(text.contains("set field=3 value=q16.16:2.000000"));
        // No residuals in this fixture; assert the section is absent.
        assert!(!text.contains("residual slot"));
    }

    #[test]
    fn delta_to_post_args_shape() {
        let delta = sample_delta();
        let args = delta_to_post_args("air-relay", &delta);
        assert_eq!(args["board"], "air-relay");
        assert_eq!(args["subject"], delta_subject(&delta));
        assert!(args["text"].as_str().unwrap().contains("source_id=7"));
    }

    #[test]
    fn sign_delta_message_round_trips_and_verifies() {
        let identity = Identity::generate();
        let delta = sample_delta();
        let created_at: DateTime<Utc> = "2026-08-27T00:00:00Z".parse().unwrap();
        let message =
            sign_delta_message(&identity, "air-relay", &delta, "gateway-1", created_at).unwrap();
        assert!(message.verify().is_ok());
        assert_eq!(message.body.author, identity.id());
        assert_eq!(message.body.subject, delta_subject(&delta));
    }

    #[test]
    fn replicate_payload_wraps_a_verifiable_message() {
        let identity = Identity::generate();
        let delta = sample_delta();
        let created_at: DateTime<Utc> = "2026-08-27T00:00:00Z".parse().unwrap();
        let payload =
            delta_to_replicate_payload(&identity, "air-relay", &delta, "gateway-1", created_at)
                .unwrap();
        match payload {
            FederationPayload::ReplicateMessage(message) => assert!(message.verify().is_ok()),
            other => panic!("expected ReplicateMessage, got {other:?}"),
        }
    }

    #[test]
    fn peer_hello_and_ack_shapes() {
        let identity = Identity::generate();
        match peer_hello(identity.id()) {
            FederationPayload::PeerHello { node, protocol } => {
                assert_eq!(node, identity.id());
                assert_eq!(protocol, PROTOCOL_VERSION);
            }
            other => panic!("expected PeerHello, got {other:?}"),
        }
        match ack("msg-42") {
            FederationPayload::Ack { id } => assert_eq!(id, "msg-42"),
            other => panic!("expected Ack, got {other:?}"),
        }
    }
}
