//! Byte-stable golden JSON samples for each of the four `FederationPayload`
//! variants this crate implements, asserted against the field order and tag
//! scheme transcribed from `agentbbs-federation/src/envelope.rs`.
//!
//! Hermetic: no I/O, no network, no subprocess. AgentId/signature hex and
//! chrono's own RFC3339 rendering are captured from the values under test
//! (not hand-guessed), so what's actually being pinned here is agentbbs's
//! wire *shape* — the `"type"` tag values, `snake_case` naming, and struct
//! field order — which is exactly the part of the contract that could
//! silently regress.

use chrono::{DateTime, Utc};
use latentmesh_agentbbs_bridge::wire::{
    AgentId, Board, FederationPayload, Identity, Message, MessageBody, MessageKind,
};

fn fixed_created_at() -> DateTime<Utc> {
    "2026-08-27T12:00:00Z".parse().unwrap()
}

/// The exact JSON fragment chrono produces for `created_at`, captured from
/// the same value under test rather than assumed.
fn json_of<T: serde::Serialize>(value: &T) -> String {
    serde_json::to_string(value).unwrap()
}

#[test]
fn announce_board_golden_shape() {
    let founder = Identity::from_seed(&[7u8; 32]).id();
    let created_at = fixed_created_at();
    let board = Board {
        slug: "air-relay".into(),
        title: "LatentMesh Air Relay".into(),
        description: "Store-and-forward bulletin for an Air stream".into(),
        locked: false,
        founder,
        created_at,
        federated: true,
    };
    let payload = FederationPayload::AnnounceBoard(board);
    let actual = json_of(&payload);

    let expected = format!(
        concat!(
            r#"{{"type":"announce_board","slug":"air-relay","title":"LatentMesh Air Relay","#,
            r#""description":"Store-and-forward bulletin for an Air stream","locked":false,"#,
            r#""founder":"{founder_hex}","created_at":{created_json},"federated":true}}"#
        ),
        founder_hex = founder.to_hex(),
        created_json = json_of(&created_at),
    );
    assert_eq!(actual, expected);
}

#[test]
fn replicate_message_golden_shape() {
    let identity = Identity::from_seed(&[9u8; 32]);
    let created_at = fixed_created_at();
    let body = MessageBody {
        board: "air-relay".into(),
        parent: None,
        subject: "air-delta source=7 epoch=3 msg=99".into(),
        body: "LatentMesh Air state delta\n".into(),
        author: identity.id(),
        handle: "gateway-1".into(),
        created_at,
        kind: MessageKind::Post,
    };
    let message = Message::sign(&identity, body).unwrap();
    let payload = FederationPayload::ReplicateMessage(message.clone());
    let actual = json_of(&payload);

    let expected = format!(
        concat!(
            r#"{{"type":"replicate_message","id":{id_json},"body":{{"board":"air-relay","#,
            r#""parent":null,"subject":"air-delta source=7 epoch=3 msg=99","#,
            r#""body":"LatentMesh Air state delta\n","author":"{author_hex}","#,
            r#""handle":"gateway-1","created_at":{created_json},"kind":"Post"}},"#,
            r#""signature":"{sig_hex}"}}"#
        ),
        id_json = json_of(&message.id),
        author_hex = identity.id().to_hex(),
        created_json = json_of(&created_at),
        sig_hex = message.signature.to_hex(),
    );
    assert_eq!(actual, expected);
}

#[test]
fn peer_hello_golden_shape() {
    let node = Identity::from_seed(&[3u8; 32]).id();
    let payload = FederationPayload::PeerHello {
        node,
        protocol: latentmesh_agentbbs_bridge::PROTOCOL_VERSION.to_string(),
    };
    let actual = json_of(&payload);
    let expected = format!(
        r#"{{"type":"peer_hello","node":"{node_hex}","protocol":"agentbbs/0.1"}}"#,
        node_hex = node.to_hex(),
    );
    assert_eq!(actual, expected);
}

#[test]
fn ack_golden_shape() {
    let payload = FederationPayload::Ack {
        id: "blake3-hex-message-id".into(),
    };
    let actual = json_of(&payload);
    assert_eq!(actual, r#"{"type":"ack","id":"blake3-hex-message-id"}"#);
}

#[test]
fn agent_id_hex_roundtrips_through_json() {
    let id = Identity::generate().id();
    let json = json_of(&id);
    let parsed: AgentId = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, id);
}
