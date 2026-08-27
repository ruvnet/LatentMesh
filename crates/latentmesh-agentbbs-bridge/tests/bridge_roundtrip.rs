//! End-to-end bridge round trip, exercised only through the crate's public
//! API (as an external caller would use it): an Air `SemanticDelta` goes in,
//! a signed agentbbs `ReplicateMessage` payload comes out over a simulated
//! in-memory transport, and the test verifies both the Ed25519 signature and
//! the rendered content. Loopback/simulation only — no live agentbbs peer,
//! per ADR-020's "What is simulated / what is hardware-pending" table.

use chrono::{DateTime, Utc};
use latentmesh_agentbbs_bridge::bridge::{self, delta_subject};
use latentmesh_agentbbs_bridge::transport::{FederationPublisher, InMemoryPeer};
use latentmesh_agentbbs_bridge::wire::{FederationPayload, Identity};
use latentmesh_air_core::{CriticalState, SymbolValue};

fn sample_delta() -> latentmesh_air_core::SemanticDelta {
    let mut before = CriticalState::new();
    before.set(10, SymbolValue::U64(1)).unwrap();
    let mut after = before.clone();
    after.set(10, SymbolValue::U64(2)).unwrap();
    after.set(11, SymbolValue::Bool(true)).unwrap();
    latentmesh_air_core::SemanticDelta::between(42, 1, 7, &before, &after, Vec::new()).unwrap()
}

#[test]
fn air_delta_in_bbs_message_out_over_simulated_transport() {
    let identity = Identity::generate();
    let delta = sample_delta();
    let created_at: DateTime<Utc> = "2026-08-27T09:30:00Z".parse().unwrap();

    // Publish path: SemanticDelta -> signed ReplicateMessage -> sealed
    // FederationEnvelope -> wire bytes handed to the sink.
    let payload =
        bridge::delta_to_replicate_payload(&identity, "air-relay", &delta, "gateway-1", created_at)
            .unwrap();

    let mut publisher = FederationPublisher::new(&identity, InMemoryPeer::new());
    let sealed = publisher.publish(payload).unwrap();
    assert_eq!(sealed.node, identity.id());
    assert_eq!(sealed.seq, 1);

    // Receive path: parse the wire bytes back, verify the envelope
    // signature, and confirm the message content and its own signature.
    let sink = publisher.into_sink();
    assert_eq!(sink.received.len(), 1);
    let opened = sink.opened_payloads().unwrap();
    assert_eq!(opened.len(), 1);

    match &opened[0] {
        FederationPayload::ReplicateMessage(message) => {
            assert!(message.verify().is_ok());
            assert_eq!(message.body.board, "air-relay");
            assert_eq!(message.body.subject, delta_subject(&delta));
            assert_eq!(message.body.author, identity.id());
            assert!(message.body.body.contains("source_id=42"));
            assert!(message.body.body.contains("set field=10 value=u64:2"));
            assert!(message.body.body.contains("set field=11 value=bool:true"));
        }
        other => panic!("expected ReplicateMessage, got {other:?}"),
    }
}

#[test]
fn post_message_args_path_matches_replicate_message_content() {
    // The two publish paths ADR-020 names (post_message MCP call vs.
    // ReplicateMessage federation payload) must render identical bulletin
    // content for the same delta, since they are two transports for the
    // same decoded fact.
    let delta = sample_delta();
    let args = bridge::delta_to_post_args("air-relay", &delta);

    let identity = Identity::generate();
    let created_at: DateTime<Utc> = "2026-08-27T09:30:00Z".parse().unwrap();
    let message =
        bridge::sign_delta_message(&identity, "air-relay", &delta, "gw", created_at).unwrap();

    assert_eq!(args["board"], "air-relay");
    assert_eq!(args["subject"], message.body.subject);
    assert_eq!(args["text"], message.body.body);
}

#[test]
fn discovery_path_announce_and_hello() {
    let identity = Identity::generate();
    let created_at: DateTime<Utc> = "2026-08-27T09:30:00Z".parse().unwrap();
    let state_hash = [0xAB_u8; 16];
    let board = bridge::air_stream_board(
        "air-relay",
        "LatentMesh Air Relay",
        identity.id(),
        &state_hash,
        created_at,
    );
    assert!(board.federated);
    assert!(board.description.contains(&hex::encode(state_hash)));

    let mut publisher = FederationPublisher::new(&identity, InMemoryPeer::new());
    publisher.publish(bridge::announce_board(board)).unwrap();
    publisher
        .publish(bridge::peer_hello(identity.id()))
        .unwrap();

    let sink = publisher.into_sink();
    let opened = sink.opened_payloads().unwrap();
    assert!(matches!(opened[0], FederationPayload::AnnounceBoard(_)));
    assert!(matches!(opened[1], FederationPayload::PeerHello { .. }));
}
