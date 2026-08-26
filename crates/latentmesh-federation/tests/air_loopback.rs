//! End-to-end federation over the Air stack (ADR-017): a rule leaves node A
//! as an LMS1 Control envelope, crosses the radio path (byte transport and
//! the BPSK IQ modem loopback), survives replay defense, passes local-scope
//! validation at node B, and lands in B's world model. Evidence label:
//! software loopback — no over-the-air claim.

use latentmesh_air_core::{SemanticEnvelope, WireProfile};
use latentmesh_air_radio::{
    AirReceiver, AirTransmitter, ChannelConfig, IqChannel, LinkConfig, ReceiverConfig,
    Transmission, TransmitterConfig,
};
use latentmesh_federation::{
    decode_rule_envelope, encode_rule_envelope, validate_candidate, AdmissionConfig, RuleScope,
    RuleVerdict, Transition, TransitionRule, WorldModel,
};

fn rule() -> TransitionRule {
    TransitionRule {
        pre: "hot".into(),
        action: "cool".into(),
        post: "warm".into(),
        support: 24,
        confidence: 0.9,
        scope: RuleScope::Global,
    }
}

fn receiver_model() -> WorldModel {
    let mut model = WorldModel::new();
    for i in 0..24 {
        model.record_holdout(Transition {
            pre: "hot".into(),
            action: "cool".into(),
            post: "warm".into(),
        });
        model.record_holdout(Transition {
            pre: format!("s{i}"),
            action: "noop".into(),
            post: format!("s{i}"),
        });
    }
    model
}

fn send_and_receive(link: LinkConfig, envelope: &SemanticEnvelope) -> SemanticEnvelope {
    let mut tx = AirTransmitter::new(TransmitterConfig::for_link(link)).unwrap();
    let mut rx = AirReceiver::new(ReceiverConfig::for_link(link)).unwrap();
    let message_bytes = envelope.encode().unwrap();
    let state_tag = u16::from_be_bytes([envelope.state_hash[0], envelope.state_hash[1]]);
    tx.enqueue_message(
        1,
        envelope.class,
        envelope.priority,
        state_tag,
        &message_bytes,
    )
    .unwrap();

    let mut channel = IqChannel::new(ChannelConfig::default()).unwrap();
    let mut received = None;
    while let Some(transmission) = tx.next_transmission().unwrap() {
        let out = match transmission {
            Transmission::Bytes { bytes, .. } => rx.ingest_frame_bytes(&bytes).unwrap(),
            Transmission::Iq { samples, .. } => {
                let impaired = channel.process(&samples);
                rx.ingest_iq_burst(&impaired).unwrap()
            }
            Transmission::Pcm { .. } => panic!("unexpected PCM for this profile"),
        };
        if let Some(message) = out {
            received = Some(message);
        }
    }
    let message = received.expect("rule message arrives");
    message.semantic_envelope().unwrap()
}

fn federate_over(link: LinkConfig) {
    let envelope = encode_rule_envelope(&rule(), 7, 1, 100, 1, 8, None).unwrap();
    let received = send_and_receive(link, &envelope);
    let candidate = decode_rule_envelope(&received).unwrap();
    assert_eq!(candidate, rule());

    let model = receiver_model();
    match validate_candidate(&model, &candidate, 0, &AdmissionConfig::default()) {
        RuleVerdict::Admit { gain, .. } => {
            assert!(gain > 0.0);
            let mut model = model;
            model.install(candidate);
            assert_eq!(model.predict("hot", "cool"), Some("warm"));
        }
        RuleVerdict::Reject { control, reason } => {
            panic!("expected admission after transport, rejected on {control}: {reason}")
        }
    }
}

#[test]
fn rule_federates_over_wifi_byte_transport() {
    federate_over(LinkConfig::for_profile(WireProfile::Wifi));
}

#[test]
fn rule_federates_over_hf_bpsk_modem_loopback() {
    federate_over(LinkConfig::for_profile(WireProfile::HfBpsk));
}

#[test]
fn replayed_rule_frames_are_rejected_by_the_air_replay_window() {
    let link = LinkConfig::for_profile(WireProfile::Wifi);
    let envelope = encode_rule_envelope(&rule(), 7, 1, 100, 1, 8, None).unwrap();
    let mut tx = AirTransmitter::new(TransmitterConfig::for_link(link)).unwrap();
    let mut rx = AirReceiver::new(ReceiverConfig::for_link(link)).unwrap();
    let bytes = envelope.encode().unwrap();
    let state_tag = u16::from_be_bytes([envelope.state_hash[0], envelope.state_hash[1]]);
    tx.enqueue_message(1, envelope.class, envelope.priority, state_tag, &bytes)
        .unwrap();

    let mut wire_frames = Vec::new();
    while let Some(t) = tx.next_transmission().unwrap() {
        if let Transmission::Bytes { bytes, .. } = t {
            wire_frames.push(bytes);
        }
    }
    // First delivery succeeds…
    let mut delivered = false;
    for frame in &wire_frames {
        if rx.ingest_frame_bytes(frame).unwrap().is_some() {
            delivered = true;
        }
    }
    assert!(delivered);
    // …replaying the same frames is refused.
    for frame in &wire_frames {
        assert!(rx.ingest_frame_bytes(frame).is_err());
    }
}
