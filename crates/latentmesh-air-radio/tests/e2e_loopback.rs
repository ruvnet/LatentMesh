use latentmesh_air_core::{CriticalState, SemanticDelta, SymbolValue, WireProfile};
use latentmesh_air_radio::{
    AirReceiver, AirTransmitter, AudioChannel, ChannelConfig, IqChannel, LinkConfig,
    ReceiverConfig, Transmission, TransmitterConfig,
};

fn delta() -> (CriticalState, CriticalState, SemanticDelta) {
    let mut base = CriticalState::new();
    base.set(1, SymbolValue::U64(827)).unwrap();
    // Latitude in deterministic microdegrees; avoid floating point on wire.
    base.set(2, SymbolValue::I64(43_412_000)).unwrap();
    let mut target = base.clone();
    target.set(3, SymbolValue::Q16_16(71 << 16)).unwrap();
    target.set(4, SymbolValue::Q16_16(22 << 16)).unwrap();
    let delta = SemanticDelta::between(42, 7, 9, &base, &target, vec![]).unwrap();
    (base, target, delta)
}

#[test]
fn hf_bpsk_fec_noisy_iq_loopback_preserves_exact_critical_state() {
    let (base, target, delta) = delta();
    let link = LinkConfig::for_profile(WireProfile::HfBpsk);
    let mut tx = AirTransmitter::new(TransmitterConfig::for_link(link)).unwrap();
    let mut rx = AirReceiver::new(ReceiverConfig::for_link(link)).unwrap();
    let mut channel = IqChannel::new(ChannelConfig {
        snr_db: 10.0,
        seed: 0x1234,
        ..ChannelConfig::default()
    })
    .unwrap();
    tx.enqueue_delta(3, &delta, 15).unwrap();
    let mut complete = None;
    while let Some(transmission) = tx.next_transmission().unwrap() {
        let Transmission::Iq { samples, .. } = transmission else {
            panic!("expected IQ burst");
        };
        complete = rx
            .ingest_iq_burst(&channel.process(&samples))
            .unwrap()
            .or(complete);
    }
    let message = complete.expect("reassembled semantic delta");
    assert_eq!(message.apply_delta(&base).unwrap(), target);
    assert_eq!(
        message.apply_delta(&base).unwrap().critical_hash(),
        target.critical_hash()
    );
}

#[test]
fn vhf_afsk_fec_noisy_pcm_loopback_preserves_exact_critical_state() {
    let (base, target, delta) = delta();
    let link = LinkConfig::for_profile(WireProfile::VhfAfsk);
    let mut tx = AirTransmitter::new(TransmitterConfig::for_link(link)).unwrap();
    let mut rx = AirReceiver::new(ReceiverConfig::for_link(link)).unwrap();
    let mut channel = AudioChannel::new(ChannelConfig {
        snr_db: 16.0,
        seed: 0x5678,
        ..ChannelConfig::default()
    })
    .unwrap();
    tx.enqueue_delta(5, &delta, 15).unwrap();
    let mut complete = None;
    while let Some(transmission) = tx.next_transmission().unwrap() {
        let Transmission::Pcm { samples, .. } = transmission else {
            panic!("expected PCM burst");
        };
        complete = rx
            .ingest_pcm_burst(&channel.process(&samples))
            .unwrap()
            .or(complete);
    }
    assert_eq!(complete.unwrap().apply_delta(&base).unwrap(), target);
}
