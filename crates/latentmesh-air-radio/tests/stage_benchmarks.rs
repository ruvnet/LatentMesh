//! Deterministic stage gates. These are engineering simulations, not claims
//! about over-the-air performance on unseen hardware or propagation paths.

use latentmesh_air_core::{CriticalState, SemanticDelta, SymbolValue, WireProfile};
use latentmesh_air_radio::{
    AirReceiver, AirTransmitter, AssistObservation, LinkConfig, ReceiverConfig, TinyNeuralAssist,
    Transmission, TransmitterConfig,
};

#[test]
fn stage_one_semantic_transport_exceeds_ten_x_at_exact_task_state() {
    let base = CriticalState::new();
    let mut target = CriticalState::new();
    target.set(1, SymbolValue::U64(827)).unwrap();
    target.set(2, SymbolValue::I64(43_412_000)).unwrap();
    target.set(3, SymbolValue::Q16_16(71 << 16)).unwrap();
    target.set(4, SymbolValue::Q16_16(22 << 16)).unwrap();
    target.set(5, SymbolValue::Bool(true)).unwrap();
    target.set(6, SymbolValue::U64(970_000)).unwrap();
    let delta = SemanticDelta::between(5, 1, 1, &base, &target, vec![]).unwrap();

    let link = LinkConfig::for_profile(WireProfile::Wifi);
    let mut tx = AirTransmitter::new(TransmitterConfig::for_link(link)).unwrap();
    let mut rx = AirReceiver::new(ReceiverConfig::for_link(link)).unwrap();
    tx.enqueue_delta(1, &delta, 15).unwrap();
    let mut transmitted = 0_usize;
    let mut received = None;
    while let Some(item) = tx.next_transmission().unwrap() {
        let Transmission::Bytes { bytes, .. } = item else {
            panic!("expected byte transport");
        };
        transmitted += bytes.len();
        received = rx.ingest_frame_bytes(&bytes).unwrap().or(received);
    }

    let reconstructed = received.unwrap().apply_delta(&base).unwrap();
    assert_eq!(reconstructed, target);
    assert_eq!(reconstructed.critical_hash(), target.critical_hash());

    // Current LatentMesh reference: 16 x 4096 Int8 scalars, excluding its
    // metadata. The sparse result includes LMS1, LMAD, outer headers and CRC.
    let dense_latent_bytes = 16 * 4_096;
    let compression_ratio = dense_latent_bytes as f64 / transmitted as f64;
    eprintln!(
        "stage1 dense={} sparse={} ratio={compression_ratio:.2}x exact_hash=true",
        dense_latent_bytes, transmitted
    );
    assert!(compression_ratio >= 10.0);
}

#[test]
fn stage_two_neural_assist_exceeds_two_x_in_simulated_phase_inversion() {
    let mut assist = TinyNeuralAssist::default();
    for epoch in 0..160 {
        for index in 0..64 {
            let expected = ((index * 17 + epoch * 13) & 1) as u8;
            let classically_correct = index % 16 == 0;
            let sign = if expected == 1 { 1 } else { -1 };
            let classical = if classically_correct {
                sign * 40
            } else {
                -sign * 40
            };
            let observation = AssistObservation {
                energy: 0.32,
                noise: 0.68,
                // A coarse synchronizer reports the high phase/frequency-error
                // regime; no expected bit is present in this feature.
                frequency_error: 0.75,
            };
            assist
                .train_verified(classical, observation, expected, 0.015)
                .unwrap();
        }
    }

    let mut classical_correct = 0_usize;
    let mut assisted_correct = 0_usize;
    let symbols = 512_usize;
    for index in 0..symbols {
        let expected = ((index * 29 + 7) & 1) as u8;
        let raw_correct = index % 8 == 0;
        let sign = if expected == 1 { 1 } else { -1 };
        let classical = if raw_correct { sign * 40 } else { -sign * 40 };
        let observation = AssistObservation {
            energy: 0.32,
            noise: 0.68,
            frequency_error: 0.75,
        };
        classical_correct += usize::from((classical > 0) == (expected == 1));
        let decision = assist.refine(classical, observation);
        assisted_correct += usize::from((decision.llr > 0) == (expected == 1));
    }
    // Airtime and sample energy are identical, so useful correct bits are the
    // useful-information-per-airtime ratio for this controlled simulation.
    let gain = assisted_correct as f64 / classical_correct as f64;
    eprintln!(
        "stage2 classical={classical_correct}/{symbols} assisted={assisted_correct}/{symbols} gain={gain:.2}x simulated=true"
    );
    assert!(gain >= 2.0);
}
