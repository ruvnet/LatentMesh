//! Transport-independent transmitter/receiver state machines and small,
//! portable reference modems for real PCM and complex baseband pipes.
//!
//! RF tuning, regulatory compliance, PTT, filtering and hardware drivers stay
//! outside this crate. In particular, an ESP32 can run the byte/codec path but
//! requires an external transceiver for HF or VHF RF.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

mod assist;
mod channel;
mod config;
mod modem;
mod phy;
mod receiver;
mod transmitter;

pub use assist::{AssistDecision, AssistObservation, TinyNeuralAssist};
pub use channel::{AudioChannel, ChannelConfig, IqChannel};
pub use config::{LinkConfig, Modulation};
pub use modem::{BpskConfig, BpskModem, CpfskConfig, CpfskModem, IqSample};
pub use phy::{BurstCodec, SoftBurstReceiver, SoftBurstState, PHY_SYNC};
pub use receiver::{AirReceiver, EnvelopeVerifier, ReceivedMessage, ReceiverConfig, ReceiverState};
pub use transmitter::{AirTransmitter, Transmission, TransmitterConfig, TransmitterState};
