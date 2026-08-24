use latentmesh_air_core::{AirError, FrameFlags, Result, WireProfile, FRAME_MAX_BYTES};

use crate::{BpskConfig, CpfskConfig};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Modulation {
    ByteTransport,
    Bpsk(BpskConfig),
    Cpfsk(CpfskConfig),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinkConfig {
    pub profile: WireProfile,
    /// Complete outer-frame MTU, including the 16-byte framing overhead.
    pub frame_mtu: usize,
    pub flags: FrameFlags,
    pub interleaver_columns: usize,
    pub modulation: Modulation,
}

impl LinkConfig {
    pub fn for_profile(profile: WireProfile) -> Self {
        match profile {
            WireProfile::Wifi => Self {
                profile,
                frame_mtu: 256,
                flags: FrameFlags::NONE,
                interleaver_columns: 1,
                modulation: Modulation::ByteTransport,
            },
            WireProfile::Ble => Self {
                profile,
                // Fits a 128-byte application record; callers may lower this
                // to an actually negotiated ATT payload.
                frame_mtu: 128,
                flags: FrameFlags::NONE,
                interleaver_columns: 1,
                modulation: Modulation::ByteTransport,
            },
            WireProfile::HfBpsk => Self {
                profile,
                frame_mtu: 64,
                flags: FrameFlags::FEC,
                interleaver_columns: 17,
                modulation: Modulation::Bpsk(BpskConfig {
                    samples_per_symbol: 160,
                    amplitude: 0.8,
                }),
            },
            WireProfile::HfAfsk => Self {
                profile,
                frame_mtu: 64,
                flags: FrameFlags::FEC,
                interleaver_columns: 17,
                modulation: Modulation::Cpfsk(CpfskConfig {
                    sample_rate: 8_000,
                    symbol_rate: 100,
                    mark_hz: 1_500,
                    space_hz: 1_700,
                    amplitude: 0.8,
                }),
            },
            WireProfile::VhfAfsk | WireProfile::HamPacket => Self {
                profile,
                frame_mtu: 128,
                flags: FrameFlags::FEC,
                interleaver_columns: 23,
                modulation: Modulation::Cpfsk(CpfskConfig {
                    sample_rate: 48_000,
                    symbol_rate: 1_200,
                    mark_hz: 1_200,
                    space_hz: 2_200,
                    amplitude: 0.8,
                }),
            },
            WireProfile::VhfCpfsk => Self {
                profile,
                frame_mtu: 128,
                flags: FrameFlags::FEC,
                interleaver_columns: 23,
                modulation: Modulation::Cpfsk(CpfskConfig {
                    sample_rate: 48_000,
                    symbol_rate: 2_400,
                    mark_hz: 6_000,
                    space_hz: 4_800,
                    amplitude: 0.8,
                }),
            },
            WireProfile::AmAudio => Self {
                profile,
                frame_mtu: 64,
                flags: FrameFlags::FEC,
                interleaver_columns: 17,
                modulation: Modulation::Cpfsk(CpfskConfig {
                    sample_rate: 8_000,
                    symbol_rate: 100,
                    mark_hz: 1_000,
                    space_hz: 1_300,
                    amplitude: 0.7,
                }),
            },
            WireProfile::FmAudio => Self {
                profile,
                frame_mtu: 128,
                flags: FrameFlags::FEC,
                interleaver_columns: 23,
                modulation: Modulation::Cpfsk(CpfskConfig {
                    sample_rate: 48_000,
                    symbol_rate: 1_200,
                    mark_hz: 1_200,
                    space_hz: 2_200,
                    amplitude: 0.8,
                }),
            },
        }
    }

    pub fn validate(&self) -> Result<()> {
        if !(16..=FRAME_MAX_BYTES).contains(&self.frame_mtu)
            || self.interleaver_columns == 0
            || self.interleaver_columns > 256
        {
            return Err(AirError::InvalidLength);
        }
        if !self.flags.contains(FrameFlags::FEC)
            && !matches!(self.modulation, Modulation::ByteTransport)
        {
            return Err(AirError::InvalidFlags);
        }
        match self.modulation {
            Modulation::ByteTransport => Ok(()),
            Modulation::Bpsk(config) => config.validate(),
            Modulation::Cpfsk(config) => config.validate(),
        }
    }
}
