use alloc::vec::Vec;

use latentmesh_air_core::{AirError, Result};

use crate::IqSample;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChannelConfig {
    pub snr_db: f32,
    pub gain: f32,
    /// Complex-baseband frequency offset in cycles per sample.
    pub frequency_offset: f32,
    /// Optional deterministic erasure period in samples.
    pub erasure_period: Option<u32>,
    pub erasure_samples: u16,
    pub seed: u64,
}

impl Default for ChannelConfig {
    fn default() -> Self {
        Self {
            snr_db: 30.0,
            gain: 1.0,
            frequency_offset: 0.0,
            erasure_period: None,
            erasure_samples: 0,
            seed: 0x4c4d_4149_522d_7631,
        }
    }
}

impl ChannelConfig {
    pub fn validate(&self) -> Result<()> {
        if !self.snr_db.is_finite()
            || !(-20.0..=80.0).contains(&self.snr_db)
            || !self.gain.is_finite()
            || !(0.0..=4.0).contains(&self.gain)
            || !self.frequency_offset.is_finite()
            || !(-0.25..=0.25).contains(&self.frequency_offset)
            || self.seed == 0
        {
            return Err(AirError::InvalidLength);
        }
        match self.erasure_period {
            Some(period) if period == 0 || u32::from(self.erasure_samples) > period => {
                Err(AirError::InvalidLength)
            }
            None if self.erasure_samples != 0 => Err(AirError::InvalidLength),
            _ => Ok(()),
        }
    }
}

#[derive(Clone, Debug)]
pub struct IqChannel {
    config: ChannelConfig,
    rng: XorShift64,
    phase: f32,
    sample_index: u64,
}

impl IqChannel {
    pub fn new(config: ChannelConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            rng: XorShift64(config.seed),
            config,
            phase: 0.0,
            sample_index: 0,
        })
    }

    pub fn process(&mut self, input: &[IqSample]) -> Vec<IqSample> {
        let sigma = noise_sigma(self.config.snr_db, self.config.gain);
        let phase_step = 2.0 * core::f32::consts::PI * self.config.frequency_offset;
        let mut output = Vec::with_capacity(input.len());
        for sample in input {
            let erased = is_erased(self.config, self.sample_index);
            let (i, q) = if erased {
                (0.0, 0.0)
            } else {
                let cosine = libm::cosf(self.phase);
                let sine = libm::sinf(self.phase);
                (
                    self.config.gain * (sample.i * cosine - sample.q * sine),
                    self.config.gain * (sample.i * sine + sample.q * cosine),
                )
            };
            output.push(IqSample {
                i: i + sigma * self.rng.normalish(),
                q: q + sigma * self.rng.normalish(),
            });
            self.phase += phase_step;
            if self.phase > core::f32::consts::PI {
                self.phase -= 2.0 * core::f32::consts::PI;
            } else if self.phase < -core::f32::consts::PI {
                self.phase += 2.0 * core::f32::consts::PI;
            }
            self.sample_index = self.sample_index.wrapping_add(1);
        }
        output
    }
}

#[derive(Clone, Debug)]
pub struct AudioChannel {
    config: ChannelConfig,
    rng: XorShift64,
    sample_index: u64,
}

impl AudioChannel {
    pub fn new(config: ChannelConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            rng: XorShift64(config.seed),
            config,
            sample_index: 0,
        })
    }

    pub fn process(&mut self, input: &[i16]) -> Vec<i16> {
        let sigma = noise_sigma(self.config.snr_db, self.config.gain);
        let mut output = Vec::with_capacity(input.len());
        for sample in input {
            let signal = if is_erased(self.config, self.sample_index) {
                0.0
            } else {
                (f32::from(*sample) / i16::MAX as f32) * self.config.gain
            };
            let value = (signal + sigma * self.rng.normalish()).clamp(-1.0, 1.0);
            output.push((value * i16::MAX as f32) as i16);
            self.sample_index = self.sample_index.wrapping_add(1);
        }
        output
    }
}

fn noise_sigma(snr_db: f32, gain: f32) -> f32 {
    let linear = libm::powf(10.0, snr_db / 10.0);
    gain / libm::sqrtf(2.0 * linear)
}

fn is_erased(config: ChannelConfig, sample_index: u64) -> bool {
    config
        .erasure_period
        .is_some_and(|period| sample_index % u64::from(period) < u64::from(config.erasure_samples))
}

#[derive(Clone, Copy, Debug)]
struct XorShift64(u64);

impl XorShift64 {
    fn next(&mut self) -> u64 {
        let mut value = self.0;
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        self.0 = value;
        value
    }

    fn uniform(&mut self) -> f32 {
        let mantissa = (self.next() >> 40) as u32;
        mantissa as f32 / 16_777_216.0
    }

    /// Irwin-Hall approximation, sufficient for deterministic channel tests.
    fn normalish(&mut self) -> f32 {
        (0..12).fold(-6.0, |sum, _| sum + self.uniform())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channel_is_reproducible_for_seed() {
        let config = ChannelConfig {
            snr_db: 8.0,
            seed: 7,
            ..ChannelConfig::default()
        };
        let samples = [IqSample { i: 1.0, q: 0.0 }; 8];
        let first = IqChannel::new(config).unwrap().process(&samples);
        let second = IqChannel::new(config).unwrap().process(&samples);
        assert_eq!(first, second);
    }
}
