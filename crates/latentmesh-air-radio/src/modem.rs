use alloc::vec::Vec;

use latentmesh_air_core::{AirError, Result, MAX_CODED_BITS};

const MAX_PHY_BITS: usize = MAX_CODED_BITS + 80;
const MAX_WAVEFORM_SAMPLES: usize = 1_000_000;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct IqSample {
    pub i: f32,
    pub q: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BpskConfig {
    pub samples_per_symbol: u16,
    pub amplitude: f32,
}

impl BpskConfig {
    pub fn validate(&self) -> Result<()> {
        if self.samples_per_symbol == 0
            || self.samples_per_symbol > 4_096
            || !self.amplitude.is_finite()
            || self.amplitude <= 0.0
            || self.amplitude > 1.0
        {
            return Err(AirError::InvalidLength);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BpskModem {
    config: BpskConfig,
}

impl BpskModem {
    pub fn new(config: BpskConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    pub const fn config(&self) -> BpskConfig {
        self.config
    }

    /// Complex baseband BPSK. The caller supplies pulse shaping and RF mixing
    /// when required by its radio hardware.
    pub fn modulate(&self, bits: &[u8]) -> Result<Vec<IqSample>> {
        if bits.len() > MAX_PHY_BITS || bits.iter().any(|bit| *bit > 1) {
            return Err(AirError::InvalidLength);
        }
        let count = bits
            .len()
            .checked_mul(usize::from(self.config.samples_per_symbol))
            .ok_or(AirError::LimitExceeded)?;
        if count > MAX_WAVEFORM_SAMPLES {
            return Err(AirError::LimitExceeded);
        }
        let mut samples = Vec::with_capacity(count);
        for bit in bits {
            let level = if *bit == 1 {
                self.config.amplitude
            } else {
                -self.config.amplitude
            };
            samples.extend(
                core::iter::repeat(IqSample { i: level, q: 0.0 })
                    .take(usize::from(self.config.samples_per_symbol)),
            );
        }
        Ok(samples)
    }

    /// Integrate-and-dump matched filter. Positive output means bit 1.
    pub fn demodulate_soft(&self, samples: &[IqSample]) -> Result<Vec<i8>> {
        let width = usize::from(self.config.samples_per_symbol);
        if samples.len() % width != 0 || samples.len() / width > MAX_PHY_BITS {
            return Err(AirError::InvalidLength);
        }
        let mut llrs = Vec::with_capacity(samples.len() / width);
        for symbol in samples.chunks_exact(width) {
            let mean = symbol.iter().map(|sample| sample.i).sum::<f32>() / width as f32;
            let normalized = mean / self.config.amplitude;
            llrs.push((normalized * 100.0).clamp(-127.0, 127.0) as i8);
        }
        Ok(llrs)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CpfskConfig {
    pub sample_rate: u32,
    pub symbol_rate: u32,
    /// Bit 1 frequency. Bell-202 convention calls this mark.
    pub mark_hz: u32,
    /// Bit 0 frequency. Bell-202 convention calls this space.
    pub space_hz: u32,
    pub amplitude: f32,
}

impl CpfskConfig {
    pub fn validate(&self) -> Result<()> {
        if self.symbol_rate == 0
            || self.sample_rate == 0
            || self.sample_rate % self.symbol_rate != 0
            || self.samples_per_symbol() < 4
            || self.samples_per_symbol() > 4_096
            || self.mark_hz == self.space_hz
            || self.mark_hz.saturating_mul(2) >= self.sample_rate
            || self.space_hz.saturating_mul(2) >= self.sample_rate
            || !self.amplitude.is_finite()
            || self.amplitude <= 0.0
            || self.amplitude > 1.0
        {
            return Err(AirError::InvalidLength);
        }
        Ok(())
    }

    pub const fn samples_per_symbol(&self) -> u32 {
        self.sample_rate / self.symbol_rate
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CpfskModem {
    config: CpfskConfig,
}

impl CpfskModem {
    pub fn new(config: CpfskConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    pub const fn config(&self) -> CpfskConfig {
        self.config
    }

    /// Continuous-phase AFSK/CPFSK into signed 16-bit mono PCM.
    pub fn modulate(&self, bits: &[u8]) -> Result<Vec<i16>> {
        if bits.len() > MAX_PHY_BITS || bits.iter().any(|bit| *bit > 1) {
            return Err(AirError::InvalidLength);
        }
        let width = self.config.samples_per_symbol() as usize;
        let count = bits
            .len()
            .checked_mul(width)
            .ok_or(AirError::LimitExceeded)?;
        if count > MAX_WAVEFORM_SAMPLES {
            return Err(AirError::LimitExceeded);
        }
        let mut pcm = Vec::with_capacity(count);
        let mut phase = 0.0_f32;
        for bit in bits {
            let frequency = if *bit == 1 {
                self.config.mark_hz
            } else {
                self.config.space_hz
            } as f32;
            let step = 2.0 * core::f32::consts::PI * frequency / self.config.sample_rate as f32;
            for _ in 0..width {
                pcm.push(
                    (libm::sinf(phase) * self.config.amplitude * i16::MAX as f32)
                        .clamp(i16::MIN as f32, i16::MAX as f32) as i16,
                );
                phase += step;
                if phase >= 2.0 * core::f32::consts::PI {
                    phase -= 2.0 * core::f32::consts::PI;
                }
            }
        }
        Ok(pcm)
    }

    /// Noncoherent two-bin correlator, robust to arbitrary carrier phase.
    pub fn demodulate_soft(&self, pcm: &[i16]) -> Result<Vec<i8>> {
        let width = self.config.samples_per_symbol() as usize;
        if pcm.len() % width != 0 || pcm.len() / width > MAX_PHY_BITS {
            return Err(AirError::InvalidLength);
        }
        let mut llrs = Vec::with_capacity(pcm.len() / width);
        for symbol in pcm.chunks_exact(width) {
            let mark = tone_energy(symbol, self.config.mark_hz, self.config.sample_rate);
            let space = tone_energy(symbol, self.config.space_hz, self.config.sample_rate);
            let total = mark + space + 1.0;
            let normalized = (mark - space) / total;
            llrs.push((normalized * 127.0).clamp(-127.0, 127.0) as i8);
        }
        Ok(llrs)
    }
}

fn tone_energy(samples: &[i16], frequency: u32, sample_rate: u32) -> f32 {
    let step = 2.0 * core::f32::consts::PI * frequency as f32 / sample_rate as f32;
    let mut i_sum = 0.0_f32;
    let mut q_sum = 0.0_f32;
    for (index, sample) in samples.iter().enumerate() {
        let phase = step * index as f32;
        let value = f32::from(*sample);
        i_sum += value * libm::cosf(phase);
        q_sum += value * libm::sinf(phase);
    }
    i_sum * i_sum + q_sum * q_sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bpsk_soft_sign_round_trip() {
        let modem = BpskModem::new(BpskConfig {
            samples_per_symbol: 8,
            amplitude: 0.5,
        })
        .unwrap();
        let bits = [0, 1, 1, 0, 1];
        let llrs = modem
            .demodulate_soft(&modem.modulate(&bits).unwrap())
            .unwrap();
        assert_eq!(
            llrs.iter()
                .map(|llr| u8::from(*llr > 0))
                .collect::<Vec<_>>(),
            bits
        );
    }

    #[test]
    fn cpfsk_soft_sign_round_trip() {
        let modem = CpfskModem::new(CpfskConfig {
            sample_rate: 48_000,
            symbol_rate: 1_200,
            mark_hz: 1_200,
            space_hz: 2_200,
            amplitude: 0.8,
        })
        .unwrap();
        let bits = [0, 1, 0, 0, 1, 1, 1];
        let llrs = modem
            .demodulate_soft(&modem.modulate(&bits).unwrap())
            .unwrap();
        assert_eq!(
            llrs.iter()
                .map(|llr| u8::from(*llr > 0))
                .collect::<Vec<_>>(),
            bits
        );
    }
}
