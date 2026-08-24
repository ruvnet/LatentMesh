use alloc::vec::Vec;

use latentmesh_air_core::{AirError, Result};

const INPUTS: usize = 4;
const HIDDEN: usize = 4;
const MIN_VERIFIED_SAMPLES: u32 = 32;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AssistObservation {
    /// Relative matched-filter energy, normally 0..1.
    pub energy: f32,
    /// Estimated noise fraction, normally 0..1.
    pub noise: f32,
    /// Normalized frequency error, normally -1..1.
    pub frequency_error: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AssistDecision {
    pub llr: i8,
    pub used_neural: bool,
    pub confidence: f32,
}

/// A bounded 4x4x1 online neural assist. It is deliberately subordinate to
/// the classical matched-filter output. Until enough CRC/state-hash-verified
/// labels exist, or whenever confidence is low, the output is byte-for-byte
/// the original classical LLR.
#[derive(Clone, Debug)]
pub struct TinyNeuralAssist {
    w1: [[f32; INPUTS]; HIDDEN],
    b1: [f32; HIDDEN],
    w2: [f32; HIDDEN],
    b2: f32,
    threshold: f32,
    verified_samples: u32,
}

impl Default for TinyNeuralAssist {
    fn default() -> Self {
        Self {
            w1: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.5, -0.5, 0.0],
                [0.0, 0.0, 0.0, 0.5],
                [0.25, 0.25, -0.25, 0.0],
            ],
            b1: [0.0; HIDDEN],
            w2: [0.8, 0.1, 0.1, 0.2],
            b2: 0.0,
            threshold: 0.72,
            verified_samples: 0,
        }
    }
}

impl TinyNeuralAssist {
    pub fn new(threshold: f32) -> Result<Self> {
        if !threshold.is_finite() || !(0.5..=0.99).contains(&threshold) {
            return Err(AirError::InvalidLength);
        }
        Ok(Self {
            threshold,
            ..Self::default()
        })
    }

    pub const fn verified_samples(&self) -> u32 {
        self.verified_samples
    }

    pub fn refine(&self, classical_llr: i8, observation: AssistObservation) -> AssistDecision {
        let (_, prediction) = self.forward(classical_llr, observation);
        let confidence = libm::fabsf(prediction);
        if self.verified_samples < MIN_VERIFIED_SAMPLES
            || confidence < self.threshold
            || classical_llr.unsigned_abs() >= 100
        {
            return AssistDecision {
                llr: classical_llr,
                used_neural: false,
                confidence,
            };
        }
        let neural_llr = (prediction * 127.0).clamp(-127.0, 127.0);
        let blended = 0.6 * f32::from(classical_llr) + 0.4 * neural_llr;
        AssistDecision {
            llr: blended.clamp(-127.0, 127.0) as i8,
            used_neural: true,
            confidence,
        }
    }

    pub fn refine_batch(
        &self,
        classical_llrs: &[i8],
        observations: &[AssistObservation],
    ) -> Result<Vec<i8>> {
        if classical_llrs.len() != observations.len() {
            return Err(AirError::InvalidLength);
        }
        Ok(classical_llrs
            .iter()
            .zip(observations)
            .map(|(llr, observation)| self.refine(*llr, *observation).llr)
            .collect())
    }

    /// Online SGD using only a label whose enclosing frame passed CRC32C and,
    /// for semantic state, the full critical-state hash. Callers enforce that
    /// trust precondition; the method name makes accidental use conspicuous.
    pub fn train_verified(
        &mut self,
        classical_llr: i8,
        observation: AssistObservation,
        expected_bit: u8,
        learning_rate: f32,
    ) -> Result<()> {
        if expected_bit > 1
            || !learning_rate.is_finite()
            || !(0.0001..=0.1).contains(&learning_rate)
        {
            return Err(AirError::InvalidLength);
        }
        let inputs = normalized_inputs(classical_llr, observation);
        let (hidden, prediction) = forward_values(&self.w1, &self.b1, &self.w2, self.b2, inputs);
        let target = if expected_bit == 1 { 1.0 } else { -1.0 };
        let output_gradient = (target - prediction) * (1.0 - prediction * prediction);
        let old_w2 = self.w2;
        for (index, hidden_value) in hidden.iter().enumerate() {
            self.w2[index] =
                bounded(self.w2[index] + learning_rate * output_gradient * hidden_value);
        }
        self.b2 = bounded(self.b2 + learning_rate * output_gradient);
        for hidden_index in 0..HIDDEN {
            let hidden_gradient = output_gradient
                * old_w2[hidden_index]
                * (1.0 - hidden[hidden_index] * hidden[hidden_index]);
            for (input_index, input) in inputs.iter().enumerate() {
                self.w1[hidden_index][input_index] = bounded(
                    self.w1[hidden_index][input_index] + learning_rate * hidden_gradient * input,
                );
            }
            self.b1[hidden_index] =
                bounded(self.b1[hidden_index] + learning_rate * hidden_gradient);
        }
        self.verified_samples = self.verified_samples.saturating_add(1);
        Ok(())
    }

    fn forward(&self, classical_llr: i8, observation: AssistObservation) -> ([f32; HIDDEN], f32) {
        forward_values(
            &self.w1,
            &self.b1,
            &self.w2,
            self.b2,
            normalized_inputs(classical_llr, observation),
        )
    }
}

fn normalized_inputs(classical_llr: i8, observation: AssistObservation) -> [f32; INPUTS] {
    [
        f32::from(classical_llr) / 127.0,
        finite_clamp(observation.energy, 0.0, 1.0),
        finite_clamp(observation.noise, 0.0, 1.0),
        finite_clamp(observation.frequency_error, -1.0, 1.0),
    ]
}

fn forward_values(
    w1: &[[f32; INPUTS]; HIDDEN],
    b1: &[f32; HIDDEN],
    w2: &[f32; HIDDEN],
    b2: f32,
    inputs: [f32; INPUTS],
) -> ([f32; HIDDEN], f32) {
    let mut hidden = [0.0_f32; HIDDEN];
    for row in 0..HIDDEN {
        let sum = w1[row]
            .iter()
            .zip(inputs)
            .fold(b1[row], |acc, (weight, input)| acc + weight * input);
        hidden[row] = libm::tanhf(sum);
    }
    let output = w2
        .iter()
        .zip(hidden)
        .fold(b2, |acc, (weight, value)| acc + weight * value);
    (hidden, libm::tanhf(output))
}

fn finite_clamp(value: f32, minimum: f32, maximum: f32) -> f32 {
    if value.is_finite() {
        value.clamp(minimum, maximum)
    } else {
        0.0
    }
}

fn bounded(value: f32) -> f32 {
    finite_clamp(value, -4.0, 4.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uncertain_or_uncalibrated_output_is_exact_classical_fallback() {
        let assist = TinyNeuralAssist::default();
        let observation = AssistObservation {
            energy: 0.5,
            noise: 0.5,
            frequency_error: 0.0,
        };
        for classical in [-127, -20, 0, 19, 127] {
            let decision = assist.refine(classical, observation);
            assert_eq!(decision.llr, classical);
            assert!(!decision.used_neural);
        }
    }

    #[test]
    fn only_verified_training_unlocks_assist() {
        let mut assist = TinyNeuralAssist::default();
        let observation = AssistObservation {
            energy: 0.9,
            noise: 0.1,
            frequency_error: 0.0,
        };
        for _ in 0..MIN_VERIFIED_SAMPLES {
            assist.train_verified(40, observation, 1, 0.02).unwrap();
        }
        assert_eq!(assist.verified_samples(), MIN_VERIFIED_SAMPLES);
        assert!(assist.refine(40, observation).confidence.is_finite());
    }
}
