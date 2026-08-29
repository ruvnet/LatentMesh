//! Sampling for the live runs (design §3/§6): the paper-mirrored
//! T=0.6 / top-p 0.95 arm and the greedy batch=1 witness arm with
//! per-step logits hashes.

use candle_core::{DType, Result, Tensor};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use sha2::{Digest, Sha256};

/// Decoding regime.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Sampling {
    /// Deterministic argmax — the witness arm, and the S0/S1a probe default
    /// (deterministic per-item outcomes maximize paired-test power).
    Greedy,
    /// Nucleus sampling; run-1 registered arm is `{ temperature: 0.6, top_p: 0.95 }`.
    TopP { temperature: f64, top_p: f64 },
}

/// Stateful sampler; seeded so sampled runs are reproducible.
#[derive(Debug)]
pub struct Sampler {
    kind: Sampling,
    rng: StdRng,
}

impl Sampler {
    pub fn new(kind: Sampling, seed: u64) -> Self {
        Self {
            kind,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Sample one token id from last-position logits (`(1, 1, vocab)` or any
    /// shape flattening to `vocab`).
    pub fn sample(&mut self, logits: &Tensor) -> Result<u32> {
        let logits = logits
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        match self.kind {
            Sampling::Greedy => Ok(argmax(&logits)),
            Sampling::TopP { temperature, top_p } => {
                Ok(self.sample_top_p(&logits, temperature, top_p))
            }
        }
    }

    fn sample_top_p(&mut self, logits: &[f32], temperature: f64, top_p: f64) -> u32 {
        // Softmax at temperature.
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<(u32, f32)> = logits
            .iter()
            .enumerate()
            .map(|(i, &l)| (i as u32, (((l - max) as f64) / temperature).exp() as f32))
            .collect();
        let sum: f32 = probs.iter().map(|(_, p)| p).sum();
        for p in probs.iter_mut() {
            p.1 /= sum;
        }
        // Nucleus truncation.
        probs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut cum = 0f32;
        let mut cutoff = probs.len();
        for (i, (_, p)) in probs.iter().enumerate() {
            cum += p;
            if cum >= top_p as f32 {
                cutoff = i + 1;
                break;
            }
        }
        probs.truncate(cutoff);
        let nucleus_sum: f32 = probs.iter().map(|(_, p)| p).sum();
        let mut draw = self.rng.gen::<f32>() * nucleus_sum;
        for (id, p) in &probs {
            draw -= p;
            if draw <= 0.0 {
                return *id;
            }
        }
        probs.last().map(|(id, _)| *id).unwrap_or(0)
    }
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &l) in logits.iter().enumerate() {
        if l > logits[best] {
            best = i;
        }
    }
    best as u32
}

/// SHA-256 over the f32 little-endian bytes of a logits tensor — the greedy
/// witness arm's per-step reproducibility receipt.
pub fn hash_logits(logits: &Tensor) -> Result<String> {
    let v = logits
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let mut hasher = Sha256::new();
    for x in &v {
        hasher.update(x.to_le_bytes());
    }
    Ok(format!("{:x}", hasher.finalize()))
}
#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn logits() -> Tensor {
        Tensor::from_vec(vec![0.1f32, 3.0, 2.9, -1.0, 0.5], 5, &Device::Cpu).unwrap()
    }

    #[test]
    fn greedy_is_argmax() {
        let mut s = Sampler::new(Sampling::Greedy, 0);
        assert_eq!(s.sample(&logits()).unwrap(), 1);
    }

    #[test]
    fn top_p_is_seed_deterministic_and_in_nucleus() {
        let run = |seed| {
            let mut s = Sampler::new(
                Sampling::TopP {
                    temperature: 0.6,
                    top_p: 0.95,
                },
                seed,
            );
            (0..32)
                .map(|_| s.sample(&logits()).unwrap())
                .collect::<Vec<_>>()
        };
        assert_eq!(run(7), run(7), "same seed must reproduce exactly");
        // With these logits at T=0.6, the 0.95 nucleus is {1, 2}; no sample
        // may escape it.
        assert!(run(7).iter().all(|t| *t == 1 || *t == 2));
    }

    #[test]
    fn tiny_top_p_degenerates_to_argmax() {
        let mut s = Sampler::new(
            Sampling::TopP {
                temperature: 0.6,
                top_p: 1e-6,
            },
            3,
        );
        assert_eq!(s.sample(&logits()).unwrap(), 1);
    }
}
