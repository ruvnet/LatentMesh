//! The trainable receiver-side LoRA (ADR-045 M5).
//!
//! Architecture is `latentmesh_runtime::lora::ResidualLora`'s, restated here
//! with candle `Var`s so it can be optimised: down-project `A (hidden, rank)`,
//! up-project `B (rank, hidden)`, `scaling = alpha / rank`, F32 arithmetic,
//! delta cast to the residual's dtype before the add. Weight init follows
//! `mlp.rs`'s convention — `A ~ U(−1/√hidden, +1/√hidden)` drawn host-side
//! from one seeded ChaCha8 stream, `B` **zero** (standard LoRA init), so a
//! freshly initialised adapter is the exact identity.
//!
//! `docs/research/034` §2 is the reason this is a new module rather than a
//! wrapper: ruvector's in-stack `micro_lora.rs` has the right *architecture*
//! but its `accumulate_gradient` is a Hebbian/REINFORCE delta rule keyed on a
//! scalar quality score, with no connection to any receiver forward or loss.
//! Only the forward-pass shape is borrowed; the training machinery is candle
//! `AdamW` over a real task loss.
//!
//! The artifact is the raw-f32 file `ResidualLora` reads
//! (`a[hidden × rank] ‖ b[rank × hidden]`, little-endian, row-major), and
//! [`golden_pairs`] produces the input/output pairs the probe side verifies
//! its own loaded adapter against — the same discipline `mlp.rs` uses, applied
//! across the composed/vendored boundary.

use crate::mlp::gaussian_vec;
use candle_core::{DType, Device, Tensor, Var};
use latentmesh_runtime::lora::{ResidualLora, LORA_ALPHA};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use sha2::Digest;

/// The receiver's hidden size (Qwen2.5-1.5B-Instruct).
pub const HIDDEN: usize = 1536;

/// The trainable adapter.
pub struct LoraAdapter {
    pub a: Var,
    pub b: Var,
    pub rank: usize,
    pub hidden: usize,
    pub alpha: f64,
}

impl LoraAdapter {
    /// Seeded deterministic init: `A ~ U(±1/√hidden)`, `B = 0`.
    pub fn new_seeded(seed: u64, rank: usize, hidden: usize, dev: &Device) -> anyhow::Result<Self> {
        anyhow::ensure!(rank > 0 && hidden > 0, "rank and hidden must be positive");
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let bound = 1.0 / (hidden as f32).sqrt();
        let a: Vec<f32> = (0..hidden * rank)
            .map(|_| rng.gen::<f32>() * 2.0 * bound - bound)
            .collect();
        let b = vec![0f32; rank * hidden];
        Self::from_parts(&a, &b, rank, hidden, dev)
    }

    fn from_parts(
        a: &[f32],
        b: &[f32],
        rank: usize,
        hidden: usize,
        dev: &Device,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(a.len() == hidden * rank && b.len() == rank * hidden);
        Ok(Self {
            a: Var::from_tensor(&Tensor::from_vec(a.to_vec(), (hidden, rank), dev)?)?,
            b: Var::from_tensor(&Tensor::from_vec(b.to_vec(), (rank, hidden), dev)?)?,
            rank,
            hidden,
            alpha: LORA_ALPHA,
        })
    }

    pub fn vars(&self) -> Vec<Var> {
        vec![self.a.clone(), self.b.clone()]
    }

    pub fn param_count(&self) -> usize {
        2 * self.hidden * self.rank
    }

    pub fn scaling(&self) -> f64 {
        self.alpha / self.rank as f64
    }

    /// The adapter's delta for a `(b, seq, hidden)` residual, in the
    /// residual's own dtype — the same three ops, in the same order, as
    /// `ResidualLora::delta`.
    pub fn delta(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let dt = xs.dtype();
        let (b, l, h) = xs.dims3()?;
        let flat = xs.to_dtype(DType::F32)?.reshape((b * l, h))?.contiguous()?;
        let d = flat
            .matmul(self.a.as_tensor())?
            .matmul(self.b.as_tensor())?;
        (d * self.scaling())?.reshape((b, l, h))?.to_dtype(dt)
    }

    /// `h + delta(h)`.
    pub fn apply(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        xs.add(&self.delta(xs)?)
    }

    /// Host copy in artifact-layout order.
    pub fn to_flat(&self) -> candle_core::Result<Vec<f32>> {
        let mut out = Vec::with_capacity(self.param_count());
        out.extend(self.a.as_tensor().flatten_all()?.to_vec1::<f32>()?);
        out.extend(self.b.as_tensor().flatten_all()?.to_vec1::<f32>()?);
        Ok(out)
    }

    /// Overwrite both `Var`s from a flat artifact-layout buffer (used to
    /// restore the best-holdout epoch before saving).
    pub fn set_flat(&self, flat: &[f32], dev: &Device) -> candle_core::Result<()> {
        assert_eq!(flat.len(), self.param_count());
        let (a, b) = flat.split_at(self.hidden * self.rank);
        self.a.set(&Tensor::from_vec(
            a.to_vec(),
            (self.hidden, self.rank),
            dev,
        )?)?;
        self.b.set(&Tensor::from_vec(
            b.to_vec(),
            (self.rank, self.hidden),
            dev,
        )?)
    }

    /// Write the raw-f32 artifact; returns its sha256 content hash.
    pub fn save_artifact(&self, path: &std::path::Path) -> anyhow::Result<String> {
        let flat = self.to_flat()?;
        let mut bytes = Vec::with_capacity(flat.len() * 4);
        for f in &flat {
            bytes.extend_from_slice(&f.to_le_bytes());
        }
        std::fs::write(path, &bytes)?;
        Ok(format!("{:x}", sha2::Sha256::digest(&bytes)))
    }

    /// L2 norms of both factors — the cheapest "did anything actually train"
    /// instrument, recorded per epoch in the training receipt.
    pub fn factor_norms(&self) -> candle_core::Result<(f32, f32)> {
        let n = |v: &Var| -> candle_core::Result<f32> {
            Ok(v.as_tensor()
                .to_dtype(DType::F32)?
                .sqr()?
                .sum_all()?
                .to_scalar::<f32>()?
                .sqrt())
        };
        Ok((n(&self.a)?, n(&self.b)?))
    }
}

/// `(artifact_sha256, inputs, outputs)` — the delta, not the sum, so a
/// zero-`B` adapter's goldens are unambiguously zero.
pub type GoldenPairs = (String, Vec<Vec<f32>>, Vec<Vec<f32>>);

/// Golden pairs produced by the trained adapter ITSELF, loaded back from the
/// artifact through the **runtime's** `ResidualLora` (the probe-side type) on
/// CPU. Verifying the probe's loaded adapter against these therefore pins the
/// two implementations to each other across the crate boundary, exactly as
/// `mlp.rs`'s goldens pin the hand-rolled probe forward to the trained net.
pub fn golden_pairs(
    artifact: &std::path::Path,
    hidden: usize,
    after_block: usize,
    seed: u64,
    n_pairs: usize,
) -> anyhow::Result<GoldenPairs> {
    let cpu = Device::Cpu;
    let lora = ResidualLora::load_artifact(artifact, hidden, after_block, LORA_ALPHA, &cpu)?;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut inputs = Vec::with_capacity(n_pairs);
    let mut outputs = Vec::with_capacity(n_pairs);
    for _ in 0..n_pairs {
        let x = gaussian_vec(&mut rng, hidden);
        let xt = Tensor::from_vec(x.clone(), (1, 1, hidden), &cpu)?;
        outputs.push(lora.delta(&xt)?.flatten_all()?.to_vec1::<f32>()?);
        inputs.push(x);
    }
    Ok((lora.content_hash, inputs, outputs))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_b_init_is_the_identity_and_is_deterministic() {
        let dev = Device::Cpu;
        let l = LoraAdapter::new_seeded(7, 1, 16, &dev).unwrap();
        assert_eq!(l.param_count(), 32);
        let (na, nb) = l.factor_norms().unwrap();
        assert!(na > 0.0, "A must not be zero-initialised");
        assert_eq!(nb, 0.0, "B must be zero-initialised");
        let xs =
            Tensor::from_vec((0..2 * 16).map(|i| i as f32).collect(), (1, 2, 16), &dev).unwrap();
        assert_eq!(
            l.apply(&xs).unwrap().to_vec3::<f32>().unwrap(),
            xs.to_vec3::<f32>().unwrap()
        );
        let again = LoraAdapter::new_seeded(7, 1, 16, &dev).unwrap();
        assert_eq!(l.to_flat().unwrap(), again.to_flat().unwrap());
        let other = LoraAdapter::new_seeded(8, 1, 16, &dev).unwrap();
        assert_ne!(l.to_flat().unwrap(), other.to_flat().unwrap());
    }

    /// The training module and the probe-side `ResidualLora` must be the SAME
    /// function. Asserted through a real artifact round-trip on non-trivial
    /// weights, at every registered rank.
    #[test]
    fn agrees_with_the_runtime_residual_lora_across_the_artifact() {
        let dev = Device::Cpu;
        let dir = std::env::temp_dir().join(format!("lm-m5-lora-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let hidden = 32usize;
        for rank in [1usize, 2, 4] {
            let l = LoraAdapter::new_seeded(0x4D35_0001, rank, hidden, &dev).unwrap();
            // Give B real values so the comparison is not trivially 0 == 0.
            let mut flat = l.to_flat().unwrap();
            let mut rng = ChaCha8Rng::seed_from_u64(0x4D35_0002 + rank as u64);
            for v in flat[hidden * rank..].iter_mut() {
                *v = rng.gen::<f32>() * 0.5 - 0.25;
            }
            l.set_flat(&flat, &dev).unwrap();
            let path = dir.join(format!("lora-r{rank}.f32bin"));
            let hash = l.save_artifact(&path).unwrap();

            let rl = ResidualLora::load_artifact(&path, hidden, 14, LORA_ALPHA, &dev).unwrap();
            assert_eq!(rl.content_hash, hash);
            assert_eq!(rl.rank, rank);
            assert_eq!(rl.to_flat().unwrap(), flat);

            let mut rng = ChaCha8Rng::seed_from_u64(0x4D35_0003);
            let x = gaussian_vec(&mut rng, 3 * hidden);
            let xs = Tensor::from_vec(x, (1, 3, hidden), &dev).unwrap();
            let ours = l
                .apply(&xs)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let theirs = rl
                .apply(&xs)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let diff: f32 = ours
                .iter()
                .zip(&theirs)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>()
                .sqrt();
            let norm: f32 = theirs.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!(
                diff / norm.max(1e-12) <= 1e-6,
                "rank {rank}: rel err {}",
                diff / norm
            );

            // The goldens the probe verifies against are the DELTA, and they
            // reload through the runtime type.
            let (gh, gin, gout) = golden_pairs(&path, hidden, 14, 0x4D35_0004, 8).unwrap();
            assert_eq!(gh, hash);
            assert_eq!(gin.len(), 8);
            assert!(gout.iter().flatten().any(|v| v.abs() > 0.0));
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
