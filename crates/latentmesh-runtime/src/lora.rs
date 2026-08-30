//! `ResidualLora` — an additive low-rank adapter on the receiver's own
//! residual stream, immediately after one decoder block (ADR-045 M5).
//!
//! Every rung before M5 trained the **payload** against a bit-identical
//! frozen receiver. M5 is the first that changes the receiver's own forward,
//! so the adapter cannot be a per-pass [`crate::models::LayerEdit`]: an edit
//! is supplied only during prefill, while a weight change must also be live
//! on every decode step. It is therefore a persistent, optional field on
//! [`crate::models::Model`], installed by
//! `ModelForCausalLM::set_residual_lora`, and `None` by default — with no
//! adapter installed the executed op sequence is exactly the vendored one, so
//! every receipt produced before M5 stays reproducible.
//!
//! **Equation.** With `h` the residual `(b, seq, hidden)` after `after_block`
//! blocks (and after any `LayerEdit` for that block — the injected content is
//! part of what the adapter sees, which is the whole point):
//!
//! ```text
//! h' = h + ((h · A) · B) · (alpha / rank)      A: (hidden, rank), B: (rank, hidden)
//! ```
//!
//! `B` is zero-initialised at training time, so a freshly initialised adapter
//! is the exact identity and "LoRA off" and "LoRA at init" are the same
//! function — the property the M5 transfer check's baseline arm relies on.
//!
//! **Arithmetic is F32 regardless of model dtype.** The adapter's own matmuls
//! run in F32 and only the delta is cast back to the model dtype before the
//! add. The training-side module (`latentmesh-train::receiver_lora`) is
//! defined by the same three operations in the same order, and the two are
//! pinned against each other by golden pairs the way every artifact in this
//! repo is.
//!
//! Artifact format: raw little-endian f32, `a[hidden × rank] ‖ b[rank ×
//! hidden]`, row-major, `x·A` convention — the repo's raw-f32bin discipline
//! (deterministic bytes, therefore a deterministic content hash).

use candle_core::{DType, Device, Error, Result, Tensor};
use sha2::{Digest, Sha256};
use std::path::Path;

/// LoRA scaling numerator, frozen. `scaling = alpha / rank`, so at rank 1 the
/// adapter's delta enters the residual at unit gain. Both the trainer and the
/// probe read this constant rather than carrying their own copy.
pub const LORA_ALPHA: f64 = 1.0;

/// Artifact layout string, recorded in receipts.
pub const ARTIFACT_LAYOUT: &str =
    "little-endian f32: a[hidden x rank row-major, x@A convention] || b[rank x hidden row-major]";

/// An additive low-rank adapter installed after one decoder block.
#[derive(Debug, Clone)]
pub struct ResidualLora {
    /// Down-projection `(hidden, rank)`, F32.
    a: Tensor,
    /// Up-projection `(rank, hidden)`, F32.
    b: Tensor,
    pub rank: usize,
    pub hidden: usize,
    /// 1-based block count after which the adapter runs (14 for the S2
    /// winner cell's receiver side).
    pub after_block: usize,
    pub alpha: f64,
    /// sha256 of the artifact file bytes, when loaded from one.
    pub content_hash: String,
}

impl ResidualLora {
    /// Build from a flat artifact-layout buffer.
    pub fn from_flat(
        flat: &[f32],
        hidden: usize,
        after_block: usize,
        alpha: f64,
        device: &Device,
        content_hash: String,
    ) -> Result<Self> {
        if hidden == 0 || flat.len() % (2 * hidden) != 0 || flat.is_empty() {
            return Err(Error::Msg(format!(
                "LoRA buffer of {} f32 is not 2 x hidden({hidden}) x rank",
                flat.len()
            )));
        }
        let rank = flat.len() / (2 * hidden);
        let (a, b) = flat.split_at(hidden * rank);
        Ok(Self {
            a: Tensor::from_vec(a.to_vec(), (hidden, rank), device)?,
            b: Tensor::from_vec(b.to_vec(), (rank, hidden), device)?,
            rank,
            hidden,
            after_block,
            alpha,
            content_hash,
        })
    }

    /// Load a raw-f32 artifact; the rank is derived from the file length.
    pub fn load_artifact(
        path: &Path,
        hidden: usize,
        after_block: usize,
        alpha: f64,
        device: &Device,
    ) -> Result<Self> {
        let bytes = std::fs::read(path).map_err(Error::wrap)?;
        if bytes.len() % 4 != 0 {
            return Err(Error::Msg(format!(
                "LoRA artifact {} is not a whole number of f32",
                path.display()
            )));
        }
        let content_hash = format!("{:x}", Sha256::digest(&bytes));
        let flat: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Self::from_flat(&flat, hidden, after_block, alpha, device, content_hash)
    }

    /// `alpha / rank`.
    pub fn scaling(&self) -> f64 {
        self.alpha / self.rank as f64
    }

    /// The adapter's delta for a `(b, seq, hidden)` residual, in the
    /// residual's own dtype. F32 matmuls, cast last.
    pub fn delta(&self, xs: &Tensor) -> Result<Tensor> {
        let dt = xs.dtype();
        let (b, l, h) = xs.dims3()?;
        if h != self.hidden {
            return Err(Error::Msg(format!(
                "LoRA hidden {} != residual hidden {h}",
                self.hidden
            )));
        }
        let flat = xs.to_dtype(DType::F32)?.reshape((b * l, h))?.contiguous()?;
        let d = flat.matmul(&self.a)?.matmul(&self.b)?;
        (d * self.scaling())?.reshape((b, l, h))?.to_dtype(dt)
    }

    /// `h + delta(h)`.
    pub fn apply(&self, xs: &Tensor) -> Result<Tensor> {
        xs.add(&self.delta(xs)?)
    }

    /// Host copy of the weights in artifact-layout order (for receipts).
    pub fn to_flat(&self) -> Result<Vec<f32>> {
        let mut out = Vec::with_capacity(2 * self.hidden * self.rank);
        out.extend(self.a.flatten_all()?.to_vec1::<f32>()?);
        out.extend(self.b.flatten_all()?.to_vec1::<f32>()?);
        Ok(out)
    }

    /// Parameter count — `2 · hidden · rank`.
    pub fn param_count(&self) -> usize {
        2 * self.hidden * self.rank
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn residual(seq: usize, hidden: usize) -> Tensor {
        let data: Vec<f32> = (0..seq * hidden).map(|i| (i as f32 + 1.0) * 0.5).collect();
        Tensor::from_vec(data, (1, seq, hidden), &Device::Cpu).unwrap()
    }

    /// Zero `B` (the training-time init) must make the adapter the EXACT
    /// identity — the property the transfer check's baseline arm relies on.
    #[test]
    fn zero_b_is_the_exact_identity() {
        let (hidden, rank) = (6usize, 2usize);
        let mut flat = vec![0f32; 2 * hidden * rank];
        for (i, v) in flat[..hidden * rank].iter_mut().enumerate() {
            *v = i as f32 * 0.125 - 0.3;
        }
        let lora = ResidualLora::from_flat(&flat, hidden, 14, LORA_ALPHA, &Device::Cpu, "t".into())
            .unwrap();
        assert_eq!(lora.rank, rank);
        let xs = residual(3, hidden);
        let out = lora.apply(&xs).unwrap();
        assert_eq!(out.to_vec3::<f32>().unwrap(), xs.to_vec3::<f32>().unwrap());
    }

    /// The equation, checked against a hand computation on a rank-1 adapter.
    #[test]
    fn rank1_matches_hand_computation() {
        let hidden = 4usize;
        // a = [1,0,0,0]^T so x·A = x[0]; b = [0,2,0,0] so the delta is
        // x[0] * [0,2,0,0] * scaling.
        let mut flat = vec![0f32; 2 * hidden];
        flat[0] = 1.0;
        flat[hidden + 1] = 2.0;
        let lora = ResidualLora::from_flat(&flat, hidden, 14, LORA_ALPHA, &Device::Cpu, "t".into())
            .unwrap();
        assert_eq!(lora.rank, 1);
        assert!((lora.scaling() - LORA_ALPHA).abs() < 1e-12);
        let xs = Tensor::from_vec(
            vec![3.0f32, 1.0, 1.0, 1.0, -2.0, 5.0, 5.0, 5.0],
            (1, 2, hidden),
            &Device::Cpu,
        )
        .unwrap();
        let out = lora.apply(&xs).unwrap().to_vec3::<f32>().unwrap();
        let s = LORA_ALPHA as f32;
        assert_eq!(out[0][0], vec![3.0, 1.0 + 3.0 * 2.0 * s, 1.0, 1.0]);
        assert_eq!(out[0][1], vec![-2.0, 5.0 + -2.0 * 2.0 * s, 5.0, 5.0]);
    }

    #[test]
    fn rank_is_derived_from_the_buffer_length() {
        let hidden = 8usize;
        for rank in [1usize, 2, 4] {
            let flat = vec![0.25f32; 2 * hidden * rank];
            let l =
                ResidualLora::from_flat(&flat, hidden, 14, LORA_ALPHA, &Device::Cpu, "t".into())
                    .unwrap();
            assert_eq!(l.rank, rank);
            assert_eq!(l.param_count(), flat.len());
            assert_eq!(l.to_flat().unwrap(), flat);
        }
        // A buffer that is not a whole number of (hidden x rank) pairs is
        // rejected rather than silently truncated.
        assert!(ResidualLora::from_flat(
            &vec![0f32; 2 * hidden + 1],
            hidden,
            14,
            LORA_ALPHA,
            &Device::Cpu,
            "t".into()
        )
        .is_err());
    }
}
