//! The M3 MLP projector (ADR-024, frozen architecture): 2-layer MLP
//! 2048→512→1536, ReLU, 1,837,056 parameters.
//!
//! Weight init is seeded (ChaCha8) and built host-side so the whole model is
//! reproducible from `(seed)` alone. The trained artifact is a raw
//! little-endian f32 file (`w1[2048×512] ‖ b1[512] ‖ w2[512×1536] ‖
//! b2[1536]`, row-major, x·W convention) — the repo's existing raw-f32bin
//! discipline, chosen over safetensors so artifact bytes (and therefore the
//! content hash) are trivially deterministic.

use candle_core::{Device, Tensor, Var};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use sha2::Digest;

pub const D_IN: usize = 2048;
pub const D_HID: usize = 512;
pub const D_OUT: usize = 1536;
/// Frozen parameter count (ADR-024 M3): 2048·512+512 + 512·1536+1536.
pub const PARAM_COUNT: usize = D_IN * D_HID + D_HID + D_HID * D_OUT + D_OUT;
/// Artifact layout string, recorded in the meta JSON and receipts.
pub const ARTIFACT_LAYOUT: &str = "little-endian f32: w1[2048x512 row-major, x@W convention] || b1[512] || w2[512x1536 row-major] || b2[1536]";

/// The projector with trainable `Var`s on a device.
pub struct Mlp {
    pub w1: Var,
    pub b1: Var,
    pub w2: Var,
    pub b2: Var,
}

impl Mlp {
    /// Seeded deterministic init: each layer U(−1/√fan_in, +1/√fan_in)
    /// (weights then bias), drawn host-side from one ChaCha8 stream.
    pub fn new_seeded(seed: u64, dev: &Device) -> candle_core::Result<Self> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut uniform = |n: usize, a: f32| -> Vec<f32> {
            (0..n).map(|_| rng.gen::<f32>() * 2.0 * a - a).collect()
        };
        let a1 = 1.0 / (D_IN as f32).sqrt();
        let a2 = 1.0 / (D_HID as f32).sqrt();
        let w1 = uniform(D_IN * D_HID, a1);
        let b1 = uniform(D_HID, a1);
        let w2 = uniform(D_HID * D_OUT, a2);
        let b2 = uniform(D_OUT, a2);
        Self::from_parts(w1, b1, w2, b2, dev)
    }

    fn from_parts(
        w1: Vec<f32>,
        b1: Vec<f32>,
        w2: Vec<f32>,
        b2: Vec<f32>,
        dev: &Device,
    ) -> candle_core::Result<Self> {
        Ok(Self {
            w1: Var::from_tensor(&Tensor::from_vec(w1, (D_IN, D_HID), dev)?)?,
            b1: Var::from_tensor(&Tensor::from_vec(b1, (D_HID,), dev)?)?,
            w2: Var::from_tensor(&Tensor::from_vec(w2, (D_HID, D_OUT), dev)?)?,
            b2: Var::from_tensor(&Tensor::from_vec(b2, (D_OUT,), dev)?)?,
        })
    }

    pub fn vars(&self) -> Vec<Var> {
        vec![
            self.w1.clone(),
            self.b1.clone(),
            self.w2.clone(),
            self.b2.clone(),
        ]
    }

    pub fn param_count(&self) -> usize {
        self.vars().iter().map(|v| v.elem_count()).sum()
    }

    /// Forward `(B, 2048) → (B, 1536)`: `relu(x·W1 + b1)·W2 + b2`.
    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let h = x
            .matmul(self.w1.as_tensor())?
            .broadcast_add(self.b1.as_tensor())?
            .relu()?;
        h.matmul(self.w2.as_tensor())?
            .broadcast_add(self.b2.as_tensor())
    }

    /// Host-side copy of all weights in artifact layout order.
    pub fn to_flat(&self) -> candle_core::Result<Vec<f32>> {
        let mut out = Vec::with_capacity(PARAM_COUNT);
        for t in [&self.w1, &self.b1, &self.w2, &self.b2] {
            out.extend(t.as_tensor().flatten_all()?.to_vec1::<f32>()?);
        }
        Ok(out)
    }

    /// Overwrite the `Var`s from a flat artifact-layout buffer (used to
    /// restore the best-validation-epoch weights before saving).
    pub fn set_flat(&self, flat: &[f32], dev: &Device) -> candle_core::Result<()> {
        assert_eq!(flat.len(), PARAM_COUNT);
        let (w1, rest) = flat.split_at(D_IN * D_HID);
        let (b1, rest) = rest.split_at(D_HID);
        let (w2, b2) = rest.split_at(D_HID * D_OUT);
        self.w1
            .set(&Tensor::from_vec(w1.to_vec(), (D_IN, D_HID), dev)?)?;
        self.b1
            .set(&Tensor::from_vec(b1.to_vec(), (D_HID,), dev)?)?;
        self.w2
            .set(&Tensor::from_vec(w2.to_vec(), (D_HID, D_OUT), dev)?)?;
        self.b2
            .set(&Tensor::from_vec(b2.to_vec(), (D_OUT,), dev)?)?;
        Ok(())
    }

    /// Write the raw-f32 artifact file; returns its sha256 content hash.
    pub fn save_artifact(&self, path: &std::path::Path) -> anyhow::Result<String> {
        let flat = self.to_flat()?;
        let mut bytes = Vec::with_capacity(flat.len() * 4);
        for f in &flat {
            bytes.extend_from_slice(&f.to_le_bytes());
        }
        std::fs::write(path, &bytes)?;
        Ok(format!("{:x}", sha2::Sha256::digest(&bytes)))
    }

    /// Load an artifact file onto `dev`; returns `(mlp, sha256)`.
    pub fn load_artifact(path: &std::path::Path, dev: &Device) -> anyhow::Result<(Self, String)> {
        let bytes = std::fs::read(path)?;
        anyhow::ensure!(
            bytes.len() == PARAM_COUNT * 4,
            "artifact {} bytes != {} params x 4",
            bytes.len(),
            PARAM_COUNT
        );
        let sha = format!("{:x}", sha2::Sha256::digest(&bytes));
        let flat: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let (w1, rest) = flat.split_at(D_IN * D_HID);
        let (b1, rest) = rest.split_at(D_HID);
        let (w2, b2) = rest.split_at(D_HID * D_OUT);
        let mlp = Self::from_parts(w1.to_vec(), b1.to_vec(), w2.to_vec(), b2.to_vec(), dev)?;
        Ok((mlp, sha))
    }
}

/// Standard-normal vector via Box–Muller over a seeded ChaCha8 stream (the
/// S1a/S2b probe generator, reused for golden inputs).
pub fn gaussian_vec(rng: &mut ChaCha8Rng, n: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(n);
    while v.len() < n {
        let u1: f64 = rng.gen_range(f64::MIN_POSITIVE..1.0);
        let u2: f64 = rng.gen::<f64>();
        let r = (-2.0 * u1.ln()).sqrt();
        let t = 2.0 * std::f64::consts::PI * u2;
        v.push((r * t.cos()) as f32);
        if v.len() < n {
            v.push((r * t.sin()) as f32);
        }
    }
    v
}

/// `(artifact_sha256, inputs, outputs)` from [`golden_pairs`].
pub type GoldenPairs = (String, Vec<Vec<f32>>, Vec<Vec<f32>>);

/// Golden input/output pairs produced by the trained network ITSELF (loaded
/// back from the artifact file, forward on CPU) — the probe-side
/// verification target, mirroring the S2b transform-artifact discipline.
pub fn golden_pairs(
    artifact: &std::path::Path,
    seed: u64,
    n_pairs: usize,
) -> anyhow::Result<GoldenPairs> {
    let cpu = Device::Cpu;
    let (mlp, sha) = Mlp::load_artifact(artifact, &cpu)?;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut inputs = Vec::with_capacity(n_pairs);
    let mut outputs = Vec::with_capacity(n_pairs);
    for _ in 0..n_pairs {
        let x = gaussian_vec(&mut rng, D_IN);
        let xt = Tensor::from_vec(x.clone(), (1, D_IN), &cpu)?;
        let y = mlp.forward(&xt)?.flatten_all()?.to_vec1::<f32>()?;
        inputs.push(x);
        outputs.push(y);
    }
    Ok((sha, inputs, outputs))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_param_count() {
        assert_eq!(PARAM_COUNT, 1_837_056);
        let mlp = Mlp::new_seeded(1, &Device::Cpu).unwrap();
        assert_eq!(mlp.param_count(), 1_837_056);
    }

    #[test]
    fn seeded_init_is_deterministic() {
        let a = Mlp::new_seeded(42, &Device::Cpu)
            .unwrap()
            .to_flat()
            .unwrap();
        let b = Mlp::new_seeded(42, &Device::Cpu)
            .unwrap()
            .to_flat()
            .unwrap();
        assert_eq!(a, b);
        let c = Mlp::new_seeded(43, &Device::Cpu)
            .unwrap()
            .to_flat()
            .unwrap();
        assert_ne!(a, c);
    }

    #[test]
    fn artifact_round_trip_and_hand_rolled_forward() {
        let dir = std::env::temp_dir().join(format!("lm-train-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mlp.f32bin");
        let mlp = Mlp::new_seeded(7, &Device::Cpu).unwrap();
        let sha_saved = mlp.save_artifact(&path).unwrap();
        let (loaded, sha_loaded) = Mlp::load_artifact(&path, &Device::Cpu).unwrap();
        assert_eq!(sha_saved, sha_loaded);
        assert_eq!(mlp.to_flat().unwrap(), loaded.to_flat().unwrap());

        // Hand-rolled plain-Rust forward (the probe-side implementation
        // spec) must agree with the candle forward to <=1e-5 relative L2.
        let flat = mlp.to_flat().unwrap();
        let (w1, rest) = flat.split_at(D_IN * D_HID);
        let (b1, rest) = rest.split_at(D_HID);
        let (w2, b2) = rest.split_at(D_HID * D_OUT);
        let mut rng = ChaCha8Rng::seed_from_u64(9);
        let x = gaussian_vec(&mut rng, D_IN);
        let mut h = b1.to_vec();
        for (i, &xi) in x.iter().enumerate() {
            for (hj, w) in h.iter_mut().zip(&w1[i * D_HID..(i + 1) * D_HID]) {
                *hj += xi * w;
            }
        }
        for hj in h.iter_mut() {
            *hj = hj.max(0.0);
        }
        let mut y = b2.to_vec();
        for (i, &hi) in h.iter().enumerate() {
            for (yj, w) in y.iter_mut().zip(&w2[i * D_OUT..(i + 1) * D_OUT]) {
                *yj += hi * w;
            }
        }
        let xt = Tensor::from_vec(x, (1, D_IN), &Device::Cpu).unwrap();
        let yc = mlp
            .forward(&xt)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let diff: f32 = y
            .iter()
            .zip(&yc)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            .sqrt();
        let norm: f32 = yc.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(diff / norm.max(1e-12) <= 1e-5, "rel err {}", diff / norm);
        std::fs::remove_dir_all(&dir).ok();
    }
}
