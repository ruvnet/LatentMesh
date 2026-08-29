//! The M4 FastGRNN sequence translator (ADR-024, frozen architecture):
//! a low-rank FastGRNN cell (Kusupati et al., NeurIPS 2018,
//! arXiv:1901.02358; reference impl `microsoft/EdgeML`
//! `pytorch/edgeml_pytorch/graph/rnn.py FastGRNNCell`) with `D_in = 2048`
//! (sender L18 states) and `D_h = 1536` (receiver L14 states) — the hidden
//! state IS the translated receiver-side state, one per input token.
//!
//! Cell equations (verbatim from the M0 training-infra scout's primary-graded
//! fetch, `docs/research/026-run2-bootstrap-scouts.json`):
//!
//! ```text
//! pre     = W x_t + U h_{t-1}         (W, U SHARED between gate and candidate)
//! z       = sigmoid(pre + b_z)
//! h_tilde = tanh(pre + b_h)
//! h_t     = z * h_{t-1} + (sigmoid(zeta)*(1-z) + sigmoid(nu)) * h_tilde
//! ```
//!
//! with low-rank `W = W1 @ W2` (`W1: [D_in, r]`, `W2: [r, D_h]`) and
//! `U = U1 @ U2` (`U1: [D_h, r]`, `U2: [r, D_h]`), trainable raw scalars
//! `zeta`, `nu`. Parameter count `r(D_in + D_h) + 2·r·D_h + 2·D_h + 2`
//! (verified 429,058 at r=64 by the scout, asserted per rank below).
//!
//! Init (frozen here, recorded in every M4 training receipt): factor matrices
//! U(−1/√fan_in, +1/√fan_in) drawn host-side from one seeded ChaCha8 stream
//! in artifact-layout order; `b_z = b_h = 0`; raw `zeta = 1.0`, raw
//! `nu = -4.0` (the EdgeML reference defaults `zetaInit=1.0, nuInit=-4.0` —
//! the scout pinned the equations but not the init, so the reference
//! implementation's defaults are adopted and cited).
//!
//! Artifact: raw little-endian f32, layout [`ARTIFACT_LAYOUT`] — the repo's
//! raw-f32bin discipline (deterministic bytes, trivially content-hashable).

use candle_core::{Device, Tensor, Var};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use sha2::Digest;

pub const D_IN: usize = 2048;
pub const D_H: usize = 1536;
/// ADR-024 frozen sub-rung ladder, ascending order.
pub const RANKS: [usize; 3] = [64, 128, 256];
/// Artifact layout string, recorded in receipts.
pub const ARTIFACT_LAYOUT: &str = "little-endian f32: w1[2048xR row-major, x@W convention] || w2[Rx1536 row-major] || u1[1536xR row-major] || u2[Rx1536 row-major] || b_z[1536] || b_h[1536] || zeta_raw[1] || nu_raw[1]";

/// Frozen parameter-count formula: `r(D_in + D_h) + 2·r·D_h + 2·D_h + 2`.
pub const fn param_count(rank: usize) -> usize {
    rank * (D_IN + D_H) + 2 * rank * D_H + 2 * D_H + 2
}

/// The low-rank FastGRNN cell with trainable `Var`s on a device.
pub struct FastGrnn {
    pub rank: usize,
    pub w1: Var,
    pub w2: Var,
    pub u1: Var,
    pub u2: Var,
    pub b_z: Var,
    pub b_h: Var,
    pub zeta: Var,
    pub nu: Var,
}

impl FastGrnn {
    /// Seeded deterministic init (see module doc for the frozen scheme).
    pub fn new_seeded(rank: usize, seed: u64, dev: &Device) -> candle_core::Result<Self> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut uniform = |n: usize, a: f32| -> Vec<f32> {
            (0..n).map(|_| rng.gen::<f32>() * 2.0 * a - a).collect()
        };
        let w1 = uniform(D_IN * rank, 1.0 / (D_IN as f32).sqrt());
        let w2 = uniform(rank * D_H, 1.0 / (rank as f32).sqrt());
        let u1 = uniform(D_H * rank, 1.0 / (D_H as f32).sqrt());
        let u2 = uniform(rank * D_H, 1.0 / (rank as f32).sqrt());
        let mut flat = Vec::with_capacity(param_count(rank));
        flat.extend_from_slice(&w1);
        flat.extend_from_slice(&w2);
        flat.extend_from_slice(&u1);
        flat.extend_from_slice(&u2);
        flat.extend(std::iter::repeat(0f32).take(2 * D_H)); // b_z, b_h
        flat.push(1.0); // zeta_raw (EdgeML default)
        flat.push(-4.0); // nu_raw (EdgeML default)
        Self::from_flat(rank, &flat, dev)
    }

    /// Build from a flat artifact-layout buffer.
    pub fn from_flat(rank: usize, flat: &[f32], dev: &Device) -> candle_core::Result<Self> {
        assert_eq!(flat.len(), param_count(rank), "flat buffer != param count");
        let (w1, rest) = flat.split_at(D_IN * rank);
        let (w2, rest) = rest.split_at(rank * D_H);
        let (u1, rest) = rest.split_at(D_H * rank);
        let (u2, rest) = rest.split_at(rank * D_H);
        let (b_z, rest) = rest.split_at(D_H);
        let (b_h, rest) = rest.split_at(D_H);
        let v = |data: &[f32], shape: (usize, usize)| -> candle_core::Result<Var> {
            Var::from_tensor(&Tensor::from_vec(data.to_vec(), shape, dev)?)
        };
        Ok(Self {
            rank,
            w1: v(w1, (D_IN, rank))?,
            w2: v(w2, (rank, D_H))?,
            u1: v(u1, (D_H, rank))?,
            u2: v(u2, (rank, D_H))?,
            b_z: Var::from_tensor(&Tensor::from_vec(b_z.to_vec(), (D_H,), dev)?)?,
            b_h: Var::from_tensor(&Tensor::from_vec(b_h.to_vec(), (D_H,), dev)?)?,
            zeta: Var::from_tensor(&Tensor::new(rest[0], dev)?)?,
            nu: Var::from_tensor(&Tensor::new(rest[1], dev)?)?,
        })
    }

    pub fn vars(&self) -> Vec<Var> {
        vec![
            self.w1.clone(),
            self.w2.clone(),
            self.u1.clone(),
            self.u2.clone(),
            self.b_z.clone(),
            self.b_h.clone(),
            self.zeta.clone(),
            self.nu.clone(),
        ]
    }

    pub fn param_count(&self) -> usize {
        self.vars().iter().map(|v| v.elem_count()).sum()
    }

    /// One cell step: `x (B, 2048)`, `h (B, 1536)` → new `h (B, 1536)`.
    pub fn step(&self, x: &Tensor, h: &Tensor) -> candle_core::Result<Tensor> {
        let pre = x
            .matmul(self.w1.as_tensor())?
            .matmul(self.w2.as_tensor())?
            .add(&h.matmul(self.u1.as_tensor())?.matmul(self.u2.as_tensor())?)?;
        let z = candle_nn::ops::sigmoid(&pre.broadcast_add(self.b_z.as_tensor())?)?;
        let h_tilde = pre.broadcast_add(self.b_h.as_tensor())?.tanh()?;
        let sz = candle_nn::ops::sigmoid(self.zeta.as_tensor())?;
        let sn = candle_nn::ops::sigmoid(self.nu.as_tensor())?;
        let coef = z
            .affine(-1.0, 1.0)? // 1 - z
            .broadcast_mul(&sz)?
            .broadcast_add(&sn)?;
        z.mul(h)?.add(&coef.mul(&h_tilde)?)
    }

    /// Full sequence forward from `h_0 = 0`: `xs (B, T, 2048)` → stacked
    /// outputs `(B, T, 1536)`. `detach_between_steps` truncates the autograd
    /// graph at every step (holdout/inference use — the training loop keeps
    /// the graph across its 16-step BPTT window by passing `false`).
    pub fn forward_seq(
        &self,
        xs: &Tensor,
        detach_between_steps: bool,
    ) -> candle_core::Result<Tensor> {
        let (b, t, _) = xs.dims3()?;
        let mut h = Tensor::zeros((b, D_H), candle_core::DType::F32, xs.device())?;
        let mut outs = Vec::with_capacity(t);
        for i in 0..t {
            // `.contiguous()`: a narrowed view keeps the parent stride, and
            // CUDA matmul requires contiguous inputs.
            let x_t = xs.narrow(1, i, 1)?.squeeze(1)?.contiguous()?;
            h = self.step(&x_t, &h)?;
            if detach_between_steps {
                h = h.detach();
            }
            outs.push(h.clone());
        }
        Tensor::stack(&outs, 1)
    }

    /// Host-side copy of all weights in artifact layout order.
    pub fn to_flat(&self) -> candle_core::Result<Vec<f32>> {
        let mut out = Vec::with_capacity(self.param_count());
        for t in [
            &self.w1, &self.w2, &self.u1, &self.u2, &self.b_z, &self.b_h, &self.zeta, &self.nu,
        ] {
            out.extend(t.as_tensor().flatten_all()?.to_vec1::<f32>()?);
        }
        Ok(out)
    }

    /// Overwrite the `Var`s from a flat artifact-layout buffer (restores the
    /// best-holdout-epoch checkpoint before saving).
    pub fn set_flat(&self, flat: &[f32], dev: &Device) -> candle_core::Result<()> {
        assert_eq!(flat.len(), param_count(self.rank));
        let r = self.rank;
        let (w1, rest) = flat.split_at(D_IN * r);
        let (w2, rest) = rest.split_at(r * D_H);
        let (u1, rest) = rest.split_at(D_H * r);
        let (u2, rest) = rest.split_at(r * D_H);
        let (b_z, rest) = rest.split_at(D_H);
        let (b_h, rest) = rest.split_at(D_H);
        self.w1
            .set(&Tensor::from_vec(w1.to_vec(), (D_IN, r), dev)?)?;
        self.w2
            .set(&Tensor::from_vec(w2.to_vec(), (r, D_H), dev)?)?;
        self.u1
            .set(&Tensor::from_vec(u1.to_vec(), (D_H, r), dev)?)?;
        self.u2
            .set(&Tensor::from_vec(u2.to_vec(), (r, D_H), dev)?)?;
        self.b_z
            .set(&Tensor::from_vec(b_z.to_vec(), (D_H,), dev)?)?;
        self.b_h
            .set(&Tensor::from_vec(b_h.to_vec(), (D_H,), dev)?)?;
        self.zeta.set(&Tensor::new(rest[0], dev)?)?;
        self.nu.set(&Tensor::new(rest[1], dev)?)?;
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

    /// Load an artifact file onto `dev` (rank inferred from the byte count);
    /// returns `(cell, sha256)`.
    pub fn load_artifact(path: &std::path::Path, dev: &Device) -> anyhow::Result<(Self, String)> {
        let bytes = std::fs::read(path)?;
        anyhow::ensure!(bytes.len() % 4 == 0, "artifact not f32-aligned");
        let n = bytes.len() / 4;
        anyhow::ensure!(
            n >= 2 * D_H + 2 && (n - 2 * D_H - 2) % (D_IN + 3 * D_H) == 0,
            "artifact {} floats does not match the FastGRNN param formula",
            n
        );
        let rank = (n - 2 * D_H - 2) / (D_IN + 3 * D_H);
        anyhow::ensure!(param_count(rank) == n);
        let sha = format!("{:x}", sha2::Sha256::digest(&bytes));
        let flat: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Ok((Self::from_flat(rank, &flat, dev)?, sha))
    }
}

/// `(artifact_sha256, inputs, outputs, pooled)` from [`golden_pairs`]:
/// per golden sequence the per-step inputs `[T][2048]`, the per-step
/// translated outputs `[T][1536]`, and the mean-pooled payload `[1536]`
/// (f64 accumulation — the exact probe-side pooling).
pub type GoldenSeqPairs = (
    String,
    Vec<Vec<Vec<f32>>>,
    Vec<Vec<Vec<f32>>>,
    Vec<Vec<f32>>,
);

/// Golden sequence pairs produced by the trained network ITSELF (artifact
/// reloaded from disk, candle CPU forward) — the probe-side verification
/// target, extending the S2b/M3 golden discipline to sequences: both every
/// per-step output AND the pooled injection payload are pinned.
pub fn golden_pairs(
    artifact: &std::path::Path,
    seed: u64,
    n_seqs: usize,
    seq_len: usize,
) -> anyhow::Result<GoldenSeqPairs> {
    let cpu = Device::Cpu;
    let (cell, sha) = FastGrnn::load_artifact(artifact, &cpu)?;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut inputs = Vec::with_capacity(n_seqs);
    let mut outputs = Vec::with_capacity(n_seqs);
    let mut pooled = Vec::with_capacity(n_seqs);
    for _ in 0..n_seqs {
        let xs: Vec<Vec<f32>> = (0..seq_len)
            .map(|_| crate::mlp::gaussian_vec(&mut rng, D_IN))
            .collect();
        let flat: Vec<f32> = xs.iter().flatten().copied().collect();
        let xt = Tensor::from_vec(flat, (1, seq_len, D_IN), &cpu)?;
        let ys = cell.forward_seq(&xt, true)?; // (1, T, D_H)
        let mut seq_out = Vec::with_capacity(seq_len);
        let mut acc = vec![0f64; D_H];
        for t in 0..seq_len {
            let y = ys.narrow(1, t, 1)?.flatten_all()?.to_vec1::<f32>()?;
            for (a, v) in acc.iter_mut().zip(&y) {
                *a += *v as f64;
            }
            seq_out.push(y);
        }
        let pool: Vec<f32> = acc.iter().map(|a| (*a / seq_len as f64) as f32).collect();
        inputs.push(xs);
        outputs.push(seq_out);
        pooled.push(pool);
    }
    Ok((sha, inputs, outputs, pooled))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_param_counts() {
        // ADR-024 / M0 scout: predicted and measured identically at r=64.
        assert_eq!(param_count(64), 429_058);
        assert_eq!(param_count(128), 855_042);
        assert_eq!(param_count(256), 1_707_010);
        let cell = FastGrnn::new_seeded(64, 1, &Device::Cpu).unwrap();
        assert_eq!(cell.param_count(), 429_058);
    }

    #[test]
    fn seeded_init_is_deterministic_with_frozen_scalars() {
        let a = FastGrnn::new_seeded(64, 42, &Device::Cpu).unwrap();
        let b = FastGrnn::new_seeded(64, 42, &Device::Cpu).unwrap();
        let fa = a.to_flat().unwrap();
        assert_eq!(fa, b.to_flat().unwrap());
        assert_ne!(
            fa,
            FastGrnn::new_seeded(64, 43, &Device::Cpu)
                .unwrap()
                .to_flat()
                .unwrap()
        );
        // Frozen init tail: b_z = b_h = 0, zeta_raw = 1.0, nu_raw = -4.0.
        let n = fa.len();
        assert!(fa[n - 2 - 2 * D_H..n - 2].iter().all(|v| *v == 0.0));
        assert_eq!(fa[n - 2], 1.0);
        assert_eq!(fa[n - 1], -4.0);
    }

    #[test]
    fn artifact_round_trip_and_hand_rolled_forward() {
        let dir = std::env::temp_dir().join(format!("lm-m4-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("fastgrnn.f32bin");
        let cell = FastGrnn::new_seeded(64, 7, &Device::Cpu).unwrap();
        let sha_saved = cell.save_artifact(&path).unwrap();
        let (loaded, sha_loaded) = FastGrnn::load_artifact(&path, &Device::Cpu).unwrap();
        assert_eq!(loaded.rank, 64);
        assert_eq!(sha_saved, sha_loaded);
        assert_eq!(cell.to_flat().unwrap(), loaded.to_flat().unwrap());

        // Hand-rolled plain-Rust sequential forward (the probe-side
        // implementation spec) must agree with the candle forward to <=1e-5
        // relative L2 at every step of a 32-step sequence.
        let flat = cell.to_flat().unwrap();
        let r = 64usize;
        let (w1, rest) = flat.split_at(D_IN * r);
        let (w2, rest) = rest.split_at(r * D_H);
        let (u1, rest) = rest.split_at(D_H * r);
        let (u2, rest) = rest.split_at(r * D_H);
        let (b_z, rest) = rest.split_at(D_H);
        let (b_h, rest) = rest.split_at(D_H);
        let (zeta_raw, nu_raw) = (rest[0], rest[1]);
        let sig = |v: f32| 1.0 / (1.0 + (-v).exp());
        let (sz, sn) = (sig(zeta_raw), sig(nu_raw));
        let seq_len = 32;
        let mut rng = ChaCha8Rng::seed_from_u64(9);
        let xs: Vec<Vec<f32>> = (0..seq_len)
            .map(|_| crate::mlp::gaussian_vec(&mut rng, D_IN))
            .collect();
        let flat_x: Vec<f32> = xs.iter().flatten().copied().collect();
        let xt = Tensor::from_vec(flat_x, (1, seq_len, D_IN), &Device::Cpu).unwrap();
        let ys = cell.forward_seq(&xt, true).unwrap();

        let mut h = vec![0f32; D_H];
        for (t, x) in xs.iter().enumerate() {
            // pre = (x@W1)@W2 + (h@U1)@U2
            let mut xr = vec![0f32; r];
            for (i, &xi) in x.iter().enumerate() {
                for (o, w) in xr.iter_mut().zip(&w1[i * r..(i + 1) * r]) {
                    *o += xi * w;
                }
            }
            let mut hr = vec![0f32; r];
            for (i, &hi) in h.iter().enumerate() {
                for (o, w) in hr.iter_mut().zip(&u1[i * r..(i + 1) * r]) {
                    *o += hi * w;
                }
            }
            let mut pre = vec![0f32; D_H];
            for (i, &v) in xr.iter().enumerate() {
                for (o, w) in pre.iter_mut().zip(&w2[i * D_H..(i + 1) * D_H]) {
                    *o += v * w;
                }
            }
            for (i, &v) in hr.iter().enumerate() {
                for (o, w) in pre.iter_mut().zip(&u2[i * D_H..(i + 1) * D_H]) {
                    *o += v * w;
                }
            }
            for j in 0..D_H {
                let z = sig(pre[j] + b_z[j]);
                let ht = (pre[j] + b_h[j]).tanh();
                h[j] = z * h[j] + (sz * (1.0 - z) + sn) * ht;
            }
            let yc = ys
                .narrow(1, t, 1)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let diff: f32 = h
                .iter()
                .zip(&yc)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>()
                .sqrt();
            let norm: f32 = yc.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!(
                diff / norm.max(1e-12) <= 1e-5,
                "step {t} rel err {}",
                diff / norm
            );
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
