//! Measured, reversible-by-choice fidelity reduction (ADR-016): compression
//! is a re-encode through `latentmesh-core`'s real quantizers — the same
//! `F16`/`Int8` machinery the wire uses — so the fidelity lost is exactly the
//! quantization error, and it is measured and recorded, never silent.

use latentmesh_core::{Encoding, Payload};

/// The result of one lossy step.
#[derive(Clone, Debug)]
pub struct CompressionOutcome {
    /// The re-encoded latent (decoded back to `f32` working form).
    pub latent: Vec<f32>,
    /// Mean absolute reconstruction error of this step.
    pub mean_abs_error: f32,
    /// Bytes the latent would occupy on the wire at the chosen encoding.
    pub wire_bytes: usize,
}

/// Re-encode `latent` at `encoding` and measure what was lost. `F32` is a
/// no-op with zero error (useful as the identity fidelity step in tests).
pub fn compress_latent(latent: &[f32], encoding: Encoding) -> CompressionOutcome {
    let payload = Payload::encode(latent, encoding);
    let decoded = payload.decode();
    let mean_abs_error = if latent.is_empty() {
        0.0
    } else {
        latent
            .iter()
            .zip(decoded.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / latent.len() as f32
    };
    CompressionOutcome {
        latent: decoded,
        mean_abs_error,
        wire_bytes: payload.wire_bytes(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_is_the_identity_step() {
        let v = vec![0.5, -1.25, 3.0];
        let out = compress_latent(&v, Encoding::F32);
        assert_eq!(out.latent, v);
        assert_eq!(out.mean_abs_error, 0.0);
    }

    #[test]
    fn int8_shrinks_wire_bytes_and_reports_nonzero_error() {
        let v: Vec<f32> = (0..256).map(|i| (i as f32) / 17.0 - 7.0).collect();
        let f32_bytes = compress_latent(&v, Encoding::F32).wire_bytes;
        let out = compress_latent(&v, Encoding::Int8);
        assert!(out.wire_bytes < f32_bytes / 3);
        assert!(out.mean_abs_error > 0.0);
        assert!(out.mean_abs_error < 0.05, "int8 error should stay small");
    }
}
