use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{LayerNorm, Linear};
use sha2::{Digest, Sha256};
use std::{collections::HashMap, fs::File, io::Read, path::Path};

pub fn file_hash(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hash = Sha256::new();
    let mut buf = [0u8; 65536];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hash.update(&buf[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

pub(crate) struct Weights {
    tensors: HashMap<String, Tensor>,
    device: Device,
    dtype: DType,
}

impl Weights {
    pub fn load(path: &Path, device: &Device, dtype: DType) -> Result<Self> {
        // Candle parses a restricted tensor pickle vocabulary; no Python runs.
        let tensors = if path.extension().and_then(|s| s.to_str()) == Some("safetensors") {
            candle_core::safetensors::load(path, &Device::Cpu)?
        } else {
            candle_core::pickle::read_all(path)?.into_iter().collect()
        };
        Ok(Self {
            tensors,
            device: device.clone(),
            dtype,
        })
    }
    pub fn tensor(&mut self, name: &str, dims: &[usize]) -> Result<Tensor> {
        let t = self
            .tensors
            .remove(name)
            .ok_or_else(|| candle_core::Error::Msg(format!("missing weight {name}")))?;
        if t.dims() != dims {
            candle_core::bail!("weight {name}: expected {dims:?}, got {:?}", t.dims());
        }
        t.to_device(&self.device)?.to_dtype(self.dtype)
    }
    pub fn linear(&mut self, name: &str, input: usize, output: usize) -> Result<Linear> {
        Ok(Linear::new(
            self.tensor(&format!("{name}.weight"), &[output, input])?,
            Some(self.tensor(&format!("{name}.bias"), &[output])?),
        ))
    }
    pub fn norm(&mut self, name: &str, dim: usize) -> Result<LayerNorm> {
        Ok(LayerNorm::new(
            self.tensor(&format!("{name}.weight"), &[dim])?,
            self.tensor(&format!("{name}.bias"), &[dim])?,
            1e-5,
        ))
    }
    pub fn finish(self) -> Result<()> {
        if !self.tensors.is_empty() {
            let mut keys: Vec<_> = self.tensors.keys().collect();
            keys.sort();
            candle_core::bail!("unexpected checkpoint weights: {keys:?}");
        }
        Ok(())
    }
}
