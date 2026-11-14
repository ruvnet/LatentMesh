//! Native inference for Stanford Memorilla. Training remains upstream.
pub mod memory;
pub mod pipeline;
mod qwen3;
pub mod service;
mod weights;
pub mod wire;

pub use candle_core::{DType, Device, Result, Tensor};
pub use memory::{MemoryCache, MemoryConfig, MemoryModule};
pub use service::{MemoryService, ReadResult};

pub fn device(name: &str) -> Result<Device> {
    match name {
        "cpu" => Ok(Device::Cpu),
        #[cfg(feature = "metal")]
        "metal" => Device::new_metal(0),
        #[cfg(feature = "cuda")]
        "cuda" => Device::new_cuda(0),
        _ => candle_core::bail!("device {name:?} is unavailable in this build"),
    }
}

pub fn dtype(name: &str) -> Result<DType> {
    match name {
        "f32" => Ok(DType::F32),
        "f16" => Ok(DType::F16),
        "bf16" => Ok(DType::BF16),
        _ => candle_core::bail!("dtype must be f32, f16 or bf16"),
    }
}
