# Native Memorilla for LatentMesh

Rust inference for the Stanford Memorilla memory model, with cached document attention and a native Qwen3 input embedding pipeline. No Python or network is used during inference.

```rust,no_run
use latentmesh_memorilla::{Device, DType, MemoryModule, MemoryService, Result};
fn main() -> Result<()> {
    let module = MemoryModule::load("/models/memorilla/stage3", &Device::Cpu, DType::F32)?;
    let mut memory = MemoryService::new(module, "tenant:acme", 1_073_741_824)?;
    // Supply vectors from the checkpoint's matching Qwen3 Embedding 4B encoder.
    memory.write("history", &[0.1; 2560])?;
    let tokens = memory.read("history", &[0.2; 2560])?;
    assert_eq!(tokens.shape, [1, 16, 4096]);
    Ok(())
}
```

CPU is the default backend. Optional `metal` and `cuda` features select additional build backends. F32 CPU and Metal memory inference have numerical parity evidence. CUDA, reduced precision, and full pretrained text task quality need deployment evaluation. Use the exact encoder and decoder pair used to train the memory checkpoint.

A scope string provides handle isolation, not authentication. Authorize access in the calling application. The cache budget excludes model weights and caller owned inputs. Wire export always uses ObserveOnly authority.

[Complete integration guide, TypeScript bindings and benchmark evidence](https://github.com/ruvnet/LatentMesh/tree/main/native/memorilla).

Acceptance test: `cargo test --locked` runs committed independent PyTorch and Transformers fixtures. Released checkpoint validation is an explicit ignored test requiring separately downloaded weights and an oracle.
