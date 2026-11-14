use latentmesh_memorilla::{DType, Device, MemoryModule, MemoryService, Result};

fn main() -> Result<()> {
    let checkpoint = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../fixtures/learned");
    let module = MemoryModule::load(checkpoint, &Device::Cpu, DType::F32)?;
    let mut service = MemoryService::new(module, "example", 1_000_000)?;
    service.write("history", &[0.1; 24])?;
    let result = service.read("history", &[0.2; 8])?;
    assert_eq!(result.shape, [1, 3, 12]);
    println!(
        "shape={:?}, first_values={:?}",
        result.shape,
        &result.values[..4]
    );
    Ok(())
}
