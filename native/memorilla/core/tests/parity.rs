use latentmesh_memorilla::{DType, Device, MemoryModule, MemoryService, Tensor};
use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Deserialize)]
struct Golden {
    documents: Vec<Vec<Vec<f32>>>,
    mask: Vec<Vec<u8>>,
    questions: Vec<Vec<f32>>,
    expected: Vec<Vec<Vec<f32>>>,
    cached: Option<Vec<Vec<Vec<f32>>>>,
    shared: Option<Vec<Vec<Vec<f32>>>>,
}
fn root(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../fixtures")
        .join(name)
}
fn flat(v: &[Vec<Vec<f32>>]) -> Vec<f32> {
    v.iter().flatten().flatten().copied().collect()
}
fn check(a: &Tensor, b: &[Vec<Vec<f32>>], tol: f32) {
    let a = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let b = flat(b);
    assert_eq!(a.len(), b.len());
    let max = a
        .iter()
        .zip(&b)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(max < tol, "maximum absolute error {max} >= {tol}");
    println!("max_absolute_error={max}");
}
fn run(weights: &Path, golden: &Path, tol: f32) {
    let m = MemoryModule::load(weights, &Device::Cpu, DType::F32).unwrap();
    let g: Golden = serde_json::from_slice(&std::fs::read(golden).unwrap()).unwrap();
    let b = g.documents.len();
    let n = g.documents[0].len();
    let d = g.questions[0].len();
    let docs = Tensor::from_vec(flat(&g.documents), (b, n, d), &Device::Cpu).unwrap();
    let mask = Tensor::from_vec(
        g.mask.iter().flatten().copied().collect::<Vec<_>>(),
        (b, n),
        &Device::Cpu,
    )
    .unwrap();
    let questions = Tensor::from_vec(
        g.questions.iter().flatten().copied().collect::<Vec<_>>(),
        (b, d),
        &Device::Cpu,
    )
    .unwrap();
    check(
        &m.forward(&docs, Some(&mask), &questions).unwrap(),
        &g.expected,
        tol,
    );
    if let Some(expected) = g.cached {
        let cache = m.write_cache(&docs, Some(&mask)).unwrap();
        check(&m.read_cache(&cache, &questions).unwrap(), &expected, tol);
        let cache = m
            .write_cache(
                &docs.narrow(0, 0, 1).unwrap(),
                Some(&mask.narrow(0, 0, 1).unwrap()),
            )
            .unwrap();
        check(
            &m.read_cache(&cache, &questions).unwrap(),
            &g.shared.unwrap(),
            tol,
        );
    } else {
        assert!(m.write_cache(&docs, Some(&mask)).is_err());
    }
}
#[test]
fn learned_matches_pytorch() {
    run(&root("learned"), &root("learned/golden.json"), 2e-5);
}
#[test]
fn retrieval_and_self_attention_match_pytorch() {
    run(&root("retrieval"), &root("retrieval/golden.json"), 2e-5);
}
#[test]
#[ignore = "set MEMORILLA_CHECKPOINT and generate the real oracle first"]
fn released_checkpoint_matches_pytorch() {
    run(
        Path::new(&std::env::var("MEMORILLA_CHECKPOINT").unwrap()),
        &root("real/golden.json"),
        2e-4,
    );
}
#[test]
fn invalid_inputs_and_scope_isolation() {
    let module = || MemoryModule::load(root("learned"), &Device::Cpu, DType::F32).unwrap();
    let mut a = MemoryService::new(module(), "tenant-a", 10000).unwrap();
    let b = MemoryService::new(module(), "tenant-b", 10000).unwrap();
    assert!(a.write("x", &[0.0; 7]).is_err());
    assert!(a.write("x", &[f32::NAN; 8]).is_err());
    a.write("x", &[0.1; 8]).unwrap();
    assert!(b.read("x", &[0.1; 8]).is_err());
    assert!(a.read("x", &[0.1; 7]).is_err());
    let used = a.cache_bytes();
    assert!(a.write("x", &vec![0.1; 8000]).is_err());
    assert_eq!(used, a.cache_bytes());
    assert!(a.read("x", &[0.1; 8]).is_ok());
    assert!(a.delete("x"));
    assert_eq!(a.cache_bytes(), 0);
    assert!(a.read("x", &[0.1; 8]).is_err());
    let docs = Tensor::zeros((1, 2, 8), DType::F32, &Device::Cpu).unwrap();
    let mask = Tensor::ones((1, 2), DType::U8, &Device::Cpu).unwrap();
    assert!(module().write_cache(&docs, Some(&mask)).is_err());
}
