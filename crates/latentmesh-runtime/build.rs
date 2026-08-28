//! Build-env guard (design doc 024 §3 / risk #4).
//!
//! candle 0.9.2 + cuda was empirically verified on this host's RTX 5080
//! (sm_120) ONLY with the CUDA 12.8 toolchain (`nvcc 12.8.93`, PATH-pinned to
//! `/usr/local/cuda-12.8/bin`). The host default nvcc is CUDA 13.0 and is
//! UNTESTED with candle 0.9.2/cudarc, so building the `cuda` feature against
//! it would silently produce an unverified artifact. This script fails the
//! build unless the `nvcc` found on PATH reports release 12.8.
//!
//! Caveat (recorded per advisor review): cargo may reuse candle-kernels build
//! artifacts cached from an earlier toolchain; the guard asserts the toolchain
//! visible at *this* build, and the runtime guard re-asserts it at process
//! start so receipts record the toolchain actually present.

fn main() {
    println!("cargo:rerun-if-env-changed=PATH");
    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        // CPU-only build: no CUDA toolchain requirement.
        println!("cargo:rustc-env=LATENTMESH_NVCC_RELEASE=none");
        return;
    }
    let out = std::process::Command::new("nvcc")
        .arg("--version")
        .output()
        .expect(
            "latentmesh-runtime[cuda]: `nvcc` not found on PATH. Pin the verified toolchain: \
             PATH=/usr/local/cuda-12.8/bin:$PATH (design doc 024 §2).",
        );
    let text = String::from_utf8_lossy(&out.stdout).to_string();
    let release = text
        .lines()
        .find_map(|l| l.split("release ").nth(1))
        .map(|r| r.split([',', ' ']).next().unwrap_or("").to_string())
        .unwrap_or_default();
    if !release.starts_with("12.8") {
        panic!(
            "latentmesh-runtime[cuda]: nvcc release {release:?} found, but only CUDA 12.8 \
             (nvcc 12.8.93) is verified with candle 0.9.2 on this GPU. \
             Pin PATH=/usr/local/cuda-12.8/bin:$PATH. Raw nvcc output:\n{text}"
        );
    }
    println!("cargo:rustc-env=LATENTMESH_NVCC_RELEASE={release}");
}
