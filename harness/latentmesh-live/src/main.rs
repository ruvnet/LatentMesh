//! latentmesh-live — workspace-side harness CLI for the live latent-exchange
//! experiment (design doc 024 §3). S2 scope: `make-splits` and `calibrate`.
//!
//! The runtime side (standalone crate `crates/latentmesh-runtime`, excluded
//! from this workspace — incompatible dep graph) produces the hidden-state
//! dumps this harness consumes; the boundary is crossed with files only.
//!
//! Usage:
//!   latentmesh-live make-splits
//!       Download GSM8K train/test (sha-verified), write the four committed
//!       index lists under data/ and the splits receipt.
//!   latentmesh-live calibrate [--dump-dir DIR] [--out FILE]
//!       Read the runtime S2 dump (default:
//!       crates/latentmesh-runtime/target/latentmesh-runs/s2), fit the 3x3
//!       depth sweep, gate A6, write the calibration receipt.

use latentmesh_live::{calibrate, gsm8k};
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("make-splits") => {
            let receipt = gsm8k::make_splits()?;
            let path = gsm8k::data_dir().join("splits-receipt.json");
            std::fs::write(&path, serde_json::to_string_pretty(&receipt)?)?;
            println!("splits receipt: {}", path.display());
            Ok(())
        }
        Some("calibrate") => {
            let mut dump_dir = gsm8k::crate_dir()
                .join("../../crates/latentmesh-runtime/target/latentmesh-runs/s2");
            let mut out: Option<PathBuf> = None;
            let mut i = 2;
            while i + 1 < args.len() {
                match args[i].as_str() {
                    "--dump-dir" => dump_dir = PathBuf::from(&args[i + 1]),
                    "--out" => out = Some(PathBuf::from(&args[i + 1])),
                    other => anyhow::bail!("unknown arg {other}"),
                }
                i += 2;
            }
            let receipt = calibrate::run(&dump_dir)?;
            let out = out.unwrap_or_else(|| dump_dir.join("s2-calibration-receipt.json"));
            std::fs::write(&out, serde_json::to_string_pretty(&receipt)?)?;
            println!("calibration receipt: {}", out.display());
            let pass = receipt["gate_A6"]["pass"].as_bool().unwrap_or(false);
            println!(
                "A6 {} (winner L{}->L{}, held-out residual {})",
                if pass {
                    "PASS"
                } else {
                    "FAIL — S2 kill-path, report honestly"
                },
                receipt["winner"]["sender_layer"],
                receipt["winner"]["receiver_layer"],
                receipt["winner"]["held_out_relative_residual"]
            );
            Ok(())
        }
        _ => {
            eprintln!(
                "usage: latentmesh-live <make-splits | calibrate [--dump-dir DIR] [--out FILE]>"
            );
            std::process::exit(2);
        }
    }
}
