//! The `latentmesh-evolve` binary: run the deterministic Darwin acceptance
//! suite and emit the MetaHarness receipt (JSON) to stdout, or to a file with
//! `--receipt <path>`. `--seed <n>` overrides the frozen default seed. The
//! `harness/evolve` suite invokes this and verifies the receipt against
//! ADR-006's acceptance bounds.

use latentmesh_evolve::{evolve, DarwinConfig, Receipt, SyntheticEnv};

fn parse_args() -> Result<(Option<String>, u64), String> {
    let mut receipt_path = None;
    let mut seed = DarwinConfig::default().seed;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--receipt" => {
                receipt_path = Some(
                    args.next()
                        .ok_or_else(|| "--receipt needs a path".to_string())?,
                );
            }
            "--seed" => {
                let raw = args
                    .next()
                    .ok_or_else(|| "--seed needs a value".to_string())?;
                seed = raw.parse().map_err(|_| format!("invalid seed: {raw}"))?;
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    Ok((receipt_path, seed))
}

fn main() {
    let (receipt_path, seed) = match parse_args() {
        Ok(parsed) => parsed,
        Err(why) => {
            eprintln!("latentmesh-evolve: {why}");
            eprintln!("usage: latentmesh-evolve [--receipt <path>] [--seed <n>]");
            std::process::exit(2);
        }
    };

    let config = DarwinConfig {
        seed,
        ..Default::default()
    };
    let env = SyntheticEnv::new(seed);
    let outcome = evolve(&env, &config, None);
    let receipt = Receipt::from_outcome(&config, &outcome);
    let json = match receipt.to_json() {
        Ok(json) => json,
        Err(why) => {
            eprintln!("latentmesh-evolve: receipt serialization failed: {why}");
            std::process::exit(1);
        }
    };

    match receipt_path {
        Some(path) => {
            if let Err(why) = std::fs::write(&path, &json) {
                eprintln!("latentmesh-evolve: cannot write {path}: {why}");
                std::process::exit(1);
            }
            eprintln!(
                "receipt written to {path} (evidence: {}, acceptance passed: {})",
                receipt.evidence, receipt.acceptance.passed
            );
        }
        None => println!("{json}"),
    }
}
