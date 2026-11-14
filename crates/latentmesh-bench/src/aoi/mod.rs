//! ADR-049 deterministic Age-of-Information / task-relevance benchmark.
//!
//! `latentmesh-bench aoi [--seeds 11,23,37,53,71] [--out PATH] [--quick]`.
//! Evidence label is `simulated-deterministic`: no radio is involved.

mod canonical;
mod grid;
mod receipt;
mod sim;
mod utility;
mod workload;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod adversarial;

use std::path::PathBuf;

use grid::Axes;
use receipt::{build_receipt, git_info, ReceiptSpec};

const DEFAULT_SEEDS: [u64; 5] = [11, 23, 37, 53, 71];
const MIN_SEEDS: usize = 5;

#[derive(Debug)]
pub struct Options {
    pub seeds: Vec<u64>,
    pub out: Option<PathBuf>,
    pub quick: bool,
}

pub fn parse_args(args: &[String]) -> Result<Options, String> {
    let mut o = Options {
        seeds: DEFAULT_SEEDS.to_vec(),
        out: None,
        quick: false,
    };
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--quick" => o.quick = true,
            "--out" => o.out = Some(it.next().ok_or("--out needs a path")?.into()),
            "--seeds" => {
                let v = it.next().ok_or("--seeds needs a list")?;
                o.seeds = v
                    .split(',')
                    .map(|s| {
                        s.trim()
                            .parse::<u64>()
                            .map_err(|e| format!("bad seed {s:?}: {e}"))
                    })
                    .collect::<Result<_, _>>()?;
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    if o.seeds.is_empty() {
        return Err("--seeds is empty".into());
    }
    if !o.quick && o.seeds.len() < MIN_SEEDS {
        return Err(format!(
            "at least {MIN_SEEDS} seeds are required outside --quick"
        ));
    }
    Ok(o)
}

/// Canonical receipt text for `opts` bound to `git`.
pub fn receipt_text(opts: &Options, git: &receipt::GitInfo) -> Result<String, String> {
    let axes = if opts.quick {
        Axes::quick()
    } else {
        Axes::full()
    };
    let seeds: Vec<String> = opts.seeds.iter().map(u64::to_string).collect();
    let command = format!(
        "latentmesh-bench aoi --seeds {}{}",
        seeds.join(","),
        if opts.quick { " --quick" } else { "" }
    );
    let spec = ReceiptSpec {
        seeds: &opts.seeds,
        axes: &axes,
        frozen: !opts.quick,
        command: &command,
        git,
    };
    let value = build_receipt(&spec)?;
    Ok(canonical::canonical_json(&value)? + "\n")
}

/// Entry point for the `aoi` subcommand.
pub fn main(args: &[String]) -> Result<(), String> {
    let opts = parse_args(args)?;
    let git = git_info()?;
    let out = opts.out.clone().unwrap_or_else(|| {
        let short: String = git.commit.chars().take(7).collect();
        PathBuf::from(format!(
            "crates/latentmesh-bench/receipts/aoi-bench-v1-{short}.json"
        ))
    });
    let text = receipt_text(&opts, &git)?;
    if let Some(dir) = out.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("create {dir:?}: {e}"))?;
    }
    std::fs::write(&out, &text).map_err(|e| format!("write {out:?}: {e}"))?;
    println!(
        "wrote {} ({} bytes, frozen={})",
        out.display(),
        text.len(),
        !opts.quick
    );
    Ok(())
}
