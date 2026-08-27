// Shell out to a Rust example driver (`cargo run --release --example
// e2e_loopback -p <crate>`) and parse the one JSON object it prints to
// stdout, exactly the way harness/evolve's scripts/run.mjs shells out to
// the latentmesh-evolve binary. Kept as one shared helper so the three
// per-suite runners (run-meshtastic.mjs, run-agentbbs.mjs,
// run-cognitum.mjs) don't each re-implement subprocess plumbing.

import { execFileSync } from "node:child_process";
import { join } from "node:path";

export const repoRoot = (here) => join(here, "..", "..", "..");

/**
 * Runs `cargo run --release --quiet -p <pkg> --example <example> [--features <features>]`
 * from `cwd`, and parses the last non-empty stdout line as JSON (the
 * example's `println!("{output}")`; `cargo run --quiet` still lets a
 * fresh-build's `Compiling ...` lines through on stderr, which this
 * function ignores by inheriting stderr rather than capturing it).
 */
export function runExampleDriver({ cwd, pkg, example, features }) {
  const args = ["run", "--release", "--quiet", "-p", pkg, "--example", example];
  if (features) args.push("--features", features);
  const stdout = execFileSync("cargo", args, {
    cwd,
    encoding: "utf8",
    stdio: ["ignore", "pipe", "inherit"],
  });
  const lines = stdout.split("\n").map((line) => line.trim()).filter(Boolean);
  if (lines.length === 0) {
    throw new Error(`cargo run -p ${pkg} --example ${example} produced no stdout`);
  }
  const last = lines[lines.length - 1];
  try {
    return JSON.parse(last);
  } catch (err) {
    throw new Error(`failed to parse driver JSON output from -p ${pkg} --example ${example}: ${err.message}\noutput: ${last}`);
  }
}
