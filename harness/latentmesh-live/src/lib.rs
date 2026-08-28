//! latentmesh-live — workspace-side harness library for the live
//! latent-exchange experiment (design doc 024 §3). See the binary
//! (`src/main.rs`) for the CLI; the library form exists so later stages
//! (S3–S6 conditions, audit, stats) and integration tests can call the
//! split/calibration machinery directly — including the §6 mechanical
//! refusal of eval/holdout items before genome freeze
//! ([`gsm8k::eval_items`], [`gsm8k::holdout_items`]).

pub mod calibrate;
pub mod calibrate_gen;
pub mod gsm8k;
