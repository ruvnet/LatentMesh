//! Bounded, deterministic wire primitives for LatentMesh Air.
//!
//! This crate intentionally performs no I/O. It can be built without `std`
//! and uses `alloc` only after validating wire-controlled lengths.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

mod crc32c;
mod envelope;
mod error;
mod fec;
mod fragment;
mod replay;
mod semantic;
mod wire;

pub use crc32c::crc32c;
pub use envelope::{SemanticEnvelope, ENVELOPE_HEADER_BYTES, ENVELOPE_SIGNATURE_BYTES};
pub use error::{AirError, Result};
pub use fec::{
    bits_to_bytes_msb, bytes_to_bits_msb, convolutional_encode, deinterleave_soft,
    hard_bits_to_llrs, interleave_bits, interleave_soft, soft_viterbi_decode, DecodeResult,
    MAX_CODED_BITS,
};
pub use fragment::{
    fragment_message, FragmentMeta, ReassembledMessage, Reassembler, ReassemblerConfig,
    MAX_FRAGMENTS, MAX_MESSAGE_BYTES,
};
pub use replay::{ReplayDecision, ReplayWindow};
pub use semantic::{
    CriticalState, Residual, SemanticDelta, SymbolUpdate, SymbolValue, CRITICAL_HASH_BYTES,
    MAX_CRITICAL_FIELDS, MAX_RESIDUALS, MAX_RESIDUAL_VALUES, MAX_SYMBOL_BYTES,
};
pub use wire::{
    state_hash_tag, FrameFlags, SemanticClass, SparseRadioFrame, WireProfile, FRAME_HEADER_BYTES,
    FRAME_MAX_BYTES, FRAME_MAX_PAYLOAD, FRAME_MIN_BYTES, PROTOCOL_MARKER,
};
