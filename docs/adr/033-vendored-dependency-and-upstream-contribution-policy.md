# 033. Vendored-dependency and upstream-contribution policy

- **Status**: Proposed
- **Date**: 2026-08-29
- **Related**: [031](031-evidence-receipt-and-statistical-protocol-governance.md) (the receipt
  contract a vendoring deviation's re-verification evidence must satisfy),
  [023](023-live-four-condition-run1-pre-registration.md) Deviations 1 and 2 (the MSRV/`half`
  conflict that motivates the vendor-and-exclude pattern, and the RoPE fix this ADR's policy is
  built from), [024](024-run2-trained-thought-adapter-ladder.md) (`latentmesh-train`'s Cargo.lock
  copied verbatim from `latentmesh-runtime`'s — the version-pinning half of this policy already
  practiced once)
- **Evidence base**: `crates/latentmesh-runtime/src/models/qwen2_a.rs` lines 1-30 (module header,
  read in full — provenance block and declared-deviations list) and `qwen2_b.rs` (the file split's
  other half); ADR-023 Deviations 1 and 2 (the workspace-exclusion ruling and the RoPE fix's
  discovery, diagnosis, and uniform-application argument)

## Context

`crates/latentmesh-runtime` vendors `candle-transformers` 0.9.2's `qwen2.rs` model file, split
across `qwen2_a.rs` (`Config`, `RotaryEmbedding`, `MLP`, `Attention` — source lines 17-215) and
`qwen2_b.rs` (`DecoderLayer`, `Model`, `ModelForCausalLM`) purely to satisfy this repository's
under-500-line-per-file rule — "the numerics are unchanged" by the split itself, per the module
header. The vendored copy carries a documented deviation ledger of four items, three numerically
inert (delegating through `with_tracing` wrappers directly to the underlying `candle_nn` types;
inlining `repeat_kv` verbatim; loosening `pub(super)` visibility so the two split files can share
types) and one genuinely **output-changing**: the vendored 0.9.2 original casts rotary-embedding
position indices to the model's storage dtype *before* the outer product with `inv_freq`; in BF16
(8 mantissa bits), integer positions above roughly 256 alias to identical float values, so distinct
token positions receive identical rotary angles — a real correctness bug, not a style deviation.
This repository's copy builds the sin/cos tables in F32 and casts only the final result, the pattern
`candle`'s own `llama.rs` already uses, and the fix was found and confirmed *live*, on this host,
diagnosing a real S1a run-1 failure (`s1a-receipt-run1-buggy-rope-noncompliant-prompt.json`):
greedy Qwen2.5-1.5B generations degenerating into token duplication ("y = = 2 x 111",
"goldfishfish") once absolute position passed ~256, cured by the F32-table fix on the identical
items. Nothing in this repository states a general policy for when vendoring is acceptable, what a
deviation ledger must contain, or what obligation a genuine upstream bug fix creates. This ADR
states that policy, using the RoPE fix as the concrete worked example throughout.

## Decision

### When vendoring is allowed

Vendoring a third-party crate's source (rather than depending on it normally) is permitted **only**
when a normal dependency is structurally impossible, not merely inconvenient. The precedent this
repository already has is exactly this bar: `candle-core` 0.9.2 requires `half >= 2.5` (itself
`rust-version = 1.81`), which cannot unify in one `Cargo.lock` with `latentmesh-core`'s
`half = 2.4.1` pin under the workspace's MSRV floor of 1.77 (ADR-023 Deviation 1). The fix that
ADR-023 actually chose was **crate exclusion** (`latentmesh-runtime` and, per ADR-024, later
`latentmesh-train`, both live outside the root workspace as their own single-crate workspaces) —
not vendoring `candle` itself. What *is* vendored in `qwen2_a.rs`/`qwen2_b.rs` is `candle`'s own
*model definition* (the Qwen2 architecture file), which is not published as a reusable library
type by `candle-transformers` in a form this repository can otherwise compose against without
copying it — model files in that crate are examples/reference implementations, not a stable public
API surface designed for downstream extension. This is the general rule: vendor a *file*, not a
*dependency graph*, and only when the alternative is either (a) a version-resolution conflict
exclusion cannot solve, or (b) upstream genuinely does not expose the needed surface as a normal
dependency.

### The deviation-ledger requirement

Every vendored file **must** carry, at its top, exactly the block `qwen2_a.rs` already has:

1. **Provenance**: source crate name, exact version, registry path, license, and which lines of the
   original the file covers (if split).
2. **An enumerated, numbered list of every deviation from verbatim**, each one classified as either
   *numerically inert* (a refactor, a visibility change, an inlined helper — behavior-preserving by
   construction or by argument) or **output-changing** (measurably alters computed results). An
   output-changing deviation must state: what the original does, what this copy does instead, why,
   and what evidence confirms the fix (`qwen2_a.rs` deviation 4 gives all four for the RoPE fix, in
   four sentences).
3. Reference to the file's own doc-comment being the single source of truth for the ledger — not
   duplicated, and liable to drift, in an ADR or a research doc. This ADR states the *policy*; the
   file itself remains the record.

**Output-changing deviations must be registered in the affected experiment's ADR, and applied
uniformly.** ADR-023 Deviation 2 is the model: the RoPE fix is stated as "output-changing — it
measurably changes generated text — but it is applied identically to every model, every condition,
and every stage from S0 onward (S0's mechanics gates were re-run and stayed green after the
change); it is a bug fix to the vendored numerics, not a condition-specific intervention, so it does
not introduce a between-condition confound." **This is the load-bearing generalization this ADR
adds**: an output-changing vendored deviation is safe to keep only if (a) it is registered in every
ADR whose experiment it touches, and (b) uniform application across every model and every condition
is *explicitly asserted and checked* (by re-running the mechanics gates, per ADR-031(a)'s gate
contract), not assumed because "it's just a bug fix." A deviation applied to only one condition, one
model, or one stage is a confound, not a fix, regardless of how obviously correct the underlying
numerics argument is.

### Version pinning

A vendored file is pinned to the exact upstream version stated in its provenance block
(`candle-transformers 0.9.2`, unambiguously). Any crate that depends on the vendoring crate's
compiled output but needs the *same* underlying library version for its own reasons (M6's
`latentmesh-train` linking `latentmesh-runtime` directly, per ADR-024) **copies the vendoring
crate's `Cargo.lock` verbatim rather than re-resolving from scratch** — already practiced
(ADR-024: "not re-resolved from scratch, to avoid pulling an unproven point release against nvcc
12.8/sm_120"). This is now a standing rule, not a one-off convenience: a lockfile carrying a proven
GPU-toolchain-compatible dependency graph is itself evidence, and re-resolution discards it for no
benefit.

### The upstream-contribution obligation

The RoPE fix is a genuine correctness bug in `candle-transformers` 0.9.2's Qwen2 rotary-embedding
implementation, and the obligation this repository takes on by finding it is to **contribute it
back to `huggingface/candle`**, not merely to carry the local patch indefinitely. A contribution
requires, in order:

1. **Verify the bug still exists in `candle` main** — **UNVERIFIED as of this ADR.** This repository
   has confirmed the bug in the *pinned 0.9.2 release* it vendors, live, on real generations. It has
   not checked whether `candle`'s current main branch (or any release after 0.9.2) already fixed
   this independently — `candle`'s own `llama.rs` already uses the F32-table pattern this repository
   adopted, which raises a live possibility that a later Qwen2 release also picked it up. **This ADR
   states plainly: whether the bug persists upstream today is not asserted, one way or the other,
   without checking.**
2. **A minimal reproducer isolated from this repository's setup** — a standalone script against
   unmodified `candle-transformers`, not this repository's Qwen2.5-3B/1.5B dual-model harness,
   demonstrating BF16 position aliasing above ~256 absolute position on any single model. The S1a
   live observation (specific degenerate completions at specific positions) is diagnostic evidence
   for this repository's own investigation; it is not, by itself, the reproducer a `candle`
   maintainer needs.
3. **The fix isolated from this repository's other deviations** — a patch against `candle`'s actual
   `qwen2.rs` (unsplit, using its own `with_tracing` wrappers), not a diff against `qwen2_a.rs`'s
   already-modified copy. Deviations 1-3 in the local ledger (the with_tracing removal, the inlined
   `repeat_kv`, the visibility change) are repository-specific and irrelevant upstream; only
   deviation 4's substance — build the RoPE table in F32, cast the final sin/cos — is the
   contribution.

None of the three steps above has been performed. This is named as an open action item, tracked in
the mission checkpoint's research backlog, not claimed as done.

### Maintenance liability

**On any future `candle`/`candle-transformers` version upgrade**, this repository's re-verification
obligation is exactly ADR-023's own precedent for introducing the fix in the first place: **S0's
mechanics gates (bit-identical logits parity, injected-logits-finite, zero-slot-equals-baseline,
norm-band checks) must be re-run and confirmed green before any receipt produced under the new
version is trusted**, per the ADR-031 receipt contract's `env.git_commit`/toolchain fields. Two
specific liabilities an upgrade creates: (a) if a later `candle` release fixes the RoPE bug
independently, this repository's local F32-table patch becomes a redundant no-op on the same code
path — not a conflict, but **not proven numerically identical to whatever fix upstream chose**
without a golden-vector comparison; the bit-parity gates are exactly the mechanism that would
surface a divergence, and are the required check before assuming redundancy is harmless. (b) any
upgrade must re-derive the deviation ledger from scratch against the new version's source — a stale
ledger describing deviations from a version no longer vendored is worse than no ledger, because it
asserts provenance that no longer holds.

## Boundary table — verified vs. asserted

| Claim | Status |
|---|---|
| BF16 RoPE position-aliasing bug present in vendored `candle-transformers` 0.9.2, live-observed | **Verified** — S1a run-1 receipt, diagnosed and fixed this session |
| F32-table fix eliminates the observed degeneration on the same items | **Verified** — S1a run-2 receipt, same items, RoPE fix applied, no degeneration |
| Fix applied uniformly across every model/condition/stage from S0 onward | **Verified** — S0's mechanics gates re-run and green after the change (ADR-023 Deviation 2) |
| Bug present in `candle` main / a `candle-transformers` release after 0.9.2 | **UNVERIFIED — not checked, not asserted either way** |
| Minimal reproducer isolated from this repository's harness | **Not built** |
| Upstream contribution submitted | **Not done** |

## Consequences

Stating the vendor-only-a-file-not-a-dependency-graph rule up front constrains any future temptation
to vendor more of `candle` than a single model file for convenience — the MSRV-conflict rationale
that justifies excluding `latentmesh-runtime`/`latentmesh-train` from the workspace already solves
the general "can't depend on candle normally" problem; vendoring is reserved for the narrower case
where even an excluded crate can't get the surface it needs as a normal dependency. Naming the
upstream-contribution obligation as unmet work, rather than silently carrying the patch forever,
keeps this repository's claim about the bug honest: a bug this repository found and fixed locally,
with an explicit trail toward (but not yet reaching) benefiting every other `candle` user hitting
the same aliasing failure above ~256 tokens.

## Implementation status

Design contract only — no code changes to `qwen2_a.rs`/`qwen2_b.rs` (their existing deviation ledger
already satisfies this ADR's requirements and needed no edit). Unmet by this ADR, named as follow-up
work: verifying the bug's status against `candle` main, building an isolated minimal reproducer, and
preparing/submitting the upstream contribution. These are tracked as a standing research-backlog item
(`.ruvnet-brain/checkpoint.json`'s research lane already lists "candle RoPE upstream PR" as backlog)
and are not scoped to any current run-2 milestone.
