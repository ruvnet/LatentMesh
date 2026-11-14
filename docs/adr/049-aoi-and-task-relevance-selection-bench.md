# 049. Age-of-Information benchmark and task-relevance selection (phase 1, benchmark only)

- **Status**: Proposed
- **Date**: 2026-11-13
- **Issue**: [#22](https://github.com/ruvnet/LatentMesh/issues/22)
- **Related**: [031](031-evidence-receipt-and-statistical-protocol-governance.md) (receipt and protocol governance), [032](032-negative-result-publication-contract.md) (a negative result is published), [046](046-representation-metaharness.md) (MetaHarness frozen-promotion pattern), [048](048-air-runtime-and-site-calibration.md) (AIR stays deterministic transport)

## Context

Issue #22 asks whether selective transmission of task-relevant fields beats whole-envelope
transmission under changing link conditions, and whether stale state should be dropped rather
than sent. The repo has no freshness curve and no selector. The motivating papers
(arXiv 2608.27688, 2608.25410) report gains on AWGN vision tasks; none of that transfers to
LoRa, Meshtastic, BLE or HF, or to agent traffic, and this ADR claims none of it.

What the code constrains (read, not assumed):

- `SemanticDelta` = `{source_id, epoch, message_id, base_hash, result_hash, updates: Vec<SymbolUpdate>, residuals: Vec<Residual>}`;
  limits `MAX_CRITICAL_FIELDS=64`, `MAX_RESIDUALS=16`, `MAX_RESIDUAL_VALUES=64`.
- `SemanticDelta::apply` verifies `base_hash` and `result_hash`. **Dropping a symbol update makes
  `result_hash` unverifiable**, so symbolic updates cannot be selectable in phase 1. Only
  residuals (which "can never create or modify a critical symbolic field") and opaque regions
  of non-delta bodies are droppable. This is narrower than the issue's wording and is stated up
  front rather than discovered later.
- `SemanticEnvelope` = 48-byte header + body + optional 64-byte signature; `SparseRadioFrame`
  carries a `u16` sequence and at most 256 bytes; `ReplayWindow` is a 64-wide `u16` window whose
  `commit` rejects duplicates and too-old sequences.
- `latentmesh-bench` has no arguments today, and depends on `rand` 0.8 but not `air-core`.

## Decision

### 1. Scope: phase 1, benchmark only, no wire change

Phases 1-4 of the issue ship here: AoI impairment matrix, mask definition, three baselines,
utility/byte/airtime/age measurement. Phase 5 (threshold evolution) is out of scope until the
benchmark is frozen (section 5). No selective *encoding* exists: the selector produces a
`TransmissionPlan` (accounting) and there is deliberately no encoder for a reduced message.
`crates/latentmesh-air-core/testdata/*` must stay byte-identical
(`git diff --exit-code crates/latentmesh-air-core/testdata` is a CI check). Existing public
items are unchanged; `selector` only adds exports to `lib.rs`.

### 2. Selector API: `crates/latentmesh-air-core/src/selector.rs`

`#![forbid(unsafe_code)]` and `no_std + alloc` as the crate already is; no new dependencies;
no floating point; no I/O.

```rust
pub const MAX_MASK_ENTRIES: usize = 80;      // MAX_CRITICAL_FIELDS + MAX_RESIDUALS
pub const MAX_MASK_WIRE_BYTES: usize = 4 + 4 * MAX_MASK_ENTRIES;   // 324
pub const MAX_REGIONS: usize = 16;           // non-delta body split into 16-byte regions
pub const REGION_BYTES: usize = 16;
pub const MAX_TASKS: usize = 16;             // registry size
pub const MAX_SELECTOR_STEPS: usize = 512;   // hard work bound per plan call

pub enum FieldClass { Required, Optional, Discardable }

pub enum ReservedField { Signature, ReplaySequence, Epoch, MessageId, SourceId,
                         StateHash, Provenance, CriticalHash, SafetyMetadata }

pub enum FieldRef { Symbol(u16), Residual(u8), Region(u8), Reserved(ReservedField) }

pub struct MaskEntry { pub field: FieldRef, pub class: FieldClass, pub rank: u8 } // rank 0..=63
pub struct UnboundMask { /* task_id: u16, entries: Vec<MaskEntry> */ }   // untrusted
pub struct BoundMask   { /* private; only constructible via TaskRegistry::bind */ }

pub struct TaskDefinition { pub task_id: u16, pub version: u8,
                            pub mandatory: Vec<FieldRef> }  // sorted, <= MAX_MASK_ENTRIES
pub struct TaskRegistry { /* <= MAX_TASKS definitions, validated on new() */ }

pub enum SelectorError { Malformed, TooManyEntries, NotCanonical, DuplicateEntry, BadClass,
                         UnknownTask, ReservedDowngrade, MandatoryDowngrade,
                         SymbolNotSelectable, WorkLimit }

impl UnboundMask  { pub fn parse(bytes: &[u8]) -> Result<Self, SelectorError>; }
impl TaskRegistry { pub fn new(defs: Vec<TaskDefinition>) -> Result<Self, SelectorError>;
                    pub fn bind(&self, m: UnboundMask) -> Result<BoundMask, SelectorError>; }

pub struct TransmissionPlan { pub task_id: u16, pub fixed_bytes: usize,
    pub kept_symbols: Vec<u16>, pub kept_residual_slots: Vec<u8>, pub kept_regions: Vec<u8>,
    pub dropped_residual_slots: Vec<u8>, pub dropped_regions: Vec<u8>,
    pub original_bytes: usize, pub planned_bytes: usize }

pub fn plan_delta(d: &SemanticDelta, m: &BoundMask, byte_cap: Option<usize>)
    -> Result<TransmissionPlan, SelectorError>;
pub fn plan_envelope(e: &SemanticEnvelope, m: &BoundMask, byte_cap: Option<usize>)
    -> Result<TransmissionPlan, SelectorError>;
```

**Mask wire form (for fuzzing and receipts).** `[task_id u16 LE][count u8][0x00]` then `count`
entries of 4 bytes `[kind u8: 0 Symbol, 1 Residual, 2 Region, 3 Reserved][id u16 LE][class<<6 | rank]`
with class 0/1/2 = Required/Optional/Discardable (3 is `BadClass`). Length is checked against
`MAX_MASK_WIRE_BYTES` and `count` against `MAX_MASK_ENTRIES` *before* any allocation.
Entries must be strictly ascending by `(kind, id)`; otherwise `NotCanonical` / `DuplicateEntry`
(no sorting on untrusted input, so work is linear and canonical bytes are unique).

**Invariants (each is a test, section 6).**

1. Always Required, never downgradable by any mask: signature, replay/sequence fields,
   `epoch`, `message_id`, `source_id`, `state_hash`/provenance, `base_hash`/`result_hash`
   (the critical-state hash), and mandatory safety metadata. These occupy `fixed_bytes` and are
   outside the selectable domain. A mask entry naming `Reserved(_)` with class other than
   `Required` returns `ReservedDowngrade`; it is never silently ignored.
2. For a `StateDelta`, every `Symbol(_)` is Required. A `Symbol` entry with class
   `Optional`/`Discardable` returns `SymbolNotSelectable` (because `result_hash` would fail).
   A `Region(_)` entry on a `StateDelta`, or a `Residual(_)` entry on a non-delta body, returns `Malformed`.
3. A field listed in the bound `TaskDefinition.mandatory` that appears in the mask with a class
   other than `Required` returns `MandatoryDowngrade`.
4. Unlisted fields default to **Required**. A mask can only remove what it names.
5. `Optional` entries are kept in ascending `(rank, field)` order while `planned_bytes <= byte_cap`
   (`None` = keep all); `Discardable` is never kept. Required bytes are kept even if they
   exceed `byte_cap` (the cap never overrides invariant 1).
6. Masks are untrusted until `TaskRegistry::bind` succeeds. `plan_*` accept only `BoundMask`, so
   an unbound mask cannot reach the planner by construction. A `task_id` not in the registry is
   `UnknownTask`. Selector metadata never reaches prompts or tool capabilities: the crate has
   no such surface and the mask carries only field indices and classes.
7. Work bound: parse is O(count), bind is O(count + mandatory), plan is O(entries + updates +
   residuals) with an explicit step counter; exceeding `MAX_SELECTOR_STEPS` returns `WorkLimit`.
   Worst case is 80 + 64 + 16 = 160 < 512 steps; the counter is a tripwire against future edits.
8. `plan_*(whole mask of all-Required).planned_bytes == original_bytes == envelope.encoded_len()`;
   a test pins the per-field size arithmetic to `SemanticDelta::encode` output so the plan
   cannot drift from the real encoding.

### 3. Deterministic AoI benchmark: `crates/latentmesh-bench/src/aoi.rs`

Invocation: `latentmesh-bench aoi [--seeds 11,23,37,53,71] [--out PATH] [--quick]`. With no
arguments `main` behaves exactly as today (existing output unchanged). `--quick` shrinks the
grid for CI smoke and writes a receipt marked `"frozen": false` that the gate rejects.
`latentmesh-bench/Cargo.toml` gains `latentmesh-air-core` (path), `sha2` and
`rand_chacha = "0.3"` (workspace-pinned). `ChaCha8Rng` is used, not `StdRng`, because `StdRng`
streams are not guaranteed stable across `rand` releases.

**Axes (independent, full Cartesian product, `grid_v1` = 648 cells).**

| axis | meaning | values |
|---|---|---|
| `age_of_information_ms` | evidence age already accrued at generation | 0, 500, 2000 |
| `airtime_budget_bytes` | on-air frame bytes allowed per 1000 ms window | 600, 1200, 2400 |
| `queue_depth` | sender FIFO capacity, drop-oldest on overflow | 4, 16 |
| `loss_rate` | iid per-frame loss, no ARQ | 0.0, 0.10, 0.30 |
| `reorder_rate` | P(message gets +1..3 generation periods of delay) | 0.0, 0.20 |
| `link_outage_ms` | one outage, start drawn in [10%,50%] of run | 0, 5000 |
| `task_relevance_fraction` | share of residual slots in the task's relevant set | 0.25, 0.5, 1.0 |

**Workload (frozen constants).** `MSG_COUNT=512`, `GEN_PERIOD_MS=250`, `LINK_BYTES_PER_S=2400`,
signed envelope, 6 symbol updates, 8 residual slots of 32 values each, per-residual random
`importance`. The relevant residual set has `ceil(rho*8)` slots chosen by a seeded
Fisher-Yates once per cell and seed; all symbols are relevant. Frames come from
`fragment_message`; **airtime = sum of encoded frame bytes** (16 B header per fragment).
All randomness (workload, per-frame loss draws, reorder draws, outage start) is materialized
up front from `ChaCha8Rng::seed_from_u64(seed.wrapping_mul(0x9E3779B97F4A7C15) ^ cell_id)`
before any baseline runs, so the three baselines see common random numbers (paired comparison).
Seeds: at least 5, default `[11,23,37,53,71]`; fewer is an error outside `--quick`.

**Baselines** (identical queue, link, stale-drop and replay handling; only the payload differs).

- `whole_payload`: full envelope, no cap.
- `priority_field`: task-agnostic; Required items always, residuals kept by descending `importance` up to the per-message cap.
- `relevance_selected`: `plan_delta` with a `BoundMask` that marks the relevant residual slots
  Required, all others Discardable (a variant with ranked Optional slots may be added, not required).
- Per-message cap for the two reduced baselines = `airtime_budget_bytes / 4` (4 messages per window); Required bytes are exempt (invariant 5).

**Link and queue.** Window-based airtime: a message needs its total frame bytes within the
remaining window budget, else it waits for the next window (head-of-line). A message larger
than the whole budget is dropped as `unschedulable_drops`. Transmission takes
`ceil(bytes*1000/LINK_BYTES_PER_S)` ms; a transmission overlapping the outage loses all its
frames. A message is lost if any fragment is lost. Delivery age
`= age_of_information_ms + (tx_end - generated_at) + reorder_extra_ms`.

**Stale-drop policy and replay consistency (all baselines).** `STALE_AGE_MS=2000`. The sender
drops a queued message whose age at transmit start exceeds `STALE_AGE_MS`. The `u16` sequence is
assigned **at generation and never reused**, so drops leave forward gaps, which `ReplayWindow`
already accepts (a gap is never a replay). The receiver runs every delivered message through
`ReplayWindow::commit`, including those it will discount as stale, so the watermark advances
identically with or without stale-drop. `replay_inconsistencies` counts: a sequence committed
twice; a sender-dropped sequence later committed; `TooOld` for a message displaced by <= 63
positions. The required value is 0.

**Registered task utility (integer arithmetic, no libm).** Decay is a piecewise-linear
half-life function with `H=1000` ms: for age `a <= STALE_AGE_MS`,
`base = 1_000_000 >> (a/1000)`, `D_ppm(a) = base - (base/2)*(a%1000)/1000`; for `a > STALE_AGE_MS`,
`D_ppm = 0`. A delivered message scores `U = D_ppm(a) * W_delivered`, where `W_delivered` is the
sum of weights of **relevant fields actually present in the delivered payload** (symbol update
weight 1, relevant residual slot weight 4; irrelevant fields weigh 0). Utility is scored on the
registered relevant set for all three baselines; in `relevance_selected` that set is exactly the
Required set, so only Required-and-delivered relevant fields count. A message is *fresh* if age
`<= STALE_AGE_MS`, *stale* otherwise, and a *useful fresh delivery* if fresh and `W_delivered >=
0.9 * W_max` (`USEFUL_COVERAGE_PPM=900000`).

**Metrics per cell, baseline, seed (all integers).** `messages_generated`, `messages_delivered`,
`bytes_on_air`, `payload_bytes_tx`, `utility_micro` (sum of `U`), `utility_per_byte_micro`
(`utility_micro / bytes_on_air`, 0 if no bytes), `stale_deliveries`, `stale_delivery_rate_ppm`
(`stale / delivered`, 0 if none delivered), `useful_fresh_deliveries`, `queue_occupancy_mean_milli`,
`queue_occupancy_max`, `queue_overflow_drops`, `stale_drops`, `unschedulable_drops`,
`evidence_age_p50_ms`, `evidence_age_p95_ms`, plus integrity counters `mandatory_violations`,
`replay_inconsistencies`, `queue_bound_violations`, `airtime_window_violations`.

### 4. Receipt: schema `latentmesh.aoi-bench.v1`

Written to `--out` (default `crates/latentmesh-bench/receipts/aoi-bench-v1-<short-commit>.json`).
Evidence label is `simulated-deterministic`; it is never `measured` (no radio is involved).
All numbers are integers and rates are parts-per-million so canonical hashing is exact.

```
schema:"latentmesh.aoi-bench.v1", schema_version:1, evidence_label, frozen:bool,
git:{commit, dirty:bool, dirty_snapshot_sha256|null}, bench:{crate_version, command},
seeds:[u64], axes:{age_of_information_ms:[], airtime_budget_bytes:[], queue_depth:[],
  loss_rate_ppm:[], reorder_rate_ppm:[], link_outage_ms:[], task_relevance_ppm:[]},
config:{rng:"ChaCha8Rng/rand_chacha-0.3", msg_count, gen_period_ms, link_bytes_per_s,
  stale_age_ms, half_life_ms, useful_coverage_ppm, weights:{symbol,relevant_residual},
  workload:{symbols,residual_slots,residual_values}, baselines:[..3 names..],
  mask_hex, primary_cells:"task_relevance_ppm<=500000"},
frozen_digest_sha256: hex,
cells:[{cell_id, axes:{<the 7 scalars>}, baselines:{whole_payload|priority_field|relevance_selected:
  {per_seed:[{seed, <metrics>}], mean:{<metrics>}}}}],
integrity:{mandatory_violations_total, replay_inconsistencies_total,
  queue_bound_violations_total, airtime_window_violations_total},
receipt_sha256: hex
```

`frozen_digest_sha256` = SHA-256 of the canonical JSON of `config` + `axes` + `seeds`.
Canonical JSON: UTF-8, object keys sorted bytewise, no whitespace, integers only (no floats,
which is why rates are ppm), ASCII strings. `receipt_sha256` is the same over the whole receipt
with that one field omitted. Rust and JS must implement the identical canonicalizer; a shared
test vector is committed in both languages.

### 5. MetaHarness promotion gate: `harness/representation`

New files only: `lib/aoi-gate.mjs`, `scripts/aoi-gate.mjs`, `test/aoi-gate.test.mjs`, and a
pinned `lib/aoi-frozen.json` (`{"schema":"latentmesh.aoi-bench.v1","frozen_digest_sha256":"..."}`)
written by the integration owner at the explicit freeze step. This gate is a separate promotion
decision; it adds no gene to `representationPolicy` and does not change ADR-046's rule.

`evaluateAoiPromotion(receipt, pinned)` returns `{verdict, criteria:{c1,c2,c3}, integrity, reasons[]}`
with verdict `PROMOTE`, `REJECT` or `INVALID`. `INVALID` (nothing is concluded) if: `pinned` is
absent (benchmark not frozen), `frozen` is false, fewer than 5 seeds, `receipt_sha256` or
`frozen_digest_sha256` do not recompute, the recomputed digest differs from the pinned one, or
`git.dirty` is true without a snapshot hash.

**Integrity gates (all must hold, else `REJECT` regardless of utility):** every integrity total
in `integrity` is 0 (no mandatory field removed, no replay inconsistency, no queue-bound or
airtime-window violation), and for every cell and seed `relevance_selected.bytes_on_air` stays
within the window budget accounting recorded by the bench. A mutation that raises utility by
violating any of these is rejected, per the issue.

**Utility criteria.** Evaluate on the primary cell set `P` (`task_relevance_ppm <= 500000`,
frozen in `config`). For each seed, aggregate over `P` by summing metrics, then compare
`relevance_selected` (S) to `whole_payload` (W). The criterion holds only if it holds in **every**
seed:

- **c1** bytes: `S.payload_bytes_tx <= 0.80 * W.payload_bytes_tx` and `S.utility_micro >= 0.98 * W.utility_micro`.
- **c2** utility at equal airtime: `S.utility_micro >= 1.20 * W.utility_micro` with
  `S.bytes_on_air <= W.bytes_on_air` (both within the per-cell budget by the integrity gate).
- **c3** staleness: `S.stale_delivery_rate <= 0.75 * W.stale_delivery_rate` (the aggregate
  `stale/delivered`) and `S.useful_fresh_deliveries >= 0.95 * W.useful_fresh_deliveries`.

Promote only if at least one of c1-c3 holds in all seeds **and** two guards hold: (a) against
`priority_field`, S's aggregate `utility_per_byte_micro` over `P` is not lower in any seed (this
is stricter than the issue text and prevents beating a strawman; relaxing it needs an ADR
amendment); (b) no-harm over the full grid: in no cell does the seed-mean
`useful_fresh_deliveries` of S fall below 0.90 of W's. Integer math only; ratios are
cross-multiplied, never divided. Thresholds are constants in the gate file and are frozen with
the benchmark.

**Negative results.** A `REJECT` with valid integrity is a result, not a failure to hide. The
receipt, gate output (`artifacts/aoi-promotion.json`) and a research note are published per
[ADR-032](032-negative-result-publication-contract.md); thresholds are not tuned after the
outcome is seen. Evolving selector thresholds with MetaHarness is explicitly out of scope until
a frozen digest is pinned and one full registered run exists.

### 6. Testing plan

- **Property tests (air-core):** for random deltas and random masks, `plan_delta` never omits a
  Required item, every `Reserved` downgrade and every `Symbol` downgrade errors, `planned_bytes
  <= original_bytes`, all-Required mask reproduces `encoded_len()`. Use a small seeded PRNG
  in-test; no new dependency in the crate.
- **Malformed-mask fuzz-style tests:** random byte strings and mutated valid masks into
  `UnboundMask::parse` and `bind`: never panics, never allocates beyond the bound, rejects
  oversize, unsorted, duplicate, bad class, unknown task, `count` larger than the buffer.
- **Replay tests (bench):** parameterised across age, loss, reorder and outage, including `u16`
  wraparound of the sequence; assert `replay_inconsistencies == 0` and that stale-drop on versus
  off yields the same `ReplayWindow::highest()` for the same deliveries.
- **Determinism test:** two runs of the same `(seeds, grid)` produce byte-identical receipts;
  digest and `receipt_sha256` match a committed Rust-and-JS cross-check vector; golden-vector
  `git diff` check passes.
- **Gate tests (JS):** synthetic receipts for PROMOTE, each criterion alone, each integrity
  failure, 4-seed receipt, tampered digest, unpinned freeze, and the priority_field guard.
- **Bounds benchmarks:** selector wall time and heap bytes at 80 entries; queue occupancy is in the receipt.

### Rollback

Disable the selector (do not call `plan_*`; `relevance_selected` is the only consumer) and keep
whole-envelope transport, which is unchanged. Remove `selector.rs` and its `lib.rs` re-exports,
`aoi.rs` and the `aoi` dispatch, and the three `harness/representation` files; no wire format,
golden vector or existing API is touched, so removal is a pure deletion. The benchmark remains
useful if the selector fails to earn its complexity.

### Non-goals and honest limits

- No LoRa, Meshtastic, BLE or HF gain claim, and no claim that the papers' 92.1 percent figure
  transfers. This receipt is a simulation of an abstract window-budget link.
- No model embeddings or token embeddings on the wire; the selector sees only field indices.
- No selective *encoding* and no envelope extension in phase 1; a later one must be backwards
  compatible or an explicitly versioned extension, with its own ADR.
- No cross-carrier adapter tests and no WASM parity tests in phase 1. The selector is
  `no_std + alloc` compatible, but no WASM build is verified here and none is claimed.
- No ARQ, no multi-hop, no regulatory duty-cycle modelling; the window budget is a stand-in.
  Preserving carrier licensing and regulatory limits is a requirement on any later radio use.
- The only droppable content is residuals and opaque regions, because symbolic updates carry
  `result_hash`. The achievable byte saving is therefore bounded by the residual share of the
  message, and the benchmark may well find that is not enough. That would be a valid outcome.

## Post-first-run review (2026-11-13)

The first full run (receipt `aoi-bench-v1-47bff02`, frozen digest `858b88d6…ec5e618`, not pinned)
returned PROMOTE on c1 and c2 in all five seeds; c3 failed. An adversarial test pass then found
the following. The rule applied: **correctness defects are fixed; design weaknesses are disclosed
and left unchanged**, because altering a comparator, guard or grid after seeing the verdict is the
post-hoc tuning ADR-032 forbids.

**Fixed (defects; the receipt is regenerated after the fixes):**

- D1. The gate did not bind `cells` to the frozen axes (count, unique `cell_id`, membership,
  coverage); a resealed receipt with cells deleted still promoted.
- D2. The gate's no-harm guard read `mean` without recomputing it from `per_seed`.
- D3. `airtime_window_violations` was checked before the add and could never fire; it is now an
  independent audit over the transmission log. Aligned-window semantics are unchanged.
- D4. The replay `TooOld` inconsistency branch was unreachable; messages displaced by 64 or more
  were lost silently. They are now counted as `replay_window_rejections` (a loss metric, not an
  integrity failure).
- D5. The `priority_field` mandatory-field check was vacuous.

**Disclosed, not changed (design weaknesses of `grid_v1`):**

- W1. The `priority_field` guard aggregates over the primary set and hides per-cell losses: in
  135 of 648 cells S has lower mean utility than `priority_field` (worst cell: budget 600,
  queue 16, rho 0.5, about 5.8 times in favour of `priority_field`). Over the budget-600 primary
  slice S/F absolute utility is 0.97.
- W2. The per-message cap (`airtime_budget_bytes / 4`) binds `priority_field` but not S, whose
  Required bytes are exempt; at budget 600 the cap is below the fixed content, so
  `priority_field` sends no residuals. Its importance scores are random and uncorrelated with
  relevance, and S has an oracle relevant set with no mask error.
- W3. S keeps every relevant field, so its per-message weight equals W's. S's advantage over W is
  throughput and age (shorter airtime, fewer fragments, less queueing), not better coverage. The
  byte saving follows from the workload's definition that irrelevant residuals weigh 0.
- W4. The c2 margin is narrow (per-seed S/W utility 1.224 to 1.229 against 1.20) and is carried by
  the budget-1200 slice, the only one where W is queue-starved. c1 carries the verdict.
- W5. The whole `age_of_information_ms = 2000` slice scores 0 utility for every baseline (a third of
  the primary set); it does not change ratios but wastes grid.
- W6. The airtime budget is an aligned 1000 ms window, not a rolling one; a rolling second can carry
  up to twice the budget.

Any promotion from `grid_v1` therefore means only "relevance selection beats whole-payload under
this abstract simulated link", not "beats a competent task-agnostic selector". A `grid_v2`
addressing W1-W6 (per-cell priority guard, a cap that binds all reduced baselines equally,
correlated importance and noisy relevance masks, a rolling budget, an AoI axis without a dead
slice) must be pre-registered as a new ADR **before** it is run.

## Consequences

Positive: AoI becomes a first-class, replayable curve; the selector's safety invariants are
type-enforced (`UnboundMask` versus `BoundMask`) and fail closed; the three parts (selector,
bench, gate) have frozen interfaces and can be built independently. Negative: a second
canonicalizer to keep in sync across Rust and JS; a 648-cell grid costs CI time (`--quick` exists
for smoke runs); results are simulation-only evidence.

**Ownership for parallel work.** A: `selector.rs` and `lib.rs` re-exports (air-core).
B: `aoi.rs`, `main.rs` dispatch, `latentmesh-bench/Cargo.toml` (may stub the selector API from
section 2 until A lands). C: the three `harness/representation` files, driven by the receipt
schema in section 4 and fixtures. Only the integration owner edits shared manifests and lockfiles.
