# 031. Evidence, receipt, and statistical-protocol governance

- **Status**: Proposed
- **Date**: 2026-08-29
- **Related**: [014](014-benchmark-and-acceptance-method.md) (evidence-label and stage-gate
  discipline this ADR formalizes into a schema), [018](018-metaharness-darwin-topology-loop.md)/
  [022](022-self-optimizing-metaharness-e2e-loop.md) (the receipt pattern this ADR generalizes
  beyond MetaHarness), [023](023-live-four-condition-run1-pre-registration.md) (the frozen
  pre-registration this ADR extracts a reusable protocol from), [024](024-run2-trained-thought-adapter-ladder.md)
  (the ladder that first needed a leakage/one-draw rule stricter than a closed-form fit ever did),
  [028](028-evolutionary-adapter-search-anti-gaming.md) (the "frozen probe is a one-shot resource"
  principle this ADR states as a general rule, not an ADR-023-specific one)
- **Evidence base**: read in full for this ADR — `crates/latentmesh-runtime/receipts/s0-receipt.json`,
  `s1a-receipt-slots8-block19-poolfull-rescaletrue-n40.json`,
  `s1a-receipt-run1-buggy-rope-noncompliant-prompt.json`,
  `s2b-receipt-cellL18toL14-slots8-poolfull-rescaletrue-n40.json`,
  `run2-m3-training-receipt-cellL18toL14.json`,
  `run2-m4-receipt-cellL18toL14-fastgrnn-r64-superseded-windowzeroinit.json`,
  `run2-a6-permnull-receipt.json`, `run-ledger.json`; ADR-023 (frozen registration + S6 results);
  ADR-024 (leakage rule, ladder discipline); `docs/research/029-a6-permutation-null.md` (the
  worked example of a protocol-safe annotation)

## Context

This repository has run a real pre-registered experiment (ADR-023), a five-rung trained-adapter
ladder against a frozen probe (ADR-024), and dozens of committed JSON receipts — but nothing
states the receipt schema or the statistical discipline as a standing rule. Both exist only as
prose scattered across ADR-014 (evidence labels, stage separation), ADR-023 (the frozen
registration itself, and the S6 results section's "nothing is estimated" discipline), and the
de-facto conventions visible in `crates/latentmesh-runtime/receipts/*.json` — over 60 files that
already agree on a shape nobody wrote down. A new contributor, or a new milestone under time
pressure, has no single document to check a receipt or a statistical decision against. This ADR
extracts what is already practiced and states it as a contract, so deviation from it is a
reviewable violation rather than an undocumented inconsistency.

## Decision

### (a) The receipt contract

Every experimental or training receipt in `crates/latentmesh-runtime/receipts/` (and any future
`crates/latentmesh-train/receipts/`-equivalent location) **must** carry the following fields,
observed without exception across every receipt read for this ADR:

| Field | Purpose | Example (verbatim from a committed receipt) |
|---|---|---|
| `stage` | Which protocol step this receipt reports | `"run2-m3-mlp-training"` |
| `design` | Path to the ADR/design-doc section this receipt tests | `"docs/adr/024-run2-trained-thought-adapter-ladder.md M3 ..."` |
| `env.evidence_label` | The ADR-014 evidence label, verbatim, machine-checkable | `"live-model, single-host, simulation-free"`; `"seeded deterministic GPU training (candle 0.9.2 AdamW) over captured per-token pairs"`; `"deterministic CPU analysis over committed dumps"` |
| `env.git_commit` | Exact source commit the run executed against | `"9a676a5d53f66425e40e3ca6f8c3bf6fb1aa9379"` |
| `env` hardware/toolchain block | GPU model + driver, nvcc version, unix timestamp — reproducibility per ADR-014 | `env.gpu`, `env.nvcc`, `env.nvcc_at_build`, `env.unix_time` |
| `dataset.sha256` (+ `.source`, `.items`/`.indices`) | Dataset pin — the exact file and exact rows consumed | matches ADR-023's frozen dataset-pin table |
| `config` / `pre_committed` | The frozen parameters this run was invoked with, restated in the receipt itself, not only in the ADR prose | e.g. `n_slots`, `placeholder_id`, `norm_band_factor`, `rescale_to_natural_median` |
| `gates` (top-level or per-item) | Per-gate `pass: bool` plus the exact evidence that produced it — never a bare boolean with no supporting number | `A7b_aligned_real_vs_random: {p: 0.5, pass: false}`; `hand_rolled_apply_matches_align_crate: {max_relative_l2_error: 0.0, pass: true}` |
| `summary` | Aggregate statistics computed from the gates/items (wins/losses, p-values, accuracy) | `primary_real_vs_random: {wins: 5, losses: 0, p_one_sided: 0.03125, alpha: 0.05, pass: true}` |
| `gate_pass` / `all_gates_pass` | One overall boolean, derivable from `gates`/`summary` but stated explicitly for a fast reader | — |
| `wall_clock_s` | Cost accounting, read directly by every GPU-accounting table in this repo's ADRs and never estimated | ADR-023 S6 §4's entire GPU-accounting table is `wall_clock_s` values copied verbatim |

**Training receipts add** (established by ADR-024's ladder, not present in ADR-023's closed-form
fits because a fit has no training loop): `training.{seed, stopping_rule, runs_performed,
note_no_discarded_runs}`, `split.{excluded_probe_overlap_rows, fit_split_seed, sha256 of both row
sets}`, and `artifact.{content_hash_sha256, golden_file, golden_pairs}` for the trained weights
themselves — the same golden-vector-verification discipline ADR-014 already requires for wire
codecs, applied to a trained artifact instead of a codec.

**Naming convention** (observed, now formalized): `<stage>-receipt-<frozen-params-as-slug>.json`,
e.g. `s2b-receipt-cellL18toL14-slots8-poolfull-rescaletrue-n40.json` — the slug encodes exactly the
frozen parameters a reader would otherwise have to open the file to learn. A companion
`<stage>-golden-*.json` file holds the bit-exact fixture a hand-rolled reimplementation is checked
against (required whenever a crate boundary — e.g. `latentmesh-align` cannot be a path dependency
of `latentmesh-runtime`, per ADR-023 Deviation 1 — forces a reimplementation of another crate's
logic). A `-superseded-<reason>` suffix marks a run abandoned before its frozen probe was invoked
(e.g. `run2-m4-fastgrnn-r64-cellL18toL14-superseded-windowzeroinit.json`, caught by the holdout
metric before any probe draw) — kept, not deleted, exactly like a passed run. A `-genpairs` (or
equivalent contingency-name) suffix marks a pre-registered contingency variant, never a silent
overwrite of the original. `run-ledger.json` is the append-only index a reader consults first.

**Append-only, never retroactively edited.** No receipt is ever edited after it is committed.
Evidence three examples already establish this as practiced, not aspirational:

1. Both S1a receipts are committed — the buggy run-1 failure
   (`s1a-receipt-run1-buggy-rope-noncompliant-prompt.json`) and the passing run-2 fix — not just
   the one that passed.
2. The superseded M4 r64 windowzeroinit training run is committed alongside the corrected one, its
   filename declaring the divergence rather than silently disappearing.
3. `docs/research/029` computed the A6 permutation-null baseline as a **new**, separate receipt
   (`run2-a6-permnull-receipt.json`) that *annotates* the recorded S2/S2c A6 numbers — stated
   explicitly in that document: "no recorded A6 number is changed... this document and its receipt
   annotate the S2/S2c receipts, exactly as ADR-024 registered." An annotation is always a new
   record with an explicit pointer to what it annotates; it is never a mutation of the original.

### (b) The statistical protocol

Extracted as a general rule from ADR-023's frozen registration and ADR-024's leakage discipline,
restated here so a future experiment need not rediscover it from two ADRs' worth of prose:

1. **Pre-registration before any outcome is known.** ADR-023 froze its statistics, thresholds, and
   dataset splits while S2 (calibration) was running concurrently, specifically so the one number
   still open (S2's `content_hash`) could not influence any already-frozen decision. This is the
   load-bearing property, not an incidental one: a threshold chosen after seeing a result is not a
   threshold.
2. **Frozen protocols are never iterated to produce a pass.** The 40-item S1a/S2b probe (item
   indices, α=0.05, one-sided exact sign test, 8-slot injection, rescale-to-median) was reused
   **verbatim** across S1a, S2b (four times, across two cells and two calibration distributions),
   M3 (twice), and M4 (three ranks) — never redrawn, never re-tuned. ADR-024 states the rule
   explicitly: a rung's failure escalates the ladder to the *next architecture*; it is "not grounds
   to retry the same architecture with different hyperparameters against the frozen probe." The one
   disciplined, pre-declared exception is a *sub-rung ladder announced in advance* (M4's `r ∈ {64,
   128, 256}`, tried once each in a pre-declared order, every receipt kept) — this is not iteration
   against an outcome, because the order and the stopping rule were fixed before any rank was
   trained.
3. **One registered draw per rung.** ADR-024's own M3 training receipt states this as a live
   constraint, not just a principle: the anchor cell (L24→L19) was "deliberately NOT probed (an
   unregistered extra draw)" at M3, because ADR-024's M3 gate registration named only the winner
   cell. ADR-028 generalizes the same fact independently: "the frozen 40-item S1a/S2b probe... is a
   *one-shot resource*. Every draw against it spends irreplaceable evidence." This ADR states it as
   a standing rule for any future frozen probe, not an ADR-024-specific observation: **a rung, a
   cell, or a condition not named in the frozen registration does not get an extra draw just
   because the infrastructure to run it exists.**
4. **Honest-fail paths are named before the outcome, not invented after.** ADR-023's Deviation-7
   stop rule ("if the generated-pairs aligned-real gate also fails at both cells, run 1 STOPS") was
   registered before S2c ran. ADR-024's M4c task-loss-ablation contingency was registered "NOW,
   before M4's probe result is known," with an explicit pre-verdict interpretive rule stating what
   an M4 null would and would not mean. A contingency invented after seeing a failing result is not
   a contingency; it is protocol drift dressed as one, and this ADR names that distinction as the
   test: was the contingency's trigger condition and action stated in a committed document **before**
   the triggering receipt existed?
5. **Superseded runs are preserved and disclosed, never silently replaced.** Restated from (a)'s
   append-only rule as a statistical (not just a filesystem) principle: a superseded run's numbers
   remain part of the evidence trail even when a corrected run supersedes its conclusion.
6. **Protocol-safe annotation is distinct from a probe draw, and both are legitimate — but only one
   spends the frozen resource.** `docs/research/029`'s permutation-null baseline is the worked
   example: it runs entirely on CPU, over already-committed dumps, computing what an *uncorrelated*
   mapping would score on the *same* held-out-residual metric A6 already used — it draws nothing
   from the frozen 40-item probe and changes no recorded A6 verdict. This is the model for any
   future "was our threshold ever calibrated against chance" question: answerable without spending
   probe evidence, provided the analysis reuses existing captured data and an existing metric rather
   than invoking the frozen protocol itself.

### (c) Reserved slot — statistical power and sequential design

`docs/research/031` (a statistical-power / sequential-design research pass) is **in flight**,
authored by a concurrent research lane at the time of this ADR's writing, and is not summarized or
pre-empted here. Candidate open questions that pass is expected to inform — named here as
questions, not answered: whether the frozen 40-item probe's power (ADR-023 already flagged that
p=0.03125 is the minimum attainable value at 5 discordant pairs — "a real pass, not an overwhelming
one") justifies its current N against the one-registered-draw-per-rung cost stated in (b) above;
whether a sequential (group-sequential or SPRT-style) design could reduce the number of items drawn
per rung while preserving the frozen-protocol guarantee in (b)(2); and whether the Clopper-Pearson
/ bootstrap-CI choices ADR-023 froze for A1-A8 should generalize to the ladder's per-rung gate. **This
ADR's rules in (b) stand as written until `docs/research/031` lands and, if warranted, amends this
ADR's test-selection guidance in a follow-up revision** — this slot exists so that revision has a
named target rather than requiring a fresh ADR to restate everything else in this document.

## What is formalized here / what stays open

| Item | Status |
|---|---|
| Receipt required-field contract, naming convention, append-only rule | **Formalized by this ADR** — matches every receipt read for it, no retroactive schema change required of existing files |
| Statistical protocol rules 1-6 above | **Formalized by this ADR**, extracted from ADR-023/024/028 practice |
| Statistical-power / sequential-design guidance | **Explicitly deferred to `docs/research/031`** — not answered here, not assumed |
| A machine-checked receipt-schema validator (e.g. a `harness/`-style CI gate that rejects a receipt missing a required field) | **Not built** — this ADR states the contract; enforcing it in CI is unscoped future work |

## Consequences

Stating the contract after the fact, from receipts that already conform to it, means this ADR
requires no migration of existing evidence — every receipt cited above already satisfies every rule
it states. The cost of writing it now rather than at ADR-023's authoring is that two milestones'
worth of receipts (S0 through M4c) had to independently reinvent the same shape without a document
to check against; a future milestone gets to check against one. Freezing rule (b)(3) — one
registered draw per rung — as a named, generalized rule (not just an ADR-024 M3 footnote) is the one
genuinely new commitment here: it applies to any future frozen-probe protocol this repository
writes, not only run 2's.

## Implementation status

Design contract only — no code. This ADR formalizes an existing, already-followed practice into a
checkable document; it does not require any receipt already committed to change, and it does not
yet add automated enforcement (see the boundary table above). The next concrete step, if pursued,
is a lightweight CI check in the style of `harness/evolve`'s acceptance-bound verifier that parses a
new receipt and rejects one missing a required field from (a) — named here as unscoped future work,
not committed to this wave.
