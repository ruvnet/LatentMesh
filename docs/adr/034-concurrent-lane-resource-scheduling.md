# 034. Concurrent-lane resource scheduling

- **Status**: Proposed
- **Date**: 2026-08-29
- **Related**: [031](031-evidence-receipt-and-statistical-protocol-governance.md) (the receipt
  contract a resumable capture's incremental writes must still satisfy), [023](023-live-four-condition-run1-pre-registration.md)
  §S6 (the GPU-accounting table this ADR's resumability rule is built from),
  [024](024-run2-trained-thought-adapter-ladder.md) (the mission this ADR's three lanes are drawn
  from, including the M3 near-miss this ADR exists to prevent a recurrence of)
- **Evidence base**: `.ruvnet-brain/checkpoint.json` (read in full — the `lanes` field is the
  already-existing, informal version of the contract this ADR formalizes, and its `incident` field
  is the near-miss this ADR cites), `crates/latentmesh-runtime/receipts/s2c-generated-dump-receipt.json`
  (the resumed-capture receipt — `resumed.prior_gpu_s_charged: 3696.03`,
  `wall_clock_s.total: 5310.37`), live reads this session: `mcp__ruvultra__ruv_gpu_status` (RTX 5080,
  `memory_total_mib: 16303`, 86% utilization, 11,840 MiB in use at the time of this ADR's authoring)
  and `mcp__ruvultra__ruv_gpu_processes` (`target/release/train_m4c_taskloss` holding 10,322 MiB —
  the M4c task-loss ablation, a registered ADR-024 contingency, running concurrently with this ADR
  being written), root `.gitignore` and the absence of a `crates/latentmesh-train/.gitignore` entry
  before this session (confirmed by direct read)

## Context

The run-2 mission (ADR-024) now runs three concurrent lanes against a single RTX 5080 with 16,303
MiB total memory: **implementation** (GPU-resident model loading, capture, and training —
`latentmesh-runtime`/`latentmesh-train`), **analysis** (CPU-only fitters and permutation nulls, e.g.
`docs/research/029`'s A6 permutation-null baseline), and **research** (network-bound literature
sweeps, e.g. `docs/research/028`/`030`). `.ruvnet-brain/checkpoint.json`'s `lanes` field already
names this three-way split informally (`"implementation": "M4c wf_d3264dd2 (GPU: ...)"`,
`"analysis": "A6 permutation-null wf_c4fe6a20 (CPU)"`, `"research": "STANDING lane"`) — this ADR
exists because that split has never been written down as a contract with rules, only practiced as an
ad-hoc convention in a checkpoint file. Two concrete incidents motivate formalizing it now. First, a
real near-miss, recorded verbatim in the checkpoint's `incident` field: **"M3 commit initially swept
7071 target/ artifacts (standalone crate not covered by root `/target` rule); caught pre-push,
reset, `.gitignore` fixed, recommitted clean."** `latentmesh-runtime` and `latentmesh-train` are both
excluded from the root workspace (ADR-023 Deviation 1, ADR-024) precisely because of the MSRV/`half`
conflict — and exclusion from the *Cargo* workspace does not exclude a crate's `target/` directory
from the *git* tree; that requires its own `.gitignore` entry, which did not exist for
`latentmesh-train` until this incident forced it. Second, live GPU contention is not hypothetical:
at the moment this ADR is being authored, `nvidia-smi`-equivalent tooling shows the card at 86%
utilization, 11,840 of 16,303 MiB in use, with `target/release/train_m4c_taskloss` (the M4c
task-loss-ablation contingency, itself an ADR-024-registered rung) alone holding 10,322 MiB — on a
16 GB card, that leaves roughly 4-5 GB of genuinely free headroom, well short of what a second
resident model (S0/S1a already ran two Qwen models concurrently, batch=1, BF16) would need without
contending for the same memory.

## Decision

**Exactly one GPU-holding workflow at a time, with the holder named in the mission checkpoint.**
`.ruvnet-brain/checkpoint.json`'s `lanes.implementation` field is the existing artifact this
requirement already uses informally (it names the workflow ID currently holding the GPU, e.g.
`"M4c wf_d3264dd2 (GPU: feasibility->train->probe)"`) — this ADR formalizes it as a **required**
field, not an incidental note: before any GPU-resident process starts (model load, capture, or
training), the checkpoint's `lanes.implementation` field must name it, and no second entry may claim
the GPU lane concurrently. This is checkable by the same live tooling used to write this ADR
(`ruv_gpu_processes`) — a GPU-holding process with no corresponding checkpoint entry is itself a
violation worth flagging, independent of whether it causes an out-of-memory failure.

**CPU-only analyses and research passes may run concurrently, and must declare themselves CPU-only
in their brief.** The A6 permutation-null baseline (`docs/research/029`) is the model: its own
document states its method as "deterministic CPU analysis over committed dumps... No model runs, no
GPU, no probe drawn" — stated up front, checkable against its own receipt's `env.evidence_label`
(per the ADR-031 receipt contract). A research lane fetching literature or a lane running arithmetic
over already-captured dumps declares that scope explicitly rather than leaving GPU-safety as an
assumption a reader has to verify by inspection.

**A lane that needs the GPU queues rather than preempts.** No lane kills or interrupts a running
GPU-holding process to make room for its own work; it waits for the checkpoint's `lanes.implementation`
field to clear, consistent with this repository's existing "never rerun to force a result"
discipline (ADR-032) extended to resource contention: preempting a running capture or training run
to steal the GPU risks losing exactly the kind of expensive, hard-to-reproduce GPU-hours this
mission has already had to account for precisely (ADR-023 S6 §4's wall-clock table).

**Long-running captures write incrementally so a kill is resumable.** The concrete precedent this
rule generalizes: S2c's generated-pairs dump was interrupted mid-run and resumed, and the resumed
receipt discloses both the prior charge and the resumed total rather than hiding the interruption —
`s2c-generated-dump-receipt.json` records `resumed.prior_gpu_s_charged: 3696.03` (the killed
attempt's already-spent GPU-seconds, preserved and charged, not discarded) alongside
`wall_clock_s.total: 5310.37` for the completed resumed run. Losing that 3,696-second charge to a
non-resumable capture would have cost roughly **1 GPU-hour** of redone work — the concrete number
this rule protects against recurring. Any future long-running GPU capture (a per-token dump, a
multi-epoch training run) must checkpoint incrementally to disk on the same principle, so that an
interruption (a preemption request, a crash, a deliberate kill to reclaim memory) loses at most the
time since the last checkpoint, not the whole run.

**Artifacts live in gitignored `target/`, with receipts — not binaries — committed.** Already the
practiced pattern (`s2c` dumps, `run2-m3-mlp-cellL18toL14.f32bin`-style trained-weight files, and the
large per-token capture files all live under a crate's `target/latentmesh-runs/` or equivalent, never
committed directly) with one documented, deliberate exception: `s2c-token-streams.jsonl` was
explicitly moved *out* of `target/` and committed (`540c34e`) because it is "the sole encoding of
~2.5 GPU-h of generations" — a capture expensive enough that losing it to a routine `cargo clean`
would be a real cost, not a convenience. This ADR states the general rule the exception already
implies: an artifact stays in gitignored `target/` **unless** it is both (a) the only surviving
encoding of GPU-hours that would be materially costly to regenerate, and (b) small enough to commit
reasonably (the token-stream JSONL, not the multi-gigabyte `.f32bin` capture files derived from it).
Receipts (JSON, always small) are committed regardless — they are the evidence trail, not the bulk
data.

**Every crate outside the root workspace needs its own `.gitignore` entry — a standing rule, not a
one-off fix.** The M3 near-miss (7,071 `target/` artifacts almost committed) happened precisely
because `latentmesh-train`'s exclusion from the Cargo workspace was not paired with a `.gitignore`
entry for its `target/` directory at the time it was created. The root `.gitignore` now carries both
`crates/latentmesh-train/target/` and `crates/latentmesh-runtime/target/` explicitly, with a comment
explaining why the workspace-level `/target` rule alone does not cover them. **This ADR states the
rule generally, for any future workspace-excluded crate this mission creates**: workspace exclusion
(a `Cargo.toml` decision) and git-ignoring the resulting build directory (a `.gitignore` decision)
are two separate steps, and creating the former without the latter is exactly the failure mode that
already happened once.

## Operational contract summary

| Rule | Enforcement mechanism |
|---|---|
| One GPU-holding workflow at a time, named in `.ruvnet-brain/checkpoint.json` `lanes.implementation` | Checkpoint field is required before GPU-resident work starts; checkable live via `ruv_gpu_processes` |
| CPU-only lanes declare scope in their brief and receipt's `evidence_label` | ADR-031 receipt contract; `docs/research/029` is the worked example |
| A lane needing the GPU queues, never preempts | No automated enforcement — a process discipline, checkable by checkpoint audit |
| Long-running captures checkpoint incrementally | Per-capture implementation requirement; `s2c-generated-dump-receipt.json`'s `resumed.*` fields are the required disclosure shape when this fires |
| Artifacts in gitignored `target/`; receipts committed; exception only for otherwise-unrecoverable GPU-hours (small enough to commit) | `.gitignore`, per-crate |
| Every workspace-excluded crate gets its own `.gitignore` entry at creation time, not after a near-miss | Root `.gitignore`, reviewed whenever a new `crates/*` workspace exclusion is added |

## Consequences

Naming the checkpoint file's `lanes` field as a required contract, rather than an informal habit,
makes a GPU-contention violation something a reader can check mechanically (compare `ruv_gpu_processes`
against the checkpoint) rather than something that surfaces only as an out-of-memory crash or a
silently-corrupted concurrent run. The resumability rule directly protects against repeating the kind
of loss the S2c interruption already demonstrated is real and costly (≈1 GPU-h), generalized to any
future long-running capture or training rung this mission runs (M4b's fresh 3B-receiver calibration
and the mandatory scale-control arm, per ADR-024, are both exactly this kind of long-running,
interruption-prone GPU work). The `.gitignore` rule is the cheapest of the six to enforce and the one
with a concrete, already-occurred failure (7,071 artifacts near-committed) as its motivating case —
stating it as a standing rule rather than a one-off fix is the point.

## Implementation status

Partially implemented already, formalized here: the `.gitignore` fix for `latentmesh-train` landed
as part of the M3 near-miss recovery (already in the tree, verified this session), and
`.ruvnet-brain/checkpoint.json`'s `lanes` field already exists and is in active use — this ADR
promotes it from informal practice to a named contract, but changes no code. Not implemented:
automated enforcement of "one GPU-holding workflow at a time" (currently a manual checkpoint-file
discipline, not a mechanically-gated one) and no tooling yet exists to diff `ruv_gpu_processes`
against the checkpoint's `lanes.implementation` field automatically. Both are named as unscoped
future work, not committed to this wave.
