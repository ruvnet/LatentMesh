# 035. M4b scale-control arm: pre-registration addendum

- **Status**: **BLOCKED PERMANENTLY (2026-08-29) — will not execute under this
  apparatus.** This ADR registered M4b as *mandatory regardless of any other
  rung's outcome*; **that mandate is now discharged as unsatisfiable, not
  waived.** PC3 established out of sample that payload content contributes
  nothing to decisions (p = 0.72, fully powered), and M4b's receiver-scale arm
  varies the payload's source model — a content factor. Running it could not
  distinguish "scale does not help" from "the apparatus cannot express any
  content at the decision level". See
  [research/048](../research/048-run2-final-synthesis.md). The pre-registration
  below is preserved unaltered and remains valid under an apparatus shown to
  steer decisions.
- **Superseded status**: Proposed (pre-registration — frozen before any M4b item is consumed, per ADR-023's
  own discipline, inherited by every ladder rung since). M4b is the ladder's one contingency
  registered as **mandatory regardless of any other rung's outcome** (ADR-024 §"Registered
  confound — receiver-scale threshold"); this document exists so M4b can execute the moment the
  GPU frees (currently held by M4g's training run — confirmed live this session via
  `ruv_gpu_processes`: `target/release/train_m4g_fuse`, 10,322 MiB, per ADR-034's one-lane
  discipline) without a second authoring pass.
- **Date**: 2026-08-29.
- **Numbering note**: `docs/research/035-probe-task-selection.md` is a different document (a
  research finding, not this ADR) that happens to share the number 035 — the same independent
  numbering-sequence situation ADR-025/026 already flagged for their own research-doc collisions.
  No relationship is implied by the shared digit.
- **Related**: [024](024-run2-trained-thought-adapter-ladder.md) (the ladder this arm belongs to —
  read in full for this ADR, all ~1,200 lines, including the M3/M4/M4c/M4d/M4g outcome and
  pre-registration sections, the manifold DIAGNOSIS, and the receiver-scale confound's original
  registration), [023](023-live-four-condition-run1-pre-registration.md) (the pre-registration
  template this ADR mirrors), [030](030-run3-causally-gated-text-pre-registration.md) (the most
  recent pre-registration — mirrored for its amended-statistics structure), [031](031-evidence-receipt-and-statistical-protocol-governance.md)
  (append-only correction discipline, the receipt contract, the frozen-probe-is-a-one-shot-resource
  principle), [032](032-negative-result-publication-contract.md) (the publishability bar this
  rung's eventual outcome, pass or fail, must clear), [034](034-concurrent-lane-resource-scheduling.md)
  (the GPU-lane discipline this rung's execution must follow — one holder at a time, named in the
  mission checkpoint, queue rather than preempt)
- **Evidence base**: `docs/research/039-bidirectional-latent-exchange.md` §1.1 (the Bicameral
  corroboration cited below), `docs/research/036-manifold-collapse-across-the-ladder.md` and
  `docs/research/040-the-pooling-gap.md` (the manifold/pooling findings that constrain which
  adapter this arm carries over), receipted GPU-second figures re-verified directly against
  `crates/latentmesh-runtime/receipts/{run2-pertoken-dump-receipt.json,
  run2-m4c-training-receipt-cellL18toL14.json,
  run2-m4c-receipt-cellL18toL14-mlp-taskloss-slots8-poolfull-rescaletrue-n40.json}` this session
  (not retyped from a prior summary — see § Cost below for the exact fields read), model inventory
  and VRAM arithmetic verified live this session (`ls ~/.cache/huggingface/hub`, `config.json`
  reads for Qwen2.5-3B/7B-Instruct, `ruv_gpu_status`/`ruv_gpu_processes`)

## Context

arXiv:2608.05164 ("Cross-Architecture Steering Transfer," surveyed in `docs/research/027` and
first registered as a confound in ADR-024 while M3 was in flight) reports cross-model
latent-steering transfer works reliably only above a **~1.7B receiver-parameter threshold**.
Every null this repository has produced — run 1's training-free affine bridges and run 2's M3
(MLP), M4 (FastGRNN ×3), M4c/M4d (task-loss variants), and M4g (fuse, executing now) — has used
Qwen2.5-1.5B-Instruct as the receiver, sitting just below that threshold. Every one of those
results is therefore scoped, by ADR-024's own registration, to "at a sub-threshold receiver" until
this arm reports.

**Independent corroboration, verified this session** (`docs/research/039` §1.1, cited directly
here since it is the second, independent reason this arm is mandatory): the Bicameral Model
(arXiv:2605.11167) **degrades on GSM8K specifically** — 49.6% → ~40% — when the capability gap
between its two coupled models is small. This is the *same task* this repository's own ladder
uses, and it is evidence for the receiver-scale confound from a source that has nothing to do with
LatentMesh's own architecture choices.

M4b is registered here as **the ladder's one mandatory-regardless contingency**: independent of
whether M4g (fuse), M4f (manifold-constrained), or M4h (de-pooling) pass or fail, this arm must
still run, because none of those other rungs vary the one factor this arm exists to isolate.

## Decision — the model pair (the harder question, addressed first)

### What is actually available on this host, verified this session

| Model | Downloaded | `hidden_size` | `num_hidden_layers` | Disk size | BF16 weight footprint (≈ params × 2 B) |
|---|---|---:|---:|---:|---:|
| Qwen2.5-1.5B-Instruct | yes | 1536 | 28 | — | ≈3.1 GB (current receiver, every prior rung) |
| Qwen2.5-3B-Instruct | yes | 2048 | 36 | — | ≈6.2 GB (current sender, every prior rung) |
| Qwen2.5-7B-Instruct | **yes** (`~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct`, verified this session) | 3584 | 28 | 15 GB on disk | ≈15.2 GB |

**The 7B checkpoint is already on this host** — the task brief's "needs download" concern for a
larger-sender option does not apply; the actual constraint is VRAM, not acquisition.

### The three options the task brief named, worked through with real numbers

**(a) 3B self-pair (Qwen2.5-3B-Instruct as both sender and receiver, two independently loaded
instances of the identical checkpoint).** VRAM: 6.2 GB + 6.2 GB = 12.4 GB weights, against the
design's own established ~14.3 GB free-VRAM budget (`docs/research/024` §2, §9 risk 10) — leaves
≈1.9 GB for KV cache and activations at batch=1, comparable to (tighter than, but not qualitatively
different from) every prior rung's headroom, and within the same mitigation envelope the design
already committed to (1024-token cap, batch=1, KV reset per episode). **Feasible today, zero new
downloads, zero new engineering.**

**(b) A larger sender, paired with the receiver promoted to 3B (e.g. sender=7B, receiver=3B — the
reading of the brief's "larger sender" option that keeps sender ≠ receiver once the receiver moves
to 3B).** VRAM: 15.2 GB + 6.2 GB = 21.4 GB weights alone — this **does not fit on a 16.3 GB
card**, with zero margin remaining even before KV cache. `docs/research/024` §1 already
adjudicated the general shape of this problem once, for a different pair, and rejected it on
identical grounds: rigor-first's judged-infeasible design was overridden specifically because it
required "sequential persist-and-swap machinery forced by 12.6/14.3 GB marginal VRAM" — machinery
this repository has never built and does not have receipted evidence for. **Rejected for this
wave**, not because the pair is conceptually wrong, but because it requires exactly the
infrastructure this project's own earlier design deliberation already declined to build.

**(c) Keep 3B→1.5B and additionally run 3B→3B as a paired comparison.** This does not name a new
data point beyond (a) — 3B→1.5B is already the ladder's existing baseline (every prior rung), so
"additionally run 3B→3B" reduces to option (a) with the existing rungs serving as the comparison
point, which they already do by construction (every M4b receipt reports against the same frozen
probe the M3/M4/M4c/M4d/M4g rungs used).

### Decision: (a), with the self-pair confound registered explicitly, not silently absorbed

**Sender = Qwen2.5-3B-Instruct (unchanged from every prior rung — the single controlled
continuity). Receiver = Qwen2.5-3B-Instruct (a second, independently loaded instance of the
identical checkpoint), clearing the ~1.7B threshold by a comfortable margin.**

**What this design can and cannot conclude, stated before any run:**

- **A NULL result is clean and strengthens the joint negative unambiguously.** A same-checkpoint
  self-pair is, if anything, an *easier* transfer problem than genuine cross-architecture transfer
  — the two models share an identical embedding space, identical residual-stream statistics at
  matched relative depth, and identical tokenizer, none of which any cross-model rung in this
  ladder has enjoyed. If even this easier configuration nulls, the receiver-scale hypothesis is
  substantially weakened as an explanation for the ladder's prior nulls, and the joint negative
  gets stronger with no interpretive hedge required.
- **A PASS is ambiguous between two explanations and must be reported as such, not resolved by
  fiat.** A same-checkpoint self-pair is structurally close to S1a's own self-pair
  identity-transform mechanics probe (ADR-023) — which already passed, at p=0.03125, using a
  1.5B model paired with itself. A 3B→3B PASS could mean "receiver scale ≥1.7B fixes transfer" (the
  registered hypothesis), or it could mean "same-checkpoint pairs transfer more easily than
  cross-checkpoint pairs, independent of absolute scale" (a different, already partially observed
  phenomenon). **This ADR pre-commits: a PASS here does not, by itself, license the claim
  "receiver scale ≥1.7B resolves the causal null for genuinely cross-model pairs."** It licenses
  the narrower claim "a ≥1.7B same-checkpoint pair can transfer," and names the clean
  cross-architecture test (below) as the follow-up required to separate the two explanations.
- **This design is NOT trivially degenerate, despite matched dimensions.** Sender and receiver
  process genuinely different token sequences at genuinely different pipeline positions even
  though they share weights: the sender processes the full reasoning-generation context and
  captures at its own generated span; the receiver processes only the question and receives
  injected content into an otherwise-unassisted continuation. Same-weights does not mean
  same-activations — this is structurally the same relationship S1a's mechanics probe already
  exercised at 1.5B scale (self-pair, non-trivial result, p=0.03), not a same-input no-op.

### The cleaner design, named as future work, not built here

A genuinely cross-architecture receiver at or above 1.7B (e.g., Qwen2.5-7B, or a different-family
model), paired with the unchanged 3B sender, is the design that would separate "scale" from
"same-checkpoint ease" cleanly. It requires either (i) quantized weight loading (int8/4-bit) to
fit both models' footprint within the ~14.3 GB budget — **not currently implemented or verified
anywhere in `latentmesh-runtime`/`latentmesh-train`**, a new engineering lift with no receipted
precedent — or (ii) sequential persist-and-swap machinery, which `docs/research/024` §1 already
declined to build once for feasibility reasons that have not changed. Neither is scoped or
scheduled by this ADR. If M4b's self-pair result is a PASS, resolving this ambiguity by building
one of these two paths becomes the natural next-ADR's job; if it is a NULL, the ambiguity is moot
for the purpose of strengthening the joint negative (per the asymmetric reading above), though the
cross-architecture design would still be worth running eventually to rule out "the self-pair
happened to be an unusually hard configuration."

## What must be re-derived vs. reused

**Everything downstream of the receiver's identity changes. Nothing about the frozen probe
protocol itself does.**

- **New hidden dimensions**: none, numerically — Qwen2.5-3B's `hidden_size=2048` matches the
  sender's own, so the injection no longer needs `latentmesh-align`'s rectangular Procrustes path
  at all (2048→2048 is square). This is disclosed as a structural simplification relative to every
  prior rung, not hidden: it removes one degree of difficulty (dimension mismatch) that every
  earlier cross-model attempt had to solve, which is a second reason a PASS here needs the
  ambiguity caveat above.
- **New layer indices**: the receiver's injection block must be re-selected. Reusing the same
  *relative* depth convention the ladder has used throughout (S2's winning cell was ≈50% relative
  depth: sender block 18 of 36, receiver block 14 of 28 for the 1.5B receiver) — for a 3B receiver
  (36 layers, same as the sender), the natural analogous choice is **receiver block 18 of 36** (the
  same absolute block index as the sender's own capture point, both at 50% relative depth). This is
  registered as the frozen injection site; no depth sweep is re-run for M4b (a 3×3 sweep here would
  be new, unregistered scope beyond what this arm needs to test).
- **New natural-norm statistics**: mandatory, per the same rule ADR-024's M5 redesign already
  established for a changed receiver — "baseline, zero-vector and random conditions must be
  re-measured fresh under the same adapted receiver; comparison against earlier rungs'
  frozen-receiver baselines is invalid for causal-gate purposes." A fresh S0-style mechanics pass
  (capture shape, injected-logits-differ-from-baseline, zero-slot-equals-baseline, injected-norm-
  within-band) is required before any training or probe draw, producing its own receipt in the
  same style as `s0-receipt.json`.
- **New calibration/capture**: a fresh teacher-forced per-token capture over the 3B-as-receiver
  pipeline, following M2's existing convention (`crates/latentmesh-runtime/src/capture.rs`'s
  `forward_capture_multi`, reused unmodified — it already supports arbitrary hidden dims and layer
  indices, no code change required for a same-family receiver swap). Same source token streams
  (`harness/latentmesh-live/data/s2c-token-streams.jsonl`) can be reused for the *sender* side
  unchanged; only the *receiver* side needs fresh forward passes, since the receiver model itself
  has changed.
- **The frozen 40-item probe's baseline/zero/random conditions must be RE-MEASURED under the new
  receiver — this is the invariant M5's redesign already identified, restated here because M4b is
  the rung that actually exercises it first.** Comparing M4b's aligned condition against the
  1.5B-receiver's baseline/zero/random accuracy numbers (23/22/24/21 style figures from M4c/M4d)
  would be comparing across two different receivers and is invalid by construction. M4b's own
  receipt must carry its own freshly measured `baseline_uninjected`, `zerovec_injected`, and
  `random` accuracy figures, scored on the same 40 frozen items, through the same 3B receiver.
- **What is reused verbatim**: the 40 GSM8K-train item indices and seed (`item_seed_chacha8:
  20897`), the one-sided statistical test family, α=0.05, the greedy/batch=1/max-400-token
  decoding, the leakage-exclusion list (still the same 13 rows, since the *item set* is unchanged —
  only the receiver model is), and the mechanical `eval-200`/`holdout-100` lock (still engaged;
  M4b does not touch it).

## Which adapter(s) to carry over

**Reconstruction-trained is the registered primary choice, for a reason grounded directly in
`docs/research/036`'s findings, not chosen by default.** The manifold pre-check established that
reconstruction-trained adapters (M3's MLP, M4's three FastGRNN ranks) land convincingly on the
manifold of pooled receiver states (cosine 0.975–0.996) while both task-loss adapters (M4c, M4d)
sit off-manifold from their untrained initialization onward (cosine −0.02/+0.05) and actively harm
the receiver (0/40 NLL wins against every control). An M4b run using a task-loss-trained adapter
would import that off-manifold, actively-harmful signature into a new receiver before this arm has
any evidence the mechanism differs at larger scale — confounding "does scale fix transfer" with
"does scale fix an already-known-broken training objective." **M4b therefore carries over an
M3-shaped reconstruction-trained MLP** (2048→512→2048 for the now-square dimension, architecture
otherwise unchanged from M3's registration), freshly trained on the new receiver's captured
per-token pairs.

**Task-loss variants are explicitly excluded from this arm's primary registration**, for the same
reason: M4c/M4d's target-mismatch finding (the adapter learns to reproduce the sender's own
reasoning span rather than the receiver's answer format) is a property of the *loss function*, not
of receiver scale, and conflating the two axes in one rung would make M4b's result impossible to
attribute cleanly. If M4c/M4d's task-loss finding is worth re-testing at 3B receiver scale, that is
named as optional follow-up comparative work, not part of M4b's registered gate.

**Conditional inheritance from M4g, registered now since M4g's outcome is not yet known as this
document is written**: if M4g (overwrite-vs-fuse) reports a PASS before M4b's training run starts,
M4b's adapter uses the **fuse** injection operator (residual add, `h[slot] += c·v`) instead of the
current overwrite (`slice_assign`), since a PASS at M4g would identify the injection operator as
the ladder's actual root cause and it would be a wasted GPU-hour to test scale-control under a
known-worse operator. If M4g reports a NULL, or has not yet reported when M4b's training run
starts, M4b uses the current **overwrite** operator, unchanged from M3/M4's registration. This
conditional is stated here, before either outcome is known, precisely so that whichever branch is
taken is a pre-registered choice, not a post-hoc one.

## Statistics

- **Primary statistic: mid-p McNemar** (Fagerland, Lydersen & Laake 2013), with the classic exact
  one-sided sign-test p reported alongside, never replacing it — the same convention ADR-030
  registered and ADR-024's M3/M4/M4c/M4d outcome annotations already retrofit. α=0.05.
- **Power expectation, stated up front, not discovered after the draw.** `docs/research/031`'s
  power-floor arithmetic applies identically here: at `n_disc≤4`, the minimum attainable one-sided
  exact-sign p is above α=0.05 regardless of true effect size. The ladder's most recent
  cross-model draws (M4c at `n_disc=6`, M4d at `n_disc=7`) were the first to clear that floor —
  **M4d is the ladder's first genuinely informative negative to date, at `n_disc=7`, minimum
  attainable p=0.0078.** This ADR sets the same expectation for M4b explicitly: if M4b's draw
  lands at `n_disc≤4`, its result must be reported with the identical structural caveat every
  earlier low-`n_disc` draw now carries — "could not have detected an effect at this discordant
  count," not "no effect exists" — rather than being read as a clean negative by default. This is
  registered before the draw happens specifically so a low-`n_disc` M4b result cannot be
  mis-reported as more informative than M4d's own worked precedent shows it would be.
- **Multiplicity position.** M4b is one more rung in the same fixed-sequence gatekeeped ladder
  ADR-024/`docs/research/031` §4.3 already established needs no additional between-rung
  correction, since it is evaluated once, after its predecessors' results are already on record,
  and does not run in parallel with any other rung. No Holm-Bonferroni correction applies to M4b
  itself (it is one test, not a parallel-variant family); if M4b's own registration later grows a
  parallel variant (e.g. testing the fuse and overwrite operators side by side rather than the
  conditional-inheritance rule above), that would require its own Holm-Bonferroni correction at
  that point, per the same rule already stated for the rest of the ladder.
- **Does ADR-030's e-process design apply here? No, by default — registered as an optional
  extension, not adopted as the primary test.** ADR-030's anytime-valid sequential test was
  designed for run 3's specific situation: one channel, one condition, needing a confirmatory scale
  beyond a single 40-item draw. M4b instead follows the ladder's standing one-registered-draw-per-
  architecture protocol (M3, M4, M4c, M4d, M4g all used exactly this shape) — treating each rung as
  a single test that escalates to the next architecture on failure, not as one architecture
  accumulating sequential evidence. Because M4b is the ladder's one mandatory-regardless rung,
  **this ADR registers the e-process as an available option, not a requirement**: if the primary
  40-item draw lands in the power-floor dead zone (`n_disc≤4`) or produces an ambiguous
  near-miss, the e-process's `adaptation-512`-drawn sequential extension (identical betting-rule
  registration text as ADR-030 §"Concrete, paste-able registration text," λ=0.30, `N_max≈300`) may
  be invoked as a **second, separately reported statistic** on the same trained adapter — never as
  a replacement for the primary 40-item draw, and never re-run against the frozen probe itself.
  This option is named now so that invoking it later, if warranted, is a pre-registered choice.

## Cost, from receipted rates

Every figure below is either read directly from a committed receipt this session, or explicitly
marked as an extrapolation with the reason stated.

| Stage | Receipted precedent | M4b estimate |
|---|---|---|
| Per-token capture (receiver side only — sender-side streams reused) | `run2-pertoken-dump-receipt.json wall_clock_s.total = 182.66 s` (both sender and receiver captured, 1.5B receiver) | **Likely higher than 182.66 s** — the 3B receiver has roughly 2× the FLOPs of the 1.5B receiver per forward pass, and only the receiver side needs re-capturing (sender-side reuses existing streams); no fresh receipt exists yet, so this is an order-of-magnitude expectation, not a pre-measurement. Budget ≈5-10 min. |
| Fresh baseline/zero/random mechanics pass (S0-style) | not separately receipted at this granularity elsewhere in the ladder | Minutes-scale, by analogy to the original `s0-receipt.json`'s 2.51 s for 3 items scaled to 40 items and a slightly larger model |
| Adapter training (M3-shaped MLP, single seeded run) | `run2-m4c-training-receipt-cellL18toL14.json wall_clock_s = 1603.48 s` (≈0.446 GPU-h, task-loss MLP against the 1.5B receiver) | **Comparable order of magnitude**, likely somewhat higher for the larger receiver's forward/backward cost; budget ≈0.5-0.8 GPU-h |
| One registered frozen-probe draw | `run2-m4c-receipt-cellL18toL14-mlp-taskloss-slots8-poolfull-rescaletrue-n40.json wall_clock_s = 422.48 s` (≈7 min, against the 1.5B receiver, 5 conditions × 40 items) | **Comparable order of magnitude, likely higher** for the larger receiver's per-token generation cost; budget ≈8-12 min |
| **Total estimated** | | **≈1-2 GPU-hours**, all figures order-of-magnitude extrapolations from receipted precedent, not fresh measurements — stated as such, not rounded to false precision |

**GPU-lane discipline (ADR-034)**: M4b's execution must be named in `.ruvnet-brain/checkpoint.json`'s
`lanes.implementation` field before any GPU-resident step starts, and must queue behind whatever
currently holds the lane (M4g at the time of this ADR's authoring) rather than preempt it.

### Honest-fail path

Unchanged from every other rung in the ladder (ADR-024, ADR-032): the primary 40-item draw runs
once, its full receipt is committed regardless of outcome, and a FAIL is reported with its exact
n_disc, mid-p, and the power-floor caveat if applicable — never retried with adjusted
hyperparameters against the same frozen probe. If the e-process option (above) is separately
invoked, its own trajectory is committed in full regardless of outcome, per the identical
never-restarted rule ADR-030 already states for that statistic. A NULL at M4b removes the
receiver-scale confound from every prior null's scope (subject to the self-pair ambiguity caveat
registered above) and is reported per ADR-032's publishability contract like any other rung's
result.

## What M4b does NOT test

- **Pooling** (M4h's registered scope) — M4b's adapter still pools the sender's per-token span into
  one vector before injection, exactly as every prior rung has. A NULL at M4b does not rule out
  that de-pooling (M4h) would succeed at the larger receiver scale too; that combination is not
  tested here.
- **Overwrite vs. fuse** (M4g's registered scope) — M4b inherits whichever operator M4g's outcome
  dictates (conditional rule above), but does not independently test the operator axis; it is a
  single-factor arm on receiver scale, not operator choice.
- **Continuous injection** (M4e's registered, deprioritized scope) — M4b's adapter delivers content
  one-shot into the 8 placeholder slots and then free-runs, exactly as every prior rung. This axis
  remains untested by M4b regardless of outcome.

**The interpretation registered in advance, before any run**: a PASS scopes every prior null
(run 1's affine bridges, M3, M4, M4c, M4d, M4g) to "at a sub-threshold receiver," subject to the
same-checkpoint ambiguity caveat registered above for what exactly a PASS here licenses claiming. A
NULL removes the ladder's most-cited confound and substantially strengthens the joint negative —
the pattern would then hold across a training-free affine map, two trained architectures, two loss
functions, one injection-operator variant (pending M4g), and now receiver scale, at a configuration
this ADR argues should have been *easier* than any genuinely cross-architecture test, not harder.

## What is frozen / what is pending

| Item | Status |
|---|---|
| Model pair (3B self-pair, with the self-pair ambiguity registered), injection site (block 18→18, same relative depth convention), adapter architecture (M3-shaped reconstruction MLP, square-dimensioned), the M4g-conditional operator-inheritance rule, statistics (mid-p McNemar primary + exact-sign secondary, power expectation, multiplicity position, e-process as optional not mandatory), cost estimate and honest-fail path, scope boundary (pooling/operator/continuity excluded) | **Frozen by this ADR** |
| Fresh receiver-side per-token capture, fresh mechanics/baseline receipt, fresh adapter training | **Not started** — queued behind whichever lane currently holds the GPU, per ADR-034 |
| The registered frozen-probe draw | **Not started** |
| The cleaner cross-architecture receiver-scale design (7B or quantized) | **Named as future work — not scoped, not scheduled** |

## Implementation status

Not implemented. This ADR is a complete pre-registration — model pair, injection site, adapter
choice (with its M4g-conditional), statistics, cost, and scope boundary are all specified in full,
sufficient to execute M4b directly from this document the moment the GPU lane frees.
