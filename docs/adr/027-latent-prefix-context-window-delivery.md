# 027. Latent-prefix context-window delivery (the M4.5 contingency)

- **Status**: Proposed. Design contract only — no code, no probe, no dataset in this wave. **Not
  part of run 2's current scope** (ADR-024's M3-M6 are unchanged by this registration).
- **Date**: 2026-08-28.
- **Related**: [009](009-online-causal-control-loop.md) (the online control loop this would be a
  different `Authority` tier's live exercise of), [023](023-live-four-condition-run1-pre-registration.md)
  (the pre-registration discipline any future activation of this rung must follow — restated
  below), [024](024-run2-trained-thought-adapter-ladder.md) (the ladder this ADR registers as a
  named contingency, "M4.5," without adding it to that ladder's actual milestones)

## Context

> **Prior-art update (2026-08-28, sweep docs/research/028)**: Interlat
> (arXiv:2511.09149) is directly relevant trained prior art for prefix-style
> latent conditioning; consult before activating this rung.

Run 1 (ADR-023 § S6) falsified mid-layer injection with a training-free linear map: a
well-fitting, bit-verified affine transform produced a receiver-side vector statistically
indistinguishable from norm-matched random noise, at both registered depth pairs, under both
calibration distributions. Run 2 (ADR-024) responds with trained nonlinear adapters (MLP,
FastGRNN, MicroLoRA), still targeting the same mid-layer overwrite mechanism run 1 used — a
placeholder-slot injection at a specific decoder block, requiring the injected vector to be valid
*mid-layer activation geometry*: correct norm, correct rotary-position phase relative to sequence
position, correct residual-stream statistics at that exact layer.

There is a second, mechanically distinct integration point `latentmesh-gate` already names but has
never specified the mechanics of: `Authority::LatentPrefix` (`crates/latentmesh-core/src/*.rs`,
between `ContextInject` and `ActionInfluencing` in the four-tier ladder `ObserveOnly →
ContextInject → LatentPrefix → ActionInfluencing`), reachable today via
`CeilingThresholds::run1()`'s recalibrated ladder (`delta_v > 0.05 → LatentPrefix`, `> 0.15 →
ActionInfluencing`, `crates/latentmesh-gate/src/lib.rs`). The gate crate names this authority
level and its threshold; nothing in the repo yet defines what "prefix" delivery actually means
mechanically at the receiver. This ADR is that specification, registered as a pre-named fallback
if M3/M4's mid-layer approach fails again, not as new work scheduled now.

## Decision

**Register latent-prefix delivery as the pre-named next contingency if run 2's mid-layer injection
gates (M3, M4) both fail** — this is the ADR-023/024 discipline of naming a fallback in advance,
applied one level up (a different *integration mechanism*, not just a different architecture
within the same mechanism M3/M4 already use).

### Mechanism

Retrieved or translated latent state enters the receiver as **soft-prefix conditioning at the
input boundary**, not a layer-`k` hidden-state overwrite. Concretely: the aligned/translated
vector(s) are prepended to the receiver's input embeddings before the first decoder block runs —
analogous to prefix-tuning / soft-prompting — rather than injected mid-stack at a specific block
the way ADR-023's 8-slot placeholder overwrite worked. `latentmesh-gate`'s `Authority::LatentPrefix`
tier is the designed authority level for exactly this mode: a step above `ContextInject` (which
this repo's `LatentFrame` doc-comment already describes as "the receiver may use the frame as
retrieval context, soft-prompt-like") but below `ActionInfluencing`, matching prefix delivery's
actual trust profile — more binding than pure retrieval context, less binding than steering tool
or action selection.

### Scientific rationale — a strictly easier transfer problem, not a retry of the same one

Mid-layer overwrite requires the injected vector to satisfy geometric constraints specific to one
exact point in the network's computation: the receiver's own attention and downstream layers
implicitly assume residual-stream statistics, rotary-position phase, and norm distribution
consistent with what *that model, at that layer, on that token position* actually produces.
Run 1's negative result is precisely that a training-free linear map does not produce a vector
satisfying those constraints in a causally useful way. **Prefix conditioning does not carry the
same requirement.** It only requires the injected content to be *useful as input* — something the
receiver's own attention mechanism can choose to attend to, discount, or ignore, the same way any
token embedding is optional context for what follows. This is a strictly easier problem:

- **Failure mode differs.** Mid-layer overwrite that fails can actively corrupt downstream
  computation (a geometrically wrong vector at layer `k` propagates errors through every
  subsequent layer). A prefix that fails to help degrades gracefully toward the no-injection
  baseline — the receiver's attention simply learns to discount an unhelpful prefix, the same way
  it discounts an unhelpful sentence of retrieved context.
- **Run 1's null result does not transfer automatically.** A vector that was causally inert as a
  mid-layer overwrite (run 1's finding) is not thereby shown to be uninformative as prefix content
  — the two integration points test different properties of the same aligned vector. This is a
  hypothesis this registration states, not a result: it has not been tested, and this ADR does not
  claim prefix delivery *will* work, only that run 1's failure does not settle the question for
  this different mechanism.

### Hard requirement: this needs its own pre-registration before any probe

**Activating this rung is out of scope for run 2 as currently pre-registered.** ADR-024's M3
through M6 stand completely unchanged by this ADR. If and when M3/M4 both fail their frozen-probe
gates and this contingency is actually pursued, it requires **its own pre-registration addendum**
— statistics, thresholds, the probe protocol (frozen or newly designed), dataset discipline, and
an explicit will-not-claim list — frozen before any probe consumes an eval item, exactly the S2-gate
discipline ADR-023 established for run 1 and ADR-024 inherited for run 2. This ADR registers the
concept, the mechanism, the authority-tier mapping, and the rationale for why it's worth naming in
advance; it does not freeze a single statistic, seed, or architecture, because none of those can be
honestly frozen before the team knows whether M3/M4 actually fail and what, specifically, a prefix
probe would need to test.

## Consequences

Naming this contingency now, while ADR-024 is fresh, means a future "M3/M4 both failed, what next"
decision has a pre-thought-through option on the table with its scientific rationale already
examined, rather than an ad hoc proposal invented under the pressure of a second negative result.
Being explicit that this requires its own pre-registration prevents the opposite failure mode: a
team that just watched two rungs fail reaching for a lower-trust integration point and skipping the
freeze-before-probe discipline specifically because urgency makes that discipline feel like
friction — the same discipline the negative result itself exists to demonstrate the value of, not
to be abandoned exactly when it's least convenient.

## What is simulated / what is hardware-pending

| Claim | Status |
|---|---|
| `Authority::LatentPrefix` exists and is reachable via `CeilingThresholds::run1()` | **Verified against current code** — `crates/latentmesh-core`'s `Authority` enum and `crates/latentmesh-gate/src/lib.rs`'s threshold ladder, both read this session |
| Prefix conditioning is a strictly easier transfer problem than mid-layer injection | **Stated as scientific rationale, not tested** — no probe of any kind has run against this hypothesis |
| Any prefix-delivery mechanism, probe, or dataset | **Not implemented, not scheduled** — this ADR is a registration, explicitly not an addition to run 2's active scope |
| A future pre-registration addendum for this rung | **Required before activation, not yet written** |

## Implementation status

Not implemented, and not scheduled. This ADR registers a fallback concept by name, mechanism, and
rationale for a future coordinator decision; it commits no code, probe, or timeline.
