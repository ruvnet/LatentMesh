# 003. Causal edge verification

- **Status**: Proposed. **The statistical machinery is implemented** (`crates/latentmesh-gate`'s `causal` module); it has not been run against a live multi-agent task. Updated 2026-08-28: prose corrected from four controls to the five the code has enforced since the 2026-08-18 `text_equivalent` revision (the doc/code drift named in `docs/research/024` §8 risk #12).
- **Date**: 2026-08-18
- **Related**: [001](001-latentmesh-architecture-and-prior-art.md) §5 (why this is the actual bet), [006](006-self-evolving-topology.md) (consumes this ADR's output as the fitness signal), [008](008-capability-governed-execution.md) (an edge failing this test never reaches execution authority)

## Context

ADR-001 §5 established the central risk: an apparent gain from "agent A sent agent B a latent state" can come from message *presence*, extra *compute*, or a generic state effect — not from `A`'s specific information. Reported audits found cases where a relay's removal cost far more than replacing it with a *mismatched* state, meaning the relay's content barely mattered. Optimizing raw benchmark deltas over that confound would evolve a topology that is elaborate and expensive without being smarter than a much simpler control.

The reframe this ADR commits to: don't ask "did B receive something from A?" Ask:

```
ΔV_{A→B} = V(B | A) − V(B | control)
```

If `ΔV_{A→B}` is near zero, the edge contributes nothing and should be killed. If positive and significant, strengthen it. If it turns negative, route around it.

## Decision

**An edge `A → B` is admitted to the topology only after passing the five-control test, and it is re-verified on a rolling basis, not granted trust once.**

### The five controls

For a candidate edge, hold everything about `B`'s task, budget, and pipeline fixed and vary only what `B` receives from `A`:

| Control | What `B` receives instead of `A`'s real state |
|---|---|
| `zero` | No state at all (a zero vector / no injection) |
| `random` | A random vector of the same shape/norm distribution |
| `mismatched` | Another episode's real state — content-shaped, but from the *wrong* task |
| `self_generated` | `B`'s own prior state, fed back to itself |
| `text_equivalent` | The same content as the real state, serialized to text and re-tokenized instead of transferred as a latent frame — beating this specifically is what makes a surviving edge a claim about **latent** communication, not merely communication-vs-silence |

`V(·)` is the task's outcome value (accuracy, success, or any scalar the caller supplies). The edge's claimed value is:

```
ΔV_{A→B} = V(B | real A state) − mean(V(B | zero), V(B | random), V(B | mismatched), V(B | self), V(B | text_equivalent))
```

An edge survives only if `ΔV_{A→B}` is **positive and statistically significant** against the control distribution (this crate uses a nonparametric permutation test over paired trial outcomes — no distributional assumption on `V`, matching the honesty stance of not asserting Gaussian task-value noise). This is the operational form of asking whether `I(A; B | task)` — the mutual information the real state carries about the task, beyond what any control carries — is actually nonzero.

### Edge lifecycle

```
propose edge → run N paired trials (real vs. each control) → permutation test
  → survives (p < α, ΔV > 0)  → admitted, weight ∝ ΔV / cost (ADR-006 §bandwidth)
  → fails                     → not admitted / existing edge scheduled for removal
  → re-test on a rolling window (task distribution drifts; a once-useful edge can decay)
```

An edge that is `mandatory` for governance (e.g. a security-review agent — ADR-008) is never removed by this test regardless of measured cognitive value; §"mandatory control vs. useful cognition" below keeps that distinction explicit rather than silently conflating "contributes nothing measurable" with "unnecessary."

### Mandatory control vs. useful cognition

A governance-mandatory agent (security review, audit logging) can legitimately show `ΔV ≈ 0` — its job is not to raise task accuracy, it's to gate/observe. This ADR's test measures *cognitive* contribution only; ADR-008's authority model is the separate, non-overridable channel for control-plane agents. Conflating "this edge doesn't help the score" with "this edge is disposable" is exactly the mistake §1 above exists to prevent.

## Consequences

- The topology-evolution fitness signal (ADR-006) becomes "number and value of edges that survived counterfactual replacement," which directly answers the reviewer objection ADR-001 §5 names, instead of a benchmark number a skeptical reader can attribute to extra compute.
- Verification cost is real: every candidate edge needs `N` paired trials across 6 conditions (real + 5 controls) before it can be trusted, which is more expensive than measuring a single benchmark delta. This is a deliberate tradeoff — cheap-and-confounded is not an acceptable substitute.
- The permutation test is nonparametric by design (no assumption that task-value noise is Gaussian or that `V` is continuous), so it applies unchanged whether `V` is a 0/1 success flag or a continuous quality score.

## Acceptance test

Ten agents with deliberately redundant capabilities; 1,000 tasks; the system is allowed to mutate its own topology (ADR-006). **Pass** if the topology converges toward a smaller graph that reduces compute by at least 30% while maintaining or improving task success, **and** the surviving communication edges each individually pass this ADR's five-control replacement test. Either half without the other is not a pass: a smaller-but-unverified graph could be cheaper by accident; a verified-but-unshrunk graph hasn't demonstrated self-organization.

## Implementation

- `crates/latentmesh-gate/src/causal.rs` — `EdgeTrial { real, zero, random, mismatched, self_generated, text_equivalent }` (each a `Vec<f64>` of paired outcome values across trials), `sign_flip_permutation_test(differences, resamples, seed) -> p`, `EdgeVerdict::{Admit{delta_v, worst_p_value}, Reject{reason}}`, `verify_edge(trial, alpha, resamples, seed) -> EdgeVerdict` running all five controls and requiring significance against the worst (most favorable to the null) control, not just the mean — a stricter bar than the mean-of-controls formula above, so a marginal edge can't hide behind one weak control. Tests: a synthetic edge with real signal (real outcomes drawn from a shifted distribution vs. controls) is admitted; a synthetic edge with no signal (all six conditions drawn from the same distribution) is rejected at the expected false-positive rate across repeated trials; the permutation test's p-value is invariant to the outcome scale (rescaling `V` doesn't change the verdict).
