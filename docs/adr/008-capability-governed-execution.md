# 008. Capability-governed execution

- **Status**: Proposed. **The admission gate is implemented** (`crates/latentmesh-gate`); RVF packaging and RVM enforcement are not wired.
- **Date**: 2026-08-18
- **Related**: [002](002-latent-packet-protocol.md) (`Authority`, `Provenance`), [003](003-causal-edge-verification.md) (a rejected edge never reaches this gate), [001](001-latentmesh-architecture-and-prior-art.md) §3 (LCGuard prior art)

## Context

Latent payloads are not human-inspectable. A malicious or corrupted latent vector could in principle manipulate a receiving model without producing any suspicious textual content — the entire premise of "govern this like code review" breaks down because there is no diff to read. LCGuard (reported, ADR-001 §3) treats KV state itself as a security boundary and attempts to strip reconstructable sensitive information before transmission — a related but distinct concern (data exfiltration) from this ADR's concern (unauthorized *influence*).

This stack already has a proven answer to "an opaque, consequential action needs a governance gate before it executes" — `cognitum-one/slack`'s AGL module (its ADR-0008/0009/0010): typed mutations, an authority ceiling that can only shrink down the lineage, a hard requirement for a rollback/observability path, and an admission function that returns the *first* violated rule. This ADR ports that shape onto latent execution instead of code mutation.

## Decision

**A latent frame may only take effect at its receiver if:**

```
execute(z)  ⟺  signature(z) ∧ authority(z) ∧ provenance(z) ∧ risk(z) < τ
```

- **`signature(z)`** — the frame's `transform_hash` (ADR-002) is checked against a known, pinned alignment transform; a frame claiming an alignment that doesn't match a registered transform is rejected outright (mirrors AGL's "authority never silently expands" — you cannot claim a better transform than what was actually run).
- **`authority(z)`** — the frame's `Authority` (`ObserveOnly < ContextInject < LatentPrefix < ActionInfluencing`, ADR-002) must not exceed the receiver's configured ceiling for that sender/edge; an edge that hasn't passed ADR-003's causal verification is capped at `ObserveOnly` regardless of what authority it requests — **causal unverified edges never reach `ActionInfluencing`**, which is the concrete link between ADR-003 and this ADR.
- **`provenance(z)`** — the frame's `Provenance.context_hash` and `parents` chain (ADR-002) must resolve; a frame with a dangling or unverifiable lineage is rejected, matching AGL's "no rollback target, no admission" — here, "no traceable origin, no execution."
- **`risk(z) < τ`** — a trajectory-level risk score (not a per-frame classifier — DreamGuard's reported approach models *trajectory* risk rather than scoring actions independently, at a reported ~25ms average guardrail overhead). This repo's `risk()` is an explicit, documented **stand-in**: a deterministic function of confidence, authority requested, and provenance depth, clearly not a trained risk model — using it as if it were validated would misrepresent what's actually here (ADR-001 §8's honesty rule applies).

`Gate::admit(frame, policy) -> Result<(), AdmissionError>` returns the **first** violated rule, in the same style as AGL's `Mutation::admissible` — deterministic, clock-free (a `now` timestamp is passed in), independently re-runnable.

## Consequences

- The gate composes directly with ADR-003: an edge that hasn't earned causal verification is structurally capped below `ActionInfluencing`, so "this connection looked useful in one run" can never itself grant execution authority — only the four-control test can raise the ceiling.
- The `risk()` stand-in is explicitly not a claim of having built DreamGuard-equivalent trajectory-risk modeling; shipping it unlabeled would be a materially misleading claim about this repo's safety posture, which is why this ADR names it as a placeholder in the decision text itself, not buried in a caveat.
- RVF (packaging/provenance) and RVM (capability-ceiling enforcement at the runtime level, the way it already enforces authority ceilings for code) are the natural production targets for this gate's output — not implemented this session; `latentmesh-gate`'s policy type is deliberately serde-friendly so it can be carried inside an RVF-packaged artifact later without a redesign.

## Implementation

- `crates/latentmesh-gate/src/lib.rs` — `Policy { authority_ceiling_by_edge: HashMap<(sender, receiver), Authority>, risk_threshold: f32, known_transforms: HashSet<transform_hash> }`, `AdmissionError::{UnknownTransform, AuthorityExceedsCeiling, UnresolvedProvenance, RiskTooHigh}`, `Gate::admit(frame, policy, now) -> Result<(), AdmissionError>`. Tests: a frame with a valid signature/authority/provenance/risk is admitted; each rule independently rejects when violated (four tests, one per rule, proving the "first violated rule" ordering is deterministic); an edge with no causal-verification record cannot be admitted above `ObserveOnly` even if it requests `ActionInfluencing`.
- `crates/latentmesh-gate/src/causal.rs` — ADR-003's `verify_edge`/`permutation_test`, consumed by `Policy` construction (a verified edge's `ΔV`/`p_value` is what raises its `authority_ceiling`).
