# 044. The RVM authority seam — what LatentMesh may never do, grounded in RVM's source

- **Status**: Proposed. **Date**: 2026-08-29.
- **Renumbered 043 → 044, and narrowed, during integration review.**
- **Defers to [043](043-cross-ecosystem-architecture-and-corrections.md)** for
  all cross-repo scope.

---

## Why this ADR was narrowed

**This document originally claimed a cross-ecosystem charter — "LatentMesh
decides what machine knowledge should move, in what representation, across
which link" — with a one-job-per-component table spanning RuVector, RuFlo,
MetaHarness, RVF, MidStream, Air and Core Memory.**

[ADR-043](043-cross-ecosystem-architecture-and-corrections.md) establishes that
writing such a spec here is the specific mistake `ruvector`'s ADR-305 was
written to fix: this repo's ADR-009 is scoped to **one bounded context** — the
Latent Communication Fabric — not the ecosystem. **That correction is accepted.
The charter claim is withdrawn.**

ADR-043 also lands a harder point this document should have made and did not:
**a cross-ecosystem spec asserting "LatentMesh owns cognitive state movement" as
settled fact overstates what this repo's own evidence supports.** Run 2's
receipts show the latent channel is decision-inert at single-layer
([research/048](../research/048-run2-final-synthesis.md)), and M5X is testing
the one remaining variable. Claiming ownership of a capability while its core
mechanism is under active falsification was the wrong posture.

**What survives is one thing ADR-043 does not cover, and it is worth keeping
separately: the authority seam, grounded in RVM's own source.**

## The seam, grounded

RVM is **capability-secure and proof-gated**. From
`ruvector/crates/rvm/crates/rvm-security/src/lib.rs` — implementing source,
not design intent — every hypercall passes a **three-stage gate**:

1. **capability check** — does the caller hold the required type and rights?
2. **proof-commitment verification** — is the transition attested?
3. **witness logging** — record the decision for audit

`PolicyDecision` is `Allow` or `Deny(RvmError)`; `enforce()` returns
`RvmResult<()>`. From `rvm-cap/README.md`: capabilities are **unforgeable**
tokens with **monotonic attenuation** — a partition may grant only what it
holds — delegation bounded at **8 levels**, with a non-transitive `GRANT_ONCE`.

### The consequence, stated precisely

> **RVM does not accept confidence. It accepts a capability token and a proof
> commitment.**

So there is **no gradient** along which latent state becomes "trusted enough."
It is evidence, or it is nothing. A design that raises a confidence score and
expects proportionally more authority has misunderstood the interface it is
calling.

**`crates/latentmesh-reasoning` enforces this structurally**: zero dependency on
`latentmesh-gate` or any authority type, so no capability-returning method could
be added without that dependency appearing in review.

⚠️ The broader "proof-gated microhypervisor" description (`concepts/rvm/CARD`)
is flagged **by the knowledge base itself** as *design intent with no
implementing source retrieved*. The three-stage gate and capability derivation
**are** source-backed; the **P3 zero-knowledge tier is not**. Do not design
against P3.

### Alignment with ADR-043

ADR-043 §3 finds independently that RVM's own docs never mention LatentMesh and
frame RVM around coherence domains for agent placement — **not** as a
downstream gate for external "candidate cognition." That is consistent with
what is written here: this ADR does **not** claim RVM will gate LatentMesh. It
records **what RVM's interface actually requires**, so that if such a seam is
ever built, it is built against the real contract rather than an imagined one.

ADR-043 §4's conclusion is adopted verbatim: **this repo's admission gate should
defer to whatever authority decision Core Memory/RVM make, not implement a
separate authority model of its own.**

## What this ADR does NOT claim

- No ownership of cross-ecosystem state movement.
- No responsibility table for components this repo does not own. The earlier
  table marked six of them "UNGROUNDED — do not build against"; ADR-043 went and
  grounded them, and its findings supersede that table entirely.
- No assertion that any RVM integration exists. None does.
