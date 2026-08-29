# 043. Cross-ecosystem responsibility boundaries — one job per component

- **Status**: Proposed. **Date**: 2026-08-29.
- **Purpose**: give every major ruvnet component exactly **one** primary
  responsibility, with LatentMesh owning **cognitive state movement**, so that
  no two components claim the same job.
- **Grounding discipline**: claims about components outside this repo are
  either **cited to source** or explicitly marked **UNGROUNDED — do not build
  against**. Nothing here is asserted from memory.

---

## The single-sentence charter

> **LatentMesh decides what machine knowledge should move, in what
> representation, across which link — and proves whether moving it improved the
> outcome.**

Everything below follows from keeping that sentence narrow.

## The boundary that matters most: LatentMesh ends where authority begins

**This is grounded, and it is stricter than "LatentMesh must not grant
authority."**

RVM is a **capability-secure, proof-gated** runtime. From
`ruvector/crates/rvm/crates/rvm-security/src/lib.rs` (implementing source, not
design intent): every hypercall passes a **three-stage gate** —

1. **capability check** — does the caller hold the required type and rights?
2. **proof-commitment verification** — is the transition attested?
3. **witness logging** — record the decision for audit

`PolicyDecision` is `Allow` or `Deny(RvmError)`, and `enforce()` returns
`RvmResult<()>`. From `rvm-cap/README.md`: capabilities are unforgeable tokens
with **monotonic attenuation** — a partition can only grant what it holds, with
delegation bounded at **8 levels** and a non-transitive `GRANT_ONCE` policy.

**The consequence for LatentMesh is concrete.** Latent state cannot be "trusted
a bit more" as confidence rises, because RVM does not accept confidence — it
accepts **a capability token plus a proof commitment**. So:

> **Latent state is evidence, never authority.** It may inform *what* a
> partition decides to attempt; it can never be the thing that permits the
> attempt. There is no gradient between the two.

`crates/latentmesh-reasoning` enforces this structurally today: **zero
dependency on `latentmesh-gate` or any authority type**, so no
capability-returning method could be added without that dependency appearing in
review.

⚠️ The broader "proof-gated microhypervisor" description (`concepts/rvm/CARD`)
is flagged **by the knowledge base itself** as *design intent with no
implementing source retrieved*. The three-stage gate and capability derivation
**are** backed by source; the P3 zero-knowledge tier is **not**. Do not design
against P3.

## One job each

| component | the one job | boundary |
|---|---|---|
| **LatentMesh** | **what state moves, in what representation, over which link** | never decides *whether an action is permitted* |
| **RVM** | **governed execution** — capability + proof + witness | never decides *what should be communicated* |
| **RuVector** | **what is known** — durable retrievable memory | never decides *what should move* |
| **MidStream** | **incremental delivery** of partial state | never decides *what is worth sending* |
| **MetaHarness** | **optimisation** of topology and policy | never *executes* the action it selects |
| **RuFlo** | **which agents exist and who works** | never decides *what cognitive state they share* |
| **RVF** | **portable signed evidence** | never *grants* what it attests |
| **Air** | **physical link adaptation** under bandwidth/energy limits | never *chooses* the payload's semantic content |

**The seam is deliberate.** RuVector answers *what is known*; LatentMesh answers
*what should move*. RuFlo answers *who works*; LatentMesh answers *what they
share*. MetaHarness answers *what to try*; RVM answers *what is allowed*.

## Grounded in this repo

ADRs **015–018** already map four of these seams, with crates implementing
each: `015` MidStream streaming, `016` RuVector persistent latent memory,
`017` Radio federated world models, `018` MetaHarness/Darwin topology loop.
This ADR does not invent the mapping; it **names the responsibility so two
components stop claiming one job.**

## What LatentMesh now owns that nothing else did

`crates/latentmesh-reasoning/src/routing.rs` — `RepresentationRouter` over
`None | Text | SymbolicDelta | SparseFeature | HiddenPrefix | MultiLayerLatent |
KvState | RecurrentCheckpoint`, scored by

```text
Score(m) = ΔV(m) / (λ_b·B + λ_l·L + λ_e·E + λ_r·R)     [normalised, additive]
```

Two properties earned by measurement rather than design taste:

- **`None` can win.** "Do not communicate" is frequently correct.
- **`Unmeasured` ≠ `Measured(0.0)`.** An unmeasured mode can be *flagged as
  worth testing* but never *selected*. Conflating the two is the error that
  cost this project a mission.

**This is the piece the evidence demanded.** Run 2 showed text moved decisions
**+0.512** where single-layer latent moved **~0** (p = 0.72, fully powered).
A system that assumes "latent is efficient, therefore send latent" is wrong on
our own data. The router exists so the runtime can *discover* which
representation earns its cost.

## UNGROUNDED — do not build against these until cited

I did not verify, this turn, the primary responsibilities of **RuFlo**,
**MetaHarness**, **RVF**, **MidStream**, **RuView**, or **Core Memory** against
their source. The table above states the *seam LatentMesh needs*, which is a
weaker claim than *what those components do*. **Before any cross-component
implementation, ground each with `search_ruvnet` and cite the path**, exactly as
the RVM row is grounded here. A responsibility table that is right about our
boundary and wrong about theirs is worse than no table.

## Acceptance test

One end-to-end workflow — RuView observation → RuVector memory → RuFlo agent
selection → LatentMesh routing → RVM governed action → RVF receipt →
MetaHarness fitness update — with **no duplicated responsibility**. Concretely,
it fails if any two components both decide the same thing, and it fails if
latent state reaches RVM as anything other than **evidence accompanied by a
capability token and proof commitment**.
