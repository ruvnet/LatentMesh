# 046. The representation metaharness — measured causal utility as a policy surface

- **Status**: Proposed. **Date**: 2026-08-29.
- **Implements the gap named in** [ADR-043 §3 row 1](043-cross-ecosystem-architecture-and-corrections.md):
  *"Causal-value-as-fitness is a **proposed integration this repo wants**, not an
  existing MetaHarness mechanism. Say so as a gap, not a fact."* This ADR closes
  that gap **on LatentMesh's side only**; it changes nothing upstream.
- **Related**: [018](018-metaharness-darwin-topology-loop.md) (the Darwin loop
  precedent), [003](003-causal-edge-verification.md) (the gate),
  [041](041-distributed-recurrent-latent-reasoning.md) (the router).

---

## 1. What upstream actually does — grounded, not remembered

Verified by reading source, not recalled:

**`@metaharness/harness` (ADR-047)**, `packages/harness/src/index.ts`:

> *"The model proposes. The harness decides. The algorithms verify."*
> Decision = argmax utility(action), and no action executes unless all four
> gates hold: **confidence ≥ threshold** ∧ risk ≤ budget ∧ cost ≤ budget ∧
> verification == pass.

Its modules are `score`, `router`, `pool` (**UCB1 contextual bandit**),
`verifier`, `safety`, `recovery`, `consensus`, `receipts` (hash-chained),
`kernel`.

**Darwin's fitness**, `packages/darwin-mode/bench/experiments/swe-fitness-selection.mjs`:

> `// Fitness: maximize resolve count, tie-break minimize cost.`

Darwin evolves **seven policy surfaces** — `planner`, `contextBuilder`,
`reviewer`, `retryPolicy`, `toolPolicy`, `memoryPolicy`, `scorePolicy` — with
the model frozen.

## 2. The two changes that make this unlike anything else

### 2.1 Confidence is a self-report; we gate on a measured effect

Upstream's first gate is `confidence ≥ threshold`. Confidence is **produced by
the same system whose output is being gated** — it is a self-report, and a
confidently wrong component passes.

We replace that gate for one specific decision — *which representation carries
state between agents* — with an **admission test the component cannot author**:
[`latentmesh-gate`](../../crates/latentmesh-gate/src/causal.rs)'s `verify_edge`
requires a candidate to beat the **worst of five controls** (zero, random
norm-matched, mismatched, self-generated, baseline) under
`sign_flip_permutation_test`. A representation is admitted on **evidence of
effect**, never on a claim of confidence.

### 2.2 `Unmeasured` is ineligible — the deliberate inverse of UCB1

Upstream's `pool` is a **UCB1 bandit**: unmeasured arms receive an *optimism
bonus* and are therefore preferentially explored. That is correct for
maximising long-run reward in a benign setting.

[`routing.rs`](../../crates/latentmesh-reasoning/src/routing.rs) does the
opposite on purpose. `GainMeasurement::Unmeasured` is a distinct variant from
`Measured(0.0)` and is **structurally excluded from selection** — it cannot be
scored, so it can never win by optimism. Exploration is an explicit operator
act, never an automatic consequence of ignorance.

**This is a design decision, not an algorithmic advance**, and must be
described that way (per [research/053 §4](../research/053-novelty-audit-and-the-remaining-experiment.md)).
It is defensible only because this is a safety-gated runtime where an
unmeasured channel silently carrying agent state is a worse failure than a
missed optimisation.

## 3. The eighth policy surface

Darwin evolves seven. This adds **`representationPolicy`**: which
representation carries state between two agents, over a lattice that includes
**`None`**.

| gene | values |
|---|---|
| `channel` | `text` · `semantic-delta` · `latent` · **`None`** |
| `admission` | `causal-gate` · `always` · `never` |
| `controls` | which of the five must be beaten |
| `alpha` | significance level for the permutation test |
| `costWeights` | λ_b, λ_l, λ_e, λ_r in the utility denominator |

**Fitness is `UtilityDensity = ΔV / (λ_b·B + λ_l·L + λ_e·E + λ_r·R)`**, where
ΔV is **measured and hypothesis-tested**, not predicted. This is the one real
distinction from BANDMAS (arXiv:2608.00458), whose contribution is
learned/replay-*predicted*. The ratio formalism itself is standard
value-of-information censoring and must not be claimed as novel.

## 4. Why this is credible rather than aspirational

**The harness has already run this decision on real data and reached a
falsifying answer about its own premise.** Fitness is computed from receipts
already committed to this repository, not from a simulation:

| receipt | what it establishes |
|---|---|
| `run3-stageA-receipt-statictext-gate-eprocess.json` | **text** crossed its e-process at item 43; beat `mismatched` 27W/1L, p=2.2e-7 |
| `run2-m4i-receipt-…-eprocess.json` | **latent, 1 layer**: accuracy 128, wealth 0.2578 — no crossing |
| `run2-m5x-receipt-2site-…-eprocess.json` | **latent, 2 layers**: accuracy 128, wealth 0.8837 — no crossing |

Scored by this fitness, the champion policy is **`channel: text`**, and the
latent channel is **`None`** — not because latent was untried, but because it
was tried, measured on a powered test (n_disc 64, min attainable p 5.4e-21),
and found decision-inert.

> **A harness that falsified its own headline mechanism and routed around it.**
> Every comparable system assumes its channel works. This one measured, found
> it didn't, and the routing table says so.

That is the property worth having, and it is the only honest claim available:
not "latent state transport works", but "the selector is trustworthy because it
told us the truth when the answer was inconvenient."

## 5. Scope — what this ADR does NOT claim

- It does **not** change MetaHarness upstream, add a surface to Darwin's seven,
  or assert that Darwin optimises causal value. ADR-043 explicitly forbids
  that reading, and it remains a gap on the upstream side.
- It does **not** revive the latent channel. The scope freeze in ADR-024's head
  is unchanged: falsified is direct activation injection into a **frozen**
  receiver, same-model, at one and two layers, for changing **decisions**.
  Learned integration, cross-model transfer, and the likelihood-level effect
  are untouched.
- `channel: latent` stays in the lattice as a **measurable option that
  currently scores zero**, so that if M5 ([ADR-045](045-m5-receiver-side-adaptation-pre-registration.md))
  or any future method makes it effective, the policy changes by
  **re-measurement, not by editing this document**.

## 6. Removability

Following the `harness/air` precedent and RuVector's ADR-256 rule: the harness
is **dev-only and fully removable**. No Rust crate reads its output; the
champion genome is transcribed by a human or ignored, and deleting
`harness/representation/` leaves the runtime byte-identical.
