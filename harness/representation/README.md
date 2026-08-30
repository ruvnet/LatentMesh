# LatentMesh Representation MetaHarness

This harness evolves `representationPolicy` — [ADR-046](../../docs/adr/046-representation-metaharness.md)'s
eighth policy surface — while freezing the model, the receipts, the promotion
rule, and the pre-registered significance level.

Its fitness is **not simulated**. `harness/air` labels its results `simulated`
because it runs a channel simulator, and that label is correct there. Here it
would be a lie: every gain, cost and admission decision is read out of receipts
already committed to `crates/latentmesh-runtime/receipts/`, so the evidence
label is `measured-committed-receipts`, and `lib/verify.mjs` refuses a receipt
that claims anything stronger — or anything weaker.

```bash
npm install --ignore-scripts
npm run validate
```

`npm run benchmark` writes `artifacts/representation-benchmark.json`.
`npm run optimize` writes `artifacts/champion-policy.json` with a hash-chained
generation trace.

## Evolvable policy (ADR-046 §3)

| gene | values |
|---|---|
| `channel` | `text` · `semantic-delta` · `latent` · `None` |
| `admission` | `causal-gate` · `always` · `never` |
| `controls` | which measured controls the channel must beat |
| `alpha` | `0.05` · `0.01` · `0.001` |
| `costWeights` | λ_b, λ_l, λ_e, λ_r in the utility denominator |

`Score(m) = ΔV(m) / Σ_r w_r · cost_r(m) / scale_r` — a normalised **additive**
denominator, mirroring `crates/latentmesh-reasoning/src/routing.rs`. Not a
product of raw physical units: see that file's `CostScale` doc for why.

## Frozen promotion rule

A candidate policy replaces the incumbent only if it (1) scores strictly higher
on the frozen yardstick, (2) requires **every** control its channel was
actually measured against, (3) does not loosen `alpha` past the pre-registered
`0.05`, and (4) routes to a mode that was admitted, or to `None`.

Clause 2 exists because "worst of the required controls" is monotone: dropping
the hardest control raises ΔV for free, so an unconstrained search is rewarded
for weakening its own gate. Strictness is a safety property, not a fitness
dimension. The counterfactual with the clause removed is computed anyway and
reported as `unconstrained_note` in the champion receipt — the constraint's
effect is visible, not asserted.

Clause 3 matters in the same direction: re-thresholding committed wealth at a
**stricter** alpha re-runs nothing and is conservative; a looser one would
retroactively weaken a pre-registered bar, so the lattice contains none.

## What the evidence actually says

Run on the three committed receipts, the champion is:

```
channel: text · admission: causal-gate · alpha: 0.05
controls: [zero, random, mismatched, self_generated]
fitness: 1.4117253864364296   selected: text
```

| channel | source | admitted ΔV | outcome |
|---|---|---|---|
| `text` | `run3-stageA-…-eprocess.json` | **+0.5116** vs `zero` | admitted; min wealth 21.158 ≥ 1/α = 20 |
| `latent` (1 site) | `run2-m4i-…-eprocess.json` | 0.0 | refused; wealth 0.2578, 66 discordant |
| `latent` (2 site) | `run2-m5x-…-eprocess.json` | 0.0 | refused; wealth 0.8837, 64 discordant |
| `semantic-delta` | *none* | — | **Unmeasured** — ineligible |

The admitted text ΔV is not a number this harness invented: it re-derives
`0.5116279069767442`, byte-identical to that receipt's own
`summary.uplift_vs_best_control.raw_delta`, and independently identifies `zero`
as the control most favourable to the null — which is what that receipt's
`summary.realised_best_control_most_favorable_to_null` says.

The latent channel scores zero **because it was measured**, not because it was
skipped. Both rungs ran a fully powered e-process on 300 items and neither
crossed. Under `admission: always` the 2-site rung's small positive raw delta
against its random control *does* get routed to — and still loses to text by
more than an order of magnitude. That path is left in the lattice deliberately:
if [ADR-045](../../docs/adr/045-m5-receiver-side-adaptation-pre-registration.md)
or any later method makes latent effective, the policy changes by
**re-measurement, not by editing a document**.

## The one behaviour to get right

`GainMeasurement` is three-state, and `Unmeasured` is **structurally**
ineligible, not ineligible by a policy check someone can forget:

- `UNMEASURED` is a frozen object with **no `deltaV` property at all**;
- `utilityDensity` refuses anything that is not a finite number;
- the router only reaches the formula inside the `Measured` branch.

So an unmeasured candidate cannot be scored, and a thing that cannot be scored
cannot win. This is the deliberate inverse of UCB1, whose optimism bonus makes
unmeasured arms *preferentially* explored. Exploration here is an operator act,
never an automatic consequence of ignorance.

`Measured(0.0)` and `Unmeasured` never collapse: one has a field the other does
not, they report as distinct `ScoreOutcome`s, and both appear in the trace. The
corpus exercises both on real data — `latent` is `Measured(0)` under a policy
requiring only the control it ran, and `Unmeasured` under one requiring
controls it never ran.

## The asymmetry, stated rather than hidden

An unmeasured **gain** is ineligible (conservative). An unmeasured **cost** is
dropped from the denominator and the remaining weights renormalise
(anti-conservative — it is charitable to the candidate). Energy was never
measured by any rung, so this applies to every candidate.

That is only safe because it applies *uniformly*: `assertCostParity` throws if
candidates differ in which resources they leave unmeasured, so an unmeasured
cost can never shrink one candidate's denominator relative to another's.

Two further disclosures the receipts carry:

- **Latency is charged asymmetrically, against text.** The text rung records
  per-condition GPU seconds, so its own two passes are charged exactly
  (4.288 s/item). The latent rungs record only wall clock, amortised over
  items × conditions (≈2.831 s/item), which *understates* their share if
  conditions differ in cost. The cheaper-looking channel is the one the harness
  goes on to reject.
- **The `random` controls are not the same control.** Text's is random *tokens*;
  latent's is a norm-matched random *vector*. Each is only ever used against its
  own channel; they are never compared to each other.

## Discrepancies between ADR prose and the committed receipts

Found while grounding every field path. The receipts win; the numbers used are
the receipts'.

1. **"the worst of five controls" (ADR-046 §2.1).** The text rung has **four**
   (`zero`, `random`, `mismatched`, `self_generated`). Its own
   `config.controls.text_equivalent_dropped` records that the fifth is
   degenerate when the channel under test is itself text. Four are used; a
   fifth is not invented.
2. **"p = 2.2e-7" for `mismatched` (ADR-046 §4).** The receipt's
   `summary.per_control[mismatched].exact_sign_one_sided` is `1.0803e-7`
   (mid-p `5.588e-8`). 2.2e-7 is that one-sided value doubled — a two-sided
   figure quoted where a one-sided one is recorded. 27W/1L is correct.
3. **"min attainable p 5.4e-21" for the 2-site rung (ADR-046 §4).** No receipt
   field holds that value. The committed
   `power_calculation.min_attainable_one_sided_sign_test_p_at_this_n_disc` is
   `1.3553e-20`, stated pre-draw at the anchor's `expected_n_discordant` of 66.
   The realised `n_discordant` was 64, whose one-sided floor is 2⁻⁶⁴ ≈ 5.4e-**20**.
   The conclusion — the endpoint was powered — is unaffected either way.
4. **`items[*].conditions[*].correct` does not exist in the text rung.** There,
   `items[].conditions` is an object of plain booleans keyed by condition name
   (`{gated_text: true, zero: false, …}`). The `.correct` shape is the *latent*
   rungs' schema. Both are handled, per file.

## Removability

Dev-only and fully removable, following the `harness/air` precedent and
RuVector's ADR-256 rule (ADR-046 §6):

- **No Rust crate reads this directory.** Nothing under `crates/` references
  `harness/representation`, its artifacts, or its schemas.
- **It is not in the Cargo workspace.** It is a private npm package with no
  dependencies and no build step; `cargo build`, `cargo test` and every crate's
  output are unaffected by its presence or absence.
- **The champion genome is transcribed by a human, or ignored.** It is a
  recommendation in a JSON file, never a runtime input. `artifacts/` is
  gitignored.
- **Deleting `harness/representation/` leaves the runtime byte-identical.**
  The only other change would be removing its CI job.
