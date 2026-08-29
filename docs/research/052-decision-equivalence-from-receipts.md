# 052. Decision-equivalence, verifiable from receipts alone

**Purpose**: let a fresh reader reach the conclusion **without taking any prose
on trust**. Every number below is a **field path into a committed receipt**.
Open the JSON, follow the path, read the value. Nothing here is derived,
summarised, or inferred.

**Date**: 2026-08-29. All receipts under `crates/latentmesh-runtime/receipts/`.

---

## Claim 1 — one-layer and two-layer injection are decision-equivalent

Both rungs used **the same 300-item stream, the same injection site, the same
reconstruction-trained adapters, and the same task**. The only difference is
**one injection site versus two**.

| field path | M4i (**1 layer**) | M5X (**2 layers**) |
|---|---|---|
| `summary/accuracy/aligned_real` | **128** | **128** |
| `summary/accuracy/baseline_uninjected` | 140 | 140 |
| `summary/accuracy/random` | 132 | 124 |
| `summary/nll_aligned_vs_random/wins` | **166** | **166** |
| `summary/nll_aligned_vs_random/losses` | **134** | **134** |
| `e_process/n_discordant` | 66 | 64 |
| `e_process/final_wealth` | 0.2578 | 0.8837 |

Receipts:
- M4i — `run2-m4i-receipt-cellL18toL14-mlp-pertokenlast-fuse-questiontail-slots8-eprocess.json`
- M5X — `run2-m5x-receipt-2site-L18toL14-L24toL19-pertokenlast-fusemany-questiontail-slots4x2-eprocess.json`

**Read the bold rows.** Accuracy under the aligned condition is **128 in both**.
The likelihood-arm split against a norm-matched random control is **166 wins /
134 losses in both** — the same integers, not a similar ratio.

**Neither e-process crossed.** The pass boundary is wealth ≥ 20
(`e_process/wealth_threshold`); the observed finals are 0.2578 and 0.8837.

## Claim 2 — this is a powered null, not a failure to measure

| field path | M5X value | meaning |
|---|---|---|
| `power_calculation/discordant_wins_needed_to_cross` | **46** | registered **before** the draw |
| `e_process/discordant_wins_aligned` | **34** | observed |
| `e_process/n_discordant` | 64 | pairs available |
| `power_calculation/max_attainable_wealth_if_all_discordant_are_wins` | 10140.5 | vs a threshold of 20 |

Minimum attainable one-sided p at 64 discordant pairs is 2⁻⁶⁴ ≈ **5.4e-20**.
The instrument could have detected an effect twenty orders of magnitude smaller
than the boundary required. **It did not fail to look; it looked and found
nothing.**

## Claim 3 — text remains causally effective on the same receiver

Receipt: `run3-stageA-receipt-statictext-gate-eprocess.json`

| field path | value |
|---|---|
| `summary/gated_text_correct` | **39** / 43 |
| `summary/per_control[zero].correct` | 17 |
| `summary/per_control[self_generated].correct` | 17 |
| `summary/per_control[random].correct` | 15 |
| `summary/per_control[mismatched].correct` | **13** |
| `summary/e_process_crossed` | **true** |
| `summary/e_process_crossed_at_item` | **43** |

The text e-process **crossed**, at item 43 of a 300-item budget. The latent
e-processes did not cross at all.

### ⚠️ CORRECTION — the control *ordering* is NOT a result (coordinator error #21)

I previously wrote that `mismatched` (13) < `random` (15) < `zero` (17) shows a
wrong-but-real message is worse than random tokens. **That ordering has no
statistical support**, and the receipt never tested it: every registered
comparison is `gated_text` vs *one* control, never control vs control.

**The paired tests, computed from the receipt's per-item flags:**

| comparison | split | two-sided p |
|---|---|---|
| `zero` vs `random` | 5W/3L | **0.73** |
| `random` vs `mismatched` | 8W/6L | **0.79** |
| `zero` vs `mismatched` | 9W/5L | **0.42** |
| `zero` vs `self_generated` | 0W/0L | identical on every item |

**The four controls are statistically indistinguishable from each other at
n = 43.** The rank ordering is noise.

### What IS supported, and it is the causal claim

| comparison | split | two-sided p |
|---|---|---|
| `gated_text` vs `zero` | 23W/1L | 3.0e-6 |
| `gated_text` vs `random` | 25W/1L | 8.1e-7 |
| `gated_text` vs `self_generated` | 23W/1L | 3.0e-6 |
| **`gated_text` vs `mismatched`** | **27W/1L** | **2.2e-7** |

**Text beats every control decisively, and beats `mismatched` hardest.** That is
the load-bearing result: `mismatched` is *another episode's genuine message*,
carrying its own number and its own fluent reasoning. Beating it at 27W/1L means
the effect is **not** "any message helps" — the receiver is responding to
*this* message's content.

**The causal claim survives intact.** What does not survive is the decorative
ordering I mistook for part of it.

## What follows, and only this

Comparing Claim 1 against Claim 3 on the same receiver:

- **Injected activations**: no decision effect at one layer, no decision effect
  at two, on a powered test.
- **Text**: crossed its boundary at item 43, with a control ordering showing
  content is being read.

**A reader who opens the four receipts and follows the paths above reaches this
without needing any sentence in this repository to be true.**

---

## ⛔ Scope of the negative result — read before citing it

**FALSIFIED**, precisely and only:

> **Direct activation injection** — a captured hidden state written into a
> **frozen** receiver's residual stream — under a **same-model** apparatus, at
> **one and two layers**, as a mechanism for changing downstream **decisions**.

**NOT falsified, and must not be cited as such:**

- **Latent communication in general.** This tested one mechanism.
- **Learned-integration methods** — task-loss-trained fusers, gated multi-layer
  receivers, receiver-side adaptation. **None was tested here.** Cache-to-Cache's
  reported 6.4–14.2pp uses a trained gated Fuser; what M5X rules out is that
  *distributing the same untrained representation across more sites* reproduces
  it.
- **Cross-model transfer as such.** Every positive control was same-model.
- **The likelihood-level effect**, which is **real and replicated** — the payload
  reliably moves the receiver's distribution toward the specific answer it
  encodes. What it does not do is change the answer.

**If a future learned-integration method succeeds, nothing above becomes wrong.**
The scope was frozen deliberately so that outcome refines the map rather than
contradicting it.
