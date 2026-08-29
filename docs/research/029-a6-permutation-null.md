# Research: the A6 permutation null — is the cross-model affine fit better than chance?

* Purpose: execute ADR-024's **registered analysis** ("Registered analysis (protocol-safe, no
  probe draw)"): compute what an *uncorrelated* cross-model mapping scores on the same held-out
  residual metric that gate A6 uses, so the recorded A6 numbers can be read against chance instead
  of against a threshold nobody calibrated.
* Date: 2026-08-28.
* Scope: the two ADR-023-registered depth cells (**L18→L14**, the S2 winner; **L24→L19**, the
  Deviation-6 anchor) on **both** calibration datasets — the gold teacher-forced S2 dump and the
  sender-generated S2c dump. Four cells, 20 seeded permutations each.
* Method / evidence label: **deterministic CPU analysis over committed dumps**. No model runs, no
  GPU, no probe drawn, and **no recorded A6 number is changed** — this document and its receipt
  *annotate* the S2/S2c receipts, exactly as ADR-024 registered.
* Artifacts: `crates/latentmesh-runtime/receipts/run2-a6-permnull-receipt.json` (per-permutation
  residuals, seeds, statistics); code `harness/latentmesh-live/src/permnull.rs`, CLI
  `latentmesh-live permnull`.

---

## 1. The question, and why it was registered

Gate A6 asks: is the fitted alignment transform *usable*? Its statistic is the held-out relative
residual

```
‖apply(X_h) − Y_h‖_F / ‖Y_h − μ_r‖_F,   apply(x) = μ_r + α·(x − μ_s)·R
```

and its pass rule is `< 0.9`. Every recorded cell passed:

| Dataset | Cell | Recorded residual | A6 (< 0.9) |
|---|---|---|---|
| gold (S2) | L18→L14 | 0.5106 | PASS |
| gold (S2) | L24→L19 | 0.5600 | PASS |
| generated (S2c) | L18→L14 | 0.4451 | PASS |
| generated (S2c) | L24→L19 | 0.4682 | PASS |

ADR-024 registered the null because the 0.9 threshold was never derived from anything — and
because the PRH-critique literature (arXiv:2602.14486) shows representation-similarity metrics can
be inflated by depth/width artifacts. If a *shuffled* pairing scored 0.55 too, A6 would have been
measuring the shape of the two hidden-state clouds, not any item-specific correspondence between
them, and run 1's "good fit, no causal transfer" paradox would have had a cheap resolution: there
was never a good fit.

## 2. Method

**Same machinery, one thing changed.** Every permuted fit calls the same
`calibrate::fit_cell` → `AlignmentTransform::fit_affine` path, on the same 80/20 split produced by
the same recorded `FIT_SPLIT_SEED` (ChaCha8 `616619523`), and is scored by the same
`held_out_residual`. The only difference from the recorded fits is which receiver row each sender
row is paired with.

**Shuffle choice: item-level, within-split.** Both dumps carry exactly one row per GSM8K item —
`manifest.item_indices` is a bijection onto rows (4000 unique indices over 4000 rows for S2; 2560
over 2560 for S2c), each row being a mean-pool over that item's solution / generated token span.
A row shuffle is therefore *identically* an item shuffle: there is no within-item slot structure
left to break, so no block permutation is needed. The loader asserts this bijection and aborts if
a future dump ever violates it (at which point a block permutation would be required instead).

The permutation is applied **within the fit block and within the held-out block separately**. That
keeps row memberships, marginal distributions, and evaluation rows byte-identical to the real fit
— only the pairing changes. The alternative, a global permutation across the split, was rejected:
it lets a held-out receiver row also appear (differently paired) in the fit set, which can only
*help* the null and would make the test anti-conservative.

**Permutation family.** Uniform permutations, not derangements — the uniform permutation is the
textbook null. The expected fixed-point count is 1 per block (2 per permutation); observed mean
2.4 (gold) and 2.2 (generated), range 0–8 — recorded per permutation in the receipt rather than
engineered away. At n = 3200 a handful of accidentally-correct pairs cannot move the statistic.

**Seeds.** Permutation `k` uses `PERM_SEED_BASE + k` with `PERM_SEED_BASE = 0x00A6_2026_0828_0001`,
a constant fixed in the source before any null residual was computed. All 80 (seed, residual,
fixed-point) triples are in the receipt.

**Reproduction check.** The real (unpermuted) fit is re-run inside the same command. All four
residuals reproduce the committed receipts to every printed digit, and all four transform content
hashes match the committed ones (`eb3f42ed…`, `e892cfb7…`, `a5a3f549…`, `9ef787d9…`) — so the null
is computed against the same artifacts the ADRs recorded, not a lookalike refit.

## 3. Results

### 3.1 Gold pairs (S2 dump: 4000 items, 3200 fit / 800 held out)

| Cell | Real residual | Null mean | Null sd | Null min | Null max | z | nulls ≤ real | p (one-sided) |
|---|---|---|---|---|---|---|---|---|
| L18→L14 | **0.510551** | 1.022358 | 0.000688 | 1.020957 | 1.023504 | −744.3 | 0 / 20 | 0.048 |
| L24→L19 | **0.559983** | 1.024787 | 0.000756 | 1.023420 | 1.026349 | −614.9 | 0 / 20 | 0.048 |

### 3.2 Generated pairs (S2c dump: 2560 items, 2048 fit / 512 held out)

| Cell | Real residual | Null mean | Null sd | Null min | Null max | z | nulls ≤ real | p (one-sided) |
|---|---|---|---|---|---|---|---|---|
| L18→L14 | **0.445104** | 1.025822 | 0.000656 | 1.024773 | 1.027069 | −885.0 | 0 / 20 | 0.048 |
| L24→L19 | **0.468172** | 1.027404 | 0.001018 | 1.025201 | 1.028780 | −549.4 | 0 / 20 | 0.048 |

`z = (real − null_mean) / null_sd`; lower residual is better, so large negative = real beats
chance. `p = (#{null ≤ real} + 1)/(N + 1)`, whose floor at N = 20 is **1/21 = 0.048** — the p-value
is pinned at its minimum for this permutation count and carries no more information; the z and the
raw gap are what separate "decisive" from "marginal" here.

**Not one of the 80 permuted fits came within 0.46 of any real residual.** Every null is above
1.0. Zero of 80 would have passed A6 (< 0.9).

### 3.3 Two reference levels worth having in the same table

| Level | Residual | Centered receiver variance explained (1 − r²) |
|---|---|---|
| Perfect map | 0.000 | 100% |
| Generated L18→L14 (best recorded) | 0.445 | 80.2% |
| Generated L24→L19 | 0.468 | 78.1% |
| Gold L18→L14 (S2 winner) | 0.511 | 73.9% |
| Gold L24→L19 | 0.560 | 68.6% |
| **A6 pass threshold** | **0.900** | **19.0%** |
| **Predict μ_r for every input (do nothing)** | **1.000** | **0%** |
| **Permutation null (measured)** | **1.022 – 1.027** | **−4.5% to −5.6%** |

The "do nothing" level is exactly 1.0 by construction, not approximately: substituting
`apply(x) = μ_r` makes the numerator `Σ‖μ_r − y‖²` and the denominator `Σ‖y − μ_r‖²` over the same
held-out rows and the same fit-set μ_r.

## 4. Verdict

**Decisive, not marginal.** On all four registered cells, on both calibration datasets, the real
sender→receiver pairing beats the item-shuffled null by ~0.5 in a metric whose null sd is ~0.0007,
with 0 of 20 nulls at or below the real value in every cell. The recorded A6 residuals are not a
depth/width artifact and not a property of the two hidden-state clouds' shapes: they depend on
*which* receiver row each sender row is paired with.

Three secondary findings, all falling out of the same run:

1. **A shuffled fit is worse than doing nothing.** Every null landed at 1.02–1.03, i.e. 2–3% above
   the μ_r-predictor's exact 1.0. With the pairing destroyed, the fitted rotation and α buy
   nothing on the fit set and cost a little generalization on the held-out set. The null is
   therefore a *hard* chance level, not a soft one.
2. **The A6 threshold was badly calibrated — but harmlessly.** Chance on this statistic is 1.0
   (analytically) / 1.02 (measured). The registered pass line of 0.9 sits ~12% below chance and
   admits transforms explaining as little as 19% of the receiver's centered variance. A6 was a
   weak gate. It happened not to matter: the observed residuals cleared it by 0.34–0.45, which is
   ~90% of the distance from the gate to a perfect map. No recorded decision changes.
3. **The z-scores are about the null's tightness, not the map's quality.** z = −745 means the
   permutation null is extremely stable at n = 3200, not that the alignment is 745 sigmas good.
   The number to quote is 0.51 vs 1.02, or 74% of variance explained — never the z.

## 5. What this does *not* license — honest interpretation

The registered claim this analysis bears on is ADR-023/024's "the geometric alignment is good."
That claim **survives, in its narrow reconstruction sense, and only in that sense.**

* **What is now ruled out.** "A6 measured nothing / any two hidden-state clouds would score this"
  is dead. So is the cheap resolution of run 1's paradox in which there was never a real fit to
  begin with. The paradox is *strengthened*: a map that genuinely recovers 74–80% of the
  receiver's centered variance still moved the receiver's behavior not at all (ADR-023 S6; the M3
  MLP repeat in ADR-024 — sign-test p between 0.31 and 0.875 across six cell/calibration/
  architecture combinations).
* **What is not ruled out — the PRH critique is only partly answered.** ADR-024 cited the
  PRH-critique literature's depth/width-artifact concern. A permutation null holds the models,
  the depths, the widths, and the text distribution fixed and destroys only item identity. It
  therefore cannot distinguish "the map transports reasoning content" from "the map transports
  *any* item-identifying covariate the two models both encode" — solution length, numeric
  magnitude, topic, prompt-family position. Those are exactly the covariates a linear map is best
  at, and exactly the ones least likely to be causally usable by the receiver. A tighter null
  would shuffle **within length-matched (or topic-matched) strata**, so that only content-specific
  correspondence remains to be destroyed; a second would fit the same machinery across two
  *unrelated* corpora to see how much of the 74% is generic shared-text structure. Neither is run
  here, and neither is scheduled — they are named so the gap is on the record, not as commitments.
* **What it says about causal usefulness: nothing.** Reconstruction quality and causal transfer
  are different questions, and run 1/M3's whole result is that they came apart. This analysis
  raises confidence that the reconstruction number is real. It gives no evidence whatsoever that a
  better-than-chance reconstruction is a step toward a causally usable one, and it must not be
  cited as if it did.
* **Scope caveat inherited unchanged.** ADR-024's registered receiver-scale confound still applies
  to everything downstream: this receiver (Qwen2.5-1.5B) sits below the ~1.7B threshold
  arXiv:2608.05164 reports for cross-model steering transfer. The null analysis is a statement
  about the *fit*, which the receiver's size does not touch, but any inference from it toward the
  transfer question stays scoped to a sub-threshold receiver until the mandatory scale-control arm
  reports.

**One-line summary for the ADR ledger**: the recorded A6 residuals (0.4451–0.5600) beat an
item-shuffled permutation null (1.0224–1.0274 ± ~0.0008, 0/20 nulls ≤ real on every cell)
decisively; A6's 0.9 threshold is itself near chance and was a weak gate that the observed values
happened to clear by a wide margin; nothing here speaks to causal usefulness.

## 6. Reproducing

```bash
cargo build --release -p latentmesh-live
./target/release/latentmesh-live permnull --perms 20 --threads 8
# defaults: gold dump target/latentmesh-runs/s2, generated dump .../s2c,
# receipt crates/latentmesh-runtime/receipts/run2-a6-permnull-receipt.json
```

CPU-only, ~4.5 minutes wall clock at 8 threads (260.5 s recorded), deterministic: the seeds are
constants, the split is the recorded one, and re-running reproduces every residual bit-for-bit —
verified here by two independent full runs whose 80 null residuals and 4 real residuals agreed to
every printed digit despite different thread interleavings.
If the dumps are absent from `crates/latentmesh-runtime/target/`, regenerate them from the
committed streams first via the documented S2/S2c dump paths (~3 min each), then re-run the above.

## 7. Sources

- [ADR-024](../adr/024-run2-trained-thought-adapter-ladder.md) — the registration this document
  executes ("Registered analysis (protocol-safe, no probe draw)"), plus the receiver-scale
  confound quoted in §5.
- [ADR-023](../adr/023-live-four-condition-run1-pre-registration.md) — gate A6's definition and
  threshold, the two registered depth cells, and the recorded S2/S2c residuals.
- [docs/research/025-run1-negative-result.md](025-run1-negative-result.md) — run 1's "good fit, no
  causal transfer" result, which this analysis sharpens rather than resolves.
- [docs/research/028-sota-continuous-sweep-1.md](028-sota-continuous-sweep-1.md) §1 — the
  probing-classifier / amnesic-probing literature that names the fit-vs-causality split.
- `crates/latentmesh-runtime/receipts/{s2-calibration-receipt,s2c-calibration-receipt,
  run2-a6-permnull-receipt}.json` — every number above, re-read from the committed JSON for this
  write-up.
