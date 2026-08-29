# 040. PC3: pre-registering the decision-change endpoint, with a mandatory power calculation

- **Status**: Proposed (pre-registration — written before any PC3 item is drawn)
- **Date**: 2026-08-29
- **Gates**: [ADR-037](037-m5x-maximal-configuration-rung.md) (M5X) and
  [ADR-035](035-m4b-scale-control-pre-registration.md) (M4b) remain **BLOCKED**
- **Protocol**: [ADR-036](036-successor-rung-evaluation-protocol.md) e-process
- **Fixes**: coordinator error #14 in [ADR-039](039-pc2-steering-control-pre-registration.md)

## Context

[PC2](039-pc2-steering-control-pre-registration.md) registered a **rare-event**
primary — decoy-emission, ~2% — yielding **n_disc = 3** and a minimum
attainable p of **0.125**. It could not reach α = 0.05 on a perfect run, and it
*was* perfect (3 wins, 0 losses). The registered primary is **UNINFORMATIVE**.

A post-hoc analysis of PC2's own receipt then answered the question at full
power (recorded in [ADR-024](024-run2-trained-thought-adapter-ladder.md)):
asking *"did the injection change the answer **at all**"* instead of *"did it
emit the specific decoy"* gives **n_disc = 77** on the same 300 items.

**That analysis is post-hoc and may not stand as a registered result.** PC3
exists to pre-register it and confirm it **out of sample**.

## THE MANDATORY POWER CALCULATION (new standing requirement)

**Every future rung must state this block before its draw.** Error #14 —
registering an endpoint without computing whether it could ever detect
anything — is the failure `docs/research/047` already documented for **10 of 14**
prior draws. It will not recur through omission.

**For PC3's primary**, derived from PC2's measured rates rather than assumed:

| quantity | value | source |
|---|---|---|
| `steer` changes the answer | 46.7% | PC2 receipt, 140/300 |
| `random` changes the answer | 50.3% | PC2 receipt, 151/300 |
| **observed discordance rate** | **25.7%** | 77/300 paired |
| **expected n_disc at N = 300** | **≈ 77** | |
| **minimum attainable p** | **6.6 × 10⁻²⁴** | 2⁻⁷⁷ |
| N for n_disc ≥ 30 (floor) | ≈ 117 items | |
| **direction of the split** | **33 wins / 44 losses** | PC2 receipt, paired |

**The direction matters and this block originally omitted it** — caught by the
PC3 owner. On PC2's post-hoc data `steer` changed answers *less* often than the
norm-matched Gaussian, so the wealth process would have **declined**, not
merely failed to cross. **A powered FAIL is therefore the strongly-expected
outcome, not a coin flip.** Stated here so the expectation is on record before
the draw rather than claimed afterwards.

**Endpoint comparator — normative.** "The answer differs from baseline" is
evaluated with **`common::answers_equal` (numeric)**, never string inequality.
`"18.0"` and `"18"` are the *same answer*; counting that as a decision change
measures formatting noise. Three PC2 items turn on exactly this (280, 788,
1390), and the choice moves the split from 33W/44L to 33W/45L. The receipt must
state which comparator the endpoint used.

**Compare the endpoint this replaces**: decoy-emission gave expected n_disc
≈ 4.5 and a minimum attainable p of 0.043 — needing **~4,224 items** for
PC1b-level power. The new endpoint reaches decisive power at **N = 300**.

## Decision

**Primary endpoint**: whether the receiver's extracted answer **differs from its
own uninjected baseline answer** — `steer` vs `random`, paired per item, under
ADR-036's e-process (λ = 0.30, α = 0.05, threshold 20.0, N_max = 300).

**Item stream**: `adaptation-512`, fixed ascending order, ADR-024's 13-item
exclusion, **taking every remaining eligible item NOT used by PC2/M4i** — a
genuine **out-of-sample** confirmation, not a re-analysis.

**AMENDED before the draw — N_max is 212, not 300.** The pool cannot supply
300: `adaptation-512` holds 512 indices, PC2/M4i/PC1b consumed the **first
300** (hard-gated identical across all three), leaving **212**, starting at
index 4485. The original text was *arithmetically unattainable*. The rung runs
the **entire remaining out-of-sample pool, N_max = 212**, which at PC2's
measured 26% discordance still projects **expected n_disc ≈ 55** — clear of the
30-item floor above, so the run stays powered. Recorded in the receipt as a
first-class gate field `n_max_reduced_from_300_to_212_pool_exhausted` carrying
the arithmetic. ADR-036's stratified-resample tranche is **not** invoked: it is
"named, not built", and inventing it mid-rung would be protocol-shopping.

**AMENDED — `restore` is DERIVED here, not byte-replayed.** PC2's `restore` arm
replayed PC1b's committed payload file byte-for-byte and gated against it. **No
PC1b/PC2 artifact covers items 4485+**, so that cross-artifact validation is
impossible for PC3. The gold payload is derived through the identical `tap()`
path (same blocks, same last-row tap, same rendering), with only the
continuation text differing between arms. The one-changed-factor claim remains
provable *within* the run but is **no longer cross-validated against a prior
committed file** — a genuine weakening of the gate, recorded as such.

**Conditions**: unchanged from PC2 (`steer`, `restore`, `baseline`, `zerovec`,
`random` norm-matched). **Secondary endpoints retained**: decoy-emission (for
continuity), decoy-NLL and gold-NLL (the likelihood arm that gave p = 2.6e-44).

### REVERSAL — decoys stay in natural-slip space

ADR-039's closing note said a successor "must draw decoys **away** from
natural-slip space to recover a genuine chance floor." **I am reversing that,
and the reasoning is why the power calculation matters:**

1. The clean-floor argument applied to the **decoy-emission** endpoint, which is
   no longer the primary. It is moot here.
2. Moving decoys away from plausible answers would lower the `baseline` rate —
   **and lower `steer` with it**. A model that never spontaneously emits an
   unrelated integer will not emit one when nudged either. It would have made
   power **worse**, not better.
3. Natural-slip decoys are the **stronger** test of steering: they are answers
   the model would plausibly accept, so failing to steer toward them is more
   damning than failing to steer toward an implausible one.

Decoy construction is therefore **unchanged from PC2** (ChaCha8 seed `0x5732`
over `{g+1, g-1, g+10, 2g}`), committed before the draw.

## Pre-registered interpretation

**PASS** (`steer` changes answers more than `random`, wealth ≥ 20) — the
apparatus moves decisions **semantically**. PC2's post-hoc null was a
false negative; **M5X and M4b unblock.**

**FAIL with real power** (expected: n_disc ≈ 77) — **confirms out of sample
that decision-level change is non-semantic**: injection perturbs ~half of all
answers, and payload content contributes nothing a norm-matched Gaussian does
not. Combined with the likelihood arm's p = 2.6e-44, this establishes the
dissociation as a **registered, out-of-sample result**:

> **The channel is semantic at the likelihood level and merely perturbative at
> the decision level.**

Consequences accepted in advance: **M5X and M4b stay blocked permanently under
this apparatus** — both vary payload content, which is demonstrably not what
moves decisions. **The ladder closes**, and the dissociation becomes the
publishable finding under [ADR-032](032-negative-result-publication-contract.md),
reported without softening.

**Underpowered** — given the calculation above this should be impossible; if
n_disc lands below 30 the rung is reported as uninformative **and the power
model itself is recorded as wrong**, which is a finding about our estimation,
not about the apparatus.

## Firewall

PC3 is **same-model, same-item, identity-transform**. It tests **the apparatus,
never transfer.** Per the symmetric rule established after PC1b, **neither a
PASS nor a FAIL may be cited as evidence for or against cross-model transfer.**

## ⛔ Inherited quarantines

- Do **not** inherit PC1b's FAIL wording (*"nulls stand as evidence about
  TRANSFER"*) — logically inverted, quarantined at the head of ADR-024.
- Do **not** let a probe binary's auto-generated verdict string stand as the
  verdict. PC2's said *"the apparatus cannot move a decision by any means"*
  while its own data showed 3 wins / 0 losses and ~47% answer changes. **The
  coordinator adjudicates the verdict against the receipt.**

## Single-owner rule

**Exactly one implementing agent.** Claim before launching; never resume an
agent *and* start a workflow for the same rung (coordinator error #11).
