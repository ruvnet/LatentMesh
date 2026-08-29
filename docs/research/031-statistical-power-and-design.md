# 031. Statistical power and design for the probe ladder (run 1/run 2/run 3)

> **STALENESS POINTER (2026-08-29) — read before using this document's tables.**
> This document's canonical draw-history table covers the ladder through the
> 10 draws that existed when it was written. It has **not** been updated with
> M4c, M4d, M4g or M4h Stage 1. Their discordant-pair counts — the numbers
> that matter most for the power argument — are M4d **7** (the only
> non-power-limited draw in the ladder), M4g **3** (min attainable p 0.125),
> M4h Stage 1 **2** (min attainable p 0.25). See ADR-024's "M4h Stage 1
> OUTCOME" section and ADR-036 for the current picture: discordance *falls*
> as adapters improve, which is why the frozen 40-item protocol was retired
> for successor rungs. The methodology in this document (mid-p McNemar, the
> e-process design, the multiplicity framing, the NLL rejection) all stands.



**Purpose**: answer four questions the run-3 (ADR-030) authoring window asked, using the actual
receipted numbers from runs 1-2 rather than textbook defaults. Every number below is either read
directly from a committed receipt (`crates/latentmesh-runtime/receipts/*.json`) or computed from
those numbers in this session (Python, shown inline where it matters). Nothing is estimated from
memory of what a paired-binary test "usually" needs.

## 0. What ADR-030 must change — summary for the impatient reader

1. **The "power note" already in ADR-030 is correct but incomplete.** It correctly says a 40-item
   sign test "detects only large effects," but it doesn't say *how* large, and the honest answer is
   worse than it implies: **at the discordant-pair counts this exact probe has produced every time
   it has run (3, 3, 3, 4, 4, 4, 4, 5, 5), the test is not merely underpowered — at n_disc ∈ {3,4}
   it is mathematically incapable of reaching p<0.05 no matter what the data show.** 6 of the 9
   cross-model null draws to date (S2b×4, M3×2) landed in that dead zone. This is a fact about the
   test's exact null distribution (§1), not a modeling assumption.
2. **Add McNemar's mid-p correction to the primary test at zero cost.** It roughly halves every
   p-value in the ladder using the identical data already collected (§2) — it would not have
   flipped any past verdict, but there is no reason to leave power on the table for run 3.
3. **Pre-register the eval-200 confirmatory scale as an anytime-valid (e-value) sequential test,
   not a fixed-N second look.** ADR-030 already commits to running 40-item and 200-item scales
   "together, not sequentially" specifically to avoid an implicit second peek — an e-process gets
   the *better* property (continuous monitoring, stop early on a real win, extend the budget on
   ambiguity) **without** that discipline problem. Concrete, paste-able registration text is in §3.
4. **State the multiplicity family explicitly, now, before any run-3 result exists.** The ladder
   has run 10 valid probe draws to date (§4); treating each rung's report as if it were the only
   test ever run will not survive review once a rung eventually passes. The fix is cheap and mostly
   already true: the ladder's *between-rung* order (M3→M4→M5) is a valid fixed-sequence procedure
   needing no correction, but *within*-rung parallel variants (M3's 2 variants, each S2b calibration
   distribution's 2 cells) need Holm-Bonferroni, and this ADR should say so before it becomes
   convenient to have said it after a pass.
5. **One more thing ADR-030 itself introduces**: "primary gate ... greater than the best
   (most favorable to the null) of the four controls" is a selection over 4 controls that needs one
   sentence pinning down *when* "best" is chosen (§5) — this is not one of the four requested
   questions but it is directly load-bearing to the ADR freezing this week.

---

## 1. The actual data: every valid probe draw to date

The frozen 40-item S1a/S2b protocol (ADR-023, inherited unchanged by ADR-024) has been run 11
times; one (S1a run 1) is excluded from any statistical accounting because it was invalidated by a
since-fixed BF16 RoPE bug and a scoring bug (ADR-023 Deviation 3) — it never produced a valid draw
against the null. The 10 valid draws, read from their receipts' `summary.primary_*` blocks:

| # | Draw | wins | losses | n_disc | exact p | mid-p | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| 1 | S1a run 2 (self-pair, identity transform) | 5 | 0 | 5 | 0.0312 | **0.0156** | PASS |
| 2 | S2b gold, L18→L14 (winner cell) | 2 | 1 | 3 | 0.5000 | 0.3125 | FAIL |
| 3 | S2b gold, L24→L19 (anchor cell) | 1 | 2 | 3 | 0.8750 | 0.6875 | FAIL |
| 4 | S2b generated, L18→L14 | 3 | 1 | 4 | 0.3125 | 0.1875 | FAIL |
| 5 | S2b generated, L24→L19 | 1 | 2 | 3 | 0.8750 | 0.6875 | FAIL |
| 6 | M3 MLP, per-token variant | 2 | 2 | 4 | 0.6875 | 0.5000 | FAIL |
| 7 | M3 MLP, pooled variant | 2 | 1 | 3 | 0.5000 | 0.3125 | FAIL |
| 8 | M4 FastGRNN r=64 | 3 | 1 | 4 | 0.3125 | 0.1875 | FAIL |
| 9 | M4 FastGRNN r=128 | 3 | 1 | 4 | 0.3125 | 0.1875 | FAIL |
| 10 | M4 FastGRNN r=256 | 4 | 1 | 5 | 0.1875 | **0.1094** | FAIL |

(Mid-p computed as `exact_p − 0.5·P(X=wins | Binomial(n_disc, 0.5))`, the standard correction —
Fagerland, Lydersen & Laake 2013, *BMC Medical Research Methodology* 13:91, "The McNemar test for
binary matched-pairs data: mid-p and asymptotic are better than exact conditional.")

Two structural facts fall out immediately:

- **Discordant-pair count never exceeded 5, out of 40 items, across 10 draws over 3 architecture
  families (identity self-pair, linear Procrustes, trained MLP, trained FastGRNN) and 9 different
  injected contents.** The observed discordance *rate* is 7.5%-12.5% (mean ≈9.5%) and is
  remarkably stable regardless of what was actually injected — including literal random noise. This
  is itself informative for run 3 design (§3): whatever this 8-slot mid-layer perturbation
  mechanism does to a GSM8K item's answer, it flips roughly 1 in 10 items regardless of content,
  and no rung so far has moved that rate.
- **9/10 draws are FAIL; the one PASS (S1a) is the mechanistically weakest test in the family** — a
  same-model self-pair sanity check (does the injection *pathway* transmit information at all), not
  a cross-model transfer claim. Every cross-model transfer test (S2b, M3, M4 — 9 draws) has failed.
  This matters directly for §4's multiplicity framing.

---

## 2. Power analysis of the actual design (RQ1) and test choice (RQ2)

### 2.1 The exact null distribution of the sign test, by discordant-pair count

The one-sided exact sign test's null distribution is `Binomial(n_disc, 0.5)`. Its **minimum
attainable p-value** at each `n_disc` (all discordant pairs win) determines whether the test can
reject at α=0.05 *at all*:

| n_disc | min attainable p (all wins) | can ever reach p<0.05? |
|---:|---:|---|
| 2 | 0.250 | **no** |
| 3 | 0.125 | **no** |
| 4 | 0.0625 | **no** |
| 5 | 0.0312 | yes — only if 5/5 |
| 6 | 0.0156 | yes — only if 6/6 |
| 8 | 0.0039 | yes — 7/8 or 8/8 |
| 10 | 0.00098 | yes — 9/10 or 10/10 |

**6 of the 9 cross-model null draws in §1 (n_disc=3 or 4) could not have passed regardless of the
true effect size**, because their discordant-pair count sat below the floor where the exact test
has any power at all. This is not a modeling claim; it is arithmetic on the observed `n_disc`. Any
future rung that lands at n_disc≤4 on this 40-item probe is in the same position — a "clean fail"
report from such a rung tells you almost nothing about whether the underlying architecture carries
signal, because the test literally could not have said otherwise.

### 2.2 Power as a function of discordant-pair count and true win-rate θ

For a true win-probability θ (P(real wins | discordant pair)) and n_disc discordant pairs, exact
power (computed in this session, `scipy`-free binomial enumeration):

| n_disc | θ=0.60 | θ=0.70 | θ=0.75 | θ=0.80 | θ=0.90 | θ=1.00 |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.078 | 0.168 | 0.237 | 0.328 | 0.590 | 1.000 |
| 10 | 0.046 | 0.149 | 0.244 | 0.376 | 0.736 | 1.000 |
| 15 | 0.091 | 0.297 | 0.461 | 0.648 | 0.944 | 1.000 |
| 20 | 0.126 | 0.416 | 0.617 | 0.804 | 0.989 | 1.000 |
| 30 | 0.291 | 0.730 | 0.894 | 0.974 | 1.000 | 1.000 |

Reading it: even a genuinely strong effect (θ=0.75, discordant pairs favor the real condition 3:1)
needs roughly **25-30 discordant pairs** for 80% power — five to six times what any draw in the
ladder has produced.

### 2.3 Translating to total N, using the ladder's own observed discordance rate

Given the empirically stable discordance rate of ~10% (range 7.5%-12.5%, §1) — not a textbook
independence assumption, the ladder's own repeated measurement — total-N requirements for various
target accuracy gaps δ (all Python-computed, `power_design(N, p_disc, δ)`, θ derived as
`0.5 + δ/(2·p_disc)`; a δ exceeding p_disc is flagged impossible rather than silently computed,
because a discordance-driven accuracy gap cannot exceed the discordance rate itself):

| N | p_disc=10% (observed) | p_disc=20% (2× observed, "if a real signal roughly doubles the flip rate") |
|---:|---|---|
| 40 (current probe) | δ=5pp: power 0.00 | δ=5pp: 0.14 · δ=10pp: 0.37 · δ=20pp: 1.00† |
| 200 (ADR-030's confirmatory scale) | δ=5pp: 0.62 | δ=5pp: 0.44 · δ=10pp: 0.95 |
| 300 | δ=5pp: 0.89 | δ=5pp: 0.61 · δ=10pp: 0.99 |
| 400 | δ=5pp: 0.95 | δ=5pp: 0.72 · δ=10pp: 1.00 |

(† this cell is a boundary artifact — at p_disc=10%, δ=20pp forces θ=1 exactly, i.e. every
discordant pair must favor real; it is mathematically valid but not a realistic operating point,
flagged rather than hidden.)

**Headline for ADR-030's existing 200-item confirmatory arm**: at the discordance rate this exact
mechanism has shown 9 times, N=200 gives ~60% power for a 5-percentage-point true effect — a
reasonable, not overwhelming, upgrade over the 40-item arm's zero power at the same effect size.
It is adequately powered for anything ≥10pp under the p_disc=20% scenario, but under the *observed*
p_disc=10% scenario a 10pp true gap is itself at the boundary artifact described above — meaning
the honest statement for ADR-030 is **"the 200-item arm is well-powered to confirm a rung that
already showed a large, unambiguous 40-item signal (θ near 1, like S1a's), and only moderately
powered (≈60%) to rescue a rung that showed a real-but-modest 40-item signal."** That is a
materially different claim than "both scales are pre-registered together" currently implies without
a number attached — recommend adding this table's headline row.

### 2.4 Is the sign test the right choice? (RQ2)

- **Mid-p McNemar**: strictly more powerful at identical data and identical Type-I error control in
  practice (Fagerland et al. 2013 show mid-p tracks nominal α better than the conservative exact
  conditional test and is "almost as powerful as" the more complex exact unconditional test).
  §1's table already shows the effect on our own numbers — e.g. draw #10 (M4 r=256) moves from
  p=0.1875 to p=0.1094, still a clean fail, but closer to the boundary in a way a future,
  slightly-stronger rung could cross where the classic exact test would not. **Recommend switching
  the primary statistic to mid-p McNemar for run 3**, disclosing the exact p alongside it (both are
  one line of arithmetic from data already collected — no new receipts, no new items).
- **Conditional logistic regression**: for a 1:1 matched pair with a single binary predictor (the
  condition indicator) and no covariates, conditional logistic regression is **mathematically
  equivalent to McNemar's test** — it will not add power here. It would matter if run 3 wanted to
  adjust for a per-item covariate (e.g. problem length, or a difficulty proxy) while still using the
  matched-pair structure — worth naming as a *future* extension, not a run-3 upgrade, since no such
  covariate is currently captured in the receipts.
- **Permutation test on the discordant subset**: for a two-arm paired binary outcome the permutation
  test *is* the exact sign test (permuting which of each discordant pair is labeled "real" is
  exactly the sign-test's reference distribution) — no separate gain available there.
- **Continuous outcome (per-item NLL) instead of binary correctness — checked against our own
  receipts, and the answer is not what the naive theory predicts.** Every receipt already carries
  `items[i].conditions.<cond>.nll_gold` (verified this session, e.g.
  `s2b-receipt-cellL18toL14-slots8-poolfull-rescaletrue-n40.json items[0]`). Running a one-sided
  paired Wilcoxon signed-rank test on `nll_real − nll_random` (n=40, no discarded items) against the
  **same four draws** as §1:

  | Draw | exact sign p (accuracy) | Wilcoxon p (raw NLL, n=40) |
  |---|---:|---:|
  | S1a run 2 (the one PASS) | **0.0312** | 0.6834 |
  | S2b gold L18→L14 | 0.5000 | 0.6341 |
  | M3 per-token | 0.6875 | 0.6239 |
  | M4 r=256 | 0.1875 | 0.4705 |

  **The continuous outcome is not more sensitive here — it is blind to the one real signal the
  ladder has produced.** ADR-023 already flagged this for S1a specifically ("the NLL secondary
  diagnostic showed no effect in run 2, 19W/21L, p=0.68"); this session confirms it holds across
  every other draw checked. The likely mechanism: `nll_gold` measures confidence in the single
  gold *final-answer token* under teacher forcing, which is a different construct from "did the
  actual greedy generation reach the correct final answer" — a generation can flip to a wrong
  answer via an arithmetic slip mid-chain that leaves the eventual gold-token NLL largely unmoved,
  or vice versa. **Recommend against pre-registering NLL as a co-primary or replacement statistic
  for run 3 without first validating it against a known-real signal** (S1a is the only candidate)
  — a continuous outcome only buys power if it is actually a finer-grained measurement of the same
  effect, and the receipted evidence says this particular one is not. If a continuous outcome is
  wanted, the better candidate is probably log-probability of the correct *numeric answer*
  marginalized over the sampled generation (closer to what "correct" actually measures), which
  would need new instrumentation, not a re-read of existing receipts — named as follow-up, not
  ready to freeze into ADR-030 this week.

---

## 3. Sequential / anytime-valid design (RQ3)

### 3.1 Why e-values over group-sequential (O'Brien-Fleming/Pocock)

Classical group-sequential designs (Pocock 1977; O'Brien & Fleming 1979; Lan & DeMets 1983
alpha-spending) control Type-I error across a **pre-specified number and timing of interim looks**.
They would work here, but they require freezing exactly how many looks and at what N before any
data is seen — a real constraint for a lab that wants to keep drawing items from `adaptation-512`
until a rung's picture is clear, or stop early on an obvious win.

**Safe, anytime-valid inference (SAVI)** — Ramdas, Grünwald, Vovk & Shafer, "Game-Theoretic
Statistics and Safe Anytime-Valid Inference," *Statistical Science* 38(4):576-601, 2023
(arXiv:2210.01948); building on Shafer, "Testing by Betting," *JRSS-A* 2021 — gives up the
requirement to pre-specify looks entirely. The test is a nonnegative martingale ("wealth process")
`W_t` starting at 1; at each new observation you update `W_t` by a pre-registered betting rule, and
you may stop and reject H0 the instant `W_t ≥ 1/α`, **at any time, for any reason, including
"we're out of budget" or "we just want to check"**, without inflating Type-I error. This is exactly
the "keep sampling until significance without alpha inflation" property the mission brief asked
about, and it is a real, established guarantee (not a heuristic), via Ville's inequality on
nonnegative martingales.

For a bounded/Bernoulli-type sequential proportion test — precisely our discordant-pair sign test,
generalized to accumulate over an unbounded item stream — Waudby-Smith & Ramdas, "Estimating Means
of Bounded Random Variables by Betting," *JRSS-B* 85(1), 2023 (arXiv:2010.09686), give the concrete
betting construction: a predictable betting fraction `λ_t` and multiplicative wealth update
`W_t = W_{t-1} · (1 + λ_t(X_t − 0.5))` for each new discordant-pair outcome `X_t ∈ {0,1}`.

### 3.2 Concrete, paste-able registration text for ADR-030

> **Primary test — e-process sign test.** In place of (or alongside, during a transition period) the
> fixed-N exact sign test, the primary `CausalDynamicText`-vs-best-control comparison is evaluated
> as a one-sided Bernoulli e-process (Waudby-Smith & Ramdas 2023; Ramdas et al. 2023) over the
> stream of discordant items drawn, in order, from `adaptation-512` (never `eval-200`/`holdout-100`
> — the existing split-discipline lock is unchanged). Wealth initializes at `W_0 = 1`. For each new
> item `i`: if concordant, `W_i = W_{i-1}` (no update); if discordant, `W_i = W_{i-1} · (1 + λ(X_i −
> 0.5))`, `X_i = 1` if the gated-text condition wins the pair, else 0, with betting fraction `λ`
> fixed in advance (recommend `λ` tuned to the smallest interesting effect, θ=0.65, i.e. `λ =
> 2θ−1 = 0.30`; a mixture/universal betting strategy that does not commit to one θ is a documented
> alternative if a single target is judged too restrictive to freeze). **PASS** the instant `W_i ≥
> 1/α = 20` (α=0.05). If `W_i` never crosses 20 by a pre-registered budget `N_max` (recommend
> `N_max` = enough draws to expect ≈30 discordant pairs at the ladder's observed ~10% discordance
> rate, i.e. `N_max ≈ 300` items — this reuses §2.2's finding that ~25-30 discordant pairs are
> needed for real power at a plausible effect size), the rung is a registered FAIL, its full `W_t`
> trajectory is committed to the receipt regardless of outcome, and the ladder proceeds to the next
> architecture exactly as today's no-rerun discipline requires. **The e-process is never restarted,
> re-parametrized, or re-run against the same rung after seeing `W_t`** — this is the direct
> analogue of "the probe is never iterated, re-drawn, or re-tuned to make a rung pass" (ADR-024),
> extended to the sequential setting.

This directly upgrades ADR-030's current "both scales pre-registered together, not sequentially"
compromise: instead of committing to exactly two fixed looks (40 and 200 items) chosen in advance
to avoid an implicit-peeking problem, the e-process gets the peeking-safety property *natively* —
it can stop at whatever item count first produces real evidence, or run out to a larger budget on
ambiguity, without the family of "how many looks did we pre-commit to" questions a reviewer would
otherwise ask about a two-scale design.

### 3.3 What this buys, concretely, against §2's numbers

An e-process is not a free lunch — for a *fixed* stopping time its power is somewhat below a
same-N fixed-sample test (the cost of anytime validity). Its value here is specifically that it
removes the need to guess N in advance: a rung with a real θ=0.9 effect (like S1a's) would very
likely cross the boundary well before N=300 (§2.2 shows n_disc=10 already gives 74% power at
θ=0.9 with a fixed test; an e-process with a well-chosen λ reaches significance at a comparable
item count), while a rung with a real-but-modest θ=0.65 effect gets to keep accumulating evidence
up to the pre-registered budget instead of being declared FAIL at whatever N the ladder happened to
freeze at (40, or 200) — the actual failure mode 6 of the 9 null draws in §1 exhibited (n_disc≤4,
zero power regardless of effect size).

---

## 4. Multiplicity across the ladder (RQ4)

### 4.1 The honest count

**10 valid probe draws to date** (§1), not ~7 — the corrected count includes the 4 S2b draws (2
depth cells × 2 calibration distributions) that a coarser count might collapse into fewer "runs."

### 4.2 Two genuinely different families, and why that split is defensible

- **Family A — "does the injection pathway transmit information at all" (1 test: S1a run 2).**
  A same-model self-pair mechanics check, logically prior to any cross-model claim. It passed. It
  is not part of the family below, because it does not test cross-model transfer at all — it tests
  whether the *plumbing* works. Treating it as a separate, single, a priori hypothesis is
  defensible on mechanistic grounds, not just convenient timing.
- **Family B — "does some trained/aligned cross-model architecture carry causal signal" (9 tests:
  S2b×4, M3×2, M4×3).** This is the family a reviewer will actually ask about, because it is the
  one the mission's headline claim depends on. **All 9 failed.** There is, right now, no
  multiplicity problem to retroactively correct, because there is no pass in this family being
  showcased — multiplicity only bites when a result is used to support a claim. The obligation is
  forward-looking: **the first time a future rung in this family passes (M4c, M4b, M5, the scale-
  control arm, or a run-3 rung if it is later judged to belong to this family), it must be reported
  as "pass #1 of N architectures tried in this family," not as if it were the only test ever run.**
  With N=9 independent-ish tests at nominal α=0.05 each, P(≥1 false pass by chance) ≈ 1−0.95⁹ ≈
  37% — not small, and exactly the number a skeptical reviewer will compute unprompted if this ADR
  doesn't.

### 4.3 The concrete, cheap correction: fixed-sequence gatekeeping already covers most of it

The methodologically load-bearing fact: **ADR-024's own ladder structure is already a valid
fixed-sequence testing procedure** (Maurer, Hothorn & Lehmacher 1995; Westfall & Krishen 2001) for
the *between-architecture* order — M3 → M4(r=64) → M4(r=128) → M4(r=256) → M5, each tested only
after the prior one fails, stopping and reporting at the first pass. **A pre-specified, ordered,
gatekeeped sequence of this kind needs no alpha correction at all between its steps** — this is
exactly what the ladder already does and already commits to in writing ("tried once each ... if
r=64 clears the gate, M4 stops there"). This is good news: most of the ladder's apparent multiple-
testing burden is already handled correctly by its existing design discipline, and ADR-030 should
say so explicitly rather than leaving it implicit.

**What is *not* gatekeeped, and does need correction, are the parallel variants run and reported
together within a single rung:**

| Parallel family | Tests | p-values | Holm-Bonferroni (m=2, α=0.05) |
|---|---|---|---|
| S2b gold-pairs cells (L18→L14, L24→L19) | 2 | 0.5000, 0.8750 | smallest needs <0.025 — fails |
| S2b generated-pairs cells | 2 | 0.3125, 0.8750 | smallest needs <0.025 — fails |
| M3 variants (per-token, pooled) | 2 | 0.6875, 0.5000 | smallest needs <0.025 — fails |

None of these corrections change any existing verdict (nothing was close to nominal significance
even uncorrected), which is exactly why now — before a future parallel family produces a marginal
pass — is the right time to freeze the rule, not after. **Recommend ADR-030 states explicitly:
within-rung parallel variants use Holm-Bonferroni at the family's own α; between-rung architecture
choices use the existing gatekeeped order and need no further correction, provided each rung is
only evaluated after its predecessor's fail is reported (never in parallel with it).**

### 4.4 A framing methodologists would recognize

This ladder is structurally closest to a **multi-arm multi-stage (MAMS) platform-trial design**
(Royston, Barthel et al., and the broader platform-trial literature) with a shared control
(random-vector injection) and sequentially added arms (architectures), where arms are dropped for
futility (a rung fails, escalate to the next architecture) rather than all arms being carried to a
single final analysis. The standard multiplicity guidance for that design class — the between-arm
family-wise rate is protected by the gatekeeping/dropping rule itself, and within-arm interim
comparisons need their own correction — is exactly what §4.3 already derives independently from
the ladder's own written rules. This is worth one citation in ADR-030 (naming the design class) so
a reviewer sees the ladder recognized a real methodology, not reinvented one ad hoc.

---

## 5. One more thing directly relevant to ADR-030's current freeze

ADR-030's primary gate compares the gated-text condition against **"the best (most favorable to
the null) of the four controls."** If "best" is selected *after* seeing all four controls' accuracy
on the same 40 (or 200) items used for the primary test, this is a mild selection effect (comparing
against whichever control happens to look strongest in this draw, which is expected to look
stronger than its own population value by chance) — a different and smaller problem than §4's
ladder multiplicity, but real. **Recommend ADR-030 add one sentence pinning down the timing**: either
(a) the control expected to be strongest is named in advance from ADR-003's own prior
characterization of the four controls, and "best" in the acceptance criterion just means "in case
the a priori ranking is wrong, use whichever actually scored highest" (defensible, low-risk), or
(b) if "best" is genuinely chosen post-hoc from the same-draw data, note that this makes the
composite test slightly conservative in the useful direction (harder to pass, since you're
comparing against the toughest-looking control) rather than anti-conservative — which is the
opposite of the usual multiple-comparisons risk and worth stating explicitly so a reviewer doesn't
have to work it out.

---

## Sources consulted

- Fagerland, M.W., Lydersen, S., Laake, P. (2013). "The McNemar test for binary matched-pairs data:
  mid-p and asymptotic are better than exact conditional." *BMC Medical Research Methodology* 13:91.
  https://link.springer.com/article/10.1186/1471-2288-13-91
- Ramdas, A., Grünwald, P., Vovk, V., Shafer, G. (2023). "Game-Theoretic Statistics and Safe
  Anytime-Valid Inference." *Statistical Science* 38(4):576-601. arXiv:2210.01948.
  https://projecteuclid.org/journals/statistical-science/volume-38/issue-4/Game-Theoretic-Statistics-and-Safe-Anytime-Valid-Inference/10.1214/23-STS894.full
- Waudby-Smith, I., Ramdas, A. (2023). "Estimating Means of Bounded Random Variables by Betting."
  *JRSS-B* 85(1). arXiv:2010.09686.
- Shafer, G. (2021). "Testing by Betting." *JRSS-A* 184(2).
- All p-values, wins/losses, and per-item NLL values: read directly from
  `crates/latentmesh-runtime/receipts/{s1a-receipt-slots8-block19-poolfull-rescaletrue-n40,
  s2b-receipt-cellL18toL14-slots8-poolfull-rescaletrue-n40[-genpairs],
  s2b-receipt-cellL24toL19-slots8-poolfull-rescaletrue-n40[-genpairs],
  run2-m3-receipt-cellL18toL14-mlp-{pertoken,pooled}-slots8-poolfull-rescaletrue-n40,
  run2-m4-receipt-cellL18toL14-fastgrnn-r{64,128,256}-slots8-poolfull-rescaletrue-n40}.json`, this
  session. Power tables and mid-p values computed in this session (exact binomial enumeration, no
  normal approximation used for anything reported as "exact").
