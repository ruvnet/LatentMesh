# 038. N-invariant manifold classification: retiring the absolute token-union rule

- **Status**: Accepted — **Implemented** in this change
- **Date**: 2026-08-29
- **Supersedes**: the support arm of the classification thresholds registered
  in [ADR-024](024-run2-trained-thought-adapter-ladder.md) (§ M4f pre-check)
- **Related**: [ADR-031](031-evidence-receipt-and-statistical-protocol-governance.md)
  (instrument governance), [ADR-036](036-successor-rung-evaluation-protocol.md)

## Context

`examples/common/lens.rs::classify()` assigns every pre-check payload a
two-axis verdict: **item-invariance × manifold**. The invariance axis was a
disjunction:

```rust
let invariant = invariance >= COLLAPSE_COSINE || token_union <= COLLAPSE_TOKEN_UNION;
```

`COLLAPSE_TOKEN_UNION = 120` is an **absolute count** of distinct tokens in the
union of the per-item top-10 sets. Its attainable ceiling is **`10 × n_items`**.
The constant's own doc comment scoped it — *"the union of the **40** top-10
sets (max possible 400)"* — and it was then reused unchanged at N=300.

**The defect surfaced as a false reassurance**, which is the dangerous
direction:

| | N | union | ceiling | % of ceiling | dominant token in | verdict |
|---|---|---|---|---|---|---|
| PC1 | 40 | 98 | 400 | **24.5%** | 40/40 (100%) | `item-invariant-but-on-manifold` |
| PC1b | 300 | 219 | 3000 | **7.3%** | 297/300 (99%) | `on-manifold-item-varying` |

PC1b's payload is **more** concentrated as a fraction, yet received the *less*
alarming label — purely because N changed. The coordinator initially read this
flip as PC1's anomaly being resolved. It was not.

`run2_pc1b_precheck_n_invariance` settled it empirically: recomputed over
**PC1b's own first 40 items**, its payload classifies
`item-invariant-but-on-manifold` — exactly as PC1's did — with a *lower* union
(**86** vs 98). `flip_is_an_n_artifact = true`.

## Decision

**Retire the absolute union count as the support arm. Replace it with the
dominant-token share.**

```rust
pub const COLLAPSE_DOMINANT_TOKEN_SHARE: f64 = 0.90;

pub fn dominant_token_share(top_sets: &[Vec<u32>]) -> f64 { /* … */ }

let invariant = invariance >= COLLAPSE_COSINE
    || dominant_share >= COLLAPSE_DOMINANT_TOKEN_SHARE;
```

`dominant_token_share` is the share of items whose top-k set contains the
single most frequent token — **a fraction of the items measured, hence
N-invariant by construction.**

### Why not "a fraction of the 10×N ceiling"

The obvious fix — normalise the union by its ceiling — **fails in the opposite
direction**, and the data says so. The union **saturates sublinearly**: PC1b
scores 98 at n=40 and 219 at n=300, against **735** if it grew linearly.
Vocabulary is finite and a concentrated emitter reuses tokens, so
fraction-of-ceiling falls monotonically with N and would label almost any
large-N draw "collapsed". A **per-item share** has no such drift.

### Calibration, stated so it is auditable

`0.90` is anchored on the one thing the matched-N diagnostic **proved**: PC1
(**1.00**) and PC1b (**0.99**) are the same geometric family and must classify
alike. It is *not* tuned to reproduce every historical label.

## Consequences

- **Verdicts are now comparable across draws of different N.** Cross-rung use
  of the `item-invariant` label, previously void wherever N differed, is
  restored for draws taken after this change.
- **Historical receipts are NOT recomputed.** They record the retired rule's
  output, and remain valid as records of what the instrument said at the time.
  Some historical labels would change under the new arm; **that is not
  silently reconciled**, per the append-only rule.
- **The union count is still reported**, under
  `collapse_top10_token_union_at_or_below_RETIRED`, with an inline explanation
  of why it no longer feeds the verdict. Continuity of the artifact series is
  preserved.
- **The classification remains non-gating** — diagnostic only. No rung's
  verdict changes. In particular **PC1b's FAIL is untouched**: its primary is
  the ADR-036 e-process on accuracy, which never consulted `classify()`.
- **PC1's item-invariance is still unexplained.** This ADR fixes the
  *instrument*, not the observation. Both payloads genuinely concentrate a
  single token (id 30543) into ~100% of items, and **why** remains open.

## Verification

- `dominant_token_share` and the N-invariance property are covered by two new
  unit tests, one using PC1's and PC1b's **real** numbers as a regression
  against exactly this defect.
- `cargo fmt --check`, `cargo clippy --all-targets --features cuda -D
  warnings`, and `cargo test --features cuda --examples` all clean: **27 test
  suites ok, 0 failed**.
- One definition only: all four call sites (`run2_manifold_precheck`,
  `run2_pc1_manifold_precheck`, `run2_pc1b_manifold_precheck`,
  `run2_pc1b_precheck_n_invariance`) call the shared helper rather than
  recomputing — the drift risk `lens.rs` was extracted to prevent.

## Incidental finding, now measured

The 02:09 rebuild that occurred under PC1b's live draw produced a
**13,102,888-byte** artifact, while the draw ran a binary reported at 30.4 MB.
The cuda-flag explanation was recorded as *plausible but unverified*. Building
this example **with `--features cuda` yields 30,379,448 bytes**, confirming it:
the on-disk artifact was a **non-cuda** build and was never the draw's binary.
`4cda2ef1…` remains PC1b's binary of record.
