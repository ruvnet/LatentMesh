# 047. The authoritative, receipt-derived power table for every probe draw

* **Purpose**: settle [ADR-024 §"UNRESOLVED DISCREPANCY — how many draws were power-incapable?"](../adr/024-run2-trained-thought-adapter-ladder.md)
  from primary evidence only. Every `n_disc` below was **recomputed by counting discordant items in
  each receipt's own per-item table**, then cross-checked against that receipt's `summary.primary_*`
  block. No number in this document is copied from prose in ADR-024, research/031, research/041 or
  research/046.
* **Date**: 2026-08-29. Branch `feat/run2-thought-adapter`. Read-only against the repo except this
  one file; not committed.
* **Method**: `wins` = items where the primary condition is scored correct and its comparator is
  not; `losses` = the reverse; `n_disc = wins + losses`; exact one-sided p =
  `P(X ≥ wins | Binomial(n_disc, 0.5))`; minimum attainable one-sided p at that `n_disc` = `2^-n_disc`
  (all discordant pairs win); mid-p = `exact_p − 0.5·P(X = wins)`. α = 0.05 throughout.
* **Headline**: **14 valid draws exist across both runs. 10 of them were structurally incapable of
  rejecting the null at any true effect size.** The circulating "6 of 9" figure is **wrong**; the
  correct figure for that family is **8 of 9**. research/046's independent recount (10 of 13
  cross-model) is **confirmed correct at the receipt level** — this document supplies the
  receipt-level reconciliation 046 explicitly declined to perform.

---

## 1. The table

Every row was recomputed from the named receipt's `items[].conditions[*].correct` booleans. All
receipts live in `crates/latentmesh-runtime/receipts/`.

| # | Draw | Run | Era | Family | Primary comparison | W | L | conc. | **n_disc** | exact p | min p (2^-n) | **capable?** | mid-p | gate |
|---:|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| 1 | S1a run 2 (self-pair, identity transform) | 1 | pre-M4c | same-model | `real` vs `random` | 5 | 0 | 35 | **5** | 0.0312 | 0.0312 | **capable** | 0.0156 | PASS |
| 2 | S2b gold-pair calibration, L18→L14 (winner cell) | 1 | pre-M4c | cross-model | `aligned_real` vs `random` | 2 | 1 | 37 | **3** | 0.5000 | 0.1250 | INCAPABLE | 0.3125 | FAIL |
| 3 | S2b gold-pair calibration, L24→L19 (anchor cell) | 1 | pre-M4c | cross-model | `aligned_real` vs `random` | 1 | 2 | 37 | **3** | 0.8750 | 0.1250 | INCAPABLE | 0.6875 | FAIL |
| 4 | S2c generated-pair calibration, L18→L14 | 1 | pre-M4c | cross-model | `aligned_real` vs `random` | 3 | 1 | 36 | **4** | 0.3125 | 0.0625 | INCAPABLE | 0.1875 | FAIL |
| 5 | S2c generated-pair calibration, L24→L19 | 1 | pre-M4c | cross-model | `aligned_real` vs `random` | 1 | 2 | 37 | **3** | 0.8750 | 0.1250 | INCAPABLE | 0.6875 | FAIL |
| 6 | M3 MLP, per-token variant | 2 | pre-M4c | cross-model | `aligned_real` vs `random` | 2 | 2 | 36 | **4** | 0.6875 | 0.0625 | INCAPABLE | 0.5000 | FAIL |
| 7 | M3 MLP, pooled variant | 2 | pre-M4c | cross-model | `aligned_real` vs `random` | 2 | 1 | 37 | **3** | 0.5000 | 0.1250 | INCAPABLE | 0.3125 | FAIL |
| 8 | M4 FastGRNN r=64 | 2 | pre-M4c | cross-model | `aligned_real` vs `random` | 3 | 1 | 36 | **4** | 0.3125 | 0.0625 | INCAPABLE | 0.1875 | FAIL |
| 9 | M4 FastGRNN r=128 | 2 | pre-M4c | cross-model | `aligned_real` vs `random` | 3 | 1 | 36 | **4** | 0.3125 | 0.0625 | INCAPABLE | 0.1875 | FAIL |
| 10 | M4 FastGRNN r=256 | 2 | pre-M4c | cross-model | `aligned_real` vs `random` | 4 | 1 | 35 | **5** | 0.1875 | 0.0312 | **capable** | 0.1094 | FAIL |
| 11 | M4c MLP task-loss | 2 | M4c+ | cross-model | `aligned_real` vs `random` | 4 | 2 | 34 | **6** | 0.3438 | 0.0156 | **capable** | 0.2266 | FAIL |
| 12 | M4d MLP task-loss + deploy-match | 2 | M4c+ | cross-model | `aligned_real` vs `random` | 5 | 2 | 33 | **7** | 0.2266 | 0.0078 | **capable** | 0.1445 | FAIL |
| 13 | M4g MLP fuse (not overwrite) | 2 | M4c+ | cross-model | `aligned_real` vs `random` | 2 | 1 | 37 | **3** | 0.5000 | 0.1250 | INCAPABLE | 0.3125 | FAIL |
| 14 | M4h Stage 1, de-pooled fuse | 2 | M4c+ | cross-model | `aligned_real` vs `random` | 2 | 0 | 38 | **2** | 0.2500 | 0.2500 | INCAPABLE | 0.1250 | FAIL |

`n_disc` multiset, all 14 draws: **{2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6, 7}**
`n_disc` multiset, 13 cross-model draws (S1a excluded): **{2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 6, 7}**

Receipt filenames, in table order:

1. `s1a-receipt-slots8-block19-poolfull-rescaletrue-n40.json`
2. `s2b-receipt-cellL18toL14-slots8-poolfull-rescaletrue-n40.json`
3. `s2b-receipt-cellL24toL19-slots8-poolfull-rescaletrue-n40.json`
4. `s2b-receipt-cellL18toL14-slots8-poolfull-rescaletrue-n40-genpairs.json`
5. `s2b-receipt-cellL24toL19-slots8-poolfull-rescaletrue-n40-genpairs.json`
6. `run2-m3-receipt-cellL18toL14-mlp-pertoken-slots8-poolfull-rescaletrue-n40.json`
7. `run2-m3-receipt-cellL18toL14-mlp-pooled-slots8-poolfull-rescaletrue-n40.json`
8. `run2-m4-receipt-cellL18toL14-fastgrnn-r64-slots8-poolfull-rescaletrue-n40.json`
9. `run2-m4-receipt-cellL18toL14-fastgrnn-r128-slots8-poolfull-rescaletrue-n40.json`
10. `run2-m4-receipt-cellL18toL14-fastgrnn-r256-slots8-poolfull-rescaletrue-n40.json`
11. `run2-m4c-receipt-cellL18toL14-mlp-taskloss-slots8-poolfull-rescaletrue-n40.json`
12. `run2-m4d-receipt-cellL18toL14-mlp-deploymatch-slots8-poolfull-rescaletrue-n40.json`
13. `run2-m4g-receipt-cellL18toL14-mlp-fuse-slots8-poolfull-rescaletrue-n40.json`
14. `run2-m4h-s1-receipt-cellL18toL14-mlp-pertokenlast-fuse-slots8-nopool-rescaletrue-n40.json`

**Note on rows 4 and 5.** Both carry `stage: "S2b-bridge-probe"`, but their `config.transform.file`
points at `transform-gen-L18-to-L14.json` / `transform-gen-L24-to-L19.json` — the *generated*-pair
calibration, i.e. the S2c distribution. research/031 §1 calls them "S2b generated"; ADR-024's
discrepancy note calls the set "the four S2b/S2c cells". They are the same four rows either way.

### 1a. Receipts examined and deliberately excluded

Four further receipts carry per-item condition tables but are **not** valid draws against the null:

| Receipt | n_items | W/L | n_disc | Why excluded |
|---|---:|---|---:|---|
| `s1a-receipt-run1-buggy-rope-noncompliant-prompt.json` | 40 | 1/1 | 2 | S1a iteration 1 — invalidated by the BF16 RoPE bug and the answer-format scoring bug (ADR-023 Deviation 3, `run-ledger.json`). Never a valid draw. |
| `s1a-receipt-slots8-block19-poolfull-rescaletrue-n40.run1-pre-fixes.json` | 40 | 1/1 | 2 | **Byte-identical duplicate** of the above (both sha256 `f6e900ca…4673`). Counting it twice would double-count an already-invalid draw. |
| `s1a-receipt-slots8-block19-poolfull-rescaletrue-n2.json` | 2 | 0/0 | 0 | 2-item smoke pilot. |
| `s2b-receipt-cellL18toL14-slots8-poolfull-rescaletrue-n2.json` | 2 | 0/0 | 0 | 2-item smoke pilot. |

**M4i and PC1 have no draw receipt.** `run2-m4i-manifold-precheck-receipt.json` exists but is a
zero-GPU geometry precheck (`stage: run2-m4f-manifold-collapse-precheck`) with no `items` array, no
conditions, and no draw against the null — as are `run2-m4h-s1-manifold-precheck-receipt.json`,
`run2-manifold-precheck-m4g-receipt.json` and `run2-manifold-precheck-receipt.json`. PC1 (registered
in ADR-024, 2026-08-29) has not run; no receipt exists. **No receipt named in ADR-024's draw
inventory is missing** — the 14 rows above are the complete valid set as of this document's date.

---

## 2. Summary counts under every defensible convention

The circulating figures differ mainly because they count *different families*. All of these are
defensible scopes; none is silently preferred here. Read the row that matches the claim you are
checking.

| Convention | draws | **incapable** | capable | Who uses it |
|---|---:|---:|---:|---|
| **All valid draws, both runs, all families** | **14** | **10** | 4 | *this document's headline* |
| All cross-model draws (S1a excluded as same-model sanity check) | 13 | **10** | 3 | research/046 §2.5 |
| Run 1 only (S1a + the four S2b/S2c cells) | 5 | 4 | 1 | — |
| Run 1, cross-model only (the four cells) | 4 | **4** | 0 | — |
| Run 2 only (M3×2, M4×3, M4c, M4d, M4g, M4h-S1) | 9 | 6 | 3 | — |
| Pre-M4c, incl. S1a | 10 | 8 | 2 | research/031 §1's table (10 rows) |
| **Pre-M4c, cross-model (S2b×4, M3×2, M4×3)** | **9** | **8** | 1 | ADR-024's flag; the "6 of 9" claim's family |
| M4c and after (M4c, M4d, M4g, M4h-S1) | 4 | 2 | 2 | — |
| Through M4d, incl. S1a | 12 | **8** | 4 | 041 §2.5's "12 draws" |
| Through M4d, cross-model | 11 | **8** | 3 | ADR-024's flag's "11 draws total" |

**Under mid-p McNemar** — which ADR-030 and ADR-035 register as the *primary* statistic going
forward — the floor moves. Minimum attainable mid-p at `n_disc` is `2^-(n_disc+1)`, so `n_disc=4`
becomes capable (0.03125 < α) while `n_disc≤3` stays incapable. Recounted on that basis:
**14 draws, 6 incapable, 8 capable** (cross-model only: 13 draws, 6 incapable, 7 capable). This does
not change any past verdict — no draw's actual mid-p reached α — but it is the right floor to quote
for any *future* rung registered under mid-p, and it is why ADR-035's power expectation for M4b
("if M4b lands at `n_disc≤4`…") is conservative rather than exact under its own registered statistic.

---

## 3. Adjudication of the circulating figures

### 3.1 "6 of 9" — **WRONG.** The correct figure is **8 of 9**.

The family is unambiguous: the 9 cross-model draws through M4 (S2b×4, M3×2, M4×3), rows 2–10 above.
Their `n_disc` values are 3, 3, 4, 3, 4, 3, 4, 4, 5. Eight of the nine sit at `n_disc ∈ {3,4}`; only
M4 r=256 (`n_disc=5`, min attainable p = 0.03125) cleared the floor. **8 of 9, not 6 of 9.**

**Where the "6" came from.** research/031 §0 item 1 reads: *"6 of the 9 cross-model null draws to
date (S2b×4, M3×2) landed in that dead zone."* The parenthetical names **six** draws — S2b×4 + M3×2
— and at the moment that sentence was drafted those six were the entire cross-model set, all six in
the dead zone. When M4×3 was appended to §1's table the denominator was updated from 6 to 9 and the
numerator was not. Two of M4's three draws (r=64 and r=128, both `n_disc=4`) also land in the dead
zone, which is what takes 6 to 8.

**A second, independent error rides along in the same sentence.** §0 states the discordant counts as
`(3, 3, 3, 4, 4, 4, 4, 5, 5)`. That is *not* the cross-model multiset; it is the **10-row S1a-inclusive
table with one `n_disc=3` draw dropped** (the true 10-row multiset is `{3,3,3,3,4,4,4,4,5,5}`; the
true cross-model 9 is `{3,3,3,3,4,4,4,4,5}`). So the quoted list simultaneously includes S1a's 5 and
omits an S2b/M3 3. Note that even inside that corrupted list, the count at `n_disc ∈ {3,4}` is **7**,
not 6 — the "6 of 9" claim is not self-consistent with the very list printed beside it.

**Every occurrence of the wrong figure, for annotation:**

| File:line | Text |
|---|---|
| `docs/research/031-statistical-power-and-design.md:29` | *"6 of the 9 cross-model null draws to date (S2b×4, M3×2) landed in that dead zone"* — plus the miscopied multiset `(3, 3, 3, 4, 4, 4, 4, 5, 5)` at line 28 |
| `docs/research/031-statistical-power-and-design.md:112` | *"6 of the 9 cross-model null draws in §1 (n_disc=3 or 4) could not have passed…"* |
| `docs/research/031-statistical-power-and-design.md:282` | *"the actual failure mode 6 of the 9 null draws in §1 exhibited"* |
| `docs/adr/030-run3-causally-gated-text-pre-registration.md:143-144` | inherits both errors: *"discordant-pair counts were {3,3,3,4,4,4,4,5,5} … 6 of 9 landed at `n_disc∈{3,4}`"* |
| `docs/adr/024-run2-trained-thought-adapter-ladder.md` | ADR-024's own flag (line 1048) states the figure was *"repeated throughout this ADR and in every summary"* — its own flag is the correction, and it is right |

research/031 §1's **table itself is correct** and always was — all ten rows' wins/losses/`n_disc`/exact-p/mid-p
reproduce exactly from the receipts (verified below, §4). The error is confined to the prose that
summarizes it.

### 3.2 "10 draws, 6 at n_disc ∈ {3,4}" (041 §2.5) — **WRONG on both numbers.**

041 §2.5 reads: *"Of the 10 valid cross-model draws through M4 (S2b×4, M3×2, M4×3, excluding S1a),
6 landed at `n_disc∈{3,4}`."* The parenthetical sums to **9**, not 10. 041 took research/031 §1's
"10 valid draws" — a count that *includes* S1a — and relabelled it "10 valid cross-model draws …
excluding S1a", which cannot both be true. Correct: **9 cross-model draws through M4, 8 of them
incapable.**

### 3.3 "12 draws, 8 capable / 4 incapable" (041 §2.5) — **the 12 is right; the split is exactly inverted.**

The family (S1a + S2b×4 + M3×2 + M4×3 + M4c + M4d) does contain **12** valid draws. But the split is
**4 capable (S1a `n_disc=5`, M4 r=256 `n_disc=5`, M4c `n_disc=6`, M4d `n_disc=7`) and 8 incapable** —
the precise inverse of what 041 states. 041's own two sentences also disagree with each other: "6
landed at `n_disc∈{3,4}`" followed by "4 of which … could not have rejected" cannot both hold, and
neither matches 12 − 8 = 4. Every number in that bullet except the "12" is wrong.

**Where it appears:** `docs/research/041-run2-synthesis-skeleton.md:162-167`. Two downstream
restatements inherit it: line 254 (figure 3, "all 12 valid draws") is fine as a *count*; line 188
(correction-log row 1, *"3 of the 5 M3/M4 draws landed at n_disc∈{3,4}"*) is **wrong** — of M3×2 +
M4×3, four of five (`n_disc` 4, 3, 4, 4; only r=256's 5 clears) are in the dead zone, so it should
read **4 of the 5**.

### 3.4 "9 pre-M4c draws, 8 incapable; 11 total after M4c/M4d" (ADR-024's flag) — **CORRECT.**

ADR-024's flagged recomputation, derived from its own per-rung annotations (S2b 3,3,4,3; M3 4,3; M4
4,4,5), reproduces the receipts exactly. Its scope is the **cross-model** set with S1a excluded, and
under that scope both of its numbers hold: 9 pre-M4c draws, 8 incapable; 11 draws through M4d, 8
incapable. The flag should be closed as **adjudicated correct**, with the one clarification that
"9 pre-M4c draws" means *cross-model* pre-M4c draws — counting S1a makes it 10 pre-M4c draws, still
8 incapable.

### 3.5 "10 of 13 cross-model incapable" (046 §2.5) — **CORRECT, and now receipt-verified.**

research/046 recounted from ADR-024's per-rung prose annotations and got 13 cross-model draws, 10
incapable, 3 capable (M4 r=256, M4c, M4d). It explicitly disclosed that *"the underlying raw JSON
receipts were not independently re-summed this session"* and left the adjudication open. **This
document performs that re-summation and 046's numbers are confirmed exactly.** 046 §7 item 1 can be
closed.

### 3.6 One further inaccuracy, surfaced by the same table

`docs/adr/035-m4b-scale-control-pre-registration.md:231-232` states that M4c (`n_disc=6`) and M4d
(`n_disc=7`) *"were the first to clear that floor."* M4 r=256 (`n_disc=5`, min attainable p =
0.03125 < α) cleared it first, if only barely — it needed a clean 5/5 sweep and drew 4/1. The
substantive claim ADR-035 rests on (M4d is the ladder's first *comfortably* informative negative)
survives; the word "first" does not. This is a wording fix, not a numbers fix.

---

## 4. Receipt-completeness findings (relevant to ADR-031's receipt contract)

**Finding 1 — every valid draw has a per-item table, and every receipted summary reproduces from
it.** All 14 draws carry `items[]` with `conditions[*].correct` booleans for every condition. For all
14, the independently recomputed `wins`, `losses` and `p_one_sided` match the receipt's own
`summary.primary_*` block to the last digit. **Zero mismatches.** No draw's `n_disc` had to be taken
on trust, and no receipt misreports its own statistics. The receipt layer is sound; the drift is
entirely in prose that restates it.

**Finding 2 — the receipt schema changed mid-ladder, and the older half is the half that drifted.**
Draws 1–11 (S1a through **M4c**) record only `wins`, `losses`, `p_one_sided`, `alpha`, `pass` in
their primary block. They carry **no `n_discordant` field, no `min_attainable_p_at_this_n_disc`, and
no mid-p**. `n_disc` for those eleven draws exists nowhere in the receipt as a stated number — it is
only *derivable* as `wins + losses`, and whether the draw was power-capable is not recorded at all.
Only from **M4d onward** (draws 12–14) do receipts self-report `n_discordant` and
`min_attainable_p_at_this_n_disc` (M4h-S1 additionally reports `mid_p_one_sided`). Where those
fields are present, they are all correct.

This is the mechanism of the drift. A quantity that is never written down as a number, only implied
by two other numbers, is a quantity that gets retyped from prose — and every one of the wrong
figures in §3 is a retyping of a derived value, not a misread of a stored one.

**Recommendation for ADR-031's receipt contract**: require every draw receipt's primary block to
carry `n_discordant`, `min_attainable_p_at_this_n_disc`, `mid_p_one_sided`, and an explicit boolean
`power_capable_at_alpha` (`min_attainable_p < alpha`). All four are pure functions of `wins` and
`losses` — zero measurement cost, zero GPU — and all four are exactly the fields whose absence let
"6 of 9" survive three documents and two ADRs. Backfilling them into draws 1–11 would be an
append-only annotation, not a rewrite of any measured value.

**Finding 3 — one duplicated receipt.** `s1a-receipt-run1-buggy-rope-noncompliant-prompt.json` and
`s1a-receipt-slots8-block19-poolfull-rescaletrue-n40.run1-pre-fixes.json` are byte-identical
(sha256 `f6e900cadddb7185ca77011bc72141cdbd047175611f6caed704fd4c9f804673`). `run-ledger.json` names
both and flags the duplication honestly, so nothing is hidden — but any future automated sweep over
`receipts/*.json` that counts draws by file will double-count an invalidated run-1 draw. Worth a
`superseded: true` marker of the kind the `*-superseded-windowzeroinit` artifacts already use.

**Finding 4 — the four manifold-precheck receipts carry no per-item table.** This is *not* a contract
violation: they are geometry diagnostics (cosine/entropy/invariance against candidate artifacts),
not draws against the null, and they report aggregate candidate statistics appropriately. Flagged
only so a future sweep does not mistake them for missing draw data.

---

## 5. What the corrected numbers mean

The correction runs in the direction ADR-024's flag anticipated: **the instrument was blinder than
the record said.** Not 6 of 9 but 8 of 9 pre-M4c cross-model draws could not have rejected at any
effect size; across the whole ladder, 10 of 14 draws (10 of 13 cross-model) were structurally
incapable. Only four draws in the entire two-run history were ever able to reject, and one of those
(S1a) is the same-model sanity check that did.

This **strengthens** the already-recorded instrument finding rather than weakening any conclusion.
The ladder's nulls were, for the most part, not evidence that the architectures carry no signal;
they were the arithmetic of a test that could not have said otherwise. The three cross-model draws
that *could* have rejected and did not (M4 r=256, M4c, M4d) remain the only genuinely informative
cross-model negatives, and M4d (`n_disc=7`, min p 0.0078) remains the most informative of them —
that framing is unchanged.

It also sharpens ADR-036's "the binding constraint is the instrument" reading: the trajectory
`M4d=7 → M4g=3 → M4h-S1=2` means the two most recent rungs are the **least** capable draws in the
entire ladder, M4h-S1 (`n_disc=2`, floor 0.25) being the weakest draw ever recorded here — weaker
even than the invalidated S1a run 1. No conclusion about de-pooled fuse transfer can rest on it.

---

## 6. How to cite this

**Quote the receipt-derived numbers from §1's table, with the receipt filename, not prose from any
other document — including this one's §5.** Concretely:

1. **Name the convention before the number.** "8 of 9" and "10 of 13" and "10 of 14" are all
   correct; they count different families. Any bare "N of M" without the family named is how this
   discrepancy started. Use the §2 table's left column verbatim as the scope label.
2. **The default scope for a write-up is the full 14-draw table**, with the cross-model 13 given
   alongside — S1a is a real draw and excluding it silently is what produced 041's "10 cross-model
   … excluding S1a" contradiction.
3. **Cite the receipt, not the summary.** Every row in §1 names its file. `n_disc` for draws 1–11 is
   `wins + losses` from the per-item table; do not expect an `n_discordant` field before M4d.
4. **State the floor alongside the p-value, always.** A draw's exact p means nothing without
   `2^-n_disc` beside it. "M4g: p=0.50" is misleading; "M4g: p=0.50, n_disc=3, floor 0.125 —
   structurally incapable of rejecting" is the honest form, and it is the form ADR-024 §"M4h Stage 1
   OUTCOME" and ADR-035 already model.
5. **Which statistic's floor.** Under the exact sign test the dead zone is `n_disc ≤ 4`; under mid-p
   McNemar (ADR-030/ADR-035's registered primary) it is `n_disc ≤ 3`. Say which one you mean —
   §2's mid-p paragraph gives both counts.
6. **When annotating the wrong figures**, ADR-031(a)'s append-only discipline applies: leave the
   original text intact and append a correction pointing here. §3's per-file table gives the exact
   line numbers.

**Reproduction.** The computation is a single ~150-line Python script over the receipt JSON —
analysis, not shipped code, so the repo's Rust-only rule does not govern it. It reads only
`crates/latentmesh-runtime/receipts/*.json`, does no I/O beyond that, and needs no GPU. Anyone
re-deriving this table should recount from `items[].conditions`, not from `summary.primary_*`, and
should report any mismatch — that cross-check is the point, and it is what establishes §4's Finding 1.
