# 047. M6 — separating manifold conformity from semantic content: pre-registration

- **Status**: **Proposed (pre-registration)** — written before any item is drawn.
  Everything above "## Outcomes" is **frozen on commit**. Outcomes go in an
  appended section; every deviation is a numbered coordinator error.
- **Date**: 2026-08-30.
- **Protocol**: [ADR-036](036-successor-rung-evaluation-protocol.md) e-process ·
  **Power**: [ADR-040](040-pc3-decision-change-endpoint-pre-registration.md) ·
  **Publication**: [ADR-032](032-negative-result-publication-contract.md) ·
  **Controls**: [ADR-003](003-causal-edge-verification.md).
- **Predecessor**: [ADR-045](045-m5-receiver-side-adaptation-pre-registration.md)
  (M5, closed — three powered nulls) and
  [research/054](../research/054-m5-receiver-adaptation-closed.md).

---

## 1. The question M5 could not answer

M5 drew three registered e-processes and none crossed. But at rank 2 the primary
was **46W/23L** — aligned beating random on the decision endpoint, trending hard
at the boundary. That looks like evidence the channel carries content.

It is not interpretable, and the reason is in M5's own accuracy field. At every
rank, accuracy ordered **`baseline` > `aligned` > `random`**:

| | baseline | aligned | random |
|---|---|---|---|
| M5 R1 | 149 | 137 | 130 |
| M5 R2 | 172 | 154 | 131 |
| M5 R4 | 160 | 154 | 137 |

**Every injection hurts. The on-manifold payload hurts least.** So "aligned beats
random" is equally explained by aligned being a *gentler perturbation* — no
content need be transmitted for it to appear.

**The defect is in the control, not the result.** `random` is a norm-matched
Gaussian: it is wrong on **two axes at once** — content-free *and*
off-manifold. When `aligned` beats it, the comparison cannot say which axis was
responsible.

## 2. The design — a 2×2 that identifies the factor

|  | **on-manifold** | **off-manifold** |
|---|---|---|
| **right content** | `aligned` | **`aligned_displaced`** |
| **wrong content** | `mismatched` | `random` |

Six conditions total: these four plus `baseline_uninjected` and
`zerovec_injected` (retained for operator-correctness continuity; under `fuse`,
zerovec is `h += 0` and must be **exactly** baseline).

**The prediction is a PATTERN, and it is what makes this a discriminator:**

- **CONTENT hypothesis** → ordering by **ROW**: `aligned ≈ aligned_displaced > mismatched ≈ random`
- **MANIFOLD hypothesis** → ordering by **COLUMN**: `aligned ≈ mismatched > aligned_displaced ≈ random`

M5 could not distinguish these because it held only the `aligned` / `random`
diagonal, where both hypotheses predict the same sign.

## 3. Prior art — what is ours and what is not

**`mismatched` is NOT ours.** **arXiv:2607.26773** (*Do Latent Channels Actually
Communicate? A Causal Audit of Latent Multi-Agent LLM*, Zhang & Emu, 2026-07-29)
already performs "controlled message replacements at the boundary where the
sender-produced representation enters the receiver" and decomposes an effect
into the part retained by an **other-example message** versus the part
attributable to example-specific content. That is this row axis, published, on a
latent inter-model channel. **We adopt their term "other-example message" for
the concept and cite them as its origin.** They vary no manifold conformity.

**Not the first factorial either.** **arXiv:2608.22140** crosses token content ×
attention allocation — and reports the two factors **coupled, not additive**
(§7.3 below registers what we do if that happens here).

**The column axis has theoretical backing but no crossed design.**
**arXiv:2605.05115** (*Manifold Steering*) shows on-manifold steering yields
natural trajectories while Euclidean steering "cuts through off-manifold
regions", holding content fixed throughout.

**The narrow novelty claim, and nothing wider:** *a controlled displacement
magnitude used as a manifold-conformity axis, crossed with content, in a latent
inter-model channel.* A literature pass (arXiv API only — **WebSearch was
unavailable, so non-arXiv venues are unchecked and these nulls are weak**) found
no prior use of precision or displacement as a deliberate perturbation of a
transmitted representation.

**Why not IGSD** (**arXiv:2606.20678**), which already instruments
content-vs-disruption via matched replacement vs zero ablation: it crosses
intervention *type* × component **within one model**. Our object is a payload
crossing **between** models, and our two factors are properties of the payload,
not of the intervention operator. Complementary, not a substitute — and a
reviewer will ask, so it is answered here.

## 4. `aligned_displaced` — the design decision that was nearly wrong

**The obvious implementation is the wrong one.** Capturing the payload from a
quantized *copy* of the model produces `w = f_Q4(tokens)` — a different function
of the same tokens. It differs from `aligned` on manifold conformity **and** on
the computation that produced it: two factors, precisely the confound this rung
exists to remove.

Worse, it may **invert** the manipulation. `w` lies on the *quantized model's own*
activation manifold, and quantization error propagating through 18 blocks is
shaped by the weights into a plausible activation. So `w` may be **more**
on-manifold than a directly displaced vector while encoding slightly different
content — a weak, uncontrolled `mismatched` dressed as the manifold cell.

**Therefore: displace the payload VECTOR at injection.** The producing
computation is bit-identical, content is held exactly, and a generic
displacement leaves the low-dimensional activation manifold essentially surely.
One factor moves.

### 4.1 Parameterise by DISPLACEMENT MAGNITUDE, not bit width

Bit width imports a hardware connotation that does no work here, and the
arithmetic shows why it is actively misleading. For symmetric absmax
quantise-dequantise with per-block-32 grids (uniform rounding error,
`step = 2·absmax/(2^b − 1)`, `err_std = step/√12`), the cosine to the
unrounded activation is:

| grid | cosine to FP16 |
|---|---|
| 8-bit | 0.99999 |
| **4-bit** | **0.99610** |
| 3-bit | 0.98248 |
| 2-bit | 0.91442 |
| norm-matched random (1536-d) | **≈ 0.026** |

**A 4-bit-rounded activation is 99.6% of the way to `aligned` and 0.4% of the
way to `random`. It is not a perturbation.** The usable band is 2–3 bits, at
which point calling it "quantization" is no longer honest.

So `aligned_displaced` is registered as a **dose ladder on target cosine**:
**0.99, 0.95, 0.90, 0.75**, with the displacement scaled to hit each target.
This yields a **dose-response curve** rather than one arbitrary point, and
states plainly what the manipulation is.

### 4.2 Registered mechanics — decided here, not at implementation time

1. Displacement is applied to the **final 1536-d injected payload**, never to
   the sender's 2048-d capture before the M3 MLP. Displacing pre-MLP would give
   `f(v+e)` — a different computation again.
2. **Displace first, then rescale** to `natural.median` (M5's rule,
   `common/m3.rs:548`). This erases the norm change and leaves **direction** as
   the only difference, making `aligned` and `aligned_displaced` norm-identical.
3. Displacement is drawn from a **per-item seeded** RNG, recorded in the receipt,
   so every arm is reproducible.
4. `mismatched` is the **previous stream item's** aligned payload, primed from a
   designated item outside the drawn stream, **no lookahead**. Under M5's rescale
   rule it lands on `natural.median` too — so `aligned`, `mismatched`,
   `aligned_displaced` and `random` all arrive at **identical norm by
   construction**.

## 5. MANDATORY MANIPULATION CHECK — gates the draw

**No draw may run until this passes.** The load-bearing cell rests on an
assumption the literature cannot settle: nobody has measured where a displaced
activation sits relative to FP16 (two searches returned literally zero results).

Extend `examples/run2_manifold_precheck.rs` (CPU-only, annotates only) with
`aligned_displaced` at each dose, plus `random`, as additional candidates. It
already computes cosine (`common/lens.rs:155`) and holds "the receiver's OWN L14
states over the same spans" as the on-manifold reference.

**Register a NEW threshold band.** `OFF_MANIFOLD_COSINE = 0.20` (`lens.rs:61`)
is a *collapse* detector and will classify every dose as on-manifold, telling us
nothing.

**Gate:** a dose is admitted only if its measured cosine to the unrounded
payload is within **±0.02** of its target, and its typicality against the
receiver's own L14 reference is **strictly between** `aligned` and `random`.
Doses failing this are **dropped and reported**, not adjusted post hoc.

## 6. MANDATORY POWER CALCULATION (ADR-040), before any draw

**These pairs have never been drawn**, so n_disc cannot be anchored on the same
comparison. Anchored instead on the closest **measured** M5 battery values, all
stored receipt fields:

| comparison | R1 | R2 | R4 |
|---|---|---|---|
| `aligned` vs `random` (off-manifold, content-free) | 77 | 69 | 67 |
| `aligned` vs `baseline` (nearest on-manifold analogue) | 60 | 50 | 58 |

`mismatched` and `aligned_displaced` are both **closer to `aligned`** than
`random` is, so expect discordance nearer the lower anchor. **Registered
expectation: n_disc ≈ 50–60 per pair.**

**The bar is n-DEPENDENT and the wealth rule is the sole authority.** No fixed
count and no fixed rate may be stored as a verdict — in M5 such a field was
wrong twice, in both directions (ADR-045 §"Power-accounting correction").
Required wins solve `(1+λ/2)^k (1−λ/2)^(n−k) ≥ 1/α` at λ = 0.30, α = 0.05:

| realised n_disc | wins needed | min attainable one-sided p |
|---|---|---|
| 40 | 32 (80.0%) | 9.1e-13 |
| **50** | **37 (74.0%)** | 8.9e-16 |
| **60** | **43 (71.7%)** | 8.7e-19 |
| 70 | 48 (68.6%) | 8.5e-22 |

**VERDICT: not power-blocked** at the registered expectation. If realised
n_disc < 30 on a pair, that pair is reported **uninformative** and the power
model recorded as wrong.

### 6.1 Two primaries, one per axis

The discriminator is the pattern, but each axis gets its own registered
e-process:

- **CONTENT axis** — `aligned` vs `mismatched` (both on-manifold; differ only in content)
- **MANIFOLD axis** — `aligned` vs `aligned_displaced` at the **0.90 dose** (both right content; differ only in conformity)

Each is a separate wealth process at λ = 0.30, threshold 20.0, N_max = 300, on
the `adaptation-512` fixed order at the question-tail site. **They are reported
separately and never pooled**; pooling across ranks was explicitly refused in
M5 and is refused here.

## 7. Registered interpretation — every branch, before the draw

**7.1 CONTENT crosses, MANIFOLD does not.** The channel carries content. First
positive content result in the programme. Does **not** license "latent
communication works" — it licenses "at this site, on this receiver, content is
detectable at the decision level."

**7.2 MANIFOLD crosses, CONTENT does not.** M5's `aligned > random` was a
disruption-magnitude artefact. This **retrospectively explains** every prior
rung's near-miss and is the outcome the disruption reading predicts.

**7.3 BOTH cross, or the doses order non-monotonically.** The factors are
**coupled, not additive** — the outcome arXiv:2608.22140 reports for its own
factorial. Registered now so it cannot be spun later: **coupling means the 2×2
did not identify the factor and the ambiguity survives.** Report it as such. Do
not present a coupled result as a content result.

**7.4 NEITHER crosses.** Consistent with M5. The likelihood arm and the
dose-response curve are still reported in full; a flat dose-response is
informative about the receiver's sensitivity even with no crossing.

**7.5 Any pair with n_disc < 30.** Uninformative for that pair. Never reported
as a null.

## 8. Mandatory co-reports — carried from M5, non-negotiable

- **Generation diagnostic before any draw**, NON-gating (ADR-045's deviation,
  approved and made permanent). Gating on task accuracy would select against the
  very confound the design excludes.
- **Full control-vs-control battery as STORED receipt fields** — 30 ordered
  pairs at six conditions. This mechanism caught coordinator error #21's shape
  on its first real draw; it is the reason M5's headline was not written up as a
  channel effect.
- **Likelihood arm** against every control with per-item sign tests. Accuracy
  alone is a **deaf** endpoint, proven three times.
- **Full wealth trajectory shape**, not just its endpoint.
- **`zerovec ≡ baseline` exact identity check** — 0W/0L on both endpoints.
- The **dose-response curve** across all admitted doses.

## 9. Known implementation hazard — fix FIRST

`common/m5.rs:139-155` maps condition index to field with a **catch-all**:

```rust
match which { 0 => q.real.0, 1 => q.base.0, 2 => q.zero.0, _ => q.rand.0 }
```

**Extending `CONDITIONS` without editing this silently maps every new condition
onto `random`**, and the battery would emit plausible, wrong numbers — the exact
failure class the battery exists to catch, hiding inside the battery. **Convert
both `correct` and `nll` to exhaustive matches before adding any condition.**

Add `six_conditions_at` **alongside** `four_conditions_at`; never edit the
latter. M4c/M4d/M4g/M4i/M5/M5X all call it, and `m3.rs:365-367` states the
house doctrine that each rung keeps its exact historical code path.

## 10. Cost and firewall

**~+15% wall clock** per draw (~37 min vs M5's ~32), no new dependency, no
second model, no VRAM change — the displacement is host-side arithmetic on a
1536-float vector. Conditions run strictly serially with `clear_kv_cache` per
prefill, so six occupy the same peak as four.

**Firewall, unchanged:** M6 is **same-model**. It tests **the apparatus, never
transfer.** Neither outcome may be cited for or against cross-model
transferability. The scope freeze in ADR-024's head and research/054 §Scope
applies unaltered — M5 closed receiver-side adaptation, and nothing here reopens
learned integration or cross-model transfer.

**Single owner.** One implementing agent, rulings recorded on the branch that
owner can read (coordinator errors #11 and #20).

---

## Outcomes

*Appended after commit. Everything above this heading is frozen; every
deviation below is a numbered coordinator error continuing M5's sequence.*

### Phase 1 — the §5 manipulation check RAN, PASSED, and killed the cell anyway

Receipt: `crates/latentmesh-runtime/receipts/run2-manifold-precheck-m6-phase1.json`
(CPU-only, annotates only, reproducible byte-identically). Implementation:
`crates/latentmesh-runtime/examples/common/displace.rs` (exact-rotation
displacement `w = ‖v‖(c·u + √(1−c²)·e)` with `e` unit and orthogonal to `u`,
per-(item, dose) ChaCha8 seeding) wired into `run2_manifold_precheck.rs`.

**The registered gate passed 4 of 4 doses, 0 dropped.** Measured cosines were
exact to six places (0.990000 / 0.950000 / 0.900000 / 0.750000, tolerance
±0.02); every typicality fell strictly inside the band `[−0.000640, 0.681357]`;
the 0.90 dose registered for the MANIFOLD primary was admitted.

**The cell was nevertheless withdrawn, on evidence beyond the gate.** That
distinction is what makes this a finding rather than a fudge: the doses did not
fail the registered criterion, and no criterion was adjusted after the fact.

### Coordinator error #24 — the registered gate could not fail

**The error is the coordinator's, not the implementing agent's.** ADR-047 §5's
gate has two arms: (a) measured cosine within ±0.02 of target, and (b)
typicality strictly between `aligned` and `random`. **They measure the same
quantity.** Arm (a) is satisfied *by construction* — `displace_to_cosine`
performs an exact rotation, so it verifies arithmetic, not an empirical risk.
And arm (b) turns out to be an affine restatement of the dose:

| dose *c* | typicality | *c* × typicality(`aligned`) | residual |
|---|---|---|---|
| 0.99 | 0.674051 | 0.674544 | **−0.000493** |
| 0.95 | 0.648818 | 0.647289 | **+0.001529** |
| 0.90 | 0.613856 | 0.613222 | **+0.000634** |
| 0.75 | 0.514531 | 0.511018 | **+0.003513** |

Typicality is `c` × typicality(`aligned`) to within 0.0035 at every dose, so
arm (b) is satisfied automatically whenever arm (a) is. A two-arm gate whose
arms are the same measurement cannot fail, and it passed 4/4 for that reason
rather than because the cell was sound.

### The structural result — the real contribution of phase 1

> **In 1536 dimensions a generic displacement direction is almost surely BOTH
> off-manifold AND content-free, so rotating toward one necessarily moves both
> factors in lockstep.**

That is why the off-diagonal "right content, wrong manifold" cell resists
construction *at all*, and it is not obvious in advance. §4 claimed the
displacement holds "content exactly" and moves one factor. It does not: the
rotation decomposes the payload into `c` × (true content) + `√(1−c²)` ×
(generic noise), so **content magnitude is scaled by `c`**. At the 0.90 dose
that is signal 0.900 against noise 0.436 — a 33% noise admixture by norm.
**§4's "content is held exactly" is FALSE as written.**

The measurements say the same thing without the geometry. Under **every**
instrument in the lens kit, a displaced payload is a partial step toward
`random`:

| | `aligned` | 0.99 | 0.95 | 0.90 | 0.75 | `random` |
|---|---|---|---|---|---|---|
| typicality vs receiver L14 | 0.6814 | 0.6741 | 0.6488 | 0.6139 | 0.5145 | −0.0006 |
| item-invariance | 0.6670 | 0.6539 | 0.6031 | 0.5390 | 0.3760 | −0.0000 |
| RMSNorm-lens entropy (nats) | 3.3199 | 3.3295 | 3.5242 | 3.6822 | 4.9180 | 5.4452 |

Monotone in every row. No measurement distinguishes a displaced vector from a
partial step along the `aligned`→`random` segment — **M5's existing diagonal,
sampled at intermediate points.** A dose-response along that segment cannot
attribute, because both factors move together. `aligned_displaced` is not a
distinct factorial cell.

**Calibration, kept because it is the load-bearing check on the other column.**
`aligned`'s typicality (0.6814) slightly **exceeds** a genuine un-pooled
receiver L14 state (0.6670). The on-manifold column is sound; the failure is
specific to the off-manifold cell, not to the instrument.

**Correction to §4.1's analytic table, verified independently.** The quantise-
dequantise cosines reproduce at absmax ≈ 2.3σ. But the "≈ 0.026" quoted for
norm-matched random in 1536-d is `1/√d = 0.0255`, the **standard deviation** of
the cosine; the **mean absolute** cosine is `√(2/(πd)) = 0.0204`. The table's
conclusion is unaffected — 4-bit is still 99.6% of the way to `aligned` — but
the figure is a σ, not a mean, and is now labelled as one.

### AMENDMENT — the `aligned_displaced` cell and the MANIFOLD primary are WITHDRAWN

- **`aligned_displaced` is dropped**, at every dose. The 2×2 of §2 collapses to
  the CONTENT axis.
- **§6.1's MANIFOLD primary at the 0.90 dose is WITHDRAWN.** M6 runs **one**
  registered e-process, not two.
- **§7.2 and §7.3 are unreachable** as written (both name the manifold arm) and
  are recorded as moot rather than reinterpreted. §7.1, §7.4 and §7.5 stand.
- **§8's battery is 20 ordered pairs at five conditions**, not 30 at six. Its
  dose-response co-report is moot.
- **§9's `six_conditions_at` is `five_conditions_at`.**

**What is NOT affected.** The CONTENT axis — `aligned` vs `mismatched`, both
on-manifold, both genuine payloads, norm-identical by the rescale rule — is
untouched by any of the above. Its power anchor (M5's measured
aligned-vs-baseline discordance, 60/50/58 at ranks 1/2/4; registered
expectation n_disc ≈ 50–60) and its n-dependent bar stand exactly as
registered. Nothing here bears on it.

### Phase 2 — implementation, in the registered order

1. **`feat/m5-receiver-lora` merged into this branch** (coordinator's call: a
   normal merge commit, no rebase, no cherry-pick, main untouched). M6 depends
   on M5's LoRA runtime, adapter loader and receipts. All **125** committed
   receipt blobs are byte-identical across the merge; the only conflict was
   `docs/adr/README.md`, where one side adds the 047 row and the other rewrites
   the 045 row — both preserved verbatim.
2. **ADR-047 §9's landmine is closed.** `common/m5.rs`'s `correct` and `nll`
   were `match which { 0 => …, _ => q.rand.… }`; both are now exhaustive
   matches on a four-armed `Cond` enum, with the receipt-key order pinned by a
   test. M6's five-armed set is a separate type, so widening either fails to
   compile rather than aliasing a new condition onto `random`.
3. **`five_conditions_at` is added ALONGSIDE `four_conditions_at`**, in
   `common/m6.rs`. `m3.rs` is not edited; M4c/M4d/M4g/M4i/M5/M5X keep their
   exact historical code path (`m3.rs:365-367`'s house doctrine).
4. **`mismatched` priming and carry-forward ported from run 3**
   (`run3_gated_text_probe.rs:313-317, 361-363, 516-517`): primed from the last
   eligible index, asserted outside the drawn stream, then carried forward
   *after* each item is evaluated, so no lookahead exists and early stopping
   stays honest. Norm-matching is free — M5's rescale rule already sends every
   injected vector to the receiver's natural inject-block median — and all
   three realised norms are stored per item so a reader can verify it.

### OPEN REGISTRATION QUESTION — which receiver M6 runs on

**ADR-047 never names it.** The evidence points at an M5-adapted receiver: the
power model is anchored entirely on M5's measured battery, §10 compares wall
clock to M5's, and phase 2's ordered steps include a transfer check, which only
exists for an adapter. But **M5 produced three adapted receivers and the frozen
text names no rank.**

The probe therefore **requires the rank as an explicit argument and refuses to
default**, because a silent default would make an unregistered choice look like
a registered one. **Recommendation: rank 2** — §1's motivating case is rank 2's
46W/23L, which is the specific ambiguity M6 exists to resolve. Awaiting the
coordinator's ruling; it is a registration gap, not an implementation choice.

**On the transfer check (phase-2 step 5).** It measures the composed→fused BF16
agreement of *the adapter*, and M6 changes no adapter — so re-running it would
reproduce M5's numbers under a name that would overwrite a frozen M5 receipt.
The probe **reads and gates on M5's committed transfer receipt** for the named
rank, and carries its mandatory NON-gating generation diagnostic into the M6
receipt so a reader of the draw alone can tell "the receiver stopped answering"
from "the channel carries nothing".

### PROPOSED SUCCESSOR (not implemented, not folded into M6) — the subspace redesign

The off-diagonal cell is unreachable by *generic* displacement, but not
unreachable in principle. The fix is to stop displacing into a generic
direction and start displacing **within a chosen subspace**:

- **Arm A** — displace within the receiver's local L14 subspace (its top
  principal directions at that block).
- **Arm B** — displace **orthogonal** to that subspace, at **matched
  magnitude**.

Both arms attenuate true content by the same factor, so content is held equal
*by construction* rather than by assertion; the only difference between them is
conformity to the receiver's own activation geometry. **That isolates the
manifold axis**, which is exactly what phase 1 showed a generic rotation cannot
do.

**This is a NEW REGISTRATION, not an amendment.** It changes the manipulation,
its analytic model and its manipulation check, so it needs its own
pre-registration with its own power calculation. It is recorded here as a
proposed successor and is **not** implemented, not run, and not folded into M6.

### Coordinator error #25 — the receiver is not named; rank 2 RULED

**Declared before the draw, not after it.** Raised by the implementing agent as an
open question, ruled by the coordinator, and recorded as **the coordinator's
error**: a pre-registration that meant "adapted" had to name a rank, because
M5's three receivers are three different experiments.

ADR-047 never states which receiver M6 runs on. Three things in the frozen text
point at an **M5-adapted** one: §6's power model is anchored entirely on M5's
measured battery, §10 compares wall clock to M5's ~32 min, and the registered
phase order includes a transfer check, which exists only for an adapter. But
**M5 produced three adapted receivers and the frozen text names no rank** — a
pre-registration that meant "adapted" would have had to name one, because M5's
three receivers are three different experiments.

**This draw runs on the M5 rank-2 adapted receiver.** The reason is §1: the
motivating case is rank 2's 46W/23L, and that specific number is the ambiguity
M6 exists to resolve. Resolving it on a different receiver would answer a
question nobody asked.

**A caveat the implementing agent first stated too strongly, corrected here
rather than quietly dropped.** The claim offered was that ADR-045's correction
#23 found the adapted receiver "not content-specific at all", and that a null
here is therefore *weaker evidence* against content transmission than the same
null on a frozen receiver. **That overstates #23, and the receipts say so:**

| M5 rank | `aligned` vs `random`, **NLL** | same pair, **ACCURACY** |
|---|---|---|
| 1 | 147/153, p = 0.657 | 42/35, p = 0.2472 |
| 2 | 156/144, p = 0.263 | **46/23, p = 0.0038** |
| 4 | 146/154, p = 0.698 | **42/25, p = 0.0249** |

**#23 rests entirely on the NLL arm — it is a statement about likelihood.** The
decision endpoint on those same pairs is not null at all; it is where 46/23
lives. And the decision arm's *content* sensitivity has never been isolated on
**any** receiver, frozen or adapted, because `random` conflates content with
manifold conformity. **That conflation is the entire reason M6 exists.** So the
inference does not hold: rank 2 is not disposed toward "no". It tests a
decision-level question that is open on both receivers, using the first control
that can separate the factors.

**The honest form of the concern, which does stand:** the adapted receiver's
likelihood arm showed no content specificity across three ranks, so *if that
generalises to decisions*, a null here would be unsurprising. That is a
hypothesis about generalisation, not an established property, and it is **not**
grounds for attaching "weaker evidence" to §7.4.

### Registered BEFORE THE RESULT WAS SEEN — the likelihood prediction

**Precise provenance, because the weaker true statement is the one worth
making:** the draw was already running when the coordinator sent this
prediction and when it was committed here. It was therefore registered *before
any output of the draw had been read* — not before the draw was launched. No
line of the result had been seen by the implementing agent or the coordinator
at commit time, so it cannot have been fitted to the outcome; but the honest
label is "before the result", not "before the draw".

If correction #23 generalises, **the LIKELIHOOD arm should stay null in M6
too.** Both endpoints are co-reported:

- **Decision crosses while likelihood stays null** → a **dissociation**, and a
  real finding.
- **Both stay null** → consistent with M5, and it closes the content axis on
  this receiver.
- **Likelihood moves while decision does not** → the M4i/M5X shape again, and
  it would mean `mismatched` behaves like `random` on likelihood, which is
  itself informative about what the payload moves.

The probe **refuses to default**: the rank is a required argument, because a
silent default would make an unregistered choice look like a registered one.
The receipt names the receiver in its headline fields. A draw at a different rank or on the frozen receiver would be a
separate draw with a separate comparator, reported separately and never pooled
— the M5 precedent, not a retry.

### Transfer check (phase-2 step 6, before the draw) — PASSED, no degeneration

Read from M5's committed `run2-m5-transfer-receipt-cellL18toL14-r2.json` rather
than regenerated: the check measures the composed→fused BF16 agreement of *the
adapter*, and M6 changes no adapter, so re-running it would reproduce the same
numbers under a name that would overwrite a frozen M5 receipt.

- `gate_pass`: **true**. Fused training-target CE **0.5043** adapter-on vs
  **0.7376** off, over 510 holdout items.
- **Generation diagnostic (NON-gating, ADR-045 error #22, mandatory):**
  accuracy **39/64 adapter-on vs 31/64 off** — the adapter *improves* the
  decision endpoint. Mean generated length 241.2 vs 840.8 chars.

**No degeneration.** The 3.4× length reduction is the caveat already recorded
against M5's draws and is not itself a failure signal: length does not predict
decision-side failure, accuracy does, and accuracy is up. The draw proceeds.

---

## OUTCOME — the draw, at rank 2

Receipt: `crates/latentmesh-runtime/receipts/run2-m6-receipt-cellL18toL14-loraR2-contentaxis-questiontail-eprocess.json`.
300 items drawn, 0 degenerate, ~35 min wall clock.

### The verdict: a POWERED NULL on the registered primary

| field | value |
|---|---|
| `final_wealth` | **3.8153** (peak 5.8980 at order 181) |
| `wealth_threshold` | 20.0 |
| `crossed` | **false** |
| `n_discordant` | **55** (34W / 21L) |
| wins needed at n = 55 | **40**, recomputed from λ and α |
| `uninformative` (n_disc < 30) | **false** |

`aligned` did not beat `mismatched` on the decision endpoint by the registered
rule. n_disc = 55 sits inside §6's registered expectation of 50–60, so the power
model held and this is a **powered null, not a failure to measure**. Per §7.4
this is consistent with M5 and closes the content axis **on this receiver**.

`zerovec ≡ baseline` on all 300 items: 0 accuracy disagreements, 300 bit-
identical NLLs, max |ΔNLL| exactly 0.0.

### The determinism result — M6 reproduces M5 rank 2 EXACTLY

Stated as a finding rather than left implicit, because it is what licenses
reading the two draws together. Across **all 12 battery pairs the two rungs
share**, plus every shared accuracy field, **every integer is identical**:

| | M6 | M5 R2 |
|---|---|---|
| accuracy `aligned` / `baseline` / `zerovec` / `random` | 154 / 172 / 172 / 131 | 154 / 172 / 172 / 131 |
| `aligned` vs `random`, accuracy | 46/23 | 46/23 |
| `aligned` vs `random`, NLL | 156/144 | 156/144 |
| `aligned` vs `baseline`, NLL | 202/98 | 202/98 |
| `random` vs `baseline`, NLL | 187/113 | 187/113 |

**The only difference between the two draws is the added condition.** This is
the cleanest cross-rung comparison the programme has produced: the apparatus is
deterministic to the integer, so any difference between M5 and M6 is
attributable to the control set and to nothing else.

### THE REGISTERED PREDICTION IS FALSIFIED — and that is the headline

Registered before the result was read: *if correction #23 generalises, the
LIKELIHOOD arm should stay null in M6 too.*

**It did not.**

| NLL contrast | split | one-sided p |
|---|---|---|
| `aligned` vs **`mismatched`** | **181/119** | **2.05e-4** |
| `aligned` vs `random` | 156/144 | 0.263 |

The likelihood arm **is** sensitive to the content contrast. It needed an
**on-manifold** control to show it. Against `random` — off-manifold and
content-free at once — the same 300 items give the same null M5 reported.

A registered prediction that fails is worth more than one that quietly
succeeds, and it is recorded here as failed.

### The caution, at equal prominence: the NLL picture is NOT internally clean

It must not be narrated as a tidy content result.

**Every injected condition improves likelihood over `baseline`, and `random`
improves it MORE than `mismatched` does** — on both the item counts and the
means:

| vs `baseline` | NLL split | p | mean NLL |
|---|---|---|---|
| `aligned` | 202/98 | 9.51e-10 | 3.8142 |
| `random` | 187/113 | 1.14e-5 | 3.8295 |
| `mismatched` | 176/124 | 1.59e-3 | 3.8901 |
| (`baseline`) | — | — | 3.9316 |

**And the three pairwise contrasts do not compose**: `aligned` > `mismatched`
(p = 2.05e-4), `aligned` ≈ `random` (p = 0.263), `random` ≈ `mismatched`
(157/143, p = 0.227). If content were straightforwardly the driver, `aligned`
should beat `random` too. It does not.

**One clarification, so a reader does not over-read the non-transitivity.**
Paired sign tests are *not* required to be transitive — they count items, not
magnitudes — so non-transitivity is not by itself evidence of a defect. The
substantive anomaly is narrower and survives that caveat: **`random` is better
for likelihood than `mismatched` on the mean as well as on the count**
(3.8295 vs 3.8901; 157/143). A content-only account does not predict that.

### HYPOTHESIS GENERATED BY THIS DATA, NOT ESTABLISHED BY IT

There is a reading that reconciles all of the above: **off-manifold payloads
produce a larger generic likelihood shift**, so `random` receives a boost that
offsets `aligned`'s content advantage and cancels that contrast to null, while
`mismatched`, being on-manifold, receives no such boost and the content
advantage becomes visible. That would explain correction #23's null and M6's
result with one mechanism.

**It is a hypothesis this data generated, not a result this data establishes**,
and it must never be cited as the latter. It is testable: it predicts that
**generic likelihood shift scales with off-manifold-ness**, which the phase-1
dose ladder could measure directly — the displaced payloads are already
constructed, already reproducible, and span exactly that axis. Recorded as a
candidate successor. **Not implemented.**

### The accuracy endpoint, stated precisely

`aligned` vs `mismatched` on accuracy is **34/21, n_disc 55, p = 0.0524**. That
number is written here without a qualifier: it is not "marginal" and not
"nearly significant". **The registered e-process is the stricter and only
registered authority, and it did not cross.**

The accuracy ordering is `baseline` 172 > `aligned` 154 > `mismatched` 141 >
`random` 131 — a decomposition in which content accounts for the 154-vs-141 gap
and manifold conformity for the 141-vs-131 gap. **This decomposition is
DESCRIPTIVE, not inferential**: the manifold contrast (`mismatched` vs
`random`, 42/32) has p = 0.148 and is not significant.

Note also that `baseline` still leads. Every injection continues to hurt the
decision endpoint, exactly as in M5.

### What M6 settles, and what it does not

**Settles:** on this receiver, at this site, the content contrast does not move
the decision endpoint by the registered rule, on a powered test. And the
apparatus is deterministic to the integer across rungs.

**Does not settle:** whether the likelihood effect is content-specific in
general. M6 shows correction #23's null was a **control artefact** rather than a
property of the receiver — but the off-manifold-inflation reading is unproven,
and until it is tested the likelihood picture stays ambiguous.

**Unchanged:** the firewall of §10. M6 is same-model and tests the apparatus,
never transfer.

### Successors still standing (neither implemented)

1. **The subspace redesign** (from phase 1) — displace WITHIN the receiver's
   local L14 subspace versus ORTHOGONAL to it at matched magnitude. Both arms
   attenuate content equally, so the contrast is conformity alone. New
   registration.
2. **The off-manifold-inflation test** (from this draw) — does generic
   likelihood shift scale with off-manifold-ness? The phase-1 dose ladder is
   the instrument. New registration.
