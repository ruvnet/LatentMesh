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
