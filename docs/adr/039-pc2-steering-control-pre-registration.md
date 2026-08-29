# 039. PC2: the steering control — pre-registration

- **Status**: Proposed (pre-registration — written before any item is drawn)
- **Date**: 2026-08-29
- **Gates**: [ADR-037](037-m5x-maximal-configuration-rung.md) (M5X) and
  [ADR-035](035-m4b-scale-control-pre-registration.md) (M4b) remain **BLOCKED**
  behind this rung
- **Protocol**: [ADR-036](036-successor-rung-evaluation-protocol.md) e-process,
  unchanged · **Publication**: [ADR-032](032-negative-result-publication-contract.md)

## Context — why another control

[PC1b](024-run2-trained-thought-adapter-ladder.md) failed at the ladder's
highest power (n_disc 64, wealth 0.1949, never above its 1.0 start). It
established a **dissociation**, not an absence:

- **Likelihood endpoint — VALIDATED.** The payload beats a *norm-matched*
  Gaussian through the same operator at the same 8 positions by **0.237 nats
  on 198/300 items** (p ≈ 2e-8). The channel demonstrably carries
  **content-dependent** information.
- **Accuracy endpoint — UNVALIDATED.** Accuracy went 140 → **127**. **No
  control in this repository has ever moved accuracy.**

Every rung's verdict rests on the accuracy endpoint. So for the endpoint that
decides every result, the apparatus cannot distinguish *"no effect"* from
*"cannot detect effects"*, and **the method remains the leading explanation for
every null in the ladder.**

**PC1 and PC1b are both *restoration* controls**: they hand the receiver the
right answer and ask whether it gets more answers right. That design has a
weakness we can no longer ignore — a restoration null is consistent with *"the
pathway cannot steer decisions"* **and** with *"the receiver's errors are not
of a kind that the gold state repairs."* It cannot separate them.

**Nothing here has tested steering.**

## Decision — PC2, a steering control

Ask the strictly easier question: **can the apparatus move the receiver's
answer at all — in any direction, including a wrong one?**

**Design.** For each item, teacher-force the receiver over its own question and
a solution **whose final numeric answer has been replaced by a decoy `d`**,
then tap block 19 at the last token — *identical derivation to PC1b in every
respect except the answer value.* Inject at the same question-tail site,
same operator (`Fuse`), same 8 slots.

**Primary endpoint**: the rate at which the receiver **emits the decoy `d`**,
`steer` vs `random`, paired per item, under ADR-036's e-process (λ = 0.30,
α = 0.05, wealth threshold 20.0, N_max = 300, `adaptation-512` fixed order,
ADR-024's 13-item exclusion).

**Conditions** (mirroring PC1b so the two are directly comparable):

| condition | payload |
|---|---|
| `steer` | receiver's L19 last-token state, teacher-forced over the **decoy** solution |
| `restore` | PC1b's payload — teacher-forced over the **gold** solution |
| `baseline` | no injection |
| `zerovec` | `h += 0` — must be bit-identical to baseline (operator check) |
| `random` | per-item seeded Gaussian, **norm-matched** to the effective steer vector |

### Why this is the right test, and why it is *easier* than PC1b

1. **It removes the confound restoration cannot.** A decoy hit is not something
   the receiver would produce for independent reasons; **`random` hits `d` at
   chance by construction**, giving a clean floor.
2. **Its effect ceiling is far higher.** Restoration is bounded by the 160/300
   items the receiver already gets wrong; steering can move **any** item.
3. **It varies exactly one thing from PC1b** — the answer value. Same site,
   same operator, same slots, same derivation, same stream. Any difference is
   attributable to what the payload *says*, not to how it is delivered.

### Decoy construction (pre-committed, to foreclose gaming)

`d` is derived deterministically from the gold answer `g` — **not** sampled,
and **never** equal to `g`: a fixed per-item ChaCha8 stream (seed `0x5732`)
selects a perturbation from `{g+1, g-1, g+10, 2g}`, re-drawing on collision
with `g` or on a non-positive result. Decoys are committed to the capture
receipt **before** the draw. A separate check records the rate at which
`baseline` already emits `d` (**expected ≈ 0**); if that exceeds 2%, the decoy
construction is declared leaky and the rung is void.

## Pre-registered interpretation — both branches, before any draw

**PASS** (steer moves answers toward `d` with real power) — **the apparatus can
steer decisions.** PC1b's failure is then re-read as *restoration being the
wrong probe*, not as a dead pathway. **M5X and M4b UNBLOCK**, and the ladder's
cross-model nulls become evidence about **transfer**, because the method has
finally been shown capable on the endpoint every verdict uses.

**FAIL with real power** — **the apparatus cannot move a decision by any
means**, not even toward an answer it was explicitly handed. Combined with
PC1b, that is decisive about the **method**: this injection paradigm is
**decision-inert while remaining likelihood-live**. Consequences, accepted in
advance:

- **M5X and M4b stay blocked permanently** under this apparatus — they vary
  the payload, and the payload is not the binding constraint.
- **The ladder closes.** Every cross-model null is explained by the method, and
  none of them is evidence about latent transferability.
- **This becomes the publishable result** — a precise, powered negative about
  activation-injection as a mechanism for cross-model reasoning transfer, with
  the likelihood/decision dissociation as its central finding. Per ADR-032 it
  is reported **without softening**.

**Underpowered** (n_disc too small to separate the arms) — reported as
uninformative. **It is not spun as either outcome**, and the power floor is
stated on its own scale per ADR-036 Decision 3.

## Firewall — inherited and extended

PC2 is **same-model, same-item, identity-transform** with gold-adjacent
content. It tests **the apparatus, never transfer**. A PASS proves the pathway
can steer a decision; it says **nothing** about whether a cross-model,
learned-alignment payload can carry reasoning. **A FAIL may not be cited as
transfer evidence either** — the symmetric rule added after PC1b.

## ⛔ Do NOT inherit PC1b's FAIL-branch wording

PC1b's registered FAIL branch said a powered failure would make the ladder
nulls *"evidence about TRANSFER rather than about plumbing."* **That is
logically inverted** and is quarantined at the head of ADR-024. A failed
positive control makes the **method** the leading explanation; ruling plumbing
out requires a **PASS**. The branches above are written to avoid re-inheriting
it.

## Single-owner rule (ADR-034, reinforced)

**This rung has exactly one implementing agent.** Coordinator error #11 put two
agents on PC1b, producing a duplicate process, concurrent edits to the running
example's source, and message identities that could not be told apart. **Claim
the rung before launching; do not resume an agent and start a workflow for the
same work.**

## AMENDMENT — the 2% leakage gate's *rationale* is wrong (written mid-draw, outcome unknown)

**Timestamp discipline**: written while the draw was at **item 9 of 300**, with
`steer` and `random` both at 1 hit and the primary undecided. I do not know the
outcome. Recording it later would be indistinguishable from protocol-shopping,
which [ADR-032](032-negative-result-publication-contract.md) forbids.

**The registered claim was**: *"`random` hits the decoy at chance by
construction, giving a clean floor."* **That is false, and the decoy rule above
is why.**

Decoys are drawn from `{g+1, g-1, g+10, 2g}` — precisely the space of **natural
arithmetic slips**: off-by-one, off-by-ten, and forgetting to halve. A model
that errs on a GSM8K item does not land uniformly over the integers; it lands
disproportionately on exactly these. So the baseline decoy-emission rate is
**structurally above chance**, and the observed early hit — item 20, gold 38,
decoy 48 (`g+10`), where **all five conditions** including `baseline` and
`zerovec` emitted 48 — is the predicted behaviour of a well-formed decoy set,
not a bug. The detector is sound: it compares the **extracted final answer**
under numeric equality (`extract_answer` → `answers_equal`), not a substring.

**What this does and does not invalidate:**

- **The primary comparison stands.** It is **paired `steer` vs `random`**, and
  both arms carry the identical baseline propensity toward natural slips. If
  `steer` reaches the decoy *more often than* `random`, that difference is
  attributable to the payload's content. The floor being elevated costs
  **power**, not validity.
- **The 2% threshold is miscalibrated**, because it was derived from the false
  "chance floor" premise. A baseline rate of, say, 8% would reflect the model's
  natural error distribution — not a leaky decoy construction.

**The gate is NOT being relaxed.** It was pre-registered and it will be
**reported exactly as it falls**: if baseline decoy-emission exceeds 2%, that
is recorded as *the registered gate tripping*. What this amendment changes is
only the **interpretation** of a trip — from *"the decoy construction is leaky
and the rung is void"* to *"the registered threshold was derived from a false
premise about the floor."*

**Deciding which reading applies is deferred to a stated, pre-committed test**,
so it cannot be chosen to suit the result: the rung is void **only if
`baseline` and `steer` are statistically indistinguishable on decoy-emission**
— i.e. the payload adds nothing over the model's own error propensity. If
`steer` separates from `random` while baseline is merely elevated, the rung is
**valid with reduced power** and is reported that way.

**A successor rung should use decoys drawn AWAY from natural-slip space** (a
random integer of similar magnitude, not an arithmetic perturbation of `g`) to
recover a genuine chance floor. Registered as a design note for PC3, not as a
change to this rung.
