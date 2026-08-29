# Research: break-even economics of latent vs. text, and the causal gate's standalone value

* Purpose: two questions that determine what LatentMesh is worth even if latent transfer keeps
  failing (ADR-023 S6, ADR-024 M3/M4 — three adversarially-verified nulls so far). Q1: does a
  working latent channel actually beat "send a text summary" once real transport costs are
  counted, and under what conditions. Q2: does the five-control causal gate (ADR-003) have value
  on its own, applied to plain text, independent of whether latent transfer ever works.
* Date: 2026-08-28/29.
* Method: Q1(a) and the cost estimate in Q2 are computed directly from committed receipts in
  `crates/latentmesh-runtime/receipts/` — every number is a wall-clock or byte figure read from a
  named receipt field, not an estimate, unless explicitly labeled otherwise. Q1(b) and Q2's
  literature survey use WebSearch/WebFetch, evidence-graded **primary** (paper/repo fetched and
  the claim verified in the source text), **inferred** (search-summary only), **uncertain**
  (could not be verified), matching `docs/research/028`'s convention.
* Read for context: ADR-002, 003, 009, 010, 019, 023, 024; `docs/research/025`, `028`.

---

## Q1 — the break-even economics of latent vs. text

### 1a. This repo's own receipted costs

**Two different "latent" objects exist in this codebase and must not be conflated in a
break-even calculation:**

1. **The dense pooled residual-stream vector** (ADR-002 `LatentFrame`, the object ADR-023/024's
   live experiments actually transfer) — one float vector per message, `dim=2048` at the sender
   capture point / `dim=1536` at the receiver injection point (`s0-receipt.json config`,
   `s2-calibration-receipt.json`).
2. **LatentMesh Air's compact semantic delta** (ADR-010/019, the radio-tier object) — typed field
   updates and quantized residual slots inside a ≤256-byte frame, deliberately **not** the same
   object as (1). ADR-010 states this outright: the `LatentFrame` benchmark's ~64.1 KiB payload
   (16×4096 Int8) "is useful between machines with ordinary network links but is the wrong unit
   for a 300 bit per second radio" — Air exists specifically because (1) doesn't fit on radio, not
   as an alternative encoding of the same content.

This means "does latent beat text" has a different answer per transport tier, and the two tiers
use physically different LatentMesh mechanisms, not the same mechanism at two different
bandwidths.

**Compute cost, GPU-local tier — measured, not estimated.** `s2c-generated-dump-receipt.json`
records, for the same 2,560-item GSM8K set, two disjoint phases over the identical
719,115-token generated span:

| Phase | Model | Wall-clock | Tokens | ms/token |
|---|---|---:|---:|---:|
| `sender_phase` (greedy decode, generating the reasoning) | Qwen2.5-3B | 5,252.82 s | 719,115 | **7.305** |
| `receiver_phase` (teacher-forced prefill, re-encoding the same span) | Qwen2.5-1.5B | 57.30 s | 719,115 | **0.0797** |

Decode costs **≈92× more per token than prefill**, on this GPU (RTX 5080), for this model pair —
a directly receipted ratio, not the standard "decode is memory-bound, prefill is compute-bound"
folklore asserted without a number. `s2-dump-receipt.json`'s independent gold-text prefill pass
(4,000 items, 848,683 tokens, sender 99.06 s / receiver 57.48 s ⇒ 0.117 ms/token and 0.068
ms/token respectively) cross-checks the prefill-side figure within the same order of magnitude.

**This is the mechanism, receipted:** a text handoff costs the sender one full decode pass over
the message (N tokens × ~7.3 ms) *plus* the receiver one prefill pass over the same message (N ×
~0.08 ms) — dominated by decode, ≈7.4 ms/token total. A latent handoff, if it reuses a hidden
state the sender already computed while producing its own answer (the design ADR-023/024 actually
implement — the pooled vector is captured off the sender's existing forward pass, not generated
freshly for transmission), costs one cached-matrix `apply()` call (ADR-002: cached projection
matrix, sub-millisecond, `O(dim²)` once, independent of message length) plus injection overhead —
effectively **flat, not `O(N)`**. For any message longer than a token or two, the latent path is
compute-cheaper at this tier, by construction, provided (and this is exactly what run 1 falsified
for the linear case) the transferred state actually carries the content.

**Byte cost, GPU-local / high-bandwidth-network tier: essentially irrelevant.** At LAN/localhost
speeds a 2–4 KB payload difference is sub-millisecond; compute dominates the tier's break-even
entirely, per the numbers above.

**Byte cost, LoRa/radio tier — where the two objects diverge sharply.** Using
`generated_tokens_mean = 280.9` (receipted, `s2c-generated-dump-receipt.json`) as a stand-in
message length, and a standard ≈4 bytes/token English-text approximation (not a repo measurement —
flagged as an estimate):

| Object | Size | Meshtastic packets (211 usable B/packet, ADR-019) | Time @ 300 bps (illustrative, ADR-010's own rate) |
|---|---:|---:|---:|
| Text summary (~281 tokens) | ≈1,124 B | 6 | ≈30 s |
| Dense `LatentFrame`, Int8, dim 2048 + header (est., no committed receipt for header bytes) | ≈2,250 B | 11 | ≈60 s |
| LatentMesh Air compact semantic delta, one signed field update (ADR-019, receipted) | ≈186 B | **1** | ≈5 s |

**The dense pooled vector is *larger* than the text it stands in for, at this message length** —
2,048 Int8 bytes (plus header) versus ~1,124 bytes of the actual reasoning text it replaces.
Text wins the byte race at radio tier for a 281-token message; the vector only wins once message
length exceeds roughly the vector's own byte footprint divided by the text bytes/token ratio
(≈2,048/4 ≈ 512 tokens for Int8, ≈4,096/4 ≈ 1,024 tokens for F16) — i.e., only for unusually long
messages. This is *exactly* why LatentMesh's own radio design doesn't send the dense vector over
Air at all: Air's compact symbolic delta beats both, but it carries a small typed fact update, not
a full reasoning trace — a different content granularity than what run 1's dense-vector experiment
tests. Comparing "latent vs. text on radio" honestly requires stating which of the two latent
objects is meant; conflating them overstates latent's radio-tier case.

### 1b. What C2C / LatentMAS report, and where the savings actually come from

- **Cache-to-Cache (C2C)**, arXiv:2510.03215 (ICLR 2026, accepted) — **primary**, abstract and
  HTML body fetched. Reports **2.0–2.5× latency speedup** over text-to-text communication (Table 3
  numbers: C2C ≈0.27–0.50 s/query vs. text-to-text ≈0.75–7.54 s/query, A100, batch 1) and 3.1–5.4%
  accuracy improvement over text. **Source of the speedup, per the paper's own text**: "eliminating
  intermediate text generation" — avoiding "exhaustive, token-by-token decoding of contextual
  explanations." The paper explicitly names decode avoidance, not prefill avoidance, matching this
  repo's own measured 92× decode/prefill cost ratio above. The paper does **not** report cache
  transfer size in bytes and does not formally analyze bandwidth cost — its efficiency claim is a
  wall-clock number on a fixed GPU/network topology, not a bytes-transferred argument. This is an
  important asymmetry for anyone citing C2C's speedup as evidence for a radio-tier bandwidth
  argument: it isn't one.
- **`docs/research/028`** (already in-repo, primary re-verification not repeated here) established
  that C2C's fuser trains on **task loss** (receiver's own next-token cross-entropy, end-to-end),
  not reconstruction loss — separately relevant to Q1 only insofar as it means C2C's reported
  numbers are for a *working* transfer method; LatentMesh's own linear/MLP nulls (ADR-023 S6,
  ADR-024 M3/M4) have not yet reached a state where this repo's own wall-clock comparison could be
  run end-to-end, only the compute *mechanism* (§1a above) can be argued from first principles plus
  our own decode/prefill measurements.
- **LatentMAS**, arXiv:2511.20639 — training-free, exposes sender KV-cache across prefill+decode
  directly to the receiver, sidestepping the loss-type question entirely (per `docs/research/028`
  §2, already primary-verified in-repo).

**Bottom line for Q1(b)**: every reported latent-communication speedup found in this pass, and the
one this repo's own receipts independently support with a measured ratio, comes from **avoiding
the sender's decode pass**, not from smaller wire payloads. This is a compute-latency claim, valid
at low-latency/high-bandwidth transport, and it does **not** transfer to the radio tier where the
scarce resource is bytes, not GPU-seconds — a distinction none of the surveyed sources state
explicitly, because none of them discuss radio-constrained transport at all.

### 1c. Break-even inequalities, stated explicitly

Let `N` = message length in tokens, `T_dec ≈ 7.3` ms/token (receipted, this repo, 3B model),
`T_pre ≈ 0.08–0.12` ms/token (receipted), `B_txt ≈ 4N` bytes (estimated), `B_lat` = latent
payload bytes (fixed, independent of `N`: ≈2,048 B Int8 / ≈4,096 B F16 for `dim=2048`, or ≈186 B
for an Air compact delta), `R` = link rate in bytes/s.

- **Compute-bound tier (GPU-local, LAN, most internet paths)**: latent wins whenever
  `N · (T_dec + T_pre) > T_apply`, i.e. essentially always once `N > 1`, **conditional on** the
  transferred state being reusable from the sender's own computation (no extra decode incurred to
  produce it) **and** carrying the causal content the receiver needs — the second condition is
  exactly what ADR-023/024's three nulls have not yet demonstrated for any tried architecture.
- **Bandwidth-bound tier (LoRa, other sub-kbps radio)**: latent (dense-vector form) wins only when
  `B_lat < B_txt`, i.e. `N > B_lat / 4` — roughly 512 tokens (Int8) or 1,024 tokens (F16) at the
  4-bytes/token estimate above. Below that message length, text wins the byte race outright, and a
  working latent channel would need to beat text on *quality* alone to justify the switch, not on
  bandwidth. LatentMesh Air's compact-delta form sidesteps this by carrying much less content per
  message (a typed fact, not a reasoning trace) — it wins bytes unconditionally but isn't
  interchangeable with the dense-vector content the run-1/run-2 experiments test.

**What latent transfer must achieve to justify itself, by tier:**

| Tier | The condition latent must clear | Status |
|---|---|---|
| GPU-local / LAN | Carry *any* nonzero causal signal (compute case is already won by ≈92× on decode avoidance) | **Not yet demonstrated** — 3 adversarially-verified nulls (linear map, MLP, FastGRNN) |
| Internet / agentbbs store-and-forward | Same causal-signal bar, plus payload ≤ what the relay's rate-limit tolerates (agentbbs specifics not receipted in this pass — flagged as unresolved) | Not evaluated |
| LoRa / Meshtastic, dense-vector form | Causal signal **and** message length > ≈512–1,024 tokens (Int8/F16) for the byte case to favor latent at all | **Doubly unmet** — not causally useful yet, and typical GSM8K-scale messages (≈281 tokens) are already byte-favorable to text at this dim |
| LoRa / Meshtastic, Air compact-delta form | Causal signal at fact-update granularity (a different, smaller claim than the dense-vector experiments test) | Separate, untested claim — Air's mechanics are simulation-validated (ADR-019) but no causal-value claim has been made for its content yet |

---

## Q2 — the causal gate's standalone value, applied to text

### Literature check: has anyone published causally-verified TEXT agent-to-agent communication with decoy controls?

Searched multi-agent-debate/collaboration ablation literature, LLM-routing-with-verification, and
MANTA-adjacent ablation practice (2025-2026, arXiv-weighted).

- **"Agents that Matter: Optimizing Multi-Agent LLMs via Removal-Based Attribution"**
  (arXiv:2605.27621) — **primary**, fetched. Closest text-communication attribution work found.
  Uses Leave-One-Out **agent** removal and model substitution to attribute contribution; explicitly
  contrasts real ablation against "introspective LLM judges" (agents reasoning in-context about a
  counterfactual absence) and finds the introspective version unfaithful. **Does not** use a
  decoy/control design at the *message* level — no random-content, mismatched-task, or
  self-generated-feedback control substituted in place of a real message while holding the agent
  present. Its causal claim is "this agent's presence/absence changes the outcome," not "this
  specific message content — versus content-matched noise — changes the outcome."
- **"Do Latent Channels Actually Communicate? A Causal Audit of Latent Multi-Agent LLM
  Communication"** (Zhang & Emu, arXiv:2607.26773) — **primary**, HTML body fetched directly and
  checked for this question specifically. This is the paper ADR-023's own methodology mirrors
  (OPE/OME/CAG/SSG vocabulary). Confirmed by direct quote: "We introduce a causal audit that
  applies controlled message replacements at the boundary where the sender-produced representation
  enters the receiver," and the audit's five message settings apply **exclusively to continuous
  internal representations (embeddings, hidden states, KV caches)**. Text communication appears
  only as a **motivating comparison** ("text communication requires an agent to project its
  internal computation into discrete token sequences") — never as a channel subjected to the same
  intervention methodology. This is the single closest paper to LatentMesh's own ADR-003
  machinery, and it explicitly does not extend the test to text.
- Broader search (multi-agent debate causal-influence metrics, MANTA-adjacent value-aware pruning,
  policy-parameterized dialogue ablations, counterfactual-graph calibration) surfaced causal-
  influence-of-communication metrics and removal-based attribution, but nothing found tests a text
  message's *specific content* against a decoy distribution (zero / random-text / mismatched-task
  text / self-generated-feedback text) using a pre-registered nonparametric test analogous to
  ADR-003's permutation test. No source found combines: (a) text as the channel, (b) content-level
  (not agent-level) controls, (c) a formal significance test against multiple decoy conditions.

**Verdict: genuinely unclaimed territory, at reasonable confidence given the search depth here.**
The five-control causal-verification methodology (ADR-003) has been applied, by the closest primary
source found, only to latent channels. No paper found runs the same decoy-controlled test on plain
text agent-to-agent messages. This matters because ADR-003's own text notes the gate "composes with
ANY channel including plain text" — the gate's value doesn't depend on latent transfer working at
all, and this literature check found no prior claim to that specific, channel-agnostic result.

### The minimum experiment

ADR-023's own machinery already implements everything needed **except** the text-decoy conditions
themselves — no new models, no alignment transform (S2 is skipped entirely), no trained adapter
(M3-M5 are skipped entirely). Replace the two latent conditions with text-decoy conditions in the
existing four-condition frame:

| Condition | Channel | Controls tested |
|---|---|---|
| `StaticText` (unchanged from ADR-023) | Text | none — cost/quality anchor |
| `StaticText+Gate` | Text | ADR-003's four applicable controls (drop `text_equivalent` — meaningless when the channel is already text): `zero` (no message), `random` (random token sequence, matched length), `mismatched` (another episode's real message), `self_generated` (receiver's own prior output fed back) |
| `CausalDynamicText` | Text | Same four controls, used as the Darwin fitness signal (ADR-006/009), mirroring `CausalDynamicLatent`'s role in ADR-023 |

**Cost estimate, from this repo's own receipted rates (§1a):** at the frozen 40-item probe scale
ADR-023/024 already use (S1a, S2b), each item needs one sender generation (real message, reused
across all controls except `random`/`zero` which need none) plus one receiver generation per
condition (real + 4 controls = 5 receiver decodes). Using the receipted 7.3 ms/token sender decode
and treating the 1.5B receiver's decode rate as the same order of magnitude as its own already-
measured 4.27 s/generation-to-400-tokens (`s1a-receipt` wall_clock_s=512.22 / 120 generations,
§ receipted above) at ≈281 mean tokens:

- Sender generation: 40 items × ≈2.05 s (281 tok × 7.3 ms) ≈ 82 s
- Receiver generations: 40 items × 5 conditions × ≈4.27 s ≈ 854 s
- **Total ≈ 936 s ≈ 0.26 GPU-hours** for a 40-item probe at ADR-023's own frozen scale.

Scaled to ADR-023's `eval-200` size (200 items, matching the run-1 A3 non-inferiority scale):
≈200/40 × 936 s ≈ **4,680 s ≈ 1.3 GPU-hours.**

Both figures are **well under** run 1's actual 3.3 GPU-hour spend (ADR-023 §S6 accounting) —
because this experiment skips calibration (S2, ≈2.5 GPU-h of the run-1 total was S2/S2c dump +
recalibration), skips the alignment bridge probes, and skips any adapter training. It is the
cheapest publishable-shaped result available in this codebase's current state, and it does not
depend on latent transfer ever working.

---

## Consequences

- The compute case for latent transfer at GPU-local/LAN tiers is not actually in question — this
  repo's own receipts put decode at ≈92× the per-token cost of prefill, matching the mechanism C2C
  reports (though C2C reports the resulting wall-clock ratio, not the decode/prefill breakdown
  itself, which is this document's own contribution). What remains unresolved, per ADR-023/024's
  three nulls, is whether *any* trained or untrained cross-model map carries usable content —
  compute was never the bottleneck the field or this repo has hit.
- The radio-tier byte argument is weaker and more conditional than the compute argument, and
  depends on which of LatentMesh's two "latent" objects is meant. Citing C2C's 2.0-2.5× speedup as
  support for a bandwidth claim over LoRa is a category error the paper itself doesn't make — it
  never measures bytes.
- Q2's gap is real by this search's evidence: the field has causally audited latent channels
  (Zhang & Emu) and has attributed agent-level contribution in text systems (Agents that Matter),
  but nothing found combines content-level decoy controls with a text channel. ADR-003's gate is
  channel-agnostic by design; this experiment is the cheap way to cash that in as a second,
  latent-transfer-independent result, at roughly 0.26–1.3 GPU-hours depending on scale — a
  rounding error against run 1's 3.3 GPU-hour spend.

## Sources

- In-repo: ADR-002, 003, 009, 010, 019, 023, 024; `docs/research/025`, `028`;
  `crates/latentmesh-runtime/receipts/{s0-receipt,s1a-receipt-slots8-block19-poolfull-rescaletrue-n40,
  s2-dump-receipt,s2c-generated-dump-receipt}.json` (every wall-clock/token figure cited above is
  read directly from these files, not retyped from a prior summary).
- Cache-to-Cache: [arXiv:2510.03215](https://arxiv.org/abs/2510.03215),
  [HTML body](https://arxiv.org/html/2510.03215v1), [github.com/thu-nics/C2C](https://github.com/thu-nics/C2C)
- LatentMAS: [arXiv:2511.20639](https://arxiv.org/abs/2511.20639)
- "Agents that Matter: Optimizing Multi-Agent LLMs via Removal-Based Attribution":
  [arXiv:2605.27621](https://arxiv.org/abs/2605.27621)
- "Do Latent Channels Actually Communicate? A Causal Audit of Latent Multi-Agent LLM
  Communication" (Zhang & Emu): [arXiv:2607.26773](https://arxiv.org/abs/2607.26773),
  [HTML body](https://arxiv.org/html/2607.26773v1)
