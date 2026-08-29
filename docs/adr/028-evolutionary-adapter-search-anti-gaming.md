# 028. Evolutionary adapter search (Darwin/AVO/flywheel) with anti-reward-hacking guardrails

- **Status**: Proposed. Design contract only — no code, no search run, no promotion in this wave.
- **Date**: 2026-08-29.
- **Related**: [024](024-run2-trained-thought-adapter-ladder.md) (the ladder this search would
  accelerate — M3 MLP and M4 FastGRNN both honest-failed by 2026-08-29, per that ADR's appended
  outcome sections; M4c task-loss ablation is running as a registered contingency),
  [022](022-self-optimizing-metaharness-e2e-loop.md) (the existing pattern this ADR follows:
  MetaHarness tooling as an optional, subprocess-invoked gate, never a workspace build/runtime
  dependency), [014](014-benchmark-and-acceptance-method.md) (the evidence-label/stage-separation
  discipline this search's fitness values must respect), [003](003-causal-edge-verification.md)
  (the five-control test — the actual ground truth this ADR refuses to let a search function
  substitute for), [023](023-live-four-condition-run1-pre-registration.md) (the frozen 40-item
  probe protocol this ADR treats as inviolable resource, not a search target)
- **Evidence base**: [docs/research/027-global-ambient-intelligence-track.md](../research/027-global-ambient-intelligence-track.md)
  §3 (the Bittensor/Filecoin/BOINC gaming-reappears-one-layer-up pattern this ADR's guardrails
  respond to, and the attack-coverage-map gap that this ADR's own search layer inherits), current
  session `metaharness_status` read (package/version inventory: `@metaharness/darwin@0.9.2`,
  `@metaharness/flywheel@0.1.10`, confirmed available this session), the `@metaharness/avo`
  package README (fetched from the locally cached tarball this session,
  `/tmp/metaharness-avo-07a2237.JEGrD0/metaharness-avo-0.1.0.tgz`, `package/README.md`), and
  ADR-322 in the `ruflo` marketplace repo (`~/.claude/plugins/marketplaces/ruflo/v3/docs/adr/ADR-322-metaharness-flywheel-integration.md`,
  read this session — the flywheel's actual promotion-gate implementation reference)

## Context

Run 2's ladder (ADR-024) explores adapter design by hand, one architecture per registered rung: M3
(MLP) and M4 (FastGRNN, three ranks) both honest-failed against the frozen probe as of this ADR's
authoring, and a task-loss ablation (M4c) is running as a registered contingency. The adapter
design space — architecture family, injection depth/site, slot count, pooling scheme, loss shape,
rank/width, training schedule — is large, and exploring it one nightly rung at a time is slow.
rUv's MetaHarness stack offers automated search over exactly this class of space:
`metaharness_evolve` (Darwin — mutate a policy surface, sandbox-score variants, keep measured
wins), `metaharness_flywheel` (an immutable-receipt, append-only, Ed25519-signed promotion ledger —
**verified empty/genesis on this repo this session**: `ledgerHead` all-zeros, no active champion,
confirming there is no prior state to accidentally build on or conflict with), and
`@metaharness/avo` (governed autonomous variation — a model proposes
inspect/search/hypothesize/edit/run/evaluate/repair actions as Darwin's child-generator, while the
outer loop retains "immutable authority over capabilities, budgets, protected invariants,
promotion, quarantine, rollback, and signed receipts" — quoted from the package's own README).

**Why this needs its own ADR, not just "turn Darwin on":** the frozen 40-item S1a/S2b probe
(ADR-023) is a *one-shot resource*. Every draw against it spends irreplaceable evidence — the whole
point of pre-registration is that the probe's statistics mean what they say only because it is
consumed rarely and deliberately, not iterated against. An automated search process with the frozen
probe as its fitness function would not be optimization; it would be Goodharting the experiment
itself, indistinguishable in kind from p-hacking. `docs/research/027` §3 already surveyed exactly
this failure class across BOINC, Filecoin, and Bittensor: **verification gaming does not get
eliminated by adding a check — it reappears one layer above the check.** Filecoin's cryptographic
layer genuinely defeated sybil/outsourcing/generation attacks, and the gaming moved to
self-dealt filler data that the protocol structurally couldn't distinguish from useful data.
Bittensor's stake-weighted-median Yuma Consensus genuinely defeats a naive colluding-minority
attack, and gaming moved to weight-copying (mitigated by commit-reveal, itself an after-the-fact
patch) and to structural stake concentration (arXiv:2507.02951: over half of subnets need under 1%
of wallets to reach the 51% collusion threshold Yuma Consensus is designed to resist). This ADR's
central design commitment is choosing, up front, which layer the search operates at, so that the
gaming this pattern predicts has nowhere useful to reappear.

## Decision

### The two-list separation — the core of this ADR

**EVOLVABLE (search is legitimate here):**

- Adapter architecture family and hyperparameters (MLP width/depth, FastGRNN rank, any future
  architecture family)
- Injection depth, injection site (which decoder block), slot count
- Pooling scheme (per-token vs. pooled, and any pooling variant in between)
- Loss shape (reconstruction vs. task-loss vs. a weighted combination — the exact axis M4c is
  currently testing by hand)
- Rank/width of any low-rank factorization
- Training schedule (epoch budget, learning-rate shape within `candle-nn`'s AdamW-only constraint,
  per ADR-024)

**Fitness for every evolvable-surface search is a metric computed on the `adaptation-512` pool
ONLY** (train-derived, per ADR-023's own split discipline) — **never** the frozen 40-item probe.
Candidate fitness can be held-out relative residual, task-loss on held-out adaptation-pool items,
or any other train-pool-computable proxy; it is never, under any circumstance, a sign-test result
against the S1a/S2b item set.

**PROTECTED / NEVER EVOLVABLE (evolving these is scientific fraud, not optimization):**

- The frozen S1a/S2b probe protocol itself — the 40 items, the exact sign test, α=0.05, slot
  count, rescale-to-median switch, greedy/batch=1 decoding (ADR-023)
- The `eval-200`/`holdout-100` splits and their mechanical lock (`eval_items()`/`holdout_items()`
  refusing until a genome-frozen receipt exists)
- ADR-003's five-control decoy definitions (`zero`/`random`/`mismatched`/`self_generated`/
  `text_equivalent`) — a search process may not redefine what counts as a valid control
- The 13-item leakage-exclusion list (ADR-024's leakage discipline)
- Receipt and evidence-label discipline itself (ADR-014/018's evidence-labelling rules) — a search
  process may not choose its own evidence label
- The pre-registration rules — freeze-before-probe — themselves

**A search process, no matter how sophisticated its fitness function or how well-defended its
sandbox, never gets write access to the second list.** This is the layer-boundary decision
`docs/research/027`'s pattern demands: Darwin/AVO evolve *within* the frozen protocol's rules, they
never evolve the rules.

### Promotion rule

The search may nominate **at most one champion per registered ladder rung** (M3, M4, M4c, or any
future rung this ADR's guardrails cover). That champion receives **exactly one frozen-probe draw**,
pre-registered exactly like every other rung in ADR-024 — same freeze-before-probe discipline, same
receipt format, same no-retry-on-failure rule. Everything else the search produces — every
candidate the Darwin loop scored, every AVO-proposed variant, every intermediate architecture — is
train-pool-only evidence, reportable and useful for understanding the search, but never itself a
probe-tested result. This rule is what keeps the search's throughput advantage (many candidates
scored cheaply) from ever translating into probe-budget consumption (only one candidate per rung
ever touches the 40 frozen items).

### Anti-gaming machinery — necessary, not sufficient

Two tools, both confirmed available in this session's tool surface, run as mandatory gates before
any champion nomination becomes a probe draw:

- **`metaharness_reward_hack_scan`** — scans an archived Darwin trajectory for gold reads,
  verification tampering, or sandbox escape. Run on **every search trajectory** before its
  champion is promoted to a probe draw, not sampled or run after the fact.
- **`metaharness_replay_verify`** — verifies a flywheel replay bundle's signed lineage and parent
  continuity (and, optionally, a pinned gate fingerprint). Run on **the promotion bundle itself**
  before the probe draw executes.

**Both tools are necessary and explicitly not sufficient.** `docs/research/027`'s own
attack-coverage map (against ADR-003's five controls, directly transferable to this search layer)
names three attack classes nothing in this repo currently defends against: **selective omission**
of real, honestly-signed data (Koyuncu et al., arXiv:2305.20043 — signing defeats fabrication, does
nothing against an adversary who submits only real data and chooses *which* real data reaches
scoring); **adaptive iterative probing** invalidating a fixed statistical test's guarantees (Dwork
et al.'s Reusable Holdout result — repeated adaptive access lets an optimizer learn which
perturbations survive a test without ever learning the true mechanism); and **colluding
verification rings**, evidenced in a different domain (peer-review bidding, Jecmen et al.,
arXiv:2402.07860) but structurally general — coordinated small rings evaded every tested
graph-structural detector. None of these are defeated by `reward_hack_scan` or `replay_verify`
alone. The one-champion-per-rung promotion rule above is this ADR's actual defense against
adaptive iterative probing specifically (it bounds the number of adaptive queries against the
frozen probe to exactly the number of registered rungs, not the number of search iterations); the
protected-list boundary is the defense against a search process re-scoping its own evidence
standard. Neither defends against selective omission of training data, which remains an open gap
this ADR does not claim to close.

### `@metaharness/avo`'s role — engineering surface only, scoped honestly

AVO's own README states its initially evolvable surfaces are "retrieval policy, model routing,
context policy, test policy, and repair strategy," and explicitly: **"Security policy and
capability expansion are not evolvable surfaces."** This maps naturally onto run 2's *engineering*
lane — the harness code, training pipeline, and capture code (`crates/latentmesh-train`,
`crates/latentmesh-runtime`'s capture extensions, `harness/latentmesh-live`) — not onto proposing
experiment semantics. AVO is registered for that engineering-surface use, under the identical
protected list above (an AVO-proposed edit to `harness/latentmesh-live/src/gsm8k.rs`'s
eval/holdout mechanical-lock logic is exactly as forbidden as a Darwin-proposed one).

**Two prerequisites this ADR states rather than assumes away**: AVO needs its own package install
(`npm install @metaharness/avo agenticow` for production RVF-backed memory, per its README) — not
currently installed in this workspace, and needs a **bench/evaluator suite** to run its
inspect/search/hypothesize/edit/run/evaluate/repair loop against — neither the package nor a
compatible evaluator suite exists in this repo today. AVO's own README is honest about the
limitation of its default proof: "The default benchmark data is a synthetic mechanism fixture and
cannot authorize an AVO-class product claim" — its 100-task SWE-bench ship gate requires
externally supplied `SWE_BENCH_RESULTS`, which has no LatentMesh analog yet. Registering AVO here
is a decision about *where* it would slot in if adopted, not a claim that it is ready to run.

### Flywheel adoption — key management is a prerequisite, not a detail

`@metaharness/flywheel`'s promotion gate (ADR-322 in the `ruflo` marketplace repo, read this
session) implements canonical JSON receipts, an O_EXCL cross-process transaction lock, and
Ed25519-signed lineage. **Adopting it here means signing every promotion receipt with a locally-held
Ed25519 key** — this is a prerequisite this ADR names explicitly rather than treating as an
implementation detail to figure out later: key generation, storage, and rotation policy for that
signing key must exist before the flywheel's signed-receipt guarantee means anything for this
repo's own promotions. The flywheel's genesis state on this repo — `ledgerHead` all-zeros, no
champion — is the honest starting point: adopting this ADR does not inherit any prior promotion
history, real or borrowed, from anywhere else this stack has run.

## Consequences

The two-list separation is the actual content of this ADR — everything else (the promotion rule,
the anti-gaming tools, AVO's scoping, the key-management prerequisite) follows from having drawn
that boundary honestly. The cost is real: restricting fitness to the `adaptation-512` pool means
the search's own signal is *not* what ultimately matters (probe-passing is), so a search process
that finds a candidate optimizing train-pool fitness beautifully could still fail the one frozen
draw it earns — this is not a flaw in the design, it is the design working as intended, the same
way M3's well-fitting-but-causally-null MLP was a correct (if disappointing) result rather than
evidence the search process should have kept trying. Naming the attack-coverage gaps explicitly
(selective omission, adaptive probing beyond the one-draw bound, verification-ring collusion)
rather than implying the two named tools close them is a direct application of
`docs/research/027`'s own finding: a defense that survives one round of gaming and is described as
complete invites the gaming to reappear one layer up, exactly where nobody was looking.

## What is simulated / what is hardware-pending

| Claim | Status |
|---|---|
| `metaharness_evolve`, `metaharness_flywheel`, `metaharness_reward_hack_scan`, `metaharness_replay_verify` are available in this session's tool surface | **Verified this session** via `metaharness_status` (package versions: darwin 0.9.2, flywheel 0.1.10) |
| Flywheel ledger genesis state (all-zeros head, no champion) on this repo | **Verified this session**, cited as reported in the coordinator's task brief and consistent with no prior promotion activity anywhere in this repo's history |
| Any actual Darwin search over run-2 adapter hyperparameters | **Not implemented** — no code in this wave |
| Any champion nomination, probe draw, or promotion | **Not implemented** — no search has run |
| `@metaharness/avo` installed and usable in this workspace | **Not installed** — package and a compatible evaluator suite are both prerequisites, not present |
| Ed25519 signing-key provisioning for flywheel receipts | **Not provisioned** — named as a prerequisite, not begun |
| Any defense against selective-omission, adaptive-probing-beyond-one-draw, or verification-ring-collusion attacks | **Not addressed** — named as an open gap, not claimed as covered |

## Implementation status

Not implemented. This ADR is a design contract naming the evolvable/protected boundary, the
promotion rule, and the anti-gaming machinery to adopt when (if) an evolutionary search over run
2's adapter design space is actually built; no crate, workflow, or search run exists yet.
