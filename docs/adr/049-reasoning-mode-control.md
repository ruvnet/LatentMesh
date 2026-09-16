# ADR-049: Opt-in reasoning-mode control experiments

Status: experimental reference implementation, not runtime promotion.

## Decision

Add `harness/mode-control`, a removable, dependency-free Node harness, following
the separation in ADR-046. It reads locally supplied sparse linear probes,
exchanges bounded Ed25519-authenticated mode advisories, and provides a pure
additive residual operator for locally calibrated adapters. All existing Rust
runtime paths, authority gates and historical experiment receipts remain unchanged.

Modes are explore, converge, challenge and unknown. They describe an advisory
stance, not task correctness, thought contents, tool permission or expert routing.
Probe coefficients, dimensions, thresholds and directions are model-specific.
No ten-neuron indices or universal steering vectors are invented or shipped.

## Research grounding and limitations

Inspired by [Metacognitive Steering](https://arxiv.org/abs/2609.16245), particularly
the distinction between exploratory regime classification and causal behavioral
improvement. Earlier [Activation Addition](https://arxiv.org/abs/2308.10248) and
[Representation Engineering](https://arxiv.org/abs/2310.01405) are prior art.
This is NOT a reproduction of the paper's multi-mode learned controller or its
99% probe result. Our three-way linear probe is an explicit reference baseline.
Existing LatentMesh null results remain null. Classification does not qualify a
channel for production; no promotion API is exposed here.

## Security boundary

Receivers pin a trusted Ed25519 key, sender, session, target checkpoint revision
and local adapter content hash. Exact schemas, 4096-byte wire limit, bounded TTL,
finite values and monotonic sequence numbers fail closed. Persist the high-water
sequence or use a fresh unpredictable session after restart. A signature proves
origin, not the truth of evidenceHash or calibrated probability. The transport
must still be authorized externally. A mode cannot confer authority.

The offline residual operator is disabled by default, validates model/revision,
rejects excessive delta norms, returns new arrays and emits deterministic hash
receipts. Receipts are integrity records, not signatures. No live model hooks,
network writes, radio traffic or credentials are added. Adapters are loaded from
locally trusted content hashes; untrusted remote payloads cannot supply directions.

## Evaluation before live integration

Freeze a model/checkpoint, data split, probe fitting procedure, intervention
layers/gains and evaluator BEFORE any held-out evaluation. Keep all states from
one task/trajectory in the same split, including feature selection, to prevent
leakage. Compare no message, explicit text mode, detected mode, mode plus local
steering, norm-matched random steering, shuffled modes and zero intervention.
Use equal task/token budgets, isolated caches and identical model weights.

Report held-out regime accuracy separately from task accuracy, paired outcomes,
abstention rate, OOD false acceptance, token/latency cost and message bytes.
Require the existing causal gate and independent authority approval before any
runtime activation write. A 30-task pilot may estimate feasibility, but cannot
establish universal superiority or 99% accuracy. No model/GPU experiment has run
as part of this change; deterministic tests validate mechanisms only.

Next integration: separately reviewed Candle readout and additive decode hook,
checkpoint-pinned probe/direction training, then held-out experiments. Do not
reuse prefill placeholder overwrite as if it were per-token dynamic steering.
