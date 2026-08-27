# 022. Self-optimizing MetaHarness e2e loop for the integration wave

- **Status**: Accepted — implemented this wave (deterministic simulation evidence only). Updated 2026-08-27.
- **Date**: 2026-08-27.
- **Related**: [014](014-benchmark-and-acceptance-method.md) (stage-gate/evidence-label discipline this ADR extends), [018](018-metaharness-darwin-topology-loop.md) (the harness pattern and receipt discipline this ADR follows), [019](019-meshtastic-transport-adapter.md), [020](020-agentbbs-store-and-forward-bridge.md), [021](021-cognitum-one-api-integration.md) (the three integrations this ADR's suites exercise)

## Context

ADR-018 established the repo's harness discipline: frozen policy domains,
deterministic seeds, evidence-labelled JSON receipts, a `harness/<name>`
directory with a Node.js runner shelling out to a release-built Rust binary
and verifying its receipt against a documented acceptance bound
(`harness/evolve/scripts/run.mjs`). `harness/air` established the same
pattern first, for the radio link. Neither harness is itself Rust — both are
thin Node orchestration around a Rust binary that does the actual work — and
that split is the established exception to this repository's Rust-only rule
for product code: the harness scripts execute and verify, they do not
implement protocol or business logic.

The three integrations in ADR-019 through ADR-021 need the same treatment:
each is loopback/simulation-testable today (per each ADR's boundary table)
but has no end-to-end evidence tying the pieces together, and no CI-style
acceptance gate comparable to ADR-014's stage table.

## Decision

**Extend `harness/`, don't invent a parallel evidence system.** New
`harness/integration/` directory, same shape as `harness/air` and
`harness/evolve`:

- **Three e2e loopback scenario suites**, one per integration, each a Node
  runner invoking the relevant Rust crate's test/example binary and
  producing an evidence-labelled receipt:
  1. **Air-over-Meshtastic-framing**: fragment a `SemanticEnvelope` at the
     233-byte MTU (ADR-019), push fragments through a mock `Data.payload`
     channel with the `0x94 0xc3` device-API framing applied and stripped,
     confirm `Reassembler` round-trips byte-identical.
  2. **agentbbs bridge round-trip**: decode a `SemanticDelta`, run it
     through `latentmesh-agentbbs-bridge`'s mapping functions (ADR-020),
     confirm the resulting `post_message`/`ReplicateMessage` JSON matches
     the pinned agentbbs contract shape; where the `agentbbs mcp` binary is
     present in the environment, additionally roundtrip it live over stdio.
  3. **cognitum client contract tests**: build and sign a canonical request
     (ADR-021) against a fixture keypair, replay it against a mock HTTP
     server built from the published OpenAPI contract, confirm signature
     verification and payload shape both hold.
- **MetaHarness gates as an optional, removable layer** — `score`,
  `genome`, `mcp-scan`, `threat-model`, invoked as pinned external CLI/MCP
  tooling (the `ruflo-metaharness` surface), subprocess-invoked from the
  harness runner exactly as `harness/evolve` subprocess-invokes `cargo run`.
  These gates are **never a build or runtime dependency of any workspace
  crate** — no `Cargo.toml` in this repository references them, directly or
  transitively. When the external tooling is absent, the harness runner
  skips those specific gate steps and labels the receipt accordingly rather
  than failing the whole suite; `cargo test --workspace` never depends on
  their presence. This mirrors this repository's existing pattern of keeping
  MetaHarness a removable augmentation over the protocol/business logic it
  evaluates, not a dependency of it.
- **Optimization loop**: reuse `latentmesh-evolve`'s deterministic,
  seeded-RNG Darwin loop pattern (ADR-018), but with the search space scoped
  to adapter parameters instead of topology — MTU packing (whether a given
  delta fits one Meshtastic packet vs. needs fragmentation), fragmentation
  threshold, and bridge batching interval — evaluated against the e2e
  suites' own receipts as the fitness signal (fewer fragments and lower
  simulated latency at unchanged round-trip correctness). Same evidence
  label discipline as ADR-018: a JSON receipt with `evidence: "simulated"`
  and an explicit `not_claimed` list; no live-agent, live-radio, or
  live-credential claim.
- **CI-style acceptance gates**, same shape as ADR-014's stage table: each
  e2e suite is a stage; a suite may only advance the claim it actually
  measures (loopback correctness is not a hardware claim, a passing
  MetaHarness `threat-model` gate is not a live-security-review claim).

## Consequences

The integration wave gets the same evidence discipline the rest of the repo
already has, instead of three one-off test scripts with no shared receipt
format. Making the MetaHarness score/genome/mcp-scan/threat-model gates
optional and subprocess-invoked keeps the hard workspace-hermeticity
property (`cargo test --workspace` succeeds offline, with no external
tooling installed) while still letting a CI environment that does have the
tooling run the fuller gate set. The optimization loop over adapter
parameters is deliberately narrow in scope — it tunes packing and batching
constants against simulated receipts, it does not claim to discover new
protocol behavior, matching ADR-018's "optimizes the causal objective in
simulation, not real-agent gains" framing.

## What is simulated / what is hardware-pending

| Claim | Status |
|---|---|
| Three e2e loopback suites (Meshtastic framing, agentbbs bridge, cognitum contract) | Buildable and testable today, no hardware, no live credentials |
| MetaHarness score/genome/mcp-scan/threat-model gates, when the external tooling is present | Buildable and testable today as an optional CI layer; absent tooling degrades the receipt, does not fail the workspace build |
| Adapter-parameter optimization loop (MTU packing, fragmentation threshold, bridge batching) | Buildable and testable today in deterministic simulation, reusing ADR-018's Darwin-loop pattern |
| Real Meshtastic hardware, real cognitum credentials, real agentbbs peers in the loop | **Not implemented, not claimed** — none exist on this host; every suite above targets loopback/simulation fixtures only |

## Implementation status

Implemented 2026-08-27, same branch, after ADR-019 through ADR-021 landed.
`harness/integration/` exists with the three e2e loopback suites (each
shelling out to an `e2e_loopback` example in the owning crate), the
metaharness-gates runner, and the Darwin-loop optimizer (selected
fragmentation threshold 211 B and bridge batching interval 500 ms —
simulation-optimal only, per the receipt's own `not_claimed` list). All
receipts land in `harness/integration/artifacts/` with `"evidence":
"simulated"` labels and seeded determinism; CI runs the suite as the
`metaharness-integration` job in `air-ci.yml`.
