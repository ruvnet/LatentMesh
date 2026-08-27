# 021. cognitum-one API integration

- **Status**: Accepted — implemented this wave (offline/mock evidence only). Updated 2026-08-27.
- **Date**: 2026-08-27.
- **Related**: [008](008-capability-governed-execution.md) (authority/promotion gate this ADR's schema reference informs), [010](010-latentmesh-air-protocol.md) (the "entity identifiers, units, observation time, confidence policy, provenance references, authority" gap this ADR's §3.3 reference addresses), [016](016-ruvector-persistent-latent-memory.md) (the feature-gated optional-backend pattern this ADR mirrors)
- **Evidence base**: [docs/research/019-meshtastic-agentbbs-cognitum-research.md](../research/019-meshtastic-agentbbs-cognitum-research.md) §3, §4.4

## Context

`cognitum-one/api` owns `api.cognitum.one`'s OpenAPI contract, API-key model,
and gateway (Phase A live: DNS/TLS resolved, landing page served; Phase B
path-based routing still pending). Its `/v1/spaces/{siteId}` design
(ADR-099 in that repo) is a shipped answer to almost exactly the problem
ADR-010 leaves open: "entity identifiers, units, observation time, confidence
policy, provenance references, and authority" required before a received
delta can be promoted into an authoritative fact. That makes it worth reading
as a design reference for LMAD's still-missing upper-layer schema, independent
of whether this repo ever calls that endpoint.

This workstation's own operational notes (`CLAUDE.local.md`) separately
describe a **V0 appliance** local API — bearer-auth `cog-gateway` on port
9000, `/api/v1/v0/*`, UDP discovery on port 5008. A full-repo grep of
`cognitum-one/api` found **zero** `/api/v1/v0/*`, `cog-gateway`, or `:9000`
references except one cross-reference inside ADR-099 itself, confirming these
are two different services, not two names for the same thing.

## Decision

**Two distinct integration surfaces. Never conflate them.**

| | Cloud fleet API | Local V0 appliance |
|---|---|---|
| Host | `api.cognitum.one` (this repo, `cognitum-one/api`) | co-located appliance, separate codebase |
| Routes | `/v1/seed/register`, `/v1/seed/heartbeat`, `/v1/spaces/{siteId}`, `/v1/oak/*` | `/api/v1/v0/*` |
| Auth | Ed25519 device keypair; canonical-request signing | bearer token, `:9000` |
| Discovery | HTTPS DNS | UDP `:5008` |
| Deployment shape | internet-registered device/fleet | physically co-located hardware |

A LatentMesh node targets one or the other by its own deployment shape, never
both through the same client path.

- **Cloud fleet registration**: `POST /v1/seed/register`
  (`{deviceId: uuid, publicKey: base64 Ed25519, firmware}`, unauthenticated
  first-contact — mirrors LMS1's own asymmetric-signature model for
  `SIGNED_ENVELOPE`), then `POST /v1/seed/heartbeat`
  (`{uptime_secs, free_memory_kb, total_vectors, epoch, wifi_ip, version}`).
  Every request past registration is signed:
  `canonical_request = "POST\n{path}\n{ISO8601 timestamp}\nsha256(body)"`,
  sent as `X-Device-Id`/`X-Device-Timestamp`/`X-Device-Signature`; server
  rejects `|server_now − timestamp| > 300s` and a duplicate signature within
  a 10-minute replay window. **Evidence correction (2026-08-27, found during
  implementation)**: only the ±300s clock-skew check was found implemented in
  the cloned `cognitum-one/api` source; the 10-minute replay window traces
  only to prose in that repo's `docs/seed-integration.md` with no
  implementation found — treat it as a documented-but-unverified server
  behavior. The repo also contains a second, different signing scheme in
  `functions/seed/index.js`, which its own README labels a "non-canonical
  reference fork; do not deploy to production" — the canonical OpenAPI/docs
  contract (implemented here) is the one to follow. **Naming note**: cognitum's heartbeat body and
  LMS1's `SemanticEnvelope.epoch` both use the field name `epoch` for
  unrelated concepts (heartbeat: a fleet-side counter; LMS1: the envelope's
  own epoch field, `envelope.rs`). This ADR keeps them explicitly distinct —
  no shared type, no field aliasing — rather than aligning the names, since
  aligning them would falsely imply the two systems share an epoch domain.
- **Richer semantic content**: `PUT /v1/spaces/{siteId}` (ADR-099) is the
  closer match once a node has more than liveness to report, and its
  envelope invariants are the design checklist for LMAD's open upper-layer
  schema — monotonic `eventSequence` (idempotent exact retries, rejected
  conflicting reuse), caller-supplied `observedAt` anchoring a
  server-bounded `ttlSeconds` (30s-24h) so transport delay cannot revive
  stale state, `P2`/`P3`-only privacy classes (raw/`P0`/`P1` rejected, not
  coerced), `edgeRuntime.{homecoreVersion, ruviewModelVersion,
  calibrationId}` + `confidence` + `inference.evidence[]` for
  provenance/confidence/model-version, a 64 KiB bound with bounded JSON
  depth/counts/strings, and — the sharpest one — **self-reported hardware
  verification claims are discarded server-side; a device cannot mark its
  own hardware "verified."** This ADR does not call `/v1/spaces/{siteId}`
  yet; it records the checklist as the reference for whoever designs LMAD's
  promotion-gate schema next.
- **Local appliance path**: target the co-located V0 appliance's own
  `/api/v1/v0/*` gateway directly — a separate client, separate auth model,
  never routed through the cloud-fleet client.
- **New crate `latentmesh-cognitum-client`, feature-gated**, mirroring
  `latentmesh-memory`'s `ruvector` feature exactly (ADR-016): `default = []`,
  an `http` feature adding the optional HTTP-client dependency, CI exercises
  `--features http` separately. **Ed25519 canonical-request signing stays in
  the default build** — `ed25519-dalek` is already a workspace dependency
  and the signing/canonicalization logic is fully offline-testable (build
  the canonical string, sign it, verify against a fixture keypair) without
  any network call. Only the HTTP transport itself goes behind the feature,
  so `cargo test --workspace` stays hermetic and green offline by default.
  Bearer/device credentials are read from the environment at call time,
  never committed, never hardcoded — consistent with the repository's
  existing security rules.

## Consequences

Because the client is feature-gated with signing in the default build, the
workspace gains a design reference and a tested signing primitive without
adding a network dependency to the default build graph — the same trade
ADR-016 made for the optional `ruvector-core` backend. The two-surface table
is the concrete artifact that prevents the V0-appliance/cloud-fleet
conflation this ADR's evidence base found no prior documentation
distinguishing. The `/v1/spaces/{siteId}` schema reference is explicitly not
an integration yet — a future ADR would need to decide whether LMAD's
promotion path actually calls it or merely borrows its invariant shape.

## What is simulated / what is hardware-pending

| Claim | Status |
|---|---|
| Canonical-request construction + Ed25519 signing against a fixture keypair | Buildable and testable today, hermetic, no network, no credential |
| `/v1/seed/register` + `/v1/seed/heartbeat` payload construction against the published OpenAPI schema | Buildable and testable today against a mock HTTP server |
| Live `POST /v1/seed/register` against `api.cognitum.one` | **Credential-pending** — needs a provisioned device identity that doesn't exist by default |
| `PUT /v1/spaces/{siteId}` call | **Not integrated** — used only as a design reference in this ADR, not called |
| Local V0 appliance `/api/v1/v0/*` client | **Not implemented** — separate follow-up if a co-located deployment shape is chosen |

## Implementation status

Implemented 2026-08-27, same branch. `crates/latentmesh-cognitum-client`
exists with `default = []`: Ed25519 canonical-request signing with an
injected clock lives in the default (offline) build; the HTTP transport is
gated behind the `http` feature (optional `ureq`). 27 tests pass by default
(no network), 30 with `--features http` including a local mock-server test.
No credentials are hardcoded anywhere; live `api.cognitum.one` calls remain
credential-pending per the table above.
