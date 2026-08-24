# Architecture Decision Records — LatentMesh

Numbered sequentially; immutable once Accepted. Status lifecycle: `Proposed` →
`Accepted` → `Implemented` / `Superseded`. This is a **research prototype**
(see the root README's Honest Status) — most ADRs below are design contracts,
not shipped systems; each states plainly what is and isn't built.

| ADR | Title | Status |
|---|---|---|
| [001](001-latentmesh-architecture-and-prior-art.md) | LatentMesh architecture and prior art — the north star, and an honest map of what LatentMAS, AVP, AAFLOW+, StateBridge, DMoA, Cisco Cognitive Fabric, and LCGuard already cover vs. the actual whitespace (causally-verified, governed, self-evolving integration) | Proposed |
| [002](002-latent-packet-protocol.md) | Latent packet protocol (`LatentFrame`) — a reference wire type for this workspace's own crates, deliberately scoped narrower than AVP (which already specifies KV/hidden-state transport); a bridge to AVP is the likely long-run wire format | Proposed (packet crate implemented) |
| [003](003-causal-edge-verification.md) | Causal edge verification — the core differentiator: an agent-to-agent latent edge is admitted to the topology only after surviving the four-control test (zero / random / mismatched / self-generated state) with statistically significant incremental value | Proposed |
| [004](004-streaming-latent-state.md) | Streaming latent state — incremental latent frames over MidStream instead of waiting for completion; downstream agents consume partial cognitive state | Proposed |
| [005](005-persistent-latent-memory.md) | Persistent latent memory — RuVector stores a continuum from raw latent trajectory to compressed trajectory to semantic prototype to symbolic rule | Proposed |
| [006](006-self-evolving-topology.md) | Self-evolving topology — MetaHarness Darwin mutates the communication graph itself, fitness-scored by causal edge value (ADR-003), not token traffic or superficial accuracy | Proposed |
| [007](007-federated-world-models.md) | Federated world models — Radio exchanges structured, compatibility-checked transition rules between nodes instead of pooling raw experience | Proposed |
| [008](008-capability-governed-execution.md) | Capability-governed execution — RVF/RVM enforce `execute(z) ⟺ signature(z) ∧ authority(z) ∧ provenance(z) ∧ risk(z) < τ`; latent execution is a governed capability, not an implicit trust | Proposed (gate implemented) |
| [009](009-online-causal-control-loop.md) | The online causal control loop — corrects ADR-001's novelty claim against a second literature pass (E2 Explainer, MANTA, BANDMAS); `G_t=(A,E,Z,M,P)`; the closed loop (execute→audit→measure→persist→evolve); a named role for every existing ruvnet component (RuFlo/MetaHarness/RuVector/MidStream/Radio/RVF+RVM/RuView/Autogenous); the one-vertical-experiment plan; ablation-based acceptance test | Proposed |
| [010](010-latentmesh-air-protocol.md) | LatentMesh Air protocol — bounded 16 through 256 byte physical frame, optional LMS1 authentication envelope, deterministic LMAD state delta, fragmentation, replay defense, and cross-language golden vectors | Accepted and partially implemented |
| [011](011-radio-adapters-and-legal-boundary.md) | Radio adapters and legal boundary — separates WiFi and BLE transports from licensed external HF, VHF, AM, FM, TNC, SDR, antenna, and operator responsibilities | Accepted and partially implemented |
| [012](012-neural-receiver-fallback.md) | Neural receiver with deterministic fallback — bounded likelihood correction subordinate to conventional synchronization, FEC, CRC, replay, authentication, and state checks | Accepted and minimally implemented |
| [013](013-esp32-firmware.md) | ESP32 firmware — native WiFi and BLE, KISS and I2S bridges, bounded queues, metrics, and default-deny external transmit policy | Accepted and implemented without hardware validation |
| [014](014-benchmark-and-acceptance-method.md) | Benchmark and acceptance — separates the tenfold semantic reduction gate from the additional twofold neural physical-layer gate and requires evidence labels | Accepted as validation contract; over-the-air gate not passed |
