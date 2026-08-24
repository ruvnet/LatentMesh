# 010. LatentMesh Air protocol

* Status: Accepted and partially implemented
* Date: 2026-08-23
* Related: [002](002-latent-packet-protocol.md), [003](003-causal-edge-verification.md), [007](007-federated-world-models.md), [008](008-capability-governed-execution.md), [011](011-radio-adapters-and-legal-boundary.md), [012](012-neural-receiver-fallback.md), [014](014-benchmark-and-acceptance-method.md)

## Context

The current LatentFrame benchmark reports about 64.1 KiB for a 16 by 4096 Int8
latent payload. That object is useful between machines with ordinary network
links but is the wrong unit for a 300 or 1200 bit per second radio. At 300 bit
per second, 64.1 KiB requires about 29 minutes before framing, error correction,
interleaving, identification, or retransmission. Putting the existing object on
HF would preserve a software abstraction while destroying the link budget.

The radio layer needs a compact deterministic representation with bounded
memory and an explicit relationship to the receiver world state. Mission
critical facts must not depend on generative reconstruction.

## Decision

Create LatentMesh Air first as a radio agnostic transport for sparse semantic
deltas, then bind the same messages to profile driven physical adapters.
The portable C implementation in `c/**` is the canonical embedded wire codec.
Rust may provide ergonomic and accelerated implementations, but both languages
must accept the same conformance vectors.

The transmitter pipeline is:

```text
WorldGraph state
  -> receiver knowledge estimate
  -> deterministic symbolic delta
  -> importance budget
  -> optional shared latent residual
  -> optionally signed semantic envelope
  -> compact Air frames
  -> optional FEC and interleaving
  -> conventional or experimental modem profile
```

The receiver pipeline is:

```text
RF or network samples
  -> conventional synchronization and demodulation
  -> optional bounded LLR correction
  -> FEC decoder
  -> Air frame CRC and replay checks
  -> semantic envelope authentication
  -> deterministic critical state reconciliation
  -> authority and causal utility gate
```

### Compact frame

The C codec currently implements a frame of at most 256 wire bytes:

| Field | Size | Purpose |
|---|---:|---|
| Magic plus version | 1 byte | Rejects unrelated and incompatible data |
| Profile plus flags | 1 byte | Names the physical profile plus ACK request, FEC, control, and signed-envelope flags |
| Stream identifier | 2 bytes | Separates replay and sequence domains |
| Sequence | 2 bytes | Duplicate and replay defense |
| Fragment index plus count | 2 bytes | Reassembles bounded messages |
| Class plus priority | 1 byte | Schedules mission value rather than treating every byte equally |
| Payload length | 1 byte | Bounds parsing |
| State tag | 2 bytes | Fast divergent state detection |
| Payload | 0 through 240 bytes | Semantic envelope fragment |
| CRC32C | 4 bytes | Detects corruption before semantic parsing |

All multibyte wire integers are big endian. A maximum payload frame has 6.25
percent framing overhead before FEC. A 64 byte payload occupies 80 wire bytes,
which is about 2.13 seconds at 300 bit per second before FEC and modem overhead.

### Semantic envelope

The assembled message contains source identity, epoch, message identity, a 64
bit logical sequence, class, priority, a 128 bit critical state hash, body
length, and optional authentication. The Rust `LMAD` version 1 body implements
typed field identifiers and deterministic values, base and result hashes, and
separate quantized residual slots. It does not yet encode entity identifiers,
units, observation time, confidence policy, provenance references, or authority.
Those remain required upper-layer schema fields before a received update can
be promoted into an authoritative WorldGraph fact.

The state hash is not a substitute for the symbolic fields. It is a divergence
alarm: if hashes disagree, peers request a deterministic reconciliation rather
than inventing missing state. Active peers periodically send compact state hashes
even when no delta is due. The interval is selected by mission risk and is part
of benchmark configuration and evidence.

Frame payload test points are 16, 32, 64, 128, and 240 bytes; a maximum frame is
256 bytes including its 16 byte header and CRC. Larger semantic messages are
bounded to 32 fragments. The scheduler should send the smallest delta whose
expected marginal task value exceeds its airtime, energy, and collision cost.

### Profiles

| Profile | Intended envelope | Current implementation status |
|---|---|---|
| WIFI | UDP over certified WiFi | ESP32 task implemented, not hardware validated |
| BLE | GATT over certified BLE | ESP32 fragmentation implemented and host tested, not hardware validated |
| AFSK HF | External audio radio path | Portable modem implemented in C simulation, no live radio validation |
| AFSK VHF | External packet TNC or audio path | KISS bridge and portable modem implemented, no live radio validation |
| CPFSK | Experimental narrowband modem | Portable simulation implemented, no live radio validation |
| BPSK IQ | External SDR IQ path | Portable simulation implemented, no live SDR validation |
| AM audio pipe | Receive audio from an external AM receiver | I2S hook implemented, no tuner or RF implementation |
| FM audio pipe | Receive audio from an external FM receiver | I2S hook implemented, no tuner or RF implementation |

AM and FM profiles do not authorize broadcast transmission. They describe
baseband pipes. A compliant external radio and responsible operator own tuning,
power, bandwidth, PTT, antenna, spectral purity, and channel access.

## Invariants

1. Critical entities, quantities, authority changes, and safety actions use a
   deterministic public representation.
2. Learned latents may carry residual evidence but never silently replace a
   critical symbolic value.
3. A receiver rejects malformed length, invalid CRC, duplicate fragments,
   replayed sequences, failed authentication, and inconsistent state.
4. The embedded codec performs no heap allocation, no tuning, and no radio
   keying. The caller owns buffers and hardware authority.
5. Authentication proves origin when keys are provisioned. It does not obscure
   meaning and must not be described as encryption.
6. Amateur profiles use a publicly documented codec. Canadian and United
   States configurations reject payloads marked encrypted.
7. A semantic scheduler is optimized against task utility per second, not byte
   throughput alone.
8. Provenance and confidence are data, not comments. A receiver that cannot
   resolve required provenance or whose confidence is below policy does not
   promote the fact into critical state.

## Consequences

The first product gate is at least ten times fewer transmitted semantic bytes
than a deterministic full state baseline at equivalent task accuracy. That
reduction must be established before attributing any gain to a neural physical
layer. No such measurement has been completed in this workspace.

The 256 byte total frame cap makes the implementation feasible on ESP32 and on slow
links. It also forces model state into explicit deltas. Large tensors, raw CSI,
audio recordings, images, and firmware images remain out of band.

CRC, FEC, signature, and state hash solve different failure modes. Removing one
because another exists is a category error. CRC detects random corruption. FEC
repairs some channel errors. A signature authenticates an envelope. A state hash
detects world model divergence.

The strongest failure mode is semantic divergence with valid transport checks.
The mitigation is deterministic critical state, compact state hashes in every
semantic message, and periodic reconciliation from a known checkpoint.

## Verification state

Portable frame, FEC, modem, replay, and reassembly logic is implemented under
`c/**`. Rust implements `LMAD` typed symbolic deltas and residual slots. C and
Rust lock the outer frame and `LMS1` envelope to shared golden vectors. ESP32
pure fragmentation, CRC16, KISS, portable codec contract, and transmit policy
tests compile with strict warnings and pass. No ESP32 binary or RF path has been
validated on hardware in this workspace. The ESP IDF 6.0.2 CI gate builds the
complete ESP32 S3 image and runs its size report, but no board has been flashed
and no RF path has been exercised. The protocol is not accepted as an over the
air system until ADR 014 passes. Entity
and units schemas, provenance resolution, confidence policy, authority binding,
periodic reconciliation, and the cross-language `LMAD` body codec remain
integration work. The frame class reserves acknowledgement and control messages,
but a semantic knowledge request body and its closed-loop scheduler are not yet
implemented.
