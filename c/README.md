# LatentMesh Air C11 core

This directory is a portable, bounded C11 transmitter and receiver for
LatentMesh Air. It produces bytes, PCM samples, or baseband IQ. It never keys a
transmitter, selects a frequency, controls RF power, or bypasses the rules of a
radio service. A legal and hardware specific transport owns those operations.

The implementation has no dynamic allocation. Callers own all state and output
buffers. The hot path has deterministic upper bounds: 256 bytes per physical
frame, 32 fragments per logical message, two concurrent reassembly slots by
default, and a 64 message replay window per tracked stream.

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
./build/latentmesh-air-loopback hf "semantic delta"
./build/latentmesh-air-bench
```

For AddressSanitizer and UndefinedBehaviorSanitizer:

```sh
cmake -S . -B build-sanitize -DLM_AIR_ENABLE_SANITIZERS=ON
cmake --build build-sanitize
ASAN_OPTIONS=detect_leaks=0 ctest --test-dir build-sanitize --output-on-failure
```

When included from ESP IDF, the same `CMakeLists.txt` calls
`idf_component_register`. The only platform dependency is the C math library.

## Physical frame contract

Every integer is unsigned and big endian. Bits in each byte and in FEC buffers
are most significant bit first. The frame is exactly `16 + payload_len` bytes.

| Offset | Size | Field | Invariant |
|---:|---:|---|---|
| 0 | 1 | magic and version | `0xA1` |
| 1 | 1 | flags and profile | flags in high nibble, profile in low nibble |
| 2 | 2 | stream ID | big endian |
| 4 | 2 | sequence | big endian, shared by all message fragments |
| 6 | 1 | fragment index | zero based and less than count |
| 7 | 1 | fragment count | 1 through 32 |
| 8 | 1 | class and priority | class in high nibble, priority in low nibble |
| 9 | 1 | payload length | 0 through 240 |
| 10 | 2 | state tag | first two bytes of the full state hash |
| 12 | N | payload | opaque fragment bytes |
| 12 + N | 4 | CRC32C | Castagnoli CRC over header and payload, big endian |

Profiles are WiFi `0`, BLE `1`, HF BPSK `2`, HF AFSK `3`, VHF AFSK `4`,
VHF CPFSK `5`, AM audio pipe `6`, FM audio pipe `7`, and ham packet pipe `8`.

The flag nibble is ACK requested `0x1`, convolutional FEC `0x2`, control
`0x4`, and signed semantic envelope `0x8`. Classes are telemetry `0`, state
delta `1`, ACK `2`, control `3`, and diagnostic `4`.

Canonical physical frame vector:

```text
a1321234010200011f04beefdeadbeefc20abd05
```

It represents HF BPSK, ACK requested plus FEC, stream `0x1234`, sequence
`0x0102`, fragment zero of one, state delta, priority 15, state tag `0xBEEF`,
payload `DEADBEEF`, and CRC32C `0xC20ABD05`.

## LMS1 semantic envelope

The fragmented physical payload is the deterministic LMS1 envelope below. Its
body is transport neutral and opaque to this library. A body may be a canonical
LMAD SemanticDelta, a compact symbolic update, or another application payload.

| Offset | Size | Field | Invariant |
|---:|---:|---|---|
| 0 | 4 | magic | ASCII `LMS1` |
| 4 | 1 | version | `1` |
| 5 | 1 | semantic flags | bit zero means signed; all other bits must be zero |
| 6 | 1 | class | 0 through 4 and equal to the physical frame class |
| 7 | 1 | priority | 0 through 15 and equal to the physical priority |
| 8 | 4 | source ID | big endian |
| 12 | 4 | epoch | big endian |
| 16 | 4 | message ID | big endian |
| 20 | 8 | logical sequence | big endian |
| 28 | 2 | body length | big endian |
| 30 | 16 | state hash | first two bytes must equal physical state tag |
| 46 | 1 | signature length | zero or 64 |
| 47 | 1 | reserved | must be zero |
| 48 | N | body | opaque application bytes |
| 48 + N | 0 or 64 | signature | application supplied signature |

Canonical unsigned semantic envelope vector:

```text
4c4d53310100010f00000001000000020000000300000000000000040004000102030405060708090a0b0c0d0e0f0000deadbeef
```

This is source 1, epoch 2, message 3, logical sequence 4, state delta,
priority 15, state hash bytes `00` through `0F`, and body `DEADBEEF`.

For a signed envelope, set semantic flag bit zero and signature length 64 before
signing. The canonical signed material is the fixed 48 byte header followed by
the body. The 64 byte detached signature follows the body. The transmitter and
receiver invoke caller supplied signing and verification callbacks. No insecure
default signature algorithm is included, and a signed envelope is rejected if a
verification callback is absent.

## Reliability and receiver policy

The FEC is a terminated constraint length 7, rate one half convolutional code
with octal generators 171 and 133. Hard decision and soft LLR Viterbi decoders
share a bounded caller owned workspace. The rectangular bit interleaver handles
nonmultiple lengths without padding ambiguity.

The receiver reassembles fragments in any order. It rejects inconsistent
metadata, duplicate fragments, CRC failures, malformed envelopes, invalid enum
values, signature failures, and delivered message replays. Sequence comparison
uses serial number arithmetic and a 64 sequence sliding window. Incomplete
reassembly is bounded and evicts the least recently touched slot.

`LM_AIR_RX_REASSEMBLY_SLOTS` and `LM_AIR_RX_REPLAY_STREAMS` are compile time
limits. Reducing the former lowers RAM at the cost of concurrent fragmented
messages. `sizeof(lm_air_rx_t)` should be checked on each embedded toolchain.

## Modems and learned assist

The modem layer provides continuous phase AFSK and CPFSK PCM modulation with
noncoherent tone energy demodulation, plus coherent BPSK float IQ primitives.
The supplied profile settings are framing and robustness defaults, not claims
that a frequency, bandwidth, emission mode, or power is lawful.

The optional learned LLR helper is a bounded five tap adaptive model trained
only from caller supplied known bits. It tracks calibration error, clips all
weights and inputs, and returns the original DSP LLR whenever prediction
confidence is below threshold. This keeps FEC and conventional demodulation as
the authority path.

## Public integration path

Packet transports such as WiFi, BLE, or an external ham modem pass
`lm_air_block_t` bytes to and from `lm_air_tx_poll` and
`lm_air_rx_ingest_block`. Audio applications feed packed block bits into the FSK
modulator and turn demodulated LLR signs back into packed bits. SDR applications
do the same with the BPSK IQ functions and may send soft LLRs directly to the
Viterbi decoder. Hardware framing, preambles, clock recovery, RF tuning, PTT,
and regulatory controls belong in the adapter or firmware layer.
