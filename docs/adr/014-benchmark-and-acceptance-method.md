# 014. Benchmark and acceptance method

* Status: Accepted as the validation contract, not yet passed over air
* Date: 2026-08-23
* Related: [003](003-causal-edge-verification.md), [010](010-latentmesh-air-protocol.md), [012](012-neural-receiver-fallback.md), [013](013-esp32-firmware.md)

## Decision

Separate protocol correctness, simulated link performance, hardware transport,
RF compliance, receiver gain, and semantic task value. A result may advance only
the stage it measures. Simulation never counts as an over the air claim.

## Stage gates

| Gate | Input | Required output | Current state |
|---|---|---|---|
| C and Rust conformance | Shared golden and malformed vectors | Identical accepted messages, rejection codes, CRC, replay, and fragmentation behavior | Outer frame and LMS1 envelope golden vectors match byte for byte; full cross-language malformed-case parity remains a CI expansion |
| ESP32 pure logic | Native C compiler | BLE, KISS, CRC, and policy suite passes strict warnings | Passed in this implementation workspace |
| ESP IDF compile | ESP IDF 6.0.2 | Clean build for ESP32 S3, size report, no warnings | Blocked because ESP IDF is absent here |
| Hardware transport | Two ESP32 boards plus host | WiFi, BLE, KISS, and I2S loopback with measured loss and latency | Not run |
| Conducted modem | SDR or radio service monitor through attenuation | BER, PER, synchronization, occupied bandwidth, spectral mask, power | Not run |
| Over the air channel | Licensed lawful station and recorded channel set | Blind BER, PER, fallback, and task metrics across unseen conditions | Not run |
| Semantic compression | Paired agents and deterministic WorldGraph oracle | At least ten times fewer semantic bytes at equivalent task accuracy and 99 percent critical agreement | Deterministic Rust fixture exceeds the byte gate with exact task state; paired real-agent task benchmark not run |
| Neural physical layer | Identical semantic payload through conventional and neural receiver conditions | At least two times useful information per airtime or energy from the neural physical layer | Frozen Rust phase-inversion simulation exceeds the gate; hardware and unseen propagation evidence not run |

## Test matrix

Run payload sizes 16, 32, 64, 128, and 240 bytes. Run WiFi, BLE, AFSK HF,
AFSK VHF, CPFSK, and BPSK IQ where hardware exists. Each physical profile uses
at least these conditions:

| Variable | Minimum levels |
|---|---|
| Signal level | Five points spanning clean link to below threshold |
| Frequency offset | Nominal plus two positive and two negative offsets appropriate to profile |
| Doppler or clock error | None, moderate, severe |
| Channel | AWGN, multipath, burst fading, impulsive interference, adjacent signal |
| Device | At least two transmitter and two receiver units for hardware claims |
| Time and location | Separate sessions, with at least one blind session excluded from adaptation |

HF evaluation additionally separates ground wave, single hop, multihop, day,
night, and disturbed conditions when available. VHF evaluation includes line of
sight, mobile multipath, and obstruction. Do not pool these into one average
without per condition results.

## Baselines

1. Conventional DSP modem, fixed FEC, full deterministic state packet.
2. Conventional DSP modem, fixed FEC, sparse semantic delta.
3. Conventional DSP plus bounded likelihood assist, fixed FEC, same semantic
   delta.
4. Adaptive FEC or constellation component after stages one through three pass.
5. Fully learned experimental waveform only after all interoperable conditions
   have evidence.

This isolates value from semantic compression versus receiver improvement. A
two times gain caused only by sending less state is still valuable, but it is
not evidence of a better physical receiver.

## Metrics

Transport metrics are payload bytes, total wire bytes, airtime, retransmissions,
queue drops, end to end latency, energy per delivered message, BER before FEC,
PER after FEC, CRC rejection, replay rejection, and authentication failure.

Receiver metrics are post FEC PER, path metric, calibrated confidence, learned
use rate, deterministic fallback rate, disagreement rate, inference latency,
memory, and energy. Report both average and worst condition deltas against DSP.

Semantic metrics use a deterministic task oracle. For semantic field `i`, let
`w_i` be its mission weight and `c_i` equal one only when the received value
matches the oracle within its declared tolerance. For observation window `T`:

```text
useful_information_per_second = sum_i(w_i * c_i) / T
```

The semantic reduction ratio is:

```text
semantic_reduction = baseline_full_state_wire_bytes
                     / Air_semantic_delta_wire_bytes
```

Task accuracy must remain inside a prespecified equivalence margin. A practical
default is an absolute reduction no greater than one percentage point, but each
task declares its margin before testing. Critical agreement has no equivalence
margin and must meet its separate threshold.

The physical layer gain is measured only after semantic bytes are fixed:

```text
phy_gain = neural_PHY useful_information per airtime_or_energy
           / conventional_PHY useful_information per airtime_or_energy
```

Critical state agreement is the fraction of critical WorldGraph facts whose
entity, value, units, time, provenance, and authority all agree. A matching hash
without matching decoded fields does not count.

## Acceptance thresholds

Acceptance is staged so semantic transport value is not confused with receiver
value:

1. Semantic reduction is at least 10.0 at equivalent task accuracy using the
   same conventional physical layer.
2. Critical WorldGraph agreement is at least 0.99, including entity, value,
   units, time, provenance, confidence policy, and authority.
3. With the semantic payload held identical, neural physical layer gain is at
   least 2.0 at identical occupied bandwidth and transmit power, measured as
   useful information per airtime or useful information per joule.
4. Both stages hold on every named blind channel family, not only the pooled
   average.
5. The lower bound of a 95 percent bootstrap confidence interval exceeds 10.0
   for semantic reduction, 2.0 for physical layer gain, and 0.99 for critical
   agreement.
6. No learned condition exceeds the conventional receiver PER by more than one
   percentage point at any validated operating point without falling back.
7. All conducted RF measurements meet the current local band, power, occupied
   bandwidth, and unwanted emission rules for the tested station.

Suggested engineering gates before the research claim are zero parser memory
errors under sanitizer and fuzz runs, zero replay acceptance, zero external RF
frames when any policy condition is false, less than one queue drop per million
frames at the declared service rate, and p95 processing time below one tenth of
the shortest symbol or transport deadline assigned to the stage.

## MetaHarness optimization protocol

The current MetaHarness implementation varies semantic byte budget, FEC mode,
interleave rows, and learned confidence threshold. Future evaluators may add a
priority threshold, bounded learning rate, and transport queue depth. The
harness may not mutate regulatory limits, call sign policy, encryption policy,
interlock behavior, critical field determinism, replay checks, signature checks,
or the blind test corpus.

For each candidate:

1. Freeze code, configuration, model identity, codebook identity, random seeds,
   and test corpus hash.
2. Run conventional and candidate conditions with paired channel realizations.
3. Compute quality, gain, latency, energy, fallback, and safety metrics.
4. Reject any candidate with a safety regression, missing evidence, or a worse
   blind condition even if its average reward rises.
5. Promote only when it beats the parent on the declared objective, safety is at
   least 0.95, and no protected metric regresses.
6. Store the full evidence as a signed benchmark receipt. Do not promote from a
   dashboard summary.

The objective should be constrained rather than a single opaque score:

```text
maximize useful_information_per_airtime_or_energy
subject to critical_agreement >= 0.99
           semantic_reduction >= 10.0
           regulatory_compliance = true
           policy_violations = 0
           energy <= device_budget
           p95_latency <= mission_deadline
```

## Reproducibility artifacts

Every reported run records source commit, firmware binary hash, board revision,
ESP IDF version, radio and TNC models, firmware versions, antenna or conducted
path, frequency, mode, power, bandwidth, sample rate, channel trace hash,
configuration, model and codebook hashes, environment notes, raw events, and
analysis script version.

## Current evidence

The command `make test` in `firmware/esp32/host_tests` compiled the pure logic
with GCC using C11, optimization, all common warnings, warnings as errors, and
pedantic mode. It printed `latentmesh-air ESP32 pure logic: PASS`. The same
command compiled all portable C sources and an exact transmitter to receiver
firmware contract loopback. It printed
`latentmesh-air ESP32 portable C contract: PASS`.

The portable C host benchmark compared a 1,800 byte full state with a 64 byte
delta through identical framing, FEC, and interleaving. It reported 37,340 air
bits across 29 blocks versus 2,328 air bits across two blocks, a 16.04 times
reduction. The C codec does not measure downstream task accuracy, so this is a
framing result rather than a passed semantic acceptance claim.

The Rust stage-one fixture compares the repository's 65,536 byte dense Int8
reference with a 173 byte transmitted semantic update and reconstructs the
exact deterministic critical state, a 378.82 times fixture reduction. The Rust
stage-two phase-inversion fixture delivers 448 of 512 correct assisted bits
versus 128 of 512 classical bits at identical symbols and simulated energy, a
7.00 times result. Both are deterministic simulation evidence. Neither is
hardware-in-the-loop or over-the-air evidence and neither establishes channel
generalization.

The JavaScript MetaHarness stage receipt is also explicitly labelled
`simulated`. Across its frozen 64-case degraded suite, the root policy reports a
222.16 times reference-byte reduction and 99.70 percent mean critical agreement,
but only 1.07 times neural physical-layer gain. The wider suite therefore fails
the 2.0 target even though the narrow Rust phase-inversion fixture passes it.
The receipt sets overall acceptance to false until the physical-layer target,
hardware evidence, confidence intervals, and blind propagation evidence all
pass.

The exact gap is that `idf.py` is not installed in this workspace. No target
binary, flash image, size report, NimBLE link, WiFi link, UART device, I2S codec,
or RF result was produced. The acceptance status is therefore protocol logic
passed, embedded and over the air acceptance not passed.

The dominant benchmark failure mode is leakage: adapting on the same channel
recordings used for the headline result. The fix is a sealed blind corpus owned
by the harness evaluator, with hardware sessions and locations unavailable to
the transmitter, receiver, and optimizer until final scoring.
