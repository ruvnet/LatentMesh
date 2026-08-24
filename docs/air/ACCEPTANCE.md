# LatentMesh Air acceptance contract

This document separates protocol implementation evidence from the larger
research claim. Passing software tests proves that the implementation is
internally consistent. It does not prove superior radio performance.

## Inputs

Each benchmark record contains the exact source state, receiver prior state,
radio profile, payload budget, channel seed, channel parameters, transmitter
power reference, bandwidth, conventional baseline, and implementation commit.

## Outputs

The runner records encoded bytes, airtime, good frames, critical state matches,
task utility, decoder confidence, fallback decisions, CPU time, peak memory,
and the content hash of every fixture and result.

## Software gates

| Gate | Pass condition |
|---|---|
| C and Rust parity | Both implementations encode the shared golden vectors byte for byte and decode each other's output |
| Malformed input | Truncation, overflow, invalid fragment counts, CRC failure, and replay are rejected without unbounded allocation |
| Channel loopback | Packet, PCM, and IQ paths recover the expected state across frozen AWGN, fading, burst loss, and frequency offset cases |
| Neural fallback | Every low confidence or out of distribution case selects classical likelihoods or rejects the frame |
| ESP32 host logic | BLE fragments, KISS escaping, queues, replay state, and transmit policy pass without ESP-IDF hardware |
| ESP32 target compile | ESP IDF 6.0.2 builds the ESP32 S3 image and produces a size report |
| Build quality | Rust formatting, linting, tests, C warnings, sanitizers, firmware host tests, and dependency checks pass |

## Hardware in the loop gate

Use at least three propagation conditions not used to tune the adaptive
receiver. Compare LatentMesh Air against a conventional modem with identical
occupied bandwidth, transmit power, antenna path, payload meaning, and trial
schedule. Randomize trial order and retain failed trials.

The research acceptance test has two independent stages. The semantic layer
must pass before receiver gain is reported as an additional multiplier.

1. **Semantic transport:** transmit at least ten times fewer total bytes than a
   full deterministic state baseline while downstream task accuracy remains
   within a predeclared one percentage point equivalence margin.
2. **Neural physical layer:** with the exact same semantic messages, occupied
   bandwidth, transmit power, antenna path, and trial schedule, deliver at least
   two times more task weighted useful information per second or per joule than
   the conventional receiver under degraded channels.
3. **Critical agreement:** critical WorldGraph state agreement is at least 99
   percent overall and at least 97 percent in every individual held out channel
   condition.
4. **Statistical evidence:** the bootstrap 95 percent interval must exclude no
   improvement for the reported semantic reduction and receiver gain, and the
   task accuracy interval must remain inside the declared equivalence margin.

The neural receiver must also demonstrate that disabling its confidence gate
worsens either critical agreement or frame error rate. Otherwise the adaptive
component has not earned its complexity. A semantic reduction result is not
evidence of a better physical layer, and a neural receiver result is not
evidence of semantic compression.

## Claim labels

Use exactly one evidence label for each public metric:

* `unit validated`
* `target compiled`
* `simulated`
* `software loopback`
* `hardware in loop`
* `over the air`

Never translate a simulated or software loopback result into an over the air
claim.
