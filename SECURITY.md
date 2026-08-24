# LatentMesh security policy

LatentMesh is a research prototype. Do not use it as the sole control path for
safety critical, emergency, medical, aviation, maritime, industrial, or public
safety communication.

## Reporting a vulnerability

Use GitHub private vulnerability reporting for `ruvnet/LatentMesh`. Include the
affected commit, reachable input, expected security invariant, observed result,
and the smallest reproducible test that does not transmit on live radio
spectrum. Do not include credentials, private RF recordings, or personal data.

## LatentMesh Air threat model

The receiver treats every byte obtained from RF, audio, IQ, BLE, WiFi, serial,
and KISS transports as hostile. Its assets are receiver availability, symbolic
state integrity, sender identity, replay freshness, model authority, radio
configuration, and operator control.

| Threat | Required control |
|---|---|
| Oversized or inconsistent lengths | Validate the fixed header and all bounds before allocation, decode, or reassembly |
| Fragment flooding | Fixed sender and assembly limits, expiry, duplicate suppression, and deterministic eviction |
| Corruption | CRC32C after FEC decode; corrupted frames never reach semantic decode |
| Replay or reordering | Sender scoped sequence window; expired frames fail closed |
| Spoofing | Optional signature verification callback and trusted sender policy; CRC is never described as authentication |
| Semantic divergence | Mission critical values use typed deterministic fields and a compact state hash |
| Neural receiver error | Neural output may adjust soft likelihoods only; low confidence falls back to classical DSP |
| Authority escalation | Received state cannot bypass provenance, causal verification, or the existing admission gate |
| Unauthorized RF transmission | ESP32 RF transmit is disabled by default; amateur profiles require explicit enablement and operator configuration |
| Secret disclosure | No secrets in frames, logs, benchmark fixtures, firmware defaults, or CI configuration |

## Security invariants

1. Parsing is bounded before memory use.
2. Integrity, freshness, and policy checks precede semantic interpretation.
3. Critical symbolic state is never generated from an opaque residual.
4. Neural inference cannot grant authority or override an integrity failure.
5. A missing verifier, callsign, radio configuration, or causal verdict reduces
   capability. It never expands it.
6. Production keying and frequency control remain in an allowlisted hardware
   adapter under operator control.

## Cryptography boundary

CRC32C detects channel corruption and is not a security primitive. Digital
signatures authenticate frames where the service and applicable radio rules
permit them. Encryption is intentionally outside the protocol core because
permitted use depends on the service, jurisdiction, and message purpose.

## Release gate

A release cannot be described as hardened unless malformed input tests,
reassembly limits, replay rejection, cross language golden vectors, sanitizer
checks, dependency scans, and the exact receiver fallback tests all pass. A
hardware radio claim additionally requires recorded hardware in the loop
evidence at the declared frequency, power, bandwidth, and channel conditions.
