# 013. ESP32 firmware architecture

* Status: Accepted and implemented without hardware validation
* Date: 2026-08-23
* Related: [010](010-latentmesh-air-protocol.md), [011](011-radio-adapters-and-legal-boundary.md), [012](012-neural-receiver-fallback.md), [014](014-benchmark-and-acceptance-method.md)

## Decision

Target ESP IDF 6.0.2 and C11 source compatibility. Keep the portable, allocation free Air codec
under `c/**`. Keep hardware, RTOS, and policy code under `firmware/esp32/**`.
The component boundary is the public `<latentmesh_air.h>` API. The firmware
loads `c/**` as an ESP IDF component and must not include private source files.

The endpoint uses one bounded transmit queue per transport and one bounded
receive queue for application consumption:

```text
application Air frame
  -> radio bus fanout
       -> WiFi queue -> UDP task
       -> BLE queue -> fragment -> notify task
       -> KISS queue -> policy -> UART task -> external TNC

WiFi receive ----\
BLE write --------> receive queue -> portable Air receiver -> application hook
KISS receive -----/

I2S receive -> PCM hook -> portable modem or external inference
I2S transmit queue -> policy -> codec output -> external audio interface
```

Queue capacity defaults to eight complete frames. The maximum transport object
defaults to 512 bytes so a 256 byte Air frame and adapter metadata fit without
heap allocation. Queue saturation increments a drop counter and rejects the new
object. A radio callback never waits indefinitely for application work.

Each physical link uses `LM_STREAM_ID + link_index` as its replay and fragment
domain. This prevents profile specific fragment boundaries from corrupting a
redundant multi link delivery. Applications deduplicate identical semantic
message identifiers after verification.

## Components

| Component | Responsibility | Trust boundary |
|---|---|---|
| `c/**` | Frame, CRC32C, FEC, interleave, modem, replay, semantic reassembly, bounded likelihood assist | Pure caller owned buffers, no hardware authority |
| `lm_air_pipeline` | Canonical C transmitter, per link profiles, receiver task, application callback, optional crypto hooks | Only verified reassembled messages reach the application hook |
| `lm_air_radio` | Queue allocation, fanout, source tagging | Rejects invalid object bounds |
| `lm_air_wifi` | WiFi station lifecycle and UDP socket | Local IP network is untrusted input |
| `lm_air_ble` | GATT service, writes, notifications | BLE central is untrusted input |
| `lm_air_ble_frag` | Out of order bounded reassembly and CRC16 | Pure logic, host tested |
| `lm_air_kiss` | KISS data command encoding and decoding | Pure logic, host tested |
| `lm_air_kiss_uart` | UART ownership and external radio policy | TNC output can cause RF transmission |
| `lm_air_i2s` | PCM input callback and guarded PCM output | Audio output may trigger VOX and cause RF transmission |
| `lm_air_policy` | Build, operator, call sign, public codec, identification, encryption, interlock checks | Additional safeguard, not a license oracle |
| `lm_air_metrics` | Atomic counters | Observability only |

## Transport decisions

WiFi uses one nonblocking UDP socket in a task. UDP was chosen for minimal
latency and simple SDR or edge host integration. Delivery, ordering, peer
authentication, and confidentiality are not provided by UDP. Air sequence,
replay, CRC, and optional signature checks remain mandatory.

BLE exposes one custom write and notify characteristic. A ten byte fragment
header carries message identifier, index, count, total length, and complete
message CRC16. The receiver supports out of order fragments with a 64 bit bitmap
and rejects inconsistent metadata. One BLE connection is configured by default
to cap RAM and simplify identity.

KISS supports data command zero and ignores control commands. The TNC owns the
link modem, CSMA behavior, PTT, and RF. The bridge does not infer that a connected
TNC is lawful or correctly configured.

I2S uses 48 kHz, signed 16 bit, mono blocks of 240 samples. That is a five
millisecond queue unit. The receive hook is weak so an application may attach
the portable modem without changing the driver. There is intentionally no PTT
GPIO in this firmware. Audio connected to a VOX radio still creates transmission
risk and therefore passes through the same external policy.

## Safe configuration

The committed configuration has blank WiFi credentials, external RF transmit
disabled, no operator attestation, no call sign, and I2S disabled. BLE and KISS
receive infrastructure are compiled. Examples include WiFi plus BLE, KISS
receive only, and an amateur transmit template that still requires a real call
sign and a physical active high interlock.

Secrets do not belong in committed `sdkconfig` files. A production node needs
secure provisioning, secure boot, flash encryption, signed OTA, key rotation,
and rollback. Those controls are not implemented here and must be completed
before deployment with sensitive state.

## Hardware scope

ESP32 S3 provides 2.4 GHz WiFi, BLE, UART, and I2S. It does not provide a general
HF or VHF RF chain, an AM or FM tuner, or an SDR ADC and DAC. External hardware
is required. Radio connectors are not electrically standardized. Level shifting,
attenuation, galvanic isolation, filtering, dummy load tests, and spectral
measurement are deployment responsibilities.

## Implementation evidence

The native suite compiles `lm_air_ble_frag.c`, `lm_air_crc.c`, `lm_air_kiss.c`,
and `lm_air_policy.c` as strict C11 with optimization, all warnings enabled,
warnings treated as errors, and pedantic diagnostics. It verifies CRC, out of
order BLE reassembly, corrupt BLE rejection, KISS escape round trip, encrypted
amateur rejection, call sign validation, first identification, and interval
expiry.

A second native test compiles all portable C sources and exercises the firmware
contract: profile defaults, transmitter initialization, emitted wire blocks,
receiver ingestion, semantic reassembly, and callback data. Both native binaries
print PASS.

Every ESP specific C translation unit also passed strict compiler syntax checks
against narrow local interface stubs. GitHub Actions then sourced the official
ESP IDF 6.0.2 container environment and completed an ESP32 S3 build and size
report. That compiled the portable component, component graph, NimBLE, WiFi,
UART and I2S interfaces together. The reported total image size was 491,387
bytes; the application binary occupied 0x77ff0 bytes and left 75 percent of the
smallest configured application partition free.

No board was flashed. Target execution, peripherals, latency, energy, RF output
and receive performance remain unvalidated. A successful target compile closes
the SDK signature gap, not the hardware gate.

Espressif identifies 6.0.2 as the current stable release in its
[official release list](https://github.com/espressif/esp-idf/releases/tag/v6.0.2).
The component manifest accepts 6.0.2 and rejects 6.1 prereleases so a moving SDK
cannot silently change the validation target.

## Consequences

The endpoint remains small, auditable, and useful with several classes of radio
without duplicating the Air codec. It also means higher order semantic selection,
WorldGraph reconciliation, large neural inference, and long lived memory execute
on a host unless an MCU benchmark justifies moving a bounded part onto the node.

The biggest failure mode is a configuration that compiles but mismatches a
specific board, ESP IDF minor release, codec voltage, or TNC UART level. The fix
path is the hardware matrix and compile plus loopback gates in ADR 014 before an
antenna is attached.
