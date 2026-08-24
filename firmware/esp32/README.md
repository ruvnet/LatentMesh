# LatentMesh Air ESP32 endpoint

This endpoint targets ESP IDF 6.0.2 for the portable `c/**` LatentMesh Air
codec. It moves complete Air wire frames over WiFi UDP, fragments them over a
BLE GATT characteristic, bridges them to an external KISS TNC, and exposes an
I2S PCM hook for an external audio interface. The ESP32 does not become an HF,
VHF, AM, or FM radio by flashing this firmware.

## Honest status

| Capability | Built | Validation in this repository | Hardware validation |
|---|---:|---|---|
| Portable Air codec dependency | Yes | Firmware contract loopback compiles all C sources and passes | None |
| WiFi station plus UDP send and receive task | Yes | ESP IDF 6.0.2 target compile | None |
| BLE GATT write, notification, fragmentation and reassembly | Yes | Native logic tests and ESP IDF 6.0.2 target compile | None |
| UART KISS send and receive bridge | Yes | Native logic tests and ESP IDF 6.0.2 target compile | None |
| I2S 48 kHz mono PCM input and output hook | Yes | ESP IDF 6.0.2 target compile | None |
| External RF transmit policy | Yes | Native policy tests cover disabled, encrypted, call sign, identification due, and interlock cases | None |
| AFSK, CPFSK, BPSK modem primitives | In `c/**` | Native simulated loopback tests | None through a radio |
| Neural receiver | In `c/**` as a bounded residual adapter | Native simulation only | Not validated on ESP32 or live IQ |

No on device or over the air result is claimed. A successful host test proves
deterministic logic, not RF performance or regulatory compliance.

## Architecture

The portable C component owns framing, CRC32C, FEC, modem math, replay control,
semantic message reassembly, and the bounded neural residual interface. This
firmware owns hardware adapters and FreeRTOS queues. The boundary is a public
`<latentmesh_air.h>` API. Firmware code never includes `c/src` internals.

Each adapter has a bounded transmit queue. All received wire frames enter one
bounded queue and the portable receiver performs CRC, replay, fragment, semantic
envelope, and optional authentication checks before invoking the application
hook. Queue saturation drops the new frame and increments a metric rather than
blocking a radio callback. Each link gets an independent stream identifier so
different BLE and WiFi fragment sizes cannot collide in one replay domain. The
semantic message identifier lets an application deduplicate redundant delivery.

WiFi and BLE are transports supplied by the certified ESP32 module. KISS and
I2S are only baseband bridges to external equipment. KISS does not key a radio
itself. I2S has no PTT implementation. A radio or TNC with VOX can still radiate
audio, so the I2S transmit queue uses the same default deny policy.

## Build

Install the current stable ESP IDF 6.0.2, export its environment, then run:

```sh
cd firmware/esp32
idf.py set-target esp32s3
idf.py menuconfig
idf.py build
idf.py -p /dev/ttyUSB0 flash monitor
```

The root `c/**` directory is loaded as an ESP IDF component automatically. A
packaged copy can be selected with:

```sh
idf.py -DLATENTMESH_C_COMPONENT=/absolute/path/to/c build
```

The version constraint is `>=6.0.2,<6.1.0`. Espressif identifies 6.0.2 as the
current stable release in its [official release list](https://github.com/espressif/esp-idf/releases/tag/v6.0.2).
The local workspace does not contain that SDK. GitHub Actions uses the official
ESP IDF 6.0.2 container and has completed the ESP32 S3 target build and size
report. The total image size was 491,387 bytes. No board was flashed.

The committed default has no WiFi credentials and external RF transmission is
off. Do not commit credentials in `sdkconfig`. Use a local defaults file, NVS
provisioning in a downstream product, or menuconfig.

Native deterministic tests do not require ESP IDF:

```sh
cd firmware/esp32/host_tests
make test
```

An application sends through the canonical portable transmitter rather than
constructing wire bytes directly:

```c
#include "lm_air_pipeline.h"
#include "lm_air_radio.h"

static const uint8_t body[] = "presence=1;confidence=0.99";
lm_air_message_t message = {
    .source_id = 7,
    .epoch = 1,
    .message_id = 42,
    .logical_sequence = 42,
    .class_id = 1,
    .priority = 15,
    .body = body,
    .body_len = sizeof(body) - 1,
};

ESP_ERROR_CHECK(lm_air_pipeline_send(
    &message,
    LM_AIR_LINK_MASK(LM_AIR_LINK_WIFI) | LM_AIR_LINK_MASK(LM_AIR_LINK_BLE),
    LM_AIR_PAYLOAD_PUBLIC_CODEC));
```

Override `lm_air_pipeline_message_hook` to consume a verified reassembled
message synchronously. Copy any body data that must outlive the callback. Pass
sign and verify hooks to `lm_air_pipeline_start` before using authenticated
messages. The included `app_main` starts without keys, so authenticated receives
fail closed and authenticated sends fail rather than silently downgrading.

## Minimal equipment

| Purpose | Item | Notes |
|---|---|---|
| Base node | ESP32 S3 DevKitC 1 or another board using a certified ESP32 S3 module | Exact pins and module approvals vary by board |
| Power and console | Data capable USB cable and a stable 5 V supply | A weak supply causes WiFi resets |
| WiFi test | WPA2 access point and a second UDP endpoint | Default UDP port is 40404 |
| BLE test | Phone or computer capable of custom GATT writes and notifications | This is not a standard BLE serial service |
| Packet radio test | External legal radio plus a TNC that exposes 3.3 V UART KISS | RS232 voltage must never be wired directly to ESP32 GPIO |
| Audio receive test | I2S audio codec or ADC breakout | Select input levels appropriate for radio speaker or discriminator audio |
| Audio transmit test | I2S DAC or codec, isolation transformer, level attenuator, and radio data input | An external TNC is safer for first tests |
| Safety | Dummy load, power meter, spectrum analyzer or service monitor, and an RF transmit interlock switch | Validate into a dummy load before using an antenna |

## Default wiring

The values below are examples for an ESP32 S3 DevKit. Verify the schematic for
the actual board before connecting anything.

| ESP32 signal | Default GPIO | Connect to | Constraint |
|---|---:|---|---|
| UART1 TX | 17 | TNC RX | 3.3 V logic only |
| UART1 RX | 18 | TNC TX | 3.3 V logic only |
| Ground | GND | TNC logic ground | Prefer isolation in permanent radio installations |
| I2S BCLK | 4 | Codec BCLK | Digital audio only |
| I2S WS | 5 | Codec LRCLK or WS | 48 kHz, mono slot |
| I2S DOUT | 6 | Codec DIN | Baseband audio, not RF |
| I2S DIN | 7 | Codec DOUT | Baseband audio, not RF |
| TX interlock | 9 in the template | Switch to 3.3 V when an operator deliberately arms TX | Internal pulldown makes an open switch safe |

Never connect a radio microphone, speaker, PTT, or RS232 connector directly to
an ESP32 pin. Voltage, polarity, grounding, and isolation differ across radios.

## Transport behavior

### WiFi UDP

When an SSID is configured, the station reconnects automatically. One task owns
the UDP socket, sends queued frames to the configured peer, and accepts frames
on the bind port. UDP supplies neither delivery nor peer authentication. Put
bench nodes on an isolated network and use signed Air messages where provenance
matters. Broadcast peer mode is for discovery benches, not deployment.

### BLE GATT

Service UUID ends in `...4c4d41495201`, and its write plus notify
characteristic ends in `...4c4d41495202`. A ten byte fragment header carries
version, message identifier, fragment index, fragment count, total length, and
CRC16. Reassembly accepts fragments out of order, rejects inconsistent metadata,
and verifies the complete payload before publishing it.

### UART KISS

The bridge supports KISS data command zero on ports zero through fifteen and
correctly escapes FEND and FESC. Other KISS control commands are ignored. The
external TNC remains responsible for modulation, channel access, PTT, RF power,
spectral purity, and frequency. The operator remains responsible for every
transmission.

### I2S audio

Receive PCM is delivered to the weak `lm_air_i2s_rx_hook` callback. A downstream
component can feed it to the portable AFSK demodulator. PCM transmit blocks are
bounded to 240 samples and pass through the external RF policy. Identification
metadata cannot prove that audible or digital identification was actually
transmitted. Treat it as an extra interlock, never as a compliance mechanism.

AM and FM broadcast support means receive audio can enter through I2S. This
project does not implement, authorize, or suggest unlicensed AM or FM broadcast
transmission.

## External RF safety invariant

KISS and I2S output is denied unless all of these are true:

1. `LM_RF_TX_ENABLE` was deliberately enabled at build time.
2. `LM_OPERATOR_ATTESTED` is enabled.
3. The configured call sign has letters and digits.
4. The public codec flag is present.
5. The encrypted flag is absent for Canadian and United States amateur profiles.
6. The optional active high hardware interlock is asserted.
7. The first packet, and a packet before the configured interval expires,
   contains clear identification metadata. KISS also verifies that the call sign
   appears literally in the packet bytes.

This gate does not know the tuned band, license privileges, occupied bandwidth,
power, antenna gain, local band plan, or whether the channel is busy. Therefore
passing the gate is necessary for this implementation but never sufficient for
lawful operation.

## Known limitations

1. ESP32 S3 only provides 2.4 GHz WiFi and Bluetooth LE. It has no HF or VHF RF
   chain, no general purpose SDR, and no AM or FM tuner.
2. ESP IDF 6.0.2 CI builds the ESP32 S3 target successfully, but the resulting
   image has not been flashed or exercised on a board. Native pure logic is also
   compiled with strict warnings and tested.
3. No radio, TNC, BLE central, WiFi peer, audio codec, antenna, or propagation
   channel has been tested in this workspace.
4. The firmware does not provision keys, credentials, clock synchronization,
   secure boot, flash encryption, OTA updates, or signed application images.
   Those are mandatory productization work.
5. UDP is unreliable. BLE notifications are not acknowledged at the application
   layer. LatentMesh sequence, replay, FEC, and state hash logic mitigate some
   corruption and duplication but do not create guaranteed delivery.
6. The bounded neural residual API is not a trained universal neural receiver.
   Live IQ adaptation belongs on a more capable host or accelerator until RAM,
   latency, energy, and fallback benchmarks pass on the target MCU.

Regulatory rationale and source links are in
`docs/adr/011-radio-adapters-and-legal-boundary.md`. This is an engineering
safety design, not legal advice.
