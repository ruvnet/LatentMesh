# 011. Radio adapters and legal boundary

* Status: Accepted and partially implemented
* Date: 2026-08-23
* Regulatory sources reviewed: 2026-08-23
* Related: [010](010-latentmesh-air-protocol.md), [013](013-esp32-firmware.md), [014](014-benchmark-and-acceptance-method.md)

## Decision

LatentMesh Air separates a codec from permission to radiate. A transport
adapter may create bytes or baseband samples. Only certified licence exempt
hardware or a separately compliant licensed station may turn them into RF.

The repository provides these adapters:

| Adapter | Firmware responsibility | External responsibility | Status |
|---|---|---|---|
| WiFi UDP | Station connection, bounded queues, UDP send and receive | Certified module configuration, network security, interference response | Implemented, not device validated |
| BLE GATT | Advertising, write, notify, fragmentation, CRC16 reassembly | Certified module integration and BLE central | Fragment logic host validated, radio not validated |
| UART KISS | KISS escaping, parsing, queues, central transmit policy | TNC, modem, PTT, frequency, power, occupied bandwidth, channel access | Host logic validated, no TNC tested |
| I2S PCM | 48 kHz mono input and output, bounded PCM queue, receive callback | Codec levels, isolation, modem integration, radio and PTT | Implemented, not hardware validated |
| AM and FM receive | Accept baseband audio from an external receiver | Receiver RF front end and lawful use | Hook only |
| HF and VHF | Move Air frames to an external modem or TNC | Licensed operator and compliant station | No live radio validation |

This project does not implement an AM or FM broadcast transmitter. It does not
turn the ESP32 into an HF or VHF radio. It does not encode a band, channel,
power, antenna gain, or license class because those values are station and
jurisdiction specific and can change independently of the protocol.

## Default deny external transmit policy

`LM_RF_TX_ENABLE` defaults off and covers KISS plus I2S output that could reach a
radio. Enabling the build flag is insufficient. Runtime requires operator
attestation, an assigned call sign, an active hardware interlock when configured,
a public codec flag, no encrypted flag for Canadian or United States amateur
profiles, and periodic clear identification metadata. KISS identification also
requires the configured call sign to occur literally in the outgoing bytes.

Normal WiFi and BLE traffic is not sent through the amateur gate because it uses
the certified ESP32 licence exempt radio stack. It remains subject to equipment
authorization, module integration, RF exposure, and interference rules.

The gate is deliberately incomplete as a legal decision system. Firmware cannot
prove the station is on an authorized frequency, within license privileges,
using permitted power and bandwidth, spectrally clean, or under a valid control
operator. A passing gate means only that repository safety preconditions were
met.

## Canada

The current authoritative constraints used for this design are:

1. The Radiocommunication Regulations section 42 requires an appropriate
   certificate or recognized authorization to operate amateur apparatus.
   Section 44 requires an Advanced Qualification to install or operate a
   transmitter or RF amplifier that is not commercially manufactured. Section
   45 incorporates ISED technical requirements.
2. Section 47 permits only communication with amateur stations, only a code or
   cipher that is not secret, and excludes music, commercially recorded
   material, broadcast programming, and communication supporting industrial,
   business, or professional activity.
3. ISED RBR 4 defines the authorized frequency and bandwidth schedules,
   interference obligations, call sign identification, and power restrictions.
   It states that a Canadian amateur station identifies by transmitting its
   assigned call sign.
4. ISED RIC 3 states that full access below 30 MHz requires Basic with Honours
   or an additional qualifying credential. The exact band schedule in current
   RBR 4 remains controlling.

Sources:

* [Justice Canada, Radiocommunication Regulations sections 42 through 49](https://laws-lois.justice.gc.ca/eng/regulations/sor-96-484/page-3.html)
* [Justice Canada, section 47 nonsecret code and content restrictions](https://laws-lois.justice.gc.ca/eng/regulations/SOR-96-484/section-47.html)
* [ISED RBR 4, Issue 3, frequency, bandwidth, interference, identification, and power](https://ised-isde.canada.ca/site/spectrum-management-telecommunications/en/licences-and-certificates/regulations-reference-rbr/rbr-4-standards-operation-radio-stations-amateur-radio-service)
* [ISED RIC 3, certification and HF privileges](https://ised-isde.canada.ca/site/spectrum-management-telecommunications/en/licences-and-certificates/radiocom-information-circulars-ric/ric-3-information-amateur-radio-service)

The learned semantic representation creates a material uncertainty. A public
algorithm is not automatically lawful merely because source code exists. The
test is whether the code or cipher is not secret, and purpose plus practical
decodability can matter. Therefore amateur operation requires a published wire
specification and decoder, deterministic critical facts, no encryption, and a
control operator who has reviewed the mode. Experimental ambiguity should be
resolved with ISED before transmission, or tested through a lawful experimental
authorization rather than assumed into compliance.

## United States

The current authoritative constraints used for this design are:

1. 47 CFR 97.113 prohibits messages encoded for the purpose of obscuring their
   meaning, with narrow exceptions stated in that rule. It also restricts
   pecuniary communications, broadcasting, and regular communication that could
   reasonably use other radio services.
2. 47 CFR 97.119 requires the assigned call sign at the end of each communication
   and at least every ten minutes during a communication. The firmware default
   interval is nine minutes to leave timing margin.
3. 47 CFR 97.109 requires a control point and constrains automatic control to
   stations permitted elsewhere in Part 97. An ESP32 task is not permission for
   automatic unattended operation.
4. 47 CFR 97.307 requires no more bandwidth than necessary, confines emissions
   to the available band or segment, and requires suppression of spurious
   emissions. The external transmitter and control operator own these duties.

Sources:

* [eCFR 47 CFR 97.113, prohibited transmissions](https://www.ecfr.gov/current/title-47/chapter-I/subchapter-D/part-97/subpart-B/section-97.113)
* [eCFR 47 CFR 97.119, station identification](https://www.ecfr.gov/current/title-47/chapter-I/subchapter-D/part-97/subpart-B/section-97.119)
* [eCFR 47 CFR 97.109, station control](https://www.ecfr.gov/current/title-47/chapter-I/subchapter-D/part-97/subpart-B/section-97.109)
* [eCFR 47 CFR 97.307, emission standards](https://www.ecfr.gov/current/title-47/chapter-I/subchapter-D/part-97/subpart-D/section-97.307)

The firmware uses a public codec requirement and blocks the encrypted payload
flag. Those are conservative engineering controls, not an FCC determination
that every learned codebook or model latent is permissible. A mode whose
practical purpose or effect is concealment stays off amateur spectrum.

## Licence exempt WiFi and BLE

United States Part 15 operation is conditional on causing no harmful
interference and accepting interference. The operator must stop if the FCC
requires cessation until the cause is corrected. Canada applies applicable RSS
standards, including RSS 247 for digital transmission and licence exempt local
area devices, together with RSS Gen requirements. A product must preserve the
module approval conditions, approved antenna configuration, labeling, RF
exposure assessment, and host integration requirements.

Sources:

* [eCFR 47 CFR 15.5, general conditions of operation](https://www.ecfr.gov/current/title-47/chapter-I/subchapter-A/part-15/subpart-A/section-15.5)
* [FCC equipment authorization overview](https://www.fcc.gov/oet/ea/rfdevice)
* [ISED RSS 247, digital transmission, frequency hopping, and licence exempt local area devices](https://ised-isde.canada.ca/site/spectrum-management-telecommunications/en/devices-and-equipment/radio-equipment-standards/radio-standards-specifications-rss/rss-247-digital-transmission-systems-dtss-frequency-hopping-systems-fhss-and-licence-exempt-local)
* [ISED RSS Gen, general radio apparatus compliance requirements](https://ised-isde.canada.ca/site/spectrum-management-telecommunications/en/devices-and-equipment/radio-equipment-standards/radio-standards-specifications-rss/rss-gen-general-requirements-compliance-radio-apparatus)

## Consequences

The repository can safely support lab simulation, receive only work, certified
WiFi and BLE modules, and interfaces to lawful stations without pretending a
software flag grants spectrum rights. Product deployments need jurisdiction
review, module integration evidence, secure provisioning, RF exposure work, and
hardware measurements.

The dominant failure mode is a developer treating an experimental semantic
codec as encryption compatible with amateur service or treating the external RF
gate as a license oracle. The fix is public deterministic critical fields, open
conformance tools, default deny transmit, physical interlock, and regulator
review before an unfamiliar learned mode is put on air.
