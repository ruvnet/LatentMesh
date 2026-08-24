#include "latentmesh_air.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition)                                                       \
    do {                                                                       \
        if (!(condition)) {                                                    \
            fprintf(stderr,                                                    \
                    "CHECK failed at %s:%d: %s\n",                            \
                    __FILE__,                                                  \
                    __LINE__,                                                  \
                    #condition);                                               \
            return 1;                                                          \
        }                                                                      \
    } while (0)

static unsigned get_bit(const uint8_t *data, size_t bit) {
    return ((unsigned)data[bit >> 3] >> (7u - (unsigned)(bit & 7u))) & 1u;
}

static void flip_bit(uint8_t *data, size_t bit) {
    data[bit >> 3] ^= (uint8_t)(1u << (7u - (bit & 7u)));
}

static lm_air_status_t ignore_message(
    void *user,
    const lm_air_message_t *message) {
    (void)user;
    (void)message;
    return LM_AIR_OK;
}

static int test_crc_and_frame(void) {
    static const uint8_t check[] = "123456789";
    static const uint8_t golden[] = {
        0xa1u, 0x32u, 0x12u, 0x34u, 0x01u, 0x02u, 0x00u,
        0x01u, 0x1fu, 0x04u, 0xbeu, 0xefu, 0xdeu, 0xadu,
        0xbeu, 0xefu, 0xc2u, 0x0au, 0xbdu, 0x05u};
    lm_air_frame_t frame;
    lm_air_frame_t decoded;
    uint8_t wire[LM_AIR_MAX_WIRE_BYTES];
    size_t wire_len = 0u;

    CHECK(lm_air_crc32c(check, sizeof(check) - 1u) == UINT32_C(0xe3069283));
    memset(&frame, 0, sizeof(frame));
    frame.profile = LM_AIR_PROFILE_HF_BPSK;
    frame.flags = LM_AIR_FLAG_ACK_REQUEST | LM_AIR_FLAG_FEC;
    frame.stream_id = 0x1234u;
    frame.sequence = 0x0102u;
    frame.fragment_index = 0u;
    frame.fragment_count = 1u;
    frame.class_id = LM_AIR_CLASS_STATE_DELTA;
    frame.priority = 15u;
    frame.state_tag = 0xbeefu;
    frame.payload_len = 4u;
    frame.payload[0] = 0xdeu;
    frame.payload[1] = 0xadu;
    frame.payload[2] = 0xbeu;
    frame.payload[3] = 0xefu;
    CHECK(lm_air_frame_encode(&frame, wire, sizeof(wire), &wire_len) ==
          LM_AIR_OK);
    CHECK(wire_len == 20u);
    CHECK(memcmp(wire, golden, sizeof(golden)) == 0);
    CHECK(lm_air_frame_decode(wire, wire_len, &decoded) == LM_AIR_OK);
    CHECK(memcmp(&frame, &decoded, sizeof(frame)) == 0);
    wire[12] ^= 1u;
    CHECK(lm_air_frame_decode(wire, wire_len, &decoded) == LM_AIR_ERR_CRC);
    wire[12] ^= 1u;
    wire[1] = (uint8_t)((wire[1] & 0xf0u) | 0x0fu);
    {
        uint32_t crc = lm_air_crc32c(wire, wire_len - 4u);
        wire[wire_len - 4u] = (uint8_t)(crc >> 24);
        wire[wire_len - 3u] = (uint8_t)(crc >> 16);
        wire[wire_len - 2u] = (uint8_t)(crc >> 8);
        wire[wire_len - 1u] = (uint8_t)crc;
    }
    CHECK(lm_air_frame_decode(wire, wire_len, &decoded) == LM_AIR_ERR_FORMAT);
    wire[1] = 0x32u;
    wire[8] = 0x5fu;
    {
        uint32_t crc = lm_air_crc32c(wire, wire_len - 4u);
        wire[wire_len - 4u] = (uint8_t)(crc >> 24);
        wire[wire_len - 3u] = (uint8_t)(crc >> 16);
        wire[wire_len - 2u] = (uint8_t)(crc >> 8);
        wire[wire_len - 1u] = (uint8_t)crc;
    }
    CHECK(lm_air_frame_decode(wire, wire_len, &decoded) == LM_AIR_ERR_FORMAT);
    return 0;
}

static int test_semantic_envelope_golden(void) {
    static const uint8_t golden[] = {
        0x4cu, 0x4du, 0x53u, 0x31u, 0x01u, 0x00u, 0x01u, 0x0fu,
        0x00u, 0x00u, 0x00u, 0x01u, 0x00u, 0x00u, 0x00u, 0x02u,
        0x00u, 0x00u, 0x00u, 0x03u, 0x00u, 0x00u, 0x00u, 0x00u,
        0x00u, 0x00u, 0x00u, 0x04u, 0x00u, 0x04u, 0x00u, 0x01u,
        0x02u, 0x03u, 0x04u, 0x05u, 0x06u, 0x07u, 0x08u, 0x09u,
        0x0au, 0x0bu, 0x0cu, 0x0du, 0x0eu, 0x0fu, 0x00u, 0x00u,
        0xdeu, 0xadu, 0xbeu, 0xefu};
    static const uint8_t body[] = {0xdeu, 0xadu, 0xbeu, 0xefu};
    lm_air_profile_config_t profile;
    lm_air_tx_t tx;
    lm_air_message_t message;
    lm_air_block_t block;
    lm_air_frame_t frame;
    lm_air_status_t status;
    lm_air_rx_t rx;
    uint8_t wire[LM_AIR_MAX_WIRE_BYTES];
    size_t wire_len;
    size_t i;

    CHECK(lm_air_profile_defaults(LM_AIR_PROFILE_WIFI, &profile) == LM_AIR_OK);
    CHECK(lm_air_tx_init(&tx, 1u, 1u, &profile, NULL) == LM_AIR_OK);
    memset(&message, 0, sizeof(message));
    message.source_id = 1u;
    message.epoch = 2u;
    message.message_id = 3u;
    message.logical_sequence = 4u;
    message.class_id = LM_AIR_CLASS_STATE_DELTA;
    message.priority = 15u;
    for (i = 0u; i < sizeof(message.state_hash); ++i) {
        message.state_hash[i] = (uint8_t)i;
    }
    message.body = body;
    message.body_len = sizeof(body);
    CHECK(lm_air_tx_begin(&tx, &message) == LM_AIR_OK);
    status = lm_air_tx_poll(&tx, &block);
    CHECK(status == LM_AIR_COMPLETE);
    CHECK(block.fec == 0u);
    CHECK(block.interleave_rows == 0u);
    CHECK(lm_air_frame_decode(block.data, block.raw_len, &frame) == LM_AIR_OK);
    CHECK(frame.payload_len == sizeof(golden));
    CHECK(memcmp(frame.payload, golden, sizeof(golden)) == 0);
    frame.payload[47] = 1u;
    CHECK(lm_air_frame_encode(&frame, wire, sizeof(wire), &wire_len) ==
          LM_AIR_OK);
    CHECK(lm_air_rx_init(&rx, ignore_message, NULL, NULL) == LM_AIR_OK);
    CHECK(lm_air_rx_ingest_wire(&rx, wire, wire_len) == LM_AIR_ERR_FORMAT);
    return 0;
}

static int test_fec_and_interleaver(void) {
    uint8_t input[73];
    uint8_t coded[LM_AIR_FEC_MAX_CODED_BYTES];
    uint8_t interleaved[LM_AIR_FEC_MAX_CODED_BYTES];
    uint8_t restored[LM_AIR_FEC_MAX_CODED_BYTES];
    uint8_t decoded[73];
    float llr[LM_AIR_FEC_MAX_CODED_BITS];
    lm_air_fec_workspace_t workspace;
    size_t coded_bits = 0u;
    size_t i;
    uint32_t metric = 0u;

    for (i = 0u; i < sizeof(input); ++i) {
        input[i] = (uint8_t)(i * 37u + 11u);
    }
    CHECK(lm_air_fec_encode(input,
                            sizeof(input),
                            coded,
                            sizeof(coded),
                            &coded_bits) == LM_AIR_OK);
    CHECK(coded_bits == (sizeof(input) * 8u + 6u) * 2u);
    CHECK(lm_air_interleave_bits(coded,
                                 coded_bits,
                                 interleaved,
                                 sizeof(interleaved),
                                 17u) == LM_AIR_OK);
    CHECK(lm_air_deinterleave_bits(interleaved,
                                   coded_bits,
                                   restored,
                                   sizeof(restored),
                                   17u) == LM_AIR_OK);
    CHECK(memcmp(coded, restored, (coded_bits + 7u) / 8u) == 0);

    for (i = 29u; i + 1u < coded_bits; i += 191u) {
        flip_bit(restored, i);
    }
    CHECK(lm_air_fec_decode_hard(restored,
                                 coded_bits,
                                 decoded,
                                 sizeof(decoded),
                                 sizeof(input),
                                 &workspace,
                                 &metric) == LM_AIR_OK);
    CHECK(memcmp(input, decoded, sizeof(input)) == 0);
    CHECK(metric > 0u);

    for (i = 0u; i < coded_bits; ++i) {
        llr[i] = get_bit(coded, i) != 0u ? 3.0f : -3.0f;
    }
    llr[19] = -llr[19];
    CHECK(lm_air_fec_decode_soft(llr,
                                 coded_bits,
                                 decoded,
                                 sizeof(decoded),
                                 sizeof(input),
                                 &workspace,
                                 &metric) == LM_AIR_OK);
    CHECK(memcmp(input, decoded, sizeof(input)) == 0);
    return 0;
}

static int test_modems(void) {
    static const uint8_t bits[] = {0xa5u, 0x3cu, 0xf0u, 0x19u};
    lm_air_fsk_config_t afsk = lm_air_afsk_bell202_config(48000.0f);
    lm_air_fsk_config_t cpfsk =
        lm_air_cpfsk_config(48000.0f, 2400.0f, 1800.0f, 600.0f, 0.7f);
    lm_air_fsk_modulator_t mod;
    float pcm[32u * 40u];
    float cpfsk_pcm[32u * 20u];
    float llr[32];
    lm_air_iq_f32_t iq[32u * 4u];
    size_t samples;
    size_t count;
    size_t i;

    CHECK(lm_air_fsk_modulator_init(&mod, &afsk) == LM_AIR_OK);
    CHECK(lm_air_fsk_modulate(
              &mod, bits, 32u, pcm, sizeof(pcm) / sizeof(pcm[0]), &samples) ==
          LM_AIR_OK);
    CHECK(samples == 1280u);
    CHECK(lm_air_fsk_demodulate(&afsk,
                                 pcm,
                                 samples,
                                 llr,
                                 sizeof(llr) / sizeof(llr[0]),
                                 &count) == LM_AIR_OK);
    CHECK(count == 32u);
    for (i = 0u; i < count; ++i) {
        CHECK((llr[i] > 0.0f) == (get_bit(bits, i) != 0u));
    }

    CHECK(lm_air_fsk_modulator_init(&mod, &cpfsk) == LM_AIR_OK);
    CHECK(lm_air_fsk_modulate(&mod,
                              bits,
                              32u,
                              cpfsk_pcm,
                              sizeof(cpfsk_pcm) / sizeof(cpfsk_pcm[0]),
                              &samples) == LM_AIR_OK);
    CHECK(lm_air_fsk_demodulate(&cpfsk,
                                 cpfsk_pcm,
                                 samples,
                                 llr,
                                 sizeof(llr) / sizeof(llr[0]),
                                 &count) == LM_AIR_OK);
    for (i = 0u; i < count; ++i) {
        CHECK((llr[i] > 0.0f) == (get_bit(bits, i) != 0u));
    }

    CHECK(lm_air_bpsk_modulate_iq(bits,
                                   32u,
                                   4u,
                                   0.75f,
                                   iq,
                                   sizeof(iq) / sizeof(iq[0]),
                                   &samples) == LM_AIR_OK);
    CHECK(lm_air_bpsk_demodulate_iq(iq,
                                     samples,
                                     4u,
                                     0.25f,
                                     llr,
                                     sizeof(llr) / sizeof(llr[0]),
                                     &count) == LM_AIR_OK);
    for (i = 0u; i < count; ++i) {
        CHECK((llr[i] > 0.0f) == (get_bit(bits, i) != 0u));
    }
    return 0;
}

static int test_llr_assist(void) {
    static const uint8_t known[] = {0xaau, 0x55u};
    float llr[16];
    float corrected[16];
    lm_air_llr_assist_t assist;
    size_t i;

    for (i = 0u; i < 16u; ++i) {
        llr[i] = get_bit(known, i) != 0u ? 2.0f : -2.0f;
    }
    lm_air_llr_assist_init(&assist, 0.04f, 0.95f);
    CHECK(lm_air_llr_assist_apply(&assist, llr, 16u, corrected, 16u) ==
          LM_AIR_OK);
    CHECK(assist.dsp_fallbacks == 16u);
    for (i = 0u; i < 64u; ++i) {
        CHECK(lm_air_llr_assist_adapt(&assist, llr, known, 16u) == LM_AIR_OK);
    }
    assist.min_confidence = 0.10f;
    CHECK(lm_air_llr_assist_apply(&assist, llr, 16u, corrected, 16u) ==
          LM_AIR_OK);
    CHECK(assist.learned_uses > 0u);
    for (i = 0u; i < 16u; ++i) {
        CHECK(isfinite(corrected[i]));
        CHECK((corrected[i] > 0.0f) == (get_bit(known, i) != 0u));
    }
    return 0;
}

typedef struct capture {
    lm_air_block_t blocks[LM_AIR_MAX_FRAGMENTS];
    size_t block_count;
    uint8_t body[2048];
    uint16_t body_len;
    uint32_t source_id;
    unsigned delivered;
} capture_t;

static lm_air_status_t capture_block(void *user, const lm_air_block_t *block) {
    capture_t *capture = (capture_t *)user;
    if (capture->block_count >= LM_AIR_MAX_FRAGMENTS) {
        return LM_AIR_ERR_CAPACITY;
    }
    capture->blocks[capture->block_count++] = *block;
    return LM_AIR_OK;
}

static lm_air_status_t capture_message(
    void *user,
    const lm_air_message_t *message) {
    capture_t *capture = (capture_t *)user;
    if (message->body_len > sizeof(capture->body)) {
        return LM_AIR_ERR_CAPACITY;
    }
    memcpy(capture->body, message->body, message->body_len);
    capture->body_len = message->body_len;
    capture->source_id = message->source_id;
    ++capture->delivered;
    return LM_AIR_OK;
}

static int toy_sign(
    void *user,
    const uint8_t *data,
    size_t data_len,
    uint8_t signature[LM_AIR_SIGNATURE_BYTES]) {
    uint32_t crc = lm_air_crc32c(data, data_len) ^ *(const uint32_t *)user;
    size_t i;
    for (i = 0u; i < LM_AIR_SIGNATURE_BYTES; ++i) {
        signature[i] = (uint8_t)(crc >> ((i & 3u) * 8u));
    }
    return 0;
}

static int toy_verify(
    void *user,
    const uint8_t *data,
    size_t data_len,
    const uint8_t signature[LM_AIR_SIGNATURE_BYTES]) {
    uint8_t expected[LM_AIR_SIGNATURE_BYTES];
    toy_sign(user, data, data_len, expected);
    return memcmp(expected, signature, sizeof(expected)) == 0 ? 0 : -1;
}

static int test_tx_rx_and_replay(void) {
    lm_air_tx_t tx;
    lm_air_rx_t rx;
    lm_air_profile_config_t profile;
    lm_air_message_t message;
    lm_air_crypto_hooks_t crypto;
    capture_t capture;
    uint8_t body[1000];
    uint32_t key = UINT32_C(0x19460721);
    size_t i;
    lm_air_status_t status = LM_AIR_MORE;

    memset(&capture, 0, sizeof(capture));
    for (i = 0u; i < sizeof(body); ++i) {
        body[i] = (uint8_t)(i ^ (i >> 3));
    }
    crypto.sign = toy_sign;
    crypto.verify = toy_verify;
    crypto.user = &key;
    CHECK(lm_air_profile_defaults(LM_AIR_PROFILE_HF_AFSK, &profile) ==
          LM_AIR_OK);
    CHECK(lm_air_tx_init(&tx, 0x3344u, 65534u, &profile, &crypto) ==
          LM_AIR_OK);
    CHECK(lm_air_rx_init(&rx, capture_message, &capture, &crypto) == LM_AIR_OK);
    memset(&message, 0, sizeof(message));
    message.source_id = UINT32_C(0x10203040);
    message.epoch = 7u;
    message.message_id = 99u;
    message.logical_sequence = UINT64_C(0x0102030405060708);
    message.class_id = 3u;
    message.priority = 14u;
    for (i = 0u; i < sizeof(message.state_hash); ++i) {
        message.state_hash[i] = (uint8_t)(0xa0u + i);
    }
    message.body = body;
    message.body_len = sizeof(body);
    message.authenticated = 1u;
    CHECK(lm_air_tx_send(&tx, &message, capture_block, &capture) == LM_AIR_OK);
    CHECK(capture.block_count > 1u);
    for (i = capture.block_count; i > 0u; --i) {
        status = lm_air_rx_ingest_block(&rx, &capture.blocks[i - 1u]);
        CHECK(status == LM_AIR_MORE || status == LM_AIR_COMPLETE);
    }
    CHECK(status == LM_AIR_COMPLETE);
    CHECK(capture.delivered == 1u);
    CHECK(capture.body_len == sizeof(body));
    CHECK(capture.source_id == message.source_id);
    CHECK(memcmp(capture.body, body, sizeof(body)) == 0);
    CHECK(lm_air_rx_ingest_block(&rx, &capture.blocks[0]) == LM_AIR_ERR_REPLAY);
    CHECK(rx.stats.replay_rejected == 1u);
    return 0;
}

static int test_authentication_failure(void) {
    static const uint8_t body[] = {1u, 2u, 3u};
    lm_air_profile_config_t profile;
    lm_air_crypto_hooks_t tx_crypto;
    lm_air_crypto_hooks_t rx_crypto;
    lm_air_message_t message;
    lm_air_tx_t tx;
    lm_air_rx_t rx;
    capture_t capture;
    uint32_t tx_key = 1u;
    uint32_t rx_key = 2u;

    memset(&capture, 0, sizeof(capture));
    tx_crypto.sign = toy_sign;
    tx_crypto.verify = toy_verify;
    tx_crypto.user = &tx_key;
    rx_crypto.sign = toy_sign;
    rx_crypto.verify = toy_verify;
    rx_crypto.user = &rx_key;
    CHECK(lm_air_profile_defaults(LM_AIR_PROFILE_WIFI, &profile) == LM_AIR_OK);
    CHECK(lm_air_tx_init(&tx, 5u, 8u, &profile, &tx_crypto) == LM_AIR_OK);
    CHECK(lm_air_rx_init(&rx, capture_message, &capture, &rx_crypto) ==
          LM_AIR_OK);
    memset(&message, 0, sizeof(message));
    message.source_id = 4u;
    message.message_id = 9u;
    message.class_id = LM_AIR_CLASS_CONTROL;
    message.priority = 15u;
    message.body = body;
    message.body_len = sizeof(body);
    message.authenticated = 1u;
    CHECK(lm_air_tx_send(&tx, &message, capture_block, &capture) == LM_AIR_OK);
    CHECK(capture.block_count == 1u);
    CHECK(lm_air_rx_ingest_block(&rx, &capture.blocks[0]) == LM_AIR_ERR_AUTH);
    CHECK(capture.delivered == 0u);
    CHECK(rx.stats.auth_failures == 1u);
    return 0;
}

static uint32_t fuzz_state = UINT32_C(0x9e3779b9);

static uint32_t fuzz_random(void) {
    fuzz_state ^= fuzz_state << 13;
    fuzz_state ^= fuzz_state >> 17;
    fuzz_state ^= fuzz_state << 5;
    return fuzz_state;
}

static int test_malformed_inputs(void) {
    lm_air_frame_t frame;
    lm_air_rx_t rx;
    capture_t capture;
    uint8_t bytes[LM_AIR_MAX_WIRE_BYTES + 16u];
    lm_air_block_t block;
    size_t iteration;

    memset(&capture, 0, sizeof(capture));
    CHECK(lm_air_rx_init(&rx, capture_message, &capture, NULL) == LM_AIR_OK);
    for (iteration = 0u; iteration < 20000u; ++iteration) {
        size_t len = fuzz_random() % sizeof(bytes);
        size_t i;
        for (i = 0u; i < len; ++i) {
            bytes[i] = (uint8_t)fuzz_random();
        }
        (void)lm_air_frame_decode(bytes, len, &frame);
        (void)lm_air_rx_ingest_wire(&rx, bytes, len);
    }
    for (iteration = 0u; iteration < 2000u; ++iteration) {
        size_t i;
        block.raw_len = (uint16_t)fuzz_random();
        block.bit_len = (uint16_t)fuzz_random();
        block.fec = (uint8_t)fuzz_random();
        block.interleave_rows = (uint8_t)fuzz_random();
        for (i = 0u; i < sizeof(block.data); ++i) {
            block.data[i] = (uint8_t)fuzz_random();
        }
        (void)lm_air_rx_ingest_block(&rx, &block);
    }
    CHECK(capture.delivered == 0u);
    return 0;
}

static int test_profiles(void) {
    lm_air_profile_config_t config;
    unsigned profile;
    for (profile = LM_AIR_PROFILE_WIFI; profile <= LM_AIR_PROFILE_HAM_PACKET;
         ++profile) {
        CHECK(lm_air_profile_defaults((uint8_t)profile, &config) == LM_AIR_OK);
        CHECK(config.fragment_payload_bytes > 0u);
        CHECK(config.fragment_payload_bytes <= LM_AIR_MAX_FRAME_PAYLOAD);
    }
    return 0;
}

int main(void) {
    CHECK(test_crc_and_frame() == 0);
    CHECK(test_semantic_envelope_golden() == 0);
    CHECK(test_fec_and_interleaver() == 0);
    CHECK(test_modems() == 0);
    CHECK(test_llr_assist() == 0);
    CHECK(test_tx_rx_and_replay() == 0);
    CHECK(test_authentication_failure() == 0);
    CHECK(test_malformed_inputs() == 0);
    CHECK(test_profiles() == 0);
    puts("latentmesh_air_tests: all checks passed");
    return 0;
}
