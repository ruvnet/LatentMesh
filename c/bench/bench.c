#include "latentmesh_air.h"

#include <stdio.h>
#include <string.h>
#include <time.h>

static double elapsed_seconds(clock_t begin, clock_t end) {
    return (double)(end - begin) / (double)CLOCKS_PER_SEC;
}

typedef struct bit_counter {
    size_t bits;
    size_t blocks;
} bit_counter_t;

static lm_air_status_t count_block(void *user, const lm_air_block_t *block) {
    bit_counter_t *counter = (bit_counter_t *)user;
    counter->bits += block->bit_len;
    ++counter->blocks;
    return LM_AIR_OK;
}

static size_t measure_hf_air_bits(size_t body_len, size_t *blocks) {
    uint8_t body[1800];
    lm_air_profile_config_t profile;
    lm_air_tx_t tx;
    lm_air_message_t message;
    bit_counter_t counter;
    size_t i;
    if (body_len > sizeof(body)) {
        return 0u;
    }
    for (i = 0u; i < body_len; ++i) {
        body[i] = (uint8_t)(i * 29u + 7u);
    }
    memset(&message, 0, sizeof(message));
    message.source_id = 1u;
    message.message_id = 1u;
    message.logical_sequence = 1u;
    message.class_id = LM_AIR_CLASS_STATE_DELTA;
    message.priority = 15u;
    message.body = body;
    message.body_len = (uint16_t)body_len;
    memset(&counter, 0, sizeof(counter));
    if (lm_air_profile_defaults(LM_AIR_PROFILE_HF_AFSK, &profile) !=
            LM_AIR_OK ||
        lm_air_tx_init(&tx, 1u, 1u, &profile, NULL) != LM_AIR_OK ||
        lm_air_tx_send(&tx, &message, count_block, &counter) != LM_AIR_OK) {
        return 0u;
    }
    *blocks = counter.blocks;
    return counter.bits;
}

int main(void) {
    enum {
        FRAME_ITERATIONS = 300000,
        FEC_ITERATIONS = 3000,
        MODEM_ITERATIONS = 300,
        MODEM_BITS = 256
    };
    lm_air_frame_t frame;
    lm_air_frame_t decoded_frame;
    uint8_t wire[LM_AIR_MAX_WIRE_BYTES];
    uint8_t coded[LM_AIR_FEC_MAX_CODED_BYTES];
    uint8_t decoded[LM_AIR_MAX_WIRE_BYTES];
    lm_air_fec_workspace_t workspace;
    lm_air_fsk_config_t afsk = lm_air_afsk_bell202_config(48000.0f);
    lm_air_fsk_modulator_t fsk_modulator;
    float pcm[MODEM_BITS * 40];
    float llr[MODEM_BITS];
    lm_air_iq_f32_t iq[MODEM_BITS * 4];
    size_t wire_len = 0u;
    size_t coded_bits = 0u;
    uint32_t metric = 0u;
    size_t sample_count = 0u;
    size_t llr_count = 0u;
    clock_t begin;
    clock_t end;
    double seconds;
    int i;
    volatile uint32_t sink = 0u;
    size_t dense_blocks;
    size_t delta_blocks;
    size_t dense_bits;
    size_t delta_bits;

    memset(&frame, 0, sizeof(frame));
    frame.profile = LM_AIR_PROFILE_WIFI;
    frame.flags = 0u;
    frame.stream_id = 42u;
    frame.sequence = 77u;
    frame.fragment_count = 1u;
    frame.class_id = 2u;
    frame.priority = 8u;
    frame.state_tag = 0xa55au;
    frame.payload_len = LM_AIR_MAX_FRAME_PAYLOAD;
    for (i = 0; i < (int)frame.payload_len; ++i) {
        frame.payload[i] = (uint8_t)(i * 13);
    }

    begin = clock();
    for (i = 0; i < FRAME_ITERATIONS; ++i) {
        frame.sequence = (uint16_t)i;
        (void)lm_air_frame_encode(&frame, wire, sizeof(wire), &wire_len);
        (void)lm_air_frame_decode(wire, wire_len, &decoded_frame);
        sink += decoded_frame.payload[0];
    }
    end = clock();
    seconds = elapsed_seconds(begin, end);
    printf("codec: %.0f encode+decode frames/s, %.1f MiB/s wire\n",
           FRAME_ITERATIONS / seconds,
           ((double)FRAME_ITERATIONS * (double)wire_len /
            (1024.0 * 1024.0)) /
               seconds);

    (void)lm_air_fec_encode(
        wire, wire_len, coded, sizeof(coded), &coded_bits);
    begin = clock();
    for (i = 0; i < FEC_ITERATIONS; ++i) {
        (void)lm_air_fec_encode(
            wire, wire_len, coded, sizeof(coded), &coded_bits);
        (void)lm_air_fec_decode_hard(coded,
                                     coded_bits,
                                     decoded,
                                     sizeof(decoded),
                                     wire_len,
                                     &workspace,
                                     &metric);
        sink += decoded[0];
    }
    end = clock();
    seconds = elapsed_seconds(begin, end);
    printf("fec k=7 r=1/2: %.0f encode+decode frames/s, %.2f Mbit/s raw\n",
           FEC_ITERATIONS / seconds,
           ((double)FEC_ITERATIONS * (double)wire_len * 8.0 / 1000000.0) /
               seconds);
    (void)lm_air_fsk_modulator_init(&fsk_modulator, &afsk);
    begin = clock();
    for (i = 0; i < MODEM_ITERATIONS; ++i) {
        (void)lm_air_fsk_modulate(&fsk_modulator,
                                  wire,
                                  MODEM_BITS,
                                  pcm,
                                  sizeof(pcm) / sizeof(pcm[0]),
                                  &sample_count);
        (void)lm_air_fsk_demodulate(&afsk,
                                     pcm,
                                     sample_count,
                                     llr,
                                     sizeof(llr) / sizeof(llr[0]),
                                     &llr_count);
        sink += llr[0] > 0.0f ? 1u : 0u;
    }
    end = clock();
    seconds = elapsed_seconds(begin, end);
    printf("afsk pcm: %.0f modulate+demodulate symbols/s\n",
           ((double)MODEM_ITERATIONS * MODEM_BITS) / seconds);

    begin = clock();
    for (i = 0; i < FRAME_ITERATIONS; ++i) {
        (void)lm_air_bpsk_modulate_iq(wire,
                                       MODEM_BITS,
                                       4u,
                                       0.8f,
                                       iq,
                                       sizeof(iq) / sizeof(iq[0]),
                                       &sample_count);
        (void)lm_air_bpsk_demodulate_iq(iq,
                                         sample_count,
                                         4u,
                                         0.25f,
                                         llr,
                                         sizeof(llr) / sizeof(llr[0]),
                                         &llr_count);
        sink += llr[0] > 0.0f ? 1u : 0u;
    }
    end = clock();
    seconds = elapsed_seconds(begin, end);
    printf("bpsk iq: %.0f modulate+demodulate symbols/s\n",
           ((double)FRAME_ITERATIONS * MODEM_BITS) / seconds);
    printf("memory: tx=%zu bytes, rx=%zu bytes, block=%zu bytes, fec=%zu "
           "bytes\n",
           sizeof(lm_air_tx_t),
           sizeof(lm_air_rx_t),
           sizeof(lm_air_block_t),
           sizeof(lm_air_fec_workspace_t));
    dense_bits = measure_hf_air_bits(1800u, &dense_blocks);
    delta_bits = measure_hf_air_bits(64u, &delta_blocks);
    printf("stage 1 framing evidence: 1800-byte state=%zu bits/%zu blocks, "
           "64-byte delta=%zu bits/%zu blocks, reduction=%.2fx\n",
           dense_bits,
           dense_blocks,
           delta_bits,
           delta_blocks,
           (double)dense_bits / (double)delta_bits);
    puts("stage 1 task-accuracy equivalence: not measured by the C codec");
    puts("stage 2 degraded-channel useful information/energy: target 2.00x, "
         "not measured by noiseless microbenchmarks");
    printf("sink=%u\n", (unsigned)sink);
    return 0;
}
