#ifndef LATENTMESH_AIR_MODEM_H
#define LATENTMESH_AIR_MODEM_H

#include "latentmesh_air/common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct lm_air_fsk_config {
    float sample_rate;
    float symbol_rate;
    float mark_hz;
    float space_hz;
    float amplitude;
} lm_air_fsk_config_t;

typedef struct lm_air_fsk_modulator {
    lm_air_fsk_config_t config;
    float phase;
} lm_air_fsk_modulator_t;

typedef struct lm_air_iq_f32 {
    float i;
    float q;
} lm_air_iq_f32_t;

lm_air_fsk_config_t lm_air_afsk_bell202_config(float sample_rate);
lm_air_fsk_config_t lm_air_cpfsk_config(
    float sample_rate,
    float symbol_rate,
    float center_hz,
    float deviation_hz,
    float amplitude);

lm_air_status_t lm_air_fsk_modulator_init(
    lm_air_fsk_modulator_t *modulator,
    const lm_air_fsk_config_t *config);

size_t lm_air_fsk_samples_for_bits(
    const lm_air_fsk_config_t *config,
    size_t bit_count);

lm_air_status_t lm_air_fsk_modulate(
    lm_air_fsk_modulator_t *modulator,
    const uint8_t *packed_bits,
    size_t bit_count,
    float *pcm,
    size_t pcm_capacity,
    size_t *pcm_samples);

lm_air_status_t lm_air_fsk_demodulate(
    const lm_air_fsk_config_t *config,
    const float *pcm,
    size_t pcm_samples,
    float *llr,
    size_t llr_capacity,
    size_t *llr_count);

lm_air_status_t lm_air_bpsk_modulate_iq(
    const uint8_t *packed_bits,
    size_t bit_count,
    unsigned samples_per_symbol,
    float amplitude,
    lm_air_iq_f32_t *iq,
    size_t iq_capacity,
    size_t *iq_count);

lm_air_status_t lm_air_bpsk_demodulate_iq(
    const lm_air_iq_f32_t *iq,
    size_t iq_count,
    unsigned samples_per_symbol,
    float noise_variance,
    float *llr,
    size_t llr_capacity,
    size_t *llr_count);

#ifdef __cplusplus
}
#endif

#endif
