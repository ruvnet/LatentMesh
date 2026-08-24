#ifndef LATENTMESH_AIR_NEURAL_H
#define LATENTMESH_AIR_NEURAL_H

#include "latentmesh_air/common.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LM_AIR_LLR_TAPS 5u

/*
 * Bounded online learned LLR corrector. It is deliberately tiny: five taps,
 * one bias, normalized LMS adaptation, bounded weights, and an explicit DSP
 * fallback when calibration or prediction margin is insufficient.
 */
typedef struct lm_air_llr_assist {
    float weights[LM_AIR_LLR_TAPS];
    float bias;
    float learning_rate;
    float mse_ema;
    float min_confidence;
    uint32_t learned_uses;
    uint32_t dsp_fallbacks;
} lm_air_llr_assist_t;

void lm_air_llr_assist_init(
    lm_air_llr_assist_t *assist,
    float learning_rate,
    float min_confidence);

float lm_air_llr_assist_predict(
    lm_air_llr_assist_t *assist,
    const float *llr,
    size_t llr_count,
    size_t index,
    float *confidence,
    int *used_learned);

lm_air_status_t lm_air_llr_assist_apply(
    lm_air_llr_assist_t *assist,
    const float *input_llr,
    size_t llr_count,
    float *output_llr,
    size_t output_capacity);

lm_air_status_t lm_air_llr_assist_adapt(
    lm_air_llr_assist_t *assist,
    const float *input_llr,
    const uint8_t *known_bits,
    size_t bit_count);

#ifdef __cplusplus
}
#endif

#endif
