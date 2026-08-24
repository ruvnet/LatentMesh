#include "latentmesh_air/neural.h"

#include <math.h>

static unsigned packed_bit(const uint8_t *data, size_t bit) {
    return ((unsigned)data[bit >> 3] >> (7u - (unsigned)(bit & 7u))) & 1u;
}

static float clampf(float value, float low, float high) {
    if (value < low) {
        return low;
    }
    if (value > high) {
        return high;
    }
    return value;
}

static float feature_at(const float *llr, size_t count, ptrdiff_t index) {
    if (index < 0 || (size_t)index >= count) {
        return 0.0f;
    }
    if (!isfinite(llr[index])) {
        return 0.0f;
    }
    return clampf(llr[index], -8.0f, 8.0f);
}

void lm_air_llr_assist_init(
    lm_air_llr_assist_t *assist,
    float learning_rate,
    float min_confidence) {
    size_t i;
    if (assist == NULL) {
        return;
    }
    for (i = 0u; i < LM_AIR_LLR_TAPS; ++i) {
        assist->weights[i] = 0.0f;
    }
    assist->weights[LM_AIR_LLR_TAPS / 2u] = 1.0f;
    assist->bias = 0.0f;
    assist->learning_rate = clampf(learning_rate, 0.0001f, 0.25f);
    assist->mse_ema = 1.0f;
    assist->min_confidence = clampf(min_confidence, 0.0f, 0.99f);
    assist->learned_uses = 0u;
    assist->dsp_fallbacks = 0u;
}

static float raw_prediction(
    const lm_air_llr_assist_t *assist,
    const float *llr,
    size_t llr_count,
    size_t index,
    float features[LM_AIR_LLR_TAPS]) {
    float value = assist->bias;
    size_t tap;
    ptrdiff_t center = (ptrdiff_t)(LM_AIR_LLR_TAPS / 2u);
    for (tap = 0u; tap < LM_AIR_LLR_TAPS; ++tap) {
        ptrdiff_t sample_index =
            (ptrdiff_t)index + (ptrdiff_t)tap - center;
        features[tap] = feature_at(llr, llr_count, sample_index);
        value += assist->weights[tap] * features[tap];
    }
    return clampf(value, -32.0f, 32.0f);
}

float lm_air_llr_assist_predict(
    lm_air_llr_assist_t *assist,
    const float *llr,
    size_t llr_count,
    size_t index,
    float *confidence,
    int *used_learned) {
    float features[LM_AIR_LLR_TAPS];
    float predicted;
    float margin;
    float calibration;
    float score;
    int use;

    if (assist == NULL || llr == NULL || index >= llr_count) {
        if (confidence != NULL) {
            *confidence = 0.0f;
        }
        if (used_learned != NULL) {
            *used_learned = 0;
        }
        return 0.0f;
    }
    predicted = raw_prediction(assist, llr, llr_count, index, features);
    margin = fabsf(predicted) / (1.0f + fabsf(predicted));
    calibration = 1.0f / (1.0f + clampf(assist->mse_ema, 0.0f, 16.0f));
    score = margin * calibration;
    use = isfinite(predicted) && score >= assist->min_confidence;
    if (use != 0) {
        ++assist->learned_uses;
    } else {
        predicted = feature_at(llr, llr_count, (ptrdiff_t)index);
        ++assist->dsp_fallbacks;
    }
    if (confidence != NULL) {
        *confidence = score;
    }
    if (used_learned != NULL) {
        *used_learned = use;
    }
    return predicted;
}

lm_air_status_t lm_air_llr_assist_apply(
    lm_air_llr_assist_t *assist,
    const float *input_llr,
    size_t llr_count,
    float *output_llr,
    size_t output_capacity) {
    size_t i;
    if (assist == NULL || input_llr == NULL || output_llr == NULL) {
        return LM_AIR_ERR_ARGUMENT;
    }
    if (llr_count > output_capacity) {
        return LM_AIR_ERR_CAPACITY;
    }
    for (i = 0u; i < llr_count; ++i) {
        output_llr[i] = lm_air_llr_assist_predict(
            assist, input_llr, llr_count, i, NULL, NULL);
    }
    return LM_AIR_OK;
}

lm_air_status_t lm_air_llr_assist_adapt(
    lm_air_llr_assist_t *assist,
    const float *input_llr,
    const uint8_t *known_bits,
    size_t bit_count) {
    size_t i;
    if (assist == NULL || input_llr == NULL || known_bits == NULL) {
        return LM_AIR_ERR_ARGUMENT;
    }
    for (i = 0u; i < bit_count; ++i) {
        float features[LM_AIR_LLR_TAPS];
        float predicted =
            raw_prediction(assist, input_llr, bit_count, i, features);
        float target = packed_bit(known_bits, i) != 0u ? 4.0f : -4.0f;
        float error = target - predicted;
        float norm = 1.0f;
        float step;
        size_t tap;
        for (tap = 0u; tap < LM_AIR_LLR_TAPS; ++tap) {
            norm += features[tap] * features[tap];
        }
        step = assist->learning_rate * error / norm;
        for (tap = 0u; tap < LM_AIR_LLR_TAPS; ++tap) {
            assist->weights[tap] = clampf(
                assist->weights[tap] + step * features[tap], -4.0f, 4.0f);
        }
        assist->bias = clampf(assist->bias + step, -4.0f, 4.0f);
        assist->mse_ema = 0.98f * assist->mse_ema +
                          0.02f * clampf((error * error) / 16.0f,
                                         0.0f,
                                         16.0f);
    }
    return LM_AIR_OK;
}
