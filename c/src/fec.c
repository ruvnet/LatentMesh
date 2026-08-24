#include "latentmesh_air/fec.h"

#include <limits.h>
#include <math.h>
#include <string.h>

#define LM_AIR_G0 0x79u /* K=7, octal 171 */
#define LM_AIR_G1 0x5bu /* K=7, octal 133 */
#define LM_AIR_METRIC_INF (INT32_MAX / 4)

static unsigned get_bit(const uint8_t *data, size_t bit) {
    return ((unsigned)data[bit >> 3] >> (7u - (unsigned)(bit & 7u))) & 1u;
}

static void set_bit(uint8_t *data, size_t bit, unsigned value) {
    uint8_t mask = (uint8_t)(1u << (7u - (bit & 7u)));
    if (value != 0u) {
        data[bit >> 3] |= mask;
    } else {
        data[bit >> 3] &= (uint8_t)~mask;
    }
}

static unsigned parity7(unsigned value) {
    value ^= value >> 4;
    value ^= value >> 2;
    value ^= value >> 1;
    return value & 1u;
}

size_t lm_air_fec_encoded_bits(size_t input_bytes) {
    if (input_bytes > LM_AIR_MAX_WIRE_BYTES) {
        return 0u;
    }
    return ((input_bytes * 8u) + LM_AIR_FEC_TAIL_BITS) * 2u;
}

lm_air_status_t lm_air_fec_encode(
    const uint8_t *input,
    size_t input_bytes,
    uint8_t *coded,
    size_t coded_capacity,
    size_t *coded_bits) {
    size_t input_bits;
    size_t output_bits;
    size_t output_bytes;
    size_t i;
    unsigned state = 0u;

    if ((input == NULL && input_bytes != 0u) || coded == NULL ||
        coded_bits == NULL) {
        return LM_AIR_ERR_ARGUMENT;
    }
    output_bits = lm_air_fec_encoded_bits(input_bytes);
    if (output_bits == 0u && input_bytes != 0u) {
        return LM_AIR_ERR_CAPACITY;
    }
    output_bytes = (output_bits + 7u) / 8u;
    if (output_bytes > coded_capacity) {
        return LM_AIR_ERR_CAPACITY;
    }
    memset(coded, 0, output_bytes);
    input_bits = input_bytes * 8u;
    for (i = 0u; i < input_bits + LM_AIR_FEC_TAIL_BITS; ++i) {
        unsigned bit = i < input_bits ? get_bit(input, i) : 0u;
        unsigned reg = ((state << 1) | bit) & 0x7fu;
        set_bit(coded, i * 2u, parity7(reg & LM_AIR_G0));
        set_bit(coded, i * 2u + 1u, parity7(reg & LM_AIR_G1));
        state = reg & 0x3fu;
    }
    *coded_bits = output_bits;
    return LM_AIR_OK;
}

typedef int32_t (*branch_metric_fn)(
    const void *observations,
    size_t observation_index,
    unsigned expected);

static int32_t hard_metric(
    const void *observations,
    size_t observation_index,
    unsigned expected) {
    const uint8_t *coded = (const uint8_t *)observations;
    return get_bit(coded, observation_index) == expected ? 0 : 1;
}

static int32_t soft_metric(
    const void *observations,
    size_t observation_index,
    unsigned expected) {
    const float *llr = (const float *)observations;
    float value = llr[observation_index];
    float signed_margin;
    float penalty;

    if (!isfinite(value)) {
        return 128;
    }
    if (value > 16.0f) {
        value = 16.0f;
    } else if (value < -16.0f) {
        value = -16.0f;
    }
    signed_margin = expected != 0u ? value : -value;
    penalty = signed_margin >= 0.0f ? 1.0f / (1.0f + signed_margin)
                                    : 1.0f + (-signed_margin);
    return (int32_t)(penalty * 32.0f + 0.5f);
}

static lm_air_status_t fec_decode(
    const void *observations,
    size_t coded_bits,
    uint8_t *output,
    size_t output_capacity,
    size_t expected_output_bytes,
    lm_air_fec_workspace_t *workspace,
    uint32_t *path_metric,
    branch_metric_fn metric_fn) {
    size_t expected_bits;
    size_t steps;
    size_t t;
    unsigned state;
    int32_t *previous;
    int32_t *next;

    if (observations == NULL || output == NULL || workspace == NULL ||
        metric_fn == NULL) {
        return LM_AIR_ERR_ARGUMENT;
    }
    if (expected_output_bytes > output_capacity ||
        expected_output_bytes > LM_AIR_MAX_WIRE_BYTES) {
        return LM_AIR_ERR_CAPACITY;
    }
    expected_bits = lm_air_fec_encoded_bits(expected_output_bytes);
    if (coded_bits != expected_bits || (coded_bits & 1u) != 0u) {
        return LM_AIR_ERR_FEC;
    }
    steps = coded_bits / 2u;
    if (steps > LM_AIR_FEC_MAX_STEPS) {
        return LM_AIR_ERR_CAPACITY;
    }

    previous = workspace->metric_a;
    next = workspace->metric_b;
    for (state = 0u; state < 64u; ++state) {
        previous[state] = state == 0u ? 0 : LM_AIR_METRIC_INF;
    }

    for (t = 0u; t < steps; ++t) {
        uint64_t decisions = 0u;
        unsigned next_state;
        for (next_state = 0u; next_state < 64u; ++next_state) {
            unsigned input_bit = next_state & 1u;
            unsigned predecessor0 = next_state >> 1;
            unsigned predecessor1 = predecessor0 | 0x20u;
            unsigned reg0 = ((predecessor0 << 1) | input_bit) & 0x7fu;
            unsigned reg1 = ((predecessor1 << 1) | input_bit) & 0x7fu;
            int32_t branch0 =
                metric_fn(observations, t * 2u, parity7(reg0 & LM_AIR_G0)) +
                metric_fn(observations,
                          t * 2u + 1u,
                          parity7(reg0 & LM_AIR_G1));
            int32_t branch1 =
                metric_fn(observations, t * 2u, parity7(reg1 & LM_AIR_G0)) +
                metric_fn(observations,
                          t * 2u + 1u,
                          parity7(reg1 & LM_AIR_G1));
            int32_t candidate0 = previous[predecessor0] >= LM_AIR_METRIC_INF
                                     ? LM_AIR_METRIC_INF
                                     : previous[predecessor0] + branch0;
            int32_t candidate1 = previous[predecessor1] >= LM_AIR_METRIC_INF
                                     ? LM_AIR_METRIC_INF
                                     : previous[predecessor1] + branch1;
            if (candidate1 < candidate0) {
                next[next_state] = candidate1;
                decisions |= UINT64_C(1) << next_state;
            } else {
                next[next_state] = candidate0;
            }
        }
        workspace->survivor[t] = decisions;
        {
            int32_t *swap = previous;
            previous = next;
            next = swap;
        }
    }

    if (previous[0] >= LM_AIR_METRIC_INF) {
        return LM_AIR_ERR_FEC;
    }
    memset(output, 0, expected_output_bytes);
    state = 0u;
    for (t = steps; t > 0u; --t) {
        size_t step = t - 1u;
        unsigned input_bit = state & 1u;
        unsigned high_predecessor =
            (unsigned)((workspace->survivor[step] >> state) & 1u);
        if (step < expected_output_bytes * 8u) {
            set_bit(output, step, input_bit);
        }
        state = (state >> 1) | (high_predecessor << 5);
    }
    if (path_metric != NULL) {
        *path_metric = (uint32_t)previous[0];
    }
    return LM_AIR_OK;
}

lm_air_status_t lm_air_fec_decode_hard(
    const uint8_t *coded,
    size_t coded_bits,
    uint8_t *output,
    size_t output_capacity,
    size_t expected_output_bytes,
    lm_air_fec_workspace_t *workspace,
    uint32_t *path_metric) {
    return fec_decode(coded,
                      coded_bits,
                      output,
                      output_capacity,
                      expected_output_bytes,
                      workspace,
                      path_metric,
                      hard_metric);
}

lm_air_status_t lm_air_fec_decode_soft(
    const float *coded_llr,
    size_t coded_bits,
    uint8_t *output,
    size_t output_capacity,
    size_t expected_output_bytes,
    lm_air_fec_workspace_t *workspace,
    uint32_t *path_metric) {
    return fec_decode(coded_llr,
                      coded_bits,
                      output,
                      output_capacity,
                      expected_output_bytes,
                      workspace,
                      path_metric,
                      soft_metric);
}

lm_air_status_t lm_air_interleave_bits(
    const uint8_t *input,
    size_t bit_count,
    uint8_t *output,
    size_t output_capacity,
    uint8_t rows) {
    size_t bytes = (bit_count + 7u) / 8u;
    size_t columns;
    size_t out_bit = 0u;
    size_t column;
    size_t row;

    if (input == NULL || output == NULL || input == output || rows == 0u) {
        return LM_AIR_ERR_ARGUMENT;
    }
    if (bytes > output_capacity) {
        return LM_AIR_ERR_CAPACITY;
    }
    memset(output, 0, bytes);
    if (bit_count == 0u) {
        return LM_AIR_OK;
    }
    columns = (bit_count + rows - 1u) / rows;
    for (column = 0u; column < columns; ++column) {
        for (row = 0u; row < rows; ++row) {
            size_t in_bit = row * columns + column;
            if (in_bit < bit_count) {
                set_bit(output, out_bit++, get_bit(input, in_bit));
            }
        }
    }
    return out_bit == bit_count ? LM_AIR_OK : LM_AIR_ERR_STATE;
}

lm_air_status_t lm_air_deinterleave_bits(
    const uint8_t *input,
    size_t bit_count,
    uint8_t *output,
    size_t output_capacity,
    uint8_t rows) {
    size_t bytes = (bit_count + 7u) / 8u;
    size_t columns;
    size_t in_bit = 0u;
    size_t column;
    size_t row;

    if (input == NULL || output == NULL || input == output || rows == 0u) {
        return LM_AIR_ERR_ARGUMENT;
    }
    if (bytes > output_capacity) {
        return LM_AIR_ERR_CAPACITY;
    }
    memset(output, 0, bytes);
    if (bit_count == 0u) {
        return LM_AIR_OK;
    }
    columns = (bit_count + rows - 1u) / rows;
    for (column = 0u; column < columns; ++column) {
        for (row = 0u; row < rows; ++row) {
            size_t out_bit = row * columns + column;
            if (out_bit < bit_count) {
                set_bit(output, out_bit, get_bit(input, in_bit++));
            }
        }
    }
    return in_bit == bit_count ? LM_AIR_OK : LM_AIR_ERR_STATE;
}
