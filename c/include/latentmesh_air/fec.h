#ifndef LATENTMESH_AIR_FEC_H
#define LATENTMESH_AIR_FEC_H

#include "latentmesh_air/common.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LM_AIR_FEC_TAIL_BITS 6u
#define LM_AIR_FEC_MAX_STEPS \
    ((LM_AIR_MAX_WIRE_BYTES * 8u) + LM_AIR_FEC_TAIL_BITS)
#define LM_AIR_FEC_MAX_CODED_BITS (LM_AIR_FEC_MAX_STEPS * 2u)
#define LM_AIR_FEC_MAX_CODED_BYTES ((LM_AIR_FEC_MAX_CODED_BITS + 7u) / 8u)

typedef struct lm_air_fec_workspace {
    uint64_t survivor[LM_AIR_FEC_MAX_STEPS];
    int32_t metric_a[64];
    int32_t metric_b[64];
} lm_air_fec_workspace_t;

size_t lm_air_fec_encoded_bits(size_t input_bytes);

lm_air_status_t lm_air_fec_encode(
    const uint8_t *input,
    size_t input_bytes,
    uint8_t *coded,
    size_t coded_capacity,
    size_t *coded_bits);

lm_air_status_t lm_air_fec_decode_hard(
    const uint8_t *coded,
    size_t coded_bits,
    uint8_t *output,
    size_t output_capacity,
    size_t expected_output_bytes,
    lm_air_fec_workspace_t *workspace,
    uint32_t *path_metric);

lm_air_status_t lm_air_fec_decode_soft(
    const float *coded_llr,
    size_t coded_bits,
    uint8_t *output,
    size_t output_capacity,
    size_t expected_output_bytes,
    lm_air_fec_workspace_t *workspace,
    uint32_t *path_metric);

lm_air_status_t lm_air_interleave_bits(
    const uint8_t *input,
    size_t bit_count,
    uint8_t *output,
    size_t output_capacity,
    uint8_t rows);

lm_air_status_t lm_air_deinterleave_bits(
    const uint8_t *input,
    size_t bit_count,
    uint8_t *output,
    size_t output_capacity,
    uint8_t rows);

#ifdef __cplusplus
}
#endif

#endif
