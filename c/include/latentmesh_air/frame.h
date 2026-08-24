#ifndef LATENTMESH_AIR_FRAME_H
#define LATENTMESH_AIR_FRAME_H

#include "latentmesh_air/common.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The low nibble is carried in byte 1 of the compact wire frame. */
typedef enum lm_air_profile {
    LM_AIR_PROFILE_WIFI = 0,
    LM_AIR_PROFILE_BLE = 1,
    LM_AIR_PROFILE_HF_BPSK = 2,
    LM_AIR_PROFILE_HF_AFSK = 3,
    LM_AIR_PROFILE_VHF_AFSK = 4,
    LM_AIR_PROFILE_VHF_CPFSK = 5,
    LM_AIR_PROFILE_AM_AUDIO = 6,
    LM_AIR_PROFILE_FM_AUDIO = 7,
    LM_AIR_PROFILE_HAM_PACKET = 8
} lm_air_profile_t;

enum {
    LM_AIR_FLAG_ACK_REQUEST = 0x1u,
    LM_AIR_FLAG_FEC = 0x2u,
    LM_AIR_FLAG_CONTROL = 0x4u,
    LM_AIR_FLAG_SIGNED = 0x8u,
    LM_AIR_FLAG_AUTHENTICATED = LM_AIR_FLAG_SIGNED
};

typedef enum lm_air_class {
    LM_AIR_CLASS_TELEMETRY = 0,
    LM_AIR_CLASS_STATE_DELTA = 1,
    LM_AIR_CLASS_ACK = 2,
    LM_AIR_CLASS_CONTROL = 3,
    LM_AIR_CLASS_DIAGNOSTIC = 4
} lm_air_class_t;

typedef struct lm_air_frame {
    uint8_t profile;
    uint8_t flags;
    uint16_t stream_id;
    uint16_t sequence;
    uint8_t fragment_index;
    uint8_t fragment_count;
    uint8_t class_id;
    uint8_t priority;
    uint16_t state_tag;
    uint8_t payload_len;
    uint8_t payload[LM_AIR_MAX_FRAME_PAYLOAD];
} lm_air_frame_t;

uint32_t lm_air_crc32c(const uint8_t *data, size_t len);

size_t lm_air_frame_wire_size(const lm_air_frame_t *frame);

lm_air_status_t lm_air_frame_encode(
    const lm_air_frame_t *frame,
    uint8_t *wire,
    size_t wire_capacity,
    size_t *wire_len);

lm_air_status_t lm_air_frame_decode(
    const uint8_t *wire,
    size_t wire_len,
    lm_air_frame_t *frame);

#ifdef __cplusplus
}
#endif

#endif
