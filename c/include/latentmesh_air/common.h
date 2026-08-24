#ifndef LATENTMESH_AIR_COMMON_H
#define LATENTMESH_AIR_COMMON_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LM_AIR_VERSION 1u
#define LM_AIR_MAX_WIRE_BYTES 256u
#define LM_AIR_FRAME_HEADER_BYTES 12u
#define LM_AIR_FRAME_CRC_BYTES 4u
#define LM_AIR_MAX_FRAME_PAYLOAD 240u
#define LM_AIR_MAX_FRAGMENTS 32u
#define LM_AIR_MAX_ASSEMBLED_BYTES \
    (LM_AIR_MAX_FRAME_PAYLOAD * LM_AIR_MAX_FRAGMENTS)
#define LM_AIR_SEMANTIC_HEADER_BYTES 48u
#define LM_AIR_SIGNATURE_BYTES 64u
#define LM_AIR_MAX_MESSAGE_BYTES \
    (LM_AIR_MAX_ASSEMBLED_BYTES - LM_AIR_SEMANTIC_HEADER_BYTES - \
     LM_AIR_SIGNATURE_BYTES)

typedef enum lm_air_status {
    LM_AIR_OK = 0,
    LM_AIR_MORE = 1,
    LM_AIR_COMPLETE = 2,
    LM_AIR_ERR_ARGUMENT = -1,
    LM_AIR_ERR_CAPACITY = -2,
    LM_AIR_ERR_FORMAT = -3,
    LM_AIR_ERR_CRC = -4,
    LM_AIR_ERR_VERSION = -5,
    LM_AIR_ERR_REPLAY = -6,
    LM_AIR_ERR_DUPLICATE = -7,
    LM_AIR_ERR_AUTH = -8,
    LM_AIR_ERR_STATE = -9,
    LM_AIR_ERR_FEC = -10,
    LM_AIR_ERR_CALLBACK = -11
} lm_air_status_t;

#ifdef __cplusplus
}
#endif

#endif
