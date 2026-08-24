#ifndef LM_AIR_PACKET_H
#define LM_AIR_PACKET_H

#include <stddef.h>
#include <stdint.h>

#ifndef LM_AIR_MAX_FRAME_SIZE
#ifdef CONFIG_LM_MAX_FRAME_SIZE
#define LM_AIR_MAX_FRAME_SIZE CONFIG_LM_MAX_FRAME_SIZE
#else
#define LM_AIR_MAX_FRAME_SIZE 512
#endif
#endif

typedef struct {
    uint16_t len;
    uint8_t source_link;
    uint8_t flags;
    uint8_t data[LM_AIR_MAX_FRAME_SIZE];
} lm_air_packet_t;

enum {
    LM_AIR_PAYLOAD_PUBLIC_CODEC = 1u << 0,
    LM_AIR_PAYLOAD_IDENTIFICATION = 1u << 1,
    LM_AIR_PAYLOAD_ENCRYPTED = 1u << 2,
    LM_AIR_PAYLOAD_CRITICAL_STATE = 1u << 3,
};

#endif
