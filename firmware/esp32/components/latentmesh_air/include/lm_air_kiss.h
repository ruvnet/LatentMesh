#ifndef LM_AIR_KISS_H
#define LM_AIR_KISS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "lm_air_packet.h"

#define LM_KISS_FEND 0xc0u
#define LM_KISS_FESC 0xdbu
#define LM_KISS_TFEND 0xdcu
#define LM_KISS_TFESC 0xddu
#define LM_KISS_CMD_DATA 0x00u

typedef struct {
    bool in_frame;
    bool escaped;
    bool overflow;
    bool command_seen;
    size_t len;
    uint8_t command;
    uint8_t data[LM_AIR_MAX_FRAME_SIZE];
} lm_kiss_decoder_t;

void lm_kiss_decoder_reset(lm_kiss_decoder_t *decoder);
int lm_kiss_encode(uint8_t port,
                   const uint8_t *payload,
                   size_t payload_len,
                   uint8_t *out,
                   size_t out_capacity,
                   size_t *out_len);
/* Returns 1 for a complete data frame, 0 for no frame, and -1 for a rejected
 * or overflowing frame.  Non-data KISS commands are consumed and ignored. */
int lm_kiss_decoder_feed(lm_kiss_decoder_t *decoder,
                         uint8_t byte,
                         uint8_t *port,
                         lm_air_packet_t *out);

#endif
