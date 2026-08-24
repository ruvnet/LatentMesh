#ifndef LM_AIR_CRC_H
#define LM_AIR_CRC_H

#include <stddef.h>
#include <stdint.h>

uint16_t lm_air_crc16_ccitt(const uint8_t *data, size_t len);

#endif
