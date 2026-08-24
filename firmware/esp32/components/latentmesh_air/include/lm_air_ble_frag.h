#ifndef LM_AIR_BLE_FRAG_H
#define LM_AIR_BLE_FRAG_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "lm_air_packet.h"

#define LM_AIR_BLE_FRAG_MAGIC 0xa1u
#define LM_AIR_BLE_FRAG_VERSION 1u
#define LM_AIR_BLE_FRAG_HEADER_SIZE 10u
#define LM_AIR_BLE_MAX_FRAGMENTS 64u

typedef enum {
    LM_AIR_BLE_MORE = 0,
    LM_AIR_BLE_COMPLETE = 1,
    LM_AIR_BLE_REJECTED = -1,
} lm_air_ble_result_t;

typedef struct {
    bool active;
    uint16_t message_id;
    uint16_t total_len;
    uint16_t expected_crc;
    uint8_t fragment_count;
    uint64_t received_bitmap;
    uint8_t flags;
    lm_air_packet_t packet;
} lm_air_ble_reassembly_t;

size_t lm_air_ble_fragment_count(size_t message_len, size_t att_payload_capacity);
int lm_air_ble_make_fragment(const lm_air_packet_t *packet,
                             uint16_t message_id,
                             uint8_t fragment_index,
                             size_t att_payload_capacity,
                             uint8_t *out,
                             size_t out_capacity,
                             size_t *out_len);
void lm_air_ble_reassembly_reset(lm_air_ble_reassembly_t *state);
lm_air_ble_result_t lm_air_ble_reassembly_ingest(lm_air_ble_reassembly_t *state,
                                                  const uint8_t *fragment,
                                                  size_t fragment_len,
                                                  lm_air_packet_t *complete);

#endif
