#ifndef LM_AIR_RADIO_H
#define LM_AIR_RADIO_H

#include <stddef.h>
#include <stdint.h>
#include "esp_err.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "lm_air_packet.h"

typedef enum {
    LM_AIR_LINK_WIFI = 0,
    LM_AIR_LINK_BLE = 1,
    LM_AIR_LINK_KISS = 2,
    LM_AIR_LINK_AUDIO = 3,
    LM_AIR_LINK_COUNT
} lm_air_link_t;

#define LM_AIR_LINK_MASK(link) (1u << (unsigned)(link))
#define LM_AIR_LINK_MASK_ALL ((1u << LM_AIR_LINK_COUNT) - 1u)

esp_err_t lm_air_radio_init(void);
QueueHandle_t lm_air_radio_tx_queue(lm_air_link_t link);
QueueHandle_t lm_air_radio_rx_queue(void);
esp_err_t lm_air_radio_publish(const lm_air_packet_t *packet,
                               uint32_t link_mask,
                               TickType_t timeout);
esp_err_t lm_air_radio_ingest(lm_air_link_t source,
                              const uint8_t *data,
                              size_t len,
                              uint8_t flags,
                              TickType_t timeout);

#endif
