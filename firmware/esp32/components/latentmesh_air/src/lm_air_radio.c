#include "lm_air_radio.h"

#include <string.h>
#include "lm_air_metrics.h"

static QueueHandle_t s_tx[LM_AIR_LINK_COUNT];
static QueueHandle_t s_rx;

esp_err_t lm_air_radio_init(void)
{
    for (unsigned i = 0; i < LM_AIR_LINK_COUNT; ++i) {
        s_tx[i] = xQueueCreate(CONFIG_LM_QUEUE_DEPTH, sizeof(lm_air_packet_t));
        if (s_tx[i] == NULL) {
            return ESP_ERR_NO_MEM;
        }
    }
    s_rx = xQueueCreate(CONFIG_LM_QUEUE_DEPTH, sizeof(lm_air_packet_t));
    return s_rx != NULL ? ESP_OK : ESP_ERR_NO_MEM;
}

QueueHandle_t lm_air_radio_tx_queue(lm_air_link_t link)
{
    return (unsigned)link < LM_AIR_LINK_COUNT ? s_tx[link] : NULL;
}

QueueHandle_t lm_air_radio_rx_queue(void)
{
    return s_rx;
}

esp_err_t lm_air_radio_publish(const lm_air_packet_t *packet,
                               uint32_t link_mask,
                               TickType_t timeout)
{
    if (packet == NULL || packet->len == 0 || packet->len > LM_AIR_MAX_FRAME_SIZE) {
        return ESP_ERR_INVALID_ARG;
    }
    esp_err_t result = ESP_OK;
    for (unsigned i = 0; i < LM_AIR_LINK_COUNT; ++i) {
        if ((link_mask & LM_AIR_LINK_MASK(i)) == 0) {
            continue;
        }
        if (s_tx[i] == NULL || xQueueSend(s_tx[i], packet, timeout) != pdTRUE) {
            lm_air_metric_add(LM_METRIC_QUEUE_DROP, 1);
            result = ESP_ERR_TIMEOUT;
        }
    }
    return result;
}

esp_err_t lm_air_radio_ingest(lm_air_link_t source,
                              const uint8_t *data,
                              size_t len,
                              uint8_t flags,
                              TickType_t timeout)
{
    if ((unsigned)source >= LM_AIR_LINK_COUNT || data == NULL || len == 0 ||
        len > LM_AIR_MAX_FRAME_SIZE || s_rx == NULL) {
        lm_air_metric_add(LM_METRIC_BAD_FRAME, 1);
        return ESP_ERR_INVALID_ARG;
    }
    lm_air_packet_t packet = {
        .len = (uint16_t)len,
        .source_link = (uint8_t)source,
        .flags = flags,
    };
    memcpy(packet.data, data, len);
    if (xQueueSend(s_rx, &packet, timeout) != pdTRUE) {
        lm_air_metric_add(LM_METRIC_QUEUE_DROP, 1);
        return ESP_ERR_TIMEOUT;
    }
    return ESP_OK;
}
