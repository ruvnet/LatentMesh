#include "lm_air_pipeline.h"

#include <stdbool.h>
#include <string.h>
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"

#include "lm_air_metrics.h"
#include "lm_air_packet.h"
#include "lm_air_radio.h"

static const char *TAG = "lm_pipeline";
static lm_air_rx_t s_rx;
static lm_air_tx_t s_tx;
static lm_air_crypto_hooks_t s_crypto;
static SemaphoreHandle_t s_tx_mutex;
static uint16_t s_next_sequence[LM_AIR_LINK_COUNT] = {1, 1, 1, 1};

__attribute__((weak)) void lm_air_pipeline_message_hook(
    const lm_air_message_t *message)
{
    (void)message;
}

static lm_air_status_t receive_message(void *user,
                                       const lm_air_message_t *message)
{
    (void)user;
    lm_air_pipeline_message_hook(message);
    return LM_AIR_OK;
}

static void receive_task(void *arg)
{
    (void)arg;
    lm_air_packet_t packet;
    for (;;) {
        if (xQueueReceive(lm_air_radio_rx_queue(), &packet, portMAX_DELAY) != pdTRUE) {
            continue;
        }
        const lm_air_status_t status =
            lm_air_rx_ingest_wire(&s_rx, packet.data, packet.len);
        if (status < 0 && status != LM_AIR_ERR_DUPLICATE &&
            status != LM_AIR_ERR_REPLAY) {
            ESP_LOGW(TAG, "Air frame rejected: %d from link %u", (int)status,
                     (unsigned)packet.source_link);
            lm_air_metric_add(LM_METRIC_BAD_FRAME, 1);
        }
    }
}

typedef struct {
    lm_air_link_t link;
    uint8_t policy_flags;
} emit_context_t;

static lm_air_status_t emit_block(void *user, const lm_air_block_t *block)
{
    emit_context_t *context = user;
    if (context == NULL || block == NULL || block->fec != 0 ||
        block->bit_len != (uint16_t)(block->raw_len * 8u) ||
        block->raw_len > LM_AIR_MAX_FRAME_SIZE) {
        return LM_AIR_ERR_FORMAT;
    }
    lm_air_packet_t packet = {
        .len = block->raw_len,
        .source_link = 0xffu,
        .flags = context->policy_flags,
    };
    memcpy(packet.data, block->data, block->raw_len);
    return lm_air_radio_publish(&packet, LM_AIR_LINK_MASK(context->link),
                                pdMS_TO_TICKS(100)) == ESP_OK
               ? LM_AIR_OK
               : LM_AIR_ERR_CALLBACK;
}

static uint8_t profile_for_link(lm_air_link_t link)
{
    switch (link) {
    case LM_AIR_LINK_WIFI:
        return LM_AIR_PROFILE_WIFI;
    case LM_AIR_LINK_BLE:
        return LM_AIR_PROFILE_BLE;
    case LM_AIR_LINK_KISS:
        return LM_AIR_PROFILE_HAM_PACKET;
    default:
        return 0xffu;
    }
}

static bool link_enabled(lm_air_link_t link)
{
    switch (link) {
    case LM_AIR_LINK_WIFI:
#if CONFIG_LM_WIFI_ENABLE
        return true;
#else
        return false;
#endif
    case LM_AIR_LINK_BLE:
#if CONFIG_LM_BLE_ENABLE
        return true;
#else
        return false;
#endif
    case LM_AIR_LINK_KISS:
#if CONFIG_LM_UART_KISS_ENABLE
        return true;
#else
        return false;
#endif
    default:
        return false;
    }
}

esp_err_t lm_air_pipeline_send(const lm_air_message_t *message,
                               uint32_t link_mask,
                               uint8_t transport_policy_flags)
{
    if (message == NULL || s_tx_mutex == NULL || link_mask == 0) {
        return ESP_ERR_INVALID_ARG;
    }
    if (xSemaphoreTake(s_tx_mutex, pdMS_TO_TICKS(1000)) != pdTRUE) {
        return ESP_ERR_TIMEOUT;
    }
    esp_err_t result = ESP_OK;
    for (unsigned i = 0; i < LM_AIR_LINK_COUNT; ++i) {
        if ((link_mask & LM_AIR_LINK_MASK(i)) == 0) {
            continue;
        }
        if (!link_enabled((lm_air_link_t)i)) {
            result = ESP_ERR_NOT_SUPPORTED;
            continue;
        }
        const uint8_t profile_id = profile_for_link((lm_air_link_t)i);
        if (profile_id == 0xffu) {
            result = ESP_ERR_NOT_SUPPORTED;
            continue;
        }
        lm_air_profile_config_t profile;
        if (lm_air_profile_defaults(profile_id, &profile) != LM_AIR_OK ||
            profile.use_fec != 0 ||
            lm_air_tx_init(&s_tx, (uint16_t)(CONFIG_LM_STREAM_ID + i),
                           s_next_sequence[i], &profile, &s_crypto) != LM_AIR_OK) {
            result = ESP_FAIL;
            continue;
        }
        emit_context_t context = {
            .link = (lm_air_link_t)i,
            .policy_flags = transport_policy_flags,
        };
        if (lm_air_tx_send(&s_tx, message, emit_block, &context) != LM_AIR_OK) {
            result = ESP_FAIL;
            continue;
        }
        s_next_sequence[i] = s_tx.next_sequence;
    }
    xSemaphoreGive(s_tx_mutex);
    return result;
}

esp_err_t lm_air_pipeline_start(const lm_air_crypto_hooks_t *crypto)
{
    memset(&s_crypto, 0, sizeof(s_crypto));
    if (crypto != NULL) {
        s_crypto = *crypto;
    }
    if (lm_air_rx_init(&s_rx, receive_message, NULL, &s_crypto) != LM_AIR_OK) {
        return ESP_FAIL;
    }
    s_tx_mutex = xSemaphoreCreateMutex();
    if (s_tx_mutex == NULL) {
        return ESP_ERR_NO_MEM;
    }
    return xTaskCreate(receive_task, "lm_air_rx", 4096, NULL, 6, NULL) == pdPASS
               ? ESP_OK
               : ESP_ERR_NO_MEM;
}
