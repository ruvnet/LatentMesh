#include <stdio.h>
#include <string.h>

#include "esp_check.h"
#include "esp_log.h"
#include "nvs_flash.h"
#if CONFIG_LM_DEMO_BEACON
#include "esp_mac.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#endif

#include "lm_air_ble.h"
#include "lm_air_i2s.h"
#include "lm_air_kiss_uart.h"
#include "lm_air_metrics.h"
#include "lm_air_pipeline.h"
#include "lm_air_policy.h"
#include "lm_air_radio.h"
#include "lm_air_wifi.h"

static const char *TAG = "latentmesh_air";

#if CONFIG_LM_DEMO_BEACON
/* Identifies this node to its peers.  The low three bytes of the station MAC
 * are enough to tell two boards apart on a bench. */
static uint32_t s_demo_source_id;

/* Overrides the weak no-op in lm_air_pipeline so a received message is visible
 * without writing any application code. */
void lm_air_pipeline_message_hook(const lm_air_message_t *message)
{
    if (message == NULL) {
        return;
    }
    ESP_LOGI(TAG, "demo rx: source=%06lX message=%lu bytes=%u authenticated=%u",
             (unsigned long)(message->source_id & 0xFFFFFFu),
             (unsigned long)message->message_id,
             (unsigned)message->body_len,
             (unsigned)message->authenticated);
}

static void demo_beacon_task(void *arg)
{
    (void)arg;
    static uint8_t body[CONFIG_LM_DEMO_BEACON_BYTES];
    uint32_t sequence = 0;

    for (;;) {
        /* Render the header into a fixed scratch buffer rather than straight
         * into body[]: body is sized by Kconfig and can legally be smaller
         * than the header, which makes a direct snprintf into it a
         * statically-provable truncation and so a -Werror=format-truncation
         * build failure across much of the configurable range. */
        char header[40];
        const int header_len = snprintf(header, sizeof(header),
                                        "latentmesh demo %06lX %lu ",
                                        (unsigned long)(s_demo_source_id & 0xFFFFFFu),
                                        (unsigned long)sequence);
        const size_t copied =
            (header_len > 0 && (size_t)header_len < sizeof(body))
                ? (size_t)header_len
                : 0u;
        if (copied != 0u) {
            memcpy(body, header, copied);
        }
        for (size_t i = copied; i < sizeof(body); i++) {
            body[i] = (uint8_t)('A' + ((i + sequence) % 26u));
        }

        lm_air_message_t message = {
            .source_id = s_demo_source_id,
            .epoch = 1,
            .message_id = sequence,
            .logical_sequence = sequence,
            .class_id = 1,
            .priority = 15,
            .body = body,
            .body_len = (uint16_t)sizeof(body),
        };
        const uint32_t links =
#if CONFIG_LM_WIFI_ENABLE
            LM_AIR_LINK_MASK(LM_AIR_LINK_WIFI) |
#endif
#if CONFIG_LM_BLE_ENABLE
            LM_AIR_LINK_MASK(LM_AIR_LINK_BLE) |
#endif
            0u;
        if (links != 0u) {
            const esp_err_t err = lm_air_pipeline_send(&message, links,
                                                       LM_AIR_PAYLOAD_PUBLIC_CODEC);
            ESP_LOGI(TAG, "demo tx: message=%lu bytes=%u result=%s",
                     (unsigned long)sequence, (unsigned)sizeof(body),
                     esp_err_to_name(err));
        }
        sequence++;
        vTaskDelay(pdMS_TO_TICKS(CONFIG_LM_DEMO_BEACON_PERIOD_MS));
    }
}

static void start_demo_beacon(void)
{
    uint8_t mac[6] = {0};
    ESP_ERROR_CHECK(esp_read_mac(mac, ESP_MAC_WIFI_STA));
    s_demo_source_id = ((uint32_t)mac[3] << 16) | ((uint32_t)mac[4] << 8) |
                       (uint32_t)mac[5];
    ESP_LOGI(TAG, "demo beacon enabled; source=%06lX period=%ums bytes=%u",
             (unsigned long)s_demo_source_id,
             (unsigned)CONFIG_LM_DEMO_BEACON_PERIOD_MS,
             (unsigned)CONFIG_LM_DEMO_BEACON_BYTES);
    xTaskCreate(demo_beacon_task, "lm_demo", 4096, NULL, 5, NULL);
}
#endif /* CONFIG_LM_DEMO_BEACON */

static esp_err_t start_transports(void)
{
#if CONFIG_LM_WIFI_ENABLE
    ESP_RETURN_ON_ERROR(lm_air_wifi_start(), TAG, "Wi-Fi start failed");
#endif
#if CONFIG_LM_BLE_ENABLE
    ESP_RETURN_ON_ERROR(lm_air_ble_start(), TAG, "BLE start failed");
#endif
#if CONFIG_LM_UART_KISS_ENABLE
    ESP_RETURN_ON_ERROR(lm_air_kiss_uart_start(), TAG, "KISS start failed");
#endif
#if CONFIG_LM_I2S_AUDIO_ENABLE
    ESP_RETURN_ON_ERROR(lm_air_i2s_start(), TAG, "I2S start failed");
#endif
    return ESP_OK;
}

void app_main(void)
{
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        err = nvs_flash_init();
    }
    ESP_ERROR_CHECK(err);

    ESP_ERROR_CHECK(lm_air_metrics_init());
    ESP_ERROR_CHECK(lm_air_radio_init());
    ESP_ERROR_CHECK(lm_air_pipeline_start(NULL));

    const lm_air_tx_policy_t policy = lm_air_policy_from_config();
    char reason[96];
    const bool external_tx_ready =
        lm_air_policy_check(&policy, LM_AIR_PAYLOAD_PUBLIC_CODEC, reason, sizeof(reason));
    ESP_LOGI(TAG, "LatentMesh Air node starting; external RF TX: %s (%s)",
             external_tx_ready ? "armed" : "blocked", reason);
    (void)external_tx_ready;

    ESP_ERROR_CHECK(start_transports());
    ESP_LOGI(TAG, "transport tasks started; max frame=%u bytes, queue depth=%u",
             (unsigned)CONFIG_LM_MAX_FRAME_SIZE,
             (unsigned)CONFIG_LM_QUEUE_DEPTH);

#if CONFIG_LM_DEMO_BEACON
    start_demo_beacon();
#endif
}
