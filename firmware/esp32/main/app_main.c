#include "esp_check.h"
#include "esp_log.h"
#include "nvs_flash.h"

#include "lm_air_ble.h"
#include "lm_air_i2s.h"
#include "lm_air_kiss_uart.h"
#include "lm_air_metrics.h"
#include "lm_air_pipeline.h"
#include "lm_air_policy.h"
#include "lm_air_radio.h"
#include "lm_air_wifi.h"

static const char *TAG = "latentmesh_air";

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
}
