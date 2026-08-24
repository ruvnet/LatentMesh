#include "lm_air_kiss_uart.h"

#include "driver/uart.h"
#include "esp_check.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "lm_air_kiss.h"
#include "lm_air_metrics.h"
#include "lm_air_policy.h"
#include "lm_air_radio.h"

static const char *TAG = "lm_kiss";
static const uart_port_t PORT = (uart_port_t)CONFIG_LM_KISS_UART_NUM;
static lm_air_tx_policy_t s_policy;
static lm_air_tx_policy_runtime_t s_policy_runtime;

static void kiss_rx_task(void *arg)
{
    (void)arg;
    lm_kiss_decoder_t decoder;
    lm_kiss_decoder_reset(&decoder);
    uint8_t bytes[128];
    for (;;) {
        const int count = uart_read_bytes(PORT, bytes, sizeof(bytes), pdMS_TO_TICKS(100));
        for (int i = 0; i < count; ++i) {
            lm_air_packet_t packet;
            uint8_t port;
            const int result = lm_kiss_decoder_feed(&decoder, bytes[i], &port, &packet);
            (void)port;
            if (result == 1) {
                if (lm_air_radio_ingest(LM_AIR_LINK_KISS, packet.data, packet.len,
                                        packet.flags, 0) == ESP_OK) {
                    lm_air_metric_add(LM_METRIC_KISS_RX, 1);
                }
            } else if (result < 0) {
                lm_air_metric_add(LM_METRIC_BAD_FRAME, 1);
            }
        }
    }
}

static void kiss_tx_task(void *arg)
{
    (void)arg;
    lm_air_packet_t packet;
    uint8_t encoded[2 * LM_AIR_MAX_FRAME_SIZE + 3];
    for (;;) {
        if (xQueueReceive(lm_air_radio_tx_queue(LM_AIR_LINK_KISS), &packet,
                          portMAX_DELAY) != pdTRUE) {
            continue;
        }
        char reason[96];
        const uint64_t now_ms = (uint64_t)esp_timer_get_time() / 1000u;
        if (!lm_air_policy_allow_packet(&s_policy, &s_policy_runtime, &packet,
                                        now_ms, reason, sizeof(reason))) {
            ESP_LOGW(TAG, "TX blocked: %s", reason);
            lm_air_metric_add(LM_METRIC_POLICY_BLOCK, 1);
            continue;
        }
        size_t encoded_len = 0;
        if (lm_kiss_encode(0, packet.data, packet.len, encoded, sizeof(encoded),
                           &encoded_len) != 0 ||
            uart_write_bytes(PORT, encoded, encoded_len) != (int)encoded_len) {
            lm_air_metric_add(LM_METRIC_QUEUE_DROP, 1);
            continue;
        }
        lm_air_metric_add(LM_METRIC_KISS_TX, 1);
    }
}

esp_err_t lm_air_kiss_uart_start(void)
{
    const uart_config_t config = {
        .baud_rate = CONFIG_LM_KISS_UART_BAUD,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_DEFAULT,
    };
    ESP_RETURN_ON_ERROR(uart_driver_install(PORT, 2048, 2048, 0, NULL, 0),
                        TAG, "driver install");
    ESP_RETURN_ON_ERROR(uart_param_config(PORT, &config), TAG, "UART config");
    ESP_RETURN_ON_ERROR(uart_set_pin(PORT, CONFIG_LM_KISS_UART_TX_GPIO,
                                     CONFIG_LM_KISS_UART_RX_GPIO,
                                     UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE),
                        TAG, "UART pins");
    s_policy = lm_air_policy_from_config();
    if (xTaskCreate(kiss_rx_task, "lm_kiss_rx", 3072, NULL, 5, NULL) != pdPASS ||
        xTaskCreate(kiss_tx_task, "lm_kiss_tx", 3072, NULL, 5, NULL) != pdPASS) {
        return ESP_ERR_NO_MEM;
    }
    return ESP_OK;
}
