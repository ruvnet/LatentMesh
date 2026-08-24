#include "lm_air_i2s.h"

#include <string.h>
#include "driver/i2s_std.h"
#include "esp_check.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/queue.h"
#include "freertos/task.h"

#include "lm_air_metrics.h"
#include "lm_air_packet.h"
#include "lm_air_policy.h"

static const char *TAG = "lm_i2s";
static i2s_chan_handle_t s_tx_channel;
static i2s_chan_handle_t s_rx_channel;
static QueueHandle_t s_pcm_tx;
static lm_air_tx_policy_t s_policy;
static lm_air_tx_policy_runtime_t s_policy_runtime;

__attribute__((weak)) void lm_air_i2s_rx_hook(const int16_t *samples,
                                               size_t sample_count)
{
    (void)samples;
    (void)sample_count;
}

static void i2s_rx_task(void *arg)
{
    (void)arg;
    int16_t samples[LM_AIR_PCM_BLOCK_SAMPLES];
    for (;;) {
        size_t bytes = 0;
        if (i2s_channel_read(s_rx_channel, samples, sizeof(samples), &bytes,
                             portMAX_DELAY) == ESP_OK && bytes > 0) {
            const size_t count = bytes / sizeof(samples[0]);
            lm_air_metric_add(LM_METRIC_AUDIO_RX_SAMPLES, count);
            lm_air_i2s_rx_hook(samples, count);
        }
    }
}

static void i2s_tx_task(void *arg)
{
    (void)arg;
    lm_air_pcm_block_t block;
    for (;;) {
        if (xQueueReceive(s_pcm_tx, &block, portMAX_DELAY) != pdTRUE) {
            continue;
        }
        char reason[96];
        if (!lm_air_policy_check(&s_policy, block.payload_flags,
                                 reason, sizeof(reason))) {
            ESP_LOGW(TAG, "PCM TX blocked: %s", reason);
            lm_air_metric_add(LM_METRIC_POLICY_BLOCK, 1);
            continue;
        }
        const uint64_t now_ms = (uint64_t)esp_timer_get_time() / 1000u;
        const bool identification =
            (block.payload_flags & LM_AIR_PAYLOAD_IDENTIFICATION) != 0;
        const bool identification_due = !s_policy_runtime.identification_seen ||
            now_ms - s_policy_runtime.last_identification_ms >=
                s_policy.callsign_interval_ms;
        if (identification_due && !identification) {
            ESP_LOGW(TAG, "%s", "PCM TX blocked: operator identification is due");
            lm_air_metric_add(LM_METRIC_POLICY_BLOCK, 1);
            continue;
        }
        if (identification) {
            s_policy_runtime.identification_seen = true;
            s_policy_runtime.last_identification_ms = now_ms;
        }
        size_t bytes = 0;
        const size_t wanted = block.sample_count * sizeof(block.samples[0]);
        if (i2s_channel_write(s_tx_channel, block.samples, wanted, &bytes,
                              portMAX_DELAY) == ESP_OK && bytes == wanted) {
            lm_air_metric_add(LM_METRIC_AUDIO_TX_SAMPLES, block.sample_count);
        } else {
            lm_air_metric_add(LM_METRIC_QUEUE_DROP, 1);
        }
    }
}

esp_err_t lm_air_i2s_submit_tx(const int16_t *samples,
                               size_t sample_count,
                               uint8_t payload_flags,
                               TickType_t timeout)
{
    if (samples == NULL || sample_count == 0 ||
        sample_count > LM_AIR_PCM_BLOCK_SAMPLES || s_pcm_tx == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    lm_air_pcm_block_t block = {
        .sample_count = (uint16_t)sample_count,
        .payload_flags = payload_flags,
    };
    memcpy(block.samples, samples, sample_count * sizeof(samples[0]));
    return xQueueSend(s_pcm_tx, &block, timeout) == pdTRUE
               ? ESP_OK
               : ESP_ERR_TIMEOUT;
}

esp_err_t lm_air_i2s_start(void)
{
    i2s_chan_config_t channel =
        I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_AUTO, I2S_ROLE_MASTER);
    ESP_RETURN_ON_ERROR(i2s_new_channel(&channel, &s_tx_channel, &s_rx_channel),
                        TAG, "new I2S channel");
    i2s_std_config_t standard = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(48000),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(
            I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = CONFIG_LM_I2S_BCLK_GPIO,
            .ws = CONFIG_LM_I2S_WS_GPIO,
            .dout = CONFIG_LM_I2S_DOUT_GPIO,
            .din = CONFIG_LM_I2S_DIN_GPIO,
            .invert_flags = {0},
        },
    };
    ESP_RETURN_ON_ERROR(i2s_channel_init_std_mode(s_tx_channel, &standard),
                        TAG, "TX standard mode");
    ESP_RETURN_ON_ERROR(i2s_channel_init_std_mode(s_rx_channel, &standard),
                        TAG, "RX standard mode");
    ESP_RETURN_ON_ERROR(i2s_channel_enable(s_tx_channel), TAG, "TX enable");
    ESP_RETURN_ON_ERROR(i2s_channel_enable(s_rx_channel), TAG, "RX enable");
    s_pcm_tx = xQueueCreate(CONFIG_LM_QUEUE_DEPTH, sizeof(lm_air_pcm_block_t));
    if (s_pcm_tx == NULL) {
        return ESP_ERR_NO_MEM;
    }
    s_policy = lm_air_policy_from_config();
    if (xTaskCreate(i2s_rx_task, "lm_i2s_rx", 3072, NULL, 6, NULL) != pdPASS ||
        xTaskCreate(i2s_tx_task, "lm_i2s_tx", 3072, NULL, 6, NULL) != pdPASS) {
        return ESP_ERR_NO_MEM;
    }
    return ESP_OK;
}
