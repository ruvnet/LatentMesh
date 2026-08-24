#ifndef LM_AIR_I2S_H
#define LM_AIR_I2S_H

#include <stddef.h>
#include <stdint.h>
#include "esp_err.h"
#include "freertos/FreeRTOS.h"

#define LM_AIR_PCM_BLOCK_SAMPLES 240u

typedef struct {
    uint16_t sample_count;
    uint8_t payload_flags;
    int16_t samples[LM_AIR_PCM_BLOCK_SAMPLES];
} lm_air_pcm_block_t;

esp_err_t lm_air_i2s_start(void);
esp_err_t lm_air_i2s_submit_tx(const int16_t *samples,
                               size_t sample_count,
                               uint8_t payload_flags,
                               TickType_t timeout);

/* Override this weak callback in an application component to pass receive
 * audio to the portable C modem/neural frontend.  It runs in the I2S task and
 * must not block. */
void lm_air_i2s_rx_hook(const int16_t *samples, size_t sample_count);

#endif
