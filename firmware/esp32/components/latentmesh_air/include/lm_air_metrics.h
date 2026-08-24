#ifndef LM_AIR_METRICS_H
#define LM_AIR_METRICS_H

#include <stdint.h>
#include "esp_err.h"

typedef enum {
    LM_METRIC_WIFI_TX,
    LM_METRIC_WIFI_RX,
    LM_METRIC_BLE_TX,
    LM_METRIC_BLE_RX,
    LM_METRIC_KISS_TX,
    LM_METRIC_KISS_RX,
    LM_METRIC_AUDIO_TX_SAMPLES,
    LM_METRIC_AUDIO_RX_SAMPLES,
    LM_METRIC_QUEUE_DROP,
    LM_METRIC_BAD_FRAME,
    LM_METRIC_POLICY_BLOCK,
    LM_METRIC_COUNT
} lm_air_metric_id_t;

typedef struct {
    uint32_t values[LM_METRIC_COUNT];
} lm_air_metrics_snapshot_t;

esp_err_t lm_air_metrics_init(void);
void lm_air_metric_add(lm_air_metric_id_t id, uint32_t delta);
void lm_air_metrics_snapshot(lm_air_metrics_snapshot_t *out);

#endif
