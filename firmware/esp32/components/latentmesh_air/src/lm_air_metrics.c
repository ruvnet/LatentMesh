#include "lm_air_metrics.h"

#include <stdatomic.h>
#include <string.h>

static atomic_uint_fast32_t s_metrics[LM_METRIC_COUNT];

esp_err_t lm_air_metrics_init(void)
{
    for (unsigned i = 0; i < LM_METRIC_COUNT; ++i) {
        atomic_store_explicit(&s_metrics[i], 0, memory_order_relaxed);
    }
    return ESP_OK;
}

void lm_air_metric_add(lm_air_metric_id_t id, uint32_t delta)
{
    if ((unsigned)id < LM_METRIC_COUNT) {
        atomic_fetch_add_explicit(&s_metrics[id], delta, memory_order_relaxed);
    }
}

void lm_air_metrics_snapshot(lm_air_metrics_snapshot_t *out)
{
    if (out == NULL) {
        return;
    }
    for (unsigned i = 0; i < LM_METRIC_COUNT; ++i) {
        out->values[i] = atomic_load_explicit(&s_metrics[i], memory_order_relaxed);
    }
}
