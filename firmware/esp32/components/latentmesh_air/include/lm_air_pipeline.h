#ifndef LM_AIR_PIPELINE_H
#define LM_AIR_PIPELINE_H

#include <stdint.h>
#include "esp_err.h"
#include "latentmesh_air.h"

esp_err_t lm_air_pipeline_start(const lm_air_crypto_hooks_t *crypto);

/* Encodes one semantic message with the canonical portable C transmitter and
 * publishes each resulting wire frame to the requested link queues. */
esp_err_t lm_air_pipeline_send(const lm_air_message_t *message,
                               uint32_t link_mask,
                               uint8_t transport_policy_flags);

/* Override in the application. The body pointer is valid only during the
 * callback; copy any state that must outlive the call. */
void lm_air_pipeline_message_hook(const lm_air_message_t *message);

#endif
