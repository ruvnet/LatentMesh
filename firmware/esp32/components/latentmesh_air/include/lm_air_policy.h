#ifndef LM_AIR_POLICY_H
#define LM_AIR_POLICY_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "lm_air_packet.h"

typedef enum {
    LM_AIR_JURISDICTION_CANADA,
    LM_AIR_JURISDICTION_US,
    LM_AIR_JURISDICTION_OTHER,
} lm_air_jurisdiction_t;

typedef struct {
    bool external_rf_tx_enabled;
    bool operator_attested;
    bool hardware_interlock_asserted;
    lm_air_jurisdiction_t jurisdiction;
    uint32_t callsign_interval_ms;
    char callsign[17];
} lm_air_tx_policy_t;

typedef struct {
    uint64_t last_identification_ms;
    bool identification_seen;
} lm_air_tx_policy_runtime_t;

lm_air_tx_policy_t lm_air_policy_from_config(void);
bool lm_air_policy_check(const lm_air_tx_policy_t *policy,
                         uint8_t payload_flags,
                         char *reason,
                         size_t reason_size);
bool lm_air_policy_allow_packet(const lm_air_tx_policy_t *policy,
                                lm_air_tx_policy_runtime_t *runtime,
                                const lm_air_packet_t *packet,
                                uint64_t now_ms,
                                char *reason,
                                size_t reason_size);
bool lm_air_callsign_valid(const char *callsign);
bool lm_air_packet_contains_callsign(const lm_air_packet_t *packet,
                                     const char *callsign);

#endif
