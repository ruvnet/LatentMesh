#include "lm_air_policy.h"

#include <ctype.h>
#include <stdio.h>
#include <string.h>

#ifdef ESP_PLATFORM
#include "driver/gpio.h"
#include "sdkconfig.h"
#endif

static bool fail(char *reason, size_t size, const char *message)
{
    if (reason != NULL && size > 0) {
        snprintf(reason, size, "%s", message);
    }
    return false;
}

static bool pass(char *reason, size_t size)
{
    if (reason != NULL && size > 0) {
        snprintf(reason, size, "policy satisfied");
    }
    return true;
}

bool lm_air_callsign_valid(const char *callsign)
{
    if (callsign == NULL) {
        return false;
    }
    const size_t len = strlen(callsign);
    if (len < 3 || len > 16) {
        return false;
    }
    bool letter = false;
    bool digit = false;
    for (size_t i = 0; i < len; ++i) {
        const unsigned char c = (unsigned char)callsign[i];
        if (isalpha(c)) {
            letter = true;
        } else if (isdigit(c)) {
            digit = true;
        } else if (c != '/') {
            return false;
        }
    }
    return letter && digit;
}

bool lm_air_policy_check(const lm_air_tx_policy_t *policy,
                         uint8_t payload_flags,
                         char *reason,
                         size_t reason_size)
{
    if (policy == NULL) {
        return fail(reason, reason_size, "missing policy");
    }
    if (!policy->external_rf_tx_enabled) {
        return fail(reason, reason_size, "external RF TX disabled at build time");
    }
    if (!policy->operator_attested) {
        return fail(reason, reason_size, "operator attestation missing");
    }
    if (!policy->hardware_interlock_asserted) {
        return fail(reason, reason_size, "hardware TX interlock is open");
    }
    if (!lm_air_callsign_valid(policy->callsign)) {
        return fail(reason, reason_size, "assigned call sign missing or invalid");
    }
    if ((payload_flags & LM_AIR_PAYLOAD_PUBLIC_CODEC) == 0) {
        return fail(reason, reason_size, "publicly documented codec flag required");
    }
    if ((payload_flags & LM_AIR_PAYLOAD_ENCRYPTED) != 0 &&
        (policy->jurisdiction == LM_AIR_JURISDICTION_CANADA ||
         policy->jurisdiction == LM_AIR_JURISDICTION_US)) {
        return fail(reason, reason_size, "obscured payload blocked for CA/US amateur profile");
    }
    return pass(reason, reason_size);
}

bool lm_air_packet_contains_callsign(const lm_air_packet_t *packet,
                                     const char *callsign)
{
    if (packet == NULL || !lm_air_callsign_valid(callsign)) {
        return false;
    }
    const size_t wanted = strlen(callsign);
    if (wanted > packet->len) {
        return false;
    }
    for (size_t i = 0; i + wanted <= packet->len; ++i) {
        size_t j = 0;
        while (j < wanted &&
               toupper((unsigned char)packet->data[i + j]) ==
                   toupper((unsigned char)callsign[j])) {
            ++j;
        }
        if (j == wanted) {
            return true;
        }
    }
    return false;
}

bool lm_air_policy_allow_packet(const lm_air_tx_policy_t *policy,
                                lm_air_tx_policy_runtime_t *runtime,
                                const lm_air_packet_t *packet,
                                uint64_t now_ms,
                                char *reason,
                                size_t reason_size)
{
    if (runtime == NULL || packet == NULL ||
        !lm_air_policy_check(policy, packet->flags, reason, reason_size)) {
        return false;
    }
    const bool identification =
        (packet->flags & LM_AIR_PAYLOAD_IDENTIFICATION) != 0 &&
        lm_air_packet_contains_callsign(packet, policy->callsign);
    const bool due = !runtime->identification_seen ||
                     now_ms - runtime->last_identification_ms >= policy->callsign_interval_ms;
    if (due && !identification) {
        return fail(reason, reason_size, "clear call-sign identification is due");
    }
    if (identification) {
        runtime->identification_seen = true;
        runtime->last_identification_ms = now_ms;
    }
    return pass(reason, reason_size);
}

lm_air_tx_policy_t lm_air_policy_from_config(void)
{
    lm_air_tx_policy_t policy = {0};
#ifdef ESP_PLATFORM
#if CONFIG_LM_RF_TX_ENABLE
    policy.external_rf_tx_enabled = true;
#if CONFIG_LM_OPERATOR_ATTESTED
    policy.operator_attested = true;
#endif
#if CONFIG_LM_JURISDICTION_CANADA
    policy.jurisdiction = LM_AIR_JURISDICTION_CANADA;
#elif CONFIG_LM_JURISDICTION_US
    policy.jurisdiction = LM_AIR_JURISDICTION_US;
#else
    policy.jurisdiction = LM_AIR_JURISDICTION_OTHER;
#endif
    policy.callsign_interval_ms = (uint32_t)CONFIG_LM_CALLSIGN_INTERVAL_SECONDS * 1000u;
    snprintf(policy.callsign, sizeof(policy.callsign), "%s", CONFIG_LM_AMATEUR_CALLSIGN);
    if (CONFIG_LM_RF_TX_INTERLOCK_GPIO < 0) {
        policy.hardware_interlock_asserted = true;
    } else {
        const gpio_num_t pin = (gpio_num_t)CONFIG_LM_RF_TX_INTERLOCK_GPIO;
        gpio_config_t io = {
            .pin_bit_mask = 1ULL << (unsigned)pin,
            .mode = GPIO_MODE_INPUT,
            .pull_down_en = GPIO_PULLDOWN_ENABLE,
        };
        policy.hardware_interlock_asserted =
            gpio_config(&io) == ESP_OK && gpio_get_level(pin) == 1;
    }
#else
    policy.hardware_interlock_asserted = false;
#endif
#else
    policy.callsign_interval_ms = 540000u;
#endif
    return policy;
}
