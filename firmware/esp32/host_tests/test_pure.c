#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "lm_air_ble_frag.h"
#include "lm_air_crc.h"
#include "lm_air_kiss.h"
#include "lm_air_policy.h"

static void test_crc(void)
{
    static const uint8_t vector[] = "123456789";
    assert(lm_air_crc16_ccitt(vector, sizeof(vector) - 1u) == 0x29b1u);
}

static void test_ble_out_of_order(void)
{
    lm_air_packet_t packet = {.len = 509};
    for (size_t i = 0; i < packet.len; ++i) {
        packet.data[i] = (uint8_t)(i * 31u);
    }
    const size_t capacity = 64;
    const size_t count = lm_air_ble_fragment_count(packet.len, capacity);
    assert(count == 10);
    uint8_t fragments[10][64];
    size_t lengths[10];
    for (size_t i = 0; i < count; ++i) {
        assert(lm_air_ble_make_fragment(&packet, 42, (uint8_t)i, capacity,
                                        fragments[i], sizeof(fragments[i]),
                                        &lengths[i]) == 0);
    }
    lm_air_ble_reassembly_t state;
    lm_air_ble_reassembly_reset(&state);
    lm_air_packet_t complete;
    static const uint8_t order[] = {9, 0, 2, 1, 4, 3, 6, 5, 8, 7};
    for (size_t i = 0; i < count; ++i) {
        const lm_air_ble_result_t result = lm_air_ble_reassembly_ingest(
            &state, fragments[order[i]], lengths[order[i]], &complete);
        assert(result == (i + 1u == count ? LM_AIR_BLE_COMPLETE : LM_AIR_BLE_MORE));
    }
    assert(complete.len == packet.len);
    assert(memcmp(complete.data, packet.data, packet.len) == 0);

    fragments[0][lengths[0] - 1u] ^= 0x01u;
    lm_air_ble_reassembly_reset(&state);
    for (size_t i = 0; i < count; ++i) {
        const lm_air_ble_result_t result = lm_air_ble_reassembly_ingest(
            &state, fragments[i], lengths[i], &complete);
        if (i + 1u == count) {
            assert(result == LM_AIR_BLE_REJECTED);
        }
    }
}

static void test_kiss(void)
{
    static const uint8_t payload[] = {0x01, LM_KISS_FEND, 0x02, LM_KISS_FESC, 0x03};
    uint8_t encoded[32];
    size_t encoded_len = 0;
    assert(lm_kiss_encode(3, payload, sizeof(payload), encoded, sizeof(encoded),
                          &encoded_len) == 0);
    lm_kiss_decoder_t decoder;
    lm_kiss_decoder_reset(&decoder);
    lm_air_packet_t decoded;
    uint8_t port = 0;
    int completed = 0;
    for (size_t i = 0; i < encoded_len; ++i) {
        const int result = lm_kiss_decoder_feed(&decoder, encoded[i], &port, &decoded);
        if (result == 1) {
            ++completed;
        }
    }
    assert(completed == 1);
    assert(port == 3);
    assert(decoded.len == sizeof(payload));
    assert(memcmp(decoded.data, payload, sizeof(payload)) == 0);
}

static void test_policy(void)
{
    lm_air_tx_policy_t disabled = lm_air_policy_from_config();
    char reason[96];
    assert(!lm_air_policy_check(&disabled, LM_AIR_PAYLOAD_PUBLIC_CODEC,
                                reason, sizeof(reason)));
    assert(strstr(reason, "disabled") != NULL);

    lm_air_tx_policy_t policy = {
        .external_rf_tx_enabled = true,
        .operator_attested = true,
        .hardware_interlock_asserted = true,
        .jurisdiction = LM_AIR_JURISDICTION_CANADA,
        .callsign_interval_ms = 540000,
        .callsign = "VE3RUV",
    };
    assert(lm_air_callsign_valid("VE3RUV"));
    assert(!lm_air_callsign_valid("not a call"));
    assert(!lm_air_policy_check(&policy,
                                LM_AIR_PAYLOAD_PUBLIC_CODEC |
                                    LM_AIR_PAYLOAD_ENCRYPTED,
                                reason, sizeof(reason)));
    assert(strstr(reason, "obscured") != NULL);
    policy.hardware_interlock_asserted = false;
    assert(!lm_air_policy_check(&policy, LM_AIR_PAYLOAD_PUBLIC_CODEC,
                                reason, sizeof(reason)));
    assert(strstr(reason, "interlock") != NULL);
    policy.hardware_interlock_asserted = true;

    lm_air_tx_policy_runtime_t runtime = {0};
    lm_air_packet_t data = {
        .len = 5,
        .flags = LM_AIR_PAYLOAD_PUBLIC_CODEC,
        .data = {'h', 'e', 'l', 'l', 'o'},
    };
    assert(!lm_air_policy_allow_packet(&policy, &runtime, &data, 1000,
                                       reason, sizeof(reason)));
    lm_air_packet_t id = {
        .len = 13,
        .flags = LM_AIR_PAYLOAD_PUBLIC_CODEC | LM_AIR_PAYLOAD_IDENTIFICATION,
        .data = {'I', 'D', ':', 'V', 'E', '3', 'R', 'U', 'V', ' ', 'L', 'M', 'A'},
    };
    assert(lm_air_policy_allow_packet(&policy, &runtime, &id, 1000,
                                      reason, sizeof(reason)));
    assert(lm_air_policy_allow_packet(&policy, &runtime, &data, 2000,
                                      reason, sizeof(reason)));
    assert(!lm_air_policy_allow_packet(&policy, &runtime, &data, 541001,
                                       reason, sizeof(reason)));
}

int main(void)
{
    test_crc();
    test_ble_out_of_order();
    test_kiss();
    test_policy();
    puts("latentmesh-air ESP32 pure logic: PASS");
    return 0;
}
