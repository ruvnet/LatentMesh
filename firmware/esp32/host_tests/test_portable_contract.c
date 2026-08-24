#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "latentmesh_air.h"

static unsigned s_received;
static uint32_t s_message_id;
static uint8_t s_body[32];
static size_t s_body_len;

static lm_air_status_t receive_message(void *user,
                                       const lm_air_message_t *message)
{
    (void)user;
    ++s_received;
    s_message_id = message->message_id;
    s_body_len = message->body_len;
    assert(s_body_len <= sizeof(s_body));
    memcpy(s_body, message->body, s_body_len);
    return LM_AIR_OK;
}

int main(void)
{
    lm_air_profile_config_t profile;
    assert(lm_air_profile_defaults(LM_AIR_PROFILE_WIFI, &profile) == LM_AIR_OK);
    assert(profile.use_fec == 0);
    assert(profile.interleave_rows == 0);

    lm_air_tx_t tx;
    lm_air_rx_t rx;
    assert(lm_air_tx_init(&tx, 3, 9, &profile, NULL) == LM_AIR_OK);
    assert(lm_air_rx_init(&rx, receive_message, NULL, NULL) == LM_AIR_OK);

    static const uint8_t body[] = "firmware contract";
    lm_air_message_t message = {
        .source_id = 7,
        .epoch = 2,
        .message_id = 99,
        .logical_sequence = 1001,
        .class_id = 1,
        .priority = 15,
        .body = body,
        .body_len = sizeof(body) - 1u,
    };
    message.state_hash[0] = 0x12;
    message.state_hash[1] = 0x34;
    assert(lm_air_tx_begin(&tx, &message) == LM_AIR_OK);

    lm_air_status_t status;
    do {
        lm_air_block_t block;
        status = lm_air_tx_poll(&tx, &block);
        assert(status == LM_AIR_MORE || status == LM_AIR_COMPLETE);
        assert(block.fec == 0);
        assert(block.bit_len == block.raw_len * 8u);
        const lm_air_status_t ingest =
            lm_air_rx_ingest_wire(&rx, block.data, block.raw_len);
        assert(ingest == LM_AIR_MORE || ingest == LM_AIR_COMPLETE);
    } while (status == LM_AIR_MORE);

    assert(s_received == 1);
    assert(s_message_id == 99);
    assert(s_body_len == sizeof(body) - 1u);
    assert(memcmp(s_body, body, s_body_len) == 0);
    puts("latentmesh-air ESP32 portable C contract: PASS");
    return 0;
}
