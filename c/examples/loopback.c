#include "latentmesh_air.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct loopback_context {
    lm_air_rx_t *rx;
    size_t blocks;
    size_t coded_bits;
    int failed;
} loopback_context_t;

static lm_air_status_t receive_message(
    void *user,
    const lm_air_message_t *message) {
    (void)user;
    printf("received source=%08x message=%u bytes=%u: %.*s\n",
           (unsigned)message->source_id,
           (unsigned)message->message_id,
           (unsigned)message->body_len,
           (int)message->body_len,
           (const char *)message->body);
    return LM_AIR_OK;
}

static lm_air_status_t emit_loopback(
    void *user,
    const lm_air_block_t *block) {
    loopback_context_t *context = (loopback_context_t *)user;
    lm_air_status_t status;
    ++context->blocks;
    context->coded_bits += block->bit_len;
    status = lm_air_rx_ingest_block(context->rx, block);
    if (status < 0) {
        fprintf(stderr, "receiver rejected block: %d\n", (int)status);
        context->failed = 1;
        return status;
    }
    return LM_AIR_OK;
}

static int parse_profile(const char *name, uint8_t *profile) {
    static const struct {
        const char *name;
        uint8_t profile;
    } names[] = {{"raw", LM_AIR_PROFILE_WIFI},
                 {"wifi", LM_AIR_PROFILE_WIFI},
                 {"ble", LM_AIR_PROFILE_BLE},
                 {"hf", LM_AIR_PROFILE_HF_AFSK},
                 {"vhf", LM_AIR_PROFILE_VHF_AFSK},
                 {"cpfsk", LM_AIR_PROFILE_VHF_CPFSK},
                 {"bpsk", LM_AIR_PROFILE_HF_BPSK},
                 {"am", LM_AIR_PROFILE_AM_AUDIO},
                 {"fm", LM_AIR_PROFILE_FM_AUDIO},
                 {"ham", LM_AIR_PROFILE_HAM_PACKET},
                 {"sdr", LM_AIR_PROFILE_HF_BPSK}};
    size_t i;
    for (i = 0u; i < sizeof(names) / sizeof(names[0]); ++i) {
        if (strcmp(name, names[i].name) == 0) {
            *profile = names[i].profile;
            return 1;
        }
    }
    return 0;
}

int main(int argc, char **argv) {
    const char *profile_name = argc > 1 ? argv[1] : "hf";
    const char *text = argc > 2 ? argv[2] : "LatentMesh Air loopback";
    uint8_t profile_id;
    lm_air_profile_config_t profile;
    lm_air_tx_t tx;
    lm_air_rx_t rx;
    lm_air_message_t message;
    loopback_context_t context;
    size_t i;

    if (!parse_profile(profile_name, &profile_id)) {
        fprintf(stderr,
                "profile must be raw, wifi, ble, hf, vhf, cpfsk, bpsk, am, "
                "fm, or sdr\n");
        return EXIT_FAILURE;
    }
    if (strlen(text) > LM_AIR_MAX_MESSAGE_BYTES) {
        fprintf(stderr, "message is too large\n");
        return EXIT_FAILURE;
    }
    if (lm_air_profile_defaults(profile_id, &profile) != LM_AIR_OK ||
        lm_air_tx_init(&tx, 7u, 1u, &profile, NULL) != LM_AIR_OK ||
        lm_air_rx_init(&rx, receive_message, NULL, NULL) != LM_AIR_OK) {
        fprintf(stderr, "initialization failed\n");
        return EXIT_FAILURE;
    }

    memset(&message, 0, sizeof(message));
    message.source_id = UINT32_C(0x5255564e);
    message.epoch = 1u;
    message.message_id = 1u;
    message.logical_sequence = 1u;
    message.class_id = 1u;
    message.priority = 12u;
    for (i = 0u; i < sizeof(message.state_hash); ++i) {
        message.state_hash[i] = (uint8_t)(i * 17u);
    }
    message.body = (const uint8_t *)text;
    message.body_len = (uint16_t)strlen(text);

    memset(&context, 0, sizeof(context));
    context.rx = &rx;
    if (lm_air_tx_send(&tx, &message, emit_loopback, &context) != LM_AIR_OK ||
        context.failed != 0) {
        return EXIT_FAILURE;
    }
    printf("profile=%s blocks=%zu coded_bits=%zu delivered=%u rejected=%u\n",
           profile_name,
           context.blocks,
           context.coded_bits,
           (unsigned)rx.stats.messages_delivered,
           (unsigned)rx.stats.frames_rejected);
    return rx.stats.messages_delivered == 1u ? EXIT_SUCCESS : EXIT_FAILURE;
}
