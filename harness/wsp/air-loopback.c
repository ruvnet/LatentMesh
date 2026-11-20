/* Binary stdin/stdout fixture using the existing Air envelope and frame codec.
 * The application body carries its Ed25519 signature. No radio or network I/O. */
#include "latentmesh_air/air.h"
#include <stdio.h>

typedef struct { lm_air_rx_t rx; size_t frames; size_t wire_bytes; } loop_t;

static lm_air_status_t deliver(void *user, const lm_air_message_t *message) {
    (void)user;
    return fwrite(message->body, 1, message->body_len, stdout) == message->body_len
        ? LM_AIR_OK : LM_AIR_ERR_CALLBACK;
}

static lm_air_status_t emit(void *user, const lm_air_block_t *block) {
    loop_t *loop = user;
    if (block->raw_len > 227u) return LM_AIR_ERR_CAPACITY;
    loop->frames++;
    loop->wire_bytes += block->raw_len;
    return lm_air_rx_ingest_block(&loop->rx, block);
}

int main(void) {
    static loop_t loop;
    lm_air_tx_t tx;
    lm_air_profile_config_t profile;
    uint8_t body[LM_AIR_MAX_MESSAGE_BYTES + 1u];
    size_t len = fread(body, 1, sizeof(body), stdin);
    if (ferror(stdin) || len == 0u || len > LM_AIR_MAX_MESSAGE_BYTES) return 1;
    lm_air_message_t message = { .source_id = 1u, .epoch = 1u, .message_id = 1u,
        .logical_sequence = 1u, .class_id = LM_AIR_CLASS_DIAGNOSTIC,
        .body = body, .body_len = (uint16_t)len };
    if (lm_air_profile_defaults(LM_AIR_PROFILE_MESHTASTIC, &profile) != LM_AIR_OK ||
        lm_air_rx_init(&loop.rx, deliver, NULL, NULL) != LM_AIR_OK ||
        lm_air_tx_init(&tx, 1u, 1u, &profile, NULL) != LM_AIR_OK ||
        lm_air_tx_send(&tx, &message, emit, &loop) != LM_AIR_OK ||
        loop.rx.stats.messages_delivered != 1u) return 1;
    fprintf(stderr, "%zu %zu\n", loop.frames, loop.wire_bytes);
    return fflush(stdout) == 0 ? 0 : 1;
}
