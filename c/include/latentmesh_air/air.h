#ifndef LATENTMESH_AIR_AIR_H
#define LATENTMESH_AIR_AIR_H

#include "latentmesh_air/fec.h"
#include "latentmesh_air/frame.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef LM_AIR_RX_REASSEMBLY_SLOTS
#define LM_AIR_RX_REASSEMBLY_SLOTS 2u
#endif

#ifndef LM_AIR_RX_REPLAY_STREAMS
#define LM_AIR_RX_REPLAY_STREAMS 8u
#endif

typedef int (*lm_air_sign_fn)(
    void *user,
    const uint8_t *data,
    size_t data_len,
    uint8_t signature[LM_AIR_SIGNATURE_BYTES]);

typedef int (*lm_air_verify_fn)(
    void *user,
    const uint8_t *data,
    size_t data_len,
    const uint8_t signature[LM_AIR_SIGNATURE_BYTES]);

typedef struct lm_air_crypto_hooks {
    lm_air_sign_fn sign;
    lm_air_verify_fn verify;
    void *user;
} lm_air_crypto_hooks_t;

typedef struct lm_air_profile_config {
    uint8_t profile;
    uint8_t fragment_payload_bytes;
    uint8_t use_fec;
    uint8_t interleave_rows;
} lm_air_profile_config_t;

lm_air_status_t lm_air_profile_defaults(
    uint8_t profile,
    lm_air_profile_config_t *config);

typedef struct lm_air_message {
    uint32_t source_id;
    uint32_t epoch;
    uint32_t message_id;
    uint64_t logical_sequence;
    uint8_t class_id;
    uint8_t priority;
    uint8_t state_hash[16];
    const uint8_t *body;
    uint16_t body_len;
    uint8_t authenticated;
} lm_air_message_t;

typedef struct lm_air_block {
    uint16_t raw_len;
    uint16_t bit_len;
    uint8_t fec;
    uint8_t interleave_rows;
    uint8_t data[LM_AIR_FEC_MAX_CODED_BYTES];
} lm_air_block_t;

typedef enum lm_air_tx_phase {
    LM_AIR_TX_IDLE = 0,
    LM_AIR_TX_EMITTING = 1,
    LM_AIR_TX_DONE = 2,
    LM_AIR_TX_FAILED = 3
} lm_air_tx_phase_t;

typedef struct lm_air_tx {
    lm_air_profile_config_t profile;
    lm_air_crypto_hooks_t crypto;
    uint16_t stream_id;
    uint16_t next_sequence;
    uint16_t active_sequence;
    uint8_t fragment_count;
    uint8_t fragment_index;
    uint8_t class_id;
    uint8_t priority;
    uint16_t state_tag;
    uint16_t envelope_len;
    lm_air_tx_phase_t phase;
    uint8_t envelope[LM_AIR_MAX_ASSEMBLED_BYTES];
} lm_air_tx_t;

typedef lm_air_status_t (*lm_air_emit_block_fn)(
    void *user,
    const lm_air_block_t *block);

lm_air_status_t lm_air_tx_init(
    lm_air_tx_t *tx,
    uint16_t stream_id,
    uint16_t initial_sequence,
    const lm_air_profile_config_t *profile,
    const lm_air_crypto_hooks_t *crypto);

lm_air_status_t lm_air_tx_begin(
    lm_air_tx_t *tx,
    const lm_air_message_t *message);

lm_air_status_t lm_air_tx_poll(lm_air_tx_t *tx, lm_air_block_t *block);

lm_air_status_t lm_air_tx_send(
    lm_air_tx_t *tx,
    const lm_air_message_t *message,
    lm_air_emit_block_fn emit,
    void *user);

typedef lm_air_status_t (*lm_air_receive_message_fn)(
    void *user,
    const lm_air_message_t *message);

typedef struct lm_air_reassembly_slot {
    uint8_t used;
    uint16_t stream_id;
    uint16_t sequence;
    uint8_t profile;
    uint8_t flags;
    uint8_t fragment_count;
    uint8_t class_id;
    uint8_t priority;
    uint16_t state_tag;
    uint32_t received_bitmap;
    uint32_t age;
    uint8_t fragment_len[LM_AIR_MAX_FRAGMENTS];
    uint8_t fragments[LM_AIR_MAX_FRAGMENTS][LM_AIR_MAX_FRAME_PAYLOAD];
} lm_air_reassembly_slot_t;

typedef struct lm_air_replay_entry {
    uint8_t used;
    uint16_t stream_id;
    uint16_t highest;
    uint64_t bitmap;
    uint32_t age;
} lm_air_replay_entry_t;

typedef struct lm_air_rx_stats {
    uint32_t frames_accepted;
    uint32_t frames_rejected;
    uint32_t messages_delivered;
    uint32_t replay_rejected;
    uint32_t duplicate_fragments;
    uint32_t reassembly_evictions;
    uint32_t auth_failures;
    uint32_t fec_failures;
} lm_air_rx_stats_t;

typedef struct lm_air_rx {
    lm_air_crypto_hooks_t crypto;
    lm_air_receive_message_fn receive;
    void *receive_user;
    uint32_t clock;
    lm_air_reassembly_slot_t slots[LM_AIR_RX_REASSEMBLY_SLOTS];
    lm_air_replay_entry_t replay[LM_AIR_RX_REPLAY_STREAMS];
    lm_air_fec_workspace_t fec_workspace;
    uint8_t scratch_a[LM_AIR_FEC_MAX_CODED_BYTES];
    uint8_t scratch_b[LM_AIR_FEC_MAX_CODED_BYTES];
    lm_air_rx_stats_t stats;
} lm_air_rx_t;

lm_air_status_t lm_air_rx_init(
    lm_air_rx_t *rx,
    lm_air_receive_message_fn receive,
    void *receive_user,
    const lm_air_crypto_hooks_t *crypto);

lm_air_status_t lm_air_rx_ingest_wire(
    lm_air_rx_t *rx,
    const uint8_t *wire,
    size_t wire_len);

lm_air_status_t lm_air_rx_ingest_block(
    lm_air_rx_t *rx,
    const lm_air_block_t *block);

void lm_air_rx_reset(lm_air_rx_t *rx);

#ifdef __cplusplus
}
#endif

#endif
