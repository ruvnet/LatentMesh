#include "latentmesh_air/air.h"

#include <string.h>

#define LM_AIR_SEMANTIC_FLAG_AUTH 0x01u

static void put_u16_be(uint8_t *dst, uint16_t value) {
    dst[0] = (uint8_t)(value >> 8);
    dst[1] = (uint8_t)value;
}

static uint16_t get_u16_be(const uint8_t *src) {
    return (uint16_t)(((uint16_t)src[0] << 8) | (uint16_t)src[1]);
}

static void put_u32_be(uint8_t *dst, uint32_t value) {
    dst[0] = (uint8_t)(value >> 24);
    dst[1] = (uint8_t)(value >> 16);
    dst[2] = (uint8_t)(value >> 8);
    dst[3] = (uint8_t)value;
}

static uint32_t get_u32_be(const uint8_t *src) {
    return ((uint32_t)src[0] << 24) | ((uint32_t)src[1] << 16) |
           ((uint32_t)src[2] << 8) | (uint32_t)src[3];
}

static void put_u64_be(uint8_t *dst, uint64_t value) {
    unsigned i;
    for (i = 0u; i < 8u; ++i) {
        dst[i] = (uint8_t)(value >> (56u - i * 8u));
    }
}

static uint64_t get_u64_be(const uint8_t *src) {
    uint64_t value = 0u;
    unsigned i;
    for (i = 0u; i < 8u; ++i) {
        value = (value << 8) | src[i];
    }
    return value;
}

lm_air_status_t lm_air_profile_defaults(
    uint8_t profile,
    lm_air_profile_config_t *config) {
    if (config == NULL || profile > 0x0fu) {
        return LM_AIR_ERR_ARGUMENT;
    }
    config->profile = profile;
    config->fragment_payload_bytes = LM_AIR_MAX_FRAME_PAYLOAD;
    config->use_fec = 0u;
    config->interleave_rows = 0u;
    switch (profile) {
        case LM_AIR_PROFILE_WIFI:
            break;
        case LM_AIR_PROFILE_BLE:
            config->fragment_payload_bytes = 180u;
            break;
        case LM_AIR_PROFILE_HF_BPSK:
            config->fragment_payload_bytes = 224u;
            config->use_fec = 1u;
            config->interleave_rows = 8u;
            break;
        case LM_AIR_PROFILE_HF_AFSK:
            config->fragment_payload_bytes = 64u;
            config->use_fec = 1u;
            config->interleave_rows = 16u;
            break;
        case LM_AIR_PROFILE_VHF_AFSK:
            config->fragment_payload_bytes = 128u;
            config->use_fec = 1u;
            config->interleave_rows = 16u;
            break;
        case LM_AIR_PROFILE_VHF_CPFSK:
            config->fragment_payload_bytes = 160u;
            config->use_fec = 1u;
            config->interleave_rows = 16u;
            break;
        case LM_AIR_PROFILE_AM_AUDIO:
            config->fragment_payload_bytes = 96u;
            config->use_fec = 1u;
            config->interleave_rows = 16u;
            break;
        case LM_AIR_PROFILE_FM_AUDIO:
            config->fragment_payload_bytes = 160u;
            config->use_fec = 1u;
            config->interleave_rows = 8u;
            break;
        case LM_AIR_PROFILE_HAM_PACKET:
            config->fragment_payload_bytes = 240u;
            break;
        case LM_AIR_PROFILE_MESHTASTIC:
            /* ADR-019 (revised after live-firmware interop testing): the raw
             * MTU used here is 227, not mesh.proto's encoded-submessage
             * ceiling DATA_PAYLOAD_LEN (233) -- that field bounds the
             * *encoded* Data submessage, and the portnum varint tag/value
             * plus the payload bytes tag/length consume ~6 bytes of
             * protobuf field overhead a raw-payload MTU must leave headroom
             * for. 227 is also the empirically-reliable ceiling measured
             * live against meshtasticd v2.7.26 (portduino, simulated
             * radio): broadcasts up to 227 bytes round-tripped
             * consistently, and 232+ bytes were rejected with
             * Routing.Error.TOO_LARGE ("Error=7, return NAK and drop
             * packet"); see latentmesh-meshtastic's MESHTASTIC_FRAME_MTU
             * doc comment and examples/meshtasticd_interop.rs for the full
             * writeup. Meshtastic does not auto-fragment application
             * payloads, so Air's own 16-byte frame overhead comes out of
             * that 227-byte budget: 227 - 16 = 211 usable fragment payload
             * bytes. Meshtastic owns FEC and interleaving itself. */
            config->fragment_payload_bytes = 211u;
            break;
        default:
            return LM_AIR_ERR_FORMAT;
    }
    return LM_AIR_OK;
}

static int valid_profile_config(const lm_air_profile_config_t *profile) {
    if (profile == NULL || profile->profile > LM_AIR_PROFILE_MESHTASTIC ||
        profile->fragment_payload_bytes == 0u ||
        profile->fragment_payload_bytes > LM_AIR_MAX_FRAME_PAYLOAD ||
        profile->use_fec > 1u || profile->interleave_rows > 64u) {
        return 0;
    }
    if (profile->interleave_rows == 1u) {
        return 0;
    }
    return 1;
}

lm_air_status_t lm_air_tx_init(
    lm_air_tx_t *tx,
    uint16_t stream_id,
    uint16_t initial_sequence,
    const lm_air_profile_config_t *profile,
    const lm_air_crypto_hooks_t *crypto) {
    if (tx == NULL || valid_profile_config(profile) == 0) {
        return LM_AIR_ERR_ARGUMENT;
    }
    memset(tx, 0, sizeof(*tx));
    tx->profile = *profile;
    if (crypto != NULL) {
        tx->crypto = *crypto;
    }
    tx->stream_id = stream_id;
    tx->next_sequence = initial_sequence;
    tx->phase = LM_AIR_TX_IDLE;
    return LM_AIR_OK;
}

lm_air_status_t lm_air_tx_begin(
    lm_air_tx_t *tx,
    const lm_air_message_t *message) {
    size_t envelope_len;
    size_t canonical_len;
    uint8_t signature_len;

    if (tx == NULL || message == NULL ||
        (message->body == NULL && message->body_len != 0u) ||
        message->class_id > LM_AIR_CLASS_DIAGNOSTIC ||
        message->priority > 0x0fu) {
        return LM_AIR_ERR_ARGUMENT;
    }
    if (tx->phase == LM_AIR_TX_EMITTING) {
        return LM_AIR_ERR_STATE;
    }
    signature_len = message->authenticated != 0u ? LM_AIR_SIGNATURE_BYTES : 0u;
    if (message->authenticated != 0u && tx->crypto.sign == NULL) {
        tx->phase = LM_AIR_TX_FAILED;
        return LM_AIR_ERR_AUTH;
    }
    envelope_len = LM_AIR_SEMANTIC_HEADER_BYTES + (size_t)message->body_len +
                   signature_len;
    if (envelope_len > LM_AIR_MAX_ASSEMBLED_BYTES) {
        return LM_AIR_ERR_CAPACITY;
    }

    memset(tx->envelope, 0, LM_AIR_SEMANTIC_HEADER_BYTES);
    tx->envelope[0] = 'L';
    tx->envelope[1] = 'M';
    tx->envelope[2] = 'S';
    tx->envelope[3] = '1';
    tx->envelope[4] = LM_AIR_VERSION;
    tx->envelope[5] = message->authenticated != 0u
                          ? LM_AIR_SEMANTIC_FLAG_AUTH
                          : 0u;
    tx->envelope[6] = message->class_id;
    tx->envelope[7] = message->priority;
    put_u32_be(tx->envelope + 8, message->source_id);
    put_u32_be(tx->envelope + 12, message->epoch);
    put_u32_be(tx->envelope + 16, message->message_id);
    put_u64_be(tx->envelope + 20, message->logical_sequence);
    put_u16_be(tx->envelope + 28, message->body_len);
    memcpy(tx->envelope + 30, message->state_hash, 16u);
    tx->envelope[46] = signature_len;
    tx->envelope[47] = 0u;
    if (message->body_len != 0u) {
        memcpy(tx->envelope + LM_AIR_SEMANTIC_HEADER_BYTES,
               message->body,
               message->body_len);
    }
    canonical_len = LM_AIR_SEMANTIC_HEADER_BYTES + message->body_len;
    if (signature_len != 0u &&
        tx->crypto.sign(tx->crypto.user,
                        tx->envelope,
                        canonical_len,
                        tx->envelope + canonical_len) != 0) {
        tx->phase = LM_AIR_TX_FAILED;
        return LM_AIR_ERR_AUTH;
    }

    tx->envelope_len = (uint16_t)envelope_len;
    tx->active_sequence = tx->next_sequence;
    tx->fragment_count = (uint8_t)(
        (envelope_len + tx->profile.fragment_payload_bytes - 1u) /
        tx->profile.fragment_payload_bytes);
    if (tx->fragment_count == 0u ||
        tx->fragment_count > LM_AIR_MAX_FRAGMENTS) {
        tx->phase = LM_AIR_TX_FAILED;
        return LM_AIR_ERR_CAPACITY;
    }
    tx->fragment_index = 0u;
    tx->class_id = message->class_id;
    tx->priority = message->priority;
    tx->state_tag = get_u16_be(message->state_hash);
    tx->phase = LM_AIR_TX_EMITTING;
    return LM_AIR_OK;
}

lm_air_status_t lm_air_tx_poll(lm_air_tx_t *tx, lm_air_block_t *block) {
    lm_air_frame_t frame;
    uint8_t raw[LM_AIR_MAX_WIRE_BYTES];
    uint8_t coded[LM_AIR_FEC_MAX_CODED_BYTES];
    size_t raw_len;
    size_t offset;
    size_t remaining;
    size_t fragment_len;
    size_t bit_len;
    lm_air_status_t status;

    if (tx == NULL || block == NULL) {
        return LM_AIR_ERR_ARGUMENT;
    }
    if (tx->phase != LM_AIR_TX_EMITTING) {
        return tx->phase == LM_AIR_TX_DONE ? LM_AIR_COMPLETE
                                           : LM_AIR_ERR_STATE;
    }
    memset(&frame, 0, sizeof(frame));
    offset = (size_t)tx->fragment_index * tx->profile.fragment_payload_bytes;
    remaining = tx->envelope_len - offset;
    fragment_len = remaining < tx->profile.fragment_payload_bytes
                       ? remaining
                       : tx->profile.fragment_payload_bytes;
    frame.profile = tx->profile.profile;
    frame.flags = 0u;
    if (tx->profile.use_fec != 0u) {
        frame.flags |= LM_AIR_FLAG_FEC;
    }
    if (tx->envelope[5] & LM_AIR_SEMANTIC_FLAG_AUTH) {
        frame.flags |= LM_AIR_FLAG_AUTHENTICATED;
    }
    frame.stream_id = tx->stream_id;
    frame.sequence = tx->active_sequence;
    frame.fragment_index = tx->fragment_index;
    frame.fragment_count = tx->fragment_count;
    frame.class_id = tx->class_id;
    frame.priority = tx->priority;
    frame.state_tag = tx->state_tag;
    frame.payload_len = (uint8_t)fragment_len;
    memcpy(frame.payload, tx->envelope + offset, fragment_len);
    status = lm_air_frame_encode(&frame, raw, sizeof(raw), &raw_len);
    if (status != LM_AIR_OK) {
        tx->phase = LM_AIR_TX_FAILED;
        return status;
    }

    memset(block, 0, sizeof(*block));
    block->raw_len = (uint16_t)raw_len;
    block->fec = tx->profile.use_fec;
    block->interleave_rows = tx->profile.interleave_rows;
    if (block->fec != 0u) {
        status = lm_air_fec_encode(
            raw, raw_len, coded, sizeof(coded), &bit_len);
    } else {
        bit_len = raw_len * 8u;
        memcpy(coded, raw, raw_len);
        status = LM_AIR_OK;
    }
    if (status != LM_AIR_OK) {
        tx->phase = LM_AIR_TX_FAILED;
        return status;
    }
    if (block->interleave_rows != 0u) {
        status = lm_air_interleave_bits(coded,
                                        bit_len,
                                        block->data,
                                        sizeof(block->data),
                                        block->interleave_rows);
    } else {
        memcpy(block->data, coded, (bit_len + 7u) / 8u);
    }
    if (status != LM_AIR_OK) {
        tx->phase = LM_AIR_TX_FAILED;
        return status;
    }
    block->bit_len = (uint16_t)bit_len;

    ++tx->fragment_index;
    if (tx->fragment_index == tx->fragment_count) {
        tx->phase = LM_AIR_TX_DONE;
        ++tx->next_sequence;
        return LM_AIR_COMPLETE;
    }
    return LM_AIR_MORE;
}

lm_air_status_t lm_air_tx_send(
    lm_air_tx_t *tx,
    const lm_air_message_t *message,
    lm_air_emit_block_fn emit,
    void *user) {
    lm_air_status_t status;
    if (emit == NULL) {
        return LM_AIR_ERR_ARGUMENT;
    }
    status = lm_air_tx_begin(tx, message);
    if (status != LM_AIR_OK) {
        return status;
    }
    do {
        lm_air_block_t block;
        status = lm_air_tx_poll(tx, &block);
        if (status < 0) {
            return status;
        }
        if (emit(user, &block) < 0) {
            tx->phase = LM_AIR_TX_FAILED;
            return LM_AIR_ERR_CALLBACK;
        }
    } while (status == LM_AIR_MORE);
    return LM_AIR_OK;
}

lm_air_status_t lm_air_rx_init(
    lm_air_rx_t *rx,
    lm_air_receive_message_fn receive,
    void *receive_user,
    const lm_air_crypto_hooks_t *crypto) {
    if (rx == NULL || receive == NULL) {
        return LM_AIR_ERR_ARGUMENT;
    }
    memset(rx, 0, sizeof(*rx));
    rx->receive = receive;
    rx->receive_user = receive_user;
    if (crypto != NULL) {
        rx->crypto = *crypto;
    }
    return LM_AIR_OK;
}

void lm_air_rx_reset(lm_air_rx_t *rx) {
    lm_air_receive_message_fn receive;
    void *receive_user;
    lm_air_crypto_hooks_t crypto;
    if (rx == NULL) {
        return;
    }
    receive = rx->receive;
    receive_user = rx->receive_user;
    crypto = rx->crypto;
    memset(rx, 0, sizeof(*rx));
    rx->receive = receive;
    rx->receive_user = receive_user;
    rx->crypto = crypto;
}

static lm_air_replay_entry_t *find_replay(
    lm_air_rx_t *rx,
    uint16_t stream_id) {
    size_t i;
    for (i = 0u; i < LM_AIR_RX_REPLAY_STREAMS; ++i) {
        if (rx->replay[i].used != 0u &&
            rx->replay[i].stream_id == stream_id) {
            return &rx->replay[i];
        }
    }
    return NULL;
}

static int32_t sequence_delta(uint16_t sequence, uint16_t reference) {
    uint16_t difference = (uint16_t)(sequence - reference);
    if (difference == 0u) {
        return 0;
    }
    if (difference < UINT16_C(0x8000)) {
        return difference;
    }
    return (int32_t)difference - INT32_C(65536);
}

static int is_replay(lm_air_rx_t *rx, uint16_t stream_id, uint16_t sequence) {
    lm_air_replay_entry_t *entry = find_replay(rx, stream_id);
    int32_t delta;
    uint16_t age;
    if (entry == NULL) {
        return 0;
    }
    delta = sequence_delta(sequence, entry->highest);
    if (delta > 0) {
        return 0;
    }
    age = (uint16_t)(-delta);
    if (age >= 64u) {
        return 1;
    }
    return (entry->bitmap & (UINT64_C(1) << age)) != 0u;
}

static void commit_replay(
    lm_air_rx_t *rx,
    uint16_t stream_id,
    uint16_t sequence) {
    lm_air_replay_entry_t *entry = find_replay(rx, stream_id);
    size_t i;
    if (entry == NULL) {
        lm_air_replay_entry_t *oldest = &rx->replay[0];
        for (i = 0u; i < LM_AIR_RX_REPLAY_STREAMS; ++i) {
            if (rx->replay[i].used == 0u) {
                oldest = &rx->replay[i];
                break;
            }
            if (rx->replay[i].age < oldest->age) {
                oldest = &rx->replay[i];
            }
        }
        entry = oldest;
        memset(entry, 0, sizeof(*entry));
        entry->used = 1u;
        entry->stream_id = stream_id;
        entry->highest = sequence;
        entry->bitmap = 1u;
        entry->age = ++rx->clock;
        return;
    }
    {
        int32_t delta = sequence_delta(sequence, entry->highest);
        if (delta > 0) {
            entry->bitmap = delta >= 64 ? 1u
                                        : (entry->bitmap << delta) | 1u;
            entry->highest = sequence;
        } else {
            uint16_t age = (uint16_t)(-delta);
            if (age < 64u) {
                entry->bitmap |= UINT64_C(1) << age;
            }
        }
        entry->age = ++rx->clock;
    }
}

static lm_air_reassembly_slot_t *find_or_allocate_slot(
    lm_air_rx_t *rx,
    const lm_air_frame_t *frame) {
    lm_air_reassembly_slot_t *oldest = &rx->slots[0];
    size_t i;
    for (i = 0u; i < LM_AIR_RX_REASSEMBLY_SLOTS; ++i) {
        lm_air_reassembly_slot_t *slot = &rx->slots[i];
        if (slot->used != 0u && slot->stream_id == frame->stream_id &&
            slot->sequence == frame->sequence) {
            return slot;
        }
    }
    for (i = 0u; i < LM_AIR_RX_REASSEMBLY_SLOTS; ++i) {
        if (rx->slots[i].used == 0u) {
            oldest = &rx->slots[i];
            break;
        }
        if (rx->slots[i].age < oldest->age) {
            oldest = &rx->slots[i];
        }
    }
    if (oldest->used != 0u) {
        ++rx->stats.reassembly_evictions;
    }
    memset(oldest, 0, sizeof(*oldest));
    oldest->used = 1u;
    oldest->stream_id = frame->stream_id;
    oldest->sequence = frame->sequence;
    oldest->profile = frame->profile;
    oldest->flags = frame->flags;
    oldest->fragment_count = frame->fragment_count;
    oldest->class_id = frame->class_id;
    oldest->priority = frame->priority;
    oldest->state_tag = frame->state_tag;
    oldest->age = ++rx->clock;
    return oldest;
}

static lm_air_status_t deliver_slot(
    lm_air_rx_t *rx,
    lm_air_reassembly_slot_t *slot,
    uint8_t outer_flags) {
    size_t total = 0u;
    size_t i;
    uint16_t body_len;
    uint8_t signature_len;
    size_t canonical_len;
    size_t expected_len;
    uint8_t *assembled = &slot->fragments[0][0];
    lm_air_message_t message;
    lm_air_status_t callback_status;

    for (i = 0u; i < slot->fragment_count; ++i) {
        size_t len = slot->fragment_len[i];
        if (total + len > LM_AIR_MAX_ASSEMBLED_BYTES) {
            return LM_AIR_ERR_CAPACITY;
        }
        memmove(assembled + total, slot->fragments[i], len);
        total += len;
    }
    if (total < LM_AIR_SEMANTIC_HEADER_BYTES || assembled[0] != 'L' ||
        assembled[1] != 'M' || assembled[2] != 'S' ||
        assembled[3] != '1' || assembled[4] != LM_AIR_VERSION ||
        assembled[47] != 0u || assembled[6] != slot->class_id ||
        assembled[7] != slot->priority ||
        get_u16_be(assembled + 30) != slot->state_tag) {
        return LM_AIR_ERR_FORMAT;
    }
    if ((assembled[5] & (uint8_t)~LM_AIR_SEMANTIC_FLAG_AUTH) != 0u) {
        return LM_AIR_ERR_FORMAT;
    }
    body_len = get_u16_be(assembled + 28);
    signature_len = assembled[46];
    expected_len = LM_AIR_SEMANTIC_HEADER_BYTES + (size_t)body_len +
                   signature_len;
    if (expected_len != total ||
        (signature_len != 0u && signature_len != LM_AIR_SIGNATURE_BYTES)) {
        return LM_AIR_ERR_FORMAT;
    }
    if (((assembled[5] & LM_AIR_SEMANTIC_FLAG_AUTH) != 0u) !=
            (signature_len == LM_AIR_SIGNATURE_BYTES) ||
        ((outer_flags & LM_AIR_FLAG_AUTHENTICATED) != 0u) !=
            (signature_len == LM_AIR_SIGNATURE_BYTES)) {
        return LM_AIR_ERR_AUTH;
    }
    canonical_len = LM_AIR_SEMANTIC_HEADER_BYTES + body_len;
    if (signature_len != 0u) {
        if (rx->crypto.verify == NULL ||
            rx->crypto.verify(rx->crypto.user,
                              assembled,
                              canonical_len,
                              assembled + canonical_len) != 0) {
            ++rx->stats.auth_failures;
            return LM_AIR_ERR_AUTH;
        }
    }

    memset(&message, 0, sizeof(message));
    message.source_id = get_u32_be(assembled + 8);
    message.epoch = get_u32_be(assembled + 12);
    message.message_id = get_u32_be(assembled + 16);
    message.logical_sequence = get_u64_be(assembled + 20);
    message.class_id = assembled[6];
    message.priority = assembled[7];
    memcpy(message.state_hash, assembled + 30, 16u);
    message.body = assembled + LM_AIR_SEMANTIC_HEADER_BYTES;
    message.body_len = body_len;
    message.authenticated = signature_len != 0u;

    commit_replay(rx, slot->stream_id, slot->sequence);
    callback_status = rx->receive(rx->receive_user, &message);
    ++rx->stats.messages_delivered;
    return callback_status < 0 ? LM_AIR_ERR_CALLBACK : LM_AIR_COMPLETE;
}

static lm_air_status_t ingest_frame(
    lm_air_rx_t *rx,
    const lm_air_frame_t *frame) {
    lm_air_reassembly_slot_t *slot;
    uint32_t complete_mask;
    lm_air_status_t status;
    if (is_replay(rx, frame->stream_id, frame->sequence) != 0) {
        ++rx->stats.replay_rejected;
        return LM_AIR_ERR_REPLAY;
    }
    slot = find_or_allocate_slot(rx, frame);
    if (slot->fragment_count != frame->fragment_count ||
        slot->profile != frame->profile || slot->flags != frame->flags ||
        slot->class_id != frame->class_id ||
        slot->priority != frame->priority || slot->state_tag != frame->state_tag) {
        return LM_AIR_ERR_FORMAT;
    }
    if ((slot->received_bitmap & (UINT32_C(1) << frame->fragment_index)) != 0u) {
        ++rx->stats.duplicate_fragments;
        return LM_AIR_ERR_DUPLICATE;
    }
    memcpy(slot->fragments[frame->fragment_index],
           frame->payload,
           frame->payload_len);
    slot->fragment_len[frame->fragment_index] = frame->payload_len;
    slot->received_bitmap |= UINT32_C(1) << frame->fragment_index;
    slot->age = ++rx->clock;
    ++rx->stats.frames_accepted;
    complete_mask = frame->fragment_count == 32u
                        ? UINT32_MAX
                        : (UINT32_C(1) << frame->fragment_count) - 1u;
    if (slot->received_bitmap != complete_mask) {
        return LM_AIR_MORE;
    }
    status = deliver_slot(rx, slot, frame->flags);
    memset(slot, 0, sizeof(*slot));
    return status;
}

lm_air_status_t lm_air_rx_ingest_wire(
    lm_air_rx_t *rx,
    const uint8_t *wire,
    size_t wire_len) {
    lm_air_frame_t frame;
    lm_air_status_t status;
    if (rx == NULL) {
        return LM_AIR_ERR_ARGUMENT;
    }
    status = lm_air_frame_decode(wire, wire_len, &frame);
    if (status != LM_AIR_OK) {
        ++rx->stats.frames_rejected;
        return status;
    }
    status = ingest_frame(rx, &frame);
    if (status < 0) {
        ++rx->stats.frames_rejected;
    }
    return status;
}

lm_air_status_t lm_air_rx_ingest_block(
    lm_air_rx_t *rx,
    const lm_air_block_t *block) {
    const uint8_t *coded;
    const uint8_t *raw;
    size_t coded_bytes;
    lm_air_status_t status;
    lm_air_frame_t frame;
    uint32_t metric;

    if (rx == NULL || block == NULL || block->raw_len < 16u ||
        block->raw_len > LM_AIR_MAX_WIRE_BYTES || block->bit_len == 0u ||
        block->bit_len > LM_AIR_FEC_MAX_CODED_BITS ||
        block->interleave_rows > 64u || block->interleave_rows == 1u ||
        block->fec > 1u) {
        return LM_AIR_ERR_ARGUMENT;
    }
    coded_bytes = ((size_t)block->bit_len + 7u) / 8u;
    if (coded_bytes > sizeof(block->data)) {
        return LM_AIR_ERR_CAPACITY;
    }
    coded = block->data;
    if (block->interleave_rows != 0u) {
        status = lm_air_deinterleave_bits(block->data,
                                          block->bit_len,
                                          rx->scratch_a,
                                          sizeof(rx->scratch_a),
                                          block->interleave_rows);
        if (status != LM_AIR_OK) {
            ++rx->stats.fec_failures;
            return status;
        }
        coded = rx->scratch_a;
    }
    if (block->fec != 0u) {
        status = lm_air_fec_decode_hard(coded,
                                        block->bit_len,
                                        rx->scratch_b,
                                        sizeof(rx->scratch_b),
                                        block->raw_len,
                                        &rx->fec_workspace,
                                        &metric);
        if (status != LM_AIR_OK) {
            ++rx->stats.fec_failures;
            return status;
        }
        raw = rx->scratch_b;
    } else {
        if (block->bit_len != (size_t)block->raw_len * 8u) {
            return LM_AIR_ERR_FORMAT;
        }
        raw = coded;
    }
    status = lm_air_frame_decode(raw, block->raw_len, &frame);
    if (status != LM_AIR_OK) {
        ++rx->stats.frames_rejected;
        return status;
    }
    if (((frame.flags & LM_AIR_FLAG_FEC) != 0u) != (block->fec != 0u)) {
        ++rx->stats.frames_rejected;
        return LM_AIR_ERR_FORMAT;
    }
    status = ingest_frame(rx, &frame);
    if (status < 0) {
        ++rx->stats.frames_rejected;
    }
    return status;
}
