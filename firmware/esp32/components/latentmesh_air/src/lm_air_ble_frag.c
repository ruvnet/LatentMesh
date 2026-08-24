#include "lm_air_ble_frag.h"

#include <string.h>
#include "lm_air_crc.h"

static uint16_t read_le16(const uint8_t *p)
{
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static void write_le16(uint8_t *p, uint16_t value)
{
    p[0] = (uint8_t)value;
    p[1] = (uint8_t)(value >> 8);
}

size_t lm_air_ble_fragment_count(size_t message_len, size_t att_payload_capacity)
{
    if (message_len == 0 || att_payload_capacity <= LM_AIR_BLE_FRAG_HEADER_SIZE) {
        return 0;
    }
    const size_t capacity = att_payload_capacity - LM_AIR_BLE_FRAG_HEADER_SIZE;
    const size_t count = (message_len + capacity - 1u) / capacity;
    return count <= LM_AIR_BLE_MAX_FRAGMENTS ? count : 0;
}

int lm_air_ble_make_fragment(const lm_air_packet_t *packet,
                             uint16_t message_id,
                             uint8_t fragment_index,
                             size_t att_payload_capacity,
                             uint8_t *out,
                             size_t out_capacity,
                             size_t *out_len)
{
    if (packet == NULL || out == NULL || out_len == NULL || packet->len == 0 ||
        packet->len > LM_AIR_MAX_FRAME_SIZE || out_capacity < att_payload_capacity) {
        return -1;
    }
    const size_t count = lm_air_ble_fragment_count(packet->len, att_payload_capacity);
    if (count == 0 || fragment_index >= count) {
        return -1;
    }
    /* Equal-sized logical chunks make offsets derivable and allow fragments
     * to arrive out of order even if peers negotiated different MTUs. */
    const size_t unit = (packet->len + count - 1u) / count;
    const size_t offset = (size_t)fragment_index * unit;
    const size_t remaining = packet->len - offset;
    const size_t payload_len = remaining < unit ? remaining : unit;
    if (LM_AIR_BLE_FRAG_HEADER_SIZE + payload_len > att_payload_capacity) {
        return -1;
    }
    out[0] = LM_AIR_BLE_FRAG_MAGIC;
    out[1] = LM_AIR_BLE_FRAG_VERSION;
    write_le16(&out[2], message_id);
    out[4] = fragment_index;
    out[5] = (uint8_t)count;
    write_le16(&out[6], packet->len);
    write_le16(&out[8], lm_air_crc16_ccitt(packet->data, packet->len));
    memcpy(&out[LM_AIR_BLE_FRAG_HEADER_SIZE], &packet->data[offset], payload_len);
    *out_len = LM_AIR_BLE_FRAG_HEADER_SIZE + payload_len;
    return 0;
}

void lm_air_ble_reassembly_reset(lm_air_ble_reassembly_t *state)
{
    if (state != NULL) {
        memset(state, 0, sizeof(*state));
    }
}

lm_air_ble_result_t lm_air_ble_reassembly_ingest(lm_air_ble_reassembly_t *state,
                                                  const uint8_t *fragment,
                                                  size_t fragment_len,
                                                  lm_air_packet_t *complete)
{
    if (state == NULL || fragment == NULL ||
        fragment_len <= LM_AIR_BLE_FRAG_HEADER_SIZE ||
        fragment[0] != LM_AIR_BLE_FRAG_MAGIC ||
        fragment[1] != LM_AIR_BLE_FRAG_VERSION) {
        return LM_AIR_BLE_REJECTED;
    }
    const uint16_t message_id = read_le16(&fragment[2]);
    const uint8_t index = fragment[4];
    const uint8_t count = fragment[5];
    const uint16_t total = read_le16(&fragment[6]);
    const uint16_t crc = read_le16(&fragment[8]);
    if (count == 0 || count > LM_AIR_BLE_MAX_FRAGMENTS || index >= count ||
        total == 0 || total > LM_AIR_MAX_FRAME_SIZE) {
        lm_air_ble_reassembly_reset(state);
        return LM_AIR_BLE_REJECTED;
    }
    if (!state->active || state->message_id != message_id) {
        lm_air_ble_reassembly_reset(state);
        state->active = true;
        state->message_id = message_id;
        state->total_len = total;
        state->expected_crc = crc;
        state->fragment_count = count;
    }
    if (state->total_len != total || state->expected_crc != crc ||
        state->fragment_count != count) {
        lm_air_ble_reassembly_reset(state);
        return LM_AIR_BLE_REJECTED;
    }
    const size_t unit = (total + count - 1u) / count;
    const size_t offset = (size_t)index * unit;
    const size_t payload_len = fragment_len - LM_AIR_BLE_FRAG_HEADER_SIZE;
    const size_t expected_len = index + 1u == count ? total - offset : unit;
    if (offset >= total || payload_len != expected_len || offset + payload_len > total) {
        lm_air_ble_reassembly_reset(state);
        return LM_AIR_BLE_REJECTED;
    }
    memcpy(&state->packet.data[offset],
           &fragment[LM_AIR_BLE_FRAG_HEADER_SIZE], payload_len);
    state->received_bitmap |= UINT64_C(1) << index;
    const uint64_t wanted = count == 64 ? UINT64_MAX : ((UINT64_C(1) << count) - 1u);
    if (state->received_bitmap != wanted) {
        return LM_AIR_BLE_MORE;
    }
    if (lm_air_crc16_ccitt(state->packet.data, total) != state->expected_crc) {
        lm_air_ble_reassembly_reset(state);
        return LM_AIR_BLE_REJECTED;
    }
    state->packet.len = total;
    state->packet.source_link = 0xffu;
    state->packet.flags = state->flags;
    if (complete != NULL) {
        *complete = state->packet;
    }
    lm_air_ble_reassembly_reset(state);
    return LM_AIR_BLE_COMPLETE;
}
