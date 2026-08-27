#include "latentmesh_air/frame.h"

#include <string.h>

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

uint32_t lm_air_crc32c(const uint8_t *data, size_t len) {
    static const uint32_t nibble_table[16] = {
        UINT32_C(0x00000000), UINT32_C(0x105ec76f),
        UINT32_C(0x20bd8ede), UINT32_C(0x30e349b1),
        UINT32_C(0x417b1dbc), UINT32_C(0x5125dad3),
        UINT32_C(0x61c69362), UINT32_C(0x7198540d),
        UINT32_C(0x82f63b78), UINT32_C(0x92a8fc17),
        UINT32_C(0xa24bb5a6), UINT32_C(0xb21572c9),
        UINT32_C(0xc38d26c4), UINT32_C(0xd3d3e1ab),
        UINT32_C(0xe330a81a), UINT32_C(0xf36e6f75)};
    uint32_t crc = UINT32_C(0xffffffff);
    size_t i;

    if (data == NULL && len != 0u) {
        return 0u;
    }
    for (i = 0u; i < len; ++i) {
        crc ^= data[i];
        crc = (crc >> 4) ^ nibble_table[crc & 0x0fu];
        crc = (crc >> 4) ^ nibble_table[crc & 0x0fu];
    }
    return ~crc;
}

static lm_air_status_t validate_frame(const lm_air_frame_t *frame) {
    if (frame == NULL) {
        return LM_AIR_ERR_ARGUMENT;
    }
    if (frame->profile > LM_AIR_PROFILE_MESHTASTIC || frame->flags > 0x0fu ||
        frame->class_id > LM_AIR_CLASS_DIAGNOSTIC ||
        frame->priority > 0x0fu) {
        return LM_AIR_ERR_FORMAT;
    }
    if (frame->fragment_count == 0u ||
        frame->fragment_count > LM_AIR_MAX_FRAGMENTS ||
        frame->fragment_index >= frame->fragment_count) {
        return LM_AIR_ERR_FORMAT;
    }
    return LM_AIR_OK;
}

size_t lm_air_frame_wire_size(const lm_air_frame_t *frame) {
    if (validate_frame(frame) != LM_AIR_OK) {
        return 0u;
    }
    return LM_AIR_FRAME_HEADER_BYTES + (size_t)frame->payload_len +
           LM_AIR_FRAME_CRC_BYTES;
}

lm_air_status_t lm_air_frame_encode(
    const lm_air_frame_t *frame,
    uint8_t *wire,
    size_t wire_capacity,
    size_t *wire_len) {
    size_t required;
    uint32_t crc;
    lm_air_status_t status = validate_frame(frame);

    if (status != LM_AIR_OK || wire == NULL || wire_len == NULL) {
        return status != LM_AIR_OK ? status : LM_AIR_ERR_ARGUMENT;
    }
    required = lm_air_frame_wire_size(frame);
    if (required > wire_capacity) {
        return LM_AIR_ERR_CAPACITY;
    }

    wire[0] = (uint8_t)(0xa0u | LM_AIR_VERSION);
    wire[1] = (uint8_t)((frame->flags << 4) | frame->profile);
    put_u16_be(wire + 2, frame->stream_id);
    put_u16_be(wire + 4, frame->sequence);
    wire[6] = frame->fragment_index;
    wire[7] = frame->fragment_count;
    wire[8] = (uint8_t)((frame->class_id << 4) | frame->priority);
    wire[9] = frame->payload_len;
    put_u16_be(wire + 10, frame->state_tag);
    if (frame->payload_len != 0u) {
        memcpy(wire + LM_AIR_FRAME_HEADER_BYTES,
               frame->payload,
               frame->payload_len);
    }
    crc = lm_air_crc32c(wire, required - LM_AIR_FRAME_CRC_BYTES);
    put_u32_be(wire + required - LM_AIR_FRAME_CRC_BYTES, crc);
    *wire_len = required;
    return LM_AIR_OK;
}

lm_air_status_t lm_air_frame_decode(
    const uint8_t *wire,
    size_t wire_len,
    lm_air_frame_t *frame) {
    size_t expected;
    uint32_t expected_crc;
    uint32_t actual_crc;

    if (wire == NULL || frame == NULL) {
        return LM_AIR_ERR_ARGUMENT;
    }
    if (wire_len < LM_AIR_FRAME_HEADER_BYTES + LM_AIR_FRAME_CRC_BYTES ||
        wire_len > LM_AIR_MAX_WIRE_BYTES) {
        return LM_AIR_ERR_FORMAT;
    }
    if (wire[0] != (uint8_t)(0xa0u | LM_AIR_VERSION)) {
        return (wire[0] & 0xf0u) == 0xa0u ? LM_AIR_ERR_VERSION
                                         : LM_AIR_ERR_FORMAT;
    }
    expected = LM_AIR_FRAME_HEADER_BYTES + (size_t)wire[9] +
               LM_AIR_FRAME_CRC_BYTES;
    if (wire_len != expected) {
        return LM_AIR_ERR_FORMAT;
    }
    expected_crc = get_u32_be(wire + wire_len - LM_AIR_FRAME_CRC_BYTES);
    actual_crc = lm_air_crc32c(wire, wire_len - LM_AIR_FRAME_CRC_BYTES);
    if (expected_crc != actual_crc) {
        return LM_AIR_ERR_CRC;
    }

    memset(frame, 0, sizeof(*frame));
    frame->profile = (uint8_t)(wire[1] & 0x0fu);
    frame->flags = (uint8_t)(wire[1] >> 4);
    frame->stream_id = get_u16_be(wire + 2);
    frame->sequence = get_u16_be(wire + 4);
    frame->fragment_index = wire[6];
    frame->fragment_count = wire[7];
    frame->class_id = (uint8_t)(wire[8] >> 4);
    frame->priority = (uint8_t)(wire[8] & 0x0fu);
    frame->payload_len = wire[9];
    frame->state_tag = get_u16_be(wire + 10);
    if (validate_frame(frame) != LM_AIR_OK) {
        return LM_AIR_ERR_FORMAT;
    }
    if (frame->payload_len != 0u) {
        memcpy(frame->payload,
               wire + LM_AIR_FRAME_HEADER_BYTES,
               frame->payload_len);
    }
    return LM_AIR_OK;
}
