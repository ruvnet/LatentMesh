#include "lm_air_kiss.h"

#include <string.h>

void lm_kiss_decoder_reset(lm_kiss_decoder_t *decoder)
{
    if (decoder != NULL) {
        memset(decoder, 0, sizeof(*decoder));
    }
}

static int append_escaped(uint8_t byte, uint8_t *out, size_t capacity, size_t *at)
{
    if (byte == LM_KISS_FEND || byte == LM_KISS_FESC) {
        if (*at + 2u > capacity) {
            return -1;
        }
        out[(*at)++] = LM_KISS_FESC;
        out[(*at)++] = byte == LM_KISS_FEND ? LM_KISS_TFEND : LM_KISS_TFESC;
    } else {
        if (*at + 1u > capacity) {
            return -1;
        }
        out[(*at)++] = byte;
    }
    return 0;
}

int lm_kiss_encode(uint8_t port,
                   const uint8_t *payload,
                   size_t payload_len,
                   uint8_t *out,
                   size_t out_capacity,
                   size_t *out_len)
{
    if (port > 15 || payload == NULL || payload_len == 0 || out == NULL ||
        out_len == NULL || out_capacity < 3) {
        return -1;
    }
    size_t at = 0;
    out[at++] = LM_KISS_FEND;
    if (append_escaped((uint8_t)(port << 4), out, out_capacity, &at) != 0) {
        return -1;
    }
    for (size_t i = 0; i < payload_len; ++i) {
        if (append_escaped(payload[i], out, out_capacity, &at) != 0) {
            return -1;
        }
    }
    if (at >= out_capacity) {
        return -1;
    }
    out[at++] = LM_KISS_FEND;
    *out_len = at;
    return 0;
}

static int finish(lm_kiss_decoder_t *decoder, uint8_t *port, lm_air_packet_t *out)
{
    if (decoder->overflow || !decoder->command_seen || decoder->len == 0) {
        const int result = decoder->overflow ? -1 : 0;
        lm_kiss_decoder_reset(decoder);
        decoder->in_frame = true;
        return result;
    }
    if ((decoder->command & 0x0fu) != LM_KISS_CMD_DATA) {
        lm_kiss_decoder_reset(decoder);
        decoder->in_frame = true;
        return 0;
    }
    if (out != NULL) {
        out->len = (uint16_t)decoder->len;
        out->source_link = 0xffu;
        out->flags = LM_AIR_PAYLOAD_PUBLIC_CODEC;
        memcpy(out->data, decoder->data, decoder->len);
    }
    if (port != NULL) {
        *port = decoder->command >> 4;
    }
    lm_kiss_decoder_reset(decoder);
    decoder->in_frame = true;
    return 1;
}

int lm_kiss_decoder_feed(lm_kiss_decoder_t *decoder,
                         uint8_t byte,
                         uint8_t *port,
                         lm_air_packet_t *out)
{
    if (decoder == NULL) {
        return -1;
    }
    if (byte == LM_KISS_FEND) {
        if (!decoder->in_frame) {
            decoder->in_frame = true;
            decoder->len = 0;
            return 0;
        }
        return finish(decoder, port, out);
    }
    if (!decoder->in_frame || decoder->overflow) {
        return 0;
    }
    if (decoder->escaped) {
        decoder->escaped = false;
        if (byte == LM_KISS_TFEND) {
            byte = LM_KISS_FEND;
        } else if (byte == LM_KISS_TFESC) {
            byte = LM_KISS_FESC;
        } else {
            decoder->overflow = true;
            return -1;
        }
    } else if (byte == LM_KISS_FESC) {
        decoder->escaped = true;
        return 0;
    }
    if (!decoder->command_seen) {
        decoder->command = byte;
        decoder->command_seen = true;
        return 0;
    }
    if (decoder->len >= LM_AIR_MAX_FRAME_SIZE) {
        decoder->overflow = true;
        return -1;
    }
    decoder->data[decoder->len++] = byte;
    return 0;
}
