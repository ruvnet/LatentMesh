#include "latentmesh_air/modem.h"

#include <math.h>
#include <stdint.h>

#define LM_AIR_TWO_PI 6.28318530717958647692f

static unsigned packed_bit(const uint8_t *data, size_t bit) {
    return ((unsigned)data[bit >> 3] >> (7u - (unsigned)(bit & 7u))) & 1u;
}

static unsigned samples_per_symbol(const lm_air_fsk_config_t *config) {
    float exact;
    unsigned rounded;
    if (config == NULL || !isfinite(config->sample_rate) ||
        !isfinite(config->symbol_rate) || config->sample_rate <= 0.0f ||
        config->symbol_rate <= 0.0f) {
        return 0u;
    }
    exact = config->sample_rate / config->symbol_rate;
    if (exact < 4.0f || exact > 4096.0f) {
        return 0u;
    }
    rounded = (unsigned)(exact + 0.5f);
    return fabsf(exact - (float)rounded) <= 0.0001f ? rounded : 0u;
}

static int valid_fsk(const lm_air_fsk_config_t *config) {
    if (samples_per_symbol(config) == 0u || !isfinite(config->mark_hz) ||
        !isfinite(config->space_hz) || !isfinite(config->amplitude)) {
        return 0;
    }
    if (config->mark_hz <= 0.0f || config->space_hz <= 0.0f ||
        config->mark_hz >= config->sample_rate * 0.5f ||
        config->space_hz >= config->sample_rate * 0.5f ||
        config->amplitude <= 0.0f || config->amplitude > 1.0f) {
        return 0;
    }
    return 1;
}

lm_air_fsk_config_t lm_air_afsk_bell202_config(float sample_rate) {
    lm_air_fsk_config_t config;
    config.sample_rate = sample_rate;
    config.symbol_rate = 1200.0f;
    config.mark_hz = 1200.0f;
    config.space_hz = 2200.0f;
    config.amplitude = 0.8f;
    return config;
}

lm_air_fsk_config_t lm_air_cpfsk_config(
    float sample_rate,
    float symbol_rate,
    float center_hz,
    float deviation_hz,
    float amplitude) {
    lm_air_fsk_config_t config;
    config.sample_rate = sample_rate;
    config.symbol_rate = symbol_rate;
    config.mark_hz = center_hz + deviation_hz;
    config.space_hz = center_hz - deviation_hz;
    config.amplitude = amplitude;
    return config;
}

lm_air_status_t lm_air_fsk_modulator_init(
    lm_air_fsk_modulator_t *modulator,
    const lm_air_fsk_config_t *config) {
    if (modulator == NULL || config == NULL) {
        return LM_AIR_ERR_ARGUMENT;
    }
    if (valid_fsk(config) == 0) {
        return LM_AIR_ERR_FORMAT;
    }
    modulator->config = *config;
    modulator->phase = 0.0f;
    return LM_AIR_OK;
}

size_t lm_air_fsk_samples_for_bits(
    const lm_air_fsk_config_t *config,
    size_t bit_count) {
    unsigned sps = samples_per_symbol(config);
    if (sps == 0u || bit_count > SIZE_MAX / sps) {
        return 0u;
    }
    return bit_count * sps;
}

lm_air_status_t lm_air_fsk_modulate(
    lm_air_fsk_modulator_t *modulator,
    const uint8_t *packed_bits,
    size_t bit_count,
    float *pcm,
    size_t pcm_capacity,
    size_t *pcm_samples) {
    size_t required;
    size_t bit;
    size_t out = 0u;
    unsigned sps;

    if (modulator == NULL || packed_bits == NULL || pcm == NULL ||
        pcm_samples == NULL) {
        return LM_AIR_ERR_ARGUMENT;
    }
    if (valid_fsk(&modulator->config) == 0) {
        return LM_AIR_ERR_FORMAT;
    }
    sps = samples_per_symbol(&modulator->config);
    required = lm_air_fsk_samples_for_bits(&modulator->config, bit_count);
    if (required > pcm_capacity) {
        return LM_AIR_ERR_CAPACITY;
    }
    for (bit = 0u; bit < bit_count; ++bit) {
        float frequency = packed_bit(packed_bits, bit) != 0u
                              ? modulator->config.mark_hz
                              : modulator->config.space_hz;
        float increment = LM_AIR_TWO_PI * frequency /
                          modulator->config.sample_rate;
        unsigned sample;
        for (sample = 0u; sample < sps; ++sample) {
            pcm[out++] = modulator->config.amplitude * sinf(modulator->phase);
            modulator->phase += increment;
            if (modulator->phase >= LM_AIR_TWO_PI) {
                modulator->phase -= LM_AIR_TWO_PI;
            }
        }
    }
    *pcm_samples = out;
    return LM_AIR_OK;
}

lm_air_status_t lm_air_fsk_demodulate(
    const lm_air_fsk_config_t *config,
    const float *pcm,
    size_t pcm_samples,
    float *llr,
    size_t llr_capacity,
    size_t *llr_count) {
    unsigned sps;
    size_t symbols;
    size_t symbol;

    if (config == NULL || pcm == NULL || llr == NULL || llr_count == NULL) {
        return LM_AIR_ERR_ARGUMENT;
    }
    if (valid_fsk(config) == 0) {
        return LM_AIR_ERR_FORMAT;
    }
    sps = samples_per_symbol(config);
    if (pcm_samples % sps != 0u) {
        return LM_AIR_ERR_FORMAT;
    }
    symbols = pcm_samples / sps;
    if (symbols > llr_capacity) {
        return LM_AIR_ERR_CAPACITY;
    }
    for (symbol = 0u; symbol < symbols; ++symbol) {
        float mark_i = 0.0f;
        float mark_q = 0.0f;
        float space_i = 0.0f;
        float space_q = 0.0f;
        unsigned sample;
        for (sample = 0u; sample < sps; ++sample) {
            float value = pcm[symbol * sps + sample];
            float t = (float)sample / config->sample_rate;
            float mark_phase = LM_AIR_TWO_PI * config->mark_hz * t;
            float space_phase = LM_AIR_TWO_PI * config->space_hz * t;
            mark_i += value * cosf(mark_phase);
            mark_q += value * sinf(mark_phase);
            space_i += value * cosf(space_phase);
            space_q += value * sinf(space_phase);
        }
        {
            float mark_energy = mark_i * mark_i + mark_q * mark_q;
            float space_energy = space_i * space_i + space_q * space_q;
            float total = mark_energy + space_energy + 1.0e-12f;
            llr[symbol] = 8.0f * (mark_energy - space_energy) / total;
        }
    }
    *llr_count = symbols;
    return LM_AIR_OK;
}

lm_air_status_t lm_air_bpsk_modulate_iq(
    const uint8_t *packed_bits,
    size_t bit_count,
    unsigned samples_per_symbol_value,
    float amplitude,
    lm_air_iq_f32_t *iq,
    size_t iq_capacity,
    size_t *iq_count) {
    size_t required;
    size_t bit;
    size_t out = 0u;
    if (packed_bits == NULL || iq == NULL || iq_count == NULL ||
        samples_per_symbol_value == 0u || !isfinite(amplitude) ||
        amplitude <= 0.0f || amplitude > 1.0f) {
        return LM_AIR_ERR_ARGUMENT;
    }
    if (bit_count > SIZE_MAX / samples_per_symbol_value) {
        return LM_AIR_ERR_CAPACITY;
    }
    required = bit_count * samples_per_symbol_value;
    if (required > iq_capacity) {
        return LM_AIR_ERR_CAPACITY;
    }
    for (bit = 0u; bit < bit_count; ++bit) {
        float value = packed_bit(packed_bits, bit) != 0u ? amplitude : -amplitude;
        unsigned sample;
        for (sample = 0u; sample < samples_per_symbol_value; ++sample) {
            iq[out].i = value;
            iq[out].q = 0.0f;
            ++out;
        }
    }
    *iq_count = out;
    return LM_AIR_OK;
}

lm_air_status_t lm_air_bpsk_demodulate_iq(
    const lm_air_iq_f32_t *iq,
    size_t iq_count,
    unsigned samples_per_symbol_value,
    float noise_variance,
    float *llr,
    size_t llr_capacity,
    size_t *llr_count) {
    size_t symbols;
    size_t symbol;
    if (iq == NULL || llr == NULL || llr_count == NULL ||
        samples_per_symbol_value == 0u || !isfinite(noise_variance) ||
        noise_variance <= 0.0f || iq_count % samples_per_symbol_value != 0u) {
        return LM_AIR_ERR_ARGUMENT;
    }
    symbols = iq_count / samples_per_symbol_value;
    if (symbols > llr_capacity) {
        return LM_AIR_ERR_CAPACITY;
    }
    for (symbol = 0u; symbol < symbols; ++symbol) {
        float sum = 0.0f;
        unsigned sample;
        for (sample = 0u; sample < samples_per_symbol_value; ++sample) {
            sum += iq[symbol * samples_per_symbol_value + sample].i;
        }
        llr[symbol] = 2.0f * sum / noise_variance;
        if (llr[symbol] > 32.0f) {
            llr[symbol] = 32.0f;
        } else if (llr[symbol] < -32.0f) {
            llr[symbol] = -32.0f;
        }
    }
    *llr_count = symbols;
    return LM_AIR_OK;
}
