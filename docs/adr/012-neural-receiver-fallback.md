# 012. Neural receiver with deterministic fallback

* Status: Accepted and minimally implemented
* Date: 2026-08-23
* Related: [010](010-latentmesh-air-protocol.md), [014](014-benchmark-and-acceptance-method.md)

## Context

A fully learned receiver can improve a known channel distribution and fail
silently outside it. HF makes that risk acute because ionospheric paths,
selective fading, Doppler, interference, and impulsive noise produce conditions
that are difficult to bound in training. Replacing synchronization, demodulation,
and FEC at once would remove the strongest diagnostic and fallback path.

The neural receiver must therefore begin as an assist to conventional DSP, not
as the sole source of decoded facts.

## Decision

Keep conventional synchronization, channel measurement, modulation envelope,
and FEC. Place a bounded learned corrector on soft bit likelihoods. The FEC
decoder receives either corrected likelihoods or the untouched DSP likelihoods.
It never receives an unbounded model action.

The portable C implementation provides a five tap normalized LMS likelihood
corrector with one bias, bounded inputs, bounded weights, bounded output, an
exponential calibration error estimate, a minimum confidence threshold, and an
explicit per bit DSP fallback counter. It contains no heap allocation and adapts
only six scalar parameters.

```text
IQ or PCM
  -> deterministic synchronization and demodulation
  -> DSP likelihoods
  -> bounded learned likelihood corrector
       -> sufficient calibrated confidence: corrected likelihood
       -> otherwise: original DSP likelihood
  -> deterministic FEC decoder
  -> CRC, replay, signature, and semantic state checks
```

This small implementation is an integration scaffold and an independently
testable safety pattern. It is not a universal neural receiver and is not
described as one.

## Promotion stages

| Stage | Learned output affects delivery | Required evidence |
|---|---:|---|
| Shadow | No | Logs predictions, calibration, latency, and disagreement on recorded channels |
| Assist | Only above a frozen threshold, with per bit fallback | Lower post FEC packet error on validation channels with no regression bound exceeded |
| Primary with fallback | Yes, but deterministic path remains available per frame | Hardware in loop evidence across unseen channel families and a tested circuit breaker |
| End to end experimental | Separate research profile only | Spectral compliance, hardware evidence, and task utility acceptance under ADR 014 |

ESP32 firmware remains at the integration stage. The C likelihood corrector has
host simulation coverage. No trained model, live IQ path, inference timing,
power measurement, or on device accuracy evidence exists yet.

## Fallback rules

Use the DSP likelihood for a bit or whole frame when any of these is true:

1. Input or output is not finite.
2. Calibrated confidence is below threshold.
3. Recent corrected packet error exceeds the deterministic path by the allowed
   regression budget.
4. CRC or semantic state disagreement rises above its control limit.
5. Channel features are outside the validated envelope.
6. Inference misses its deadline or exceeds its energy budget.
7. Adaptation parameters hit a bound repeatedly.
8. A signed model or configuration identity does not match the approved one.

The first implementation covers finite values, confidence, bounded parameters,
and per bit fallback. Distribution shift, deadline, energy, signed model, and
system level regression circuit breakers remain integration work.

## Online adaptation

Adapt only on trusted labels derived from known pilots, verified frames, or an
explicit training sequence. Never adapt to an unauthenticated decoded payload.
Update a small parameter set at radio timescales. Store the preadaptation state,
learning window, channel summary, error change, and rollback decision. MetaHarness
may propose thresholds or bounded adaptation parameters, but deployment requires
the same held out and safety gates as a code change.

An improvement in average bit error is insufficient if it concentrates errors
in critical fields. Report packet error, critical state agreement, calibration,
fallback rate, latency, energy, and task utility.

## Consequences

The deterministic receiver remains a standards compatible reference and a safe
degradation mode. The learned component can demonstrate value early without
forcing a new waveform. The cost is that some joint optimization gain is left
on the table.

The main failure mode is selection bias: the confidence score can be high on a
channel family missing from training. The first fix path is a shadow corpus that
contains radios, bands, locations, times, weather, interference types, and
negative cases not used for adaptation, followed by conformal or similarly
bounded coverage calibration before promotion.
