# Staged benchmark contract

The C microbenchmark separates mechanical evidence from system claims. It must
not label a smaller payload as semantic success unless an external task
evaluator proves equivalent downstream accuracy.

## Stage 1 semantic transport

Inputs are a conventional state representation, a semantic delta, identical
receiver prior state, and a deterministic downstream task evaluator. Outputs
are task score, critical fact agreement, encoded air bits, blocks, and airtime.

The pass conditions are:

```text
task_score_delta >= task_score_baseline - allowed_tolerance
critical_fact_agreement >= 0.99
air_bits_baseline / air_bits_delta >= 10.0
```

Critical facts remain deterministic. Every scored delta must carry confidence,
provenance, and the full state hash in its canonical LMAD body. The LMS1
envelope carries the state hash again, and the physical frame carries its first
two bytes as an early mismatch tag. Missing provenance or confidence makes a
trial invalid rather than inaccurate.

`latentmesh-air-bench` currently provides exact framing evidence for an 1800
byte state and a 64 byte delta through the same HF AFSK profile, including the
same LMS1 overhead, CRC32C, FEC, and interleaver. It explicitly reports that
task accuracy is not measured. The size result is therefore transport evidence,
not proof of the tenfold semantic target.

## Stage 2 neural PHY

Inputs are the same accepted semantic deltas, identical bandwidth and power or
energy budgets, held out degraded channel traces, and a conventional DSP
baseline. Outputs are accepted task useful bits, airtime, joules, critical fact
agreement, false acceptance rate, and fallback rate.

Two normalized metrics are required:

```text
useful_information_per_second = accepted_task_useful_bits / airtime_seconds
useful_information_per_joule = accepted_task_useful_bits / measured_joules
```

The pass condition is at least `2.0` times the conventional receiver on one of
those metrics without reducing critical fact agreement below 0.99. Training
traces, pilot bits, and adaptation intervals cannot overlap held out scoring
traces. A neural prediction below its configured confidence threshold must use
the DSP LLR and must be counted as a fallback.

The current modem microbenchmarks are noiseless compute throughput tests. They
do not establish degraded channel gain or energy efficiency, so the program
prints the twofold goal as an unmeasured target. Hardware in the loop or a
versioned channel corpus is required before that claim can pass.
