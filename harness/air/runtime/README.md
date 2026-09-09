# AIR runtime qualification: software slice

Fast radio work stays local; slow observers cannot hold up packet decoding.
A small offline model corrects site-specific prediction errors. This directory
provides runnable calibration and qualification tools, not an implemented 5G
neural receiver. See [ADR-048](../../../docs/adr/048-air-runtime-and-site-calibration.md).

## Build and test

From repository root (C11 compiler; Python 3 with NumPy 2.3.5 tested):

```sh
gcc -std=c11 -Wall -Wextra -Werror -pedantic -I c/include c/src/*.c c/tests/test_runtime.c -lm -o /tmp/lm-runtime-test
/tmp/lm-runtime-test
gcc -std=c11 -O2 -I c/include c/src/*.c c/bench/runtime.c -lm -o /tmp/lm-runtime-bench
/tmp/lm-runtime-bench
python3 -m unittest discover -s harness/air/runtime -v
```

CMake also registers the runtime test and UNIX benchmark. ESP-IDF does not
compile this host-only runtime. Include `latentmesh_air/runtime.h` explicitly.
Initialize with a monotonic nanosecond clock. Configure only while quiescent.
Use `lm_rt_ingest` in place of the existing block-ingest call and drain its
observer queue from a separate consumer. `lm_rt_decide` is an independently
tested policy helper, not yet wired to a scheduler. Do not claim CUDA support.

## Calibrate and apply

Input is a JSON array of paths; at least three whole-TX groups and six records:

```json
[{"site":"lab","tx":"tx1","rt_dbm":-80,"measured_dbm":-61,
  "distance_m":12,"bounces":2,"frequency_ghz":6.75}]
```

```sh
python3 harness/air/runtime/calibrate.py paths.json model.json
python3 harness/air/runtime/apply_calibration.py model.json paths.json corrected.json
python3 harness/air/runtime/qualify.py measured-receipt.json qualification.json
```

The single-row example only documents format; it is not a valid training set.
Apply does not require `measured_dbm`. Production qualification always remains
false. A passing calibration gate means 30% and 1 dB held-out RMSE improvement,
not BLER improvement. Inspect the constant-offset baseline as well.

Qualification receipt schema is documented in `qualify.py`. Provide matching
trial IDs for conventional and split arms. Record measured positive joules,
seconds and bandwidth; do not substitute zero or estimated latency for samples.
Utility is task-specific and must use the same definition in both arms. Metrics
are reported separately per joule, second and Hz-second; their arbitrary product
is not called spectral efficiency. Campaign gates require 10,000 paired trials,
100/150 us P99.9 deadlines, fallback below 1e-4, nonregressing BLER and 15% utility
improvement on all three denominators. The tool reports a stopping condition;
it does not control hardware. Input hashes bind files, not their truthfulness.

## Evidence and limitations

Portable C integration and existing regression tests pass. Python tests exercise
synthetic held-out fitting, artifact application, bad inputs, receipt gates and
stopping rules. ASan/UBSan passed with LeakSanitizer disabled because the runner
cannot inspect `/proc` under its tracing environment. No dependency advisory
scan, ThreadSanitizer, CMake run, Rust build, CUDA build, PUSCH replay, power
measurement, two-site dataset evaluation or OTA test has passed here.

No neural-channel improvement, real-time guarantee, production readiness or
independent reproduction is claimed. Keep M3/M5X nulls unchanged.
