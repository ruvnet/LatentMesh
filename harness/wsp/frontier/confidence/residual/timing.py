"""Measure warmed native and optional program inference on identical pretokenized inputs."""
import argparse
import tempfile
import time

import onnxruntime as ort

from frozen import HERE, LOCK_SHA, locked
from test1 import Encoder, Path, ensure, examples, feed, incumbent, install_program, json_write, np, readout, session_options
from stream import OUTPUTS


def measure(cache, out):
    lock = locked()
    rows, _ = examples(cache)
    opts = session_options()
    opts.log_severity_level = 3
    target = Encoder(cache, "target")
    sessions = {"native":target.original}
    projection, offset = incumbent("local")[:2]
    runs = tuple(("native","transfer","local"))
    with tempfile.TemporaryDirectory() as tmp:
        for arm in runs[1:]:
            method = lock["methods"][arm]
            model = Path(tmp)/(arm+".onnx")
            install_program(ensure(cache,"target-model.onnx"), model, arm,
                            (projection, offset, *readout(method), method["points"]), LOCK_SHA)
            sessions[arm] = ort.InferenceSession(str(model), opts, providers=["CPUExecutionProvider"])
    samples = []
    for prev, cur, *_ in rows[:128]:
        previous = sessions["transfer"].run(["ctx_next"], feed(target.tokenizer, prev))[0]
        native = feed(target.tokenizer, cur)
        samples.append((native, {**native, "ctx_previous":previous}))
    def infer(name, sample):
        is_base = name == "native"
        sessions[name].run(["logits"] if is_base else OUTPUTS, sample[0 if is_base else 1])
    for sample in samples[:8]:
        for name in runs:
            infer(name, sample)
    times = {name:[] for name in runs}
    for i, sample in enumerate(samples):
        shift = i % len(runs)
        for name in runs[shift:]+runs[:shift]:
            start = time.perf_counter_ns()
            infer(name, sample)
            times[name].append((time.perf_counter_ns() - start)/1e6)
    result = {"sample":len(samples), "warmup":8, "threads":2, "unit":"ms",
              "scope":"pretokenized final turn and ready previous state, rotated order",
              "latency":{k:{"p50":round(float(np.median(v)),4),"p95":round(float(np.percentile(v,95)),4)}
                         for k,v in times.items()}}
    json_write(out, result)
    print(result["latency"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    measure(args.cache, args.out)
