"""Check the native logits, added emotion outputs and CPU cost through ONNX Runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from install import coefficients
from data import ensure, session_options


def validate(cache, out, path, regime, mode, arm):
    original = ort.InferenceSession(str(ensure(cache, "target-model.onnx")), session_options(), providers=["CPUExecutionProvider"])
    installed = ort.InferenceSession(str(path), session_options(), providers=["CPUExecutionProvider"])
    tokenizer = Tokenizer.from_file(str(ensure(cache, "target-tokenizer.json")))
    tokenizer.no_padding()
    tokenizer.enable_truncation(max_length=96)
    sample = json.loads((out / "parity-input.json").read_bytes())
    raw_sets = json.loads((out / "test.json").read_bytes())
    col = raw_sets["arms"].index(arm)
    sources = [v for subset in raw_sets["sets"].values() for v in subset["scores"][:32]]
    if len(sources) != len(sample):
        raise ValueError("parity sample and held out features differ")
    for row, scores in zip(sample, sources, strict=True):
        if row["score"] != scores[0]:
            raise ValueError("parity samples are reordered")
        row["score"] = scores[col]
    w = coefficients(regime, mode, arm)
    checkpoint = hashlib.sha256(path.read_bytes()).hexdigest()
    base_times, patch_times = [], []
    native_drift, error, tag_errors, route_errors, active = 0., 0., 0, 0, 0
    for i, row in enumerate(sample[:8] + sample):
        b = tokenizer.encode(row["text"])
        inputs = {"input_ids": np.asarray([b.ids], dtype=np.int64), "attention_mask": np.asarray([b.attention_mask], dtype=np.int64)}
        values = {}
        for key in (("base", "patch") if i % 2 == 0 else ("patch", "base")):
            t = time.perf_counter_ns()
            if key == "base":
                values[key] = original.run(["logits"], inputs)
            else:
                values[key] = installed.run(["logits", "wspc_evidence", "wspc_tag", "wspc_route"], inputs)
            elapsed = (time.perf_counter_ns() - t) / 1e6
            if i >= 8:
                (base_times if key == "base" else patch_times).append(elapsed)
        if i < 8:
            continue
        native = values["base"][0]
        added, z, tag, route = values["patch"]
        expected = w[0] * row["score"] + w[1] * row["confidence"] + w[2]
        native_drift = max(native_drift, float(np.max(np.abs(native-added))), float(np.max(np.abs(native-np.asarray(row["logits"])))))
        error = max(error, abs(float(z.item()) - expected))
        tag_errors += bool(tag.item()) != (expected > 0)
        route_errors += int(route.item()) != (3 if expected > 0 else int(native.argmax()))
        active += bool(tag.item())
    if native_drift != 0 or error > 1e-8 or tag_errors or route_errors:
        raise ValueError(f"embedded program diverged: {native_drift}, {error}, {tag_errors}, {route_errors}")
    return {"arm": arm, "regime": regime, "mode": mode, "modelSha256": checkpoint, "rows": len(sample), "activeTags": active,
            "maxNativeLogitDrift": native_drift, "maxEvidenceError": error, "tagDisagreements": int(tag_errors), "routeDisagreements": int(route_errors),
            "cpuMs": {"baseP50": float(np.median(base_times)), "patchP50": float(np.median(patch_times)),
                      "baseP95": float(np.percentile(base_times, 95)), "patchP95": float(np.percentile(patch_times, 95)),
                      "medianPairedChange": float(np.median(np.asarray(patch_times)-base_times))}}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for arg in ("cache", "run", "model"):
        cli.add_argument(arg, type=Path)
    cli.add_argument("regime", choices=("transfer", "adapted"))
    cli.add_argument("mode", choices=("protected", "tag", "separate"))
    cli.add_argument("--arm", choices=("wsp48", "float13"), default="wsp48")
    args = cli.parse_args()
    print(json.dumps(validate(args.cache, args.run, args.model, args.regime, args.mode, args.arm)))
