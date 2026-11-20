"""Check all three installable readouts on the same 128 deterministic validation turns."""
import argparse
from pathlib import Path
import tempfile

import numpy as np
import onnxruntime as ort

from corpus import describe, history, partition, sha
from cohorts import Encoder, ensure, json_write
from data import session_options
from install_receiver import MODES, committed, install, readout


def check(cache, output):
    seen = history(cache)
    partition(cache, "train", seen)
    _, val, _ = partition(cache, "validation", seen)
    sample = sorted(val, key=lambda r: sha(("daily-install-check/" + r.id).encode()))[:128]
    enc = Encoder(cache, "target")
    inputs, states, logits = [], [], []
    for row in sample:
        tokens = enc.tokenizer.encode(row.text)
        feed = {"input_ids": np.asarray([tokens.ids], dtype="int64"), "attention_mask": np.asarray([tokens.attention_mask], dtype="int64")}
        inputs.append(feed)
        states.append(enc.graph.run([enc.hidden], feed)[0].astype(np.float64)[0])
        logits.append(enc.original.run(["logits"], feed)[0])
    del enc
    methods = {}
    for arm, policy in committed()["methods"].items():
        w = readout(policy)
        expected = np.asarray(states) @ w[:-1] + w[-1]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "receiver.onnx"
            cert = install(ensure(cache, "target-model.onnx"), path, arm)
            sess = ort.InferenceSession(str(path), session_options(), providers=["CPUExecutionProvider"])
            drift = error = 0.
            disagreements = 0
            for i, feed in enumerate(inputs):
                native, score, *tags = sess.run(["logits", "dd_score", *MODES.values()], feed)
                drift = max(drift, float(np.max(np.abs(native-logits[i]))))
                error = max(error, abs(score.item()-expected[i]))
                for mode, tag in zip(MODES, tags, strict=True):
                    cut = policy["points"][mode]["threshold"]
                    disagreements += int(bool(tag.item()) != (cut is not None and expected[i] > cut))
            if drift != 0 or error > 1e-8 or disagreements:
                raise ValueError("compiled candidate disagrees: " + arm)
        methods[arm] = {"installation": cert, "nativeDrift": drift, "maxScoreError": error, "tagDisagreements": disagreements}
    json_write(output, {"validation": describe(sample), "methods": methods})


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("cache", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    check(args.cache, args.output)
