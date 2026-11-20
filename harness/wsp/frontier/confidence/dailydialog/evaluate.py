"""Run frozen and supervised receivers against the entire official DailyDialog test."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import onnxruntime as ort

from corpus import DATA_SHA, HERE, describe, history, partition
from cohorts import ARMS, Encoder, ensure, file_hash, json_write, load_heads
from data import session_options
from install_receiver import FROZEN_SHA, LOCK_SHA, committed, install, readout
from install import PRECISION_LOCK_SHA256, install as install_frozen
sys.path.insert(0, str(HERE.parents[1]))
from evidence import archive

ADAPTED_SHA = "d39c71b00321fbddfb8afebe6bb61076d694a3103b48c654d5bc58ad2d3d3e5c"
FROZEN_ARMS = ("scoreOnly", "float13", "local", "wsp48", "softKD")


def session(path, expected):
    if file_hash(path) != expected:
        raise ValueError("unexpected installed ONNX checkpoint")
    return ort.InferenceSession(str(path), session_options(), providers=["CPUExecutionProvider"])


def run(cache, out):
    out.mkdir(parents=True, exist_ok=True)
    locked = committed()
    prior_path = HERE.parent / "precision" / "lock.json"
    if file_hash(prior_path) != PRECISION_LOCK_SHA256 or locked["parentLock"] != PRECISION_LOCK_SHA256:
        raise ValueError("wrong historical readouts")
    prior = json.loads(prior_path.read_bytes())["policies"]
    seen = history(cache)
    for part in ("train", "validation"):
        partition(cache, part, seen)
    rows, clean, exclusions = partition(cache, "test", seen)
    keys = {r.id for r in clean}
    frozen_path, adapted_path = out / "frozen.onnx", out / "transfer.onnx"
    base = ensure(cache, "target-model.onnx")
    if not frozen_path.exists():
        install_frozen(base, "all", "f05", frozen_path, "scoreOnly")
    receipt = out / "graph.json"
    if not adapted_path.exists():
        json_write(receipt, install(base, adapted_path))
    elif not receipt.exists():
        with tempfile.TemporaryDirectory(dir=out) as temp:
            fresh = install(base, Path(temp) / "receiver.onnx")
        if fresh["modelSha256"] != file_hash(adapted_path):
            raise ValueError("cached candidate differs from a clean install")
        json_write(receipt, fresh)
    frozen_graph, adapted_graph = session(frozen_path, FROZEN_SHA), session(adapted_path, ADAPTED_SHA)
    enc = Encoder(cache, "target")
    heads = load_heads(HERE.parent / "heads")
    methods = {k: readout(v) for k, v in locked["methods"].items()}
    predictions = {"frozen." + k: [] for k in FROZEN_ARMS}
    predictions.update({k + "." + mode: [] for k in methods for mode in ("majority", "f05")})
    truth, native_decisions, mask = [], [], []
    proof = {"checkedTurns": 0, "instrumentedDrift": 0., "frozenDrift": 0., "adaptedDrift": 0.,
             "frozenTagErrors": 0, "adaptedMajorityErrors": 0, "adaptedF05Errors": 0,
             "maxFrozenEvidenceError": 0., "maxAdaptedEvidenceError": 0.}
    original_logits = hashlib.sha256()
    for i, r in enumerate(rows):
        token = enc.tokenizer.encode(r.text)
        feed = {"input_ids": np.asarray([token.ids], dtype="int64"),
                "attention_mask": np.asarray([token.attention_mask], dtype="int64")}
        original = enc.original.run(["logits"], feed)[0]
        instrumented, state = enc.graph.run(["logits", enc.hidden], feed)
        f_native, f_score, f_tag = frozen_graph.run(["logits", "wspc_evidence", "wspc_tag"], feed)
        a_native, a_score, a_tag, a_f05 = adapted_graph.run(["logits", "dd_score", "dd_anger", "dd_anger_f05"], feed)
        if not all(np.isfinite(v).all() for v in (original, instrumented, state, f_native, f_score, a_native, a_score)):
            raise ValueError("nonfinite inference result")
        for field, other in (("instrumentedDrift", instrumented), ("frozenDrift", f_native), ("adaptedDrift", a_native)):
            proof[field] = max(proof[field], float(np.max(np.abs(original-other))))
        original_logits.update(np.asarray(original, dtype="<f4").tobytes())
        h = state[0].astype(np.float64)
        scores = heads[:, :-1] @ h + heads[:, -1]
        native = original[0].astype(np.float64)
        confidence = 1. / np.exp(native - native.max()).sum()
        for name in FROZEN_ARMS:
            source = "float13" if name == "scoreOnly" else name
            a, b, c = prior[name]["f05"]["coefficients"]
            z = a*scores[ARMS.index(source)] + b*confidence + c
            if name == "scoreOnly":
                proof["frozenTagErrors"] += int(bool(f_tag.item()) != (z > 0))
                proof["maxFrozenEvidenceError"] = max(proof["maxFrozenEvidenceError"], abs(f_score.item()-z))
            predictions["frozen." + name].append(z > 0)
        for name, weight in methods.items():
            z = h @ weight[:-1] + weight[-1]
            for mode in ("majority", "f05"):
                cut = locked["methods"][name]["points"][mode]["threshold"]
                emit = cut is not None and z > cut
                predictions[name + "." + mode].append(emit)
                if name == "transfer":
                    actual = a_tag if mode == "majority" else a_f05
                    field = "adaptedMajorityErrors" if mode == "majority" else "adaptedF05Errors"
                    proof[field] += int(bool(actual.item()) != emit)
            if name == "transfer":
                proof["maxAdaptedEvidenceError"] = max(proof["maxAdaptedEvidenceError"], abs(a_score.item()-z))
        truth.append(r.anger)
        mask.append(r.id in keys)
        native_decisions.append(str(int(native.argmax())))
        proof["checkedTurns"] += 1
        if (i+1) % 1000 == 0:
            print("Scored", i+1, "of", len(rows), flush=True)
    if (any(proof[k] != 0 for k in ("instrumentedDrift","frozenDrift","adaptedDrift","frozenTagErrors",
                                    "adaptedMajorityErrors","adaptedF05Errors")) or
            max(proof["maxFrozenEvidenceError"], proof["maxAdaptedEvidenceError"]) > 1e-8):
        raise ValueError("installed policies diverge from original logits or independent math: " + str(proof))
    def bits(values):
        return "".join("1" if v else "0" for v in values)
    record = {"schema": "latentmesh.dailydialog-observations/1", "datasetSha256": DATA_SHA,
              "lockSha256": LOCK_SHA, "frozenGraphSha256": FROZEN_SHA, "adaptedGraphSha256": ADAPTED_SHA,
              "official": describe(rows), "clean": describe(clean), "exclusions": exclusions,
              "groups": [r.group for r in rows], "truth": bits(truth), "isClean": bits(mask),
              "native": "".join(native_decisions), "nativeLogitsSha256": original_logits.hexdigest(),
              "predictions": {k: bits(v) for k,v in predictions.items()}, "parity": proof}
    json_write(out / "observations.json", record)
    archive(out / "observations.json", out / "observations.archive.json")
    print("Parity", proof, flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("cache", type=Path)
    cli.add_argument("out", type=Path)
    args = cli.parse_args()
    run(args.cache, args.out)
