"""Fold the negotiated WSP readout into a copy of the original ONNX receiver and verify it."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper
from tokenizers import Tokenizer

from data import ensure, file_hash, json_write, packed, session_options, sha


def install(cache, out):
    base = ensure(cache, "target-model.onnx")
    graph = onnx.load(base)
    initial = {t.name: hashlib.sha256(t.SerializeToString()).hexdigest() for t in graph.graph.initializer}
    node_outputs = {name for n in graph.graph.node for name in n.output}
    cls = "/classifier/Gather_output_0"
    if cls not in node_outputs or any(name.startswith("lmwsp_") for name in (node_outputs | set(initial))):
        raise ValueError("unexpected backbone or existing WSP graph")
    lock = json.loads((out / "lock.json").read_bytes())
    if (file_hash(out / "patches.json") != lock["patchesSha256"] or file_hash(out / "public.json") != lock["publicSha256"]):
        raise ValueError("frozen inputs changed")
    selected = json.loads((out / "patches.json").read_bytes())["wsp48"]
    with np.load(out / "basis.npz", allow_pickle=False) as f:
        if sha(packed(f["center"], f["basis"], f["std"])) != json.loads((out / "public.json").read_bytes())["receiverBasis"]:
            raise ValueError("receiver basis differs from signed context")
        v = np.asarray(selected["coef"])
        w = (v[:-1] / f["std"]) @ f["basis"]
        bias = v[-1] - f["center"] @ w
    prefix = "lmwsp_"
    def param(name, x):
        graph.graph.initializer.append(numpy_helper.from_array(np.asarray(x), prefix + name))
    def node(op, inputs, output, **attrs):
        graph.graph.node.append(helper.make_node(op, inputs, [prefix + output], name=prefix + output, **attrs))
    cut = selected["policy"]["cut"]
    if cut == "never":
        cut = np.finfo(float).max
    elif cut == "always":
        cut = -np.finfo(float).max
    param("weights", w.reshape(-1, 1)); param("bias", np.float64(bias)); param("threshold", np.float64(cut))
    param("anger", np.int64(3))
    node("Cast", [cls], "state", to=onnx.TensorProto.DOUBLE)
    node("MatMul", [prefix + "state", prefix + "weights"], "linear")
    node("Add", [prefix + "linear", prefix + "bias"], "score")
    node("Greater", [prefix + "score", prefix + "threshold"], "above")
    node("ArgMax", ["logits"], "native", axis=1, keepdims=1)
    gate = prefix + "above"
    if selected["policy"]["negativeOnly"]:
        param("zero", np.int64(0))
        node("Equal", [prefix + "native", prefix + "zero"], "negative")
        node("And", [gate, prefix + "negative"], "override")
        gate = prefix + "override"
    node("Where", [gate, prefix + "anger", prefix + "native"], "route")
    for name, kind in [("score", onnx.TensorProto.DOUBLE), ("route", onnx.TensorProto.INT64)]:
        graph.graph.output.append(helper.make_tensor_value_info(prefix + name, kind, [None, 1]))
    dest = out / "receiver-with-anger.onnx"
    onnx.checker.check_model(graph)
    onnx.save(graph, dest)
    saved = onnx.load(dest)
    after = {t.name: hashlib.sha256(t.SerializeToString()).hexdigest() for t in saved.graph.initializer}
    if any(after.get(name) != h for name, h in initial.items()):
        raise AssertionError("original initializer changed")
    cfg = session_options()
    original = ort.InferenceSession(str(base), cfg, providers=["CPUExecutionProvider"])
    installed = ort.InferenceSession(str(dest), cfg, providers=["CPUExecutionProvider"])
    tokenizer = Tokenizer.from_file(str(ensure(cache, "target-tokenizer.json")))
    tokenizer.no_padding()
    tokenizer.enable_truncation(max_length=96)
    samples = json.loads((out / "graph-sample.json").read_bytes())
    references = json.loads((out / "standalone-graph.json").read_bytes())
    drift, score_error, route_diff = 0., 0., 0
    for case, expected in zip(samples, references, strict=True):
        token = tokenizer.encode(case["text"])
        inp = {"input_ids": np.asarray([token.ids], dtype="int64"), "attention_mask": np.asarray([token.attention_mask], dtype="int64")}
        native = original.run(["logits"], inp)[0]
        new, value, route = installed.run(["logits", prefix + "score", prefix + "route"], inp)
        drift = max(drift, float(np.max(np.abs(native - new))))
        score_error = max(score_error, abs(value.item() - expected["value"]))
        route_diff += int(route.item() != expected["route"])
    report = {"schema": "latentmesh.graph-parity/1", "rows": len(samples), "originalInitializersIdentical": len(initial),
              "foldedReadoutScalars": len(w) + 1, "artifactExtraBytes": dest.stat().st_size - base.stat().st_size,
              "candidateOnnxSha256": file_hash(dest), "nativeMaxLogitDrift": drift,
              "maxScoreDifferenceFromWSP": score_error, "routeDisagreements": route_diff}
    json_write(out / "graph.json", report)
    print(report)
    if route_diff or drift > 1e-6 or score_error > 2e-4:
        raise ValueError("installed graph did not reproduce the registered numerical checks")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    install(args.cache, args.out)
