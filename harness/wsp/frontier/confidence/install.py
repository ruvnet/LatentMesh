"""Install one calibrated WSP program as additional outputs on the exact reference ONNX receiver."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from artifact import apply

REFERENCE_ADAPTER = "f825228564f424efcb35e28bb52f38972e29d73425b88eb714d7c97b4aca99d3"
LOCK_SHA256 = "68e02cc51c5c4f1073ab582d240bcdde6f94887402adcc0526252c17cd15aacb"
PRECISION_LOCK_SHA256 = "6765e09c6436e16457436b88d7611a8b1982eda4c61f610221ed9a7af2e8856d"


def coefficients(regime, mode, arm="wsp48"):
    precision = regime == "all"
    valid = ("f05", "majority") if precision else ("protected", "tag", "separate")
    arms = ("float13", "wsp48", "scoreOnly") if precision else ("float13", "wsp48")
    if mode not in valid or regime not in ("transfer", "adapted", "all") or arm not in arms:
        raise ValueError("unsupported receiver regime, mode or arm")
    raw = ((HERE / "precision" if precision else HERE) / "lock.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != (PRECISION_LOCK_SHA256 if precision else LOCK_SHA256):
        raise ValueError("unexpected receiver calibration lock")
    policies = json.loads(raw)["policies"]
    table = policies if precision else policies[regime]
    values = np.asarray(table[arm][mode]["coefficients"], dtype=np.float64)
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError("invalid three parameter WSP program")
    if arm == "scoreOnly" and values[1] != 0:
        raise ValueError("score only program contains a confidence coefficient")
    return values


def fingerprint(items, count):
    h = hashlib.sha256()
    for i in range(count):
        raw = items[i].SerializeToString()
        h.update(len(raw).to_bytes(8, "big"))
        h.update(raw)
    return h.hexdigest()


def float_readout():
    index = (HERE / "heads" / "index.json").read_bytes()
    expected = "e5035042fc80e9640c221ceff53127891d5985ddd53ff647eb308c446d76e396"
    if hashlib.sha256(index).hexdigest() != expected:
        raise ValueError("incorrect frozen head manifest")
    raw = base64.b64decode((HERE / "heads/float13.b64").read_text().strip(), validate=True)
    if len(raw) != 769 * 8 or hashlib.sha256(raw).hexdigest() != json.loads(index)["arms"]["float13"]:
        raise ValueError("incorrect 13 float source head")
    return np.frombuffer(raw, dtype="<f8")


def extend(model, w, arm):
    graph = model.graph
    if [i.name for i in graph.output] != ["logits", "lmwsp_score", "lmwsp_route"]:
        raise ValueError("missing reference WSP outputs")
    names = {n for node in graph.node for n in node.output} | {t.name for t in graph.initializer}
    if any(n.startswith("wspc_") for n in names) or not {"lmwsp_native", "lmwsp_anger"} <= names:
        raise ValueError("unexpected receiver namespace")
    fields = ("node", "initializer", "input", "output")
    original = {k: len(getattr(graph, k)) for k in fields}
    proof = {k: fingerprint(getattr(graph, k), original[k]) for k in fields}
    weights = w[:1] if arm == "scoreOnly" else w[:2].reshape(2, 1)
    for name, value in (("weights", weights), ("bias", w[2:]), ("zero", np.array(0., dtype=np.float64))):
        graph.initializer.append(numpy_helper.from_array(value, "wspc_" + name))

    def add(op, inputs, output, **attrs):
        graph.node.append(helper.make_node(op, inputs, ["wspc_" + output], name="wspc_" + output, **attrs))

    source = "lmwsp_score"
    if arm in ("float13", "scoreOnly"):
        source_head = float_readout()
        graph.initializer.append(numpy_helper.from_array(source_head[:-1].reshape(768, 1), "wspc_source_weights"))
        graph.initializer.append(numpy_helper.from_array(source_head[-1:], "wspc_source_bias"))
        add("MatMul", ["lmwsp_state", "wspc_source_weights"], "source_linear")
        add("Add", ["wspc_source_linear", "wspc_source_bias"], "source_score")
        source = "wspc_source_score"
    if arm == "scoreOnly":
        add("Mul", [source, "wspc_weights"], "linear")
    else:
        add("Cast", ["logits"], "logits_f64", to=TensorProto.DOUBLE)
        add("Softmax", ["wspc_logits_f64"], "probabilities", axis=1)
        add("ReduceMax", ["wspc_probabilities"], "confidence", axes=[1], keepdims=1)
        add("Concat", [source, "wspc_confidence"], "features", axis=1)
        add("MatMul", ["wspc_features", "wspc_weights"], "linear")
    add("Add", ["wspc_linear", "wspc_bias"], "evidence")
    add("Greater", ["wspc_evidence", "wspc_zero"], "tag")
    add("Where", ["wspc_tag", "lmwsp_anger", "lmwsp_native"], "route")
    for name, kind in (("evidence", TensorProto.DOUBLE), ("tag", TensorProto.BOOL), ("route", TensorProto.INT64)):
        graph.output.append(helper.make_tensor_value_info("wspc_" + name, kind, [None, 1]))
    if any(fingerprint(getattr(graph, k), n) != proof[k] for k, n in original.items()):
        raise ValueError("existing computation changed")
    onnx.checker.check_model(model)
    return {"preserved": original, "prefixSha256": proof, "newNodes": len(graph.node)-original["node"], "learnedScalars": 2 if arm == "scoreOnly" else 3}


def install(original_base, regime, mode, destination, arm="wsp48"):
    with tempfile.TemporaryDirectory() as work:
        first = Path(work) / "first.onnx"
        initial_proof = apply(original_base, HERE.parent / "reference" / "adapter.json", REFERENCE_ADAPTER, first)
        model = onnx.load(first)
        proof = extend(model, coefficients(regime, mode, arm), arm)
        raw = model.SerializeToString()
        record = {"arm": arm, "regime": regime, "mode": mode, "graph": proof, "originalNativeGraph": initial_proof,
                  "calibrationLock": PRECISION_LOCK_SHA256 if regime == "all" else LOCK_SHA256, "modelSha256": hashlib.sha256(raw).hexdigest(),
                  "bytesBeyondPreviousAdapter": len(raw) - first.stat().st_size}
    # Link the complete file into place on the destination filesystem.
    # Existing files remain intact even on concurrent builds or disk exhaustion.
    with tempfile.NamedTemporaryFile(dir=destination.parent) as f:
        f.write(raw)
        f.flush()
        os.link(f.name, destination)
    return record


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("base", type=Path)
    cli.add_argument("regime", choices=("transfer", "adapted", "all"))
    cli.add_argument("mode", choices=("protected", "tag", "separate", "f05", "majority"))
    cli.add_argument("destination", type=Path)
    cli.add_argument("--arm", choices=("wsp48", "float13", "scoreOnly"), default="wsp48")
    args = cli.parse_args()
    print(json.dumps(install(args.base, args.regime, args.mode, args.destination, args.arm)))
