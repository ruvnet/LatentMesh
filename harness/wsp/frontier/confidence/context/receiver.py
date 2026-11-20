"""Install a gradient free optional contextual readout in the original RoBERTa graph."""
import argparse
import base64
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "dailydialog"))
from install_receiver import BASE_SHA, LOCK_SHA as PARENT_SHA, digest, fingerprint
from stream import OUTPUTS

LOCK_SHA = "f7b72714a143c4706882d20f0b265e015eae504fa8c63083848e16a08451592f"


def unpack(blob, size):
    raw = base64.b64decode(blob["base64"], validate=True)
    array = np.frombuffer(raw, dtype="<f8")
    if array.size != size or not np.isfinite(array).all() or digest(raw) != blob["sha256"]:
        raise ValueError("incorrect contextual readout parameters")
    return array


def parameters(arm):
    raw = (HERE / "lock.json").read_bytes()
    if digest(raw) != LOCK_SHA:
        raise ValueError("wrong contextual lock")
    lock = json.loads(raw)
    raw = (HERE / "projection.json").read_bytes()
    if lock["parentLock"] != PARENT_SHA or digest(raw) != lock["projectionSha256"]:
        raise ValueError("wrong receiver projection or parent")
    projection = unpack(json.loads(raw), 768*8 + 8)
    policy = lock["arms"][arm]
    weights = unpack(policy["readout"], 768 + 1 + 8 + 8)
    return (projection[:-8].reshape(768,8), projection[-8:], weights[:768],
            weights[768], weights[769:777], weights[777:], policy["points"])


def install(base_path, destination, arm):
    projection, offset, w, bias, prev_w, cross_w, points = parameters(arm)
    raw = base_path.read_bytes()
    if digest(raw) != BASE_SHA:
        raise ValueError("unexpected native checkpoint")
    model = onnx.load_model_from_string(raw)
    graph = model.graph
    counts = {k: len(getattr(graph, k)) for k in ("node", "initializer", "input", "output")}
    prefix = {k: fingerprint(getattr(graph, k), n) for k, n in counts.items()}
    names = {s for n in graph.node for s in n.output} | {s.name for s in graph.input} | {s.name for s in graph.initializer}
    hidden = "/classifier/Gather_output_0"
    if hidden not in names or any(s.startswith("ctx_") for s in names):
        raise ValueError("unexpected graph namespace")

    def constant(name, value):
        graph.initializer.append(numpy_helper.from_array(np.asarray(value, dtype=np.float64), name))

    def op(name, kind, ins, **attrs):
        graph.node.append(helper.make_node(kind, ins, [name], name=name, **attrs))

    def info(name, kind, dims):
        return helper.make_tensor_value_info(name, kind, dims)

    # A caller may omit the context on the first turn and when requesting only logits.
    graph.input.append(info("ctx_previous", TensorProto.DOUBLE, [None, 8]))
    constant("ctx_previous", np.zeros((1, 8)))
    constant("ctx_project", projection)
    constant("ctx_offset", offset)
    constant("ctx_current_w", w[:, None])
    constant("ctx_bias", bias)
    constant("ctx_prev_w", prev_w[:, None])
    constant("ctx_cross_w", cross_w[:, None])
    op("ctx_h", "Cast", [hidden], to=TensorProto.DOUBLE)
    op("ctx_projection", "MatMul", ["ctx_h", "ctx_project"])
    op("ctx_next", "Add", ["ctx_projection", "ctx_offset"])
    op("ctx_current_dot", "MatMul", ["ctx_h", "ctx_current_w"])
    op("ctx_current", "Add", ["ctx_current_dot", "ctx_bias"])
    op("ctx_prior", "MatMul", ["ctx_previous", "ctx_prev_w"])
    op("ctx_product", "Mul", ["ctx_previous", "ctx_next"])
    op("ctx_interaction", "MatMul", ["ctx_product", "ctx_cross_w"])
    op("ctx_sum", "Add", ["ctx_current", "ctx_prior"])
    op("ctx_score", "Add", ["ctx_sum", "ctx_interaction"])
    graph.output.append(info("ctx_score", TensorProto.DOUBLE, [None, 1]))
    for mode, name in (("majority", "ctx_anger"), ("f05", "ctx_anger_f05")):
        threshold = points[mode]["threshold"]
        if threshold is None:
            op(name, "Less", ["ctx_score", "ctx_score"])
        else:
            if not np.isfinite(threshold):
                raise ValueError("nonfinite cutoff")
            constant(name + "_threshold", threshold)
            op(name, "Greater", ["ctx_score", name + "_threshold"])
        graph.output.append(info(name, TensorProto.BOOL, [None, 1]))
    graph.output.append(info("ctx_next", TensorProto.DOUBLE, [None, 8]))
    if any(fingerprint(getattr(graph, k), n) != prefix[k] for k, n in counts.items()):
        raise ValueError("changed native graph")
    onnx.checker.check_model(model)
    result = model.SerializeToString()
    with tempfile.NamedTemporaryFile(dir=destination.parent) as handle:
        handle.write(result)
        handle.flush()
        os.link(handle.name, destination)
    return {"arm":arm, "lockSha256":LOCK_SHA, "baseSha256":BASE_SHA, "modelSha256":digest(result),
            "addedBytes":len(result)-len(raw), "stateBytes":8*8, "addedNodes":len(graph.node)-counts["node"],
            "originalNodes":counts["node"], "originalTensors":counts["initializer"]}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("base", type=Path)
    cli.add_argument("destination", type=Path)
    cli.add_argument("--arm", choices=("transfer", "local"), default="transfer")
    a = cli.parse_args()
    print(json.dumps(install(a.base, a.destination, a.arm)))
