"""Install a hash bound optional anger tag on the unedited quantized RoBERTa receiver."""
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
from install import fingerprint

LOCK_SHA = "a0ebc489e00815f033e5626881859e128b74076d244984cb5bc1735714c68067"
BASE_SHA = "046f7e4cc46b399558fa9b2de966f6b0d42c69e4b01f582bfb4099b01a0bb5d7"
FROZEN_SHA = "e3d8c7b05aab5578e0e42fba5a1835678e2deff78fd998347097e6a277f7c815"
MODES = {"majority": "dd_anger", "f05": "dd_anger_f05"}


def digest(b):
    return hashlib.sha256(b).hexdigest()


def committed():
    raw = (HERE / "lock.json").read_bytes()
    if digest(raw) != LOCK_SHA:
        raise ValueError("wrong DailyDialog calibration lock")
    return json.loads(raw)


def readout(policy):
    raw = base64.b64decode(policy["foldedBase64"], validate=True)
    if len(raw) != 769 * 8 or digest(raw) != policy["foldedSha256"]:
        raise ValueError("invalid folded WSP readout")
    w = np.frombuffer(raw, dtype="<f8")
    if not np.isfinite(w).all():
        raise ValueError("nonfinite readout")
    return w


def install(base_path, destination, arm="transfer"):
    policy = committed()["methods"][arm]
    w = readout(policy)
    raw = base_path.read_bytes()
    if digest(raw) != BASE_SHA:
        raise ValueError("wrong native checkpoint")
    model = onnx.load_model_from_string(raw)
    graph = model.graph
    fields = ("node", "initializer", "input", "output")
    counts = {k: len(getattr(graph, k)) for k in fields}
    prefix = {k: fingerprint(getattr(graph, k), counts[k]) for k in fields}
    available = {s for n in graph.node for s in n.output} | {t.name for t in graph.initializer}
    hidden = "/classifier/Gather_output_0"
    if hidden not in available or any(s.startswith("dd_") for s in available):
        raise ValueError("incompatible graph namespace")
    def constant(name, value):
        graph.initializer.append(numpy_helper.from_array(np.asarray(value, dtype=np.float64), name))
    def op(name, operation, inputs, **attrs):
        graph.node.append(helper.make_node(operation, inputs, [name], name=name, **attrs))
    def output(name, kind):
        graph.output.append(helper.make_tensor_value_info(name, kind, [None, 1]))
    constant("dd_weights", w[:-1].reshape(768, 1))
    constant("dd_bias", w[-1])
    op("dd_state", "Cast", [hidden], to=TensorProto.DOUBLE)
    op("dd_linear", "MatMul", ["dd_state", "dd_weights"])
    op("dd_score", "Add", ["dd_linear", "dd_bias"])
    output("dd_score", TensorProto.DOUBLE)
    for mode, name in MODES.items():
        value = policy["points"][mode]["threshold"]
        if value is None:
            op(name, "Less", ["dd_score", "dd_score"])
        else:
            if not np.isfinite(value):
                raise ValueError("nonfinite cutoff")
            constant(name + "_threshold", value)
            op(name, "Greater", ["dd_score", name + "_threshold"])
        output(name, TensorProto.BOOL)
    if any(fingerprint(getattr(graph, k), n) != prefix[k] for k, n in counts.items()):
        raise ValueError("changed native graph prefix")
    onnx.checker.check_model(model)
    result = model.SerializeToString()
    with tempfile.NamedTemporaryFile(dir=destination.parent) as f:
        f.write(result)
        f.flush()
        os.link(f.name, destination)
    return {"arm": arm, "lockSha256": LOCK_SHA, "baseSha256": BASE_SHA, "modelSha256": digest(result),
            "addedBytes": len(result)-len(raw), "originalNodes": counts["node"],
            "originalInitializers": counts["initializer"], "addedNodes": len(graph.node)-counts["node"]}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("base", type=Path)
    cli.add_argument("destination", type=Path)
    cli.add_argument("--arm", choices=("transfer", "local", "pca"), default="transfer")
    args = cli.parse_args()
    print(json.dumps(install(args.base, args.destination, args.arm)))
