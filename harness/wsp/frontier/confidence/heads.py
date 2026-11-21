"""Export the eight frozen reference readouts in the original CLS coordinates."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
FRONTIER = HERE.parent
sys.path.insert(0, str(FRONTIER))
from data import file_hash, json_write, packed, sha

ARMS = ("wsp48", "softKD", "float13", "hard416", "binary4", "text4", "observed8", "local")
BASE_ARTIFACT = "f825228564f424efcb35e28bb52f38972e29d73425b88eb714d7c97b4aca99d3"


def export(run, destination):
    import onnx
    from onnx import numpy_helper
    lock = json.loads((FRONTIER / "lock.json").read_bytes())
    public = json.loads((FRONTIER / "public.json").read_bytes())
    if file_hash(run / "patches.json") != lock["patchesSha256"] or file_hash(FRONTIER / "public.json") != lock["publicSha256"]:
        raise ValueError("incorrect reference coefficients or setup")
    patches = json.loads((run / "patches.json").read_bytes())
    with np.load(run / "basis.npz", allow_pickle=False) as src:
        center, basis, std = (src[key] for key in ("center", "basis", "std"))
    if sha(packed(center, basis, std)) != public["receiverBasis"]:
        raise ValueError("reference projection differs")
    portable = FRONTIER / "reference" / "adapter.json"
    if file_hash(portable) != BASE_ARTIFACT:
        raise ValueError("incorrect portable reference")
    graph = onnx.GraphProto.FromString(base64.b64decode(json.loads(portable.read_bytes())["additionBase64"], validate=True))
    constants = {i.name: numpy_helper.to_array(i).ravel() for i in graph.initializer}
    destination.mkdir(parents=True, exist_ok=True)
    index = {"baseModel": public["assets"]["target-model.onnx"], "parentLock": file_hash(FRONTIER / "lock.json"),
             "serialization": "768 little endian float64 CLS weights followed by a float64 intercept", "arms": {}}
    for name in ARMS:
        v = np.asarray(patches[name]["coef"])
        w = (v[:-1] / std[:len(v)-1]) @ basis[:len(v)-1]
        intercept = v[-1] - center @ w
        values = np.r_[w, intercept]
        if name == "wsp48" and not np.array_equal(values, np.r_[constants["lmwsp_weights"], constants["lmwsp_bias"]]):
            raise ValueError("WSP readout differs from previously installed ONNX")
        raw = np.asarray(values, dtype="<f8").tobytes()
        (destination / (name + ".b64")).write_text(base64.b64encode(raw).decode() + "\n")
        index["arms"][name] = hashlib.sha256(raw).hexdigest()
    json_write(destination / "index.json", index)


def load_heads(directory):
    index = json.loads((directory / "index.json").read_bytes())
    if set(index["arms"]) != set(ARMS) or index["parentLock"] != file_hash(FRONTIER / "lock.json"):
        raise ValueError("incorrect arm manifest")
    heads = []
    for name in ARMS:
        raw = base64.b64decode((directory / (name + ".b64")).read_text().strip(), validate=True)
        if len(raw) != 769 * 8 or sha(raw) != index["arms"][name]:
            raise ValueError("incorrect folded readout " + name)
        heads.append(np.frombuffer(raw, dtype="<f8"))
    return np.stack(heads)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_run", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    export(args.reference_run, args.destination)
