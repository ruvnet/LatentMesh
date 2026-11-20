"""Export or apply the small pinned ONNX addition and certify the original computation."""
import argparse
import base64
import hashlib
import json
from pathlib import Path

import onnx


def digest(data):
    return hashlib.sha256(data).hexdigest()


def serialized(items):
    return [x.SerializeToString() for x in items]


def seal(base, candidate):
    # Native outputs have the same ordered producer graph, inputs, operator semantics
    # and constants. The appended pure operators can only read that graph.
    onnx.checker.check_model(candidate)
    before, after = base.graph, candidate.graph
    if (base.ir_version != candidate.ir_version or serialized(base.opset_import) != serialized(candidate.opset_import)
            or serialized(base.functions) != serialized(candidate.functions)
            or serialized(before.input) != serialized(after.input)
            or serialized(before.sparse_initializer) != serialized(after.sparse_initializer)
            or serialized(before.output) != serialized(after.output[:len(before.output)])
            or serialized(before.node) != serialized(after.node[:len(before.node)])
            or serialized(before.initializer) != serialized(after.initializer[:len(before.initializer)])):
        raise ValueError("original computational graph changed")
    available = {x.name for x in before.input} | {x.name for x in before.initializer}
    available.update(name for node in before.node for name in node.output if name)
    new_tensors = after.initializer[len(before.initializer):]
    permitted_names = {"lmwsp_weights", "lmwsp_bias", "lmwsp_threshold", "lmwsp_anger", "lmwsp_zero"}
    for tensor in new_tensors:
        if tensor.name not in permitted_names or tensor.name in available or tensor.external_data:
            raise ValueError("unexpected or shadowed initializer")
        if list(tensor.dims) != ([768, 1] if tensor.name == "lmwsp_weights" else []):
            raise ValueError("invalid initializer shape")
        available.add(tensor.name)
    new_nodes = after.node[len(before.node):]
    allowed = {"Cast", "MatMul", "Add", "Greater", "ArgMax", "Equal", "And", "Where"}
    for node in new_nodes:
        if node.domain or node.op_type not in allowed or not node.name.startswith("lmwsp_") or len(node.output) != 1:
            raise ValueError("unexpected operation in appended branch")
        if any(name not in available for name in node.input):
            raise ValueError("unresolved input in appended branch")
        name = node.output[0]
        if name in available or not name.startswith("lmwsp_"):
            raise ValueError("existing value shadowed")
        available.add(name)
    if [o.name for o in after.output[len(before.output):]] != ["lmwsp_score", "lmwsp_route"]:
        raise ValueError("unexpected new outputs")
    return {"originalNodes": len(before.node), "originalInitializers": len(before.initializer),
            "originalInputs": [x.name for x in before.input], "originalOutputs": [x.name for x in before.output],
            "appendedPureNodes": len(new_nodes), "appendedInitializers": len(new_tensors),
            "nativeComputationIdentical": True}


def make(base_path, candidate_path, artifact_path):
    base_bytes, target = base_path.read_bytes(), candidate_path.read_bytes()
    base, candidate = onnx.load_model_from_string(base_bytes), onnx.load_model_from_string(target)
    proof = seal(base, candidate)
    extra = onnx.GraphProto()
    extra.name = "latentmesh_frontier_readout"
    extra.node.extend(candidate.graph.node[len(base.graph.node):])
    extra.initializer.extend(candidate.graph.initializer[len(base.graph.initializer):])
    extra.output.extend(candidate.graph.output[len(base.graph.output):])
    encoded = extra.SerializeToString()
    artifact = {"schema": "latentmesh.onnx-readout-addition/1", "baseSha256": digest(base_bytes),
                "modelSha256": digest(target), "additionSha256": digest(encoded), "additionBytes": len(encoded),
                "certificate": proof, "additionBase64": base64.b64encode(encoded).decode("ascii")}
    artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")
    return proof


def apply(base_path, artifact_path, expected_sha, destination):
    raw = artifact_path.read_bytes()
    if len(raw) > 20000 or digest(raw) != expected_sha:
        raise ValueError("unexpected artifact fingerprint or size")
    artifact = json.loads(raw)
    base_bytes = base_path.read_bytes()
    if artifact["schema"] != "latentmesh.onnx-readout-addition/1" or digest(base_bytes) != artifact["baseSha256"]:
        raise ValueError("wrong backbone or schema")
    b = base64.b64decode(artifact["additionBase64"], validate=True)
    if len(b) != artifact["additionBytes"] or digest(b) != artifact["additionSha256"]:
        raise ValueError("addition digest mismatch")
    extra = onnx.GraphProto.FromString(b)
    model, original = onnx.load_model_from_string(base_bytes), onnx.load_model_from_string(base_bytes)
    model.graph.node.extend(extra.node)
    model.graph.initializer.extend(extra.initializer)
    model.graph.output.extend(extra.output)
    evidence = seal(original, model)
    combined = model.SerializeToString()
    if evidence != artifact["certificate"] or digest(combined) != artifact["modelSha256"]:
        raise ValueError("reconstructed graph differs from attested artifact")
    with destination.open("xb") as f:
        f.write(combined)
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="action", required=True)
    make_p = modes.add_parser("make")
    for key in ("base", "candidate", "artifact"):
        make_p.add_argument(key, type=Path)
    apply_p = modes.add_parser("apply")
    apply_p.add_argument("base", type=Path)
    apply_p.add_argument("artifact", type=Path)
    apply_p.add_argument("expected_sha")
    apply_p.add_argument("destination", type=Path)
    args = parser.parse_args()
    result = make(args.base, args.candidate, args.artifact) if args.action == "make" else apply(
        args.base, args.artifact, args.expected_sha, args.destination)
    print(json.dumps(result))
