"""Replay two independent prediction archives and validate their registered wire and graph checks."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import zlib

from score import analyze


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def archive(source, destination):
    raw = source.read_bytes()
    record = {"encoding": "zlib+base64", "bytes": len(raw), "sha256": digest(raw),
              "data": base64.b64encode(zlib.compress(raw, 9)).decode("ascii")}
    destination.write_text(json.dumps(record, indent=2) + "\n")


def check_host(folder, setup):
    public, lock = (read(setup / name) for name in ("public.json", "lock.json"))
    combined = read(folder / "observations.archive.json")
    require(combined["encoding"] == "zlib+base64", "unknown prediction archive format")
    raw = zlib.decompress(base64.b64decode(combined["data"], validate=True))
    require(len(raw) == combined["bytes"] and digest(raw) == combined["sha256"], "prediction archive mismatch")
    observations = json.loads(raw)
    require(digest((setup / "public.json").read_bytes()) == lock["publicSha256"], "setup mismatch")
    require(digest((setup / "lock.json").read_bytes()) == observations["lockSha256"], "candidate lock mismatch")
    require(lock["compactMethods"] == observations["compactMethods"], "registered methods differ")
    methods = {"wsp48", "oracle", *lock["compactMethods"]}
    for name, case in lock["methods"].items():
        transport = ({"useful": 0, "application": 0, "frames": 0, "wire": 0} if name in {"local", "intact", "oracle"}
                     else {"useful": 52, "application": 154, "frames": 1, "wire": 218})
        require(case["bytes"] == transport, f"wire budget differs for {name}")
    require(set(lock["methods"]) == methods and lock["momentMaxCoefficientError"] < 1e-4, "registered solve differs")
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "observations.json"
        path.write_bytes(raw)
        scorecard = analyze(path)
    require(scorecard == read(folder / "decision.json"), "recomputed decision differs")
    official = read(folder / "official-scorer.json")
    require(official["macroF1"] == scorecard["official"]["macroF1"] and official["rows"] == 1421,
            "benchmark authors' scores differ")
    gold = ("\n".join(observations["official"]["truth"]) + "\n").encode()
    require(digest(gold) == official["officialTestLabelsSha256"] == public["assets"]["emotion-test-labels.txt"],
            "official test labels differ")
    require(official["scriptSha256"] == "86c824e466ffba0cd407655fbbda759c6a9c09e541be4bfaaf547df8255d2793",
            "official scorer revision differs")
    require(set(observations["official"]["methods"]) == methods and observations["official"]["exactDevelopmentTextOverlaps"] == 0,
            "official arms or partition differ")
    graph = read(folder / "graph.json")
    require(graph["rows"] == 64 and graph["originalInitializersIdentical"] == 355 and graph["nativeMaxLogitDrift"] == 0
            and graph["routeDisagreements"] == 0 and graph["maxScoreDifferenceFromWSP"] < 2e-4, "graph parity differs")
    ids = [observations["official"]["ids"]]
    for key, expected_n in (("go", 700), ("tweet", 900)):
        domain = observations["domains"][key]
        require(set(domain["methods"]) == methods and len(domain["truth"]) == 512 and len(domain["labels"]) == expected_n,
                f"arms or cohort size differ for {key}")
        ids.extend((domain["primaryIds"], domain["retentionIds"], domain["truth"], domain["labels"]))
    print(folder.name, "official macro F1", scorecard["official"]["macroF1"]["wsp48"],
          "both domains meet point thresholds", scorecard["advance"])
    return public, ids


def verify(root):
    ref, ids = check_host(root / "reference", root)
    remote, other_ids = check_host(root / "ruos", root / "ruos")
    require(ids == other_ids, "hosts evaluated different populations")
    for field in ("assets", "publicPartitionHashes", "mapping", "anchorTexts"):
        require(ref[field] == remote[field], f"host setup differs in {field}")
    artifact_path = root / "reference" / "adapter.json"
    artifact = read(artifact_path)
    graph = read(root / "reference" / "graph.json")
    require(digest(artifact_path.read_bytes()) == "f825228564f424efcb35e28bb52f38972e29d73425b88eb714d7c97b4aca99d3",
            "reference installer fingerprint differs")
    addition = base64.b64decode(artifact["additionBase64"], validate=True)
    require(len(addition) == artifact["additionBytes"] and digest(addition) == artifact["additionSha256"]
            and artifact["modelSha256"] == graph["candidateOnnxSha256"]
            and artifact["baseSha256"] == ref["assets"]["target-model.onnx"], "portable artifact differs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    record = commands.add_parser("archive")
    record.add_argument("source", type=Path)
    record.add_argument("destination", type=Path)
    commands.add_parser("verify").add_argument("root", type=Path)
    args = parser.parse_args()
    if args.action == "archive":
        archive(args.source, args.destination)
    else:
        verify(args.root)
