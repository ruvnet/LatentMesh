"""Recompute the gate locally. Use --exact to require the reference predictions and fingerprints."""
import argparse
import hashlib
import json
from pathlib import Path

from score import analyze


def read(path):
    return json.loads(path.read_bytes())


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(reference, run, exact=False):
    lock = read(reference / "lock.json")
    ref = read(reference / "predictions.json")
    if (file_hash(reference / "freeze.json") != lock["freezeSha256"]
            or ref["freezeHash"] != lock["freezeSha256"] or ref["candidatesSha256"] != lock["candidatesSha256"]
            or {k: v["patchSha256"] for k, v in ref["costs"].items()} != lock["patches"]):
        raise ValueError("reference diverged from the published candidate lock")
    current = read(run / "predictions.json")
    if file_hash(run / "freeze.json") != current["freezeHash"]:
        raise ValueError("the run used a different setup from its recorded freeze")
    if (run / "candidates.json").exists() and file_hash(run / "candidates.json") != current["candidatesSha256"]:
        raise ValueError("the run's candidate file differs from the scored hash")
    ref_freeze, run_freeze = read(reference / "freeze.json"), read(run / "freeze.json")
    portable = {k for k in ref_freeze if k not in {"basisSha256", "directionSha256"}}
    if any(run_freeze[k] != ref_freeze[k] for k in portable) or current["splits"] != ref["splits"]:
        raise ValueError("pinned inputs, calibration bank or held out populations differ")
    for arm, v in ref["costs"].items():
        for field in ("count", "edits", "usefulBytes", "paddingBytes", "applicationBytes", "wireBytes", "frames"):
            if current["costs"][arm][field] != v[field]:
                raise ValueError(f"{arm} changed its {field}")
    decision = analyze(run / "predictions.json")
    if decision != read(run / "decision.json"):
        raise ValueError("paired observations do not reproduce the archived scorecard")
    same = current == ref
    if exact and not same:
        raise ValueError("the full reference evidence was not reproduced exactly")
    for field in ("advantagePass", "retentionPass", "advance"):
        if decision["decision"][field] != read(reference / "decision.json")["decision"][field]:
            raise ValueError(f"registered gate outcome diverged on {field}")
    differences = {key: sum(a != b for a, b in zip(current["correct"]["hybrid"][key], ref["correct"]["hybrid"][key]))
                   for key in ("primary", "retention")}
    print("Exact reference evidence." if same else f"Independent gate reproduction; hybrid prediction differences: {differences}.")
    print("Registered advance decision:", decision["decision"]["advance"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("--exact", action="store_true")
    args = parser.parse_args()
    validate(args.reference, args.run, args.exact)
