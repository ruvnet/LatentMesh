"""Score the fresh target calibration comparison from paired categorical predictions."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys
import zlib

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from score import cp_upper, digits, distribution, rn

CONTROLS = ("softKD", "float13", "hard416", "binary4", "text4", "observed8", "local", "intact")


def summarize(raw, lock):
    data = json.loads(raw)
    if data["lockSha256"] != hashlib.sha256(lock.read_bytes()).hexdigest():
        raise ValueError("target calibration lock mismatch")
    expected = {*CONTROLS, "wsp48", "oracle"}
    out = {"observationsSha256": hashlib.sha256(raw).hexdigest(), "skills": {}}
    for key, pop in data["populations"].items():
        if set(pop["arms"]) != expected:
            raise ValueError("missing arm")
        truth = digits(pop["labels"], "012" if key == "protected" else "01")
        native = digits(pop["native"], "012")
        pairs = {name: [digits(b, "01") for b in pair] for name, pair in pop["arms"].items()}
        if len(native) != len(truth) or any(len(p) != 2 or len(p[0]) != len(truth) or len(p[1]) != len(truth)
                or np.any(p[1] > p[0]) for p in pairs.values()):
            raise ValueError("invalid or expanding override")
        if key == "protected":
            if np.bincount(truth).tolist() != [600, 600, 600]:
                raise ValueError("protected population differs")
            methods = {}
            for name, pair in pairs.items():
                metrics = []
                for p in pair:
                    displaced = (native == truth) & (p == 1)
                    counts = [int(np.sum(displaced & (truth == k))) for k in range(3)]
                    metrics.append({"lost": sum(counts), "lossPp": rn(100 * displaced.mean()), "lostByClass": counts,
                                    "upper95Pp": rn(np.mean([cp_upper(i, 600, 0.05 / 3) for i in counts]))})
                methods[name] = metrics
            out["protected"] = {"rows": len(truth), "nativeCorrect": int(np.sum(native == truth)), "methods": methods}
            continue
        if len(truth) == 0 or truth.sum() * 2 != len(truth):
            raise ValueError("skill population must be balanced")
        correct = {name: [(v == truth).astype(np.int16) for v in pair] for name, pair in pairs.items()}
        accuracy = {name: [rn(100 * c.mean()) for c in pair] for name, pair in correct.items()}
        strongest = max(accuracy[name][1] for name in CONTROLS)
        best = [name for name in CONTROLS if accuracy[name][1] == strongest]
        lower, ci = {}, {}
        for name in CONTROLS:
            difference = correct["wsp48"][1] - correct[name][1]
            bootstrap = distribution(difference, truth)
            lower[name] = rn(np.quantile(bootstrap, 0.05 / (2 * len(CONTROLS))))
            if name in best:
                ci[name] = [rn(v) for v in np.percentile(bootstrap, [2.5, 97.5])]
        delta = correct["wsp48"][1] - correct["wsp48"][0]
        out["skills"][key] = {"rows": len(truth), "accuracy": accuracy, "best": best,
              "advantagePp": rn(100 * np.mean(correct["wsp48"][1] - correct[best[0]][1])), "paired95VsBest": ci,
              "changeVsOriginalPp": rn(100 * delta.mean()), "paired95VsOriginal": [rn(v) for v in np.percentile(distribution(delta, truth), [2.5, 97.5])],
              "simultaneousLowerPp": lower}
    if set(out["skills"]) != {"go", "carer"} or "protected" not in out:
        raise ValueError("a registered endpoint is missing")
    risk = out["protected"]["methods"]["wsp48"][1]
    out["retentionPass"] = risk["lossPp"] <= 1 and risk["upper95Pp"] < 1
    out["advance"] = out["retentionPass"] and all(s["advantagePp"] >= 2 and min(s["simultaneousLowerPp"].values()) > 2 for s in out["skills"].values())
    return out


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("mode", choices=("score", "verify"))
    cli.add_argument("folder", type=Path)
    args = cli.parse_args()
    if args.mode == "score":
        raw = (args.folder / "observations.json").read_bytes()
    else:
        record = json.loads((args.folder / "observations.archive.json").read_bytes())
        raw = zlib.decompress(base64.b64decode(record["data"], validate=True))
        if len(raw) != record["bytes"] or hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise ValueError("prediction archive corrupted")
    result = summarize(raw, args.folder / "lock.json")
    if args.mode == "verify":
        if result != json.loads((args.folder / "decision.json").read_bytes()):
            raise ValueError("published score differs")
        print("Target calibration decision reproduced", result["retentionPass"], result["advance"])
    else:
        print(json.dumps(result, indent=2, allow_nan=False))
