"""Replay the natural prevalence anger tag comparison with paired dialogue uncertainty."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys
import zlib

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from scorecard import DRAWS, SEED, interval, rn, vector


def performance(tp, fp, positives):
    n = tp + fp
    p = np.divide(tp, n, out=np.full(np.shape(tp), np.nan, dtype=float), where=n > 0)
    return {"precision": 100 * p, "recall": 100 * tp / positives, "f05": 500 * tp / (4 * n + positives),
            "f1": 200 * tp / (n + positives)}


def assess(raw, lock_path):
    x = json.loads(raw)
    if x["lockSha256"] != hashlib.sha256(lock_path.read_bytes()).hexdigest():
        raise ValueError("wrong precision lock")
    lock = json.loads(lock_path.read_bytes())
    truth, native, gold = vector(x["truth"], "01"), vector(x["native"], "012"), vector(x["sentiment"], "012")
    groups, inverse = np.unique(x["groups"], return_inverse=True)
    if (set(groups) & (set(lock["sets"]["fit"]["groupIDs"]) | set(lock["sets"]["tune"]["groupIDs"])) or
            len(native) != len(truth) or len(gold) != len(truth) or len(inverse) != len(truth)):
        raise ValueError("invalid protected dialogue populations")
    n = len(groups)
    weights = np.random.default_rng(SEED).multinomial(n, np.full(n,1/n), size=DRAWS).astype(float)
    def cluster(values):
        return weights @ np.bincount(inverse, weights=values, minlength=n)
    positives = int(truth.sum())
    bootstrap_positives = cluster(truth)
    if np.any(bootstrap_positives == 0):
        raise ValueError("sample has no anger episodes")
    arms = {name + "." + mode: bits for name, method in x["predictions"].items() for mode,bits in method.items()}
    arms.update({"previous." + name: bits for name, bits in x["previous"].items()})
    if set(x["predictions"]) != set(lock["policies"]) or any(set(v) != {"f05", "majority"} for v in x["predictions"].values()):
        raise ValueError("missing calibrated control")
    metrics, sampled = {}, {}
    for method, bits in arms.items():
        pred = vector(bits, "01")
        if len(pred) != len(truth):
            raise ValueError("length mismatch")
        hits = pred * truth
        wrong = pred * (1 - truth)
        tp, fp = int(hits.sum()), int(wrong.sum())
        scores = performance(tp, fp, positives)
        loss = int(np.sum(pred * (native == gold)))
        metrics[method] = {k: rn(v) if np.isfinite(v) else None for k,v in scores.items()}
        metrics[method].update({"tp": tp, "fp": fp, "fn": positives - tp, "tags": tp + fp, "lostIfExclusive": loss,
                                "lossIfExclusivePp": rn(100 * loss/len(pred))})
        sampled[method] = performance(cluster(hits), cluster(wrong), bootstrap_positives)
    comparisons = {}
    for selected in ("float13.f05", "float13.majority"):
        comparisons[selected] = {}
        for base in ("previous.adapted", "previous.transfer", "wsp48.f05", "softKD.f05", "local.f05", "scoreOnly.f05"):
            comparisons[selected][base] = {}
            for metric in ("precision", "f05"):
                s, b = metrics[selected][metric], metrics[base][metric]
                delta = sampled[selected][metric] - sampled[base][metric]
                usable = delta[np.isfinite(delta)]
                comparisons[selected][base][metric] = {"differencePp": rn(s-b) if s is not None and b is not None else None,
                    "paired95Pp": interval(usable) if len(usable) >= .99 * DRAWS else None, "validDraws": len(usable)}
    return {"observationsSha256": hashlib.sha256(raw).hexdigest(), "lockSha256": x["lockSha256"], "ids": x["ids"],
        "rows": len(truth), "dialogues": n, "positive": positives, "prevalence": rn(100 * positives/len(truth)),
        "nativeCorrect": int(np.sum(native==gold)), "methods": metrics, "comparisons": comparisons}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("mode", choices=("score", "verify"))
    cli.add_argument("folder", type=Path)
    cli.add_argument("lock", type=Path)
    args = cli.parse_args()
    if args.mode == "score":
        raw = (args.folder / "observations.json").read_bytes()
    else:
        a = json.loads((args.folder / "observations.archive.json").read_bytes())
        raw = zlib.decompress(base64.b64decode(a["data"], validate=True))
        if len(raw) != a["bytes"] or hashlib.sha256(raw).hexdigest() != a["sha256"]:
            raise ValueError("invalid precision archive")
    result = assess(raw, args.lock)
    if args.mode == "verify":
        if result != json.loads((args.folder / "decision.json").read_bytes()):
            raise ValueError("precision scorecard differs")
        print("Verified all emotion tag on", result["rows"], "utterances from", result["dialogues"], "dialogues")
    else:
        print(json.dumps(result, allow_nan=False, separators=(",", ":")))
