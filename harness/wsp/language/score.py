"""Replay all decisions and paired uncertainty from the public correctness bitmaps."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

CONTROLS = ["hard48", "hard64", "decimal52", "text4", "phrase", "native"]


def bitmap(value):
    if not isinstance(value, str) or not value or len(value) % 2 or set(value) - {"0", "1"}:
        raise ValueError("expected an even, nonempty correctness bitmap")
    return np.fromiter((int(x) for x in value), dtype=np.int16)


def paired_interval(first, other):
    if len(first) != len(other):
        raise ValueError("unpaired examples")
    diff = first - other
    rng = np.random.default_rng(72191)
    bootstrap = np.zeros(10000)
    for group in (diff[::2], diff[1::2]):
        idx = rng.integers(len(group), size=(len(bootstrap), len(group)))
        bootstrap += 50 * group[idx].mean(axis=1)
    return [round(x, 4) for x in np.percentile(bootstrap, [2.5, 97.5]).tolist()]


def analyze(path):
    raw = path.read_bytes()
    data = json.loads(raw)
    pairs = {arm: {name: bitmap(s) for name, s in parts.items()} for arm, parts in data["correct"].items()}
    baseline = pairs["intact"]
    n, m = len(baseline["primary"]), len(baseline["retention"])
    metrics = {}
    for name, pair in pairs.items():
        p, r = pair["primary"], pair["retention"]
        if len(p) != n or len(r) != m:
            raise ValueError("arms need identical examples")
        metrics[name] = {"newCorrect": int(p.sum()), "retainedCorrect": int(r.sum()),
                         "newAccuracyPct": round(100 * p.mean(), 4), "retainedAccuracyPct": round(100 * r.mean(), 4),
                         "angerCorrect": int(p[::2].sum()), "otherNegativeCorrect": int(p[1::2].sum())}
    top = max(metrics[k]["newCorrect"] for k in CONTROLS)
    best = [k for k in CONTROLS if metrics[k]["newCorrect"] == top]
    hybrid = pairs["hybrid"]
    dx = metrics["hybrid"]["newCorrect"] - top
    dr = metrics["hybrid"]["retainedCorrect"] - metrics["intact"]["retainedCorrect"]
    return {"schema": "latentmesh.language-decision/1", "predictionsSha256": hashlib.sha256(raw).hexdigest(),
            "freezeSha256": data["freezeHash"], "candidatesSha256": data["candidatesSha256"],
            "rows": {"primary": n, "retention": m}, "metrics": metrics, "source": data["source"],
            "decision": {"strongestCompactControls": best, "advantagePp": round(100 * dx / n, 4),
                         "paired95PpByBestControl": {k: paired_interval(hybrid["primary"], pairs[k]["primary"]) for k in best},
                         "retainedChangePp": round(100 * dr / m, 4),
                         "paired95RetentionPp": paired_interval(hybrid["retention"], baseline["retention"]),
                         "advantagePass": 100 * dx >= 2 * n, "retentionPass": 100 * dr >= -m,
                         "advance": 100 * dx >= 2 * n and 100 * dr >= -m}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    print(json.dumps(analyze(parser.parse_args().predictions), indent=2))
