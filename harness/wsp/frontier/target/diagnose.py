"""Post hoc threshold-independent ranking of frozen scores on the two skill populations."""
import argparse
from pathlib import Path

import numpy as np
from scipy.stats import norm

from trial import Encoder, encode, init, json_write, skill_split, split_digest, values


def ranking(positive, negative):
    # Mann-Whitney placements count ties at one half. Their within-class
    # variances give the univariate DeLong asymptotic standard error.
    pairs = (positive[:, None] > negative).astype(float) + 0.5 * (positive[:, None] == negative)
    pos, neg = pairs.mean(axis=1), pairs.mean(axis=0)
    auc = pairs.mean()
    error = np.sqrt(pos.var(ddof=1) / len(pos) + neg.var(ddof=1) / len(neg))
    interval = np.clip(auc + np.array([-1, 1]) * norm.ppf(.975) * error, 0, 1)
    return {"auc": round(float(auc), 5), "approximate95": [round(float(x), 5) for x in interval],
            "positiveMedian": round(float(np.median(positive)), 5), "negativeMedian": round(float(np.median(negative)), 5)}


def diagnose(cache, run, target):
    import json
    basis, patches, _ = init(run)
    observed = json.loads((target / "observations.json").read_bytes())["populations"]
    enc = Encoder(cache, "target")
    output = {"scope": "post hoc rank diagnostic, frozen weights, no threshold selection", "domains": {}}
    for key, rows in skill_split(cache).items():
        if split_digest(rows) != observed[key]["ids"]:
            raise ValueError("fresh diagnostic partition differs")
        x, _, raw_labels = encode(rows, enc, basis)
        truth = raw_labels == ("anger" if key == "go" else 3)
        report = {}
        for name in ("wsp48", "softKD", "float13"):
            scores = values(x, patches[name])
            report[name] = ranking(scores[truth], scores[~truth])
        output["domains"][key] = {"rows": len(rows), "methods": report}
    json_write(target / "ranking.json", output)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "parent_run", "target_run"):
        args.add_argument(name, type=Path)
    options = args.parse_args()
    diagnose(options.cache, options.parent_run, options.target_run)
