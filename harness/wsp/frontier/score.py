"""Score publicly shareable paired predictions without text, checkpoints or an inference runtime."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta

NBOOT = 20000


def digits(s, allowed):
    if not isinstance(s, str) or not s or set(s) - set(allowed):
        raise ValueError("invalid categorical observation")
    return np.fromiter((int(ch) for ch in s), dtype=np.int16)


def rn(value):
    return round(float(value), 5)


def distribution(diff, labels):
    rng = np.random.default_rng(72191)
    result = np.zeros(NBOOT)
    for label in np.unique(labels):
        stratum = diff[labels == label]
        counts = np.bincount(stratum + 1, minlength=3)
        draws = rng.multinomial(len(stratum), counts / len(stratum), size=NBOOT)
        result += 100 * (draws[:, 2] - draws[:, 0]) / len(diff)
    return result


def confidence_metrics(truth, pred):
    counts = np.bincount(truth * 4 + pred, minlength=16).reshape(4, 4)
    return rn(f1(counts))


def f1(matrix):
    tp = np.diagonal(matrix, axis1=-2, axis2=-1)
    denominators = matrix.sum(axis=-1) + matrix.sum(axis=-2)
    return 100 * np.divide(2 * tp, denominators, out=np.zeros_like(tp, dtype=float), where=denominators > 0).mean(axis=-1)


def official_interval(truth, first, second):
    rng = np.random.default_rng(72191)
    first_conf = np.zeros((NBOOT, 4, 4))
    second_conf = np.zeros_like(first_conf)
    for label in range(4):
        take = truth == label
        counts = np.bincount(first[take] * 4 + second[take], minlength=16)
        samples = rng.multinomial(int(take.sum()), counts / take.sum(), size=NBOOT).reshape(NBOOT, 4, 4)
        first_conf[:, label] = samples.sum(axis=2)
        second_conf[:, label] = samples.sum(axis=1)
    return [rn(v) for v in np.percentile(f1(first_conf) - f1(second_conf), [2.5, 97.5])]


def cp_upper(lost, n, tail):
    return 100.0 if lost == n else 100 * beta.ppf(1 - tail, lost + 1, n - lost)


def analyze(path):
    raw = path.read_bytes()
    data = json.loads(raw)
    controls = data["compactMethods"]
    if set(controls) != {"softKD", "float13", "hard416", "binary4", "text4", "observed8", "local", "intact"}:
        raise ValueError("registered control missing")
    tail = .05 / (2 * len(controls) + 2)
    output = {"schema": "latentmesh.frontier-decision/1", "observationsSha256": hashlib.sha256(raw).hexdigest(),
              "lockSha256": data["lockSha256"], "simultaneousTail": tail, "domains": {}}
    for key in ("go", "tweet"):
        samples = data["domains"][key]
        truth, labels, native = digits(samples["truth"], "01"), digits(samples["labels"], "012"), digits(samples["native"], "012")
        if truth.sum() * 2 != len(truth) or len(native) != len(labels):
            raise ValueError("unbalanced or malformed evaluation population")
        methods, correct = {}, {}
        for name, case in samples["methods"].items():
            pos, override = digits(case["positive"], "01"), digits(case["override"], "01")
            if len(pos) != len(truth) or len(override) != len(labels):
                raise ValueError("sample length mismatch")
            if name == "intact" and (pos.any() or override.any()):
                raise ValueError("intact receiver cannot emit anger")
            correct[name] = (pos == truth).astype(np.int16)
            retained = (native == labels) & (override == 0)
            methods[name] = {"newCorrect": int(correct[name].sum()), "newAccuracy": rn(100 * correct[name].mean()),
                             "retainedCorrect": int(retained.sum()), "retainedAccuracy": rn(100 * retained.mean()),
                             "retainedLost": int(((native == labels) & (override == 1)).sum()),
                             "skillOverrides": int(pos.sum()), "retentionOverrides": int(override.sum())}
        strongest = max(methods[n]["newCorrect"] for n in controls)
        best = [n for n in controls if methods[n]["newCorrect"] == strongest]
        hybrid = methods["wsp48"]
        lower = {}
        intervals = {}
        for name in controls:
            dist = distribution(correct["wsp48"] - correct[name], truth)
            lower[name] = rn(np.quantile(dist, tail))
            if name in best:
                intervals[name] = [rn(v) for v in np.percentile(dist, [2.5, 97.5])]
        lost = hybrid["retainedLost"]
        upper = rn(cp_upper(lost, len(labels), tail))
        advantage = hybrid["newCorrect"] - strongest
        output["domains"][key] = {"skillRows": len(truth), "retentionRows": len(labels), "metrics": methods, "strongest": best,
            "advantagePp": rn(100 * advantage / len(truth)), "retainedLossPp": rn(100 * lost / len(labels)),
            "paired95VsBest": intervals, "simultaneousLowerPp": lower, "retentionUpperPp": upper,
            "passesPoint": bool(100 * advantage >= 2 * len(truth) and 100 * lost <= len(labels)),
            "passesUncertainty": bool(all(v > 2 for v in lower.values()) and upper < 1)}
    off = data["official"]
    truth = digits(off["truth"], "0123")
    original = digits(off["specialist"], "0123")
    if set(truth.tolist()) != {0, 1, 2, 3} or len(truth) != len(original):
        raise ValueError("official benchmark incomplete")
    scores = {}
    for name, encoded in off["methods"].items():
        p = digits(encoded, "0123")
        if len(p) != len(truth) or np.any((p != original) & (p != 0)):
            raise ValueError("invalid specialist override")
        scores[name] = confidence_metrics(truth, p)
    base = confidence_metrics(truth, original)
    assert scores["intact"] == base
    candidate = digits(off["methods"]["wsp48"], "0123")
    ci = official_interval(truth, candidate, original)
    output["official"] = {"rows": len(truth), "exactDevelopmentTextOverlaps": off["exactDevelopmentTextOverlaps"],
        "macroF1": scores, "intactMacroF1": base, "candidateVsIntactPp": rn(scores["wsp48"] - base), "paired95Pp": ci,
        "publishedTimeLMs2021": 80.2, "candidateVsPublishedPp": rn(scores["wsp48"] - 80.2),
        "clearsReference": bool(scores["wsp48"] > max(80.2, base) and ci[0] > 0)}
    output["advance"] = all(v["passesPoint"] for v in output["domains"].values())
    output["comparativeEvidence"] = all(v["passesUncertainty"] for v in output["domains"].values())
    return output


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("observations", type=Path)
    print(json.dumps(analyze(cli.parse_args().observations), indent=2, allow_nan=False))
