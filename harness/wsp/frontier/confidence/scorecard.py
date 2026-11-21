"""Replay receiver decisions and paired uncertainty using public categorical observations."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import zlib

import numpy as np

MODES = ("protected", "tag", "separate")
CONTROLS = ("softKD", "float13", "hard416", "binary4", "text4", "observed8", "local", "scoreOnly", "confidenceOnly", "intact")
DRAWS, SEED = 10000, 51873
TAIL = .05 / (len(CONTROLS) + 1)


def rn(x):
    return round(float(x), 5)


def vector(s, alphabet):
    if not s or set(s) - set(alphabet):
        raise ValueError("unexpected categorical value")
    return np.fromiter((-1 if c == "x" else int(c) for c in s), dtype=np.int8)


class Paired:
    def __init__(self, skills, native_gold, groups, clustered):
        self.skill = skills
        self.sentiment = native_gold
        self.clustered = clustered
        if not clustered:
            return
        _, self.inverse = np.unique(groups, return_inverse=True)
        n = len(np.unique(groups))
        self.draw = np.random.default_rng(SEED).multinomial(n, np.full(n, 1/n), size=DRAWS).astype(float)
        self.den = [self.total(skills == k) for k in (0, 1)]
        self.sent_den = self.total(native_gold >= 0)
        if any(np.any(d == 0) for d in [*self.den, self.sent_den]):
            raise ValueError("empty stratum in dialogue resampling")

    def total(self, values):
        return self.draw @ np.bincount(self.inverse, weights=values, minlength=self.draw.shape[1])

    def difference(self, first, second):
        d = (first == self.skill).astype(np.int8) - (second == self.skill).astype(np.int8)
        if self.clustered:
            return 50 * sum(self.total(d * (self.skill == k)) / self.den[k] for k in (0, 1))
        rng = np.random.default_rng(SEED)
        out = np.zeros(DRAWS)
        for k in (0, 1):
            sample = d[self.skill == k]
            probs = np.bincount(sample + 1, minlength=3) / len(sample)
            draws = rng.multinomial(len(sample), probs, size=DRAWS)
            out += 50 * (draws[:, 2] - draws[:, 0]) / len(sample)
        return out

    def loss(self, displaced):
        if self.clustered:
            return 100 * self.total(displaced) / self.sent_den
        n = int(np.sum(self.sentiment >= 0))
        rng = np.random.default_rng(SEED)
        out = np.zeros(DRAWS)
        for k in (0, 1, 2):
            take = self.sentiment == k
            out += rng.binomial(int(take.sum()), displaced[take].mean(), size=DRAWS)
        return 100 * out / n


def interval(draw):
    return [rn(x) for x in np.quantile(draw, [.025, .975])]


def measure(pred, skill, sentiment, native, meld):
    p, n = skill == 1, skill == 0
    keep = sentiment >= 0
    native_ok = keep & (native == sentiment)
    tp, fp = int(np.sum(pred & p)), int(np.sum(pred & n))
    lost = int(np.sum(pred & native_ok))
    m = {"tp": tp, "fn": int(p.sum())-tp, "fpVsSadness": fp, "tn": int(n.sum())-fp,
         "balancedAccuracy": rn(50 * (tp/p.sum() + 1 - fp/n.sum())), "angerRecall": rn(100 * tp/p.sum()),
         "lostIfExclusive": lost, "lossIfExclusivePp": rn(100 * lost / keep.sum())}
    if meld:
        count = int(pred.sum())
        m["allTagged"] = count
        m["allUtterancePrecision"] = rn(100 * tp / count) if count else None
    return m


def panel(key, x):
    meld = key != "social"
    skill, sentiment = vector(x["skill"], "01x"), vector(x["sentiment"], "012x")
    native = vector(x["native"], "012")
    groups = np.asarray(x["groups"])
    if len({len(skill), len(sentiment), len(native), len(groups)}) != 1 or not all((skill == k).any() for k in (0,1)):
        raise ValueError("invalid evaluation populations")
    if not (np.all(sentiment >= 0) if meld else np.all((skill >= 0) != (sentiment >= 0))):
        raise ValueError("invalid joint gold labels")
    sampler = Paired(skill, sentiment, groups, meld)
    methods = set(CONTROLS) | {"wsp48", "priorSource", "priorProtected"}
    if set(x["methods"]) != methods:
        raise ValueError("registered arms differ")
    native_ok = (sentiment >= 0) & (native == sentiment)
    nr = int((sentiment >= 0).sum())
    out = {"ids": x["ids"], "total": len(native), "anger": int(np.sum(skill == 1)), "sadness": int(np.sum(skill == 0)),
           "protected": nr, "nativeCorrect": int(native_ok.sum()), "nativeAccuracy": rn(100 * native_ok.sum()/nr),
           "dialogues": len(np.unique(groups)) if meld else None, "modes": {}}
    if meld:
        pos = skill == 1
        budget = nr // 100
        native_correct_angry = int(np.sum(native_ok & pos))
        max_true = int(pos.sum() - native_correct_angry + min(native_correct_angry, budget))
        out["exclusiveOracle"] = {"allowedLosses": budget, "nativeCorrectAngry": native_correct_angry, "maxTrueAnger": max_true,
             "maxAngerRecall": rn(100 * max_true/pos.sum()), "maxBalancedAccuracy": rn(50 * (1 + max_true/pos.sum()))}
    for mode in MODES:
        pred = {m: vector(v[mode], "01") for m,v in x["methods"].items()}
        if any(len(v) != len(native) for v in pred.values()) or pred["intact"].any():
            raise ValueError("invalid prediction vector")
        metrics = {m: measure(v, skill, sentiment, native, meld) for m,v in pred.items()}
        strongest = max(metrics[m]["balancedAccuracy"] for m in CONTROLS)
        best = [m for m in CONTROLS if metrics[m]["balancedAccuracy"] == strongest]
        refs = list(CONTROLS) if mode == "protected" else list(dict.fromkeys([*best, "float13", "scoreOnly", "softKD", "priorProtected", "priorSource"]))
        contrasts = {}
        for m in refs:
            draw = sampler.difference(pred["wsp48"], pred[m])
            contrasts[m] = {"differencePp": rn(metrics["wsp48"]["balancedAccuracy"] - metrics[m]["balancedAccuracy"]),
                            "paired95Pp": interval(draw), "simultaneousLowerPp": rn(np.quantile(draw, TAIL))}
        upper = rn(np.quantile(sampler.loss(pred["wsp48"] & native_ok), 1 - TAIL))
        out["modes"][mode] = {"metrics": metrics, "best": best, "contrasts": contrasts, "exclusiveLossUpperPp": upper}
    p = out["modes"]["protected"]
    out["pointGate"] = bool(all(p["contrasts"][m]["differencePp"] >= 2 for m in CONTROLS) and p["metrics"]["wsp48"]["lossIfExclusivePp"] <= 1)
    out["confidenceGate"] = bool(out["pointGate"] and all(p["contrasts"][m]["simultaneousLowerPp"] > 2 for m in CONTROLS) and p["exclusiveLossUpperPp"] < 1)
    return out


def analyze(raw, lock_path):
    d = json.loads(raw)
    if d["schema"] != "latentmesh.confidence-observations/1" or d["lockSha256"] != hashlib.sha256(lock_path.read_bytes()).hexdigest():
        raise ValueError("wrong calibration lock")
    if set(d["panels"]) != {"unseen", "matched", "social"}:
        raise ValueError("registered panel missing")
    for field in ("ids", "skill", "sentiment", "native", "groups"):
        if d["panels"]["unseen"][field] != d["panels"]["matched"][field]:
            raise ValueError("dialogue panels are not paired")
    panels = {k: panel(k, v) for k,v in d["panels"].items()}
    return {"schema": "latentmesh.confidence-scorecard/1", "observationsSha256": hashlib.sha256(raw).hexdigest(),
            "lockSha256": d["lockSha256"], "panels": panels, "advance": panels["unseen"]["confidenceGate"]}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action", choices=("score", "verify"))
    cli.add_argument("folder", type=Path)
    cli.add_argument("lock", type=Path)
    args = cli.parse_args()
    if args.action == "verify":
        archive = json.loads((args.folder / "observations.archive.json").read_bytes())
        raw = zlib.decompress(base64.b64decode(archive["data"], validate=True))
        if len(raw) != archive["bytes"] or hashlib.sha256(raw).hexdigest() != archive["sha256"]:
            raise ValueError("incorrect observation archive")
    else:
        raw = (args.folder / "observations.json").read_bytes()
    out = analyze(raw, args.lock)
    if args.action == "verify":
        if out != json.loads((args.folder / "decision.json").read_bytes()):
            raise ValueError("scorecard mismatch")
        print("Verified", args.folder.name, "advance", out["advance"])
    else:
        print(json.dumps(out, allow_nan=False, separators=(",",":")))
