"""Diagnose the best possible single cutoff on locked scores. Never return a deployable cutoff."""
import argparse
import json
from pathlib import Path

import numpy as np

from report import digest, load_archive


def positive_blocks(scores, labels):
    order = np.argsort(-scores, kind="stable")
    ranked, y = scores[order], labels[order]
    if not np.isfinite(ranked).all():
        raise ValueError("invalid ranking")
    ends = np.flatnonzero(np.r_[ranked[1:] != ranked[:-1], True])
    counts = y.cumsum()[ends]
    added = np.diff(counts, prepend=0)
    return [[int(end+1), int(hits)] for end,hits,delta in zip(ends, counts, added, strict=True) if delta]


def summarize(blocks, n, positive):
    flags = hits = 0
    ap = 0.
    required = (positive + 4)//5
    best = None
    largest = 0
    for count,tp in blocks:
        if not (flags < count <= n and hits < tp <= positive and tp-hits <= count-flags):
            raise ValueError("corrupted ranked positive blocks")
        ap += (tp-hits) * tp / count
        if tp >= required and (best is None or tp*best[0] > best[1]*count):
            best = (count,tp)
        if 2*tp >= count:
            largest = tp
        flags, hits = count, tp
    if not best or hits != positive:
        raise ValueError("missing ranked positives")
    count,tp = best
    return {"averagePrecision": round(100*ap/positive,5), "requiredHits": required,
            "maxPrecisionAt20Recall": round(100*tp/count,5), "trueAtBestPrecision": tp,
            "falseAtBestPrecision": count-tp, "maxRecallAt50Precision": round(100*largest/positive,5),
            "someCutoffPasses": bool(2*tp >= count)}


def verify(folder):
    stored = json.loads((folder/"ranking.json").read_bytes())
    archive = load_archive(folder/"observations.archive.json")
    obs = json.loads(archive)
    if (stored["observationsSha256"] != digest(archive) or stored["population"] != obs["clean"] or
            set(stored["methods"]) != {"frozen.scoreOnly","transfer","local","pca"}):
        raise ValueError("ranking uses a different population or calibration")
    for method in stored["methods"].values():
        if summarize(method["positiveBlocks"], obs["clean"]["rows"], obs["clean"]["positive"]) != method["summary"]:
            raise ValueError("ranking summary differs")
    return stored


def run(cache, folder):
    from corpus import HERE, describe, history, partition
    from cohorts import ARMS, Encoder, load_heads
    from install_receiver import committed, readout
    from install import coefficients
    raw = load_archive(folder/"observations.archive.json")
    obs = json.loads(raw)
    seen = history(cache)
    for part in ("train", "validation"):
        partition(cache, part, seen)
    _, clean, _ = partition(cache, "test", seen)
    if describe(clean) != obs["clean"]:
        raise ValueError("ranking clean cohort differs from scored trial")
    mask = np.fromiter((v == "1" for v in obs["isClean"]), dtype=bool)
    truth = np.asarray([r.anger for r in clean],dtype=int)
    if any(a != (b == "1") for a,b in zip(truth, np.asarray(list(obs["truth"]))[mask], strict=True)):
        raise ValueError("gold labels differ")
    states, _ = Encoder(cache,"target").extract(r.text for r in clean)
    frozen = load_heads(HERE.parent / "heads")[ARMS.index("float13")]
    a, _, c = coefficients("all","f05","scoreOnly")
    scores = {"frozen.scoreOnly": (states @ frozen[:-1] + frozen[-1])*a+c}
    def confirm(arm, pred):
        truth_bits = np.asarray(list(obs["predictions"][arm]))[mask] == "1"
        if not np.array_equal(truth_bits, pred):
            raise ValueError("ranking differs from the locked operating point: " + arm)
    confirm("frozen.scoreOnly", scores["frozen.scoreOnly"] > 0)
    for name, policy in committed()["methods"].items():
        w = readout(policy)
        scores[name] = states @ w[:-1] + w[-1]
        for mode in ("majority","f05"):
            cut = policy["points"][mode]["threshold"]
            confirm(name+"."+mode, scores[name] > cut if cut is not None else np.zeros(len(clean),dtype=bool))
    result = {"observationsSha256": digest(raw), "population": obs["clean"], "methods": {}}
    for name, values in scores.items():
        blocks = positive_blocks(values,truth)
        result["methods"][name] = {"positiveBlocks": blocks, "summary": summarize(blocks,len(clean),int(truth.sum()))}
    (folder/"ranking.json").write_text(json.dumps(result,separators=(",",":")) + "\n")
    return verify(folder)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action",required=True)
    new = sub.add_parser("run")
    new.add_argument("cache",type=Path)
    new.add_argument("folder",type=Path)
    sub.add_parser("verify").add_argument("folder",type=Path)
    args = p.parse_args()
    report = run(args.cache,args.folder) if args.action == "run" else verify(args.folder)
    for arm,r in report["methods"].items(): print(arm,r["summary"])
