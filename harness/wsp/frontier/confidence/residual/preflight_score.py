"""Replay average precision differences from compressed orders, tie groups and labels."""
import argparse
import base64
import json
from pathlib import Path
import zlib

import numpy as np


def encode(values):
    return base64.b64encode(zlib.compress(np.asarray(values, dtype="<u2").tobytes(),9)).decode()


def decode(field):
    return np.frombuffer(zlib.decompress(base64.b64decode(field,validate=True)),dtype="<u2").astype(np.int64)


def ap(truth, order, ends, weight):
    w = weight[order]
    tp = np.cumsum(w*truth[order])[ends]
    total = np.cumsum(w)[ends]
    precision = np.divide(tp,total,out=np.zeros(len(tp)),where=total>0)
    return float(100 * (np.diff(tp,prepend=0) @ precision) / tp[-1]) if tp[-1] else np.nan


def analyze(evidence):
    y, groups = decode(evidence["truth"]), decode(evidence["groups"])
    n = len(y)
    if (evidence["schema"] != "latentmesh.full-sender-screen/1" or
            evidence["sourceSha256"] != "9f96b2966d2d83c4a2fe53bcd349c7bccd65f82ecf42d92570be38b17e6c73ea" or
            evidence["idsSha256"] != "f85ad492c9f3f9bbabc2ecffa66a1a3d0fef565421afd4e09efd20cc9f5ce11c" or
            n != 5617 or len(groups) != n or y.sum() != 53 or y.max() != 1 or
            not np.array_equal(np.unique(groups),np.arange(884))):
        raise ValueError("invalid validation records")
    rank = {k:(decode(v["order"]),decode(v["ends"])) for k,v in evidence["ranks"].items()}
    for order, ends in rank.values():
        if not np.array_equal(np.sort(order), np.arange(n)) or not np.all(np.diff(ends)>0) or ends[-1] != n-1:
            raise ValueError("incorrect ordering or tie blocks")
    def scores(w):
        return [ap(y,*rank[k],w) for k in ("local","fullSender")]
    a,b = scores(np.ones(n,dtype=np.int64))
    rng = np.random.default_rng(11202026)
    delta = []
    for weights in rng.multinomial(884, np.full(884,1/884),size=2000):
        old,new = scores(weights[groups])
        delta.append(new-old)
    if not np.isfinite(delta).all():
        raise ValueError("undefined bootstrap AP")
    interval = np.quantile(delta,[.025,.975])
    return {"cohort":"DailyDialog validation, exploratory", "rows":n, "angry":int(y.sum()), "dialogues":884,
            "localAP":round(a,6), "fullSenderAP":round(b,6), "gain":round(b-a,6), "paired95":interval.round(6).tolist(),
            "partialCorrelation":round(evidence["partialCorrelation"],6), "passesScreen":bool(interval[0]>0)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder",type=Path)
    folder = parser.parse_args().folder
    result = analyze(json.loads((folder/"preflight-ranks.json").read_bytes()))
    if result != json.loads((folder/"preflight.json").read_bytes()):
        raise ValueError("development screen replay differs")
    print(result)
