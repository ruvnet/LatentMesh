"""Recompute whole-dialogue uncertainty and verify the hash bound, published observations."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys
import zlib

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "precision"))
from assess import DRAWS, SEED, interval, performance, rn, vector

ARMS = ("frozen.scoreOnly", "frozen.float13", "frozen.local", "frozen.wsp48", "frozen.softKD",
        "transfer.majority", "local.majority", "pca.majority", "transfer.f05", "local.f05", "pca.f05")
PAIRS = {"frozen.scoreOnly": ("frozen.local", "frozen.wsp48", "frozen.softKD"),
         "transfer.majority": ("local.majority", "pca.majority", "frozen.scoreOnly"),
         "transfer.f05": ("local.f05", "pca.f05")}
METRICS = ("precision", "recall", "f05")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def id_hash(ids):
    return digest(json.dumps(ids, separators=(",", ":")).encode())


def verify_observations(x, graph):
    if (x["schema"] != "latentmesh.dailydialog-observations/1" or
            x["datasetSha256"] != "2de93416a95060057d8b95ef9e2ab9b17a317906f1e352414001a1887f081193" or
            x["lockSha256"] != digest((HERE / "lock.json").read_bytes()) or
            x["frozenGraphSha256"] != "e3d8c7b05aab5578e0e42fba5a1835678e2deff78fd998347097e6a277f7c815" or
            x["adaptedGraphSha256"] != graph["modelSha256"] or graph["lockSha256"] != x["lockSha256"]):
        raise ValueError("wrong dataset, lock or installed graph")
    n = len(x["groups"])
    if (set(x["predictions"]) != set(ARMS) or any(len(v) != n for v in x["predictions"].values()) or
            not all(len(x[k]) == n for k in ("truth", "isClean", "native")) or
            len(vector(x["native"], "012")) != n or not all(isinstance(g,int) and g >= 0 for g in x["groups"]) or
            sorted(x["groups"]) != x["groups"]):
        raise ValueError("missing observations or invalid dialogue order")
    ids = []
    previous = None
    for g in x["groups"]:
        turn = turn + 1 if g == previous else 0
        ids.append(f"test/{g}/{turn}")
        previous = g
    if (id_hash(ids) != x["official"]["idsSha256"] or
            id_hash([v for v,flag in zip(ids, x["isClean"], strict=True) if flag == "1"]) != x["clean"]["idsSha256"]):
        raise ValueError("inconsistent population IDs")
    p = x["parity"]
    if p["checkedTurns"] != n or any(not np.isfinite(v) or v < 0 for v in p.values()):
        raise ValueError("invalid parity measurements")
    for field in ("frozenTagErrors", "adaptedMajorityErrors", "adaptedF05Errors"):
        if not isinstance(p[field], int) or p[field] > n:
            raise ValueError("invalid tag mismatch count")
    e = x["exclusions"]
    if (e["alignedTurns"] != n or e["rawTurns"] < n or
            n - e["priorTextOverlaps"] - e["repeatOrEmpty"] != x["clean"]["rows"]):
        raise ValueError("inconsistent exclusion counts")


def ci(values):
    usable = values[np.isfinite(values)]
    return interval(usable) if len(usable) >= .99 * DRAWS else None


def panel(x, name):
    truth = vector(x["truth"], "01").astype(np.int64)
    mask = vector(x["isClean"], "01").astype(bool) if name == "clean" else np.ones(len(truth), dtype=bool)
    truth = truth[mask]
    labels, groups = np.unique(np.asarray(x["groups"])[mask], return_inverse=True)
    n, pos = len(labels), int(truth.sum())
    if pos == 0 or not (len(truth) == x[name]["rows"] and pos == x[name]["positive"] and n == x[name]["dialogues"]):
        raise ValueError("inconsistent population labels or groups")
    # One draw of whole dialogues supplies every method and every paired difference.
    # Store group sufficient statistics and process the draw matrix in small blocks.
    stats = [np.bincount(groups, weights=truth, minlength=n)]
    methods = {}
    for arm in ARMS:
        pred = vector(x["predictions"][arm], "01")[mask].astype(np.int64)
        tp = int((pred * truth).sum())
        flags = int(pred.sum())
        stats.extend([np.bincount(groups, weights=pred*truth, minlength=n), np.bincount(groups, weights=pred, minlength=n)])
        metrics = performance(tp, flags-tp, pos)
        methods[arm] = {k: rn(v) if np.isfinite(v) else None for k,v in metrics.items()}
        methods[arm].update(tp=tp, fp=flags-tp, fn=pos-tp, tags=flags)
    table = np.column_stack(stats)
    rng = np.random.default_rng(SEED)
    samples = []
    for first in range(0, DRAWS, 250):
        w = rng.multinomial(n, np.full(n,1/n), size=min(250,DRAWS-first))
        samples.append(w.astype(float) @ table)
    draws = np.vstack(samples)
    draws_positive = draws[:, :1]
    hits, flags = draws[:,1::2], draws[:,2::2]
    with np.errstate(invalid="ignore", divide="ignore"):
        bootstrap = performance(hits, flags-hits, draws_positive)
    for i, arm in enumerate(ARMS):
        methods[arm]["interval95"] = {k: ci(bootstrap[k][:,i]) for k in METRICS}
    comparisons = {}
    for candidate, comparators in PAIRS.items():
        comparisons[candidate] = {}
        for other in comparators:
            at, bt = ARMS.index(candidate), ARMS.index(other)
            fields = {}
            for k in METRICS:
                a, b = methods[candidate][k], methods[other][k]
                fields[k] = {"points": rn(a-b) if a is not None and b is not None else None,
                             "paired95": ci(bootstrap[k][:,at]-bootstrap[k][:,bt])}
            comparisons[candidate][other] = fields
    gates = {}
    for a in ("frozen.scoreOnly", "transfer.majority", "local.majority", "pca.majority"):
        m = methods[a]
        gates[a] = bool(m["tags"] and 2*m["tp"] >= m["tags"] and 5*m["tp"] >= pos)
    p = x["parity"]
    gates["frozen.scoreOnly"] &= (p["frozenDrift"] == 0 and p["frozenTagErrors"] == 0 and
                                  p["instrumentedDrift"] == 0 and p["maxFrozenEvidenceError"] <= 1e-8)
    gates["transfer.majority"] &= (p["adaptedDrift"] == 0 and p["adaptedMajorityErrors"] == 0 and
                                   p["instrumentedDrift"] == 0 and p["maxAdaptedEvidenceError"] <= 1e-8)
    lower = comparisons["transfer.majority"]["local.majority"]["f05"]["paired95"]
    return {**x[name], "prevalence": rn(100*pos/len(truth)), "methods": methods, "paired": comparisons,
            "utilityGate": gates, "transferBenefit": bool(lower and lower[0] > 0)}


def load_archive(path):
    archive = json.loads(path.read_bytes())
    raw = zlib.decompress(base64.b64decode(archive["data"], validate=True))
    if archive["encoding"] != "zlib+base64" or len(raw) != archive["bytes"] or digest(raw) != archive["sha256"]:
        raise ValueError("corrupted prediction archive")
    return raw


def run(mode, folder):
    raw = load_archive(folder / "observations.archive.json")
    x = json.loads(raw)
    graph = json.loads((folder / "graph.json").read_bytes())
    verify_observations(x, graph)
    result = {"observationsSha256": digest(raw), "lockSha256": x["lockSha256"], "parity": x["parity"],
              "exclusions": x["exclusions"], "panels": {name: panel(x, name) for name in ("clean","official")}}
    path = folder / "decision.json"
    if mode == "verify":
        if result != json.loads(path.read_bytes()):
            raise ValueError("decision differs from the paired predictions")
    else:
        path.write_text(json.dumps(result, allow_nan=False, separators=(",",":")) + "\n")
    p = result["panels"]["clean"]
    print(mode, p["rows"], "turns;", p["positive"], "angry;", p["utilityGate"], "transfer benefit:", p["transferBenefit"])


def compare(first, second):
    records = [json.loads(load_archive(d / "observations.archive.json")) for d in (first, second)]
    graphs = [json.loads((d / "graph.json").read_bytes()) for d in (first, second)]
    for rec, g in zip(records, graphs, strict=True):
        verify_observations(rec, g)
    if graphs[0] != graphs[1]:
        raise ValueError("compiled ONNX files differ")
    keys = ("datasetSha256", "lockSha256", "frozenGraphSha256", "adaptedGraphSha256",
            "official", "clean", "exclusions", "groups", "truth", "isClean")
    for k in keys:
        if records[0][k] != records[1][k]:
            raise ValueError("independent host cohorts or installed programs differ: " + k)
    controls = [json.loads((d / "controls.json").read_bytes()) for d in (first, second)]
    if controls[0]["validation"] != controls[1]["validation"] or controls[0]["validation"]["rows"] != 128:
        raise ValueError("control installation parity cohorts differ")
    for name in ("transfer", "local", "pca"):
        a, b = (r["methods"][name] for r in controls)
        if a["installation"] != b["installation"] or (name == "transfer" and a["installation"] != graphs[0]):
            raise ValueError("installation digest or graph receipt differs: " + name)
        for val in (a,b):
            if val["nativeDrift"] or val["tagDisagreements"] or not (0 <= val["maxScoreError"] <= 1e-8):
                raise ValueError("native logit or tag parity failure in control installation")
    for arm in ("frozen.scoreOnly", "transfer.majority"):
        a, b = (rec["predictions"][arm] for rec in records)
        print(arm, "cross host decisions differing on official test:", sum(x != y for x,y in zip(a,b,strict=True)))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="mode", required=True)
    for action in ("score", "verify"):
        commands.add_parser(action).add_argument("folder", type=Path)
    pair = commands.add_parser("compare")
    pair.add_argument("first", type=Path)
    pair.add_argument("second", type=Path)
    args = p.parse_args()
    if args.mode == "compare":
        compare(args.first, args.second)
    else:
        run(args.mode, args.folder)
