"""Recompute whole conversation intervals from compact paired decisions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "dailydialog"))
from report import DRAWS, SEED, ci, load_archive, performance, rn, vector

ARMS = ("context.transfer", "context.local", "current.transfer", "current.local", "reset.transfer")
MODES = ("majority", "f05")
METHODS = tuple(a+"."+m for a in ARMS for m in MODES)
FIELDS = ("precision","recall","f05")


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def valid(x):
    n = 5509
    if (x["schema"] != "latentmesh.context-observations/1" or x["lockSha256"] != file_sha(HERE/"lock.json")
            or x["datasetSha256"] != "926548fbcc83a8d1ac8e2329d1313724f35c51dae3d46b906dbbf7eb144e636e"
            or x["goldSha256"] != "7ae49a1577be91bf95a8ae71437eccce995a075d56e31c0a8858e1bad2cbe49a"
            or set(x["predictions"]) != set(METHODS) or len(x["nativeSha256"]) != 64):
        raise ValueError("wrong observation identity or method set")
    for s in [x["truth"], x["clean"], *x["predictions"].values()]:
        if len(s) != n or set(s) - {"0","1"}:
            raise ValueError("invalid observation vector")
    if any(type(v) != int or v < 0 for v in x["exclusions"].values()) or sum(x["exclusions"].values()) != n-x["clean"].count("1"):
        raise ValueError("invalid population exclusions")
    p = x["parity"]
    if any(not np.isfinite(z) or z<0 for z in p.values()) or p["checkedFinal"]!=n or p["checkedPrevious"]!=n:
        raise ValueError("incomplete parity record")
    for arm in ("transfer","local"):
        g = x["graphs"][arm]
        if (g["arm"] != arm or g["lockSha256"] != x["lockSha256"] or g["stateBytes"] != 64
                or g["baseSha256"] != "046f7e4cc46b399558fa9b2de966f6b0d42c69e4b01f582bfb4099b01a0bb5d7"
                or len(g["modelSha256"]) != 64 or min(g["addedBytes"],g["addedNodes"]) <= 0):
            raise ValueError("invalid graph receipt")
    return (all(p[k] == 0 for k in ("instrumentedDrift","localDrift","transferDrift","tagMismatches"))
            and p["stateMaxError"] < 1e-8 and p["scoreMaxError"] < 1e-8)


def panel(x, clean, parity):
    mask = vector(x["clean"],"01").astype(bool) if clean else np.ones(len(x["truth"]),dtype=bool)
    y = vector(x["truth"],"01")[mask]
    z = np.column_stack([vector(x["predictions"][k],"01")[mask] for k in METHODS])
    if not y.any():
        raise ValueError("no true positives in population")
    # Sampling these joint patterns is exactly sampling complete conversations with replacement.
    patterns, count = np.unique(np.column_stack((y,z)), axis=0, return_counts=True)
    py, flags = patterns[:,0], patterns[:,1:]
    rng = np.random.default_rng(SEED)
    draws = rng.multinomial(len(y),count/len(y),size=DRAWS)
    pred_pos = draws @ flags
    tp = draws @ (flags * py[:,None])
    total_pos = (draws @ py)[:,None]
    boot = performance(tp,pred_pos-tp,total_pos)
    points = {}
    for j,name in enumerate(METHODS):
        true = int(np.sum(z[:,j]*y,dtype=np.int64))
        false = int(z[:,j].sum())-true
        rates = performance(true,false,int(y.sum()))
        points[name] = {"tp":true,"fp":false,"fn":int(y.sum())-true,
                        **{k:rn(rates[k]) if np.isfinite(rates[k]) else None for k in FIELDS},
                        "interval95":{k:ci(boot[k][:,j]) for k in FIELDS}}
    pairs = {}
    for mode in MODES:
        name = "context.transfer."+mode
        ai = METHODS.index(name)
        for arm in ("context.local","current.transfer","reset.transfer"):
            key = arm+"."+mode
            bj = METHODS.index(key)
            plus = (z[:,ai]==1) & (z[:,bj]==0)
            minus = (z[:,ai]==0) & (z[:,bj]==1)
            pairs[name+"/"+key] = {"f05Gain":rn(points[name]["f05"]-points[key]["f05"]),
                                   "paired95":ci(boot["f05"][:,ai]-boot["f05"][:,bj]),
                                   "newTrue":int(y[plus].sum()),"newFalse":int((1-y[plus]).sum()),
                                   "lostTrue":int(y[minus].sum()),"removedFalse":int((1-y[minus]).sum())}
    m = points["context.transfer.majority"]
    lo = pairs["context.transfer.majority/context.local.majority"]["paired95"]
    gate = {"precision":2*m["tp"]>=m["tp"]+m["fp"] and m["tp"]+m["fp"]>0, "recall":5*m["tp"]>=int(y.sum()),
            "transferGain":bool(lo and lo[0]>0), "nativeAndReadout":bool(parity)}
    return {"conversations":len(y),"angry":int(y.sum()),"prevalence":rn(100*y.mean()),"methods":points,
            "paired":pairs,"criteria":gate,"passed":all(gate.values())}


def report(folder, verify):
    raw = load_archive(folder/"observations.archive.json")
    x = json.loads(raw)
    parity = valid(x)
    result = {"lockSha256":x["lockSha256"], "observationsSha256":hashlib.sha256(raw).hexdigest(),
              "exclusions":x["exclusions"], "parity":x["parity"], "graphs":x["graphs"],
              "clean":panel(x,True,parity), "official":panel(x,False,parity)}
    path = folder/"decision.json"
    if verify:
        if result != json.loads(path.read_bytes()):
            raise ValueError("scorecard differs from paired decisions")
    else:
        path.write_text(json.dumps(result, allow_nan=False, separators=(",",":"))+"\n")
    return result


def compare(paths):
    xs = [json.loads(load_archive(d/"observations.archive.json")) for d in paths]
    if not all(valid(x) for x in xs):
        raise ValueError("numerical parity failed")
    a,b = xs
    for k in ("truth","clean","graphs","datasetSha256","goldSha256","lockSha256"):
        if a[k]!=b[k]:
            raise ValueError("independent execution differs in " + k)
    return {k:sum(x!=y for x,y in zip(a["predictions"][k], b["predictions"][k],strict=True)) for k in METHODS}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="phase",required=True)
    for command in ("score","verify"):
        sub.add_parser(command).add_argument("folder",type=Path)
    sub.add_parser("compare").add_argument("folders",nargs=2,type=Path)
    a = cli.parse_args()
    if a.phase == "compare":
        print(json.dumps(compare(a.folders)))
    else:
        p = report(a.folder,a.phase=="verify")["clean"]
        print(a.phase,p["conversations"],p["angry"],p["criteria"])
