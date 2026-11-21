"""Replay paired decisions and the transfer gate without checkpoints or network access."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from frozen import ARMS, GOLD_SHA, HERE, LOCK_SHA, MODES, PROGRAMS, TEST_SHA, locked
sys.path.insert(0, str(HERE.parent / "dailydialog"))
from report import ci, load_archive, performance, rn, vector

METHODS = tuple(arm+"."+mode for arm in ARMS for mode in MODES)


def parity(x):
    p = x["parity"]
    e = x["exclusions"]
    if (x["schema"] != "latentmesh.residual-observations/1" or x["lockSha256"] != LOCK_SHA or
            x["rawSha256"] != TEST_SHA or x["goldSha256"] != GOLD_SHA or
            set(x["predictions"]) != set(METHODS) or p["checked"] != 2755 or p["uniquePrevious"] < 1 or
            any(len(vector(a, "01")) != 2755 for a in [x["truth"],x["clean"],*x["predictions"].values()]) or
            x["truth"].count("1") != 150 or x["clean"].count("0") != sum(e.values()) or
            set(x["graphs"]) != set(PROGRAMS)):
        raise ValueError("incorrect experiment fingerprint, predictions or population")
    checks = [p["instrumentedNativeDifferences"] == 0, p["maxScoreError"] < 1e-9, p["maxStateError"] < 1e-9]
    for arm in PROGRAMS:
        g = x["graphs"][arm]
        if (g["lockSha256"] != LOCK_SHA or g["baseSha256"] != "046f7e4cc46b399558fa9b2de966f6b0d42c69e4b01f582bfb4099b01a0bb5d7" or
                g["stateBytes"] != 64 or g["addedNodes"] != 13 or g["addedBytes"] <= 0):
            raise ValueError("unexpected graph configuration")
        checks.extend((p["nativeDifferences"][arm] == 0, p["tagDifferences"][arm] == 0))
    return all(checks)


def population(x, mask, exact):
    y = vector(x["truth"], "01")[mask].astype(np.int64)
    z = np.column_stack([vector(x["predictions"][arm], "01")[mask] for arm in METHODS]).astype(np.int64)
    patterns, counts = np.unique(np.column_stack((y, z)), axis=0, return_counts=True)
    truth, flags = patterns[:, 0], patterns[:, 1:]
    draws = np.random.default_rng(62017).multinomial(len(y), counts / len(y), size=10000)
    boot_p = (draws @ truth)[:, None]
    boot_tp = draws @ (flags * truth[:, None])
    boot = performance(boot_tp, draws @ flags - boot_tp, boot_p)
    points = {}
    for i, arm in enumerate(METHODS):
        tp = int(y @ z[:, i])
        fp = int(z[:, i].sum()) - tp
        scores = performance(tp, fp, int(y.sum()))
        points[arm] = {"tp":tp, "fp":fp, "fn":int(y.sum())-tp,
                       **{k:rn(v) if np.isfinite(v) else None for k,v in scores.items()}}
    paired = {}
    for mode in MODES:
        j = METHODS.index("transfer."+mode)
        for arm in ARMS[1:]:
            i = METHODS.index(arm+"."+mode)
            paired[arm+"."+mode] = {"gain":rn(points[METHODS[j]]["f05"] - points[METHODS[i]]["f05"]),
                                     "ci95":ci(boot["f05"][:,j] - boot["f05"][:,i]),
                                     "disagreements":int(np.count_nonzero(z[:,i] != z[:,j]))}
    m = points["transfer.majority"]
    def positive_lower(arm):
        c = paired[arm+".majority"]["ci95"]
        return c is not None and c[0] > 0
    gate = {"precision":m["tp"] > 0 and 2*m["tp"] >= m["tp"]+m["fp"], "recall":5*m["tp"] >= int(y.sum()),
            "beatsMatchedLocal":positive_lower("local"), "exactNativeAndReadout":exact}
    return {"conversations":len(y), "angry":int(y.sum()), "methods":points, "paired":paired, "gate":gate,
            "passes":all(gate.values()), "senderAttribution":positive_lower("shuffled"),
            "replacesIncumbent":all(gate.values()) and positive_lower("shuffled") and positive_lower("incumbent.local")}


def report(folder, verify):
    locked()
    raw = load_archive(folder / "observations.archive.json")
    x = json.loads(raw)
    exact = parity(x)
    result = {"observationsSha256":hashlib.sha256(raw).hexdigest(), "lockSha256":LOCK_SHA, "parity":x["parity"],
              "graphs":x["graphs"], "exclusions":x["exclusions"], "nativeSha256":x["nativeSha256"],
              "clean":population(x, vector(x["clean"],"01").astype(bool), exact),
              "official":population(x, np.ones(2755, dtype=bool), exact)}
    path = folder / "decision.json"
    if verify:
        if result != json.loads(path.read_bytes()):
            raise ValueError("replayed decision differs from archive")
    else:
        path.write_text(json.dumps(result, allow_nan=False, separators=(",",":"))+"\n")
    return result, x


def compare(folders):
    (a, left), (b, right) = [report(f, True) for f in folders]
    for key in ("truth", "clean", "graphs", "exclusions"):
        if left[key] != right[key]:
            raise ValueError("hosts differ in " + key)
    disagreement = {k:sum(x != y for x,y in zip(left["predictions"][k], right["predictions"][k], strict=True)) for k in METHODS}
    print(json.dumps({"crossHostTags":disagreement,"identicalNativeDigest":a["nativeSha256"]==b["nativeSha256"],
                      "passes":[a["clean"]["passes"], b["clean"]["passes"]]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    for name in ("score", "verify"):
        subs.add_parser(name).add_argument("folder", type=Path)
    subs.add_parser("compare").add_argument("folders", nargs=2, type=Path)
    a = parser.parse_args()
    if a.action == "compare":
        compare(a.folders)
    else:
        result, _ = report(a.folder, a.action == "verify")
        print("Clean conversations", result["clean"]["conversations"], "passed", result["clean"]["passes"])
