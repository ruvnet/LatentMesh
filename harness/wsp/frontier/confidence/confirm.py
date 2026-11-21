"""Confirm the fixed 13 float tag on unused MELD development dialogues."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import zlib

import numpy as np

from scorecard import Paired, interval, measure, rn, vector

HERE = Path(__file__).resolve().parent
ARMS_TO_COMPARE = ("float13", "wsp48", "softKD", "local")


def prepare(cache, output):
    from scipy.special import softmax
    from cohorts import ASSETS, MELD, Encoder, carer, dedup, json_write, load_heads, meld, norm, raw_sources, sha
    ASSETS["meld-dev.csv"] = (f"https://raw.githubusercontent.com/declare-lab/MELD/{MELD}/data/MELD/dev_sent_emo.csv",
                              "2e89c6f8aa182d6f62f8c6331aece905ac7273ca4999660bfb5213e1d0370c1c")
    go, valid, emo, sentiment = raw_sources(cache)
    seen = go | {norm(r[1]) for r in valid + emo + sentiment}
    for split in ("train", "validation", "test"):
        seen.update(norm(r.text) for r in carer(cache, split))
    for split in ("train", "test"):
        seen.update(norm(r.text) for r in meld(cache, split))
    raw = meld(cache, "dev")
    rows = dedup(raw, seen)
    lock_raw = (HERE / "lock.json").read_bytes()
    lock = json.loads(lock_raw)
    heads = load_heads(HERE / "heads")
    states, logits = Encoder(cache, "target").extract(r.text for r in rows)
    scores = states @ heads[:, :-1].T + heads[:, -1]
    confidence = softmax(logits, axis=1).max(axis=1)
    result = {"lockSha256": sha(lock_raw), "ids": sha(json.dumps([r.id for r in rows], separators=(",", ":")).encode()),
        "rawRows": len(raw), "skill": "".join("x" if r.skill == -1 else str(r.skill) for r in rows),
        "sentiment": "".join(str(r.sentiment) for r in rows), "groups": [r.group for r in rows],
        "native": "".join(map(str, logits.argmax(axis=1))), "predictions": {}}
    for arm in ARMS_TO_COMPARE:
        col = lock["arms"].index(arm)
        a, b, d = lock["policies"]["transfer"][arm]["separate"]["coefficients"]
        result["predictions"][arm] = "".join(map(str, (a * scores[:, col] + b * confidence + d > 0).astype(int)))
    json_write(output / "confirm.json", result)


def analyze(raw):
    x = json.loads(raw)
    if x["lockSha256"] != hashlib.sha256((HERE / "lock.json").read_bytes()).hexdigest() or set(x["predictions"]) != set(ARMS_TO_COMPARE):
        raise ValueError("confirmation input differs from frozen methods")
    skill, sentiment, native = vector(x["skill"], "01x"), vector(x["sentiment"], "012"), vector(x["native"], "012")
    pred = {k: vector(v, "01") for k, v in x["predictions"].items()}
    if len({len(v) for v in [native, skill, sentiment, x["groups"], *pred.values()]}) != 1:
        raise ValueError("inconsistent lengths")
    sampler = Paired(skill, sentiment, x["groups"], True)
    metrics = {k: measure(v, skill, sentiment, native, True) for k,v in pred.items()}
    return {"observationsSha256": hashlib.sha256(raw).hexdigest(), "ids": x["ids"], "rawRows": x["rawRows"], "rows": len(skill),
            "dialogues": len(set(x["groups"])), "anger": int((skill == 1).sum()), "sadness": int((skill == 0).sum()),
            "nativeCorrect": int((native == sentiment).sum()), "metrics": metrics,
            "float13VsWsp48Pp": rn(metrics["float13"]["balancedAccuracy"] - metrics["wsp48"]["balancedAccuracy"]),
            "paired95Pp": interval(sampler.difference(pred["float13"], pred["wsp48"]))}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="mode", required=True)
    run = sub.add_parser("run")
    run.add_argument("cache", type=Path)
    run.add_argument("folder", type=Path)
    sub.add_parser("verify").add_argument("folder", type=Path)
    args = cli.parse_args()
    if args.mode == "run":
        import sys
        sys.path.insert(0, str(HERE.parent))
        from evidence import archive
        args.folder.mkdir(parents=True, exist_ok=True)
        prepare(args.cache, args.folder)
        record = args.folder / "confirm.json"
        archive(record, args.folder / "confirm.archive.json")
        result = analyze(record.read_bytes())
        (args.folder / "confirmation.json").write_text(json.dumps(result, indent=2) + "\n")
    else:
        a = json.loads((args.folder / "confirm.archive.json").read_bytes())
        raw = zlib.decompress(base64.b64decode(a["data"], validate=True))
        if len(raw) != a["bytes"] or hashlib.sha256(raw).hexdigest() != a["sha256"]:
            raise ValueError("confirmation archive mismatch")
        result = analyze(raw)
        if result != json.loads((args.folder / "confirmation.json").read_bytes()):
            raise ValueError("confirmation scores differ")
    print("Confirmation:", result["rows"], "rows, 13 floats vs 48 int8:", result["float13VsWsp48Pp"], result["paired95Pp"])
