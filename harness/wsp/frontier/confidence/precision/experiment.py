"""Make distinct fit, selection and test dialogue features for all-emotion tagging."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import softmax

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from cohorts import (ARMS, ASSETS, MELD, Encoder, carer, dedup, file_hash, fresh_rows, json_write, load_heads,
                     meld, norm, raw_sources, sha)

ASSETS["meld-dev.csv"] = (f"https://raw.githubusercontent.com/declare-lab/MELD/{MELD}/data/MELD/dev_sent_emo.csv",
                          "2e89c6f8aa182d6f62f8c6331aece905ac7273ca4999660bfb5213e1d0370c1c")


def partitions(cache, phase):
    old = fresh_rows(cache, "fit")
    fitted = {r.group for role in ("skillFit", "guardFit") for r in old["adapted." + role]}
    tuned = {r.group for role in ("skillTune", "guardTune") for r in old["adapted." + role]}
    if fitted & tuned:
        raise ValueError("old fit and tune dialogues overlap")
    go, val, test, tweets = raw_sources(cache)
    seen = go | {norm(r[1]) for r in val + test + tweets}
    for split in ("train", "validation", "test"):
        seen.update(norm(r.text) for r in carer(cache, split))
    for split in ("dev", "test"):
        seen.update(norm(r.text) for r in meld(cache, split))
    all_rows = dedup(meld(cache, "train"), seen)
    if phase == "test":
        return {"test": [r for r in all_rows if r.group not in fitted | tuned]}
    pool = [r for r in all_rows if r.group in fitted]
    def first(positive, cap):
        choices = [r for r in pool if (r.skill == 1) == positive]
        return sorted(choices, key=lambda r: sha(("receiver-precision-v1/" + r.id).encode()))[:cap]
    return {"fit": first(True, 192) + first(False, 576), "tune": [r for r in all_rows if r.group in tuned]}


def extract(cache, output, phase):
    heads = HERE.parent / "heads"
    fixed = load_heads(heads)
    result = {"headsSha256": file_hash(heads / "index.json"), "arms": ARMS, "sets": {}}
    if phase == "test":
        result["lockSha256"] = file_hash(output / "lock.json")
        lock = json.loads((output / "lock.json").read_bytes())
        if lock["headsSha256"] != result["headsSha256"]:
            raise ValueError("incorrect source heads")
    encoder = Encoder(cache, "target")
    for role, rows in partitions(cache, phase).items():
        states, logits = encoder.extract(r.text for r in rows)
        scores = states @ fixed[:, :-1].T + fixed[:, -1]
        confidence = softmax(logits, axis=1).max(axis=1)
        groups = [r.group for r in rows]
        if phase == "test" and any(set(groups) & set(lock["sets"][k]["groupIDs"]) for k in ("fit", "tune")):
            raise ValueError("test dialogues used in development")
        result["sets"][role] = {"ids": sha(json.dumps([r.id for r in rows], separators=(",", ":")).encode()),
             "groups": groups, "truth": [int(r.skill == 1) for r in rows], "sentiment": [r.sentiment for r in rows],
             "native": logits.argmax(axis=1).tolist(), "confidence": confidence.tolist(), "scores": scores.tolist()}
        if phase == "test":
            json_write(output / "sample.json", [{"text": r.text, "logits": l.tolist(), "scores": scores[i].tolist(), "confidence": float(confidence[i])}
                        for i, (r, l) in enumerate(zip(rows[:64], logits[:64], strict=True))])
        print(role, len(rows), len(set(groups)), sum(r.skill == 1 for r in rows), flush=True)
    json_write(output / (phase + ".json"), result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "test"))
    parser.add_argument("cache", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    extract(args.cache, args.output, args.phase)
