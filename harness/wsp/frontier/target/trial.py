"""Calibrate and independently test the registered protection rule on frozen receiver readouts."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

PARENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PARENT))
from data import (ASSETS, Encoder, NEG, balanced, compact, ensure, file_hash, groups, json_write, norm,
                  packed, partitions, project, sha, split_digest, tweet_rows)

ASSETS["carer-test.parquet"] = (
    "https://huggingface.co/datasets/dair-ai/emotion/resolve/cab853a1dbdf4c42c2b3ef2173804746df8825fe/split/test-00000-of-00001.parquet",
    "6f8407fa1ca9c310f55781f082ed73812f6551e8dda2c61973123a121869245b")


def ranked(rows):
    return sorted(rows, key=lambda row: sha(("latentmesh-target-v1/" + row[0]).encode()))


def unique(rows, seen):
    result = []
    for row in rows:
        key = norm(row[1])
        if key not in seen:
            result.append(row)
        seen.add(key)
    return result


def raw_sources(cache):
    go = {norm(line.split("\t")[0]) for split in ("train", "dev", "test")
          for line in ensure(cache, split + ".tsv").read_text().splitlines()}
    valid, test = (tweet_rows(cache, "emotion", s) for s in ("val", "test"))
    return go, valid, test, tweet_rows(cache, "sentiment", "test")


def protected_split(cache, prior_ids):
    go, valid, emotion, sentiment = raw_sources(cache)
    seen = go | {norm(r[1]) for r in valid}
    unique(emotion, seen)
    eligible = unique(sentiment, seen)
    prior = groups(eligible, [{0}, {1}, {2}], 300)
    if split_digest(prior) != prior_ids:
        raise ValueError("previous protected population could not be reproduced")
    used = {r[0] for r in prior}
    ordered = [ranked([r for r in eligible if r[2] == label and r[0] not in used]) for label in range(3)]
    if min(map(len, ordered)) < 800:
        raise ValueError("insufficient unused target examples")
    return [r for block in ordered for r in block[:200]], [r for block in ordered for r in block[200:800]]


def balanced_new(rows, positive, negative, cap):
    a, b = (ranked([r for r in rows if r[2] in group]) for group in (positive, negative))
    n = min(len(a), len(b), cap)
    return [r for pair in zip(a[:n], b[:n]) for r in pair]


def skill_split(cache):
    import duckdb
    dev = partitions(cache, False)["dev"]
    used = {r[0] for r in balanced(dev, {"anger"}, NEG, 32)}
    go_rows = balanced_new([r for r in dev if r[0] not in used], {"anger"}, NEG, 200)
    go, val, emotion, sentiment = raw_sources(cache)
    seen = go | {norm(r[1]) for r in val + emotion + sentiment}
    with duckdb.connect(":memory:") as db:
        raw = db.execute("select text, label from read_parquet(?)", [str(ensure(cache, "carer-test.parquet"))]).fetchall()
    carer = unique([(f"carer/test/{i}", text, int(label)) for i, (text, label) in enumerate(raw)], seen)
    return {"go": go_rows, "carer": balanced_new(carer, {3}, {0}, 256)}


def encode(rows, encoder, basis):
    hidden, logits = encoder.extract(r[1] for r in rows)
    return project(hidden, basis), logits.argmax(axis=1), np.asarray([r[2] for r in rows])


def values(design, params):
    coef = params["coef"]
    return design[:, :len(coef)-1] @ coef[:-1] + coef[-1]


def route(scores, native, policy):
    eligible = native == 0 if policy["negativeOnly"] else np.ones(len(scores), dtype=bool)
    cut = policy["cut"]
    if cut == "never":
        return np.zeros(len(scores), dtype=bool)
    return eligible if cut == "always" else eligible & (scores > cut)


def init(run):
    lock = json.loads((run / "lock.json").read_bytes())
    if file_hash(run / "patches.json") != lock["patchesSha256"] or file_hash(run / "public.json") != lock["publicSha256"]:
        raise ValueError("source weights or setup differ from the preceding run")
    patches = json.loads((run / "patches.json").read_bytes())
    public = json.loads((run / "public.json").read_bytes())
    with np.load(run / "basis.npz", allow_pickle=False) as f:
        basis = dict(f)
    if sha(packed(basis["center"], basis["basis"], basis["std"])) != public["receiverBasis"]:
        raise ValueError("feature basis mismatch")
    old = json.loads((run / "observations.json").read_bytes())
    if old["lockSha256"] != file_hash(run / "lock.json"):
        raise ValueError("previous population belongs to a different lock")
    return basis, patches, old["domains"]["tweet"]["retentionIds"]


def calibrate(cache, run, out):
    basis, patches, prior = init(run)
    rows, _ = protected_split(cache, prior)
    design, native, gold = encode(rows, Encoder(cache, "target"), basis)
    policies = {}
    for name, params in patches.items():
        policy = dict((k, params["policy"][k]) for k in ("cut", "negativeOnly"))
        scores = values(design, params)
        good = (native == gold) & ((native == 0) if policy["negativeOnly"] else True)
        old = policy["cut"]
        if old != "never" and good.any():
            limit = float(np.max(scores[good]))
            policy["cut"] = limit if old == "always" else max(old, limit)
        if np.any(route(scores, native, policy) & (native == gold)):
            raise ValueError("calibration displaced a correct native decision")
        policies[name] = policy
    json_write(out / "lock.json", {"parentLock": file_hash(run / "lock.json"), "parentProtectedIds": prior,
               "calibrationIds": split_digest(rows), "calibrationRows": len(rows),
               "nativeCorrectByGold": [int(np.sum((native == gold) & (gold == k))) for k in range(3)], "policies": policies})


def bits(array):
    return "".join(np.asarray(array).astype(int).astype(str))


def evaluate(cache, run, out):
    basis, patches, prior = init(run)
    lock = json.loads((out / "lock.json").read_bytes())
    calibration, test = protected_split(cache, prior)
    if (lock["parentLock"] != file_hash(run / "lock.json") or lock["parentProtectedIds"] != prior
            or lock["calibrationIds"] != split_digest(calibration)):
        raise ValueError("wrong target calibration lock")
    encoder = Encoder(cache, "target")
    populations = {**skill_split(cache), "protected": test}
    result = {"lockSha256": file_hash(out / "lock.json"), "populations": {}}
    for key, rows in populations.items():
        design, native, labels = encode(rows, encoder, basis)
        gold = labels if key == "protected" else labels == ("anger" if key == "go" else 3)
        predictions = {}
        for name, params in patches.items():
            scores = values(design, params)
            predictions[name] = [bits(route(scores, native, p)) for p in (params["policy"], lock["policies"][name])]
        result["populations"][key] = {"ids": split_digest(rows), "native": bits(native), "labels": bits(gold), "arms": predictions}
        print(key, len(rows), flush=True)
    json_write(out / "observations.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["calibrate", "evaluate"])
    for k in ("cache", "parent_run", "output"):
        parser.add_argument(k, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (calibrate if args.phase == "calibrate" else evaluate)(args.cache, args.parent_run, args.output)
