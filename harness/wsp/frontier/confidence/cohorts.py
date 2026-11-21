"""Extract disjoint receiver calibration and held out features from pinned public data."""
import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import duckdb
import numpy as np
from scipy.special import softmax

from heads import ARMS, FRONTIER, HERE, load_heads
sys.path.insert(0, str(FRONTIER / "target"))
from trial import protected_split, raw_sources, unique
from data import ASSETS, Encoder, ensure, file_hash, groups, json_write, norm, sha

MELD = "e8cedf27b5d2877e198332c957127e16eb214afe"
CARER = "cab853a1dbdf4c42c2b3ef2173804746df8825fe"
for split, digest in (("train", "d2fa2d6529cf03cac2989efec05c9b27d8fd2f4c8fc5974c7ae88aa537fa02db"),
                      ("test", "8d37103938f7067600839fe29d5a114a6cd1bcdafb75bec101e06464c5006888")):
    ASSETS[f"meld-{split}.csv"] = (f"https://raw.githubusercontent.com/declare-lab/MELD/{MELD}/data/MELD/{split}_sent_emo.csv", digest)
for split, digest in (("train", "10817f0f2ea42358bc62f69a09dfb8bd71701727df6d5a387bea742f3ea06417"),
                      ("validation", "c70f0e660b5ebd1ea9a37d2a851f516f08a6d6477cdfc11be204e22a2f1102fd")):
    ASSETS[f"carer-{split}.parquet"] = (f"https://huggingface.co/datasets/dair-ai/emotion/resolve/{CARER}/split/{split}-00000-of-00001.parquet", digest)

SENTIMENTS = ("negative", "neutral", "positive")


@dataclass(frozen=True)
class Row:
    id: str
    text: str
    skill: int              # anger = 1, sadness = 0, other or unavailable = -1
    sentiment: int = -1
    group: int = -1


def rank(key):
    return sha(("receiver-confidence-v1/" + key).encode())


def carer(cache, split):
    with duckdb.connect(":memory:") as db:
        db.execute("set threads=1")
        rows = db.execute("select text, label from read_parquet(?)", [str(ensure(cache, f"carer-{split}.parquet"))]).fetchall()
    return [Row(f"carer/{split}/{i}", text, {3: 1, 0: 0}.get(label, -1)) for i, (text, label) in enumerate(rows)]


def meld(cache, split):
    with ensure(cache, f"meld-{split}.csv").open(encoding="utf-8-sig", newline="") as fp:
        return [Row(f"meld/{split}/{r['Dialogue_ID']}/{r['Utterance_ID']}", r["Utterance"],
                    {"anger": 1, "sadness": 0}.get(r["Emotion"], -1), SENTIMENTS.index(r["Sentiment"]), int(r["Dialogue_ID"]))
                for r in csv.DictReader(fp)]


def dedup(rows, excluded):
    seen = set(excluded)
    result = []
    for row in rows:
        key = norm(row.text)
        if key and key not in seen:
            result.append(row)
        seen.add(key)
    if len({r.id for r in result}) != len(result):
        raise ValueError("duplicate public row ID")
    return result


def take(rows, field, classes, cap):
    strata = [sorted((r for r in rows if getattr(r, field) == label), key=lambda r: rank(r.id)) for label in classes]
    n = min(cap, *map(len, strata))
    return [r for pair in zip(*(s[:n] for s in strata), strict=True) for r in pair]


def without(rows, used):
    ids = {r.id for r in used}
    return [r for r in rows if r.id not in ids]


def fresh_rows(cache, phase):
    prev = json.loads((FRONTIER / "target" / "lock.json").read_bytes())
    old_guard, old_test = protected_split(cache, prev["parentProtectedIds"])
    go, valid, emo, sentiment = raw_sources(cache)
    previous = go | {norm(r[1]) for r in valid + emo + sentiment}
    previous.update(norm(r.text) for r in carer(cache, "test"))
    raw_ct = carer(cache, "train")
    ct = dedup(raw_ct, previous)
    earlier = previous | {norm(r.text) for r in raw_ct}
    raw_mt = meld(cache, "train")
    mt = dedup(raw_mt, earlier)
    def tweet(r):
        return Row(r[0], r[1], -1, r[2])
    if phase == "fit":
        sfit = take(ct, "skill", (1, 0), 192)
        stune = take(without(ct, sfit), "skill", (1, 0), 96)
        guard = [tweet(r) for r in old_guard]
        gfit = [r for label in range(3) for r in [q for q in guard if q.sentiment == label][:80]]
        out = {"transfer.skillFit": sfit, "transfer.skillTune": stune,
               "transfer.guardFit": gfit, "transfer.guardTune": without(guard, gfit)}
        for partition, parity, ns, ng in (("Fit", 0, 192, 80), ("Tune", 1, 96, 120)):
            pool = [r for r in mt if int(rank(f"meld/train/{r.group}")[:2], 16) % 2 == parity]
            skill = take(pool, "skill", (1, 0), ns)
            out["adapted.skill" + partition] = skill
            out["adapted.guard" + partition] = take(without(pool, skill), "sentiment", range(3), ng)
        if {r.group for k in ("skillFit", "guardFit") for r in out["adapted." + k]} & {
                r.group for k in ("skillTune", "guardTune") for r in out["adapted." + k]}:
            raise ValueError("MELD development partitions share dialogues")
    else:
        full = meld(cache, "test")
        excluded = earlier | {norm(r.text) for r in raw_mt}
        raw_val = carer(cache, "validation")
        out = {"meld": dedup(full, excluded), "carer": [r for r in dedup(raw_val, earlier) if r.skill >= 0]}
        historical = go | {norm(r[1]) for r in valid}
        unique(emo, historical)
        eligible = unique(sentiment, historical)
        old900 = groups(eligible, [{0}, {1}, {2}], 300)
        excluded_ids = {r[0] for r in old900 + old_guard + old_test}
        exclude_text = {norm(r.text) for r in out["carer"] + out["meld"]}
        pool = [tweet(r) for r in eligible if r[0] not in excluded_ids and norm(r[1]) not in exclude_text]
        out["tweet"] = take(pool, "sentiment", range(3), 300)
        if len(out["tweet"]) != 900:
            raise ValueError("incomplete fresh sentiment panel")
        print("MELD test raw/eligible", len(full), len(out["meld"]), "CARER validation raw/eligible skill", len(raw_val), len(out["carer"]), flush=True)
    texts = [norm(r.text) for part in out.values() for r in part]
    if len(set(texts)) != len(texts):
        raise ValueError("duplicate normalized text across panels")
    return out


def extract(cache, dest, phase):
    manifest = HERE / "heads" / "index.json"
    fixed = load_heads(manifest.parent)
    if json.loads(manifest.read_bytes())["baseModel"] != file_hash(ensure(cache, "target-model.onnx")):
        raise ValueError("wrong receiver")
    if phase == "test":
        lock = json.loads((dest / "lock.json").read_bytes())
        if lock["headsSha256"] != file_hash(manifest):
            raise ValueError("wrong receiver readouts")
    rows = fresh_rows(cache, phase)
    if phase == "test":
        fit_texts = {norm(r.text) for sub in fresh_rows(cache, "fit").values() for r in sub}
        if any(norm(r.text) in fit_texts for sub in rows.values() for r in sub):
            raise ValueError("development and evaluation share normalized texts")
    enc = Encoder(cache, "target")
    result = {"headsSha256": file_hash(manifest), "arms": ARMS, "sets": {}}
    samples = []
    for key, group in rows.items():
        states, logits = enc.extract(r.text for r in group)
        scores = states @ fixed[:, :-1].T + fixed[:, -1]
        confidence = softmax(logits, axis=1).max(axis=1)
        native = logits.argmax(axis=1).tolist()
        result["sets"][key] = {"ids": sha(json.dumps([r.id for r in group], separators=(",", ":")).encode()),
            "skill": [r.skill for r in group], "sentiment": [r.sentiment for r in group], "group": [r.group for r in group],
            "native": native, "confidence": confidence.tolist(), "scores": scores.tolist()}
        if phase == "test":
            samples.extend({"text": r.text, "logits": l.tolist(), "score": float(s[0]), "confidence": float(c)}
                for r, l, s, c in zip(group[:32], logits[:32], scores[:32], confidence[:32], strict=True))
        print(key, len(group), flush=True)
    if phase == "test":
        result["lockSha256"] = file_hash(dest / "lock.json")
        json_write(dest / "parity-input.json", samples)
    json_write(dest / (phase + ".json"), result)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("fit", "test"))
    cli.add_argument("cache", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    extract(args.cache, args.output, args.phase)
