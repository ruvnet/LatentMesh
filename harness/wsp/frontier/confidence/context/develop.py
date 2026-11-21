"""Fit the registered one turn readouts on the previous DailyDialog training IDs."""
import argparse
import base64
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.decomposition import PCA

HERE = Path(__file__).resolve().parent
DAILY = HERE.parent / "dailydialog"
sys.path.insert(0, str(DAILY))
from corpus import describe, history, partition, sha
from development import WITNESS, fit_rows
from cohorts import ARMS, Encoder, file_hash, json_write, load_heads
from features import packed
from install_receiver import committed as daily_lock, LOCK_SHA as DAILY_SHA


def predecessors(all_rows):
    previous = {}
    by_id = {}
    for row in all_rows:
        by_id[row.id] = previous.get(row.group)
        previous[row.group] = row.text
    return by_id


def prepare(cache, out):
    out.mkdir(parents=True, exist_ok=True)
    seen = history(cache)
    all_train, train, train_log = partition(cache, "train", seen)
    all_tune, tune, tune_log = partition(cache, "validation", seen)
    fit = fit_rows(train)
    prev = predecessors(all_train) | predecessors(all_tune)
    locked = daily_lock()["data"]
    for name, rows in (("fit", fit), ("tune", tune)):
        if describe(rows) != locked[name]:
            raise ValueError("training or validation cohort changed")
    texts = list(dict.fromkeys(s for row in fit + tune for s in (row.text, prev[row.id]) if s is not None))
    encoded, _ = Encoder(cache, "target").extract(texts)
    states = dict(zip(texts, encoded, strict=True))
    x = np.stack([states[row.text] for row in fit])
    pc = PCA(n_components=48, svd_solver="full").fit(x)
    # Store the standardization as a single affine map, shared by both installations.
    context_map = pc.components_[:8].T / np.sqrt(pc.explained_variance_[:8])
    offset = -pc.mean_ @ context_map
    state8 = dict(zip(texts, encoded @ context_map + offset, strict=True))
    zero = np.zeros(8)
    heads = load_heads(HERE.parent / "heads")
    data = {"trainScreened": describe(train), "screen": {"train": train_log, "validation": tune_log}}
    for name, rows in (("fit", fit), ("tune", tune)):
        h = np.stack([states[r.text] for r in rows])
        current = np.stack([state8[r.text] for r in rows])
        past = np.stack([state8.get(prev[r.id], zero) for r in rows])
        data[name] = {**describe(rows), "pc": np.column_stack((pc.transform(h), past, current * past)).tolist(),
                      "truth": [int(r.anger) for r in rows]}
        for arm, source in WITNESS.items():
            w = heads[ARMS.index(source)]
            data[name][arm] = (h @ w[:-1] + w[-1]).tolist()
        if name == "fit":
            np.savez(out / "basis.npz", basis=pc.components_, mean=pc.mean_, context=context_map,
                     offset=offset, witness=h[:32], past=past[:32])
    json_write(out / "features.json", data)
    print("Labeled training and validation:", len(fit), len(tune), "unique input texts:", len(texts), flush=True)


def blob(*arrays):
    raw = packed(*arrays)
    return {"base64": base64.b64encode(raw).decode("ascii"), "sha256": sha(raw)}


def fold(out):
    training = np.load(out / "basis.npz")
    record = json.loads((out / "policies.json").read_bytes())
    head = load_heads(HERE.parent / "heads")
    context, offset = training["context"], training["offset"]
    json_write(out / "projection.json", blob(context, offset))
    lock = {"schema": "latentmesh.context-readout/1", "parentLock": DAILY_SHA, "data": record["data"],
            "featuresSha256": file_hash(out / "features.json"), "projectionSha256": file_hash(out / "projection.json"), "arms": {}}
    h, previous = training["witness"], training["past"]
    qc = h @ context + offset
    for arm, source in WITNESS.items():
        ws = head[ARMS.index(source)]
        policy = record["methods"][arm]
        a = np.array(policy["slopes"])
        if a.shape != (65,):
            raise ValueError("unexpected WSP input count")
        w = a[:48] @ training["basis"]
        b = policy["intercept"] - w @ training["mean"] + a[64] * ws[-1]
        w += a[64] * ws[:-1]
        x = np.column_stack(((h - training["mean"]) @ training["basis"].T, previous, qc * previous, h @ ws[:-1] + ws[-1]))
        scores = h @ w + b + previous @ a[48:56] + (qc * previous) @ a[56:64]
        error = float(np.max(abs(scores - (x @ a + policy["intercept"]))))
        if not np.isfinite(error) or error > 1e-9:
            raise ValueError("folded contextual readout diverges")
        lock["arms"][arm] = {"source": source, "width": 65, "modelSha256": policy["modelSha256"], "foldError": error,
                             "readout": blob(w, [b], a[48:56], a[56:64]), "points": policy["points"]}
    json_write(out / "lock.json", lock)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="phase", required=True)
    pre = sub.add_parser("prepare")
    pre.add_argument("cache", type=Path)
    pre.add_argument("out", type=Path)
    sub.add_parser("fold").add_argument("out", type=Path)
    args = cli.parse_args()
    if args.phase == "prepare":
        prepare(args.cache, args.out)
    else:
        fold(args.out)
