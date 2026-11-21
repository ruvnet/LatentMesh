"""Prepare or fold a target readout using training and validation alone."""
import argparse
import base64
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from corpus import HERE, describe, history, partition, sha
from cohorts import ARMS, Encoder, file_hash, json_write, load_heads
from features import packed

NAMES = ("transfer", "local", "pca")
WITNESS = {"transfer": "float13", "local": "local"}


def fit_rows(train):
    order = lambda r: sha(("daily-wsp-v1/" + r.id).encode())
    return sorted((r for r in train if r.anger), key=order)[:192] + sorted((r for r in train if not r.anger), key=order)[:576]


def prepare(cache, out):
    out.mkdir(parents=True, exist_ok=True)
    seen = history(cache)
    _, train, train_log = partition(cache, "train", seen)
    _, val, val_log = partition(cache, "validation", seen)
    fit = fit_rows(train)
    frozen = load_heads(HERE.parent / "heads")
    enc = Encoder(cache, "target")
    x, _ = enc.extract(r.text for r in fit)
    y, _ = enc.extract(r.text for r in val)
    pca = PCA(n_components=64, svd_solver="full").fit(x)
    np.savez(out / "training-basis.npz", basis=pca.components_, center=pca.mean_, witness=x[:64])
    result = {"fit": describe(fit), "tune": describe(val), "trainScreened": describe(train),
              "screen": {"train": train_log, "validation": val_log}}
    for label, rows, states in (("fit", fit, x), ("tune", val, y)):
        result[label]["pc"] = pca.transform(states).tolist()
        result[label]["truth"] = [int(r.anger) for r in rows]
        for name, source in WITNESS.items():
            w = frozen[ARMS.index(source)]
            result[label][name] = (states @ w[:-1] + w[-1]).tolist()
    json_write(out / "features.json", result)
    print("Fit", describe(fit), "validation", describe(val), flush=True)


def fold(out):
    record = json.loads((out / "policies.json").read_bytes())
    frozen = load_heads(HERE.parent / "heads")
    with np.load(out / "training-basis.npz") as data:
        basis, center, check = (data[k] for k in ("basis", "center", "witness"))
    source_sha = file_hash(HERE.parent / "precision" / "lock.json")
    lock = {"schema": "latentmesh.dailydialog/1", "parentLock": source_sha,
            "data": record["data"], "fitFeaturesSha256": file_hash(out / "features.json"), "methods": {}}
    for name in NAMES:
        policy = record["methods"][name]
        slopes = np.asarray(policy["slopes"])
        weight = slopes[:64] @ basis
        bias = policy["intercept"] - weight @ center
        original = (check - center) @ basis.T @ slopes[:64] + policy["intercept"]
        if name in WITNESS:
            source = frozen[ARMS.index(WITNESS[name])]
            weight = weight + slopes[-1] * source[:-1]
            bias += slopes[-1] * source[-1]
            original += slopes[-1] * (check @ source[:-1] + source[-1])
        raw = packed(weight, np.asarray([bias]))
        max_error = float(np.max(np.abs(original - (check @ weight + bias))))
        if len(raw) != 769 * 8 or not np.all(np.isfinite(np.r_[weight, bias])) or max_error > 1e-9:
            raise ValueError("folded readout diverges")
        lock["methods"][name] = {"source": WITNESS.get(name), "featureWidth": len(slopes), "modelSha256": policy["modelSha256"],
                                 "foldedSha256": sha(raw), "foldedBase64": base64.b64encode(raw).decode(),
                                 "maxFoldError": max_error, "points": policy["points"]}
    json_write(out / "lock.json", lock)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="phase", required=True)
    ps = sub.add_parser("prepare")
    ps.add_argument("cache", type=Path)
    ps.add_argument("out", type=Path)
    sub.add_parser("fold").add_argument("out", type=Path)
    args = cli.parse_args()
    if args.phase == "prepare":
        prepare(args.cache, args.out)
    else:
        fold(args.out)
