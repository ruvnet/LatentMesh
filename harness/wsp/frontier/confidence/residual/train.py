"""Fit a sender on dialogue-held-out residuals, then enroll two signed programs."""
import argparse
import json

from sklearn.decomposition import PCA
from design import (HERE, CONTEXT, PAIRS, Encoder, apply, blob, compact, describe, file_hash,
                    fit_rows, folded, history, json_write, np, pair_state, partition,
                    predecessors, products, ridge, score_program, sha, Path, ASSETS)


def fingerprint(s):
    return sha(s.encode())


def enroll(cache, parent, out):
    out.mkdir(parents=True, exist_ok=True)
    lock = json.loads((CONTEXT / "lock.json").read_bytes())
    if file_hash(parent / "features.json") != lock["featuresSha256"]:
        raise ValueError("reused DailyDialog development features differ")
    data = json.loads((parent / "features.json").read_bytes())
    seen = history(cache)
    all_train, train, _ = partition(cache, "train", seen)
    all_tune, tune, _ = partition(cache, "validation", seen)
    fit = fit_rows(train)
    if any(describe(rows) != lock["data"][name] for rows, name in [(fit, "fit"), (tune, "tune")]):
        raise ValueError("incorrect development rows")
    prior = predecessors(all_train) | predecessors(all_tune)
    fit_groups = {r.group for r in fit}
    anchors = []
    taken = set(fit_groups)
    for r in sorted(train, key=lambda r: fingerprint("residual-anchor/" + r.id)):
        if r.group not in taken and prior[r.id] is not None:
            anchors.append(r)
            taken.add(r.group)
            if len(anchors) == 416:
                break
    if len(anchors) != 416:
        raise ValueError("insufficient independent anchors")
    ids = [r.id for r in anchors]
    x = np.column_stack((data["fit"]["pc"], data["fit"]["local"]))
    y = 2 * np.asarray(data["fit"]["truth"]) - 1
    folds = np.array([int(fingerprint(f"residual-fold/{r.group}"), 16) % 5 for r in fit])
    residual = np.empty(len(y))
    for fold in range(5):
        mask = folds == fold
        residual[mask] = y[mask] - apply(x[mask], ridge(x[~mask], y[~mask]))
    print("Fit rows / dialogues per fold", [int((folds == k).sum()) for k in range(5)],
          [len({r.group for r, f in zip(fit, folds) if f == k}) for k in range(5)], flush=True)
    enc = Encoder(cache, "source")
    def source_states(rows):
        return np.stack([pair_state(enc, prior[r.id], r.text) for r in rows])
    h = source_states(fit)
    pc = PCA(n_components=32, svd_solver="full").fit(h)
    head = ridge(pc.transform(h), residual)
    w = head[:-1] @ pc.components_
    teacher = np.r_[w, head[-1] - w @ pc.mean_]
    teacher_fit = apply(h, teacher)
    teacher_anchor = apply(source_states(anchors), teacher)
    teacher_tune = apply(source_states(tune), teacher)
    del enc
    print("Sender scores ready", flush=True)
    z = np.load(parent / "basis.npz")
    if blob(z["context"], z["offset"]) != json.loads((CONTEXT / "projection.json").read_bytes()):
        raise ValueError("cached receiver projection differs")
    enc = Encoder(cache, "target")
    texts = list(dict.fromkeys(s for r in anchors for s in (prior[r.id], r.text)))
    states, _ = enc.extract(texts)
    shared = dict(zip(texts, states @ z["context"] + z["offset"], strict=True))
    anchor_features = products(np.stack([shared[r.text] for r in anchors]), np.stack([shared[prior[r.id]] for r in anchors]))
    scale = np.asarray(data["fit"]["pc"])[:, :8].std(axis=0, ddof=1)
    splits = {}
    for name, scores in (("fit", teacher_fit), ("tune", teacher_tune)):
        common = np.array(data[name]["pc"])
        splits[name] = products(common[:, :8] / scale, common[:, 48:56])
        data[name]["teacher"] = scores.tolist()
    # Verify the common-feature reconstruction against direct projected fit witnesses.
    direct = products(z["witness"] @ z["context"] + z["offset"], z["past"])
    np.testing.assert_allclose(direct, splits["fit"][:len(direct)], atol=1e-10)
    permutation = sorted(range(len(ids)), key=lambda j: fingerprint("residual-shuffle/" + ids[j]))
    source = {"transfer": ridge(anchor_features, teacher_anchor).tolist(),
              "shuffled": ridge(anchor_features, teacher_anchor[permutation]).tolist()}
    contract = {"schema":"latentmesh.residual-wire/1", "protocolSha256":file_hash(HERE / "protocol.md"),
                "anchorIdsSha256":sha(compact(ids)), "anchors":len(ids), "coordinates":PAIRS,
                "models":{a:ASSETS[a+"-model.onnx"][1] for a in ("source","target")},
                "projectionSha256":lock["projectionSha256"]}
    json_write(out / "enrollment.json", {"contract":contract, "source":source})
    fit_local = ridge(splits["fit"], residual).astype(np.float32).astype(np.float64)
    json_write(out / "development.json", {"teacher":blob(teacher), "local":fit_local.tolist(),
               "foldSizes":np.bincount(folds).tolist(), "anchorCount":len(anchors), "anchorIdsSha256":contract["anchorIdsSha256"]})
    np.savez(out / "products.npz", **splits)
    json_write(out / "base-features.json", data)


def received_features(out):
    data = json.loads((out / "base-features.json").read_bytes())
    fields = coefficients(out)
    with np.load(out / "products.npz") as coords:
        for name in ("fit", "tune"):
            for arm, values in fields.items():
                data[name][arm] = apply(coords[name], values).tolist()
    json_write(out / "features.json", data)


def coefficients(out):
    decoded = json.loads((out / "received.json").read_bytes())
    local = json.loads((out / "development.json").read_bytes())["local"]
    return {k:np.asarray(v) for k,v in {**decoded, "local":local}.items()}


def freeze(parent, out):
    data = json.loads((out / "features.json").read_bytes())
    record = json.loads((out / "policies.json").read_bytes())
    development = json.loads((out / "development.json").read_bytes())
    coeff = coefficients(out)
    result = {"schema":"latentmesh.residual-readout/1", "parentLockSha256":file_hash(CONTEXT / "lock.json"),
              "protocolSha256":file_hash(HERE / "protocol.md"), "packetsSha256":file_hash(out / "packets.json"),
              "data":record["data"], "foldSizes":development["foldSizes"], "anchorIdsSha256":development["anchorIdsSha256"],
              "senderHead":development["teacher"], "methods":{}}
    z = np.load(parent / "basis.npz")
    h, past = z["witness"], z["past"]
    u = h @ z["context"] + z["offset"]
    for arm, policy in record["methods"].items():
        a = np.asarray(policy["slopes"])
        if a.shape != (65,):
            raise ValueError("wrong WSP readout width")
        affine = coeff.get(arm, np.zeros(13))
        params = folded(a, policy["intercept"], z["basis"], z["mean"], affine)
        expected = np.asarray(data["fit"]["pc"][:len(h)]) @ a[:64] + policy["intercept"]
        if arm != "teacher":
            expected += a[64] * np.asarray(data["fit"][arm][:len(h)])
        error = np.max(np.abs(expected - score_program(h, u, past, params)))
        if error > 1e-9 or not np.isfinite(error):
            raise ValueError("folded readout differs from unfurled WSP score")
        extras = {"sourceScale":float(a[64])} if arm == "teacher" else {"affine":affine.tolist()}
        result["methods"][arm] = {**extras, "program":blob(*params), "points":policy["points"],
                                   "modelSha256":policy["modelSha256"], "foldMaxError":float(error)}
    for name in ("fit", "tune"):
        q = data[name]
        result.setdefault("teacherFidelity", {})[name] = {}
        for arm in coeff:
            y, v = np.array(q["teacher"]), np.array(q[arm])
            result["teacherFidelity"][name][arm] = {"pearson":float(np.corrcoef(y, v)[0,1]),
                           "r2":float(1 - np.square(y-v).sum() / np.square(y-y.mean()).sum())}
    json_write(out / "lock.json", result)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cmds = cli.add_subparsers(dest="action", required=True)
    a = cmds.add_parser("enroll")
    a.add_argument("cache", type=Path)
    a.add_argument("parent", type=Path)
    f = cmds.add_parser("freeze")
    f.add_argument("parent", type=Path)
    for cmd in (a, f, cmds.add_parser("features")):
        cmd.add_argument("out", type=Path)
    args = cli.parse_args()
    if args.action == "enroll":
        enroll(args.cache, args.parent, args.out)
    elif args.action == "features":
        received_features(args.out)
    else:
        freeze(args.parent, args.out)
