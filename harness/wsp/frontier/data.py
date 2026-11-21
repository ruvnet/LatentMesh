"""Construct the public fit, calibration and held out partitions for the registered trial."""
import argparse
import itertools
import json
import sys
import unicodedata
from pathlib import Path

import numpy as np
import onnxruntime as ort
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "language"))
from assets import ASSETS, DATA, SOURCE, TARGET, ensure, file_hash
from features import Encoder, NEG, balanced, compact, json_write, packed, partitions, sha, source_scores

TWEET = "4fbd22cd78421f05b1ecdb4fc5725bc7a7bd8f66"
EMOTION = "90bb3b59b9d78b239630b6dfaaf321966d1abe54"
PREFIX = "latentmesh-frontier-v1/"
EMOTIONS = ["anger", "joy", "optimism", "sadness"]
for (task, split, kind), digest in {
    ("emotion", "val", "text"): "e2e30c86b8cbb97944d6543aedc06eace3bb275cb2f381aba787b838b4f23ca5",
    ("emotion", "val", "labels"): "313730630160b7e0a6b4235b800c76683f4aeeb72d094eb69646630cd5cfe338",
    ("emotion", "test", "text"): "7e1070f5d3e3fcece5bc73680bff9981e90d8f7b2f1009bfe7a01d059d1c6091",
    ("emotion", "test", "labels"): "245072348c711961785be6d395997f97cf7fcda3effeae7805664171dc75f913",
    ("sentiment", "test", "text"): "09a93a55c63fd93f97485ef7302889d7edb4091cd49733aa37da094f0bfa0675",
    ("sentiment", "test", "labels"): "6afb4afe9374d1f983bcf9a7c79b108d0f37fdf020a83f30488309bed215db9d",
}.items():
    url = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/{TWEET}/datasets/{task}/{split}_{kind}.txt"
    ASSETS[f"{task}-{split}-{kind}.txt"] = (url, digest)
for local, remote, digest in [
    ("model.onnx", "onnx/model_int8.onnx", "9f96b2966d2d83c4a2fe53bcd349c7bccd65f82ecf42d92570be38b17e6c73ea"),
    ("tokenizer.json", "tokenizer.json", "727009a8214ddfa5af1deedf1006d4d06e8e51e54aa5f03566263d4e19bfcdce"),
    ("config.json", "config.json", "77e756522bc0d17cac3f1f4b1b29aee4b984649533a114bcb5f5b30d4a1ffb6b"),
]:
    ASSETS["emotion-" + local] = (f"https://huggingface.co/onnx-community/twitter-roberta-base-emotion-ONNX/resolve/{EMOTION}/{remote}", digest)


def norm(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def order(row):
    return sha((PREFIX + row[0]).encode())


def groups(rows, strata, limit):
    parts = [sorted((r for r in rows if r[2] in group), key=order) for group in strata]
    return [r for block in zip(*(p[:limit] for p in parts)) for r in block]


def plan(cache):
    training = partitions(cache, False)["train"]
    old = balanced(training, {"anger"}, NEG, 256)
    used = {r[0] for r in old}
    pool = [r for r in training if r[0] not in used]
    plan_rows = {"source": old}
    for strata, names in [([{"anger"}, NEG], [("bank", 208), ("local", 80), ("calSkill", 112), ("testSkill", 256)]),
                          ([{"joy"}, {"sadness"}], [("guard", 128), ("calRetain", 200), ("testRetain", 350)])]:
        remaining = pool
        for name, cap in names:
            rows = groups(remaining, strata, cap)
            if len(rows) != 2 * cap:
                raise ValueError(f"insufficient rows for registered pool {name}")
            plan_rows[name] = rows
            taken = {r[0] for r in rows}
            remaining = [r for r in remaining if r[0] not in taken]
    plan_rows["bank"].sort(key=lambda r: sha(("latentmesh-frontier-bank/" + r[0]).encode()))
    all_rows = [r for rows in plan_rows.values() for r in rows]
    assert len({norm(r[1]) for r in all_rows}) == len(all_rows)
    return plan_rows


def tweet_rows(cache, task, split):
    texts = ensure(cache, f"{task}-{split}-text.txt").read_text().splitlines()
    labs = ensure(cache, f"{task}-{split}-labels.txt").read_text().splitlines()
    return [(f"{task}/{split}/{i}", text, int(label)) for i, (text, label) in enumerate(zip(texts, labs, strict=True))]


def session_options():
    opt = ort.SessionOptions()
    opt.intra_op_num_threads = 2
    opt.inter_op_num_threads = 1
    opt.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return opt


class Specialist:
    def __init__(self, cache):
        self.graph = ort.InferenceSession(str(ensure(cache, "emotion-model.onnx")), session_options(), providers=["CPUExecutionProvider"])
        self.tok = Tokenizer.from_file(str(ensure(cache, "emotion-tokenizer.json")))
        self.tok.no_padding()
        self.tok.enable_truncation(max_length=96)

    def logits(self, text):
        tokens = ["@user" if t.startswith("@") and len(t) > 1 else "http" if t.startswith("http") else t for t in text.split(" ")]
        item = self.tok.encode(" ".join(tokens))
        inputs = {"input_ids": np.array([item.ids], dtype="int64"),
                  "attention_mask": np.array([item.attention_mask], dtype="int64")}
        if any(i.name == "token_type_ids" for i in self.graph.get_inputs()):
            inputs["token_type_ids"] = np.zeros_like(inputs["input_ids"])
        return self.graph.run(["logits"], inputs)[0][0]

    def labels(self, rows):
        return np.asarray([int(np.argmax(self.logits(text))) for _, text, _ in rows])


def split_digest(rows):
    return sha(compact([r[0] for r in rows]))


def project(states, f):
    return ((states - f["center"]) @ f["basis"].T) / f["std"]


def targets(rows):
    return np.asarray([1.0 if r[2] == "anger" else -1.0 for r in rows])


def retain(rows):
    return [2 if r[2] == "joy" else 0 for r in rows]


def soft_fit(x, margins):
    # Binary soft target cross entropy with an unpenalized intercept.
    p = (margins + 1) / 2
    design = np.column_stack([x, np.ones(len(x))])
    def loss(w):
        z = design @ w
        cost = np.mean(np.logaddexp(0, z) - p * z) + 0.005 * np.dot(w[:-1], w[:-1])
        grad = design.T @ (expit(z) - p) / len(x)
        grad[:-1] += 0.01 * w[:-1]
        return cost, grad
    result = minimize(loss, np.zeros(design.shape[1]), method="L-BFGS-B", jac=True,
                      options={"maxiter": 1000, "ftol": 1e-13, "gtol": 1e-9})
    if not result.success:
        raise ValueError("distillation failed to converge")
    return result.x


def prepare(cache, out):
    p = plan(cache)
    src = Encoder(cache, "source")
    train, _ = src.extract(r[1] for r in p["source"])
    scaler = StandardScaler().fit(train)
    clf = LogisticRegression(C=0.01, solver="lbfgs", max_iter=1000).fit(scaler.transform(train), targets(p["source"]) > 0)
    axis = dict(mean=scaler.mean_, scale=scaler.scale_, coef=clf.coef_[0], bias=clf.intercept_)
    raw, _ = src.extract(r[1] for r in p["bank"])
    y = source_scores(raw, axis)
    del src
    enc = Encoder(cache, "target")
    hs, logits = {}, {}
    for name in ("bank", "local", "guard", "calSkill", "calRetain"):
        hs[name], logits[name] = enc.extract(r[1] for r in p[name])
    pca = PCA(n_components=47, svd_solver="full").fit(hs["bank"])
    f = dict(center=pca.mean_, basis=pca.components_, std=np.sqrt(pca.explained_variance_))
    np.savez(out / "basis.npz", **f)
    xs = {name: project(states, f) for name, states in hs.items()}
    guard_ok = logits["guard"].argmax(axis=1) == retain(p["guard"])
    gx = xs["guard"][guard_ok]
    lx, ly = xs["local"], targets(p["local"])
    combined = np.vstack((xs["bank"], lx, gx))
    scores = np.r_[y, ly, -np.ones(len(gx))]
    kd = soft_fit(combined, scores)
    # Sufficient-statistics witness for the column-normalized 13 parameter WSP control.
    x = np.column_stack((combined[:, :12], np.ones(len(combined))))
    col = np.max(np.abs(x), axis=0)
    a = x / col
    gram = a.T @ a / len(x) + 1e-4 * np.eye(13)
    moments = a.T @ scores / len(x)
    moment_coef = np.linalg.solve(gram, moments.astype('<f4').astype(float)) / col
    valid = groups(tweet_rows(cache, "emotion", "val"), [{i} for i in range(4)], 24)
    raw_labels = Specialist(cache).labels(valid)
    truth = np.asarray([r[2] for r in valid])
    scores_by_map = [(int(np.sum(np.take(perm, raw_labels) == truth)), perm) for perm in itertools.permutations(range(4))]
    best_correct = max(v[0] for v in scores_by_map)
    mapping = min(v[1] for v in scores_by_map if v[0] == best_correct)
    cfg = json.loads(ensure(cache, "emotion-config.json").read_bytes())["id2label"]
    rawbank = compact([[r[0], r[1]] for r in p["bank"]])
    public = {"schema": "latentmesh.frontier.public/1", "sourceModel": list(SOURCE), "receiverModel": list(TARGET),
              "goRevision": DATA, "tweetRevision": TWEET, "specialistRevision": EMOTION, "receiverWidth": 47,
              "assets": {k: v[1] for k, v in ASSETS.items()}, "fittedSourceDirection": sha(packed(*axis.values())),
              "receiverBasis": sha(packed(*f.values())), "anchorTexts": sha(rawbank),
              "publicPartitionHashes": {k: split_digest(v) for k, v in p.items() if not k.startswith("test")},
              "mapping": list(mapping), "configuredMapping": [EMOTIONS.index(cfg[str(i)]) for i in range(4)],
              "mappingValidation": {"correct": best_correct, "rows": len(valid), "ids": split_digest(valid)},
              "rows": {"bank": len(y), "local": len(lx), "protectedCorrect": len(gx), "calSkill": len(xs["calSkill"]),
                       "calRetain": len(xs["calRetain"])},
              "provisioningBytes": {"anchorIdsAndText": len(rawbank), "basisFloat64": len(packed(*f.values())),
                  "sourceDirectionFloat64": len(packed(*axis.values())),
                  "senderSharedFitFeaturesFloat64": int(combined.nbytes),
                  "localAndGuardIdsTextLabels": len(compact([list(r) for k in ("local", "guard") for r in p[k]]))}}
    json_write(out / "public.json", public)
    json_write(out / "fit.json", {"public": public, "bank": xs["bank"].tolist(), "teacher": y.tolist(),
               "local": lx.tolist(), "localLabels": ly.tolist(), "guard": gx.tolist(), "kd": kd.tolist(),
               "momentCoef": moment_coef.tolist(), "calSkill": xs["calSkill"].tolist(), "truth": (targets(p["calSkill"]) > 0).tolist(),
               "calRetain": xs["calRetain"].tolist(), "retained": retain(p["calRetain"]),
               "nativeSkill": logits["calSkill"].argmax(axis=1).tolist(), "nativeRetain": logits["calRetain"].argmax(axis=1).tolist()})
    print("Fit and calibration inputs ready", public["rows"], public["mappingValidation"], flush=True)


def heldout(cache, out):
    pub = json.loads((out / "public.json").read_bytes())
    lock = json.loads((out / "lock.json").read_bytes())
    if file_hash(out / "public.json") != lock["publicSha256"] or file_hash(out / "patches.json") != lock["patchesSha256"]:
        raise ValueError("calibration or public context changed")
    p = plan(cache)
    with np.load(out / "basis.npz", allow_pickle=False) as npz:
        f = dict(npz)
    if sha(packed(f["center"], f["basis"], f["std"])) != pub["receiverBasis"]:
        raise ValueError("receiver basis changed")
    emo = tweet_rows(cache, "emotion", "test")
    exclude = {norm(r[1]) for r in tweet_rows(cache, "emotion", "val")}
    for split in ("train", "dev", "test"):
        exclude.update(norm(line.split("\t")[0]) for line in ensure(cache, split + ".tsv").read_text().splitlines())
    used_in_fit = {norm(r[1]) for key, rows in p.items() if not key.startswith("test") for r in rows}
    used_in_fit.update(norm(r[1]) for r in groups(tweet_rows(cache,"emotion","val"),[{i} for i in range(4)],24))
    overlaps = sum(norm(r[1]) in used_in_fit for r in emo)
    def unique(rows):
        result = []
        for r in rows:
            key = norm(r[1])
            if key not in exclude:
                result.append(r)
            exclude.add(key)
        return result
    primary = groups(unique(emo), [{0}, {3}], 256)
    retained = groups(unique(tweet_rows(cache, "sentiment", "test")), [{0}, {1}, {2}], 300)
    assert not {norm(r[1]) for r in primary} & {norm(r[1]) for r in retained}
    enc = Encoder(cache, "target")
    def dataset(rows):
        h, logits = enc.extract(r[1] for r in rows)
        return {"x": project(h, f).tolist(), "native": logits.argmax(axis=1).tolist(), "ids": split_digest(rows)}
    result = {"go": {"skill": dataset(p["testSkill"]), "truth": (targets(p["testSkill"]) > 0).tolist(),
                     "retain": dataset(p["testRetain"]), "labels": retain(p["testRetain"])}}
    # Use one encoder pass per official text; binary rows index that same output.
    entirety = dataset(emo)
    indices = {r[0]: i for i, r in enumerate(emo)}
    ri = [indices[r[0]] for r in primary]
    result["tweet"] = {"skill": {"x": [entirety["x"][i] for i in ri], "native": [entirety["native"][i] for i in ri],
                                 "ids": split_digest(primary)},
                       "truth": [r[2] == 0 for r in primary], "retain": dataset(retained), "labels": [r[2] for r in retained]}
    result["official"] = {**entirety, "truth": [r[2] for r in emo],
                          "specialist": np.take(pub["mapping"], Specialist(cache).labels(emo)).tolist(),
                          "exactDevelopmentTextOverlaps": overlaps}
    json_write(out / "heldout.json", result)
    # The graph parity sample is private, reconstructed from the public sources at replay time.
    json_write(out / "graph-sample.json", [{"text": r[1], "x": x, "native": k}
        for rows, subset in ((p["testSkill"], result["go"]["skill"]), (primary, result["tweet"]["skill"]))
        for r, x, k in zip(rows[:32], subset["x"][:32], subset["native"][:32], strict=True)])
    print("Reserved cohorts extracted", [(k, len(result[k]["truth"]), len(result[k]["labels"])) for k in ("go", "tweet")],
          "official", len(emo), flush=True)


if __name__ == "__main__":
    args = argparse.ArgumentParser(__doc__)
    args.add_argument("phase", choices=["prepare", "heldout"])
    args.add_argument("cache", type=Path)
    args.add_argument("out", type=Path)
    cfg = args.parse_args()
    cfg.out.mkdir(parents=True, exist_ok=True)
    (prepare if cfg.phase == "prepare" else heldout)(cfg.cache, cfg.out)
