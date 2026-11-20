"""Fit from train/dev only or extract reserved test cases using frozen arrays."""
import argparse
import hashlib
import json
import time
import unicodedata
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from tokenizers import Tokenizer

from assets import ASSETS, DATA, SOURCE, TARGET, ensure, file_hash

NEG = {"annoyance", "disappointment", "disgust", "fear"}
PHRASE = "angry furious enraged|annoyed afraid disappointed"
PREFIX = "latentmesh-language-v1/"


def compact(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def json_write(path, data):
    path.write_bytes(compact(data) + b"\n")


def packed(*arrays):
    return b"".join(np.asarray(a, dtype="<f8").tobytes(order="C") for a in arrays)


def partitions(cache, include_test):
    names = ensure(cache, "emotions.txt").read_text().splitlines()
    texts, ids, data = set(), set(), {}
    for split in (["train", "dev", "test"] if include_test else ["train", "dev"]):
        keep = []
        for line in ensure(cache, split + ".tsv").read_text().splitlines():
            text, labels, ident = line.split("\t")
            key = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
            duplicate = ident in ids or key in texts
            ids.add(ident)
            texts.add(key)
            if not duplicate and "," not in labels:
                keep.append((ident, text, names[int(labels)]))
        data[split] = keep
    return data


def balanced(rows, positive, negative, cap):
    def rank(row):
        return sha((PREFIX + row[0]).encode())
    pos = sorted((x for x in rows if x[2] in positive), key=rank)
    neg = sorted((x for x in rows if x[2] in negative), key=rank)
    n = min(len(pos), len(neg), cap)
    if n == 0:
        raise ValueError("empty stratum")
    return [row for pair in zip(pos[:n], neg[:n]) for row in pair]


class Encoder:
    def __init__(self, cache, role):
        model = ensure(cache, role + "-model.onnx")
        self.tokenizer = Tokenizer.from_file(str(ensure(cache, role + "-tokenizer.json")))
        self.tokenizer.no_padding()
        self.tokenizer.enable_truncation(max_length=96)
        self.hidden = {"source": "/Gather_output_0", "target": "/classifier/Gather_output_0"}[role]
        graph = onnx.load(model)
        if not any(self.hidden in n.output for n in graph.graph.node):
            raise ValueError("unknown checkpoint graph")
        graph.graph.output.append(onnx.helper.make_tensor_value_info(self.hidden, onnx.TensorProto.FLOAT, [None, 768]))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        providers = ["CPUExecutionProvider"]
        self.graph = ort.InferenceSession(graph.SerializeToString(), options, providers=providers)
        self.original = ort.InferenceSession(str(model), options, providers=providers)
        self.timings = []
        self.checked = 0

    def extract(self, texts):
        hidden, logits = [], []
        for text in texts:
            start = time.perf_counter()
            encoded = self.tokenizer.encode(text)
            inputs = {"input_ids": np.array([encoded.ids], dtype="int64"),
                      "attention_mask": np.array([encoded.attention_mask], dtype="int64")}
            native, state = self.graph.run(["logits", self.hidden], inputs)
            self.timings.append((time.perf_counter() - start) * 1000)
            if self.checked < 8:
                original = self.original.run(["logits"], inputs)[0]
                if np.argmax(original) != np.argmax(native):
                    raise ValueError("instrumentation changed native classification")
                self.checked += 1
            hidden.append(state[0])
            logits.append(native[0])
        return np.asarray(hidden, dtype=np.float64), np.asarray(logits, dtype=np.float64)


def source_scores(states, frozen):
    margins = ((states - frozen["mean"]) / frozen["scale"]) @ frozen["coef"] + frozen["bias"][0]
    return np.tanh(margins / 2)


def native_scores(logits):
    return np.tanh((logits[:, 0] - logits[:, 1]) / 2)


def project(states, frozen):
    return (states - frozen["center"]) @ frozen["basis"].T


def cosine(states, vector):
    return (states @ vector) / (np.linalg.norm(states, axis=1) * np.linalg.norm(vector))


def fingerprint_splits(rows):
    return sha(compact([r[0] for r in rows]))


def fit(cache, out):
    data = partitions(cache, False)
    train = balanced(data["train"], {"anger"}, NEG, 256)
    bank = sorted(balanced(data["dev"], {"anger"}, NEG, 32),
                  key=lambda row: sha(("latentmesh-anchor-order-v1/" + row[0]).encode()))
    if len(bank) != 64:
        raise ValueError("protocol requires 64 anchors")
    sender = Encoder(cache, "source")
    train_h, _ = sender.extract(r[1] for r in train)
    scale = StandardScaler().fit(train_h)
    clf = LogisticRegression(C=0.01, solver="lbfgs", max_iter=1000).fit(
        scale.transform(train_h), [r[2] == "anger" for r in train])
    if clf.n_iter_.max() >= 1000:
        raise ValueError("source fit did not converge")
    source_h, source_logits = sender.extract(r[1] for r in bank)
    del sender
    receiver = Encoder(cache, "target")
    target_h, _ = receiver.extract(r[1] for r in bank)
    prototypes, _ = receiver.extract(PHRASE.split("|"))
    pca = PCA(n_components=12, svd_solver="full").fit(target_h)
    f = {"mean": scale.mean_, "scale": scale.scale_, "coef": clf.coef_[0], "bias": clf.intercept_,
         "center": pca.mean_, "basis": pca.components_}
    np.savez(out / "frozen.npz", **f)
    direction = packed(f["mean"], f["scale"], f["coef"], f["bias"])
    basis = packed(f["center"], f["basis"])
    ordered = compact([[ident, text] for ident, text, _ in bank])
    manifest = {"schema": "latentmesh.language-freeze/1", "skill": "goemotions/anger-v1",
                "source": list(SOURCE), "receiver": list(TARGET), "data": DATA,
                "assetHashes": {key: value[1] for key, value in ASSETS.items()},
                "directionSha256": sha(direction), "basisSha256": sha(basis),
                "anchorBankSha256": sha(ordered), "sourceTrainingIdsSha256": fingerprint_splits(train),
                "sourceTrainingRows": len(train), "sharedAnchors": len(bank), "receiverWidth": 12,
                "provisioningBytes": {"orderedIdsAndTextJson": len(ordered),
                    "textOnlyUtf8": sum(len(r[1].encode()) for r in bank), "sourceDirectionFloat64": len(direction),
                    "receiverBasisFloat64": len(basis)}}
    json_write(out / "freeze.json", manifest)
    json_write(out / "anchors.json", {"manifest": manifest, "features": project(target_h, f).tolist(),
        "direction": source_scores(source_h, f).tolist(), "native": native_scores(source_logits).tolist(),
        "phraseCache": {"text": PHRASE, "scores": (cosine(target_h, prototypes[0]) - cosine(target_h, prototypes[1])).tolist()},
        "permutation": np.random.default_rng(72191).permutation(48).tolist()})
    print("Froze source direction and receiver basis", sha((out / "freeze.json").read_bytes()), flush=True)


def heldout(cache, out):
    manifest = json.loads((out / "freeze.json").read_bytes())
    with np.load(out / "frozen.npz", allow_pickle=False) as f:
        frozen = dict(f)
    if (sha(packed(*(frozen[k] for k in ("mean", "scale", "coef", "bias")))) != manifest["directionSha256"]
            or sha(packed(frozen["center"], frozen["basis"])) != manifest["basisSha256"]):
        raise ValueError("frozen parameters changed")
    data = partitions(cache, True)
    primary = balanced(data["test"], {"anger"}, NEG, 160)
    retention = balanced(data["test"], {"joy"}, {"sadness"}, 150)
    rows = primary + retention
    sender = Encoder(cache, "source")
    src_h, src_logits = sender.extract(r[1] for r in primary)
    direction, native = source_scores(src_h, frozen), native_scores(src_logits)
    truth = [r[2] == "anger" for r in primary]
    receiver = Encoder(cache, "target")
    dst_h, dst_logits = receiver.extract(r[1] for r in rows)
    metrics = {"directionAUROC": roc_auc_score(truth, direction), "nativeNegativeAUROC": roc_auc_score(truth, native),
               "directionCorrect": int(np.sum((direction > 0) == truth)), "nativeNegativeCorrect": int(np.sum((native > 0) == truth))}
    json_write(out / "heldout.json", {"freezeSha256": file_hash(out / "freeze.json"),
        "primaryIdsSha256": fingerprint_splits(primary), "retentionIdsSha256": fingerprint_splits(retention),
        "features": project(dst_h, frozen).tolist(), "truth": truth, "retainedLabels": [2 if r[2] == "joy" else 0 for r in retention],
        "nativeLabels": dst_logits.argmax(axis=1).tolist(), "sourceMetrics": metrics})
    timing = {}
    for role, encoder in [("source", sender), ("receiver", receiver)]:
        samples = encoder.timings[8:]
        timing[role] = {"rowsAfterWarmup": len(samples), "nativeDecisionsChecked": encoder.checked,
                        "msP50": float(np.percentile(samples, 50)), "msP95": float(np.percentile(samples, 95))}
    json_write(out / "latency-encoders.json", timing)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["fit", "heldout"])
    parser.add_argument("cache", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (fit if args.stage == "fit" else heldout)(args.cache, args.out)
