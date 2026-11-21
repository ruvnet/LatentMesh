"""Run the fixed full-sender development screen and export its paired rank evidence."""
import argparse
import json

from scipy.special import logsumexp
from sklearn.metrics import average_precision_score

from design import (ASSETS, CONTEXT, Path, apply, compact, describe, ensure, file_hash,
                    fit_rows, history, json_write, np, partition, ridge, sha)
from data import Specialist
from preflight_score import encode, analyze


def run(cache, parent, out):
    previous = json.loads((CONTEXT / "lock.json").read_bytes())
    if file_hash(parent/"features.json") != previous["featuresSha256"]:
        raise ValueError("wrong development inputs")
    source = Specialist(cache)
    labels = json.loads(ensure(cache,"emotion-config.json").read_bytes())["id2label"]
    anger = int(next(i for i,name in labels.items() if name == "anger"))
    other = [int(i) for i in labels if int(i) != anger]
    seen = history(cache)
    _, train, _ = partition(cache, "train", seen)
    _, valid, _ = partition(cache, "validation", seen)
    rows = {"fit":fit_rows(train), "tune":valid}
    data = json.loads((parent/"features.json").read_bytes())
    x, m, y = {}, {}, {}
    for k, examples in rows.items():
        if describe(examples) != previous["data"][k]:
            raise ValueError("wrong training or validation IDs")
        raw = np.stack([source.logits(r.text) for r in examples]).astype(np.float64)
        m[k] = raw[:,anger] - logsumexp(raw[:,other],axis=1)
        y[k] = 2*np.asarray(data[k]["truth"]) - 1
        x[k] = np.column_stack((data[k]["pc"],data[k]["local"]))
    base = apply(x["tune"], ridge(x["fit"], y["fit"]))
    joint = {k:np.column_stack((x[k],m[k])) for k in rows}
    augmented = apply(joint["tune"], ridge(joint["fit"], y["fit"]))
    remaining = y["tune"] - base
    conditional = m["tune"] - apply(x["tune"], ridge(x["fit"],m["fit"]))
    truth = (y["tune"]+1)//2
    groups = np.unique([r.group for r in valid],return_inverse=True)[1]
    evidence = {"schema":"latentmesh.full-sender-screen/1", "sourceSha256":ASSETS["emotion-model.onnx"][1],
                "idsSha256":sha(compact([r.id for r in valid])), "truth":encode(truth), "groups":encode(groups),
                "partialCorrelation":float(np.corrcoef(remaining, conditional)[0,1]), "ranks":{}}
    for arm, scores in (("local",base),("fullSender",augmented)):
        order = np.argsort(-scores,kind="stable")
        ends = np.r_[np.flatnonzero(np.diff(scores[order])),len(scores)-1]
        evidence["ranks"][arm] = {"order":encode(order),"ends":encode(ends)}
    result = analyze(evidence)
    for k, scores in (("local",base),("fullSender",augmented)):
        np.testing.assert_allclose(100*average_precision_score(truth, scores), result[k+"AP"], atol=1e-5)
    json_write(out/"preflight-ranks.json", evidence)
    json_write(out/"preflight.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache","parent","out"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    run(args.cache, args.parent, args.out)
