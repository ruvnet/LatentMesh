"""Fit the matched readouts on screened DailyDialog rows and freeze their serving weights."""
import argparse
import json
from pathlib import Path

import numpy as np

from dialogue_data import ASSETS, CONTEXT, HERE, describe, development, file_hash, json_write, supervision
from design import blob
from cohorts import ARMS, load_heads


def prepare(cache,parent,out):
    from sender import Sender
    out.mkdir(parents=True,exist_ok=True)
    previous = json.loads((CONTEXT/"lock.json").read_bytes())
    if file_hash(parent/"features.json") != previous["featuresSha256"]:
        raise ValueError("unexpected original development features")
    data = json.loads((parent/"features.json").read_bytes())
    known,_ = supervision(cache)
    original, indices, audit = development(cache,known)
    declared = json.loads((out/"audit.json").read_bytes())
    if audit != declared["development"]:
        raise ValueError("declared supervised exclusions changed")
    enc = Sender(cache)
    result = {"trainScreened":data["trainScreened"],"screen":audit}
    for k in ("fit","tune"):
        example_ids = indices[k]
        rows = [original[k][j] for j in example_ids]
        margin = enc.margins([r.text for r in rows])
        item = data[k]
        common = [item["pc"][j]+[item["local"][j]] for j in example_ids]
        result[k] = {**describe(rows),"pc":common,"truth":[int(r.anger) for r in rows],
                     "local":[0.0]*len(rows),"transfer":margin.tolist()}
        print(k,len(rows),sum(r.anger for r in rows),flush=True)
    json_write(out/"features.json",result)


def freeze(parent,out):
    record = json.loads((out/"policies.json").read_bytes())
    previous = json.loads((CONTEXT/"lock.json").read_bytes())
    z=np.load(parent/"basis.npz")
    if blob(z["context"],z["offset"]) != json.loads((CONTEXT/"projection.json").read_bytes()):
        raise ValueError("receiver projection changed")
    native = load_heads(CONTEXT.parent/"heads")[ARMS.index("local")]
    result = {"schema":"latentmesh.dialogue-sender/1","protocolSha256":file_hash(HERE/"protocol.md"),
              "auditSha256":file_hash(out/"audit.json"),"parentSha256":file_hash(CONTEXT/"lock.json"),
              "projectionSha256":previous["projectionSha256"],"sourceSha256":ASSETS["emoberta-base.bin"][1],
              "baseSha256":ASSETS["target-model.onnx"][1],"developmentFeaturesSha256":file_hash(out/"features.json"),
              "data":record["data"],"methods":{}}
    h,p = z["witness"],z["past"]
    u = h @ z["context"] + z["offset"]
    for arm, method in record["methods"].items():
        a = np.asarray(method["slopes"])
        if a.shape != (66,) or (arm=="local" and a[-1]!=0):
            raise ValueError("unexpected WSP width or local auxiliary input")
        w0 = a[:48] @ z["basis"]
        bias = method["intercept"] - w0 @ z["mean"] + a[64]*native[-1]
        w = w0 + a[64]*native[:-1]
        aux = np.linspace(-4,4,len(h)) if arm=="transfer" else np.zeros(len(h))
        x = np.column_stack(((h-z["mean"])@z["basis"].T,p,u*p,h@native[:-1]+native[-1],aux))
        reference = x@a + method["intercept"]
        score = h@w + bias + p@a[48:56] + (u*p)@a[56:64] + aux*a[-1]
        difference = float(np.abs(reference-score).max())
        if not np.isfinite(difference) or difference>1e-9:
            raise ValueError("folded head does not match WSP weights")
        result["methods"][arm] = {"readout":blob(w,[bias],a[48:56],a[56:64]),"sourceScale":float(a[-1]),
                                  "points":method["points"],"modelSha256":method["modelSha256"],"foldError":difference}
    json_write(out/"lock.json",result)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    cmds=parser.add_subparsers(dest="action",required=True)
    prep=cmds.add_parser("prepare")
    prep.add_argument("cache",type=Path)
    for c in (prep,cmds.add_parser("freeze")):
        c.add_argument("parent",type=Path)
        c.add_argument("out",type=Path)
    args=parser.parse_args()
    if args.action=="prepare":
        prepare(args.cache,args.parent,args.out)
    else:
        freeze(args.parent,args.out)
