"""Recompute paired F0.5 and the prespecified compression decision from predictions."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from frozen_dialogue import GOLD_SHA, HERE, LOCK_SHA, MODES, RAW_SHA, SCREEN_ARMS, digest, locked
sys.path.insert(0,str(HERE.parent/"dailydialog"))
from report import performance, rn, vector
sys.path.insert(0,str(HERE.parent/"residual"))
from preflight_score import ap, decode


def summarize(truth,pred,arms):
    names=[a+"."+m for a in arms for m in MODES]
    if set(names)!=set(pred) or not truth.sum() or np.any((truth!=0)&(truth!=1)):
        raise ValueError("missing arm or invalid truth")
    y=truth.astype(np.int64)
    z=np.column_stack([vector(pred[k],"01") for k in names]).astype(np.int64)
    if len(z)!=len(y):
        raise ValueError("population lengths differ")
    patterns,counts=np.unique(np.column_stack((y,z)),axis=0,return_counts=True)
    py,flags=patterns[:,0],patterns[:,1:]
    draws=np.random.default_rng(11212026).multinomial(len(y),counts/len(y),size=10000)
    tp=draws @ (flags*py[:,None])
    f05=performance(tp,draws@flags-tp,(draws@py)[:,None])["f05"]
    point,paired={},{}
    for j,name in enumerate(names):
        true=int(y@z[:,j])
        n=int(z[:,j].sum())
        rates=performance(true,n-true,int(y.sum()))
        point[name]={"tp":true,"fp":n-true,"fn":int(y.sum())-true,
                     **{k:rn(v) if np.isfinite(v) else None for k,v in rates.items()}}
    for m in MODES:
        ai=names.index("transfer."+m)
        for arm in arms[1:]:
            k=arm+"."+m
            bi=names.index(k)
            plus=(z[:,ai]==1)&(z[:,bi]==0)
            minus=(z[:,ai]==0)&(z[:,bi]==1)
            interval=np.quantile(f05[:,ai]-f05[:,bi],[.025,.975])
            paired[k]={"gain":rn(point[names[ai]]["f05"]-point[k]["f05"]),
                       "ci95":[rn(v) for v in interval],"positiveLower":bool(interval[0]>0),"discordant":int((plus|minus).sum()),
                       "newTrue":int(y[plus].sum()),"newFalse":int((1-y[plus]).sum()),
                       "lostTrue":int(y[minus].sum()),"removedFalse":int((1-y[minus]).sum())}
    return {"rows":len(y),"angry":int(y.sum()),"methods":point,"paired":paired}


def report(folder,verify):
    locked()
    audit=json.loads((HERE/"audit.json").read_bytes())
    obs_path=folder/"observations.json"
    x=json.loads(obs_path.read_bytes())
    panel=audit["holdout"]["panels"]["screen"]
    if (x["schema"]!="latentmesh.dialogue-screen-observations/1" or x["lockSha256"]!=LOCK_SHA or
            x["idsSha256"]!=panel["idsSha256"] or x["rawSha256"]!=RAW_SHA or x["goldSha256"]!=GOLD_SHA or
            len(x["truth"])!=panel["rows"] or set(x["ranks"])!={*SCREEN_ARMS,"senderAlone"}):
        raise ValueError("screen provenance differs")
    y=vector(x["truth"],"01")
    result=summarize(y,x["predictions"],SCREEN_ARMS)
    result["descriptiveAP"]={}
    for name,blob in x["ranks"].items():
        ranks=decode(blob)
        if len(ranks)!=len(y) or not np.array_equal(np.unique(ranks),np.arange(ranks.max()+1)):
            raise ValueError("invalid score ordering")
        order=np.argsort(-ranks,kind="stable")
        ends=np.r_[np.flatnonzero(np.diff(ranks[order])),len(ranks)-1]
        result["descriptiveAP"][name]=rn(ap(y,order,ends,np.ones(len(y),dtype=np.int64)))
    result.update({"observationsSha256":digest(obs_path),"lockSha256":LOCK_SHA,
                   "passesFullSender":result["paired"]["local.majority"]["positiveLower"]})
    path=folder/"decision.json"
    if verify:
        if result!=json.loads(path.read_bytes()):
            raise ValueError("paired score differs from archive")
    else:
        path.write_text(json.dumps(result,allow_nan=False,separators=(",",":"))+"\n")
    return result,x


def compare(left,right):
    a,x=report(left,True)
    b,y=report(right,True)
    if (x["truth"],x["idsSha256"])!=(y["truth"],y["idsSha256"]):
        raise ValueError("hosts used different cases")
    print(json.dumps({"passed":[a["passesFullSender"],b["passesFullSender"]],
                      "tagDifferences":{k:sum(p!=q for p,q in zip(x["predictions"][k],y["predictions"][k],strict=True))
                                        for k in x["predictions"]}}))


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    sub=cli.add_subparsers(dest="action",required=True)
    for name in ("score","verify"):
        sub.add_parser(name).add_argument("folder",type=Path)
    sub.add_parser("compare").add_argument("folders",nargs=2,type=Path)
    args=cli.parse_args()
    if args.action=="compare":
        compare(*args.folders)
    else:
        r,_=report(args.folder,args.action=="verify")
        print(r["rows"],r["angry"],r["paired"]["local.majority"],"passed",r["passesFullSender"])
