"""Run the locked unrestricted sender screen, leaving the future packet panel blind."""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from frozen_dialogue import GOLD_SHA, HERE, LOCK_SHA, MODES, RAW_SHA, locked
from dialogue_data import gold, json_write, panels, supervision
from design import Encoder, incumbent, unpack
from sender import Sender
from preflight_score import encode


def packed_ranks(values):
    if not np.isfinite(values).all() or len(values)>65535:
        raise ValueError("invalid evaluation margins")
    return encode(np.unique(values,return_inverse=True)[1])


def bits(values):
    return "".join("1" if v else "0" for v in values)


def score(cache,out):
    out.mkdir(parents=True,exist_ok=True)
    lock=locked()
    known,counts=supervision(cache)
    splits,actual=panels(cache,known)
    audit=json.loads((HERE/"audit.json").read_bytes())
    if actual!=audit["holdout"] or counts!=audit["source"]:
        raise ValueError("prospective panel or supervision changed")
    rows=splits["screen"]
    enc=Sender(cache)
    start=time.perf_counter()
    margin=enc.margins([r[2] for r in rows])
    source_seconds=time.perf_counter()-start
    del enc
    print("Full sender finished",len(rows),"current utterances",flush=True)
    target=Encoder(cache,"target")
    start=time.perf_counter()
    ph,_=target.extract([r[1] for r in rows])
    h,_=target.extract([r[2] for r in rows])
    target_seconds=time.perf_counter()-start
    matrix,offset,w,b,wp,wi,points=incumbent("local")
    u,v=h@matrix+offset,ph@matrix+offset
    def head(weight,bias,past,cross):
        return h@weight+bias+v@past+(u*v)@cross
    pred,ranks={},{}
    def record(arm,values,cutoffs):
        ranks[arm]=packed_ranks(values)
        for mode in MODES:
            threshold=cutoffs[mode]["threshold"]
            pred[arm+"."+mode]=bits(values>threshold) if threshold is not None else "0"*len(rows)
    record("incumbent",head(w,b,wp,wi),points)
    for arm,policy in lock["methods"].items():
        raw=unpack(policy["readout"],785)
        scores=head(raw[:768],raw[768],raw[769:777],raw[777:])+policy["sourceScale"]*margin
        record(arm,scores,policy["points"])
    ranks["senderAlone"]=packed_ranks(margin)
    obs={"schema":"latentmesh.dialogue-screen-observations/1","lockSha256":LOCK_SHA,"rawSha256":RAW_SHA,
         "goldSha256":GOLD_SHA,"idsSha256":actual["panels"]["screen"]["idsSha256"],"truth":bits(gold(cache,rows)),
         "predictions":pred,"ranks":ranks}
    json_write(out/"observations.json",obs)
    json_write(out/"timing.json",{"rows":len(rows),"sourceSeconds":round(source_seconds,4),
         "targetSeconds":round(target_seconds,4),"sourceEval":"one current turn, FP32, batches of 16, four CPU threads",
         "targetEval":"previous and current turn separately, INT8, two CPU threads", "includesCheckpointLoad":False})


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("cache",type=Path)
    cli.add_argument("out",type=Path)
    a=cli.parse_args()
    score(a.cache,a.out)
