"""Measure how much of the frozen sender score a local receiver can reconstruct."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def design(split, include_local):
    base = np.asarray(split["pc"], dtype=np.float64)
    columns = [np.ones(base.shape[0]), base]
    if include_local:
        columns.append(split["local"])
    return np.column_stack(columns)


def explain(fit, tune, include_local):
    x = design(fit,include_local)
    y = np.asarray(fit["transfer"])
    v = np.asarray(tune["transfer"])
    beta, _, rank, _ = np.linalg.lstsq(x,y,rcond=None)
    error = v - design(tune,include_local) @ beta
    squared = np.mean(error**2)
    variance = v.var()
    if variance <= 0 or not np.isfinite(squared):
        raise ValueError("sender has no measurable validation variation")
    return {"columnsWithBias":x.shape[1],"rank":int(rank),"rSquared":round(float(1-squared/variance),7),
            "rmseInSenderUnits":round(float(np.sqrt(squared)),7)}


def measure(features):
    data = json.loads(features.read_bytes())
    results = {name:explain(data["fit"],data["tune"],uses_local)
               for name,uses_local in (("receiverFeatures",False),("receiverFeaturesAndLocalScore",True))}
    return {"featuresSha256":hashlib.sha256(features.read_bytes()).hexdigest(),"fit":data["fit"]["rows"],
            "validation":data["tune"]["rows"],"results":results}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("features",type=Path)
    cli.add_argument("output",type=Path)
    cli.add_argument("--verify",action="store_true")
    args = cli.parse_args()
    result = measure(args.features)
    if args.verify:
        if result != json.loads(args.output.read_bytes()):
            raise ValueError("sender reconstruction diagnostic differs")
    else:
        args.output.write_text(json.dumps(result,separators=(",",":"))+"\n")
    print(json.dumps(result["results"]))
