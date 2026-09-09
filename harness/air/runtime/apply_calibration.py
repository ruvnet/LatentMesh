"""Apply a JSON residual artifact to path records from a calibrated site.
Usage: python apply_calibration.py model.json paths.json corrected.json
Power correction is offline; it cannot alter symbols or grant radio authority.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from calibrate import FEATURES, predict

def apply(artifact, rows):
    if artifact.get("schema") != "latentmesh.site-residual.v1" or artifact.get("features") != FEATURES:
        raise ValueError("incompatible artifact")
    m = artifact["model"]
    for key in ("mean", "scale", "weights"):
        a = np.asarray(m[key], dtype=float)
        if a.shape != (4,) or not np.isfinite(a).all(): raise ValueError("invalid model")
    if not np.isfinite(m["bias"]) or min(m["scale"]) <= 0: raise ValueError("invalid scale/bias")
    output = []
    for row in rows:
        if row["site"] not in artifact["sites"]: raise ValueError("uncalibrated site")
        x = np.array([[row[k] for k in FEATURES]], dtype=float)
        if not np.isfinite(x).all() or x[0,1] <= 0 or x[0,2] < 0 or x[0,3] <= 0:
            raise ValueError("invalid path")
        power = float(row["rt_dbm"] + predict(m, x)[0])
        if not np.isfinite(power): raise ValueError("nonfinite prediction")
        output.append(dict(row, calibrated_dbm=power))
    return output

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "input", "output"): p.add_argument(name, type=Path)
    a = p.parse_args()
    result = apply(json.loads(a.model.read_text()), json.loads(a.input.read_text()))
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
