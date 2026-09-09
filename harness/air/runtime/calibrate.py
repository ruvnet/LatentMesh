"""Offline site residual calibration. Not a reproduction of unreleased SPARC.

Input JSON: [{site, tx, rt_dbm, measured_dbm, distance_m, bounces, frequency_ghz}].
Hold out whole transmitters; select ridge penalty inside remaining TX groups.
Never consume receiver labels or OTA holdouts for calibration selection.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np

FEATURES = ["rt_dbm", "distance_m", "bounces", "frequency_ghz"]

def matrices(rows):
    if len(rows) < 6:
        raise ValueError("at least six observations required")
    for r in rows:
        if not isinstance(r.get("site"), str) or not r["site"] or not isinstance(r.get("tx"), str) or not r["tx"]:
            raise ValueError("nonempty site and tx required")
        for key in FEATURES + ["measured_dbm"]:
            if type(r.get(key)) not in (int, float) or not math.isfinite(r[key]):
                raise ValueError("finite numeric features and labels required")
        if r["distance_m"] <= 0 or r["frequency_ghz"] <= 0 or r["bounces"] < 0:
            raise ValueError("invalid physical features")
    x = np.array([[r[k] for k in FEATURES] for r in rows], dtype=float)
    y = np.array([r["measured_dbm"] - r["rt_dbm"] for r in rows])
    groups = np.array([json.dumps([r["site"], r["tx"]]) for r in rows])
    if len(set(groups)) < 3:
        raise ValueError("at least three independent transmitter groups required")
    return x, y, groups

def fit(x, y, alpha):
    mean, scale = x.mean(0), x.std(0)
    scale[scale < 1e-12] = 1
    z = (x - mean) / scale
    bias = float(y.mean())
    weights = np.linalg.solve(z.T @ z + alpha * np.eye(x.shape[1]), z.T @ (y - bias))
    return dict(mean=mean.tolist(), scale=scale.tolist(), weights=weights.tolist(), bias=bias)

def predict(model, x):
    return ((x - model["mean"]) / model["scale"]) @ model["weights"] + model["bias"]

def select(x, y, groups):
    scores = []
    for alpha in (0.1, 1., 10., 100.):
        errors = []
        for group in sorted(set(groups)):
            test = groups == group
            m = fit(x[~test], y[~test], alpha)
            errors.extend((predict(m, x[test]) - y[test]) ** 2)
        scores.append((float(np.mean(errors)), alpha))
    return min(scores)[1]

def evaluate(rows):
    x, y, groups = matrices(rows)
    estimated, offset = np.zeros_like(y), np.zeros_like(y)
    folds = []
    for group in sorted(set(groups)):
        test = groups == group
        alpha = select(x[~test], y[~test], groups[~test])
        model = fit(x[~test], y[~test], alpha)
        estimated[test] = predict(model, x[test])
        offset[test] = y[~test].mean()
        folds.append(dict(held_out=group, alpha=alpha, count=int(test.sum())))
    rmse = lambda v: float(np.sqrt(np.mean(v ** 2)))
    raw, residual, constant = rmse(y), rmse(y - estimated), rmse(y - offset)
    alpha = select(x, y, groups)
    return dict(schema="latentmesh.site-residual.v1", features=FEATURES,
                model=fit(x, y, alpha), sites=sorted({r["site"] for r in rows}),
                folds=folds, raw_rmse_db=raw, residual_rmse_db=residual,
                constant_offset_rmse_db=constant,
                calibration_gate=raw > 0 and residual <= .7 * raw and raw - residual >= 1,
                production_qualified=False,
                missing=["held-out OTA BLER/utility ranking", "material-tuned and boosting baselines", "independent hardware qualification"])

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    result = evaluate(json.loads(raw))
    result["input_sha256"] = hashlib.sha256(raw).hexdigest()
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

if __name__ == "__main__":
    main()
