"""Fail-closed qualification of measured per-invocation paired-arm receipts.

Each row: trial_id, arm (conventional|split), a_ns, b_ns, fallback (bool),
block_error (bool), utility, joules, seconds, bandwidth_hz.
Metadata: evidence (ota|fixture|replay), capture_sha256, hardware, seed.
Caller provenance is recorded, not independently authenticated by this tool.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np

def qualify(document):
    rows = document["rows"]
    meta = document["metadata"]
    arms = {"conventional": [], "split": []}
    ids = {k: set() for k in arms}
    for row in rows:
        arm = row["arm"]
        if arm not in arms or type(row["trial_id"]) is not int or row["trial_id"] in ids[arm]:
            raise ValueError("invalid or duplicate arm/trial")
        ids[arm].add(row["trial_id"])
        for key in ("a_ns", "b_ns", "utility", "joules", "seconds", "bandwidth_hz"):
            value = row[key]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("invalid measurement")
        if min(row["joules"], row["seconds"], row["bandwidth_hz"]) <= 0:
            raise ValueError("energy/time/bandwidth must be measured and positive")
        if type(row["fallback"]) is not bool or type(row["block_error"]) is not bool:
            raise ValueError("boolean outcomes required")
        arms[arm].append(row)
    if not ids["split"] or ids["split"] != ids["conventional"]:
        raise ValueError("matched trial IDs required")
    summary = {}
    for arm, data in arms.items():
        data.sort(key=lambda r: r["trial_id"])
        summary[arm] = dict(count=len(data),
            a_p999_ns=float(np.quantile([r["a_ns"] for r in data], .999, method="higher")),
            b_p999_ns=float(np.quantile([r["b_ns"] for r in data], .999, method="higher")),
            bler=sum(r["block_error"] for r in data) / len(data),
            fallback_rate=sum(r["fallback"] for r in data) / len(data),
            utility_per_joule=sum(r["utility"] for r in data) / sum(r["joules"] for r in data),
            utility_per_second=sum(r["utility"] for r in data) / sum(r["seconds"] for r in data),
            utility_per_hz_second=sum(r["utility"] for r in data) / sum(r["seconds"] * r["bandwidth_hz"] for r in data))
    base, split = summary["conventional"], summary["split"]
    gates = dict(samples=split["count"] >= 10000,
                 a_deadline=split["a_p999_ns"] < 150000,
                 b_deadline=split["b_p999_ns"] < 100000,
                 fallback=split["fallback_rate"] < 1e-4,
                 bler=split["bler"] <= base["bler"])
    for key in ("utility_per_joule", "utility_per_second", "utility_per_hz_second"):
        gates[key] = base[key] > 0 and split[key] >= 1.15 * base[key]
    sha = meta.get("capture_sha256", "")
    provenance = (meta.get("evidence") == "ota" and isinstance(sha, str) and len(sha) == 64
                  and all(c in "0123456789abcdef" for c in sha) and bool(meta.get("hardware")))
    # One campaign cannot meet five-seed/cross-site independent qualification.
    return dict(schema="latentmesh.runtime-qualification.v1", summary=summary, gates=gates,
                campaign_candidate=bool(provenance and all(gates.values())),
                production_qualified=False,
                stop=split["count"] >= 10000 and (split["a_p999_ns"] > 180000 or
                     split["b_p999_ns"] > 120000 or split["bler"] > base["bler"] + .01),
                missing=["independent provenance verification", "five-seed cross-site qualification", "confidence intervals"])

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path); p.add_argument("output", type=Path)
    a = p.parse_args(); raw = a.input.read_bytes()
    result = qualify(json.loads(raw)); result["input_sha256"] = hashlib.sha256(raw).hexdigest()
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

if __name__ == "__main__":
    main()
