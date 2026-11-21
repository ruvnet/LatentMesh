"""Measure conditional signal on DailyDialog validation, independently of the Test1 decision."""
import argparse
import json

from design import Path, apply, file_hash, json_write, np, ridge


def examine(parent, output):
    native = json.loads((parent / "features.json").read_bytes())
    candidate = json.loads((output / "features.json").read_bytes())
    if any(native[k]["idsSha256"] != candidate[k]["idsSha256"] for k in ("fit", "tune")):
        raise ValueError("diagnostic cohort mismatch")
    def xy(k):
        s = native[k]
        return np.column_stack((s["pc"],s["local"])), 2*np.asarray(s["truth"]) - 1
    fit_x, fit_y = xy("fit")
    val_x, val_y = xy("tune")
    remaining = val_y - apply(val_x, ridge(fit_x, fit_y))
    scores = {k:np.asarray(candidate["tune"][k]) for k in ("teacher", "transfer", "shuffled", "local")}
    summary = {}
    for name, values in scores.items():
        summary[name] = {"residualPearson":float(np.corrcoef(remaining, values)[0,1]),
                         "senderPearson":float(np.corrcoef(scores["teacher"], values)[0,1])}
    record = {"population":"DailyDialog validation", "rows":int(len(val_y)), "positives":int((val_y==1).sum()),
              "fitIdsSha256":native["fit"]["idsSha256"], "validationIdsSha256":native["tune"]["idsSha256"],
              "parentFeaturesSha256":file_hash(parent/"features.json"), "featuresSha256":file_hash(output/"features.json"),
              "methods":summary}
    json_write(output/"diagnostic.json", record)
    print(json.dumps(summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("output", type=Path)
    a = parser.parse_args()
    examine(a.parent, a.output)
