"""Score locked contextual and current only WSP programs on the final EmoContext test."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time

import duckdb
import numpy as np
import onnxruntime as ort
from receiver import HERE, LOCK_SHA, OUTPUTS, install, parameters
from corpus import ASSETS, history, key, partition
from cohorts import Encoder, ensure, file_hash, json_write
from data import session_options
from install_receiver import committed, readout
sys.path.insert(0, str(HERE.parents[1]))
from evidence import archive

TSV_SHA = "926548fbcc83a8d1ac8e2329d1313724f35c51dae3d46b906dbbf7eb144e636e"
GOLD_SHA = "7ae49a1577be91bf95a8ae71437eccce995a075d56e31c0a8858e1bad2cbe49a"
ASSETS["emocontext-test.tsv"] = ("https://raw.githubusercontent.com/baaesh/semeval19_task3/a861bb50458b42ee215a7edd21dcbff93347fc0f/shble/data/raw/test.txt", TSV_SHA)
ASSETS["emocontext-test.parquet"] = ("https://huggingface.co/datasets/SemEvalWorkshop/emo/resolve/24bc90b7c8668db0c9171cef8c28e83275ff865b/emo2019/test/0000.parquet", GOLD_SHA)
LABELS = ("others", "happy", "sad", "angry")
ARMS = ("transfer", "local")
MODES = ("majority", "f05")


def test_rows(cache):
    with ensure(cache, "emocontext-test.tsv").open(encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp, delimiter="\t", quoting=csv.QUOTE_NONE)
        if reader.fieldnames != ["id", "turn1", "turn2", "turn3", "label"]:
            raise ValueError("wrong EmoContext header")
        raw = list(reader)
    # Reading file order explicitly is important for a cross-source gold check.
    with duckdb.connect(":memory:") as db:
        gold = [r[0] for r in db.execute("select label from read_parquet(?, file_row_number=true) order by file_row_number",
                                          [str(ensure(cache,"emocontext-test.parquet"))]).fetchall()]
    if len(raw) != 5509 or [int(r["id"]) for r in raw] != list(range(5509)) or len(gold) != len(raw):
        raise ValueError("unexpected evaluation IDs")
    if any(y not in range(4) or r["label"] != LABELS[y] for r,y in zip(raw, gold, strict=True)):
        raise ValueError("independent gold labels differ")
    seen = history(cache)
    for split in ("train","validation","test"):
        partition(cache, split, seen)
    historic = seen.copy()
    rows = []
    counts = {"historicalTarget":0, "repeatOrEmptyTarget":0}
    for r in raw:
        normalized = key(r["turn3"])
        if normalized in historic:
            clean = False
            counts["historicalTarget"] += 1
        elif not normalized or normalized in seen:
            clean = False
            counts["repeatOrEmptyTarget"] += 1
        else:
            clean = True
        rows.append((r["turn2"], r["turn3"], r["label"]=="angry", clean))
        seen.add(normalized)
    return rows, counts


def feed(tokenizer, text):
    tokens = tokenizer.encode(text)
    return {"input_ids":np.array([tokens.ids], dtype=np.int64), "attention_mask":np.array([tokens.attention_mask], dtype=np.int64)}


def flags(score, points):
    return [points[m]["threshold"] is not None and score > points[m]["threshold"] for m in MODES]


def times(sessions, samples):
    # Measure both on the same warmed, pretokenized final turns using real previous state.
    outputs = {"base":["logits"], "context":OUTPUTS}
    budget = len(samples)
    for native, contextual in samples[:8]:
        sessions["base"].run(outputs["base"], native)
        sessions["context"].run(outputs["context"], contextual)
    results = {"base":[], "context":[]}
    for k, (native, contextual) in enumerate(samples):
        order = ("base","context") if k % 2 == 0 else ("context","base")
        for name in order:
            inp = native if name == "base" else contextual
            start = time.perf_counter_ns()
            sessions[name].run(outputs[name], inp)
            results[name].append((time.perf_counter_ns()-start)/1e6)
    return {"sample":budget, "unit":"ms", "warmup":min(8,budget),
            "sessions":{k:{"p50":round(float(np.median(a)),4), "p95":round(float(np.percentile(a,95)),4)} for k,a in results.items()}}


def evaluate(cache, out):
    out.mkdir(parents=True, exist_ok=True)
    rows, exclusions = test_rows(cache)
    base = ensure(cache, "target-model.onnx")
    opts = session_options()
    opts.log_severity_level = 3
    sessions, receipts = {}, {}
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        for arm in ARMS:
            path = Path(tmp) / (arm+".onnx")
            receipts[arm] = install(base, path, arm)
            if file_hash(path) != receipts[arm]["modelSha256"]:
                raise ValueError("installation differs from receipt")
            sessions[arm] = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    enc = Encoder(cache,"target")
    previous_params = committed()["methods"]
    old = {arm: (readout(previous_params[arm]), previous_params[arm]["points"]) for arm in ARMS}
    params = {arm:parameters(arm) for arm in ARMS}
    matrix, offset = params["transfer"][:2]
    predictions = {name+"."+m:[] for name in ("context.transfer","context.local","current.transfer","current.local","reset.transfer") for m in MODES}
    proof = {"checkedFinal":0, "checkedPrevious":0, "instrumentedDrift":0., "transferDrift":0., "localDrift":0.,
             "stateMaxError":0., "scoreMaxError":0., "tagMismatches":0}
    native_digest = hashlib.sha256()
    samples = []
    def drift(name, x, y):
        error = float(np.max(np.abs(x-y)))
        if not np.isfinite(error):
            raise ValueError("nonfinite inference")
        proof[name] = max(proof[name], error)
    def record(name, tag):
        for mode, value in zip(MODES, tag, strict=True):
            predictions[name+"."+mode].append("1" if value else "0")
    for i,(prev,text,_,_) in enumerate(rows):
        p, cur = feed(enc.tokenizer,prev), feed(enc.tokenizer,text)
        prev_hidden = enc.graph.run([enc.hidden],p)[0].astype(np.float64)
        previous = prev_hidden @ matrix + offset
        native = enc.original.run(["logits"],cur)[0]
        replay, h = enc.graph.run(["logits",enc.hidden],cur)
        h = h.astype(np.float64)
        drift("instrumentedDrift",native,replay)
        native_digest.update(native.astype("<f4").tobytes())
        next_state = h @ matrix + offset
        for arm in ARMS:
            graph = sessions[arm]
            cached = graph.run(["ctx_next"],p)[0]  # no feed on a conversation boundary
            drift("stateMaxError",previous,cached)
            inputs = {**cur, "ctx_previous":cached}
            logits, score, a, b, state = graph.run(OUTPUTS,inputs)
            drift(arm+"Drift",native,logits)
            drift("stateMaxError",next_state,state)
            _, _, weight, bias, wp, wi, points = params[arm]
            zero = float((h @ weight).item()+bias)
            actual_score = zero + (previous @ wp).item() + ((next_state*previous) @ wi).item()
            drift("scoreMaxError",np.array([actual_score]),score)
            expected = flags(actual_score,points)
            observed = [a.item(),b.item()]
            proof["tagMismatches"] += sum(i != j for i,j in zip(expected,observed,strict=True))
            record("context."+arm,observed)
            if arm == "transfer":
                record("reset.transfer",flags(zero,points))
                if i < 128:
                    samples.append((cur,inputs))
            w, current_points = old[arm]
            record("current."+arm,flags(float((h @ w[:-1]).item()+w[-1]),current_points))
        proof["checkedFinal"] += 1
        proof["checkedPrevious"] += 1
        if (i+1) % 1000 == 0:
            print("Scored",i+1,"of",len(rows),flush=True)
    packed = {"schema":"latentmesh.context-observations/1", "lockSha256":LOCK_SHA, "datasetSha256":TSV_SHA, "goldSha256":GOLD_SHA,
              "graphs":receipts, "nativeSha256":native_digest.hexdigest(), "parity":proof,
              "exclusions":exclusions, "truth":"".join("1" if r[2] else "0" for r in rows),
              "clean":"".join("1" if r[3] else "0" for r in rows), "predictions":{k:"".join(v) for k,v in predictions.items()}}
    js = out / "observations.json"
    json_write(js,packed)
    archive(js, out/"observations.archive.json")
    json_write(out/"latency.json", times({"base":enc.original,"context":sessions["transfer"]}, samples))
    print("Final turns checked",len(rows),flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("cache", type=Path)
    cli.add_argument("out", type=Path)
    args = cli.parse_args()
    evaluate(args.cache, args.out)
