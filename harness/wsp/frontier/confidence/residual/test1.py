"""Run the committed optional receivers and unrestricted teacher on official Test1."""
import argparse
from collections import Counter
import csv
import hashlib
import tempfile

import onnxruntime as ort

from frozen import ARMS, GOLD_SHA, HERE, LOCK_SHA, MODES, PROGRAMS, TEST_SHA, locked
from design import (ASSETS, Encoder, Path, apply, ensure, file_hash, history, incumbent, json_write,
                    np, pair_state, partition, score_program, unpack)
from corpus import key
from runtime import readout, verify_wire
from receiver import install_program
from stream import OUTPUTS
from context.evaluate import TSV_SHA as TEST2_SHA, feed, flags
from evidence import archive
from data import session_options

ASSETS["emocontext-dev.tsv"] = ("https://raw.githubusercontent.com/baaesh/semeval19_task3/a861bb50458b42ee215a7edd21dcbff93347fc0f/shble/data/raw/dev.txt", TEST_SHA)
ASSETS["emocontext-dev.csv"] = ("https://huggingface.co/datasets/oneonlee/cleansed_emocontext/resolve/d96a144ecefc8600c28e711bbbb318f4fb8f5dbd/cleansed_dev.csv", GOLD_SHA)


def examples(cache):
    seen = history(cache)
    for split in ("train", "validation", "test"):
        partition(cache, split, seen)
    with ensure(cache, "emocontext-test.tsv").open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            seen.update(key(row[k]) for k in ("turn1", "turn2", "turn3"))
    with ensure(cache, "emocontext-dev.tsv").open(encoding="utf-8-sig", newline="") as f:
        raw = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
    with ensure(cache, "emocontext-dev.csv").open(encoding="utf-8-sig", newline="") as f:
        gold = [r["label"] for r in csv.DictReader(f)]
    if (len(raw) != 2755 or [r["id"] for r in raw] != list(map(str, range(2755))) or
        gold != [r["label"] for r in raw] or Counter(gold) != {"angry":150,"happy":142,"sad":125,"others":2338}):
        raise ValueError("Test1 ordering, labels or official counts differ")
    history_count = repeats = 0
    encountered = set()
    rows = []
    for r in raw:
        target = key(r["turn3"])
        overlap = target in seen
        duplicate = not target or target in encountered
        history_count += int(overlap)
        repeats += int(not overlap and duplicate)
        rows.append((r["turn2"], r["turn3"], r["label"] == "angry", not (overlap or duplicate)))
        encountered.add(target)
    return rows, {"historicalTarget":history_count, "repeatOrEmptyTarget":repeats}


def bits(values):
    return "".join("1" if x else "0" for x in values)


def evaluate(cache, out):
    out.mkdir(parents=True, exist_ok=True)
    lock = locked()
    verify_wire()
    rows, exclusions = examples(cache)
    source = Encoder(cache, "source")
    teacher = unpack(lock["senderHead"], 769)
    online = [float(apply(pair_state(source, prev, cur), teacher)) for prev, cur, _, _ in rows]
    del source
    print("Unrestricted source scores complete", flush=True)
    target = Encoder(cache, "target")
    old = {a:incumbent(a) for a in ("transfer", "local")}
    matrix, offset = old["local"][:2]
    params = {a:readout(m) for a,m in lock["methods"].items()}
    opts = session_options()
    opts.log_severity_level = 3
    graphs, receipts = {}, {}
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        for arm in PROGRAMS:
            path = Path(tmp)/(arm+".onnx")
            args = (matrix, offset, *params[arm], lock["methods"][arm]["points"])
            receipts[arm] = install_program(ensure(cache,"target-model.onnx"), path, arm, args, LOCK_SHA)
            if file_hash(path) != receipts[arm]["modelSha256"]:
                raise ValueError("installed model fingerprint differs")
            graphs[arm] = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    saved = {}
    for text in dict.fromkeys(r[0] for r in rows):
        tokens = feed(target.tokenizer, text)
        expected = target.graph.run([target.hidden], tokens)[0].astype(np.float64) @ matrix + offset
        state = graphs["transfer"].run(["ctx_next"], tokens)[0]
        np.testing.assert_allclose(state, expected, rtol=0, atol=1e-9)
        saved[text] = state
    pred = {arm+"."+mode:[] for arm in ARMS for mode in MODES}
    stats = {"checked":len(rows), "instrumentedNativeDifferences":0, "uniquePrevious":len(saved),
             "nativeDifferences":{a:0 for a in PROGRAMS}, "tagDifferences":{a:0 for a in PROGRAMS},
             "maxScoreError":0., "maxStateError":0.}
    native_hash = hashlib.sha256()
    def record(arm, choices):
        for mode, choice in zip(MODES, choices, strict=True):
            pred[arm+"."+mode].append(bool(choice))
    for i, (prev, cur, _, _) in enumerate(rows):
        tokens = feed(target.tokenizer, cur)
        native = target.original.run(["logits"], tokens)[0]
        instrumented, h = target.graph.run(["logits", target.hidden], tokens)
        stats["instrumentedNativeDifferences"] += int(instrumented.tobytes() != native.tobytes())
        native_hash.update(native.astype("<f4").tobytes())
        h = h[0].astype(np.float64)
        current, past = h @ matrix + offset, saved[prev][0]
        for arm in PROGRAMS:
            logits, score, m, f, state = graphs[arm].run(OUTPUTS, {**tokens, "ctx_previous":saved[prev]})
            stats["nativeDifferences"][arm] += int(logits.tobytes() != native.tobytes())
            calc = float(score_program(h, current, past, params[arm]))
            stats["maxScoreError"] = max(stats["maxScoreError"], abs(calc-score.item()))
            stats["maxStateError"] = max(stats["maxStateError"], float(np.abs(current-state[0]).max()))
            choices = [m.item(), f.item()]
            expected = flags(calc, lock["methods"][arm]["points"])
            stats["tagDifferences"][arm] += int(choices != expected)
            record(arm, choices)
        method = lock["methods"]["teacher"]
        diagnostic = score_program(h, current, past, params["teacher"]) + method["sourceScale"] * online[i]
        record("teacher", flags(diagnostic, method["points"]))
        for arm, (_, _, w, b, v, d, points) in old.items():
            record("incumbent."+arm, flags(float(h@w + b + past@v + (current*past)@d), points))
        if (i+1) % 750 == 0:
            print("Scored", i+1, "conversations", flush=True)
    obs = {"schema":"latentmesh.residual-observations/1", "lockSha256":LOCK_SHA, "rawSha256":TEST_SHA,
           "goldSha256":GOLD_SHA, "priorTestSha256":TEST2_SHA, "exclusions":exclusions, "graphs":receipts,
           "nativeSha256":native_hash.hexdigest(), "parity":stats, "truth":bits(r[2] for r in rows),
           "clean":bits(r[3] for r in rows), "predictions":{k:bits(a) for k,a in pred.items()}}
    json_write(out / "observations.json", obs)
    archive(out / "observations.json", out / "observations.archive.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    evaluate(args.cache, args.out)
