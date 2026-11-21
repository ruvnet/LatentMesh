"""Resolve the source supervision and allocate two disjoint, label-blind conversation panels."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import tarfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "residual"))
from design import ASSETS, CONTEXT, compact, describe, ensure, file_hash, fit_rows, history, json_write, partition, predecessors, sha
from corpus import key

MODEL_REV = "64377bdd2a1d7bc5ecdac9a4fbd219002663df1e"
ERC_REV = "faf25370d8805a1975bda768cb94288d5d72c490"
CORPUS_REV = "a36101638a8121b422ce4a2a17746b25f23335b8"
RAW_ROOT = "https://raw.githubusercontent.com/baaesh/semeval19_task3/a861bb50458b42ee215a7edd21dcbff93347fc0f/shble/data/raw/"
GOLD_ROOT = "https://huggingface.co/datasets/oneonlee/cleansed_emocontext/resolve/d96a144ecefc8600c28e711bbbb318f4fb8f5dbd/"
for local, name, digest in (
        ("emoberta-base.bin","pytorch_model.bin","f89836cedb395f1321d50bb4358e6baefe94a4d24ae15727d00e21befb2fdd5f"),
        ("emoberta-config.json","config.json","e6faff7aa04cb2cd1910ce3d2d539da3f4d6b8c219f3bda56a9c31c8ed701167"),
        ("emoberta-tokenizer.json","tokenizer.json","2989a1589191d31125c941504362a8dd916ddd1c9cf7602acb58843b6a7d966a")):
    ASSETS[local] = (f"https://huggingface.co/tae898/emoberta-base/resolve/{MODEL_REV}/{name}",digest)
ASSETS.update({
    "erc-supervision.json":(f"https://raw.githubusercontent.com/tae898/erc/{ERC_REV}/utterance-ordered-MELD_IEMOCAP.json","ac6f3dc7b33c0c890b0664c17cd3fcf8f641d31610b669dba1058c6a4b0dae69"),
    "emoberta-corpora.tar.gz":(f"https://codeload.github.com/tae898/multimodal-datasets/tar.gz/{CORPUS_REV}","ddaa3ac5f76329094863daf6074f04d5873e4cf81596d071e0484d1203d35791"),
    "emocontext-train.tsv":(RAW_ROOT+"train.txt","d56b8371d8682e9ec2131c6e17410628d481140bb27fb2d3ee440375e1136e94"),
    "emocontext-train.csv":(GOLD_ROOT+"cleansed_train.csv","cae11b65ecf87158be24bffb8486e389e94cf8ac03e1b209cc62ff2bd4577e51"),
    "emocontext-dev.tsv":(RAW_ROOT+"dev.txt","b5131ebf90a9772ea86db121d421cbe26b04a779947ad38239c91870b004b208"),
    "emocontext-test.tsv":(RAW_ROOT+"test.txt","926548fbcc83a8d1ac8e2329d1313724f35c51dae3d46b906dbbf7eb144e636e")})


def supervision(cache):
    manifest = json.loads(ensure(cache,"erc-supervision.json").read_bytes())
    field_by_path, counts = {}, {}
    for split, dialogues in manifest.items():
        counts[split] = {}
        for lines in dialogues.values():
            for ident in lines:
                corpus, original_split, _, utterance = ident.split("/")
                p = f"multimodal-datasets-{CORPUS_REV}/{corpus}/raw-texts/{original_split}/{utterance}.json"
                if p in field_by_path:
                    raise ValueError("duplicate source utterance ID")
                field_by_path[p] = split
                counts[split][corpus] = counts[split].get(corpus,0)+1
    known = set()
    with tarfile.open(ensure(cache,"emoberta-corpora.tar.gz"),"r|gz") as records:
        for entry in records:
            if entry.name not in field_by_path:
                continue
            if not entry.isfile():
                raise ValueError("expected transcript record")
            text = json.load(records.extractfile(entry))["Utterance"]
            if not isinstance(text,str):
                raise ValueError("missing source utterance")
            known.add(key(text))
            del field_by_path[entry.name]
    if field_by_path or sum(sum(v.values()) for v in counts.values()) != 23747:
        raise ValueError("incomplete published supervision")
    known.discard("")
    return known, counts


def development(cache, known):
    past_seen = history(cache)
    all_fit, eligible, _ = partition(cache,"train",past_seen)
    all_tune, tune, _ = partition(cache,"validation",past_seen)
    prev = predecessors(all_fit) | predecessors(all_tune)
    original = {"fit":fit_rows(eligible), "tune":tune}
    prior_lock = json.loads((CONTEXT / "lock.json").read_bytes())
    indices, audit = {}, {}
    for name, rows in original.items():
        if describe(rows) != prior_lock["data"][name]:
            raise ValueError("original development cohort changed")
        keep = [j for j,r in enumerate(rows) if key(r.text) not in known and key(prev[r.id] or "") not in known]
        indices[name] = keep
        audit[name] = {**describe([rows[j] for j in keep]), "before":len(rows), "removed":len(rows)-len(keep)}
    return original, indices, audit


def read_tsv(cache, name):
    with ensure(cache,name).open(encoding="utf-8-sig",newline="") as f:
        reader = csv.DictReader(f,delimiter="\t",quoting=csv.QUOTE_NONE)
        if reader.fieldnames != ["id","turn1","turn2","turn3","label"]:
            raise ValueError("unexpected conversation schema")
        for row in reader:
            yield row


def panel_ids(rows):
    return sha(compact([r[0] for r in rows]))


def panels(cache, known):
    earlier = history(cache)
    for split in ("train","validation","test"):
        partition(cache,split,earlier)
    for fn in ("emocontext-dev.tsv","emocontext-test.tsv"):
        for row in read_tsv(cache,fn):
            earlier.update(key(row[k]) for k in ("turn1","turn2","turn3"))
    # Drop the label immediately. This path only sees model inputs and official IDs.
    raw = [(int(r["id"]),r["turn2"],r["turn3"]) for r in read_tsv(cache,"emocontext-train.tsv")]
    if [r[0] for r in raw] != list(range(30160)):
        raise ValueError("unexpected official row IDs")
    order = sorted(raw,key=lambda r:hashlib.sha256(f"dialogue-sender-v1/{r[0]}".encode()).digest())
    removed = dict.fromkeys(("emptyOrEqual","sourceSupervision","priorExperiments","sharedPanelInput"),0)
    used, clean = set(), []
    for row in order:
        prev,cur = key(row[1]),key(row[2])
        if not prev or not cur or prev == cur:
            reason = "emptyOrEqual"
        elif prev in known or cur in known:
            reason = "sourceSupervision"
        elif prev in earlier or cur in earlier:
            reason = "priorExperiments"
        elif prev in used or cur in used:
            reason = "sharedPanelInput"
        else:
            clean.append(row)
            used.update((prev,cur))
            continue
        removed[reason] += 1
    n = min(4096,len(clean)//2)
    if n == 0 or len(raw) != sum(removed.values()) + len(clean):
        raise ValueError("invalid populations")
    splits = {"screen":clean[:n], "reserve":clean[n:2*n]}
    audit = {"officialRows":len(raw),"eligible":len(clean),"exclusions":removed,"panels":{}}
    unique_across = set()
    for name, rows in splits.items():
        terms = {key(t) for _,p,c in rows for t in (p,c)}
        if len(terms)!=2*len(rows) or terms & (known | unique_across | earlier):
            raise ValueError("a scored input overlaps")
        audit["panels"][name] = {"rows":len(rows),"idsSha256":panel_ids(rows),
                                 "sourceTextOverlaps":len(known & terms),"distinctModelInputs":len(terms)}
        unique_across.update(terms)
    return splits, audit


def gold(cache, rows):
    ids = {r[0] for r in rows}
    labels = {int(r["id"]):r["label"] for r in read_tsv(cache,"emocontext-train.tsv") if int(r["id"]) in ids}
    with ensure(cache,"emocontext-train.csv").open(encoding="utf-8-sig",newline="") as f:
        checked = {i:r["label"] for i,r in enumerate(csv.DictReader(f)) if i in ids}
    if labels != checked or labels.keys() != ids or set(labels.values()) - {"angry","others","happy","sad"}:
        raise ValueError("independent labels differ")
    return [int(labels[r[0]]=="angry") for r in rows]


def make_audit(cache):
    known, counts = supervision(cache)
    _,_,development_stats = development(cache,known)
    _,held_out = panels(cache,known)
    return {"source":counts,"uniqueNormalizedSourceTexts":len(known),
            "supervisionSha256":ASSETS["erc-supervision.json"][1],
            "transcriptsSha256":ASSETS["emoberta-corpora.tar.gz"][1],
            "development":development_stats,"holdout":held_out}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache",type=Path)
    parser.add_argument("output",type=Path)
    a=parser.parse_args()
    json_write(a.output,make_audit(a.cache))
