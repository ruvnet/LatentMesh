"""Strictly read the published DailyDialog splits and isolate previously unseen turns."""
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import re
import sys
import unicodedata
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from cohorts import ASSETS, MELD, carer, ensure, meld, raw_sources, sha
from features import compact

DATA_SHA = "2de93416a95060057d8b95ef9e2ab9b17a317906f1e352414001a1887f081193"
ASSETS["dailydialog.zip"] = ("https://aclanthology.org/attachments/I17-1099.Datasets.zip", DATA_SHA)
ASSETS["meld-dev.csv"] = (f"https://raw.githubusercontent.com/declare-lab/MELD/{MELD}/data/MELD/dev_sent_emo.csv",
                          "2e89c6f8aa182d6f62f8c6331aece905ac7273ca4999660bfb5213e1d0370c1c")
SPLITS = {"train": "6191257f8250da0e045bfd1d017645e15b2358df975ec4c5c431a5a1f99662a4",
          "validation": "f068cf7526a56ab5f26e99b74ce9edbc2766f7b060a3c6cef035920b0dd28153",
          "test": "5f9526a5ce4b8af4ca237aa3ff38cbf03becbc596c70b0974d7f5f9af2ea0615"}
TOKENS = re.compile(r"\w+|[^\w\s]")


def key(text):
    s = unicodedata.normalize("NFKC", text).casefold().replace("’", "'").replace("‘", "'")
    return " ".join(TOKENS.findall(s))


@dataclass(frozen=True)
class Turn:
    id: str
    group: int
    text: str
    anger: bool


def history(cache):
    go, val, test, tweets = raw_sources(cache)
    seen = {key(s) for s in go} | {key(r[1]) for r in val + test + tweets}
    for split in ("train", "validation", "test"):
        seen.update(key(r.text) for r in carer(cache, split))
    for split in ("train", "dev", "test"):
        seen.update(key(r.text) for r in meld(cache, split))
    return seen


def partition(cache, split, seen):
    with ZipFile(ensure(cache, "dailydialog.zip")) as outer:
        raw = outer.read("EMNLP_dataset/" + split + ".zip")
    if sha(raw) != SPLITS[split]:
        raise ValueError("incorrect DailyDialog split")
    with ZipFile(BytesIO(raw)) as inner:
        def lines(stem):
            return inner.read(f"{split}/dialogues_{stem}{split}.txt").decode("utf-8-sig").splitlines()
        dialogue_lines, emotion_lines = lines(""), lines("emotion_")
    all_rows, clean, malformed = [], [], []
    earlier = set(seen)
    n_earlier = n_repeat = n_text = 0
    for i, (line, emotion) in enumerate(zip(dialogue_lines, emotion_lines, strict=True)):
        fields = line.split("__eou__")
        if fields[-1].strip():
            raise ValueError("missing end of utterance marker")
        texts = [s.strip() for s in fields[:-1]]
        n_text += len(texts)
        labels = [int(v) for v in emotion.split()]
        if any(y not in range(7) for y in labels):
            raise ValueError("unknown emotion ID")
        if len(texts) != len(labels):
            malformed.append(i)
            seen.update(key(s) for s in texts)
            continue
        for j, (text, y) in enumerate(zip(texts, labels, strict=True)):
            row = Turn(f"{split}/{i}/{j}", i, text, y == 1)
            all_rows.append(row)
            normalized = key(text)
            if normalized in earlier:
                n_earlier += 1
            elif not normalized or normalized in seen:
                n_repeat += 1
            else:
                clean.append(row)
            seen.add(normalized)
    return all_rows, clean, {"rawTurns": n_text, "rawDialogues": len(dialogue_lines),
                             "unalignableDialogues": malformed, "alignedTurns": len(all_rows),
                             "priorTextOverlaps": n_earlier, "repeatOrEmpty": n_repeat}


def describe(rows):
    return {"rows": len(rows), "positive": sum(r.anger for r in rows), "dialogues": len({r.group for r in rows}),
            "idsSha256": sha(compact([r.id for r in rows]))}
