"""Immutable upstream assets. Download by digest into a caller supplied cache."""
import hashlib
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

SOURCE = ("Xenova/distilbert-base-uncased-finetuned-sst-2-english", "0b6928efcb76139cae2c6881d49cda67fe119f42")
TARGET = ("Xenova/twitter-roberta-base-sentiment-latest", "f3ec4d0925f90c3ca7ee7814f52d6ee7cf180445")
DATA = "2adf640a14f11025ae5a9d0ec493b78530d276d3"
ASSETS = {}
for role, (repo, rev), digests in [
    ("source", SOURCE, ("79b45c7fd8c2673aa53615aba07caeb2de798f63a95a2393a365982429c94f7f",
                        "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66")),
    ("target", TARGET, ("046f7e4cc46b399558fa9b2de966f6b0d42c69e4b01f582bfb4099b01a0bb5d7",
                        "1e6506713f00e34406a757acb80f9f3233c1c3950857d32bbc41bcd419d5d8b6")),
]:
    for local, remote, digest in zip(("model.onnx", "tokenizer.json"),
                                     ("onnx/model_quantized.onnx", "tokenizer.json"), digests):
        ASSETS[f"{role}-{local}"] = (f"https://huggingface.co/{repo}/resolve/{rev}/{remote}", digest)

for name, digest in {
    "train.tsv": "1c254a142be5c00e80d819b9ae1bbd36d94b2eeb8f4b1271846508d57e57d9c5",
    "dev.tsv": "575489c079c9de1097062a01738f998590d6b7ead66dd1c9fd1d2ba01fd8bc62",
    "test.tsv": "0587b2dd8b27b97352adbfc3fb083d46005c8946657fdc2b1ca8b1cc7f1f8be4",
    "emotions.txt": "45c3ef86782d2a4d7fedcd6d8c111aa0d0e94720689bd164fac94fefb4495a89",
}.items():
    ASSETS[name] = (f"https://raw.githubusercontent.com/google-research/google-research/{DATA}/goemotions/data/{name}", digest)


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def ensure(cache, name):
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / name
    url, expected = ASSETS[name]
    if path.exists():
        if file_hash(path) != expected:
            raise ValueError(f"digest mismatch for cached {name}")
        return path
    with tempfile.NamedTemporaryFile(dir=cache, delete=False) as handle:
        temp = Path(handle.name)
    try:
        with urllib.request.urlopen(url, timeout=90) as stream, temp.open("wb") as dest:
            shutil.copyfileobj(stream, dest)
        if file_hash(temp) != expected:
            raise ValueError(f"digest mismatch for downloaded {name}")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    return path
