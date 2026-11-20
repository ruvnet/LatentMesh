"""Cross-check an archived run using the authors' pinned TweetEval scorer and gold labels."""
import argparse
import base64
import hashlib
import importlib.util
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import zlib


def check(folder, script, gold):
    receipt = json.loads((folder / "official-scorer.json").read_bytes())
    if hashlib.sha256(script.read_bytes()).hexdigest() != receipt["scriptSha256"]:
        raise ValueError("unexpected TweetEval evaluator")
    gold_bytes = gold.read_bytes()
    if hashlib.sha256(gold_bytes).hexdigest() != receipt["officialTestLabelsSha256"]:
        raise ValueError("unexpected official test labels")
    spec = importlib.util.spec_from_file_location("tweeteval_evaluator", script)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    archive = json.loads((folder / "observations.archive.json").read_bytes())
    raw = zlib.decompress(base64.b64decode(archive["data"], validate=True))
    if hashlib.sha256(raw).hexdigest() != archive["sha256"]:
        raise ValueError("unexpected prediction archive")
    official = json.loads(raw)["official"]
    if "".join(gold_bytes.decode().splitlines()) != official["truth"]:
        raise ValueError("test label order differs")
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        (directory / "emotion").mkdir()
        (directory / "emotion/test_labels.txt").write_bytes(gold_bytes)
        args = SimpleNamespace(tweeteval_path=directory, predictions_path=directory, task="emotion")
        scores = {}
        for method, predictions in official["methods"].items():
            (directory / "emotion.txt").write_text("\n".join(predictions) + "\n")
            f1, _ = evaluator.single_task_results(args)
            scores[method] = round(100 * f1, 5)
    if scores != receipt["macroF1"]:
        raise ValueError("authors' evaluator differs from archived scorecard")
    print(folder.name, "TweetEval official macro F1", scores["wsp48"], "soft distillation", scores["softKD"])


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for arg in ("folder", "script", "gold"):
        cli.add_argument(arg, type=Path)
    args = cli.parse_args()
    check(args.folder, args.script, args.gold)
