"""Check resets, explicit state ownership, output order and batching against installed ONNX."""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import onnxruntime as ort
from receiver import install
from stream import step
from corpus import ensure
from data import session_options
from tokenizers import Tokenizer


def run(cache):
    base = ensure(cache,"target-model.onnx")
    opts = session_options()
    opts.log_severity_level = 3
    original = ort.InferenceSession(str(base),opts,providers=["CPUExecutionProvider"])
    tokenizer = Tokenizer.from_file(str(ensure(cache,"target-tokenizer.json")))
    tokenizer.enable_padding(pad_id=1,pad_token="<pad>")
    def encode(texts):
        batch = tokenizer.encode_batch(texts)
        return {"input_ids":np.array([x.ids for x in batch],dtype="int64"),
                "attention_mask":np.array([x.attention_mask for x in batch],dtype="int64")}
    first = encode(["The bus is late.", "Please pass the salt."])
    current = encode(["That figures.", "Thank you."])
    checks = []
    for arm in ("transfer","local"):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"receiver.onnx"
            receipt = install(base,path,arm)
            sess = ort.InferenceSession(str(path),opts,providers=["CPUExecutionProvider"])
        old = step(sess,first)[-1]
        default = step(sess,current)
        explicit = step(sess,current,np.zeros((2,8)))
        contextual = step(sess,current,old)
        mixed = old.copy()
        mixed[1] = 0
        second_reset = step(sess,current,mixed)
        repeat = step(sess,current,old)
        for i in range(len(default)):
            np.testing.assert_array_equal(default[i],explicit[i])
            np.testing.assert_array_equal(contextual[i],repeat[i])
            np.testing.assert_array_equal(contextual[i][0],second_reset[i][0])
            np.testing.assert_array_equal(default[i][1],second_reset[i][1])
        np.testing.assert_array_equal(original.run(["logits"],current)[0],contextual[0])
        np.testing.assert_array_equal(default[-1],contextual[-1])
        if np.allclose(contextual[1],default[1]):
            raise ValueError("state did not change either contextual score")
        for invalid in (old[:1],np.full((2,8),np.nan)):
            try:
                step(sess,current,invalid)
            except ValueError:
                continue
            raise AssertionError("invalid explicit state accepted")
        checks.append({"arm":arm,"modelSha256":receipt["modelSha256"],"batch":2,
                       "mixedReset":True,"repeatable":True,"nativeParity":True,"invalidStateRejected":True})
    return checks


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cache",type=Path)
    p.add_argument("receipt",type=Path)
    args = p.parse_args()
    args.receipt.write_text(json.dumps(run(args.cache),separators=(",",":"))+"\n")
