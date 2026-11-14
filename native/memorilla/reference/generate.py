"""Generate deterministic oracle fixtures using the unmodified upstream memory module.

Python is test tooling only. The shipped Rust and Node runtimes do not invoke it.
Run with an installed CPU PyTorch. No network or model download occurs here.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types
import torch

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = "d3e3f8f1b6dd2560971dfc989ffd017c40d5b825"

def upstream_module():
    if sys.version_info < (3, 11):
        raise SystemExit("Reference generation requires Python 3.11 or newer")
    source=(ROOT / "reference/memory.py").read_bytes()
    blob=hashlib.sha1(b"blob "+str(len(source)).encode()+b"\0"+source).hexdigest()
    if blob != "4e2add0b5da95ddb82039517f00921b96d6a78a0":
        raise SystemExit("Reference source differs from the pinned upstream Git blob")
    # Only checkpoint utility imports are stubbed; all model math is upstream.
    utils = types.ModuleType("memorilla.utils")
    utils.CHECKPOINT_FILE = "memory.pt"
    utils.CONFIG_FILE = "config.json"
    utils.checkpoint_file = lambda p: Path(p) / "memory.pt" if Path(p).is_dir() else Path(p)
    utils.load_memory_state = lambda p: torch.load(utils.checkpoint_file(p), map_location="cpu", weights_only=True)
    sys.modules["memorilla"] = types.ModuleType("memorilla")
    sys.modules["memorilla.utils"] = utils
    spec = importlib.util.spec_from_file_location("memorilla.memory", ROOT / "reference/memory.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

def fixture(module, path, model=None):
    torch.manual_seed(23)
    torch.set_num_threads(1)
    if model is None:
        model = module.MemoryModule(embedding_dim=8, output_dim=12, num_memories=3,
            num_heads=2, num_self_attn_layers=1 if path.name=="retrieval" else 0,
            retrieval_init=path.name=="retrieval", dropout=0.1)
        model.save(path)
    model = model.float().eval()
    d = model.config.embedding_dim
    docs = torch.randn(2,5,d)
    mask = torch.tensor([[False,False,True,False,True],[False,True,False,False,False]])
    questions = torch.randn(2,d)
    with torch.no_grad():
        result = model(docs,mask,questions)
        data = dict(upstream=UPSTREAM, torch=torch.__version__, config=vars(model.config),
            documents=docs.tolist(), mask=mask.int().tolist(), questions=questions.tolist(), expected=result.tolist())
        if model.supports_cache:
            cache=model.write_cache(docs,mask)
            data["cached"]=model.read_cache(cache,questions).tolist()
            shared=model.write_cache(docs[:1],mask[:1])
            data["shared"]=model.read_cache(shared,questions).tolist()
    path.mkdir(parents=True,exist_ok=True)
    (path/"golden.json").write_text(json.dumps(data,separators=(",",":"))+"\n")
    print(path, "output", list(result.shape), flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--checkpoint",type=Path)
    args=parser.parse_args()
    module=upstream_module()
    if args.checkpoint:
        fixture(module,ROOT/"fixtures/real",module.MemoryModule.from_pretrained(args.checkpoint))
    else:
        for name in ("learned","retrieval"):
            fixture(module,ROOT/"fixtures"/name)
