"""Verify and read the published program, enrollment and protocol."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOCK_SHA = "0de8e2ddb2eaff7999f2d517c495ba275618c3b36bd26309fbaa4ac7871fcd35"
TEST_SHA = "b5131ebf90a9772ea86db121d421cbe26b04a779947ad38239c91870b004b208"
GOLD_SHA = "4038220b2e82333f8bfcc11cc9566ba9d15bde57a8601d206920392ad5edd5e7"
PROGRAMS = ("transfer", "local", "shuffled")
ARMS = (*PROGRAMS, "teacher", "incumbent.transfer", "incumbent.local")
MODES = ("majority", "f05")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def locked():
    path = HERE / "lock.json"
    if digest(path) != LOCK_SHA:
        raise ValueError("incorrect residual lock")
    lock = json.loads(path.read_bytes())
    for p, field in ((HERE / "protocol.md", "protocolSha256"), (HERE / "packets.json", "packetsSha256"),
                     (HERE.parent / "context" / "lock.json", "parentLockSha256")):
        if digest(p) != lock[field]:
            raise ValueError("program provenance differs")
    return lock
