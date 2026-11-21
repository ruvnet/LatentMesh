"""Verify the frozen files without importing a model runtime."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOCK_SHA = "d5961d7130beed45afcef62f45befffcae65ee9c15828472feb2fc27b0b56214"
GOLD_SHA = "cae11b65ecf87158be24bffb8486e389e94cf8ac03e1b209cc62ff2bd4577e51"
RAW_SHA = "d56b8371d8682e9ec2131c6e17410628d481140bb27fb2d3ee440375e1136e94"
MODES = ("majority","f05")
SCREEN_ARMS = ("transfer","local","incumbent")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def locked():
    name=HERE/"lock.json"
    if digest(name)!=LOCK_SHA:
        raise ValueError("full sender lock differs")
    data=json.loads(name.read_bytes())
    for path, field in ((HERE/"protocol.md","protocolSha256"),(HERE/"audit.json","auditSha256"),
                        (HERE.parent/"context"/"lock.json","parentSha256"),
                        (HERE.parent/"context"/"projection.json","projectionSha256")):
        if digest(path)!=data[field]:
            raise ValueError("wrong input for "+field)
    return data
