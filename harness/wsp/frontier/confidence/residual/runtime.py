"""Install a frozen, optional contextual WSP program onto the pinned receiver checkpoint."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from frozen import HERE, LOCK_SHA, PROGRAMS, locked
sys.path.insert(0, str(HERE.parent / "context"))
from receiver import install_program, parameters as incumbent, unpack


def readout(method):
    w = unpack(method["program"], 841)
    return w[:768], w[768], w[769:777], w[777:].reshape(8, 8)


def verify_wire():
    subprocess.run(["node", str(HERE / "wire.mjs"), "verify", str(HERE)], check=True, stdout=sys.stderr)


def install(base, destination, arm):
    if arm not in PROGRAMS:
        raise ValueError("unrecognized deployable program")
    lock = locked()
    if arm != "local":
        verify_wire()
    projection, offset = incumbent("local")[:2]
    method = lock["methods"][arm]
    params = (projection, offset, *readout(method), method["points"])
    return install_program(base, destination, arm, params, LOCK_SHA)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--arm", choices=PROGRAMS, default="transfer")
    args = parser.parse_args()
    print(json.dumps(install(args.base, args.destination, args.arm)))
