#!/usr/bin/env bash
# Run from any directory. Use Python 3.12 with requirements.txt installed.
set -euo pipefail
if [ "$#" != 2 ]; then
  echo "Usage: bash replay.sh CACHE_DIRECTORY OUTPUT_DIRECTORY" >&2
  exit 2
fi
here=$(cd "$(dirname "$0")" && pwd)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
python "$here/features.py" fit "$1" "$2"
node "$here/run.mjs" compile "$2"
python "$here/features.py" heldout "$1" "$2"
node "$here/run.mjs" predict "$2"
python "$here/score.py" "$2/predictions.json" > "$2/decision.json"
python "$here/check.py" "$here" "$2"
