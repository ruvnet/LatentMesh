#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 || $# -gt 3 || ( $# -eq 3 && $3 != "--refit" ) ]]; then
  echo "usage: bash replay.sh MODEL_CACHE NEW_RUN_DIRECTORY [--refit]" >&2
  exit 2
fi
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$1"
mkdir "$2"
cache=$(cd "$1" && pwd)
out=$(cd "$2" && pwd)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
if [[ $# -eq 3 ]]; then
  python "$here/develop.py" prepare "$cache" "$out"
  node "$here/../dailydialog/fit.mjs" "$out" --matched
  python "$here/develop.py" fold "$out"
fi
python "$here/evaluate.py" "$cache" "$out"
python "$here/score.py" score "$out"
python "$here/score.py" verify "$out"
python "$here/test_stream.py" "$cache" "$out/stream.json"
