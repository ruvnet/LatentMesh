#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 || $# -gt 3 || ( $# -eq 3 && $3 != "--refit" ) ]]; then
  echo "usage: bash replay.sh MODEL_AND_DATA_CACHE NEW_RUN_DIRECTORY [--refit]" >&2
  exit 2
fi
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$1"
mkdir "$2"
cache=$(cd "$1" && pwd)
out=$(cd "$2" && pwd)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
if [[ $# -eq 3 ]]; then
  python "$here/development.py" prepare "$cache" "$out"
  node "$here/fit.mjs" "$out"
  python "$here/development.py" fold "$out"
  mv "$out/lock.json" "$out/local-fit-lock.json"
fi
python "$here/evaluate.py" "$cache" "$out"
python "$here/report.py" score "$out"
python "$here/report.py" verify "$out"
python "$here/check_controls.py" "$cache" "$out/controls.json"
