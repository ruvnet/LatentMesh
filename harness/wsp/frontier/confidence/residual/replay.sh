#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo 'usage: bash replay.sh MODEL_CACHE NEW_OUTPUT_DIRECTORY' >&2
  exit 2
fi
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$1"
mkdir "$2"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python "$here/test1.py" "$1" "$2"
python "$here/score_results.py" score "$2"
python "$here/score_results.py" verify "$2"
