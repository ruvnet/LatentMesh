#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo 'usage: bash replay.sh MODEL_CACHE NEW_OUTPUT_DIRECTORY' >&2
  exit 2
fi
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$1"
mkdir "$2"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
python "$here/dialogue_data.py" "$1" "$2/audit.json"
cmp "$here/audit.json" "$2/audit.json"
python "$here/full_screen.py" "$1" "$2"
python "$here/dialogue_score.py" score "$2"
python "$here/dialogue_score.py" verify "$2"
