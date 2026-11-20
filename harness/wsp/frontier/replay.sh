#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo "usage: bash replay.sh ASSET_CACHE NEW_RUN_DIRECTORY" >&2
  exit 2
fi
scriptdir=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$1"
mkdir "$2"
cachedir=$(cd "$1" && pwd)
rundir=$(cd "$2" && pwd)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python "$scriptdir/data.py" prepare "$cachedir" "$rundir"
node "$scriptdir/run.mjs" compile "$rundir"
python "$scriptdir/data.py" heldout "$cachedir" "$rundir"
node "$scriptdir/run.mjs" predict "$rundir"
python "$scriptdir/score.py" "$rundir/observations.json" > "$rundir/decision.json"
python "$scriptdir/install.py" "$cachedir" "$rundir"
python "$scriptdir/artifact.py" make "$cachedir/target-model.onnx" "$rundir/receiver-with-anger.onnx" "$rundir/adapter.json"
python "$scriptdir/evidence.py" archive "$rundir/observations.json" "$rundir/observations.archive.json"
