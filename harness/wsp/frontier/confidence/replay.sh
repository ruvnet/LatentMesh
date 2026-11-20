#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo "usage: bash replay.sh ASSET_CACHE NEW_RUN_DIRECTORY" >&2
  exit 2
fi
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$1"
mkdir "$2"
cache=$(cd "$1" && pwd)
out=$(cd "$2" && pwd)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python "$here/cohorts.py" fit "$cache" "$out"
node "$here/fit.mjs" "$out"
# Score the exact precommitted programs on any host. Keep the independently
# fitted lock so its numerical differences can be inspected.
mv "$out/lock.json" "$out/local-fit-lock.json"
cp "$here/lock.json" "$out/lock.json"
python "$here/cohorts.py" test "$cache" "$out"
node "$here/predict.mjs" "$out"
python "$here/scorecard.py" score "$out" "$here/lock.json" > "$out/decision.json"
python "$here/../evidence.py" archive "$out/observations.json" "$out/observations.archive.json"
python "$here/scorecard.py" verify "$out" "$here/lock.json"
python "$here/confirm.py" run "$cache" "$out"
python "$here/install.py" "$cache/target-model.onnx" transfer separate "$out/receiver.onnx" --arm float13
python "$here/parity.py" "$cache" "$out" "$out/receiver.onnx" transfer separate --arm float13

precision="$out/precision"
python "$here/precision/experiment.py" fit "$cache" "$precision"
node "$here/precision/policy.mjs" fit "$precision"
mv "$precision/lock.json" "$precision/local-fit-lock.json"
cp "$here/precision/lock.json" "$precision/lock.json"
python "$here/precision/experiment.py" test "$cache" "$precision"
node "$here/precision/policy.mjs" predict "$precision"
python "$here/precision/assess.py" score "$precision" "$here/precision/lock.json" > "$precision/decision.json"
python "$here/../evidence.py" archive "$precision/observations.json" "$precision/observations.archive.json"
python "$here/precision/assess.py" verify "$precision" "$here/precision/lock.json"
for arm in float13 scoreOnly; do
  python "$here/install.py" "$cache/target-model.onnx" all f05 "$precision/$arm.onnx" --arm "$arm" > "$precision/$arm-install.json"
  python "$here/parity.py" "$cache" "$precision" "$precision/$arm.onnx" all f05 --arm "$arm" > "$precision/$arm-parity.json"
done
