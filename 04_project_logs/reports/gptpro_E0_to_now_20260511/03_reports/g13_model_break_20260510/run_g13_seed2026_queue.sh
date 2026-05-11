#!/usr/bin/env bash
set -u
BASE=/root/autodl-tmp/data_process
FAIR="$BASE/02_code/final_code/model/training/fair_vehicle_event_comparison_20260427"
TRAIN="$BASE/02_code/final_code/model/training"
PY=/root/miniconda3/bin/python
REPORT="$BASE/04_project_logs/reports/g13_model_breakthrough_20260510"
LOGDIR="$REPORT/full_seed2026_logs"
STATUSDIR="$REPORT/full_seed2026_status"
mkdir -p "$LOGDIR" "$STATUSDIR"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$TRAIN:$FAIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE
MAX_PARALLEL=5

echo "===== G13 seed2026 queue started $(date '+%F %T') ====="
echo "BASE=$BASE"
echo "MAX_PARALLEL=$MAX_PARALLEL"
nvidia-smi || true

running_count() { jobs -rp | wc -l; }
throttle() {
  while [ "$(running_count)" -ge "$MAX_PARALLEL" ]; do
    sleep 20
  done
}
run_bg() {
  local candidate="$1"
  throttle
  echo "[$(date '+%F %T')] START ${candidate} seed2026"
  (
    cd "$FAIR" || exit 99
    "$PY" run_g13_breakthrough_candidates.py --mode run --candidate "$candidate" --seed 2026
  ) > "$LOGDIR/${candidate}_seed2026.log" 2>&1 &
  echo $! > "$STATUSDIR/${candidate}_seed2026.pid"
}

for candidate in G13A G13B G13C G13F G13H G13I; do
  run_bg "$candidate"
done

echo "[$(date '+%F %T')] WAIT G13 seed2026"
while [ "$(running_count)" -gt 0 ]; do
  wait -n
  sleep 5
done

echo "===== G13 seed2026 queue finished $(date '+%F %T') ====="
nvidia-smi || true
tail -n +1 "$REPORT/g13_run_log.csv" 2>/dev/null || true
