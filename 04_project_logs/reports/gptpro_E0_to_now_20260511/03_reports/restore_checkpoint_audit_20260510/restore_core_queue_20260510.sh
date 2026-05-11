#!/usr/bin/env bash
set -u
BASE=/root/autodl-tmp/data_process
FAIR="$BASE/02_code/final_code/model/training/fair_vehicle_event_comparison_20260427"
TRAIN="$BASE/02_code/final_code/model/training"
PY=/root/miniconda3/bin/python
REPORT="$BASE/04_project_logs/reports/restore_checkpoint_audit_20260510"
LOGDIR="$REPORT/remote_restore_logs"
STATUSDIR="$REPORT/remote_restore_status"
mkdir -p "$LOGDIR" "$STATUSDIR"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$TRAIN:$FAIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE
MAX_PARALLEL=5

echo "===== restore queue started $(date '+%F %T') ====="
echo "BASE=$BASE"
echo "MAX_PARALLEL=$MAX_PARALLEL"
nvidia-smi || true

running_count() {
  jobs -rp | wc -l
}

throttle() {
  while [ "$(running_count)" -ge "$MAX_PARALLEL" ]; do
    sleep 20
  done
}

run_bg() {
  local name="$1"
  shift
  throttle
  echo "[$(date '+%F %T')] START $name :: $*"
  (
    cd "$FAIR" || exit 99
    "$PY" restore_core_checkpoint_one.py "$@"
  ) > "$LOGDIR/${name}.log" 2>&1 &
  echo $! > "$STATUSDIR/${name}.pid"
}

wait_phase() {
  local phase="$1"
  echo "[$(date '+%F %T')] WAIT $phase"
  while [ "$(running_count)" -gt 0 ]; do
    wait -n
    sleep 5
  done
  echo "[$(date '+%F %T')] DONE $phase"
  nvidia-smi || true
}

# ??????????????????
for seed in 2026 2027 2028; do
  run_bg "E2_seed${seed}" --candidate E2 --seed "$seed"
done
for seed in 2026 2027 2028; do
  run_bg "E4_seed${seed}" --candidate E4 --seed "$seed"
done
for seed in 2026 2027 2028; do
  run_bg "E7C_seed${seed}" --candidate E7C --seed "$seed"
done
run_bg "E10C_seed2026" --candidate E10C --seed 2026
wait_phase "phase1_base_runs"

teacher_ckpt() {
  local seed="$1"
  local dir
  dir=$(ls -td "$BASE"/tmp/event_conditioned_runs/RESTORE_E4_*_seed${seed}_* 2>/dev/null | head -1 || true)
  if [ -n "$dir" ] && [ -f "$dir/best_model.pt" ]; then
    echo "$dir/best_model.pt"
  fi
}

# ?????????? E4 ???????????
for seed in 2026 2027 2028; do
  ckpt=$(teacher_ckpt "$seed")
  if [ -z "${ckpt:-}" ]; then
    echo "[$(date '+%F %T')] SKIP E5A/E6 seed${seed}: missing E4 teacher checkpoint"
    continue
  fi
  run_bg "E5A_seed${seed}" --candidate E5A --seed "$seed" --teacher-checkpoint "$ckpt"
  run_bg "E6_seed${seed}" --candidate E6 --seed "$seed" --teacher-checkpoint "$ckpt"
done
ckpt=$(teacher_ckpt 2026)
if [ -n "${ckpt:-}" ]; then
  run_bg "E11A_seed2026" --candidate E11A --seed 2026 --teacher-checkpoint "$ckpt"
else
  echo "[$(date '+%F %T')] SKIP E11A seed2026: missing E4 teacher checkpoint"
fi
wait_phase "phase2_distill_runs"

echo "===== restore queue finished $(date '+%F %T') ====="
find "$REPORT/remote_restore_records" -type f -name '*.json' 2>/dev/null | sort | tail -40
