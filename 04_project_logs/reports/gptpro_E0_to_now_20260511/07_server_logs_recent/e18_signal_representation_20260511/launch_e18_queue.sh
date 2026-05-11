#!/usr/bin/env bash
set -u
cd /root/autodl-tmp/data_process
LOG_DIR="04_project_logs/reports/server_logs/e18_signal_representation_20260511"
RECORD_DIR="04_project_logs/reports/e18_signal_representation_job_records_20260511"
mkdir -p "$LOG_DIR" "$RECORD_DIR"
: > "$LOG_DIR/queue_pids.txt"
: > "$LOG_DIR/queue_status.txt"
EXPERIMENTS=(E18A E18B E18C E18D E18E E18F E18G E18H E18I E18J E18K E18L)
MAX_PARALLEL=4
run_one() {
  local exp="$1"
  local log="$LOG_DIR/${exp}.log"
  local record="$RECORD_DIR/${exp}_seed2026.csv"
  {
    echo "START ${exp} $(date '+%F %T')"
    PYTHONUTF8=1 CUDA_VISIBLE_DEVICES=0 /root/miniconda3/bin/python \
      02_code/final_code/model/training/fair_vehicle_event_comparison_20260427/run_e18_signal_representation.py \
      --experiments "$exp" --seeds 2026 --execute --device cuda --run-record "$record"
    code=$?
    echo "END ${exp} $(date '+%F %T') exit=${code}"
    exit $code
  } > "$log" 2>&1
}
for exp in "${EXPERIMENTS[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do
    wait -n
  done
  run_one "$exp" &
  echo "$exp $!" | tee -a "$LOG_DIR/queue_pids.txt"
  echo "LAUNCHED $exp $(date '+%F %T')" >> "$LOG_DIR/queue_status.txt"
  sleep 3
done
wait
echo "ALL_DONE $(date '+%F %T')" >> "$LOG_DIR/queue_status.txt"
