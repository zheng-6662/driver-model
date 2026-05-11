#!/usr/bin/env bash
set -u
REMOTE_ROOT="/root/autodl-tmp/data_process"
PY="/root/miniconda3/bin/python"
RUN="$REMOTE_ROOT/02_code/final_code/model/training/fair_vehicle_event_comparison_20260427/run_g13_breakthrough_candidates.py"
OUT="/root/autodl-tmp/data_process/04_project_logs/reports/g13_model_breakthrough_20260510/g13_hi_2027_2028_parallel_20260510"
mkdir -p "$OUT/logs" "$OUT/status"
echo "start $(date '+%F %T')" > "$OUT/master.log"
echo "host $(hostname)" >> "$OUT/master.log"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader >> "$OUT/master.log" 2>&1
JOBS=("G13H 2027" "G13H 2028" "G13I 2027" "G13I 2028")
for item in "${JOBS[@]}"; do
  cand="${item%% *}"
  seed="${item##* }"
  report="$OUT/${cand}_seed${seed}"
  log="$OUT/logs/${cand}_seed${seed}.log"
  (
    echo "[$(date '+%F %T')] START $cand seed$seed" >> "$OUT/master.log"
    cd "$REMOTE_ROOT" || exit 2
    PYTHONUTF8=1 "$PY" "$RUN" --mode run --candidate "$cand" --seed "$seed" --report-dir "$report" > "$log" 2>&1
    code=$?
    echo "$code" > "$OUT/status/${cand}_seed${seed}.exit"
    echo "[$(date '+%F %T')] END $cand seed$seed code=$code" >> "$OUT/master.log"
  ) &
  echo "$! $cand $seed $log" >> "$OUT/pids.txt"
  sleep 8
done
wait
echo "done $(date '+%F %T')" >> "$OUT/master.log"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader >> "$OUT/master.log" 2>&1
