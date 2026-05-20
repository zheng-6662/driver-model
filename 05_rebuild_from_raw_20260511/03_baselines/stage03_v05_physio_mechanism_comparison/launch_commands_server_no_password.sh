#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/data_process
export DATA_PROCESS_ROOT=/root/autodl-tmp/data_process
export PYTHONUNBUFFERED=1
export PYTHON_BIN=${PYTHON_BIN:-/root/miniconda3/bin/python}
mkdir -p /root/autodl-tmp/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_v05_physio_mechanism_comparison/logs

screen -dmS v05p_T1_2026 bash -lc 'cd /root/autodl-tmp/data_process && export DATA_PROCESS_ROOT=/root/autodl-tmp/data_process && CUDA_VISIBLE_DEVICES=0 ${PYTHON_BIN:-/root/miniconda3/bin/python} 05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_v05_physio_mechanism_comparison.py --run-one T1 --seed 2026 --device cuda 2>&1 | tee /root/autodl-tmp/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_v05_physio_mechanism_comparison/logs/T1_seed2026.log'
screen -dmS v05p_T3_2026 bash -lc 'cd /root/autodl-tmp/data_process && export DATA_PROCESS_ROOT=/root/autodl-tmp/data_process && CUDA_VISIBLE_DEVICES=1 ${PYTHON_BIN:-/root/miniconda3/bin/python} 05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_v05_physio_mechanism_comparison.py --run-one T3 --seed 2026 --device cuda 2>&1 | tee /root/autodl-tmp/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_v05_physio_mechanism_comparison/logs/T3_seed2026.log'
screen -dmS v05p_T4_2026 bash -lc 'cd /root/autodl-tmp/data_process && export DATA_PROCESS_ROOT=/root/autodl-tmp/data_process && CUDA_VISIBLE_DEVICES=0 ${PYTHON_BIN:-/root/miniconda3/bin/python} 05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_v05_physio_mechanism_comparison.py --run-one T4 --seed 2026 --device cuda 2>&1 | tee /root/autodl-tmp/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_v05_physio_mechanism_comparison/logs/T4_seed2026.log'
