#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/data_process
export DATA_PROCESS_ROOT=/root/autodl-tmp/data_process
export PYTHONUNBUFFERED=1
export PYTHON_BIN=${PYTHON_BIN:-/root/miniconda3/bin/python}
mkdir -p /root/autodl-tmp/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_v05_physio_mechanism_comparison/logs

screen -dmS v05p_B1_2026 bash -lc 'cd /root/autodl-tmp/data_process && export DATA_PROCESS_ROOT=/root/autodl-tmp/data_process && CUDA_VISIBLE_DEVICES=0 ${PYTHON_BIN:-/root/miniconda3/bin/python} 05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_v05_physio_mechanism_comparison.py --run-one B1 --seed 2026 --device cuda 2>&1 | tee /root/autodl-tmp/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_v05_physio_mechanism_comparison/logs/B1_seed2026.log'
screen -dmS v05p_S1_2026 bash -lc 'cd /root/autodl-tmp/data_process && export DATA_PROCESS_ROOT=/root/autodl-tmp/data_process && CUDA_VISIBLE_DEVICES=1 ${PYTHON_BIN:-/root/miniconda3/bin/python} 05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_v05_physio_mechanism_comparison.py --run-one S1 --seed 2026 --device cuda 2>&1 | tee /root/autodl-tmp/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_v05_physio_mechanism_comparison/logs/S1_seed2026.log'
screen -dmS v05p_S2_2026 bash -lc 'cd /root/autodl-tmp/data_process && export DATA_PROCESS_ROOT=/root/autodl-tmp/data_process && CUDA_VISIBLE_DEVICES=0 ${PYTHON_BIN:-/root/miniconda3/bin/python} 05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_v05_physio_mechanism_comparison.py --run-one S2 --seed 2026 --device cuda 2>&1 | tee /root/autodl-tmp/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_v05_physio_mechanism_comparison/logs/S2_seed2026.log'
screen -dmS v05p_S3_2026 bash -lc 'cd /root/autodl-tmp/data_process && export DATA_PROCESS_ROOT=/root/autodl-tmp/data_process && CUDA_VISIBLE_DEVICES=1 ${PYTHON_BIN:-/root/miniconda3/bin/python} 05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_v05_physio_mechanism_comparison.py --run-one S3 --seed 2026 --device cuda 2>&1 | tee /root/autodl-tmp/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_v05_physio_mechanism_comparison/logs/S3_seed2026.log'
