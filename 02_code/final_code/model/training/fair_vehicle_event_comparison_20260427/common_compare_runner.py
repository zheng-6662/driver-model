# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
TRAINING_DIR = THIS_DIR.parent
MANIFEST_PATH = TRAINING_DIR / "protocol_allphase_control_v2_context_full2s" / "sample_manifest.csv"
PROJECT_ROOT = TRAINING_DIR.parents[3]
STYLE_VECTOR_PATH = PROJECT_ROOT / "04_project_logs" / "reports" / "style_probe_artifacts" / "driver_style_vectors.csv"

if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from run_event_conditioned_trajectory_baseline import parse_args, train_one_run  # noqa: E402
from prediction_plotting import save_prediction_plots_for_run  # noqa: E402


# 这张表是全部公平对照实验的共同 fullrun 配置。
# 如果后面要统一改 epoch、batch、学习率、显卡等，只改这里，不要分别改各个入口脚本。
COMMON_FULLRUN_CONFIG: dict[str, Any] = {
    "manifest": str(MANIFEST_PATH),
    "seed": 2026,
    "device": "cuda",
    "init_checkpoint": None,
    "epochs": 40,
    "min_epochs": 40,
    "patience": 99,
    "batch_size": 64,
    "lr": 1e-3,
    "weight_decay": 0.0,
    "grad_clip": 1.0,
    "selection_mode": "legacy_rmse",
    "d_model": 128,
    "nhead": 2,
    "enc_layers": 2,
    "dec_layers": 2,
    "ffn_dim": 256,
    "dropout": 0.1,
    "event_embed_dim": 96,
    "event_bin_size": 20,
    "structure_width": 0.065,
    "gate_temperature": 0.040,
    "event_residual_scale": 1.0,
    "use_privileged_teacher": False,
    "max_train_samples": None,
    "max_val_samples": None,
    "max_test_samples": None,
    "smoke_test": False,
}


VARIANT_CONFIGS: dict[str, dict[str, Any]] = {
    "vehicle_direct": {
        "display_name": "01 只有车辆数据，直接预测方向盘轨迹",
        "run_prefix": "FAIR01_只有车辆数据_直接预测轨迹",
        "conditioning_mode": "vehicle_direct",
        "teacher_forcing_ratio": 0.0,
        "event_loss_weight": 0.0,
    },
    "event_injection_teacher_forcing": {
        "display_name": "02 车辆数据 + 显式事件信息注入",
        "run_prefix": "FAIR02_车辆数据_显式事件信息注入",
        "conditioning_mode": "structured_v2",
        "teacher_forcing_ratio": 1.0,
        "event_loss_weight": 0.5,
    },
    "event_injection_no_teacher_forcing": {
        "display_name": "03 车辆数据 + 显式事件信息注入 - 教师强制",
        "run_prefix": "FAIR03_车辆数据_显式事件信息注入_无教师强制",
        "conditioning_mode": "structured_v2",
        "teacher_forcing_ratio": 0.0,
        "event_loss_weight": 0.5,
    },
    "event_injection_coarse_fine_heads": {
        "display_name": "04 车辆数据 + 显式事件信息注入（粗细两个状态头）",
        "run_prefix": "FAIR04_车辆数据_显式事件信息注入_粗细两个状态头",
        "conditioning_mode": "structured_v2_coarse_fine",
        "teacher_forcing_ratio": 1.0,
        "event_loss_weight": 0.5,
    },
    "vehicle_teacher_state_continuous_style": {
        "display_name": "05 车辆数据 + 教师状态 + 连续驾驶风格",
        "run_prefix": "FAIR05_车辆数据_教师状态_连续驾驶风格",
        "conditioning_mode": "structured_v2_coarse_fine",
        "teacher_forcing_ratio": 1.0,
        "event_loss_weight": 0.5,
        "enable_teacher_state_context": True,
        "teacher_state_mode": "pca_latent",
        "teacher_state_dim": 4,
        "enable_driver_style_context": True,
        "driver_style_vector_csv": str(STYLE_VECTOR_PATH),
        "driver_style_embed_dim": 4,
        "driver_style_include_iqr": True,
    },
    "vehicle_direct_teacher_state_continuous_style": {
        "display_name": "06 车辆数据 + 教师状态 + 连续驾驶风格（直接预测）",
        "run_prefix": "FAIR06_车辆数据_教师状态_连续驾驶风格_直接预测",
        "conditioning_mode": "vehicle_direct",
        "teacher_forcing_ratio": 0.0,
        "event_loss_weight": 0.0,
        "enable_teacher_state_context": True,
        "teacher_state_mode": "pca_latent",
        "teacher_state_dim": 4,
        "enable_driver_style_context": True,
        "driver_style_vector_csv": str(STYLE_VECTOR_PATH),
        "driver_style_embed_dim": 4,
        "driver_style_include_iqr": True,
    },
    "vehicle_direct_teacher_state": {
        "display_name": "07 车辆数据 + 教师状态（直接预测）",
        "run_prefix": "FAIR07_车辆数据_教师状态_直接预测",
        "conditioning_mode": "vehicle_direct",
        "teacher_forcing_ratio": 0.0,
        "event_loss_weight": 0.0,
        "enable_teacher_state_context": True,
        "teacher_state_mode": "pca_latent",
        "teacher_state_dim": 4,
    },
    "vehicle_direct_continuous_style": {
        "display_name": "08 车辆数据 + 连续驾驶风格（直接预测）",
        "run_prefix": "FAIR08_车辆数据_连续驾驶风格_直接预测",
        "conditioning_mode": "vehicle_direct",
        "teacher_forcing_ratio": 0.0,
        "event_loss_weight": 0.0,
        "enable_driver_style_context": True,
        "driver_style_vector_csv": str(STYLE_VECTOR_PATH),
        "driver_style_embed_dim": 4,
        "driver_style_include_iqr": True,
    },
    "vehicle_direct_coarse_fine": {
        "display_name": "09 车辆数据 + 粗细双头（无显式事件）",
        "run_prefix": "FAIR09_车辆数据_粗细双头_无显式事件",
        "conditioning_mode": "vehicle_direct_coarse_fine",
        "teacher_forcing_ratio": 0.0,
        "event_loss_weight": 0.0,
    },
    "vehicle_direct_coarse_fine_teacher_state": {
        "display_name": "10 车辆数据 + 粗细双头 + 教师状态（无显式事件）",
        "run_prefix": "FAIR10_车辆数据_粗细双头_教师状态_无显式事件",
        "conditioning_mode": "vehicle_direct_coarse_fine",
        "teacher_forcing_ratio": 0.0,
        "event_loss_weight": 0.0,
        "enable_teacher_state_context": True,
        "teacher_state_mode": "pca_latent",
        "teacher_state_dim": 4,
    },
    "vehicle_direct_coarse_fine_continuous_style": {
        "display_name": "11 车辆数据 + 粗细双头 + 连续驾驶风格（无显式事件）",
        "run_prefix": "FAIR11_车辆数据_粗细双头_连续驾驶风格_无显式事件",
        "conditioning_mode": "vehicle_direct_coarse_fine",
        "teacher_forcing_ratio": 0.0,
        "event_loss_weight": 0.0,
        "enable_driver_style_context": True,
        "driver_style_vector_csv": str(STYLE_VECTOR_PATH),
        "driver_style_embed_dim": 4,
        "driver_style_include_iqr": True,
    },
    "vehicle_direct_coarse_fine_teacher_state_continuous_style": {
        "display_name": "12 车辆数据 + 粗细双头 + 教师状态 + 连续驾驶风格（无显式事件）",
        "run_prefix": "FAIR12_车辆数据_粗细双头_教师状态_连续驾驶风格_无显式事件",
        "conditioning_mode": "vehicle_direct_coarse_fine",
        "teacher_forcing_ratio": 0.0,
        "event_loss_weight": 0.0,
        "enable_teacher_state_context": True,
        "teacher_state_mode": "pca_latent",
        "teacher_state_dim": 4,
        "enable_driver_style_context": True,
        "driver_style_vector_csv": str(STYLE_VECTOR_PATH),
        "driver_style_embed_dim": 4,
        "driver_style_include_iqr": True,
    },
    "vehicle_direct_coarse_fine_raw_physio_continuous_style": {
        "display_name": "13 车辆数据 + 粗细双头 + 原始生理数据 + 连续驾驶风格（无显式事件）",
        "run_prefix": "FAIR13_车辆数据_粗细双头_原始生理数据_连续驾驶风格_无显式事件",
        "conditioning_mode": "vehicle_direct_coarse_fine",
        "teacher_forcing_ratio": 0.0,
        "event_loss_weight": 0.0,
        "enable_teacher_state_context": True,
        "teacher_state_mode": "raw_physio",
        "teacher_state_dim": 12,
        "enable_driver_style_context": True,
        "driver_style_vector_csv": str(STYLE_VECTOR_PATH),
        "driver_style_embed_dim": 4,
        "driver_style_include_iqr": True,
    },
    "vehicle_direct_coarse_fine_raw_physio_no_eeg_continuous_style": {
        "display_name": "E7C 车辆数据 + 粗细双头 + raw 无 EEG 生理信号 + 连续驾驶风格",
        "run_prefix": "E7C_raw_no_EEG_physio_continuous_style",
        "conditioning_mode": "vehicle_direct_coarse_fine",
        "teacher_forcing_ratio": 0.0,
        "event_loss_weight": 0.0,
        "enable_teacher_state_context": True,
        "teacher_state_mode": "raw_physio_no_eeg",
        "teacher_state_dim": 4,
        "enable_driver_style_context": True,
        "driver_style_vector_csv": str(STYLE_VECTOR_PATH),
        "driver_style_embed_dim": 4,
        "driver_style_include_iqr": True,
    },
    "vehicle_direct_coarse_fine_raw_hr_only_continuous_style": {
        "display_name": "E10A 车辆数据 + 粗细双头 + HR-only + 连续驾驶风格",
        "run_prefix": "E10A_HR_only_continuous_style",
        "conditioning_mode": "vehicle_direct_coarse_fine",
        "teacher_forcing_ratio": 0.0,
        "event_loss_weight": 0.0,
        "enable_teacher_state_context": True,
        "teacher_state_mode": "raw_hr_only",
        "teacher_state_dim": 1,
        "enable_driver_style_context": True,
        "driver_style_vector_csv": str(STYLE_VECTOR_PATH),
        "driver_style_embed_dim": 4,
        "driver_style_include_iqr": True,
    },
    "vehicle_direct_coarse_fine_raw_eda_only_continuous_style": {
        "display_name": "E10B 车辆数据 + 粗细双头 + EDA-only + 连续驾驶风格",
        "run_prefix": "E10B_EDA_only_continuous_style",
        "conditioning_mode": "vehicle_direct_coarse_fine",
        "teacher_forcing_ratio": 0.0,
        "event_loss_weight": 0.0,
        "enable_teacher_state_context": True,
        "teacher_state_mode": "raw_eda_only",
        "teacher_state_dim": 2,
        "enable_driver_style_context": True,
        "driver_style_vector_csv": str(STYLE_VECTOR_PATH),
        "driver_style_embed_dim": 4,
        "driver_style_include_iqr": True,
    },
    "vehicle_direct_coarse_fine_raw_emg_only_continuous_style": {
        "display_name": "E10C 车辆数据 + 粗细双头 + EMG-only + 连续驾驶风格",
        "run_prefix": "E10C_EMG_only_continuous_style",
        "conditioning_mode": "vehicle_direct_coarse_fine",
        "teacher_forcing_ratio": 0.0,
        "event_loss_weight": 0.0,
        "enable_teacher_state_context": True,
        "teacher_state_mode": "raw_emg_only",
        "teacher_state_dim": 1,
        "enable_driver_style_context": True,
        "driver_style_vector_csv": str(STYLE_VECTOR_PATH),
        "driver_style_embed_dim": 4,
        "driver_style_include_iqr": True,
    },
    "vehicle_direct_coarse_fine_raw_eeg_only_continuous_style": {
        "display_name": "E7B 车辆数据 + 粗细双头 + raw EEG 单独 + 连续驾驶风格",
        "run_prefix": "E7B_raw_EEG_only_continuous_style",
        "conditioning_mode": "vehicle_direct_coarse_fine",
        "teacher_forcing_ratio": 0.0,
        "event_loss_weight": 0.0,
        "enable_teacher_state_context": True,
        "teacher_state_mode": "raw_eeg_only",
        "teacher_state_dim": 8,
        "enable_driver_style_context": True,
        "driver_style_vector_csv": str(STYLE_VECTOR_PATH),
        "driver_style_embed_dim": 4,
        "driver_style_include_iqr": True,
    },
    "vehicle_direct_coarse_fine_semantic_driver_state_continuous_style": {
        "display_name": "14 车辆数据 + 粗细双头 + 生理状态量 + 连续驾驶风格（无显式事件）",
        "run_prefix": "FAIR14_车辆数据_粗细双头_生理状态量_连续驾驶风格_无显式事件",
        "conditioning_mode": "vehicle_direct_coarse_fine",
        "teacher_forcing_ratio": 0.0,
        "event_loss_weight": 0.0,
        "enable_teacher_state_context": True,
        "teacher_state_mode": "semantic_driver_state",
        "teacher_state_dim": 6,
        "enable_driver_style_context": True,
        "driver_style_vector_csv": str(STYLE_VECTOR_PATH),
        "driver_style_embed_dim": 4,
        "driver_style_include_iqr": True,
    },
    "vehicle_direct_coarse_fine_semantic_driver_state_eeg_only_continuous_style": {
        "display_name": "E7A 车辆数据 + 粗细双头 + EEG 单独生理状态 + 连续驾驶风格",
        "run_prefix": "E7A_EEG_only_semantic_state_continuous_style",
        "conditioning_mode": "vehicle_direct_coarse_fine",
        "teacher_forcing_ratio": 0.0,
        "event_loss_weight": 0.0,
        "enable_teacher_state_context": True,
        "teacher_state_mode": "semantic_driver_state_eeg_only",
        "teacher_state_dim": 6,
        "enable_driver_style_context": True,
        "driver_style_vector_csv": str(STYLE_VECTOR_PATH),
        "driver_style_embed_dim": 4,
        "driver_style_include_iqr": True,
    },
    "vehicle_direct_coarse_fine_semantic_driver_state_no_eeg_continuous_style": {
        "display_name": "17 车辆数据 + 粗细双头 + 无脑电生理状态量 + 连续驾驶风格（无显式事件）",
        "run_prefix": "FAIR17_车辆数据_粗细双头_无脑电生理状态量_连续驾驶风格_无显式事件",
        "conditioning_mode": "vehicle_direct_coarse_fine",
        "teacher_forcing_ratio": 0.0,
        "event_loss_weight": 0.0,
        "enable_teacher_state_context": True,
        "teacher_state_mode": "semantic_driver_state_no_eeg",
        "teacher_state_dim": 6,
        "enable_driver_style_context": True,
        "driver_style_vector_csv": str(STYLE_VECTOR_PATH),
        "driver_style_embed_dim": 4,
        "driver_style_include_iqr": True,
    },
    "vehicle_direct_coarse_fine_baseline_driver_state": {
        "display_name": "15 车辆数据 + 粗细双头 + 基线校正生理状态量（无显式事件）",
        "run_prefix": "FAIR15_车辆数据_粗细双头_基线校正生理状态量_无显式事件",
        "conditioning_mode": "vehicle_direct_coarse_fine",
        "teacher_forcing_ratio": 0.0,
        "event_loss_weight": 0.0,
        "enable_teacher_state_context": True,
        "teacher_state_mode": "semantic_driver_state_local_delta",
        "teacher_state_dim": 12,
    },
    "vehicle_direct_coarse_fine_baseline_driver_state_continuous_style": {
        "display_name": "16 车辆数据 + 粗细双头 + 基线校正生理状态量 + 连续驾驶风格（无显式事件）",
        "run_prefix": "FAIR16_车辆数据_粗细双头_基线校正生理状态量_连续驾驶风格_无显式事件",
        "conditioning_mode": "vehicle_direct_coarse_fine",
        "teacher_forcing_ratio": 0.0,
        "event_loss_weight": 0.0,
        "enable_teacher_state_context": True,
        "teacher_state_mode": "semantic_driver_state_local_delta",
        "teacher_state_dim": 12,
        "enable_driver_style_context": True,
        "driver_style_vector_csv": str(STYLE_VECTOR_PATH),
        "driver_style_embed_dim": 4,
        "driver_style_include_iqr": True,
    },
}


def _apply_config(args: Namespace, config: dict[str, Any]) -> None:
    for key, value in config.items():
        if key == "display_name":
            continue
        setattr(args, key, value)


def build_args(variant_key: str) -> Namespace:
    if variant_key not in VARIANT_CONFIGS:
        raise KeyError(f"Unknown comparison variant: {variant_key}")
    original_argv = sys.argv[:]
    try:
        sys.argv = [original_argv[0]]
        args = parse_args()
    finally:
        sys.argv = original_argv
    _apply_config(args, COMMON_FULLRUN_CONFIG)
    _apply_config(args, VARIANT_CONFIGS[variant_key])
    return args


def run_variant(variant_key: str) -> dict[str, Any]:
    args = build_args(variant_key)
    display_name = str(VARIANT_CONFIGS[variant_key]["display_name"])
    print(f"当前实验: {display_name}")
    print(f"运行前缀: {args.run_prefix}")
    print(f"共同配置文件: {MANIFEST_PATH}")
    result = train_one_run(args)
    try:
        plot_result = save_prediction_plots_for_run(
            run_root=result["run_root"],
            split="test",
            case_file=THIS_DIR / "shared_prediction_cases_test.csv",
            max_cases=8,
            batch_size=int(args.batch_size),
            device=str(args.device),
        )
        result["prediction_plot_result"] = plot_result
        print(f"??????: {plot_result['figures_dir']}")
        print(f"?????: {plot_result['overview_path']}")
        print(f"??????: {plot_result['case_file']}")
    except Exception as exc:
        result["prediction_plot_error"] = str(exc)
        print(f"???????: {exc}")
    print(f"输出文件夹: {result['run_root']}")
    print(f"最佳验证集 steer RMSE: {result['best_val_steer_rmse']}")
    print(f"测试集 steer RMSE: {result['test_metrics']['steer_rmse']}")
    return result
