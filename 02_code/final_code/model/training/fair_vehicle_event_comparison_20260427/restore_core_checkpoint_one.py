# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
TRAINING_DIR = THIS_DIR.parent
PROJECT_ROOT = TRAINING_DIR.parents[3]
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from prediction_plotting import save_prediction_plots_for_run  # noqa: E402
from run_event_conditioned_trajectory_baseline import parse_args, train_one_run  # noqa: E402


REPORT_DIR = PROJECT_ROOT / "04_project_logs" / "reports" / "restore_checkpoint_audit_20260510"
STYLE_VECTOR_PATH = PROJECT_ROOT / "04_project_logs" / "reports" / "style_probe_artifacts" / "driver_style_vectors.csv"
SHARED_CASE_FILE = THIS_DIR / "shared_prediction_cases_test.csv"


COMMON_CONFIG: dict[str, Any] = {
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


CANDIDATES: dict[str, dict[str, Any]] = {
    "E2": {
        "label": "粗细双头 + 连续驾驶风格",
        "run_prefix": "RESTORE_E2_粗细双头_连续驾驶风格",
        "conditioning_mode": "vehicle_direct_coarse_fine",
        "teacher_forcing_ratio": 0.0,
        "event_loss_weight": 0.0,
        "enable_driver_style_context": True,
        "driver_style_vector_csv": str(STYLE_VECTOR_PATH),
        "driver_style_embed_dim": 4,
        "driver_style_include_iqr": True,
    },
    "E4": {
        "label": "含 EEG 生理状态 + 连续驾驶风格教师",
        "run_prefix": "RESTORE_E4_含EEG生理状态_连续驾驶风格",
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
    "E5A": {
        "label": "脑电教师蒸馏学生，不加生理输入",
        "run_prefix": "RESTORE_E5A_脑电教师_无生理学生",
        "conditioning_mode": "vehicle_direct_coarse_fine",
        "teacher_forcing_ratio": 0.0,
        "event_loss_weight": 0.0,
        "enable_driver_style_context": True,
        "driver_style_vector_csv": str(STYLE_VECTOR_PATH),
        "driver_style_embed_dim": 4,
        "driver_style_include_iqr": True,
        "distill_weight": 0.20,
        "distill_tail_weight": 0.05,
    },
    "E6": {
        "label": "脑电教师学生 + 幅值方向物理损失",
        "run_prefix": "RESTORE_E6_脑电教师_物理损失",
        "conditioning_mode": "vehicle_direct_coarse_fine",
        "teacher_forcing_ratio": 0.0,
        "event_loss_weight": 0.0,
        "enable_driver_style_context": True,
        "driver_style_vector_csv": str(STYLE_VECTOR_PATH),
        "driver_style_embed_dim": 4,
        "driver_style_include_iqr": True,
        "distill_weight": 0.20,
        "distill_tail_weight": 0.05,
        "steer_amp_loss_weight": 0.10,
        "steer_direction_loss_weight": 0.05,
        "steer_amp_target_ratio": 0.85,
    },
    "E7C": {
        "label": "raw HR+EDA+EMG 融合 + 连续驾驶风格",
        "run_prefix": "RESTORE_E7C_raw无EEG生理融合",
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
    "E10C": {
        "label": "EMG-only + 连续驾驶风格",
        "run_prefix": "RESTORE_E10C_EMG单信号",
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
    "E11A": {
        "label": "脑电教师 + EMG-only 学生 + 连续驾驶风格",
        "run_prefix": "RESTORE_E11A_脑电教师_EMG学生",
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
        "distill_weight": 0.20,
        "distill_tail_weight": 0.05,
    },
}


def _fresh_args() -> argparse.Namespace:
    original_argv = sys.argv[:]
    try:
        sys.argv = [original_argv[0]]
        return parse_args()
    finally:
        sys.argv = original_argv


def _apply(args: argparse.Namespace, config: dict[str, Any]) -> None:
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)


def _compact_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    test = metrics["test_metrics"]
    selection = test["selection_summary"]
    return {
        "test_steer_rmse": float(test["steer_rmse"]),
        "primary_rmse": float(selection.get("rmse_primary_abs_steer", selection.get("primary_rmse_score", float("nan")))),
        "tail_rmse": float(selection.get("rmse_tail_abs_steer", float("nan"))),
        "peak_err_s": float(selection.get("peak_time_abs_err_s", float("nan"))),
        "selection": float(selection.get("selection_score", float("nan"))),
    }


def build_train_args(candidate_id: str, seed: int, teacher_checkpoint: str = "", smoke: bool = False) -> argparse.Namespace:
    if candidate_id not in CANDIDATES:
        raise KeyError(f"未知候选版本: {candidate_id}")
    args = _fresh_args()
    _apply(args, COMMON_CONFIG)
    _apply(args, CANDIDATES[candidate_id])
    args.seed = int(seed)
    args.run_prefix = f"{CANDIDATES[candidate_id]['run_prefix']}_seed{seed}"
    if smoke:
        args.smoke_test = True
        args.smoke_epochs = 1
        args.smoke_train_samples = 96
        args.smoke_val_samples = 32
        args.smoke_test_samples = 32
    if candidate_id in {"E5A", "E6", "E11A"}:
        if not teacher_checkpoint:
            raise ValueError(f"{candidate_id} 需要传入 --teacher-checkpoint")
        args.distill_teacher_checkpoint = str(teacher_checkpoint)
    return args


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, choices=sorted(CANDIDATES))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--teacher-checkpoint", default="")
    parser.add_argument("--report-dir", default=str(REPORT_DIR))
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parsed = parser.parse_args()

    report_dir = Path(parsed.report_dir)
    run_record_dir = report_dir / "remote_restore_records"
    run_record_dir.mkdir(parents=True, exist_ok=True)

    train_args = build_train_args(
        parsed.candidate,
        seed=int(parsed.seed),
        teacher_checkpoint=str(parsed.teacher_checkpoint),
        smoke=bool(parsed.smoke),
    )
    start = time.time()
    result = train_one_run(train_args)
    plot_result: dict[str, Any] = {}
    if not parsed.skip_plots:
        try:
            plot_result = save_prediction_plots_for_run(
                run_root=result["run_root"],
                split="test",
                case_file=SHARED_CASE_FILE,
                max_cases=8,
                batch_size=int(train_args.batch_size),
                device=str(train_args.device),
            )
        except Exception as exc:
            plot_result = {"plot_error": str(exc)}

    record = {
        "candidate": parsed.candidate,
        "label": CANDIDATES[parsed.candidate]["label"],
        "seed": int(parsed.seed),
        "run_root": str(result["run_root"]),
        "best_model": str(Path(result["run_root"]) / "best_model.pt"),
        "teacher_checkpoint": str(parsed.teacher_checkpoint),
        "elapsed_sec": round(time.time() - start, 3),
        **_compact_metrics(result),
        "plot_result": plot_result,
    }
    out_path = run_record_dir / f"{parsed.candidate}_seed{parsed.seed}_{int(time.time())}.json"
    out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
