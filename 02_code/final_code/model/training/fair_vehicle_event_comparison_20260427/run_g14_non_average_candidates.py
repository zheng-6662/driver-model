# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from common_compare_runner import build_args
from prediction_plotting import save_prediction_plots_for_run
from run_event_conditioned_trajectory_baseline import train_one_run


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[4]
REPORT_DIR = PROJECT_ROOT / "04_project_logs" / "reports" / "g14_non_average_prediction_20260510"
DEFAULT_RUN_RECORD = REPORT_DIR / "g14_candidate_run_log.csv"


@dataclass(frozen=True)
class Candidate:
    experiment_id: str
    base_variant: str
    label: str
    purpose: str
    extra_args: dict[str, Any]
    needs_teacher: bool = False


COMMON_FULL = {
    "device": "cuda",
    "epochs": 40,
    "min_epochs": 40,
    "patience": 99,
    "batch_size": 64,
    "lr": 1e-3,
    "selection_mode": "legacy_rmse",
    "teacher_forcing_ratio": 0.0,
    "event_loss_weight": 0.0,
}


CANDIDATES: dict[str, Candidate] = {
    "G14A": Candidate(
        experiment_id="G14A",
        base_variant="vehicle_direct_coarse_fine_continuous_style",
        label="连续风格 + 响应先判别 + 四候选轨迹",
        purpose="验证多候选输出能否减少单条平均化预测，先不加入肌电。",
        extra_args={
            "enable_response_type_head": True,
            "enable_response_type_condition": True,
            "response_type_loss_weight": 0.25,
            "num_trajectory_candidates": 4,
            "candidate_delta_scale": 0.75,
            "trajectory_loss_weight": 0.35,
            "multi_candidate_loss_weight": 0.85,
            "candidate_selector_loss_weight": 0.15,
        },
    ),
    "G14B": Candidate(
        experiment_id="G14B",
        base_variant="vehicle_direct_coarse_fine_raw_emg_only_continuous_style",
        label="连续风格 + 肌电 + 响应先判别 + 四候选轨迹",
        purpose="验证肌电是否能帮助模型在推理前判断响应强弱、方向和形态，从而减少平均化。",
        extra_args={
            "enable_response_type_head": True,
            "enable_response_type_condition": True,
            "response_type_loss_weight": 0.25,
            "num_trajectory_candidates": 4,
            "candidate_delta_scale": 0.75,
            "trajectory_loss_weight": 0.35,
            "multi_candidate_loss_weight": 0.85,
            "candidate_selector_loss_weight": 0.15,
        },
    ),
    "G14C": Candidate(
        experiment_id="G14C",
        base_variant="vehicle_direct_coarse_fine_raw_emg_only_continuous_style",
        label="肌电 + 四候选轨迹 + 幅值方向约束",
        purpose="在 G14B 基础上直接压制真实大幅响应被预测成小幅、主峰方向跑错的问题。",
        extra_args={
            "enable_response_type_head": True,
            "enable_response_type_condition": True,
            "response_type_loss_weight": 0.25,
            "num_trajectory_candidates": 4,
            "candidate_delta_scale": 0.75,
            "trajectory_loss_weight": 0.30,
            "multi_candidate_loss_weight": 0.90,
            "candidate_selector_loss_weight": 0.15,
            "steer_amp_loss_weight": 0.08,
            "steer_direction_loss_weight": 0.04,
            "steer_amp_target_ratio": 0.85,
            "steer_direction_major_only": True,
        },
    ),
    "G14D": Candidate(
        experiment_id="G14D",
        base_variant="vehicle_direct_coarse_fine_raw_emg_only_continuous_style",
        label="肌电 + 响应先判别 + 八候选轨迹",
        purpose="检验候选数量增加后，是否能覆盖更多反向修正、多段修正和大幅响应，而不是继续平均化。",
        extra_args={
            "enable_response_type_head": True,
            "enable_response_type_condition": True,
            "response_type_loss_weight": 0.25,
            "num_trajectory_candidates": 8,
            "candidate_delta_scale": 0.85,
            "trajectory_loss_weight": 0.25,
            "multi_candidate_loss_weight": 1.00,
            "candidate_selector_loss_weight": 0.20,
        },
    ),
    "G14E": Candidate(
        experiment_id="G14E",
        base_variant="vehicle_direct_coarse_fine_raw_emg_only_continuous_style",
        label="肌电 + 稳定候选选择 + 四候选轨迹",
        purpose="把候选选择从临时最小误差改成固定响应类型标签，检验模型是否能更稳定地选中方向、幅值和形态更合理的轨迹。",
        extra_args={
            "enable_response_type_head": True,
            "enable_response_type_condition": True,
            "response_type_loss_weight": 0.35,
            "num_trajectory_candidates": 4,
            "candidate_delta_scale": 0.75,
            "candidate_base_mode": "learned_delta",
            "multi_candidate_target_mode": "response_type",
            "trajectory_loss_weight": 0.15,
            "multi_candidate_loss_weight": 1.00,
            "candidate_selector_loss_weight": 0.35,
        },
    ),
    "G14F": Candidate(
        experiment_id="G14F",
        base_variant="vehicle_direct_coarse_fine_raw_emg_only_continuous_style",
        label="肌电 + 训练集响应原型 + 原型残差修正",
        purpose="让候选轨迹先对应训练集中的小幅、高幅、反向修正、多段修正原型，再学习残差，避免候选自由漂移。",
        extra_args={
            "enable_response_type_head": True,
            "enable_response_type_condition": True,
            "response_type_loss_weight": 0.35,
            "num_trajectory_candidates": 4,
            "candidate_delta_scale": 0.35,
            "candidate_base_mode": "response_prototype",
            "multi_candidate_target_mode": "response_type",
            "trajectory_loss_weight": 0.05,
            "multi_candidate_loss_weight": 1.20,
            "candidate_selector_loss_weight": 0.45,
        },
    ),
    "G14G": Candidate(
        experiment_id="G14G",
        base_variant="vehicle_direct_coarse_fine_raw_emg_only_continuous_style",
        label="脑电教师 + 肌电 + 响应原型候选",
        purpose="重新组合脑电教师和肌电学生：脑电只在训练阶段提供软目标，肌电和响应原型在推理阶段帮助选择候选轨迹。",
        extra_args={
            "enable_response_type_head": True,
            "enable_response_type_condition": True,
            "response_type_loss_weight": 0.30,
            "num_trajectory_candidates": 4,
            "candidate_delta_scale": 0.35,
            "candidate_base_mode": "response_prototype",
            "multi_candidate_target_mode": "response_type",
            "trajectory_loss_weight": 0.05,
            "multi_candidate_loss_weight": 1.10,
            "candidate_selector_loss_weight": 0.40,
            "distill_weight": 0.15,
            "distill_tail_weight": 0.05,
            "distill_hardcase_weighting": True,
            "distill_hardcase_extra_weight": 0.50,
        },
        needs_teacher=True,
    ),
    "G14H": Candidate(
        experiment_id="G14H",
        base_variant="vehicle_direct_coarse_fine_raw_emg_only_continuous_style",
        label="肌电 + 响应原型候选 + 幅值方向约束",
        purpose="在 G14F 的训练集响应原型基础上，再压制大幅响应被预测过小和主峰方向错误的问题。",
        extra_args={
            "enable_response_type_head": True,
            "enable_response_type_condition": True,
            "response_type_loss_weight": 0.30,
            "num_trajectory_candidates": 4,
            "candidate_delta_scale": 0.45,
            "candidate_base_mode": "response_prototype",
            "multi_candidate_target_mode": "response_type",
            "trajectory_loss_weight": 0.08,
            "multi_candidate_loss_weight": 1.10,
            "candidate_selector_loss_weight": 0.40,
            "steer_amp_loss_weight": 0.08,
            "steer_direction_loss_weight": 0.04,
            "steer_amp_target_ratio": 0.85,
            "steer_direction_major_only": True,
        },
    ),
}


def _find_teacher_checkpoint(seed: int) -> Path | None:
    csv_paths = [
        PROJECT_ROOT / "04_project_logs" / "reports" / "style_physio_eeg_e3_e4_runs_20260507.csv",
        PROJECT_ROOT / "04_project_logs" / "reports" / "g13_model_breakthrough_20260510" / "g13_teacher_run_log.csv",
    ]
    for csv_path in csv_paths:
        if not csv_path.exists():
            continue
        for row in pd.read_csv(csv_path).to_dict("records"):
            if str(row.get("seed", "")) != str(seed):
                continue
            run_root = Path(str(row.get("run_root", "")))
            ckpt = run_root / "best_model.pt"
            if ckpt.exists():
                return ckpt
            teacher_ckpt = Path(str(row.get("teacher_checkpoint", "")))
            if teacher_ckpt.exists():
                return teacher_ckpt
    run_root = PROJECT_ROOT / "tmp" / "event_conditioned_runs"
    patterns = [
        f"G14T_*_seed{seed}_*",
        f"G14T_兼容脑电教师_seed{seed}_*",
        f"RESTORE_E4_*_seed{seed}_*",
        f"E4_粗细双头 + 生理状态量 + 连续驾驶风格_seed{seed}_*",
        f"G13T_脑电教师准备_seed{seed}_*",
    ]
    for pattern in patterns:
        for run_dir in sorted(run_root.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True):
            ckpt = run_dir / "best_model.pt"
            if ckpt.exists():
                return ckpt
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="执行 G14 非平均化预测候选模型。")
    parser.add_argument("--experiments", nargs="+", default=["G14A", "G14B", "G14C", "G14D"], choices=sorted(CANDIDATES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026])
    parser.add_argument("--run-record", default=str(DEFAULT_RUN_RECORD))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def apply_config(args: argparse.Namespace, values: dict[str, Any]) -> None:
    for key, value in values.items():
        setattr(args, key, value)


def planned_rows(cli_args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in cli_args.seeds:
        for exp_id in cli_args.experiments:
            cand = CANDIDATES[exp_id]
            run_args = build_args(cand.base_variant)
            apply_config(run_args, COMMON_FULL)
            apply_config(run_args, cand.extra_args)
            run_args.seed = int(seed)
            run_args.smoke_test = bool(cli_args.smoke)
            if cli_args.smoke:
                run_args.epochs = 1
                run_args.min_epochs = 1
                run_args.patience = 1
            run_args.run_prefix = f"{cand.experiment_id}_{cand.label}_seed{seed}"
            rows.append(
                {
                    "experiment_id": cand.experiment_id,
                    "seed": int(seed),
                    "label": cand.label,
                    "purpose": cand.purpose,
                    "base_variant": cand.base_variant,
                    "run_prefix": run_args.run_prefix,
                    "smoke_test": bool(cli_args.smoke),
                    "epochs": int(run_args.epochs),
                    "batch_size": int(run_args.batch_size),
                    "lr": float(run_args.lr),
                    "conditioning_mode": str(run_args.conditioning_mode),
                    "enable_teacher_state_context": bool(getattr(run_args, "enable_teacher_state_context", False)),
                    "teacher_state_mode": str(getattr(run_args, "teacher_state_mode", "")),
                    "enable_driver_style_context": bool(getattr(run_args, "enable_driver_style_context", False)),
                    "enable_response_type_condition": bool(getattr(run_args, "enable_response_type_condition", False)),
                    "num_trajectory_candidates": int(getattr(run_args, "num_trajectory_candidates", 1)),
                    "candidate_base_mode": str(getattr(run_args, "candidate_base_mode", "learned_delta")),
                    "multi_candidate_target_mode": str(getattr(run_args, "multi_candidate_target_mode", "oracle")),
                    "trajectory_loss_weight": float(getattr(run_args, "trajectory_loss_weight", 1.0)),
                    "multi_candidate_loss_weight": float(getattr(run_args, "multi_candidate_loss_weight", 0.0)),
                    "candidate_selector_loss_weight": float(getattr(run_args, "candidate_selector_loss_weight", 0.0)),
                    "steer_amp_loss_weight": float(getattr(run_args, "steer_amp_loss_weight", 0.0)),
                    "steer_direction_loss_weight": float(getattr(run_args, "steer_direction_loss_weight", 0.0)),
                    "needs_teacher": bool(cand.needs_teacher),
                    "teacher_checkpoint": "",
                    "run_root": "",
                }
            )
    return rows


def write_rows(path: Path, rows: list[dict[str, Any]], merge_existing: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output_rows = [dict(row) for row in rows]
    if merge_existing and path.exists():
        output_rows = pd.read_csv(path).to_dict("records") + output_rows
        deduped: dict[tuple[Any, ...], dict[str, Any]] = {}
        for row in output_rows:
            key = (row.get("experiment_id"), row.get("seed"), row.get("smoke_test"))
            deduped[key] = row
        output_rows = list(deduped.values())
    fieldnames: list[str] = []
    for row in output_rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)


def run_one(row: dict[str, Any], no_plots: bool) -> dict[str, Any]:
    cand = CANDIDATES[str(row["experiment_id"])]
    run_args = build_args(cand.base_variant)
    apply_config(run_args, COMMON_FULL)
    apply_config(run_args, cand.extra_args)
    if cand.needs_teacher:
        teacher_checkpoint = _find_teacher_checkpoint(seed=int(row["seed"]))
        if teacher_checkpoint is None:
            raise FileNotFoundError(f"{cand.experiment_id} 需要同 seed 脑电教师 checkpoint，但没有找到。")
        run_args.distill_teacher_checkpoint = str(teacher_checkpoint)
        row["teacher_checkpoint"] = str(teacher_checkpoint)
    run_args.seed = int(row["seed"])
    run_args.smoke_test = bool(row["smoke_test"])
    if bool(row["smoke_test"]):
        run_args.epochs = 1
        run_args.min_epochs = 1
        run_args.patience = 1
    run_args.run_prefix = str(row["run_prefix"])
    result = train_one_run(run_args)
    row["run_root"] = str(result["run_root"])
    row["best_val_steer_rmse"] = float(result["best_val_steer_rmse"])
    row["test_steer_rmse"] = float(result["test_metrics"]["steer_rmse"])
    selection = result["test_metrics"]["selection_summary"]
    row["test_tail_rmse"] = float(selection.get("rmse_tail_abs_steer", float("nan")))
    row["test_selection"] = float(selection.get("selection_score", float("nan")))
    row["best_epoch"] = int(result["best_epoch"])
    if not no_plots:
        try:
            plot_result = save_prediction_plots_for_run(
                run_root=result["run_root"],
                split="test",
                case_file=THIS_DIR / "shared_prediction_cases_test.csv",
                max_cases=8,
                batch_size=int(run_args.batch_size),
                device=str(run_args.device),
            )
            row["prediction_figures_dir"] = str(plot_result["figures_dir"])
            row["prediction_overview"] = str(plot_result["overview_path"])
        except Exception as exc:
            row["prediction_plot_error"] = str(exc)
    return row


def main() -> None:
    args = parse_args()
    rows = planned_rows(args)
    record_path = Path(args.run_record)
    if not args.execute:
        plan_path = record_path.with_name(f"{record_path.stem}_plan.csv")
        write_rows(plan_path, rows)
        print(f"已生成计划: {plan_path}")
        for row in rows:
            print(f"{row['experiment_id']} seed={row['seed']} {row['label']} candidates={row['num_trajectory_candidates']}")
        return

    executed: list[dict[str, Any]] = []
    for row in rows:
        print(f"开始 {row['experiment_id']} seed={row['seed']}：{row['label']}", flush=True)
        executed.append(run_one(row, no_plots=bool(args.no_plots)))
        write_rows(record_path, executed, merge_existing=True)
    print(f"完成 {len(executed)} 个 G14 候选实验。记录: {record_path}")


if __name__ == "__main__":
    main()
