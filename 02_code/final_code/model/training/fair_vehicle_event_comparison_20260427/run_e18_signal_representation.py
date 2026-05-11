# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import pandas as pd

from common_compare_runner import build_args
from prediction_plotting import save_prediction_plots_for_run
from run_event_conditioned_trajectory_baseline import train_one_run


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[4]
REPORTS_DIR = PROJECT_ROOT / "04_project_logs" / "reports"
DEFAULT_RUN_RECORD = REPORTS_DIR / "style_physio_eeg_e18_signal_representation_runs_20260511.csv"


def _common_label(signal: str, method: str) -> str:
    return f"粗细双头 + 连续风格 + {signal} + {method}"


E18_VARIANTS: dict[str, dict[str, Any]] = {
    "E18A": {
        "signal": "心率",
        "method": "基线校正",
        "teacher_state_mode": "signal_current_delta_hr_only",
        "teacher_state_dim": 4,
        "question": "心率相对自身基线变化是否比心率原始值更有用。",
    },
    "E18B": {
        "signal": "皮电",
        "method": "基线校正",
        "teacher_state_mode": "signal_current_delta_eda_only",
        "teacher_state_dim": 6,
        "question": "皮电相对自身基线变化是否能减轻原始皮电直接输入的噪声问题。",
    },
    "E18C": {
        "signal": "肌电",
        "method": "基线校正",
        "teacher_state_mode": "signal_current_delta_emg_only",
        "teacher_state_dim": 4,
        "question": "肌电相对自身基线变化是否比肌电原始值更能反映控制紧张或操纵意图。",
    },
    "E18D": {
        "signal": "脑电",
        "method": "基线校正",
        "teacher_state_mode": "signal_current_delta_eeg_only",
        "teacher_state_dim": 18,
        "question": "脑电当前特征和同记录前序事件相对变化，是否比脑电原始直接输入更合理。",
    },
    "E18E": {
        "signal": "心率",
        "method": "数据自动表示",
        "teacher_state_mode": "signal_pca_hr_only",
        "teacher_state_dim": 2,
        "question": "不用人工权重，让训练集从心率当前值、变化值和有效性里提取低维表示是否有用。",
    },
    "E18F": {
        "signal": "皮电",
        "method": "数据自动表示",
        "teacher_state_mode": "signal_pca_eda_only",
        "teacher_state_dim": 3,
        "question": "不用人工权重，让训练集从皮电当前值、变化值和有效性里提取低维表示是否有用。",
    },
    "E18G": {
        "signal": "肌电",
        "method": "数据自动表示",
        "teacher_state_mode": "signal_pca_emg_only",
        "teacher_state_dim": 2,
        "question": "不用人工权重，让训练集从肌电当前值、变化值和有效性里提取低维表示是否有用。",
    },
    "E18H": {
        "signal": "脑电",
        "method": "数据自动表示",
        "teacher_state_mode": "signal_pca_eeg_only",
        "teacher_state_dim": 4,
        "question": "不用人工权重，让训练集从脑电当前值、前序事件变化和有效性里提取低维表示是否有用。",
    },
    "E18I": {
        "signal": "心率",
        "method": "任务相关状态",
        "teacher_state_mode": "signal_current_delta_hr_only",
        "teacher_state_dim": 4,
        "task_related": True,
        "question": "心率是否更适合帮助模型判断大幅、反向、多段、后段漂移等响应类型。",
    },
    "E18J": {
        "signal": "皮电",
        "method": "任务相关状态",
        "teacher_state_mode": "signal_current_delta_eda_only",
        "teacher_state_dim": 6,
        "task_related": True,
        "question": "皮电是否更适合帮助模型判断大幅、反向、多段、后段漂移等响应类型。",
    },
    "E18K": {
        "signal": "肌电",
        "method": "任务相关状态",
        "teacher_state_mode": "signal_current_delta_emg_only",
        "teacher_state_dim": 4,
        "task_related": True,
        "question": "肌电是否更适合帮助模型判断控制相关的响应类型，并改善困难样本。",
    },
    "E18L": {
        "signal": "脑电",
        "method": "任务相关状态",
        "teacher_state_mode": "signal_current_delta_eeg_only",
        "teacher_state_dim": 18,
        "task_related": True,
        "question": "脑电直接输入不佳时，是否可以通过任务相关响应类型监督发挥作用。",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行无人工权重的生理/脑电单信号表示筛选。")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026])
    parser.add_argument("--experiments", nargs="+", default=list(E18_VARIANTS), choices=sorted(E18_VARIANTS))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    parser.add_argument("--run-record", default=str(DEFAULT_RUN_RECORD))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def build_variant_args(experiment_id: str, seed: int, device: str | None, smoke: bool) -> argparse.Namespace:
    info = E18_VARIANTS[experiment_id]
    run_args = build_args("vehicle_direct_coarse_fine_continuous_style")
    run_args.seed = int(seed)
    if device is not None:
        run_args.device = str(device)
    run_args.smoke_test = bool(smoke)
    run_args.run_prefix = f"{experiment_id}_signal_representation_seed{seed}"
    run_args.conditioning_mode = "vehicle_direct_coarse_fine"
    run_args.teacher_forcing_ratio = 0.0
    run_args.event_loss_weight = 0.0
    run_args.enable_teacher_state_context = True
    run_args.teacher_state_mode = str(info["teacher_state_mode"])
    run_args.teacher_state_dim = int(info["teacher_state_dim"])
    run_args.enable_driver_style_context = True
    run_args.driver_style_embed_dim = 4
    run_args.driver_style_include_iqr = True
    if bool(info.get("task_related", False)):
        run_args.enable_response_type_head = True
        run_args.enable_response_type_condition = True
        run_args.response_type_use_context = True
        run_args.response_type_loss_weight = 0.20
        run_args.response_type_hidden_dim = 96
    return run_args


def _planned_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        for experiment_id in args.experiments:
            info = E18_VARIANTS[experiment_id]
            run_args = build_variant_args(experiment_id, int(seed), args.device, bool(args.smoke))
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "seed": int(seed),
                    "signal": info["signal"],
                    "method": info["method"],
                    "label": _common_label(str(info["signal"]), str(info["method"])),
                    "question": info["question"],
                    "run_prefix": str(run_args.run_prefix),
                    "manifest": str(run_args.manifest),
                    "device": str(run_args.device),
                    "epochs": int(run_args.epochs),
                    "batch_size": int(run_args.batch_size),
                    "lr": float(run_args.lr),
                    "conditioning_mode": str(run_args.conditioning_mode),
                    "teacher_state_mode": str(run_args.teacher_state_mode),
                    "teacher_state_dim": int(run_args.teacher_state_dim),
                    "enable_teacher_state_context": bool(run_args.enable_teacher_state_context),
                    "enable_driver_style_context": bool(run_args.enable_driver_style_context),
                    "enable_response_type_head": bool(getattr(run_args, "enable_response_type_head", False)),
                    "enable_response_type_condition": bool(getattr(run_args, "enable_response_type_condition", False)),
                    "response_type_use_context": bool(getattr(run_args, "response_type_use_context", False)),
                    "response_type_loss_weight": float(getattr(run_args, "response_type_loss_weight", 0.0)),
                    "smoke_test": bool(run_args.smoke_test),
                    "run_root": "",
                }
            )
    return rows


def _write_rows(path: Path, rows: list[dict[str, Any]], merge_existing: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output_rows = [dict(row) for row in rows]
    if merge_existing and path.exists():
        existing = pd.read_csv(path).to_dict("records")
        output_rows = [dict(row) for row in existing] + output_rows
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
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)


def _run_one(row: dict[str, Any], no_plots: bool) -> dict[str, Any]:
    run_args = build_variant_args(str(row["experiment_id"]), int(row["seed"]), str(row["device"]), bool(row["smoke_test"]))
    result = train_one_run(run_args)
    row["run_root"] = str(result["run_root"])
    row["best_val_steer_rmse"] = float(result["best_val_steer_rmse"])
    row["test_steer_rmse"] = float(result["test_metrics"]["steer_rmse"])
    row["test_tail_rmse"] = float(result["test_metrics"]["selection_summary"]["rmse_tail_abs_steer"])
    row["test_primary_rmse"] = float(result["test_metrics"]["selection_summary"]["overall_primary_steer_rmse"])
    row["test_selection"] = float(result["test_metrics"]["selection_summary"]["selection_score"])
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
    rows = _planned_rows(args)
    record_path = Path(args.run_record)
    if not args.execute:
        plan_path = record_path.with_name(f"{record_path.stem}_plan.csv")
        _write_rows(plan_path, rows)
        print(f"计划运行 {len(rows)} 组。")
        print(f"计划表: {plan_path}")
        for row in rows:
            print(
                f"{row['experiment_id']} seed={row['seed']} "
                f"signal={row['signal']} method={row['method']} "
                f"mode={row['teacher_state_mode']} task={row['enable_response_type_condition']}"
            )
        return

    executed: list[dict[str, Any]] = []
    for row in rows:
        print(f"开始运行 {row['experiment_id']} seed={row['seed']} - {row['label']}", flush=True)
        executed.append(_run_one(row, no_plots=bool(args.no_plots)))
        _write_rows(record_path, executed, merge_existing=True)
    print(f"完成 {len(executed)} 组。")
    print(f"运行记录: {record_path}")


if __name__ == "__main__":
    main()
