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
DEFAULT_RUN_RECORD = REPORTS_DIR / "style_physio_eeg_e10_non_eeg_signal_runs_20260509.csv"

E10_VARIANTS: dict[str, dict[str, str]] = {
    "E10A": {
        "variant_key": "vehicle_direct_coarse_fine_raw_hr_only_continuous_style",
        "label": "HR-only + 连续驾驶风格",
        "question": "HR 单独作为非 EEG 生理信号是否有预测价值",
    },
    "E10B": {
        "variant_key": "vehicle_direct_coarse_fine_raw_eda_only_continuous_style",
        "label": "EDA-only + 连续驾驶风格",
        "question": "EDA tonic/phasic 作为一组单独输入是否有预测价值",
    },
    "E10C": {
        "variant_key": "vehicle_direct_coarse_fine_raw_emg_only_continuous_style",
        "label": "EMG-only + 连续驾驶风格",
        "question": "EMG RMS 单独作为非 EEG 生理信号是否有预测价值",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan or execute E10 non-EEG physiological signal attribution runs. "
            "These runs keep the FAIR protocol fixed and only change HR/EDA/EMG input groups."
        )
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026])
    parser.add_argument("--experiments", nargs="+", default=["E10A", "E10B", "E10C"], choices=sorted(E10_VARIANTS))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    parser.add_argument("--run-record", default=str(DEFAULT_RUN_RECORD))
    parser.add_argument("--smoke", action="store_true", help="Run smoke settings from the training script.")
    parser.add_argument("--execute", action="store_true", help="Actually train. Omit for dry-run plan.")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def _planned_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        for experiment_id in args.experiments:
            info = E10_VARIANTS[experiment_id]
            run_args = build_args(info["variant_key"])
            run_args.seed = int(seed)
            if args.device is not None:
                run_args.device = str(args.device)
            run_args.smoke_test = bool(args.smoke)
            run_args.run_prefix = f"{experiment_id}_{info['label']}_seed{seed}"
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "seed": int(seed),
                    "variant_key": info["variant_key"],
                    "label": info["label"],
                    "question": info["question"],
                    "run_prefix": run_args.run_prefix,
                    "manifest": str(run_args.manifest),
                    "device": str(run_args.device),
                    "epochs": int(run_args.epochs),
                    "batch_size": int(run_args.batch_size),
                    "lr": float(run_args.lr),
                    "conditioning_mode": str(run_args.conditioning_mode),
                    "teacher_state_mode": str(getattr(run_args, "teacher_state_mode", "")),
                    "teacher_state_dim": int(getattr(run_args, "teacher_state_dim", 0)),
                    "enable_teacher_state_context": bool(getattr(run_args, "enable_teacher_state_context", False)),
                    "enable_driver_style_context": bool(getattr(run_args, "enable_driver_style_context", False)),
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
            key = (
                row.get("experiment_id"),
                row.get("seed"),
                row.get("variant_key"),
                row.get("smoke_test"),
            )
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
    run_args = build_args(str(row["variant_key"]))
    run_args.seed = int(row["seed"])
    run_args.device = str(row["device"])
    run_args.run_prefix = str(row["run_prefix"])
    run_args.smoke_test = bool(row["smoke_test"])
    result = train_one_run(run_args)
    row["run_root"] = str(result["run_root"])
    row["best_val_steer_rmse"] = float(result["best_val_steer_rmse"])
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
        print(f"Dry-run only. Planned {len(rows)} runs.")
        print(f"Plan CSV: {plan_path}")
        for row in rows:
            print(
                f"{row['experiment_id']} seed={row['seed']} "
                f"mode={row['teacher_state_mode']} smoke={row['smoke_test']} "
                f"question={row['question']}"
            )
        return

    executed: list[dict[str, Any]] = []
    for row in rows:
        print(f"Running {row['experiment_id']} seed={row['seed']} - {row['label']}", flush=True)
        executed.append(_run_one(row, no_plots=bool(args.no_plots)))
        _write_rows(record_path, executed, merge_existing=True)
    print(f"Completed {len(executed)} runs.")
    print(f"Run record: {record_path}")


if __name__ == "__main__":
    main()

