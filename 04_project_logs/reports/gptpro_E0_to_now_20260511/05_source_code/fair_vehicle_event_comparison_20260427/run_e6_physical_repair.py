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
DEFAULT_TEACHER_RUNS = REPORTS_DIR / "style_physio_eeg_e3_e4_runs_20260507.csv"
DEFAULT_RUN_RECORD = REPORTS_DIR / "style_physio_eeg_e6_physical_repair_runs_20260508.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan or execute E6 physical-repair runs. "
            "E6 keeps the E5A EEG-teacher/no-EEG-student route and adds small "
            "amplitude/direction auxiliary losses."
        )
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026])
    parser.add_argument("--teacher-runs-csv", default=str(DEFAULT_TEACHER_RUNS))
    parser.add_argument("--run-record", default=str(DEFAULT_RUN_RECORD))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    parser.add_argument("--distill-weight", type=float, default=0.20)
    parser.add_argument("--distill-tail-weight", type=float, default=0.05)
    parser.add_argument("--steer-amp-loss-weight", type=float, default=0.10)
    parser.add_argument("--steer-direction-loss-weight", type=float, default=0.05)
    parser.add_argument("--steer-amp-target-ratio", type=float, default=0.85)
    parser.add_argument("--smoke", action="store_true", help="Run smoke settings from the training script.")
    parser.add_argument("--execute", action="store_true", help="Actually train. Omit for dry-run plan.")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def _find_teacher_checkpoint(teacher_runs_csv: str | Path, seed: int) -> Path:
    path = Path(teacher_runs_csv)
    if not path.exists():
        raise FileNotFoundError(f"teacher runs CSV not found: {path}")
    runs = pd.read_csv(path)
    rows = runs[
        runs["experiment_id"].astype(str).eq("E4")
        & runs["seed"].astype(int).eq(int(seed))
        & runs["run_root"].fillna("").astype(str).str.len().gt(0)
    ]
    if rows.empty:
        raise ValueError(f"No completed E4 teacher row found for seed={seed} in {path}")
    run_root = Path(str(rows.iloc[0]["run_root"]))
    ckpt = run_root / "best_model.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"E4 teacher checkpoint not found: {ckpt}")
    return ckpt


def _planned_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        teacher_ckpt = _find_teacher_checkpoint(args.teacher_runs_csv, int(seed))
        run_args = build_args("vehicle_direct_coarse_fine_continuous_style")
        run_args.seed = int(seed)
        if args.device is not None:
            run_args.device = str(args.device)
        run_args.distill_teacher_checkpoint = str(teacher_ckpt)
        run_args.distill_weight = float(args.distill_weight)
        run_args.distill_tail_weight = float(args.distill_tail_weight)
        run_args.steer_amp_loss_weight = float(args.steer_amp_loss_weight)
        run_args.steer_direction_loss_weight = float(args.steer_direction_loss_weight)
        run_args.steer_amp_target_ratio = float(args.steer_amp_target_ratio)
        run_args.use_privileged_teacher = False
        run_args.smoke_test = bool(args.smoke)
        run_args.run_prefix = f"E6_physical_repair_EEGteacher_noEEGstudent_seed{seed}"
        rows.append(
            {
                "experiment_id": "E6",
                "seed": int(seed),
                "variant_key": "vehicle_direct_coarse_fine_continuous_style",
                "label": "EEG教师 + 无EEG学生 + 幅值/方向物理损失",
                "run_prefix": run_args.run_prefix,
                "teacher_checkpoint": str(teacher_ckpt),
                "manifest": str(run_args.manifest),
                "device": str(run_args.device),
                "epochs": int(run_args.epochs),
                "batch_size": int(run_args.batch_size),
                "lr": float(run_args.lr),
                "distill_weight": float(run_args.distill_weight),
                "distill_tail_weight": float(run_args.distill_tail_weight),
                "steer_amp_loss_weight": float(run_args.steer_amp_loss_weight),
                "steer_direction_loss_weight": float(run_args.steer_direction_loss_weight),
                "steer_amp_target_ratio": float(run_args.steer_amp_target_ratio),
                "conditioning_mode": str(run_args.conditioning_mode),
                "teacher_state_mode": str(getattr(run_args, "teacher_state_mode", "")),
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
                row.get("distill_weight"),
                row.get("distill_tail_weight"),
                row.get("steer_amp_loss_weight"),
                row.get("steer_direction_loss_weight"),
                row.get("steer_amp_target_ratio"),
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
    run_args.distill_teacher_checkpoint = str(row["teacher_checkpoint"])
    run_args.distill_weight = float(row["distill_weight"])
    run_args.distill_tail_weight = float(row["distill_tail_weight"])
    run_args.steer_amp_loss_weight = float(row["steer_amp_loss_weight"])
    run_args.steer_direction_loss_weight = float(row["steer_direction_loss_weight"])
    run_args.steer_amp_target_ratio = float(row["steer_amp_target_ratio"])
    run_args.use_privileged_teacher = False
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
                f"amp_w={row['steer_amp_loss_weight']} dir_w={row['steer_direction_loss_weight']} "
                f"teacher={row['teacher_checkpoint']} smoke={row['smoke_test']}"
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
