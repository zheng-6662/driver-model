# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from common_compare_runner import build_args
from prediction_plotting import save_prediction_plots_for_run
from run_event_conditioned_trajectory_baseline import train_one_run


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_RUN_RECORD = THIS_DIR / "e3_e4_multiseed_last_runs.csv"

E3_E4_VARIANTS: dict[str, dict[str, str]] = {
    "E3": {
        "variant_key": "vehicle_direct_coarse_fine_semantic_driver_state_no_eeg_continuous_style",
        "label": "粗细双头 + 无脑电生理状态量 + 连续驾驶风格",
    },
    "E4": {
        "variant_key": "vehicle_direct_coarse_fine_semantic_driver_state_continuous_style",
        "label": "粗细双头 + 生理状态量 + 连续驾驶风格",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan or execute the E3/E4 multi-seed physiology/EEG control runs. "
            "Dry-run is the default; pass --execute to train."
        )
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    parser.add_argument("--experiments", nargs="+", default=["E3", "E4"], choices=sorted(E3_E4_VARIANTS))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    parser.add_argument("--execute", action="store_true", help="Actually run training. Omit for plan-only dry-run.")
    parser.add_argument("--run-record", default=str(DEFAULT_RUN_RECORD))
    parser.add_argument("--no-plots", action="store_true", help="Skip prediction plot export after each run.")
    return parser.parse_args()


def _planned_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        for experiment_id in args.experiments:
            info = E3_E4_VARIANTS[experiment_id]
            run_args = build_args(info["variant_key"])
            run_args.seed = int(seed)
            if args.device is not None:
                run_args.device = str(args.device)
            run_args.run_prefix = f"{experiment_id}_{info['label']}_seed{seed}"
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "seed": int(seed),
                    "variant_key": info["variant_key"],
                    "label": info["label"],
                    "run_prefix": run_args.run_prefix,
                    "manifest": str(run_args.manifest),
                    "device": str(run_args.device),
                    "epochs": int(run_args.epochs),
                    "batch_size": int(run_args.batch_size),
                    "lr": float(run_args.lr),
                    "conditioning_mode": str(run_args.conditioning_mode),
                    "teacher_state_mode": str(getattr(run_args, "teacher_state_mode", "")),
                    "enable_teacher_state_context": bool(getattr(run_args, "enable_teacher_state_context", False)),
                    "enable_driver_style_context": bool(getattr(run_args, "enable_driver_style_context", False)),
                    "run_root": "",
                }
            )
    return rows


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else [
        "experiment_id",
        "seed",
        "variant_key",
        "label",
        "run_prefix",
        "manifest",
        "device",
        "epochs",
        "batch_size",
        "lr",
        "conditioning_mode",
        "teacher_state_mode",
        "enable_teacher_state_context",
        "enable_driver_style_context",
        "run_root",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _run_one(row: dict[str, Any], no_plots: bool) -> dict[str, Any]:
    run_args = build_args(str(row["variant_key"]))
    run_args.seed = int(row["seed"])
    run_args.device = str(row["device"])
    run_args.run_prefix = str(row["run_prefix"])
    result = train_one_run(run_args)
    row["run_root"] = str(result["run_root"])
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
        except Exception as exc:  # Keep later runs from being lost if plotting fails.
            row["prediction_plot_error"] = str(exc)
    return row


def main() -> None:
    args = parse_args()
    rows = _planned_rows(args)
    record_path = Path(args.run_record)
    if not args.execute:
        _write_rows(record_path, rows)
        print(f"Dry-run only. Planned {len(rows)} runs.")
        print(f"Plan CSV: {record_path}")
        for row in rows:
            print(
                f"{row['experiment_id']} seed={row['seed']} "
                f"variant={row['variant_key']} prefix={row['run_prefix']}"
            )
        return

    executed: list[dict[str, Any]] = []
    for row in rows:
        print(f"Running {row['experiment_id']} seed={row['seed']} - {row['label']}", flush=True)
        executed.append(_run_one(row, no_plots=bool(args.no_plots)))
        _write_rows(record_path, executed)
    print(f"Completed {len(executed)} runs.")
    print(f"Run record: {record_path}")


if __name__ == "__main__":
    main()
