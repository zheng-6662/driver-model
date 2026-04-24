from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
ACTIVE_SCRIPT = REPO_ROOT / "02_code" / "final_code" / "model" / "training" / "future_steer_event_rollpeak_transformer_v5_8_amp_tuned_fixed.py"
RECALC_TOOL = REPO_ROOT / "02_code" / "tools" / "recalc_v58_checkpoint_with_current_metrics.py"
METRICS_SCRIPT = REPO_ROOT / "02_code" / "final_code" / "model" / "diagnostics" / "future_steer_event_rollpeak_transformer_v5_8_diag_eval.py"
REPORT_ROOT = REPO_ROOT / "04_project_logs" / "reports" / "input_group_ablation_20260421"
RESULT_ROOT = REPO_ROOT / "03_results" / "tmp"


COMMON_BASE_ENV = {
    "DRIVER_MODEL_INPUT_PIPELINE_VERSION": "fixed_v20260421",
    "DRIVER_MODEL_MANUAL_COARSE_UPSAMPLE": "1",
    "DRIVER_MODEL_STEER_ANGLE_UNIT": "rad",
    "DRIVER_MODEL_STEER_PLOT_UNIT": "deg",
    "DRIVER_MODEL_REV_AUX_TARGET": "strong",
    "DRIVER_MODEL_REV_SAMPLE_WEIGHT_MODE": "hybrid",
    "DRIVER_MODEL_REV_HYBRID_WEAK_COEF": "0.60",
    "DRIVER_MODEL_REV_HYBRID_STRONG_COEF": "0.40",
    "DRIVER_MODEL_REV_BRIDGE_MODE": "static",
    "DRIVER_MODEL_W_FIRSTREV_LOCAL": "0.0",
    "DRIVER_MODEL_USE_PEDALS": "0",
    "DRIVER_MODEL_USE_VY": "0",
    "DRIVER_MODEL_USE_VROLL": "0",
    "DRIVER_MODEL_USE_MU": "0",
    "DRIVER_MODEL_USE_Z": "1",
    "DRIVER_MODEL_USE_IS_CURVE_CTX": "0",
}


INPUT_GROUP_MATRIX = {
    "baseline_fixed_input": {},
    "plus_pedals": {
        "DRIVER_MODEL_USE_PEDALS": "1",
    },
    "plus_lat_dyn": {
        "DRIVER_MODEL_USE_VY": "1",
        "DRIVER_MODEL_USE_VROLL": "1",
    },
    "plus_road_cond": {
        "DRIVER_MODEL_USE_MU": "1",
    },
    "minus_z": {
        "DRIVER_MODEL_USE_Z": "0",
    },
}


BRIDGE_MATRIX = {
    "bridge_55_45": {
        "DRIVER_MODEL_REV_SAMPLE_WEIGHT_MODE": "hybrid",
        "DRIVER_MODEL_REV_HYBRID_WEAK_COEF": "0.55",
        "DRIVER_MODEL_REV_HYBRID_STRONG_COEF": "0.45",
        "DRIVER_MODEL_REV_BRIDGE_MODE": "static",
    },
    "bridge_50_50": {
        "DRIVER_MODEL_REV_SAMPLE_WEIGHT_MODE": "hybrid",
        "DRIVER_MODEL_REV_HYBRID_WEAK_COEF": "0.50",
        "DRIVER_MODEL_REV_HYBRID_STRONG_COEF": "0.50",
        "DRIVER_MODEL_REV_BRIDGE_MODE": "static",
    },
    "bridge_schedule_B_to_A": {
        "DRIVER_MODEL_REV_SAMPLE_WEIGHT_MODE": "hybrid",
        "DRIVER_MODEL_REV_HYBRID_WEAK_COEF": "0.60",
        "DRIVER_MODEL_REV_HYBRID_STRONG_COEF": "0.40",
        "DRIVER_MODEL_REV_BRIDGE_MODE": "b_to_a_linear",
    },
}


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def discover_vehicle_data_root() -> str | None:
    dataset_root = REPO_ROOT / "01_datasets"
    if not dataset_root.exists():
        return None
    for match in dataset_root.rglob("*_vehicle_aligned_cleaned.csv"):
        return str(match.parent.parent.parent)
    return None


def discover_style_csv() -> str | None:
    matches = sorted(REPO_ROOT.rglob("driver_style_cluster_result.xlsx"))
    if matches:
        return str(matches[0])
    return None


def resolve_required_path(label: str, explicit_value: str | None, env_key: str, discovered_value: str | None) -> str:
    candidates = [explicit_value, os.environ.get(env_key), discovered_value]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).resolve()
        if path.exists():
            return str(path)
    raise FileNotFoundError(
        f"Could not resolve required {label}; pass the explicit path or set {env_key}."
    )


def build_group_env(matrix: str, group_name: str) -> dict[str, str]:
    env = dict(COMMON_BASE_ENV)
    if matrix == "input_ablation":
        env.update(INPUT_GROUP_MATRIX[group_name])
    elif matrix == "bridge":
        env.update(BRIDGE_MATRIX[group_name])
    else:
        raise ValueError(f"Unsupported matrix={matrix!r}")
    return env


def choose_groups(matrix: str, include_minus_z: bool, explicit_groups: list[str] | None) -> list[str]:
    if explicit_groups:
        return explicit_groups
    if matrix == "input_ablation":
        groups = ["baseline_fixed_input", "plus_pedals", "plus_lat_dyn", "plus_road_cond"]
        if include_minus_z:
            groups.append("minus_z")
        return groups
    return ["bridge_55_45", "bridge_50_50", "bridge_schedule_B_to_A"]


def newest_run_dir(root: Path) -> str | None:
    if not root.exists():
        return None
    candidates = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("TRAIN_V5_4_STATECOND_REV")]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0])


def run_command(command: list[str], env: dict[str, str], cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_f:
        process = subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return int(process.returncode)


def maybe_run_recalc(python_exe: str, run_dir: Path, split_mode: str, eval_batch_size: int) -> dict[str, Any]:
    checkpoints = {
        "best_by_loss": run_dir / "checkpoints" / "best_model_v5_8_by_loss.pth",
        "best_by_structured": run_dir / "checkpoints" / "best_model_v5_8_by_structured.pth",
    }
    results: dict[str, Any] = {}
    for label, checkpoint_path in checkpoints.items():
        if not checkpoint_path.exists():
            continue
        prefix = f"recalc_{label}"
        command = [
            python_exe,
            str(RECALC_TOOL),
            "--script-path",
            str(ACTIVE_SCRIPT),
            "--metrics-script-path",
            str(METRICS_SCRIPT),
            "--checkpoint-path",
            str(checkpoint_path),
            "--output-dir",
            str(run_dir / "figures"),
            "--output-prefix",
            prefix,
            "--split-mode",
            split_mode,
            "--eval-batch-size",
            str(eval_batch_size),
        ]
        if split_mode == "smoke_random80":
            command.extend(["--smoke-max-samples", "512"])
        proc = subprocess.run(command, cwd=str(REPO_ROOT), check=False)
        results[label] = {
            "checkpoint_path": str(checkpoint_path),
            "returncode": int(proc.returncode),
            "summary_path": str(run_dir / "figures" / f"{prefix}_summary.json"),
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the approved input-ablation or bridge matrix on the active script.")
    parser.add_argument("--matrix", choices=["input_ablation", "bridge"], default="input_ablation")
    parser.add_argument("--groups", nargs="*", default=None)
    parser.add_argument("--include-minus-z", action="store_true")
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-recalc", action="store_true")
    parser.add_argument("--split-mode", choices=["protocol_safe", "smoke_random80"], default="protocol_safe")
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--report-dir", default=str(REPORT_ROOT))
    parser.add_argument("--result-root", default=str(RESULT_ROOT))
    parser.add_argument("--driver-root", default=None)
    parser.add_argument("--style-csv", default=None)
    args = parser.parse_args()

    report_dir = Path(args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    result_root = Path(args.result_root).resolve()
    resolved_driver_root = resolve_required_path(
        label="driver root",
        explicit_value=args.driver_root,
        env_key="DRIVER_MODEL_ROOT",
        discovered_value=discover_vehicle_data_root(),
    )
    resolved_style_csv = resolve_required_path(
        label="style csv",
        explicit_value=args.style_csv,
        env_key="DRIVER_MODEL_STYLE_CSV",
        discovered_value=discover_style_csv(),
    )

    groups = choose_groups(args.matrix, args.include_minus_z, args.groups)
    manifest_path = report_dir / f"{args.matrix}_manifest.json"
    existing_manifest = load_json(manifest_path) or {}
    existing_groups = existing_manifest.get("groups", []) if existing_manifest.get("matrix") == args.matrix else []
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "matrix": args.matrix,
        "groups": existing_groups if isinstance(existing_groups, list) else [],
        "script_path": str(ACTIVE_SCRIPT),
        "python_exe": args.python_exe,
        "dry_run": bool(args.dry_run),
        "run_recalc": bool(args.run_recalc),
        "resolved_driver_root": resolved_driver_root,
        "resolved_style_csv": resolved_style_csv,
    }
    group_records_by_name = {
        group_record.get("group_name"): group_record
        for group_record in manifest["groups"]
        if isinstance(group_record, dict) and group_record.get("group_name")
    }

    for group_name in groups:
        group_env_overrides = build_group_env(args.matrix, group_name)
        group_env_overrides["DRIVER_MODEL_ROOT"] = resolved_driver_root
        group_env_overrides["DRIVER_MODEL_STYLE_CSV"] = resolved_style_csv
        group_result_root = result_root / ("bridge_training_20260421" if args.matrix == "bridge" else "input_group_ablation_20260421") / group_name
        group_env = os.environ.copy()
        group_env.update(group_env_overrides)
        group_env["DRIVER_MODEL_RESULT_ROOT"] = str(group_result_root)
        command = [args.python_exe, str(ACTIVE_SCRIPT)]
        log_path = report_dir / f"{group_name}.log"

        group_record: dict[str, Any] = {
            "group_name": group_name,
            "matrix": args.matrix,
            "env_overrides": group_env_overrides,
            "result_root": str(group_result_root),
            "command": command,
            "status": "planned" if args.dry_run else "running",
            "train_log_path": str(log_path),
        }
        existing_group_record = group_records_by_name.get(group_name)
        if existing_group_record is None:
            manifest["groups"].append(group_record)
            group_records_by_name[group_name] = group_record
        else:
            existing_group_record.clear()
            existing_group_record.update(group_record)
            group_record = existing_group_record

        if args.dry_run:
            continue

        returncode = run_command(command, env=group_env, cwd=REPO_ROOT, log_path=log_path)
        group_record["returncode"] = int(returncode)
        group_record["status"] = "completed" if returncode == 0 else "failed"
        group_record["run_dir"] = newest_run_dir(group_result_root)

        if args.run_recalc and group_record["run_dir"] and returncode == 0:
            group_record["recalc"] = maybe_run_recalc(
                python_exe=args.python_exe,
                run_dir=Path(group_record["run_dir"]),
                split_mode=args.split_mode,
                eval_batch_size=int(args.eval_batch_size),
            )

    save_json(manifest_path, manifest)
    print(json.dumps({"manifest_path": str(manifest_path), "groups": manifest["groups"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
