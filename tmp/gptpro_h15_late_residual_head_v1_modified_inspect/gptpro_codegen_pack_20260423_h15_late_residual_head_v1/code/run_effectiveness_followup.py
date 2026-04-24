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
SUMMARY_TOOL = REPO_ROOT / "02_code" / "tools" / "summarize_effectiveness_followup.py"
DEFAULT_REPORT_DIR = REPO_ROOT / "04_project_logs" / "reports" / "effectiveness_followup_20260422"
DEFAULT_RESULT_ROOT = REPO_ROOT / "03_results" / "tmp" / "effectiveness_followup_20260422"
DEFAULT_PYTHON = Path(r"D:\ProgramData\anaconda3\envs\predict_2\python.exe")

BASELINE_FIXED_INPUT_STRUCTURED = (
    REPO_ROOT
    / "03_results"
    / "tmp"
    / "input_group_ablation_20260421"
    / "baseline_fixed_input"
    / "TRAIN_V5_4_STATECOND_REV_20260421_223235"
    / "checkpoints"
    / "best_model_v5_8_by_structured.pth"
)
RUN_A_STRUCTURED = (
    REPO_ROOT
    / "03_results"
    / "tmp"
    / "runA_structured_full"
    / "TRAIN_V5_4_STATECOND_REV_20260420_110255"
    / "checkpoints"
    / "best_model_v5_8_by_structured.pth"
)

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
    "DRIVER_MODEL_STEER_COARSE_FINE": "0",
    "DRIVER_MODEL_PHASE_ADAPTIVE_TREND": "0",
    "DRIVER_MODEL_HARD_LATE_FINE": "0",
    "DRIVER_MODEL_LATE_REV_GATE": "0",
    "DRIVER_MODEL_STRONG_POS_GATE": "0",
}

ACTION_CATALOG: dict[str, dict[str, Any]] = {
    "D0_BASELINE": {
        "phase": "D0",
        "kind": "recalc_only",
        "enabled_by_default": True,
        "base_reference": "baseline_fixed_input",
        "checkpoint_path": str(BASELINE_FIXED_INPUT_STRUCTURED),
        "output_subdir": "d0/baseline_fixed_input",
        "output_prefix": "d0_baseline_fixed_input_best_by_structured",
        "selection_source": "best_by_structured",
        "notes": "Fit/tail numeric anchor recalc with absolute-time windows.",
    },
    "D0_RUNA": {
        "phase": "D0",
        "kind": "recalc_only",
        "enabled_by_default": True,
        "base_reference": "Run A",
        "checkpoint_path": str(RUN_A_STRUCTURED),
        "output_subdir": "d0/runA",
        "output_prefix": "d0_runA_best_by_structured",
        "selection_source": "best_by_structured",
        "notes": "Structure anchor recalc with absolute-time windows.",
    },
    "H15_SMOKE": {
        "phase": "Stage1",
        "kind": "train",
        "enabled_by_default": True,
        "mode": "smoke",
        "base_reference": "baseline_fixed_input",
        "result_subdir": "h15_smoke",
        "env_overrides": {
            "DRIVER_MODEL_FUTURE_SEC": "1.5",
            "DRIVER_MODEL_BATCH_SIZE": "64",
            "DRIVER_MODEL_EPOCHS": "40",
            "DRIVER_MODEL_OPTIMIZER": "adam",
            "DRIVER_MODEL_LR": "1e-3",
            "DRIVER_MODEL_WEIGHT_DECAY": "0.0",
            "DRIVER_MODEL_SCHEDULER": "none",
            "DRIVER_MODEL_WARMUP_EPOCHS": "0",
            "DRIVER_MODEL_GRAD_CLIP_NORM": "0.0",
            "DRIVER_MODEL_SMOKE": "1",
            "DRIVER_MODEL_SMOKE_MAX_SAMPLES": "256",
            "DRIVER_MODEL_SMOKE_EPOCHS": "2",
            "DRIVER_MODEL_SMOKE_BATCH_SIZE": "32",
        },
        "notes": "Wiring validation for 1.5s horizon and absolute-time window outputs.",
    },
    "H15": {
        "phase": "Stage1",
        "kind": "train",
        "enabled_by_default": True,
        "mode": "full",
        "base_reference": "baseline_fixed_input",
        "result_subdir": "h15_full",
        "env_overrides": {
            "DRIVER_MODEL_FUTURE_SEC": "1.5",
            "DRIVER_MODEL_BATCH_SIZE": "64",
            "DRIVER_MODEL_EPOCHS": "40",
            "DRIVER_MODEL_OPTIMIZER": "adam",
            "DRIVER_MODEL_LR": "1e-3",
            "DRIVER_MODEL_WEIGHT_DECAY": "0.0",
            "DRIVER_MODEL_SCHEDULER": "none",
            "DRIVER_MODEL_WARMUP_EPOCHS": "0",
            "DRIVER_MODEL_GRAD_CLIP_NORM": "0.0",
        },
        "notes": "Primary 1.5s horizon probe from the fixed-input baseline route.",
    },
    "OPT_A_20": {
        "phase": "Stage1",
        "kind": "train",
        "enabled_by_default": True,
        "mode": "full",
        "base_reference": "baseline_fixed_input",
        "result_subdir": "opt_a_20",
        "env_overrides": {
            "DRIVER_MODEL_FUTURE_SEC": "2.0",
            "DRIVER_MODEL_BATCH_SIZE": "64",
            "DRIVER_MODEL_EPOCHS": "60",
            "DRIVER_MODEL_OPTIMIZER": "adamw",
            "DRIVER_MODEL_LR": "1e-3",
            "DRIVER_MODEL_WEIGHT_DECAY": "1e-4",
            "DRIVER_MODEL_SCHEDULER": "cosine",
            "DRIVER_MODEL_WARMUP_EPOCHS": "3",
            "DRIVER_MODEL_GRAD_CLIP_NORM": "1.0",
        },
        "notes": "Optimization-only rescue check on the original 2.0s task.",
    },
    "OPT_A_H15": {
        "phase": "Stage2",
        "kind": "train",
        "enabled_by_default": False,
        "mode": "full",
        "base_reference": "H15",
        "result_subdir": "opt_a_h15",
        "env_overrides": {
            "DRIVER_MODEL_FUTURE_SEC": "1.5",
            "DRIVER_MODEL_BATCH_SIZE": "64",
            "DRIVER_MODEL_EPOCHS": "60",
            "DRIVER_MODEL_OPTIMIZER": "adamw",
            "DRIVER_MODEL_LR": "1e-3",
            "DRIVER_MODEL_WEIGHT_DECAY": "1e-4",
            "DRIVER_MODEL_SCHEDULER": "cosine",
            "DRIVER_MODEL_WARMUP_EPOCHS": "3",
            "DRIVER_MODEL_GRAD_CLIP_NORM": "1.0",
        },
        "notes": "Optimization bundle on the 1.5s branch if H15 qualifies.",
    },
    "OPT_B_H15": {
        "phase": "Stage2",
        "kind": "train",
        "enabled_by_default": False,
        "mode": "full",
        "base_reference": "H15",
        "result_subdir": "opt_b_h15",
        "env_overrides": {
            "DRIVER_MODEL_FUTURE_SEC": "1.5",
            "DRIVER_MODEL_BATCH_SIZE": "64",
            "DRIVER_MODEL_EPOCHS": "60",
            "DRIVER_MODEL_OPTIMIZER": "adamw",
            "DRIVER_MODEL_LR": "3e-4",
            "DRIVER_MODEL_WEIGHT_DECAY": "1e-4",
            "DRIVER_MODEL_SCHEDULER": "cosine",
            "DRIVER_MODEL_WARMUP_EPOCHS": "3",
            "DRIVER_MODEL_GRAD_CLIP_NORM": "1.0",
        },
        "notes": "Lower-LR optimization bundle on the 1.5s branch if H15 qualifies.",
    },
    "H10": {
        "phase": "Stage3",
        "kind": "train",
        "enabled_by_default": False,
        "mode": "full",
        "base_reference": "baseline_fixed_input",
        "result_subdir": "h10_full",
        "env_overrides": {
            "DRIVER_MODEL_FUTURE_SEC": "1.0",
            "DRIVER_MODEL_BATCH_SIZE": "64",
            "DRIVER_MODEL_EPOCHS": "40",
            "DRIVER_MODEL_OPTIMIZER": "adam",
            "DRIVER_MODEL_LR": "1e-3",
            "DRIVER_MODEL_WEIGHT_DECAY": "0.0",
            "DRIVER_MODEL_SCHEDULER": "none",
            "DRIVER_MODEL_WARMUP_EPOCHS": "0",
            "DRIVER_MODEL_GRAD_CLIP_NORM": "0.0",
        },
        "notes": "Conditional diagnostic ceiling run; not part of the default sequence.",
    },
    "OPT_C_BEST": {
        "phase": "Stage3",
        "kind": "train",
        "enabled_by_default": False,
        "mode": "full",
        "base_reference": "winner",
        "result_subdir": "opt_c_best",
        "env_overrides": {
            "DRIVER_MODEL_WEIGHT_DECAY": "5e-4",
        },
        "notes": "Conditional regularization slot; apply only after the current winner is chosen.",
    },
    "CAP_192_BEST": {
        "phase": "Stage3",
        "kind": "train",
        "enabled_by_default": False,
        "mode": "full",
        "base_reference": "winner",
        "result_subdir": "cap_192_best",
        "env_overrides": {
            "DRIVER_MODEL_D_MODEL": "192",
            "DRIVER_MODEL_N_HEAD": "4",
            "DRIVER_MODEL_FFN_DIM": "384",
        },
        "notes": "Width-only bump from the current best non-collapse configuration.",
    },
    "WINNER_CONFIRM": {
        "phase": "Reserve",
        "kind": "train",
        "enabled_by_default": False,
        "mode": "full",
        "base_reference": "winner",
        "result_subdir": "winner_confirm",
        "env_overrides": {},
        "notes": "Reserve slot for re-running the current winner with the same seed and env.",
    },
}

DEFAULT_ACTIONS = ["D0_BASELINE", "D0_RUNA", "H15_SMOKE", "H15", "OPT_A_20"]


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else None


def merge_records(
    existing_records: list[dict[str, Any]] | None,
    new_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for record in existing_records or []:
        action_name = record.get("action_name")
        if isinstance(action_name, str):
            merged[action_name] = record
    for record in new_records:
        action_name = record.get("action_name")
        if isinstance(action_name, str):
            merged[action_name] = record
    ordered_names = [name for name in ACTION_CATALOG if name in merged]
    ordered_names.extend(name for name in merged if name not in ACTION_CATALOG)
    return [merged[name] for name in ordered_names]


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
    for candidate in (explicit_value, os.environ.get(env_key), discovered_value):
        if not candidate:
            continue
        path = Path(candidate).resolve()
        if path.exists():
            return str(path)
    raise FileNotFoundError(f"Could not resolve required {label}; pass it explicitly or set {env_key}.")


def newest_run_dir(root: Path) -> str | None:
    if not root.exists():
        return None
    candidates = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("TRAIN_V5_4_STATECOND_REV")]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0])


def load_winner_env_from_run_dir(run_dir: Path) -> dict[str, str]:
    run_config_path = run_dir / "run_config.json"
    run_config = load_json(run_config_path)
    if run_config is None:
        raise FileNotFoundError(f"Failed to load winner run_config.json from {run_config_path}")

    mapping = {
        "FUTURE_SEC": "DRIVER_MODEL_FUTURE_SEC",
        "BATCH_SIZE": "DRIVER_MODEL_BATCH_SIZE",
        "EPOCHS": "DRIVER_MODEL_EPOCHS",
        "OPTIMIZER": "DRIVER_MODEL_OPTIMIZER",
        "LR": "DRIVER_MODEL_LR",
        "WEIGHT_DECAY": "DRIVER_MODEL_WEIGHT_DECAY",
        "SCHEDULER": "DRIVER_MODEL_SCHEDULER",
        "WARMUP_EPOCHS": "DRIVER_MODEL_WARMUP_EPOCHS",
        "GRAD_CLIP_NORM": "DRIVER_MODEL_GRAD_CLIP_NORM",
        "D_MODEL": "DRIVER_MODEL_D_MODEL",
        "N_HEAD": "DRIVER_MODEL_N_HEAD",
        "FFN_DIM": "DRIVER_MODEL_FFN_DIM",
        "DROPOUT": "DRIVER_MODEL_DROPOUT",
        "INPUT_PIPELINE_VERSION": "DRIVER_MODEL_INPUT_PIPELINE_VERSION",
        "REV_AUX_TARGET": "DRIVER_MODEL_REV_AUX_TARGET",
        "REV_SAMPLE_WEIGHT_MODE": "DRIVER_MODEL_REV_SAMPLE_WEIGHT_MODE",
        "REV_HYBRID_WEAK_COEF": "DRIVER_MODEL_REV_HYBRID_WEAK_COEF",
        "REV_HYBRID_STRONG_COEF": "DRIVER_MODEL_REV_HYBRID_STRONG_COEF",
        "REV_BRIDGE_MODE": "DRIVER_MODEL_REV_BRIDGE_MODE",
    }
    bool_mapping = {
        "USE_PEDALS": "DRIVER_MODEL_USE_PEDALS",
        "USE_VY": "DRIVER_MODEL_USE_VY",
        "USE_VROLL": "DRIVER_MODEL_USE_VROLL",
        "USE_MU": "DRIVER_MODEL_USE_MU",
        "USE_Z": "DRIVER_MODEL_USE_Z",
        "USE_IS_CURVE_CTX": "DRIVER_MODEL_USE_IS_CURVE_CTX",
        "ENABLE_STEER_COARSE_FINE": "DRIVER_MODEL_STEER_COARSE_FINE",
        "ENABLE_PHASE_ADAPTIVE_TREND": "DRIVER_MODEL_PHASE_ADAPTIVE_TREND",
        "ENABLE_HARD_LATE_FINE": "DRIVER_MODEL_HARD_LATE_FINE",
        "ENABLE_LATE_REV_GATE": "DRIVER_MODEL_LATE_REV_GATE",
        "ENABLE_STRONG_POS_GATE": "DRIVER_MODEL_STRONG_POS_GATE",
        "ENABLE_MANUAL_COARSE_UPSAMPLE": "DRIVER_MODEL_MANUAL_COARSE_UPSAMPLE",
    }

    env_overrides: dict[str, str] = {}
    for config_key, env_key in mapping.items():
        if config_key in run_config:
            env_overrides[env_key] = str(run_config[config_key])
    for config_key, env_key in bool_mapping.items():
        if config_key in run_config:
            env_overrides[env_key] = "1" if bool(run_config[config_key]) else "0"
    return env_overrides


def run_subprocess(command: list[str], env: dict[str, str], cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_f:
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return int(proc.returncode)


def run_recalc_for_checkpoint(
    python_exe: str,
    checkpoint_path: Path,
    output_dir: Path,
    output_prefix: str,
    eval_batch_size: int,
) -> dict[str, Any]:
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
        str(output_dir),
        "--output-prefix",
        output_prefix,
        "--split-mode",
        "protocol_safe",
        "--eval-batch-size",
        str(eval_batch_size),
    ]
    proc = subprocess.run(command, cwd=str(REPO_ROOT), check=False)
    return {
        "returncode": int(proc.returncode),
        "summary_path": str(output_dir / f"{output_prefix}_summary.json"),
        "cases_path": str(output_dir / f"{output_prefix}_cases.csv"),
        "top_bad_cases_path": str(output_dir / f"{output_prefix}_top_bad_cases.csv"),
    }


def run_recalc_for_train_run(python_exe: str, run_dir: Path, eval_batch_size: int) -> dict[str, Any]:
    results: dict[str, Any] = {}
    checkpoint_suffixes = {
        "best_by_loss": "by_loss",
        "best_by_structured": "by_structured",
    }
    for label, suffix in checkpoint_suffixes.items():
        checkpoint_path = run_dir / "checkpoints" / f"best_model_v5_8_{suffix}.pth"
        if not checkpoint_path.exists():
            continue
        results[label] = run_recalc_for_checkpoint(
            python_exe=python_exe,
            checkpoint_path=checkpoint_path,
            output_dir=run_dir / "figures",
            output_prefix=f"recalc_{label}",
            eval_batch_size=eval_batch_size,
        )
    return results


def build_action_env(
    report_dir: Path,
    action_name: str,
    action_def: dict[str, Any],
    driver_root: str,
    style_csv: str,
    result_root: Path,
    winner_env: dict[str, str] | None,
) -> dict[str, str]:
    env = dict(os.environ)
    env.update(COMMON_BASE_ENV)
    if winner_env:
        env.update(winner_env)
    env["DRIVER_MODEL_ROOT"] = driver_root
    env["DRIVER_MODEL_STYLE_CSV"] = style_csv
    env["DRIVER_MODEL_RESULT_ROOT"] = str(result_root)
    env.update(action_def.get("env_overrides", {}))
    if action_def.get("mode") == "smoke":
        env.setdefault("DRIVER_MODEL_SMOKE", "1")
    env["PYTHONUTF8"] = "1"
    env["EFFECTIVENESS_REPORT_DIR"] = str(report_dir)
    env["EFFECTIVENESS_ACTION_NAME"] = action_name
    return env


def execute_recalc_only_action(
    action_name: str,
    action_def: dict[str, Any],
    python_exe: str,
    report_dir: Path,
    eval_batch_size: int,
) -> dict[str, Any]:
    checkpoint_path = Path(action_def["checkpoint_path"]).resolve()
    output_dir = report_dir / action_def["output_subdir"]
    recalc_result = run_recalc_for_checkpoint(
        python_exe=python_exe,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        output_prefix=action_def["output_prefix"],
        eval_batch_size=eval_batch_size,
    )
    return {
        "action_name": action_name,
        "phase": action_def["phase"],
        "kind": action_def["kind"],
        "status": "completed" if recalc_result["returncode"] == 0 else "failed",
        "base_reference": action_def.get("base_reference"),
        "checkpoint_path": str(checkpoint_path),
        "selection_source": action_def.get("selection_source"),
        "notes": action_def.get("notes"),
        "recalc": recalc_result,
    }


def execute_train_action(
    action_name: str,
    action_def: dict[str, Any],
    python_exe: str,
    report_dir: Path,
    result_root_base: Path,
    driver_root: str,
    style_csv: str,
    eval_batch_size: int,
    winner_env: dict[str, str] | None,
) -> dict[str, Any]:
    action_result_root = result_root_base / action_def["result_subdir"]
    env = build_action_env(
        report_dir=report_dir,
        action_name=action_name,
        action_def=action_def,
        driver_root=driver_root,
        style_csv=style_csv,
        result_root=action_result_root,
        winner_env=winner_env,
    )
    log_path = report_dir / "logs" / f"{action_name.lower()}.log"
    command = [python_exe, str(ACTIVE_SCRIPT)]
    returncode = run_subprocess(command, env=env, cwd=REPO_ROOT, log_path=log_path)
    run_dir = newest_run_dir(action_result_root)
    record: dict[str, Any] = {
        "action_name": action_name,
        "phase": action_def["phase"],
        "kind": action_def["kind"],
        "mode": action_def.get("mode"),
        "status": "completed" if returncode == 0 else "failed",
        "base_reference": action_def.get("base_reference"),
        "notes": action_def.get("notes"),
        "log_path": str(log_path),
        "command": command,
        "result_root": str(action_result_root),
        "run_dir": run_dir,
        "env_overrides": action_def.get("env_overrides", {}),
    }
    if returncode == 0 and run_dir is not None:
        record["recalc"] = run_recalc_for_train_run(
            python_exe=python_exe,
            run_dir=Path(run_dir),
            eval_batch_size=eval_batch_size,
        )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the 2026-04-22 effectiveness follow-up workflow.")
    parser.add_argument("--actions", nargs="*", default=None, help="Explicit action names to execute.")
    parser.add_argument("--list-actions", action="store_true", help="List available action names and exit.")
    parser.add_argument("--python-exe", default=str(DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else Path(sys.executable).resolve()))
    parser.add_argument("--dry-run", action="store_true", help="Write the manifest without executing actions.")
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--result-root", default=str(DEFAULT_RESULT_ROOT))
    parser.add_argument("--driver-root", default=None)
    parser.add_argument("--style-csv", default=None)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--refresh-summary", action="store_true", help="Refresh the effectiveness summary after execution.")
    parser.add_argument("--winner-run-dir", default=None, help="Run directory whose run_config should seed winner-based conditional actions.")
    args = parser.parse_args()

    if args.list_actions:
        print(json.dumps(ACTION_CATALOG, indent=2, ensure_ascii=False))
        return

    selected_actions = args.actions or list(DEFAULT_ACTIONS)
    unknown = [name for name in selected_actions if name not in ACTION_CATALOG]
    if unknown:
        raise ValueError(f"Unknown actions: {unknown}")

    report_dir = Path(args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    result_root = Path(args.result_root).resolve()
    result_root.mkdir(parents=True, exist_ok=True)

    driver_root = resolve_required_path(
        label="driver root",
        explicit_value=args.driver_root,
        env_key="DRIVER_MODEL_ROOT",
        discovered_value=discover_vehicle_data_root(),
    )
    style_csv = resolve_required_path(
        label="style csv",
        explicit_value=args.style_csv,
        env_key="DRIVER_MODEL_STYLE_CSV",
        discovered_value=discover_style_csv(),
    )

    manifest: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "report_dir": str(report_dir),
        "result_root": str(result_root),
        "python_exe": str(Path(args.python_exe).resolve()),
        "driver_root": driver_root,
        "style_csv": style_csv,
        "dry_run": bool(args.dry_run),
        "winner_run_dir": args.winner_run_dir,
        "default_actions": DEFAULT_ACTIONS,
        "requested_actions": selected_actions,
        "action_catalog": ACTION_CATALOG,
        "records": [],
    }
    manifest_path = report_dir / "effectiveness_followup_manifest.json"
    existing_manifest = load_json(manifest_path)
    if existing_manifest is not None:
        manifest["records"] = list(existing_manifest.get("records", []))

    winner_env = load_winner_env_from_run_dir(Path(args.winner_run_dir).resolve()) if args.winner_run_dir else None

    new_records: list[dict[str, Any]] = []
    if not args.dry_run:
        for action_name in selected_actions:
            action_def = ACTION_CATALOG[action_name]
            if action_def["kind"] == "recalc_only":
                record = execute_recalc_only_action(
                    action_name=action_name,
                    action_def=action_def,
                    python_exe=str(Path(args.python_exe).resolve()),
                    report_dir=report_dir,
                    eval_batch_size=args.eval_batch_size,
                )
            else:
                action_winner_env = winner_env if action_def.get("base_reference") == "winner" else None
                if action_def.get("base_reference") == "winner" and not action_winner_env:
                    raise ValueError(f"{action_name} requires --winner-run-dir")
                record = execute_train_action(
                    action_name=action_name,
                    action_def=action_def,
                    python_exe=str(Path(args.python_exe).resolve()),
                    report_dir=report_dir,
                    result_root_base=result_root,
                    driver_root=driver_root,
                    style_csv=style_csv,
                    eval_batch_size=args.eval_batch_size,
                    winner_env=action_winner_env,
                )
            new_records.append(record)

    manifest["records"] = merge_records(manifest.get("records"), new_records)
    save_json(manifest_path, manifest)

    if args.refresh_summary and not args.dry_run:
        subprocess.run(
            [
                str(Path(args.python_exe).resolve()),
                str(SUMMARY_TOOL),
                "--manifest",
                str(manifest_path),
                "--report-dir",
                str(report_dir),
            ],
            cwd=str(REPO_ROOT),
            check=False,
        )

    print(json.dumps({"manifest_path": str(manifest_path), "requested_actions": selected_actions}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
