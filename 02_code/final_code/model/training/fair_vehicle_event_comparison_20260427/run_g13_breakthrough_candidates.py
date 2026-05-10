from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
TRAINING_DIR = THIS_DIR.parent
PROJECT_ROOT = THIS_DIR.parents[4]
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from prediction_plotting import save_prediction_plots_for_run


TRAIN_SCRIPT = TRAINING_DIR / "run_event_conditioned_trajectory_baseline.py"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "04_project_logs" / "reports" / "g13_model_breakthrough_20260510"
LOCAL_PREVIOUS_E4_CSV = PROJECT_ROOT / "04_project_logs" / "reports" / "style_physio_eeg_e3_e4_runs_20260507.csv"
RUN_ROOT = PROJECT_ROOT / "tmp" / "event_conditioned_runs"


@dataclass(frozen=True)
class Candidate:
    experiment_id: str
    label: str
    purpose: str
    extra_args: tuple[str, ...]
    needs_teacher: bool = False


COMMON_ARGS: tuple[str, ...] = (
    "--device",
    "cuda",
    "--epochs",
    "40",
    "--min-epochs",
    "40",
    "--patience",
    "99",
    "--batch-size",
    "64",
    "--lr",
    "0.001",
    "--conditioning-mode",
    "vehicle_direct_coarse_fine",
    "--selection-mode",
    "legacy_rmse",
    "--enable-driver-style-context",
)


CANDIDATES: dict[str, Candidate] = {
    "G13A": Candidate(
        experiment_id="G13A",
        label="连续风格 + 响应类型辅助学习",
        purpose="验证只增加响应类型监督，是否能让模型更理解大幅、反向、多段、晚峰等响应形态。",
        extra_args=(
            "--enable-response-type-head",
            "--response-type-loss-weight",
            "0.20",
        ),
    ),
    "G13B": Candidate(
        experiment_id="G13B",
        label="连续风格 + 肌电 + 响应类型辅助学习",
        purpose="验证肌电和响应类型监督一起使用时，是否比 E10C 更稳定。",
        extra_args=(
            "--enable-teacher-state-context",
            "--teacher-state-mode",
            "raw_emg_only",
            "--teacher-state-dim",
            "1",
            "--enable-response-type-head",
            "--response-type-loss-weight",
            "0.20",
        ),
    ),
    "G13C": Candidate(
        experiment_id="G13C",
        label="连续风格 + 肌电 + 响应类型影响轨迹预测",
        purpose="让模型先预测响应类型，再把这个判断反馈给轨迹预测头，检验按响应类型分开处理是否可行。",
        extra_args=(
            "--enable-teacher-state-context",
            "--teacher-state-mode",
            "raw_emg_only",
            "--teacher-state-dim",
            "1",
            "--enable-response-type-head",
            "--enable-response-type-condition",
            "--response-type-loss-weight",
            "0.20",
        ),
    ),
    "G13F": Candidate(
        experiment_id="G13F",
        label="肌电 + 响应类型 + 幅值方向物理约束",
        purpose="直接处理用户指出的趋势像但幅值/方向物理意义不对的问题。",
        extra_args=(
            "--enable-teacher-state-context",
            "--teacher-state-mode",
            "raw_emg_only",
            "--teacher-state-dim",
            "1",
            "--enable-response-type-head",
            "--enable-response-type-condition",
            "--response-type-loss-weight",
            "0.20",
            "--steer-amp-loss-weight",
            "0.10",
            "--steer-direction-loss-weight",
            "0.05",
            "--steer-amp-target-ratio",
            "0.85",
            "--steer-direction-major-only",
        ),
    ),
    "G13G": Candidate(
        experiment_id="G13G",
        label="脑电教师 + 连续风格 + 响应类型辅助学习",
        purpose="检验脑电教师蒸馏收益能否和响应类型监督结合，但推理时仍不使用生理输入。",
        extra_args=(
            "--enable-response-type-head",
            "--response-type-loss-weight",
            "0.20",
            "--distill-weight",
            "0.20",
            "--distill-tail-weight",
            "0.05",
        ),
        needs_teacher=True,
    ),
    "G13H": Candidate(
        experiment_id="G13H",
        label="脑电教师 + 肌电学生 + 响应类型辅助学习",
        purpose="重新评估脑电教师和肌电学生的组合，重点看困难样本和响应类型而不只看整体误差。",
        extra_args=(
            "--enable-teacher-state-context",
            "--teacher-state-mode",
            "raw_emg_only",
            "--teacher-state-dim",
            "1",
            "--enable-response-type-head",
            "--response-type-loss-weight",
            "0.20",
            "--distill-weight",
            "0.20",
            "--distill-tail-weight",
            "0.05",
        ),
        needs_teacher=True,
    ),
    "G13I": Candidate(
        experiment_id="G13I",
        label="脑电教师 + 肌电学生 + 困难响应加权 + 物理约束",
        purpose="把脑电教师重点用于大幅、反向、多段等困难响应，同时用物理约束压制幅值不足和错侧。",
        extra_args=(
            "--enable-teacher-state-context",
            "--teacher-state-mode",
            "raw_emg_only",
            "--teacher-state-dim",
            "1",
            "--enable-response-type-head",
            "--enable-response-type-condition",
            "--response-type-loss-weight",
            "0.20",
            "--distill-weight",
            "0.20",
            "--distill-tail-weight",
            "0.05",
            "--distill-reliability-weighting",
            "--distill-hardcase-weighting",
            "--distill-hardcase-extra-weight",
            "0.50",
            "--steer-amp-loss-weight",
            "0.10",
            "--steer-direction-loss-weight",
            "0.05",
            "--steer-amp-target-ratio",
            "0.85",
            "--steer-direction-major-only",
        ),
        needs_teacher=True,
    ),
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    if path.exists():
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            existing = next(reader, None)
        if existing:
            fieldnames = existing
    write_header = not path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _run_command(command: list[str], log_path: Path, cwd: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("COMMAND: " + " ".join(command) + "\n\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        return process.wait()


def _latest_run_dir(prefix: str) -> Path:
    matches = sorted(RUN_ROOT.glob(f"{prefix}_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"没有找到运行目录: {prefix}_*")
    return matches[0]


def _compact_metrics(metrics_path: Path) -> dict[str, float]:
    metrics = _read_json(metrics_path)
    test = metrics["test"]
    selection = test["selection_summary"]
    return {
        "test_steer_rmse": float(test["steer_rmse"]),
        "primary_rmse": float(selection.get("rmse_primary_abs_steer", selection.get("primary_rmse_score", float("nan")))),
        "tail_rmse": float(selection.get("rmse_tail_abs_steer", float("nan"))),
        "peak_err_s": float(selection.get("peak_time_abs_err_s", float("nan"))),
        "selection": float(selection.get("selection_score", float("nan"))),
    }


def _find_teacher_checkpoint(seed: int, report_dir: Path) -> Path | None:
    teacher_log = report_dir / "g13_teacher_run_log.csv"
    for csv_path in (teacher_log, LOCAL_PREVIOUS_E4_CSV):
        if not csv_path.exists():
            continue
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("seed", "")) != str(seed):
                    continue
                if row.get("experiment_id") not in {"G13T", "E4", ""}:
                    continue
                run_root = Path(row.get("run_root", ""))
                candidate = run_root / "best_model.pt"
                if candidate.exists():
                    return candidate
                teacher_checkpoint = Path(row.get("teacher_checkpoint", ""))
                if teacher_checkpoint.exists():
                    return teacher_checkpoint
    patterns = [
        f"G13T_脑电教师准备_seed{seed}_*",
        f"RESTORE_E4_*_seed{seed}_*",
        f"E4_粗细双头 + 生理状态量 + 连续驾驶风格_seed{seed}_*",
    ]
    for pattern in patterns:
        for run_dir in sorted(RUN_ROOT.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True):
            candidate = run_dir / "best_model.pt"
            if candidate.exists():
                return candidate
    return None


def _teacher_command(seed: int, smoke: bool = False) -> tuple[str, list[str]]:
    prefix = f"G13T_脑电教师准备_seed{seed}"
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--run-prefix",
        prefix,
        "--seed",
        str(seed),
        *COMMON_ARGS,
        "--enable-teacher-state-context",
        "--teacher-state-mode",
        "semantic_driver_state",
        "--teacher-state-dim",
        "6",
    ]
    if smoke:
        command.extend(["--smoke-test", "--smoke-epochs", "1", "--smoke-train-samples", "96", "--smoke-val-samples", "32", "--smoke-test-samples", "32"])
    return prefix, command


def _candidate_command(candidate: Candidate, seed: int, teacher_checkpoint: Path | None, smoke: bool = False) -> tuple[str, list[str]]:
    prefix = f"{candidate.experiment_id}_{candidate.label}_seed{seed}"
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--run-prefix",
        prefix,
        "--seed",
        str(seed),
        *COMMON_ARGS,
        *candidate.extra_args,
    ]
    if candidate.needs_teacher:
        if teacher_checkpoint is None:
            raise FileNotFoundError(f"{candidate.experiment_id} 需要脑电教师检查点，但当前没有找到 seed{seed} 的 best_model.pt。")
        command.extend(["--distill-teacher-checkpoint", str(teacher_checkpoint)])
    if smoke:
        command.extend(["--smoke-test", "--smoke-epochs", "1", "--smoke-train-samples", "96", "--smoke-val-samples", "32", "--smoke-test-samples", "32"])
    return prefix, command


def _record_run(
    report_dir: Path,
    candidate: Candidate,
    seed: int,
    run_root: Path,
    command: list[str],
    log_path: Path,
    teacher_checkpoint: Path | None = None,
) -> None:
    metrics = _compact_metrics(run_root / "metrics.json")
    plot_info: dict[str, str] = {}
    try:
        plot_info = save_prediction_plots_for_run(run_root=run_root, split="test")
    except Exception as exc:
        plot_info = {"prediction_error": str(exc)}
    row = {
        "experiment_id": candidate.experiment_id,
        "label": candidate.label,
        "purpose": candidate.purpose,
        "seed": seed,
        "run_root": str(run_root),
        "teacher_checkpoint": "" if teacher_checkpoint is None else str(teacher_checkpoint),
        "log_path": str(log_path),
        "command": " ".join(command),
        **metrics,
        **plot_info,
    }
    _write_csv_row(report_dir / "g13_run_log.csv", row)


def _record_teacher(report_dir: Path, seed: int, run_root: Path, command: list[str], log_path: Path) -> None:
    metrics = _compact_metrics(run_root / "metrics.json")
    row = {
        "experiment_id": "G13T",
        "label": "脑电教师准备",
        "purpose": "为脑电教师蒸馏候选重新生成同种子教师检查点。",
        "seed": seed,
        "run_root": str(run_root),
        "teacher_checkpoint": str(run_root / "best_model.pt"),
        "log_path": str(log_path),
        "command": " ".join(command),
        **metrics,
    }
    _write_csv_row(report_dir / "g13_teacher_run_log.csv", row)


def run_one(candidate_id: str, seed: int, report_dir: Path, smoke: bool = False, ensure_teacher: bool = False) -> Path:
    candidate = CANDIDATES[candidate_id]
    report_dir.mkdir(parents=True, exist_ok=True)
    log_dir = report_dir / "run_logs"
    teacher_checkpoint = None
    if candidate.needs_teacher:
        teacher_checkpoint = _find_teacher_checkpoint(seed=seed, report_dir=report_dir)
        if teacher_checkpoint is None and ensure_teacher:
            teacher_prefix, teacher_command = _teacher_command(seed=seed, smoke=smoke)
            teacher_log = log_dir / f"{teacher_prefix}.log"
            code = _run_command(teacher_command, log_path=teacher_log, cwd=TRAINING_DIR)
            if code != 0:
                raise RuntimeError(f"脑电教师训练失败，退出码={code}，日志={teacher_log}")
            teacher_run_root = _latest_run_dir(teacher_prefix)
            _record_teacher(report_dir=report_dir, seed=seed, run_root=teacher_run_root, command=teacher_command, log_path=teacher_log)
            teacher_checkpoint = teacher_run_root / "best_model.pt"
        if teacher_checkpoint is None:
            raise FileNotFoundError("没有脑电教师检查点。可以加 --ensure-teacher 先重训教师。")
    prefix, command = _candidate_command(candidate=candidate, seed=seed, teacher_checkpoint=teacher_checkpoint, smoke=smoke)
    log_path = log_dir / f"{prefix}.log"
    code = _run_command(command, log_path=log_path, cwd=TRAINING_DIR)
    if code != 0:
        raise RuntimeError(f"{candidate_id} 训练失败，退出码={code}，日志={log_path}")
    run_root = _latest_run_dir(prefix)
    _record_run(
        report_dir=report_dir,
        candidate=candidate,
        seed=seed,
        run_root=run_root,
        command=command,
        log_path=log_path,
        teacher_checkpoint=teacher_checkpoint,
    )
    return run_root


def write_plan(report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# G13 模型突破阶段执行记录",
        "",
        "## 当前代码状态",
        "- 核心训练代码保留了响应类型辅助学习、响应类型条件化、肌电输入和脑电教师蒸馏接口。",
        "- 之前的实验批量脚本目录被清空，因此本轮补充了 G13 专用运行脚本和预测图生成脚本。",
        "- 旧的 `tmp/event_conditioned_runs` 检查点文件当前缺失，因此脑电教师路线需要重新生成同种子教师检查点，不能直接复用旧 E4 检查点。",
        "",
        "## 第一轮候选",
    ]
    for candidate in CANDIDATES.values():
        lines.append(f"- {candidate.experiment_id}：{candidate.label}。目的：{candidate.purpose}")
    lines.extend(
        [
            "",
            "## 记录要求",
            "- 每个版本记录整体误差、主响应误差、尾段误差、综合选择指标、预测图和运行命令。",
            "- 训练命令不包含服务器密码；服务器密码不得写入任何日志、报告或代码。",
        ]
    )
    (report_dir / "g13_execution_note_cn.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["plan", "smoke", "run"], default="plan")
    parser.add_argument("--candidate", choices=sorted(CANDIDATES.keys()), default="G13C")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--ensure-teacher", action="store_true")
    args = parser.parse_args()
    report_dir = Path(args.report_dir)
    write_plan(report_dir)
    if args.mode == "plan":
        print(report_dir / "g13_execution_note_cn.md")
        return
    run_one(
        candidate_id=str(args.candidate),
        seed=int(args.seed),
        report_dir=report_dir,
        smoke=(args.mode == "smoke"),
        ensure_teacher=bool(args.ensure_teacher),
    )


if __name__ == "__main__":
    main()
