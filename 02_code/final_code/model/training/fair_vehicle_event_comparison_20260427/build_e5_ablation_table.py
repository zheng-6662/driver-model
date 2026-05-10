# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[4]
REPORTS_DIR = PROJECT_ROOT / "04_project_logs" / "reports"
DEFAULT_E0_E2 = REPORTS_DIR / "style_physio_eeg_e0_e2_summary_fresh_3seed_20260507" / "mean_std_metrics.csv"
DEFAULT_E5 = REPORTS_DIR / "style_physio_eeg_e5_distill_summary_20260508" / "mean_std_metrics.csv"
DEFAULT_DELTA = REPORTS_DIR / "style_physio_eeg_e5_distill_summary_20260508" / "paired_sample_deltas.csv"
DEFAULT_OUT_DIR = REPORTS_DIR / "style_physio_eeg_e5_distill_summary_20260508"

NAMES = {
    "E0": "直接预测 + 连续驾驶风格",
    "E1": "粗细双头，不用连续驾驶风格",
    "E2": "粗细双头 + 连续驾驶风格",
    "E3": "无 EEG 生理状态 + 连续驾驶风格",
    "E4": "含 EEG 生理状态 + 连续驾驶风格",
    "E5A": "EEG 教师 / 无 EEG 学生 + 连续驾驶风格",
}

QUESTIONS = {
    "E0": "raw RMSE 保护性对照",
    "E1": "验证连续驾驶风格是否有用",
    "E2": "当前可部署强基准",
    "E3": "无 EEG 生理状态是否直接有用",
    "E4": "EEG 生理状态是否含有有效信号",
    "E5A": "训练用 EEG 教师、推理不用 EEG 是否有效",
}

CONCLUSIONS = {
    "E0": "保留为直接预测 raw RMSE 对照",
    "E1": "弱于 E2，支持连续驾驶风格有效",
    "E2": "作为当前强基准和主要对照",
    "E3": "弱于 E2，当前无 EEG 生理状态不能直接当主线",
    "E4": "优于 E3，说明 EEG 有信号，但推理依赖 EEG",
    "E5A": "当前最强候选；仍需人工看图确认形状风险",
}

ORDER = ["E0", "E1", "E2", "E3", "E4", "E5A"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Chinese ablation table for E0-E5A.")
    parser.add_argument("--e0-e2-mean", default=str(DEFAULT_E0_E2))
    parser.add_argument("--e5-mean", default=str(DEFAULT_E5))
    parser.add_argument("--paired-deltas", default=str(DEFAULT_DELTA))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    return parser.parse_args()


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def _load_means(e0_e2_path: Path, e5_path: Path) -> pd.DataFrame:
    e0_e2 = pd.read_csv(e0_e2_path)
    e5 = pd.read_csv(e5_path)
    rows = []
    for exp_id in ORDER:
        source = e0_e2 if exp_id == "E1" else e5
        matched = source[source["experiment_id"].astype(str).eq(exp_id)]
        if matched.empty:
            continue
        row = matched.iloc[0]
        rows.append(
            {
                "版本": exp_id,
                "实验含义": NAMES.get(exp_id, exp_id),
                "回答的问题": QUESTIONS.get(exp_id, ""),
                "seeds": int(row["n_seeds"]),
                "test RMSE": f"{_fmt(row['test_steer_rmse_mean'])}±{_fmt(row['test_steer_rmse_std'])}",
                "primary": _fmt(row["primary_rmse_mean"]),
                "tail": _fmt(row["tail_rmse_mean"]),
                "peak_err_s": _fmt(row["peak_err_s_mean"]),
                "selection": _fmt(row["selection_mean"]),
                "当前结论": CONCLUSIONS.get(exp_id, ""),
            }
        )
    return pd.DataFrame(rows)


def _paired_note(delta_path: Path) -> str:
    delta = pd.read_csv(delta_path)
    e5_e2 = delta[delta["pair"].astype(str).eq("E5A:E2")]
    if e5_e2.empty:
        return "暂无 E5A vs E2 配对结果。"
    overall = e5_e2["delta_rmse_2s_abs_steer_mean"].astype(float)
    tail = e5_e2["delta_rmse_tail_abs_steer_mean"].astype(float)
    ci_high = e5_e2["delta_rmse_2s_abs_steer_mean_ci95_high"].astype(float)
    return (
        f"E5A vs E2：整体 2 秒误差平均差 `{_fmt(overall.mean())}`，"
        f"E5A 更好的 seed `{int((overall < 0).sum())}/{len(overall)}`，"
        f"CI 完全小于 0 的 seed `{int((ci_high < 0).sum())}/{len(ci_high)}`；"
        f"尾段误差平均差 `{_fmt(tail.mean())}`。"
    )


def _markdown_table(df: pd.DataFrame) -> str:
    lines = [
        "| " + " | ".join(df.columns.astype(str)) + " |",
        "| " + " | ".join(["---"] * len(df.columns)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(value) for value in row.tolist()) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = _load_means(Path(args.e0_e2_mean), Path(args.e5_mean))
    paired_note = _paired_note(Path(args.paired_deltas))
    table.to_csv(out_dir / "paper_ablation_table_e5a_20260508.csv", index=False, encoding="utf-8-sig")
    lines = [
        "# E0-E5A 论文/组会消融表",
        "",
        _markdown_table(table),
        "",
        "## 配对样本补充说明",
        "",
        f"- {paired_note}",
        "",
        "## 保守汇报口径",
        "",
        "- E2 证明连续驾驶风格在粗细双头结构中稳定有效。",
        "- E4 证明 EEG 相比无 EEG 生理状态有信号。",
        "- E5A 证明 EEG 可以作为训练期教师信号，帮助推理时不需要 EEG 的学生模型。",
        "- E5A 是当前主候选，但仍需要人工看图确认曲线形状后再进入最终结论。",
    ]
    (out_dir / "paper_ablation_table_e5a_20260508.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"ablation_csv: {out_dir / 'paper_ablation_table_e5a_20260508.csv'}")
    print(f"ablation_md: {out_dir / 'paper_ablation_table_e5a_20260508.md'}")


if __name__ == "__main__":
    main()
