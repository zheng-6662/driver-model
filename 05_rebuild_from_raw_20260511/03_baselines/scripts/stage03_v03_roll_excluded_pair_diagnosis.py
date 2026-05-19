# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(os.environ.get("DATA_PROCESS_ROOT", r"F:/data_set_process/data_process"))
ROOT = Path(os.environ.get("REBUILD_ROOT", str(PROJECT_ROOT / "05_rebuild_from_raw_20260511")))
SCRIPT_DIR = ROOT / "03_baselines" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import stage03_v03_vehicle_only_baselines as base  # noqa: E402


SOURCE_ROOT = ROOT / "03_baselines" / "stage03_v03_excluded_stratified_inclusion"
DATASET_ROOT = ROOT / "03_processed_datasets" / "extreme_condition_v0_3_excluded_stratified_inclusion"
OUT_ROOT = ROOT / "03_baselines" / "stage03_v03_roll_excluded_pair_diagnosis"
TABLE_DIR = OUT_ROOT / "tables"
FIG_DIR = OUT_ROOT / "figures"
REPORT_DIR = ROOT / "09_reports"
NOTES_DIR = ROOT / "00_project_notes"
DAILY_LOG = NOTES_DIR / "daily_logs" / "2026-05-19.md"
ARTIFACT_INDEX = NOTES_DIR / "ARTIFACT_INDEX_CN.md"

REF_VARIANT = "v03_plus_review_ref"
ROLL_VARIANT = "v03_plus_review_excluded_roll_no_lateral"

matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False


def ensure_dirs() -> None:
    for path in [TABLE_DIR, FIG_DIR, REPORT_DIR, NOTES_DIR / "daily_logs"]:
        path.mkdir(parents=True, exist_ok=True)


def load_variant(variant_id: str) -> dict[str, Any]:
    variant_dir = SOURCE_ROOT / variant_id
    table_dir = variant_dir / "tables"
    metrics = pd.read_csv(table_dir / f"{variant_id}_baseline_metrics.csv", encoding="utf-8-sig")
    per = pd.read_csv(table_dir / f"{variant_id}_per_sample_metrics.csv", encoding="utf-8-sig")
    meta = pd.read_csv(DATASET_ROOT / variant_id / "tables" / f"{variant_id}_manifest.csv", encoding="utf-8-sig", low_memory=False)
    pred = np.load(variant_dir / f"{variant_id}_predictions.npz", allow_pickle=True)
    best_model = str(metrics[metrics["split"].astype(str).eq("test")].sort_values("rmse_steer").iloc[0]["model_name"])
    best_per = per[(per["split"].astype(str).eq("test")) & (per["model_name"].astype(str).eq(best_model))].copy()
    sample_ids = [str(x) for x in pred["sample_id"].tolist()]
    sample_index = {sid: i for i, sid in enumerate(sample_ids)}
    return {
        "variant_id": variant_id,
        "metrics": metrics,
        "per": best_per,
        "meta": meta,
        "pred": pred,
        "best_model": best_model,
        "sample_index": sample_index,
        "prediction": pred[f"pred_{best_model}"],
        "y_true": pred["y_true"],
        "y_mask": pred["y_mask"].astype(bool),
        "label_time": pred["label_time"].astype(float),
    }


def bool_col(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def paired_common(ref: dict[str, Any], roll: dict[str, Any]) -> pd.DataFrame:
    r = ref["per"].copy()
    g = roll["per"].copy()
    common = sorted(set(r["sample_id"].astype(str)) & set(g["sample_id"].astype(str)))
    r = r[r["sample_id"].astype(str).isin(common)].copy()
    g = g[g["sample_id"].astype(str).isin(common)].copy()
    keep = [
        "sample_id",
        "subject",
        "session_stamp",
        "v0_3_category",
        "v0_3_category_cn",
        "condition_context_cn",
        "sample_rmse",
        "gt_peak_signed",
        "pred_peak_signed",
        "gt_peak_abs",
        "pred_peak_abs",
        "large_response",
        "wrong_side_large",
        "severe_amp_under_large",
    ]
    paired = r[keep].merge(g[keep], on="sample_id", suffixes=("_ref", "_roll"))
    paired["sample_rmse_delta_roll_minus_ref"] = paired["sample_rmse_roll"] - paired["sample_rmse_ref"]
    paired["abs_pred_peak_delta_roll_minus_ref"] = paired["pred_peak_abs_roll"] - paired["pred_peak_abs_ref"]
    paired["wrong_side_improved"] = bool_col(paired["wrong_side_large_ref"]) & ~bool_col(paired["wrong_side_large_roll"])
    paired["wrong_side_worsened"] = ~bool_col(paired["wrong_side_large_ref"]) & bool_col(paired["wrong_side_large_roll"])
    paired["severe_under_improved"] = bool_col(paired["severe_amp_under_large_ref"]) & ~bool_col(paired["severe_amp_under_large_roll"])
    paired["severe_under_worsened"] = ~bool_col(paired["severe_amp_under_large_ref"]) & bool_col(paired["severe_amp_under_large_roll"])
    paired["large_response"] = bool_col(paired["large_response_ref"]) | bool_col(paired["large_response_roll"])
    paired["major_physical_improved"] = paired["wrong_side_improved"] | paired["severe_under_improved"]
    paired["major_physical_worsened"] = paired["wrong_side_worsened"] | paired["severe_under_worsened"]
    return paired.sort_values("sample_rmse_delta_roll_minus_ref")


def group_summary(paired: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, g in paired.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        large = g[g["large_response"]]
        row.update(
            {
                "n": int(len(g)),
                "mean_delta_roll_minus_ref": float(g["sample_rmse_delta_roll_minus_ref"].mean()),
                "median_delta_roll_minus_ref": float(g["sample_rmse_delta_roll_minus_ref"].median()),
                "improved_n": int((g["sample_rmse_delta_roll_minus_ref"] < 0).sum()),
                "worsened_n": int((g["sample_rmse_delta_roll_minus_ref"] > 0).sum()),
                "large_n": int(len(large)),
                "large_mean_delta": float(large["sample_rmse_delta_roll_minus_ref"].mean()) if len(large) else np.nan,
                "wrong_side_improved_n": int(g["wrong_side_improved"].sum()),
                "wrong_side_worsened_n": int(g["wrong_side_worsened"].sum()),
                "severe_under_improved_n": int(g["severe_under_improved"].sum()),
                "severe_under_worsened_n": int(g["severe_under_worsened"].sum()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["large_n", "n"], ascending=[False, False])


def plot_overlay(sample_ids: list[str], ref: dict[str, Any], roll: dict[str, Any], paired: pd.DataFrame, out_path: Path, title: str) -> None:
    if not sample_ids:
        return
    n = min(len(sample_ids), 12)
    cols = 3
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.2, rows * 3.2), squeeze=False)
    t = ref["label_time"]
    pair_index = paired.set_index("sample_id")
    for ax, sid in zip(axes.ravel(), sample_ids[:n]):
        if sid not in ref["sample_index"] or sid not in roll["sample_index"]:
            ax.axis("off")
            continue
        ir = ref["sample_index"][sid]
        ig = roll["sample_index"][sid]
        mask = ref["y_mask"][ir]
        row = pair_index.loc[sid]
        ax.plot(t[mask], ref["y_true"][ir][mask], color="#111827", linewidth=2.2, label="真实")
        ax.plot(t[mask], ref["prediction"][ir][mask], color="#2563eb", linewidth=1.8, label="参考版本")
        ax.plot(t[mask], roll["prediction"][ig][mask], color="#dc2626", linewidth=1.8, label="横滚版本")
        ax.axhline(0, color="#9ca3af", linewidth=0.8)
        ax.grid(True, alpha=0.22)
        ax.set_title(
            f"{row['subject_ref']} | Δ={row['sample_rmse_delta_roll_minus_ref']:+.3f} | "
            f"真实峰={row['gt_peak_signed_ref']:+.2f}",
            fontsize=9,
        )
        ax.tick_params(labelsize=8)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def fmt(value: Any) -> str:
    try:
        v = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(v):
        return "NA"
    return f"{v:.4f}"


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 20) -> str:
    view = df[columns].head(max_rows).copy()
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in view.iterrows():
        vals = []
        for col in columns:
            val = row[col]
            vals.append(fmt(val) if isinstance(val, (float, int, np.floating, np.integer)) else str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(paired: pd.DataFrame, roll_only: pd.DataFrame, subject_summary: pd.DataFrame, context_summary: pd.DataFrame) -> None:
    report_path = REPORT_DIR / "stage03_v03_roll_excluded_pair_diagnosis_user_summary_cn.md"
    large = paired[paired["large_response"]]
    improved = paired[paired["sample_rmse_delta_roll_minus_ref"] < 0]
    worsened = paired[paired["sample_rmse_delta_roll_minus_ref"] > 0]
    physical_imp = paired[paired["major_physical_improved"]]
    physical_worse = paired[paired["major_physical_worsened"]]
    lines = [
        "# 横滚/姿态 excluded 版本 paired 诊断（用户查看版）",
        "",
        "## 这次为什么做",
        "",
        "横滚/姿态 excluded 版本整体 RMSE 比参考版本差，但大响应错侧率和严重幅值不足率更好。本报告只比较两版共同测试样本，并单独检查新增横滚/姿态 excluded 样本，判断它到底是在改善关键极限样本，还是只是改变测试集组成。",
        "",
        "## 共同测试样本结论",
        "",
        f"- 共同测试样本数：{len(paired)}。",
        f"- 横滚版本逐样本 RMSE 改善：{len(improved)} 个；恶化：{len(worsened)} 个。",
        f"- 共同测试样本平均 ΔRMSE（横滚版本 - 参考版本）：{fmt(paired['sample_rmse_delta_roll_minus_ref'].mean())}。负数代表横滚版本更好。",
        f"- 大响应共同样本数：{len(large)}；大响应平均 ΔRMSE：{fmt(large['sample_rmse_delta_roll_minus_ref'].mean()) if len(large) else 'NA'}。",
        f"- 错侧/严重幅值不足至少一项改善的样本：{len(physical_imp)} 个；至少一项恶化的样本：{len(physical_worse)} 个。",
        "",
        "## 新增横滚/姿态 excluded 测试样本",
        "",
        f"- 横滚版本中新增 excluded 测试样本数：{len(roll_only)}。",
        f"- 其中大响应样本数：{int(bool_col(roll_only['large_response']).sum()) if len(roll_only) else 0}。",
        f"- 新增 excluded 的平均逐样本 RMSE：{fmt(roll_only['sample_rmse'].mean()) if len(roll_only) else 'NA'}。",
        f"- 新增 excluded 的大响应错侧率：{fmt(bool_col(roll_only['wrong_side_large']).mean()) if len(roll_only) else 'NA'}。",
        f"- 新增 excluded 的严重幅值不足率：{fmt(bool_col(roll_only['severe_amp_under_large']).mean()) if len(roll_only) else 'NA'}。",
        "",
        "## 分被试结果",
        "",
        markdown_table(subject_summary, ["subject_ref", "n", "mean_delta_roll_minus_ref", "large_n", "large_mean_delta", "wrong_side_improved_n", "severe_under_improved_n"], 30),
        "",
        "## 分工况来源结果",
        "",
        markdown_table(context_summary, ["condition_context_cn_ref", "n", "mean_delta_roll_minus_ref", "large_n", "large_mean_delta", "wrong_side_improved_n", "severe_under_improved_n"], 30),
        "",
        "## 当前判断",
        "",
        "- 如果只看整体 RMSE，横滚/姿态版本不能直接替代参考版本。",
        "- 如果看大响应物理问题，横滚/姿态版本有继续研究价值，尤其要看错侧和严重幅值不足是否集中改善在强姿态/大响应样本。",
        "- 更合理的后续方向不是把横滚/姿态 excluded 全部混入普通训练，而是把它作为极限姿态子集：单独复核、加权训练，或者做响应类型分支。",
        "",
        "## 可查看文件",
        "",
        f"- paired 明细表：`{(TABLE_DIR / 'roll_vs_ref_common_test_paired_metrics.csv').as_posix()}`",
        f"- 改善最多样本表：`{(TABLE_DIR / 'top_roll_improved_common_test.csv').as_posix()}`",
        f"- 恶化最多样本表：`{(TABLE_DIR / 'top_roll_worsened_common_test.csv').as_posix()}`",
        f"- 改善样本对比图：`{(FIG_DIR / 'roll_vs_ref_top_improved_common_test.png').as_posix()}`",
        f"- 恶化样本对比图：`{(FIG_DIR / 'roll_vs_ref_top_worsened_common_test.png').as_posix()}`",
        f"- 物理指标改善样本对比图：`{(FIG_DIR / 'roll_vs_ref_physical_improved_common_test.png').as_posix()}`",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8-sig")


def append_notes() -> None:
    section = (
        "## 2026-05-19 横滚/姿态 excluded paired 诊断\n\n"
        "- 当前阶段：检查横滚/姿态 excluded 版本是否真的改善大响应物理问题。\n"
        "- 已完成：共同测试样本 paired 对比、新增横滚/姿态 excluded 测试样本统计、改善/恶化样本对比图。\n"
        f"- 用户查看版报告：`{(REPORT_DIR / 'stage03_v03_roll_excluded_pair_diagnosis_user_summary_cn.md').as_posix()}`。\n"
        f"- 输出目录：`{OUT_ROOT.as_posix()}`。\n"
    )
    for path in [DAILY_LOG, NOTES_DIR / "PROJECT_STATUS_CN.md", NOTES_DIR / "TASK_QUEUE_CN.md"]:
        if not path.exists():
            continue
        raw = path.read_text(encoding="utf-8")
        if "## 2026-05-19 横滚/姿态 excluded paired 诊断" not in raw:
            path.write_text(raw.rstrip() + "\n\n" + section, encoding="utf-8")
    if ARTIFACT_INDEX.exists():
        raw = ARTIFACT_INDEX.read_text(encoding="utf-8")
        block = (
            "## 横滚/姿态 excluded paired 诊断\n\n"
            f"- 用户查看版报告：`{(REPORT_DIR / 'stage03_v03_roll_excluded_pair_diagnosis_user_summary_cn.md').as_posix()}`\n"
            f"- paired 明细表：`{(TABLE_DIR / 'roll_vs_ref_common_test_paired_metrics.csv').as_posix()}`\n"
            f"- 输出目录：`{OUT_ROOT.as_posix()}`\n"
        )
        if "## 横滚/姿态 excluded paired 诊断" not in raw:
            ARTIFACT_INDEX.write_text(raw.rstrip() + "\n\n" + block, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    ref = load_variant(REF_VARIANT)
    roll = load_variant(ROLL_VARIANT)
    paired = paired_common(ref, roll)
    paired.to_csv(TABLE_DIR / "roll_vs_ref_common_test_paired_metrics.csv", index=False, encoding="utf-8-sig")
    paired.head(30).to_csv(TABLE_DIR / "top_roll_improved_common_test.csv", index=False, encoding="utf-8-sig")
    paired.tail(30).sort_values("sample_rmse_delta_roll_minus_ref", ascending=False).to_csv(
        TABLE_DIR / "top_roll_worsened_common_test.csv", index=False, encoding="utf-8-sig"
    )
    physical = paired[paired["major_physical_improved"]].sort_values("sample_rmse_delta_roll_minus_ref")
    physical.to_csv(TABLE_DIR / "roll_physical_improved_common_test.csv", index=False, encoding="utf-8-sig")

    subject_summary = group_summary(paired, ["subject_ref"])
    context_summary = group_summary(paired, ["condition_context_cn_ref"])
    category_summary = group_summary(paired, ["v0_3_category_cn_ref"])
    subject_summary.to_csv(TABLE_DIR / "roll_vs_ref_by_subject.csv", index=False, encoding="utf-8-sig")
    context_summary.to_csv(TABLE_DIR / "roll_vs_ref_by_context.csv", index=False, encoding="utf-8-sig")
    category_summary.to_csv(TABLE_DIR / "roll_vs_ref_by_category.csv", index=False, encoding="utf-8-sig")

    ref_ids = set(ref["per"]["sample_id"].astype(str))
    roll_only = roll["per"][~roll["per"]["sample_id"].astype(str).isin(ref_ids)].copy()
    roll_only.to_csv(TABLE_DIR / "roll_only_excluded_test_metrics.csv", index=False, encoding="utf-8-sig")

    plot_overlay(
        paired.head(12)["sample_id"].astype(str).tolist(),
        ref,
        roll,
        paired,
        FIG_DIR / "roll_vs_ref_top_improved_common_test.png",
        "横滚/姿态版本改善最多的共同测试样本",
    )
    plot_overlay(
        paired.tail(12).sort_values("sample_rmse_delta_roll_minus_ref", ascending=False)["sample_id"].astype(str).tolist(),
        ref,
        roll,
        paired,
        FIG_DIR / "roll_vs_ref_top_worsened_common_test.png",
        "横滚/姿态版本恶化最多的共同测试样本",
    )
    plot_overlay(
        physical.head(12)["sample_id"].astype(str).tolist(),
        ref,
        roll,
        paired,
        FIG_DIR / "roll_vs_ref_physical_improved_common_test.png",
        "横滚/姿态版本物理指标改善的共同测试样本",
    )

    summary = {
        "common_test_n": int(len(paired)),
        "improved_n": int((paired["sample_rmse_delta_roll_minus_ref"] < 0).sum()),
        "worsened_n": int((paired["sample_rmse_delta_roll_minus_ref"] > 0).sum()),
        "mean_delta_roll_minus_ref": float(paired["sample_rmse_delta_roll_minus_ref"].mean()),
        "large_common_n": int(paired["large_response"].sum()),
        "large_mean_delta": float(paired.loc[paired["large_response"], "sample_rmse_delta_roll_minus_ref"].mean()),
        "major_physical_improved_n": int(paired["major_physical_improved"].sum()),
        "major_physical_worsened_n": int(paired["major_physical_worsened"].sum()),
        "roll_only_excluded_test_n": int(len(roll_only)),
        "roll_only_large_n": int(bool_col(roll_only["large_response"]).sum()) if len(roll_only) else 0,
        "roll_only_mean_rmse": float(roll_only["sample_rmse"].mean()) if len(roll_only) else np.nan,
    }
    (TABLE_DIR / "roll_excluded_pair_diagnosis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(paired, roll_only, subject_summary, context_summary)
    append_notes()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
