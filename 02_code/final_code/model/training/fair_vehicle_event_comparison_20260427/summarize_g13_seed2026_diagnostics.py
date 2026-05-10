# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[4]
REPORTS_DIR = PROJECT_ROOT / "04_project_logs" / "reports"
G13_DIR = REPORTS_DIR / "g13_model_breakthrough_20260510"
RESTORE_DIR = REPORTS_DIR / "restore_checkpoint_audit_20260510"
G11_DIR = REPORTS_DIR / "style_physio_eeg_g11_bad_case_attribution_20260509"
DEFAULT_OUT_DIR = G13_DIR / "g13_seed2026_diagnostics"


BASELINE_ORDER = ["E2", "E5A", "E6", "E10C", "E11A"]
G13_ORDER = ["G13A", "G13B", "G13C", "G13F", "G13H", "G13I"]
ALL_ORDER = BASELINE_ORDER + G13_ORDER


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总 G13 seed2026 与旧基准的诊断指标。")
    parser.add_argument("--g13-index", default=str(G13_DIR / "g13_seed2026_full_index.csv"))
    parser.add_argument("--restore-index", default=str(RESTORE_DIR / "restored_run_index_20260510.csv"))
    parser.add_argument("--g11-catalog", default=str(G11_DIR / "bad_case_catalog.csv"))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--major-amp", type=float, default=0.20)
    parser.add_argument("--large-amp", type=float, default=0.30)
    parser.add_argument("--under-ratio", type=float, default=0.70)
    parser.add_argument("--severe-under-ratio", type=float, default=0.45)
    parser.add_argument("--sign-eps", type=float, default=0.03)
    parser.add_argument("--tail-drift-risk", type=float, default=0.15)
    parser.add_argument("--bad-rmse-threshold", type=float, default=0.75)
    return parser.parse_args()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, encoding="utf-8-sig")


def _normalize_restore_index(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows = df[(df["seed"].astype(int) == seed) & (df["candidate"].isin(BASELINE_ORDER))].copy()
    rows = rows.rename(
        columns={
            "candidate": "experiment_id",
            "label": "experiment_name",
            "test_steer_rmse": "test_rmse",
            "selection": "selection_score",
        }
    )
    return rows[
        [
            "experiment_id",
            "seed",
            "experiment_name",
            "local_run_root",
            "test_rmse",
            "primary_rmse",
            "tail_rmse",
            "peak_err_s",
            "selection_score",
        ]
    ]


def _normalize_g13_index(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows = df[(df["seed"].astype(int) == seed) & (df["experiment_id"].isin(G13_ORDER))].copy()
    rows = rows.rename(
        columns={
            "label": "experiment_name",
            "test_steer_rmse": "test_rmse",
            "selection": "selection_score",
        }
    )
    return rows[
        [
            "experiment_id",
            "seed",
            "experiment_name",
            "local_run_root",
            "test_rmse",
            "primary_rmse",
            "tail_rmse",
            "peak_err_s",
            "selection_score",
        ]
    ]


def _safe_bool_mean(series: pd.Series) -> float:
    if len(series) == 0:
        return math.nan
    return float(series.astype(float).mean())


def _safe_num(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return math.nan


def _fmt_num(value: Any, digits: int = 4) -> str:
    value = _safe_num(value)
    if math.isnan(value):
        return "-"
    return f"{value:.{digits}f}"


def _fmt_pct(value: Any, digits: int = 1) -> str:
    value = _safe_num(value)
    if math.isnan(value):
        return "-"
    return f"{value * 100:.{digits}f}%"


def _signed(value: Any, digits: int = 4) -> str:
    value = _safe_num(value)
    if math.isnan(value):
        return "-"
    return f"{value:+.{digits}f}"


def _safe_sign(value: float, eps: float) -> int:
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0


def _load_sequence_detail(run_root: Path, experiment_id: str, args: argparse.Namespace) -> pd.DataFrame:
    seq_path = run_root / "prediction_figures" / "test" / "prediction_sequences.npz"
    if not seq_path.exists():
        raise FileNotFoundError(seq_path)
    arrays = np.load(str(seq_path), allow_pickle=True)
    channel_names = arrays["channel_names"].astype(str).tolist()
    steer_idx = channel_names.index("steer_rel") if "steer_rel" in channel_names else 0
    pred_pool = arrays["pred"][:, :, steer_idx].astype(float)
    true_pool = arrays["true"][:, :, steer_idx].astype(float)
    mask_pool = arrays["mask"] > 0.5
    sample_keys = arrays["sample_key"].astype(str).tolist()

    rows: list[dict[str, Any]] = []
    for i, sample_key in enumerate(sample_keys):
        valid = mask_pool[i]
        true = true_pool[i][valid]
        pred = pred_pool[i][valid]
        if len(true) == 0:
            continue

        true_peak_idx = int(np.argmax(np.abs(true)))
        pred_peak_idx = int(np.argmax(np.abs(pred)))
        true_peak_value = float(true[true_peak_idx])
        pred_at_true_peak = float(pred[true_peak_idx])
        pred_peak_value = float(pred[pred_peak_idx])
        true_amp = abs(true_peak_value)
        pred_amp = abs(pred_peak_value)
        amp_ratio = pred_amp / max(true_amp, 1e-6)

        tail_start = max(0, int(len(true) * 0.75))
        true_tail_mean = float(np.mean(true[tail_start:]))
        pred_tail_mean = float(np.mean(pred[tail_start:]))
        tail_drift_abs = abs(pred_tail_mean - true_tail_mean)

        significant = np.abs(true) >= args.major_amp
        opposite = significant & (true * pred < -(args.sign_eps**2))
        opposite_rate = float(opposite.sum() / significant.sum()) if int(significant.sum()) else math.nan

        true_peak_sign = _safe_sign(true_peak_value, args.sign_eps)
        pred_at_true_peak_sign = _safe_sign(pred_at_true_peak, args.sign_eps)
        pred_peak_sign = _safe_sign(pred_peak_value, args.sign_eps)

        is_major = true_amp >= args.major_amp
        is_large = true_amp >= args.large_amp

        rows.append(
            {
                "experiment_id": experiment_id,
                "sample_key": sample_key,
                "true_amp": true_amp,
                "pred_amp": pred_amp,
                "amp_ratio": amp_ratio,
                "is_major_response": is_major,
                "is_large_response": is_large,
                "under_amp": bool(is_major and amp_ratio < args.under_ratio),
                "severe_under_amp": bool(is_large and amp_ratio < args.severe_under_ratio),
                "true_peak_sign": true_peak_sign,
                "pred_at_true_peak_sign": pred_at_true_peak_sign,
                "pred_peak_sign": pred_peak_sign,
                "opposite_at_true_peak": bool(is_major and true_peak_sign != 0 and pred_at_true_peak_sign == -true_peak_sign),
                "opposite_at_pred_peak": bool(is_major and true_peak_sign != 0 and pred_peak_sign == -true_peak_sign),
                "opposite_side_rate": opposite_rate,
                "opposite_side_heavy": bool(is_major and not math.isnan(opposite_rate) and opposite_rate >= 0.20),
                "tail_drift_abs": tail_drift_abs,
                "tail_drift_risk": bool(tail_drift_abs >= args.tail_drift_risk),
            }
        )
    return pd.DataFrame(rows)


def _load_model_sample_detail(meta: pd.Series, args: argparse.Namespace, g11_keys: set[str]) -> pd.DataFrame:
    run_root = Path(str(meta["local_run_root"]))
    metrics_path = run_root / "prediction_figures" / "test" / "prediction_sample_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    metrics = _read_csv(metrics_path)
    phys = _load_sequence_detail(run_root, str(meta["experiment_id"]), args)
    merged = metrics.merge(phys, on="sample_key", how="left", suffixes=("", "_phys"))
    merged["experiment_id"] = str(meta["experiment_id"])
    merged["experiment_name"] = str(meta["experiment_name"])
    merged["seed"] = int(meta["seed"])
    merged["is_g11_case"] = merged["sample_key"].astype(str).isin(g11_keys)
    merged["model_bad_rmse"] = pd.to_numeric(merged["rmse_2s_abs_steer"], errors="coerce") >= args.bad_rmse_threshold
    return merged


def _summarize_physical(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for exp_id, group in df.groupby("experiment_id", sort=False):
        major = group[group["is_major_response"].fillna(False)]
        large = group[group["is_large_response"].fillna(False)]
        rows.append(
            {
                "experiment_id": exp_id,
                "experiment_name": str(group["experiment_name"].iloc[0]),
                "n_samples": int(len(group)),
                "major_response_rate": float(len(major) / max(len(group), 1)),
                "large_response_rate": float(len(large) / max(len(group), 1)),
                "median_amp_ratio_major": float(pd.to_numeric(major["amp_ratio"], errors="coerce").median())
                if len(major)
                else math.nan,
                "mean_amp_ratio_major": float(pd.to_numeric(major["amp_ratio"], errors="coerce").mean())
                if len(major)
                else math.nan,
                "under_amp_rate_major": _safe_bool_mean(major["under_amp"]) if len(major) else math.nan,
                "severe_under_amp_rate_large": _safe_bool_mean(large["severe_under_amp"]) if len(large) else math.nan,
                "opposite_at_true_peak_rate_major": _safe_bool_mean(major["opposite_at_true_peak"])
                if len(major)
                else math.nan,
                "opposite_at_pred_peak_rate_major": _safe_bool_mean(major["opposite_at_pred_peak"])
                if len(major)
                else math.nan,
                "opposite_side_heavy_rate_major": _safe_bool_mean(major["opposite_side_heavy"])
                if len(major)
                else math.nan,
                "tail_drift_risk_rate": _safe_bool_mean(group["tail_drift_risk"]),
                "tail_drift_abs_mean": float(pd.to_numeric(group["tail_drift_abs"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows)


def _summarize_subject(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (exp_id, subj), group in df.groupby(["experiment_id", "subj"], sort=False):
        major = group[group["is_major_response"].fillna(False)]
        large = group[group["is_large_response"].fillna(False)]
        rows.append(
            {
                "experiment_id": exp_id,
                "experiment_name": str(group["experiment_name"].iloc[0]),
                "subj": subj,
                "n_samples": int(len(group)),
                "rmse_2s_mean": float(pd.to_numeric(group["rmse_2s_abs_steer"], errors="coerce").mean()),
                "tail_rmse_mean": float(pd.to_numeric(group["rmse_tail_abs_steer"], errors="coerce").mean()),
                "peak_err_s_mean": float(pd.to_numeric(group["peak_time_abs_err_s"], errors="coerce").mean()),
                "under_amp_rate_major": _safe_bool_mean(major["under_amp"]) if len(major) else math.nan,
                "severe_under_amp_rate_large": _safe_bool_mean(large["severe_under_amp"]) if len(large) else math.nan,
                "opposite_at_true_peak_rate_major": _safe_bool_mean(major["opposite_at_true_peak"])
                if len(major)
                else math.nan,
                "tail_drift_risk_rate": _safe_bool_mean(group["tail_drift_risk"]),
                "g11_case_rate": _safe_bool_mean(group["is_g11_case"]),
                "model_bad_rmse_rate": _safe_bool_mean(group["model_bad_rmse"]),
            }
        )

    out = pd.DataFrame(rows)
    macro_rows: list[dict[str, Any]] = []
    numeric_cols = [
        "n_samples",
        "rmse_2s_mean",
        "tail_rmse_mean",
        "peak_err_s_mean",
        "under_amp_rate_major",
        "severe_under_amp_rate_large",
        "opposite_at_true_peak_rate_major",
        "tail_drift_risk_rate",
        "g11_case_rate",
        "model_bad_rmse_rate",
    ]
    for exp_id, group in out.groupby("experiment_id", sort=False):
        row = {
            "experiment_id": exp_id,
            "experiment_name": str(group["experiment_name"].iloc[0]),
            "subj": "subject_macro_mean",
        }
        for col in numeric_cols:
            row[col] = float(pd.to_numeric(group[col], errors="coerce").mean())
        macro_rows.append(row)
    return pd.concat([out, pd.DataFrame(macro_rows)], ignore_index=True)


def _summarize_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (exp_id, group_name), group in df.groupby(["experiment_id", group_col], sort=False):
        major = group[group["is_major_response"].fillna(False)]
        large = group[group["is_large_response"].fillna(False)]
        rows.append(
            {
                "experiment_id": exp_id,
                "experiment_name": str(group["experiment_name"].iloc[0]),
                group_col: group_name,
                "n_samples": int(len(group)),
                "rmse_2s_mean": float(pd.to_numeric(group["rmse_2s_abs_steer"], errors="coerce").mean()),
                "tail_rmse_mean": float(pd.to_numeric(group["rmse_tail_abs_steer"], errors="coerce").mean()),
                "peak_err_s_mean": float(pd.to_numeric(group["peak_time_abs_err_s"], errors="coerce").mean()),
                "under_amp_rate_major": _safe_bool_mean(major["under_amp"]) if len(major) else math.nan,
                "severe_under_amp_rate_large": _safe_bool_mean(large["severe_under_amp"]) if len(large) else math.nan,
                "opposite_at_true_peak_rate_major": _safe_bool_mean(major["opposite_at_true_peak"])
                if len(major)
                else math.nan,
                "tail_drift_risk_rate": _safe_bool_mean(group["tail_drift_risk"]),
                "g11_case_rate": _safe_bool_mean(group["is_g11_case"]),
                "model_bad_rmse_rate": _safe_bool_mean(group["model_bad_rmse"]),
            }
        )
    return pd.DataFrame(rows)


def _summarize_g11(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    g11 = df[df["is_g11_case"].fillna(False)].copy()
    for exp_id, group in g11.groupby("experiment_id", sort=False):
        major = group[group["is_major_response"].fillna(False)]
        large = group[group["is_large_response"].fillna(False)]
        rows.append(
            {
                "experiment_id": exp_id,
                "experiment_name": str(group["experiment_name"].iloc[0]),
                "n_g11": int(len(group)),
                "g11_rmse_mean": float(pd.to_numeric(group["rmse_2s_abs_steer"], errors="coerce").mean()),
                "g11_tail_rmse_mean": float(pd.to_numeric(group["rmse_tail_abs_steer"], errors="coerce").mean()),
                "g11_peak_err_s_mean": float(pd.to_numeric(group["peak_time_abs_err_s"], errors="coerce").mean()),
                "g11_under_amp_rate_major": _safe_bool_mean(major["under_amp"]) if len(major) else math.nan,
                "g11_severe_under_amp_rate_large": _safe_bool_mean(large["severe_under_amp"]) if len(large) else math.nan,
                "g11_opposite_at_true_peak_rate_major": _safe_bool_mean(major["opposite_at_true_peak"])
                if len(major)
                else math.nan,
                "g11_tail_drift_risk_rate": _safe_bool_mean(group["tail_drift_risk"]),
            }
        )
    return pd.DataFrame(rows)


def _add_order(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_order"] = out["experiment_id"].map({name: i for i, name in enumerate(ALL_ORDER)}).fillna(999)
    return out.sort_values(["_order", "experiment_id"]).drop(columns=["_order"])


def _write_markdown(
    out_path: Path,
    overall: pd.DataFrame,
    physical: pd.DataFrame,
    subject: pd.DataFrame,
    morphology: pd.DataFrame,
    g11: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    overall = _add_order(overall)
    physical = _add_order(physical)
    g11 = _add_order(g11)

    e10c = overall[overall["experiment_id"].eq("E10C")].iloc[0]
    e6 = overall[overall["experiment_id"].eq("E6")].iloc[0]
    g13h = overall[overall["experiment_id"].eq("G13H")].iloc[0]
    g13i = overall[overall["experiment_id"].eq("G13I")].iloc[0]

    lines: list[str] = []
    lines.append("# G13 seed2026 筛选诊断总结")
    lines.append("")
    lines.append("## 当前状态")
    lines.append("")
    lines.append("- 旧设置和核心 checkpoint 已经完成恢复。这里的旧基准使用本次恢复后重新训练得到的同协议 seed2026 结果。")
    lines.append("- G13 第一批完整训练已经跑完：G13A、G13B、G13C、G13F、G13H、G13I。")
    lines.append("- 本报告不新增训练，只读取已保存的预测序列和逐样本指标，补充看分被试、分响应类型、G11 困难样本、幅值不足和错侧问题。")
    lines.append("- 注意：G13H/G13I 都属于“训练时脑电教师 + 推理时肌电学生输入 + 连续驾驶风格”的组合；推理阶段不使用脑电。")
    lines.append("")
    lines.append("## 版本含义")
    lines.append("")
    lines.append("| 版本 | 含义 |")
    lines.append("| --- | --- |")
    version_desc = {
        "E2": "车辆数据 + 连续驾驶风格旧基准",
        "E5A": "脑电教师蒸馏，学生推理不使用生理信号",
        "E6": "E5A 基础上加入幅值/方向物理约束",
        "E10C": "车辆数据 + 连续驾驶风格 + 肌电单信号",
        "E11A": "脑电教师蒸馏 + 肌电学生输入旧诊断版",
        "G13A": "连续风格 + 响应类型辅助学习",
        "G13B": "连续风格 + 肌电 + 响应类型辅助学习",
        "G13C": "连续风格 + 肌电 + 响应类型影响轨迹预测",
        "G13F": "肌电 + 响应类型 + 幅值方向物理约束",
        "G13H": "脑电教师 + 肌电学生 + 响应类型辅助学习",
        "G13I": "脑电教师 + 肌电学生 + 困难响应加权 + 物理约束",
    }
    for exp_id in ALL_ORDER:
        lines.append(f"| {exp_id} | {version_desc.get(exp_id, '')} |")
    lines.append("")

    lines.append("## 整体指标")
    lines.append("")
    lines.append("数值越小越好。`相对 E10C` 和 `相对 E6` 为负数表示更好。")
    lines.append("")
    lines.append("| 版本 | test RMSE | 主响应误差 | 尾段误差 | 峰值时间误差 | 综合选择指标 | 相对 E10C | 相对 E6 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for _, row in overall.iterrows():
        delta_e10c = _safe_num(row["test_rmse"]) - _safe_num(e10c["test_rmse"])
        delta_e6 = _safe_num(row["test_rmse"]) - _safe_num(e6["test_rmse"])
        lines.append(
            f"| {row['experiment_id']} | {_fmt_num(row['test_rmse'])} | {_fmt_num(row['primary_rmse'])} | "
            f"{_fmt_num(row['tail_rmse'])} | {_fmt_num(row['peak_err_s'])} | {_fmt_num(row['selection_score'])} | "
            f"{_signed(delta_e10c)} | {_signed(delta_e6)} |"
        )
    lines.append("")

    lines.append("## 物理风险")
    lines.append("")
    lines.append(
        f"口径：主要响应为真实最大幅值 >= {args.major_amp}；大幅响应为真实最大幅值 >= {args.large_amp}；"
        f"幅值不足为预测/真实最大幅值 < {args.under_ratio}；严重幅值不足为大幅样本中预测/真实最大幅值 < {args.severe_under_ratio}。"
    )
    lines.append("")
    lines.append("| 版本 | 主要响应幅值比中位数 | 幅值不足率 | 大幅样本严重幅值不足率 | 真实主峰错侧率 | 明显跨零线相反率 | 后段漂移风险率 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for _, row in physical.iterrows():
        lines.append(
            f"| {row['experiment_id']} | {_fmt_num(row['median_amp_ratio_major'], 3)} | "
            f"{_fmt_pct(row['under_amp_rate_major'])} | {_fmt_pct(row['severe_under_amp_rate_large'])} | "
            f"{_fmt_pct(row['opposite_at_true_peak_rate_major'])} | {_fmt_pct(row['opposite_side_heavy_rate_major'])} | "
            f"{_fmt_pct(row['tail_drift_risk_rate'])} |"
        )
    lines.append("")

    lines.append("## G11 困难样本")
    lines.append("")
    lines.append("这里专门看之前归因出的 111 个困难样本。")
    lines.append("")
    lines.append("| 版本 | G11 RMSE | G11 尾段误差 | G11 峰值时间误差 | G11 幅值不足率 | G11 错侧率 | G11 后段漂移风险 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for _, row in g11.iterrows():
        lines.append(
            f"| {row['experiment_id']} | {_fmt_num(row['g11_rmse_mean'])} | {_fmt_num(row['g11_tail_rmse_mean'])} | "
            f"{_fmt_num(row['g11_peak_err_s_mean'])} | {_fmt_pct(row['g11_under_amp_rate_major'])} | "
            f"{_fmt_pct(row['g11_opposite_at_true_peak_rate_major'])} | {_fmt_pct(row['g11_tail_drift_risk_rate'])} |"
        )
    lines.append("")

    lines.append("## 分被试结论")
    lines.append("")
    macro = subject[subject["subj"].astype(str).eq("subject_macro_mean")].copy()
    macro = _add_order(macro)
    lines.append("| 版本 | 被试平均 RMSE | 被试平均尾段误差 | 被试平均坏样本率 | 被试平均幅值不足率 | 被试平均后段漂移风险 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for _, row in macro.iterrows():
        lines.append(
            f"| {row['experiment_id']} | {_fmt_num(row['rmse_2s_mean'])} | {_fmt_num(row['tail_rmse_mean'])} | "
            f"{_fmt_pct(row['model_bad_rmse_rate'])} | {_fmt_pct(row['under_amp_rate_major'])} | {_fmt_pct(row['tail_drift_risk_rate'])} |"
        )
    lines.append("")

    lines.append("## 分响应类型重点")
    lines.append("")
    focus = morphology[morphology["eval_morphology_label"].isin(["reverse_correction", "multi_correction", "single_lobe"])].copy()
    focus = focus[focus["experiment_id"].isin(["E6", "E10C", "G13H", "G13I"])]
    focus = _add_order(focus)
    lines.append("| 版本 | 响应类型 | 样本数 | RMSE | 尾段误差 | 幅值不足率 | 后段漂移风险 |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for _, row in focus.iterrows():
        lines.append(
            f"| {row['experiment_id']} | {row['eval_morphology_label']} | {int(row['n_samples'])} | "
            f"{_fmt_num(row['rmse_2s_mean'])} | {_fmt_num(row['tail_rmse_mean'])} | "
            f"{_fmt_pct(row['under_amp_rate_major'])} | {_fmt_pct(row['tail_drift_risk_rate'])} |"
        )
    lines.append("")

    g13h_rmse_delta_e10c = _safe_num(g13h["test_rmse"]) - _safe_num(e10c["test_rmse"])
    g13h_rmse_delta_e6 = _safe_num(g13h["test_rmse"]) - _safe_num(e6["test_rmse"])
    g13i_rmse_delta_e10c = _safe_num(g13i["test_rmse"]) - _safe_num(e10c["test_rmse"])

    lines.append("## 初步判断")
    lines.append("")
    lines.append(
        f"- G13H 是 seed2026 里整体误差最好的新候选：整体 RMSE 比 E10C 低 {_fmt_num(abs(g13h_rmse_delta_e10c))}，"
        f"比 E6 低 {_fmt_num(abs(g13h_rmse_delta_e6))}。它说明“脑电教师 + 肌电学生 + 响应类型辅助学习”这条组合路线比旧 E11A 更合理。"
    )
    lines.append(
        "- 但是 G13H 不是无条件突破：它的幅值不足率最高，G11 困难样本也没有超过 E6。"
        "这说明它可能把一部分样本预测得更保守，从而压低平均误差，但还没有真正解决用户指出的“大幅响应被预测成轻微响应”的问题。"
    )
    lines.append(
        f"- G13I 的整体 RMSE 也比 E10C 低 {_fmt_num(abs(g13i_rmse_delta_e10c))}。"
        "它的幅值不足、错侧和后段漂移风险比 G13H 更均衡，但尾段误差和 G11 RMSE 仍不如 E6。"
        "因此 G13I 更像“物理风险平衡候选”，不是当前最强整体候选。"
    )
    lines.append("- E6 仍然是 G11 困难样本 RMSE 和尾段误差上的重要参照，不能因为 G13H 平均误差更好就把 E6 的困难样本优势忽略掉。")
    lines.append("- G13A 只加响应类型辅助学习，整体不如 E10C/E6；G13B/G13C/G13F 目前都没有体现出值得直接补三种子的整体优势。")
    lines.append("- 当前还不能只用“响应类型辅助学习”替代原有主线；更有希望的是把脑电教师、肌电输入和响应类型监督组合起来。")
    lines.append("")

    lines.append("## 下一步建议")
    lines.append("")
    lines.append("1. 补 G13H 的 seed2027/2028，验证整体误差优势是否稳定。")
    lines.append("2. 同时补 G13I 的 seed2027/2028，原因不是它当前最强，而是它比 G13H 更关注幅值和错侧风险，可以检验“物理风险更均衡但整体略弱”的路线是否稳定。")
    lines.append("3. G13A/B/C/F 暂时不补三种子，除非后续人工看图发现它们在关键大幅/反向/多段样本上有 E10C/E6/G13H/G13I 没有的明显形态优势。")
    lines.append("4. 后续汇报时不要说“模型已经彻底解决物理问题”，更准确的说法是：G13H 在整体误差上出现新突破，G13I 在物理风险上更均衡，但困难样本仍没有完全超过 E6。")
    lines.append("")

    lines.append("## 产物")
    lines.append("")
    lines.append("- `g13_seed2026_overall_comparison.csv`：整体指标与基准差值。")
    lines.append("- `g13_seed2026_physical_summary.csv`：幅值不足、错侧、跨零线、后段漂移。")
    lines.append("- `g13_seed2026_subject_summary.csv`：cwh/gf/tyy 分被试和被试平均。")
    lines.append("- `g13_seed2026_morphology_summary.csv`：按响应类型分组。")
    lines.append("- `g13_seed2026_g11_summary.csv`：G11 困难样本专表。")
    lines.append("- `g13_seed2026_sample_detail.csv`：逐样本明细，供后续画图和抽查。")
    lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    g13_idx = _normalize_g13_index(_read_csv(Path(args.g13_index)), args.seed)
    restore_idx = _normalize_restore_index(_read_csv(Path(args.restore_index)), args.seed)
    overall = pd.concat([restore_idx, g13_idx], ignore_index=True)
    overall = overall[overall["experiment_id"].isin(ALL_ORDER)].copy()
    overall = _add_order(overall)

    e10c_rmse = float(overall.loc[overall["experiment_id"].eq("E10C"), "test_rmse"].iloc[0])
    e6_rmse = float(overall.loc[overall["experiment_id"].eq("E6"), "test_rmse"].iloc[0])
    overall["delta_test_rmse_vs_E10C"] = pd.to_numeric(overall["test_rmse"], errors="coerce") - e10c_rmse
    overall["delta_test_rmse_vs_E6"] = pd.to_numeric(overall["test_rmse"], errors="coerce") - e6_rmse

    g11_catalog = _read_csv(Path(args.g11_catalog))
    g11_keys = set(g11_catalog["sample_key"].astype(str))

    detail_frames = []
    for _, row in overall.iterrows():
        detail_frames.append(_load_model_sample_detail(row, args, g11_keys))
    detail = pd.concat(detail_frames, ignore_index=True)
    detail = _add_order(detail)

    physical = _add_order(_summarize_physical(detail))
    subject = _summarize_subject(detail)
    subject = _add_order(subject)
    morphology = _summarize_group(detail, "eval_morphology_label")
    morphology = _add_order(morphology)
    road = _summarize_group(detail, "road_type_anchor")
    road = _add_order(road)
    g11 = _add_order(_summarize_g11(detail))

    overall.to_csv(out_dir / "g13_seed2026_overall_comparison.csv", index=False, encoding="utf-8-sig")
    physical.to_csv(out_dir / "g13_seed2026_physical_summary.csv", index=False, encoding="utf-8-sig")
    subject.to_csv(out_dir / "g13_seed2026_subject_summary.csv", index=False, encoding="utf-8-sig")
    morphology.to_csv(out_dir / "g13_seed2026_morphology_summary.csv", index=False, encoding="utf-8-sig")
    road.to_csv(out_dir / "g13_seed2026_road_summary.csv", index=False, encoding="utf-8-sig")
    g11.to_csv(out_dir / "g13_seed2026_g11_summary.csv", index=False, encoding="utf-8-sig")
    detail.to_csv(out_dir / "g13_seed2026_sample_detail.csv", index=False, encoding="utf-8-sig")

    _write_markdown(
        out_path=out_dir / "g13_seed2026_screening_summary_cn.md",
        overall=overall,
        physical=physical,
        subject=subject,
        morphology=morphology,
        g11=g11,
        args=args,
    )
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
