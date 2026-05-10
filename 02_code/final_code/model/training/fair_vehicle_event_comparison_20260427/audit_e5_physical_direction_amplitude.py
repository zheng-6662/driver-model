# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[4]
REPORTS_DIR = PROJECT_ROOT / "04_project_logs" / "reports"
DEFAULT_SUMMARY_DIR = REPORTS_DIR / "style_physio_eeg_e5_distill_summary_20260508"
DEFAULT_INDEX = DEFAULT_SUMMARY_DIR / "prediction_figure_index.csv"
DEFAULT_OUT_DIR = DEFAULT_SUMMARY_DIR / "physical_direction_amplitude_audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit steering direction and amplitude errors from saved predictions.")
    parser.add_argument("--prediction-index", default=str(DEFAULT_INDEX))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--major-amp", type=float, default=0.20)
    parser.add_argument("--large-amp", type=float, default=0.30)
    parser.add_argument("--point-eps", type=float, default=0.03)
    parser.add_argument("--area-mean-threshold", type=float, default=0.05)
    parser.add_argument("--under-ratio", type=float, default=0.70)
    parser.add_argument("--severe-under-ratio", type=float, default=0.45)
    parser.add_argument("--over-ratio", type=float, default=1.50)
    return parser.parse_args()


def _read_prediction_index(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"prediction index not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_sequences(run_root: Path) -> dict[str, Any]:
    path = run_root / "prediction_figures" / "test" / "prediction_sequences.npz"
    if not path.exists():
        raise FileNotFoundError(f"prediction sequence file not found: {path}")
    arrays = np.load(str(path), allow_pickle=True)
    channel_names = arrays["channel_names"].astype(str).tolist()
    steer_idx = channel_names.index("steer_rel") if "steer_rel" in channel_names else 0
    return {
        "pred": arrays["pred"][:, :, steer_idx].astype(float),
        "true": arrays["true"][:, :, steer_idx].astype(float),
        "mask": arrays["mask"] > 0.5,
        "sample_key": arrays["sample_key"].astype(str).tolist(),
    }


def _safe_sign(value: float, eps: float) -> int:
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0


def _sample_row(
    experiment_id: str,
    experiment_name: str,
    seed: int,
    sample_key: str,
    true: np.ndarray,
    pred: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    true_peak_idx = int(np.argmax(np.abs(true)))
    pred_peak_idx = int(np.argmax(np.abs(pred)))
    true_peak_signed = float(true[true_peak_idx])
    pred_at_true_peak = float(pred[true_peak_idx])
    pred_peak_signed = float(pred[pred_peak_idx])
    true_amp = abs(true_peak_signed)
    pred_amp = abs(pred_peak_signed)
    amp_ratio = pred_amp / max(true_amp, 1e-6)

    true_area_mean = float(np.mean(true))
    pred_area_mean = float(np.mean(pred))
    true_area_sign = _safe_sign(true_area_mean, args.area_mean_threshold)
    pred_area_sign = _safe_sign(pred_area_mean, args.area_mean_threshold)

    true_peak_sign = _safe_sign(true_peak_signed, args.point_eps)
    pred_at_true_peak_sign = _safe_sign(pred_at_true_peak, args.point_eps)
    pred_peak_sign = _safe_sign(pred_peak_signed, args.point_eps)

    significant = np.abs(true) >= args.major_amp
    opposite_side = significant & (true * pred < -(args.point_eps**2))
    significant_n = int(significant.sum())
    opposite_side_rate = float(opposite_side.sum() / significant_n) if significant_n else np.nan

    is_major = bool(true_amp >= args.major_amp)
    is_large = bool(true_amp >= args.large_amp)
    row: dict[str, Any] = {
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "seed": seed,
        "sample_key": sample_key,
        "true_amp": true_amp,
        "pred_amp": pred_amp,
        "amp_ratio": amp_ratio,
        "true_peak_signed": true_peak_signed,
        "pred_at_true_peak": pred_at_true_peak,
        "pred_peak_signed": pred_peak_signed,
        "true_peak_idx": true_peak_idx,
        "pred_peak_idx": pred_peak_idx,
        "true_peak_time_s": true_peak_idx * 0.005,
        "pred_peak_time_s": pred_peak_idx * 0.005,
        "true_area_mean": true_area_mean,
        "pred_area_mean": pred_area_mean,
        "true_peak_sign": true_peak_sign,
        "pred_at_true_peak_sign": pred_at_true_peak_sign,
        "pred_peak_sign": pred_peak_sign,
        "true_area_sign": true_area_sign,
        "pred_area_sign": pred_area_sign,
        "significant_point_count": significant_n,
        "opposite_side_rate": opposite_side_rate,
        "is_major_response": is_major,
        "is_large_response": is_large,
        "peak_side_wrong_at_true_peak": bool(is_major and true_peak_sign != 0 and pred_at_true_peak_sign == -true_peak_sign),
        "peak_side_wrong_at_pred_peak": bool(is_major and true_peak_sign != 0 and pred_peak_sign == -true_peak_sign),
        "area_side_wrong": bool(true_area_sign != 0 and pred_area_sign == -true_area_sign),
        "under_amp": bool(is_major and amp_ratio < args.under_ratio),
        "severe_under_amp": bool(is_large and amp_ratio < args.severe_under_ratio),
        "over_amp": bool(is_major and amp_ratio > args.over_ratio),
        "opposite_side_heavy": bool(is_major and not np.isnan(opposite_side_rate) and opposite_side_rate >= 0.20),
    }
    return row


def _build_detail_rows(rows: list[dict[str, str]], args: argparse.Namespace) -> pd.DataFrame:
    detail_rows: list[dict[str, Any]] = []
    for item in rows:
        experiment_id = str(item["experiment_id"])
        seed = int(item["seed"])
        run_root = Path(str(item["run_root"]))
        seq = _load_sequences(run_root)
        for idx, sample_key in enumerate(seq["sample_key"]):
            valid = seq["mask"][idx]
            true = seq["true"][idx][valid]
            pred = seq["pred"][idx][valid]
            if len(true) == 0:
                continue
            detail_rows.append(
                _sample_row(
                    experiment_id=experiment_id,
                    experiment_name=str(item.get("experiment_name", experiment_id)),
                    seed=seed,
                    sample_key=sample_key,
                    true=true,
                    pred=pred,
                    args=args,
                )
            )
    return pd.DataFrame(detail_rows)


def _summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (experiment_id, seed), group in df.groupby(["experiment_id", "seed"], sort=True):
        major = group[group["is_major_response"]]
        large = group[group["is_large_response"]]
        area_eval = group[group["true_area_sign"] != 0]
        item: dict[str, Any] = {
            "experiment_id": experiment_id,
            "experiment_name": str(group["experiment_name"].iloc[0]),
            "seed": int(seed),
            "n_samples": int(len(group)),
            "n_major": int(len(major)),
            "n_large": int(len(large)),
            "major_rate": float(len(major) / max(len(group), 1)),
            "median_amp_ratio_major": float(major["amp_ratio"].median()) if len(major) else np.nan,
            "mean_amp_ratio_major": float(major["amp_ratio"].mean()) if len(major) else np.nan,
            "under_amp_rate_major": float(major["under_amp"].mean()) if len(major) else np.nan,
            "severe_under_amp_rate_large": float(large["severe_under_amp"].mean()) if len(large) else np.nan,
            "over_amp_rate_major": float(major["over_amp"].mean()) if len(major) else np.nan,
            "peak_side_wrong_at_true_peak_rate_major": float(major["peak_side_wrong_at_true_peak"].mean())
            if len(major)
            else np.nan,
            "peak_side_wrong_at_pred_peak_rate_major": float(major["peak_side_wrong_at_pred_peak"].mean())
            if len(major)
            else np.nan,
            "area_side_wrong_rate": float(area_eval["area_side_wrong"].mean()) if len(area_eval) else np.nan,
            "opposite_side_heavy_rate_major": float(major["opposite_side_heavy"].mean()) if len(major) else np.nan,
            "mean_opposite_side_rate_major": float(major["opposite_side_rate"].mean()) if len(major) else np.nan,
        }
        rows.append(item)

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary

    mean_rows: list[dict[str, Any]] = []
    numeric_cols = [
        "n_samples",
        "n_major",
        "n_large",
        "major_rate",
        "median_amp_ratio_major",
        "mean_amp_ratio_major",
        "under_amp_rate_major",
        "severe_under_amp_rate_large",
        "over_amp_rate_major",
        "peak_side_wrong_at_true_peak_rate_major",
        "peak_side_wrong_at_pred_peak_rate_major",
        "area_side_wrong_rate",
        "opposite_side_heavy_rate_major",
        "mean_opposite_side_rate_major",
    ]
    for experiment_id, group in summary.groupby("experiment_id", sort=True):
        item = {
            "experiment_id": experiment_id,
            "experiment_name": str(group["experiment_name"].iloc[0]),
            "seed": "mean",
        }
        for col in numeric_cols:
            item[col] = float(pd.to_numeric(group[col], errors="coerce").mean())
        mean_rows.append(item)
    return pd.concat([summary, pd.DataFrame(mean_rows)], ignore_index=True)


def _fmt_pct(value: Any) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value) * 100:.1f}%"


def _fmt_num(value: Any) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):.3f}"


def _write_report(summary: pd.DataFrame, out_path: Path, args: argparse.Namespace) -> None:
    mean = summary[summary["seed"].astype(str).eq("mean")].copy()
    order = ["E0", "E2", "E3", "E4", "E5A", "E5B", "E6", "E7A", "E7B", "E7C", "E8"]
    mean["order"] = mean["experiment_id"].map({name: i for i, name in enumerate(order)}).fillna(99)
    mean = mean.sort_values(["order", "experiment_id"])

    lines: list[str] = []
    lines.append("# 方向与幅值物理复核")
    lines.append("")
    lines.append("## 为什么补这个检查")
    lines.append("")
    lines.append("人工看图发现：有些曲线只是趋势相似，但方向盘响应的物理含义可能不对。例如真实曲线在零线上方、预测在零线下方，或者真实是明显大幅转向，预测却只有很小幅度。")
    lines.append("")
    lines.append("因此这里不重新训练，只基于已有 `prediction_sequences.npz` 做额外诊断。这个诊断不替代 RMSE，而是补充检查“方向是否错号”和“幅值是否明显不足”。")
    lines.append("")
    lines.append("## 统计口径")
    lines.append("")
    lines.append(f"- 主要响应样本：真实方向盘最大绝对幅值 >= `{args.major_amp}`。")
    lines.append(f"- 大幅响应样本：真实方向盘最大绝对幅值 >= `{args.large_amp}`。")
    lines.append(f"- 幅值不足：主要响应样本中，预测最大幅值 / 真实最大幅值 < `{args.under_ratio}`。")
    lines.append(f"- 严重幅值不足：大幅响应样本中，预测最大幅值 / 真实最大幅值 < `{args.severe_under_ratio}`。")
    lines.append(f"- 主峰错号：在真实主峰时刻，预测值和真实主峰方向相反。零线附近用 `{args.point_eps}` 作为容忍阈值。")
    lines.append("")
    lines.append("## 三种子均值")
    lines.append("")
    lines.append("| 版本 | 主响应样本占比 | 主响应幅值比中位数 | 幅值不足率 | 大幅样本严重幅值不足率 | 真实主峰时刻错号率 | 预测主峰错号率 | 零线两侧明显相反率 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for _, row in mean.iterrows():
        lines.append(
            "| "
            + str(row["experiment_id"])
            + " | "
            + _fmt_pct(row["major_rate"])
            + " | "
            + _fmt_num(row["median_amp_ratio_major"])
            + " | "
            + _fmt_pct(row["under_amp_rate_major"])
            + " | "
            + _fmt_pct(row["severe_under_amp_rate_large"])
            + " | "
            + _fmt_pct(row["peak_side_wrong_at_true_peak_rate_major"])
            + " | "
            + _fmt_pct(row["peak_side_wrong_at_pred_peak_rate_major"])
            + " | "
            + _fmt_pct(row["opposite_side_heavy_rate_major"])
            + " |"
        )
    lines.append("")
    lines.append("## 读数解释")
    lines.append("")
    lines.append("- 当前模型确实不只是“尖峰时间”问题。尖峰时间总体还可以，但幅值不足和局部错号更接近人工看图时发现的问题。")
    lines.append("- 如果一个版本 RMSE 略好，但幅值比明显偏低，说明它可能在做更保守的平均预测：趋势大致像，但把真实的大动作压小。")
    lines.append("- 如果主峰错号或零线两侧明显相反率偏高，就不能只说趋势对，因为方向盘向左和向右的物理意义相反。")
    lines.append("- E5B 弱于 E5A 的结果支持一个判断：当前无 EEG 生理状态直接加入学生端可能带来噪声或冲突，不能证明它有稳定额外价值。")
    lines.append("- E4 相对 E3 更好，仍说明 EEG 本身可能有信息；但 EEG 加其他生理信号的模型不好，不等于 EEG 不行，更可能是其他生理信号质量、表示方式或融合方式有问题。")
    lines.append("")
    lines.append("## 下一步建议")
    lines.append("")
    lines.append("1. 后续模型选择不能只看 RMSE，需要同时报告方向错号率、幅值不足率和预测图。")
    lines.append("2. 如果继续改模型，优先针对“幅值被压小”和“零线方向错号”改损失或结构，而不是继续堆输入。")
    lines.append("3. 生理数据下一步应做信号分组消融：EEG 单独、无 EEG 生理单独、EEG + 单类生理，而不是把所有生理信号一次性拼接。")
    lines.append("4. 当前不建议把 E5B 推为主线；E5A 仍是候选，但必须带着上述物理风险进行保守表述。")
    lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    prediction_index = Path(args.prediction_index)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_prediction_index(prediction_index)
    detail = _build_detail_rows(rows, args)
    summary = _summarize(detail)

    detail.to_csv(out_dir / "physical_direction_amplitude_detail.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "physical_direction_amplitude_summary.csv", index=False, encoding="utf-8-sig")
    _write_report(summary, out_dir / "physical_direction_amplitude_report_cn.md", args)
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
