# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import font_manager


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[4]
REMOTE_ROOT = Path("/root/autodl-tmp/data_process")
REPORTS_DIR = PROJECT_ROOT / "04_project_logs" / "reports"
G11_CATALOG = REPORTS_DIR / "style_physio_eeg_g11_bad_case_attribution_20260509" / "bad_case_catalog.csv"

DEFAULT_E18_RUNS = REPORTS_DIR / "style_physio_eeg_e18_signal_representation_runs_20260511.csv"
DEFAULT_OUT_DIR = REPORTS_DIR / "style_physio_eeg_e18_signal_representation_seed2026_summary_20260511"

REFERENCE_RUN_FILES = [
    REPORTS_DIR / "style_physio_eeg_e0_e2_fresh_3seed_runs_20260507.csv",
    REPORTS_DIR / "style_physio_eeg_e10_non_eeg_signal_runs_20260509.csv",
    REPORTS_DIR / "style_physio_eeg_e16_eeg_style_runs_20260511.csv",
    REPORTS_DIR / "style_physio_eeg_e17_semantic_single_signal_runs_20260511.csv",
]

REFERENCE_INFO: dict[str, dict[str, Any]] = {
    "E2": {"signal": "无", "method": "无生理", "label": "粗细双头 + 连续风格"},
    "E10A": {"signal": "心率", "method": "原始值", "label": "心率原始值 + 连续风格"},
    "E10B": {"signal": "皮电", "method": "原始值", "label": "皮电原始值 + 连续风格"},
    "E10C": {"signal": "肌电", "method": "原始值", "label": "肌电原始值 + 连续风格"},
    "E16B": {"signal": "脑电", "method": "原始值", "label": "脑电原始值 + 连续风格"},
    "E17A": {"signal": "心率", "method": "旧人工状态", "label": "心率旧人工状态 + 连续风格"},
    "E17B": {"signal": "皮电", "method": "旧人工状态", "label": "皮电旧人工状态 + 连续风格"},
    "E17C": {"signal": "肌电", "method": "旧人工状态", "label": "肌电旧人工状态 + 连续风格"},
    "E17D": {"signal": "脑电", "method": "旧人工状态", "label": "脑电旧人工状态 + 连续风格"},
}

E18_INFO: dict[str, dict[str, Any]] = {
    "E18A": {"signal": "心率", "method": "基线校正", "label": "心率当前值 + 相对基线变化"},
    "E18B": {"signal": "皮电", "method": "基线校正", "label": "皮电当前值 + 相对基线变化"},
    "E18C": {"signal": "肌电", "method": "基线校正", "label": "肌电当前值 + 相对基线变化"},
    "E18D": {"signal": "脑电", "method": "基线校正", "label": "脑电当前值 + 前序事件变化"},
    "E18E": {"signal": "心率", "method": "数据自动表示", "label": "心率数据自动表示"},
    "E18F": {"signal": "皮电", "method": "数据自动表示", "label": "皮电数据自动表示"},
    "E18G": {"signal": "肌电", "method": "数据自动表示", "label": "肌电数据自动表示"},
    "E18H": {"signal": "脑电", "method": "数据自动表示", "label": "脑电数据自动表示"},
    "E18I": {"signal": "心率", "method": "任务相关状态", "label": "心率任务相关状态"},
    "E18J": {"signal": "皮电", "method": "任务相关状态", "label": "皮电任务相关状态"},
    "E18K": {"signal": "肌电", "method": "任务相关状态", "label": "肌电任务相关状态"},
    "E18L": {"signal": "脑电", "method": "任务相关状态", "label": "脑电任务相关状态"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总 E18 无人工权重生理表示筛选结果。")
    parser.add_argument("--e18-runs", default=str(DEFAULT_E18_RUNS))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    return parser.parse_args()


def configure_font() -> None:
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
    ]
    font_path = next((p for p in candidates if p.exists()), None)
    if font_path:
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams["axes.unicode_minus"] = False


def fmt(value: Any, digits: int = 4) -> str:
    try:
        f = float(value)
    except Exception:
        return "NA"
    if not math.isfinite(f):
        return "NA"
    return f"{f:.{digits}f}"


def fmt_pct(value: Any) -> str:
    try:
        f = float(value)
    except Exception:
        return "NA"
    if not math.isfinite(f):
        return "NA"
    return f"{100.0 * f:.1f}%"


def df_to_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "无数据。"
    work = df.copy()
    for col in work.columns:
        if pd.api.types.is_float_dtype(work[col]):
            work[col] = work[col].map(lambda x: fmt(x) if pd.notna(x) else "NA")
        else:
            work[col] = work[col].map(lambda x: "" if pd.isna(x) else str(x))
    headers = list(map(str, work.columns))
    rows = work.astype(str).values.tolist()
    widths = [len(h) for h in headers]
    for row in rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(value))
    return "\n".join(
        [
            "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |",
            "| " + " | ".join("-" * w for w in widths) + " |",
            *["| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(row)) + " |" for row in rows],
        ]
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def localize_path(path_text: str) -> Path:
    raw = str(path_text).strip()
    raw_norm = raw.replace("\\", "/")
    remote_norm = str(REMOTE_ROOT).replace("\\", "/")
    if raw_norm.startswith(remote_norm):
        rel = raw_norm[len(remote_norm) :].lstrip("/")
        return PROJECT_ROOT / rel
    return Path(raw)


def read_records(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "smoke_test" in df.columns:
        smoke = df["smoke_test"].fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})
        df = df[~smoke].copy()
    if "seed" in df.columns:
        df = df[pd.to_numeric(df["seed"], errors="coerce").eq(2026)].copy()
    return df


def metrics_from_run_root(run_root: Path) -> dict[str, Any] | None:
    metrics_path = run_root / "metrics.json"
    if not metrics_path.exists():
        return None
    metrics = read_json(metrics_path)
    test = metrics["test"]
    selection = test["selection_summary"]
    return {
        "test_rmse": float(test["steer_rmse"]),
        "primary_rmse": float(selection["overall_primary_steer_rmse"]),
        "tail_rmse": float(selection["rmse_tail_abs_steer"]),
        "peak_err_s": float(selection["peak_time_abs_err_s"]),
        "selection": float(selection["selection_score"]),
        "best_epoch": int(metrics.get("best_epoch", -1)),
        "prediction_overview": str(run_root / "prediction_figures" / "test" / "overview.png"),
        "sample_metrics_csv": str(run_root / "prediction_figures" / "test" / "prediction_sample_metrics.csv"),
        "sequence_npz": str(run_root / "prediction_figures" / "test" / "prediction_sequences.npz"),
    }


def rows_from_records(path: Path, info_map: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    df = read_records(path)
    rows: list[dict[str, Any]] = []
    if df.empty or "experiment_id" not in df.columns:
        return rows
    for version, info in info_map.items():
        match = df[df["experiment_id"].astype(str).eq(version)]
        if match.empty:
            continue
        row = match.iloc[-1]
        run_root = localize_path(str(row["run_root"]))
        metrics = metrics_from_run_root(run_root)
        if metrics is None:
            continue
        rows.append(
            {
                "version": version,
                "seed": 2026,
                "signal": info["signal"],
                "method": info["method"],
                "label": info["label"],
                "run_root": str(run_root),
                **metrics,
            }
        )
    return rows


def load_reference_rows() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in REFERENCE_RUN_FILES:
        rows.extend(rows_from_records(path, REFERENCE_INFO))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates(subset=["version"], keep="last")


def load_e18_rows(path: Path) -> pd.DataFrame:
    rows = rows_from_records(path, E18_INFO)
    found = {row["version"] for row in rows}
    for version, info in E18_INFO.items():
        if version in found:
            continue
        candidates = sorted((PROJECT_ROOT / "tmp" / "event_conditioned_runs").glob(f"{version}_*seed2026_*"))
        for run_root in reversed(candidates):
            metrics = metrics_from_run_root(run_root)
            if metrics is None:
                continue
            rows.append(
                {
                    "version": version,
                    "seed": 2026,
                    "signal": info["signal"],
                    "method": info["method"],
                    "label": info["label"],
                    "run_root": str(run_root),
                    **metrics,
                }
            )
            break
    return pd.DataFrame(rows)


def add_physical_metrics(sample_df: pd.DataFrame, seq_path: Path) -> pd.DataFrame:
    seq = np.load(seq_path, allow_pickle=True)
    pred = seq["pred"].astype(np.float32)
    true = seq["true"].astype(np.float32)
    mask = seq["mask"].astype(np.float32)
    ctx = seq["ctx_raw"].astype(np.float32)
    anchors = ctx[:, 0].astype(np.float32)
    true_abs = true[:, :, 0] + anchors.reshape(-1, 1)
    pred_abs = pred[:, :, 0] + anchors.reshape(-1, 1)
    rows: list[dict[str, Any]] = []
    for i in range(pred.shape[0]):
        valid = int(mask[i].sum())
        valid = max(1, min(valid, pred.shape[1]))
        t = true_abs[i, :valid]
        p = pred_abs[i, :valid]
        peak_i = int(np.argmax(np.abs(t)))
        true_peak = float(t[peak_i])
        pred_at_peak = float(p[peak_i])
        true_peak_abs = abs(true_peak)
        pred_peak_abs = float(np.max(np.abs(p)))
        ratio = pred_peak_abs / (true_peak_abs + 1e-6)
        rows.append(
            {
                "sample_key": str(seq["sample_key"][i]),
                "true_peak_abs": true_peak_abs,
                "pred_peak_abs": pred_peak_abs,
                "amp_ratio_pred_over_gt": ratio,
                "under_amp": int(true_peak_abs >= 0.10 and ratio < 0.70),
                "severe_under_amp": int(true_peak_abs >= 0.10 and ratio < 0.45),
                "opposite_at_true_peak": int(
                    true_peak_abs >= 0.10 and abs(pred_at_peak) >= 0.03 and np.sign(pred_at_peak) != np.sign(true_peak)
                ),
                "true_peak_abs_bin": "large_>=0.3" if true_peak_abs >= 0.30 else ("medium_0.1-0.3" if true_peak_abs >= 0.10 else "tiny_<0.1"),
            }
        )
    phys = pd.DataFrame(rows)
    out = sample_df.merge(phys, on="sample_key", how="left")
    out["tail_drift_risk"] = (pd.to_numeric(out.get("tail_pre_ratio_abs_steer"), errors="coerce") > 1.20).astype(int)
    return out


def summarize_group(df: pd.DataFrame, version: str, family: str) -> list[dict[str, Any]]:
    if family not in df.columns:
        return []
    rows: list[dict[str, Any]] = []
    for label, part in df.groupby(family, dropna=False):
        rows.append(
            {
                "version": version,
                "group_family": family,
                "group_label": str(label),
                "sample_count": int(len(part)),
                "rmse": float(part["rmse_2s_abs_steer"].mean()),
                "tail_rmse": float(part["rmse_tail_abs_steer"].mean()),
                "under_amp_rate": float(part["under_amp"].mean()),
                "severe_under_amp_rate": float(part["severe_under_amp"].mean()),
                "opposite_peak_rate": float(part["opposite_at_true_peak"].mean()),
                "tail_drift_risk_rate": float(part["tail_drift_risk"].mean()),
            }
        )
    return rows


def render_table(table: pd.DataFrame, out_path: Path) -> None:
    configure_font()
    cols = ["version", "signal", "method", "test_rmse", "primary_rmse", "tail_rmse", "selection", "delta_vs_raw"]
    view = table[cols].copy()
    rename = {
        "version": "版本",
        "signal": "信号",
        "method": "表示方式",
        "test_rmse": "test RMSE",
        "primary_rmse": "primary",
        "tail_rmse": "tail",
        "selection": "selection",
        "delta_vs_raw": "比原始值",
    }
    view = view.rename(columns=rename)
    for col in ["test RMSE", "primary", "tail", "selection", "比原始值"]:
        view[col] = view[col].map(lambda x: fmt(x))
    fig_h = max(4.8, 0.42 * len(view) + 1.2)
    fig, ax = plt.subplots(figsize=(15, fig_h))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.axis("off")
    ax.text(0.0, 1.05, "无人工权重生理表示筛选（seed2026）", fontsize=18, fontweight="bold", transform=ax.transAxes)
    table_artist = ax.table(
        cellText=view.values,
        colLabels=view.columns,
        cellLoc="center",
        colLoc="center",
        bbox=[0, 0, 1, 0.96],
    )
    table_artist.auto_set_font_size(False)
    table_artist.set_fontsize(9)
    for (row, col), cell in table_artist.get_celld().items():
        cell.set_edgecolor("#D9DEE7")
        if row == 0:
            cell.set_facecolor("#E9EEF5")
            cell.set_text_props(fontweight="bold", color="#17202A")
        else:
            method = str(view.iloc[row - 1]["表示方式"])
            if method == "基线校正":
                cell.set_facecolor("#F4FAF1")
            elif method == "数据自动表示":
                cell.set_facecolor("#F1F6FF")
            elif method == "任务相关状态":
                cell.set_facecolor("#FFF5EA")
            else:
                cell.set_facecolor("white")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    refs = load_reference_rows()
    e18 = load_e18_rows(Path(args.e18_runs))
    if e18.empty:
        raise RuntimeError("没有找到 E18 seed2026 结果。")

    e2 = float(refs.loc[refs["version"].eq("E2"), "test_rmse"].iloc[0]) if not refs[refs["version"].eq("E2")].empty else math.nan
    raw_by_signal = {
        str(row["signal"]): float(row["test_rmse"])
        for _, row in refs[refs["method"].astype(str).eq("原始值")].iterrows()
    }
    old_state_by_signal = {
        str(row["signal"]): float(row["test_rmse"])
        for _, row in refs[refs["method"].astype(str).eq("旧人工状态")].iterrows()
    }

    e18["delta_vs_E2"] = e18["test_rmse"].map(lambda x: float(x) - e2 if math.isfinite(e2) else math.nan)
    e18["delta_vs_raw"] = [
        float(row["test_rmse"]) - raw_by_signal.get(str(row["signal"]), math.nan)
        for _, row in e18.iterrows()
    ]
    e18["delta_vs_old_manual_state"] = [
        float(row["test_rmse"]) - old_state_by_signal.get(str(row["signal"]), math.nan)
        for _, row in e18.iterrows()
    ]
    e18["rank_by_test_rmse"] = e18["test_rmse"].rank(method="min")

    physical_rows: list[dict[str, Any]] = []
    g11_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    morph_rows: list[dict[str, Any]] = []
    detail_frames: list[pd.DataFrame] = []
    figure_rows: list[dict[str, Any]] = []
    g11 = pd.read_csv(G11_CATALOG) if G11_CATALOG.exists() else pd.DataFrame()
    g11_keys = set(g11["sample_key"].astype(str)) if not g11.empty and "sample_key" in g11.columns else set()

    for _, row in e18.iterrows():
        version = str(row["version"])
        sample_path = Path(str(row["sample_metrics_csv"]))
        seq_path = Path(str(row["sequence_npz"]))
        if not sample_path.exists() or not seq_path.exists():
            continue
        sample = pd.read_csv(sample_path)
        detail = add_physical_metrics(sample, seq_path)
        detail["version"] = version
        detail["signal"] = str(row["signal"])
        detail["method"] = str(row["method"])
        detail_frames.append(detail)
        physical_rows.append(
            {
                "version": version,
                "under_amp_rate": float(detail["under_amp"].mean()),
                "severe_under_amp_rate": float(detail["severe_under_amp"].mean()),
                "opposite_peak_rate": float(detail["opposite_at_true_peak"].mean()),
                "tail_drift_risk_rate": float(detail["tail_drift_risk"].mean()),
                "large_rmse": float(detail[detail["true_peak_abs_bin"].eq("large_>=0.3")]["rmse_2s_abs_steer"].mean()),
                "reverse_rmse": float(detail[detail["eval_morphology_label"].astype(str).eq("reverse_correction")]["rmse_2s_abs_steer"].mean()),
                "multi_rmse": float(detail[detail["eval_morphology_label"].astype(str).eq("multi_correction")]["rmse_2s_abs_steer"].mean()),
            }
        )
        g11_part = detail[detail["sample_key"].astype(str).isin(g11_keys)].copy()
        if len(g11_part):
            g11_rows.append(
                {
                    "version": version,
                    "sample_count": int(len(g11_part)),
                    "g11_rmse": float(g11_part["rmse_2s_abs_steer"].mean()),
                    "g11_tail_rmse": float(g11_part["rmse_tail_abs_steer"].mean()),
                    "g11_under_amp_rate": float(g11_part["under_amp"].mean()),
                    "g11_severe_under_amp_rate": float(g11_part["severe_under_amp"].mean()),
                    "g11_opposite_peak_rate": float(g11_part["opposite_at_true_peak"].mean()),
                }
            )
        subject_rows.extend(summarize_group(detail, version, "subj"))
        morph_rows.extend(summarize_group(detail, version, "eval_morphology_label"))
        figure_rows.append(
            {
                "version": version,
                "signal": str(row["signal"]),
                "method": str(row["method"]),
                "prediction_overview": str(row["prediction_overview"]),
                "run_root": str(row["run_root"]),
            }
        )

    physical = pd.DataFrame(physical_rows)
    g11_summary = pd.DataFrame(g11_rows)
    subject = pd.DataFrame(subject_rows)
    morphology = pd.DataFrame(morph_rows)
    detail_all = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    figures = pd.DataFrame(figure_rows)

    combined = e18.merge(physical, on="version", how="left").merge(g11_summary, on="version", how="left")
    refs.to_csv(out_dir / "e18_seed2026_reference_compare.csv", index=False, encoding="utf-8-sig")
    e18.to_csv(out_dir / "e18_seed2026_overall.csv", index=False, encoding="utf-8-sig")
    physical.to_csv(out_dir / "e18_seed2026_physical_summary.csv", index=False, encoding="utf-8-sig")
    g11_summary.to_csv(out_dir / "e18_seed2026_g11_summary.csv", index=False, encoding="utf-8-sig")
    subject.to_csv(out_dir / "e18_seed2026_subject_summary.csv", index=False, encoding="utf-8-sig")
    morphology.to_csv(out_dir / "e18_seed2026_morphology_summary.csv", index=False, encoding="utf-8-sig")
    detail_all.to_csv(out_dir / "e18_seed2026_sample_detail.csv", index=False, encoding="utf-8-sig")
    figures.to_csv(out_dir / "e18_prediction_figure_index.csv", index=False, encoding="utf-8-sig")
    combined.to_csv(out_dir / "e18_seed2026_combined_decision_table.csv", index=False, encoding="utf-8-sig")

    table_png = out_dir / "e18_signal_representation_seed2026_table.png"
    render_table(e18.sort_values(["method", "signal"]), table_png)

    best = e18.sort_values("test_rmse").iloc[0].to_dict()
    by_method = e18.groupby("method", dropna=False)["test_rmse"].mean().reset_index().sort_values("test_rmse")
    by_signal = e18.groupby("signal", dropna=False)["test_rmse"].mean().reset_index().sort_values("test_rmse")
    improve_raw = e18[pd.to_numeric(e18["delta_vs_raw"], errors="coerce") < 0].copy()
    candidates = combined[
        (pd.to_numeric(combined["delta_vs_raw"], errors="coerce") < 0)
        | (pd.to_numeric(combined.get("g11_rmse", pd.Series(dtype=float)), errors="coerce") < 0.55)
        | (pd.to_numeric(combined.get("severe_under_amp_rate", pd.Series(dtype=float)), errors="coerce") < 0.20)
    ].copy()

    report = f"""# E18 无人工权重生理表示筛选报告（seed2026）

## 1. 为什么做这一轮

之前的单信号“语义状态”仍然套用了人工权重公式。即使只保留心率、皮电、肌电或脑电，其他信号被置成平均状态后，状态公式本身仍然会影响结果。因此 E17 不能强力证明某个信号本身无效，只能说明“旧人工权重状态没有稳定胜出”。

E18 改成三类无人工权重表示：基线校正、数据自动表示、任务相关状态。所有版本都保留粗细双头和连续驾驶风格，只改变生理/脑电表示方式。

## 2. 整体结果

{df_to_md(e18[["version", "signal", "method", "test_rmse", "primary_rmse", "tail_rmse", "selection", "delta_vs_E2", "delta_vs_raw", "delta_vs_old_manual_state"]].sort_values("test_rmse"))}

当前 E18 中整体 RMSE 最低的是 `{best.get("version", "NA")}`，信号为 `{best.get("signal", "NA")}`，表示方式为 `{best.get("method", "NA")}`，test RMSE 为 `{fmt(best.get("test_rmse"))}`。

## 3. 按表示方式看

{df_to_md(by_method)}

## 4. 按信号看

{df_to_md(by_signal)}

## 5. 比同信号原始输入更好的版本

{df_to_md(improve_raw[["version", "signal", "method", "test_rmse", "delta_vs_raw", "delta_vs_E2"]].sort_values("delta_vs_raw"))}

## 6. 物理风险摘要

{df_to_md(physical.sort_values("severe_under_amp_rate") if not physical.empty else physical)}

## 7. G11 困难样本摘要

{df_to_md(g11_summary.sort_values("g11_rmse") if not g11_summary.empty else g11_summary)}

## 8. 初步晋级建议

以下版本满足至少一个初筛信号，后续可以结合预测图再决定是否补 seed2027/2028：

{df_to_md(candidates[["version", "signal", "method", "test_rmse", "delta_vs_raw", "g11_rmse", "severe_under_amp_rate", "tail_drift_risk_rate"]].sort_values("test_rmse") if not candidates.empty else candidates)}

注意：本轮只有 seed2026，所有结论都只能作为筛选信号。是否补 2027/2028，必须同时看整体 RMSE、G11 困难样本、严重幅值不足率、错侧率、尾段漂移和预测图。

## 9. 产物

- `e18_seed2026_overall.csv`
- `e18_seed2026_physical_summary.csv`
- `e18_seed2026_g11_summary.csv`
- `e18_seed2026_subject_summary.csv`
- `e18_seed2026_morphology_summary.csv`
- `e18_seed2026_sample_detail.csv`
- `e18_seed2026_combined_decision_table.csv`
- `e18_prediction_figure_index.csv`
- `e18_signal_representation_seed2026_table.png`
"""
    (out_dir / "e18_signal_representation_seed2026_report_cn.md").write_text(report, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
