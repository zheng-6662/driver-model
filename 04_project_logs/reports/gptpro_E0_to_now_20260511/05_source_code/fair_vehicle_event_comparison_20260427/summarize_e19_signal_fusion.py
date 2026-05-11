# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[4]
REMOTE_ROOT = Path("/root/autodl-tmp/data_process")
REPORTS_DIR = PROJECT_ROOT / "04_project_logs" / "reports"
G11_CATALOG = REPORTS_DIR / "style_physio_eeg_g11_bad_case_attribution_20260509" / "bad_case_catalog.csv"

DEFAULT_E19_RUNS = REPORTS_DIR / "style_physio_eeg_e19_signal_fusion_runs_20260511.csv"
DEFAULT_OUT_DIR = REPORTS_DIR / "style_physio_eeg_e19_signal_fusion_seed2026_summary_20260511"


REFERENCE_RUN_FILES = [
    REPORTS_DIR / "style_physio_eeg_e0_e2_fresh_3seed_runs_20260507.csv",
    REPORTS_DIR / "style_physio_eeg_e7_signal_group_runs_20260508.csv",
    REPORTS_DIR / "style_physio_eeg_e10_non_eeg_signal_runs_20260509.csv",
    REPORTS_DIR / "style_physio_eeg_e16_eeg_style_runs_20260511.csv",
    REPORTS_DIR / "style_physio_eeg_e18_signal_representation_runs_20260511.csv",
]


VERSION_INFO: dict[str, dict[str, Any]] = {
    "E2": {"type": "基础对照", "method": "无生理/脑电", "label": "粗细双头 + 连续风格"},
    "E7C": {"type": "旧融合对照", "method": "心率+皮电+肌电原始拼接", "label": "非脑电 raw 简单融合"},
    "E10A": {"type": "单信号对照", "method": "心率原始值", "label": "心率原始单信号"},
    "E10B": {"type": "单信号对照", "method": "皮电原始值", "label": "皮电原始单信号"},
    "E10C": {"type": "强候选对照", "method": "肌电原始值", "label": "肌电原始单信号"},
    "E16B": {"type": "单信号对照", "method": "脑电原始值", "label": "脑电原始单信号"},
    "E18C": {"type": "上一轮较好", "method": "肌电基线校正", "label": "肌电当前值+相对基线变化"},
    "E18H": {"type": "上一轮参考", "method": "脑电数据自动表示", "label": "脑电自动低维表示"},
    "E18K": {"type": "上一轮参考", "method": "肌电任务相关状态", "label": "肌电辅助响应类型"},
    "E19A": {"type": "本轮融合", "method": "四信号基线校正融合", "label": "四信号当前值+相对变化+有效率"},
    "E19B": {"type": "本轮融合", "method": "四信号数据自动表示融合", "label": "四信号自动低维表示"},
    "E19C": {"type": "本轮融合", "method": "四信号任务相关融合", "label": "四信号辅助响应类型"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总 E19 四信号融合 seed2026 结果。")
    parser.add_argument("--e19-runs", default=str(DEFAULT_E19_RUNS))
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


def rows_from_records(path: Path, allowed_versions: set[str]) -> list[dict[str, Any]]:
    df = read_records(path)
    rows: list[dict[str, Any]] = []
    if df.empty or "experiment_id" not in df.columns:
        return rows
    for version in sorted(allowed_versions):
        match = df[df["experiment_id"].astype(str).eq(version)]
        if match.empty:
            continue
        row = match.iloc[-1]
        run_root = localize_path(str(row["run_root"]))
        metrics = metrics_from_run_root(run_root)
        if metrics is None:
            continue
        info = VERSION_INFO[version]
        rows.append(
            {
                "version": version,
                "seed": 2026,
                "type": info["type"],
                "method": info["method"],
                "label": info["label"],
                "run_root": str(run_root),
                **metrics,
            }
        )
    return rows


def load_all_rows(e19_runs: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    reference_versions = set(VERSION_INFO) - {"E19A", "E19B", "E19C"}
    for path in REFERENCE_RUN_FILES:
        rows.extend(rows_from_records(path, reference_versions))
    rows.extend(rows_from_records(e19_runs, {"E19A", "E19B", "E19C"}))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates(subset=["version"], keep="last")


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
    cols = [
        "version",
        "type",
        "method",
        "test_rmse",
        "primary_rmse",
        "tail_rmse",
        "selection",
        "g11_rmse",
        "severe_under_amp_rate",
        "delta_vs_E10C",
    ]
    view = table[cols].copy()
    view = view.rename(
        columns={
            "version": "版本",
            "type": "类别",
            "method": "表示方式",
            "test_rmse": "test RMSE",
            "primary_rmse": "primary",
            "tail_rmse": "tail",
            "selection": "selection",
            "g11_rmse": "G11 RMSE",
            "severe_under_amp_rate": "严重幅值不足率",
            "delta_vs_E10C": "比E10C",
        }
    )
    for col in ["test RMSE", "primary", "tail", "selection", "G11 RMSE", "比E10C"]:
        view[col] = view[col].map(lambda x: fmt(x))
    view["严重幅值不足率"] = view["严重幅值不足率"].map(lambda x: fmt_pct(x))

    fig_h = max(5.2, 0.48 * len(view) + 1.4)
    fig, ax = plt.subplots(figsize=(16.5, fig_h))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.axis("off")
    ax.text(0.0, 1.05, "四类生理/脑电信号融合结果（seed2026）", fontsize=18, fontweight="bold", transform=ax.transAxes)
    table_artist = ax.table(
        cellText=view.values,
        colLabels=view.columns,
        cellLoc="center",
        colLoc="center",
        bbox=[0, 0, 1, 0.96],
    )
    table_artist.auto_set_font_size(False)
    table_artist.set_fontsize(8.8)
    for (row, col), cell in table_artist.get_celld().items():
        cell.set_edgecolor("#D9DEE7")
        if row == 0:
            cell.set_facecolor("#E9EEF5")
            cell.set_text_props(fontweight="bold", color="#17202A")
        else:
            version = str(view.iloc[row - 1]["版本"])
            if version.startswith("E19"):
                cell.set_facecolor("#EAF7EE")
            elif version in {"E10C", "E18C"}:
                cell.set_facecolor("#FFF5D9")
            elif version == "E2":
                cell.set_facecolor("#F4F6F8")
            else:
                cell.set_facecolor("white")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = load_all_rows(Path(args.e19_runs))
    e19 = all_rows[all_rows["version"].astype(str).str.startswith("E19")].copy()
    if e19.empty:
        raise RuntimeError("没有找到 E19 seed2026 正式结果。")

    physical_rows: list[dict[str, Any]] = []
    g11_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    morphology_rows: list[dict[str, Any]] = []
    detail_frames: list[pd.DataFrame] = []
    figure_rows: list[dict[str, Any]] = []
    g11 = pd.read_csv(G11_CATALOG) if G11_CATALOG.exists() else pd.DataFrame()
    g11_keys = set(g11["sample_key"].astype(str)) if not g11.empty and "sample_key" in g11.columns else set()

    for _, row in all_rows.iterrows():
        version = str(row["version"])
        sample_path = Path(str(row["sample_metrics_csv"]))
        seq_path = Path(str(row["sequence_npz"]))
        if not sample_path.exists() or not seq_path.exists():
            continue
        sample = pd.read_csv(sample_path)
        detail = add_physical_metrics(sample, seq_path)
        detail["version"] = version
        detail["method"] = str(row["method"])
        detail["type"] = str(row["type"])
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
        morphology_rows.extend(summarize_group(detail, version, "eval_morphology_label"))
        figure_rows.append(
            {
                "version": version,
                "type": str(row["type"]),
                "method": str(row["method"]),
                "prediction_overview": str(row["prediction_overview"]),
                "run_root": str(row["run_root"]),
            }
        )

    physical = pd.DataFrame(physical_rows)
    g11_summary = pd.DataFrame(g11_rows)
    subject = pd.DataFrame(subject_rows)
    morphology = pd.DataFrame(morphology_rows)
    detail_all = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    figures = pd.DataFrame(figure_rows)

    combined = all_rows.merge(physical, on="version", how="left").merge(g11_summary, on="version", how="left")
    e2_rmse = float(combined.loc[combined["version"].eq("E2"), "test_rmse"].iloc[0]) if not combined[combined["version"].eq("E2")].empty else math.nan
    e10c_rmse = float(combined.loc[combined["version"].eq("E10C"), "test_rmse"].iloc[0]) if not combined[combined["version"].eq("E10C")].empty else math.nan
    e18c_rmse = float(combined.loc[combined["version"].eq("E18C"), "test_rmse"].iloc[0]) if not combined[combined["version"].eq("E18C")].empty else math.nan
    combined["delta_vs_E2"] = combined["test_rmse"].map(lambda x: float(x) - e2_rmse if math.isfinite(e2_rmse) else math.nan)
    combined["delta_vs_E10C"] = combined["test_rmse"].map(lambda x: float(x) - e10c_rmse if math.isfinite(e10c_rmse) else math.nan)
    combined["delta_vs_E18C"] = combined["test_rmse"].map(lambda x: float(x) - e18c_rmse if math.isfinite(e18c_rmse) else math.nan)

    e19_combined = combined[combined["version"].astype(str).str.startswith("E19")].copy()
    all_rows.to_csv(out_dir / "e19_seed2026_loaded_overall.csv", index=False, encoding="utf-8-sig")
    combined.to_csv(out_dir / "e19_seed2026_combined_decision_table.csv", index=False, encoding="utf-8-sig")
    e19_combined.to_csv(out_dir / "e19_seed2026_overall.csv", index=False, encoding="utf-8-sig")
    physical.to_csv(out_dir / "e19_seed2026_physical_summary.csv", index=False, encoding="utf-8-sig")
    g11_summary.to_csv(out_dir / "e19_seed2026_g11_summary.csv", index=False, encoding="utf-8-sig")
    subject.to_csv(out_dir / "e19_seed2026_subject_summary.csv", index=False, encoding="utf-8-sig")
    morphology.to_csv(out_dir / "e19_seed2026_morphology_summary.csv", index=False, encoding="utf-8-sig")
    detail_all.to_csv(out_dir / "e19_seed2026_sample_detail.csv", index=False, encoding="utf-8-sig")
    figures.to_csv(out_dir / "e19_prediction_figure_index.csv", index=False, encoding="utf-8-sig")

    display_versions = ["E2", "E7C", "E10A", "E10B", "E10C", "E16B", "E18C", "E18H", "E18K", "E19A", "E19B", "E19C"]
    display = combined[combined["version"].isin(display_versions)].copy()
    display["sort_key"] = display["version"].map({v: i for i, v in enumerate(display_versions)})
    display = display.sort_values("sort_key")
    table_png = out_dir / "e19_signal_fusion_seed2026_table.png"
    render_table(display, table_png)

    best_e19 = e19_combined.sort_values("test_rmse").iloc[0].to_dict()
    best_all = combined.sort_values("test_rmse").iloc[0].to_dict()
    e19_rank = e19_combined[["version", "type", "method", "test_rmse", "primary_rmse", "tail_rmse", "selection", "delta_vs_E2", "delta_vs_E10C", "delta_vs_E18C", "g11_rmse", "severe_under_amp_rate", "opposite_peak_rate", "tail_drift_risk_rate"]].sort_values("test_rmse")
    subject_e19 = subject[subject["version"].astype(str).str.startswith("E19")].copy()
    morphology_e19 = morphology[morphology["version"].astype(str).str.startswith("E19")].copy()

    decision_lines: list[str] = []
    if math.isfinite(e10c_rmse) and float(best_e19["test_rmse"]) < e10c_rmse:
        decision_lines.append("至少有一个四信号融合版本整体 RMSE 低于 E10C，可以进一步看预测图和困难样本后决定是否补种子。")
    else:
        decision_lines.append("四信号融合 seed2026 暂时没有在整体 RMSE 上超过 E10C，不能直接替代肌电原始单信号主线。")
    if not e19_combined.empty and "g11_rmse" in e19_combined.columns:
        best_g11 = e19_combined.sort_values("g11_rmse").iloc[0]
        decision_lines.append(f"G11 困难样本中，四信号融合当前最好的是 {best_g11['version']}，G11 RMSE={fmt(best_g11['g11_rmse'])}。")
    decision = "\n".join(f"- {line}" for line in decision_lines)

    report = f"""# E19 四类生理/脑电信号融合实验报告（seed2026）

## 1. 这轮为什么做

上一轮 E18 是把心率、皮电、肌电、脑电分开看，目的是判断每类信号换成“基线校正、数据自动表示、任务相关状态”以后是否有价值。本轮 E19 进一步回答另一个问题：如果四类信号不再用旧的人工权重公式，而是用更合理的方式融合，是否能让它们互补，而不是互相加噪声。

本轮仍然固定为：粗细双头、连续驾驶风格、同一套样本划分、同一训练参数、seed2026。只改变四类信号融合方式。

## 2. 三个融合版本是什么

- E19A：四信号基线校正融合。输入四类信号的当前值、相对基线变化，以及心率/皮电/肌电/脑电各自的有效率。
- E19B：四信号数据自动表示融合。先把四类信号的当前值、变化值和有效率放在一起，再只用训练集提取低维表示，减少人工权重影响。
- E19C：四信号任务相关融合。让四类信号参与响应类型判断，再让预测到的响应类型辅助轨迹预测。

## 3. E19 结果

{df_to_md(e19_rank)}

E19 内部整体 RMSE 最低的是 `{best_e19.get("version", "NA")}`，test RMSE=`{fmt(best_e19.get("test_rmse"))}`。本表里所有已加载版本中，整体 RMSE 最低的是 `{best_all.get("version", "NA")}`，test RMSE=`{fmt(best_all.get("test_rmse"))}`。

## 4. 和关键对照放在一起

{df_to_md(display[["version", "type", "method", "test_rmse", "primary_rmse", "tail_rmse", "selection", "g11_rmse", "severe_under_amp_rate", "delta_vs_E2", "delta_vs_E10C", "delta_vs_E18C"]])}

## 5. 物理风险指标

{df_to_md(physical[physical["version"].astype(str).str.startswith("E19")].sort_values("severe_under_amp_rate"))}

## 6. G11 困难样本

{df_to_md(g11_summary[g11_summary["version"].astype(str).str.startswith("E19")].sort_values("g11_rmse") if not g11_summary.empty else g11_summary)}

## 7. 分被试结果

{df_to_md(subject_e19.sort_values(["version", "group_label"]) if not subject_e19.empty else subject_e19)}

## 8. 分响应类型结果

{df_to_md(morphology_e19.sort_values(["version", "group_label"]) if not morphology_e19.empty else morphology_e19)}

## 9. 当前判断

{decision}

这轮只有 seed2026，结论只能作为筛选信号。是否补 seed2027/2028，需要同时看整体 RMSE、G11 困难样本、严重幅值不足率、错侧率、尾段漂移风险和预测图。

## 10. 产物

- `e19_seed2026_overall.csv`
- `e19_seed2026_combined_decision_table.csv`
- `e19_seed2026_physical_summary.csv`
- `e19_seed2026_g11_summary.csv`
- `e19_seed2026_subject_summary.csv`
- `e19_seed2026_morphology_summary.csv`
- `e19_seed2026_sample_detail.csv`
- `e19_prediction_figure_index.csv`
- `e19_signal_fusion_seed2026_table.png`
"""
    (out_dir / "e19_signal_fusion_seed2026_report_cn.md").write_text(report, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
