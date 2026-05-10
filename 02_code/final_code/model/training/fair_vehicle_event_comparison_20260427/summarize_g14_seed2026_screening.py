# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[4]
REPORTS_DIR = PROJECT_ROOT / "04_project_logs" / "reports"
G14_DIR = REPORTS_DIR / "g14_non_average_prediction_20260510"
RUN_LOG_DIRS = [
    G14_DIR / "g14_seed2026_parallel",
    G14_DIR / "g14_seed2026_stable_candidate_parallel",
]
OUT_DIR = G14_DIR / "g14_seed2026_screening_summary"
G11_CATALOG = REPORTS_DIR / "style_physio_eeg_g11_bad_case_attribution_20260509" / "bad_case_catalog.csv"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any, digits: int = 4) -> str:
    try:
        f = float(value)
    except Exception:
        return "NA"
    if not math.isfinite(f):
        return "NA"
    return f"{f:.{digits}f}"


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


def load_g14_run_log() -> pd.DataFrame:
    frames = []
    for run_log_dir in RUN_LOG_DIRS:
        if not run_log_dir.exists():
            continue
        for path in sorted(run_log_dir.glob("*_run_log.csv")):
            df = pd.read_csv(path)
            df["run_log_csv"] = str(path)
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["run_root_local"] = out["run_root"].astype(str).str.replace(
        "/root/autodl-tmp/data_process",
        str(PROJECT_ROOT).replace("\\", "/"),
        regex=False,
    )
    return out


def load_baselines() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    e6_path = REPORTS_DIR / "style_physio_eeg_e6_physical_repair_summary_20260508" / "e6_seed_wise_metrics.csv"
    if e6_path.exists():
        df = pd.read_csv(e6_path)
        keep = df[(df["seed"].eq(2026)) & (df["experiment_id"].astype(str).isin(["E2", "E5A", "E6"]))].copy()
        for _, row in keep.iterrows():
            rows.append(
                {
                    "version": str(row["experiment_id"]),
                    "meaning": {"E2": "连续风格基础模型", "E5A": "脑电教师蒸馏", "E6": "脑电教师蒸馏+物理约束"}[str(row["experiment_id"])],
                    "test_rmse": float(row["test_rmse"]),
                    "tail_rmse": float(row["tail"]),
                    "selection": float(row["selection"]),
                }
            )
    e10_path = REPORTS_DIR / "style_physio_eeg_e10c_emg_only_3seed_summary_20260509" / "seed_wise_metrics.csv"
    if e10_path.exists():
        df = pd.read_csv(e10_path)
        keep = df[(df["seed"].eq(2026)) & (df["experiment_id"].astype(str).eq("E10C"))].copy()
        for _, row in keep.iterrows():
            rows.append(
                {
                    "version": "E10C",
                    "meaning": "肌电单信号+连续风格",
                    "test_rmse": float(row["test_steer_rmse"]),
                    "tail_rmse": float(row["tail_rmse"]),
                    "selection": float(row["selection"]),
                }
            )
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
                "opposite_at_true_peak": int(true_peak_abs >= 0.10 and abs(pred_at_peak) >= 0.03 and np.sign(pred_at_peak) != np.sign(true_peak)),
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
                "severe_under_amp_rate": float(part["severe_under_amp"].mean()),
                "opposite_peak_rate": float(part["opposite_at_true_peak"].mean()),
                "tail_drift_risk_rate": float(part["tail_drift_risk"].mean()),
            }
        )
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    g14 = load_g14_run_log()
    baselines = load_baselines()
    overall_rows: list[dict[str, Any]] = []
    physical_rows: list[dict[str, Any]] = []
    g11_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    morph_rows: list[dict[str, Any]] = []
    detail_frames: list[pd.DataFrame] = []
    g11 = pd.read_csv(G11_CATALOG) if G11_CATALOG.exists() else pd.DataFrame()
    g11_keys = set(g11["sample_key"].astype(str)) if not g11.empty else set()

    for _, row in g14.iterrows():
        version = str(row["experiment_id"])
        run_root = Path(str(row["run_root_local"]))
        sample_path = run_root / "prediction_figures" / "test" / "prediction_sample_metrics.csv"
        seq_path = run_root / "prediction_figures" / "test" / "prediction_sequences.npz"
        metrics_path = run_root / "metrics.json"
        if not sample_path.exists() or not seq_path.exists():
            continue
        sample = pd.read_csv(sample_path)
        detail = add_physical_metrics(sample, seq_path)
        detail["version"] = version
        detail["label"] = str(row["label"])
        detail_frames.append(detail)
        metrics = read_json(metrics_path) if metrics_path.exists() else {}
        selection = (metrics.get("test") or {}).get("selection_summary") or {}
        overall_rows.append(
            {
                "version": version,
                "label": str(row["label"]),
                "test_rmse": float(row["test_steer_rmse"]),
                "tail_rmse": float(row["test_tail_rmse"]),
                "selection": float(row["test_selection"]),
                "primary_rmse": float(selection.get("overall_primary_steer_rmse", np.nan)),
                "best_epoch": int(row["best_epoch"]),
                "run_root_local": str(run_root),
                "prediction_overview": str(run_root / "prediction_figures" / "test" / "overview.png"),
            }
        )
        physical_rows.append(
            {
                "version": version,
                "severe_under_amp_rate": float(detail["severe_under_amp"].mean()),
                "under_amp_rate": float(detail["under_amp"].mean()),
                "opposite_peak_rate": float(detail["opposite_at_true_peak"].mean()),
                "tail_drift_risk_rate": float(detail["tail_drift_risk"].mean()),
                "large_rmse": float(detail[detail["true_peak_abs_bin"].eq("large_>=0.3")]["rmse_2s_abs_steer"].mean()),
                "reverse_rmse": float(detail[detail["eval_morphology_label"].astype(str).eq("reverse_correction")]["rmse_2s_abs_steer"].mean()),
                "multi_rmse": float(detail[detail["eval_morphology_label"].astype(str).eq("multi_correction")]["rmse_2s_abs_steer"].mean()),
            }
        )
        g11_part = detail[detail["sample_key"].astype(str).isin(g11_keys)].copy()
        g11_rows.append(
            {
                "version": version,
                "sample_count": int(len(g11_part)),
                "g11_rmse": float(g11_part["rmse_2s_abs_steer"].mean()),
                "g11_tail_rmse": float(g11_part["rmse_tail_abs_steer"].mean()),
                "g11_severe_under_amp_rate": float(g11_part["severe_under_amp"].mean()),
                "g11_opposite_peak_rate": float(g11_part["opposite_at_true_peak"].mean()),
            }
        )
        subject_rows.extend(summarize_group(detail, version, "subj"))
        morph_rows.extend(summarize_group(detail, version, "eval_morphology_label"))

    overall = pd.DataFrame(overall_rows)
    physical = pd.DataFrame(physical_rows)
    g11_summary = pd.DataFrame(g11_rows)
    subject = pd.DataFrame(subject_rows)
    morph = pd.DataFrame(morph_rows)
    detail_all = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()

    if not g11.empty and not g11_summary.empty:
        g11_existing = pd.DataFrame(
            [
                {"version": "E10C", "g11_rmse": float(g11["E10C_rmse_2s"].mean()), "g11_tail_rmse": float(g11["E10C_tail_rmse"].mean())},
                {"version": "E6", "g11_rmse": float(g11["E6_rmse_2s"].mean()), "g11_tail_rmse": float(g11["E6_tail_rmse"].mean())},
                {"version": "E5A", "g11_rmse": float(g11["E5A_rmse_2s"].mean()), "g11_tail_rmse": float(g11["E5A_tail_rmse"].mean())},
                {"version": "E2", "g11_rmse": float(g11["E2_rmse_2s"].mean()), "g11_tail_rmse": float(g11["E2_tail_rmse"].mean())},
            ]
        )
    else:
        g11_existing = pd.DataFrame()

    baselines.to_csv(OUT_DIR / "g14_seed2026_baseline_compare.csv", index=False, encoding="utf-8-sig")
    overall.to_csv(OUT_DIR / "g14_seed2026_overall.csv", index=False, encoding="utf-8-sig")
    physical.to_csv(OUT_DIR / "g14_seed2026_physical_summary.csv", index=False, encoding="utf-8-sig")
    g11_summary.to_csv(OUT_DIR / "g14_seed2026_g11_summary.csv", index=False, encoding="utf-8-sig")
    g11_existing.to_csv(OUT_DIR / "g14_seed2026_g11_existing_baselines.csv", index=False, encoding="utf-8-sig")
    subject.to_csv(OUT_DIR / "g14_seed2026_subject_summary.csv", index=False, encoding="utf-8-sig")
    morph.to_csv(OUT_DIR / "g14_seed2026_morphology_summary.csv", index=False, encoding="utf-8-sig")
    detail_all.to_csv(OUT_DIR / "g14_seed2026_sample_detail.csv", index=False, encoding="utf-8-sig")

    table = overall[["version", "label", "test_rmse", "tail_rmse", "selection", "best_epoch"]].sort_values("test_rmse")
    baseline_table = baselines[["version", "meaning", "test_rmse", "tail_rmse", "selection"]].sort_values("test_rmse")
    g11_table = pd.concat(
        [
            g11_existing.assign(label="已有强基准"),
            g11_summary[["version", "g11_rmse", "g11_tail_rmse"]].assign(label="G14候选"),
        ],
        ignore_index=True,
    ).sort_values("g11_rmse") if not g11_summary.empty else pd.DataFrame()
    physical_table = physical.sort_values("severe_under_amp_rate")
    best = table.iloc[0].to_dict() if not table.empty else {}

    report = f"""# G14 seed2026 筛选报告：多候选轨迹与响应先判别

## 1. 本轮为什么做

G14 第一阶段的相似历史事件上限诊断说明：训练集中存在足够相似的真实响应，但模型在推理前判断不出当前样本应该属于哪种方向、幅值和形态。因此本轮不再继续单条平均轨迹小改，而是直接训练“响应先判别 + 多候选轨迹”的版本。

## 2. 本轮版本

- G14A：连续风格 + 响应先判别 + 四候选轨迹，不加肌电；
- G14B：连续风格 + 肌电 + 响应先判别 + 四候选轨迹；
- G14C：G14B 基础上加入幅值和方向物理约束；
- G14D：连续风格 + 肌电 + 响应先判别 + 八候选轨迹；
- G14E：肌电 + 固定响应类型监督候选选择 + 四候选轨迹；
- G14F：肌电 + 训练集响应原型 + 原型残差修正；
- G14G：脑电教师 + 肌电 + 响应原型候选；
- G14H：G14F 基础上加入幅值和方向物理约束。

所有版本均在服务器完成 seed2026 完整 40 轮训练。
其中 G14G 使用重新训练的兼容脑电教师，因为旧恢复版脑电教师 checkpoint 的输入维度与当前 G14 代码不一致，不能直接作为本轮教师。

## 3. 与已有强基准对比

{df_to_md(baseline_table)}

## 4. G14 seed2026 总体结果

{df_to_md(table)}

当前 G14 中整体最好的版本是 `{best.get("version", "NA")}`，测试 RMSE 为 `{fmt(best.get("test_rmse"))}`。

## 5. G11 困难样本结果

{df_to_md(g11_table)}

## 6. 物理风险摘要

{df_to_md(physical_table)}

## 7. 结论

本轮没有形成新的强主线。

- G14C 是本轮 G14 中最接近可用的版本，测试 RMSE `0.4603`，略好于 E2 seed2026 的 `0.4644`，但仍弱于 E10C `0.4421`、E5A `0.4404` 和 E6 `0.4395`。
- 加肌电的 G14B 明显优于不加肌电的 G14A，说明肌电仍然有帮助；但多候选结构本身没有把肌电价值充分放大。
- 八候选 G14D 没有比四候选更好，说明“候选数量变多”本身不是突破点。
- G14C 加幅值方向约束后整体略好于 G14B，但尾段和综合选择指标仍然不够好。
- G14E 的严重幅值不足率最低，但尾段漂移风险最高，说明固定响应类型监督会放大幅值，却没有解决尾段物理合理性。
- G14F/G14H 使用训练集响应原型后，比 G14E 稳定，但仍没有超过 G14C；G14H 加物理约束后也没有带来额外收益。
- G14G 加入脑电教师和肌电学生后，整体 RMSE 仍没有超过 E5A/E6/E10C，也没有超过 G14C；但它的尾段误差 `0.3477` 是本轮 G14 中最好的，甚至好于当前强基准的尾段误差，综合选择指标 `0.8158` 也明显好于 E10C。这说明脑电教师 + 肌电 + 响应原型对尾段有诊断价值，但还不足以成为新的整体主线。

更重要的是：这轮说明“多候选轨迹”和“响应原型”有局部价值，但当前选择头还没有学会稳定选中好候选。下一步如果继续 G14，不应该简单增加候选数，也不应该直接补 G14A-H 的 2027/2028；应先解释为什么 G14G 能改善尾段却不能改善整体和困难样本，再决定是否做更有针对性的尾段/候选选择模型：

1. 让候选选择头先接受更直接的响应方向、幅值等级和形态监督；
2. 训练时不要只用“当前候选谁误差最低”当选择标签，而要构造更稳定的响应类型标签；
3. 考虑把第一阶段的相似事件参考作为候选来源或辅助目标，而不是只靠模型内部自由生成候选；
4. 若要继续训练，应优先做“尾段为什么变好”和“困难样本为什么没有同步变好”的归因，而不是继续堆候选版本。

## 8. 产物

- `g14_seed2026_overall.csv`
- `g14_seed2026_physical_summary.csv`
- `g14_seed2026_g11_summary.csv`
- `g14_seed2026_subject_summary.csv`
- `g14_seed2026_morphology_summary.csv`
- `g14_seed2026_sample_detail.csv`
"""
    (OUT_DIR / "g14_seed2026_screening_report_cn.md").write_text(report, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
