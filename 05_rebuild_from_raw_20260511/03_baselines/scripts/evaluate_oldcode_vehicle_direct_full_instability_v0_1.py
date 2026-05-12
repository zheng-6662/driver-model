# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


ROOT = Path(r"F:/data_set_process/data_process")
REBUILD_ROOT = ROOT / "05_rebuild_from_raw_20260511"
OLD_TRAIN_DIR = ROOT / "02_code" / "final_code" / "model" / "training"
BASELINE_SCRIPT_DIR = REBUILD_ROOT / "03_baselines" / "scripts"

for path in [OLD_TRAIN_DIR, BASELINE_SCRIPT_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_stage3_vehicle_baselines as old_eval  # noqa: E402
from event_conditioned_baseline_model import (  # noqa: E402
    EventConditionedDataset,
    EventConditionedTrajectoryModel,
    build_event_schema_targets,
)
from event_conditioned_eval_support import build_primary_selection_bundle  # noqa: E402
from run_event_conditioned_trajectory_baseline import (  # noqa: E402
    FUTURE_LEN,
    annotate_event_meta,
    build_sample_bundle_from_manifest,
)


DEFAULT_MANIFEST = (
    REBUILD_ROOT
    / "03_processed_datasets"
    / "vehicle_instability_allraw_highconf_v0_1"
    / "tables"
    / "oldcode_manifest_session_level_split_clean_vehicle_v0_1.csv"
)
DEFAULT_RUN_ROOT = (
    ROOT
    / "tmp"
    / "event_conditioned_runs"
    / "OLD_FULL_INSTABILITY_HIGHCONF_VEHICLE_DIRECT_CLEAN_V0_1_20260512_181413"
)
DEFAULT_OUT_DIR = REBUILD_ROOT / "03_baselines" / "oldcode_vehicle_direct_full_clean_on_instability_v0_1"
DEFAULT_FIXED_SET = (
    REBUILD_ROOT
    / "03_baselines"
    / "oldcode_vehicle_baselines_on_instability_v0_1"
    / "tables"
    / "oldcode_instability_fixed_plot_sample_set.csv"
)
REPORT_DIR = REBUILD_ROOT / "09_reports"


def ensure_dirs(out_dir: Path) -> dict[str, Path]:
    paths = {
        "tables": out_dir / "tables",
        "figures": out_dir / "figures",
        "logs": out_dir / "logs",
        "reports": REPORT_DIR,
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def as_np_stats(norm_stats: dict[str, Any]) -> dict[str, np.ndarray]:
    return {key: np.asarray(value, dtype=np.float32) for key, value in norm_stats.items()}


def normalize_x_with_checkpoint(X_pool: np.ndarray, norm_stats: dict[str, np.ndarray]) -> np.ndarray:
    return (
        (X_pool - norm_stats["feat_mean"].reshape(1, 1, -1))
        / norm_stats["feat_std"].reshape(1, 1, -1)
    ).astype(np.float32)


def meta_for_old_eval(meta_df: pd.DataFrame) -> pd.DataFrame:
    out = meta_df.reset_index(drop=True).copy()
    if "subject" not in out.columns and "subj" in out.columns:
        out["subject"] = out["subj"].astype(str)
    if "event_uid" not in out.columns and "instability_event_uid" in out.columns:
        out["event_uid"] = out["instability_event_uid"].astype(str)
    if "session_stamp" not in out.columns:
        sample_parts = out.get("sample_key", pd.Series([""] * len(out))).astype(str).str.split("::", expand=True)
        if sample_parts.shape[1] >= 2:
            out["session_stamp"] = sample_parts[1].astype(str)
        else:
            out["session_stamp"] = out.get("recording_id", pd.Series(["unknown"] * len(out))).astype(str)
    if "window_config_id" not in out.columns:
        out["window_config_id"] = "pre2_label2_old_main"
    if "sample_id" not in out.columns:
        out["sample_id"] = out["event_uid"].astype(str) + "__pre2_label2_old_main"
    return out


def load_bundle(manifest: Path, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    X_pool, y_pool, curve_pool, ctx_pool, mask_pool, meta_df, dropped = build_sample_bundle_from_manifest(
        manifest_path=manifest,
        max_train_samples=None,
        max_val_samples=None,
        max_test_samples=None,
        seed=seed,
    )
    if dropped:
        raise RuntimeError(f"Unexpected dropped samples when evaluating full run: {dropped}")
    meta_df = annotate_event_meta(meta_df, y_pool, mask_pool)
    meta_df = meta_for_old_eval(meta_df)
    return X_pool, y_pool, curve_pool, ctx_pool, mask_pool, meta_df


def build_model_from_checkpoint(
    checkpoint: dict[str, Any],
    input_dim: int,
    context_dim: int,
    device: str,
) -> EventConditionedTrajectoryModel:
    args = checkpoint["args"]
    model = EventConditionedTrajectoryModel(
        input_dim=int(input_dim),
        context_dim=int(context_dim),
        future_len=FUTURE_LEN,
        event_bin_size=int(args.get("event_bin_size", 20)),
        d_model=int(args.get("d_model", 128)),
        nhead=int(args.get("nhead", 2)),
        enc_layers=int(args.get("enc_layers", 2)),
        dec_layers=int(args.get("dec_layers", 2)),
        ffn_dim=int(args.get("ffn_dim", 256)),
        dropout=float(args.get("dropout", 0.1)),
        event_embed_dim=int(args.get("event_embed_dim", 96)),
        out_dim=2,
        conditioning_mode=str(args.get("conditioning_mode", "vehicle_direct")),
        structure_width=float(args.get("structure_width", 0.065)),
        gate_temperature=float(args.get("gate_temperature", 0.040)),
        event_residual_scale=float(args.get("event_residual_scale", 1.0)),
        enable_response_type_head=bool(args.get("enable_response_type_head", False)),
        enable_response_type_condition=bool(args.get("enable_response_type_condition", False)),
        response_type_use_context=bool(args.get("response_type_use_context", False)),
        response_type_hidden_dim=int(args.get("response_type_hidden_dim", 96)),
        num_trajectory_candidates=int(args.get("num_trajectory_candidates", 1)),
        candidate_delta_scale=float(args.get("candidate_delta_scale", 1.0)),
        candidate_base_mode=str(args.get("candidate_base_mode", "learned_delta")),
        candidate_prototypes=None,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model


def predict_checkpoint(
    ckpt_path: Path,
    checkpoint_tag: str,
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    curve_pool: np.ndarray,
    ctx_pool: np.ndarray,
    mask_pool: np.ndarray,
    meta_df: pd.DataFrame,
    device: str,
) -> dict[str, Any]:
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    norm_stats = as_np_stats(checkpoint["norm_stats"])
    X_norm = normalize_x_with_checkpoint(X_pool, norm_stats)
    event_targets = build_event_schema_targets(
        y_pool=y_pool,
        mask_pool=mask_pool,
        future_len=FUTURE_LEN,
        event_bin_size=int(checkpoint["args"].get("event_bin_size", 20)),
    )
    dataset = EventConditionedDataset(
        X_norm=X_norm,
        y_pool=y_pool,
        curve_pool=curve_pool,
        ctx_pool=ctx_pool,
        mask_pool=mask_pool,
        norm_stats=norm_stats,
        event_targets=event_targets,
        meta_df=meta_df,
    )
    loader = DataLoader(dataset, batch_size=int(checkpoint["args"].get("batch_size", 64)), shuffle=False, num_workers=0)
    model = build_model_from_checkpoint(
        checkpoint=checkpoint,
        input_dim=X_pool.shape[-1],
        context_dim=ctx_pool.shape[-1],
        device=device,
    )
    y_mean = torch.tensor(norm_stats["y_mean"], dtype=torch.float32, device=device)
    y_std = torch.tensor(norm_stats["y_std"], dtype=torch.float32, device=device)
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            src = batch["src"].to(device)
            ctx = batch["ctx"].to(device)
            curve_norm = batch["curve_norm"].to(device)
            y_hat, _ = model(src=src, ctx=ctx, curve_norm=curve_norm, event_teacher=None, privileged_event_teacher=None)
            y_hat_den = y_hat * y_std.view(1, 1, -1) + y_mean.view(1, 1, -1)
            preds.append(y_hat_den.detach().cpu().numpy())
    pred = np.concatenate(preds, axis=0).astype(np.float32)

    metrics = []
    per_sample = []
    label_time = (np.arange(y_pool.shape[1], dtype=np.float32) + 1.0) / 200.0
    for split_name in ["train", "val", "test"]:
        idx = np.where(meta_df["split"].astype(str).to_numpy() == split_name)[0]
        if idx.size == 0:
            continue
        split_meta = meta_df.iloc[idx].reset_index(drop=True)
        rows = old_eval.sample_metric_rows(
            y_pool[idx, :, 0],
            pred[idx, :, 0],
            mask_pool[idx].astype(bool),
            label_time,
            split_meta,
            checkpoint_tag,
            "session_level_split",
            split_name,
            "pre2_label2_old_main",
            large_thr=float(np.nanpercentile(np.max(np.abs(y_pool[idx, :, 0]), axis=1), 75)),
            difficult_thr=float(np.nanpercentile(np.max(np.abs(y_pool[idx, :, 0]), axis=1), 80)),
        )
        per_sample.extend(rows)
        if rows:
            aggregate = old_eval.aggregate_metrics(pd.DataFrame(rows))
            summary = aggregate.iloc[0].to_dict() if len(aggregate) else {}
        else:
            summary = {}
        selection = build_primary_selection_bundle(
            pred=pred[idx],
            true=y_pool[idx],
            mask=mask_pool[idx],
            ctx_raw=ctx_pool[idx],
            meta_df=split_meta,
            split_name=split_name,
            seed=int(checkpoint["args"].get("seed", 2026)),
        )["selection_summary"]
        metrics.append(
            {
                "checkpoint_tag": checkpoint_tag,
                "checkpoint_path": str(ckpt_path),
                "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
                "split": split_name,
                "n_samples": int(idx.size),
                **{f"sample_{k}": v for k, v in summary.items()},
                **{f"selection_{k}": v for k, v in selection.items()},
            }
        )
    per_sample_df = pd.DataFrame(per_sample)
    if len(per_sample_df):
        per_sample_df["checkpoint_tag"] = checkpoint_tag
    return {
        "checkpoint_tag": checkpoint_tag,
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "pred": pred,
        "metrics": pd.DataFrame(metrics),
        "per_sample": per_sample_df,
    }


def summarize_groups(per_sample: pd.DataFrame, meta_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    enriched = per_sample.merge(
        meta_df[
            [
                "event_uid",
                "eval_morphology_label",
                "event_type",
                "event_level",
                "road_type_anchor",
                "mechanism_tag",
                "phase_type",
            ]
        ].drop_duplicates("event_uid"),
        on="event_uid",
        how="left",
    )
    agg_cols = {
        "sample_rmse": "mean",
        "wrong_side": "mean",
        "severe_amp_under": "mean",
        "large_response_recalled": "mean",
        "peak_time_abs_error_s": "mean",
        "tail_abs_error": "mean",
        "reversal_count_exact": "mean",
        "is_large_response": "sum",
        "event_uid": "count",
    }
    by_subject = (
        enriched[enriched["split"] == "test"]
        .groupby(["checkpoint_tag", "subject"], dropna=False)
        .agg(agg_cols)
        .rename(columns={"event_uid": "n_samples", "is_large_response": "n_large_response"})
        .reset_index()
        .sort_values(["checkpoint_tag", "sample_rmse", "subject"])
    )
    by_response = (
        enriched[enriched["split"] == "test"]
        .groupby(["checkpoint_tag", "eval_morphology_label"], dropna=False)
        .agg(agg_cols)
        .rename(columns={"event_uid": "n_samples", "is_large_response": "n_large_response"})
        .reset_index()
        .sort_values(["checkpoint_tag", "sample_rmse", "eval_morphology_label"])
    )
    return by_subject, by_response


def plot_grid(
    plot_df: pd.DataFrame,
    pred_map: dict[str, np.ndarray],
    y_pool: np.ndarray,
    mask_pool: np.ndarray,
    meta_df: pd.DataFrame,
    output_path: Path,
    title: str,
    max_rows: int = 12,
) -> None:
    rows = plot_df.head(max_rows).copy()
    fig, axes = plt.subplots(3, 4, figsize=(16, 9), sharex=True)
    axes_flat = axes.ravel()
    x = np.arange(y_pool.shape[1], dtype=np.float32) / 200.0
    meta_by_uid = meta_df.reset_index().set_index("event_uid", drop=False)
    for ax, (_, row) in zip(axes_flat, rows.iterrows()):
        uid = str(row["event_uid"])
        if uid not in meta_by_uid.index:
            ax.axis("off")
            continue
        sample_idx = int(meta_by_uid.loc[uid, "index"])
        valid_len = int(mask_pool[sample_idx].sum())
        valid = slice(0, valid_len)
        ax.plot(x[valid], y_pool[sample_idx, valid, 0], color="#111827", lw=1.8, label="true")
        for tag, pred in pred_map.items():
            color = "#2563eb" if "active" in tag else "#dc2626"
            style = "-" if "active" in tag else "--"
            ax.plot(x[valid], pred[sample_idx, valid, 0], color=color, lw=1.3, ls=style, label=tag)
        subj = meta_df.loc[sample_idx, "subject"]
        morph = meta_df.loc[sample_idx, "eval_morphology_label"]
        rmse = row.get("sample_rmse", np.nan)
        ax.set_title(f"{subj} | {morph} | RMSE={rmse:.3f}", fontsize=9)
        ax.axhline(0.0, color="#9ca3af", lw=0.7)
        ax.grid(alpha=0.25, lw=0.5)
    for ax in axes_flat[len(rows) :]:
        ax.axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=3, frameon=False)
    fig.suptitle(title, y=0.992, fontsize=13)
    fig.supxlabel("Time after instability anchor (s)")
    fig.supylabel("Steering delta")
    fig.tight_layout(rect=(0.02, 0.03, 0.98, 0.93))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_report(
    out_dir: Path,
    run_root: Path,
    manifest: Path,
    metrics: pd.DataFrame,
    by_subject: pd.DataFrame,
    by_response: pd.DataFrame,
    fixed_plot: Path,
    bad_plot: Path,
    summary: dict[str, Any],
) -> None:
    test_cols = [
        "checkpoint_tag",
        "checkpoint_epoch",
        "split",
        "n_samples",
        "sample_rmse_steer",
        "sample_peak_direction_accuracy",
        "sample_wrong_side_rate",
        "sample_large_response_recall",
        "sample_severe_amp_under_rate",
        "sample_peak_time_mae_s",
        "sample_tail_abs_error_mean",
        "sample_reversal_count_exact_match_rate",
        "selection_selection_score",
        "selection_overall_primary_steer_rmse",
        "selection_rmse_tail_abs_steer",
        "selection_peak_time_abs_err_s",
    ]
    existing_cols = [c for c in test_cols if c in metrics.columns]
    test_table = metrics[metrics["split"] == "test"][existing_cols].copy()
    active_row = test_table[test_table["checkpoint_tag"] == "active_legacy_best"].head(1)
    structure_row = test_table[test_table["checkpoint_tag"] == "structure_best"].head(1)
    active_rmse = float(active_row["sample_rmse_steer"].iloc[0]) if len(active_row) else float("nan")
    active_wrong = float(active_row["sample_wrong_side_rate"].iloc[0]) if len(active_row) else float("nan")
    active_under = float(active_row["sample_severe_amp_under_rate"].iloc[0]) if len(active_row) else float("nan")
    structure_rmse = float(structure_row["sample_rmse_steer"].iloc[0]) if len(structure_row) else float("nan")
    active_epoch = int(active_row["checkpoint_epoch"].iloc[0]) if len(active_row) else -1
    structure_epoch = int(structure_row["checkpoint_epoch"].iloc[0]) if len(structure_row) else -1

    report = f"""# 旧 `vehicle_direct` 全量车辆-only 对照：全原始失稳高置信样本 clean v0.1

生成时间：2026-05-12

## 这次跑了什么

按用户要求，使用旧流程深度模型入口 `run_event_conditioned_trajectory_baseline.py`，在全原始车辆 CSV 重筛得到的高置信车辆失稳样本上跑全量 `vehicle_direct` 车辆-only 对照。

- 训练入口：`{(OLD_TRAIN_DIR / 'run_event_conditioned_trajectory_baseline.py').as_posix()}`
- clean manifest：`{manifest.as_posix()}`
- run 目录：`{run_root.as_posix()}`
- 输入模态：车辆历史 + 旧入口上下文字段；未使用生理、脑电、连续风格、驾驶员风格向量或教师状态。
- split：session-level split，train/val/test = 611/156/139。
- 样本定义：非方向盘车辆动力学 onset，即 `ay/roll_rate` 触发的失稳锚点；方向盘只作为事件后的响应标签。
- 服务器：未使用服务器，未读取服务器指令与密码文件。

注意：此前直接让旧深度入口读取原始车辆 CSV 的 run 已判定无效，因为旧代码会把原始 CSV 中的交替缺失点直接填 0，导致方向盘标签出现不真实的高频跳变。本报告只使用 clean manifest 结果。

## 关键结果

旧脚本按 `legacy_rmse` 选择的 active checkpoint 是 epoch {active_epoch}；同一个 run 中另有 structure-aware checkpoint epoch {structure_epoch}，下面一起列出，便于看旧选择规则的影响。

{test_table.to_string(index=False)}

最主要的 active checkpoint 测试集结果：

- test RMSE：{active_rmse:.6f}
- 主峰错侧率：{active_wrong:.6f}
- 严重幅值不足率：{active_under:.6f}
- structure-aware checkpoint 的 test RMSE：{structure_rmse:.6f}，但它不是本次旧脚本 `legacy_rmse` 选择的 active checkpoint。

## 固定图和坏样本图

- 固定预测图：`{fixed_plot.as_posix()}`
- 坏样本图：`{bad_plot.as_posix()}`

固定图使用此前旧 ridge 诊断固定下来的 pre2 + session-level test 样本，避免只挑好看的样本。坏样本图按 active checkpoint 在 test 集上的逐样本 RMSE 排序取前 12 个。

## 分被试结果

{by_subject.to_string(index=False)}

## 分响应类型结果

{by_response.to_string(index=False)}

## 如何解释

这次结果说明：旧 `vehicle_direct` 深度入口可以在 906 个可用高置信失稳事件上完整训练和评估，且在 session-level test 的整体 RMSE 上明显低于旧 ridge/no-learning 诊断。但是它仍然有较高的严重幅值不足和错侧问题，特别是坏样本图需要继续检查是否集中在大幅响应、反向修正或多段修正。

这不是“新流程强车辆基线”的最终结论。它只是旧代码在新失稳样本上的历史对照，后续仍应把同一批高置信失稳事件整理成新流程正式 manifest，再建立无泄漏、无驾驶员 ID、物理指标齐全的强车辆基线。

## 不能下的结论

- 不能据此证明连续风格有效。
- 不能据此证明生理或脑电有效。
- 不能把旧 `vehicle_direct` 的 RMSE 当作最终上限。
- 不能忽略本次锚点来自车辆动力学 onset，而不是失稳发生前预警。
"""
    (REPORT_DIR / "oldcode_vehicle_direct_full_clean_on_instability_v0_1_cn.md").write_text(report, encoding="utf-8")

    user_summary = f"""# 阶段 3 用户查看版：旧 `vehicle_direct` 全量车辆-only 对照 clean v0.1

生成时间：2026-05-12

## 为什么做

你要求先用之前的旧代码测试这批重新筛选出来的车辆失稳样本，所以这次没有进入风格、生理、脑电路线，而是只跑旧 `vehicle_direct` 车辆-only 深度模型，看旧代码在新失稳样本上到底能做到什么程度。

## 检查了什么

- 906 个可用高置信车辆失稳事件。
- session-level 切分：train 611、val 156、test 139。
- 旧 `vehicle_direct` 全量训练，不是 smoke run。
- 固定预测图和坏样本图。
- 分被试结果和分响应类型结果。

## 目前发现

旧脚本选择的 active checkpoint 测试集 RMSE 为 {active_rmse:.6f}。需要注意，这次是用 clean manifest 跑出的可信旧代码对照；此前直接读原始 CSV 的 run 因缺失点被旧代码填 0，已标为无效诊断。当前结果仍然有物理错误：错侧率 {active_wrong:.6f}，严重幅值不足率 {active_under:.6f}。所以它能拟合一部分轨迹，但还不能说明方向、幅值和复杂响应都可靠。

## 哪些结果可信

- 本次确实是全量 run，不是 96/32/32 的 smoke run。
- 输入只来自车辆，不含生理、脑电、连续风格。
- 固定图和坏样本图是按固定规则生成的，不是挑好看的图。

## 哪些结果还不能下结论

- 不能说这是新流程最终强车辆基线。
- 不能说旧模型已经解决车辆失稳响应预测。
- 不能说连续风格、生理、脑电有效。
- 不能把 `structure_best` 的结果直接当主结果，因为旧脚本本次 active 选择规则是 `legacy_rmse`。

## 下一阶段是否可以继续

可以继续，但建议下一步不是直接上生理，而是把这 906 个高置信失稳事件整理成新流程正式 `samples_master`，再建立新流程强车辆基线。旧代码结果只作为历史对照和坏样本来源。

## 推荐优先查看

1. `{(REPORT_DIR / 'oldcode_vehicle_direct_full_clean_on_instability_v0_1_cn.md').as_posix()}`
2. `{fixed_plot.as_posix()}`
3. `{bad_plot.as_posix()}`
4. `{(out_dir / 'tables' / 'oldcode_vehicle_direct_full_metrics.csv').as_posix()}`
5. `{(out_dir / 'tables' / 'oldcode_vehicle_direct_full_per_sample_metrics.csv').as_posix()}`
"""
    (REPORT_DIR / "stage03_oldcode_vehicle_direct_full_clean_user_summary_cn.md").write_text(user_summary, encoding="utf-8")

    (out_dir / "logs" / "oldcode_vehicle_direct_full_eval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fixed-sample-set", type=Path, default=DEFAULT_FIXED_SET)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    paths = ensure_dirs(args.out_dir)
    run_summary = json.loads((args.run_root / "run_summary.json").read_text(encoding="utf-8"))
    seed = int(run_summary.get("selection_compare", {}).get("active", {}).get("epoch", 2026))
    X_pool, y_pool, curve_pool, ctx_pool, mask_pool, meta_df = load_bundle(args.manifest, seed=seed)

    checkpoint_specs = [
        ("active_legacy_best", args.run_root / "best_model.pt"),
        ("structure_best", args.run_root / "best_model_structure.pt"),
    ]
    results = [
        predict_checkpoint(
            ckpt_path=path,
            checkpoint_tag=tag,
            X_pool=X_pool,
            y_pool=y_pool,
            curve_pool=curve_pool,
            ctx_pool=ctx_pool,
            mask_pool=mask_pool,
            meta_df=meta_df,
            device=str(args.device),
        )
        for tag, path in checkpoint_specs
    ]

    metrics = pd.concat([item["metrics"] for item in results], ignore_index=True)
    per_sample = pd.concat([item["per_sample"] for item in results], ignore_index=True)
    metrics.to_csv(paths["tables"] / "oldcode_vehicle_direct_full_metrics.csv", index=False, encoding="utf-8-sig")
    per_sample.to_csv(paths["tables"] / "oldcode_vehicle_direct_full_per_sample_metrics.csv", index=False, encoding="utf-8-sig")
    meta_df.to_csv(paths["tables"] / "oldcode_vehicle_direct_full_manifest_used.csv", index=False, encoding="utf-8-sig")

    by_subject, by_response = summarize_groups(per_sample, meta_df)
    by_subject.to_csv(paths["tables"] / "oldcode_vehicle_direct_full_by_subject_test.csv", index=False, encoding="utf-8-sig")
    by_response.to_csv(paths["tables"] / "oldcode_vehicle_direct_full_by_response_type_test.csv", index=False, encoding="utf-8-sig")

    pred_map = {item["checkpoint_tag"]: item["pred"] for item in results}
    np.savez_compressed(
        paths["tables"] / "oldcode_vehicle_direct_full_predictions_test_arrays.npz",
        true=y_pool.astype(np.float32),
        mask=mask_pool.astype(np.float32),
        **{f"pred_{tag}": pred.astype(np.float32) for tag, pred in pred_map.items()},
    )

    active_sample = per_sample[(per_sample["checkpoint_tag"] == "active_legacy_best") & (per_sample["split"] == "test")].copy()
    if args.fixed_sample_set.exists():
        fixed_base = pd.read_csv(args.fixed_sample_set)
        fixed_df = fixed_base[["rank", "event_uid"]].merge(active_sample, on="event_uid", how="left").sort_values("rank")
    else:
        fixed_df = active_sample.sort_values("gt_peak_abs", ascending=False).head(12).copy()
        fixed_df["rank"] = np.arange(1, len(fixed_df) + 1)
    bad_df = active_sample.sort_values("sample_rmse", ascending=False).head(12).copy()
    fixed_df.to_csv(paths["tables"] / "oldcode_vehicle_direct_full_fixed_plot_samples.csv", index=False, encoding="utf-8-sig")
    bad_df.to_csv(paths["tables"] / "oldcode_vehicle_direct_full_bad_plot_samples.csv", index=False, encoding="utf-8-sig")

    fixed_plot = paths["figures"] / "oldcode_vehicle_direct_full_fixed_predictions_test.png"
    bad_plot = paths["figures"] / "oldcode_vehicle_direct_full_bad_samples_test.png"
    plot_grid(
        fixed_df,
        pred_map=pred_map,
        y_pool=y_pool,
        mask_pool=mask_pool,
        meta_df=meta_df,
        output_path=fixed_plot,
        title="Old vehicle_direct full run: fixed test predictions",
    )
    plot_grid(
        bad_df,
        pred_map=pred_map,
        y_pool=y_pool,
        mask_pool=mask_pool,
        meta_df=meta_df,
        output_path=bad_plot,
        title="Old vehicle_direct full run: worst active-checkpoint test samples",
    )

    summary = {
        "run_root": str(args.run_root),
        "manifest": str(args.manifest),
        "out_dir": str(args.out_dir),
        "sample_count": int(len(meta_df)),
        "split_counts": meta_df["split"].value_counts().to_dict(),
        "checkpoint_epochs": {item["checkpoint_tag"]: item["checkpoint_epoch"] for item in results},
        "test_metrics": metrics[metrics["split"] == "test"].to_dict(orient="records"),
        "fixed_plot": str(fixed_plot),
        "bad_plot": str(bad_plot),
        "server_used": False,
        "credential_file_read": False,
    }
    write_report(
        out_dir=args.out_dir,
        run_root=args.run_root,
        manifest=args.manifest,
        metrics=metrics,
        by_subject=by_subject,
        by_response=by_response,
        fixed_plot=fixed_plot,
        bad_plot=bad_plot,
        summary=summary,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
