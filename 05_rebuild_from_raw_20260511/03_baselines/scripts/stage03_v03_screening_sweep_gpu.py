# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(os.environ.get("DATA_PROCESS_ROOT", r"F:/data_set_process/data_process"))
ROOT = Path(os.environ.get("REBUILD_ROOT", str(PROJECT_ROOT / "05_rebuild_from_raw_20260511")))
SCRIPT_DIR = ROOT / "03_baselines" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import stage03_v03_screening_sweep as sweep  # noqa: E402
import stage03_v03_vehicle_only_baselines as base  # noqa: E402
import stage03_v03_vehicle_only_inclusion_ablation as incl  # noqa: E402


OUT_ROOT = ROOT / "03_baselines" / "stage03_v03_screening_sweep_gpu"
DATASET_ROOT = ROOT / "03_processed_datasets" / "extreme_condition_v0_3_screening_sweep_gpu"
REPORT_DIR = ROOT / "09_reports"
NOTES_DIR = ROOT / "00_project_notes"
DAILY_LOG = NOTES_DIR / "daily_logs" / "2026-05-19.md"
ARTIFACT_INDEX = NOTES_DIR / "ARTIFACT_INDEX_CN.md"

SUMMARY_PATH = OUT_ROOT / "tables" / "v03_screening_sweep_gpu_summary.csv"
RANKING_PATH = OUT_ROOT / "tables" / "v03_screening_sweep_gpu_ranking.csv"
EXTRA_SOURCE_PATH = OUT_ROOT / "tables" / "v03_screening_sweep_gpu_extra_source_counts.csv"

RANDOM_SEED = 20260519


class LinearHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def ensure_dirs() -> None:
    for path in [OUT_ROOT / "tables", OUT_ROOT / "logs", REPORT_DIR, NOTES_DIR / "daily_logs"]:
        path.mkdir(parents=True, exist_ok=True)


def configure_modules() -> None:
    incl.OUT_ROOT = OUT_ROOT
    incl.DATASET_ROOT = DATASET_ROOT
    incl.REPORT_DIR = REPORT_DIR
    incl.NOTES_DIR = NOTES_DIR
    incl.DAILY_LOG = DAILY_LOG
    incl.ARTIFACT_INDEX = ARTIFACT_INDEX
    base.TABLE_DIR = OUT_ROOT / "tables"
    base.FIG_DIR = OUT_ROOT / "figures"
    base.LOG_DIR = OUT_ROOT / "logs"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def masked_mse(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.bool()
    diff = (pred - y).pow(2)
    if valid.any():
        return diff[valid].mean()
    return diff.mean()


def rmse_np(y: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    valid = mask & np.isfinite(y) & np.isfinite(pred)
    if not valid.any():
        return float("nan")
    return float(np.sqrt(np.mean(np.square(y[valid] - pred[valid]))))


def train_one_model(
    model: nn.Module,
    model_name: str,
    x_mat: np.ndarray,
    y: np.ndarray,
    y_mask: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    patience: int,
) -> tuple[str, np.ndarray, dict[str, Any]]:
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_ds = TensorDataset(
        torch.from_numpy(x_mat[train_idx]).float(),
        torch.from_numpy(y[train_idx]).float(),
        torch.from_numpy(y_mask[train_idx].astype(np.float32)).float(),
    )
    loader = DataLoader(train_ds, batch_size=min(batch_size, len(train_ds)), shuffle=True, drop_last=False)
    x_val = torch.from_numpy(x_mat[val_idx]).float().to(device)
    y_val = torch.from_numpy(y[val_idx]).float().to(device)
    mask_val = torch.from_numpy(y_mask[val_idx].astype(np.float32)).float().to(device)

    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    wait = 0
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb, mb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            mb = mb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = masked_mse(model(xb), yb, mb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            val_pred = model(x_val)
            val_loss = masked_mse(val_pred, y_val, mask_val).item()
        val_rmse = float(np.sqrt(max(val_loss, 0.0)))
        if val_rmse + 1e-6 < best_val:
            best_val = val_rmse
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(x_mat), 2048):
            xb = torch.from_numpy(x_mat[start : start + 2048]).float().to(device)
            preds.append(model(xb).cpu().numpy().astype(np.float32))
    pred = np.vstack(preds)
    info = {"model_name": model_name, "best_val_rmse": best_val, "best_epoch": best_epoch}
    return model_name, pred, info


def train_torch_models(
    X: pd.DataFrame,
    y: np.ndarray,
    y_mask: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    x_mat, feature_names = base.compact_numpy_features(X, train_idx)
    y_filled = np.where(y_mask, y, 0.0).astype(np.float32)
    model_specs = [
        ("torch_linear_vehicle_context", LinearHead(x_mat.shape[1], y.shape[1]), 260, 1e-3, 1e-4, 512, 35),
        ("torch_mlp256_vehicle_context", MLPHead(x_mat.shape[1], y.shape[1], hidden=256, dropout=0.08), 360, 1e-3, 1e-4, 512, 45),
        ("torch_mlp512_vehicle_context", MLPHead(x_mat.shape[1], y.shape[1], hidden=512, dropout=0.10), 420, 8e-4, 1e-4, 512, 50),
    ]
    preds: dict[str, np.ndarray] = {}
    info_rows: list[dict[str, Any]] = []
    print(f"fit torch baselines device={device} features={x_mat.shape[1]} train={len(train_idx)} val={len(val_idx)}", flush=True)
    for offset, (name, model, epochs, lr, wd, bs, patience) in enumerate(model_specs):
        set_seed(RANDOM_SEED + offset)
        model_name, pred, info = train_one_model(
            model,
            name,
            x_mat,
            y_filled,
            y_mask,
            train_idx,
            val_idx,
            device,
            epochs=epochs,
            lr=lr,
            weight_decay=wd,
            batch_size=bs,
            patience=patience,
        )
        preds[model_name] = pred
        info_rows.append({**info, "feature_count": int(x_mat.shape[1]), "feature_names_count": len(feature_names)})
        print(f"  {model_name} best_val_rmse={info['best_val_rmse']:.6f} epoch={info['best_epoch']}", flush=True)
    return preds, info_rows


def select_by_val(metrics: pd.DataFrame) -> dict[str, Any]:
    val = metrics[metrics["split"].astype(str).eq("val")].sort_values("rmse_steer")
    if val.empty:
        return {}
    model_name = str(val.iloc[0]["model_name"])
    test = metrics[metrics["split"].astype(str).eq("test") & metrics["model_name"].astype(str).eq(model_name)]
    if test.empty:
        return {}
    row = test.iloc[0].to_dict()
    row["val_selected_model"] = model_name
    row["val_rmse_for_selected"] = float(val.iloc[0]["rmse_steer"])
    return row


def run_variant_gpu(variant: dict[str, Any], sample_split: dict[str, str], session_split: dict[str, str], device: torch.device) -> dict[str, Any]:
    variant_id = str(variant["variant_id"])
    out_dir = OUT_ROOT / variant_id
    table_dir = out_dir / "tables"
    log_dir = out_dir / "logs"
    for path in [table_dir, log_dir]:
        path.mkdir(parents=True, exist_ok=True)

    summary_json = log_dir / f"{variant_id}_gpu_summary.json"
    if summary_json.exists():
        try:
            payload = json.loads(summary_json.read_text(encoding="utf-8"))
            if isinstance(payload.get("result"), dict):
                print(f"reuse {variant_id}", flush=True)
                return payload["result"]
        except Exception:
            pass

    feature_backup = list(base.VEHICLE_FEATURES)
    drop_features = set(str(x) for x in variant.get("drop_features") or [])
    if drop_features:
        base.VEHICLE_FEATURES = [x for x in feature_backup if x not in drop_features]
    try:
        x, x_mask, y, y_mask, meta, dataset_summary = incl.build_variant_dataset(variant, sample_split, session_split)
        train_idx = np.where(meta["split"].astype(str).to_numpy() == "train")[0]
        val_idx = np.where(meta["split"].astype(str).to_numpy() == "val")[0]
        test_idx = np.where(meta["split"].astype(str).to_numpy() == "test")[0]
        if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
            raise RuntimeError(f"{variant_id} split invalid: {dataset_summary.get('split_counts')}")
        X, _ = base.flatten_history_features(x, x_mask, meta)
        preds = base.build_no_learning_predictions(y, y_mask, x, x_mask, meta, train_idx)
        torch_preds, train_info = train_torch_models(X, y, y_mask, train_idx, val_idx, device)
        preds.update(torch_preds)
        metrics, per_sample = base.evaluate_all(y, y_mask, base.LABEL_TIME, meta, preds, train_idx)
    finally:
        base.VEHICLE_FEATURES = feature_backup

    metrics.to_csv(table_dir / f"{variant_id}_gpu_metrics.csv", index=False, encoding="utf-8-sig")
    per_sample.to_csv(table_dir / f"{variant_id}_gpu_per_sample_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(train_info).to_csv(table_dir / f"{variant_id}_gpu_train_info.csv", index=False, encoding="utf-8-sig")

    selected = select_by_val(metrics)
    if not selected:
        selected = metrics[metrics["split"].astype(str).eq("test")].sort_values("rmse_steer").iloc[0].to_dict()
        selected["val_selected_model"] = str(selected["model_name"])
        selected["val_rmse_for_selected"] = float("nan")

    result = {
        "variant_id": variant_id,
        "name_cn": variant["name_cn"],
        "sample_count": int(dataset_summary["sample_count"]),
        "extra_episode_count": int(dataset_summary.get("extra_episode_count", 0)),
        "dropped_count": int(dataset_summary.get("dropped_count", 0)),
        "split_counts_json": json.dumps(dataset_summary.get("split_counts", {}), ensure_ascii=False),
        "category_counts_json": json.dumps(dataset_summary.get("category_counts", {}), ensure_ascii=False),
        "val_selected_model": str(selected["val_selected_model"]),
        "val_rmse_for_selected": float(selected["val_rmse_for_selected"]),
        "test_rmse_steer": float(selected["rmse_steer"]),
        "test_primary_rmse_0_2s": float(selected["primary_rmse_0_2s"]),
        "test_tail_rmse_2_5s": float(selected["tail_rmse_2_5s"]),
        "test_wrong_side_rate_large": float(selected["wrong_side_rate_large"]),
        "test_severe_amp_under_rate_large": float(selected["severe_amp_under_rate_large"]),
        "test_large_response_recall": float(selected["large_response_recall"]),
    }
    summary_json.write_text(json.dumps({"dataset_summary": dataset_summary, "result": result}, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def score_rows(summary: pd.DataFrame) -> pd.DataFrame:
    df = summary.copy()
    for col in [
        "test_rmse_steer",
        "test_wrong_side_rate_large",
        "test_severe_amp_under_rate_large",
        "test_large_response_recall",
        "test_tail_rmse_2_5s",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    base_row = df[df["variant_id"].eq("s00_base_nolat")]
    if len(base_row):
        b = base_row.iloc[0]
        df["delta_rmse_vs_base"] = df["test_rmse_steer"] - float(b["test_rmse_steer"])
        df["delta_wrong_side_vs_base"] = df["test_wrong_side_rate_large"] - float(b["test_wrong_side_rate_large"])
        df["delta_severe_under_vs_base"] = df["test_severe_amp_under_rate_large"] - float(b["test_severe_amp_under_rate_large"])
        df["delta_large_recall_vs_base"] = df["test_large_response_recall"] - float(b["test_large_response_recall"])
    else:
        df["delta_rmse_vs_base"] = np.nan
        df["delta_wrong_side_vs_base"] = np.nan
        df["delta_severe_under_vs_base"] = np.nan
        df["delta_large_recall_vs_base"] = np.nan
    df["screening_score"] = (
        -df["delta_rmse_vs_base"].fillna(0.0)
        - 0.35 * df["delta_wrong_side_vs_base"].fillna(0.0)
        - 0.25 * df["delta_severe_under_vs_base"].fillna(0.0)
        + 0.15 * df["delta_large_recall_vs_base"].fillna(0.0)
    )
    return df.sort_values(["screening_score", "test_rmse_steer"], ascending=[False, True])


def fmt(v: Any) -> str:
    try:
        x = float(v)
    except Exception:
        return str(v)
    if not np.isfinite(x):
        return "NA"
    return f"{x:.4f}"


def markdown_table(df: pd.DataFrame, limit: int = 12) -> str:
    cols = [
        "variant_id",
        "name_cn",
        "sample_count",
        "extra_episode_count",
        "val_selected_model",
        "test_rmse_steer",
        "delta_rmse_vs_base",
        "test_wrong_side_rate_large",
        "test_severe_amp_under_rate_large",
        "test_large_response_recall",
        "screening_score",
    ]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.head(limit)[cols].iterrows():
        vals = []
        for col in cols:
            val = row[col]
            vals.append(fmt(val) if col not in ["variant_id", "name_cn", "val_selected_model"] else str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(ranking: pd.DataFrame, device: torch.device) -> None:
    report = REPORT_DIR / "stage03_v03_screening_sweep_gpu_user_summary_cn.md"
    best_score = ranking.iloc[0].to_dict()
    best_rmse = ranking.sort_values("test_rmse_steer").iloc[0].to_dict()
    base_row = ranking[ranking["variant_id"].eq("s00_base_nolat")].iloc[0].to_dict()
    lines = [
        "# v0.3 样本筛选策略 GPU 快筛",
        "",
        "## 为什么改成 GPU",
        "",
        "之前的连续筛选脚本沿用 sklearn 核岭回归，默认只能走 CPU。它适合做传统基线，但连续扫描十几个样本策略太慢。本轮改用 PyTorch 车辆-only 小网络，在 GPU 上训练同一结构，用来快速判断哪些样本筛选方向值得继续。",
        "",
        "注意：这张表用于筛选样本方向，不和旧 sklearn 核岭回归表直接混作同一模型结论。",
        "",
        "## 运行设置",
        "",
        f"- 设备：`{device}`。",
        "- 输入：车辆历史 + 事件/上下文表格特征，不含连续风格、生理或脑电。",
        "- 模型：线性头、256 隐层网络、512 隐层网络；按验证集 RMSE 选模型，再报告测试集。",
        "",
        "## 基础版本",
        "",
        f"- `s00_base_nolat` 样本数 {int(base_row['sample_count'])}，test RMSE={fmt(base_row['test_rmse_steer'])}，大响应错侧率={fmt(base_row['test_wrong_side_rate_large'])}，严重幅值不足率={fmt(base_row['test_severe_amp_under_rate_large'])}，大响应召回={fmt(base_row['test_large_response_recall'])}。",
        "",
        "## 排名前 12 的筛选策略",
        "",
        markdown_table(ranking, 12),
        "",
        "## 自动读法",
        "",
        f"- 按综合分数，最好的是 `{best_score['variant_id']}`，test RMSE={fmt(best_score['test_rmse_steer'])}，综合分数={fmt(best_score['screening_score'])}。",
        f"- 单看整体 RMSE，最低的是 `{best_rmse['variant_id']}`，test RMSE={fmt(best_rmse['test_rmse_steer'])}。",
        "- 如果 RMSE 降低但大响应召回/严重幅值不足恶化，说明它更像普通拟合改善，不一定适合作为极限主样本。",
        "- 如果物理指标改善但 RMSE 不占优，可以考虑作为极限姿态专用样本集，而不是和普通样本混训。",
        "",
        "## 产物位置",
        "",
        f"- 汇总表：`{SUMMARY_PATH}`",
        f"- 排名表：`{RANKING_PATH}`",
        f"- 输出目录：`{OUT_ROOT}`",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")


def append_notes(ranking: pd.DataFrame, device: torch.device) -> None:
    best = ranking.iloc[0].to_dict()
    block = (
        "## 2026-05-19 v0.3 样本筛选策略 GPU 快筛\n\n"
        "- 当前阶段：车辆-only 样本筛选策略对比，不涉及连续风格、生理或脑电。\n"
        f"- 本轮动作：停止 CPU 核岭 sweep，改用 PyTorch GPU 快筛，设备 `{device}`。\n"
        f"- 当前综合排序第一：`{best['variant_id']}`，test RMSE={fmt(best['test_rmse_steer'])}，综合分数={fmt(best['screening_score'])}。\n"
        f"- 用户查看版报告：`{REPORT_DIR / 'stage03_v03_screening_sweep_gpu_user_summary_cn.md'}`。\n"
        f"- 输出目录：`{OUT_ROOT}`。\n"
    )
    for path in [NOTES_DIR / "PROJECT_STATUS_CN.md", NOTES_DIR / "TASK_QUEUE_CN.md", DAILY_LOG]:
        if path.exists():
            raw = path.read_text(encoding="utf-8")
            if "## 2026-05-19 v0.3 样本筛选策略 GPU 快筛" not in raw:
                path.write_text(raw.rstrip() + "\n\n" + block, encoding="utf-8")
    if ARTIFACT_INDEX.exists():
        raw = ARTIFACT_INDEX.read_text(encoding="utf-8")
        artifact = (
            "## v0.3 样本筛选策略 GPU 快筛\n\n"
            f"- 用户查看版报告：`{REPORT_DIR / 'stage03_v03_screening_sweep_gpu_user_summary_cn.md'}`\n"
            f"- 汇总表：`{SUMMARY_PATH}`\n"
            f"- 排名表：`{RANKING_PATH}`\n"
            f"- 输出目录：`{OUT_ROOT}`\n"
        )
        if "## v0.3 样本筛选策略 GPU 快筛" not in raw:
            ARTIFACT_INDEX.write_text(raw.rstrip() + "\n\n" + artifact, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    configure_modules()
    set_seed(RANDOM_SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; GPU screening requires a CUDA device.")
    device = torch.device("cuda")
    print(f"cuda={torch.cuda.get_device_name(0)}", flush=True)

    groups, group_counts = sweep.load_uid_groups()
    group_counts.to_csv(EXTRA_SOURCE_PATH, index=False, encoding="utf-8-sig")
    variants = sweep.make_variants(groups)
    sample_split, session_split = incl.load_reference_split()
    rows: list[dict[str, Any]] = []
    for variant in variants:
        print(f"run-gpu {variant['variant_id']} extra={len(set(variant.get('extra_episode_uids') or []))}", flush=True)
        rows.append(run_variant_gpu(variant, sample_split, session_split, device))
    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
    ranking = score_rows(summary)
    ranking.to_csv(RANKING_PATH, index=False, encoding="utf-8-sig")
    write_report(ranking, device)
    append_notes(ranking, device)
    print(
        ranking[
            [
                "variant_id",
                "sample_count",
                "extra_episode_count",
                "val_selected_model",
                "test_rmse_steer",
                "test_wrong_side_rate_large",
                "test_severe_amp_under_rate_large",
                "test_large_response_recall",
                "screening_score",
            ]
        ]
        .head(15)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
