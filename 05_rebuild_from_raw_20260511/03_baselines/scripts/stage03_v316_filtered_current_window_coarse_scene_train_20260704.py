#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v316 filtered current-window coarse-scene curve train.

目的：
- 按用户要求“完整跑一次看看”；
- 在 v315 保留清单上重跑当前 0-2 秒窗口模型；
- 复用 v307 的粗场景条件曲线模型结构，但把 84 个来源可疑事件隔离出当前窗口训练/验证/测试统计；
- 与第300版参照和第307版旧结果在同一套保留测试集上比较。

边界：
- 本脚本训练模型，但不重切窗口；
- v315 的 77 个重锚定候选不会直接改锚点参与训练；
- 候选选择只看过滤后的验证集，不使用测试误差选模型。
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import pickle
import random
import shutil
import sys
import time
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch


SEED = 20260704
ROOT = Path(__file__).resolve().parents[2]
BASELINES = ROOT / "03_baselines"
SCRIPTS = BASELINES / "scripts"
V307_SCRIPT = SCRIPTS / "stage03_v307_coarse_scene_label_conditioned_curve_model_20260704.py"
V315_POLICY = (
    BASELINES
    / "v315_rapid_steering_filter_reanchor_plan_20260704"
    / "tables"
    / "v315_current_window_training_policy_all_delay0.csv"
)
V307_PRED = (
    BASELINES
    / "v307_coarse_scene_label_conditioned_curve_model_20260704"
    / "v307_coarse_scene_label_conditioned_predictions.npz"
)

OUT = BASELINES / "v316_filtered_current_window_coarse_scene_train_20260704"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"
MODELS = OUT / "models"


def import_module_from_path(module_name: str, path: Path):
    """按路径导入第307版脚本，复用已跑通的数据和模型函数。"""

    if not path.exists():
        raise FileNotFoundError(f"缺少依赖脚本：{path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法导入脚本：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V307 = import_module_from_path("stage03_v307_for_v316_filtered_train", V307_SCRIPT)
V304 = V307.V304
FUTURE_GRID = V307.FUTURE_GRID


def patch_output_globals() -> None:
    """让复用函数写入第316版输出目录。"""

    V307.SEED = SEED
    V307.OUT = OUT
    V307.TABLES = TABLES
    V307.FIGURES = FIGURES
    V307.REPORTS = REPORTS
    V307.LOGS = LOGS
    V307.MODELS = MODELS
    V307.patch_v304_output_globals()


def ensure_dirs() -> None:
    """创建输出目录。"""

    for folder in (TABLES, FIGURES, REPORTS, LOGS, MODELS):
        folder.mkdir(parents=True, exist_ok=True)


def clean_out_dir() -> None:
    """只清理第316版自己的输出目录。"""

    resolved_out = OUT.resolve()
    resolved_base = BASELINES.resolve()
    if resolved_base not in resolved_out.parents:
        raise RuntimeError(f"拒绝清理非预期目录：{resolved_out}")
    if OUT.exists():
        shutil.rmtree(OUT)
    ensure_dirs()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """表格使用 utf-8-sig，方便中文软件查看。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_json(payload: Dict[str, object], path: Path) -> None:
    """保存 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def file_sha256(path: Path) -> str:
    """计算文件哈希。"""

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def set_seed(seed: int) -> None:
    """固定随机种子。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False


def to_bool(value: object) -> bool:
    """兼容表格里的布尔文本。"""

    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y", "是"}


def markdown_table(df: pd.DataFrame) -> str:
    """不用额外依赖生成报告表格。"""

    if df.empty:
        return "（空表）"
    cols = list(df.columns)

    def cell(value: object) -> str:
        if isinstance(value, float):
            text = f"{value:.6g}" if np.isfinite(value) else ""
        else:
            text = str(value)
        return text.replace("|", "｜").replace("\n", " ")

    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(cell(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def load_policy_table() -> pd.DataFrame:
    """读取第315版处理策略表。"""

    if not V315_POLICY.exists():
        raise FileNotFoundError(f"缺少第315版处理策略表：{V315_POLICY}")
    policy = pd.read_csv(V315_POLICY, encoding="utf-8-sig")
    if policy["event_uid"].duplicated().any():
        dup = policy.loc[policy["event_uid"].duplicated(), "event_uid"].head(5).tolist()
        raise AssertionError(f"第315版策略表事件重复：{dup}")
    for col in ["v315_current_window_train_keep", "v315_isolate_from_current_window"]:
        if col in policy.columns:
            policy[col] = policy[col].map(to_bool)
    return policy


def apply_v315_policy_to_data(data, event_table: pd.DataFrame, policy: pd.DataFrame):
    """把第315版保留/隔离策略映射到 rolling manifest。"""

    keep_map = policy.set_index("event_uid")["v315_current_window_train_keep"].map(to_bool)
    policy_cols = [
        "v315_policy",
        "v315_policy_cn",
        "v315_next_action_cn",
        "v315_current_window_train_keep",
        "v315_isolate_from_current_window",
        "v315_candidate_after_reanchor",
        "v315_candidate_anchor_shift_s",
        "v315_candidate_observation_s",
    ]
    policy_cols = [c for c in policy_cols if c in policy.columns]
    policy_small = policy.set_index("event_uid")[policy_cols]

    manifest = data.manifest.copy()
    mapped = manifest["event_uid"].astype(str).map(keep_map)
    if mapped.isna().any():
        missing = manifest.loc[mapped.isna(), "event_uid"].drop_duplicates().head(10).tolist()
        raise AssertionError(f"rolling manifest 中存在第315版策略未覆盖事件：{missing}")
    manifest["v315_original_split"] = manifest["split"].astype(str)
    manifest["v315_current_window_train_keep"] = mapped.astype(bool).to_numpy()
    for col in policy_small.columns:
        manifest[col] = manifest["event_uid"].astype(str).map(policy_small[col])

    isolate = ~manifest["v315_current_window_train_keep"].astype(bool)
    manifest.loc[isolate, "split"] = "isolated_" + manifest.loc[isolate, "v315_original_split"].astype(str)

    filtered_data = V304.V300.clone_rolling_data(data, manifest=manifest)
    event_table_out = event_table.copy()
    event_table_out["v315_current_window_train_keep"] = event_table_out["event_uid"].astype(str).map(keep_map).astype(bool)
    return filtered_data, event_table_out


def prepare_v316_data(hard_event_extra: float, policy: pd.DataFrame):
    """读取数据、应用第315版过滤策略，并构造第316版输入。"""

    raw_data = V304.V238.load_v236_data()
    data0, event_table0 = V304.V300.apply_v299_within_subject_split(raw_data)
    data, event_table = apply_v315_policy_to_data(data0, event_table0, policy)
    no_subject = V304.V300.prepare_variant(
        "no_subject",
        data,
        {
            "uses_subject_onehot": False,
            "description_cn": "第316版只使用第315版保留事件训练当前0到2秒窗口；结构沿用第307版粗场景条件模型",
        },
    )

    x_base = V304.V238.build_base_design_matrix(data)
    roll_raw, roll_feature_names, signal_audit = V304.V302.build_roll_cause_summary(x_base, data.feature_names)
    train_mask = data.manifest["split"].astype(str).to_numpy() == "train"
    roll_scaled, impute_mean, scale_mean, scale_std = V304.fit_transform_roll_features(roll_raw, train_mask)
    event_label, event_label_name, class_names, labels = V307.load_coarse_scene_labels_for_all_samples(data.manifest)

    train_labels = event_label[train_mask]
    counts = np.bincount(train_labels, minlength=len(class_names)).astype(np.float32)
    counts[counts < 1] = 1.0
    class_weight = counts.sum() / (len(class_names) * counts)
    class_weight = np.clip(class_weight, 0.35, 4.0).astype(np.float32)
    curve_mult = V304.build_curve_sample_multiplier(data.manifest, event_label_name, hard_event_extra)

    write_csv(signal_audit, TABLES / "v316_roll_cause_signal_coverage.csv")
    write_csv(
        pd.DataFrame(
            [
                {
                    "roll_cause_feature_n": int(roll_scaled.shape[1]),
                    "raw_nan_rate": float(np.mean(~np.isfinite(roll_raw))),
                    "scaled_nan_rate": float(np.mean(~np.isfinite(roll_scaled))),
                    "coarse_scene_class_n": int(len(class_names)),
                    "hard_event_extra": float(hard_event_extra),
                    "filtered_train_rows": int(train_mask.sum()),
                    "filtered_train_events": int(data.manifest.loc[train_mask, "event_uid"].nunique()),
                }
            ]
        ),
        TABLES / "v316_input_audit.csv",
    )
    write_csv(
        pd.DataFrame({"coarse_scene_label": class_names, "class_index": list(range(len(class_names))), "class_weight": class_weight}),
        TABLES / "v316_coarse_scene_class_mapping.csv",
    )

    return V304.RollPrepared(
        data=data,
        prepared=no_subject,
        roll_raw=roll_raw.astype(np.float32),
        roll_scaled=roll_scaled.astype(np.float32),
        roll_feature_names=roll_feature_names,
        roll_impute_mean=impute_mean,
        roll_scale_mean=scale_mean,
        roll_scale_std=scale_std,
        event_label=event_label,
        event_label_name=event_label_name,
        class_names=class_names,
        class_weight=class_weight,
        curve_sample_multiplier=curve_mult,
        labels_table=labels,
        event_table=event_table,
    )


def build_v315_split_audit(manifest: pd.DataFrame, event_table: pd.DataFrame) -> pd.DataFrame:
    """生成过滤前后按划分的事件/样本数量。"""

    rows: List[Dict[str, object]] = []
    for split_name in ["train", "val", "test"]:
        original = manifest["v315_original_split"].astype(str).eq(split_name)
        keep = manifest["split"].astype(str).eq(split_name)
        isolated = manifest["split"].astype(str).eq(f"isolated_{split_name}")
        rows.append(
            {
                "split": split_name,
                "original_rows": int(original.sum()),
                "keep_rows": int(keep.sum()),
                "isolated_rows": int(isolated.sum()),
                "original_events": int(manifest.loc[original, "event_uid"].nunique()),
                "keep_events": int(manifest.loc[keep, "event_uid"].nunique()),
                "isolated_events": int(manifest.loc[isolated, "event_uid"].nunique()),
            }
        )
    event_split_n = manifest.groupby("event_uid")["split"].nunique()
    event_delay_n = manifest.groupby("event_uid")["delay_ms"].nunique()
    rows.append(
        {
            "split": "audit",
            "original_rows": int(len(manifest)),
            "keep_rows": int(manifest["v315_current_window_train_keep"].astype(bool).sum()),
            "isolated_rows": int((~manifest["v315_current_window_train_keep"].astype(bool)).sum()),
            "original_events": int(manifest["event_uid"].nunique()),
            "keep_events": int(event_table["v315_current_window_train_keep"].astype(bool).sum()),
            "isolated_events": int((~event_table["v315_current_window_train_keep"].astype(bool)).sum()),
            "event_in_multiple_splits_n": int((event_split_n > 1).sum()),
            "event_without_6_delay_rows_n": int((event_delay_n != 6).sum()),
            "duplicate_event_delay_rows_n": int(manifest.duplicated(["event_uid", "delay_ms"]).sum()),
        }
    )
    return pd.DataFrame(rows)


def load_previous_v307_prediction(manifest: pd.DataFrame) -> Tuple[np.ndarray, str]:
    """读取第307版旧选中预测，作为同一过滤测试集上的对照。"""

    if not V307_PRED.exists():
        raise FileNotFoundError(f"缺少第307版预测包：{V307_PRED}")
    with np.load(V307_PRED, allow_pickle=True) as z:
        event_uid = z["event_uid"].astype(str)
        delay_ms = z["delay_ms"].astype(int)
        pred = z["pred_v307_selected"].astype(np.float32)
        name = str(z["best_v307_model"][0])
    if not np.array_equal(manifest["event_uid"].astype(str).to_numpy(), event_uid):
        raise AssertionError("第307版预测 event_uid 与当前顺序不一致")
    if not np.array_equal(manifest["delay_ms"].astype(int).to_numpy(), delay_ms):
        raise AssertionError("第307版预测 delay_ms 与当前顺序不一致")
    return pred, f"v307_previous_{name}"


def metric_row(summary: pd.DataFrame, model_name: str, split: str, group: str):
    """取摘要表单行。"""

    one = summary[
        summary["model_name"].astype(str).eq(model_name)
        & summary["split"].astype(str).eq(split)
        & summary["group"].astype(str).eq(group)
    ]
    if one.empty:
        return None
    return one.iloc[0]


def value_line(summary: pd.DataFrame, model_name: str, split: str, group: str) -> str:
    """格式化均方根误差。"""

    row = metric_row(summary, model_name, split, group)
    if row is None or not np.isfinite(float(row["sample_rmse_mean"])):
        return "无"
    return f"{float(row['sample_rmse_mean']):.6f}"


def write_report(
    selection: pd.DataFrame,
    delay0_summary: pd.DataFrame,
    event_metrics: pd.DataFrame,
    guardrail: Dict[str, object],
    selected_name: str,
    v300_name: str,
    old_v307_name: str,
) -> Path:
    """写第316版中文报告。"""

    selected_rows = delay0_summary[
        delay0_summary["model_name"].isin([v300_name, old_v307_name, selected_name])
        & delay0_summary["split"].eq("test")
        & delay0_summary["group"].isin(["all", "within_bad_top10", "within_bad_top20", "strong_steer"])
    ][["model_name", "split", "group", "n", "sample_rmse_mean", "sample_rmse_median", "sample_rmse_p90"]]

    lines = [
        "# 第316版过滤当前窗口样本后完整重训",
        "",
        "## 结论",
        "",
        f"- 本轮按第315版保留清单完整重训，候选选择只看过滤后的验证集。",
        f"- 选中候选：`{selected_name}`。",
        f"- 参照第300版：`{v300_name}`。",
        f"- 旧第307版对照：`{old_v307_name}`。",
        f"- 训练保留事件：`{guardrail['train_keep_event_n']}`；验证保留事件：`{guardrail['val_keep_event_n']}`；测试保留事件：`{guardrail['test_keep_event_n']}`。",
        "",
        "## 过滤后测试集核心结果",
        "",
        f"- 全部：第300版 `{value_line(delay0_summary, v300_name, 'test', 'all')}`；旧第307版 `{value_line(delay0_summary, old_v307_name, 'test', 'all')}`；第316版 `{value_line(delay0_summary, selected_name, 'test', 'all')}`。",
        f"- 原困难前10：第300版 `{value_line(delay0_summary, v300_name, 'test', 'within_bad_top10')}`；旧第307版 `{value_line(delay0_summary, old_v307_name, 'test', 'within_bad_top10')}`；第316版 `{value_line(delay0_summary, selected_name, 'test', 'within_bad_top10')}`。",
        f"- 原困难前20：第300版 `{value_line(delay0_summary, v300_name, 'test', 'within_bad_top20')}`；旧第307版 `{value_line(delay0_summary, old_v307_name, 'test', 'within_bad_top20')}`；第316版 `{value_line(delay0_summary, selected_name, 'test', 'within_bad_top20')}`。",
        "",
        "## 测试集摘要表",
        "",
        markdown_table(selected_rows),
        "",
        "## 验证选模表",
        "",
        markdown_table(selection.drop(columns=["config_json"], errors="ignore")),
        "",
        "## 事件辅助头",
        "",
        markdown_table(event_metrics[event_metrics["model_name"].eq(selected_name)]),
        "",
        "## 边界",
        "",
        "- 本轮没有重切第315版重锚定候选，只是把它们隔离出当前窗口任务。",
        "- 第315版隔离清单不参与训练、验证选模或测试主统计。",
        "- 旧第307版在同一过滤后测试集上只作为对照，不参与选模。",
    ]
    path = REPORTS / "v316_filtered_current_window_coarse_scene_train_cn.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_file_inventory() -> pd.DataFrame:
    """记录输出文件清单。"""

    rows = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name != "file_inventory.csv":
            rows.append(
                {
                    "relative_path": str(path.relative_to(OUT)),
                    "size_bytes": int(path.stat().st_size),
                    "sha256": file_sha256(path),
                }
            )
    inv = pd.DataFrame(rows)
    write_csv(inv, LOGS / "file_inventory.csv")
    return inv


def make_zip_package() -> Tuple[Path, bool]:
    """打包并自检。"""

    zip_path = OUT / "v316_filtered_current_window_coarse_scene_train_20260704.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(OUT.rglob("*")):
            if path == zip_path or not path.is_file():
                continue
            zf.write(path, path.relative_to(OUT))
    with zipfile.ZipFile(zip_path, "r") as zf:
        bad = zf.testzip()
    return zip_path, bad is None


def main() -> None:
    start_time = time.time()
    patch_output_globals()
    clean_out_dir()
    torch.set_num_threads(1)
    set_seed(SEED)

    print("[v316] 读取第315版保留清单并构造过滤后训练数据", flush=True)
    policy = load_policy_table()
    prepared_base = prepare_v316_data(hard_event_extra=0.0, policy=policy)
    split_audit = build_v315_split_audit(prepared_base.data.manifest, prepared_base.event_table)
    write_csv(split_audit, TABLES / "v316_v315_filtered_split_audit.csv")

    y_true_curve = prepared_base.data.y_future[:, :, 0].astype(np.float32)
    pred_v300, v300_name, v300_guard = V304.load_v300_prediction_all(prepared_base.data.manifest)
    pred_v307_old, old_v307_name = load_previous_v307_prediction(prepared_base.data.manifest)

    input_hashes = pd.DataFrame(
        [
            {"input_name": "v307_script_reused", "path": str(V307_SCRIPT), "sha256": file_sha256(V307_SCRIPT)},
            {"input_name": "v315_policy_table", "path": str(V315_POLICY), "sha256": file_sha256(V315_POLICY)},
            {"input_name": "v300_predictions", "path": str(V304.V300_PRED), "sha256": file_sha256(V304.V300_PRED)},
            {"input_name": "v307_predictions", "path": str(V307_PRED), "sha256": file_sha256(V307_PRED)},
        ]
    )
    write_csv(input_hashes, LOGS / "input_hashes.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[v316] 使用设备：{device}", flush=True)

    configs: List[Tuple[str, Dict[str, object]]] = [
        (
            "v316_filtered_scene_init_aux003_film005_h64",
            {
                "hidden_dim": 64,
                "n_heads": 4,
                "n_layers": 3,
                "mixer_layers": 2,
                "mlp_hidden": 112,
                "roll_hidden": 128,
                "event_embed_dim": 64,
                "dropout": 0.06,
                "film_scale": 0.05,
                "smooth_weight": 0.02,
                "aux_weight": 0.03,
                "hard_event_extra": 0.0,
                "init_from_v300": True,
                "v300_init_model_name": v300_name,
                "lr": 2e-4,
                "min_lr": 1e-5,
                "weight_decay": 3e-4,
                "batch_size": 384,
                "max_epochs": 55,
                "patience": 9,
            },
        ),
        (
            "v316_filtered_scene_init_aux005_film010_h64",
            {
                "hidden_dim": 64,
                "n_heads": 4,
                "n_layers": 3,
                "mixer_layers": 2,
                "mlp_hidden": 128,
                "roll_hidden": 160,
                "event_embed_dim": 64,
                "dropout": 0.08,
                "film_scale": 0.10,
                "smooth_weight": 0.025,
                "aux_weight": 0.05,
                "hard_event_extra": 0.0,
                "init_from_v300": True,
                "v300_init_model_name": v300_name,
                "lr": 2e-4,
                "min_lr": 1e-5,
                "weight_decay": 4e-4,
                "batch_size": 384,
                "max_epochs": 60,
                "patience": 10,
            },
        ),
        (
            "v316_filtered_scene_init_aux006_film010_hard110_h64",
            {
                "hidden_dim": 64,
                "n_heads": 4,
                "n_layers": 3,
                "mixer_layers": 2,
                "mlp_hidden": 128,
                "roll_hidden": 160,
                "event_embed_dim": 64,
                "dropout": 0.08,
                "film_scale": 0.10,
                "smooth_weight": 0.025,
                "aux_weight": 0.06,
                "hard_event_extra": 0.10,
                "init_from_v300": True,
                "v300_init_model_name": v300_name,
                "lr": 2e-4,
                "min_lr": 1e-5,
                "weight_decay": 4e-4,
                "batch_size": 384,
                "max_epochs": 60,
                "patience": 10,
            },
        ),
    ]

    runs: List[object] = []
    for idx, (model_name, config) in enumerate(configs):
        prepared = copy.copy(prepared_base)
        prepared.curve_sample_multiplier = V304.build_curve_sample_multiplier(
            prepared_base.data.manifest,
            prepared_base.event_label_name,
            hard_event_extra=float(config["hard_event_extra"]),
        )
        print(
            f"[v316] 训练 {model_name} | 辅助={config['aux_weight']} | 调制={config['film_scale']} | 加权={config['hard_event_extra']}",
            flush=True,
        )
        run = V304.train_v304_candidate(model_name, config, prepared, device, seed=SEED + idx)
        runs.append(run)
        write_csv(run.training_history, TABLES / f"{model_name}_training_history.csv")
        torch.save(
            {
                "model_name": run.model_name,
                "state_dict": run.state_dict,
                "config": run.config,
                "roll_feature_names": prepared.roll_feature_names,
                "roll_impute_mean": prepared.roll_impute_mean,
                "roll_scale_mean": prepared.roll_scale_mean,
                "roll_scale_std": prepared.roll_scale_std,
                "class_names": prepared.class_names,
                "class_weight": prepared.class_weight,
                "best_epoch": run.best_epoch,
                "best_val_loss": run.best_val_loss,
                "training_seconds": run.training_seconds,
                "seed": SEED + idx,
            },
            MODELS / f"{model_name}.pt",
        )
        print(f"[v316] {model_name} 最优轮次={run.best_epoch} 最优验证损失={run.best_val_loss:.6f}", flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("[v316] 计算指标并只用过滤后验证集选模", flush=True)
    pred_by_model: Dict[str, np.ndarray] = {
        v300_name: pred_v300.astype(np.float32),
        old_v307_name: pred_v307_old.astype(np.float32),
    }
    for run in runs:
        pred_by_model[run.model_name] = run.pred_curve.astype(np.float32)

    metrics = V304.V238.compute_metrics_table(
        y_true_curve=y_true_curve,
        pred_by_model=pred_by_model,
        manifest=prepared_base.data.manifest,
        eval_modes=["original_remaining", "receding_2s_diagnostic"],
    )
    write_csv(metrics, TABLES / "v316_metrics_by_delay_and_bucket.csv")

    per_tables = []
    for model_name, pred_curve in pred_by_model.items():
        per = V304.V238.build_per_sample_metrics(
            y_true_curve=y_true_curve,
            pred_curve=pred_curve,
            manifest=prepared_base.data.manifest,
            model_name=model_name,
        )
        per_tables.append(per)
    per_sample = pd.concat(per_tables, ignore_index=True)
    per_sample = V304.V300.attach_v299_event_labels(per_sample, prepared_base.event_table)
    write_csv(per_sample, TABLES / "v316_per_sample_metrics_original_remaining.csv")

    delay0_summary = V304.V300.build_delay0_group_summary(per_sample)
    write_csv(delay0_summary, TABLES / "v316_delay0_group_summary.csv")

    selection = V304.build_selection_from_metrics(metrics, delay0_summary, runs, v300_name)
    write_csv(selection, TABLES / "v316_model_selection_validation.csv")
    selected_name = str(selection.iloc[0]["model_name"])

    event_metrics = V304.build_event_aux_metrics(prepared_base, runs)
    write_csv(event_metrics, TABLES / "v316_coarse_scene_aux_metrics.csv")

    original_remaining_valid, _ = V304.V238.build_original_remaining_mask(prepared_base.data.manifest)
    npz_payload = {
        "y_true_steering_delta": y_true_curve.astype(np.float32),
        "pred_v300_reference": pred_v300.astype(np.float32),
        "v300_reference_model": np.array([v300_name]),
        "pred_v307_previous": pred_v307_old.astype(np.float32),
        "v307_previous_model": np.array([old_v307_name]),
        "pred_v316_selected": pred_by_model[selected_name].astype(np.float32),
        "best_v316_model": np.array([selected_name]),
        "delay_ms": prepared_base.data.manifest["delay_ms"].astype(int).to_numpy(dtype=np.int32),
        "split": prepared_base.data.manifest["split"].astype(str).to_numpy(),
        "v315_original_split": prepared_base.data.manifest["v315_original_split"].astype(str).to_numpy(),
        "v315_current_window_train_keep": prepared_base.data.manifest["v315_current_window_train_keep"].astype(bool).to_numpy(),
        "event_uid": prepared_base.data.manifest["event_uid"].astype(str).to_numpy(),
        "subject": prepared_base.data.manifest["subject"].astype(str).to_numpy(),
        "future_grid_s": FUTURE_GRID.astype(np.float32),
        "original_remaining_valid": original_remaining_valid.astype(bool),
        "coarse_scene_label": prepared_base.event_label_name.astype(str),
        "coarse_scene_class_index": prepared_base.event_label.astype(np.int64),
        "coarse_scene_class_names": np.array(prepared_base.class_names),
    }
    for run in runs:
        npz_payload[f"pred_{run.model_name}"] = run.pred_curve.astype(np.float32)
        npz_payload[f"event_logits_{run.model_name}"] = run.event_logits.astype(np.float32)
    np.savez_compressed(OUT / "v316_filtered_current_window_predictions.npz", **npz_payload)

    with (MODELS / "v316_scalers_and_selection.pkl").open("wb") as f:
        pickle.dump(
            {
                "selection": selection.to_dict(orient="records"),
                "selected_name": selected_name,
                "v300_reference_model": v300_name,
                "v307_previous_model": old_v307_name,
                "roll_feature_names": prepared_base.roll_feature_names,
                "roll_impute_mean": prepared_base.roll_impute_mean,
                "roll_scale_mean": prepared_base.roll_scale_mean,
                "roll_scale_std": prepared_base.roll_scale_std,
                "class_names": prepared_base.class_names,
                "v300_guardrail": v300_guard,
            },
            f,
        )

    figure_paths = [
        V304.plot_training_history(runs),
        V304.plot_delay0_group_bars(delay0_summary, selected_name, v300_name),
        V304.plot_event_aux(event_metrics, selected_name),
    ]

    event_split_n = prepared_base.data.manifest.groupby("event_uid")["split"].nunique()
    event_delay_n = prepared_base.data.manifest.groupby("event_uid")["delay_ms"].nunique()
    selected_row = selection.iloc[0].to_dict()
    split_counts = prepared_base.data.manifest[prepared_base.data.manifest["delay_ms"].astype(int).eq(0)].groupby("split")["event_uid"].nunique()
    guardrail = {
        "pass": bool((event_split_n <= 1).all() and (event_delay_n == 6).all()),
        "version": "v316_filtered_current_window_coarse_scene_train_20260704",
        "training_run": True,
        "model_structure_changed_from_v307": False,
        "output_target_unchanged": "21_point_steering_delta_curve",
        "uses_v315_keep_manifest": True,
        "directly_reanchors_samples": False,
        "uses_test_error_as_features": False,
        "candidate_selection_uses_test": False,
        "same_event_never_repeated_across_splits": bool((event_split_n <= 1).all()),
        "event_in_multiple_splits_n": int((event_split_n > 1).sum()),
        "event_without_6_delay_rows_n": int((event_delay_n != 6).sum()),
        "event_n_total": int(prepared_base.data.manifest["event_uid"].nunique()),
        "event_n_kept_current_window": int(prepared_base.event_table["v315_current_window_train_keep"].astype(bool).sum()),
        "event_n_isolated_current_window": int((~prepared_base.event_table["v315_current_window_train_keep"].astype(bool)).sum()),
        "train_keep_event_n": int(split_counts.get("train", 0)),
        "val_keep_event_n": int(split_counts.get("val", 0)),
        "test_keep_event_n": int(split_counts.get("test", 0)),
        "v300_reference_model": v300_name,
        "v307_previous_model": old_v307_name,
        "selected_v316_model": selected_name,
        "selected_passes_v316_noharm_gate": bool(selected_row.get("passes_v304_noharm_gate", False)),
        "selected_val_all_delta_vs_v300": float(selected_row.get("delay0_val_all_delta_vs_v300", math.nan)),
        "selected_val_bad10_delta_vs_v300": float(selected_row.get("delay0_val_bad10_delta_vs_v300", math.nan)),
        "selected_val_bad20_delta_vs_v300": float(selected_row.get("delay0_val_bad20_delta_vs_v300", math.nan)),
        "device": str(device),
        "runtime_seconds": float(time.time() - start_time),
        "figure_paths": [str(p) for p in figure_paths],
    }
    write_json(guardrail, LOGS / "guardrail_check.json")
    report_path = write_report(selection, delay0_summary, event_metrics, guardrail, selected_name, v300_name, old_v307_name)
    guardrail["report_path"] = str(report_path)
    write_json(guardrail, LOGS / "guardrail_check.json")

    write_file_inventory()
    zip_path, zip_ok = make_zip_package()
    guardrail["zip_path"] = str(zip_path)
    guardrail["zip_testzip"] = bool(zip_ok)
    write_json(guardrail, LOGS / "guardrail_check.json")
    write_file_inventory()

    print("[v316] 完成", flush=True)
    print(json.dumps(guardrail, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
