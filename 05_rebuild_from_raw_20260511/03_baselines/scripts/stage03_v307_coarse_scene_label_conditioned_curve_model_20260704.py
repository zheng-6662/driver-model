#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v307 coarse scene-label conditioned curve model.

目的：
- 复用 v304 已验证的 fixed-label conditioned 曲线模型结构；
- 把 v304 的细 event_primary_type 条件输入替换为 v306 coarse_scene_label；
- 检查“下坡过弯 / 平路过弯 / 连续变道 / 紧急变道失稳”这种粗场景标签是否比 v300/v304 更适合当前任务；
- validation-only 选择候选，不用 test 做选模。

边界：
- v306 中过弯标签来自当前 scene_type，边界更接近实验条件；
- v306 中直道内连续/紧急子类仍部分来自 v305/v301 自动 seed，需要人工或实验条件确认；
- 因此 v307 是 coarse-scene seed 条件模型，不直接宣称最终部署模型。
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
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch


SEED = 20260704
ROOT = Path(__file__).resolve().parents[2]
BASELINES = ROOT / "03_baselines"
SCRIPTS = BASELINES / "scripts"
V304_SCRIPT = SCRIPTS / "stage03_v304_fixed_event_label_conditioned_curve_model_20260703.py"
V306_LABELS = BASELINES / "v306_coarse_predefined_scene_label_table_20260704" / "tables" / "v306_coarse_scene_event_labels.csv"

OUT = BASELINES / "v307_coarse_scene_label_conditioned_curve_model_20260704"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"
MODELS = OUT / "models"

COARSE_HARD_TYPES = {"continuous_lane_change", "emergency_lane_change_instability"}
COARSE_SCENE_ORDER = [
    "curve_downhill",
    "curve_flat",
    "continuous_lane_change",
    "emergency_lane_change_instability",
    "other_or_uncertain",
]


def import_module_from_path(module_name: str, path: Path):
    """按路径导入 v304 脚本，复用其中已经跑通的模型和评估代码。"""

    if not path.exists():
        raise FileNotFoundError(f"缺少依赖脚本：{path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法导入脚本：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V304 = import_module_from_path("stage03_v304_for_v307_coarse_scene", V304_SCRIPT)
FUTURE_GRID = V304.FUTURE_GRID


def patch_v304_output_globals() -> None:
    """让复用的 v304 helper 写入 v307 输出目录。"""

    V304.SEED = SEED
    V304.OUT = OUT
    V304.TABLES = TABLES
    V304.FIGURES = FIGURES
    V304.REPORTS = REPORTS
    V304.LOGS = LOGS
    V304.MODELS = MODELS
    V304.HARD_EVENT_TYPES = COARSE_HARD_TYPES


def ensure_dirs() -> None:
    """创建 v307 输出目录。"""

    for folder in (TABLES, FIGURES, REPORTS, LOGS, MODELS):
        folder.mkdir(parents=True, exist_ok=True)


def clean_out_dir() -> None:
    """只清理 v307 自己的输出。"""

    if OUT.exists():
        shutil.rmtree(OUT)
    ensure_dirs()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """CSV 使用 utf-8-sig，方便 Windows/Excel 直接查看中文。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_json(payload: Dict[str, object], path: Path) -> None:
    """保存 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def file_sha256(path: Path) -> str:
    """计算文件 sha256。"""

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


def load_coarse_scene_labels_for_all_samples(manifest: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    """把 v306 coarse_scene_label 映射到同一 event 的全部 rolling delay 行。"""

    if not V306_LABELS.exists():
        raise FileNotFoundError(f"缺少 v306 粗场景标签表：{V306_LABELS}")
    labels = pd.read_csv(V306_LABELS, encoding="utf-8-sig")
    if labels["event_uid"].duplicated().any():
        dup = labels.loc[labels["event_uid"].duplicated(), "event_uid"].head(5).tolist()
        raise AssertionError(f"v306 标签表 event_uid 重复：{dup}")
    label_map = labels.set_index("event_uid")["coarse_scene_label"].astype(str)
    names = manifest["event_uid"].astype(str).map(label_map).astype(str).to_numpy()
    if pd.isna(names).any() or (pd.Series(names).eq("nan")).any():
        missing = manifest.loc[pd.Series(names).eq("nan").to_numpy(), "event_uid"].drop_duplicates().head(10).tolist()
        raise AssertionError(f"存在无法映射 v306 coarse_scene_label 的 rolling 样本：{missing}")
    present = pd.Series(names).unique().tolist()
    class_names = [name for name in COARSE_SCENE_ORDER if name in present]
    for name in sorted(set(present) - set(class_names)):
        class_names.append(name)
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    y = np.array([class_to_idx[name] for name in names], dtype=np.int64)
    return y, names, class_names, labels


def prepare_v307_data(hard_event_extra: float):
    """读取数据、构造 roll-cause summary、映射 v306 粗场景标签。"""

    raw_data = V304.V238.load_v236_data()
    data, event_table = V304.V300.apply_v299_within_subject_split(raw_data)
    no_subject = V304.V300.prepare_variant(
        "no_subject",
        data,
        {
            "uses_subject_onehot": False,
            "description_cn": "v307 主线只使用车辆/道路/phase，不拼 subject one-hot；额外使用 v306 coarse_scene_label 条件输入",
        },
    )

    x_base = V304.V238.build_base_design_matrix(data)
    roll_raw, roll_feature_names, signal_audit = V304.V302.build_roll_cause_summary(x_base, data.feature_names)
    train_mask = data.manifest["split"].astype(str).to_numpy() == "train"
    roll_scaled, impute_mean, scale_mean, scale_std = V304.fit_transform_roll_features(roll_raw, train_mask)
    event_label, event_label_name, class_names, labels = load_coarse_scene_labels_for_all_samples(data.manifest)

    train_labels = event_label[train_mask]
    counts = np.bincount(train_labels, minlength=len(class_names)).astype(np.float32)
    counts[counts < 1] = 1.0
    class_weight = counts.sum() / (len(class_names) * counts)
    class_weight = np.clip(class_weight, 0.35, 4.0).astype(np.float32)
    curve_mult = V304.build_curve_sample_multiplier(data.manifest, event_label_name, hard_event_extra)

    write_csv(signal_audit, TABLES / "v307_roll_cause_signal_coverage.csv")
    write_csv(
        pd.DataFrame(
            [
                {
                    "roll_cause_feature_n": int(roll_scaled.shape[1]),
                    "raw_nan_rate": float(np.mean(~np.isfinite(roll_raw))),
                    "scaled_nan_rate": float(np.mean(~np.isfinite(roll_scaled))),
                    "coarse_scene_class_n": int(len(class_names)),
                    "hard_event_extra": float(hard_event_extra),
                }
            ]
        ),
        TABLES / "v307_input_audit.csv",
    )
    write_csv(
        pd.DataFrame({"coarse_scene_label": class_names, "class_index": list(range(len(class_names))), "class_weight": class_weight}),
        TABLES / "v307_coarse_scene_class_mapping.csv",
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


def metric_row(summary: pd.DataFrame, model_name: str, split: str, group: str) -> pd.Series | None:
    """从 delay0 group summary 里取一行。"""

    one = summary[
        summary["model_name"].astype(str).eq(model_name)
        & summary["split"].astype(str).eq(split)
        & summary["group"].astype(str).eq(group)
    ]
    if one.empty:
        return None
    return one.iloc[0]


def write_report(
    selection: pd.DataFrame,
    delay0_summary: pd.DataFrame,
    event_metrics: pd.DataFrame,
    guardrail: Dict[str, object],
    selected_name: str,
    v300_name: str,
) -> Path:
    """写 v307 中文报告。"""

    path = REPORTS / "v307_coarse_scene_label_conditioned_curve_model_cn.md"

    def group_line(model_name: str, split: str, group: str) -> str:
        row = metric_row(delay0_summary, model_name, split, group)
        if row is None:
            return "NA"
        return f"{float(row['sample_rmse_mean']):.4f}"

    selected_rows = delay0_summary[
        delay0_summary["model_name"].isin([v300_name, selected_name])
        & delay0_summary["split"].eq("test")
        & delay0_summary["group"].isin(["all", "within_bad_top10", "within_bad_top20", "strong_steer", "vehicle_ambiguous"])
    ][["model_name", "split", "group", "n", "sample_rmse_mean", "sample_rmse_median", "sample_rmse_p90"]]
    selected_event = event_metrics[event_metrics["model_name"].eq(selected_name)]
    lines = [
        "# v307 coarse scene-label conditioned 曲线模型",
        "",
        "## 这一步做了什么",
        "",
        "v307 复用 v304 的 fixed-label conditioned 模型结构，但把条件标签从 v301/v305 的细事件类型替换为 v306 的粗场景标签。",
        "",
        "粗标签包括：下坡过弯、平路过弯、连续变道/连续左右修正、紧急变道/猛打方向失稳、其他/不确定。",
        "",
        "## validation-only 选择",
        "",
        selection.to_markdown(index=False),
        "",
        f"validation 选择出的 v307 候选：`{selected_name}`。",
        f"v300 参照模型：`{v300_name}`。",
        "",
        "## test delay0 对比",
        "",
        selected_rows.to_markdown(index=False),
        "",
        "简表：",
        "",
        f"- test/all：v300 `{group_line(v300_name, 'test', 'all')}` -> v307 `{group_line(selected_name, 'test', 'all')}`。",
        f"- test/within_bad_top10：v300 `{group_line(v300_name, 'test', 'within_bad_top10')}` -> v307 `{group_line(selected_name, 'test', 'within_bad_top10')}`。",
        f"- test/within_bad_top20：v300 `{group_line(v300_name, 'test', 'within_bad_top20')}` -> v307 `{group_line(selected_name, 'test', 'within_bad_top20')}`。",
        "",
        "## 粗场景辅助头",
        "",
        selected_event.to_markdown(index=False),
        "",
        "## 当前判断",
        "",
        "- 如果 v307 接近或优于 v304，说明粗场景标签已经保留了主要条件信息，后续人工审核成本可明显降低。",
        "- 如果 v307 明显弱于 v304，说明急左/急右/复合制动等细粒度信息仍有价值，需要在粗场景内保留少量二级标签。",
        "- v307 中直道内连续/紧急子类仍有 v305/v301 seed 成分，不能直接写成最终人工标签。",
        "",
        "## guardrail",
        "",
        "```json",
        json.dumps(guardrail, ensure_ascii=False, indent=2),
        "```",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_file_inventory() -> pd.DataFrame:
    """记录输出文件清单。"""

    rows = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            rows.append({"relative_path": str(path.relative_to(OUT)), "size_bytes": path.stat().st_size, "sha256": file_sha256(path)})
    inv = pd.DataFrame(rows)
    write_csv(inv, LOGS / "file_inventory.csv")
    return inv


def make_zip_package() -> Tuple[Path, bool]:
    """打包并校验 v307 产物。"""

    zip_path = OUT / "v307_coarse_scene_label_conditioned_curve_model_20260704.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(OUT.rglob("*")):
            if path == zip_path or not path.is_file():
                continue
            zf.write(path, path.relative_to(OUT))
    with zipfile.ZipFile(zip_path, "r") as zf:
        ok = zf.testzip() is None
    return zip_path, ok


def main() -> None:
    start_time = time.time()
    patch_v304_output_globals()
    clean_out_dir()
    torch.set_num_threads(1)
    set_seed(SEED)

    print("[v307] 读取数据并构造 coarse scene-label conditioned 输入")
    prepared_base = prepare_v307_data(hard_event_extra=0.0)
    split_audit = V304.V300.build_split_audit(prepared_base.data.manifest, prepared_base.event_table)
    write_csv(split_audit, TABLES / "v307_within_subject_split_audit.csv")

    y_true_curve = prepared_base.data.y_future[:, :, 0].astype(np.float32)
    pred_v300, v300_name, v300_guard = V304.load_v300_prediction_all(prepared_base.data.manifest)

    input_hashes = pd.DataFrame(
        [
            {"input_name": "v304_script_reused", "path": str(V304_SCRIPT), "sha256": file_sha256(V304_SCRIPT)},
            {"input_name": "v306_coarse_scene_labels", "path": str(V306_LABELS), "sha256": file_sha256(V306_LABELS)},
            {"input_name": "v300_predictions", "path": str(V304.V300_PRED), "sha256": file_sha256(V304.V300_PRED)},
        ]
    )
    write_csv(input_hashes, LOGS / "input_hashes.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[v307] 使用设备：{device}")

    configs: List[Tuple[str, Dict[str, object]]] = [
        (
            "v307_coarse_scene_init_aux003_film005_h64",
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
            "v307_coarse_scene_init_aux005_film010_h64",
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
            "v307_coarse_scene_init_aux006_film010_hard110_h64",
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
            f"[v307] training {model_name} | aux={config['aux_weight']} | film={config['film_scale']} | hard_extra={config['hard_event_extra']}"
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
        print(f"[v307] {model_name} best_epoch={run.best_epoch} best_val_loss={run.best_val_loss:.6f}")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("[v307] 计算指标与 validation-only 选择")
    pred_by_model: Dict[str, np.ndarray] = {v300_name: pred_v300.astype(np.float32)}
    for run in runs:
        pred_by_model[run.model_name] = run.pred_curve.astype(np.float32)

    metrics = V304.V238.compute_metrics_table(
        y_true_curve=y_true_curve,
        pred_by_model=pred_by_model,
        manifest=prepared_base.data.manifest,
        eval_modes=["original_remaining", "receding_2s_diagnostic"],
    )
    write_csv(metrics, TABLES / "v307_metrics_by_delay_and_bucket.csv")

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
    write_csv(per_sample, TABLES / "v307_per_sample_metrics_original_remaining.csv")

    delay0_summary = V304.V300.build_delay0_group_summary(per_sample)
    write_csv(delay0_summary, TABLES / "v307_delay0_group_summary.csv")

    selection = V304.build_selection_from_metrics(metrics, delay0_summary, runs, v300_name)
    write_csv(selection, TABLES / "v307_model_selection_validation.csv")
    selected_name = str(selection.iloc[0]["model_name"])

    event_metrics = V304.build_event_aux_metrics(prepared_base, runs)
    write_csv(event_metrics, TABLES / "v307_coarse_scene_aux_metrics.csv")

    print("[v307] 保存预测数组和图像")
    original_remaining_valid, _ = V304.V238.build_original_remaining_mask(prepared_base.data.manifest)
    npz_payload = {
        "y_true_steering_delta": y_true_curve.astype(np.float32),
        "pred_v300_reference": pred_v300.astype(np.float32),
        "v300_reference_model": np.array([v300_name]),
        "pred_v307_selected": pred_by_model[selected_name].astype(np.float32),
        "best_v307_model": np.array([selected_name]),
        "delay_ms": prepared_base.data.manifest["delay_ms"].astype(int).to_numpy(dtype=np.int32),
        "split": prepared_base.data.manifest["split"].astype(str).to_numpy(),
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
    np.savez_compressed(OUT / "v307_coarse_scene_label_conditioned_predictions.npz", **npz_payload)

    with (MODELS / "v307_scalers_and_selection.pkl").open("wb") as f:
        pickle.dump(
            {
                "selection": selection.to_dict(orient="records"),
                "selected_name": selected_name,
                "v300_reference_model": v300_name,
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
    labels_table = prepared_base.labels_table
    guardrail = {
        "pass": bool((event_split_n <= 1).all() and (event_delay_n == 6).all()),
        "version": "v307_coarse_scene_label_conditioned_curve_model_20260704",
        "model_structure_changed": True,
        "output_target_unchanged": "21_point_steering_delta_curve",
        "uses_roll_cause_summary_as_input": True,
        "uses_coarse_scene_labels_as_features": True,
        "coarse_scene_label_source": str(V306_LABELS),
        "curve_scene_labels_from_current_scene_type": True,
        "noncurve_subtypes_require_manual_or_experiment_confirmation": True,
        "uses_future_behavior_seed_for_some_noncurve_subtypes": bool(
            labels_table["uses_future_behavior_seed_for_noncurve_subtype"].astype(bool).any()
            if "uses_future_behavior_seed_for_noncurve_subtype" in labels_table.columns
            else True
        ),
        "deployable_without_noncurve_manual_confirmation": False,
        "uses_test_error_as_features": False,
        "candidate_selection_uses_test": False,
        "same_event_never_repeated_across_splits": bool((event_split_n <= 1).all()),
        "event_in_multiple_splits_n": int((event_split_n > 1).sum()),
        "event_without_6_delay_rows_n": int((event_delay_n != 6).sum()),
        "event_n": int(prepared_base.data.manifest["event_uid"].nunique()),
        "rolling_sample_n": int(len(prepared_base.data.manifest)),
        "roll_cause_feature_n": int(prepared_base.roll_scaled.shape[1]),
        "coarse_scene_class_n": int(len(prepared_base.class_names)),
        "coarse_scene_class_names": prepared_base.class_names,
        "v300_reference_model": v300_name,
        "selected_v307_model": selected_name,
        "selected_passes_v307_noharm_gate": bool(selected_row.get("passes_v304_noharm_gate", False)),
        "selected_val_all_delta_vs_v300": float(selected_row.get("delay0_val_all_delta_vs_v300", math.nan)),
        "selected_val_bad10_delta_vs_v300": float(selected_row.get("delay0_val_bad10_delta_vs_v300", math.nan)),
        "selected_val_bad20_delta_vs_v300": float(selected_row.get("delay0_val_bad20_delta_vs_v300", math.nan)),
        "device": str(device),
        "runtime_seconds": float(time.time() - start_time),
        "figure_paths": [str(p) for p in figure_paths],
    }
    write_json(guardrail, LOGS / "guardrail_check.json")
    report_path = write_report(selection, delay0_summary, event_metrics, guardrail, selected_name, v300_name)
    guardrail["report_path"] = str(report_path)
    write_json(guardrail, LOGS / "guardrail_check.json")

    write_file_inventory()
    zip_path, zip_ok = make_zip_package()
    guardrail["zip_path"] = str(zip_path)
    guardrail["zip_testzip"] = bool(zip_ok)
    write_json(guardrail, LOGS / "guardrail_check.json")
    write_file_inventory()

    print("[v307] 完成")
    print(json.dumps(guardrail, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
