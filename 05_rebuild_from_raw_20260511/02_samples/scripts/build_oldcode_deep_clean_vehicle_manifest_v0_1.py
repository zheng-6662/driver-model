# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_stage2_samples import FS, fill_nan_linear, interp_to_grid, to_seconds  # noqa: E402


PROJECT_ROOT = Path(r"F:/data_set_process/data_process")
REBUILD_ROOT = PROJECT_ROOT / "05_rebuild_from_raw_20260511"
DATASET_DIR = REBUILD_ROOT / "03_processed_datasets" / "vehicle_instability_allraw_highconf_v0_1"
TABLE_DIR = DATASET_DIR / "tables"
CLEAN_ROOT = DATASET_DIR / "oldcode_deep_vehicle_csv_v0_1"
LOG_DIR = DATASET_DIR / "logs"
REPORT_DIR = REBUILD_ROOT / "09_reports"

SOURCE_MANIFESTS = {
    "random_event_split": TABLE_DIR / "oldcode_manifest_random_event_split.csv",
    "session_level_split": TABLE_DIR / "oldcode_manifest_session_level_split.csv",
    "subject_level_split": TABLE_DIR / "oldcode_manifest_subject_level_split.csv",
}

OLD_REQUIRED_OUTPUTS = [
    "zx|roll",
    "zx|SteeringWheel",
    "zx|vyaw",
    "zx|vx",
    "zx|z",
    "zx|ay",
    "zx|ax",
    "zx1|lanecurvatureXY",
    "zx|yaw",
]

OPTIONAL_OUTPUTS: list[str] = []

INPUT_ALIASES = {
    "zx|roll": ["zx|roll"],
    "zx|SteeringWheel": ["zx|SteeringWheel"],
    "zx|vyaw": ["zx|vyaw"],
    "zx|vx": ["zx|vx"],
    "zx|z": ["zx|z"],
    "zx|ay": ["zx|ay"],
    "zx|ax": ["zx|ax"],
    "zx1|lanecurvatureXY": ["zx1|lanecurvatureXY", "zx|lanecurvatureXY"],
    "zx|yaw": ["zx|yaw"],
    "lateraldistance": ["lateraldistance", "zx1|lateraldistance", "zx|lateraldistance"],
    "road_type_fixed": ["road_type_fixed", "road_type", "roadType_fixed"],
    "ref_nn_ok": ["ref_nn_ok", "ref_ok", "refnn_ok"],
}


def ensure_dirs() -> None:
    for path in [CLEAN_ROOT, TABLE_DIR, LOG_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def choose_existing(columns: set[str], aliases: list[str]) -> str | None:
    for name in aliases:
        if name in columns:
            return name
    return None


def read_numeric_interpolated(df: pd.DataFrame, source_col: str | None, x_rel: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if source_col is None or source_col not in df.columns:
        return np.zeros_like(grid, dtype=np.float64)
    values = pd.to_numeric(df[source_col], errors="coerce").to_numpy(dtype=np.float64)
    interpolated = interp_to_grid(x_rel, values, grid)
    return fill_nan_linear(interpolated).astype(np.float64)


def build_storage_time_like(raw_storage: pd.Series, grid: np.ndarray) -> pd.Series:
    parsed = pd.to_datetime(raw_storage, errors="coerce")
    if parsed.notna().any():
        base = parsed[parsed.notna()].iloc[0]
        times = base + pd.to_timedelta(grid, unit="s")
        return pd.Series(times.astype(str))
    return pd.Series([f"{float(t):.6f}" for t in grid])


def clean_one_vehicle(raw_path: Path, subject: str) -> dict[str, Any]:
    clean_path = CLEAN_ROOT / subject / raw_path.name
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    needed_inputs = {"StorageTime"}
    for aliases in INPUT_ALIASES.values():
        needed_inputs.update(aliases)
    try:
        df = pd.read_csv(raw_path, usecols=lambda c: c in needed_inputs)
        if "StorageTime" not in df.columns:
            raise ValueError("missing StorageTime")
        t_abs = to_seconds(df["StorageTime"])
        finite = np.isfinite(t_abs)
        if finite.sum() < 2:
            raise ValueError("not enough valid StorageTime rows")
        t0 = float(np.nanmin(t_abs))
        duration = float(np.nanmax(t_abs) - t0)
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("non-positive duration")
        grid = np.arange(0.0, duration + 0.5 / FS, 1.0 / FS, dtype=np.float64)
        x_rel = t_abs - t0
        columns = set(df.columns)
        out = pd.DataFrame({"StorageTime": build_storage_time_like(df["StorageTime"], grid)})
        source_map: dict[str, str | None] = {}
        for out_col in OLD_REQUIRED_OUTPUTS + OPTIONAL_OUTPUTS:
            src = choose_existing(columns, INPUT_ALIASES[out_col])
            source_map[out_col] = src
            out[out_col] = read_numeric_interpolated(df, src, x_rel, grid).astype(np.float32)
        out.to_csv(clean_path, index=False, encoding="utf-8-sig")
        return {
            "raw_vehicle_file": str(raw_path),
            "clean_vehicle_file": str(clean_path),
            "subject": subject,
            "status": "ok",
            "rows": int(len(out)),
            "duration_s": float(duration),
            "source_map": source_map,
            "missing_required_outputs": [col for col in OLD_REQUIRED_OUTPUTS if source_map.get(col) is None],
            "nan_after_clean": int(out[OLD_REQUIRED_OUTPUTS].isna().sum().sum()),
        }
    except Exception as exc:
        return {
            "raw_vehicle_file": str(raw_path),
            "clean_vehicle_file": str(clean_path),
            "subject": subject,
            "status": "error",
            "error": repr(exc),
        }


def build_clean_manifests(status_df: pd.DataFrame) -> dict[str, Any]:
    ok_map = {
        str(row.raw_vehicle_file): str(row.clean_vehicle_file)
        for row in status_df.itertuples(index=False)
        if str(row.status) == "ok"
    }
    manifest_outputs: dict[str, Any] = {}
    for split_name, src_path in SOURCE_MANIFESTS.items():
        df = pd.read_csv(src_path)
        out = df.copy()
        out["raw_vehicle_file_before_cleaning"] = out["vehicle_file"].astype(str)
        out["vehicle_file"] = out["vehicle_file"].astype(str).map(ok_map)
        missing = out["vehicle_file"].isna()
        if missing.any():
            bad = out.loc[missing, "raw_vehicle_file_before_cleaning"].drop_duplicates().tolist()
            raise RuntimeError(f"clean vehicle file missing for {len(bad)} source files: {bad[:3]}")
        out["protocol_version"] = out["protocol_version"].astype(str) + "_clean_vehicle_v0_1"
        out_path = TABLE_DIR / f"oldcode_manifest_{split_name}_clean_vehicle_v0_1.csv"
        out.to_csv(out_path, index=False, encoding="utf-8-sig")
        manifest_outputs[split_name] = {
            "path": str(out_path),
            "rows": int(len(out)),
            "split_counts": out["split"].value_counts().to_dict(),
            "unique_vehicle_files": int(out["vehicle_file"].nunique()),
        }
    return manifest_outputs


def verify_against_npz(clean_manifest_path: Path) -> dict[str, Any]:
    old_train_dir = PROJECT_ROOT / "02_code" / "final_code" / "model" / "training"
    if str(old_train_dir) not in sys.path:
        sys.path.insert(0, str(old_train_dir))
    from run_event_conditioned_trajectory_baseline import build_sample_bundle_from_manifest  # noqa: E402

    _, y_pool, _, _, _, meta_df, dropped = build_sample_bundle_from_manifest(
        manifest_path=clean_manifest_path,
        max_train_samples=None,
        max_val_samples=None,
        max_test_samples=None,
        seed=2026,
    )
    idx_df = pd.read_csv(TABLE_DIR / "sample_index_pre2_label2_old_main.csv")
    z = np.load(DATASET_DIR / "arrays" / "pre2_label2_old_main.npz", allow_pickle=True)
    npz_by_uid = {
        str(row.event_uid): int(row.array_row)
        for row in idx_df[["event_uid", "array_row"]].itertuples(index=False)
    }
    diffs: list[float] = []
    checked = 0
    for i, row in meta_df.iterrows():
        uid = str(row.get("instability_event_uid", ""))
        if uid not in npz_by_uid:
            continue
        j = npz_by_uid[uid]
        a = y_pool[i, :, 0].astype(np.float32)
        # The deep old loader predicts anchor_idx+1 ... anchor_idx+400,
        # while the diagnostic npz keeps t=0.000 ... 2.000 as 401 points.
        b = z["label_steer_delta"][j, 1:401].astype(np.float32)
        diffs.append(float(np.nanmax(np.abs(a - b))))
        checked += 1
        if checked >= 50:
            break
    return {
        "manifest": str(clean_manifest_path),
        "dropped": int(dropped),
        "checked_samples": int(checked),
        "max_abs_diff_vs_npz_first50": float(max(diffs) if diffs else np.nan),
        "mean_abs_diff_vs_npz_first50": float(np.mean(diffs) if diffs else np.nan),
    }


def write_report(summary: dict[str, Any]) -> None:
    status_table = pd.DataFrame(summary["vehicle_clean_status"])
    report = f"""# 旧深度入口车辆 CSV 清洗 manifest v0.1

生成时间：2026-05-12

## 为什么补这一步

旧 `vehicle_direct` 深度入口直接读取 `vehicle_file`，并在旧代码里把 CSV 中的缺失值直接填成 0。当前原始车辆 CSV 存在大量交替缺失点，如果直接读原始 CSV，会把方向盘标签变成高频 0 跳变，固定图会出现不真实的黑色填充块。因此本步骤把原始车辆文件先插值成旧深度入口可读的 200Hz CSV，再生成新的 clean manifest。

## 输入

- 原始旧 manifest：`{TABLE_DIR.as_posix()}/oldcode_manifest_*_split.csv`
- 原始车辆 CSV：manifest 中记录的 `vehicle_file`

## 输出

- 清洗车辆 CSV：`{CLEAN_ROOT.as_posix()}`
- clean manifest：`{TABLE_DIR.as_posix()}/oldcode_manifest_*_clean_vehicle_v0_1.csv`
- 状态表：`{(TABLE_DIR / 'oldcode_deep_clean_vehicle_status_v0_1.csv').as_posix()}`

## 清洗结果

- 原始车辆文件数：{summary['unique_raw_vehicle_files']}
- 清洗成功文件数：{summary['clean_ok_files']}
- 清洗失败文件数：{summary['clean_error_files']}
- session-level clean manifest 行数：{summary['manifest_outputs']['session_level_split']['rows']}
- session-level split：{summary['manifest_outputs']['session_level_split']['split_counts']}

## 与旧 `.npz` 标签一致性检查

{json.dumps(summary['npz_consistency_check'], ensure_ascii=False, indent=2)}

如果最大差异接近 0，说明旧深度入口读取 clean manifest 得到的 2 秒方向盘标签，已经和我们前面插值生成的 `pre2_label2_old_main.npz` 标签一致。

## 结论

后续旧 `vehicle_direct` 全量对照必须使用 clean manifest；此前直接用原始 CSV 的 full run 只能作为失败诊断，不能作为模型结果引用。
"""
    (REPORT_DIR / "oldcode_deep_clean_vehicle_manifest_v0_1_cn.md").write_text(report, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    base_manifest = pd.read_csv(SOURCE_MANIFESTS["session_level_split"])
    unique_files = (
        base_manifest[["subj", "vehicle_file"]]
        .drop_duplicates()
        .sort_values(["subj", "vehicle_file"])
        .reset_index(drop=True)
    )
    status_rows = [
        clean_one_vehicle(Path(str(row.vehicle_file)), subject=str(row.subj))
        for row in unique_files.itertuples(index=False)
    ]
    status_df = pd.DataFrame(status_rows)
    status_path = TABLE_DIR / "oldcode_deep_clean_vehicle_status_v0_1.csv"
    status_df.to_csv(status_path, index=False, encoding="utf-8-sig")
    manifest_outputs = build_clean_manifests(status_df)
    check = verify_against_npz(Path(manifest_outputs["session_level_split"]["path"]))
    summary = {
        "unique_raw_vehicle_files": int(len(unique_files)),
        "clean_ok_files": int((status_df["status"] == "ok").sum()),
        "clean_error_files": int((status_df["status"] != "ok").sum()),
        "vehicle_clean_status_path": str(status_path),
        "vehicle_clean_status": status_df.to_dict(orient="records"),
        "manifest_outputs": manifest_outputs,
        "npz_consistency_check": check,
    }
    summary_path = LOG_DIR / "oldcode_deep_clean_vehicle_manifest_summary_v0_1.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "vehicle_clean_status"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
