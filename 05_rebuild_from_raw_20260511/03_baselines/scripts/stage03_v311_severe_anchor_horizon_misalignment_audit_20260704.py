#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v311 severe anchor / horizon misalignment audit.

目的：
- 解释 v309 severe 样本为什么很难靠 v310 的 loss/权重修好；
- 检查严重错例中是否存在“真实动作主要发生在 2s 之后，但标签或模型把它提前到 0-2s”的窗口错位；
- 给下一轮修改提供明确方向：如果错位成立，应优先改标签/锚点/预测窗口，而不是继续调网络 loss。

边界：
- 本脚本只做审计，不训练模型；
- 使用 v309 图册已经读取过的 raw vehicle CSV；
- v309 severe 表只作为诊断集合，不用于训练或选模。
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import zipfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASELINES = ROOT / "03_baselines"
V309_DIR = BASELINES / "v309_recent_best_prediction_effect_gallery_20260704"
V309_TEST_TABLE = V309_DIR / "tables" / "v309_test_delay0_prediction_effect_table.csv"
V309_SEVERE = V309_DIR / "tables" / "v309_severe_direction_or_intent_errors.csv"

OUT = BASELINES / "v311_severe_anchor_horizon_misalignment_audit_20260704"
TABLES = OUT / "tables"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"


def ensure_dirs() -> None:
    """创建输出目录。"""

    for folder in (TABLES, REPORTS, LOGS):
        folder.mkdir(parents=True, exist_ok=True)


def clean_out_dir() -> None:
    """只清理 v311 自己的输出。"""

    if OUT.exists():
        import shutil

        shutil.rmtree(OUT)
    ensure_dirs()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """CSV 使用 utf-8-sig，方便 Windows/Excel 查看中文。"""

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


def signed_peak(values: np.ndarray) -> float:
    """返回绝对值最大的有符号峰值。"""

    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return float("nan")
    valid = arr[finite]
    return float(valid[int(np.argmax(np.abs(valid)))])


def sign_label(value: float, eps: float = 0.05) -> str:
    """把连续值转成方向标签。"""

    if not np.isfinite(value):
        return "NA"
    if value > eps:
        return "+"
    if value < -eps:
        return "-"
    return "0"


def load_raw_window(raw_path: str, observation_s: float, end_s: float = 6.0) -> pd.DataFrame:
    """读取锚点附近 raw vehicle 数据。"""

    path = Path(str(raw_path))
    if not path.exists() or not np.isfinite(float(observation_s)):
        return pd.DataFrame()
    needed = ["StorageTime", "zx|SteeringWheel", "zx|ay", "zx|vyaw", "zx|roll"]
    try:
        df = pd.read_csv(path, usecols=lambda c: c in needed)
    except Exception:
        return pd.DataFrame()
    if df.empty or "StorageTime" not in df.columns:
        return pd.DataFrame()
    t = pd.to_datetime(df["StorageTime"], errors="coerce")
    if t.isna().all():
        return pd.DataFrame()
    rel_record = (t - t.iloc[0]).dt.total_seconds().to_numpy(dtype=float)
    rel_anchor = rel_record - float(observation_s)
    keep = (rel_anchor >= -3.2) & (rel_anchor <= end_s)
    out = df.loc[keep].copy()
    out["rel_anchor_s"] = rel_anchor[keep]
    return out.reset_index(drop=True)


def raw_peak_features(row: pd.Series) -> Dict[str, object]:
    """从 raw 后续曲线提取 0-2s 与 2-6s 的峰值对比。"""

    raw = load_raw_window(str(row.get("raw_vehicle_csv", "")), float(row.get("observation_s", float("nan"))))
    if raw.empty or "zx|SteeringWheel" not in raw.columns:
        return {
            "raw_available": False,
            "raw_0_2_peak": math.nan,
            "raw_2_6_peak": math.nan,
            "raw_2_6_peak_abs": math.nan,
            "raw_2_6_peak_time_s": math.nan,
            "raw_2_6_ay_peak_abs": math.nan,
            "raw_2_6_yaw_peak_abs": math.nan,
            "raw_2_6_roll_peak_abs": math.nan,
        }

    t = raw["rel_anchor_s"].to_numpy(dtype=float)
    steer = raw["zx|SteeringWheel"].to_numpy(dtype=float)
    finite = np.isfinite(t) & np.isfinite(steer)
    if not finite.any():
        return {"raw_available": False}
    anchor_idx = int(np.argmin(np.abs(t[finite])))
    finite_indices = np.where(finite)[0]
    anchor_value = float(steer[finite_indices[anchor_idx]])
    steer_delta = steer - anchor_value
    win_0_2 = (t >= 0.0) & (t <= 2.0)
    win_2_6 = (t > 2.0) & (t <= 6.0)
    peak_0_2 = signed_peak(steer_delta[win_0_2])
    peak_2_6 = signed_peak(steer_delta[win_2_6])
    if win_2_6.any() and np.isfinite(peak_2_6):
        post_vals = steer_delta[win_2_6]
        post_t = t[win_2_6]
        post_finite = np.isfinite(post_vals)
        peak_t = float(post_t[post_finite][int(np.argmax(np.abs(post_vals[post_finite])))]) if post_finite.any() else math.nan
    else:
        peak_t = math.nan

    def abs_peak_col(col: str) -> float:
        if col not in raw.columns or not win_2_6.any():
            return math.nan
        vals = raw.loc[win_2_6, col].to_numpy(dtype=float)
        finite_vals = vals[np.isfinite(vals)]
        if finite_vals.size == 0:
            return math.nan
        return float(np.nanmax(np.abs(finite_vals)))

    return {
        "raw_available": True,
        "raw_0_2_peak": float(peak_0_2),
        "raw_0_2_peak_abs": float(abs(peak_0_2)) if np.isfinite(peak_0_2) else math.nan,
        "raw_2_6_peak": float(peak_2_6),
        "raw_2_6_peak_abs": float(abs(peak_2_6)) if np.isfinite(peak_2_6) else math.nan,
        "raw_2_6_peak_time_s": peak_t,
        "raw_2_6_ay_peak_abs": abs_peak_col("zx|ay"),
        "raw_2_6_yaw_peak_abs": abs_peak_col("zx|vyaw"),
        "raw_2_6_roll_peak_abs": abs_peak_col("zx|roll"),
    }


def classify_misalignment(row: pd.Series) -> Dict[str, object]:
    """按 0-2s 与 2-6s 峰值关系给严重错例打审计标签。"""

    horizon_peak = float(row.get("true_peak", math.nan))
    v307_peak = float(row.get("v307_peak", math.nan))
    post_peak = float(row.get("raw_2_6_peak", math.nan))
    horizon_abs = abs(horizon_peak) if np.isfinite(horizon_peak) else math.nan
    post_abs = abs(post_peak) if np.isfinite(post_peak) else math.nan

    horizon_flat_post_large = bool(np.isfinite(horizon_abs) and np.isfinite(post_abs) and horizon_abs < 0.40 and post_abs >= 1.00)
    post2_dominant = bool(
        np.isfinite(horizon_abs)
        and np.isfinite(post_abs)
        and post_abs >= max(1.00, 1.50 * max(horizon_abs, 0.10))
    )
    horizon_post_opposite = bool(
        np.isfinite(horizon_peak)
        and np.isfinite(post_peak)
        and horizon_abs >= 0.40
        and post_abs >= 0.60
        and sign_label(horizon_peak) != sign_label(post_peak)
    )
    model_follows_post_not_horizon = bool(
        np.isfinite(v307_peak)
        and np.isfinite(post_peak)
        and np.isfinite(horizon_peak)
        and abs(v307_peak) >= 0.40
        and sign_label(v307_peak) == sign_label(post_peak)
        and sign_label(v307_peak) != sign_label(horizon_peak)
    )
    predicts_future_too_early = bool((post2_dominant or horizon_flat_post_large or horizon_post_opposite) and model_follows_post_not_horizon)
    label_horizon_mismatch_suspected = bool(post2_dominant or horizon_flat_post_large or horizon_post_opposite)

    tags: List[str] = []
    if horizon_flat_post_large:
        tags.append("horizon_flat_post_large")
    if post2_dominant:
        tags.append("post2_dominant")
    if horizon_post_opposite:
        tags.append("horizon_post_opposite")
    if model_follows_post_not_horizon:
        tags.append("model_follows_post_not_horizon")
    if predicts_future_too_early:
        tags.append("predicts_future_too_early")
    if not tags:
        tags.append("no_clear_anchor_misalignment")

    return {
        "horizon_peak_abs": horizon_abs,
        "post2_peak_abs": post_abs,
        "post2_over_horizon_abs_ratio": float(post_abs / max(horizon_abs, 0.10)) if np.isfinite(post_abs) and np.isfinite(horizon_abs) else math.nan,
        "horizon_sign": sign_label(horizon_peak),
        "post2_sign": sign_label(post_peak),
        "v307_sign": sign_label(v307_peak),
        "horizon_flat_post_large": horizon_flat_post_large,
        "post2_dominant": post2_dominant,
        "horizon_post_opposite": horizon_post_opposite,
        "model_follows_post_not_horizon": model_follows_post_not_horizon,
        "predicts_future_too_early": predicts_future_too_early,
        "label_horizon_mismatch_suspected": label_horizon_mismatch_suspected,
        "misalignment_tags": ";".join(tags),
    }


def write_report(audit: pd.DataFrame, summary: pd.DataFrame, guardrail: Dict[str, object]) -> Path:
    """写中文审计报告。"""

    path = REPORTS / "v311_severe_anchor_horizon_misalignment_audit_cn.md"
    top = audit.sort_values(
        ["predicts_future_too_early", "label_horizon_mismatch_suspected", "post2_over_horizon_abs_ratio"],
        ascending=[False, False, False],
    ).head(20)
    cols = [
        "severe_rank",
        "screenshot_rank",
        "event_uid",
        "coarse_scene_label_cn",
        "true_peak",
        "v307_peak",
        "raw_2_6_peak",
        "post2_over_horizon_abs_ratio",
        "misalignment_tags",
    ]
    cols = [c for c in cols if c in top.columns]
    lines = [
        "# v311 severe anchor / horizon misalignment audit",
        "",
        "## 这一步做了什么",
        "",
        "本审计专门解释 v309/v310 暴露的严重错例：比较模型真正预测范围 `0-2s` 内的真实峰值，和 raw 车辆数据中 `2-6s` 后续峰值。",
        "",
        "如果一个样本 `0-2s` 内真实几乎没动作，但 `2s` 后才有大动作，而 v307 在 `0-2s` 就预测大动作，那么问题更像是锚点/标签窗口错位，而不是普通网络拟合不足。",
        "",
        "## 汇总",
        "",
        summary.to_markdown(index=False),
        "",
        "## 优先复核样本 Top 20",
        "",
        top[cols].to_markdown(index=False),
        "",
        "## 当前判断",
        "",
        "- `predicts_future_too_early=True` 的样本，模型很可能把 2s 后才发生的动作提前预测到了 0-2s。",
        "- `horizon_flat_post_large=True` 的样本，不应该仅靠加大转向/失稳标签权重解决，因为真实预测窗口内目标就是平的。",
        "- `horizon_post_opposite=True` 的样本，粗标签可能描述的是后续事件方向，而不是当前 0-2s 目标方向。",
        "- 下一轮更应改成 horizon-aligned 标签或锚点重定义，而不是继续堆 loss 权重。",
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


def make_zip_package() -> tuple[Path, bool]:
    """打包并校验产物。"""

    zip_path = OUT / "v311_severe_anchor_horizon_misalignment_audit_20260704.zip"
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
    started = time.time()
    clean_out_dir()
    if not V309_TEST_TABLE.exists():
        raise FileNotFoundError(f"缺少 v309 test table：{V309_TEST_TABLE}")
    if not V309_SEVERE.exists():
        raise FileNotFoundError(f"缺少 v309 severe table：{V309_SEVERE}")

    test_table = pd.read_csv(V309_TEST_TABLE, encoding="utf-8-sig")
    severe = pd.read_csv(V309_SEVERE, encoding="utf-8-sig")
    merged = severe.merge(
        test_table[
            [
                "event_uid",
                "subject",
                "recording",
                "scene_type",
                "route_event",
                "observation_s",
                "raw_vehicle_csv",
                "strong_steer",
                "vehicle_strong",
                "within_bad_top10_by_v249",
                "within_bad_top20_by_v249",
            ]
        ],
        on="event_uid",
        how="left",
        suffixes=("", "_v309"),
    )

    rows = []
    for _, row in merged.iterrows():
        base = row.to_dict()
        raw_feat = raw_peak_features(row)
        base.update(raw_feat)
        base.update(classify_misalignment(pd.Series(base)))
        rows.append(base)
    audit = pd.DataFrame(rows)
    write_csv(audit, TABLES / "v311_severe_anchor_horizon_misalignment_audit.csv")

    summary_rows = []
    groups = {
        "all_severe": audit,
        "user_screenshot": audit[audit["error_tags"].astype(str).str.contains("shown_in_user_screenshot")],
        "opposite_peak_direction": audit[audit["error_tags"].astype(str).str.contains("opposite_peak_direction")],
        "false_large_maneuver": audit[audit["error_tags"].astype(str).str.contains("false_large_maneuver")],
        "missed_extreme_amplitude": audit[audit["error_tags"].astype(str).str.contains("missed_extreme_amplitude")],
    }
    for group_name, df in groups.items():
        summary_rows.append(
            {
                "group": group_name,
                "n": int(len(df)),
                "raw_available_n": int(df["raw_available"].astype(bool).sum()) if len(df) else 0,
                "label_horizon_mismatch_suspected_n": int(df["label_horizon_mismatch_suspected"].astype(bool).sum()) if len(df) else 0,
                "predicts_future_too_early_n": int(df["predicts_future_too_early"].astype(bool).sum()) if len(df) else 0,
                "horizon_flat_post_large_n": int(df["horizon_flat_post_large"].astype(bool).sum()) if len(df) else 0,
                "horizon_post_opposite_n": int(df["horizon_post_opposite"].astype(bool).sum()) if len(df) else 0,
                "post2_dominant_n": int(df["post2_dominant"].astype(bool).sum()) if len(df) else 0,
                "post2_over_horizon_ratio_median": float(df["post2_over_horizon_abs_ratio"].replace([np.inf, -np.inf], np.nan).median()) if len(df) else math.nan,
            }
        )
    summary = pd.DataFrame(summary_rows)
    write_csv(summary, TABLES / "v311_misalignment_summary.csv")

    guardrail = {
        "pass": True,
        "version": "v311_severe_anchor_horizon_misalignment_audit_20260704",
        "training_run": False,
        "uses_v309_severe_table_for_diagnostic_only": True,
        "uses_test_error_as_training_feature": False,
        "candidate_selection_uses_test": False,
        "severe_event_n": int(len(audit)),
        "raw_available_n": int(audit["raw_available"].astype(bool).sum()),
        "label_horizon_mismatch_suspected_n": int(audit["label_horizon_mismatch_suspected"].astype(bool).sum()),
        "predicts_future_too_early_n": int(audit["predicts_future_too_early"].astype(bool).sum()),
        "runtime_seconds": float(time.time() - started),
    }
    report_path = write_report(audit, summary, guardrail)
    guardrail["report_path"] = str(report_path)
    write_json(guardrail, LOGS / "guardrail_check.json")
    write_file_inventory()
    zip_path, zip_ok = make_zip_package()
    guardrail["zip_path"] = str(zip_path)
    guardrail["zip_testzip"] = bool(zip_ok)
    write_json(guardrail, LOGS / "guardrail_check.json")
    write_file_inventory()
    print(json.dumps(guardrail, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
