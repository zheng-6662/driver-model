# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


def configure_fonts() -> None:
    for font_path in [
        Path(r"C:/Windows/Fonts/msyh.ttc"),
        Path(r"C:/Windows/Fonts/simhei.ttf"),
        Path(r"C:/Windows/Fonts/simsun.ttc"),
    ]:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            prop = font_manager.FontProperties(fname=str(font_path))
            plt.rcParams["font.sans-serif"] = [prop.get_name()]
            break
    plt.rcParams["axes.unicode_minus"] = False


def plot_episode(vehicle: pd.DataFrame, row: pd.Series, out_path: Path, config: dict[str, Any]) -> bool:
    if vehicle.empty:
        return False
    t0 = pd.to_numeric(pd.Series([row.get("t_steer_onset")]), errors="coerce").iloc[0]
    if not math.isfinite(float(t0)):
        t0 = pd.to_numeric(pd.Series([row.get("nearest_aed_trigger_time")]), errors="coerce").iloc[0]
    if not math.isfinite(float(t0)):
        return False
    t0 = float(t0)
    start = t0 - float(config.get("pre_window_sec", 2.0))
    end = t0 + float(config.get("correction_window_sec", 5.0))
    win = vehicle[(vehicle["time_rel_s"] >= start) & (vehicle["time_rel_s"] <= end)].copy()
    if len(win) < 20:
        return False
    t = win["time_rel_s"].to_numpy(dtype=float) - t0
    signals = [
        ("steer_smooth", "方向盘角"),
        ("steer_rate", "方向盘角速度"),
        ("speed", "速度"),
        ("ay", "横向加速度"),
        ("yaw_rate", "横摆角速度"),
        ("roll_rate", "侧倾角速度"),
        ("lat_offset", "横向偏移"),
        ("brake", "制动"),
        ("mu", "附着系数"),
    ]
    available = [(col, name) for col, name in signals if col in win.columns]
    if not available:
        return False
    fig, axes = plt.subplots(len(available), 1, figsize=(12, max(7, 1.25 * len(available))), dpi=130, sharex=True)
    if len(available) == 1:
        axes = [axes]
    markers = [
        (0.0, "t_steer_onset", "#111111"),
        (float(config.get("early_observation_sec", 0.5)), "t_obs_end", "#4C78A8"),
        (row.get("t_steer_peak"), "t_steer_peak", "#E45756"),
        (row.get("t_correction_onset"), "t_correction", "#54A24B"),
        (row.get("nearest_aed_trigger_time"), "aed", "#F58518"),
        (row.get("nearest_old_anchor_time"), "old", "#B279A2"),
    ]
    for ax, (col, label) in zip(axes, available):
        ax.plot(t, win[col].to_numpy(dtype=float), linewidth=0.9)
        ax.set_ylabel(label)
        ax.grid(alpha=0.18)
        for marker_time, marker_label, color in markers:
            try:
                mt = float(marker_time)
            except Exception:
                continue
            if not math.isfinite(mt):
                continue
            x = mt - t0 if marker_label not in {"t_steer_onset", "t_obs_end"} else mt
            if start - t0 <= x <= end - t0:
                ax.axvline(x, color=color, linestyle="--", linewidth=0.9)
    axes[-1].set_xlabel("相对方向盘启动时刻的时间（秒）")
    title = (
        f"{row.get('episode_class','')} | {row.get('subject_id','')} {row.get('session_stamp','')} "
        f"| {row.get('road_context','')} | {row.get('class_reason_cn','')}"
    )
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return True


def select_review_rows(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    fig_cfg = config.get("review_figures", {}) or {}
    selected = []
    for cls, n in fig_cfg.items():
        sub = df[df["episode_class"].eq(cls)].copy()
        if sub.empty:
            continue
        if cls.startswith("P1"):
            sub["_score"] = pd.to_numeric(sub.get("steering_impulse_score"), errors="coerce").fillna(0) + pd.to_numeric(sub.get("vehicle_dynamic_score"), errors="coerce").fillna(0) + pd.to_numeric(sub.get("correction_score"), errors="coerce").fillna(0)
            sub = sub.sort_values("_score", ascending=False)
        elif cls.startswith("U"):
            sub["_score"] = (
                pd.to_numeric(sub.get("steering_impulse_score"), errors="coerce").fillna(0)
                + pd.to_numeric(sub.get("vehicle_dynamic_score"), errors="coerce").fillna(0)
            )
            sub["_distance_to_middle"] = (sub["_score"] - sub["_score"].median()).abs()
            sub = sub.sort_values("_distance_to_middle")
        else:
            sub["_score"] = pd.to_numeric(sub.get("steering_impulse_score"), errors="coerce").fillna(0)
            sub = sub.sort_values("_score", ascending=False)
        selected.append(sub.head(int(n)))
    return pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()
