# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from typing import Any


P1 = "P1_strong_steer_vehicle_response_correction"
P2 = "P2_strong_steer_weak_vehicle_response"
C = "C_normal_curve_or_normal_lane_change"
N = "N_trigger_no_effect_or_no_response"
U = "U_unclear"
X = "X_exclude"


def classify_episode(row: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    if row.get("row_source") == "trigger_context_without_steering_episode":
        row["episode_class"] = N
        row.setdefault("class_reason_cn", "场景触发附近没有方向盘快速动作 episode")
        return row

    cls_cfg = config.get("classification", {}) or {}
    reasons: list[str] = []
    pre_ok = bool(row.get("pre_window_complete", False))
    label_ok = bool(row.get("label_window_complete", False))
    coord_ok = bool(row.get("coordinate_continuity_ok", True))
    steering_score = float(row.get("steering_impulse_score", 0.0) or 0.0)
    vehicle_score = float(row.get("vehicle_dynamic_score", 0.0) or 0.0)
    correction_score = float(row.get("correction_score", 0.0) or 0.0)
    has_vehicle = bool(row.get("has_vehicle_response", False))
    has_correction = bool(row.get("has_correction", False))

    if not pre_ok or not label_ok:
        row["episode_class"] = X
        row["class_reason_cn"] = "前置输入窗口或后续标签窗口不完整"
        return row
    if not coord_ok and vehicle_score > 1.0:
        row["episode_class"] = X
        row["class_reason_cn"] = "横向偏移存在疑似非物理跳变，暂不作为正样本"
        return row

    is_curve = bool(row.get("is_curve_context", False))
    is_lane_like = bool(row.get("is_middle_section_context", False) or row.get("is_longstraight_context", False) or row.get("is_fix_road_context", False))

    p1 = (
        steering_score >= float(cls_cfg.get("p1_min_steering_impulse_score", 2.0))
        and vehicle_score >= float(cls_cfg.get("p1_min_vehicle_dynamic_score", 2.0))
        and correction_score >= float(cls_cfg.get("p1_min_correction_score", 1.5))
        and has_vehicle
        and has_correction
    )
    if p1:
        row["episode_class"] = P1
        row["class_reason_cn"] = "强方向盘启动、车辆动态增强和纠正过程同时成立"
        return row

    p2 = (
        steering_score >= float(cls_cfg.get("p2_min_steering_impulse_score", 2.0))
        and correction_score >= float(cls_cfg.get("p2_min_correction_score", 1.0))
        and not has_vehicle
    )
    if p2:
        row["episode_class"] = P2
        row["class_reason_cn"] = "方向盘动作和纠正较明显，但车辆横向/横摆/侧倾响应较弱"
        return row

    if is_curve and vehicle_score <= float(cls_cfg.get("normal_curve_vehicle_score_max", 2.2)):
        row["episode_class"] = C
        row["class_reason_cn"] = "弯道上下文中车辆动态不强，更像正常平滑过弯"
        return row
    if is_lane_like and vehicle_score < 1.5 and correction_score < 1.0:
        row["episode_class"] = C
        row["class_reason_cn"] = "道路/变道上下文中方向盘变化较像正常驾驶动作"
        return row

    if steering_score < 1.2:
        row["episode_class"] = U
        row["class_reason_cn"] = "方向盘动作分数偏低，需人工确认是否真实 episode"
        return row
    if vehicle_score < 1.0 and correction_score < 1.0:
        row["episode_class"] = U
        row["class_reason_cn"] = "方向盘动作存在，但车辆响应和纠正证据都偏弱"
        return row

    row["episode_class"] = U
    row["class_reason_cn"] = "方向盘、车辆动态或纠正证据不完全一致，需要人工复核"
    return row

