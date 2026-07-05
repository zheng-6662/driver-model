"""
v234 短观察后预测评估层构建包。

背景：
v233 将部分样本标为 observe_later_review：旧锚点前证据弱，但后续变化很大。
这类样本不应与纯提前预测混在同一个评估层里，也不应简单改写事件锚点。

本脚本构建一个只读评估层定义：
- 纯提前预测参考层：observe_delay = 0.0s
- 短观察后预测候选层：observe_delay = 0.5s / 1.0s / 1.5s / 2.0s

每个层都从原始车辆 CSV 重建 2.0s horizon 的真实目标曲线：
target_delta_from_observe = SteeringWheel(t) - SteeringWheel(observe_anchor)

边界：
- 不训练模型
- 不修改标签
- 不改 formal headline
- 不把旧 formal prediction 硬评到新观察层
- 不重启硬响应类型分类
- 不把简单多候选轨迹输出作为主线
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import zipfile
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(r"F:\data_set_process\data_process")
REBUILD = ROOT / "05_rebuild_from_raw_20260511"
BASELINES = REBUILD / "03_baselines"
V232_SCRIPT = BASELINES / "scripts" / "stage03_v232_late_anchor_reanchor_candidates_20260624.py"
V233_DIR = BASELINES / "v233_adaptive_anchor_observation_policy_20260624"
V233_POLICY_PATH = V233_DIR / "tables" / "v233_anchor_observation_policy_table.csv"

OUT = BASELINES / "v234_short_observation_prediction_layer_20260624"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"
for folder in (TABLES, FIGURES, REPORTS, LOGS):
    folder.mkdir(parents=True, exist_ok=True)

OBSERVE_DELAYS = [0.0, 0.5, 1.0, 1.5, 2.0]
SHORT_OBSERVE_DELAYS = [0.5, 1.0, 1.5, 2.0]
HORIZON_TIMES = np.round(np.arange(0.0, 2.0 + 1e-9, 0.1), 4)
CONTEXT_GRID = np.round(np.arange(-3.0, 4.0001, 0.05), 4)

mpl.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
mpl.rcParams["axes.unicode_minus"] = False


def load_v232_module():
    spec = importlib.util.spec_from_file_location("v232_reanchor_module_for_v234", V232_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 v232 脚本：{V232_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def safe_float(value) -> float:
    try:
        if pd.isna(value):
            return math.nan
        return float(value)
    except Exception:
        return math.nan


def finite_ratio(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or abs(b) < 1e-12:
        return math.nan
    return float(a / b)


def signal_at(v232, raw, old_anchor_s: float, col: str, rel_old_s: float) -> tuple[float, float]:
    """返回某个旧锚点相对时刻的最近非空信号值和毫秒误差。"""
    t_rel_old = raw.df["t_rel_record_s"].to_numpy(dtype=float) - old_anchor_s
    values = pd.to_numeric(raw.df[col], errors="coerce").to_numpy(dtype=float)
    sampled, err_ms = v232.nearest_values(t_rel_old, values, np.array([rel_old_s], dtype=float))
    return float(sampled[0]), float(err_ms[0])


def sample_series_at(v232, raw, old_anchor_s: float, col: str, rel_old_targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t_rel_old = raw.df["t_rel_record_s"].to_numpy(dtype=float) - old_anchor_s
    values = pd.to_numeric(raw.df[col], errors="coerce").to_numpy(dtype=float)
    sampled, err_ms = v232.nearest_values(t_rel_old, values, rel_old_targets.astype(float))
    return sampled.astype(float), err_ms.astype(float)


def zscore_for_plot(series: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    std = np.nanstd(arr)
    if not np.isfinite(std) or std < 1e-12:
        return np.zeros_like(arr, dtype=float)
    return (arr - np.nanmean(arr)) / std


def layer_name(delay: float) -> str:
    if abs(delay) < 1e-9:
        return "pure_pre_anchor_d0p0"
    return f"short_observe_d{str(delay).replace('.', 'p')}s"


def build_one_sample(v232, row: pd.Series, raw_cache: dict[str, object]) -> tuple[list[dict], list[dict], pd.DataFrame]:
    sample_id = str(row["sample_id"])
    old_anchor_s = float(row["old_anchor_s"])
    raw_path = Path(str(row["raw_vehicle_csv"]))
    if str(raw_path) not in raw_cache:
        raw_cache[str(raw_path)] = v232.read_raw_vehicle(raw_path)
    raw = raw_cache[str(raw_path)]

    context = v232.signal_grid(raw, old_anchor_s, [col for col in v232.RAW_CANDIDATE_COLS if col not in ("ID", "StorageTime")], CONTEXT_GRID)
    context.insert(0, "sample_id", sample_id)
    context.insert(1, "old_anchor_s", old_anchor_s)
    steering = pd.to_numeric(context["zx|SteeringWheel"], errors="coerce")
    context["steering_smooth"] = v232.rolling_smooth(steering)
    baseline = safe_float(row.get("steering_baseline_old_minus8_to_minus6", math.nan))
    if not np.isfinite(baseline):
        baseline = float(np.nanmedian(context.loc[(context["t_rel_old_anchor_s"] >= -3) & (context["t_rel_old_anchor_s"] <= -2), "steering_smooth"]))
    context["steering_baseline"] = baseline
    context["steering_delta_from_baseline"] = context["steering_smooth"] - baseline
    rel = context["t_rel_old_anchor_s"].to_numpy(dtype=float)
    delta = context["steering_delta_from_baseline"].to_numpy(dtype=float)
    context["steering_delta_rate"] = np.gradient(delta, rel) if np.isfinite(delta).sum() >= 3 else math.nan

    old_anchor_steer, old_anchor_err_ms = signal_at(v232, raw, old_anchor_s, "zx|SteeringWheel", 0.0)
    future_peak_reference = safe_float(row.get("future_peak_abs_delta", math.nan))
    suggested_delay = safe_float(row.get("suggested_observe_delay_s", math.nan))

    sample_rows: list[dict] = []
    curve_rows: list[dict] = []

    for delay in OBSERVE_DELAYS:
        observe_anchor_s = old_anchor_s + delay
        observe_abs_time = raw.start_time + pd.to_timedelta(observe_anchor_s, unit="s")
        observe_steer, observe_err_ms = signal_at(v232, raw, old_anchor_s, "zx|SteeringWheel", delay)
        observe_delta_from_old_anchor = observe_steer - old_anchor_steer if np.isfinite(observe_steer) and np.isfinite(old_anchor_steer) else math.nan
        observe_delta_from_baseline = observe_steer - baseline if np.isfinite(observe_steer) and np.isfinite(baseline) else math.nan

        rel_old_targets = delay + HORIZON_TIMES
        target_steer, target_err_ms = sample_series_at(v232, raw, old_anchor_s, "zx|SteeringWheel", rel_old_targets)
        target_delta_from_observe = target_steer - observe_steer
        target_delta_from_old_anchor = target_steer - old_anchor_steer

        peak_abs = float(np.nanmax(np.abs(target_delta_from_observe))) if np.isfinite(target_delta_from_observe).any() else math.nan
        peak_idx = int(np.nanargmax(np.abs(target_delta_from_observe))) if np.isfinite(target_delta_from_observe).any() else -1
        peak_t = float(HORIZON_TIMES[peak_idx]) if peak_idx >= 0 else math.nan
        signed_peak = float(target_delta_from_observe[peak_idx]) if peak_idx >= 0 else math.nan
        zero_hold_rmse = float(np.sqrt(np.nanmean(np.square(target_delta_from_observe)))) if np.isfinite(target_delta_from_observe).any() else math.nan
        target_end_delta = float(target_delta_from_observe[-1]) if len(target_delta_from_observe) else math.nan
        visible_fraction = finite_ratio(abs(observe_delta_from_baseline), future_peak_reference)
        remaining_fraction = finite_ratio(peak_abs, future_peak_reference)

        if abs(delay) < 1e-9:
            layer_status = "reference_pure_pre_anchor"
        elif np.isfinite(suggested_delay) and abs(delay - suggested_delay) < 1e-9:
            layer_status = "suggested_short_observation"
        else:
            layer_status = "candidate_short_observation"

        sample_rows.append(
            {
                "sample_id": sample_id,
                "subject": row.get("subject", ""),
                "recording": row.get("recording", ""),
                "display_pool": row.get("display_pool", ""),
                "display_model": row.get("display_model", ""),
                "scene_type": row.get("scene_type", ""),
                "route_event": row.get("route_event", ""),
                "old_anchor_s": old_anchor_s,
                "observe_delay_s": delay,
                "observe_anchor_s": observe_anchor_s,
                "observe_anchor_abs_time": observe_abs_time.isoformat(sep=" "),
                "layer_name": layer_name(delay),
                "layer_status": layer_status,
                "old_anchor_steering": old_anchor_steer,
                "old_anchor_steering_time_error_ms": old_anchor_err_ms,
                "observe_steering": observe_steer,
                "observe_steering_time_error_ms": observe_err_ms,
                "observe_delta_from_old_anchor": observe_delta_from_old_anchor,
                "observe_delta_from_baseline": observe_delta_from_baseline,
                "visible_fraction_of_future_peak": visible_fraction,
                "target_peak_abs_delta_from_observe": peak_abs,
                "target_peak_t_s": peak_t,
                "target_peak_signed_delta_from_observe": signed_peak,
                "target_end_delta_from_observe": target_end_delta,
                "zero_hold_rmse_after_observe": zero_hold_rmse,
                "remaining_peak_fraction_of_original_future_peak": remaining_fraction,
                "future_peak_abs_reference": future_peak_reference,
                "pre3_to_post03_peak_ratio": safe_float(row.get("pre3_to_post03_peak_ratio", math.nan)),
                "eval_rmse_old_formal": safe_float(row.get("eval_rmse", math.nan)),
                "eval_tail_rmse_old_formal": safe_float(row.get("eval_tail_rmse", math.nan)),
                "policy_reason_cn": row.get("policy_reason_cn", ""),
                "suggested_delay_reason_cn": row.get("suggested_delay_reason_cn", ""),
                "needs_human_confirm": True,
                "human_layer_decision": "",
                "human_observe_delay_s": "",
                "human_use_for_training": "",
                "human_note_cn": "",
            }
        )

        for h, abs_t, steer_value, steer_err, d_obs, d_old in zip(
            HORIZON_TIMES,
            rel_old_targets,
            target_steer,
            target_err_ms,
            target_delta_from_observe,
            target_delta_from_old_anchor,
        ):
            curve_rows.append(
                {
                    "sample_id": sample_id,
                    "layer_name": layer_name(delay),
                    "observe_delay_s": delay,
                    "old_anchor_s": old_anchor_s,
                    "observe_anchor_s": observe_anchor_s,
                    "horizon_t_s": float(h),
                    "t_rel_old_anchor_s": float(abs_t),
                    "StorageTime": (raw.start_time + pd.to_timedelta(old_anchor_s + float(abs_t), unit="s")).isoformat(sep=" "),
                    "steering": float(steer_value) if np.isfinite(steer_value) else math.nan,
                    "steering_time_error_ms": float(steer_err),
                    "target_delta_from_observe": float(d_obs) if np.isfinite(d_obs) else math.nan,
                    "target_delta_from_old_anchor": float(d_old) if np.isfinite(d_old) else math.nan,
                }
            )

    return sample_rows, curve_rows, context


def plot_sample(row: pd.Series, context: pd.DataFrame, sample_layer: pd.DataFrame, curve_df: pd.DataFrame) -> Path:
    sample_id = str(row["sample_id"])
    x = context["t_rel_old_anchor_s"].to_numpy(dtype=float)

    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=False)
    fig.suptitle(
        f"{sample_id}\n短观察后预测层：旧锚点={float(row['old_anchor_s']):.3f}s，候选延迟=0.5/1.0/1.5/2.0s",
        fontsize=12,
    )

    ax = axes[0]
    ax.axvline(0, color="#dc2626", lw=1.6, label="old anchor")
    for delay in SHORT_OBSERVE_DELAYS:
        ax.axvline(delay, lw=1.2, linestyle="--", label=f"observe +{delay:.1f}s")
    ax.plot(x, context["zx|SteeringWheel"], color="#94a3b8", lw=0.9, label="Steering raw")
    ax.plot(x, context["steering_smooth"], color="#1d4ed8", lw=1.3, label="Steering smooth")
    ax.axhline(float(row["steering_baseline_old_minus8_to_minus6"]), color="#475569", lw=0.9, ls=":", label="baseline")
    ax.set_title("观察点定义：旧锚点和后移观察点", loc="left", fontsize=10)
    ax.set_xlim(-3, 4)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8, ncol=3)

    ax = axes[1]
    colors = {0.0: "#111827", 0.5: "#2563eb", 1.0: "#16a34a", 1.5: "#f97316", 2.0: "#7c3aed"}
    for delay in OBSERVE_DELAYS:
        layer = layer_name(delay)
        sub = curve_df[(curve_df["sample_id"] == sample_id) & (curve_df["layer_name"] == layer)]
        if sub.empty:
            continue
        label = "pure +0.0s" if delay == 0 else f"observe +{delay:.1f}s"
        ax.plot(sub["horizon_t_s"], sub["target_delta_from_observe"], lw=1.5, color=colors[delay], label=label)
    ax.axhline(0, color="#6b7280", lw=0.8)
    ax.set_title("各观察点之后 2 秒真实目标曲线：相对观察点方向盘增量", loc="left", fontsize=10)
    ax.set_xlabel("horizon after observe point (s)")
    ax.set_ylabel("delta steering")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8, ncol=3)

    ax = axes[2]
    layer_order = sample_layer.sort_values("observe_delay_s")
    xpos = np.arange(len(layer_order))
    ax.bar(xpos - 0.18, layer_order["visible_fraction_of_future_peak"], width=0.35, label="visible fraction at observe")
    ax.bar(xpos + 0.18, layer_order["remaining_peak_fraction_of_original_future_peak"], width=0.35, label="remaining target fraction")
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"+{d:.1f}s" for d in layer_order["observe_delay_s"]])
    ax.set_ylim(0, max(1.2, float(np.nanmax(layer_order[["visible_fraction_of_future_peak", "remaining_peak_fraction_of_original_future_peak"]].to_numpy())) + 0.2))
    ax.set_title("看见多少早期证据，以及后续还剩多少要预测", loc="left", fontsize=10)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[3]
    ax.axvline(0, color="#dc2626", lw=1.3, label="old anchor")
    for delay in [0.5, 1.0, 1.5]:
        ax.axvline(delay, lw=1.0, linestyle="--", label=f"+{delay:.1f}s")
    for col in ["zx|ay", "zx|vyaw", "zx|yaw", "zx|roll", "zx1|lanecurvatureXY", "zx1|lateraldistance"]:
        if col in context.columns and context[col].notna().sum() > 0:
            ax.plot(x, zscore_for_plot(context[col]), lw=1.0, label=f"{col} (z)")
    ax.set_title("车辆/道路信号：后移观察点是否出现可见证据", loc="left", fontsize=10)
    ax.set_xlim(-3, 4)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8, ncol=3)

    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    out_path = FIGURES / f"{int(row['short_obs_rank']):02d}_{sample_id}_short_observation_layer.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def make_contact_sheet(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    thumb_w, thumb_h = 820, 690
    pad = 24
    label_h = 38
    cols = 2
    rows = math.ceil(len(paths) / cols)
    canvas = Image.new("RGB", (cols * thumb_w + (cols + 1) * pad, rows * (thumb_h + label_h) + (rows + 1) * pad), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    for i, path in enumerate(paths):
        image = Image.open(path).convert("RGB")
        image.thumbnail((thumb_w, thumb_h), Image.LANCZOS)
        x = pad + (i % cols) * (thumb_w + pad)
        y = pad + (i // cols) * (thumb_h + label_h + pad)
        draw.text((x, y), path.name[:88], fill=(20, 20, 20), font=font)
        canvas.paste(image, (x, y + label_h))
    out_path = FIGURES / "v234_short_observation_layer_contact_sheet.png"
    canvas.save(out_path, quality=95)
    return out_path


def write_report(assign_df: pd.DataFrame, layer_df: pd.DataFrame, contact_sheet: Path | None) -> Path:
    report_path = REPORTS / "v234_short_observation_prediction_layer_cn.md"
    suggested = layer_df[layer_df["layer_status"].eq("suggested_short_observation")].copy()
    lines: list[str] = []
    lines.append("# v234 短观察后预测评估层构建包")
    lines.append("")
    lines.append("## 目的")
    lines.append("")
    lines.append("本包把 v233 的 `observe_later_review` 样本单独构造成“短观察后预测”评估层。")
    lines.append("它不是统一后移事件锚点，也不是训练新模型，而是把任务可观测性分层：纯提前预测和短观察后预测分开报告。")
    lines.append("")
    lines.append("## 方法边界")
    lines.append("")
    lines.append("- 不训练模型、不改标签、不改 formal headline。")
    lines.append("- 不把旧 formal prediction 硬评到新观察层；旧 prediction 是从旧锚点出发的，不适合直接评估后移观察点。")
    lines.append("- 不重启硬响应类型分类；不把简单多候选轨迹输出作为主线。")
    lines.append("")
    lines.append("## 输出文件")
    lines.append("")
    lines.append(f"- 层定义表：`{TABLES / 'v234_short_observation_layer_definition.csv'}`")
    lines.append(f"- 样本层分配表：`{TABLES / 'v234_short_observation_layer_assignments.csv'}`")
    lines.append(f"- 真实目标曲线长表：`{TABLES / 'v234_short_observation_target_curves.csv'}`")
    lines.append(f"- 人工审核模板：`{TABLES / 'v234_short_observation_manual_review_template.csv'}`")
    lines.append(f"- 图目录：`{FIGURES}`")
    if contact_sheet is not None:
        lines.append(f"- 图拼接总览：`{contact_sheet}`")
    lines.append(f"- ZIP 包：`{OUT / 'v234_short_observation_prediction_layer_pack.zip'}`")
    lines.append("")
    lines.append("## 默认建议层摘要")
    lines.append("")
    lines.append("|rank|sample_id|old_anchor_s|suggest_delay|visible_frac|remaining_frac|remaining_peak|zero_hold_rmse|")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|")
    for row in suggested.sort_values("short_obs_rank").itertuples(index=False):
        lines.append(
            f"|{int(row.short_obs_rank)}|`{row.sample_id}`|{float(row.old_anchor_s):.3f}|{float(row.observe_delay_s):.1f}|"
            f"{float(row.visible_fraction_of_future_peak):.3f}|{float(row.remaining_peak_fraction_of_original_future_peak):.3f}|"
            f"{float(row.target_peak_abs_delta_from_observe):.3f}|{float(row.zero_hold_rmse_after_observe):.3f}|"
        )
    lines.append("")
    lines.append("## 解释")
    lines.append("")
    lines.append("`visible_frac` 表示到观察点时已经看见的方向盘证据占原后续峰值的比例。")
    lines.append("`remaining_frac` 表示从观察点之后 2 秒内仍然剩余的目标峰值占原后续峰值的比例。")
    lines.append("如果 `visible_frac` 上升而 `remaining_frac` 仍不低，说明短观察后预测既有可见证据，也仍有真实未来要预测，不是简单补全已经发生的轨迹。")
    lines.append("")
    lines.append("## 下一步")
    lines.append("")
    lines.append("人工先审核默认 `0.5s` 层是否合理；如果某些样本 0.5 秒仍太早或太晚，可在模板里填写 `human_selected_observe_delay_s=1.0/1.5`。")
    lines.append("人工确认后，下一步才生成 v235 的短观察层数据清单或重新评估对应模型。")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def write_zip() -> Path:
    zip_path = OUT / "v234_short_observation_prediction_layer_pack.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for folder in (TABLES, FIGURES, REPORTS, LOGS):
            for path in folder.rglob("*"):
                if path.is_file():
                    zf.write(path, path.relative_to(OUT))
    return zip_path


def main() -> None:
    v232 = load_v232_module()
    policy_df = pd.read_csv(V233_POLICY_PATH, encoding="utf-8-sig")
    obs_df = policy_df[policy_df["anchor_observation_policy"].eq("observe_later_review")].copy()
    obs_df = obs_df.sort_values(["future_peak_abs_delta", "eval_tail_rmse"], ascending=False).reset_index(drop=True)
    obs_df.insert(0, "short_obs_rank", range(1, len(obs_df) + 1))

    raw_cache: dict[str, object] = {}
    all_layer_rows: list[dict] = []
    all_curve_rows: list[dict] = []
    context_frames: list[pd.DataFrame] = []
    errors: list[dict] = []

    for row in obs_df.itertuples(index=False):
        s = pd.Series(row._asdict())
        try:
            sample_rows, curve_rows, context = build_one_sample(v232, s, raw_cache)
            for item in sample_rows:
                item["short_obs_rank"] = int(s["short_obs_rank"])
            for item in curve_rows:
                item["short_obs_rank"] = int(s["short_obs_rank"])
            context.insert(0, "short_obs_rank", int(s["short_obs_rank"]))
            all_layer_rows.extend(sample_rows)
            all_curve_rows.extend(curve_rows)
            context_frames.append(context)
        except Exception as exc:
            errors.append({"sample_id": str(s.get("sample_id", "")), "error": repr(exc)})

    layer_df = pd.DataFrame(all_layer_rows)
    curve_df = pd.DataFrame(all_curve_rows)
    context_df = pd.concat(context_frames, ignore_index=True) if context_frames else pd.DataFrame()

    layer_def_rows = [
        {
            "layer_name": layer_name(delay),
            "observe_delay_s": delay,
            "layer_type": "pure_pre_anchor_reference" if delay == 0 else "short_observation_candidate",
            "horizon_s": 2.0,
            "horizon_step_s": 0.1,
            "target_definition": "SteeringWheel(t) - SteeringWheel(observe_anchor)",
            "usage_cn": "纯提前预测参考层" if delay == 0 else "短观察后预测候选层",
        }
        for delay in OBSERVE_DELAYS
    ]
    layer_def_df = pd.DataFrame(layer_def_rows)

    suggested = layer_df[layer_df["layer_status"].eq("suggested_short_observation")].copy()
    review_template = suggested[[
        "short_obs_rank",
        "sample_id",
        "old_anchor_s",
        "observe_delay_s",
        "observe_anchor_s",
        "visible_fraction_of_future_peak",
        "remaining_peak_fraction_of_original_future_peak",
        "target_peak_abs_delta_from_observe",
        "zero_hold_rmse_after_observe",
        "policy_reason_cn",
    ]].copy()
    review_template["human_layer_decision"] = ""
    review_template["human_selected_observe_delay_s"] = ""
    review_template["human_use_for_training"] = ""
    review_template["human_note_cn"] = ""

    layer_def_path = TABLES / "v234_short_observation_layer_definition.csv"
    assignments_path = TABLES / "v234_short_observation_layer_assignments.csv"
    curves_path = TABLES / "v234_short_observation_target_curves.csv"
    context_path = TABLES / "v234_short_observation_context_grid.csv"
    review_path = TABLES / "v234_short_observation_manual_review_template.csv"

    layer_def_df.to_csv(layer_def_path, index=False, encoding="utf-8-sig")
    layer_df.to_csv(assignments_path, index=False, encoding="utf-8-sig")
    curve_df.to_csv(curves_path, index=False, encoding="utf-8-sig")
    context_df.to_csv(context_path, index=False, encoding="utf-8-sig")
    review_template.to_csv(review_path, index=False, encoding="utf-8-sig")

    figure_paths: list[Path] = []
    for row in obs_df.itertuples(index=False):
        s = pd.Series(row._asdict())
        sample_id = str(s["sample_id"])
        sample_context = context_df[context_df["sample_id"].eq(sample_id)].copy()
        sample_layer = layer_df[layer_df["sample_id"].eq(sample_id)].copy()
        if sample_context.empty or sample_layer.empty:
            continue
        figure_paths.append(plot_sample(s, sample_context, sample_layer, curve_df))
    contact_sheet = make_contact_sheet(figure_paths)
    report_path = write_report(layer_df, layer_df, contact_sheet)
    zip_path = write_zip()

    manifest = {
        "version": "v234_short_observation_prediction_layer_20260624",
        "observe_later_sample_count": int(len(obs_df)),
        "layer_count": int(len(layer_def_df)),
        "assignment_rows": int(len(layer_df)),
        "curve_rows": int(len(curve_df)),
        "figure_count": int(len(figure_paths)),
        "errors": errors,
        "outputs": {
            "layer_definition": str(layer_def_path),
            "assignments": str(assignments_path),
            "curves": str(curves_path),
            "context": str(context_path),
            "review_template": str(review_path),
            "report": str(report_path),
            "contact_sheet": str(contact_sheet) if contact_sheet else "",
            "zip": str(zip_path),
        },
        "method_boundaries": [
            "no_training",
            "no_label_modification",
            "no_formal_headline_change",
            "do_not_eval_old_formal_prediction_on_shifted_layer",
            "short_observation_layer_separate_from_pure_prediction",
        ],
    }
    (LOGS / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (LOGS / "v234_errors.json").write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")

    print("DONE v234")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(suggested[[
        "short_obs_rank",
        "sample_id",
        "old_anchor_s",
        "observe_delay_s",
        "visible_fraction_of_future_peak",
        "remaining_peak_fraction_of_original_future_peak",
        "target_peak_abs_delta_from_observe",
        "zero_hold_rmse_after_observe",
    ]].sort_values("short_obs_rank").to_string(index=False))


if __name__ == "__main__":
    main()
