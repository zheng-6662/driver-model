"""
v233 自适应锚点/观察时长策略审核包。

背景：
用户指出，部分大变化事件在事件前几秒确实看不出区别。对这类样本，继续要求模型在旧锚点
处立刻预测完整大响应并不合理；应区别于“锚点晚了、需要提前重锚定”的样本，考虑后移观察点
或延长观察时长。

本脚本只做审核包：
- 不训练模型
- 不修改标签
- 不改 formal headline
- 不把“硬响应类型分类”或“简单多候选轨迹输出”作为主线

输出：
- 样本级策略表：reanchor_earlier / observe_later / standard / ambiguous
- 后移观察延迟表：0, 0.5, 1.0, 1.5, 2.0, 3.0 秒下已经显露多少反应
- 人工审核图和中文报告
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
V232_OUT = BASELINES / "v232_late_anchor_reanchor_candidates_20260624"
OUT = BASELINES / "v233_adaptive_anchor_observation_policy_20260624"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"

for folder in (TABLES, FIGURES, REPORTS, LOGS):
    folder.mkdir(parents=True, exist_ok=True)

OBSERVE_DELAYS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]

mpl.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
mpl.rcParams["axes.unicode_minus"] = False


def load_v232_module():
    spec = importlib.util.spec_from_file_location("v232_reanchor_module", V232_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 v232 脚本：{V232_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def nearest_value(df: pd.DataFrame, rel_col: str, value_col: str, target: float) -> float:
    if df.empty or value_col not in df.columns:
        return math.nan
    rel = pd.to_numeric(df[rel_col], errors="coerce").to_numpy(dtype=float)
    values = pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(rel) & np.isfinite(values)
    if not np.any(valid):
        return math.nan
    idx_pool = np.flatnonzero(valid)
    idx = idx_pool[int(np.argmin(np.abs(rel[idx_pool] - target)))]
    return float(values[idx])


def finite_ratio(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or abs(b) < 1e-12:
        return math.nan
    return float(a / b)


def classify_policy(row: pd.Series) -> tuple[str, str]:
    """把样本分到策略类。阈值保守，目的是做人工审核入口，不是自动改任务定义。"""
    priority = str(row["review_priority"])
    pre3 = float(row["pre_3_0_peak_abs_delta"])
    post03 = float(row["post_0_3_peak_abs_delta"])
    post38 = float(row["post_3_8_peak_abs_delta"])
    phase = float(row["old_anchor_phase_ratio_abs_delta_over_peak"])
    pre_ratio = finite_ratio(pre3, post03)
    large_future = max(post03 if np.isfinite(post03) else 0.0, post38 if np.isfinite(post38) else 0.0)

    # 已经由人工确认的晚锚点和强晚锚点证据，仍然优先走提前重锚定。
    if priority.startswith("P0") or priority.startswith("P1"):
        return "reanchor_earlier_review", "旧锚点前已有强证据或人工确认，优先审核提前重锚定"

    # P2 但旧锚点处已经处于响应进程，仍应先人工看是否需要提前重锚定。
    if priority.startswith("P2") and ((np.isfinite(pre_ratio) and pre_ratio >= 0.45) or (np.isfinite(phase) and phase >= 0.35)):
        return "reanchor_earlier_or_ambiguous_review", "存在中等晚锚点证据，需人工判定提前重锚定还是保留原锚点"

    # 大响应、但旧锚点前 3 秒证据弱：这就是用户说的“前几秒看不出区别”的核心集合。
    if large_future >= 1.5 and np.isfinite(pre_ratio) and pre_ratio <= 0.35 and np.isfinite(phase) and phase <= 0.35:
        return "observe_later_review", "后续变化很大但旧锚点前证据弱，适合审核后移观察点/延长观察时长"

    if large_future >= 1.5:
        return "large_change_standard_or_ambiguous", "后续变化较大，但不满足低前证据条件；需结合图判断"

    return "standard_anchor_review", "未见明显需要变更锚点或观察时长的证据"


def choose_observe_delay(delay_rows: pd.DataFrame, future_peak: float) -> tuple[float, str]:
    if not np.isfinite(future_peak) or future_peak <= 0:
        return math.nan, "future_peak 不可用"
    # 目标不是看完整反应，而是让模型至少看到可区分早期证据。
    abs_threshold = max(0.35, 0.25 * future_peak)
    for row in delay_rows.sort_values("observe_delay_s").itertuples(index=False):
        if float(row.observe_delay_s) <= 0:
            continue
        if np.isfinite(row.abs_delta_at_delay) and float(row.abs_delta_at_delay) >= abs_threshold:
            return float(row.observe_delay_s), f"首次达到可见证据阈值 abs_delta>={abs_threshold:.3f}"
    return math.nan, "0-3s 内未达到可见证据阈值，需人工看其他车辆动力学信号"


def zscore_for_plot(series: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    std = np.nanstd(arr)
    if not np.isfinite(std) or std < 1e-12:
        return np.zeros_like(arr, dtype=float)
    return (arr - np.nanmean(arr)) / std


def plot_policy_case(row: pd.Series, grid: pd.DataFrame) -> Path:
    sample_id = str(row["sample_id"])
    sample_grid = grid[grid["sample_id"] == sample_id].copy()
    x = sample_grid["t_rel_old_anchor_s"].to_numpy(dtype=float)
    observe_delay = float(row["suggested_observe_delay_s"]) if np.isfinite(row["suggested_observe_delay_s"]) else math.nan
    candidate_shift = float(row["anchor_shift_s"]) if np.isfinite(row["anchor_shift_s"]) else math.nan

    fig, axes = plt.subplots(5, 1, figsize=(15, 13), sharex=True)
    fig.suptitle(
        f"{sample_id}\npolicy={row['anchor_observation_policy']}, old_anchor={float(row['old_anchor_s']):.3f}s, "
        f"suggested_observe_delay={observe_delay if np.isfinite(observe_delay) else 'NA'}",
        fontsize=12,
    )

    def mark(ax):
        ax.axvline(0, color="#dc2626", lw=1.6, label="old anchor")
        if np.isfinite(candidate_shift):
            ax.axvline(candidate_shift, color="#16a34a", lw=1.4, label="earlier reanchor candidate")
        if np.isfinite(observe_delay):
            ax.axvline(observe_delay, color="#2563eb", lw=1.4, label="observe-later point")
        ax.axvspan(-3, 0, color="#9ca3af", alpha=0.10, linewidth=0)
        ax.axvspan(0, 3, color="#60a5fa", alpha=0.08, linewidth=0)
        ax.grid(True, alpha=0.25)

    ax = axes[0]
    mark(ax)
    ax.plot(x, sample_grid["zx|SteeringWheel"], color="#94a3b8", lw=0.9, label="Steering raw")
    ax.plot(x, sample_grid["steering_smooth"], color="#1d4ed8", lw=1.3, label="Steering smooth")
    ax.axhline(float(row["steering_baseline_old_minus8_to_minus6"]), color="#475569", ls="--", lw=0.9, label="baseline")
    ax.set_title("方向盘：旧锚点前是否已有可区分变化", loc="left", fontsize=10)
    ax.legend(loc="upper right", fontsize=8, ncol=3)

    ax = axes[1]
    mark(ax)
    ax.plot(x, sample_grid["steering_delta_from_baseline"], color="#7c3aed", lw=1.2, label="delta from baseline")
    ax.plot(x, sample_grid["steering_delta_rate"], color="#ea580c", lw=1.0, label="delta rate")
    ax.set_title("相对基线变化：后移观察点是否能看到早期证据", loc="left", fontsize=10)
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    ax = axes[2]
    mark(ax)
    for col in ["zx1|v_km/h", "zx|v_km/h", "zx|vx", "zx|vy"]:
        if col in sample_grid.columns and sample_grid[col].notna().sum() > 0:
            ax.plot(x, sample_grid[col], lw=1.0, label=col)
    ax.set_title("速度 / 速度分量", loc="left", fontsize=10)
    ax.legend(loc="upper right", fontsize=8, ncol=3)

    ax = axes[3]
    mark(ax)
    for col in ["zx|ay", "zx|vyaw", "zx|ayaw", "zx|yaw", "zx|roll", "zx|pitch"]:
        if col in sample_grid.columns and sample_grid[col].notna().sum() > 0:
            ax.plot(x, zscore_for_plot(sample_grid[col]), lw=1.0, label=f"{col} (z)")
    ax.set_title("车辆动力学信号（z-score）", loc="left", fontsize=10)
    ax.legend(loc="upper right", fontsize=8, ncol=3)

    ax = axes[4]
    mark(ax)
    for col in ["zx1|lanecurvatureXY", "zx|lanecurvatureXY", "zx1|lateraldistance", "zx|lateraldistance", "zx|AcceleratorPedal", "zx|BrakePedal"]:
        if col in sample_grid.columns and sample_grid[col].notna().sum() > 0:
            ax.plot(x, zscore_for_plot(sample_grid[col]), lw=1.0, label=f"{col} (z)")
    ax.set_title("道路/车道/踏板信号（z-score）", loc="left", fontsize=10)
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    ax.set_xlim(-8, 6)
    ax.set_xlabel("seconds relative to old anchor")

    fig.text(0.01, 0.01, f"{row['policy_reason_cn']}；{row['suggested_delay_reason_cn']}", fontsize=9)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = FIGURES / f"{int(row['policy_rank']):02d}_{sample_id}_policy.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def make_contact_sheet(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    thumb_w, thumb_h = 820, 710
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
    out_path = FIGURES / "v233_adaptive_anchor_policy_contact_sheet.png"
    canvas.save(out_path, quality=95)
    return out_path


def write_report(policy_df: pd.DataFrame, review_df: pd.DataFrame, contact_sheet: Path | None) -> Path:
    report_path = REPORTS / "v233_adaptive_anchor_observation_policy_cn.md"
    lines: list[str] = []
    lines.append("# v233 自适应锚点 / 观察时长策略审核包")
    lines.append("")
    lines.append("## 目的")
    lines.append("")
    lines.append("本包回应用户的新判断：有些大变化事件在事件前几秒确实看不出区别。")
    lines.append("这类样本不应强行归为锚点晚，也不应要求模型在没有可见证据时预测完整大响应；可以单独审核是否后移观察点或延长观察时长。")
    lines.append("")
    lines.append("## 方法边界")
    lines.append("")
    lines.append("- 本轮不训练模型、不修改标签、不改 formal headline。")
    lines.append("- 不重启硬响应类型分类。")
    lines.append("- 不把简单多候选轨迹输出作为主线。")
    lines.append("- 后移观察点不是为了刷分，而是把任务拆成不同可观测性层级：提前预测、短观察后预测、已响应补全。")
    lines.append("")
    lines.append("## 输出文件")
    lines.append("")
    lines.append(f"- 样本策略表：`{TABLES / 'v233_anchor_observation_policy_table.csv'}`")
    lines.append(f"- 人工审核表：`{TABLES / 'v233_anchor_observation_policy_review_table.csv'}`")
    lines.append(f"- 观察延迟表：`{TABLES / 'v233_observe_delay_grid.csv'}`")
    lines.append(f"- 图目录：`{FIGURES}`")
    if contact_sheet is not None:
        lines.append(f"- 策略图拼接总览：`{contact_sheet}`")
    lines.append(f"- ZIP 包：`{OUT / 'v233_adaptive_anchor_observation_policy_pack.zip'}`")
    lines.append("")
    lines.append("## 策略分布")
    lines.append("")
    counts = policy_df["anchor_observation_policy"].value_counts()
    lines.append("|policy|count|")
    lines.append("|---|---:|")
    for policy, count in counts.items():
        lines.append(f"|{policy}|{int(count)}|")
    lines.append("")
    lines.append("## 人工审核重点")
    lines.append("")
    if review_df.empty:
        lines.append("本轮没有需要人工审核的策略候选。")
    else:
        lines.append("|rank|policy|sample_id|old_anchor_s|suggest_delay|pre3/post03|post_peak|reason|")
        lines.append("|---:|---|---|---:|---:|---:|---:|---|")
        for row in review_df.itertuples(index=False):
            lines.append(
                f"|{int(row.policy_rank)}|{row.anchor_observation_policy}|`{row.sample_id}`|"
                f"{float(row.old_anchor_s):.3f}|"
                f"{'' if not np.isfinite(row.suggested_observe_delay_s) else f'{float(row.suggested_observe_delay_s):.1f}'}|"
                f"{float(row.pre3_to_post03_peak_ratio):.3f}|{float(row.future_peak_abs_delta):.3f}|{row.policy_reason_cn}|"
            )
    lines.append("")
    lines.append("## 解释")
    lines.append("")
    lines.append("如果样本属于 `reanchor_earlier_review`，说明旧锚点前已经能看到较强变化，应优先审核是否提前重锚定。")
    lines.append("如果样本属于 `observe_later_review`，说明旧锚点前证据弱但后续变化大，适合审核是否把观察点后移 0.5-2 秒，再做预测。")
    lines.append("这两类不能混在一起：前者是标注/事件起点问题，后者是任务可观测性问题。")
    lines.append("")
    lines.append("## 下一步")
    lines.append("")
    lines.append("人工先看 `observe_later_review` 的图：如果旧锚点前确实看不出区别，但后移 0.5-1.5 秒后可见响应证据，则应建立一个单独的“短观察后预测”评估层，而不是把它和纯提前预测混在同一指标里。")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def write_zip() -> Path:
    zip_path = OUT / "v233_adaptive_anchor_observation_policy_pack.zip"
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
    eval_df = pd.read_csv(v232.EVAL_PATH, encoding="utf-8-sig")
    casebook_df = pd.read_csv(v232.CASEBOOK_PATH, encoding="utf-8-sig")
    v231_meta = pd.read_csv(v232.V231_META_PATH, encoding="utf-8-sig")
    targets = v232.build_targets(eval_df, casebook_df, v231_meta)

    raw_cache = {}
    scored_rows = []
    grid_frames = []
    errors = []
    for row in targets.itertuples(index=False):
        target = pd.Series(row._asdict())
        try:
            scored, grid = v232.score_one_sample(target, raw_cache)
            scored_rows.append(scored)
            grid_frames.append(grid)
        except Exception as exc:
            errors.append({"sample_id": target.get("sample_id", ""), "error": repr(exc)})

    scored = pd.DataFrame(scored_rows)
    grid_all = pd.concat(grid_frames, ignore_index=True) if grid_frames else pd.DataFrame()

    delay_rows = []
    policy_rows = []
    for row in scored.itertuples(index=False):
        s = pd.Series(row._asdict())
        sample_id = str(s["sample_id"])
        sample_grid = grid_all[grid_all["sample_id"] == sample_id].copy()
        future_peak = max(float(s["post_0_3_peak_abs_delta"]), float(s["post_3_8_peak_abs_delta"]))
        pre_ratio = finite_ratio(float(s["pre_3_0_peak_abs_delta"]), float(s["post_0_3_peak_abs_delta"]))

        sample_delay_rows = []
        for delay in OBSERVE_DELAYS:
            delta_at_delay = nearest_value(sample_grid, "t_rel_old_anchor_s", "steering_delta_from_baseline", delay)
            rate_at_delay = nearest_value(sample_grid, "t_rel_old_anchor_s", "steering_delta_rate", delay)
            row_delay = {
                "sample_id": sample_id,
                "old_anchor_s": float(s["old_anchor_s"]),
                "observe_delay_s": delay,
                "observe_anchor_s": float(s["old_anchor_s"]) + delay,
                "steering_delta_at_delay": delta_at_delay,
                "abs_delta_at_delay": abs(delta_at_delay) if np.isfinite(delta_at_delay) else math.nan,
                "steering_delta_rate_at_delay": rate_at_delay,
                "future_peak_abs_delta": future_peak,
                "visible_fraction_of_future_peak": finite_ratio(abs(delta_at_delay), future_peak) if np.isfinite(delta_at_delay) else math.nan,
            }
            sample_delay_rows.append(row_delay)
            delay_rows.append(row_delay)

        policy, reason = classify_policy(s)
        sample_delay_df = pd.DataFrame(sample_delay_rows)
        suggested_delay, suggested_reason = choose_observe_delay(sample_delay_df, future_peak)
        # 只有 observe_later 才建议后移观察点；其他策略保留延迟证据但不建议采用。
        if policy != "observe_later_review":
            suggested_delay = math.nan
            suggested_reason = "非后移观察候选；先按对应策略人工审核"

        policy_row = s.to_dict()
        policy_row.update(
            {
                "future_peak_abs_delta": future_peak,
                "pre3_to_post03_peak_ratio": pre_ratio,
                "anchor_observation_policy": policy,
                "policy_reason_cn": reason,
                "suggested_observe_delay_s": suggested_delay,
                "suggested_observe_anchor_s": float(s["old_anchor_s"]) + suggested_delay if np.isfinite(suggested_delay) else math.nan,
                "suggested_delay_reason_cn": suggested_reason,
                "human_policy_decision": "",
                "human_final_anchor_s": "",
                "human_observe_delay_s": "",
                "human_use_for_training": "",
                "human_note_cn": "",
            }
        )
        policy_rows.append(policy_row)

    policy_df = pd.DataFrame(policy_rows)
    order = {
        "reanchor_earlier_review": 0,
        "reanchor_earlier_or_ambiguous_review": 1,
        "observe_later_review": 2,
        "large_change_standard_or_ambiguous": 3,
        "standard_anchor_review": 4,
    }
    policy_df["policy_order"] = policy_df["anchor_observation_policy"].map(order).fillna(9)
    policy_df = policy_df.sort_values(
        ["policy_order", "future_peak_abs_delta", "eval_tail_rmse"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    policy_df.insert(0, "policy_rank", range(1, len(policy_df) + 1))

    review_df = policy_df[policy_df["anchor_observation_policy"].isin(
        ["reanchor_earlier_review", "reanchor_earlier_or_ambiguous_review", "observe_later_review"]
    )].copy()

    delay_df = pd.DataFrame(delay_rows)
    policy_path = TABLES / "v233_anchor_observation_policy_table.csv"
    review_path = TABLES / "v233_anchor_observation_policy_review_table.csv"
    delay_path = TABLES / "v233_observe_delay_grid.csv"
    policy_df.to_csv(policy_path, index=False, encoding="utf-8-sig")
    review_df.to_csv(review_path, index=False, encoding="utf-8-sig")
    delay_df.to_csv(delay_path, index=False, encoding="utf-8-sig")
    (LOGS / "v233_errors.json").write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")

    figure_paths = []
    for row in review_df.itertuples(index=False):
        # 图太多会降低审核效率；保留前 14 个，其中 observe_later 会按排序自然包含。
        if len(figure_paths) >= 14:
            break
        figure_paths.append(plot_policy_case(pd.Series(row._asdict()), grid_all))
    contact_sheet = make_contact_sheet(figure_paths)
    report_path = write_report(policy_df, review_df, contact_sheet)
    zip_path = write_zip()

    manifest = {
        "version": "v233_adaptive_anchor_observation_policy_20260624",
        "sample_count": int(len(policy_df)),
        "review_count": int(len(review_df)),
        "observe_later_count": int((policy_df["anchor_observation_policy"] == "observe_later_review").sum()),
        "reanchor_review_count": int(policy_df["anchor_observation_policy"].str.contains("reanchor").sum()),
        "figure_count": int(len(figure_paths)),
        "errors": errors,
        "outputs": {
            "policy": str(policy_path),
            "review": str(review_path),
            "delay": str(delay_path),
            "report": str(report_path),
            "contact_sheet": str(contact_sheet) if contact_sheet else "",
            "zip": str(zip_path),
        },
    }
    (LOGS / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("DONE v233")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(policy_df[[
        "policy_rank",
        "anchor_observation_policy",
        "sample_id",
        "old_anchor_s",
        "suggested_observe_delay_s",
        "pre3_to_post03_peak_ratio",
        "future_peak_abs_delta",
        "policy_reason_cn",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
