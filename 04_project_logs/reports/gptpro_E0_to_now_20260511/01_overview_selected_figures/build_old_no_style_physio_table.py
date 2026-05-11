# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch


ROOT = Path("F:/data_set_process/data_process")
REPORT = ROOT / "04_project_logs" / "reports"
OUT_DIR = REPORT / "physio_to_g14_progress_review_20260511"
SELECTED_DIR = OUT_DIR / "selected_figures"
SELECTED_DIR.mkdir(parents=True, exist_ok=True)


def configure_font() -> None:
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    ]
    font_path = next((p for p in candidates if p.exists()), None)
    if font_path:
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams["axes.unicode_minus"] = False


def load_current(version: str) -> dict[str, str]:
    path = REPORT / "current_model_version_result_log_20260509.csv"
    df = pd.read_csv(path, dtype=str)
    row = df[df["version"].eq(version)]
    if row.empty:
        raise ValueError(f"版本不存在: {version}")
    return row.iloc[0].to_dict()


def load_fair10() -> dict[str, str]:
    path = (
        ROOT
        / "tmp"
        / "event_conditioned_runs"
        / "FAIR10_车辆数据_粗细双头_教师状态_无显式事件_20260428_035102"
        / "metrics.json"
    )
    metrics = json.loads(path.read_text(encoding="utf-8"))
    test = metrics["test"]
    sel = test["selection_summary"]
    return {
        "test_rmse": f'{test["steer_rmse"]:.4f}',
        "primary_rmse": f'{sel["overall_primary_steer_rmse"]:.4f}',
        "tail_rmse": f'{sel["rmse_tail_abs_steer"]:.4f}',
        "selection": f'{sel["selection_score"]:.4f}',
    }


def load_old_fair_physio(model: str) -> dict[str, str]:
    path = REPORT / "gptpro_physio_strategy_pack_20260429" / "evidence" / "fair_physio_comparison_table.csv"
    df = pd.read_csv(path, dtype=str)
    row = df[df["model"].eq(model)]
    if row.empty:
        raise ValueError(f"旧生理对照表中不存在: {model}")
    r = row.iloc[0].to_dict()
    return {
        "test_rmse": r["test_steer_rmse"],
        "primary_rmse": r["primary_rmse"],
        "tail_rmse": r["tail_rmse"],
        "selection": r["selection"],
    }


def current_metrics(version: str) -> dict[str, str]:
    row = load_current(version)
    return {
        "test_rmse": row["test_rmse"],
        "primary_rmse": row["primary_rmse"],
        "tail_rmse": row["tail_rmse"],
        "selection": row["selection"],
    }


def build_rows() -> pd.DataFrame:
    specs = [
        {
            "验证问题": "基础对照",
            "版本": "E1",
            "模型": "粗细双头，不加连续风格，不加生理",
            "连续风格": "无",
            "生理/脑电": "无",
            "结果来源": "当前三种子",
            "metrics": current_metrics("E1"),
            "结论": "粗细双头基础版本",
        },
        {
            "验证问题": "旧无风格生理对照",
            "版本": "FAIR10",
            "模型": "粗细双头 + 生理/脑电PCA状态，不加连续风格",
            "连续风格": "无",
            "生理/脑电": "PCA状态，含生理和脑电",
            "结果来源": "旧单次 seed2026",
            "metrics": load_fair10(),
            "结论": "效果较弱，不适合作为主线证据",
        },
        {
            "验证问题": "旧无风格生理对照",
            "版本": "FAIR15",
            "模型": "粗细双头 + 基线校正生理状态，不加连续风格",
            "连续风格": "无",
            "生理/脑电": "基线校正生理状态",
            "结果来源": "旧单次 seed2026",
            "metrics": load_old_fair_physio("FAIR15"),
            "结论": "效果较弱，说明这种生理构造单独不够",
        },
        {
            "验证问题": "连续风格基准",
            "版本": "E2",
            "模型": "粗细双头 + 连续风格，不加生理",
            "连续风格": "有",
            "生理/脑电": "无",
            "结果来源": "当前三种子",
            "metrics": current_metrics("E2"),
            "结论": "当前主要基础基准",
        },
        {
            "验证问题": "无脑电生理状态",
            "版本": "E3",
            "模型": "粗细双头 + 无脑电生理状态 + 连续风格",
            "连续风格": "有",
            "生理/脑电": "无脑电生理状态",
            "结果来源": "当前三种子",
            "metrics": current_metrics("E3"),
            "结论": "当前构造下不作主线",
        },
        {
            "验证问题": "含脑电生理状态",
            "版本": "E4",
            "模型": "粗细双头 + 含脑电生理状态 + 连续风格",
            "连续风格": "有",
            "生理/脑电": "含脑电生理状态",
            "结果来源": "当前三种子",
            "metrics": current_metrics("E4"),
            "结论": "好于 E3，说明脑电有信息",
        },
        {
            "验证问题": "非脑电融合",
            "版本": "E7C",
            "模型": "粗细双头 + 心率/皮电/肌电融合 + 连续风格",
            "连续风格": "有",
            "生理/脑电": "心率+皮电+肌电",
            "结果来源": "当前三种子",
            "metrics": current_metrics("E7C"),
            "结论": "简单融合不适合作为主线",
        },
        {
            "验证问题": "心率单信号",
            "版本": "E10A",
            "模型": "粗细双头 + 心率单信号 + 连续风格",
            "连续风格": "有",
            "生理/脑电": "心率",
            "结果来源": "当前三种子",
            "metrics": current_metrics("E10A"),
            "结论": "弱证据，暂不作为主线",
        },
        {
            "验证问题": "皮电单信号",
            "版本": "E10B",
            "模型": "粗细双头 + 皮电单信号 + 连续风格",
            "连续风格": "有",
            "生理/脑电": "皮电",
            "结果来源": "当前三种子",
            "metrics": current_metrics("E10B"),
            "结论": "效果较弱，不作为主线",
        },
        {
            "验证问题": "肌电单信号",
            "版本": "E10C",
            "模型": "粗细双头 + 肌电单信号 + 连续风格",
            "连续风格": "有",
            "生理/脑电": "肌电",
            "结果来源": "当前三种子",
            "metrics": current_metrics("E10C"),
            "结论": "当前非脑电生理最强候选",
        },
    ]

    rows = []
    for spec in specs:
        metrics = spec.pop("metrics")
        rows.append(
            {
                **spec,
                "test steer RMSE": metrics["test_rmse"],
                "primary RMSE": metrics["primary_rmse"],
                "tail RMSE": metrics["tail_rmse"],
                "selection": metrics["selection"],
            }
        )
    return pd.DataFrame(rows)


def render(df: pd.DataFrame) -> None:
    csv_path = OUT_DIR / "06_physio_no_style_old_control_table.csv"
    png_path = OUT_DIR / "06_physio_no_style_old_control_table.png"
    selected_png = SELECTED_DIR / "19_physio_no_style_old_control_table.png"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    plot_df = df[
        [
            "验证问题",
            "版本",
            "模型",
            "连续风格",
            "生理/脑电",
            "结果来源",
            "test steer RMSE",
            "primary RMSE",
            "tail RMSE",
            "selection",
        ]
    ].copy()

    fig_w = 20.5
    fig_h = 1.35 + len(plot_df) * 0.72
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=180)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    ax.axis("off")

    bg = FancyBboxPatch(
        (0.01, 0.01),
        0.98,
        0.98,
        boxstyle="round,pad=0.012,rounding_size=0.03",
        linewidth=1.0,
        edgecolor="#d6dbe1",
        facecolor="#ffffff",
        transform=ax.transAxes,
        zorder=-1,
    )
    ax.add_patch(bg)

    table = ax.table(
        cellText=plot_df.values.tolist(),
        colLabels=list(plot_df.columns),
        colWidths=[0.12, 0.08, 0.30, 0.07, 0.15, 0.12, 0.10, 0.10, 0.09, 0.08],
        loc="center",
        cellLoc="left",
        bbox=[0.02, 0.105, 0.96, 0.79],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.0)
    data = plot_df.values.tolist()
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d9dee5")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#eef1f5")
            cell.get_text().set_color("#20242a")
            cell.get_text().set_weight("bold")
            cell.get_text().set_fontsize(9.4)
        else:
            version = str(data[row - 1][1])
            source = str(data[row - 1][5])
            if source.startswith("旧"):
                face = "#fff4df"
            elif version == "E10C":
                face = "#e8f5ec"
            elif version in {"E3", "E7C"}:
                face = "#fdecec"
            elif version == "E4":
                face = "#eaf3ff"
            else:
                face = "#ffffff" if row % 2 else "#f8fafc"
            cell.set_facecolor(face)
            cell.get_text().set_color("#222831")
            cell.get_text().set_fontsize(8.8)
        cell._loc = "center" if col >= 3 else "left"

    ax.text(
        0.025,
        0.955,
        "补充表：旧无连续风格生理对照 + 当前单信号归因结果",
        color="#111827",
        fontsize=15,
        fontweight="bold",
        transform=ax.transAxes,
        va="center",
    )
    ax.text(
        0.025,
        0.045,
        "说明：棕色行为旧单次无连续风格结果，不能和当前三种子结果等价使用；心率、皮电、肌电单信号行为当前三种子结果，均包含连续风格。",
        color="#4b5563",
        fontsize=9.2,
        transform=ax.transAxes,
        va="center",
    )

    fig.savefig(png_path, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.08)
    fig.savefig(selected_png, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(png_path)
    print(csv_path)
    print(selected_png)


def main() -> None:
    configure_font()
    render(build_rows())


if __name__ == "__main__":
    main()
