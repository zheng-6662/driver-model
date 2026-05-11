# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch


ROOT = Path("F:/data_set_process/data_process")
REPORT = ROOT / "04_project_logs" / "reports"
OUT_DIR = REPORT / "physio_to_g14_progress_review_20260511"
OUT_DIR.mkdir(parents=True, exist_ok=True)


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


def load_version_log() -> pd.DataFrame:
    path = REPORT / "current_model_version_result_log_20260509.csv"
    return pd.read_csv(path, dtype=str)


def pick(df: pd.DataFrame, version: str) -> dict[str, str]:
    row = df[df["version"].eq(version)].iloc[0].to_dict()
    return row


def build_rows() -> pd.DataFrame:
    df = load_version_log()
    specs = [
        {
            "验证问题": "粗细双头基础",
            "版本": "E1",
            "模型": "粗细双头，不加连续风格，不加生理",
            "连续风格": "无",
            "生理/脑电": "无",
            "作用": "结构基础对照",
        },
        {
            "验证问题": "连续风格作用",
            "版本": "E2",
            "模型": "粗细双头 + 连续风格，不加生理",
            "连续风格": "有",
            "生理/脑电": "无",
            "作用": "连续风格基准",
        },
        {
            "验证问题": "非脑电生理状态",
            "版本": "E3",
            "模型": "粗细双头 + 无脑电生理状态 + 连续风格",
            "连续风格": "有",
            "生理/脑电": "无脑电生理状态",
            "作用": "验证非脑电状态是否直接有效",
        },
        {
            "验证问题": "含脑电生理状态",
            "版本": "E4",
            "模型": "粗细双头 + 含脑电生理状态 + 连续风格",
            "连续风格": "有",
            "生理/脑电": "含脑电生理状态",
            "作用": "验证脑电是否带来信息",
        },
        {
            "验证问题": "非脑电原始融合",
            "版本": "E7C",
            "模型": "粗细双头 + 心率/皮电/肌电融合 + 连续风格",
            "连续风格": "有",
            "生理/脑电": "心率+皮电+肌电",
            "作用": "验证简单多生理融合",
        },
        {
            "验证问题": "心率单信号",
            "版本": "E10A",
            "模型": "粗细双头 + 心率单信号 + 连续风格",
            "连续风格": "有",
            "生理/脑电": "心率",
            "作用": "单信号归因",
        },
        {
            "验证问题": "皮电单信号",
            "版本": "E10B",
            "模型": "粗细双头 + 皮电单信号 + 连续风格",
            "连续风格": "有",
            "生理/脑电": "皮电",
            "作用": "单信号归因",
        },
        {
            "验证问题": "肌电单信号",
            "版本": "E10C",
            "模型": "粗细双头 + 肌电单信号 + 连续风格",
            "连续风格": "有",
            "生理/脑电": "肌电",
            "作用": "当前非脑电最强候选",
        },
    ]
    rows = []
    for spec in specs:
        row = pick(df, spec["版本"])
        rows.append(
            {
                "验证问题": spec["验证问题"],
                "模型": spec["模型"],
                "连续风格": spec["连续风格"],
                "生理/脑电": spec["生理/脑电"],
                "test steer RMSE": row["test_rmse"],
                "primary RMSE": row["primary_rmse"],
                "tail RMSE": row["tail_rmse"],
                "selection": row["selection"],
                "结论": row["decision"],
            }
        )
    return pd.DataFrame(rows)


def render(df: pd.DataFrame) -> None:
    csv_path = OUT_DIR / "05_physio_ablation_table.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    plot_df = df[["验证问题", "模型", "生理/脑电", "test steer RMSE", "primary RMSE", "tail RMSE", "selection"]].copy()

    fig_w = 17.2
    fig_h = 1.25 + len(plot_df) * 0.72
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=180)
    fig.patch.set_facecolor("#101112")
    ax.set_facecolor("#101112")
    ax.axis("off")

    bg = FancyBboxPatch(
        (0.01, 0.01),
        0.98,
        0.98,
        boxstyle="round,pad=0.012,rounding_size=0.035",
        linewidth=0,
        facecolor="#171819",
        transform=ax.transAxes,
        zorder=-1,
    )
    ax.add_patch(bg)

    table = ax.table(
        cellText=plot_df.values.tolist(),
        colLabels=list(plot_df.columns),
        colWidths=[0.15, 0.32, 0.18, 0.13, 0.13, 0.12, 0.11],
        loc="center",
        cellLoc="left",
        bbox=[0.02, 0.10, 0.96, 0.80],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    data = plot_df.values.tolist()
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#2b2d2f")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#252627")
            cell.get_text().set_color("#eeeeee")
            cell.get_text().set_weight("bold")
            cell.get_text().set_fontsize(10.2)
        else:
            model = str(data[row - 1][1])
            if "肌电单信号" in model:
                face = "#18241d"
            elif "含脑电" in model:
                face = "#17212a"
            elif "无脑电生理" in model or "心率/皮电/肌电" in model:
                face = "#241b1b"
            else:
                face = "#171819" if row % 2 else "#151617"
            cell.set_facecolor(face)
            cell.get_text().set_color("#d7d7d7")
            cell.get_text().set_fontsize(9.2)
        cell._loc = "center" if col >= 3 else "left"

    ax.text(
        0.025,
        0.955,
        "生理信号作用验证表：这里的生理版本都接在粗细双头结构上",
        color="#f2f2f2",
        fontsize=15,
        fontweight="bold",
        transform=ax.transAxes,
        va="center",
    )
    ax.text(
        0.025,
        0.045,
        "说明：该表使用 current_model_version_result_log_20260509.csv 的三种子结果。红色系表示当前生理表示较弱，绿色表示当前最强非脑电生理信号，蓝色表示含脑电状态。",
        color="#aeb4bb",
        fontsize=9.2,
        transform=ax.transAxes,
        va="center",
    )

    png_path = OUT_DIR / "05_physio_ablation_table.png"
    fig.savefig(png_path, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(png_path)
    print(csv_path)


def main() -> None:
    configure_font()
    render(build_rows())


if __name__ == "__main__":
    main()
