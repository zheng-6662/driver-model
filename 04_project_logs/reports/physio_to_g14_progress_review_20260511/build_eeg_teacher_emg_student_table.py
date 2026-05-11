# -*- coding: utf-8 -*-
from __future__ import annotations

import textwrap
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
        font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
        plt.rcParams["font.family"] = font_name
    plt.rcParams["axes.unicode_minus"] = False


def load_current_rows() -> dict[str, dict[str, str]]:
    path = REPORT / "current_model_version_result_log_20260509.csv"
    df = pd.read_csv(path, dtype=str)
    return {row["version"]: row.to_dict() for _, row in df.iterrows()}


def metric(row: dict[str, str], key: str) -> str:
    value = row.get(key, "")
    return "-" if value in {"", "nan", None} else str(value)


def build_rows() -> pd.DataFrame:
    current = load_current_rows()
    specs = [
        {
            "验证问题": "连续风格基准",
            "版本": "E2",
            "模型": "粗细双头 + 连续驾驶风格",
            "训练阶段使用信息": "车辆历史 + 连续风格",
            "推理阶段使用信息": "车辆历史 + 连续风格",
            "结果来源": "当前三种子",
            "判断": "作为后续脑电/肌电比较基准",
        },
        {
            "验证问题": "无脑电状态对照",
            "版本": "E3",
            "模型": "粗细双头 + 无脑电生理状态 + 连续风格",
            "训练阶段使用信息": "车辆历史 + 连续风格 + 无脑电生理状态",
            "推理阶段使用信息": "车辆历史 + 连续风格 + 无脑电生理状态",
            "结果来源": "当前三种子",
            "判断": "当前构造较弱，不作为主线",
        },
        {
            "验证问题": "全程使用脑电",
            "版本": "E4",
            "模型": "粗细双头 + 含脑电生理状态 + 连续风格",
            "训练阶段使用信息": "车辆历史 + 连续风格 + 含脑电状态",
            "推理阶段使用信息": "车辆历史 + 连续风格 + 含脑电状态",
            "结果来源": "当前三种子",
            "判断": "好于 E3，证明脑电含有效信息，但直接全程输入不是最强",
        },
        {
            "验证问题": "脑电教师",
            "版本": "E5A",
            "模型": "脑电教师蒸馏，无脑电学生",
            "训练阶段使用信息": "脑电教师监督学生",
            "推理阶段使用信息": "车辆历史 + 连续风格，不用脑电",
            "结果来源": "当前三种子",
            "判断": "当前最重要证据：训练用脑电，推理不用脑电也能提升",
        },
        {
            "验证问题": "脑电教师 + 物理约束",
            "版本": "E6",
            "模型": "E5A + 幅值/方向物理损失",
            "训练阶段使用信息": "脑电教师 + 物理约束",
            "推理阶段使用信息": "车辆历史 + 连续风格，不用脑电",
            "结果来源": "当前三种子",
            "判断": "物理更均衡，但整体 RMSE 略弱于 E5A",
        },
        {
            "验证问题": "肌电推理信号",
            "版本": "E10C",
            "模型": "粗细双头 + 肌电单信号 + 连续风格",
            "训练阶段使用信息": "车辆历史 + 连续风格 + 肌电",
            "推理阶段使用信息": "车辆历史 + 连续风格 + 肌电",
            "结果来源": "当前三种子",
            "判断": "当前最强非脑电推理期生理信号",
        },
        {
            "验证问题": "脑电教师 + 肌电学生",
            "版本": "E11A",
            "模型": "脑电教师 + 肌电学生 + 连续风格",
            "训练阶段使用信息": "脑电教师 + 肌电学生",
            "推理阶段使用信息": "车辆历史 + 连续风格 + 肌电",
            "结果来源": "当前单次 seed2026",
            "判断": "整体未超过 E5A/E10C，只作为诊断证据",
        },
    ]

    rows = []
    for spec in specs:
        version = spec["版本"]
        row = current[version]
        rows.append(
            {
                **spec,
                "test steer RMSE": metric(row, "test_rmse"),
                "primary RMSE": metric(row, "primary_rmse"),
                "tail RMSE": metric(row, "tail_rmse"),
                "selection": metric(row, "selection"),
            }
        )
    return pd.DataFrame(rows)


def wrap_cell(value: object, width: int) -> str:
    return "\n".join(textwrap.wrap(str(value), width=width, break_long_words=False, replace_whitespace=False))


def render_table(df: pd.DataFrame) -> None:
    csv_path = OUT_DIR / "07_eeg_teacher_emg_student_comparison_table.csv"
    png_path = OUT_DIR / "07_eeg_teacher_emg_student_comparison_table.png"
    selected_png = SELECTED_DIR / "20_eeg_teacher_emg_student_comparison_table.png"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    plot_df = df[
        [
            "验证问题",
            "版本",
            "模型",
            "训练阶段使用信息",
            "推理阶段使用信息",
            "结果来源",
            "test steer RMSE",
            "primary RMSE",
            "tail RMSE",
            "selection",
        ]
    ].copy()

    wrap_widths = {
        "验证问题": 9,
        "版本": 8,
        "模型": 22,
        "训练阶段使用信息": 18,
        "推理阶段使用信息": 18,
        "结果来源": 12,
        "test steer RMSE": 14,
        "primary RMSE": 14,
        "tail RMSE": 14,
        "selection": 14,
    }
    for col, width in wrap_widths.items():
        plot_df[col] = plot_df[col].map(lambda value, w=width: wrap_cell(value, w))

    fig_w = 21.0
    fig_h = 1.6 + len(plot_df) * 0.92
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
        colWidths=[0.095, 0.060, 0.205, 0.170, 0.170, 0.105, 0.100, 0.095, 0.085, 0.085],
        loc="center",
        cellLoc="left",
        bbox=[0.02, 0.18, 0.96, 0.70],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.8)

    raw_rows = df.to_dict("records")
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d9dee5")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#eef1f5")
            cell.get_text().set_color("#20242a")
            cell.get_text().set_weight("bold")
            cell.get_text().set_fontsize(9.0)
        else:
            version = raw_rows[row - 1]["版本"]
            source = raw_rows[row - 1]["结果来源"]
            if source.startswith("当前单次"):
                face = "#fff4df"
            elif version in {"E5A", "E6", "E10C"}:
                face = "#e8f5ec"
            elif version == "E4":
                face = "#eaf3ff"
            elif version == "E3":
                face = "#fdecec"
            else:
                face = "#ffffff" if row % 2 else "#f8fafc"
            cell.set_facecolor(face)
            cell.get_text().set_color("#222831")
            cell.get_text().set_fontsize(8.4)
        cell._loc = "center" if col >= 5 else "left"

    ax.text(
        0.025,
        0.945,
        "第二阶段：脑电使用方式对比（全程脑电、脑电教师、肌电学生）",
        color="#111827",
        fontsize=15,
        fontweight="bold",
        transform=ax.transAxes,
        va="center",
    )

    note = (
        "说明：E4 表示训练和推理阶段都使用含脑电状态；E5A/E6 表示训练阶段使用脑电教师，"
        "推理阶段不用脑电；E10C 表示肌电作为推理期可用生理信号；E11A 只有 seed2026，"
        "不能和三种子主结论等价使用。"
    )
    conclusion = (
        "结论：脑电确实含有有效信息，但目前更适合作为训练阶段教师信号，而不是默认推理阶段必须全程输入。"
        "肌电是当前最强的非脑电推理期生理信号；脑电教师 + 肌电学生的简单组合没有超过 E5A/E10C，"
        "后续应转向困难样本和响应类型的选择性利用。"
    )
    ax.text(
        0.025,
        0.105,
        textwrap.fill(note, width=105),
        color="#4b5563",
        fontsize=9.2,
        transform=ax.transAxes,
        va="center",
    )
    ax.text(
        0.025,
        0.055,
        textwrap.fill(conclusion, width=102),
        color="#111827",
        fontsize=11.0,
        fontweight="bold",
        transform=ax.transAxes,
        va="center",
    )

    fig.savefig(png_path, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.08)
    fig.savefig(selected_png, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(png_path)
    print(csv_path)
    print(selected_png)


def write_slide_text(df: pd.DataFrame) -> None:
    md_path = OUT_DIR / "07_补充说明_脑电教师与肌电学生_CN.md"
    rows_md = dataframe_to_markdown(df)
    text = f"""# 第二阶段：脑电使用方式对比

整理时间：2026-05-11

## 这一页要回答的问题

上一页已经说明：生理数据不是简单无效，其中脑电和肌电最值得继续看。下一步的问题不是继续堆更多生理特征，而是比较这些信号应该如何进入模型：

1. 脑电是否要在推理阶段全程使用；
2. 脑电是否更适合只在训练阶段作为教师；
3. 肌电能否作为推理阶段可用的学生信号；
4. 脑电教师和肌电学生简单组合后是否能进一步超过主线。

## 推荐放入 PPT 的表

{rows_md}

## 页面底部总结文字

对比 E3/E4 可以看到，含脑电状态明显好于无脑电生理状态，说明脑电中确实包含和方向盘响应相关的有效信息。但 E4 并不是最强结果，更关键的是 E5A：训练阶段利用脑电教师监督学生，推理阶段不再需要脑电，反而取得了更低的整体误差和尾段误差。因此，脑电当前更适合作为训练阶段的教师信息，而不是默认部署时必须全程输入的传感器。

另一方面，E10C 说明肌电单信号是当前最强的非脑电推理期生理信号，三种子结果稳定；但 E11A 的“脑电教师 + 肌电学生”只有 seed2026，且整体没有超过 E5A/E10C。因此这一阶段的结论不是“脑电和肌电简单叠加就更好”，而是：脑电教师和肌电组合可能有诊断价值，后续需要针对困难样本和响应类型做选择性利用。

## 口头汇报稿

上一页主要验证的是生理数据本身是否有价值。结果显示，脑电和肌电是比较值得继续看的两个信号。所以我进一步比较了脑电的几种使用方式：一种是推理阶段也全程使用脑电状态，也就是 E4；一种是只在训练阶段使用脑电作为教师，推理阶段不用脑电，也就是 E5A/E6；还有一种是让脑电教师去指导肌电学生，也就是 E11A。

从结果看，E4 相比无脑电状态 E3 有明显改善，说明脑电确实有信息。但直接全程使用脑电不是最优，E5A 在推理阶段不需要脑电，却取得了更好的整体误差和尾段误差，说明脑电更适合做训练阶段的教师信号。肌电方面，E10C 三种子结果稳定，是当前最强的非脑电推理期生理信号；不过 E11A 这个脑电教师 + 肌电学生版本没有超过 E5A 或 E10C，所以目前不能把它作为新主线，只能作为后续困难样本分析的诊断依据。

## 一句话结论

脑电的主要价值目前体现在训练阶段教师监督；肌电是推理阶段最值得保留的非脑电生理信号；但脑电教师和肌电学生的简单叠加尚未带来稳定收益。
"""
    md_path.write_text(text, encoding="utf-8")
    print(md_path)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        values = [str(row[col]).replace("\n", "<br>").replace("|", "｜") for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    configure_font()
    df = build_rows()
    render_table(df)
    write_slide_text(df)


if __name__ == "__main__":
    main()
