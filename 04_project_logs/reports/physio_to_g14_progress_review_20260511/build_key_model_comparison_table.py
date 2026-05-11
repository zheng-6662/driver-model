# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import textwrap
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
        font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
        plt.rcParams["font.family"] = font_name
    plt.rcParams["axes.unicode_minus"] = False


def add_row(rows: list[dict[str, object]], model: str, test: float, primary: float, tail: float, selection: float, source: str) -> None:
    rows.append(
        {
            "模型": model,
            "test steer RMSE": float(test),
            "primary RMSE": float(primary),
            "tail RMSE": float(tail),
            "selection": float(selection),
            "数据来源": source,
        }
    )


def load_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    # 连续驾驶风格验证。
    df = pd.read_csv(REPORT / "style_physio_eeg_e0_e2_summary_fresh_3seed_20260507" / "mean_std_metrics.csv")
    names = {
        "E0": "01 E0 直接预测+连续风格",
        "E1": "02 E1 粗细双头，无连续风格",
        "E2": "03 E2 粗细双头+连续风格",
    }
    for eid in ["E0", "E1", "E2"]:
        r = df[df["experiment_id"].eq(eid)].iloc[0]
        add_row(rows, names[eid], r.test_steer_rmse_mean, r.primary_rmse_mean, r.tail_rmse_mean, r.selection_mean, "三种子均值")

    # 脑电与脑电教师。
    df = pd.read_csv(REPORT / "style_physio_eeg_e5_distill_summary_20260508" / "mean_std_metrics.csv")
    names = {
        "E3": "04 E3 粗细双头+无脑电生理状态+连续风格",
        "E4": "05 E4 粗细双头+含脑电生理状态+连续风格",
        "E5A": "06 E5A 脑电教师，无脑电推理",
    }
    for eid in ["E3", "E4", "E5A"]:
        r = df[df["experiment_id"].eq(eid)].iloc[0]
        add_row(rows, names[eid], r.test_steer_rmse_mean, r.primary_rmse_mean, r.tail_rmse_mean, r.selection_mean, "三种子均值")

    # 脑电教师 + 物理约束。
    df = pd.read_csv(REPORT / "style_physio_eeg_e6_physical_repair_summary_20260508" / "e6_seed_wise_metrics.csv")
    part = df[df["experiment_id"].eq("E6")]
    add_row(rows, "07 E6 脑电教师+物理约束", part["test_rmse"].mean(), part["primary"].mean(), part["tail"].mean(), part["selection"].mean(), "三种子均值")

    # 非脑电生理信号。
    df = pd.read_csv(REPORT / "style_physio_eeg_e10c_emg_only_3seed_summary_20260509" / "seed_wise_metrics.csv")
    for eid, label in [
        ("E7C", "08 E7C 粗细双头+心率/皮电/肌电融合"),
        ("E10A", "09 E10A 粗细双头+心率单信号"),
        ("E10B", "10 E10B 粗细双头+皮电单信号"),
        ("E10C", "11 E10C 粗细双头+肌电单信号+连续风格"),
    ]:
        part = df[df["experiment_id"].eq(eid)]
        source = "三种子均值" if len(part) >= 3 else "seed2026"
        add_row(rows, label, part["test_steer_rmse"].mean(), part["primary_rmse"].mean(), part["tail_rmse"].mean(), part["selection"].mean(), source)

    # 脑电教师 + 肌电学生。
    df = pd.read_csv(REPORT / "style_physio_eeg_e11_emg_distill_summary_20260509" / "seed_wise_metrics.csv")
    part = df[df["experiment_id"].eq("E11A")]
    if not part.empty:
        r = part.iloc[0]
        add_row(rows, "12 E11A 脑电教师+肌电学生（seed2026）", r.test_steer_rmse, r.primary_rmse, r.tail_rmse, r.selection, "seed2026")

    # G13：从 metrics.json 读取同口径 primary RMSE。
    df = pd.read_csv(REPORT / "g13_model_breakthrough_20260510" / "g13_hi_multiseed_summary_20260510" / "g13_hi_seed_wise_metrics.csv")
    for eid, label in [
        ("G13H", "13 G13H 响应类型辅助+脑电教师+肌电"),
        ("G13I", "14 G13I 困难响应加权+物理约束"),
    ]:
        vals = []
        for _, r in df[df["experiment_id"].eq(eid)].iterrows():
            metrics_path = Path(str(r["local_run_root"])) / "metrics.json"
            if metrics_path.exists():
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                sel = metrics["test"]["selection_summary"]
                vals.append(
                    {
                        "test": float(metrics["test"]["steer_rmse"]),
                        "primary": float(sel["overall_primary_steer_rmse"]),
                        "tail": float(sel["rmse_tail_abs_steer"]),
                        "selection": float(sel["selection_score"]),
                    }
                )
        vdf = pd.DataFrame(vals)
        add_row(rows, label, vdf["test"].mean(), vdf["primary"].mean(), vdf["tail"].mean(), vdf["selection"].mean(), "三种子均值")

    # G14：seed2026 筛选。
    df = pd.read_csv(REPORT / "g14_non_average_prediction_20260510" / "g14_seed2026_screening_summary" / "g14_seed2026_overall.csv")
    for eid, label in [
        ("G14C", "15 G14C 多候选轨迹+幅值方向约束（seed2026）"),
        ("G14G", "16 G14G 脑电教师+肌电+响应原型（seed2026）"),
    ]:
        r = df[df["version"].eq(eid)].iloc[0]
        add_row(rows, label, r.test_rmse, r.primary_rmse, r.tail_rmse, r.selection, "seed2026")

    return pd.DataFrame(rows)


def render_table(df: pd.DataFrame) -> None:
    out_csv = OUT_DIR / "04_key_model_comparison_table.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    plot_df = df[["模型", "test steer RMSE", "primary RMSE", "tail RMSE", "selection"]].copy()
    for col in ["test steer RMSE", "primary RMSE", "tail RMSE", "selection"]:
        plot_df[col] = plot_df[col].map(lambda x: f"{float(x):.4f}")
    plot_df["模型"] = plot_df["模型"].map(lambda s: "\n".join(textwrap.wrap(str(s), width=24, break_long_words=False)))

    nrows = len(plot_df)
    fig_w = 15.6
    fig_h = 1.2 + nrows * 0.56
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
        colWidths=[0.44, 0.18, 0.17, 0.15, 0.13],
        loc="center",
        cellLoc="left",
        bbox=[0.02, 0.08, 0.96, 0.84],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    cell_text = plot_df.values.tolist()
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#2b2d2f")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#252627")
            cell.get_text().set_color("#eeeeee")
            cell.get_text().set_weight("bold")
            cell.get_text().set_fontsize(10.5)
        else:
            model_text = str(cell_text[row - 1][0]) if row - 1 < len(cell_text) else ""
            if any(k in model_text for k in ["E5A", "E6", "E10C"]):
                face = "#18241d"
            elif any(k in model_text for k in ["G14G", "G14C"]):
                face = "#17212a"
            else:
                face = "#171819" if row % 2 else "#151617"
            cell.set_facecolor(face)
            cell.get_text().set_color("#d7d7d7")
            cell.get_text().set_fontsize(9.25)
        cell._loc = "center" if col > 0 else "left"

    ax.text(
        0.025,
        0.955,
        "关键模型结果对比（从生理信号验证到 G14）",
        color="#f2f2f2",
        fontsize=15,
        fontweight="bold",
        transform=ax.transAxes,
        va="center",
    )
    ax.text(
        0.025,
        0.035,
        "说明：三种子版本显示均值；标注 seed2026 的版本只完成单种子。四个指标均按越低越好理解。绿色为当前主候选，蓝色为 G14 诊断候选。",
        color="#aeb4bb",
        fontsize=9.2,
        transform=ax.transAxes,
        va="center",
    )

    out_png = OUT_DIR / "04_key_model_comparison_table.png"
    fig.savefig(out_png, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(out_png)
    print(out_csv)


def main() -> None:
    configure_font()
    render_table(load_rows())


if __name__ == "__main__":
    main()
