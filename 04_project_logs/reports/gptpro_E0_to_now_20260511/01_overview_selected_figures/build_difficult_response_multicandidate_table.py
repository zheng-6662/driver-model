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

G11_DIR = REPORT / "style_physio_eeg_g11_bad_case_attribution_20260509"
G12_DIR = REPORT / "style_physio_eeg_g12_response_gate_subject_audit_20260510"
G13_DIR = REPORT / "g13_model_breakthrough_20260510" / "g13_hi_multiseed_summary_20260510"
G14_DIR = REPORT / "g14_non_average_prediction_20260510"
G14_RETRIEVAL_DIR = G14_DIR / "retrieval_reference_stage1"


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


def fmt(x: object, digits: int = 4) -> str:
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def fmt_mean_std(mean: object, std: object) -> str:
    return f"{fmt(mean)}±{fmt(std)}"


def current_metric(version: str, metric: str) -> str:
    df = pd.read_csv(REPORT / "current_model_version_result_log_20260509.csv", dtype=str)
    row = df[df["version"].eq(version)].iloc[0]
    return str(row[metric])


def g11_failure_summary() -> str:
    df = pd.read_csv(G11_DIR / "failure_type_summary.csv")
    top = df.head(8)
    parts = [f"{r.failure_type}{int(r.n)}" for _, r in top.iterrows()]
    return "111个困难样本；" + "、".join(parts[:5]) + f"；零线两侧相反{int(top[top.failure_type.eq('零线两侧相反')].iloc[0].n)}"


def g12_subject_summary() -> str:
    df = pd.read_csv(G12_DIR / "subject_delta_vs_E2_mean3seed.csv")
    part = df[df["model_id"].eq("E10C")]
    parts = [f"{r.subject} {fmt(r.delta_rmse_vs_E2)}" for _, r in part.iterrows()]
    return "E10C相对E2三被试均改善：" + "，".join(parts)


def g12_classifier_summary() -> str:
    df = pd.read_csv(G12_DIR / "response_type_classifier_metrics.csv")
    feature = "车辆历史_连续风格_肌电"
    hard = df[(df["feature_set"].eq(feature)) & (df["target"].eq("困难风险代理")) & (df["split"].eq("test"))].iloc[0]
    reverse = df[(df["feature_set"].eq(feature)) & (df["target"].eq("反向修正")) & (df["split"].eq("test"))].iloc[0]
    multi = df[(df["feature_set"].eq(feature)) & (df["target"].eq("多段修正")) & (df["split"].eq("test"))].iloc[0]
    return (
        f"困难风险macroF1 {fmt(hard.macro_f1)}，G11 recall {fmt(hard.g11_bad_case_recall)}但precision {fmt(hard.g11_bad_case_precision)}；"
        f"反向修正recall {fmt(reverse.positive_recall)}，多段修正recall {fmt(multi.positive_recall)}"
    )


def g12_selector_summary() -> str:
    df = pd.read_csv(G12_DIR / "light_selector_metrics_seed2026.csv")
    row = df[df["candidate"].eq("加权融合_车辆历史_连续风格_肌电")].iloc[0]
    return (
        f"加权融合test RMSE {fmt(row.rmse_2s)}，G11 RMSE {fmt(row.g11_bad_case_rmse)}；"
        f"best expert accuracy约{fmt(row.best_expert_accuracy)}"
    )


def g13_summary(version: str) -> str:
    summary = pd.read_csv(G13_DIR / "g13_hi_three_seed_summary.csv")
    g11 = pd.read_csv(G13_DIR / "g13_hi_g11_mean.csv")
    r = summary[summary["experiment_id"].eq(version)].iloc[0]
    g = g11[g11["experiment_id"].eq(version)].iloc[0]
    seed2026 = pd.read_csv(G13_DIR / "g13_hi_seed_wise_metrics.csv")
    s = seed2026[(seed2026["experiment_id"].eq(version)) & (seed2026["seed"].eq(2026))].iloc[0]
    return (
        f"seed2026 {fmt(s.test_rmse)}；三种子test {fmt_mean_std(r.test_rmse_mean, r.test_rmse_std)}，"
        f"tail {fmt(r.tail_rmse_mean)}，G11 RMSE {fmt(g.g11_rmse_mean)}"
    )


def g14_retrieval_summary() -> str:
    df = pd.read_csv(G14_RETRIEVAL_DIR / "g14_retrieval_metrics.csv")
    deploy = df[(df["feature_set"].eq("触发前车辆和事件信息")) & (df["k"].eq(10))].iloc[0]
    upper = df[(df["feature_set"].eq("未来响应标签上限诊断")) & (df["k"].eq(10))].iloc[0]
    return (
        f"可部署检索test {fmt(deploy.test_rmse)}但G11 {fmt(deploy.g11_rmse)}；"
        f"未来标签上限test {fmt(upper.test_rmse)}，G11 {fmt(upper.g11_rmse)}"
    )


def g14_model_summary(version: str) -> str:
    df = pd.read_csv(G14_DIR / "g14_seed2026_screening_summary" / "g14_seed2026_overall.csv")
    r = df[df["version"].eq(version)].iloc[0]
    return f"seed2026 test {fmt(r.test_rmse)}，tail {fmt(r.tail_rmse)}，selection {fmt(r.selection)}"


def build_rows() -> pd.DataFrame:
    e5a = current_metric("E5A", "test_rmse")
    e10c = current_metric("E10C", "test_rmse")
    rows = [
        {
            "阶段": "主线参照",
            "版本/分析": "E5A / E10C",
            "要回答的问题": "后续突破需要超过什么基准",
            "关键数字": f"E5A {e5a}；E10C {e10c}",
            "证据等级": "当前三种子主线",
            "汇报判断": "E5A/E10C 是这一阶段之后所有结构尝试的参照线",
        },
        {
            "阶段": "困难样本归因",
            "版本/分析": "G11",
            "要回答的问题": "为什么平均误差下降后预测图仍不好看",
            "关键数字": g11_failure_summary(),
            "证据等级": "诊断分析",
            "汇报判断": "瓶颈不是一点RMSE，而是幅值、方向、后段回正和多段修正等物理形态",
        },
        {
            "阶段": "被试泛化",
            "版本/分析": "G12 subject",
            "要回答的问题": "E10C是否只靠某一个被试撑起来",
            "关键数字": g12_subject_summary(),
            "证据等级": "三种子分被试统计",
            "汇报判断": "肌电收益不是单一被试偶然，tyy改善更明显",
        },
        {
            "阶段": "响应类型诊断",
            "版本/分析": "G12 classifier",
            "要回答的问题": "触发前信息能否判断响应形态/困难风险",
            "关键数字": g12_classifier_summary(),
            "证据等级": "诊断分析",
            "汇报判断": "有可判别信号，但精度不足，不能直接稳定决定每个样本该走哪条轨迹",
        },
        {
            "阶段": "模型选择诊断",
            "版本/分析": "G12 selector",
            "要回答的问题": "已有模型之间是否互补，能否按样本选择",
            "关键数字": g12_selector_summary(),
            "证据等级": "seed2026诊断",
            "汇报判断": "模型互补存在，但当前选择器还不会稳定选对专家",
        },
        {
            "阶段": "响应类型辅助学习",
            "版本/分析": "G13H",
            "要回答的问题": "把响应类型作为辅助任务能否形成新主线",
            "关键数字": g13_summary("G13H"),
            "证据等级": "当前三种子",
            "汇报判断": "单seed有希望，但三种子后不如E5A/E10C，不能升级主线",
        },
        {
            "阶段": "困难加权与物理约束",
            "版本/分析": "G13I",
            "要回答的问题": "困难样本加权和物理约束能否修复G13H问题",
            "关键数字": g13_summary("G13I"),
            "证据等级": "当前三种子",
            "汇报判断": "物理风险略均衡，但整体更弱，说明简单加权不够",
        },
        {
            "阶段": "相似历史上限",
            "版本/分析": "G14检索",
            "要回答的问题": "训练集中是否存在可参考的相似响应",
            "关键数字": g14_retrieval_summary(),
            "证据等级": "诊断分析",
            "汇报判断": "训练集中有相似响应；真正瓶颈是推理前判断不出当前响应类型、方向和幅值",
        },
        {
            "阶段": "多候选轨迹",
            "版本/分析": "G14C",
            "要回答的问题": "输出多条候选轨迹能否避免平均化",
            "关键数字": g14_model_summary("G14C"),
            "证据等级": "seed2026筛选",
            "汇报判断": "G14中整体最好，但仍不如E5A/E10C",
        },
        {
            "阶段": "响应原型候选",
            "版本/分析": "G14G",
            "要回答的问题": "响应原型能否改善后段和回正",
            "关键数字": g14_model_summary("G14G"),
            "证据等级": "seed2026筛选",
            "汇报判断": "尾段最好，有诊断价值，但整体和困难样本没有同步突破",
        },
    ]
    return pd.DataFrame(rows)


def wrap_cell(value: object, width: int) -> str:
    return "\n".join(textwrap.wrap(str(value), width=width, break_long_words=True, break_on_hyphens=False))


def render_table(df: pd.DataFrame) -> None:
    csv_path = OUT_DIR / "08_difficult_response_multicandidate_table.csv"
    png_path = OUT_DIR / "08_difficult_response_multicandidate_table.png"
    selected_png = SELECTED_DIR / "21_difficult_response_multicandidate_table.png"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    plot_df = df.copy()
    wrap_widths = {
        "阶段": 10,
        "版本/分析": 12,
        "要回答的问题": 18,
        "关键数字": 34,
        "证据等级": 13,
        "汇报判断": 25,
    }
    for col, width in wrap_widths.items():
        plot_df[col] = plot_df[col].map(lambda value, w=width: wrap_cell(value, w))

    fig_w = 21.0
    fig_h = 1.9 + len(plot_df) * 0.86
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
        colWidths=[0.105, 0.095, 0.190, 0.300, 0.120, 0.230],
        loc="center",
        cellLoc="left",
        bbox=[0.02, 0.18, 0.96, 0.70],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.6)

    raw_rows = df.to_dict("records")
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d9dee5")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#eef1f5")
            cell.get_text().set_color("#20242a")
            cell.get_text().set_weight("bold")
            cell.get_text().set_fontsize(9.4)
        else:
            stage = raw_rows[row - 1]["阶段"]
            source = raw_rows[row - 1]["证据等级"]
            if stage == "主线参照":
                face = "#e8f5ec"
            elif "诊断" in source:
                face = "#eaf3ff"
            elif "seed2026" in source:
                face = "#fff4df"
            elif stage in {"响应类型辅助学习", "困难加权与物理约束"}:
                face = "#fdecec"
            else:
                face = "#ffffff" if row % 2 else "#f8fafc"
            cell.set_facecolor(face)
            cell.get_text().set_color("#222831")
            cell.get_text().set_fontsize(8.25)
        cell._loc = "center" if col in {0, 1, 4} else "left"

    ax.text(
        0.025,
        0.945,
        "第三阶段：困难样本归因、响应类型辅助学习与多候选轨迹",
        color="#111827",
        fontsize=15,
        fontweight="bold",
        transform=ax.transAxes,
        va="center",
    )

    note = (
        "说明：蓝色为诊断分析，浅红为三种子复验后未形成新主线的结构尝试，"
        "浅黄为 seed2026 筛选结果，绿色为当前主线参照。G14 的上限诊断使用未来响应标签，"
        "只能说明训练集中存在相似响应，不能作为可部署模型结果。"
    )
    conclusion = (
        "结论：这一阶段把问题从“生理信号是否有用”推进到“推理期能否判断当前样本属于哪种响应形态”。"
        "G11/G12 证明主要瓶颈是幅值、方向、后段漂移和响应类型选择；G13/G14 说明简单加辅助任务或多候选轨迹还没有稳定超过 E5A/E10C，"
        "后续应围绕响应方向、幅值和形态的先判别与选择性轨迹输出继续推进。"
    )
    ax.text(
        0.025,
        0.105,
        textwrap.fill(note, width=112),
        color="#4b5563",
        fontsize=9.2,
        transform=ax.transAxes,
        va="center",
    )
    ax.text(
        0.025,
        0.052,
        textwrap.fill(conclusion, width=98),
        color="#111827",
        fontsize=10.5,
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


def write_slide_text(df: pd.DataFrame) -> None:
    md_path = OUT_DIR / "08_补充说明_困难样本响应类型与多候选_CN.md"
    rows_md = dataframe_to_markdown(df)
    text = f"""# 第三阶段：困难样本归因、响应类型辅助学习与多候选轨迹

整理时间：2026-05-11

## 这一页要回答的问题

前两页已经说明：连续风格有效，脑电更适合作为训练阶段教师，肌电是当前最强的非脑电推理期生理信号。第三阶段要回答的是：为什么这些主线模型的平均误差已经不错，但预测图仍然会出现不符合物理直觉的情况。

这一页的核心问题是：

1. 困难样本到底难在哪里；
2. 模型是否可以提前判断响应方向、幅值和形态；
3. 响应类型辅助学习能否解决这些问题；
4. 多候选轨迹能否避免把多种可能响应平均成一条平滑但不真实的曲线。

## 推荐放入 PPT 的表

{rows_md}

## 页面底部总结文字

G11 的意义是把问题从平均 RMSE 拆开，发现真正影响预测图质量的是后段漂移、幅值不足、峰值时序偏移、零线两侧相反、反向修正和多段修正等物理形态问题。也就是说，模型不是完全不会预测趋势，而是经常把真实的大幅响应、反向修正或多段响应压成一条比较平滑的平均轨迹。

G12 进一步说明，E10C 的肌电收益不是单个被试偶然造成的；同时，触发前信息对困难风险和部分响应类型确实有可判别信号，但当前选择器还不能稳定判断每个样本应该选哪个模型或哪条轨迹。因此后续不能只继续堆输入，而要让模型先判断响应方向、幅值和形态。

基于这个判断，G13 尝试了响应类型辅助学习、脑电教师 + 肌电学生、困难响应加权和物理约束。G13H 在 seed2026 上接近强候选，但补到三种子后没有超过 E5A/E10C；G13I 物理风险略均衡，但整体更弱。G14 进一步尝试相似历史检索和多候选轨迹。检索上限很强，说明训练集中确实存在相似响应；但可部署检索和多候选轨迹当前仍不能稳定选对困难样本，所以 G14 也不升级为新主线。

## 口头汇报稿

接下来我把问题从“生理信号有没有用”转到“为什么预测图还有明显物理问题”。G11 先做困难样本归因，找出了 111 个困难样本。这里最集中的失败类型是后段漂移、整体误差大、峰值时序偏移和幅值不足，这说明模型不是只差一点平均误差，而是经常没有学准方向、幅值、后段回正以及反向/多段修正这些响应形态。

然后 G12 做了两个检查。第一，E10C 相对 E2 在 cwh、gf、tyy 三个测试被试上都有改善，所以肌电不是只靠某个被试撑起来。第二，我们尝试用触发前信息判断响应类型和困难风险，发现困难风险可以做到较高召回，但精度还不够，轻量选择器也不能稳定选对专家模型。这说明模型之间确实有互补性，但当前还缺一个可靠的响应类型判断器。

基于这个结论，我继续做了 G13 和 G14。G13H/G13I 尝试把响应类型辅助学习、脑电教师、肌电学生、困难响应加权和物理约束结合起来。seed2026 看起来有一点希望，但三种子后没有超过 E5A/E10C，所以不能作为新主线。G14 则进一步尝试相似历史事件和多候选轨迹。未来标签上限诊断效果非常好，说明训练集中有相似响应；但可部署条件下模型仍然不知道该选哪类响应，多候选轨迹当前也没有稳定超过主线。因此这一阶段的结论是：方向是对的，但当前实现还没有解决“推理期响应类型选择”的核心问题。

## 一句话结论

这一阶段证明，主要瓶颈已经从“有没有生理信息”转为“能不能在推理期判断当前响应的方向、幅值和形态”；G13/G14 有诊断价值，但没有形成新主线，后续应围绕响应类型先判别和选择性轨迹输出继续推进。
"""
    md_path.write_text(text, encoding="utf-8")
    print(md_path)


def main() -> None:
    configure_font()
    df = build_rows()
    render_table(df)
    write_slide_text(df)


if __name__ == "__main__":
    main()
