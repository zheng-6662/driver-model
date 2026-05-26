from __future__ import annotations

import html
import re
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(r"F:/data_set_process/data_process/05_rebuild_from_raw_20260511")
AUDIT_DIR = ROOT / "03_baselines" / "stage03_goal2_clean_task_audit" / "outputs" / "exclusion_recovery_audit"
BREAKDOWN_CSV = AUDIT_DIR / "goal2_exclusion_reason_breakdown.csv"
OUT_DIR = AUDIT_DIR / "manual_review_images_by_priority"

FOLDER_ORDER = {
    "A_优先人工恢复复核": "00_A_优先看_旧结论可能误伤",
    "B_较可能可恢复": "01_B_较可能可恢复_看图确认",
    "C1_弯道高度变化重点复核": "02_C1_弯道高度变化_重点复核",
    "C2_高度姿态重点复核": "03_C2_高度姿态明显_谨慎复核",
    "D_暂不恢复_疑似路边或路外": "04_D_暂不恢复_疑似路边路外",
    "U_原因不清_需要复核": "05_U_原因不清_需要复核",
}


def safe_name(text: str, max_len: int = 130) -> str:
    text = re.sub(r"[\\/:*?\"<>|]+", "_", str(text))
    text = re.sub(r"\s+", "_", text).strip("_")
    return text[:max_len] or "unknown"


def link_for(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def build_gallery(folder: Path, rows: list[dict[str, str]], title: str, root: Path) -> None:
    lines = [
        "<html><meta charset='utf-8'><body>",
        f"<h1>{html.escape(title)}</h1>",
        "<p>建议人工标记：保留 / 排除 / 不确定。图片文件名保留 episode_uid 和当前高度字段。</p>",
        "<style>body{font-family:Arial,'Microsoft YaHei',sans-serif;} .item{margin:20px 0;padding:12px;border:1px solid #ddd;} img{max-width:1200px;width:100%;height:auto;} .meta{white-space:pre-wrap;font-size:13px;color:#333;}</style>",
    ]
    for row in rows:
        rel = link_for(Path(row["copied_image_path"]), root)
        meta = "\n".join(
            [
                f"episode_uid: {row.get('episode_uid','')}",
                f"recovery_priority: {row.get('recovery_priority','')}",
                f"height_issue: {row.get('v2_0_height_pose_issue','')}",
                f"z_residual: {row.get('z_residual_range_v1_3','')}",
                f"z_rise: {row.get('z_rise_from_start_v1_4','')}",
                f"z_drop: {row.get('z_drop_from_start_v1_4','')}",
                f"reason: {row.get('actual_exclusion_reasons','')}",
            ]
        )
        lines.extend(
            [
                "<div class='item'>",
                f"<h3>{html.escape(str(row.get('episode_uid','')))}</h3>",
                f"<div class='meta'>{html.escape(meta)}</div>",
                f"<p><a href='{html.escape(rel)}'>打开图片</a></p>",
                f"<img src='{html.escape(rel)}'>",
                "</div>",
            ]
        )
    lines.append("</body></html>")
    (folder / "index.html").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(BREAKDOWN_CSV, encoding="utf-8-sig", low_memory=False)
    df["manual_keep"] = ""
    df["manual_note"] = ""

    copied_rows: list[dict[str, str]] = []
    missing_rows: list[dict[str, str]] = []

    for priority, group in df.groupby("recovery_priority", sort=False):
        folder_name = FOLDER_ORDER.get(str(priority), safe_name(priority))
        folder = OUT_DIR / folder_name
        folder.mkdir(parents=True, exist_ok=True)
        group_rows: list[dict[str, str]] = []
        group = group.sort_values(["z_residual_range_v1_3", "z_rise_from_start_v1_4", "episode_uid"], ascending=[True, True, True])
        for idx, (_, row) in enumerate(group.iterrows(), start=1):
            src = Path(str(row.get("review_image_path", "")).strip())
            out_row = row.to_dict()
            if not src.exists():
                out_row["missing_reason"] = "review_image_path 不存在或为空"
                missing_rows.append(out_row)
                continue
            episode_uid = safe_name(row.get("episode_uid", "unknown"), max_len=90)
            height = safe_name(row.get("v2_0_height_pose_issue", "height"), max_len=20)
            stem = f"{idx:04d}_{episode_uid}_{height}{src.suffix.lower()}"
            dst = folder / stem
            if not dst.exists():
                shutil.copy2(src, dst)
            out_row["copied_image_path"] = str(dst)
            out_row["review_folder"] = str(folder)
            copied_rows.append(out_row)
            group_rows.append(out_row)

        pd.DataFrame(group_rows).to_csv(folder / "index.csv", index=False, encoding="utf-8-sig")
        readme = [
            f"# {folder_name}",
            "",
            f"- 样本数：{len(group)}",
            f"- 已复制图片：{len(group_rows)}",
            f"- 缺少图片路径：{len(group) - len(group_rows)}",
            "",
            "人工审核建议：",
            "- A/B：优先看，确认是否可恢复进训练。",
            "- C1/C2：重点看高度变化是否属于真实路边/斜坡，还是正常道路坡度/车辆姿态变化。",
            "- D/U：一般不优先恢复，除非图像非常明确。",
        ]
        (folder / "README.md").write_text("\n".join(readme), encoding="utf-8")
        build_gallery(folder, group_rows, folder_name, OUT_DIR)

    copied = pd.DataFrame(copied_rows)
    missing = pd.DataFrame(missing_rows)
    copied.to_csv(OUT_DIR / "manual_review_images_index.csv", index=False, encoding="utf-8-sig")
    missing.to_csv(OUT_DIR / "manual_review_images_missing.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    for priority, group in df.groupby("recovery_priority", sort=False):
        copied_n = int((copied["recovery_priority"].eq(priority)).sum()) if not copied.empty else 0
        summary_rows.append(
            {
                "recovery_priority": priority,
                "total_rows": int(len(group)),
                "copied_images": copied_n,
                "missing_images": int(len(group) - copied_n),
                "folder": str(OUT_DIR / FOLDER_ORDER.get(str(priority), safe_name(priority))),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "manual_review_images_summary.csv", index=False, encoding="utf-8-sig")

    lines = [
        "<html><meta charset='utf-8'><body>",
        "<h1>Goal2 被排除样本人工审核图片索引</h1>",
        "<p>本目录按恢复优先级整理。优先看 A 和 B 两档。</p>",
        summary.to_html(index=False, escape=False),
        "<h2>分类入口</h2><ul>",
    ]
    for row in summary_rows:
        folder = Path(row["folder"])
        lines.append(
            f"<li><a href='{html.escape(link_for(folder / 'index.html', OUT_DIR))}'>{html.escape(str(row['recovery_priority']))}</a> "
            f"复制 {row['copied_images']} / {row['total_rows']}</li>"
        )
    lines.extend(["</ul>", "</body></html>"])
    (OUT_DIR / "index.html").write_text("\n".join(lines), encoding="utf-8")

    report_lines = [
        "# Goal2 被排除样本人工审核图片整理",
        "",
        f"- 输出目录：`{OUT_DIR}`",
        f"- 总样本：`{len(df)}`",
        f"- 已复制图片：`{len(copied)}`",
        f"- 缺少图片路径：`{len(missing)}`",
        "",
        "## 分类数量",
        "",
        summary.to_markdown(index=False),
        "",
        "## 审核方式",
        "",
        "1. 优先打开 `index.html`，再进入 A/B 分类。",
        "2. 如果看图后认为样本可以恢复，建议在对应 `index.csv` 里填写 `manual_keep=保留`。",
        "3. 如果认为明显下马路/路边/上斜坡，填写 `manual_keep=排除`。",
        "4. 如果判断不清，填写 `manual_keep=不确定`，后续结合道路源文件再判断。",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"wrote {OUT_DIR}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
