# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path


ROOT = Path("F:/data_set_process/data_process/04_project_logs/reports")
PACK = ROOT / "gptpro_E0_to_now_20260511"
OUT = ROOT / "gptpro_E0_to_now_20260511_split_zips"

COMMON_FILES = ["00_README_CN.md", "01_timeline_CN.csv", "02_reading_guide_CN.md"]

PARTS = [
    {
        "name": "overview_logs_code_protocol",
        "title": "总览、日志、源码、协议包",
        "paths": [
            "01_overview_selected_figures",
            "02_project_logs",
            "04_run_records",
            "05_source_code",
            "06_protocol_and_style",
            "07_server_logs_recent",
            "03_file_manifest.csv",
            "05_package_check.json",
        ],
        "note": "先看这个包。它包含总览、精选图、项目日志、运行记录、源码、样本 manifest 和连续驾驶风格材料。",
    },
    {
        "name": "E0_to_E8_baselines_physio_eeg",
        "title": "E0-E8 基准、连续风格、脑电教师和早期生理路线包",
        "paths": [
            "03_reports/e0_e2_summary_fresh_3seed_20260507",
            "03_reports/e3_e4_summary_final_20260507",
            "03_reports/e5_distill_summary_20260508",
            "03_reports/e6_physical_repair_summary_20260508",
            "03_reports/e7_signal_group_summary_20260508",
            "03_reports/e8_reliable_phys_summary_20260508",
        ],
        "note": "这一包用于看早期证据链：连续驾驶风格、含/不含脑电、脑电教师蒸馏、物理修复、信号分组和可靠性尝试。",
    },
    {
        "name": "G9_E10_E11_signal_calibration",
        "title": "G9、E10、E11 信号归因与校准包",
        "paths": [
            "03_reports/g9_cand_conv_20260508",
            "03_reports/e10_non_eeg_signal_summary_20260509",
            "03_reports/e10_single_sig_3seed_summary_20260509",
            "03_reports/e10c_emg_only_3seed_summary_20260509",
            "03_reports/e11_emg_distill_summary_20260509",
        ],
        "note": "这一包用于看非脑电生理信号归因、肌电 E10C、脑电教师加肌电学生，以及 G9 校准和候选收敛尝试。",
    },
    {
        "name": "G11_G12_G13_G14_diagnostics",
        "title": "G11-G14 困难样本、被试泛化、突破模型和非平均化预测包",
        "paths": [
            "03_reports/g11_badcase_attr_20260509",
            "03_reports/g12_gate_subject_20260510",
            "03_reports/g13_model_break_20260510",
            "03_reports/g14_nonavg_pred_20260510",
            "03_reports/restore_checkpoint_audit_20260510",
        ],
        "note": "这一包用于分析模型卡住的原因：困难样本、响应类型、被试分布、G13/G14 新结构为什么没有真正突破。",
    },
    {
        "name": "E15_to_E19_recent_physio_representations",
        "title": "E15-E19 最近生理/脑电表示和融合补充包",
        "paths": [
            "03_reports/e15_e16_single_sig_summary_20260511",
            "03_reports/e17_semantic_single_sig_seed2026_summary_20260511",
            "03_reports/e18_sig_repr_seed2026_summary_20260511",
            "03_reports/e19_sig_fusion_seed2026_summary_20260511",
        ],
        "note": "这一包用于看最近关于单信号、人工语义状态、去人工权重表示、四信号融合的补充结论。",
    },
    {
        "name": "prediction_overview_figures",
        "title": "正式运行预测总览图包",
        "paths": ["08_prediction_overviews"],
        "note": "这一包主要是预测曲线总览图，便于 GPTPro 不只看指标，也看趋势、幅值、错侧和尾段问题。",
    },
]


def iter_files(rel_path: str):
    path = PACK / rel_path
    if not path.exists():
        return
    if path.is_file():
        yield path
    else:
        yield from (item for item in path.rglob("*") if item.is_file())


def main() -> None:
    if not PACK.exists():
        raise RuntimeError(f"missing package directory: {PACK}")
    if OUT.exists():
        resolved = OUT.resolve()
        reports = ROOT.resolve()
        if reports not in resolved.parents or resolved.name != "gptpro_E0_to_now_20260511_split_zips":
            raise RuntimeError(f"refuse to remove unexpected path: {resolved}")
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    summary = []
    for index, part in enumerate(PARTS, start=1):
        zip_path = OUT / f"{index:02d}_{part['name']}.zip"
        part_root = part["name"]
        file_count = 0
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            part_readme = (
                f"# {part['title']}\n\n"
                f"{part['note']}\n\n"
                f"这是从完整证据包 `gptpro_E0_to_now_20260511` 拆分出的第 {index} 个压缩包。"
                "正式提问词不在包内，需要下一个对话单独写。\n"
            )
            archive.writestr(f"{part_root}/PART_README_CN.md", part_readme.encode("utf-8-sig"))
            file_count += 1
            for common in COMMON_FILES:
                path = PACK / common
                if path.exists():
                    archive.write(path, f"{part_root}/{path.name}")
                    file_count += 1
            for rel_path in part["paths"]:
                for file in iter_files(rel_path):
                    arcname = file.relative_to(PACK).as_posix()
                    archive.write(file, f"{part_root}/{arcname}")
                    file_count += 1
        with zipfile.ZipFile(zip_path, "r") as archive:
            bad = archive.testzip()
            if bad:
                raise RuntimeError(f"bad member in {zip_path}: {bad}")
        summary.append(
            {
                "zip": zip_path.name,
                "title": part["title"],
                "note": part["note"],
                "bytes": zip_path.stat().st_size,
                "MB": round(zip_path.stat().st_size / 1024 / 1024, 2),
                "file_count": file_count,
            }
        )

    (OUT / "00_split_zip_manifest_CN.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8-sig",
    )
    readme = (
        "# 拆分压缩包说明\n\n"
        "原完整包是 `gptpro_E0_to_now_20260511.zip`，较大。这里按用途拆成 6 个压缩包。"
        "建议先发第 1 个包，再按 GPTPro 需要补发第 2-6 个包。\n\n"
    )
    for row in summary:
        readme += (
            f"- `{row['zip']}`：{row['title']}，约 {row['MB']} MB，"
            f"{row['file_count']} 个文件。{row['note']}\n"
        )
    (OUT / "00_README_CN.md").write_text(readme, encoding="utf-8-sig")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
