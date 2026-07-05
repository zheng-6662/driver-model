# 2026-06-23 GPTPro v230 action items

## 必做

1. 新增 `05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_v230_failure_case_manual_review_casebook_20260623.py`。
2. 只读 v225/v226/v228/v229 产物，生成 `v230_failure_case_manual_review_casebook_20260623/`。
3. 选择 compact casebook：
   - 每个 pool 至少 5 个强反应低估；
   - 至少 4 个极端峰值低估或极端峰值尾段难例；
   - 至少 5 个强响应幅值/尾段；
   - 至少 3 个反转或多次修正；
   - 至少 3 个过零/换向边界；
   - 至少 3 个普通曲线可控 baseline sufficient。
4. 生成必需表：
   - `v230_case_selection_index.csv`
   - `v230_manual_review_template.csv`
   - `v230_failure_casebook_table.csv`
   - `v230_bucket_to_claim_mapping.csv`
   - `v230_case_figure_inventory.csv`
   - `v230_formal_boundary_check.csv`
5. 生成必需报告：
   - `v230_failure_case_manual_review_casebook_cn.md`
   - `v230_advisor_discussion_notes_cn.md`
   - `v230_paper_failure_case_section_draft_cn.md`
6. 复制既有图到：
   - `figures/selected_casebook_figures/loose_main_pool/`
   - `figures/selected_casebook_figures/strict_main_pool/`
   - `figures/selected_casebook_figures/cross_pool_repeated_cases/`
   - `figures/selected_casebook_figures/baseline_sufficient_controls/`
7. 生成日志：
   - `run_manifest.json`
   - `input_file_hashes.json`
   - `guardrail_check.json`
   - `forbidden_scan_report.json`
   - `file_inventory.json`
   - `figure_copy_check.json`
   - `consistency_check.json`
8. 生成 `v230_failure_case_manual_review_casebook_pack.zip`。
9. 运行并记录验证：`py_compile`、完整脚本、ZIP `testzip()==None`、required files、guardrail、forbidden scan、consistency、figure-copy。

## 停止条件

v230 包完成并通过验证后停止；不继续启动模型训练或任何新预测。
