# GPTPro Action Items: v222a closeout candidate gap audit

日期：2026-06-22

## Implementation

1. 新建 `stage03_v222a_closeout_candidate_gap_audit_20260622.py`。
2. 读取已有 `v221_formal_model_leaderboard_20260622`、`v222a_candidate_curve_cache_20260622`、`v222a_noharm_gate_diagnostic_20260622` 输出。
3. 固定 formal headline：`loose_main_pool=avg_joint_focus`，`strict_main_pool=peak_floor_090`。
4. 将 `v222a_bounded_residual`、`v222a_noharm_gate`、`oracle_safe_gate` 写为 diagnostic only。
5. 逐样本构造 `B/M/O` 对比：
   - `B`：formal baseline；
   - `M`：v222a selected residual / no-harm gate candidate；
   - `O`：best allowed candidate oracle diagnostic，仅诊断使用。
6. 分配 taxonomy label：
   - `baseline_sufficient`
   - `safe_under_fix`
   - `under_tradeoff`
   - `pure_harm`
   - `selector_failed`
   - `candidate_missing`
7. 生成 bucket summary、future route decision、case figures、中文报告、manifest、zip。

## Validation

1. `python -m py_compile` 新脚本。
2. 运行 closeout audit 脚本。
3. 检查所有必需表格、图目录、报告、manifest、zip 存在。
4. 用 `zipfile.testzip()` 验证 zip。
5. 检查 formal headline 不包含 `W3_B4_original_soft`。
6. 检查输出没有把 oracle / true_label / fallback / diagnostic-only 项写成 deployable model。

## State Sync

1. 更新 `05_rebuild_from_raw_20260511/00_project_notes/PROJECT_STATUS_CN.md`。
2. 更新 `05_rebuild_from_raw_20260511/00_project_notes/TASK_QUEUE_CN.md`。
3. 更新 `05_rebuild_from_raw_20260511/00_project_notes/ARTIFACT_INDEX_CN.md`。
4. 更新 `05_rebuild_from_raw_20260511/00_project_notes/daily_logs/2026-06-22.md`。
5. 更新 `04_project_logs/references/current-state.md`。
6. 更新 `04_project_logs/reports/progress/decision_log.md`。

## Next GPTPro Prompt

完成 closeout pack 后，把以下信息报告给 GPTPro：

- formal headline decision；
- taxonomy / bucket 主导失败类型；
- high-tail-error 样本中 `candidate_missing_rate`；
- `future_route_decision.csv` 对 `v222b_allowed`、`v223_allowed` 的结论；
- 验证命令与 zip 路径；
- 请 GPTPro 给下一步 bounded 指令，且不得要求 test tuning 或违反候选池 / 泄漏规则。
