# v221-v222 GPTPro Execution Plan Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 按 GPTPro 最新建议，先完成不训练新模型的 v221 统一评估框架，再根据 v221 结果决定是否进入 v222a 机制风险门控软融合与受限残差。

**Architecture:** v221 是自包含评估脚本，直接读取 v216/v217/v218/v219 已有 CSV 产物，生成统一 formal leaderboard、分组指标、失败样本清单、中文报告、HTML 和 ZIP。v222a 只有在 v221 验证通过后才启动；其输入特征必须通过泄漏字段断言，且所有选择只用 validation。

**Tech Stack:** Python 3, pandas, numpy, matplotlib, zipfile, 现有 `05_rebuild_from_raw_20260511/03_baselines/v216*` 到 `v219*` 表格。

---

## Scope Lock

- v221 不训练模型、不改候选池、不调用缺失的旧源码。
- v218 强峰值训练输出只作为诊断对照，不作为“下一步主线”。
- formal leaderboard 禁止出现 `W3_B4_original_soft`、oracle、true-label fallback、diagnostic-only row。
- v222a 不在 v221 通过前启动。

## Tasks

### Task 1: Restore Plan and Inspect Inputs

- [ ] Confirm v216/v217/v218/v219 output tables exist.
- [ ] Confirm `gptpro_answer/2026.6.22回答.txt` is absent in current checkout and rely on the conversation-provided instruction plus v220/v216-v219 artifacts.

### Task 2: Implement v221 Script

- [ ] Create `05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_v221_formal_model_leaderboard_20260622.py`.
- [ ] Read per-sample tables and metrics-by-split tables.
- [ ] Assign `formal`, `diagnostic`, and `excluded` scopes.
- [ ] Assert forbidden formal models are absent.
- [ ] Compute overall, bucket, no-harm, and universal-failure tables.
- [ ] Write Chinese report, HTML index, run manifest, and ZIP.

### Task 3: Verify v221

- [ ] Run `python -m py_compile` on the new script.
- [ ] Run the script end-to-end.
- [ ] Verify ZIP `bad_file=None`.
- [ ] Verify formal leaderboard excludes forbidden strings.

### Task 4: Sync Notes

- [ ] Update `PROJECT_STATUS_CN.md`.
- [ ] Update `TASK_QUEUE_CN.md`.
- [ ] Update `ARTIFACT_INDEX_CN.md`.
- [ ] Update `daily_logs/2026-06-22.md`.

### Task 5: Decide v222a

- [ ] Read v221 decision table.
- [ ] If v221 outputs are complete, keep v222a as the next task; do not start neural v222b/v223.
