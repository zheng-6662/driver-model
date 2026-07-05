# GPTPro Reply Decision: v222a closeout candidate gap audit

日期：2026-06-22

## Accepted

- 停止 `v222a bounded residual / no-harm gate` formal 主线：本地 no-harm gate 结果已经满足 GPTPro 指出的停止条件，validation 两池通过但 locked test 两池失败，且 loose / strict 呈现相反失败模式。
- 将 `v222a_bounded_residual`、`v222a_noharm_gate`、`oracle_safe_gate` 全部降级为 diagnostic only：符合当前 formal leaderboard 不得引入 test 不稳定 gate 或 oracle 诊断项的 guardrails。
- 执行 `v222a closeout + candidate gap audit`：该步骤不训练新模型、不调阈值、不新增 router，只使用已有 v221 / v222a / no-harm 输出做归因分析，符合当前项目边界。
- 锁定 formal headline：`loose_main_pool=avg_joint_focus`，`strict_main_pool=peak_floor_090`，并在交付表中显式说明 v222a 相关项只作诊断。
- 生成逐样本 taxonomy、bucket failure summary、future route decision：这些交付物直接回答失败来自 selector/gate 还是 candidate pool，不会把诊断 oracle 当作 deployable model。

## Rejected

- 不执行任何 `v222b`、`v223`、neural gate、多候选 router 或 threshold retuning：GPTPro 回复明确禁止，且当前 local evidence 不支持扩大模型线。
- 不根据 locked test 反向调整 gate：这会违反 test discipline。
- 不把 `W3_B4_original_soft`、oracle、true-label fallback 或 diagnostic rows 纳入 formal headline / gate / usage / deployable model：违反 AGENTS.md guardrails。

## Deferred

- 是否未来讨论 `v223`：推迟到 `future_route_decision.csv` 基于 high-tail-error 样本的 `candidate_missing_rate` 和 allowed oracle 改善能力作出判断。
- 是否未来讨论 `v222b`：推迟到新的 subject-group repeated validation 证据；当前结论默认禁止。

## Local Evidence Files

- `05_rebuild_from_raw_20260511/03_baselines/v222a_noharm_gate_diagnostic_20260622/reports/v222a_noharm_gate_diagnostic_report_cn.md`
- `05_rebuild_from_raw_20260511/03_baselines/v222a_noharm_gate_diagnostic_20260622/tables/test_locked_gate_report.csv`
- `05_rebuild_from_raw_20260511/03_baselines/v222a_noharm_gate_diagnostic_20260622/tables/oracle_safe_gate_report.csv`
- `gptpro_reviews/20260622_v222a_closeout_gptpro_response.md`

## Local Decision

采纳 GPTPro 的 closeout-only 路线。下一步只实现：

`05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_v222a_closeout_candidate_gap_audit_20260622.py`

输出目录：

`05_rebuild_from_raw_20260511/03_baselines/v222a_closeout_candidate_gap_audit_20260622/`

完成后同步 note layer，并把 closeout pack 与结论再报告给 GPTPro。
