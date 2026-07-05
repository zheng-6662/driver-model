# GPTPro Response: v222a closeout candidate gap audit

日期：2026-06-22

来源：ChatGPT Desktop / GPTPro 项目会话可见回复转写。由于当前界面的复制按钮不稳定，本文件按可见回复人工整理；后续执行以本地文件、验证命令和项目 guardrails 为准。

## 核心结论

正式停止 `v222a bounded residual / no-harm gate` 主线。

`v222a` 只保留为 diagnostic / case study，不能进入 formal headline，也不能作为继续做 `v222b` / `v223` 的理由。

理由：

- validation 上两个 pool 都能通过；
- locked test 上两个 pool 都失败；
- `loose_main_pool` 保住 under 改善但伤害 RMSE / tail；
- `strict_main_pool` 保住 RMSE / tail 但 under 反向变差；
- oracle upper bound 有价值，说明 residual 局部有用；
- 但 learned gate 泛化失败，deployable feature + validation selection 不足；
- 停止条件已经满足，继续调 gate threshold 属于局部困境。

## 下一步唯一允许方向

执行 `v222a closeout + candidate gap audit`。

目标不是继续优化 `v222a`，而是回答：

> 当前失败主要是 selector/gate 学不出来，还是现有 candidate pool 本身缺少正确曲线？

这个实验不能训练新模型，不能调新阈值，不能新增 router。

建议脚本：

`stage03_v222a_closeout_candidate_gap_audit_20260622.py`

建议输出目录：

`v222a_closeout_candidate_gap_audit_20260622/`

## 执行边界

禁止：

- 训练 `v222b`；
- 训练 `v223`；
- 做多候选 router；
- 重新选择 `tau`；
- 根据 locked test 反调 gate；
- 新增 test-based config；
- 使用 `oracle` / `true_label` / `fallback` / `W3_B4_original_soft` 作为 deployable model。

允许：

- 读取已有 `v221` leaderboard；
- 读取已有 `v222a` candidate cache；
- 读取已有 no-harm gate diagnostic 输出；
- 读取 candidate predictions；
- 读取 per-sample errors；
- 做逐样本归因、分桶、画图、写报告。

## 必须交付

表格：

- `tables/formal_headline_decision.csv`
- `tables/v222a_stop_evidence.csv`
- `tables/oracle_vs_learned_gap.csv`
- `tables/candidate_gap_audit.csv`
- `tables/per_sample_failure_taxonomy.csv`
- `tables/bucket_failure_summary.csv`
- `tables/future_route_decision.csv`

图目录：

- `figures/top_selector_failed_cases/`
- `figures/top_candidate_missing_cases/`
- `figures/top_safe_under_fix_cases/`
- `figures/top_baseline_sufficient_cases/`

报告与打包：

- `reports/v222a_closeout_candidate_gap_audit_cn.md`
- `logs/closeout_manifest.json`
- `v222a_closeout_candidate_gap_audit_pack.zip`

## Formal headline 必须锁定

`formal_headline_decision.csv` 中必须写明：

- `loose_main_pool` formal model = `avg_joint_focus`
- `strict_main_pool` formal model = `peak_floor_090`
- `v222a_bounded_residual` = diagnostic only
- `v222a_noharm_gate` = diagnostic only
- `oracle_safe_gate` = upper-bound diagnostic only
- `ridge_residual_peakfloor` = low-under diagnostic reference

## Per-sample taxonomy

逐样本比较：

- `B` = formal baseline
- `M` = v222a selected residual/gate candidate
- `O` = best allowed candidate oracle diagnostic

`O` 只能用于诊断，不能作为 deployable model。

每个样本必须分配一个 primary label：

1. `baseline_sufficient`

   baseline 已经足够好，`v222a` 不必要。

   建议规则：`B_tail_rmse <= pool_baseline_tail_median` 且 `B_under == 0`

2. `safe_under_fix`

   residual/gate 安全修复 under。

   建议规则：`B_under == 1`，`M_under == 0`，`M_tail_rmse <= B_tail_rmse + 0.03`，`M_rmse <= B_rmse + 0.02`

3. `under_tradeoff`

   修复 under 但伤害 tail 或整条曲线。

   建议规则：`B_under == 1`，`M_under == 0`，且满足 `M_tail_rmse > B_tail_rmse + 0.03` 或 `M_rmse > B_rmse + 0.02`

4. `pure_harm`

   没有修复 under，并且伤害 baseline。

   建议规则：`M_under >= B_under`，且满足 `M_tail_rmse > B_tail_rmse + 0.03` 或 `M_rmse > B_rmse + 0.02`

5. `selector_failed`

   oracle 显示存在安全更好候选，但 learned gate 没抓住。

   建议规则：`O_tail_rmse <= B_tail_rmse - 0.03` 或 `O_rmse <= B_rmse - 0.02` 或 `O_under < B_under`，且 `M` 没达到 `O` 的主要收益。

6. `candidate_missing`

   baseline 很差，且 allowed oracle candidates 也救不了。

   建议规则：`B_tail_rmse > pool_p75`，`O_tail_rmse > B_tail_rmse - 0.03`，且 `O_under >= B_under`

`candidate_missing` 很重要。只有当它在 wrong / high-tail 样本中占主导，未来才有理由讨论新的 candidate generator / `v223`。

## candidate_gap_audit.csv 必须包含字段

- `pool`
- `split`
- `sample_id`
- `scenario_type`
- `strong_steer`
- `reverse`
- `zero_cross`
- `multi_correction`
- `vehicle_strong`
- `normal_curve`
- `baseline_name`
- `baseline_rmse`
- `baseline_tail_rmse`
- `baseline_under`
- `baseline_strong_under`
- `v222a_name`
- `v222a_rmse`
- `v222a_tail_rmse`
- `v222a_under`
- `v222a_strong_under`
- `oracle_best_allowed_candidate`
- `oracle_rmse`
- `oracle_tail_rmse`
- `oracle_under`
- `oracle_strong_under`
- `gain_v222a_rmse`
- `gain_v222a_tail`
- `gain_oracle_rmse`
- `gain_oracle_tail`
- `taxonomy_label`

## bucket_failure_summary.csv 必须包含 bucket 维度

- `pool`
- `scenario_type`
- `strong_steer`
- `reverse`
- `zero_cross`
- `multi_correction`
- `vehicle_strong`
- `normal_curve`
- `extreme_peak`
- `high_tail_error`
- `taxonomy_label`

每个 bucket 输出：

- `n`
- `baseline_rmse`
- `baseline_tail`
- `v222a_rmse`
- `v222a_tail`
- `oracle_rmse`
- `oracle_tail`
- `baseline_under`
- `v222a_under`
- `oracle_under`
- `selector_failed_rate`
- `candidate_missing_rate`
- `under_tradeoff_rate`
- `safe_under_fix_rate`
- `pure_harm_rate`

## future_route_decision.csv 规则

该表直接决定未来是否可以讨论 `v222b` / `v223`。

`v222b` 默认：

`v222b_allowed = False`

只有满足以下全部条件，未来才可以重新考虑 `v222b`：

- `selector_failed_rate` 不是主要失败源；
- no-harm gate 在新的 subject-group repeated validation 中稳定通过；
- 不再依赖单一 validation split。

当前：

`v222b_allowed = False`

理由：learned gate validation pass but locked test failed；更大 neural gate 很可能 overfit selector signal。

`v223` 默认：

`v223_allowed = False`

只有当 closeout audit 显示：

- high-tail-error 样本中的 `candidate_missing_rate > 50%`；
- 且 `oracle_best_allowed_candidate` 仍不能清晰改善 baseline；

才允许未来讨论 `v223`。

如果主问题是 `selector_failed`、`under_tradeoff` 或 `pure_harm`，则 `v223` 仍然禁止。

## 停止条件

`v222a` 主线停止条件已经触发：

- validation formal pass = True；
- locked test formal pass = False；
- 两个 pool 呈现不一致 / 相反失败模式。

因此：

- STOP `v222a` formal model development；
- STOP `v222a` threshold tuning；
- STOP `v222a` no-harm gate optimization；
- STOP `v222a bounded residual as headline`。

closeout audit 的停止条件：

- 必须交付物完成后自动停止；
- closeout 不是新模型线；
- 不允许继续到 `v222a_gate_v2`、`v222a_new_tau`、`v222a_multi_router`、`v222a_neural_gate` 或 `v222b`，除非 `future_route_decision.csv` 明确满足 unlock 条件。
