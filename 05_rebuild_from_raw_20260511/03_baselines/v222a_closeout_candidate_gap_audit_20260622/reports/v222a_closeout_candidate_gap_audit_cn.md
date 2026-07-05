# v222a closeout + candidate gap audit 报告

## 结论

- `v222a bounded residual / no-harm gate` formal 主线停止。
- formal headline 锁定为：`loose_main_pool=avg_joint_focus`，`strict_main_pool=peak_floor_090`。
- `v222a_bounded_residual`、`v222a_noharm_gate`、`oracle_safe_gate` 均为 diagnostic-only。
- 本轮没有训练 v222b/v223，没有新增 router，没有重新选择 tau，也没有根据 locked test 反调配置。

## v222a 停止证据

- loose_main_pool: validation pass=True，locked test pass=False，test RMSE delta=0.010559，tail delta=0.027764，under reduction=0.043478。
- strict_main_pool: validation pass=True，locked test pass=False，test RMSE delta=-0.008975，tail delta=-0.005264，under reduction=-0.017241。

## Oracle vs learned gate

- loose_main_pool: learned tail gain=-0.027764，oracle tail gain=0.105286，oracle-minus-learned tail gap=0.133050，selector_failed_rate=0.408，candidate_missing_rate=0.027。
- strict_main_pool: learned tail gain=0.005264，oracle tail gain=0.106719，oracle-minus-learned tail gap=0.101455，selector_failed_rate=0.414，candidate_missing_rate=0.029。

## Failure taxonomy

- loose_main_pool: baseline_sufficient=88, selector_failed=75, safe_under_fix=7, candidate_missing=5, pure_harm=5, under_tradeoff=4
- strict_main_pool: baseline_sufficient=96, selector_failed=72, candidate_missing=5, pure_harm=1

## Future route decision

- loose_main_pool: main_failure=baseline_sufficient，v222b_allowed=False，v223_allowed=False，high_tail_candidate_missing_rate=0.11904761904761904。
- strict_main_pool: main_failure=baseline_sufficient，v222b_allowed=False，v223_allowed=False，high_tail_candidate_missing_rate=0.13513513513513514。
- combined: main_failure=baseline_sufficient，v222b_allowed=False，v223_allowed=False，high_tail_candidate_missing_rate=0.12658227848101267。

## 诊断口径

- O 是 best allowed formal candidate oracle diagnostic，逐样本选择规则为：先避免 severe-under，再最小 tail RMSE，再最小 sample RMSE。
- `candidate_missing` 的含义是 baseline high-tail 且固定 formal candidate oracle 也不能清晰改善 baseline。
- `selector_failed` 的含义是候选池存在可改善样本的候选，但 learned gate 输出没有抓住对应收益。
- `vehicle_strong` 只作为 closeout bucket：由历史 ay/yaw/curvature 或 future curvature 输入强度的 pool 内 75 分位派生，不作为新模型特征。

## 关键产物

- `tables/formal_headline_decision.csv`
- `tables/v222a_stop_evidence.csv`
- `tables/oracle_vs_learned_gap.csv`
- `tables/candidate_gap_audit.csv`
- `tables/per_sample_failure_taxonomy.csv`
- `tables/bucket_failure_summary.csv`
- `tables/future_route_decision.csv`
- `logs/closeout_manifest.json`
- `logs/sha256_manifest.csv`
- `v222a_closeout_candidate_gap_audit_pack.zip`
- case figures: 61 张，索引见 `tables/case_figure_index.csv`
