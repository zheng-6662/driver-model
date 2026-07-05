# v240 锁定 attention 候选审查报告

## 本轮边界

- 本轮不训练模型，不改 attention 配置，不用 test 选择模型。
- 本轮只读取 v239 已锁定的 `v239_light_attention_h32` 预测和权重，做样本级审查。
- 本轮不创建 gate/router/selector，不做响应类型硬路由，不改变 formal headline。

## 锁定数字结论

- `all`：mean tail delta=-0.059007，max tail delta=-0.016376，tail no-harm all-delay=True。
- `observe_later_like`：mean tail delta=-0.142789，max tail delta=-0.089425，tail no-harm all-delay=True。
- `strong_steer`：mean tail delta=-0.036692，max tail delta=+0.048013，tail no-harm all-delay=False。
- `normal_predictable`：mean tail delta=-0.069787，max tail delta=-0.031002，tail no-harm all-delay=True。
- `strong_400_1000_positive_regression_cases`：mean tail delta=+0.279932，max tail delta=+1.648318，tail no-harm all-delay=False。

## 重点发现

- observe_later_like 锁定审查通过：平均 tail delta `-0.142789`，所有 delay tail no-harm 为 `True`。
- normal_predictable 锁定 no-harm 通过：平均 tail delta `-0.069787`，所有 delay tail no-harm 为 `True`。
- strong_steer 仍有例外：400/1000ms 正向退化样本数 `82`，需要人工看 casebook。
- attention 代表样本平均历史最后 1 秒权重 `0.544`，道路 0-1.2 秒权重 `0.607`。

## 下一步决策

- `attention_candidate_survives_locked_audit`: `True`；normal_predictable and observe_later_like pass locked test no-harm by delay; attention remains a valid next-stage candidate.
- `formal_replacement_allowed`: `False`；v240 is locked audit/casebook only. Formal headline remains v225/v226 until robustness and manual case review are completed.
- `strong_exception_requires_review`: `True`；strong_steer 400/1000ms contains 82 positive-regression test cases; inspect casebook before claiming strong bucket solved.
- `recommended_next_task`: `v241_attention_case_manual_review_and_robustness_ci`；Use v240 casebook for manual review, then run robustness/CI; do not expand architecture before resolving strong exceptions.

## 代表图

- attention casebook 图数：`21`，目录：`figures/attention_casebook/`。
- 每张图包含 true/v236/v238/v239 曲线、历史 attention 权重和道路预瞄 attention 权重。

## Guardrail

- `stage`: `v240_locked_attention_audit`
- `trained_new_model`: `False`
- `changed_model_config`: `False`
- `test_used_for_selection`: `False`
- `gate_router_selector_created`: `False`
- `response_type_hard_routing_created`: `False`
- `formal_headline_changed`: `False`
- `same_event_uid_cross_split_count`: `0`
- `pass`: `True`

## 输出

- `tables/v240_locked_overall_summary.csv`
- `tables/v240_subbucket_noharm_audit.csv`
- `tables/v240_per_sample_locked_metrics.csv`
- `tables/v240_top_observe_later_improvements.csv`
- `tables/v240_strong_400_1000_regressions.csv`
- `tables/v240_attention_casebook_index.csv`
- `tables/v240_attention_time_focus_summary.csv`
- `tables/v240_next_decision.csv`
- ZIP：`v240_locked_attention_audit_pack.zip`
