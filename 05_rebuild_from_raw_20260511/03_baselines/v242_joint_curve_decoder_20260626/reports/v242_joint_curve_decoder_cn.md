# v242 联合曲线解码模型训练报告

## 本轮做了什么

- 保留 `original_remaining` masked target。
- 把逐点预测改成一次输出 21 个 future points 的联合曲线解码。
- 增加轻量曲线差分约束，让模型学习曲线形态，而不是只追逐单点误差。
- 不做 gate/router/selector，不做 response-type hard routing，不删除样本。
- 只用 validation 选择模型；test 只做固定后的报告。
- 训练设备：`cuda`。

## Validation 选择结果

- best diagnostic model：`v242_joint_curve_h96_smooth005`，validation score=1.040291，accepted_as_next_candidate=False。
- vs v236：normal max tail delta=-0.124933，observe mean tail delta=-0.269463，strong 0-600 mean tail delta=-0.201033。
- vs v241：normal max tail delta=+0.039176，observe mean tail delta=-0.005649，strong 400/1000 mean tail delta=+0.014069。
- 没有 v242 候选同时通过 v236 no-harm 和 v241 upgrade；本轮只作为诊断训练。

## Test original_remaining 对照

### observe_later_like
- delay=0ms：v242-v236 tail delta=-0.345093，v242-v241 tail delta=-0.037164
- delay=200ms：v242-v236 tail delta=-0.308988，v242-v241 tail delta=+0.026917
- delay=400ms：v242-v236 tail delta=-0.417920，v242-v241 tail delta=+0.006321
- delay=600ms：v242-v236 tail delta=-0.377988，v242-v241 tail delta=-0.005997
- delay=800ms：v242-v236 tail delta=-0.286458，v242-v241 tail delta=+0.000240
- delay=1000ms：v242-v236 tail delta=-0.293047，v242-v241 tail delta=-0.052675

### strong_steer
- delay=0ms：v242-v236 tail delta=-0.334856，v242-v241 tail delta=+0.035465
- delay=200ms：v242-v236 tail delta=-0.243751，v242-v241 tail delta=+0.047760
- delay=400ms：v242-v236 tail delta=-0.181720，v242-v241 tail delta=+0.040468
- delay=600ms：v242-v236 tail delta=-0.179174，v242-v241 tail delta=+0.017999
- delay=800ms：v242-v236 tail delta=-0.119919，v242-v241 tail delta=+0.013149
- delay=1000ms：v242-v236 tail delta=-0.055062，v242-v241 tail delta=-0.014600

### normal_predictable
- delay=0ms：v242-v236 tail delta=-0.237371，v242-v241 tail delta=+0.029923
- delay=200ms：v242-v236 tail delta=-0.140178，v242-v241 tail delta=+0.011244
- delay=400ms：v242-v236 tail delta=-0.114593，v242-v241 tail delta=+0.020563
- delay=600ms：v242-v236 tail delta=-0.135585，v242-v241 tail delta=+0.012490
- delay=800ms：v242-v236 tail delta=-0.110204，v242-v241 tail delta=+0.023834
- delay=1000ms：v242-v236 tail delta=-0.073804，v242-v241 tail delta=+0.023507

## 逐样本回退摘要

- 下面统计的是 test 样本内 v242 相对 v241 的逐样本 tail RMSE 是否变差。
- `all`：n=1104，tail 回退 588 条，回退率=0.533，mean delta=+0.022558，max delta=+0.814949。
- `observe_later_like`：n=162，tail 回退 86 条，回退率=0.531，mean delta=-0.010393，max delta=+0.568164。
- `normal_predictable`：n=594，tail 回退 311 条，回退率=0.524，mean delta=+0.020260，max delta=+0.689269。
- `strong_steer`：n=480，tail 回退 258 条，回退率=0.537，mean delta=+0.023373，max delta=+0.814949。
- `strong_400_1000`：n=160，tail 回退 80 条，回退率=0.500，mean delta=+0.012934，max delta=+0.814949。

## 下一步决策

- `best_diagnostic_joint_curve_model`: `v242_joint_curve_h96_smooth005`；Best by validation selection score; not automatically a formal replacement.
- `accept_joint_curve_model_as_next_candidate`: `False`；No v242 joint-curve candidate passed both v236 no-harm and v241-upgrade checks. Keep v241 as the stronger candidate.
- `accepted_model_name`: ``；Empty means v242 should remain diagnostic only.
- `formal_replacement_allowed`: `False`；v242 is a training experiment; formal headline remains locked until locked audit and robustness checks pass.
- `recommended_next_task`: `v243_manual_review_or_loss_redesign_for_sample_regressions`；Do not use test to retune; fixed candidate must be locked-audited before any formal claim.

## Guardrail

- `stage`: `v242_joint_curve_decoder`
- `task_base`: `v238_original_remaining_masked_point_level_target`
- `model_type`: `sample_level_joint_curve_decoder_with_cross_attention`
- `full_transformer_used`: `False`
- `gate_router_selector_created`: `False`
- `response_type_hard_routing_created`: `False`
- `observe_later_like_deleted`: `False`
- `formal_headline_changed`: `False`
- `test_used_for_selection`: `False`
- `same_event_uid_cross_split_count`: `0`
- `validation_noharm_rule_used`: `True`
- `pass`: `True`

## 输出

- `tables/v242_model_selection_validation_noharm.csv`
- `tables/v242_metrics_by_delay_and_bucket.csv`
- `tables/v242_compare_vs_v236_v239_v241_original_remaining.csv`
- `tables/v242_per_sample_delta_vs_v241.csv`
- `tables/v242_per_sample_delta_summary_vs_v241.csv`
- `tables/v242_next_decision.csv`
- ZIP：`v242_joint_curve_decoder_pack.zip`
