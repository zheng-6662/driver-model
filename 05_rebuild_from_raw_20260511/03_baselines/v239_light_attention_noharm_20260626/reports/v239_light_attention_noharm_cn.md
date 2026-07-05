# v239 轻量 temporal attention + no-harm 约束报告

## 本轮做了什么

- 继续保留 v238 的 `original_remaining` masked point-level target。
- 新增轻量 temporal attention：历史序列和道路预瞄序列分别做 soft attention。
- attention 只是在同一个模型内部给时间点加权，不是 gate/router/selector，也不是响应类型硬分类。
- 只用 validation no-harm 规则判断 attention 是否可作为下一步候选；test 不参与选择。
- 训练设备：`cuda`。

## Validation 选择

- best diagnostic model：`v239_light_attention_h32`，validation score=1.077325，no-harm pass=True。
- 有 attention 候选通过 no-harm：`v239_light_attention_h32`。

## Test original_remaining 重点对照

### observe_later_like
- delay=0ms：attention tail delta=-0.153984，sample delta=-0.073721
- delay=200ms：attention tail delta=-0.089425，sample delta=-0.043512
- delay=400ms：attention tail delta=-0.161281，sample delta=-0.109955
- delay=600ms：attention tail delta=-0.196791，sample delta=-0.157283
- delay=800ms：attention tail delta=-0.133886，sample delta=-0.118977
- delay=1000ms：attention tail delta=-0.121369，sample delta=-0.121369

### strong_steer
- delay=0ms：attention tail delta=-0.141946，sample delta=-0.073474
- delay=200ms：attention tail delta=-0.033997，sample delta=-0.020663
- delay=400ms：attention tail delta=+0.009627，sample delta=+0.010641
- delay=600ms：attention tail delta=-0.067563，sample delta=-0.047442
- delay=800ms：attention tail delta=-0.034286，sample delta=-0.026010
- delay=1000ms：attention tail delta=+0.048014，sample delta=+0.048014

### normal_predictable
- delay=0ms：attention tail delta=-0.173278，sample delta=-0.137202
- delay=200ms：attention tail delta=-0.071748，sample delta=-0.057233
- delay=400ms：attention tail delta=-0.043820，sample delta=-0.028729
- delay=600ms：attention tail delta=-0.036963，sample delta=-0.022837
- delay=800ms：attention tail delta=-0.061909，sample delta=-0.053664
- delay=1000ms：attention tail delta=-0.031002，sample delta=-0.031002

## 下一步决策

- `accept_attention_as_candidate`: `True`；v239_light_attention_h32 passed validation no-harm and can be treated as the next candidate, still not formal headline.
- `keep_original_remaining_task`: `True`；v239 keeps v238 original_remaining masked target; this remains the correct task construction.
- `formal_replacement_allowed`: `False`；This run is a prototype experiment; formal headline remains locked to v225/v226.
- `recommended_next_task`: `v240_locked_test_report_for_attention_candidate`；Continue only through validation-bounded no-harm; do not expand to router/gate or full Transformer.

## Guardrail

- `task_base`: `v238_original_remaining_masked_point_level_target`
- `attention_type`: `light_temporal_attention_inside_single_model`
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

- `tables/v239_model_selection_validation_noharm.csv`
- `tables/v239_metrics_by_delay_and_bucket.csv`
- `tables/v239_compare_vs_v236_original_remaining.csv`
- `tables/v239_attention_training_history.csv`
- `tables/v239_next_model_decision.csv`
- ZIP：`v239_light_attention_noharm_pack.zip`
