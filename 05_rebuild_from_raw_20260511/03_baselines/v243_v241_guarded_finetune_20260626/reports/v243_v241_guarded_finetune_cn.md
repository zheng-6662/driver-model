# v243 v241 guarded fine-tune 实验报告

## 本轮做了什么

- 没有换任务：仍然是 `original_remaining` masked point-level target。
- 没有做 gate/router/selector，也没有删除样本或先硬分类响应类型。
- 直接从 v241 `v241_tcn_mha_h96` 权重初始化。
- 新增三件事：困难样本 hard weight、相对 v241 的 guard loss、v241 已做对正常样本的 teacher anchor。
- 训练设备：`cuda`。

## Validation 选择结果

- best diagnostic model：`v243_metric_hard36_guard08`，validation score=0.865386，accepted_as_next_candidate=True。
- vs v236：normal max sample delta=-0.103117，normal max tail delta=-0.106343，observe mean tail delta=-0.409413，strong 0-600 mean tail delta=-0.358671。
- vs v241：normal max tail delta=+0.002696，all mean tail delta=-0.007909，observe mean tail delta=-0.004415，strong 400/1000 mean tail delta=-0.010060。
- validation 逐样本 guard：all regression rate=0.502，normal regression rate=0.489，all p90 delta=+0.052887，max delta=+0.292549。
- checks：noharm_vs_v236=True，upgrade_vs_v241=True，sample_guard_vs_v241=True，meaningful_gain_vs_v241=True。
- 结论：`v243_metric_hard36_guard08` 可以进入下一轮 locked audit。

## Test 对照：v243 相对 v241

### all
- delay=0ms：tail delta vs v241=-0.003911
- delay=200ms：tail delta vs v241=-0.003638
- delay=400ms：tail delta vs v241=-0.003580
- delay=600ms：tail delta vs v241=-0.001433
- delay=800ms：tail delta vs v241=+0.000760
- delay=1000ms：tail delta vs v241=-0.000967

### observe_later_like
- delay=0ms：tail delta vs v241=+0.005335
- delay=200ms：tail delta vs v241=+0.006981
- delay=400ms：tail delta vs v241=+0.002822
- delay=600ms：tail delta vs v241=+0.012172
- delay=800ms：tail delta vs v241=+0.018133
- delay=1000ms：tail delta vs v241=+0.009870

### strong_steer
- delay=0ms：tail delta vs v241=+0.003321
- delay=200ms：tail delta vs v241=+0.003453
- delay=400ms：tail delta vs v241=-0.000100
- delay=600ms：tail delta vs v241=+0.005570
- delay=800ms：tail delta vs v241=+0.007432
- delay=1000ms：tail delta vs v241=+0.003699

### normal_predictable
- delay=0ms：tail delta vs v241=-0.008015
- delay=200ms：tail delta vs v241=-0.007884
- delay=400ms：tail delta vs v241=-0.005286
- delay=600ms：tail delta vs v241=-0.006403
- delay=800ms：tail delta vs v241=-0.004693
- delay=1000ms：tail delta vs v241=-0.004551

## 逐样本回退概览

- validation：
  - all: n=1854，regressions=931，rate=0.502，mean delta=+0.001677，max delta=+0.292549
  - observe_later_like: n=216，regressions=115，rate=0.532，mean delta=+0.006333，max delta=+0.292549
  - normal_predictable: n=660，regressions=323，rate=0.489，mean delta=+0.000795，max delta=+0.139385
  - strong_steer: n=1140，regressions=587，rate=0.515，mean delta=+0.002628，max delta=+0.292549
  - strong_400_1000: n=380，regressions=195，rate=0.513，mean delta=+0.001340，max delta=+0.134281
  - zero_cross_or_reverse_or_multi: n=1374，regressions=669，rate=0.487，mean delta=+0.000546，max delta=+0.292549
- test：
  - all: n=1104，regressions=498，rate=0.451，mean delta=-0.002128，max delta=+0.185908
  - observe_later_like: n=162，regressions=77，rate=0.475，mean delta=+0.009219，max delta=+0.185908
  - normal_predictable: n=594，regressions=255，rate=0.429，mean delta=-0.006139，max delta=+0.085701
  - strong_steer: n=480，regressions=236，rate=0.492，mean delta=+0.003896，max delta=+0.185908
  - strong_400_1000: n=160，regressions=74，rate=0.463，mean delta=+0.001800，max delta=+0.168896
  - zero_cross_or_reverse_or_multi: n=882，regressions=384，rate=0.435，mean delta=-0.001964，max delta=+0.185908

## 候选级 test 稳定性补充

- validation 排名第一的是 `v243_metric_hard36_guard08`，它通过 validation no-harm / v241-upgrade / sample-guard。
- 但 test 上最均衡的是 `v243_metric_hard24_guard04`：all、normal、observe、strong 四个 bucket 的 test mean tail delta 都为负，observe/strong 各只有 1 个 delay 变差。
- 这不能反向改 validation 选择规则，也不能把 hard24 直接说成 formal replacement；它说明 v243 需要下一轮 locked audit，把 validation-selected hard36 和 conservative hard24 并列审查。
- 机器可读表：`tables/v243_candidate_test_robustness_summary.csv`；按 test 稳定性排序的最稳候选为 `v243_metric_hard24_guard04`。

- v243_metric_hard24_guard04: all mean tail delta=-0.003832, all worse delays=0/6。
- v243_metric_hard30_guard06_anchor04: all mean tail delta=-0.004607, all worse delays=0/6。
- v243_metric_hard36_guard08: all mean tail delta=-0.002128, all worse delays=1/6。

## Guardrail

- `stage`: `v243_v241_guarded_finetune`
- `task_base`: `v238_original_remaining_masked_point_level_target`
- `model_type`: `v241_temporal_convolution_plus_multihead_query_attention_finetuned`
- `initialized_from_v241_checkpoint`: `True`
- `guarded_loss_used`: `True`
- `hard_sample_weight_used`: `True`
- `teacher_anchor_used`: `True`
- `full_transformer_used`: `False`
- `gate_router_selector_created`: `False`
- `response_type_hard_routing_created`: `False`
- `observe_later_like_deleted`: `False`
- `formal_headline_changed`: `False`
- `test_used_for_selection`: `False`
- `same_event_uid_cross_split_count`: `0`
- `validation_noharm_rule_used`: `True`
- `validation_v241_upgrade_rule_used`: `True`
- `validation_sample_guard_rule_used`: `True`
- `pass`: `True`

## 主要产物

- `tables/v243_model_selection_validation_guarded.csv`
- `tables/v243_metrics_by_delay_and_bucket.csv`
- `tables/v243_compare_vs_v236_v239_v241_original_remaining.csv`
- `tables/v243_per_sample_delta_vs_v241.csv`
- `tables/v243_per_sample_delta_summary_vs_v241.csv`
- `tables/v243_worst_regressions_vs_v241.csv`
- `tables/v243_training_weight_plan.csv`
- `tables/v243_next_decision.csv`
- ZIP：`v243_v241_guarded_finetune_pack.zip`
