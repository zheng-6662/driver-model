# v241 更强时序模型受控实验报告

## 本轮做了什么

- 保留 v238/v239 的 `original_remaining` masked point-level target。
- 将 v239 的轻量 attention 升级为 temporal convolution + multi-head query attention。
- 模型仍然是单一连续预测器，不做 gate/router/selector，不先硬判断响应类型。
- 只用 validation 选择模型；test 只做固定后的对照报告。
- 训练设备：`cuda`。

## Validation 选择结果

- best diagnostic model：`v241_tcn_mha_h96`，validation score=0.872780，accepted_as_stronger_candidate=True。
- vs v236：normal max sample delta=-0.143055，normal max tail delta=-0.164108，observe mean tail delta=-0.263814，strong 0-600 mean tail delta=-0.221641。
- vs v239：normal max tail delta=-0.023461，observe mean tail delta=-0.124918，strong 400/1000 mean tail delta=-0.187700。
- 通过 stronger-candidate 检查的模型：`v241_tcn_mha_h96`。

## Test original_remaining 对照

### observe_later_like
- delay=0ms：v241-v236 tail delta=-0.307929，v241-v239 tail delta=-0.153944
- delay=200ms：v241-v236 tail delta=-0.335905，v241-v239 tail delta=-0.246481
- delay=400ms：v241-v236 tail delta=-0.424241，v241-v239 tail delta=-0.262960
- delay=600ms：v241-v236 tail delta=-0.371992，v241-v239 tail delta=-0.175201
- delay=800ms：v241-v236 tail delta=-0.286698，v241-v239 tail delta=-0.152812
- delay=1000ms：v241-v236 tail delta=-0.240372，v241-v239 tail delta=-0.119003

### strong_steer
- delay=0ms：v241-v236 tail delta=-0.370320，v241-v239 tail delta=-0.228374
- delay=200ms：v241-v236 tail delta=-0.291511，v241-v239 tail delta=-0.257514
- delay=400ms：v241-v236 tail delta=-0.222188，v241-v239 tail delta=-0.231814
- delay=600ms：v241-v236 tail delta=-0.197173，v241-v239 tail delta=-0.129609
- delay=800ms：v241-v236 tail delta=-0.133068，v241-v239 tail delta=-0.098782
- delay=1000ms：v241-v236 tail delta=-0.040462，v241-v239 tail delta=-0.088476

### normal_predictable
- delay=0ms：v241-v236 tail delta=-0.267294，v241-v239 tail delta=-0.094016
- delay=200ms：v241-v236 tail delta=-0.151421，v241-v239 tail delta=-0.079674
- delay=400ms：v241-v236 tail delta=-0.135156，v241-v239 tail delta=-0.091336
- delay=600ms：v241-v236 tail delta=-0.148075，v241-v239 tail delta=-0.111112
- delay=800ms：v241-v236 tail delta=-0.134037，v241-v239 tail delta=-0.072128
- delay=1000ms：v241-v236 tail delta=-0.097311，v241-v239 tail delta=-0.066310

## 逐样本回退摘要

- 下面统计的是 test 样本内 v241 相对 v239 的逐样本 tail RMSE 是否变差。均值改善不等于每个样本都改善。
- `all`：n=1104，tail 回退 368 条，回退率=0.333，mean delta=-0.128199，max delta=+1.076289。
- `observe_later_like`：n=162，tail 回退 44 条，回退率=0.272，mean delta=-0.185067，max delta=+0.423391。
- `normal_predictable`：n=594，tail 回退 223 条，回退率=0.375，mean delta=-0.085762，max delta=+1.076289。
- `strong_steer`：n=480，tail 回退 142 条，回退率=0.296，mean delta=-0.172428，max delta=+0.539752。
- `strong_400_1000`：n=160，tail 回退 47 条，回退率=0.294，mean delta=-0.160145，max delta=+0.538267。

## 下一步决策

- `best_diagnostic_stronger_model`: `v241_tcn_mha_h96`；Best by validation selection score; this does not itself imply formal replacement.
- `accept_stronger_model_as_next_candidate`: `True`；v241_tcn_mha_h96 passed validation no-harm and v239-upgrade checks; it can enter locked audit.
- `accepted_model_name`: `v241_tcn_mha_h96`；Empty means no stronger model should replace v239 yet.
- `formal_replacement_allowed`: `False`；v241 is a stronger-model trial; formal headline remains locked until locked audit and robustness checks pass.
- `recommended_next_task`: `v242_locked_test_report_for_stronger_temporal_candidate`；Do not use test to retune; either locked-audit the accepted candidate or return to strong-case review.

## Guardrail

- `stage`: `v241_stronger_temporal_model`
- `task_base`: `v238_original_remaining_masked_point_level_target`
- `model_type`: `temporal_convolution_plus_multihead_query_attention`
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

- `tables/v241_model_selection_validation_noharm.csv`
- `tables/v241_metrics_by_delay_and_bucket.csv`
- `tables/v241_compare_vs_v236_v238_v239_original_remaining.csv`
- `tables/v241_per_sample_delta_vs_v239.csv`
- `tables/v241_per_sample_delta_summary_vs_v239.csv`
- `tables/v241_worst_regressions_vs_v239.csv`
- `tables/v241_next_decision.csv`
- ZIP：`v241_stronger_temporal_model_pack.zip`
