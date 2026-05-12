# 阶段 3 统一车辆-only 对照 v0.1

生成时间：2026-05-12

## 目的

本报告把正式失稳样本 `vehicle_instability_highconf_v0_1` 的主窗口 `pre2_label2_old_main` + `session_level_split` 上已经完成的车辆-only 对照放到同一张表中。这里仍然只讨论车辆历史和事件/道路上下文，不讨论连续风格、生理或 EEG 的有效性。

## 输入边界

- 使用已经生成的阶段 3 指标文件，不重新训练模型。
- 所有候选均不使用生理、脑电、连续风格或驾驶员 ID。
- `eval_label_*` 只允许用于评价分层和图表，不作为模型输入。
- 本轮未连接服务器，未读取服务器指令与密码文件。

## test 集核心指标

| display_name | source_group | rmse_steer | peak_direction_accuracy | wrong_side_rate | large_response_recall | severe_amp_under_rate | tail_abs_error_mean | reversal_count_exact_match_rate | multi_segment_rate_abs_gap | difficult_top20_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| KNN template | template_memory_risk | 0.516941 | 0.827338 | 0.172662 | 0.520000 | 0.244604 | 0.501961 | 0.007194 | 0.316547 | 1.040524 |
| RBF KRR | strong_vehicle_diagnostic | 0.540287 | 0.784173 | 0.215827 | 0.600000 | 0.251799 | 0.530500 | 0.043165 | 0.201439 | 1.015198 |
| peak-scaled template | template_memory_risk | 0.555055 | 0.791367 | 0.208633 | 0.600000 | 0.079137 | 0.552014 | 0.093525 | 0.251799 | 1.066179 |
| vehicle Transformer | true_vehicle_transformer | 0.567162 | 0.820144 | 0.179856 | 0.440000 | 0.266187 | 0.526112 | 0.201439 | 0.676259 | 1.089107 |
| direction-gated KNN | template_memory_risk | 0.579581 | 0.791367 | 0.208633 | 0.640000 | 0.071942 | 0.574765 | 0.129496 | 0.309353 | 1.085050 |
| old vehicle_direct active | old_vehicle_direct_clean | 0.637366 | 0.870504 | 0.129496 | 0.142857 | 0.683453 | 0.530855 | 0.086331 | 0.640288 | 1.053696 |
| old vehicle_direct structure | old_vehicle_direct_clean | 0.647720 | 0.856115 | 0.143885 | 0.200000 | 0.561151 | 0.496645 | 0.057554 | 0.057554 | 1.060033 |
| formal ridge | formal_linear | 0.649341 | 0.769784 | 0.230216 | 0.080000 | 0.582734 | 0.663662 | 0.093525 | 0.273381 | 1.265239 |
| rich ridge context | strong_vehicle_diagnostic | 0.652941 | 0.820144 | 0.179856 | 0.240000 | 0.330935 | 0.651683 | 0.035971 | 0.323741 | 1.254761 |
| event mean | no_learning | 0.677212 | 0.568345 | 0.431655 | 0.000000 | 0.913669 | 0.719166 | 0.086331 | 0.309353 | 1.318363 |
| rich ridge history | strong_vehicle_diagnostic | 0.680683 | 0.755396 | 0.244604 | 0.160000 | 0.374101 | 0.696528 | 0.057554 | 0.323741 | 1.307168 |
| zero hold | no_learning | 0.683514 | 0.517986 | 0.482014 | 0.000000 | 1.000000 | 0.725312 | 0.064748 | 0.676259 | 1.322311 |
| train mean | no_learning | 0.685789 | 0.482014 | 0.517986 | 0.000000 | 0.942446 | 0.724539 | 0.258993 | 0.676259 | 1.326782 |
| ridge history | formal_linear | 0.707027 | 0.712230 | 0.287770 | 0.120000 | 0.510791 | 0.696813 | 0.079137 | 0.309353 | 1.344790 |
| history trend | no_learning | 1.073892 | 0.237410 | 0.762590 | 0.640000 | 0.129496 | 1.486614 | 0.064748 | 0.676259 | 1.675081 |

## 当前判断

- RMSE 最低的是 `KNN template`，test RMSE=0.516941。
- `KNN template` 的 test RMSE=0.516941，但训练集 RMSE 近 0，属于模板记忆风险候选，不能直接当最终主线。
- `RBF KRR` 的 test RMSE=0.540287，大幅响应召回=0.600000，但反向修正计数匹配率=0.043165，说明它的复杂响应结构仍弱。
- `vehicle Transformer` 是真正的车辆-only Transformer：test RMSE=0.567162，优于 formal ridge 的 0.649341，也优于旧 `vehicle_direct active` 的 0.637366，但它的多段修正预测率与 GT 差距=0.676259，暂不能作为最终强车辆主线。
- 因此当前阶段 3 结论不是“某个模型胜出”，而是：车辆-only 已经有多个强对照，下一步必须用物理错误和稳健性验证冻结主车辆参照。

## 候选决策表

| display_name | decision | continue_priority | rmse_steer | rmse_improvement_pct_vs_formal | decision_note_cn |
| --- | --- | --- | --- | --- | --- |
| zero hold | reference | low | 0.683514 | -5.262610 | 无学习下界，只作参照。 |
| history trend | reference | low | 1.073892 | -65.381755 | 无学习趋势外推，尾段和错侧风险高，只作参照。 |
| train mean | reference | low | 0.685789 | -5.613022 | 训练集平均轨迹，证明平均化响应的下界。 |
| event mean | reference | low | 0.677212 | -4.292133 | 按事件类型平均，仍不能代表个体响应。 |
| ridge history | reference | low | 0.707027 | -8.883784 | 浅层车辆历史基线，保留为线性参照。 |
| formal ridge | keep_as_formal_reference | required | 0.649341 | 0.000000 | 正式 shallow vehicle baseline，是当前所有车辆模型的最低公平参照。 |
| old vehicle_direct active | old_flow_reference_only | medium | 0.637366 | 1.844291 | 旧代码 clean vehicle-only 历史对照，使用旧架构，不作为新流程主线。 |
| old vehicle_direct structure | old_flow_reference_only | medium | 0.647720 | 0.249668 | 旧代码结构 checkpoint 历史对照，可看物理指标但不继承旧流程假设。 |
| rich ridge history | not_selected_currently | low | 0.680683 | -4.826687 | 丰富车辆历史线性模型，没有超过强候选，不作为主线。 |
| rich ridge context | not_selected_currently | low | 0.652941 | -0.554396 | 丰富车辆上下文线性模型，方向指标尚可但 RMSE 不优。 |
| RBF KRR | strong_diagnostic_candidate_needs_controls | high | 0.540287 | 16.794625 | 非参数强候选，RMSE 和大幅召回好，但反向修正匹配很差，需稳健性验证。 |
| KNN template | diagnostic_upper_bound_memory_risk | medium | 0.516941 | 20.389968 | test RMSE 最低，但 train RMSE 近 0，模板记忆风险高，暂作诊断上限。 |
| direction-gated KNN | diagnostic_upper_bound_memory_risk | medium | 0.579581 | 10.743306 | KNN 变体，仍有模板记忆风险，暂作诊断。 |
| peak-scaled template | diagnostic_upper_bound_memory_risk | medium | 0.555055 | 14.520303 | 模板缩放候选，幅值指标好但仍需检查物理错误。 |
| vehicle Transformer | true_transformer_candidate_needs_structured_fix | high | 0.567162 | 12.655737 | 真正车辆-only Transformer，强于 formal ridge，但多段修正预测为 0，需继续结构化改进。 |

## 关键产物

- 指标总表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_vehicle_instability_unified_vehicle_comparison_v0_1\tables\unified_vehicle_comparison_metrics_test.csv`
- 相对 formal ridge 差异：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_vehicle_instability_unified_vehicle_comparison_v0_1\tables\unified_vehicle_comparison_delta_vs_formal_test.csv`
- 候选决策表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_vehicle_instability_unified_vehicle_comparison_v0_1\tables\unified_vehicle_candidate_decision_table.csv`
- 坏样本重合表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_vehicle_instability_unified_vehicle_comparison_v0_1\tables\unified_vehicle_top_bad_overlap.csv`
- 关键指标图：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_vehicle_instability_unified_vehicle_comparison_v0_1\figures\unified_vehicle_key_metrics_test.png`
- 物理错误热图：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_vehicle_instability_unified_vehicle_comparison_v0_1\figures\unified_vehicle_physical_failure_heatmap_test.png`
- RMSE/错侧权衡图：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_vehicle_instability_unified_vehicle_comparison_v0_1\figures\unified_vehicle_rmse_vs_wrong_side_test.png`
- 坏样本重合图：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_vehicle_instability_unified_vehicle_comparison_v0_1\figures\unified_vehicle_top_bad_overlap.png`

## 下一步

1. 先做 subject-level split 或窗口敏感性检查，验证 RBF/KNN/Transformer 的收益是否稳定。
2. 对 top bad overlap 中反复失败的样本绘图复盘，确认是事件锚点问题、车辆历史信息不足、还是模型结构问题。
3. 在冻结强车辆主参照前，继续阻塞连续风格、生理和 EEG 有效性结论。
