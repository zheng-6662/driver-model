# v237 rolling target / phase audit 报告

## 结论

- v237 是 audit-only，没有训练新模型，没有生成新预测，没有调 alpha/threshold/tau，也没有创建 gate/router/selector。
- v236 receding 指标已复现，consistency 最大差异 `2.3841858e-07`，容差 `1e-05`。
- target sanity check 全部 pass：v236 的 target 与 prediction 都在 `steering_delta_from_observation` 空间。
- v238_allowed = `True`。

## observe_later_like: receding vs original_remaining

- 0ms: receding_tail=1.178022，remaining_tail=1.178022，remaining_points=21
- 200ms: receding_tail=1.136082，remaining_tail=1.070851，remaining_points=19
- 400ms: receding_tail=1.877300，remaining_tail=1.307550，remaining_points=17
- 600ms: receding_tail=1.912394，remaining_tail=1.180526，remaining_points=15
- 800ms: receding_tail=1.441137，remaining_tail=0.942026，remaining_points=13
- 1000ms: receding_tail=4.074430，remaining_tail=1.199416，remaining_points=11

## strong_steer: receding vs original_remaining

- 0ms: receding_tail=1.018893，remaining_tail=1.018893，strong_under=0.762500
- 200ms: receding_tail=0.927601，remaining_tail=0.888611，strong_under=0.362500
- 400ms: receding_tail=0.872109，remaining_tail=0.802943，strong_under=0.227848
- 600ms: receding_tail=0.873736，remaining_tail=0.772004，strong_under=0.250000
- 800ms: receding_tail=0.819825，remaining_tail=0.698652，strong_under=0.338462
- 1000ms: receding_tail=0.814272，remaining_tail=0.529581，strong_under=0.238806

## observe_later 子桶

- observe_later_and_extreme_peak: n=3，receding_tail=1.089481，remaining_tail=0.729074，reverse_rate=1.000，zero_cross_rate=1.000
- observe_later_and_high_tail_error: n=19，receding_tail=4.823908，remaining_tail=1.400434，reverse_rate=0.684，zero_cross_rate=0.947
- observe_later_and_multi_correction: n=6，receding_tail=0.817944，remaining_tail=0.518549，reverse_rate=0.333，zero_cross_rate=0.833
- observe_later_and_reverse: n=15，receding_tail=5.419585，remaining_tail=1.549610，reverse_rate=1.000，zero_cross_rate=1.000
- observe_later_and_strong_steer: n=22，receding_tail=1.078108，remaining_tail=0.702932，reverse_rate=0.636，zero_cross_rate=0.955
- observe_later_and_zero_cross: n=25，receding_tail=4.229306，remaining_tail=1.232088，reverse_rate=0.600，zero_cross_rate=1.000
- observe_later_normal_direction: n=1，receding_tail=0.653833，remaining_tail=0.607351，reverse_rate=0.000，zero_cross_rate=0.000

## 1000ms failure audit

- observe_later_like test 样本数：27，命中 new phase 规则比例：0.889
- `rjy_Entity_Recording_2025_09_28_19_51_44_v108_039`: tail 0ms=1.218690 -> 1000ms=20.489019，delta=+19.270330，new_phase=True
- `rjy_Entity_Recording_2025_09_28_20_02_20_v108_037`: tail 0ms=0.948906 -> 1000ms=1.609251，delta=+0.660345，new_phase=True
- `rjy_Entity_Recording_2025_09_28_20_15_42_v108_002`: tail 0ms=0.393218 -> 1000ms=1.013341，delta=+0.620122，new_phase=True
- `tyy_Entity_Recording_2025_09_28_14_23_43_v108_002`: tail 0ms=1.317812 -> 1000ms=1.767759，delta=+0.449947，new_phase=True
- `rjy_Entity_Recording_2025_09_28_20_02_20_v108_015`: tail 0ms=0.686734 -> 1000ms=1.130754，delta=+0.444021，new_phase=True
- `tyy_Entity_Recording_2025_09_28_14_57_17_v108_004`: tail 0ms=0.369859 -> 1000ms=0.789407，delta=+0.419548，new_phase=True
- `rjy_Entity_Recording_2025_09_28_20_15_42_v108_010`: tail 0ms=0.397763 -> 1000ms=0.795394，delta=+0.397630，new_phase=True
- `rjy_Entity_Recording_2025_09_28_19_51_44_v108_014`: tail 0ms=1.559387 -> 1000ms=1.928083，delta=+0.368696，new_phase=True

## Ridge underfit

- all: v236_rmse=1.220571，old_formal=0.468061，gap=+0.752510，peak_shrinkage=0.685097，pred_var/target_var=1.687677
- observe_later_like: v236_rmse=0.958695，old_formal=0.785293，gap=+0.173402，peak_shrinkage=0.341520，pred_var/target_var=0.161242
- normal_predictable: v236_rmse=1.483471，old_formal=0.343042，gap=+1.140429，peak_shrinkage=1.177494，pred_var/target_var=6.907819
- strong_steer: v236_rmse=0.827447，old_formal=0.611697，gap=+0.215749，peak_shrinkage=0.418264，pred_var/target_var=0.307122
- alpha selected at max boundary: `True`，selected alpha=`1000`。

## 下一步决策

- recommended_next_task: `v238_small_rolling_model`
- reason: target_definition_sanity_all_pass=True; split_leakage_all_pass=True; original_remaining_observe_later_improves=True; strong_steer_improvement_maintained=True; 1000ms_degradation_phase_explained=True; ridge_underfit_evidence=True

## 输出

- `tables/v237_target_definition_sanity_check.csv`
- `tables/v237_receding_vs_original_remaining_metrics.csv`
- `tables/v237_observe_later_subbucket_profile.csv`
- `tables/v237_1000ms_failure_audit.csv`
- `tables/v237_ridge_underfit_audit.csv`
- ZIP：`v237_rolling_target_phase_audit_pack.zip`
