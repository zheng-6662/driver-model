# 最新产物索引：2026-07-05 v320 rank-budget repair gate
- 脚本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v320_rank_budget_repair_gate_20260705.py`
- 产物目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v320_rank_budget_repair_gate_20260705`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v320_rank_budget_repair_gate_20260705\reports\v320_rank_budget_repair_gate_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v320_rank_budget_repair_gate_20260705\v320_rank_budget_repair_gate_20260705.zip`
- 关键表：
  - `tables/v320_train_oof_quota_search.csv`：训练折外排序配额搜索表。
  - `tables/v320_selected_quota_config.csv`：最终固定配额配置。
  - `tables/v320_validation_budget_check.csv`：验证预算通过表。
  - `tables/v320_validation_budget_metrics.csv`：验证收益、风险和覆盖率指标。
  - `tables/v320_validation_coverage_budget.csv`：验证分组校正覆盖率。
  - `tables/v320_validation_channel_contribution.csv`：验证通道贡献。
  - `tables/v320_test_budget_metrics.csv`：测试收益、风险和覆盖率指标。
  - `tables/v320_test_coverage_budget.csv`：测试分组校正覆盖率。
  - `tables/v320_test_channel_contribution.csv`：测试通道贡献。
  - `tables/v320_test_candidate_family_usage.csv`：测试候选家族使用与收益。
- 关键图：
  - `figures/v320_validation_group_rmse.png`
  - `figures/v320_validation_candidate_usage.png`
- 守卫结果：
  - `guardrail.pass=True`
  - `goal_validation_passed=True`
  - `test_reported=True`
  - `candidate_selection_uses_test=False`
  - `uses_test_error_as_features=False`
  - `uses_future_truth_as_input=False`
  - `uses_hard20_as_gate_input=False`
  - `zip_testzip=True`
- 关键结果：
  - validation：全部样本收益 `0.001961`，普通样本收益 `0`，强方向盘收益 `0.003729`，困难前20收益 `0.005495`，困难前10收益 `0.002038`。
  - validation 覆盖：全部 `0.075829`，普通 `0`，强方向盘 `0.133929`，困难前20 `0.177778`，困难前10 `0.083333`。
  - test：全部样本收益 `0.000311`，普通样本收益 `0`，强方向盘收益 `0.000613`，困难前20收益 `-0.001521`，困难前10收益 `-0.005689`。
  - test 覆盖：全部 `0.076577`，普通 `0`，强方向盘 `0.128000`，困难前20 `0.130435`，困难前10 `0.083333`。
- 结论：v320 验证通过并解决了“全不改”与普通样本误伤；但测试困难组没有稳定泛化，下一步应诊断候选排序和候选家族风险预算。
- 验证：`D:\ProgramData\anaconda3\envs\predict_2\python.exe -m py_compile` 通过；完整运行通过；ZIP 自检通过。

---

# 最新产物索引：2026-07-04 v316 filtered current-window coarse-scene train
- 脚本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v316_filtered_current_window_coarse_scene_train_20260704.py`
- 产物目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v316_filtered_current_window_coarse_scene_train_20260704`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v316_filtered_current_window_coarse_scene_train_20260704\reports\v316_filtered_current_window_coarse_scene_train_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v316_filtered_current_window_coarse_scene_train_20260704\v316_filtered_current_window_coarse_scene_train_20260704.zip`
- 主要表：
  - `tables/v316_delay0_group_summary.csv`：过滤后 delay0 分组指标。
  - `tables/v316_model_selection_validation.csv`：过滤后 validation 选模表。
  - `tables/v316_per_sample_metrics_original_remaining.csv`：逐样本指标。
  - `tables/v316_metrics_by_delay_and_bucket.csv`：全 delay 和桶指标。
  - `tables/v316_v315_filtered_split_audit.csv`：第315版过滤后的划分审计。
  - `tables/v316_coarse_scene_aux_metrics.csv`：粗场景辅助头指标。
  - `tables/*training_history.csv`：3 个候选训练历史。
- 主要模型：
  - `models/v316_filtered_scene_init_aux003_film005_h64.pt`
  - `models/v316_filtered_scene_init_aux005_film010_h64.pt`
  - `models/v316_filtered_scene_init_aux006_film010_hard110_h64.pt`
- 预测包：`v316_filtered_current_window_predictions.npz`
- 关键结果：
  - 保留当前窗口事件 `1083`，隔离事件 `84`
  - train/val/test 保留事件 `650/211/222`
  - 选中候选：`v316_filtered_scene_init_aux003_film005_h64`
  - 过滤后 test/all：v300 `0.525580`，旧 v307 `0.496950`，v316 `0.502633`
  - 过滤后 test/bad10：v300 `0.859987`，旧 v307 `0.777797`，v316 `0.800171`
  - 过滤后 test/bad20：v300 `0.703038`，旧 v307 `0.651121`，v316 `0.660814`
  - 保留 severe 33 个：v300 `0.805638`，旧 v307 `0.877334`，v316 `0.886424`
- 结论：v316 验证了过滤清单可用于干净训练边界，但单纯过滤重训未超过旧 v307，也没有解决保留 severe 的幅值/相位/极端跟随问题。
- 验证：`python -m py_compile` 通过；完整训练通过；`logs/guardrail_check.json` 中 `pass=True`；ZIP 自检 `zip_testzip=True`。

---

# 最新产物索引：2026-07-04 v315 rapid steering filter / reanchor plan
- 脚本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v315_rapid_steering_filter_reanchor_plan_20260704.py`
- 产物目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v315_rapid_steering_filter_reanchor_plan_20260704`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v315_rapid_steering_filter_reanchor_plan_20260704\reports\v315_rapid_steering_filter_reanchor_plan_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v315_rapid_steering_filter_reanchor_plan_20260704\v315_rapid_steering_filter_reanchor_plan_20260704.zip`
- 主要表：
  - `tables/v315_current_window_training_policy_all_delay0.csv`：1167 个 delay0 事件的训练前处理策略。
  - `tables/v315_current_window_keep_manifest.csv`：1083 个当前窗口来源成立、可进入当前窗口训练的事件。
  - `tables/v315_current_window_isolate_manifest.csv`：84 个当前窗口来源可疑、需隔离的事件。
  - `tables/v315_reanchor_candidate_manifest.csv`：77 个候选重锚定事件。
  - `tables/v315_weak_fast_source_exclusion_candidates.csv`：7 个全程快转证据弱、候选剔除事件。
  - `tables/v315_split_filter_summary.csv`：按 train/val/test 的保留与隔离统计。
  - `tables/v315_policy_summary.csv`：处理策略分布。
  - `tables/v315_scene_policy_summary.csv`：粗场景与处理策略交叉汇总。
  - `tables/v315_reanchor_shift_summary.csv`：重锚定候选偏移统计。
- 主要图：
  - `figures/v315_split_keep_isolate_counts.png`
  - `figures/v315_policy_counts.png`
- 关键结果：`event_n=1167`，`current_window_keep_n=1083`，`current_window_isolate_n=84`，`reanchor_candidate_n=77`，`weak_fast_source_exclusion_candidate_n=7`，`severe_isolate_n=4`，`screenshot_isolate_n=1`。
- 按划分：train `650/702` 保留，val `211/233` 保留，test `222/232` 保留。
- 结论：当前窗口模型下一轮应使用保留清单；隔离清单不应继续作为当前强动作监督。重锚定候选不能直接改表训练，必须重新切窗口和目标曲线。
- 验证：`python -m py_compile` 通过；完整运行通过；`logs/guardrail_check.json` 中 `pass=True`；ZIP 自检 `zip_testzip=True`。

---

# 最新产物索引：2026-07-04 v314 rapid steering source sample audit
- 脚本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v314_rapid_steering_source_sample_audit_20260704.py`
- 产物目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v314_rapid_steering_source_sample_audit_20260704`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v314_rapid_steering_source_sample_audit_20260704\reports\v314_rapid_steering_source_sample_audit_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v314_rapid_steering_source_sample_audit_20260704\v314_rapid_steering_source_sample_audit_20260704.zip`
- 主要表：
  - `tables/v314_rapid_steering_source_audit_all_delay0.csv`：1167 个 delay0 事件的方向盘快转来源全量审计表。
  - `tables/v314_rapid_steering_source_sample_cases.csv`：72 个固定种子抽样排查样本。
  - `tables/v314_source_category_summary.csv`：来源分级汇总。
  - `tables/v314_scene_by_source_category_summary.csv`：粗场景与来源分级交叉汇总。
  - `tables/v314_steering_rate_quantiles.csv`：方向盘转动速度分位数。
- 主要图：
  - `figures/v314_current_window_steering_rate_distribution.png`
  - `figures/v314_source_category_counts.png`
  - `figures/sample_cases/*.png`：72 张抽样排查图。
- 关键结果：`event_n=1167`，`current_fast_steer_supported_n=1083`，`suspect_not_current_fast_steer_n=84`，`severe_n=37`，`severe_suspect_not_current_fast_steer_n=4`，`screenshot_n=5`，`screenshot_suspect_not_current_fast_steer_n=1`。
- 结论：用户强调的“方向盘快速转动来源”总体成立，但 `84` 个样本存在当前窗口快转证据不足或来源错位；其中截图 #020 是典型“当前平缓、后续才快转”。后续应先隔离这类样本，再处理来源成立但模型仍预测差的幅值/相位问题。
- 验证：`python -m py_compile` 通过；完整运行通过；`logs/guardrail_check.json` 中 `pass=True`；ZIP 自检 `zip_testzip=True`。

---

# 最新产物索引：2026-07-04 v312 horizon-aligned label / anchor audit
- 脚本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v312_horizon_aligned_label_anchor_audit_20260704.py`
- 产物目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v312_horizon_aligned_label_anchor_audit_20260704`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v312_horizon_aligned_label_anchor_audit_20260704\reports\v312_horizon_aligned_label_anchor_audit_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v312_horizon_aligned_label_anchor_audit_20260704\v312_horizon_aligned_label_anchor_audit_20260704.zip`
- 关键表：
  - `tables/v312_horizon_aligned_delay0_event_labels.csv`：1167 个 delay0 事件的 local 0-2s 标签与 late 2-6s 标签。
  - `tables/v312_v309_severe_horizon_label_overlay.csv`：v309 severe 37 个事件叠加 horizon-aligned 标签和建议动作。
  - `tables/v312_coarse_local_late_crosstab.csv`：粗标签、local 标签、late 标签交叉表。
  - `tables/v312_horizon_alignment_summary_by_split.csv`：按 split 的对齐状态分布。
- 关键图：
  - `figures/v312_local_0_2_label_distribution.png`
  - `figures/v312_horizon_alignment_by_split.png`
- 关键结果：`event_n=1167`，`coarse_label_horizon_mismatch_n=227`，`local_flat_late_large_n=49`，`local_late_direction_conflict_n=98`，`severe_overlay_n=37`，`severe_coarse_label_horizon_mismatch_n=11`。
- 边界：`local_0_2_motion_label` 来自真实 0-2s 目标曲线，`late_2_6_context_label` 来自 raw 后续 2-6s；二者都不能直接当作原锚点部署输入，必须先转成可部署人工/实验条件标签或仅用于诊断。

---

# 最新产物索引：2026-07-04 v310/v311 severe-case 修改与审计
- v310 训练脚本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v310_severe_error_targeted_curve_model_20260704.py`
- v310 产物目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v310_severe_error_targeted_curve_model_20260704`
- v310 报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v310_severe_error_targeted_curve_model_20260704\reports\v310_severe_error_targeted_curve_model_cn.md`
- v310 关键表：
  - `tables/v310_model_selection_validation.csv`
  - `tables/v310_v309_severe_group_summary.csv`
  - `tables/v310_v309_severe_event_comparison.csv`
- v310 结果：guardrail 通过；选中 `v310_v307init_shape_guard_lo`；test/all `0.496138 -> 0.494998`，test/bad10 `0.777797 -> 0.775882`，但 v309 severe 37 个 `0.888400 -> 0.890705`，截图 5 个 `1.755055 -> 1.775683`，因此 v310 不能视为 severe 修复。
- v311 审计脚本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v311_severe_anchor_horizon_misalignment_audit_20260704.py`
- v311 产物目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v311_severe_anchor_horizon_misalignment_audit_20260704`
- v311 报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v311_severe_anchor_horizon_misalignment_audit_20260704\reports\v311_severe_anchor_horizon_misalignment_audit_cn.md`
- v311 关键表：
  - `tables/v311_severe_anchor_horizon_misalignment_audit.csv`
  - `tables/v311_misalignment_summary.csv`
- v311 结果：37 个 severe 都读到 raw 后续；`11/37` 有 horizon/label mismatch 嫌疑；截图 5 个中 `3/5` 命中；`false_large_maneuver` 中 `2/3` 命中；下一步建议做 horizon-aligned label / anchor 修正。

---

# 最新产物索引：2026-07-04 v309 严重方向/意图错误复核表
- 产物目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v309_recent_best_prediction_effect_gallery_20260704`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v309_recent_best_prediction_effect_gallery_20260704\reports\v309_severe_direction_or_intent_errors_cn.md`
- CSV 明细：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v309_recent_best_prediction_effect_gallery_20260704\tables\v309_severe_direction_or_intent_errors.csv`
- 来源：用户在 v309 近期最好预测图册中指出部分样本存在“完全偏离方向”的严重错误。
- 关键结果：`test_delay0_event_n=232`，`severe_candidate_n=37`，`user_screenshot_match_n=5`。
- 截图命中事件：`#014`、`#017`、`#019`、`#020`、`#023`。
- 错误类型：`opposite_peak_direction`、`false_large_maneuver`、`missed_extreme_amplitude`、`large_event_high_rmse`、`regression_vs_v300`。

---

## 最新指针：2026-07-02 v300 within-subject full joint-curve retrain

- 总结：v300 固定 v299 within-subject 事件级划分，并映射回全部 rolling delay 样本后完整重训 joint curve decoder。结果 `guardrail.pass=True`，`event_n=1167`，`rolling_sample_n=7002`，`train/val/test rolling=4212/1398/1392`，`train/val/test event=702/233/232`，`event_in_multiple_splits_n=0`，`duplicate_event_delay_rows_n=0`。训练候选包括 `v300_full_joint_h64_no_subject`、`v300_full_joint_h64_subject_onehot`、`v300_full_joint_h96_subject_onehot`；validation 选中 `v300_full_joint_h64_no_subject`，说明 subject one-hot 没有在本轮带来稳定收益。delay0 test/all RMSE `0.5198`，旧 v249 诊断参照 `0.3246`；delay0 test/within_bad_top10 RMSE `0.8600`，旧 v249 诊断参照 `1.0383`。旧 v249 只作诊断参照，且 within-test 中旧 v249 原 train 暴露比例 `0.5819`，不能作为正式公平基线。
- 入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v300_within_subject_full_joint_curve_train_20260702.py`
- 产物：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v300_within_subject_full_joint_curve_train_20260702`
- 报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v300_within_subject_full_joint_curve_train_20260702\reports\v300_within_subject_full_joint_curve_train_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v300_within_subject_full_joint_curve_train_20260702\v300_within_subject_full_joint_curve_train_20260702.zip`
- 核心表：`tables\v300_model_selection_validation.csv`，`tables\v300_metrics_by_delay_and_bucket.csv`，`tables\v300_delay0_group_summary.csv`，`tables\v300_delay0_event_wide_comparison.csv`，`tables\v300_per_sample_metrics_original_remaining.csv`，`tables\v300_within_subject_split_audit.csv`，`tables\v300_input_variant_audit.csv`
- 核心图：`figures\v300_training_history.png`，`figures\v300_test_delay0_group_rmse.png`，`figures\v300_test_selected_bad_top6_curves.png`
- 校验：`logs\guardrail_check.json`，`logs\input_hashes.csv`，`logs\file_inventory.csv`；ZIP 自检 `testzip=True`。

---

# 最新产物索引：2026-07-04 v308 coarse scene 视觉人工复核包

- 产物目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v308_coarse_scene_visual_manual_review_20260704`
- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v308_coarse_scene_visual_manual_review_20260704.py`
- HTML 图册入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v308_coarse_scene_visual_manual_review_20260704\index.html`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v308_coarse_scene_visual_manual_review_20260704\reports\v308_coarse_scene_visual_manual_review_cn.md`
- 打包文件：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v308_coarse_scene_visual_manual_review_20260704\v308_coarse_scene_visual_manual_review_20260704.zip`
- 关键表：
  - `tables/v308_visual_review_manifest.csv`：748 个 high + medium 复核事件的图像路径、候选标签、图上统计量。
  - `tables/v308_manual_review_decision_template.csv`：人工复核结果填写模板。
- 关键图：
  - `figures/priority_review/*.png`：逐事件复核曲线图，共 `748` 张。
- 日志与校验：
  - `logs/guardrail_check.json`：`review_event_n=748`，`image_n=748`，`uses_future_response_for_manual_review=true`，`future_response_used_as_model_input=false`。
  - ZIP 自检：`testzip=None`，`png_n=748`，`has_index=True`。
- 用途：
  - 给用户人工复核 v306 的 `continuous_lane_change`、`emergency_lane_change_instability`、`other_or_uncertain` seed。
  - 支持在浏览器中看图、筛选、填写复核结论并导出 CSV。
  - 不能作为训练输入本身；图中锚点后真实响应只用于人工标注确认。

---

# 最新产物索引：2026-07-04 v309 近期最好模型预测效果图册

- 产物目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v309_recent_best_prediction_effect_gallery_20260704`
- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v309_recent_best_prediction_effect_gallery_20260704.py`
- HTML 图册入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v309_recent_best_prediction_effect_gallery_20260704\index.html`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v309_recent_best_prediction_effect_gallery_20260704\reports\v309_recent_best_prediction_effect_gallery_cn.md`
- 打包文件：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v309_recent_best_prediction_effect_gallery_20260704\v309_recent_best_prediction_effect_gallery_20260704.zip`
- 关键表：
  - `tables/v309_test_delay0_prediction_effect_table.csv`：test delay0 逐事件 v300/v307 RMSE 对比表。
  - `tables/v309_gallery_sample_manifest.csv`：54 个代表性图册样本清单。
  - `tables/v309_group_rmse_from_npz.csv`：按 all、bad_top10、bad_top20、strong_steer、vehicle_ambiguous 的均值 RMSE。
- 关键图：
  - `figures/v309_group_rmse_v307_vs_v300.png`
  - `figures/v309_event_scatter_v307_vs_v300.png`
  - `figures/prediction_cases/*.png`：逐事件预测效果图，共 `54` 张。
- 日志与校验：
  - `logs/guardrail_check.json`：`test_delay0_event_n=232`，`gallery_case_n=54`，`model_prediction_horizon_s=[0.0,2.0]`，`extended_true_future_is_model_prediction=false`。
  - ZIP 自检：`testzip=None`，`png_n=56`，`has_index=True`。
- 用途：
  - 观察近期最好版本 v307 selected model 的真实预测效果。
  - 对比 v307 与 v300 在代表性样本上的差异。
  - 展示 2s 后真实后续走势，帮助理解模型 0-2s 预测之后事件如何发展；2s 后不是模型预测范围。

---

## 最新指针：2026-07-02 v299 within-subject split residual calibration

- 总结：v299 按用户要求改成同一被试内随机切分 train/val/test，且同一 `event_uid` 不跨 split。结果 `guardrail.pass=True`，`event_n=1167`，`train/val/test=702/233/232`，`duplicate_event_uid_n=0`，`event_in_multiple_splits_n=0`，18 个被试都同时出现在三个 split。固定 v249 预测上做快速 residual 校准，val 选中 `base_curve_meta_subject__extra_trees_d5`，test all delta `-0.0067 RMSE`，test within_bad_top10 delta `-0.0738 RMSE`。边界：`full_v249_retrained=False`，新 within-test 中 `58.2%` 原本属于旧 v249 train split，所以本轮是潜力审计，不是 formal retrain 结果。
- 入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v299_within_subject_split_residual_calibration_20260702.py`
- 产物：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v299_within_subject_split_residual_calibration_20260702`
- 报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v299_within_subject_split_residual_calibration_20260702\reports\v299_within_subject_split_residual_calibration_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v299_within_subject_split_residual_calibration_20260702_pack.zip`
- 核心表：`tables\v299_within_subject_split_event_table.csv`，`tables\v299_within_subject_split_subject_counts.csv`，`tables\v299_within_vs_original_split_crosstab.csv`，`tables\v299_within_subject_residual_summary.csv`，`tables\v299_within_subject_residual_event_deltas.csv`，`tables\v299_chosen_by_val.csv`，`tables\v299_chosen_test_delta_by_original_split.csv`
- 核心图：`figures\v299_within_subject_split_counts.png`，`figures\v299_test_delta_by_method.png`，`figures\v299_test_bad_top6_curves.png`
- 校验：`logs\guardrail_check.json`，`logs\input_hashes.csv`，`logs\file_inventory.csv`

---

## 最新指针：2026-07-02 v298 event label explanatory audit

- 总结：v298 检查当前 1167 个事件的事件/响应标签解释力。结果 `guardrail.pass=True`，`event_n=1167`。粗响应标签能识别风险：`oracle_strength_label` 对 test bad_top10 的 AUC 为 `0.7735`；但 label-known 残差修正上限很弱，最佳 `oracle_shape` 只让 test bad_top10 delta `-0.0093 RMSE`，test all delta `-0.0013`。历史规则标签只能匹配 all `22.7%`、test `28.3%` 当前事件，覆盖不足。结论：当前没有足够覆盖、可部署、锚点前可知的事件标签；粗响应标签是风险提示，不是轨迹修正方案。
- 入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v298_event_label_explanatory_audit_20260702.py`
- 产物：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v298_event_label_explanatory_audit_20260702`
- 报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v298_event_label_explanatory_audit_20260702\reports\v298_event_label_explanatory_audit_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v298_event_label_explanatory_audit_20260702_pack.zip`
- 核心表：`tables\v298_event_label_audit_table.csv`，`tables\v298_historical_rule_label_time_match.csv`，`tables\v298_label_family_catalog.csv`，`tables\v298_label_level_bad_rmse_summary.csv`，`tables\v298_label_risk_auc_from_train_rates.csv`，`tables\v298_label_numeric_eta_summary.csv`，`tables\v298_label_known_residual_correction_summary.csv`，`tables\v298_label_known_residual_event_deltas.csv`，`tables\v298_bad_sample_label_casebook.csv`，`tables\v298_event_label_route_decision.csv`
- 核心图：`figures\v298_oracle_label_bad_rate.png`，`figures\v298_label_known_residual_delta.png`，`figures\v298_history_label_match_coverage.png`
- 校验：`logs\guardrail_check.json`，`logs\input_hashes.csv`，`logs\file_inventory.csv`

---

## 最新指针：2026-07-02 v297 subject style stability audit

- 总结：v297 审计同一被试多次独立事件是否存在稳定驾驶风格/响应倾向。结果 `guardrail.pass=True`，`event_n=1167`，`key_subject_eta_train_mean=0.05984`，`same_subject_mean_distance_ratio=0.71030`，`rolling_history_test_relative_rmse_improvement_mean_history3=0.06988`，但 `rolling_history_test_positive_target_rate_history3=0.28571`，`binary_history_test_auc_mean_history3=0.53109`。结论：存在弱 subject/style 信号，但不足以作为主线；下一步应优先事件级标签/实验条件标签。
- 入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v297_subject_style_stability_audit_20260702.py`
- 产物：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v297_subject_style_stability_audit_20260702`
- 报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v297_subject_style_stability_audit_20260702\reports\v297_subject_style_stability_audit_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v297_subject_style_stability_audit_20260702_pack.zip`
- 核心表：`tables\v297_event_response_descriptors.csv`，`tables\v297_subject_recording_eta.csv`，`tables\v297_pair_distance_summary.csv`，`tables\v297_rolling_history_predictability.csv`，`tables\v297_binary_history_auc.csv`，`tables\v297_oracle_label_candidate_counts.csv`，`tables\v297_style_route_decision.csv`
- 核心图：`figures\v297_subject_eta_by_descriptor.png`，`figures\v297_rolling_history_improvement.png`，`figures\v297_same_vs_different_subject_distance.png`
- 校验：`logs\guardrail_check.json`，`logs\input_hashes.csv`，`logs\file_inventory.csv`

## 历史指针：2026-07-02 v294 post-response candidate wait ranker

- 总结：v294 将 v293 的 post-response 生理可见性转成 RMSE 候选选择任务：等待 1/2/3/5 秒后，用 query/prototype 的 post 生理响应匹配 v292 的 40 个 vehicle-similar train prototype 候选，并由 val 选择是否覆盖 latest。结果 `route_viable_now=false`：val no-harm active 策略存在，但 test bad_top10 delta `+0.0070`；test-best diagnostic 只有 `-0.0112`，且 val bad_top10 `+0.1239`、val all `+0.0606`，不可部署。结论是 post 生理能识别风险，但不能稳定选择正确未来候选。
- 入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v294_post_response_candidate_wait_ranker_20260702.py`
- 产物：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v294_post_response_candidate_wait_ranker_20260702`
- 报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v294_post_response_candidate_wait_ranker_20260702\reports\v294_post_response_candidate_wait_ranker_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v294_post_response_candidate_wait_ranker_20260702_pack.zip`
- 核心表：`tables\v294_candidate_pool_oracle_summary.csv`，`tables\v294_train_only_feature_screen.csv`，`tables\v294_feature_block_audit.csv`，`tables\v294_wait_ranker_threshold_summary.csv`，`tables\v294_wait_ranker_selected_per_event_thresholds.csv`，`tables\v294_wait_ranker_chosen_by_val.csv`，`tables\v294_route_decision.csv`
- 核心图：`figures\v294_chosen_wait_ranker_delta.png`，`figures\v294_top_diagnostic_wait_rankers.png`
- 校验：`logs\guardrail_check.json`，`logs\input_hashes.csv`，`logs\file_inventory.csv`

## 最新指针：2026-07-02 v293 physiology response visibility / latency audit

- 总结：v293 检查生理差异是在 observation 前可见，还是 observation 后才出现。结果 `guardrail.pass=True`、`zip_testzip=True`，`event_n=1167`，`feature_n=540`，`screen_feature_n=540`，`ok_rate=0.91945`。主差样本 `bad_top10` 在 pre 窗口不可见，best pre test AUC `0.4896`；但 early-post 响应明显，best early-post test AUC `0.7726`，`window_post0_3` AUC `0.7254`，`window_post0_2` AUC `0.7053`。结论是生理价值主要来自 event 后短时间响应，不是原锚点前静态差异。
- 入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v293_physio_response_visibility_latency_audit_20260702.py`
- 产物：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v293_physio_response_visibility_latency_audit_20260702`
- 报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v293_physio_response_visibility_latency_audit_20260702\reports\v293_physio_response_visibility_latency_audit_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v293_physio_response_visibility_latency_audit_20260702_pack.zip`
- 核心表：`tables\v293_prepost_physio_visibility_features.csv`，`tables\v293_train_only_feature_screen.csv`，`tables\v293_feature_sets.csv`，`tables\v293_window_signal_screen_summary.csv`，`tables\v293_visibility_classifier_summary.csv`，`tables\v293_visibility_decision.csv`
- 核心图：`figures\v293_phase_test_auc.png`，`figures\v293_window_signal_screen_corr.png`，`figures\v293_window_valid_ratio.png`
- 校验：`logs\guardrail_check.json`，`logs\input_hashes.csv`，`logs\file_inventory.csv`

## 最新指针：2026-07-02 v292 source-physio pairwise candidate ranker

- 总结：v292 固定使用 v278 `listrank_vehicle` 的 40 个 vehicle-similar train prototype 候选，用 query/prototype 的 ECG/RESP/EDA 源生理 pair 差异学习候选排序。结果 `route_viable_now=false`：candidate-pool oracle 在 test bad_top10 上有 `-0.0784 RMSE` 空间，在 bad_top10_vehicle_ambiguous 上有 `-0.0881` 空间，但 validation 没有 no-harm active pairwise selector。test-best diagnostic `bio_all_top_pair_only__hgb_d3` 在 test bad_top10 上为 `-0.0248`，但 val bad_top10 伤害 `+0.1402`、val all 伤害 `+0.0367`，不可部署。
- 入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v292_source_physio_pairwise_candidate_ranker_20260702.py`
- 产物：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v292_source_physio_pairwise_candidate_ranker_20260702`
- 报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v292_source_physio_pairwise_candidate_ranker_20260702\reports\v292_source_physio_pairwise_candidate_ranker_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v292_source_physio_pairwise_candidate_ranker_20260702_pack.zip`
- 核心表：`tables\v292_pairwise_candidate_table.csv`，`tables\v292_bio_feature_sets.csv`，`tables\v292_feature_block_audit.csv`，`tables\v292_candidate_pool_oracle_summary.csv`，`tables\v292_pairwise_threshold_summary.csv`，`tables\v292_pairwise_chosen_by_val.csv`，`tables\v292_route_decision.csv`
- 核心图：`figures\v292_candidate_pool_oracle_delta.png`，`figures\v292_chosen_selector_badtop10_delta.png`，`figures\v292_diagnostic_top_selectors.png`
- 校验：`logs\guardrail_check.json`，`logs\input_hashes.csv`，`logs\file_inventory.csv`

## 最新指针：2026-07-02 v291 multi-signal physiology supervised probe

- 总结：v291 合并 v288 ECG、v289 RESP、v290 EDA 三路 causal 源信号，并用 v278 的 `latest / listrank_vehicle / listrank_vehicle_bio / listrank_vehicle_style_bio` 现成方法池做监督 selector 与分类探针。结果 `route_viable_now=false`：method-pool oracle 在 test bad_top10 有 `-0.0402 RMSE` 事后上限，但 validation 没有任何 no-harm active selector；test-best diagnostic 仅 `-0.0093 RMSE` 且不可部署。源生理识别 test bad_top10 的最好 AUC 只有 `0.5394`。
- 入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v291_multisignal_physio_supervised_probe_20260702.py`
- 产物：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v291_multisignal_physio_supervised_probe_20260702`
- 报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v291_multisignal_physio_supervised_probe_20260702\reports\v291_multisignal_physio_supervised_probe_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v291_multisignal_physio_supervised_probe_20260702_pack.zip`
- 核心表：`tables\v291_multisignal_event_table.csv`，`tables\v291_train_only_bio_feature_screen.csv`，`tables\v291_feature_blocks.csv`，`tables\v291_method_pool_summary.csv`，`tables\v291_selector_threshold_summary.csv`，`tables\v291_selector_chosen_by_val.csv`，`tables\v291_classification_probe_summary.csv`，`tables\v291_route_decision.csv`
- 核心图：`figures\v291_method_pool_badtop10_delta.png`，`figures\v291_selector_chosen_badtop10_delta.png`，`figures\v291_feature_screen_source_counts.png`
- 校验：`logs\guardrail_check.json`，`logs\input_hashes.csv`，`logs\file_inventory.csv`

## 最新指针：2026-07-02 v290 EDA/SCR usable-subset source route audit

- 总结：v290 回到 cleaned 200Hz EDA 源信号，直接读取 `EDA_raw200/EDA_filt200/EDA_Tonic/EDA_Phasic`，重建 tonic/phasic/SCR-like peak/短窗动态/质量特征，并显式评估 EDA 可用子集。结果 `route_viable_now=false` 且 `eda_subset_route_viable_now=false`：EDA 可用事件 `906/1167`，deployable top1 在 test bad_top10 上仍比 latest 差 `+0.1760`，在 bad_top10_vehicle_ambiguous 上差 `+0.1601`；test-best top1 仍差 `+0.1409`，best corr `0.0306`。EDA/SCR 可用子集也没有把差样本拉回来。
- 入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v290_eda_scr_usable_subset_route_audit_20260702.py`
- 产物：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v290_eda_scr_usable_subset_route_audit_20260702`
- 报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v290_eda_scr_usable_subset_route_audit_20260702\reports\v290_eda_scr_usable_subset_route_audit_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v290_eda_scr_usable_subset_route_audit_20260702_pack.zip`
- 核心表：`tables\v290_eda_source_features.csv`，`tables\v290_train_only_feature_screen.csv`，`tables\v290_eda_quality_by_recording.csv`，`tables\v290_route_group_summary.csv`，`tables\v290_val_chosen_generalization.csv`，`tables\v290_route_gate_decision.csv`，`tables\v290_eda_subset_route_decision.csv`
- 核心图：`figures\v290_badtop10_val_test_delta.png`，`figures\v290_eda_feature_screen_summary.png`，`figures\v290_eda_usable_subset_delta.png`
- 校验：`logs\guardrail_check.json`，`logs\input_hashes.csv`，`logs\file_inventory.csv`

## 最新指针：2026-07-02 v289 RESP source phase route audit

- 总结：v289 回到 cleaned 200Hz RESP 源信号，重建呼吸周期、相位、幅值、质量和因果同步偏移特征，复用 v278 vehicle top40 route gate。结果 `route_viable_now=false`：deployable top1 在 test bad_top10 上仍比 latest 差 `+0.1553`，在 bad_top10_vehicle_ambiguous 上差 `+0.1251`；test-best top1 仍差 `+0.0625`，best corr `0.0463`。RESP 源信号比 ECG 更接近 latest，但仍不能稳定转成可部署候选选择。
- 入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v289_resp_source_phase_route_audit_20260702.py`
- 产物：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v289_resp_source_phase_route_audit_20260702`
- 报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v289_resp_source_phase_route_audit_20260702\reports\v289_resp_source_phase_route_audit_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v289_resp_source_phase_route_audit_20260702_pack.zip`
- 核心表：`tables\v289_resp_source_features.csv`，`tables\v289_train_only_feature_screen.csv`，`tables\v289_resp_quality_by_recording.csv`，`tables\v289_route_group_summary.csv`，`tables\v289_val_chosen_generalization.csv`，`tables\v289_route_gate_decision.csv`
- 核心图：`figures\v289_badtop10_val_test_delta.png`，`figures\v289_resp_offset_group_summary.png`，`figures\v289_resp_feature_screen_summary.png`
- 校验：`logs\guardrail_check.json`，`logs\input_hashes.csv`，`logs\file_inventory.csv`

## 最新指针：2026-07-02 v288 ECG source-signal route audit

- 总结：v288 回到 cleaned 200Hz ECG 源信号，重新提取 R 峰/RR、短窗形态、质量和因果同步偏移特征，复用 v278 vehicle top40 route gate。结果 `route_viable_now=false`：deployable top1 在 test bad_top10 上仍比 latest 差 `+0.1556`，在 bad_top10_vehicle_ambiguous 上差 `+0.1510`；test-best top1 仍差 `+0.0903`，best corr `0.0620`。
- 入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v288_ecg_source_signal_route_audit_20260702.py`
- 产物：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v288_ecg_source_signal_route_audit_20260702`
- 报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v288_ecg_source_signal_route_audit_20260702\reports\v288_ecg_source_signal_route_audit_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v288_ecg_source_signal_route_audit_20260702_pack.zip`
- 核心表：`tables\v288_ecg_source_features.csv`，`tables\v288_train_only_feature_screen.csv`，`tables\v288_ecg_quality_by_recording.csv`，`tables\v288_route_group_summary.csv`，`tables\v288_val_chosen_generalization.csv`，`tables\v288_route_gate_decision.csv`
- 核心图：`figures\v288_badtop10_val_test_delta.png`，`figures\v288_ecg_offset_group_summary.png`，`figures\v288_ecg_feature_screen_summary.png`
- 校验：`logs\guardrail_check.json`，`logs\input_hashes.csv`，`logs\file_inventory.csv`

## 最新指针：2026-07-02 v287 physiology temporal-window route audit

- 总结：v287 复用 v285 已因果抽取的 raw 200Hz shape-state 特征，按时间窗口、信号族、特征类型和窗口×信号组合拆成 47 个 feature set，重新过 v278 vehicle top40 route gate。结果 `route_viable_now=false`：deployable top1 在 test bad_top10 上比 latest 差 `+0.2379`，在 bad_top10_vehicle_ambiguous 上差 `+0.2314`。test-best 诊断显示 ECG 最近 1-2 秒有弱信号：`combo_pre2_0_ecg_top16` top1 仍差 `+0.0941`，`combo_pre1_0_ecg_top16` corr `0.0854`，但都不是可部署改善。
- 入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v287_physio_temporal_window_route_audit_20260702.py`
- 产物：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v287_physio_temporal_window_route_audit_20260702`
- 报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v287_physio_temporal_window_route_audit_20260702\reports\v287_physio_temporal_window_route_audit_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v287_physio_temporal_window_route_audit_20260702_pack.zip`
- 核心表：
  - `tables\v287_enriched_train_only_feature_screen.csv`
  - `tables\v287_feature_set_audit.csv`
  - `tables\v287_train_scaler_audit.csv`
  - `tables\v287_route_gate_per_event.csv`
  - `tables\v287_route_group_summary.csv`
  - `tables\v287_val_chosen_generalization.csv`
  - `tables\v287_route_gate_decision.csv`
  - `tables\v287_group_winner_summary.csv`
- 核心图：
  - `figures\v287_window_badtop10_top1_delta.png`
  - `figures\v287_signal_bad_ambiguous_corr.png`
  - `figures\v287_group_type_winners.png`
- 校验：`logs\guardrail_check.json`，`logs\input_hashes.csv`，`logs\file_inventory.csv`

## 最新指针：2026-07-02 v286 raw-200Hz online subject-aware calibration

- 总结：v286 在 v285 subject-disjoint route gate 失败后，单独测试 subject-aware / online adaptation 边界。它使用 v285 train-only 选出的 130 个 raw285 特征做同驾驶员历史 KNN 校准。test bad_top10 中 fixed wait-latest 为 `0.6950`，online raw285 KNN vehicle 为 `0.7358`，vehicle+raw285 后 raw285 KNN 为 `0.7197`，均未超过 wait-latest；raw285 KNN 相对纯 subject mean online 反而变差。
- 入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v286_raw200_online_subject_calibration_20260702.py`
- 产物：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v286_raw200_online_subject_calibration_20260702`
- 报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v286_raw200_online_subject_calibration_20260702\reports\v286_raw200_online_subject_calibration_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v286_raw200_online_subject_calibration_20260702_pack.zip`
- 核心表：
  - `tables\v286_event_online_predictions.csv`
  - `tables\v286_selected_wait_gate_by_strategy.csv`
  - `tables\v286_online_strategy_summary.csv`
  - `tables\v286_feature_block_audit.csv`
  - `tables\v286_raw285_feature_source_audit.csv`
  - `tables\v286_raw285_fill_scale_audit.csv`
  - `tables\v286_global_model_feature_audit.csv`
  - `tables\v286_online_history_audit.csv`
- 核心图：
  - `figures\v286_raw285_online_badtop10.png`
- 校验：`logs\guardrail_check.json`，`logs\input_hashes.csv`，`logs\file_inventory.csv`

## 最新指针：2026-07-02 v285 raw-200Hz signal-shape physiology route gate

- 总结：v285 不复用 v260 biomarker，而是从 cleaned 200Hz 连续信号直接构造质量、短窗形态、导数/突变、节律/相位、跨信号耦合和 causal past percentile 等 1146 个 raw shape-state 特征，并在 v278 vehicle top40 候选池中过 route gate。结果 `route_viable_now=false`：deployable top1 在 test bad_top10 上比 latest 差 `+0.1958`，在 bad_top10_vehicle_ambiguous 上差 `+0.1826`；test-best top1 也差 `+0.1578`。
- 入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v285_raw200_shape_state_route_gate_20260702.py`
- 产物：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v285_raw200_shape_state_route_gate_20260702`
- 报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v285_raw200_shape_state_route_gate_20260702\reports\v285_raw200_shape_state_route_gate_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v285_raw200_shape_state_route_gate_20260702_pack.zip`
- 核心表：
  - `tables\v285_raw200_shape_state_features.csv`
  - `tables\v285_raw200_shape_state_features_with_targets.csv`
  - `tables\v285_train_only_feature_screen.csv`
  - `tables\v285_feature_screen_summary.csv`
  - `tables\v285_feature_set_audit.csv`
  - `tables\v285_train_scaler_audit.csv`
  - `tables\v285_route_gate_per_event.csv`
  - `tables\v285_route_group_summary.csv`
  - `tables\v285_val_chosen_generalization.csv`
  - `tables\v285_route_gate_decision.csv`
- 核心图：
  - `figures\v285_badtop10_val_test_delta.png`
  - `figures\v285_feature_screen_by_family.png`
  - `figures\v285_bad_ambiguous_corr.png`
- 校验：`logs\guardrail_check.json`，`logs\input_hashes.csv`，`logs\file_inventory.csv`

## 最新指针：2026-07-02 v284 dynamic low-identity physiology route gate

- 总结：v284 按 v283 的要求重新构造低身份、动态生理状态特征，并在 v278 vehicle top40 候选池里重新计算生理距离排序。结果 `route_viable_now=false`：deployable top1 在 test bad_top10 上比 latest 差 `+0.1697`，在 test bad_top10_vehicle_ambiguous 上差 `+0.1903`；test-best top1 diagnostic 也差 `+0.1525`。best corr `0.0553` 只说明有弱排序苗头，未形成可部署选择。
- 入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v284_dynamic_low_identity_physio_route_gate_20260702.py`
- 产物：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v284_dynamic_low_identity_physio_route_gate_20260702`
- 报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v284_dynamic_low_identity_physio_route_gate_20260702\reports\v284_dynamic_low_identity_physio_route_gate_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v284_dynamic_low_identity_physio_route_gate_20260702_pack.zip`
- 核心表：
  - `tables\v284_train_only_feature_screen.csv`
  - `tables\v284_feature_set_audit.csv`
  - `tables\v284_route_gate_per_event.csv`
  - `tables\v284_route_group_summary.csv`
  - `tables\v284_val_chosen_generalization.csv`
  - `tables\v284_route_gate_decision.csv`
- 核心图：
  - `figures\v284_badtop10_val_test_delta.png`
  - `figures\v284_bad_ambiguous_corr.png`
  - `figures\v284_feature_screen_identity_vs_behavior.png`
- 校验：`logs\guardrail_check.json`，`logs\input_hashes.csv`，`logs\file_inventory.csv`

## 最新指针：2026-07-02 v283 physiology route lineage / gap audit

- 总结：v283 合并 v254b-v282 的结构化证据，形成生理路线级收口。结论：`current_goal_achieved=false`，`old_feature_selector_route_closed=true`，`physio_source_alignment_ready=true`，`next_route_requires_feature_redefinition=true`。旧特征/旧候选选择器路线不应继续微调；若继续生理目标，必须先构造低身份但行为相关的新生理状态表示，并先通过车辆歧义样本 route gate。
- 入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v283_physio_route_lineage_gap_audit_20260702.py`
- 产物：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v283_physio_route_lineage_gap_audit_20260702`
- 报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v283_physio_route_lineage_gap_audit_20260702\reports\v283_physio_route_lineage_gap_audit_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v283_physio_route_lineage_gap_audit_20260702_pack.zip`
- 核心表：
  - `tables\v283_alignment_quality_summary.csv`
  - `tables\v283_route_lineage_summary.csv`
  - `tables\v283_next_route_requirements.csv`
  - `tables\v283_decision_summary.csv`
  - `tables\v283_signal_quality_by_family.csv`
- 核心图：
  - `figures\v283_physio_route_lineage_status.png`
  - `figures\v283_signal_quality_by_family.png`
  - `figures\v283_badtop10_macro_f1_delta.png`
- 校验：`logs\guardrail_check.json`，`logs\input_hashes.csv`，`logs\file_inventory.csv`

## 最新指针：2026-07-02 v282 physiology ambiguity route gate

- 总结：v282 用 v272 事件级诊断表做“生理能否在车辆 top40 相似候选内稳定消歧”的路线门控审计。结果 `route_viable_now=false`：可部署 bio top1 在 test bad_top10 上比 latest 差 `+0.1989`，在 test bad_top10_vehicle_ambiguous 上差 `+0.2347`；非部署 bio top3 上限在歧义差样本上 val/test 不同向；test bad_top10 生理距离-真实误差排序相关最高仅 `0.00985`。结论是旧生理特征层不适合继续做同类候选选择微调。
- 入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v282_physio_ambiguity_route_gate_20260702.py`
- 产物：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v282_physio_ambiguity_route_gate_20260702`
- 报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v282_physio_ambiguity_route_gate_20260702\reports\v282_physio_ambiguity_route_gate_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v282_physio_ambiguity_route_gate_20260702_pack.zip`
- 核心表：
  - `tables\v282_route_group_summary.csv`
  - `tables\v282_val_chosen_generalization.csv`
  - `tables\v282_split_consistency.csv`
  - `tables\v282_test_bad_ambiguous_event_audit.csv`
  - `tables\v282_route_gate_decision.csv`
- 核心图：
  - `figures\v282_badtop10_val_test_bio_delta.png`
  - `figures\v282_bad_ambiguous_bio_rank_corr.png`
  - `figures\v282_val_chosen_test_delta.png`
- 校验：`logs\guardrail_check.json`，`logs\input_hashes.csv`，`logs\file_inventory.csv`

## 最新指针：2026-07-02 v281 bio-top3 constrained selector

- 总结：v281 将 v272 的 `vehicle top40 -> bio top3` 少量 oracle 上限转成可训练 selector。结果显示 bio top3 oracle 在 test bad_top10 上可到 `0.6738`，但 validation 上该上限本身比 latest 差，因此 val-selected deployable 仍回到 fixed wait-latest `0.6950`；test-best diagnostic 为 `0.6842`，不能作为部署结论。
- 入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v281_bio_top3_constrained_selector_20260702.py`
- 产物：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v281_bio_top3_constrained_selector_20260702`
- 报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v281_bio_top3_constrained_selector_20260702\reports\v281_bio_top3_constrained_selector_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v281_bio_top3_constrained_selector_20260702_pack.zip`
- 核心表：
  - `tables\v281_bio_top3_candidates.csv`
  - `tables\v281_feature_set_audit.csv`
  - `tables\v281_predictions.csv`
  - `tables\v281_bio_top3_oracle_summary.csv`
  - `tables\v281_threshold_search.csv`
  - `tables\v281_chosen_configs.csv`
  - `tables\v281_decision_summary.csv`
- 核心图：`figures\v281_test_badtop10_bio_top3_selector.png`
- 校验：`logs\guardrail_check.json`，`logs\input_hashes.csv`，`logs\file_inventory.csv`

## 最新指针：2026-07-02 v280 cross-fit physiology reliability filter

- 总结：v280 用 recording-group OOF train top 修正 v279 可靠性模型训练候选偏乐观的问题。结果 test-best diagnostic 为 `0.6891`，deployable 仍为 `0.6950`；bio reliability 仍未超过 vehicle reliability。
- 入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v280_crossfit_physio_reliability_filter_20260702.py`
- 产物：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v280_crossfit_physio_reliability_filter_20260702`
- 报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v280_crossfit_physio_reliability_filter_20260702\reports\v280_crossfit_physio_reliability_filter_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v280_crossfit_physio_reliability_filter_20260702_pack.zip`

## 最新指针：2026-07-02 v279 physiology reliability filter

- 总结：v279 不让生理直接选轨迹，而是判断 v278 vehicle listwise 第一候选是否可信。test-best diagnostic 达到 `0.6791`，但 val 不支持部署；bio reliability 没有赢过 vehicle reliability。
- 入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v279_physio_reliability_filter_for_listrank_20260702.py`
- 产物：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v279_physio_reliability_filter_for_listrank_20260702`
- 报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v279_physio_reliability_filter_for_listrank_20260702\reports\v279_physio_reliability_filter_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v279_physio_reliability_filter_for_listrank_20260702_pack.zip`

## 最新指针：2026-07-02 v278 listwise candidate rank loss

- 总结：v278 把候选选择从绝对收益回归改成同事件组内排序标签，比较 vehicle-only、vehicle+bio、vehicle+style+bio。vehicle-only listwise test-best diagnostic 可把 test bad_top10 从 `0.6950` 降到 `0.6832`，覆盖率 `10.53%`，但 val 上不可部署；最佳 bio 特征组 diagnostic 只有 `0.6950`，未超过 vehicle-only。结论：候选选择损失有潜力，但当前生理/风格没有提供候选排序增量。

### v278 listwise candidate rank loss

- 入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v278_listwise_candidate_rank_loss_20260702.py`
- 产物：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v278_listwise_candidate_rank_loss_20260702`
- 报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v278_listwise_candidate_rank_loss_20260702\reports\v278_listwise_candidate_rank_loss_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v278_listwise_candidate_rank_loss_20260702_pack.zip`
- 核心表：
  - `tables\v278_event_feature_audit_from_v277.csv`
  - `tables\v278_listrank_feature_set_audit.csv`
  - `tables\v278_candidate_listrank_predictions_compact.csv`
  - `tables\v278_selected_by_strategy.csv`
  - `tables\v278_threshold_search.csv`
  - `tables\v278_chosen_configs.csv`
  - `tables\v278_decision_summary.csv`
- 核心图：
  - `figures\v278_test_badtop10_listrank_candidate_loss.png`
- 校验：
  - `logs\guardrail_check.json`
  - `logs\input_file_hashes.csv`
  - `logs\file_inventory.csv`
- 摘要：
  - event_n `1167`。
  - candidate_rows `46680`。
  - feature_set_n `3`。
  - search_rows `96`。
  - fixed wait-latest `0.6950`，oracle `0.6125`。
  - vehicle-only test-best diagnostic `0.6832`，覆盖率 `0.1053`，不可部署。
  - best deployable `0.6950`，覆盖率 `0`。
  - best bio feature diagnostic `0.6950`，未超过 vehicle-only。

### v277 style + calibrated physiology candidate gain model

- 入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v277_style_bio_candidate_gain_model_20260702.py`
- 产物：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v277_style_bio_candidate_gain_model_20260702`
- 报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v277_style_bio_candidate_gain_model_20260702\reports\v277_style_bio_candidate_gain_model_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v277_style_bio_candidate_gain_model_20260702_pack.zip`
- 核心表：
  - `tables\v277_event_feature_audit.csv`
  - `tables\v277_model_feature_set_audit.csv`
  - `tables\v277_candidate_gain_predictions_compact.csv`
  - `tables\v277_selected_by_strategy.csv`
  - `tables\v277_threshold_search.csv`
  - `tables\v277_chosen_configs.csv`
  - `tables\v277_decision_summary.csv`
- 核心图：
  - `figures\v277_test_badtop10_style_bio_candidate_gain.png`
- 校验：
  - `logs\guardrail_check.json`
  - `logs\input_file_hashes.csv`
  - `logs\file_inventory.csv`
- 摘要：
  - event_n `1167`。
  - candidate_rows `46680`。
  - style query feature cap `96`。
  - bio271 query feature cap `96`。
  - feature_set_n `6`。
  - search_rows `195`。
  - fixed wait-latest `0.6950`，oracle `0.6125`。
  - val-best deployable `0.6950`，test 覆盖率 `0`，未超过 fixed wait-latest。
  - test-best diagnostic `0.7008`，比 fixed wait-latest 更差。

### v276 bio-assisted candidate gain model

- 入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v276_bio_assisted_candidate_gain_model_20260702.py`
- 产物：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v276_bio_assisted_candidate_gain_model_20260702`
- 报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v276_bio_assisted_candidate_gain_model_20260702\reports\v276_bio_assisted_candidate_gain_model_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v276_bio_assisted_candidate_gain_model_20260702_pack.zip`
- 核心表：
  - `tables\v276_feature_set_audit.csv`
  - `tables\v276_candidate_gain_predictions.csv`
  - `tables\v276_top_candidate_by_event.csv`
  - `tables\v276_threshold_search.csv`
  - `tables\v276_chosen_configs.csv`
  - `tables\v276_selected_by_strategy.csv`
  - `tables\v276_decision_summary.csv`
- 核心图：
  - `figures\v276_test_badtop10_candidate_gain_model.png`
- 校验：
  - `logs\guardrail_check.json`
  - `logs\input_file_hashes.csv`
  - `logs\file_inventory.csv`
- 摘要：
  - event_n `1167`。
  - candidate_rows `46680`。
  - feature_set_n `3`。
  - search_rows `96`。
  - fixed wait-latest `0.6950`，oracle `0.6125`。
  - test-best gain diagnostic `0.6858`，覆盖率 `0.0526`，但不可部署。
  - val-best deployable `0.6950`，覆盖率 `0`，未超过 fixed wait-latest。

### v275 stable bio consensus override

- 入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v275_stable_bio_consensus_override_20260702.py`
- 产物：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v275_stable_bio_consensus_override_20260702`
- 报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v275_stable_bio_consensus_override_20260702\reports\v275_stable_bio_consensus_override_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v275_stable_bio_consensus_override_20260702_pack.zip`
- 核心表：
  - `tables\v275_decision_summary.csv`
  - `tables\v275_chosen_consensus_configs.csv`
  - `tables\v275_consensus_search.csv`
  - `tables\v275_consensus_grid.csv`
  - `tables\v275_selected_by_strategy.csv`
  - `tables\v275_consensus_summary.csv`
- 核心图：
  - `figures\v275_test_badtop10_stable_consensus.png`
- 校验：
  - `logs\guardrail_check.json`
  - `logs\input_file_hashes.csv`
  - `logs\file_inventory.csv`
- 摘要：
  - event_n `1167`。
  - candidate_rows `35010`。
  - grid_rows / search_rows `750`。
  - fixed wait-latest `0.6950`，oracle `0.6125`。
  - bio-prefilter candidate oracle `0.6466`。
  - test-best consensus diagnostic `0.6881`，覆盖率 `0.1053`，但不可部署。
  - val-best deployable `0.6950`，覆盖率 `0`，未超过 fixed wait-latest。

### v274 no-harm bio override

- 入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v274_noharm_bio_override_20260702.py`
- 产物：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v274_noharm_bio_override_20260702`
- 报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v274_noharm_bio_override_20260702\reports\v274_noharm_bio_override_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v274_noharm_bio_override_20260702_pack.zip`
- 核心表：
  - `tables\v274_decision_summary.csv`
  - `tables\v274_chosen_thresholds.csv`
  - `tables\v274_override_summary.csv`
  - `tables\v274_threshold_search.csv`
  - `tables\v274_event_candidate_predictions.csv`
  - `tables\v274_selected_by_strategy.csv`
- 核心图：
  - `figures\v274_test_badtop10_noharm_override.png`
- 校验：
  - `logs\guardrail_check.json`
  - `logs\input_file_hashes.csv`
  - `logs\file_inventory.csv`
- 摘要：
  - event_n `1167`。
  - candidate_event_model_rows `35010`。
  - threshold_search_rows `3780`。
  - fixed wait-latest `0.6950`，oracle `0.6125`。
  - bio-prefilter candidate oracle `0.6466`。
  - test-best override diagnostic `0.6902`，覆盖率 `0.0870`。
  - val-best deployable `0.6950`，未超过 fixed wait-latest。

### v273 bio-prefiltered pair reranker

- 入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v273_bio_prefiltered_pair_reranker_20260702.py`
- 产物：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v273_bio_prefiltered_pair_reranker_20260702`
- 报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v273_bio_prefiltered_pair_reranker_20260702\reports\v273_bio_prefiltered_pair_reranker_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v273_bio_prefiltered_pair_reranker_20260702_pack.zip`
- 核心表：
  - `tables\v273_bio_prefilter_neighbors.csv`
  - `tables\v273_pair_predictions_compact.csv`
  - `tables\v273_selected_by_strategy.csv`
  - `tables\v273_pair_reranker_summary.csv`
  - `tables\v273_val_chosen_summary.csv`
  - `tables\v273_feature_block_audit.csv`
  - `tables\v273_decision_summary.csv`
- 核心图：
  - `figures\v273_test_badtop10_bio_prefiltered_pair.png`
- 校验：
  - `logs\guardrail_check.json`
  - `logs\input_file_hashes.csv`
  - `logs\file_inventory.csv`
- 摘要：
  - vehicle_pool_k `40`，bio_prefilter_k `5`。
  - pair_row_n `35010`。
  - fixed wait-latest `0.6950`，oracle `0.6125`。
  - bio-prefilter candidate oracle `0.6466`。
  - test-best deployable diagnostic `0.7964`。
  - val-best vehicle+bio `0.8664`。

### v272 physiology ambiguity disambiguation

- 入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v272_physio_ambiguity_disambiguation_20260702.py`
- 产物：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v272_physio_ambiguity_disambiguation_20260702`
- 报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v272_physio_ambiguity_disambiguation_20260702\reports\v272_physio_ambiguity_disambiguation_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v272_physio_ambiguity_disambiguation_20260702_pack.zip`
- 核心表：
  - `tables\v272_neighbor_rank_diagnostics_by_event.csv`
  - `tables\v272_ambiguity_reduction_summary.csv`
  - `tables\v272_decision_summary.csv`
  - `tables\v272_feature_set_input_audit.csv`
- 核心图：
  - `figures\v272_test_badtop10_ambiguity_decision.png`
  - `figures\v272_test_badtop10_bio_rank_capture.png`
- 校验：
  - `logs\guardrail_check.json`
  - `logs\input_file_hashes.csv`
  - `logs\file_inventory.csv`
- 摘要：
  - diagnostic_row_n `28008`。
  - test bad_top10 vehicle nearest `0.8785`。
  - vehicle candidate oracle k40 `0.6166`。
  - val 选 bio top1 `0.8940`。
  - test-best bio top1 diagnostic `0.8744`。
  - test-best bio top3 oracle `0.6738`。

### v271 calibrated raw physiology state

- 入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v271_calibrated_raw_physio_state_20260702.py`
- 产物：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v271_calibrated_raw_physio_state_20260702`
- 报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v271_calibrated_raw_physio_state_20260702\reports\v271_calibrated_raw_physio_state_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v271_calibrated_raw_physio_state_20260702_pack.zip`
- 核心表：
  - `tables\v271_raw_delay0_alignment_audit.csv`
  - `tables\v271_summary_calibration_audit.csv`
  - `tables\v271_sequence_centering_audit.csv`
  - `tables\v271_centered_raw_pca_audit.csv`
  - `tables\v271_raw_feature_screening_train_only.csv`
  - `tables\v271_raw_feature_set_audit.csv`
  - `tables\v271_event_context_table.csv`
  - `tables\v271_wait_selected_by_strategy.csv`
  - `tables\v271_wait_summary.csv`
  - `tables\v271_pair_selected_by_strategy.csv`
  - `tables\v271_pair_reranker_summary.csv`
  - `tables\v271_pair_val_chosen_summary.csv`
  - `tables\v271_pair_predictions_compact.csv`
  - `tables\v271_decision_summary.csv`
- 核心图：
  - `figures\v271_test_badtop10_decision_summary.png`
- 校验：
  - `logs\guardrail_check.json`
  - `logs\input_file_hashes.csv`
  - `logs\file_inventory.csv`
- 摘要：
  - calibration setting `subject_recording_unsupervised_transductive`。
  - event_n `1167`。
  - raw sequence shape delay0 `[1167, 6, 400]`。
  - raw physio ok rate `0.9195`。
  - raw feature_n `505`，raw_set_n `6`。
  - pair_row_n `280080`。
  - fixed wait-latest `0.6950`，oracle `0.6125`。
  - pair candidate oracle k40 `0.6166`。
  - wait test-best `0.6950`，等价于全 wait-latest。
  - pair test-best deployable `0.7853`。
  - val-best vehicle+raw `0.9232`。

### v270 raw physiology state latent

- 入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v270_raw_physio_state_latent_20260702.py`
- 产物：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v270_raw_physio_state_latent_20260702`
- 报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v270_raw_physio_state_latent_20260702\reports\v270_raw_physio_state_latent_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v270_raw_physio_state_latent_20260702_pack.zip`
- 核心表：
  - `tables\v270_raw_delay0_alignment_audit.csv`
  - `tables\v270_raw_pca_audit.csv`
  - `tables\v270_raw_feature_screening_train_only.csv`
  - `tables\v270_raw_feature_set_audit.csv`
  - `tables\v270_event_context_table.csv`
  - `tables\v270_wait_selected_by_strategy.csv`
  - `tables\v270_wait_summary.csv`
  - `tables\v270_pair_selected_by_strategy.csv`
  - `tables\v270_pair_reranker_summary.csv`
  - `tables\v270_pair_val_chosen_summary.csv`
  - `tables\v270_pair_predictions_compact.csv`
  - `tables\v270_decision_summary.csv`
- 核心图：
  - `figures\v270_test_badtop10_decision_summary.png`
- 校验：
  - `logs\guardrail_check.json`
  - `logs\input_file_hashes.csv`
  - `logs\file_inventory.csv`
- 摘要：
  - event_n `1167`。
  - raw sequence shape delay0 `[1167, 6, 400]`。
  - raw physio ok rate `0.9195`。
  - raw feature_n `277`，raw_set_n `4`。
  - fixed wait-latest `0.6950`，oracle `0.6125`。
  - pair candidate oracle k40 `0.6166`。
  - wait test-best `0.6950`，等价于全 wait-latest。
  - pair test-best deployable `0.7866`。
  - val-best vehicle+raw `0.8142`。

### v269 reliable / identity-removed physiology

- 入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v269_reliable_identity_removed_physio_20260702.py`
- 产物：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v269_reliable_identity_removed_physio_20260702`
- 报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v269_reliable_identity_removed_physio_20260702\reports\v269_reliable_identity_removed_physio_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v269_reliable_identity_removed_physio_20260702_pack.zip`
- 核心表：
  - `tables\v269_feature_screening_train_only.csv`
  - `tables\v269_feature_set_audit.csv`
  - `tables\v269_event_context_table.csv`
  - `tables\v269_wait_gate_selected_by_strategy.csv`
  - `tables\v269_wait_gate_summary.csv`
  - `tables\v269_wait_gate_audit.csv`
  - `tables\v269_pair_predictions_compact.csv`
  - `tables\v269_pair_selected_by_strategy.csv`
  - `tables\v269_pair_reranker_summary.csv`
  - `tables\v269_pair_val_chosen_summary.csv`
  - `tables\v269_pair_feature_block_audit.csv`
  - `tables\v269_decision_summary.csv`
- 核心图：
  - `figures\v269_wait_gate_test_badtop10.png`
  - `figures\v269_pair_reranker_test_badtop10.png`
- 校验：
  - `logs\guardrail_check.json`
  - `logs\input_file_hashes.csv`
  - `logs\file_inventory.csv`
- 摘要：
  - candidate_bio_feature_n `352`，feature_set_n `4`。
  - pair_row_n `186720`。
  - test bad_top10 fixed wait-latest `0.6950`，oracle `0.6125`，candidate oracle k40 `0.6166`。
  - wait gate best `0.6950`，但实际接近全 wait-latest。
  - pair test-best deployable `0.7781`。
  - val-best vehicle+bio `0.8365`，相对 v267 `0.8495` 只小幅改善。

---

## 最新指针：2026-07-02 v268 physiology quality / alignment / identifiability audit

- 总结：v268 不再训练新模型，而是审计现有生理链路。结果显示 200Hz 连续源层时序稳定、事件窗口覆盖可用，但当前派生生理表征存在不可用/近常数列，并且 subject/recording 身份信号远强于行为信号；这解释了为什么 v266/v267 虽有候选 oracle headroom，但 bio reranker 不能稳定选中最佳候选。

### v268 physiology quality / alignment / identifiability audit

- 入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v268_physio_quality_identifiability_audit_20260702.py`
- 产物：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v268_physio_quality_identifiability_audit_20260702`
- 报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v268_physio_quality_identifiability_audit_20260702\reports\v268_physio_quality_identifiability_audit_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v268_physio_quality_identifiability_audit_20260702_pack.zip`
- 核心表：
  - `tables\v268_source_recording_quality_summary.csv`
  - `tables\v268_source_recording_quality_by_subject.csv`
  - `tables\v268_source_signal_availability_quality.csv`
  - `tables\v268_source_signal_quality_by_family.csv`
  - `tables\v268_event_coverage_by_split_delay.csv`
  - `tables\v268_event_coverage_by_recording.csv`
  - `tables\v268_event_feature_missingness_by_family.csv`
  - `tables\v268_bio_identity_behavior_eta_detail.csv`
  - `tables\v268_bio_identity_behavior_eta_summary.csv`
  - `tables\v268_v260_eta_reference.csv`
  - `tables\v268_candidate_rank_diagnostics_by_event.csv`
  - `tables\v268_candidate_rank_diagnostics_summary.csv`
  - `tables\v268_conclusion_flags.csv`
- 核心图：
  - `figures\v268_signal_availability.png`
  - `figures\v268_identity_vs_behavior_eta.png`
  - `figures\v268_test_badtop10_candidate_rank_quality.png`
- 校验：
  - `logs\guardrail_check.json`
  - `logs\input_file_hashes.csv`
  - `logs\file_inventory.csv`
- 摘要：
  - source recording `82` 个，subject `18` 个，median_hz `200.000`，gap/duplicate 均为 `0`。
  - `HRV_RMSSD`、`RESP_BPM`、`RESP_Amplitude` usable 均为 `0/82`；EDA usable `73/82`。
  - min split-delay ok_rate `0.889`，post-observation rate `0`。
  - median family identity/behavior eta ratio `68.74`。
  - test bad_top10 上 `pred_pair_vehicle_bio_hgb` chosen_minus_latest `+0.1509`，true best top3 rate `0.211`。
  - 结论：当前 blocker 更像派生生理表征质量与身份混淆问题，不是源采样断裂或模型深度不足。

---

## 最新指针：2026-07-02 v267 supervised bio prototype reranker

- 总结：v267 在 v266 的 vehicle-matched prototype headroom 基础上，进一步训练监督式 query-prototype pair reranker。结果显示，bio 有弱增量但仍不能低于 fixed wait-latest，不能完成 goal。

### v267 supervised bio prototype reranker

- 入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v267_supervised_bio_prototype_reranker_20260702.py`
- 产物：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v267_supervised_bio_prototype_reranker_20260702`
- 报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v267_supervised_bio_prototype_reranker_20260702\reports\v267_supervised_bio_prototype_reranker_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v267_supervised_bio_prototype_reranker_20260702_pack.zip`
- 核心表：
  - `tables\v267_pair_predictions_compact.csv`
  - `tables\v267_selected_pair_reranker_by_strategy.csv`
  - `tables\v267_pair_reranker_summary.csv`
  - `tables\v267_val_chosen_pair_strategy_summary.csv`
  - `tables\v267_feature_block_audit.csv`
  - `tables\v267_feature_fill_audit.csv`
  - `tables\v267_pair_construction_audit.csv`
- 核心图：
  - `figures\v267_test_badtop10_pair_reranker.png`
  - `figures\v267_val_test_badtop10_generalization.png`
- 校验：
  - `logs\guardrail_check.json`
  - `logs\input_file_hashes.csv`
  - `logs\file_inventory.csv`
- 摘要：
  - pair_n `46680`，train_pair_n `26960`，val_pair_n `12360`，test_pair_n `7360`。
  - candidate oracle k40 `0.6166`，fixed wait-latest `0.6950`。
  - val-best pair vehicle `0.8746`。
  - val-best pair vehicle+bio `0.8495`。
  - bio 监督式 reranker 有弱增量，但仍远未达到本质改善门槛。

---

## 最新指针：2026-07-02 GPTPro phase02 + v266 vehicle-matched bio residual prototype

- 总结：GPTPro phase02 已通过 ChatGPT 桌面软件 Pro / Pro 扩展完成。其建议的三条路线中，v265 已覆盖 wait-benefit，v264 已覆盖 subject-aware online，v266 覆盖 vehicle-matched residual prototype reranking。v266 显示候选库有 headroom，但当前 bio260 不能把 headroom 稳定转化为可部署收益。

### GPTPro phase02 外部复核

- 提问词：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\gptpro_reviews\20260702_phase02_prompt.md`
- 回复：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\gptpro_reviews\20260702_phase02_response.md`
- 原始可访问性树：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\gptpro_reviews\20260702_phase02_response_raw_accessibility.txt`
- 状态：已发送并归档，来源为 ChatGPT 桌面软件 Pro / Pro 扩展。

### v266 vehicle-matched bio residual prototype

- 入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v266_vehicle_matched_bio_residual_prototype_20260702.py`
- 产物：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v266_vehicle_matched_bio_residual_prototype_20260702`
- 报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v266_vehicle_matched_bio_residual_prototype_20260702\reports\v266_vehicle_matched_bio_residual_prototype_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v266_vehicle_matched_bio_residual_prototype_20260702_pack.zip`
- 核心表：
  - `tables\v266_event_context_table.csv`
  - `tables\v266_vehicle_matched_neighbors.csv`
  - `tables\v266_selected_prototype_by_strategy.csv`
  - `tables\v266_prototype_strategy_summary.csv`
  - `tables\v266_val_chosen_strategy_summary.csv`
  - `tables\v266_feature_block_audit.csv`
  - `tables\v266_feature_fill_audit.csv`
- 核心图：
  - `figures\v266_test_badtop10_main_comparison.png`
  - `figures\v266_candidate_oracle_headroom_by_k.png`
- 校验：
  - `logs\guardrail_check.json`
  - `logs\input_file_hashes.csv`
  - `logs\file_inventory.csv`
- 摘要：
  - test bad_top10 candidate oracle k40 `0.6166`，接近 full oracle `0.6125`，低于 fixed wait-latest `0.6950`。
  - val-best vehicle-only prototype `0.8890`。
  - val-best vehicle+bio prototype `0.8374`。
  - bio 有小幅重排改善，但没有低于 fixed wait-latest，不能算 goal 达成。

---

## 最新指针：2026-07-02 v265 physiology uncertainty / wait frontier

- 总结：v265 验证生理是否能作为不确定性/风险校准信号，在固定等待预算下更准确挑出需要 wait-latest 的样本。结果显示，生理 badprob 有弱 AUC，但不能稳定转化为同等等待预算下的 RMSE 优势。

### v265 physiology uncertainty / wait frontier

- 入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v265_physio_uncertainty_wait_frontier_20260702.py`
- 产物：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v265_physio_uncertainty_wait_frontier_20260702`
- 报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v265_physio_uncertainty_wait_frontier_20260702\reports\v265_physio_uncertainty_wait_frontier_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v265_physio_uncertainty_wait_frontier_20260702_pack.zip`
- 核心表：
  - `tables\v265_event_risk_scores.csv`
  - `tables\v265_selected_wait_frontier_by_policy.csv`
  - `tables\v265_wait_frontier_summary.csv`
  - `tables\v265_val_thresholds_by_wait_rate.csv`
  - `tables\v265_score_diagnostics.csv`
  - `tables\v265_feature_block_audit.csv`
- 核心图：
  - `figures\v265_test_badtop10_wait_frontier.png`
- 校验：
  - `logs\guardrail_check.json`
  - `logs\input_file_hashes.csv`
  - `logs\file_inventory.csv`
- 摘要：
  - test bad_top10 的最佳 RMSE 全部退化为 fixed wait-latest `0.6950`。
  - bio 风险分数有弱 bad_top10 AUC：vehicle+bio_badprob `0.6175`，bio_only_badprob `0.6376`。
  - 但该信号不能稳定产生同等等待预算下的策略收益。

---

## 最新指针：2026-07-02 v264 online subject-aware physiology calibration

- 总结：v264 是生理主线的边界实验，不是正式 subject-disjoint 替代结果。它允许同一驾驶员更早事件的已知结果做 online calibration，用来验证“生理是否只在 subject-aware 设定下有用”。结果显示，同驾驶员历史反馈有价值，但 physiology KNN 没有额外收益。

### v264 online subject-aware physiology calibration

- 入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v264_online_subject_physio_calibration_20260702.py`
- 产物：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v264_online_subject_physio_calibration_20260702`
- 报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v264_online_subject_physio_calibration_20260702\reports\v264_online_subject_physio_calibration_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v264_online_subject_physio_calibration_20260702_pack.zip`
- 核心表：
  - `tables\v264_event_online_predictions.csv`
  - `tables\v264_selected_wait_gate_by_strategy.csv`
  - `tables\v264_online_strategy_summary.csv`
  - `tables\v264_online_history_audit.csv`
  - `tables\v264_feature_block_audit.csv`
  - `tables\v264_feature_fill_audit.csv`
- 核心图：
  - `figures\v264_online_subject_physio_badtop10.png`
- 校验：
  - `logs\guardrail_check.json`
  - `logs\input_file_hashes.csv`
  - `logs\file_inventory.csv`
- 摘要：
  - test bad_top10 global vehicle gate `0.7528`。
  - online_subject_mean_vehicle `0.7112`，说明同驾驶员历史反馈有帮助。
  - online_physio_knn_vehicle `0.7112`，没有超过纯 subject mean。
  - online_physio_knn_vehicle_bio `0.7698`，比 subject mean 和 fixed wait-latest 都差。
  - 结论：当前 bio260 没有提供额外 online 个体内状态区分能力。

### GPTPro phase02 复核提示

- 提问词：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\gptpro_reviews\20260702_phase02_prompt.md`
- 状态：已通过 ChatGPT 桌面软件 Pro / Pro 扩展发送并归档。
- 回复：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\gptpro_reviews\20260702_phase02_response.md`
- 说明：此前 `gptpro-browser-bridge` 无法确认 Chrome Pro/进阶模式而拒绝发送；之后用户确认 GPTPro 指 ChatGPT 软件，已改走桌面软件并完成。

---

## 最新指针：2026-07-02 v260-v263 生理重构与决策复核

- 总结：v260-v263 继续围绕“充分利用生理数据弥补锚点前信息不足”推进。v260 重构 200Hz 事件级 bio260 biomarker，v261 用全量 bio260 做 anchor selector，v262 用 subject-invariant bio260 特征复核，v263 将任务简化为 0ms wait gate。结论是：bio260 有弱诊断信号，但尚未形成差样本本质改善；当前 goal 仍未达成。

### v260 event biomarker physiology rebuild

- 入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v260_event_biomarker_physio_rebuild_20260702.py`
- 产物：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v260_event_biomarker_physio_rebuild_20260702`
- 报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v260_event_biomarker_physio_rebuild_20260702\reports\v260_event_biomarker_physio_rebuild_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v260_event_biomarker_physio_rebuild_20260702_pack.zip`
- 核心表：
  - `tables\v260_event_biomarker_features.csv`
  - `tables\v260_behavior_classification_diagnostics.csv`
  - `tables\v260_future_summary_regression_diagnostics.csv`
  - `tables\v260_biomarker_eta2_by_target_feature.csv`
  - `tables\v260_alignment_coverage_summary.csv`
  - `tables\v260_feature_block_audit.csv`
- 核心图：
  - `figures\v260_subject_disjoint_test_macro_f1.png`
  - `figures\v260_eta2_top_features.png`
- 校验：
  - `logs\guardrail_check.json`
  - `logs\input_file_hashes.csv`
  - `logs\file_inventory.csv`
- 摘要：bio260 相比旧 physio200 在 bad_top10 诊断上略有改善，但 vehicle+bio260 仍未形成正式未来行为预测增量。

### v261 bio260 anchor selector

- 入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v261_bio260_anchor_selector_20260702.py`
- 产物：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v261_bio260_anchor_selector_20260702`
- 报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v261_bio260_anchor_selector_20260702\reports\v261_bio260_anchor_selector_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v261_bio260_anchor_selector_20260702_pack.zip`
- 核心表：
  - `tables\v261_candidate_predictions_compact.csv`
  - `tables\v261_selected_anchor_by_strategy.csv`
  - `tables\v261_anchor_selector_summary.csv`
  - `tables\v261_bio260_merge_audit.csv`
  - `tables\v261_feature_block_audit.csv`
  - `tables\v261_feature_fill_audit.csv`
  - `tables\v261_v258_badtop10_reference.csv`
- 核心图：
  - `figures\v261_anchor_selector_test_badtop10.png`
- 摘要：test bad_top10 中 vehicle selector `0.9425`，vehicle+bio260 `0.9765`，badweighted `0.9837`，说明全量 bio260 没有帮助锚点选择。

### v262 subject-invariant bio260 selector

- 入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v262_subject_invariant_bio260_selector_20260702.py`
- 产物：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v262_subject_invariant_bio260_selector_20260702`
- 报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v262_subject_invariant_bio260_selector_20260702\reports\v262_subject_invariant_bio260_selector_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v262_subject_invariant_bio260_selector_20260702_pack.zip`
- 核心表：
  - `tables\v262_candidate_predictions_compact.csv`
  - `tables\v262_selected_anchor_by_strategy.csv`
  - `tables\v262_anchor_selector_summary.csv`
  - `tables\v262_bio260_merge_audit.csv`
  - `tables\v262_feature_block_audit.csv`
  - `tables\v262_feature_fill_audit.csv`
  - `tables\v262_feature_selection_audit.csv`
  - `tables\v262_v261_badtop10_reference.csv`
- 核心图：
  - `figures\v262_subject_invariant_bio260_test_badtop10.png`
- 摘要：subject-invariant sp64 让 bad_top10 selector tail 从 `0.9419` 降到 `0.9059`，但幅度很小，仍远弱于 fixed wait-latest `0.6950`。

### v263 bio260 wait gate

- 入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v263_bio260_wait_gate_20260702.py`
- 产物：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v263_bio260_wait_gate_20260702`
- 报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v263_bio260_wait_gate_20260702\reports\v263_bio260_wait_gate_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v263_bio260_wait_gate_20260702_pack.zip`
- 核心表：
  - `tables\v263_event_wait_gate_predictions.csv`
  - `tables\v263_selected_wait_gate_by_strategy.csv`
  - `tables\v263_wait_gate_summary.csv`
  - `tables\v263_threshold_tuning_audit.csv`
  - `tables\v263_feature_selection_audit.csv`
  - `tables\v263_feature_block_audit.csv`
  - `tables\v263_feature_fill_audit.csv`
  - `tables\v263_bio260_merge_audit.csv`
- 核心图：
  - `figures\v263_bio260_wait_gate_test_badtop10.png`
- 摘要：0ms wait gate 中 vehicle gate `0.7528`，vehicle+bio260_sp64 gate `0.8748`；val 阈值最优几乎等于全等 latest，说明收益来自多观察，不是生理判断。

---

## 最新指针：2026-07-01 v254a physio deep signal audit

- 入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v254a_physio_deep_signal_audit_20260701.py`
- 产物：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v254a_physio_deep_signal_audit_20260701`
- 报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v254a_physio_deep_signal_audit_20260701\reports\v254a_physio_deep_signal_audit_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v254a_physio_deep_signal_audit_20260701_pack.zip`
- 摘要：v254a 固定 v252/v253 样本和 split，从 10Hz 生理表提取锚点前多窗口深层统计，检查生理是否含有行为相关状态结构。结果：10Hz 覆盖率 `0.919` 且无未来泄漏；生理含强 subject/recording 结构，但未来行为标签 eta² 很低，test 上 physio10hz 和 vehicle+physio10hz 均未超过 vehicle_only。下一步应重做生理表征和个体内归一化，而不是继续简单拼接。

---

## 2026-07-02 v254b 200Hz 连续生理事件相关表征（最新）

- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v254b_physio_200hz_event_representation_20260702.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v254b_physio_200hz_event_representation_20260702`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v254b_physio_200hz_event_representation_20260702\reports\v254b_physio_200hz_event_representation_cn.md`
- 压缩包：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v254b_physio_200hz_event_representation_20260702_pack.zip`
- 核心表：
  - `tables\v254b_event_physio200_features.csv`
  - `tables\v254b_future_behavior_targets.csv`
  - `tables\v254b_behavior_classification_diagnostics.csv`
  - `tables\v254b_future_summary_regression_diagnostics.csv`
  - `tables\v254b_physio200_eta2_by_target_feature.csv`
  - `tables\v254b_alignment_coverage_summary.csv`
  - `tables\v254b_feature_block_audit.csv`
  - `tables\v254b_split_protocol_table.csv`
- 核心图：
  - `figures\v254b_macro_f1_subject_disjoint_vs_subject_aware.png`
  - `figures\v254b_top_eta2_physio200.png`
- 日志与校验：
  - `logs\guardrail_check.json`
  - `logs\input_file_hashes.csv`
  - `logs\file_inventory.csv`
- 结果摘要：
  - 200Hz 生理覆盖率：test `0.8967`，train `0.8887`，val `1.0`。
  - subject-disjoint high_future_abs_q75：vehicle_only macro-F1 `0.7408`，vehicle+physio200_curated `0.6169`。
  - subject-disjoint bad_top10_v250_diagnostic：vehicle_only `0.4958`，vehicle+physio200_curated `0.5170`。
  - subject-aware bad_top10_v250_diagnostic：vehicle_only `0.4578`，vehicle+physio200_norm `0.6095`。
- 用途：证明 200Hz 手工事件表征仍没有正式跨驾驶员增量，但存在弱个体化诊断信号。

---

## 2026-07-02 v255 生理状态条件化候选轨迹选择（最新）

- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v255_physio_conditioned_candidate_ranker_20260702.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v255_physio_conditioned_candidate_ranker_20260702`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v255_physio_conditioned_candidate_ranker_20260702\reports\v255_physio_conditioned_candidate_ranker_cn.md`
- 压缩包：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v255_physio_conditioned_candidate_ranker_20260702_pack.zip`
- 核心表：
  - `tables\v255_selected_candidate_per_query.csv`
  - `tables\v255_candidate_selection_summary.csv`
  - `tables\v255_threshold_tuning_summary.csv`
  - `tables\v255_ranker_feature_audit.csv`
  - `tables\v255_test_pair_predictions_compact.csv`
  - `tables\v255_physio_feature_standardization_audit.csv`
  - `tables\v255_pair_feature_fill_values.csv`
- 核心图：
  - `figures\v255_badtop10_candidate_selection_rmse.png`
  - `figures\v255_test_delta_vs_vehicle_rank1.png`
- 日志与校验：
  - `logs\guardrail_check.json`
  - `logs\input_file_hashes.csv`
  - `logs\file_inventory.csv`
- 结果摘要：
  - subject-disjoint bad_top10：vehicle_rank1 `0.9934`，learned_physio_state_guarded `0.9934`，oracle `0.3678`。
  - subject-aware bad_top10：vehicle_rank1 `0.9838`，learned_physio_state_guarded `0.9838`，oracle `0.4403`。
  - no-harm 阈值均退回 `1e9`，表示 learned ranker 在 val 上不敢重排。
- 用途：证明候选池 oracle 上限仍大，但当前生理状态无法可靠选择候选未来。

---

## 2026-07-02 v256 raw 200Hz 生理 CNN 融合预测（最新）

- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v256_raw_physio_cnn_fusion_20260702.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v256_raw_physio_cnn_fusion_20260702`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v256_raw_physio_cnn_fusion_20260702\reports\v256_raw_physio_cnn_fusion_cn.md`
- 压缩包：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v256_raw_physio_cnn_fusion_20260702_pack.zip`
- 核心表：
  - `tables\v256_prediction_metrics_by_bucket.csv`
  - `tables\v256_per_sample_prediction_metrics.csv`
  - `tables\v256_training_log.csv`
  - `tables\v256_physio_sequence_alignment_audit.csv`
  - `tables\v256_subject_disjoint_vehicle_standardization_audit.csv`
  - `tables\v256_subject_aware_vehicle_standardization_audit.csv`
- 核心张量：
  - `tensors\v256_predictions.npz`
  - `tensors\v256_physio_seq_20s_20hz.npz`（较大，ZIP 打包时跳过，可由脚本复现）
- 核心图：
  - `figures\v256_test_bucket_tail_rmse.png`
- 日志与校验：
  - `logs\guardrail_check.json`
  - `logs\input_file_hashes.csv`
  - `logs\file_inventory.csv`
- 结果摘要：
  - 生理序列形状 `[7002, 6, 400]`，`physio_ok_rate=0.91945`。
  - subject-disjoint bad_top10：vehicle tail RMSE `0.8411`，vehicle+physio CNN `0.9138`，delta `+0.0727`。
  - subject-aware bad_top10：vehicle `0.9272`，vehicle+physio CNN `0.9114`，delta `-0.0158`。
  - 纯生理 CNN 在正式 test 上明显弱于车辆。
- 用途：证明 raw 生理时序 CNN 也未能带来正式跨驾驶员增量；当前生理路线不满足“差样本本质改善”目标。

---

## 2026-07-02 GPTPro 方法复核提问归档（未发送）

- 提问词：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\gptpro_reviews\20260702_phase01_prompt.md`
- 状态：未发送。
- 原因：`gptpro-browser-bridge` 无法确认 Chrome 当前为 Pro/进阶模式，按规则拒绝发送。


## 2026-07-01 v254a physio deep signal audit

- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v254a_physio_deep_signal_audit_20260701.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v254a_physio_deep_signal_audit_20260701`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v254a_physio_deep_signal_audit_20260701\reports\v254a_physio_deep_signal_audit_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v254a_physio_deep_signal_audit_20260701_pack.zip`
- 核心表：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v254a_physio_deep_signal_audit_20260701\tables\v254a_event_physio10hz_deep_features.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v254a_physio_deep_signal_audit_20260701\tables\v254a_alignment_coverage_summary.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v254a_physio_deep_signal_audit_20260701\tables\v254a_physio_signal_quality_summary.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v254a_physio_deep_signal_audit_20260701\tables\v254a_physio_eta2_by_target_feature.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v254a_physio_deep_signal_audit_20260701\tables\v254a_behavior_classification_diagnostics.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v254a_physio_deep_signal_audit_20260701\tables\v254a_future_summary_regression_diagnostics.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v254a_physio_deep_signal_audit_20260701\tables\v254a_feature_block_audit.csv`
- 图：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v254a_physio_deep_signal_audit_20260701\figures\v254a_behavior_classification_macro_f1.png`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v254a_physio_deep_signal_audit_20260701\figures\v254a_future_summary_regression_r2.png`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v254a_physio_deep_signal_audit_20260701\figures\v254a_top_physio_eta2.png`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v254a_physio_deep_signal_audit_20260701\figures\v254a_physio10hz_window_rows.png`
- 日志与校验：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v254a_physio_deep_signal_audit_20260701\logs\guardrail_check.json`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v254a_physio_deep_signal_audit_20260701\logs\input_file_hashes.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v254a_physio_deep_signal_audit_20260701\logs\file_inventory.csv`
- 结论：当前 10Hz/1Hz窗口统计没有跨 subject 行为增量；后续应重做基于 200Hz 连续层的事件相关生理表征和个体内归一化。

---

## 2026-07-01 v253b physio/style state tie-break audit

- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v253b_physio_state_tiebreak_audit_20260701.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v253b_physio_state_tiebreak_audit_20260701`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v253b_physio_state_tiebreak_audit_20260701\reports\v253b_physio_state_tiebreak_audit_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v253b_physio_state_tiebreak_audit_20260701_pack.zip`
- 核心表：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v253b_physio_state_tiebreak_audit_20260701\tables\v253b_tiebreak_summary.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v253b_physio_state_tiebreak_audit_20260701\tables\v253b_tiebreak_per_strategy.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v253b_physio_state_tiebreak_audit_20260701\tables\v253b_pool_distance_future_correlation_summary.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v253b_physio_state_tiebreak_audit_20260701\tables\v253b_pool_distance_future_correlation_by_sample.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v253b_physio_state_tiebreak_audit_20260701\tables\v253b_vehicle_candidate_pool_summary.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v253b_physio_state_tiebreak_audit_20260701\tables\v253b_subject_split_table.csv`
- 图：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v253b_physio_state_tiebreak_audit_20260701\figures\v253b_badtop10_tiebreak_selected_future_rmse.png`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v253b_physio_state_tiebreak_audit_20260701\figures\v253b_tiebreak_delta_vs_vehicle_rank1.png`
- 日志与校验：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v253b_physio_state_tiebreak_audit_20260701\logs\guardrail_check.json`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v253b_physio_state_tiebreak_audit_20260701\logs\input_file_hashes.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v253b_physio_state_tiebreak_audit_20260701\logs\file_inventory.csv`
- 结论：当前状态特征手工 tie-break 不成立；但 oracle 上限很大，下一步应做可学习的状态条件多模态/不确定性模型，或先做生理质量与对齐审计。

---

## 2026-07-01 v253a state-signal disambiguation audit

- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v253_state_signal_disambiguation_audit_20260701.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v253_state_signal_disambiguation_audit_20260701`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v253_state_signal_disambiguation_audit_20260701\reports\v253_state_signal_disambiguation_audit_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v253_state_signal_disambiguation_audit_20260701_pack.zip`
- 核心表：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v253_state_signal_disambiguation_audit_20260701\tables\v253a_old_style_match_audit.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v253_state_signal_disambiguation_audit_20260701\tables\v253a_current_style_features_last60_guard3.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v253_state_signal_disambiguation_audit_20260701\tables\v253a_current_physio_features_1hz.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v253_state_signal_disambiguation_audit_20260701\tables\v253a_feature_block_audit.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v253_state_signal_disambiguation_audit_20260701\tables\v253a_neighbor_divergence_by_feature_group.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v253_state_signal_disambiguation_audit_20260701\tables\v253a_summary_by_feature_group_bucket_delay.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v253_state_signal_disambiguation_audit_20260701\tables\v253a_key_comparison_vs_vehicle_only.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v253_state_signal_disambiguation_audit_20260701\tables\v253a_error_ambiguity_correlation_by_feature_group.csv`
- 图：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v253_state_signal_disambiguation_audit_20260701\figures\v253a_state_signal_badtop10_disambiguation.png`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v253_state_signal_disambiguation_audit_20260701\figures\v253a_state_signal_delta_vs_vehicle_only.png`
- 日志与校验：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v253_state_signal_disambiguation_audit_20260701\logs\guardrail_check.json`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v253_state_signal_disambiguation_audit_20260701\logs\input_file_hashes.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v253_state_signal_disambiguation_audit_20260701\logs\file_inventory.csv`
- 结论：状态信号可进入下一阶段，但当前直接拼接输入没有降低未来分叉；建议作为多模态概率预测的条件变量和不确定性校准信号。

---

## 2026-07-01 v252 input-similarity future-divergence audit

- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v252_input_similarity_future_divergence_20260701.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v252_input_similarity_future_divergence_20260701`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v252_input_similarity_future_divergence_20260701\reports\v252_input_similarity_future_divergence_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v252_input_similarity_future_divergence_20260701_pack.zip`
- 核心表：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v252_input_similarity_future_divergence_20260701\tables\v252_neighbor_divergence_by_sample.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v252_input_similarity_future_divergence_20260701\tables\v252_neighbor_detail.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v252_input_similarity_future_divergence_20260701\tables\v252_summary_by_delay_bucket.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v252_input_similarity_future_divergence_20260701\tables\v252_error_ambiguity_correlation.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v252_input_similarity_future_divergence_20260701\tables\v252_high_ambiguity_error_overlap.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v252_input_similarity_future_divergence_20260701\tables\v252_casebook_index.csv`
- 图：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v252_input_similarity_future_divergence_20260701\figures\v252_error_vs_neighbor_future_divergence.png`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v252_input_similarity_future_divergence_20260701\figures\v252_neighbor_divergence_by_error_group.png`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v252_input_similarity_future_divergence_20260701\figures\v252_delay_future_divergence_summary.png`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v252_input_similarity_future_divergence_20260701\figures\v252_casebook_high_error_high_ambiguity.png`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v252_input_similarity_future_divergence_20260701\figures\v252_casebook_worst_regression_neighbors.png`
- 日志与校验：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v252_input_similarity_future_divergence_20260701\logs\guardrail_check.json`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v252_input_similarity_future_divergence_20260701\logs\input_file_hashes.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v252_input_similarity_future_divergence_20260701\logs\file_inventory.csv`
- 结论：v252 是任务可辨识性证据，不是模型提升。它支持下一步从单条确定性曲线回归转向概率/多模态/不确定性预测。

---

## 2026-07-01 v251 locked robustness audit for v250_minimal_lateral7

- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v251_locked_robustness_v250_20260701.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v251_locked_robustness_v250_20260701`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v251_locked_robustness_v250_20260701\reports\v251_locked_robustness_v250_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v251_locked_robustness_v250_20260701_pack.zip`
- 核心表：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v251_locked_robustness_v250_20260701\tables\v251_sample_locked_delta.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v251_locked_robustness_v250_20260701\tables\v251_bucket_delay_locked_summary.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v251_locked_robustness_v250_20260701\tables\v251_subject_delay_locked_summary.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v251_locked_robustness_v250_20260701\tables\v251_subject_locked_summary.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v251_locked_robustness_v250_20260701\tables\v251_recording_locked_summary.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v251_locked_robustness_v250_20260701\tables\v251_event_bootstrap_ci.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v251_locked_robustness_v250_20260701\tables\v251_worst_regressions.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v251_locked_robustness_v250_20260701\tables\v251_top_improvements.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v251_locked_robustness_v250_20260701\tables\v251_bad_top10_casebook_index.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v251_locked_robustness_v250_20260701\tables\v251_next_decision.csv`
- 图：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v251_locked_robustness_v250_20260701\figures\v251_test_bucket_delay_tail_delta.png`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v251_locked_robustness_v250_20260701\figures\v251_subject_bucket_tail_delta.png`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v251_locked_robustness_v250_20260701\figures\v251_bootstrap_ci_all_delay.png`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v251_locked_robustness_v250_20260701\figures\v251_bad_top10_casebook.png`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v251_locked_robustness_v250_20260701\figures\v251_worst_regression_casebook.png`
- 日志与校验：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v251_locked_robustness_v250_20260701\logs\guardrail_check.json`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v251_locked_robustness_v250_20260701\logs\input_file_hashes.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v251_locked_robustness_v250_20260701\logs\file_inventory.csv`
- 结论：v251 支持 `v250_minimal_lateral7` 进入下一主线候选打包；仍不是 formal replacement。

---

## 2026-06-30 v250 history-channel ablation

- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v250_history_channel_ablation_20260630.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v250_history_channel_ablation_20260630`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v250_history_channel_ablation_20260630\reports\v250_history_channel_ablation_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v250_history_channel_ablation_20260630_pack.zip`
- 模型文件：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v250_history_channel_ablation_20260630\models\v250_best_channel_ablation_diagnostic.pt`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v250_history_channel_ablation_20260630\models\v250_scalers_and_selection.pkl`
- 预测数组：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v250_history_channel_ablation_20260630\v250_channel_ablation_predictions.npz`
- 核心表：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v250_history_channel_ablation_20260630\tables\v250_channel_groups.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v250_history_channel_ablation_20260630\tables\v250_model_selection_validation_channel_ablation.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v250_history_channel_ablation_20260630\tables\v250_metrics_by_delay_and_bucket.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v250_history_channel_ablation_20260630\tables\v250_compare_vs_v241_original_remaining.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v250_history_channel_ablation_20260630\tables\v250_shape_summary.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v250_history_channel_ablation_20260630\tables\v250_input_neighborhood_ambiguity_by_channel.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v250_history_channel_ablation_20260630\tables\v250_input_neighborhood_ambiguity_summary.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v250_history_channel_ablation_20260630\tables\v250_next_decision.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v250_history_channel_ablation_20260630\tables\v250_training_history.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v250_history_channel_ablation_20260630\tables\v250_split_integrity_check.csv`
- 图：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v250_history_channel_ablation_20260630\figures\v250_tail_delta_by_channel_group.png`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v250_history_channel_ablation_20260630\figures\v250_neighbor_ambiguity_by_channel_group.png`
- 日志与校验：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v250_history_channel_ablation_20260630\logs\guardrail_check.json`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v250_history_channel_ablation_20260630\logs\input_file_hashes.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v250_history_channel_ablation_20260630\logs\file_inventory.csv`
- 结论：`v250_minimal_lateral7` 是新的 channel-ablation 候选，说明历史通道精简有效；仍需 locked robustness，不是 formal replacement。

---

## 2026-06-30 v249 shape-aware curve model 训练与审计

- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v249_shape_aware_curve_model_20260630.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v249_shape_aware_curve_model_20260630`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v249_shape_aware_curve_model_20260630\reports\v249_shape_aware_curve_model_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v249_shape_aware_curve_model_20260630_pack.zip`
- 模型文件：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v249_shape_aware_curve_model_20260630\models\v249_best_shape_aware_diagnostic.pt`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v249_shape_aware_curve_model_20260630\models\v249_scalers_and_selection.pkl`
- 预测数组：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v249_shape_aware_curve_model_20260630\v249_shape_aware_predictions.npz`
- 核心表：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v249_shape_aware_curve_model_20260630\tables\v249_model_selection_validation_shape.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v249_shape_aware_curve_model_20260630\tables\v249_metrics_by_delay_and_bucket.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v249_shape_aware_curve_model_20260630\tables\v249_compare_vs_v241_original_remaining.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v249_shape_aware_curve_model_20260630\tables\v249_shape_summary.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v249_shape_aware_curve_model_20260630\tables\v249_per_sample_shape_delta_vs_v241.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v249_shape_aware_curve_model_20260630\tables\v249_worst_regressions_vs_v241.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v249_shape_aware_curve_model_20260630\tables\v249_top_improvements_vs_v241.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v249_shape_aware_curve_model_20260630\tables\v249_input_neighborhood_ambiguity_audit.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v249_shape_aware_curve_model_20260630\tables\v249_next_decision.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v249_shape_aware_curve_model_20260630\tables\v249_training_history.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v249_shape_aware_curve_model_20260630\tables\v249_split_integrity_check.csv`
- 图：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v249_shape_aware_curve_model_20260630\figures\v249_shape_casebook_test_hard.png`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v249_shape_aware_curve_model_20260630\figures\v249_tail_delta_by_bucket.png`
- 日志与校验：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v249_shape_aware_curve_model_20260630\logs\guardrail_check.json`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v249_shape_aware_curve_model_20260630\logs\leakage_check.json`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v249_shape_aware_curve_model_20260630\logs\run_manifest.json`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v249_shape_aware_curve_model_20260630\logs\input_file_hashes.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v249_shape_aware_curve_model_20260630\logs\file_inventory.csv`
- 结论：best diagnostic model 为 `v249c_shape_conditioned_residual`，但 `accepted_as_shape_candidate=False`。v249 不替代 v241；它的主要价值是证明简单 shape-aware deterministic objective 仍会压平强变化样本，并暴露 hard case 的输入邻域歧义。

---

## 2026-06-30 v248 best-anchor 后残余轨迹形状误差审查

- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v248_best_anchor_residual_shape_audit_20260630.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v248_best_anchor_residual_shape_audit_20260630`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v248_best_anchor_residual_shape_audit_20260630\reports\v248_best_anchor_residual_shape_audit_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v248_best_anchor_residual_shape_audit_20260630_pack.zip`
- 核心表：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v248_best_anchor_residual_shape_audit_20260630\tables\v248_anchor_vs_shape_summary.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v248_best_anchor_residual_shape_audit_20260630\tables\v248_best_anchor_residual_decomposition.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v248_best_anchor_residual_shape_audit_20260630\tables\v248_shape_error_categories.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v248_best_anchor_residual_shape_audit_20260630\tables\v248_peak_underestimation_table.csv`
- 图：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v248_best_anchor_residual_shape_audit_20260630\figures\v248_best_anchor_still_bad_casebook.png`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v248_best_anchor_residual_shape_audit_20260630\figures\v248_improved_but_still_wrong_casebook.png`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v248_best_anchor_residual_shape_audit_20260630\figures\v248_peak_underestimation_casebook.png`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v248_best_anchor_residual_shape_audit_20260630\figures\v248_error_decomposition_scatter.png`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v248_best_anchor_residual_shape_audit_20260630\figures\v248_shape_category_summary.png`
- 日志与校验：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v248_best_anchor_residual_shape_audit_20260630\logs\guardrail_check.json`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v248_best_anchor_residual_shape_audit_20260630\logs\run_manifest.json`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v248_best_anchor_residual_shape_audit_20260630\logs\input_file_hashes.csv`
- 结论：best anchor 只能解释一部分差样本；剩余 hard case 的预测曲线仍明显偏平、幅值不足、斜率不足，部分样本方向/转折错误。v249 不应继续做 v222a gate、删样本、轻量 residual 或单纯锚点 selector，而应做 shape-aware 曲线建模。

---

## 2026-06-26 v242 联合曲线解码模型训练

- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v242_joint_curve_decoder_20260626.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v242_joint_curve_decoder_20260626`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v242_joint_curve_decoder_20260626\reports\v242_joint_curve_decoder_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v242_joint_curve_decoder_20260626\v242_joint_curve_decoder_pack.zip`
- 模型文件：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v242_joint_curve_decoder_20260626\models\v242_best_joint_curve_diagnostic.pt`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v242_joint_curve_decoder_20260626\models\v242_scalers_and_selection.pkl`
- 预测数组：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v242_joint_curve_decoder_20260626\v242_joint_curve_predictions.npz`
- 核心表：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v242_joint_curve_decoder_20260626\tables\v242_model_selection_validation_noharm.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v242_joint_curve_decoder_20260626\tables\v242_metrics_by_delay_and_bucket.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v242_joint_curve_decoder_20260626\tables\v242_compare_vs_v236_v239_v241_original_remaining.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v242_joint_curve_decoder_20260626\tables\v242_per_sample_delta_vs_v241.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v242_joint_curve_decoder_20260626\tables\v242_per_sample_delta_summary_vs_v241.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v242_joint_curve_decoder_20260626\tables\v242_worst_regressions_vs_v241.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v242_joint_curve_decoder_20260626\tables\v242_top_improvements_vs_v241.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v242_joint_curve_decoder_20260626\tables\v242_next_decision.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v242_joint_curve_decoder_20260626\tables\v242_split_integrity_check.csv`
- 图：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v242_joint_curve_decoder_20260626\figures\v242_joint_curve_tail_compare_observe_later_like.png`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v242_joint_curve_decoder_20260626\figures\v242_joint_curve_tail_compare_strong_steer.png`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v242_joint_curve_decoder_20260626\figures\v242_joint_curve_tail_compare_normal_predictable.png`
- 日志与校验：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v242_joint_curve_decoder_20260626\logs\guardrail_check.json`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v242_joint_curve_decoder_20260626\logs\leakage_check.json`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v242_joint_curve_decoder_20260626\logs\run_manifest.json`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v242_joint_curve_decoder_20260626\logs\input_file_hashes.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v242_joint_curve_decoder_20260626\logs\file_inventory.json`
- 结论：v242 相对 v236 有效，但没有超过 v241；`accepted_as_next_candidate=False`，当前最强候选仍为 `v241_tcn_mha_h96`。
---

## 2026-06-26 v241 更强时序模型受控实验

- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v241_stronger_temporal_model_20260626.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v241_stronger_temporal_model_20260626`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v241_stronger_temporal_model_20260626\reports\v241_stronger_temporal_model_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v241_stronger_temporal_model_20260626\v241_stronger_temporal_model_pack.zip`
- 模型文件：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v241_stronger_temporal_model_20260626\models\v241_best_stronger_temporal_diagnostic.pt`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v241_stronger_temporal_model_20260626\models\v241_scalers_and_selection.pkl`
- 预测数组：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v241_stronger_temporal_model_20260626\v241_stronger_temporal_predictions.npz`
- 核心表：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v241_stronger_temporal_model_20260626\tables\v241_model_selection_validation_noharm.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v241_stronger_temporal_model_20260626\tables\v241_metrics_by_delay_and_bucket.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v241_stronger_temporal_model_20260626\tables\v241_compare_vs_v236_v238_v239_original_remaining.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v241_stronger_temporal_model_20260626\tables\v241_per_sample_delta_vs_v239.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v241_stronger_temporal_model_20260626\tables\v241_per_sample_delta_summary_vs_v239.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v241_stronger_temporal_model_20260626\tables\v241_worst_regressions_vs_v239.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v241_stronger_temporal_model_20260626\tables\v241_top_improvements_vs_v239.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v241_stronger_temporal_model_20260626\tables\v241_next_decision.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v241_stronger_temporal_model_20260626\tables\v241_split_integrity_check.csv`
- 图：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v241_stronger_temporal_model_20260626\figures\v241_stronger_tail_compare_observe_later_like.png`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v241_stronger_temporal_model_20260626\figures\v241_stronger_tail_compare_strong_steer.png`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v241_stronger_temporal_model_20260626\figures\v241_stronger_tail_compare_normal_predictable.png`
- 日志与校验：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v241_stronger_temporal_model_20260626\logs\guardrail_check.json`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v241_stronger_temporal_model_20260626\logs\leakage_check.json`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v241_stronger_temporal_model_20260626\logs\run_manifest.json`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v241_stronger_temporal_model_20260626\logs\input_file_hashes.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v241_stronger_temporal_model_20260626\logs\file_inventory.json`
- 结论：`v241_tcn_mha_h96` 可作为下一阶段 stronger temporal candidate，推荐进入 `v242_locked_test_report_for_stronger_temporal_candidate`；仍非 formal replacement。
---

## 2026-06-26 v240 locked attention audit

- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v240_locked_attention_audit_20260626.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v240_locked_attention_audit_20260626`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v240_locked_attention_audit_20260626\reports\v240_locked_attention_audit_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v240_locked_attention_audit_20260626\v240_locked_attention_audit_pack.zip`
- 核心表：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v240_locked_attention_audit_20260626\tables\v240_locked_overall_summary.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v240_locked_attention_audit_20260626\tables\v240_subbucket_noharm_audit.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v240_locked_attention_audit_20260626\tables\v240_per_sample_locked_metrics.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v240_locked_attention_audit_20260626\tables\v240_top_observe_later_improvements.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v240_locked_attention_audit_20260626\tables\v240_top_normal_improvements.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v240_locked_attention_audit_20260626\tables\v240_worst_regressions.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v240_locked_attention_audit_20260626\tables\v240_worst_v239_residuals.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v240_locked_attention_audit_20260626\tables\v240_strong_400_1000_regressions.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v240_locked_attention_audit_20260626\tables\v240_attention_casebook_index.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v240_locked_attention_audit_20260626\tables\v240_attention_time_focus_summary.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v240_locked_attention_audit_20260626\tables\v240_next_decision.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v240_locked_attention_audit_20260626\tables\v240_split_integrity_check.csv`
- 图目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v240_locked_attention_audit_20260626\figures\attention_casebook`
- 日志与校验：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v240_locked_attention_audit_20260626\logs\guardrail_check.json`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v240_locked_attention_audit_20260626\logs\leakage_check.json`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v240_locked_attention_audit_20260626\logs\run_manifest.json`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v240_locked_attention_audit_20260626\logs\input_file_hashes.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v240_locked_attention_audit_20260626\logs\file_inventory.json`
- 结论：`attention_candidate_survives_locked_audit=True`，但 `formal_replacement_allowed=False`；observe_later_like 和 normal_predictable 通过 locked no-harm，strong_steer 的 400ms/1000ms 例外需要进入 v241 人工复核。
---

## 2026-06-26 v239 轻量 temporal attention + no-harm 约束实验

- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v239_light_attention_noharm_20260626.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v239_light_attention_noharm_20260626`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v239_light_attention_noharm_20260626\reports\v239_light_attention_noharm_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v239_light_attention_noharm_20260626\v239_light_attention_noharm_pack.zip`
- 模型文件：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v239_light_attention_noharm_20260626\models\v239_best_light_attention_diagnostic.pt`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v239_light_attention_noharm_20260626\models\v239_scalers_and_selection.pkl`
- 预测数组：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v239_light_attention_noharm_20260626\v239_light_attention_predictions.npz`
- 核心表：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v239_light_attention_noharm_20260626\tables\v239_task_construction_audit.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v239_light_attention_noharm_20260626\tables\v239_point_training_rows_by_delay.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v239_light_attention_noharm_20260626\tables\v239_model_selection_validation_noharm.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v239_light_attention_noharm_20260626\tables\v239_metrics_by_delay_and_bucket.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v239_light_attention_noharm_20260626\tables\v239_compare_vs_v236_original_remaining.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v239_light_attention_noharm_20260626\tables\v239_attention_training_history.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v239_light_attention_noharm_20260626\tables\v239_next_model_decision.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v239_light_attention_noharm_20260626\tables\v239_split_integrity_check.csv`
- 图：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v239_light_attention_noharm_20260626\figures\v239_attention_tail_compare_observe_later_like.png`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v239_light_attention_noharm_20260626\figures\v239_attention_tail_compare_strong_steer.png`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v239_light_attention_noharm_20260626\figures\v239_attention_tail_compare_normal_predictable.png`
- 日志与校验：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v239_light_attention_noharm_20260626\logs\guardrail_check.json`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v239_light_attention_noharm_20260626\logs\leakage_check.json`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v239_light_attention_noharm_20260626\logs\run_manifest.json`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v239_light_attention_noharm_20260626\logs\input_file_hashes.csv`
- 结论：`v239_light_attention_h32` 通过 validation no-harm，可作为下一候选；但本轮仍非 formal replacement，下一步建议 `v240_locked_test_report_for_attention_candidate`。
---

## 2026-06-26 v238 任务构造与小型 rolling 模型重搭

- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v238_task_model_redesign_20260626.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v238_task_model_redesign_20260626`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v238_task_model_redesign_20260626\reports\v238_task_model_redesign_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v238_task_model_redesign_20260626\v238_task_model_redesign_pack.zip`
- 模型文件：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v238_task_model_redesign_20260626\models\v238_selected_point_model.pkl`
- 预测数组：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v238_task_model_redesign_20260626\v238_original_remaining_predictions.npz`
- 核心表：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v238_task_model_redesign_20260626\tables\v238_task_construction_audit.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v238_task_model_redesign_20260626\tables\v238_point_training_rows_by_delay.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v238_task_model_redesign_20260626\tables\v238_model_selection_validation_only.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v238_task_model_redesign_20260626\tables\v238_metrics_by_delay_and_bucket.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v238_task_model_redesign_20260626\tables\v238_compare_v236_original_remaining.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v238_task_model_redesign_20260626\tables\v238_selected_per_sample_metrics.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v238_task_model_redesign_20260626\tables\v238_next_model_decision.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v238_task_model_redesign_20260626\tables\v238_split_integrity_check.csv`
- 图：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v238_task_model_redesign_20260626\figures\v238_compare_tail_original_remaining_observe_later_like.png`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v238_task_model_redesign_20260626\figures\v238_compare_tail_original_remaining_strong_steer.png`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v238_task_model_redesign_20260626\figures\v238_compare_tail_original_remaining_normal_predictable.png`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v238_task_model_redesign_20260626\figures\v238_validation_model_selection.png`
- 日志与校验：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v238_task_model_redesign_20260626\logs\guardrail_check.json`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v238_task_model_redesign_20260626\logs\leakage_check.json`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v238_task_model_redesign_20260626\logs\run_manifest.json`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v238_task_model_redesign_20260626\logs\input_file_hashes.csv`
- 结论：接受 v238 的 `original_remaining` masked point-level 任务构造；不接受当前 selected MLP 作为正式替代模型；下一步建议 `v239_noharm_constrained_original_remaining_model`。
---

## 2026-06-25 v237 rolling target / phase audit（audit-only 决策点）

- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v237_rolling_target_phase_audit_20260624.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v237_rolling_target_phase_audit_20260624`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v237_rolling_target_phase_audit_20260624\reports\v237_rolling_target_phase_audit_cn.md`
- 审计 ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v237_rolling_target_phase_audit_20260624\v237_rolling_target_phase_audit_pack.zip`
- 核心表：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v237_rolling_target_phase_audit_20260624\tables\v237_target_definition_sanity_check.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v237_rolling_target_phase_audit_20260624\tables\v237_receding_vs_original_remaining_metrics.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v237_rolling_target_phase_audit_20260624\tables\v237_metrics_by_delay_and_bucket_recheck.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v237_rolling_target_phase_audit_20260624\tables\v237_observe_later_subbucket_profile.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v237_rolling_target_phase_audit_20260624\tables\v237_phase_transition_profile.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v237_rolling_target_phase_audit_20260624\tables\v237_reverse_multi_correction_delay_profile.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v237_rolling_target_phase_audit_20260624\tables\v237_1000ms_failure_audit.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v237_rolling_target_phase_audit_20260624\tables\v237_ridge_underfit_audit.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v237_rolling_target_phase_audit_20260624\tables\v237_alpha_validation_curve_audit.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v237_rolling_target_phase_audit_20260624\tables\v237_next_model_decision.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v237_rolling_target_phase_audit_20260624\tables\v237_v236_receding_metric_reproduction.csv`
- 图目录：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v237_rolling_target_phase_audit_20260624\figures\observe_later_receding_vs_remaining_curves`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v237_rolling_target_phase_audit_20260624\figures\strong_steer_delay_curves`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v237_rolling_target_phase_audit_20260624\figures\reverse_multicorrection_delay_curves`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v237_rolling_target_phase_audit_20260624\figures\phase_transition_examples`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v237_rolling_target_phase_audit_20260624\figures\ridge_underfit_examples`
- 日志：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v237_rolling_target_phase_audit_20260624\logs\run_manifest.json`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v237_rolling_target_phase_audit_20260624\logs\input_file_hashes.json`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v237_rolling_target_phase_audit_20260624\logs\guardrail_check.json`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v237_rolling_target_phase_audit_20260624\logs\leakage_check.json`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v237_rolling_target_phase_audit_20260624\logs\consistency_check.json`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v237_rolling_target_phase_audit_20260624\logs\forbidden_scan_report.json`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v237_rolling_target_phase_audit_20260624\logs\file_inventory.json`
- 结果摘要：v237 未训练模型、未生成新预测、未搜索 alpha/threshold/tau、未创建 gate/router/selector。target sanity 全 pass；v236 receding 指标复现最大差异 `2.3841858e-07`；`observe_later_like` 的 1000ms receding failure 主要与 horizon 后移进入新阶段有关，new phase 命中率 `0.888889`；strong_steer rolling 收益保持；Ridge underfit 和 alpha 最大边界证据成立。`v238_allowed=True` 仅表示下一步可候选小 rolling 模型，尚未执行 v238。
- 验证摘要：`py_compile` 通过；完整运行通过；required files `missing=[]`；guardrail/leakage/consistency 均 pass；forbidden hits `[]`；ZIP `testzip=None`、条目数 `29`。
---

## 2026-06-24 连续车辆源数据审计（独立于样本集）

- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\01_audit\scripts\vehicle_source_audit_20260624.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\01_audit\vehicle_source_audit_20260624`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\vehicle_source_audit_20260624_cn.md`
- 文件盘点：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\01_audit\vehicle_source_audit_20260624\tables\vehicle_file_inventory.csv`
- 源层汇总：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\01_audit\vehicle_source_audit_20260624\tables\source_layer_summary.csv`
- 文件级车辆质量表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\01_audit\vehicle_source_audit_20260624\tables\file_vehicle_quality_summary.csv`
- 字段级数值分布表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\01_audit\vehicle_source_audit_20260624\tables\vehicle_numeric_column_summary.csv`
- 同记录跨源层 lineage 对照：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\01_audit\vehicle_source_audit_20260624\tables\recording_cluster_summary.csv`
- 被试汇总：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\01_audit\vehicle_source_audit_20260624\tables\subject_vehicle_summary.csv`
- 道路类型汇总：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\01_audit\vehicle_source_audit_20260624\tables\road_type_summary.csv`
- 发现项：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\01_audit\vehicle_source_audit_20260624\tables\vehicle_source_audit_findings.csv`
- 图：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\01_audit\vehicle_source_audit_20260624\figures\vehicle_recording_duration_hist.png`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\01_audit\vehicle_source_audit_20260624\figures\vehicle_duration_by_subject.png`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\01_audit\vehicle_source_audit_20260624\figures\vehicle_audit_flag_counts.png`
- 运行清单：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\01_audit\vehicle_source_audit_20260624\logs\run_manifest.json`
- 结果摘要：候选文件 `358` 个，纳入连续车辆审计 `182` 个，覆盖 `18` 名被试、`91` 个记录键、约 `25.31` 小时。主 `vehicle_aligned_cleaned` 层 `91` 个文件字段完整、时间轴稳定；补充 `(2)_vehicle_fixed_200Hz` 层 `91` 个文件存在字段缺失和采样/行数不一致风险。`车辆清理后` 目录另含 `82` 个 PhysioLAB 生理文件和 `85` 个 EEG/加速度文件，后续必须按字段白名单识别车辆源。
- 验证摘要：`py_compile` 通过；完整运行 `processed_files=182`、`errors=[]`；3 张图像文件均非空。

---

## 2026-06-24 v236 rolling/reanchor 数据集与 joint Ridge 小基线（最新）

- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v236_rolling_reanchor_dataset_and_baseline_20260624.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v236_rolling_reanchor_dataset_and_baseline_20260624`
- 中文说明：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v236_rolling_reanchor_dataset_and_baseline_20260624\reports\v236_rolling_reanchor_baseline_cn.md`
- rolling 样本 manifest：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v236_rolling_reanchor_dataset_and_baseline_20260624\tables\v236_rolling_sample_manifest.csv`
- delay 样本计数：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v236_rolling_reanchor_dataset_and_baseline_20260624\tables\v236_delay_sample_counts.csv`
- split 泄漏检查表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v236_rolling_reanchor_dataset_and_baseline_20260624\tables\v236_train_val_test_event_split_check.csv`
- by-delay 指标：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v236_rolling_reanchor_dataset_and_baseline_20260624\tables\v236_baseline_metrics_by_delay.csv`
- by-delay-and-bucket 指标：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v236_rolling_reanchor_dataset_and_baseline_20260624\tables\v236_baseline_metrics_by_delay_and_bucket.csv`
- observe_later 曲线：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v236_rolling_reanchor_dataset_and_baseline_20260624\tables\v236_observe_later_improvement_curve.csv`
- strong event 曲线：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v236_rolling_reanchor_dataset_and_baseline_20260624\tables\v236_strong_event_improvement_curve.csv`
- normal no-harm 检查：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v236_rolling_reanchor_dataset_and_baseline_20260624\tables\v236_normal_sample_noharm_check.csv`
- old 0ms formal 对照：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v236_rolling_reanchor_dataset_and_baseline_20260624\tables\v236_metric_vs_old_0ms_formal_reference.csv`
- 模型选择表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v236_rolling_reanchor_dataset_and_baseline_20260624\tables\v236_model_selection_validation_only.csv`
- 逐样本指标：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v236_rolling_reanchor_dataset_and_baseline_20260624\tables\v236_selected_per_sample_metrics.csv`
- 特征表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v236_rolling_reanchor_dataset_and_baseline_20260624\tables\v236_feature_schema.csv`
- 数据与预测数组：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v236_rolling_reanchor_dataset_and_baseline_20260624\v236_rolling_dataset_arrays_and_predictions.npz`
- selected Ridge 模型：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v236_rolling_reanchor_dataset_and_baseline_20260624\models\v236_joint_ridge_selected.pkl`
- 图：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v236_rolling_reanchor_dataset_and_baseline_20260624\figures\v236_tail_rmse_by_delay_buckets.png`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v236_rolling_reanchor_dataset_and_baseline_20260624\figures\v236_strong_under_by_delay_buckets.png`
- Guardrail：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v236_rolling_reanchor_dataset_and_baseline_20260624\logs\guardrail_check.json`
- Leakage：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v236_rolling_reanchor_dataset_and_baseline_20260624\logs\leakage_check.json`
- 文件清单：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v236_rolling_reanchor_dataset_and_baseline_20260624\logs\file_inventory.json`
- ZIP 包：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v236_rolling_reanchor_dataset_and_baseline_20260624\v236_rolling_reanchor_dataset_and_baseline_pack.zip`
- 结果摘要：`7002` 个 rolling 样本覆盖 `1167` 个事件；同一 event_uid 跨 split 数为 `0`；selected alpha=`1000`。strong_steer 桶随 delay 改善明显，但 observe_later_like 桶没有达到 later observation 持续改善的成功条件；v236 0ms baseline 弱于旧 formal，因此本轮只能作为 rolling 任务可行性小基线。
- 验证摘要：`py_compile` 通过；完整运行通过；必需文件 `missing=[]`；guardrail `pass`；leakage `pass`；ZIP `testzip bad=None`、文件数 `22`。

---

## 2026-06-24 v235 删除 observe_later_like 样本后的受控重训实验（最新）

- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v235_remove_observe_later_retrain_20260624.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v235_remove_observe_later_retrain_20260624`
- 中文说明：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v235_remove_observe_later_retrain_20260624\reports\v235_remove_observe_later_retrain_cn.md`
- 删除来源表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v235_remove_observe_later_retrain_20260624\tables\v235_remove_id_source_from_v234.csv`
- 删除计数表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v235_remove_observe_later_retrain_20260624\tables\v235_removed_sample_counts.csv`
- 过滤后 validation 选择表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v235_remove_observe_later_retrain_20260624\tables\v235_validation_selection_filtered.csv`
- 删除后重训模型指标：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v235_remove_observe_later_retrain_20260624\tables\v235_selected_metrics_filtered.csv`
- 旧模型同一过滤子集指标：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v235_remove_observe_later_retrain_20260624\tables\v235_old_selected_metrics_filtered.csv`
- 被删除样本 holdout 诊断指标：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v235_remove_observe_later_retrain_20260624\tables\v235_selected_metrics_removed_holdout.csv`
- 主对照表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v235_remove_observe_later_retrain_20260624\tables\v235_comparison_summary.csv`
- 模型 manifest：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v235_remove_observe_later_retrain_20260624\tables\v235_model_manifest.csv`
- selected 模型路径表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v235_remove_observe_later_retrain_20260624\tables\v235_selected_model_paths.csv`
- leakage guard：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v235_remove_observe_later_retrain_20260624\tables\v235_leakage_guard_result.csv`
- 对比图：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v235_remove_observe_later_retrain_20260624\figures\v235_test_rmse_comparison.png`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v235_remove_observe_later_retrain_20260624\figures\v235_test_tail_rmse_comparison.png`
- ZIP 包：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v235_remove_observe_later_retrain_20260624\v235_remove_observe_later_retrain_pack.zip`
- GPTPro 代码审查包目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v235_remove_observe_later_retrain_20260624\gptpro_code_review_pack_20260624`
- GPTPro 代码审查 ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v235_remove_observe_later_retrain_20260624\v235_gptpro_code_review_pack_20260624.zip`
- GPTPro 审查提示：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v235_remove_observe_later_retrain_20260624\gptpro_code_review_pack_20260624\gptpro_review_prompt_cn.md`
- 运行日志：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v235_remove_observe_later_retrain_20260624\logs\run_manifest.json`
- 结果摘要：loose 删除 `121/1167`，strict 删除 `117/963`；旧 v222a selected full test RMSE loose/strict 为 `0.555940/0.571966`，旧模型同一过滤 test 子集为 `0.482685/0.506547`，删除后重训为 `0.474318/0.504151`。重训额外收益小于直接删除带来的分布变化，说明 v235 是诊断对照，不是正式方法替代。
- 边界说明：本轮重训 v222a light residual/融合层和新增 absolute Ridge 对照，不是端到端底座候选网络重训；不改 formal headline；不把删难样本后的指标写作模型能力提升。
- 验证摘要：`py_compile` 通过；完整脚本运行通过；feature schema guard `pass`；selection 使用 filtered validation only；图像非空检查通过；ZIP `testzip bad=None`、文件数 `24`。

---

## 2026-06-24 v234 短观察后预测评估层构建包（最新）

- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v234_short_observation_prediction_layer_20260624.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v234_short_observation_prediction_layer_20260624`
- 中文说明：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v234_short_observation_prediction_layer_20260624\reports\v234_short_observation_prediction_layer_cn.md`
- 层定义表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v234_short_observation_prediction_layer_20260624\tables\v234_short_observation_layer_definition.csv`
- 样本层分配表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v234_short_observation_prediction_layer_20260624\tables\v234_short_observation_layer_assignments.csv`
- 真实目标曲线长表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v234_short_observation_prediction_layer_20260624\tables\v234_short_observation_target_curves.csv`
- 上下文网格：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v234_short_observation_prediction_layer_20260624\tables\v234_short_observation_context_grid.csv`
- 人工审核模板：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v234_short_observation_prediction_layer_20260624\tables\v234_short_observation_manual_review_template.csv`
- 图目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v234_short_observation_prediction_layer_20260624\figures`
- 图拼接总览：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v234_short_observation_prediction_layer_20260624\figures\v234_short_observation_layer_contact_sheet.png`
- ZIP 包：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v234_short_observation_prediction_layer_20260624\v234_short_observation_prediction_layer_pack.zip`
- 运行日志：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v234_short_observation_prediction_layer_20260624\logs\run_manifest.json`
- 结果摘要：v234 将 v233 的 `10` 个 `observe_later_review` 样本构造成短观察后预测评估层；定义 `5` 个层（0.0s 纯提前参考层 + 0.5/1.0/1.5/2.0s 短观察层），生成 `50` 条层分配、`1050` 个 2.0s horizon 真实目标曲线点和 `10` 张样本图。默认 0.5s 层下，多数样本仍有较大 `remaining_peak`，说明短观察后仍有真实未来要预测，不是简单补全。
- 边界说明：本轮只构建评估层和真实目标曲线，不训练模型、不修改标签、不改 formal headline；不把旧 formal prediction 硬评到新观察层；不重启硬响应类型分类；不把简单多候选轨迹输出作为主线。
- 验证摘要：`py_compile` 通过；完整脚本运行通过；`errors=[]`；图像非空检查通过；ZIP `testzip bad=None`、文件数 `17`。

---

## 2026-06-24 v233 自适应锚点 / 观察时长策略审核包（最新）

- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v233_adaptive_anchor_observation_policy_20260624.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v233_adaptive_anchor_observation_policy_20260624`
- 中文说明：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v233_adaptive_anchor_observation_policy_20260624\reports\v233_adaptive_anchor_observation_policy_cn.md`
- 样本策略表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v233_adaptive_anchor_observation_policy_20260624\tables\v233_anchor_observation_policy_table.csv`
- 人工审核表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v233_adaptive_anchor_observation_policy_20260624\tables\v233_anchor_observation_policy_review_table.csv`
- 观察延迟表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v233_adaptive_anchor_observation_policy_20260624\tables\v233_observe_delay_grid.csv`
- 图目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v233_adaptive_anchor_observation_policy_20260624\figures`
- 策略图拼接总览：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v233_adaptive_anchor_observation_policy_20260624\figures\v233_adaptive_anchor_policy_contact_sheet.png`
- ZIP 包：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v233_adaptive_anchor_observation_policy_20260624\v233_adaptive_anchor_observation_policy_pack.zip`
- 运行日志：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v233_adaptive_anchor_observation_policy_20260624\logs\run_manifest.json`
- 结果摘要：v233 在 v232 的 `29` 个样本上区分“提前重锚定”和“后移观察点”。策略分布为 `observe_later_review=10`、`reanchor_earlier_review=5`、`reanchor_earlier_or_ambiguous_review=6`、`large_change_standard_or_ambiguous=1`、`standard_anchor_review=7`。后移观察类覆盖“旧锚点前证据弱但后续变化大”的样本，建议先作为单独的“短观察后预测”评估层，不与纯提前预测混在同一指标里。
- 边界说明：本轮只生成策略审核表和图，不训练模型、不改标签、不改 formal headline；不重启硬响应类型分类；不把简单多候选轨迹输出作为主线。
- 验证摘要：`py_compile` 通过；完整脚本运行通过；`errors=[]`；图像非空检查通过；ZIP `testzip bad=None`、文件数 `20`。

---

## 2026-06-24 v232 过晚锚点重锚定候选审核包（最新）

- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v232_late_anchor_reanchor_candidates_20260624.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v232_late_anchor_reanchor_candidates_20260624`
- 中文说明：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v232_late_anchor_reanchor_candidates_20260624\reports\v232_late_anchor_reanchor_candidates_cn.md`
- 目标样本表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v232_late_anchor_reanchor_candidates_20260624\tables\v232_target_samples.csv`
- 全量打分表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v232_late_anchor_reanchor_candidates_20260624\tables\v232_reanchor_candidate_all_scored.csv`
- 人工审核表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v232_late_anchor_reanchor_candidates_20260624\tables\v232_reanchor_candidate_review_table.csv`
- 0.05 秒信号网格：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v232_late_anchor_reanchor_candidates_20260624\tables\v232_reanchor_grid_0p05s.csv`
- 关键时刻表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v232_late_anchor_reanchor_candidates_20260624\tables\v232_reanchor_key_points.csv`
- 图目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v232_late_anchor_reanchor_candidates_20260624\figures`
- 候选图拼接总览：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v232_late_anchor_reanchor_candidates_20260624\figures\v232_reanchor_candidates_contact_sheet.png`
- ZIP 包：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v232_late_anchor_reanchor_candidates_20260624\v232_late_anchor_reanchor_candidates_pack.zip`
- 运行日志：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v232_late_anchor_reanchor_candidates_20260624\logs\run_manifest.json`
- 结果摘要：v232 从 v230 casebook + v231 六样本形成 `29` 个唯一样本范围，全部完成原始车辆信号打分；输出 `11` 个 P0/P1/P2 重锚定人工审核候选，其中 P0=1、P1=4、P2=6。`rjy...010` 按用户人工反馈标为 P0，算法候选从旧锚点 `143.100s` 提前到 `138.950s`，移动 `-4.15s`，但仍需人工确认后才能用于 label window 重建。
- 边界说明：本轮只生成重锚定候选和证据，不训练模型、不改 formal headline、不自动修改标签；硬响应类型分类和简单多候选轨迹输出均不作为下一步主线。
- 验证摘要：`py_compile` 通过；完整脚本运行通过；`errors=[]`；图像非空检查通过；ZIP `testzip bad=None`、文件数 `19`。

---

## 2026-06-24 v231 六个最差样本锚点上下文人工审核包（最新）

- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v231_worst_case_anchor_context_20260624`
- 中文说明：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v231_worst_case_anchor_context_20260624\reports\v231_worst_case_anchor_context_cn.md`
- 元数据总表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v231_worst_case_anchor_context_20260624\tables\v231_anchor_metadata.csv`
- 窗口摘要表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v231_worst_case_anchor_context_20260624\tables\v231_window_summary.csv`
- 信号对齐关键时刻表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v231_worst_case_anchor_context_20260624\tables\v231_anchor_key_points.csv`
- 信号对齐 0.1 秒稀疏窗口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v231_worst_case_anchor_context_20260624\tables\v231_anchor_window_sparse_8s.csv`
- 原始 200Hz 密集窗口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v231_worst_case_anchor_context_20260624\tables\v231_anchor_window_dense_pm3s.csv`
- 图清单：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v231_worst_case_anchor_context_20260624\tables\v231_anchor_context_figures.csv`
- 图目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v231_worst_case_anchor_context_20260624\figures`
- 用户反馈修正：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v231_worst_case_anchor_context_20260624\reports\v231_user_feedback_method_correction_cn.md`
- 用户反馈覆盖表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v231_worst_case_anchor_context_20260624\tables\v231_user_feedback_overrides.csv`
- 结果摘要：按用户要求调取 6 个最差/代表性差样本的 `anchor_s`、绝对锚点时间、锚点前后原始车辆信号和方向盘相对锚点增量。关键点表按“每个信号取目标时刻最近非空值”输出，并保留毫秒误差；密集表保留原始行不填补。初判显示 2 个样本优先查锚点/窗口，3 个样本主要是反转/多次修正形态问题，1 个样本更像幅值/形态预测问题。
- 用户修正摘要：`rjy...010` 已人工确认锚点晚了；“先硬判断响应类型再预测轨迹”不作为下一步主线，因为此前已经尝试过，且存在分类错误传播风险。后续方法提升应优先考虑软门控/概率混合、多假设轨迹输出、连续相位或延迟建模。
- 用户第二轮修正摘要：过晚锚点需要进一步重锚定，不能只标记；一次性输出多个候选轨迹此前也已尝试且效果不好，即使 best candidate 仍有偏差，因此简单多候选轨迹输出也不作为主线。下一步应生成重锚定候选表和人工确认字段。
- 边界说明：这是服务于行为预测方法提升的人工审核包，不是失败机制论文包；本轮不训练新模型、不改 formal headline。

---

## 2026-06-23 v230 失败案例人工复核 / 论文案例证据包（最新）

- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v230_failure_case_manual_review_casebook_20260623.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v230_failure_case_manual_review_casebook_20260623`
- 主报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v230_failure_case_manual_review_casebook_20260623\reports\v230_failure_case_manual_review_casebook_cn.md`
- 导师讨论笔记：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v230_failure_case_manual_review_casebook_20260623\reports\v230_advisor_discussion_notes_cn.md`
- 论文失败案例小节草稿：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v230_failure_case_manual_review_casebook_20260623\reports\v230_paper_failure_case_section_draft_cn.md`
- case 选择索引：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v230_failure_case_manual_review_casebook_20260623\tables\v230_case_selection_index.csv`
- 人工复核模板：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v230_failure_case_manual_review_casebook_20260623\tables\v230_manual_review_template.csv`
- casebook 表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v230_failure_case_manual_review_casebook_20260623\tables\v230_failure_casebook_table.csv`
- claim 映射：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v230_failure_case_manual_review_casebook_20260623\tables\v230_bucket_to_claim_mapping.csv`
- 图清单：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v230_failure_case_manual_review_casebook_20260623\tables\v230_case_figure_inventory.csv`
- formal 边界检查：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v230_failure_case_manual_review_casebook_20260623\tables\v230_formal_boundary_check.csv`
- 图目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v230_failure_case_manual_review_casebook_20260623\figures\selected_casebook_figures`
- guardrail：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v230_failure_case_manual_review_casebook_20260623\logs\guardrail_check.json`
- forbidden scan：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v230_failure_case_manual_review_casebook_20260623\logs\forbidden_scan_report.json`
- consistency：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v230_failure_case_manual_review_casebook_20260623\logs\consistency_check.json`
- figure copy check：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v230_failure_case_manual_review_casebook_20260623\logs\figure_copy_check.json`
- 压缩包：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v230_failure_case_manual_review_casebook_20260623\v230_failure_case_manual_review_casebook_pack.zip`
- 结果摘要：v230 只做失败案例人工复核和论文案例证据打包；共 `46` 个 case，复制既有图 `85` 张，case 图缺失 `13` 个已记录；不训练、不新预测、不重选 formal headline。
- 验证摘要：`py_compile` 通过；完整运行通过；ZIP `bad_file=None`、文件数 `103`；required files `[]`；guardrail `pass=True`；consistency `pass=True`；forbidden hits `[]`；人工复核字段全空。
---

## 2026-06-23 v229 两个月路线经验复盘与失败分类包（最新）

- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v229_two_month_lessons_failure_taxonomy_20260623.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v229_two_month_lessons_failure_taxonomy_20260623`
- 主报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v229_two_month_lessons_failure_taxonomy_20260623\reports\v229_two_month_lessons_failure_taxonomy_cn.md`
- GPTPro 中文提问稿：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v229_two_month_lessons_failure_taxonomy_20260623\reports\v229_gptpro_next_prompt_cn.md`
- 阶段经验表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v229_two_month_lessons_failure_taxonomy_20260623\tables\v229_phase_lessons_table.csv`
- 失败桶统计：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v229_two_month_lessons_failure_taxonomy_20260623\tables\v229_failure_taxonomy_by_pool_event.csv`
- 高尾失败案例：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v229_two_month_lessons_failure_taxonomy_20260623\tables\v229_top_tail_failure_cases.csv`
- bucket 风险摘要：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v229_two_month_lessons_failure_taxonomy_20260623\tables\v229_bucket_risk_summary.csv`
- selector/candidate 诊断：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v229_two_month_lessons_failure_taxonomy_20260623\tables\v229_selector_candidate_diagnosis.csv`
- 下一步决策矩阵：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v229_two_month_lessons_failure_taxonomy_20260623\tables\v229_next_action_decision_matrix.csv`
- guardrail：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v229_two_month_lessons_failure_taxonomy_20260623\logs\guardrail_check.json`
- manifest：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v229_two_month_lessons_failure_taxonomy_20260623\logs\run_manifest.json`
- 输入哈希：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v229_two_month_lessons_failure_taxonomy_20260623\logs\input_file_hashes.json`
- 压缩包：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v229_two_month_lessons_failure_taxonomy_20260623\v229_two_month_lessons_failure_taxonomy_pack.zip`
- 结果摘要：v229 只读复盘 v220/v225/v228，不训练、不生成新预测、不重选 formal headline；核心结论是方向/普通响应较稳，但强反应幅值、极端峰值、尾段、反转/多次修正仍是主要失败区，下一步应先让 GPTPro 审阅路线边界和失败分类。
- 验证摘要：`py_compile` 通过；脚本完整运行通过；ZIP `bad_file=None`、文件数 `15`；必需文件缺失 `[]`；guardrail `pass=True`。
---

## 2026-06-22 v226 formal robustness / confidence-interval audit（最新）

- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v226_formal_robustness_ci_audit_20260622.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v226_formal_robustness_ci_audit_20260622`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v226_formal_robustness_ci_audit_20260622\reports\v226_formal_robustness_ci_audit_cn.md`
- 核心表：`formal_metric_ci_sample_bootstrap.csv`、`formal_metric_ci_subject_block_bootstrap.csv`、`formal_tail_error_concentration.csv`、`formal_readiness_decision.csv`
- 核心日志：`metric_reproduction_check.json`、`leakage_guard_report.json`、`forbidden_scan_report.json`、`table_alignment_check.json`、`file_inventory.json`
- 压缩包：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v226_formal_robustness_ci_audit_20260622\v226_formal_robustness_ci_audit_pack.zip`
- 验证摘要：ZIP `bad_file=None`，required files `[]`，formal lock / metric reproduction / leakage guard / forbidden scan / table alignment 全部 pass，figure count `4/4/2/2/2`。

---

## 2026-06-23 v227 heartbeat GPTPro 阻塞归档（最新）

- response blocked：`F:\data_set_process\data_process\gptpro_reviews\20260623_v227_heartbeat_gptpro_response_blocked.md`
- decision blocked：`F:\data_set_process\data_process\gptpro_reviews\20260623_v227_heartbeat_gptpro_decision_blocked.md`
- action items blocked：`F:\data_set_process\data_process\gptpro_reviews\20260623_v227_heartbeat_gptpro_action_items_blocked.md`
- 关联 prompt：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v227_paper_claim_readiness_pack_20260622\reports\v227_next_gptpro_prompt_ascii.md`
- 关联 ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v227_paper_claim_readiness_pack_20260622\v227_paper_claim_readiness_pack.zip`
- 验证摘要：本轮 heartbeat 复核 v227 ZIP `bad_file=None`，文件数 35；Desktop 未见有效 GPTPro 回复，Chrome bridge 因无法验证 Pro/进阶模式拒绝发送。
- 用途：记录 2026-06-23 goal-mode 自动续跑的外部阻塞状态；不是 GPTPro 新指令。

---

## 2026-06-22 v225 formal route reconstruction evidence pack

- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v225_formal_route_reconstruction_evidence_pack_20260622.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v225_formal_route_reconstruction_evidence_pack_20260622`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v225_formal_route_reconstruction_evidence_pack_20260622\reports\v225_formal_route_reconstruction_evidence_cn.md`
- formal model lock：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v225_formal_route_reconstruction_evidence_pack_20260622\tables\formal_model_lock.csv`
- overall 指标：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v225_formal_route_reconstruction_evidence_pack_20260622\tables\formal_reconstruction_metrics_overall.csv`
- by-pool 指标：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v225_formal_route_reconstruction_evidence_pack_20260622\tables\formal_reconstruction_metrics_by_pool.csv`
- by-bucket 指标：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v225_formal_route_reconstruction_evidence_pack_20260622\tables\formal_reconstruction_metrics_by_bucket.csv`
- route-event 逐样本指标：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v225_formal_route_reconstruction_evidence_pack_20260622\tables\formal_reconstruction_metrics_by_route_event.csv`
- formal 逐样本评估：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v225_formal_route_reconstruction_evidence_pack_20260622\tables\per_sample_formal_reconstruction_eval.csv`
- failure case index：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v225_formal_route_reconstruction_evidence_pack_20260622\tables\formal_failure_case_index.csv`
- diagnostic-only v222a closeout summary：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v225_formal_route_reconstruction_evidence_pack_20260622\tables\diagnostic_only_v222a_closeout_summary.csv`
- excluded diagnostic audit：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v225_formal_route_reconstruction_evidence_pack_20260622\tables\excluded_diagnostic_models_audit.csv`
- figure 目录：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v225_formal_route_reconstruction_evidence_pack_20260622\figures\formal_examples`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v225_formal_route_reconstruction_evidence_pack_20260622\figures\worst_tail_cases`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v225_formal_route_reconstruction_evidence_pack_20260622\figures\strong_under_cases`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v225_formal_route_reconstruction_evidence_pack_20260622\figures\baseline_sufficient_cases`
- manifest：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v225_formal_route_reconstruction_evidence_pack_20260622\logs\run_manifest.json`
- metric reproduction：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v225_formal_route_reconstruction_evidence_pack_20260622\logs\metric_reproduction_check.json`
- leakage guard：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v225_formal_route_reconstruction_evidence_pack_20260622\logs\leakage_guard_report.json`
- forbidden scan：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v225_formal_route_reconstruction_evidence_pack_20260622\logs\forbidden_scan_report.json`
- table alignment：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v225_formal_route_reconstruction_evidence_pack_20260622\logs\table_alignment_check.json`
- file inventory：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v225_formal_route_reconstruction_evidence_pack_20260622\logs\file_inventory.json`
- 压缩包：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v225_formal_route_reconstruction_evidence_pack_20260622\v225_formal_route_reconstruction_evidence_pack.zip`
- 结果摘要：formal headline 只保留 `loose_main_pool=avg_joint_focus` 和 `strict_main_pool=peak_floor_090`。locked test 复现为 loose RMSE/tail `0.544884/0.629752`，strict RMSE/tail `0.571770/0.658306`；diagnostic-only 模型不进入 formal 表。
- 验证摘要：`py_compile` 通过，脚本完整运行通过，ZIP `bad_file=None`，必需文件无缺失，figure count 为 `12/12/8/8`，metric reproduction / leakage guard / forbidden scan / table alignment 全 pass。

---

## 2026-06-22 v221 统一评估框架

- 脚本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v221_formal_model_leaderboard_20260622.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v221_formal_model_leaderboard_20260622`
- HTML 入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v221_formal_model_leaderboard_20260622\v221_formal_model_leaderboard_index.html`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v221_formal_model_leaderboard_20260622\reports\v221_formal_model_leaderboard_report_cn.md`
- 关键表：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v221_formal_model_leaderboard_20260622\tables\v221_candidate_decision_summary.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v221_formal_model_leaderboard_20260622\tables\v221_model_overall_metrics.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v221_formal_model_leaderboard_20260622\tables\v221_model_bucket_metrics.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v221_formal_model_leaderboard_20260622\tables\v221_noharm_vs_reference.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v221_formal_model_leaderboard_20260622\tables\v221_universal_failure_cases.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v221_formal_model_leaderboard_20260622\tables\v221_extreme_peak_worst_models.csv`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v221_formal_model_leaderboard_20260622\tables\v221_per_sample_model_errors.csv`
- 压缩包：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v221_formal_model_leaderboard_20260622\v221_formal_model_leaderboard_pack.zip`
- 验证：ZIP `bad_file=None`，包含 `13` 个文件；formal overall 表不含 `W3_B4_original_soft`、oracle、fallback 或 true-label 行。
- 核心用途：给 v222a 固定候选、基线和停止线；避免继续盲目训练大模型或硬切换 router。

---

## 2026-06-22 v222a closeout candidate gap audit

- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v222a_closeout_candidate_gap_audit_20260622.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_closeout_candidate_gap_audit_20260622`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_closeout_candidate_gap_audit_20260622\reports\v222a_closeout_candidate_gap_audit_cn.md`
- formal headline 决策：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_closeout_candidate_gap_audit_20260622\tables\formal_headline_decision.csv`
- v222a 停止证据：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_closeout_candidate_gap_audit_20260622\tables\v222a_stop_evidence.csv`
- oracle vs learned gap：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_closeout_candidate_gap_audit_20260622\tables\oracle_vs_learned_gap.csv`
- 候选缺口逐样本审计：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_closeout_candidate_gap_audit_20260622\tables\candidate_gap_audit.csv`
- 逐样本 failure taxonomy：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_closeout_candidate_gap_audit_20260622\tables\per_sample_failure_taxonomy.csv`
- bucket failure summary：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_closeout_candidate_gap_audit_20260622\tables\bucket_failure_summary.csv`
- future route decision：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_closeout_candidate_gap_audit_20260622\tables\future_route_decision.csv`
- guardrail 检查：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_closeout_candidate_gap_audit_20260622\tables\leakage_guard_result.csv`
- case 图索引：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_closeout_candidate_gap_audit_20260622\tables\case_figure_index.csv`
- case 图目录：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_closeout_candidate_gap_audit_20260622\figures\top_selector_failed_cases`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_closeout_candidate_gap_audit_20260622\figures\top_candidate_missing_cases`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_closeout_candidate_gap_audit_20260622\figures\top_safe_under_fix_cases`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_closeout_candidate_gap_audit_20260622\figures\top_baseline_sufficient_cases`
- manifest：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_closeout_candidate_gap_audit_20260622\logs\closeout_manifest.json`
- sha256 manifest：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_closeout_candidate_gap_audit_20260622\logs\sha256_manifest.csv`
- ZIP 校验：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_closeout_candidate_gap_audit_20260622\logs\zip_verify.json`
- 压缩包：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_closeout_candidate_gap_audit_20260622\v222a_closeout_candidate_gap_audit_pack.zip`
- 结果摘要：v222a formal 主线停止；formal headline 保持 `loose=avg_joint_focus`、`strict=peak_floor_090`。test 上 selector_failed 约 `0.410615` combined，candidate_missing 约 `0.027933` combined，high-tail candidate_missing 约 `0.126582` combined；因此当前不解锁 v222b/v223。
- 验证摘要：`py_compile` 通过，脚本完整运行通过，ZIP `bad_file=None` 且 `74` 文件，guard 全 pass，禁用名检查无命中，case 图抽检正常。

---

## 2026-05-20 v0.5 服务器处理后样本重筛 + 被试划分旧流程车辆-only

- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage03_v05_server_aligned_subject_oldflow_fair09_user_summary_cn.md`
- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v05_server_aligned_subject_oldflow_fair09.py`
- v0.5 样本筛选表目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\extreme_condition_episodes_server_aligned_v0_5\tables`
- 旧流程 manifest：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_processed_datasets\stage03_v05_server_aligned_subject_oldflow_fair09\tables\oldflow_fair09_vehicle_only_server_aligned_v05_subject_split_manifest.csv`
- 运行记录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_server_aligned_subject_oldflow_fair09\tables\server_aligned_v05_subject_oldflow_fair09_run_record.csv`
- 分被试样本级指标：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_server_aligned_subject_oldflow_fair09\tables\test_subject_sample_metrics_v0_5.csv`
- 分道路类型样本级指标：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_server_aligned_subject_oldflow_fair09\tables\test_road_type_sample_metrics_v0_5.csv`
- 分机制标签样本级指标：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_server_aligned_subject_oldflow_fair09\tables\test_mechanism_sample_metrics_v0_5.csv`
- 本地运行目录：`F:\data_set_process\data_process\tmp\event_conditioned_runs\V05_SERVER_ALIGNED_SUBJECT_FAIR09_vehicle_only_seed2026_20260520_120144`
- 预测总览图：`F:\data_set_process\data_process\tmp\event_conditioned_runs\V05_SERVER_ALIGNED_SUBJECT_FAIR09_vehicle_only_seed2026_20260520_120144\prediction_figures\test\overview.png`
- 服务器训练日志：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\00_project_notes\server_logs\stage03_v05_server_aligned_subject_train_latest.log`
- 结果摘要：loader 保留 1376 个样本，train/val/test=953/260/163；车辆-only test RMSE=0.3386，primary=0.2184，tail=0.3105，selection=0.8206。
- Git 提交：`77f5809 stage03: run server-aligned subject fair09 vehicle baseline`

---

## 2026-05-19 v0.3 样本筛选策略 GPU 快速对比完成

- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage03_v03_screening_sweep_gpu_user_summary_cn.md`
- 汇总表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_screening_sweep_gpu\tables\v03_screening_sweep_gpu_summary.csv`
- 排序表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_screening_sweep_gpu\tables\v03_screening_sweep_gpu_ranking.csv`
- 额外样本来源统计：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_screening_sweep_gpu\tables\v03_screening_sweep_gpu_extra_source_counts.csv`
- 服务器日志：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\00_project_notes\server_logs\stage03_v03_screening_sweep_gpu_20260519_211258.log`
- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v03_screening_sweep_gpu.py`
- 当前结论：`s16_weakpost_lat` 排名第一，但需继续复核横向偏移坐标风险和 16 个新增样本。

---

## 2026-05-19 v0.3 样本筛选策略 GPU 快速对比

- 脚本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v03_screening_sweep_gpu.py`
- CPU 对照脚本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v03_screening_sweep.py`
- 服务器任务：screen `v03gpu`
- 服务器日志：`/root/autodl-tmp/data_process/05_rebuild_from_raw_20260511/00_project_notes/server_logs/stage03_v03_screening_sweep_gpu_20260519_211258.log`
- 预计本地结果目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_screening_sweep_gpu`
- 预计用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage03_v03_screening_sweep_gpu_user_summary_cn.md`
- 状态：运行中，完成后拉回结果并更新本索引。

---

## 2026-05-18 车辆响应锚点前方向盘动作重新筛选 v0.2

- 用户查看版：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage02_vehicle_response_presteer_rescreen_user_summary_cn.md`
- 技术报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\vehicle_response_presteer_rescreen_v0_2_cn.md`
- 总表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\vehicle_response_presteer_rescreen_v0_2\tables\vehicle_response_presteer_candidates_v0_2.csv`
- P1 最干净核心样本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\vehicle_response_presteer_rescreen_v0_2\tables\primary_roll_presteer_events_P1_v0_2.csv`
- P2 最干净次级样本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\vehicle_response_presteer_rescreen_v0_2\tables\secondary_lateral_presteer_events_P2_v0_2.csv`
- 汇总表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\vehicle_response_presteer_rescreen_v0_2\tables\vehicle_response_presteer_summary_v0_2.csv`
- 分场景表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\vehicle_response_presteer_rescreen_v0_2\tables\vehicle_response_presteer_by_module_v0_2.csv`
- 时间差表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\vehicle_response_presteer_rescreen_v0_2\tables\vehicle_response_presteer_latency_quantiles_v0_2.csv`
- 复核图索引：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\vehicle_response_presteer_rescreen_v0_2\tables\vehicle_response_presteer_review_panel_index_v0_2.csv`

## 2026-05-18 车辆响应锚点前方向盘动作重新筛选 v0.2

- 用户查看版：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage02_vehicle_response_presteer_rescreen_user_summary_cn.md`
- 技术报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\vehicle_response_presteer_rescreen_v0_2_cn.md`
- 总表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\vehicle_response_presteer_rescreen_v0_2\tables\vehicle_response_presteer_candidates_v0_2.csv`
- P1 核心样本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\vehicle_response_presteer_rescreen_v0_2\tables\primary_roll_presteer_events_P1_v0_2.csv`
- P2 次级样本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\vehicle_response_presteer_rescreen_v0_2\tables\secondary_lateral_presteer_events_P2_v0_2.csv`
- 汇总表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\vehicle_response_presteer_rescreen_v0_2\tables\vehicle_response_presteer_summary_v0_2.csv`
- 分场景表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\vehicle_response_presteer_rescreen_v0_2\tables\vehicle_response_presteer_by_module_v0_2.csv`
- 时间差表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\vehicle_response_presteer_rescreen_v0_2\tables\vehicle_response_presteer_latency_quantiles_v0_2.csv`
- 复核图索引：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\vehicle_response_presteer_rescreen_v0_2\tables\vehicle_response_presteer_review_panel_index_v0_2.csv`

## 2026-05-14 方向盘动作候选漏斗审计 v0.1

- 报告：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\funnel_audit_v0_1\steering_episode_funnel_audit_v0_1.md`
- 逐记录漏斗表：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\funnel_audit_v0_1\steering_funnel_by_record_v0_1.csv`
- 候选明细表：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\funnel_audit_v0_1\steering_funnel_candidates_v0_1.csv`
- 宽松候选池：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\funnel_audit_v0_1\loose_steering_candidates_v0_1.csv`
- 严格通过表：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\funnel_audit_v0_1\strict_steering_episode_candidates_v0_1.csv`

## 2026-05-14 方向盘动作 episode 样本重建 v0.6

- 汇总报告：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\event_episode_summary_v0_6.md`
- 主 episode 表：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\event_episodes_all_v0_6.csv`
- P1 表：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\primary_positive_episodes_P1_v0_6.csv`
- P2 表：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\secondary_episodes_P2_v0_6.csv`
- C 表：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\context_control_C_v0_6.csv`
- N 表：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\trigger_no_effect_N_v0_6.csv`
- U 表：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\manual_review_U_v0_6.csv`
- X 表：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\excluded_X_v0_6.csv`
- 复核图目录：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\review_figures`
- 复核图索引：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\review_figure_index_v0_6.csv`
- 日志：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\build_event_episodes_v0_6.log`

## 2026-05-14 方向盘动作 episode 样本重建 v0.6

- 汇总报告：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\event_episode_summary_v0_6.md`
- 主 episode 表：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\event_episodes_all_v0_6.csv`
- P1 表：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\primary_positive_episodes_P1_v0_6.csv`
- P2 表：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\secondary_episodes_P2_v0_6.csv`
- C 表：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\context_control_C_v0_6.csv`
- N 表：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\trigger_no_effect_N_v0_6.csv`
- U 表：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\manual_review_U_v0_6.csv`
- X 表：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\excluded_X_v0_6.csv`
- 复核图目录：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\review_figures`
- 复核图索引：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\review_figure_index_v0_6.csv`
- 日志：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\build_event_episodes_v0_6.log`

## 2026-05-14 方向盘动作 episode 样本重建 v0.6

- 汇总报告：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\event_episode_summary_v0_6.md`
- 主 episode 表：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\event_episodes_all_v0_6.csv`
- P1 表：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\primary_positive_episodes_P1_v0_6.csv`
- P2 表：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\secondary_episodes_P2_v0_6.csv`
- C 表：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\context_control_C_v0_6.csv`
- N 表：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\trigger_no_effect_N_v0_6.csv`
- U 表：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\manual_review_U_v0_6.csv`
- X 表：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\excluded_X_v0_6.csv`
- 复核图目录：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\review_figures`
- 复核图索引：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\review_figure_index_v0_6.csv`
- 日志：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\build_event_episodes_v0_6.log`

## 2026-05-14 方向盘到车辆动态时间差审计

- 用户查看版：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage02_steer_to_vehicle_dynamics_latency_user_summary_cn.md`
- 技术报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\steer_to_vehicle_dynamics_latency_v0_1_cn.md`
- 明细表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\steer_to_vehicle_dynamics_latency_v0_1\tables\steer_to_dynamics_latency_events_v0_1.csv`
- 汇总表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\steer_to_vehicle_dynamics_latency_v0_1\tables\steer_to_dynamics_latency_summary_v0_1.csv`
- 分组表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\steer_to_vehicle_dynamics_latency_v0_1\tables\steer_to_dynamics_latency_by_bucket_module_v0_1.csv`
- 分位数表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\steer_to_vehicle_dynamics_latency_v0_1\tables\steer_to_dynamics_latency_quantiles_v0_1.csv`
- 代表性复核图索引：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\steer_to_vehicle_dynamics_latency_v0_1\tables\latency_review_panel_index_v0_1.csv`
- 代表性复核图数量：24

图表：
- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\steer_to_vehicle_dynamics_latency_v0_1\figures\steer_to_dynamics_latency_histogram_v0_1.png`
- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\steer_to_vehicle_dynamics_latency_v0_1\figures\steer_to_dynamics_latency_by_bucket_v0_1.png`
- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\steer_to_vehicle_dynamics_latency_v0_1\figures\steer_to_dynamics_latency_by_module_box_v0_1.png`

## 2026-05-14 方向盘到车辆动态时间差审计

- 用户查看版：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage02_steer_to_vehicle_dynamics_latency_user_summary_cn.md`
- 技术报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\steer_to_vehicle_dynamics_latency_v0_1_cn.md`
- 明细表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\steer_to_vehicle_dynamics_latency_v0_1\tables\steer_to_dynamics_latency_events_v0_1.csv`
- 汇总表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\steer_to_vehicle_dynamics_latency_v0_1\tables\steer_to_dynamics_latency_summary_v0_1.csv`
- 分组表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\steer_to_vehicle_dynamics_latency_v0_1\tables\steer_to_dynamics_latency_by_bucket_module_v0_1.csv`
- 分位数表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\steer_to_vehicle_dynamics_latency_v0_1\tables\steer_to_dynamics_latency_quantiles_v0_1.csv`
- 代表性复核图索引：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\steer_to_vehicle_dynamics_latency_v0_1\tables\latency_review_panel_index_v0_1.csv`
- 代表性复核图数量：24

图表：
- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\steer_to_vehicle_dynamics_latency_v0_1\figures\steer_to_dynamics_latency_histogram_v0_1.png`
- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\steer_to_vehicle_dynamics_latency_v0_1\figures\steer_to_dynamics_latency_by_bucket_v0_1.png`
- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\steer_to_vehicle_dynamics_latency_v0_1\figures\steer_to_dynamics_latency_by_module_box_v0_1.png`

## 2026-05-14 方向盘到车辆动态时间差审计

- 用户查看版：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage02_steer_to_vehicle_dynamics_latency_user_summary_cn.md`
- 技术报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\steer_to_vehicle_dynamics_latency_v0_1_cn.md`
- 明细表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\steer_to_vehicle_dynamics_latency_v0_1\tables\steer_to_dynamics_latency_events_v0_1.csv`
- 汇总表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\steer_to_vehicle_dynamics_latency_v0_1\tables\steer_to_dynamics_latency_summary_v0_1.csv`
- 分组表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\steer_to_vehicle_dynamics_latency_v0_1\tables\steer_to_dynamics_latency_by_bucket_module_v0_1.csv`
- 分位数表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\steer_to_vehicle_dynamics_latency_v0_1\tables\steer_to_dynamics_latency_quantiles_v0_1.csv`
- 代表性复核图索引：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\steer_to_vehicle_dynamics_latency_v0_1\tables\latency_review_panel_index_v0_1.csv`
- 代表性复核图数量：24

图表：
- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\steer_to_vehicle_dynamics_latency_v0_1\figures\steer_to_dynamics_latency_histogram_v0_1.png`
- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\steer_to_vehicle_dynamics_latency_v0_1\figures\steer_to_dynamics_latency_by_bucket_v0_1.png`
- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\steer_to_vehicle_dynamics_latency_v0_1\figures\steer_to_dynamics_latency_by_module_box_v0_1.png`

# 阶段产物索引

## 最新更新：2026-05-13 08:09

## Stage 7j session 多折稳定性验证 v0.1

- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/scripts/stage07j_session_cv_stability_v0_1.py`
- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage07j_session_cv_stability_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage07j_session_cv_stability_v0_1_cn.md`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07j_session_cv_stability_v0_1/logs/stage07j_session_cv_stability_summary.json`
- session CV split 表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07j_session_cv_stability_v0_1/tables/stage07j_session_cv_split_table.csv`
- 候选指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07j_session_cv_stability_v0_1/tables/stage07j_candidate_metrics.csv`
- 逐样本指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07j_session_cv_stability_v0_1/tables/stage07j_candidate_per_sample_metrics.csv`
- 候选分数表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07j_session_cv_stability_v0_1/tables/stage07j_candidate_score_table.csv`
- policy fold 指标：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07j_session_cv_stability_v0_1/tables/stage07j_policy_fold_metrics.csv`
- policy 汇总：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07j_session_cv_stability_v0_1/tables/stage07j_policy_aggregate.csv`
- 原始 val gate 选择表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07j_session_cv_stability_v0_1/tables/stage07j_original_val_gate_selection_table.csv`
- feature audit：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07j_session_cv_stability_v0_1/tables/stage07j_feature_audit.csv`
- allowed features：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07j_session_cv_stability_v0_1/tables/stage07j_allowed_features.csv`
- fold RBF fit info：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07j_session_cv_stability_v0_1/tables/stage07j_fold_rbf_fit_info.csv`
- gate 表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07j_session_cv_stability_v0_1/tables/stage07j_gate_table.csv`
- policy fold delta 图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07j_session_cv_stability_v0_1/figures/stage07j_policy_fold_deltas.png`
- 选中模型计数图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07j_session_cv_stability_v0_1/figures/stage07j_selected_model_counts.png`
- val/test delta 散点图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07j_session_cv_stability_v0_1/figures/stage07j_candidate_val_test_delta_scatter.png`
- GPTPro 提问和回复：本阶段未调用。
- 服务器日志：本阶段未使用服务器，未读取服务器指令与密码文件。
- 重要 Git commit：`11296297 Add stage7j session cv stability audit`。
- 适合用户/老师直接查看：优先看用户查看版总结、policy 汇总、gate 表和 fold delta 图。

## Stage 7i 稳定性校准候选选择 v0.1

- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/scripts/stage07i_stability_calibrated_selection_v0_1.py`
- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage07i_stability_calibrated_selection_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage07i_stability_calibrated_selection_v0_1_cn.md`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07i_stability_calibrated_selection_v0_1/logs/stage07i_stability_calibrated_selection_summary.json`
- 候选分数表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07i_stability_calibrated_selection_v0_1/tables/stage07i_candidate_score_table.csv`
- policy split 指标：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07i_stability_calibrated_selection_v0_1/tables/stage07i_policy_split_metrics.csv`
- policy test 汇总：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07i_stability_calibrated_selection_v0_1/tables/stage07i_policy_test_summary.csv`
- 逐样本收益表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07i_stability_calibrated_selection_v0_1/tables/stage07i_selected_policy_gain_samples.csv`
- gate 表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07i_stability_calibrated_selection_v0_1/tables/stage07i_gate_table.csv`
- policy 对照图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07i_stability_calibrated_selection_v0_1/figures/stage07i_policy_summary.png`
- 稳定性分数组成图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07i_stability_calibrated_selection_v0_1/figures/stage07i_stability_score_components.png`
- 逐样本收益分布图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07i_stability_calibrated_selection_v0_1/figures/stage07i_selected_gain_distribution.png`
- GPTPro 提问和回复：本阶段未调用。
- 服务器日志：本阶段未使用服务器，未读取服务器指令与密码文件。
- 重要 Git commit：`d294a520 Add stage7i stability calibrated selection`。
- 适合用户/老师直接查看：优先看用户查看版总结、gate 表、policy test 汇总和 policy 对照图。

## 最新新增：Stage 7b 非 oracle top-K selector 轻量实验 v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage07b_non_oracle_topk_selector_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage07b_non_oracle_topk_selector_v0_1_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/scripts/stage07b_non_oracle_topk_selector_v0_1.py`
- feature audit：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07b_non_oracle_topk_selector_v0_1/tables/stage07b_feature_audit.csv`
- allowed features：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07b_non_oracle_topk_selector_v0_1/tables/stage07b_allowed_features.csv`
- model info：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07b_non_oracle_topk_selector_v0_1/tables/stage07b_model_info.csv`
- selector decisions：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07b_non_oracle_topk_selector_v0_1/tables/stage07b_selector_decisions.csv`
- all policy metrics：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07b_non_oracle_topk_selector_v0_1/tables/stage07b_all_policy_metrics.csv`
- selected policy metrics：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07b_non_oracle_topk_selector_v0_1/tables/stage07b_selected_policy_metrics.csv`
- decision diagnostics：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07b_non_oracle_topk_selector_v0_1/tables/stage07b_decision_diagnostics.csv`
- coverage-risk：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07b_non_oracle_topk_selector_v0_1/tables/stage07b_coverage_risk.csv`
- gate 表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07b_non_oracle_topk_selector_v0_1/tables/stage07b_gate_table.csv`
- RMSE 图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07b_non_oracle_topk_selector_v0_1/figures/stage07b_selector_test_rmse.png`
- choice accuracy 图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07b_non_oracle_topk_selector_v0_1/figures/stage07b_selector_choice_accuracy.png`
- coverage-risk 图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07b_non_oracle_topk_selector_v0_1/figures/stage07b_coverage_risk.png`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07b_non_oracle_topk_selector_v0_1/logs/stage07b_selector_summary.json`
- 服务器日志：无，本轮未使用远程服务器。
- GPTPro 提问和回复：无。
- 重要 Git commit：`d431cd11 Add stage7b non-oracle topk selector`。
- 适合用户/老师直接查看：用户查看版总结、gate 表、selected policy metrics、feature audit。

## 最新新增：Stage 7a 非 oracle 多候选选择协议 v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage07a_non_oracle_selection_protocol_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage07a_non_oracle_selection_protocol_v0_1_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/scripts/stage07a_non_oracle_selection_protocol_v0_1.py`
- 候选池 manifest：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07a_non_oracle_selection_protocol_v0_1/tables/stage07a_candidate_pool_manifest.csv`
- 特征守卫表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07a_non_oracle_selection_protocol_v0_1/tables/stage07a_feature_guard_table.csv`
- 选择流程表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07a_non_oracle_selection_protocol_v0_1/tables/stage07a_selection_protocol.csv`
- 评价计划表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07a_non_oracle_selection_protocol_v0_1/tables/stage07a_evaluation_plan.csv`
- 固定图协议：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07a_non_oracle_selection_protocol_v0_1/tables/stage07a_fixed_plot_protocol.csv`
- gate 表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07a_non_oracle_selection_protocol_v0_1/tables/stage07a_gate_table.csv`
- 候选池 RMSE 图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07a_non_oracle_selection_protocol_v0_1/figures/stage07a_candidate_pool_rmse.png`
- gate 状态图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07a_non_oracle_selection_protocol_v0_1/figures/stage07a_protocol_gate_status.png`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07a_non_oracle_selection_protocol_v0_1/logs/stage07a_protocol_summary.json`
- 服务器日志：无，本轮未使用远程服务器。
- GPTPro 提问和回复：无。
- 重要 Git commit：`dfacb38d Add stage7a non-oracle selection protocol`。
- 适合用户/老师直接查看：用户查看版总结、特征守卫表、选择流程表、gate 表。

## 最新新增：Stage 6e 多候选 oracle gap 复核 v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage06e_multicandidate_oracle_gap_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage06e_multicandidate_oracle_gap_v0_1_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/scripts/stage06e_multicandidate_oracle_gap_v0_1.py`
- source manifest：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06e_multicandidate_oracle_gap_v0_1/tables/multicandidate_source_manifest.csv`
- 候选可用性表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06e_multicandidate_oracle_gap_v0_1/tables/multicandidate_model_availability.csv`
- 全指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06e_multicandidate_oracle_gap_v0_1/tables/multicandidate_all_metrics.csv`
- oracle gap 表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06e_multicandidate_oracle_gap_v0_1/tables/multicandidate_oracle_gap_table.csv`
- oracle winner 明细：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06e_multicandidate_oracle_gap_v0_1/tables/multicandidate_oracle_winner_detail.csv`
- oracle winner 汇总：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06e_multicandidate_oracle_gap_v0_1/tables/multicandidate_oracle_winner_summary.csv`
- oracle gain 样本明细：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06e_multicandidate_oracle_gap_v0_1/tables/multicandidate_oracle_gain_sample_detail.csv`
- gate 表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06e_multicandidate_oracle_gap_v0_1/tables/multicandidate_oracle_gap_gate_table.csv`
- RMSE gap 图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06e_multicandidate_oracle_gap_v0_1/figures/multicandidate_oracle_gap_rmse.png`
- winner count 图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06e_multicandidate_oracle_gap_v0_1/figures/multicandidate_oracle_winner_counts.png`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06e_multicandidate_oracle_gap_v0_1/logs/multicandidate_oracle_gap_summary.json`
- 服务器日志：无，本轮未使用远程服务器。
- GPTPro 提问和回复：无。
- 重要 Git commit：`cb4d8eec Add stage6e multicandidate oracle gap audit`。
- 适合用户/老师直接查看：用户查看版总结、oracle gap 表、winner 汇总、RMSE gap 图。

## 最新新增：Stage 6d RBF/KNN reliability gate v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage06d_reliability_gate_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage06d_reliability_gate_v0_1_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/scripts/stage06d_reliability_gate_v0_1.py`
- 全阈值指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06d_reliability_gate_v0_1/tables/reliability_gate_all_threshold_metrics.csv`
- 选中 policy 表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06d_reliability_gate_v0_1/tables/reliability_gate_selected_policies.csv`
- policy 指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06d_reliability_gate_v0_1/tables/reliability_gate_policy_metrics.csv`
- gate 表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06d_reliability_gate_v0_1/tables/reliability_gate_gate_table.csv`
- best confusion 表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06d_reliability_gate_v0_1/tables/reliability_gate_best_confusion.csv`
- RMSE 图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06d_reliability_gate_v0_1/figures/reliability_gate_test_rmse.png`
- 物理指标图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06d_reliability_gate_v0_1/figures/reliability_gate_physical_metrics.png`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06d_reliability_gate_v0_1/logs/reliability_gate_summary.json`
- 服务器日志：无，本轮未使用远程服务器。
- GPTPro 提问和回复：无。
- 重要 Git commit：`4264db88 Add stage6d reliability gate`。
- 适合用户/老师直接查看：用户查看版总结、gate 表、policy 指标表、物理指标图。

## 最新新增：Stage 6c selector feature revision v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage06c_selector_feature_revision_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage06c_selector_feature_revision_v0_1_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/scripts/stage06c_selector_feature_revision_v0_1.py`
- 特征协议表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06c_selector_feature_revision_v0_1/tables/selector_revision_feature_manifest.csv`
- 指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06c_selector_feature_revision_v0_1/tables/selector_revision_metrics.csv`
- 阈值扫描表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06c_selector_feature_revision_v0_1/tables/selector_revision_threshold_sweep.csv`
- 候选明细表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06c_selector_feature_revision_v0_1/tables/selector_revision_candidate_details.csv`
- 最佳 selector 明细：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06c_selector_feature_revision_v0_1/tables/selector_revision_best_detail.csv`
- 最佳 selector 混淆表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06c_selector_feature_revision_v0_1/tables/selector_revision_best_confusion.csv`
- gate 表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06c_selector_feature_revision_v0_1/tables/selector_revision_gate_table.csv`
- RMSE 图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06c_selector_feature_revision_v0_1/figures/selector_revision_test_rmse.png`
- 物理指标图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06c_selector_feature_revision_v0_1/figures/selector_revision_physical_metrics.png`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06c_selector_feature_revision_v0_1/logs/selector_feature_revision_summary.json`
- 服务器日志：无，本轮未使用远程服务器。
- GPTPro 提问和回复：无。
- 重要 Git commit：`eae76c2f Add stage6c selector feature revision`。
- 适合用户/老师直接查看：用户查看版总结、gate 表、指标表、物理指标图。

## 最新新增：Stage 6b RBF/keypoint 选择器错误复盘 v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage06b_keypoint_selector_error_review_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage06b_keypoint_selector_error_review_v0_1_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/scripts/stage06b_keypoint_selector_error_review_v0_1.py`
- 选择器样本明细：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06b_keypoint_selector_error_review_v0_1/tables/keypoint_selector_sample_detail.csv`
- 混淆表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06b_keypoint_selector_error_review_v0_1/tables/keypoint_selector_confusion_table.csv`
- 分组摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06b_keypoint_selector_error_review_v0_1/tables/keypoint_selector_group_summary.csv`
- top regret 样本：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06b_keypoint_selector_error_review_v0_1/tables/keypoint_selector_top_regret_samples.csv`
- 漏选 keypoint 收益样本：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06b_keypoint_selector_error_review_v0_1/tables/keypoint_selector_missed_keypoint_gain_samples.csv`
- 错选 keypoint 样本：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06b_keypoint_selector_error_review_v0_1/tables/keypoint_selector_false_keypoint_samples.csv`
- 下一步动作表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06b_keypoint_selector_error_review_v0_1/tables/keypoint_selector_next_actions.csv`
- 混淆矩阵图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06b_keypoint_selector_error_review_v0_1/figures/keypoint_selector_confusion_matrix.png`
- top regret 图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06b_keypoint_selector_error_review_v0_1/figures/keypoint_selector_top_regret_samples.png`
- probability vs gain 图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06b_keypoint_selector_error_review_v0_1/figures/keypoint_selector_probability_vs_gain.png`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06b_keypoint_selector_error_review_v0_1/logs/keypoint_selector_error_review_summary.json`
- 服务器日志：无，本轮未使用远程服务器。
- GPTPro 提问和回复：无。
- 重要 Git commit：`753525fd Add stage6b keypoint selector error review`。
- 适合用户/老师直接查看：用户查看版总结、混淆表、top regret 样本、probability vs gain 图。

## 最新新增：阶段 6 车辆-only 结构化路线审计 v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage06_vehicle_only_structured_route_audit_user_summary_cn.md`
- 阶段通用用户总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage06_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage06_vehicle_only_structured_route_audit_v0_1_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/scripts/stage06_vehicle_only_structured_route_audit_v0_1.py`
- 车辆-only候选 scorecard：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06_vehicle_only_structured_route_audit_v0_1/tables/vehicle_structured_candidate_scorecard.csv`
- 相对 RBF delta 表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06_vehicle_only_structured_route_audit_v0_1/tables/vehicle_structured_metric_delta_vs_rbf.csv`
- gate 表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06_vehicle_only_structured_route_audit_v0_1/tables/vehicle_structured_route_gate_table.csv`
- 下一步动作表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06_vehicle_only_structured_route_audit_v0_1/tables/vehicle_structured_next_actions.csv`
- RMSE 汇总图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06_vehicle_only_structured_route_audit_v0_1/figures/vehicle_structured_route_rmse_summary.png`
- 相对 RBF delta 图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06_vehicle_only_structured_route_audit_v0_1/figures/vehicle_structured_route_delta_vs_rbf.png`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06_vehicle_only_structured_route_audit_v0_1/logs/vehicle_structured_route_audit_summary.json`
- 服务器日志：无，本轮未使用远程服务器。
- GPTPro 提问和回复：无。
- 重要 Git commit：`b4d7ac20 Add stage6 vehicle structured route audit`。
- 适合用户/老师直接查看：用户查看版总结、gate 表、scorecard、相对 RBF delta 图。

## 最新新增：阶段 4 连续风格协议与候选特征 v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage04_user_summary_cn.md`
- 同内容阶段 4 协议总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage04_continuous_style_protocol_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage04_continuous_style_protocol_v0_1_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/scripts/stage04_continuous_style_protocol_v0_1.py`
- 候选风格 long 表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_continuous_style_protocol_v0_1/tables/style_feature_candidate_long.csv`
- 候选风格 wide 表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_continuous_style_protocol_v0_1/tables/style_feature_candidate_wide.csv`
- train-only 标准化后 wide 表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_continuous_style_protocol_v0_1/tables/style_feature_candidate_wide_trainz_session_split.csv`
- train-only 标准化参数：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_continuous_style_protocol_v0_1/tables/style_train_only_scaler_session_split.csv`
- 来源协议表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_continuous_style_protocol_v0_1/tables/style_source_protocol_table.csv`
- 泄漏边界表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_continuous_style_protocol_v0_1/tables/style_leakage_guard_table.csv`
- 置乱对照计划：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_continuous_style_protocol_v0_1/tables/style_permutation_plan.csv`
- split 可行性表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_continuous_style_protocol_v0_1/tables/style_split_feasibility.csv`
- 被试/道路耦合审计：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_continuous_style_protocol_v0_1/tables/style_subject_road_coupling_audit.csv`
- 阶段 4 gate 表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_continuous_style_protocol_v0_1/tables/style_protocol_gate_table.csv`
- 风格窗口可用性图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_continuous_style_protocol_v0_1/figures/style_feature_availability_by_window.png`
- split-道路分布图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_continuous_style_protocol_v0_1/figures/style_split_road_distribution_heatmap.png`
- 被试-道路耦合热图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_continuous_style_protocol_v0_1/figures/style_subject_road_coupling_heatmap.png`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_continuous_style_protocol_v0_1/logs/run_summary.json`
- 服务器日志：无，本轮未使用远程服务器。
- GPTPro 提问和回复：无。
- 重要 Git commit：`012f4803 Add continuous style protocol audit`。
- 适合用户/老师直接查看：用户查看版总结、阶段 4 gate 表、泄漏边界表、风格窗口可用性图、被试-道路耦合热图。

## 最新新增：阶段 3 RBF 主参照冻结审计 v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_rbf_reference_freeze_audit_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_rbf_reference_freeze_audit_v0_1_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_vehicle_instability_rbf_reference_freeze_audit_v0_1.py`
- RBF 指标画像：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_reference_freeze_audit_v0_1/tables/rbf_reference_metric_profile.csv`
- RBF 失败画像：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_reference_freeze_audit_v0_1/tables/rbf_reference_failure_profile.csv`
- RBF top bad 样本：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_reference_freeze_audit_v0_1/tables/rbf_reference_top_bad_samples.csv`
- 冻结 gate 表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_reference_freeze_audit_v0_1/tables/rbf_reference_freeze_gate_table.csv`
- 稳健性快照：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_reference_freeze_audit_v0_1/tables/rbf_reference_robustness_snapshot.csv`
- 失败画像图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_reference_freeze_audit_v0_1/figures/rbf_reference_failure_profile.png`
- 关键指标图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_reference_freeze_audit_v0_1/figures/rbf_reference_key_metrics.png`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_reference_freeze_audit_v0_1/logs/rbf_reference_freeze_audit_summary.json`
- 服务器日志：无，本轮未使用远程服务器。
- GPTPro 提问和回复：无。
- 重要 Git commit：`112824f7 Add rbf reference freeze audit`。
- 适合用户/老师直接查看：用户查看版总结、冻结 gate 表、失败画像图、关键指标图、RBF top bad 样本。

## 最新新增：阶段 3 车辆-only 主参照决策表 v0.2

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_vehicle_only_decision_table_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_vehicle_only_decision_table_v0_2_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_vehicle_instability_vehicle_only_decision_table_v0_2.py`
- 候选决策表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_vehicle_only_decision_table_v0_2/tables/vehicle_only_candidate_decision_table_v0_2.csv`
- gate 状态表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_vehicle_only_decision_table_v0_2/tables/vehicle_only_stage3_gate_status_v0_2.csv`
- 角色汇总表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_vehicle_only_decision_table_v0_2/tables/vehicle_only_decision_role_summary_v0_2.csv`
- 阶段 3 指标库存：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_vehicle_only_decision_table_v0_2/tables/vehicle_only_stage3_metric_inventory_v0_2.csv`
- 指标概览图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_vehicle_only_decision_table_v0_2/figures/vehicle_only_decision_key_metrics_test.png`
- RMSE vs 错侧图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_vehicle_only_decision_table_v0_2/figures/vehicle_only_decision_rmse_vs_wrong_side_test.png`
- 角色计数图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_vehicle_only_decision_table_v0_2/figures/vehicle_only_decision_role_counts.png`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_vehicle_only_decision_table_v0_2/logs/vehicle_only_decision_table_summary.json`
- 服务器日志：无，本轮未使用远程服务器。
- GPTPro 提问和回复：无。
- 重要 Git commit：`e04bdb2f Add vehicle-only decision table`。
- 适合用户/老师直接查看：用户查看版总结、gate 状态表、候选决策表、指标概览图、RMSE vs 错侧图。

## 最新新增：阶段 3 top-K 可靠性选择/回退 v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_topk_reliability_selector_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_topk_reliability_selector_v0_1_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_vehicle_instability_topk_reliability_selector_v0_1.py`
- 指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_reliability_selector_v0_1/tables/topk_reliability_selector_metrics.csv`
- 逐样本指标：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_reliability_selector_v0_1/tables/topk_reliability_selector_per_sample_metrics.csv`
- 选择器特征表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_reliability_selector_v0_1/tables/topk_reliability_selector_feature_table.csv`
- 决策表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_reliability_selector_v0_1/tables/topk_reliability_selector_decisions.csv`
- validation 选择表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_reliability_selector_v0_1/tables/topk_reliability_selector_validation_selection.csv`
- 阈值扫描表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_reliability_selector_v0_1/tables/topk_reliability_selector_threshold_sweep.csv`
- 分被试汇总：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_reliability_selector_v0_1/tables/topk_reliability_selector_subject_summary.csv`
- 分道路模块汇总：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_reliability_selector_v0_1/tables/topk_reliability_selector_road_module_summary.csv`
- 固定预测图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_reliability_selector_v0_1/figures/topk_reliability_selector_fixed_predictions_test.png`
- 坏样本图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_reliability_selector_v0_1/figures/topk_reliability_selector_bad_samples_test.png`
- oracle 增益样本图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_reliability_selector_v0_1/figures/topk_reliability_selector_oracle_gain_samples_test.png`
- 指标概览图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_reliability_selector_v0_1/figures/topk_reliability_selector_metric_summary_test.png`
- 决策计数图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_reliability_selector_v0_1/figures/topk_reliability_selector_decision_counts_test.png`
- fallback 风险散点图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_reliability_selector_v0_1/figures/topk_reliability_selector_fallback_scatter_test.png`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_reliability_selector_v0_1/logs/topk_reliability_selector_summary.json`
- 服务器日志：无，本轮未使用远程服务器。
- GPTPro 提问和回复：无。
- 重要 Git commit：`fbb8d94d Add topk reliability selector`。
- 适合用户/老师直接查看：用户查看版总结、指标概览图、固定预测图、坏样本图、决策计数图、fallback 风险散点图。

## 最新新增：阶段 3 top-K top1/bestK 差距复盘 v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_topk_gap_review_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_topk_gap_review_v0_1_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_vehicle_instability_topk_gap_review_v0_1.py`
- 样本详情：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_gap_review_v0_1/tables/topk_gap_review_sample_detail.csv`
- 总体摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_gap_review_v0_1/tables/topk_gap_review_overall_summary.csv`
- train 阈值：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_gap_review_v0_1/tables/topk_gap_review_thresholds.csv`
- 相关性表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_gap_review_v0_1/tables/topk_gap_review_correlations.csv`
- 分桶汇总：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_gap_review_v0_1/tables/topk_gap_review_bucket_summary.csv`
- top gap 样本：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_gap_review_v0_1/tables/topk_gap_review_top_gap_samples.csv`
- top1 比 RBF 更差样本：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_gap_review_v0_1/tables/topk_gap_review_top1_worse_than_rbf_samples.csv`
- top gap 图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_gap_review_v0_1/figures/topk_gap_top_samples.png`
- 风险散点图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_gap_review_v0_1/figures/topk_gap_risk_scatter.png`
- 分支混淆图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_gap_review_v0_1/figures/topk_gap_branch_confusion.png`
- 物理错误图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_gap_review_v0_1/figures/topk_gap_error_flags.png`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_gap_review_v0_1/logs/topk_gap_review_summary.json`
- 服务器日志：无，本轮未使用远程服务器。
- GPTPro 提问和回复：无。
- 重要 Git commit：`1ace03f2 Add topk gap review`。
- 适合用户/老师直接查看：用户查看版总结、总体摘要、top gap 样本、top gap 图、风险散点图、分支混淆图。

## 最新新增：阶段 3 top-K 车辆-only Transformer v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_topk_vehicle_transformer_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_topk_vehicle_transformer_v0_1_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_vehicle_instability_topk_vehicle_transformer_v0_1.py`
- 指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_vehicle_transformer_v0_1/tables/topk_vehicle_transformer_metrics.csv`
- 逐样本指标：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_vehicle_transformer_v0_1/tables/topk_vehicle_transformer_per_sample_metrics.csv`
- 分支诊断：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_vehicle_transformer_v0_1/tables/topk_vehicle_transformer_branch_diagnostics.csv`
- 可靠性分箱：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_vehicle_transformer_v0_1/tables/topk_vehicle_transformer_reliability_bins.csv`
- 训练历史：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_vehicle_transformer_v0_1/tables/topk_vehicle_transformer_training_history.csv`
- 与参照对照表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_vehicle_transformer_v0_1/tables/topk_vehicle_transformer_comparison_with_references.csv`
- 模型信息：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_vehicle_transformer_v0_1/tables/topk_vehicle_transformer_model_info.csv`
- 固定预测图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_vehicle_transformer_v0_1/figures/topk_fixed_predictions_test.png`
- 坏样本图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_vehicle_transformer_v0_1/figures/topk_bad_samples_test.png`
- top1/bestK 差距图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_vehicle_transformer_v0_1/figures/topk_top1_bestk_gap_samples_test.png`
- 指标概览图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_vehicle_transformer_v0_1/figures/topk_metric_summary_test.png`
- 可靠性散点图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_vehicle_transformer_v0_1/figures/topk_reliability_scatter_test.png`
- checkpoint：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_vehicle_transformer_v0_1/checkpoints/B_response3s_strict_core_topk_vehicle_transformer_top1_no_subject_best.pt`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_vehicle_transformer_v0_1/logs/topk_vehicle_transformer_summary.json`
- 服务器日志：无，本轮未使用远程服务器。
- GPTPro 提问和回复：无。
- 重要 Git commit：`03165475 Add topk vehicle transformer`。
- 适合用户/老师直接查看：用户查看版总结、指标表、固定预测图、top1/bestK 差距图、可靠性散点图。

## 最新新增：阶段 3 RBF/keypoint 多候选车辆-only 复盘 v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_rbf_keypoint_multihypothesis_review_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_rbf_keypoint_multihypothesis_review_v0_1_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_vehicle_instability_rbf_keypoint_multihypothesis_review_v0_1.py`
- 统一指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_multihypothesis_review_v0_1/tables/multihypothesis_metrics.csv`
- 逐样本指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_multihypothesis_review_v0_1/tables/multihypothesis_per_sample_metrics.csv`
- 选择摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_multihypothesis_review_v0_1/tables/choice_summary.csv`
- 选择详情：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_multihypothesis_review_v0_1/tables/choice_detail.csv`
- test 误选样本：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_multihypothesis_review_v0_1/tables/test_misselected_samples.csv`
- test oracle 增益样本：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_multihypothesis_review_v0_1/tables/test_oracle_gap_samples.csv`
- 固定预测图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_multihypothesis_review_v0_1/figures/multihypothesis_fixed_predictions_test.png`
- selector 坏样本图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_multihypothesis_review_v0_1/figures/multihypothesis_selector_bad_samples_test.png`
- oracle 增益样本图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_multihypothesis_review_v0_1/figures/multihypothesis_oracle_gap_samples_test.png`
- 选择混淆图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_multihypothesis_review_v0_1/figures/selector_choice_confusion_test.png`
- oracle 增益柱图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_multihypothesis_review_v0_1/figures/oracle_gap_top_samples.png`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_multihypothesis_review_v0_1/logs/multihypothesis_review_summary.json`
- 服务器日志：无，本轮未使用远程服务器。
- GPTPro 提问和回复：无。
- 重要 Git commit：`01033e3e Add rbf keypoint multihypothesis review`。
- 适合用户/老师直接查看：用户查看版总结、统一指标表、选择摘要、固定预测图、oracle 增益样本图。

## 最新新增：阶段 3 RBF vs keypoint train/val 选择器 v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_rbf_keypoint_selector_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_rbf_keypoint_selector_v0_1_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_vehicle_instability_rbf_keypoint_selector_v0_1.py`
- selector 训练表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_selector_v0_1/tables/rbf_keypoint_selector_training_table.csv`
- selector 决策表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_selector_v0_1/tables/rbf_keypoint_selector_decisions.csv`
- selector 统一指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_selector_v0_1/tables/rbf_keypoint_selector_metrics.csv`
- 阈值扫描表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_selector_v0_1/tables/rbf_keypoint_selector_threshold_sweep.csv`
- 选择后逐样本指标：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_selector_v0_1/tables/rbf_keypoint_selector_selected_per_sample_metrics.csv`
- 数值特征表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_selector_v0_1/tables/rbf_keypoint_selector_numeric_features.csv`
- 类别特征表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_selector_v0_1/tables/rbf_keypoint_selector_categorical_features.csv`
- test 指标图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_selector_v0_1/figures/rbf_keypoint_selector_test_metrics.png`
- 阈值扫描图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_selector_v0_1/figures/rbf_keypoint_selector_threshold_sweep.png`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_selector_v0_1/logs/rbf_keypoint_selector_summary.json`
- 服务器日志：无，本轮未使用远程服务器。
- GPTPro 提问和回复：无。
- 重要 Git commit：`7e3d53f6 Add rbf keypoint selector`。
- 适合用户/老师直接查看：用户查看版总结、selector 统一指标表、selector 决策表、test 指标图、阈值扫描图。

## 最新新增：阶段 3 keypoint+residual vs RBF 坏样本差异复盘 v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_keypoint_vs_rbf_bad_sample_review_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_keypoint_vs_rbf_bad_sample_review_v0_1_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_vehicle_instability_keypoint_vs_rbf_bad_sample_review_v0_1.py`
- 样本级差异表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_keypoint_vs_rbf_bad_sample_review_v0_1/tables/keypoint_vs_rbf_sample_delta.csv`
- 错误变化计数表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_keypoint_vs_rbf_bad_sample_review_v0_1/tables/keypoint_vs_rbf_change_counts.csv`
- 总体摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_keypoint_vs_rbf_bad_sample_review_v0_1/tables/keypoint_vs_rbf_overall_summary.csv`
- 分被试摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_keypoint_vs_rbf_bad_sample_review_v0_1/tables/keypoint_vs_rbf_subject_summary.csv`
- Top 改善样本：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_keypoint_vs_rbf_bad_sample_review_v0_1/tables/keypoint_vs_rbf_top_improved.csv`
- Top 退化样本：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_keypoint_vs_rbf_bad_sample_review_v0_1/tables/keypoint_vs_rbf_top_degraded.csv`
- RMSE 差异图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_keypoint_vs_rbf_bad_sample_review_v0_1/figures/keypoint_vs_rbf_rmse_delta_top_samples.png`
- 错误变化计数图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_keypoint_vs_rbf_bad_sample_review_v0_1/figures/keypoint_vs_rbf_error_change_counts.png`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_keypoint_vs_rbf_bad_sample_review_v0_1/logs/keypoint_vs_rbf_bad_sample_review_summary.json`
- 服务器日志：无，本轮未使用远程服务器。
- GPTPro 提问和回复：无。
- 重要 Git commit：待提交。
- 适合用户/老师直接查看：用户查看版总结、样本级差异表、RMSE 差异图、错误变化计数图。

## 最新新增：阶段 3 B 轨道车辆-only 关键点 + 残差 Transformer v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_keypoint_residual_vehicle_transformer_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_keypoint_residual_vehicle_transformer_v0_1_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_vehicle_instability_keypoint_residual_vehicle_transformer_v0_1.py`
- 指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_keypoint_residual_vehicle_transformer_v0_1/tables/keypoint_residual_vehicle_transformer_metrics.csv`
- 逐样本指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_keypoint_residual_vehicle_transformer_v0_1/tables/keypoint_residual_vehicle_transformer_per_sample_metrics.csv`
- 关键点误差表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_keypoint_residual_vehicle_transformer_v0_1/tables/keypoint_residual_vehicle_transformer_keypoint_metrics.csv`
- 模型信息表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_keypoint_residual_vehicle_transformer_v0_1/tables/keypoint_residual_vehicle_transformer_model_info.csv`
- 训练历史：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_keypoint_residual_vehicle_transformer_v0_1/tables/keypoint_residual_vehicle_transformer_training_history.csv`
- val 选择表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_keypoint_residual_vehicle_transformer_v0_1/tables/keypoint_residual_vehicle_transformer_val_selected_models.csv`
- B 轨道固定预测图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_keypoint_residual_vehicle_transformer_v0_1/figures/B_response3s_strict_core_fixed_predictions_test.png`
- B 轨道坏样本图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_keypoint_residual_vehicle_transformer_v0_1/figures/B_response3s_strict_core_keypoint_residual_bad_samples_test.png`
- 指标概览图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_keypoint_residual_vehicle_transformer_v0_1/figures/keypoint_residual_vehicle_transformer_metric_summary_test.png`
- checkpoint：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_keypoint_residual_vehicle_transformer_v0_1/checkpoints/B_response3s_strict_core_keypoint_residual_vehicle_transformer_no_subject_best.pt`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_keypoint_residual_vehicle_transformer_v0_1/logs/keypoint_residual_vehicle_transformer_summary.json`
- 服务器日志：无，本轮未使用远程服务器。
- GPTPro 提问和回复：无。
- 重要 Git commit：待提交。
- 适合用户/老师直接查看：用户查看版总结、指标表、关键点误差表、B 轨道固定预测图、B 轨道坏样本图、指标概览图。

## 最新新增：阶段 3 B 轨道车辆-only 响应分解/结构化 Transformer v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_structured_vehicle_transformer_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_structured_vehicle_transformer_v0_1_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_vehicle_instability_structured_vehicle_transformer_v0_1.py`
- 指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_structured_vehicle_transformer_v0_1/tables/structured_vehicle_transformer_metrics.csv`
- 逐样本指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_structured_vehicle_transformer_v0_1/tables/structured_vehicle_transformer_per_sample_metrics.csv`
- 辅助响应标签准确率表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_structured_vehicle_transformer_v0_1/tables/structured_vehicle_transformer_aux_metrics.csv`
- 模型信息表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_structured_vehicle_transformer_v0_1/tables/structured_vehicle_transformer_model_info.csv`
- 训练历史：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_structured_vehicle_transformer_v0_1/tables/structured_vehicle_transformer_training_history.csv`
- val 选择表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_structured_vehicle_transformer_v0_1/tables/structured_vehicle_transformer_val_selected_models.csv`
- B 轨道固定预测图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_structured_vehicle_transformer_v0_1/figures/B_response3s_strict_core_fixed_predictions_test.png`
- B 轨道坏样本图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_structured_vehicle_transformer_v0_1/figures/B_response3s_strict_core_structured_bad_samples_test.png`
- 指标概览图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_structured_vehicle_transformer_v0_1/figures/structured_vehicle_transformer_metric_summary_test.png`
- checkpoint：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_structured_vehicle_transformer_v0_1/checkpoints/B_response3s_strict_core_structured_vehicle_transformer_aux_no_subject_best.pt`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_structured_vehicle_transformer_v0_1/logs/structured_vehicle_transformer_summary.json`
- 服务器日志：无，本轮未使用远程服务器。
- GPTPro 提问和回复：无。
- 重要 Git commit：待提交。
- 适合用户/老师直接查看：用户查看版总结、指标表、B 轨道固定预测图、B 轨道坏样本图、指标概览图。

## 最新新增：阶段 3 干净响应任务车辆-only Transformer v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_clean_task_vehicle_transformer_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_clean_task_vehicle_transformer_v0_1_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_vehicle_instability_clean_task_vehicle_transformer_v0_1.py`
- 指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_transformer_v0_1/tables/clean_task_vehicle_transformer_metrics.csv`
- 逐样本指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_transformer_v0_1/tables/clean_task_vehicle_transformer_per_sample_metrics.csv`
- 模型信息表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_transformer_v0_1/tables/clean_task_vehicle_transformer_model_info.csv`
- 训练历史：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_transformer_v0_1/tables/clean_task_vehicle_transformer_training_history.csv`
- val 选择表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_transformer_v0_1/tables/clean_task_vehicle_transformer_val_selected_models.csv`
- 指标概览图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_transformer_v0_1/figures/clean_task_vehicle_transformer_metric_summary_test.png`
- B 轨道固定图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_transformer_v0_1/figures/B_response3s_strict_core_fixed_predictions_test.png`
- B 轨道坏样本图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_transformer_v0_1/figures/B_response3s_strict_core_transformer_bad_samples_test.png`
- checkpoint 目录：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_transformer_v0_1/checkpoints`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_transformer_v0_1/logs/clean_task_vehicle_transformer_summary.json`
- 服务器日志：无，本轮未使用远程服务器。
- GPTPro 提问和回复：无。
- 重要 Git commit：待提交。
- 适合用户/老师直接查看：用户查看版总结、B 轨道固定图、B 轨道坏样本图、指标概览图。

## 最新新增：阶段 3 车辆-only 响应分解标签 v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_response_decomposition_labels_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_response_decomposition_labels_v0_1_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_vehicle_instability_response_decomposition_labels_v0_1.py`
- 样本级响应分解标签：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_response_decomposition_labels_v0_1/tables/response_decomposition_sample_labels.csv`
- train-only 阈值表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_response_decomposition_labels_v0_1/tables/response_decomposition_train_thresholds.csv`
- 轨道汇总：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_response_decomposition_labels_v0_1/tables/response_decomposition_track_summary.csv`
- split 汇总：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_response_decomposition_labels_v0_1/tables/response_decomposition_split_summary.csv`
- 响应形态汇总：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_response_decomposition_labels_v0_1/tables/response_decomposition_morphology_summary.csv`
- 响应族汇总：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_response_decomposition_labels_v0_1/tables/response_decomposition_response_family_summary.csv`
- 道路模块汇总：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_response_decomposition_labels_v0_1/tables/response_decomposition_road_module_summary.csv`
- 被试汇总：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_response_decomposition_labels_v0_1/tables/response_decomposition_subject_summary.csv`
- 图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_response_decomposition_labels_v0_1/figures/response_decomposition_morphology_counts.png`
- 图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_response_decomposition_labels_v0_1/figures/response_decomposition_peak_time_amp_scatter.png`
- 图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_response_decomposition_labels_v0_1/figures/b_track_mean_gt_trajectories_by_morphology.png`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_response_decomposition_labels_v0_1/logs/response_decomposition_labels_summary.json`
- 服务器日志：无，本轮未使用服务器。
- GPTPro 提问和回复：无。
- 重要 Git commit：待提交。
- 适合用户/老师直接查看：用户查看版总结、轨道汇总、响应形态计数图、B 轨道分形态平均轨迹图。

## 最新新增：阶段 3 响应任务定义决策 v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_response_task_decision_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_response_task_decision_v0_1_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/scripts/build_vehicle_instability_response_task_decision_v0_1.py`
- 事件级任务决策表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_response_task_decision_v0_1/tables/event_response_task_decision_table.csv`
- 样本级任务 manifest：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_response_task_decision_v0_1/tables/sample_response_task_manifest.csv`
- 任务类别计数：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_response_task_decision_v0_1/tables/response_task_decision_counts.csv`
- 任务轨道计数：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_response_task_decision_v0_1/tables/response_task_track_counts.csv`
- split 汇总：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_response_task_decision_v0_1/tables/response_task_split_summary.csv`
- subject 汇总：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_response_task_decision_v0_1/tables/response_task_subject_summary.csv`
- 图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_response_task_decision_v0_1/figures/response_task_decision_counts.png`
- 图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_response_task_decision_v0_1/figures/response_task_sample_roles_by_window.png`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_response_task_decision_v0_1/logs/response_task_decision_summary.json`

## 最新新增：阶段 3 标签窗口覆盖审计 v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_label_window_coverage_audit_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_label_window_coverage_audit_v0_1_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_vehicle_instability_label_window_coverage_audit_v0_1.py`
- 样本级窗口指标：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_label_window_coverage_audit_v0_1/tables/label_window_sample_metrics.csv`
- 事件级窗口策略表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_label_window_coverage_audit_v0_1/tables/label_window_event_policy_table.csv`
- Top 坏事件叠加表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_label_window_coverage_audit_v0_1/tables/label_window_bad_event_overlay.csv`
- 窗口级统计：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_label_window_coverage_audit_v0_1/tables/label_window_window_summary.csv`
- 推荐策略计数：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_label_window_coverage_audit_v0_1/tables/label_window_policy_counts.csv`
- 图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_label_window_coverage_audit_v0_1/figures/label_window_policy_counts.png`
- 图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_label_window_coverage_audit_v0_1/figures/label_window_peak_tail_scatter_pre3.png`
- 图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_label_window_coverage_audit_v0_1/figures/label_window_coverage_rates_by_window.png`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_label_window_coverage_audit_v0_1/logs/label_window_coverage_audit_summary.json`

## 最新新增：阶段 3 复发坏样本失败来源归因 v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_bad_event_failure_attribution_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_bad_event_failure_attribution_v0_1_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_vehicle_instability_bad_event_failure_attribution_v0_1.py`
- 归因明细表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_bad_event_failure_attribution_v0_1/tables/bad_event_failure_attribution_table.csv`
- 归因旗标统计：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_bad_event_failure_attribution_v0_1/tables/bad_event_failure_flag_counts.csv`
- 主归因统计：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_bad_event_failure_attribution_v0_1/tables/bad_event_primary_attribution_counts.csv`
- 归因旗标热图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_bad_event_failure_attribution_v0_1/figures/bad_event_failure_attribution_flags.png`
- 主归因计数图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_bad_event_failure_attribution_v0_1/figures/bad_event_primary_attribution_counts.png`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_bad_event_failure_attribution_v0_1/logs/bad_event_failure_attribution_summary.json`
- 服务器日志：无，本轮未使用服务器。
- GPTPro 提问和回复：无。
- 重要 Git commit：待提交。
- 适合用户/老师直接查看：用户查看版总结、归因明细表、归因旗标热图、单事件曲线总览拼图。

## 最新新增：阶段 3 复发坏样本详细曲线复盘 v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_bad_event_curve_review_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_bad_event_curve_review_v0_1_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_vehicle_instability_bad_event_curve_review_v0_1.py`
- 总览拼图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_bad_event_curve_review_v0_1/figures/bad_event_curve_contact_sheet.png`
- 总览拼图 PDF：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_bad_event_curve_review_v0_1/figures/bad_event_curve_contact_sheet.pdf`
- 单事件曲线目录：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_bad_event_curve_review_v0_1/figures/per_event`
- 图索引：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_bad_event_curve_review_v0_1/tables/bad_event_curve_figure_index.csv`
- 模型逐事件误差表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_bad_event_curve_review_v0_1/tables/bad_event_curve_model_error_table.csv`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_bad_event_curve_review_v0_1/logs/bad_event_curve_review_summary.json`
- 服务器日志：无，本轮未使用服务器。
- GPTPro 提问和回复：无。
- 重要 Git commit：待提交。
- 适合用户/老师直接查看：用户查看版总结、总览拼图、单事件曲线目录、模型逐事件误差表。

## 最新新增：阶段 3 稳健性坏样本复盘 v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_robustness_bad_sample_review_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_robustness_bad_sample_review_v0_1_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_vehicle_instability_robustness_bad_sample_review_v0_1.py`
- 复发坏样本总表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_robustness_bad_sample_review_v0_1/tables/robustness_bad_event_recurrence.csv`
- 代表坏样本表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_robustness_bad_sample_review_v0_1/tables/robustness_representative_bad_events.csv`
- 带错误标记的逐样本表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_robustness_bad_sample_review_v0_1/tables/robustness_test_per_sample_with_error_flags.csv`
- 物理错误汇总：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_robustness_bad_sample_review_v0_1/tables/robustness_error_flag_summary_by_config_model.csv`
- 分被试坏样本汇总：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_robustness_bad_sample_review_v0_1/tables/robustness_subject_bad_summary.csv`
- 坏样本矩阵表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_robustness_bad_sample_review_v0_1/tables/robustness_bad_event_matrix.csv`
- 复发事件图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_robustness_bad_sample_review_v0_1/figures/robustness_recurrent_bad_events.png`
- 物理错误热图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_robustness_bad_sample_review_v0_1/figures/robustness_error_flag_heatmap.png`
- 分被试坏样本率图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_robustness_bad_sample_review_v0_1/figures/robustness_subject_bad_rate.png`
- 坏样本矩阵图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_robustness_bad_sample_review_v0_1/figures/robustness_bad_event_matrix.png`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_robustness_bad_sample_review_v0_1/logs/robustness_bad_sample_review_summary.json`
- 服务器日志：无，本轮未使用服务器。
- GPTPro 提问和回复：无。
- 重要 Git commit：待提交。
- 适合用户/老师直接查看：用户查看版总结、代表坏样本表、复发坏样本图、物理错误热图、坏样本矩阵图。

## 最新新增：阶段 3 强车辆基线稳健性验证 v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_strong_vehicle_robustness_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_strong_vehicle_robustness_v0_1_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_vehicle_instability_strong_vehicle_robustness_v0_1.py`
- 决策表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_strong_vehicle_robustness_v0_1/tables/strong_vehicle_robustness_decision_table.csv`
- 指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_strong_vehicle_robustness_v0_1/tables/strong_vehicle_robustness_metrics.csv`
- 逐样本指标：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_strong_vehicle_robustness_v0_1/tables/strong_vehicle_robustness_per_sample_metrics.csv`
- 模型信息：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_strong_vehicle_robustness_v0_1/tables/strong_vehicle_robustness_model_info.csv`
- RMSE 热图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_strong_vehicle_robustness_v0_1/figures/strong_vehicle_robustness_rmse_heatmap.png`
- 大幅响应召回热图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_strong_vehicle_robustness_v0_1/figures/strong_vehicle_robustness_large_recall_heatmap.png`
- 反向修正匹配热图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_strong_vehicle_robustness_v0_1/figures/strong_vehicle_robustness_reversal_heatmap.png`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_strong_vehicle_robustness_v0_1/logs/strong_vehicle_robustness_summary.json`
- 服务器日志：无，本轮未使用服务器。
- GPTPro 提问和回复：无。
- 重要 Git commit：待提交。
- 适合用户/老师直接查看：用户查看版总结、决策表、RMSE 热图、大幅响应召回热图、反向修正热图。

## 最新新增：阶段 3 车辆-only 统一对照 v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_unified_vehicle_comparison_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_unified_vehicle_comparison_v0_1_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_vehicle_instability_unified_vehicle_comparison_v0_1.py`
- test 指标总表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_unified_vehicle_comparison_v0_1/tables/unified_vehicle_comparison_metrics_test.csv`
- all-split 指标总表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_unified_vehicle_comparison_v0_1/tables/unified_vehicle_comparison_metrics_all_splits.csv`
- 相对 formal ridge 差异表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_unified_vehicle_comparison_v0_1/tables/unified_vehicle_comparison_delta_vs_formal_test.csv`
- 候选决策表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_unified_vehicle_comparison_v0_1/tables/unified_vehicle_candidate_decision_table.csv`
- 指标排名表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_unified_vehicle_comparison_v0_1/tables/unified_vehicle_metric_rankings_test.csv`
- 坏样本重合表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_unified_vehicle_comparison_v0_1/tables/unified_vehicle_top_bad_overlap.csv`
- 关键指标图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_unified_vehicle_comparison_v0_1/figures/unified_vehicle_key_metrics_test.png`
- 物理错误热图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_unified_vehicle_comparison_v0_1/figures/unified_vehicle_physical_failure_heatmap_test.png`
- RMSE/错侧权衡图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_unified_vehicle_comparison_v0_1/figures/unified_vehicle_rmse_vs_wrong_side_test.png`
- 坏样本重合图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_unified_vehicle_comparison_v0_1/figures/unified_vehicle_top_bad_overlap.png`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_unified_vehicle_comparison_v0_1/logs/unified_vehicle_comparison_summary.json`
- 服务器日志：无，本轮未使用服务器。
- GPTPro 提问和回复：无。
- 重要 Git commit：待提交。
- 适合用户/老师直接查看：用户查看版总结、test 指标总表、候选决策表、关键指标图、物理错误热图。

## 最新新增：阶段 3 车辆-only Transformer 时序基线 v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_vehicle_transformer_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_vehicle_transformer_v0_1_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_vehicle_instability_vehicle_transformer_v0_1.py`
- 指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_vehicle_transformer_v0_1/tables/vehicle_transformer_metrics.csv`
- 逐样本指标：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_vehicle_transformer_v0_1/tables/vehicle_transformer_per_sample_metrics.csv`
- 训练历史：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_vehicle_transformer_v0_1/tables/vehicle_transformer_training_history.csv`
- 模型信息：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_vehicle_transformer_v0_1/tables/vehicle_transformer_model_info.csv`
- 固定预测图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_vehicle_transformer_v0_1/figures/vehicle_transformer_fixed_predictions_test.png`
- 坏样本图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_vehicle_transformer_v0_1/figures/vehicle_transformer_bad_samples_test.png`
- checkpoint：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_vehicle_transformer_v0_1/checkpoints/vehicle_transformer_context_no_subject_best.pt`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_vehicle_transformer_v0_1/logs/vehicle_transformer_summary.json`
- 服务器日志：无，本轮未使用服务器。
- GPTPro 提问和回复：无。
- 重要 Git commit：待提交。
- 适合用户/老师直接查看：用户查看版总结、固定预测图、坏样本图、指标表。

## 最新新增：阶段 3 强车辆-only 时序/结构化基线 v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_strong_vehicle_baselines_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_strong_vehicle_baselines_v0_1_cn.md`
- 指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_strong_vehicle_baselines_v0_1/tables/strong_vehicle_baseline_metrics.csv`
- 逐样本指标：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_strong_vehicle_baselines_v0_1/tables/strong_vehicle_baseline_per_sample_metrics.csv`
- 模型信息：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_strong_vehicle_baselines_v0_1/tables/strong_vehicle_model_info.csv`
- val 选择模型错误分型：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_strong_vehicle_baselines_v0_1/tables/selected_model_error_flag_summary.csv`
- 与 formal ridge 逐样本差异：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_strong_vehicle_baselines_v0_1/tables/selected_vs_formal_per_sample_delta.csv`
- 固定预测图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_strong_vehicle_baselines_v0_1/figures/strong_vehicle_fixed_predictions_test.png`
- 坏样本图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_strong_vehicle_baselines_v0_1/figures/strong_vehicle_bad_samples_test.png`
- test 指标柱状图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_strong_vehicle_baselines_v0_1/figures/strong_vehicle_model_metric_bars_test.png`
- 与 formal ridge 的 RMSE 差异图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_strong_vehicle_baselines_v0_1/figures/strong_vehicle_selected_vs_formal_rmse_delta.png`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_vehicle_instability_strong_vehicle_baselines_v0_1.py`
- 服务器日志：无，本轮未使用服务器。
- GPTPro 提问和回复：无。
- 重要 Git commit：待提交。
- 适合用户/老师直接查看：用户查看版总结、固定预测图、坏样本图、指标表。

## 最新新增：阶段 3 车辆基线坏样本物理错误分型 v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_error_analysis_user_summary_cn.md`
- 正式中文报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_error_analysis_v0_1_cn.md`
- 脚本入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_vehicle_instability_error_analysis_v0_1.py`
- 逐样本错误标签表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_error_analysis_v0_1/tables/per_sample_error_taxonomy.csv`
- 错误标签汇总：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_error_analysis_v0_1/tables/error_flags_summary.csv`
- 分被试汇总：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_error_analysis_v0_1/tables/subject_summary.csv`
- 分响应类型汇总：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_error_analysis_v0_1/tables/morphology_summary.csv`
- 与旧 deep 对照：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_error_analysis_v0_1/tables/old_comparison_summary.csv`
- 错误标签柱状图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_error_analysis_v0_1/figures/formal_error_flag_counts.png`
- 与旧 deep RMSE 散点图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_error_analysis_v0_1/figures/formal_vs_old_deep_rmse_scatter.png`
- top bad 错误矩阵：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_error_analysis_v0_1/figures/top_bad_sample_error_matrix.png`
- 分被试错误热图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_error_analysis_v0_1/figures/subject_error_rate_heatmap.png`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_error_analysis_v0_1/logs/stage03_error_analysis_summary.json`
- 关键结果：反向修正计数不匹配 126/139，尾段漂移 87/139，严重幅值不足 81/139，多段修正结构不匹配 46/139，错侧 32/139，大幅响应漏召回 23/139；旧 deep 与 formal ridge top20%坏样本重叠 21/28。
- 重要边界：错误标签只用于解释 test 集失败类型，不参与训练、split、标准化或任何生理/风格有效性结论。
- 适合用户/老师直接查看：优先看用户查看版总结、错误标签柱状图、top bad 错误矩阵和逐样本错误标签表。

## 最新新增：阶段 3 正式车辆失稳样本车辆-only 基线 v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_formal_baselines_user_summary_cn.md`
- 正式中文报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_formal_baselines_v0_1_cn.md`
- 脚本入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_vehicle_instability_formal_baselines_v0_1.py`
- 指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_formal_baselines_v0_1/tables/formal_baseline_metrics.csv`
- 逐样本指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_formal_baselines_v0_1/tables/formal_baseline_per_sample_metrics.csv`
- 模型信息表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_formal_baselines_v0_1/tables/formal_baseline_model_info.csv`
- 固定预测图样本表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_formal_baselines_v0_1/tables/formal_baseline_fixed_plot_samples.csv`
- 坏样本图样本表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_formal_baselines_v0_1/tables/formal_baseline_bad_plot_samples.csv`
- 固定预测图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_formal_baselines_v0_1/figures/formal_baseline_fixed_predictions_test.png`
- 坏样本图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_formal_baselines_v0_1/figures/formal_baseline_bad_samples_test.png`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_formal_baselines_v0_1/logs/formal_baseline_summary.json`
- 关键结果：主窗口 `pre2_label2_old_main` + session-level test 中，`ridge_vehicle_context_no_subject` RMSE=0.649341、错侧率=0.230216、大幅响应召回=0.080000、严重幅值不足率=0.582734、反向修正计数完全匹配率=0.093525。
- 重要边界：该结果是车辆-only 浅层基线，不使用生理、脑电、连续风格或驾驶员 ID；不能支持风格/生理有效性结论。
- 适合用户/老师直接查看：优先看用户查看版总结、固定预测图、坏样本图和指标表。

## 最新新增：车辆失稳高置信正式样本清单 v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage02_vehicle_instability_highconf_user_summary_cn.md`
- 数据版本卡：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/dataset_version_card_vehicle_instability_highconf_v0_1_cn.md`
- 正式中文说明：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/vehicle_instability_highconf_samples_v0_1_cn.md`
- 脚本入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/scripts/build_vehicle_instability_highconf_samples_v0_1.py`
- 样本主表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_highconf_v0_1/tables/samples_master.csv`
- 样本 JSONL：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_highconf_v0_1/tables/samples_master.jsonl`
- 事件锚点表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_highconf_v0_1/tables/event_anchor_table.csv`
- split 表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_highconf_v0_1/tables/split_table.csv`
- split 可行性报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_highconf_v0_1/tables/split_feasibility_report.csv`
- 排除原因表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_highconf_v0_1/tables/sample_exclusion_reasons.csv`
- eval-only 响应类型统计：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_highconf_v0_1/tables/label_eval_only_response_summary.csv`
- 窗口配置表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_highconf_v0_1/tables/window_config_table.csv`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_highconf_v0_1/logs/vehicle_instability_highconf_samples_summary_v0_1.json`
- 关键数量：908 个高置信事件，906 个进入正式 v0.1，2 个因 3 秒历史窗口不足排除；3 个窗口共 2718 行；主窗口 session-level split 为 train 611、val 156、test 139。
- 重要边界：`eval_label_*` 字段只允许用于评估分层、固定图和困难样本分析，不允许作为模型输入、split 决策或标准化依据。
- 适合用户/老师直接查看：优先看用户查看版总结、数据版本卡、split 可行性报告和排除原因表。

## 最新新增：旧 `vehicle_direct` 全量车辆-only clean 对照 v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_oldcode_vehicle_direct_full_clean_user_summary_cn.md`
- 正式中文报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/oldcode_vehicle_direct_full_clean_on_instability_v0_1_cn.md`
- clean manifest 说明：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/oldcode_deep_clean_vehicle_manifest_v0_1_cn.md`
- clean manifest 脚本入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/scripts/build_oldcode_deep_clean_vehicle_manifest_v0_1.py`
- 评估和画图脚本入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/scripts/evaluate_oldcode_vehicle_direct_full_instability_v0_1.py`
- clean session-level manifest：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_processed_datasets/vehicle_instability_allraw_highconf_v0_1/tables/oldcode_manifest_session_level_split_clean_vehicle_v0_1.csv`
- clean random split manifest：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_processed_datasets/vehicle_instability_allraw_highconf_v0_1/tables/oldcode_manifest_random_event_split_clean_vehicle_v0_1.csv`
- clean subject-level manifest：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_processed_datasets/vehicle_instability_allraw_highconf_v0_1/tables/oldcode_manifest_subject_level_split_clean_vehicle_v0_1.csv`
- clean 文件状态表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_processed_datasets/vehicle_instability_allraw_highconf_v0_1/tables/oldcode_deep_clean_vehicle_status_v0_1.csv`
- clean manifest 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_processed_datasets/vehicle_instability_allraw_highconf_v0_1/logs/oldcode_deep_clean_vehicle_manifest_summary_v0_1.json`
- 训练日志：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/oldcode_vehicle_direct_full_clean_on_instability_v0_1/logs/oldcode_vehicle_direct_full_clean_train_stdout.log`
- 评估摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/oldcode_vehicle_direct_full_clean_on_instability_v0_1/logs/oldcode_vehicle_direct_full_eval_summary.json`
- 指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/oldcode_vehicle_direct_full_clean_on_instability_v0_1/tables/oldcode_vehicle_direct_full_metrics.csv`
- 逐样本指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/oldcode_vehicle_direct_full_clean_on_instability_v0_1/tables/oldcode_vehicle_direct_full_per_sample_metrics.csv`
- 分被试结果：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/oldcode_vehicle_direct_full_clean_on_instability_v0_1/tables/oldcode_vehicle_direct_full_by_subject_test.csv`
- 分响应类型结果：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/oldcode_vehicle_direct_full_clean_on_instability_v0_1/tables/oldcode_vehicle_direct_full_by_response_type_test.csv`
- 固定预测图样本表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/oldcode_vehicle_direct_full_clean_on_instability_v0_1/tables/oldcode_vehicle_direct_full_fixed_plot_samples.csv`
- 坏样本图样本表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/oldcode_vehicle_direct_full_clean_on_instability_v0_1/tables/oldcode_vehicle_direct_full_bad_plot_samples.csv`
- 固定预测图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/oldcode_vehicle_direct_full_clean_on_instability_v0_1/figures/oldcode_vehicle_direct_full_fixed_predictions_test.png`
- 坏样本图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/oldcode_vehicle_direct_full_clean_on_instability_v0_1/figures/oldcode_vehicle_direct_full_bad_samples_test.png`
- 本地旧训练 run：`F:/data_set_process/data_process/tmp/event_conditioned_runs/OLD_FULL_INSTABILITY_HIGHCONF_VEHICLE_DIRECT_CLEAN_V0_1_20260512_181413`
- 重要结论：旧 `vehicle_direct` active checkpoint 在 session-level test 上 RMSE=0.637366，但严重幅值不足率=0.683453、大幅响应召回=0.142857、反向修正计数完全匹配率=0.086331；只能作为旧流程历史对照和坏样本来源，不能替代新流程强车辆基线。
- 重要风险记录：raw manifest 直读原始 CSV 的旧 deep run 已判定无效，因为旧 loader 会把原始交替缺失点填 0；正式结果只采用 clean manifest。
- 适合用户/老师直接查看：优先看用户查看版总结、固定预测图、坏样本图和正式中文报告。

## 最新新增：道路设定引导的车辆失稳事件判定 v0.1

- 中文说明：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/road_guided_instability_v0_1_cn.md`
- 数据版本卡：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/dataset_version_card_vehicle_instability_road_guided_v0_1_cn.md`
- 脚本入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/scripts/build_road_guided_instability_events_v0_1.py`
- 全量判定表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/road_guided_instability_v0_1/tables/road_guided_instability_events_v0_1.csv`
- 自动采用表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/road_guided_instability_v0_1/tables/road_guided_auto_accepted_events_v0_1.csv`
- 中间复核队列表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/road_guided_instability_v0_1/tables/road_guided_review_queue_v0_1.csv`
- 汇总表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/road_guided_instability_v0_1/tables/road_guided_instability_summary_v0_1.csv`
- 道路模块交叉表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/road_guided_instability_v0_1/tables/road_guided_module_summary_v0_1.csv`
- 人工抽查校准表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/road_guided_instability_v0_1/tables/road_guided_manual_calibration_v0_1.csv`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/road_guided_instability_v0_1/logs/road_guided_instability_run_summary_v0_1.json`
- 旧日志依据：`F:/data_set_process/data_process/04_project_logs/reports/trigger_response_lag_20260421/TASK_DEFINITION_AND_EVENT_LOGIC.md`
- 道路设计依据：`F:/data_set_process/data_process/01_datasets/多模态数据/被试数据集合/道路信息/full_centerline_layout.csv`
- 重要 Git commit：`ad981f6 Add road-guided instability event adjudication`
- 适合用户/老师直接查看：优先看中文说明和自动采用表；这版用于替代“全人工逐条标注”的第一轮车辆失稳事件筛选。

## 最新新增：全部原始车辆 CSV 失稳样本重筛 v0.1

- 中文说明：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/all_raw_vehicle_instability_rescreen_v0_1_cn.md`
- 数据版本卡：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/dataset_version_card_all_raw_vehicle_instability_rescreen_v0_1_cn.md`
- 脚本入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/scripts/rescreen_all_raw_vehicle_instability_v0_1.py`
- 全量候选表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_all_raw_rescreen_v0_1/tables/all_raw_vehicle_instability_candidates_v0_1.csv`
- 高置信主清单：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_all_raw_rescreen_v0_1/tables/all_raw_vehicle_instability_primary_high_confidence_v0_1.csv`
- 自动采用扩展清单：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_all_raw_rescreen_v0_1/tables/all_raw_vehicle_instability_auto_accepted_v0_1.csv`
- 中间复核队列：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_all_raw_rescreen_v0_1/tables/all_raw_vehicle_instability_review_queue_v0_1.csv`
- 低证据剔除表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_all_raw_rescreen_v0_1/tables/all_raw_vehicle_instability_rejected_v0_1.csv`
- 文件读取状态：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_all_raw_rescreen_v0_1/tables/all_raw_vehicle_rescreen_file_status_v0_1.csv`
- 汇总表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_all_raw_rescreen_v0_1/tables/all_raw_vehicle_instability_summary_v0_1.csv`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_all_raw_rescreen_v0_1/logs/all_raw_vehicle_instability_rescreen_run_summary_v0_1.json`
- 重要 Git commit：`12c30cf Rescreen all raw vehicle instability events`
- 适合用户/老师直接查看：优先看中文说明、数据版本卡和高置信主清单；该版本覆盖全部 91 个原始车辆 CSV。

更新时间：2026-05-12 14:03:26

## 阶段 0：旧流程冻结与重建准则

- 阶段 0 规则说明：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/00_project_notes/stage00_old_flow_freeze_and_rules_cn.md`
- 阶段 0 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage00_user_summary_cn.md`
- 旧流程参考定位：`04_project_logs/reports/progress/experiment_registry.md`、`04_project_logs/reports/physio_to_g14_progress_review_20260511/`
- GPTPro 原始数据重建建议：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/gptpro_replies/20260512_rebuild_steering_reply_summary_cn.md`

## 阶段 1：原始数据审计

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage01_user_summary_cn.md`
- 审计中文总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/raw_data_audit_summary_cn.md`
- 文件清单：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/01_audit/tables/raw_file_inventory.csv`
- 字段报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/01_audit/tables/raw_schema_report.csv`
- 被试/记录/模态矩阵：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/01_audit/tables/subject_session_modality_matrix.csv`
- 时间连续性报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/01_audit/tables/timestamp_continuity_report.csv`
- 采样率报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/01_audit/tables/sampling_rate_report.csv`
- 模态重叠报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/01_audit/tables/modality_overlap_report.csv`
- 信号质量报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/01_audit/tables/signal_quality_report.csv`
- EEG 初审报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/01_audit/tables/eeg_artifact_report.csv`
- 泄漏风险报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/01_audit/tables/leakage_risk_report.csv`
- 审计脚本入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/01_audit/scripts/raw_csv_audit.py`
- 审计图目录：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/01_audit/figures/audit`
- 审计运行日志：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/01_audit/logs/raw_csv_audit.log`
- 阶段 0/1 完成审计：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/00_project_notes/STAGE00_01_COMPLETION_AUDIT_CN.md`

## 阶段 2：事件锚点与样本清单重建

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage02_user_summary_cn.md`
- 事件锚点重建总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/event_anchor_rebuild_summary_cn.md`
- 数据版本卡：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/dataset_version_card_v0_2_cn.md`
- 处理后车辆窗口说明：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/processed_vehicle_windows_v0_2_cn.md`
- 候选事件总表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/tables/candidate_events_master.csv`
- 样本总表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/tables/samples_master.csv`
- 样本 JSONL：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/tables/samples_master.jsonl`
- split 表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/tables/split_table.csv`
- split 可行性报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/tables/split_feasibility_report.csv`
- 锚点来源统计：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/tables/anchor_source_inventory.csv`
- 道路设计清单：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/tables/road_design_inventory.csv`
- 窗口配置对比：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/tables/window_config_comparison.csv`
- 锚点来源近邻对照：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/tables/anchor_source_comparison.csv`
- 车辆重采样状态：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/tables/vehicle_resample_status.csv`
- 阶段 2 脚本入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/scripts/build_stage2_samples.py`
- 处理后车辆窗口脚本入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/scripts/build_processed_vehicle_windows.py`
- 阶段 2 图目录：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/figures`
- 处理后车辆窗口数组：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_processed_datasets/vehicle_road_curvature_v0_2/arrays`
- 处理后车辆窗口索引：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_processed_datasets/vehicle_road_curvature_v0_2/tables`
- 人工事件标注审查包 v0.1 中文说明：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/manual_event_label_review_pack_v0_1_cn.md`
- 人工事件标注审查 HTML：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/manual_event_label_review_v0_1/review_index.html`
- 人工标注模板：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/manual_event_label_review_v0_1/tables/manual_event_labels_template_v0_1.csv`
- 人工审查记录清单：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/manual_event_label_review_v0_1/tables/session_review_manifest_v0_1.csv`
- 人工审查时间线图目录：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/manual_event_label_review_v0_1/figures`
- 人工标注包脚本入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/scripts/build_manual_event_label_review_pack.py`
- 人工标注包运行日志：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/manual_event_label_review_v0_1/logs`
- 键盘式人工标注播放器页面：`http://127.0.0.1:8766/`
- 键盘式人工标注播放器说明：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/manual_event_keyboard_player_v0_1_cn.md`；当前页面已升级为候选段审查模式，初版整段播放器可从 `/legacy` 查看。
- 键盘式人工标注播放器脚本入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/scripts/run_manual_event_keyboard_player.py`
- 键盘式人工标签输出：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/manual_event_keyboard_player_v0_1/tables/keyboard_event_labels_v0_1.csv`
- 键盘式人工标注播放器日志：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/manual_event_keyboard_player_v0_1/logs`
- Codex 自动事件审阅说明：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/codex_event_review_v0_1_cn.md`
- Codex 自动事件审阅脚本入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/scripts/build_codex_event_review_v0_1.py`
- Codex 自动审阅总表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/codex_event_review_v0_1/tables/codex_reviewed_event_labels_v0_1.csv`
- Codex 自动采用标签：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/codex_event_review_v0_1/tables/codex_auto_accepted_event_labels_v0_1.csv`
- Codex 需要复核队列：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/codex_event_review_v0_1/tables/codex_needs_human_review_v0_1.csv`
- Codex 自动审阅汇总表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/codex_event_review_v0_1/tables/codex_event_review_summary_v0_1.csv`
- Codex 自动审阅图目录：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/codex_event_review_v0_1/figures`
- Codex 自动审阅运行日志：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/codex_event_review_v0_1/logs`

## 阶段 2 修正：车辆失稳事件候选 v0.1

- 说明：`codex_event_review_v0_1` 的 404 个样本是弯道/道路曲率候选，已降级为道路上下文参考；当前主线改为车辆失稳候选。
- 车辆失稳中文说明：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/instability_event_review_v0_1_cn.md`
- 车辆失稳数据版本卡：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/dataset_version_card_vehicle_instability_v0_1_cn.md`
- 车辆失稳审阅脚本：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/scripts/build_instability_event_review_v0_1.py`
- 全量车辆失稳候选表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/instability_event_review_v0_1/tables/instability_reviewed_events_v0_1.csv`
- 自动采用车辆失稳候选：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/instability_event_review_v0_1/tables/instability_auto_accepted_events_v0_1.csv`
- 需要人工复核车辆失稳候选：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/instability_event_review_v0_1/tables/instability_needs_human_review_v0_1.csv`
- 车辆失稳审阅汇总表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/instability_event_review_v0_1/tables/instability_event_review_summary_v0_1.csv`
- 车辆失稳概览图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/instability_event_review_v0_1/figures/instability_event_score_overview_v0_1.png`
- 车辆失稳示例图目录：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/instability_event_review_v0_1/figures`
- 车辆失稳运行日志：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/instability_event_review_v0_1/logs/build_instability_event_review_v0_1.json`
- 本地车辆失稳审查页面：`http://127.0.0.1:8766/`
- 车辆失稳人工标签输出：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/manual_event_keyboard_player_v0_1/tables/keyboard_instability_event_labels_v0_1.csv`

## 阶段 3：无学习基线与纯车辆基线

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_user_summary_cn.md`
- 阶段 3 基线总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_baseline_summary_cn.md`
- 阶段 3 脚本入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/scripts/evaluate_stage3_vehicle_baselines.py`
- 汇总指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_baselines_v0_2/tables/stage03_baseline_metrics.csv`
- 逐样本指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_baselines_v0_2/tables/stage03_per_sample_metrics.csv`
- 各窗口/切分测试集最好行：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_baselines_v0_2/tables/stage03_best_test_by_window_split.csv`
- ridge 训练信息：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_baselines_v0_2/tables/stage03_ridge_model_info.csv`
- 固定画图样本集：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_baselines_v0_2/tables/stage03_fixed_plot_sample_set.csv`
- 固定预测图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_baselines_v0_2/figures/stage03_fixed_predictions_pre2_session_test.png`
- 坏样本图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_baselines_v0_2/figures/stage03_bad_samples_pre2_session_test_ridge.png`
- 运行 stdout：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_baselines_v0_2/logs/evaluate_stage3_vehicle_baselines.stdout.log`
- 运行 stderr：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_baselines_v0_2/logs/evaluate_stage3_vehicle_baselines.stderr.log`
- JSON 摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_baselines_v0_2/logs/stage03_baseline_summary.json`
- v0.3 诊断总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_diagnostics_v0_3_cn.md`
- v0.3 诊断脚本入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_vehicle_diagnostics_v0_3.py`
- v0.3 无被试 ID 模型指标：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_diagnostics_v0_3/tables/stage03_stronger_vehicle_metrics_v0_3.csv`
- v0.3 模型对照表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_diagnostics_v0_3/tables/stage03_vehicle_model_comparison_v0_3.csv`
- v0.3 坏样本诊断表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_diagnostics_v0_3/tables/stage03_bad_sample_diagnostics_v0_3.csv`
- v0.3 固定图样本诊断表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_diagnostics_v0_3/tables/stage03_fixed_plot_diagnostics_v0_3.csv`
- v0.3 错误桶：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_diagnostics_v0_3/tables/stage03_error_bucket_summary_pre2_session_v0_3.csv`
- v0.3 小样本过拟合测试：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_diagnostics_v0_3/tables/stage03_small_overfit_report_v0_3.csv`
- v0.3 模型对照图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_diagnostics_v0_3/figures/stage03_pre2_session_model_rmse_comparison_v0_3.png`
- v0.3 坏样本诊断图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_diagnostics_v0_3/figures/stage03_pre2_session_bad_sample_diagnostic_v0_3.png`
- v0.3 运行日志：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_diagnostics_v0_3/logs`
- v0.4 RBF KRR 候选模型卡：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_rbf_krr_candidate_model_card_v0_4_cn.md`
- v0.4 脚本入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_rbf_krr_model_card_v0_4.py`
- v0.4 候选指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_rbf_krr_model_card_v0_4/tables/stage03_rbf_krr_candidate_metrics_v0_4.csv`
- v0.4 分被试表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_rbf_krr_model_card_v0_4/tables/stage03_rbf_krr_per_subject_v0_4.csv`
- v0.4 分响应组表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_rbf_krr_model_card_v0_4/tables/stage03_rbf_krr_response_group_summary_v0_4.csv`
- v0.4 画图样本表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_rbf_krr_model_card_v0_4/tables/stage03_rbf_krr_plot_sample_set_v0_4.csv`
- v0.4 pre2 固定样本轨迹图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_rbf_krr_model_card_v0_4/figures/stage03_rbf_krr_fixed_predictions_pre2_session_v0_4.png`
- v0.4 pre2 坏样本轨迹图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_rbf_krr_model_card_v0_4/figures/stage03_rbf_krr_bad_samples_pre2_session_v0_4.png`
- v0.4 pre3 固定样本轨迹图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_rbf_krr_model_card_v0_4/figures/stage03_rbf_krr_fixed_predictions_pre3_session_v0_4.png`
- v0.4 pre3 坏样本轨迹图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_rbf_krr_model_card_v0_4/figures/stage03_rbf_krr_bad_samples_pre3_session_v0_4.png`
- v0.4 运行日志：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_rbf_krr_model_card_v0_4/logs`

## 服务器日志

- 本阶段未使用服务器。

## 重要 Git commit

- `e9d302f Add raw rebuild stage 0 and 1 audit`
- `bae5618 Add stage 0 and 1 completion audit`
- `9bef223 Record completion audit commit`
- `114208d Add stage 2 samples and processed vehicle windows`
- `b61e427 Add stage 3 vehicle baseline evaluation`
- `a2379c5 Record stage 3 artifact commit`
- `82d6a1a Add stage 3 no-subject vehicle diagnostics`
- `db1ff13 Record stage 3 no-subject diagnostics commit`
- `6c3c9f3 Add stage 3 RBF KRR model card`
- `9907aa5 Add manual event labeling review pack`
- `cf9d06b Add keyboard event labeling player`
- `5b8abbf Focus manual labeler on candidate event segments`
- `4019653 Clarify manual labeler event line legend`
- `2819f3f Add codex automatic event review`
- `d0dbf5d Rebuild event review around vehicle instability`

## 适合用户/老师直接查看的材料

1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage01_user_summary_cn.md`
2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage02_user_summary_cn.md`
3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/processed_vehicle_windows_v0_2_cn.md`
4. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/dataset_version_card_v0_2_cn.md`
5. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_user_summary_cn.md`
6. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_baseline_summary_cn.md`
7. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_diagnostics_v0_3_cn.md`
8. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_rbf_krr_candidate_model_card_v0_4_cn.md`
9. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_rbf_krr_model_card_v0_4/figures/stage03_rbf_krr_fixed_predictions_pre2_session_v0_4.png`
10. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_rbf_krr_model_card_v0_4/figures/stage03_rbf_krr_bad_samples_pre2_session_v0_4.png`
11. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_rbf_krr_model_card_v0_4/figures/stage03_rbf_krr_fixed_predictions_pre3_session_v0_4.png`
12. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_rbf_krr_model_card_v0_4/figures/stage03_rbf_krr_bad_samples_pre3_session_v0_4.png`
13. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_diagnostics_v0_3/figures/stage03_pre2_session_model_rmse_comparison_v0_3.png`
14. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_diagnostics_v0_3/figures/stage03_pre2_session_bad_sample_diagnostic_v0_3.png`
15. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_baselines_v0_2/figures/stage03_fixed_predictions_pre2_session_test.png`
16. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_baselines_v0_2/figures/stage03_bad_samples_pre2_session_test_ridge.png`
17. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/figures/stage02_anchor_overlay_example.png`
18. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/tables/samples_master.csv`
19. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_processed_datasets/vehicle_road_curvature_v0_2/tables/processed_vehicle_window_outputs.csv`
20. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/00_project_notes/STAGE00_01_COMPLETION_AUDIT_CN.md`
21. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/manual_event_label_review_v0_1/review_index.html`
22. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/manual_event_label_review_pack_v0_1_cn.md`
23. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/manual_event_label_review_v0_1/tables/manual_event_labels_template_v0_1.csv`
24. `http://127.0.0.1:8766/`
25. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/manual_event_keyboard_player_v0_1_cn.md`
26. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/manual_event_keyboard_player_v0_1/tables/keyboard_event_labels_v0_1.csv`
27. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/codex_event_review_v0_1_cn.md`
28. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/codex_event_review_v0_1/tables/codex_auto_accepted_event_labels_v0_1.csv`
29. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/codex_event_review_v0_1/tables/codex_needs_human_review_v0_1.csv`

## 阶段 2 追加：道路事件位置与锚点重建审计 v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage02_road_anchor_audit_user_summary_cn.md`
- 中文审计报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/road_event_anchor_audit_v0_1_cn.md`
- 脚本入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/scripts/build_road_event_anchor_audit_v0_1.py`
- 道路模块位置表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/road_event_anchor_audit_v0_1/tables/road_event_position_map_v0_1.csv`
- 每条记录道路映射摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/road_event_anchor_audit_v0_1/tables/session_road_mapping_summary_v0_1.csv`
- 每条记录道路模块进入/离开时间：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/road_event_anchor_audit_v0_1/tables/session_module_entry_exit_v0_1.csv`
- 旧锚点对齐表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/road_event_anchor_audit_v0_1/tables/old_new_anchor_alignment_v0_1.csv`
- 道路引导候选对齐表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/road_event_anchor_audit_v0_1/tables/road_guided_anchor_alignment_v0_1.csv`
- 审计汇总表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/road_event_anchor_audit_v0_1/tables/road_event_anchor_audit_summary_v0_1.csv`
- 道路模块位置图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/road_event_anchor_audit_v0_1/figures/road_event_position_map_v0_1.png`
- 锚点审计概览图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/road_event_anchor_audit_v0_1/figures/road_anchor_audit_overview_v0_1.png`
- 代表样本面板目录：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/road_event_anchor_audit_v0_1/figures/representative_panels`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/road_event_anchor_audit_v0_1/logs/road_event_anchor_audit_run_summary_v0_1.json`

## 阶段 2/3 追加：旧代码测试全原始车辆失稳高置信样本 v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_oldcode_instability_user_summary_cn.md`
- 旧车辆代码诊断报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/oldcode_vehicle_baseline_on_instability_v0_1_cn.md`
- 旧深度模型 smoke 报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/oldcode_vehicle_direct_smoke_on_instability_v0_1_cn.md`
- 旧代码兼容数据包说明：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/instability_oldcode_dataset_v0_1_cn.md`
- 窗口生成脚本：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/scripts/build_instability_oldcode_windows_v0_1.py`
- 旧车辆基线诊断脚本：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/scripts/evaluate_oldcode_vehicle_baselines_on_instability_v0_1.py`
- 处理后车辆窗口数组：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_processed_datasets/vehicle_instability_allraw_highconf_v0_1/arrays`
- 旧代码 manifest：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_processed_datasets/vehicle_instability_allraw_highconf_v0_1/tables/oldcode_manifest_session_level_split.csv`
- 旧代码可用性表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_processed_datasets/vehicle_instability_allraw_highconf_v0_1/tables/instability_highconf_events_oldcode_eligibility_v0_1.csv`
- 窗口索引和 split 表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_processed_datasets/vehicle_instability_allraw_highconf_v0_1/tables`
- 旧车辆基线指标：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/oldcode_vehicle_baselines_on_instability_v0_1/tables/oldcode_instability_baseline_metrics.csv`
- 逐样本指标：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/oldcode_vehicle_baselines_on_instability_v0_1/tables/oldcode_instability_per_sample_metrics.csv`
- 各窗口/切分最佳测试行：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/oldcode_vehicle_baselines_on_instability_v0_1/tables/oldcode_instability_best_test_by_window_split.csv`
- 固定预测图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/oldcode_vehicle_baselines_on_instability_v0_1/figures/oldcode_fixed_predictions_pre2_session_test.png`
- 坏样本图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/oldcode_vehicle_baselines_on_instability_v0_1/figures/oldcode_bad_samples_pre2_session_test_ridge.png`
- 旧 manifest loader smoke 记录：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_processed_datasets/vehicle_instability_allraw_highconf_v0_1/logs/oldcode_manifest_loader_smoke_check.json`
- 旧深度模型 smoke run 目录：`F:/data_set_process/data_process/tmp/event_conditioned_runs/OLD_SMOKE_INSTABILITY_HIGHCONF_V0_1_20260512_165950`


## 场景触发点审计 v0.2（2026-05-12）

| 产物 | 路径 | 说明 |
|---|---|---|
| 审计脚本 | `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/scripts/build_scene_trigger_audit_v0_2.py` | 解析 `.aed` 交通对象、触发点并和旧锚点对齐 |
| 用户版说明 | `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage02_scene_trigger_user_summary_cn.md` | 面向用户/老师的白话说明 |
| 完整报告 | `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/scene_trigger_audit_v0_2_cn.md` | 包含 longstraight 交通对象、触发点和旧锚点对齐结论 |
| 交通对象表 | `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/scene_trigger_audit_v0_2/tables/aed_traffic_objects_v0_2.csv` | `.aed` 中交通车、车流源等对象 |
| 场景触发点表 | `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/scene_trigger_audit_v0_2/tables/aed_traffic_triggers_v0_2.csv` | Activate、Stop、ChangeLane 触发点 |
| 触发点时间映射 | `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/scene_trigger_audit_v0_2/tables/scene_trigger_session_times_v0_2.csv` | 触发点换算到每条被试记录相对时间轴 |
| 旧锚点对齐表 | `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/scene_trigger_audit_v0_2/tables/old_anchor_vs_scene_trigger_v0_2.csv` | 旧 v400 锚点与最近场景触发点时间差 |
| longstraight 图 | `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/scene_trigger_audit_v0_2/figures/longstraight_scene_trigger_map_v0_2.png` | 25/26 车道交通对象与触发点图 |
| 时间差图 | `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/scene_trigger_audit_v0_2/figures/old_anchor_scene_trigger_delta_hist_v0_2.png` | 旧锚点相对最近场景触发点的时间差分布 |


| longstraight 被试车道投影图 | `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/scene_trigger_audit_v0_2/figures/longstraight_ego_lane_projection_v0_2.png` | 把被试车横向位置与交通触发车道放在同一横向坐标上 |
| longstraight 被试车道表 | `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/scene_trigger_audit_v0_2/tables/longstraight_ego_lane_at_scene_triggers_v0_2.csv` | 每个 longstraight 场景触发点处的被试车道估计 |
## 阶段 2 追加：场景设计与被试方向锚点工作图 v0.3

- 用户查看版报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/scene_design_working_map_v0_3_cn.md`
- 场景事件来源工作表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/scene_trigger_audit_v0_2/tables/ego_direction_scene_event_source_map_v0_3.csv`
- 更新后的用户总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage02_scene_trigger_user_summary_cn.md`
- 依赖的完整触发审计报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/scene_trigger_audit_v0_2_cn.md`
- 依赖的触发审计表目录：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/scene_trigger_audit_v0_2/tables`
- 说明：根据用户最新补充，`longstraight` 25/26 普通连续车流按背景处理，但 MAN TGL 25->26 变道和 Chrysler300 Stop 要进入候选锚点审查；`fix_road` 也已确认存在显式变道触发。
## 阶段 2 追加：被试方向设计点与候选锚点重建 v0.4

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage02_ego_direction_design_anchor_user_summary_cn.md`
- 完整中文报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/ego_direction_design_anchor_rebuild_v0_4_cn.md`
- 小论文场景依据摘录：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/small_paper_scene_design_extract_v0_4.md`
- 审计脚本：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/scripts/build_ego_direction_design_anchors_v0_4.py`
- 小论文场景表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/ego_direction_design_anchor_v0_4/tables/small_paper_scene_design_tables_v0_4.csv`
- 配置车道/附着表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/ego_direction_design_anchor_v0_4/tables/cfg_lane_mu_geometry_v0_4.csv`
- 被试方向低附着段表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/ego_direction_design_anchor_v0_4/tables/cfg_ego_direction_mu_segments_v0_4.csv`
- 候选锚点清单：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/ego_direction_design_anchor_v0_4/tables/ego_direction_design_anchor_candidates_v0_4.csv`
- 场景模块汇总表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/ego_direction_design_anchor_v0_4/tables/ego_direction_design_anchor_module_summary_v0_4.csv`
- 说明：本轮确认小论文中弯道、低附着、急停、施工/维修、汇入等场景设计可作为锚点重建依据；根据后续用户补充，`middle_section`、`longstraight` 和 `fix_road` 也已纳入高优先级候选锚点审查。

### middle_section 连续超车修正

- 修正说明：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/middle_section_continuous_overtaking_correction_20260512_cn.md`
- 更新后的候选锚点清单：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/ego_direction_design_anchor_v0_4/tables/ego_direction_design_anchor_candidates_v0_4.csv`
- 更新后的模块汇总表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/ego_direction_design_anchor_v0_4/tables/ego_direction_design_anchor_module_summary_v0_4.csv`
- 说明：用户确认道路连接段存在连续超车事件，因此 `middle_section` 已从“背景/过渡段”修正为“连续超车负荷事件段”。当前新增连续超车段入口、中点、横向偏移变化峰值、横向加速度峰值、横摆角速度峰值五类候选。

### longstraight 与维修路段变道触发修正

- 修正说明：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/longstraight_fixroad_lanechange_trigger_correction_20260512_cn.md`
- 更新后的候选锚点清单：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/ego_direction_design_anchor_v0_4/tables/ego_direction_design_anchor_candidates_v0_4.csv`
- 更新后的模块汇总表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/ego_direction_design_anchor_v0_4/tables/ego_direction_design_anchor_module_summary_v0_4.csv`
- 更新后的场景事件来源工作表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/scene_trigger_audit_v0_2/tables/ego_direction_scene_event_source_map_v0_3.csv`
- 说明：用户确认 `longstraight` 和维修路段都涉及变道触发点。当前 `longstraight` 已新增显式变道/停车候选，`fix_road` 已新增两类显式变道候选。候选总数更新为 4519 行。

## 阶段 2 追加：事件候选筛选 v0.5

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage02_event_filter_user_summary_cn.md`
- 完整中文报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/event_candidate_filter_v0_5_cn.md`
- 筛选脚本：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/scripts/filter_event_anchor_candidates_v0_5.py`
- 全部候选评分表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/event_candidate_filter_v0_5/tables/event_candidate_scores_v0_5.csv`
- 去重后复核清单：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/event_candidate_filter_v0_5/tables/event_candidates_for_review_v0_5.csv`
- 高置信复核清单：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/event_candidate_filter_v0_5/tables/event_candidates_high_confidence_v0_5.csv`
- 分场景汇总：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/event_candidate_filter_v0_5/tables/event_candidate_module_summary_v0_5.csv`
- 分类型汇总：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/event_candidate_filter_v0_5/tables/event_candidate_decision_summary_v0_5.csv`
- 复核图索引：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/event_candidate_filter_v0_5/tables/event_candidate_review_panel_index_v0_5.csv`
- 概览图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/event_candidate_filter_v0_5/figures/event_candidate_filter_overview_v0_5.png`
- 代表性复核图目录：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/event_candidate_filter_v0_5/figures/review_panels`
- 说明：本轮没有训练模型，只是把 4519 个候选锚点按设计证据、车身响应、窗口可用性和旧锚点接近程度进行初筛。去重后建议复核 534 个，高置信复核 314 个。

## GPTPro 事件锚点审查证据包（2026-05-12）

- 证据包目录：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/gptpro_event_anchor_review_pack_20260512`
- 压缩包：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/gptpro_event_anchor_review_pack_20260512.zip`
- README：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/gptpro_event_anchor_review_pack_20260512/00_README_CN.md`
- GPTPro 提问词：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/gptpro_event_anchor_review_pack_20260512/01_GPTPRO_PROMPT_CN.md`
- 说明：该包包含事件筛选中文报告、核心表格、概览图和按场景精选的 19 张复核图，不包含原始数据、模型 checkpoint、服务器密码或连接凭据。

## GPTPro 事件锚点审查回复与 v0.6 规则（2026-05-12）

- GPTPro 回复归档：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/gptpro_event_anchor_reply_20260512/20260512_event_anchor_v05_response_manualpaste.md`
- 已填充决策记录：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/gptpro_event_anchor_reply_20260512/20260512_event_anchor_v05_decision_filled.md`
- 已填充行动项：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/gptpro_event_anchor_reply_20260512/20260512_event_anchor_v05_action_items_filled.md`
- v0.6 筛选规则草案：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/event_v0_6_screening_rule_from_gptpro_20260512_cn.md`
- 说明：GPTPro 支持先重审事件锚点，建议 v0.6 输出四类事件表，并先用小而干净的核心样本训练车辆/道路-only 基线。
## 阶段 3 追加：干净响应任务车辆-only 基线 v0.1（2026-05-13）

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_clean_task_vehicle_baselines_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_clean_task_vehicle_baselines_v0_1_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_vehicle_instability_clean_task_vehicle_baselines_v0_1.py`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_baselines_v0_1/logs/clean_task_vehicle_baselines_summary.json`
- 指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_baselines_v0_1/tables/clean_task_vehicle_metrics.csv`
- 逐样本指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_baselines_v0_1/tables/clean_task_vehicle_per_sample_metrics.csv`
- 模型信息表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_baselines_v0_1/tables/clean_task_vehicle_model_info.csv`
- 任务轨道汇总：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_baselines_v0_1/tables/clean_task_track_summary.csv`
- val 选择模型表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_baselines_v0_1/tables/clean_task_vehicle_val_selected_models.csv`
- test 指标概览图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_baselines_v0_1/figures/clean_task_vehicle_metric_summary_test.png`
- A 轨道固定预测图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_baselines_v0_1/figures/A_instant2s_core_fixed_predictions_test.png`
- A 轨道坏样本图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_baselines_v0_1/figures/A_instant2s_core_bad_samples_test.png`
- B 轨道固定预测图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_baselines_v0_1/figures/B_response3s_strict_core_fixed_predictions_test.png`
- B 轨道坏样本图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_baselines_v0_1/figures/B_response3s_strict_core_bad_samples_test.png`
- 重要 Git commit：待提交。
- 适合用户/老师直接查看的材料：优先看用户查看版总结、test 指标概览图和 B 轨道坏样本图。
## 阶段 3 追加：B 轨道 RBF KRR 坏样本物理复查 v0.1（2026-05-13）

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_clean_task_bad_sample_review_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_clean_task_bad_sample_review_v0_1_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_vehicle_instability_clean_task_bad_sample_review_v0_1.py`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_bad_sample_review_v0_1/logs/clean_task_bad_sample_review_summary.json`
- B 轨道坏样本总表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_bad_sample_review_v0_1/tables/b_track_rbf_bad_sample_table.csv`
- 失败标记汇总：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_bad_sample_review_v0_1/tables/b_track_rbf_failure_summary.csv`
- top bad 样本表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_bad_sample_review_v0_1/tables/b_track_rbf_top_bad_samples.csv`
- 分响应形态汇总：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_bad_sample_review_v0_1/tables/b_track_rbf_failure_by_morphology.csv`
- 分被试汇总：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_bad_sample_review_v0_1/tables/b_track_rbf_failure_by_subject.csv`
- 分道路模块汇总：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_bad_sample_review_v0_1/tables/b_track_rbf_failure_by_road_module.csv`
- 失败标记率图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_bad_sample_review_v0_1/figures/b_track_rbf_failure_flag_rates.png`
- top bad RMSE 图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_bad_sample_review_v0_1/figures/b_track_rbf_top_bad_rmse.png`
- 主峰幅值散点图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_bad_sample_review_v0_1/figures/b_track_rbf_peak_amp_scatter.png`
- 重要 Git commit：待提交。
- 适合用户/老师直接查看的材料：优先看用户查看版总结、失败标记率图、top bad 样本表和主峰幅值散点图。

## 阶段 2 回补：episode-first 事件样本 v0.6（2026-05-13）

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage02_episode_first_v0_6_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/episode_first_event_v0_6_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/scripts/build_episode_first_events_v0_6.py`
- episode 总表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/episode_first_event_v0_6/tables/episode_candidates_v0_6.csv`
- 第一版最干净核心候选：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/episode_first_event_v0_6/tables/primary_training_events_v0_6.csv`
- 坐标需复核扩展候选：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/episode_first_event_v0_6/tables/coordinate_flagged_expansion_events_v0_6.csv`
- 弱响应/负样本候选：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/episode_first_event_v0_6/tables/trigger_no_effect_or_weak_response_v0_6.csv`
- 分桶汇总表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/episode_first_event_v0_6/tables/episode_decision_summary_v0_6.csv`
- 类型汇总表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/episode_first_event_v0_6/tables/episode_label_summary_v0_6.csv`
- 场景汇总表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/episode_first_event_v0_6/tables/episode_module_summary_v0_6.csv`
- 代表图索引：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/episode_first_event_v0_6/tables/episode_review_panel_index_v0_6.csv`
- 概览图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/episode_first_event_v0_6/figures/episode_first_v0_6_summary.png`
- 分组代表图目录：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/episode_first_event_v0_6/figures/episode_review_panels`
- 重要 Git commit：待提交。
- 适合用户/老师直接查看的材料：优先看用户查看版总结、概览图、严格核心表、坐标需复核扩展表和分组代表图目录。

## 阶段 3 追加：episode-first v0.6 纯车辆/道路预测对照 v0.1（2026-05-13）

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_episode_first_vehicle_baselines_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_episode_first_vehicle_baselines_v0_1_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_episode_first_vehicle_baselines_v0_1.py`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_episode_first_vehicle_baselines_v0_1/logs/episode_first_vehicle_baselines_summary.json`
- 指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_episode_first_vehicle_baselines_v0_1/tables/episode_first_vehicle_metrics.csv`
- 逐样本指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_episode_first_vehicle_baselines_v0_1/tables/episode_first_vehicle_per_sample_metrics.csv`
- 模型信息表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_episode_first_vehicle_baselines_v0_1/tables/episode_first_vehicle_model_info.csv`
- 轨道汇总表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_episode_first_vehicle_baselines_v0_1/tables/episode_first_track_summary.csv`
- val 选择表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_episode_first_vehicle_baselines_v0_1/tables/episode_first_vehicle_val_selected_models.csv`
- test 指标概览图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_episode_first_vehicle_baselines_v0_1/figures/episode_first_vehicle_metric_summary_test.png`
- 3 秒不使用横向偏移坏样本图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_episode_first_vehicle_baselines_v0_1/figures/EP3_expanded_no_lateral_3s_bad_samples_test.png`
- 重要 Git commit：待提交。
- 适合用户/老师直接查看的材料：优先看用户查看版总结、val 选择表、指标概览图和 3 秒不使用横向偏移坏样本图。

## 目标完成审计：事件锚点筛选与样本重建（2026-05-13）

- 完成审计报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/goal_completion_audit_event_v0_6_20260513_cn.md`
- 审计结论：v0.6 样本清单、四类/多类分桶、复核图、分层统计、物理指标和纯车辆/道路预测对照均已完成；车辆-only 指标未优于旧 B 轨道，但证明 v0.6 更集中在复杂真实 episode，下一阶段应进入车辆-only 响应分解模型。

## 最新新增：阶段 4 连续驾驶风格探索性增量对照 v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage04_style_increment_exploratory_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage04_style_increment_exploratory_v0_1_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/scripts/stage04_style_increment_exploratory_v0_1.py`
- 指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_increment_exploratory_v0_1/tables/style_increment_metrics.csv`
- 逐样本表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_increment_exploratory_v0_1/tables/style_increment_per_sample_metrics.csv`
- 置乱汇总：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_increment_exploratory_v0_1/tables/style_increment_permutation_summary.csv`
- gate 表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_increment_exploratory_v0_1/tables/style_increment_gate_table.csv`
- 固定预测图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_increment_exploratory_v0_1/figures/style_increment_fixed_predictions_test.png`
- 坏样本图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_increment_exploratory_v0_1/figures/style_increment_bad_samples_test.png`
- 服务器日志：无，本轮未使用远程服务器。
- GPTPro 提问和回复：无。
- 重要 Git commit：待提交。
- 适合用户/老师直接查看：用户查看版总结、gate 表、指标概览图、固定预测图、坏样本图。

## 最新新增：阶段 4 连续风格跨 split 复核 v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage04_style_cross_split_validation_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage04_style_cross_split_validation_v0_1_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/scripts/stage04_style_cross_split_validation_v0_1.py`
- 指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_cross_split_validation_v0_1/tables/style_cross_split_metrics.csv`
- 逐样本表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_cross_split_validation_v0_1/tables/style_cross_split_per_sample_metrics.csv`
- gate 表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_cross_split_validation_v0_1/tables/style_cross_split_gate_table.csv`
- 指标图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_cross_split_validation_v0_1/figures/style_cross_split_metric_summary_test.png`
- subject-level 坏样本图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_cross_split_validation_v0_1/figures/style_cross_split_subject_bad_samples_test.png`
- 服务器日志：无，本轮未使用远程服务器。
- GPTPro 提问和回复：无。
- 重要 Git commit：待提交。
- 适合用户/老师直接查看：用户查看版总结、gate 表、指标图、subject-level 坏样本图。

## 最新新增：阶段 4 连续风格路线收口决策 v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage04_style_route_decision_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage04_style_route_decision_v0_1_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/scripts/stage04_style_route_decision_v0_1.py`
- 证据摘要表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_route_decision_v0_1/tables/style_route_evidence_summary.csv`
- gate 表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_route_decision_v0_1/tables/style_route_decision_gate_table.csv`
- 下一步动作表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_route_decision_v0_1/tables/style_route_next_actions.csv`
- RMSE delta 图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_route_decision_v0_1/figures/style_route_rmse_delta_summary.png`
- 服务器日志：无，本轮未使用远程服务器。
- GPTPro 提问和回复：无。
- 重要 Git commit：`4064bf64 Add style route decision`。
- 适合用户/老师直接查看：用户查看版总结、gate 表、RMSE delta 图、下一步动作表。
# R2E-Steering 阶段产物索引
## 最新更新：2026-05-13 07:42

## Stage 7h val/test 选择不稳定诊断 v0.1

- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/scripts/stage07h_val_test_selection_diagnostics_v0_1.py`
- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage07h_val_test_selection_diagnostics_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage07h_val_test_selection_diagnostics_v0_1_cn.md`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07h_val_test_selection_diagnostics_v0_1/logs/stage07h_val_test_selection_diagnostics_summary.json`
- 候选 split 稳定性：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07h_val_test_selection_diagnostics_v0_1/tables/stage07h_candidate_split_stability.csv`
- 类别分布长表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07h_val_test_selection_diagnostics_v0_1/tables/stage07h_categorical_shift_long.csv`
- 类别分布摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07h_val_test_selection_diagnostics_v0_1/tables/stage07h_categorical_shift_summary.csv`
- 数值分布摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07h_val_test_selection_diagnostics_v0_1/tables/stage07h_numeric_shift_summary.csv`
- 逐样本候选收益：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07h_val_test_selection_diagnostics_v0_1/tables/stage07h_candidate_gain_samples.csv`
- 分 bucket 候选收益：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07h_val_test_selection_diagnostics_v0_1/tables/stage07h_candidate_gain_by_bucket.csv`
- keypoint target 指标副本：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07h_val_test_selection_diagnostics_v0_1/tables/stage07h_keypoint_target_metrics_copy.csv`
- gate 表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07h_val_test_selection_diagnostics_v0_1/tables/stage07h_gate_table.csv`
- 候选 val/test 稳定性图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07h_val_test_selection_diagnostics_v0_1/figures/stage07h_candidate_val_test_stability.png`
- val/test 类别偏移图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07h_val_test_selection_diagnostics_v0_1/figures/stage07h_val_test_categorical_shift.png`
- 候选逐样本收益图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07h_val_test_selection_diagnostics_v0_1/figures/stage07h_candidate_gain_by_split.png`
- keypoint target RMSE 图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07h_val_test_selection_diagnostics_v0_1/figures/stage07h_keypoint_target_rmse_by_split.png`
- GPTPro 提问和回复：本阶段未调用。
- 服务器日志：本阶段未使用服务器。
- 重要 Git commit：`d990f8e3 Add stage7h selection diagnostics`。
- 适合用户/老师直接查看：优先看用户查看版总结、gate 表、候选稳定性表、类别/数值偏移表和候选稳定性图。

## 最新更新：2026-05-13 07:33

## Stage 7g keypoint/segment 车辆-only 候选 v0.1

- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/scripts/stage07g_keypoint_segment_candidates_v0_1.py`
- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage07g_keypoint_segment_candidates_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage07g_keypoint_segment_candidates_v0_1_cn.md`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07g_keypoint_segment_candidates_v0_1/logs/stage07g_keypoint_segment_candidates_summary.json`
- feature audit：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07g_keypoint_segment_candidates_v0_1/tables/stage07g_feature_audit.csv`
- allowed features：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07g_keypoint_segment_candidates_v0_1/tables/stage07g_allowed_features.csv`
- 关键点预测表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07g_keypoint_segment_candidates_v0_1/tables/stage07g_keypoint_prediction_table.csv`
- 关键点 target 指标：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07g_keypoint_segment_candidates_v0_1/tables/stage07g_keypoint_target_metrics.csv`
- 候选指标：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07g_keypoint_segment_candidates_v0_1/tables/stage07g_candidate_metrics.csv`
- 候选逐样本指标：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07g_keypoint_segment_candidates_v0_1/tables/stage07g_candidate_per_sample_metrics.csv`
- validation selection：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07g_keypoint_segment_candidates_v0_1/tables/stage07g_validation_selection_table.csv`
- oracle 诊断：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07g_keypoint_segment_candidates_v0_1/tables/stage07g_oracle_diag.csv`
- gate 表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07g_keypoint_segment_candidates_v0_1/tables/stage07g_gate_table.csv`
- 指标图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07g_keypoint_segment_candidates_v0_1/figures/stage07g_metric_summary_test.png`
- 关键点散点图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07g_keypoint_segment_candidates_v0_1/figures/stage07g_keypoint_target_scatter.png`
- 固定预测图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07g_keypoint_segment_candidates_v0_1/figures/stage07g_fixed_predictions_test.png`
- oracle gain 预测图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07g_keypoint_segment_candidates_v0_1/figures/stage07g_oracle_gain_predictions_test.png`
- GPTPro 提问和回复：本阶段未调用。
- 服务器日志：本阶段未使用服务器。
- 重要 Git commit：`52de7176 Add stage7g keypoint segment candidates`。
- 适合用户/老师直接查看：优先看用户查看版总结、gate 表、候选指标、关键点散点图和 oracle gain 预测图。

## 最新更新：2026-05-13 07:19

## Stage 7f response-factorized vehicle-only candidate v0.1

- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/scripts/stage07f_response_factorized_candidates_v0_1.py`
- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage07f_response_factorized_candidates_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage07f_response_factorized_candidates_v0_1_cn.md`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07f_response_factorized_candidates_v0_1/logs/stage07f_response_factorized_candidates_summary.json`
- feature audit：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07f_response_factorized_candidates_v0_1/tables/stage07f_feature_audit.csv`
- allowed features：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07f_response_factorized_candidates_v0_1/tables/stage07f_allowed_features.csv`
- factor 预测指标：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07f_response_factorized_candidates_v0_1/tables/stage07f_factor_prediction_metrics.csv`
- factor 预测明细：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07f_response_factorized_candidates_v0_1/tables/stage07f_factor_predictions_long.csv`
- 候选逐样本指标：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07f_response_factorized_candidates_v0_1/tables/stage07f_candidate_per_sample_metrics.csv`
- policy 指标：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07f_response_factorized_candidates_v0_1/tables/stage07f_policy_metrics.csv`
- policy 逐样本指标：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07f_response_factorized_candidates_v0_1/tables/stage07f_policy_per_sample_metrics.csv`
- policy 和候选总表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07f_response_factorized_candidates_v0_1/tables/stage07f_policy_and_candidate_metrics.csv`
- validation selection：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07f_response_factorized_candidates_v0_1/tables/stage07f_validation_selection_table.csv`
- response-factorized oracle 诊断：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07f_response_factorized_candidates_v0_1/tables/stage07f_response_factorized_oracle_diag.csv`
- combo oracle 诊断：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07f_response_factorized_candidates_v0_1/tables/stage07f_combo_oracle_diag.csv`
- prototype trace：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07f_response_factorized_candidates_v0_1/tables/stage07f_prototype_trace.csv`
- gate 表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07f_response_factorized_candidates_v0_1/tables/stage07f_gate_table.csv`
- 指标图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07f_response_factorized_candidates_v0_1/figures/stage07f_metric_summary_test.png`
- 固定预测图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07f_response_factorized_candidates_v0_1/figures/stage07f_fixed_predictions_test.png`
- oracle gain 预测图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07f_response_factorized_candidates_v0_1/figures/stage07f_oracle_gain_predictions_test.png`
- GPTPro 提问和回复：本阶段未调用。
- 服务器日志：本阶段未使用服务器。
- 重要 Git commit：`12cef06b Add stage7f response factorized candidates`。
- 适合用户/老师直接查看：优先看用户查看版总结、gate 表、factor 预测指标、固定预测图和 oracle gain 预测图。

## 最新更新：2026-05-13 06:50

## Stage 7c 候选轨迹导出与差异审计 v0.1

- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/scripts/stage07c_candidate_trajectory_export_v0_1.py`
- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage07c_candidate_trajectory_export_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage07c_candidate_trajectory_export_v0_1_cn.md`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07c_candidate_trajectory_export_v0_1/logs/stage07c_candidate_trajectory_export_summary.json`
- 轨迹数组：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07c_candidate_trajectory_export_v0_1/arrays/stage07c_candidate_trajectories.npz`
- 指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07c_candidate_trajectory_export_v0_1/tables/candidate_export_metrics.csv`
- 逐样本指标：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07c_candidate_trajectory_export_v0_1/tables/candidate_export_per_sample_metrics.csv`
- 候选两两差异明细：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07c_candidate_trajectory_export_v0_1/tables/candidate_pairwise_disagreement_long.csv`
- 候选两两差异摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07c_candidate_trajectory_export_v0_1/tables/candidate_pairwise_disagreement_summary.csv`
- 候选特征与标签诊断：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07c_candidate_trajectory_export_v0_1/tables/candidate_feature_and_label_diagnosis.csv`
- oracle 摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07c_candidate_trajectory_export_v0_1/tables/candidate_oracle_summary.csv`
- gate 表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07c_candidate_trajectory_export_v0_1/tables/candidate_export_gate_table.csv`
- 固定样本图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07c_candidate_trajectory_export_v0_1/figures/candidate_fixed_predictions_test.png`
- 高候选差异图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07c_candidate_trajectory_export_v0_1/figures/candidate_high_disagreement_predictions_test.png`
- oracle gain 图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07c_candidate_trajectory_export_v0_1/figures/candidate_oracle_gain_predictions_test.png`
- 指标图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07c_candidate_trajectory_export_v0_1/figures/candidate_metric_summary_test.png`
- 差异-上限散点图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07c_candidate_trajectory_export_v0_1/figures/candidate_disagreement_vs_oracle_gain_test.png`
- GPTPro 提问和回复：本阶段未调用。
- 服务器日志：本阶段未使用服务器。
- 重要 Git commit：`48b8c438 Add stage7c candidate trajectory export`。
- 适合用户/老师直接查看：优先看用户查看版总结、gate 表、指标图、oracle gain 图和轨迹数组说明。
# R2E-Steering 阶段产物索引
## 最新更新：2026-05-13 06:58

## Stage 7d 非 oracle selector v0.2

- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/scripts/stage07d_non_oracle_selector_v0_2.py`
- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage07d_non_oracle_selector_v0_2_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage07d_non_oracle_selector_v0_2_cn.md`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07d_non_oracle_selector_v0_2/logs/stage07d_non_oracle_selector_summary.json`
- feature audit：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07d_non_oracle_selector_v0_2/tables/stage07d_feature_audit.csv`
- allowed features：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07d_non_oracle_selector_v0_2/tables/stage07d_allowed_features.csv`
- policy metrics：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07d_non_oracle_selector_v0_2/tables/stage07d_policy_metrics.csv`
- decision diagnostics：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07d_non_oracle_selector_v0_2/tables/stage07d_decision_diagnostics.csv`
- selected decisions：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07d_non_oracle_selector_v0_2/tables/stage07d_selected_policy_decisions.csv`
- validation selection：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07d_non_oracle_selector_v0_2/tables/stage07d_validation_selection_table.csv`
- gate 表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07d_non_oracle_selector_v0_2/tables/stage07d_gate_table.csv`
- 指标图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07d_non_oracle_selector_v0_2/figures/stage07d_policy_metrics_test.png`
- val delta 图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07d_non_oracle_selector_v0_2/figures/stage07d_validation_rmse_delta.png`
- 选择计数图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07d_non_oracle_selector_v0_2/figures/stage07d_selected_choice_counts.png`
- GPTPro 提问和回复：本阶段未调用。
- 服务器日志：本阶段未使用服务器。
- 重要 Git commit：`eb785f4a Add stage7d non-oracle selector`。
- 适合用户/老师直接查看：优先看用户查看版总结、gate 表、val delta 图和 policy metrics 图。
# R2E-Steering 阶段产物索引
## 最新更新：2026-05-13 07:05

## Stage 7e 候选生成重设计审计 v0.1

- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/scripts/stage07e_candidate_generation_redesign_v0_1.py`
- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage07e_candidate_generation_redesign_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage07e_candidate_generation_redesign_v0_1_cn.md`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07e_candidate_generation_redesign_v0_1/logs/stage07e_candidate_generation_redesign_summary.json`
- 响应类型表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07e_candidate_generation_redesign_v0_1/tables/stage07e_response_label_table.csv`
- 样本候选缺口表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07e_candidate_generation_redesign_v0_1/tables/stage07e_sample_candidate_gap_table.csv`
- bucket 覆盖表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07e_candidate_generation_redesign_v0_1/tables/stage07e_existing_candidate_coverage_by_bucket.csv`
- oracle winner 分布：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07e_candidate_generation_redesign_v0_1/tables/stage07e_oracle_winner_distribution.csv`
- 候选生成蓝图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07e_candidate_generation_redesign_v0_1/tables/stage07e_candidate_generation_blueprint.csv`
- 下一实验计划：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07e_candidate_generation_redesign_v0_1/tables/stage07e_next_experiment_plan.csv`
- gate 表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07e_candidate_generation_redesign_v0_1/tables/stage07e_gate_table.csv`
- oracle gain by family 图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07e_candidate_generation_redesign_v0_1/figures/stage07e_oracle_gain_by_response_family_test.png`
- winner 分布图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07e_candidate_generation_redesign_v0_1/figures/stage07e_oracle_winner_distribution_test.png`
- 样本缺口散点图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07e_candidate_generation_redesign_v0_1/figures/stage07e_candidate_gap_scatter_test.png`
- 候选生成蓝图图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07e_candidate_generation_redesign_v0_1/figures/stage07e_candidate_generation_blueprint.png`
- GPTPro 提问和回复：本阶段未调用。
- 服务器日志：本阶段未使用服务器。
- 重要 Git commit：`98552bf3 Add stage7e candidate generation redesign`。
- 适合用户/老师直接查看：优先看用户查看版总结、候选生成蓝图、gate 表、oracle gain 图和 winner 分布图。


## v0.3 全量原始数据极限工况 episode 重筛

- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage02_extreme_condition_episode_v0_3_user_summary_cn.md`
- 技术报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\extreme_condition_episode_v0_3_cn.md`
- 总表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\extreme_condition_episodes_v0_3\tables\extreme_condition_episodes_all_v0_3.csv`
- 复核图索引：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\extreme_condition_episodes_v0_3\tables\extreme_condition_review_panel_index_v0_3.csv`
- 分类统计图：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\extreme_condition_episodes_v0_3\figures\extreme_condition_category_counts_v0_3.png`


## v0.3 全量原始数据极限工况 episode 重筛

- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage02_extreme_condition_episode_v0_3_user_summary_cn.md`
- 技术报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\extreme_condition_episode_v0_3_cn.md`
- 总表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\extreme_condition_episodes_v0_3\tables\extreme_condition_episodes_all_v0_3.csv`
- 复核图索引：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\extreme_condition_episodes_v0_3\tables\extreme_condition_review_panel_index_v0_3.csv`
- 分类统计图：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\extreme_condition_episodes_v0_3\figures\extreme_condition_category_counts_v0_3.png`


## v0.3 全量原始数据极限工况 episode 重筛

- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage02_extreme_condition_episode_v0_3_user_summary_cn.md`
- 技术报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\extreme_condition_episode_v0_3_cn.md`
- 总表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\extreme_condition_episodes_v0_3\tables\extreme_condition_episodes_all_v0_3.csv`
- 复核图索引：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\extreme_condition_episodes_v0_3\tables\extreme_condition_review_panel_index_v0_3.csv`
- 分类统计图：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\extreme_condition_episodes_v0_3\figures\extreme_condition_category_counts_v0_3.png`


## v0.3 全量原始数据极限工况 episode 重筛

- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage02_extreme_condition_episode_v0_3_user_summary_cn.md`
- 技术报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\extreme_condition_episode_v0_3_cn.md`
- 总表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\extreme_condition_episodes_v0_3\tables\extreme_condition_episodes_all_v0_3.csv`
- 复核图索引：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\extreme_condition_episodes_v0_3\tables\extreme_condition_review_panel_index_v0_3.csv`
- 分类统计图：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\extreme_condition_episodes_v0_3\figures\extreme_condition_category_counts_v0_3.png`


## v0.3 车辆-only 数据集与基线

- 数据集数组：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_processed_datasets\extreme_condition_v0_3_vehicle_only\arrays\v03_vehicle_only_pre2_label5_20hz.npz`
- 数据集 manifest：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_processed_datasets\extreme_condition_v0_3_vehicle_only\tables\v03_vehicle_only_manifest.csv`
- 指标表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_vehicle_only_baselines\tables\v03_vehicle_only_baseline_metrics.csv`
- 逐样本指标：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_vehicle_only_baselines\tables\v03_vehicle_only_per_sample_metrics.csv`
- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage03_v03_vehicle_only_baselines_user_summary_cn.md`
- 固定预测图：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_vehicle_only_baselines\figures\v03_vehicle_only_fixed_predictions_test.png`
- 坏样本图：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_vehicle_only_baselines\figures\v03_vehicle_only_bad_samples_test.png`


## v0.3 车辆-only 数据集与基线（中文修正版）

- 数据集数组：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_processed_datasets\extreme_condition_v0_3_vehicle_only\arrays\v03_vehicle_only_pre2_label5_20hz.npz`
- 数据集 manifest：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_processed_datasets\extreme_condition_v0_3_vehicle_only\tables\v03_vehicle_only_manifest.csv`
- 总指标表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_vehicle_only_baselines\tables\v03_vehicle_only_baseline_metrics.csv`
- 逐样本指标：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_vehicle_only_baselines\tables\v03_vehicle_only_per_sample_metrics.csv`
- 分样本类型表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_vehicle_only_baselines\tables\v03_vehicle_only_best_model_by_category_test.csv`
- 分被试表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_vehicle_only_baselines\tables\v03_vehicle_only_best_model_by_subject_test.csv`
- 分工况上下文表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_vehicle_only_baselines\tables\v03_vehicle_only_best_model_by_context_test.csv`
- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage03_v03_vehicle_only_baselines_user_summary_cn.md`
- 固定预测图：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_vehicle_only_baselines\figures\v03_vehicle_only_fixed_predictions_test.png`
- 坏样本图：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_vehicle_only_baselines\figures\v03_vehicle_only_bad_samples_test.png`


## v0.3 车辆-only 数据集与基线（中文修正版）

- 数据集数组：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_processed_datasets\extreme_condition_v0_3_vehicle_only\arrays\v03_vehicle_only_pre2_label5_20hz.npz`
- 数据集 manifest：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_processed_datasets\extreme_condition_v0_3_vehicle_only\tables\v03_vehicle_only_manifest.csv`
- 总指标表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_vehicle_only_baselines\tables\v03_vehicle_only_baseline_metrics.csv`
- 逐样本指标：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_vehicle_only_baselines\tables\v03_vehicle_only_per_sample_metrics.csv`
- 分样本类型表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_vehicle_only_baselines\tables\v03_vehicle_only_best_model_by_category_test.csv`
- 分被试表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_vehicle_only_baselines\tables\v03_vehicle_only_best_model_by_subject_test.csv`
- 分工况上下文表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_vehicle_only_baselines\tables\v03_vehicle_only_best_model_by_context_test.csv`
- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage03_v03_vehicle_only_baselines_user_summary_cn.md`
- 固定预测图：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_vehicle_only_baselines\figures\v03_vehicle_only_fixed_predictions_test.png`
- 坏样本图：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_vehicle_only_baselines\figures\v03_vehicle_only_bad_samples_test.png`

## v0.3 样本纳入范围消融

- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage03_v03_vehicle_only_inclusion_ablation_user_summary_cn.md`
- 汇总表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_vehicle_only_inclusion_ablation\tables\v03_vehicle_only_inclusion_ablation_summary.csv`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_vehicle_only_inclusion_ablation`

## v0.3 excluded 分层加入实验

- 用户查看版报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_v03_excluded_stratified_inclusion_user_summary_cn.md`
- 汇总表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_v03_excluded_stratified_inclusion/tables/v03_excluded_stratified_inclusion_summary.csv`
- 输出目录：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_v03_excluded_stratified_inclusion`
- 服务器日志本地副本：`F:/data_set_process/data_process/04_project_logs/reports/server_logs/v03_excluded_stratified_20260519/run.log`

## 横滚/姿态 excluded paired 诊断

- 用户查看版报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_v03_roll_excluded_pair_diagnosis_user_summary_cn.md`
- paired 明细表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_v03_roll_excluded_pair_diagnosis/tables/roll_vs_ref_common_test_paired_metrics.csv`
- 输出目录：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_v03_roll_excluded_pair_diagnosis`

## v0.3 极限工况样本人工复核清单

- 复核指南：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/manual_review_guide_extreme_condition_v0_3_cn.md`
- 优先复核清单：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/extreme_condition_episodes_v0_3/tables/manual_review_priority_list_v0_3.csv`
- 复核图目录：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/extreme_condition_episodes_v0_3/figures/review_panels`

## 新人工规则下的 v0.3 自动候选分组

- 中文说明：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/new_rule_auto_candidate_groups_v0_3_cn.md`
- 全量分组表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/extreme_condition_episodes_v0_3/tables/new_rule_auto_candidate_groups_v0_3.csv`
- 分组数量表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/extreme_condition_episodes_v0_3/tables/new_rule_auto_candidate_group_summary_v0_3.csv`
- 每组代表样本：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/extreme_condition_episodes_v0_3/tables/new_rule_auto_candidate_representatives_v0_3.csv`
- 分组图片目录：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/extreme_condition_episodes_v0_3/figures/new_rule_review_groups_v0_3`
- 分组图片说明：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/extreme_condition_episodes_v0_3/figures/new_rule_review_groups_v0_3/00_先看这里_图片说明.md`
- 分组图片索引：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/extreme_condition_episodes_v0_3/figures/new_rule_review_groups_v0_3/new_rule_review_image_index_v0_3.csv`

## v0.3 方向盘角速度候选复核

- 中文说明：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/new_rule_fast_steer_candidates_v0_3_cn.md`
- 候选表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/extreme_condition_episodes_v0_3/tables/new_rule_fast_steer_candidates_v0_3.csv`
- 数量表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/extreme_condition_episodes_v0_3/tables/new_rule_fast_steer_candidate_summary_v0_3.csv`
- 图片目录：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/extreme_condition_episodes_v0_3/figures/fast_steer_review_v0_3`
- 图片说明：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/extreme_condition_episodes_v0_3/figures/fast_steer_review_v0_3/00_先看这里_方向盘角速度候选说明.md`
- 图片索引：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/extreme_condition_episodes_v0_3/figures/fast_steer_review_v0_3/fast_steer_review_image_index_v0_3.csv`

## v0.3 快速转向候选按车辆响应强弱拆分

- 中文说明：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/fast_steer_vehicle_response_split_v0_3_cn.md`
- 拆分表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/extreme_condition_episodes_v0_3/tables/fast_steer_vehicle_response_split_v0_3.csv`
- 数量表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/extreme_condition_episodes_v0_3/tables/fast_steer_vehicle_response_split_summary_v0_3.csv`
- 图片目录：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/extreme_condition_episodes_v0_3/figures/fast_steer_vehicle_response_split_v0_3`
- 图片说明：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/extreme_condition_episodes_v0_3/figures/fast_steer_vehicle_response_split_v0_3/00_先看这里_快速转向按车辆响应拆分说明.md`
- 图片索引：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/extreme_condition_episodes_v0_3/figures/fast_steer_vehicle_response_split_v0_3/fast_steer_vehicle_response_split_image_index_v0_3.csv`

## v0.3 快速转向候选锚点时序审计

- 中文说明：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/fast_steer_anchor_timing_audit_v0_3_cn.md`
- 审计表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/extreme_condition_episodes_v0_3/tables/fast_steer_anchor_timing_audit_v0_3.csv`
- 汇总表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/extreme_condition_episodes_v0_3/tables/fast_steer_anchor_timing_audit_summary_v0_3.csv`
- 图片目录：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/extreme_condition_episodes_v0_3/figures/fast_steer_anchor_timing_split_v0_3`
- 图片说明：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/extreme_condition_episodes_v0_3/figures/fast_steer_anchor_timing_split_v0_3/00_先看这里_锚点时序复核说明.md`
- 图片索引：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/extreme_condition_episodes_v0_3/figures/fast_steer_anchor_timing_split_v0_3/fast_steer_anchor_timing_image_index_v0_3.csv`

## v0.3 临时加入锚点后响应弱样本训练

- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage03_v03_fast_weakpost_temp_train_user_summary_cn.md`
- 汇总表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_fast_weakpost_temp_train\tables\v03_fast_weakpost_temp_train_summary.csv`
- 临时加入 episode 清单：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_fast_weakpost_temp_train\tables\v03_fast_weakpost_extra_episode_uids.csv`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_fast_weakpost_temp_train`
## v0.3 样本筛选策略连续对比

- 脚本：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_v03_screening_sweep.py`
- 服务器日志：`/root/autodl-tmp/data_process/05_rebuild_from_raw_20260511/00_project_notes/server_logs/stage03_v03_screening_sweep_20260519_203455.log`
- 预期输出目录：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_v03_screening_sweep`
- 当前状态：服务器运行中，结果待拉回。

## 2026-05-19 v0.4 极限工况样本重新筛选

- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage02_extreme_condition_refilter_v0_4_user_summary_cn.md`
- 总表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\extreme_condition_episodes_v0_4\tables\extreme_condition_episodes_refiltered_v0_4.csv`
- 主+次级训练候选：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\extreme_condition_episodes_v0_4\tables\train_candidate_episodes_v0_4.csv`
- 复核图目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\extreme_condition_episodes_v0_4\figures\review_panels`

## 2026-05-20 v0.4 重筛样本车辆-only GPU 基线

- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage03_v04_vehicle_only_gpu_user_summary_cn.md`
- 汇总表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v04_vehicle_only_gpu_baseline\tables\v04_vehicle_only_gpu_summary.csv`
- 排名表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v04_vehicle_only_gpu_baseline\tables\v04_vehicle_only_gpu_ranking.csv`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v04_vehicle_only_gpu_baseline`
## 2026-05-20 v0.4 主训练+次级+待复核 GPU 基线

- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage03_v04_review_gpu_user_summary_cn.md`
- 汇总表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v04_review_gpu_baseline\tables\v04_review_gpu_summary.csv`
- 排名表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v04_review_gpu_baseline\tables\v04_review_gpu_ranking.csv`
- 逐模型指标：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v04_review_gpu_baseline\v04_primary_secondary_review_nolat\tables\v04_primary_secondary_review_nolat_gpu_metrics.csv`
- 逐样本指标：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v04_review_gpu_baseline\v04_primary_secondary_review_nolat\tables\v04_primary_secondary_review_nolat_gpu_per_sample_metrics.csv`
- 服务器日志：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\00_project_notes\server_logs\stage03_v04_review_gpu_20260520_102550.log`
- 脚本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v04_review_gpu_baseline.py`


## 2026-05-20 13:48:32 v0.5 生理机制验证

已更新 v0.5 生理机制验证材料。

- 报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage03_v05_physio_mechanism_comparison_user_summary_cn.md`
- 运行状态表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\tables\v05_physio_run_status.csv`
- 对比表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\tables\v05_physio_comparison_table.csv`
- 机制表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\tables\v05_physio_mechanism_table.csv`


## 2026-05-20 13:58:46 v0.5 生理机制验证

已更新 v0.5 生理机制验证材料。

- 报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage03_v05_physio_mechanism_comparison_user_summary_cn.md`
- 运行状态表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\tables\v05_physio_run_status.csv`
- 对比表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\tables\v05_physio_comparison_table.csv`
- 机制表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\tables\v05_physio_mechanism_table.csv`


## 2026-05-20 15:10:44 v0.5 生理机制验证

已更新 v0.5 生理机制验证材料。

- 报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage03_v05_physio_mechanism_comparison_user_summary_cn.md`
- 运行状态表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\tables\v05_physio_run_status.csv`
- 对比表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\tables\v05_physio_comparison_table.csv`
- 机制表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\tables\v05_physio_mechanism_table.csv`


## 2026-05-20 15:14:24 v0.5 生理机制验证

已更新 v0.5 生理机制验证材料。

- 报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage03_v05_physio_mechanism_comparison_user_summary_cn.md`
- 运行状态表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\tables\v05_physio_run_status.csv`
- 对比表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\tables\v05_physio_comparison_table.csv`
- 机制表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\tables\v05_physio_mechanism_table.csv`

## 2026-05-20 v0.5 连续风格与生理机制验证

- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage03_v05_physio_mechanism_comparison_user_summary_cn.md`
- 实验脚本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v05_physio_mechanism_comparison.py`
- 旧流程生理对齐修正：`F:\data_set_process\data_process\02_code\final_code\model\training\run_event_conditioned_trajectory_baseline.py`
- 实验注册表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\tables\v05_physio_experiment_registry.csv`
- 生理可用性表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\tables\v05_physio_availability_check.csv`
- 运行状态表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\tables\v05_physio_run_status.csv`
- 总指标对比表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\tables\v05_physio_comparison_table.csv`
- 分被试表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\tables\v05_physio_subject_metrics.csv`
- 机制判断表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\tables\v05_physio_mechanism_table.csv`
- 服务器日志目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\logs`
- 启动命令模板：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\launch_commands_server_no_password.sh`

## 2026-05-20 v0.5 脑电原始数据审计与锚点前特征提取

- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage03_v05_eeg_feature_extraction_user_summary_cn.md`
- 提取脚本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v05_extract_eeg_features.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_eeg_features`
- 记录级脑电清单：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_eeg_features\tables\v05_eeg_recording_inventory.csv`
- v0.5 锚点前 2 秒脑电特征表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_eeg_features\tables\v05_eeg_features_pre_anchor_hist2s.csv`
- 脑电可用性汇总：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_eeg_features\tables\v05_eeg_feature_availability_summary.csv`
- 技术说明：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_eeg_features\stage03_v05_eeg_feature_extraction_report_cn.md`
## 2026-05-20 v0.5 生理/脑电补齐实验

- 用户查看版总结：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage03_v05_physio_eeg_completion_user_summary_cn.md`
- 脑电特征提取说明：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage03_v05_eeg_feature_extraction_user_summary_cn.md`
- 脑电锚点前特征表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_eeg_features\tables\v05_eeg_features_pre_anchor_hist2s.csv`
- 多版本结果白底表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\figures\v05_physio_eeg_result_table_white.png`
- 多版本指标柱状图：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\figures\v05_physio_eeg_metric_overview.png`
- 脑电直接输入/全生理融合曲线图：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\figures\v05_multiversion_overlay_eeg_direct.png`
- 教师蒸馏曲线图：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\figures\v05_multiversion_overlay_teacher.png`
- 完整指标表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\tables\v05_physio_comparison_table.csv`
- 运行状态表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\tables\v05_physio_run_status.csv`
- 汇图脚本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v05_build_multiversion_result_plots.py`
## 2026-05-20 完整记录级 episode 重建 v1.0

这一版不再把一条实验记录固定当成一个事件，也不继续以旧锚点或 `.aed` 触发点作为主入口。它从完整原始车辆 CSV 中重建车辆状态时间线，并允许一条完整实验记录自动切出多个 episode。

- 用户查看版说明：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage02_record_episode_reconstruction_v1_0_user_summary_cn.md`
- 构建脚本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\scripts\build_record_level_episode_reconstruction_v1_0.py`
- 配置文件：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\configs\record_episode_reconstruction_v1_0.json`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_0`
- 全量 episode 总表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_0\tables\record_level_episodes_all_v1_0.csv`
- 文件读取清单：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_0\tables\record_level_file_inventory_v1_0.csv`
- 分组统计表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_0\tables\record_episode_group_summary_v1_0.csv`
- 上下文统计表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_0\tables\record_episode_context_summary_v1_0.csv`
- 分被试统计表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_0\tables\record_episode_by_subject_v1_0.csv`
- 多信号复核图目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_0\figures\review_panels`
- 静态 3D 轨迹图目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_0\figures\trajectory_3d_static`

当前结果摘要：91 条原始车辆记录全部成功读取，自动检测到 1766 个 episode。其中核心极限样本 973 个，保守/弱操作极限样本 406 个，需要复核 335 个，边界复核 45 个。道路/场景字段只作为解释上下文，不作为最终事件真值。

## 2026-05-20 完整记录级 episode 人工复核整理 v1.1

用户查看 v1.0 复核图后判断：大部分样本可以保留，“需要复核”和“边界复核”类基本可以舍去。因此 v1.1 不重新检测 episode，只把 v1.0 结果整理成主训练候选、对照样本、舍弃/暂缓三类。

- 用户查看版说明：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage02_record_episode_reconstruction_v1_1_user_summary_cn.md`
- 构建脚本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\scripts\build_record_episode_reviewed_v1_1.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_1_reviewed`
- 全量带复核决策表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_1_reviewed\tables\record_level_episodes_all_reviewed_v1_1.csv`
- 主训练候选表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_1_reviewed\tables\train_candidate_extreme_episodes_v1_1.csv`
- 舍弃/暂缓表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_1_reviewed\tables\discarded_review_episodes_v1_1.csv`
- 复核图索引：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_1_reviewed\tables\record_episode_figure_index_v1_1.csv`

当前结果摘要：v1.1 总 episode 仍为 1766 个；主训练候选 1383 个，对照样本 3 个，舍弃/暂缓 380 个。主训练候选由核心极限样本、保守/弱操作极限样本、次级训练样本组成；正常弯道或普通操控只作为对照，不进入主训练。

## 2026-05-20 v1.1 完整记录级样本车辆-only GPU 基线

- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage03_v11_vehicle_only_gpu_user_summary_cn.md`
- 汇总表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v11_vehicle_only_gpu_baseline\tables\v11_vehicle_only_gpu_summary.csv`
- 排名表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v11_vehicle_only_gpu_baseline\tables\v11_vehicle_only_gpu_ranking.csv`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v11_vehicle_only_gpu_baseline`

## 2026-05-21 完整记录级 episode 样本集 v1.2

- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage02_record_episode_reconstruction_v1_2_user_summary_cn.md`
- 全量表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_2_cleaned\tables\record_level_episodes_all_v1_2.csv`
- 主训练候选：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_2_cleaned\tables\train_candidate_target_episodes_v1_2.csv`
- 疑似上下马路/路外恢复：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_2_cleaned\tables\suspected_offroad_or_road_recovery_episodes_v1_2.csv`
- 超长误合并：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_2_cleaned\tables\long_merged_episodes_v1_2.csv`
- 复核图目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_2_cleaned\figures\review_panels_v1_2`

## 2026-05-21 完整记录级 episode 样本集 v1.3

- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage02_record_episode_reconstruction_v1_3_user_summary_cn.md`
- 全量表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_3_cleaned\tables\record_level_episodes_all_v1_3.csv`
- 主训练候选：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_3_cleaned\tables\train_candidate_target_episodes_v1_3.csv`
- 疑似路边恢复或上下马路：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_3_cleaned\tables\suspected_roadedge_or_offroad_episodes_v1_3.csv`
- 长弯道/平滑坡度/弯道高动态复核：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_3_cleaned\tables\review_curve_or_grade_episodes_v1_3.csv`
- 复核图目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_3_cleaned\figures\review_panels_v1_3`

## 2026-05-21 完整记录级 episode 样本集 v1.4

- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage02_record_episode_reconstruction_v1_4_user_summary_cn.md`
- 全量表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_4_zdrop_reviewed\tables\record_level_episodes_all_v1_4.csv`
- 主训练候选：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_4_zdrop_reviewed\tables\train_candidate_target_episodes_v1_4.csv`
- 高度大幅下降保留样本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_4_zdrop_reviewed\tables\train_z_drop_extreme_keep_episodes_v1_4.csv`
- 上下马路但无明显大幅下降抛弃样本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_4_zdrop_reviewed\tables\discard_roadedge_without_large_zdrop_episodes_v1_4.csv`
- 复核图目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_4_zdrop_reviewed\figures\review_panels_v1_4`

## 2026-05-21 完整记录级 episode 样本集 v1.5

- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage02_record_episode_reconstruction_v1_5_user_summary_cn.md`
- 全量表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_5_curve_separated\tables\record_level_episodes_all_v1_5.csv`
- 主训练候选：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_5_curve_separated\tables\train_candidate_target_episodes_v1_5.csv`
- 弯道高度下降单独复核：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_5_curve_separated\tables\review_curve_z_drop_separate_episodes_v1_5.csv`
- 全部弯道上下文表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_5_curve_separated\tables\all_curve_context_episodes_v1_5.csv`
- 复核图目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_5_curve_separated\figures\review_panels_v1_5`

## 2026-05-21 完整记录级 episode 样本集 v1.6

- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage02_record_episode_reconstruction_v1_6_user_summary_cn.md`
- 全量表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_6_curve_roll_split\tables\record_level_episodes_all_v1_6.csv`
- 非弯道主训练候选：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_6_curve_roll_split\tables\train_candidate_noncurve_episodes_v1_6.csv`
- 弯道侧倾候选：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_6_curve_roll_split\tables\curve_roll_candidate_clean_episodes_v1_6.csv`
- 弯道高度异常排除：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_6_curve_roll_split\tables\discard_curve_slope_or_z_abnormal_episodes_v1_6.csv`
- 复核图目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_6_curve_roll_split\figures\review_panels_v1_6`

## 2026-05-22 完整记录级 episode 样本集 v1.7

- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage02_record_episode_reconstruction_v1_7_user_summary_cn.md`
- 全量表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_7_curve_zprofile_revised\tables\record_level_episodes_all_v1_7.csv`
- 非弯道主训练候选：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_7_curve_zprofile_revised\tables\train_candidate_noncurve_episodes_v1_7.csv`
- 平滑下坡弯道侧倾候选：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_7_curve_zprofile_revised\tables\curve_smooth_downhill_roll_candidate_episodes_v1_7.csv`
- 弯道高度轨迹异常排除：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_7_curve_zprofile_revised\tables\discard_curve_z_profile_abnormal_episodes_v1_7.csv`
- 复核图目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_7_curve_zprofile_revised\figures\review_panels_v1_7`

## 2026-05-22 完整记录级 episode 样本集 v1.8

- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage02_record_episode_reconstruction_v1_8_user_summary_cn.md`
- 全量表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_8_anchor_curve_revised\tables\record_level_episodes_all_v1_8.csv`
- 全部训练候选：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_8_anchor_curve_revised\tables\train_candidate_all_episodes_v1_8.csv`
- 非弯道主训练候选：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_8_anchor_curve_revised\tables\train_candidate_noncurve_episodes_v1_8.csv`
- 弯道训练候选：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_8_anchor_curve_revised\tables\train_candidate_curve_episodes_v1_8.csv`
- 平滑下坡弯道侧倾候选：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_8_anchor_curve_revised\tables\curve_smooth_downhill_roll_candidate_episodes_v1_8.csv`
- 弯道高度异常排除：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_8_anchor_curve_revised\tables\discard_curve_height_or_z_abnormal_episodes_v1_8.csv`
- 复核图目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_8_anchor_curve_revised\figures\review_panels_v1_8`

## 2026-05-22 完整记录级 episode 样本集 v1.9 道路坐标判弯道

- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage02_record_episode_reconstruction_v1_9_user_summary_cn.md`
- 全量表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_9_coord_curve_revised\tables\record_level_episodes_all_v1_9.csv`
- 全部训练候选：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_9_coord_curve_revised\tables\train_candidate_all_episodes_v1_9.csv`
- 道路坐标弯道训练候选：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_9_coord_curve_revised\tables\train_candidate_curve_coord_episodes_v1_9.csv`
- 非弯道训练候选：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_9_coord_curve_revised\tables\train_candidate_noncurve_episodes_v1_9.csv`
- 冲突审计表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_9_coord_curve_revised\tables\metadata_vs_coord_curve_audit_v1_9.csv`
- 复核图目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_9_coord_curve_revised\figures\review_panels_v1_9`

## 2026-05-22 v1.9 非弯道高度微小变化审计

- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage02_v1_9_noncurve_height_micro_motion_audit_cn.md`
- 分组统计表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_9_coord_curve_revised\tables\noncurve_height_micro_motion_by_module_v1_9.csv`
- 统计图：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\figures\v1_9_noncurve_height_micro_motion_audit.png`


## 2026-05-22 完整记录级 episode 样本集 v2.0 全量无历史继承重审

- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage02_record_episode_reconstruction_v2_0_user_summary_cn.md`
- 全量表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v2_0_no_history_reaudit\tables\record_level_episodes_all_v2_0.csv`
- 全部训练候选：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v2_0_no_history_reaudit\tables\train_candidate_all_episodes_v2_0.csv`
- 非弯道训练候选：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v2_0_no_history_reaudit\tables\train_candidate_noncurve_episodes_v2_0.csv`
- 弯道训练候选：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v2_0_no_history_reaudit\tables\train_candidate_curve_coord_episodes_v2_0.csv`
- 重新纳入训练样本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v2_0_no_history_reaudit\tables\recovered_from_v1_9_nontrain_episodes_v2_0.csv`
- 待复核样本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v2_0_no_history_reaudit\tables\manual_review_episodes_v2_0.csv`
- 对照样本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v2_0_no_history_reaudit\tables\control_or_weak_episodes_v2_0.csv`
- 复核图目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v2_0_no_history_reaudit\figures\review_panels_v2_0`

## 2026-05-22 v2.0 全量无历史继承重审样本车辆-only GPU 基线

- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage03_v20_no_history_vehicle_only_gpu_user_summary_cn.md`
- 汇总表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v20_no_history_vehicle_only_gpu_baseline\tables\v20_no_history_vehicle_only_gpu_summary.csv`
- 排名表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v20_no_history_vehicle_only_gpu_baseline\tables\v20_no_history_vehicle_only_gpu_ranking.csv`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v20_no_history_vehicle_only_gpu_baseline`

## 2026-05-22 v2.0 待复核样本纳入训练车辆-only GPU 基线

- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage03_v20_review_inclusion_vehicle_only_gpu_user_summary_cn.md`
- 汇总表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v20_review_inclusion_vehicle_only_gpu\tables\v20_review_inclusion_vehicle_only_gpu_summary.csv`
- 排名表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v20_review_inclusion_vehicle_only_gpu\tables\v20_review_inclusion_vehicle_only_gpu_ranking.csv`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v20_review_inclusion_vehicle_only_gpu`

## 2026-05-25 goal1 v2.0 训练任务重定义执行

- 报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage03_goal1_v2_task_redesign_user_summary_cn.md`
- 最终报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_goal1_v2_task_redesign\outputs\final_task_redesign_report.md`
- manifest：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_goal1_v2_task_redesign\manifests`
- E0-E5 输出：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_goal1_v2_task_redesign\outputs`
- 汇总表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_goal1_v2_task_redesign\outputs\goal1_experiment_summary.csv`

## 2026-05-26 Goal2 clean vehicle-only 任务审计

- 报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage03_goal2_clean_task_user_summary_cn.md`
- 输出：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_goal2_clean_task_audit\outputs`
- manifest：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_goal2_clean_task_audit\manifests`

## 2026-05-26 Goal2 被排除样本原因拆解

- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage03_goal2_exclusion_recovery_audit_cn.md`
- 逐样本原因表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_goal2_clean_task_audit\outputs\exclusion_recovery_audit\goal2_exclusion_reason_breakdown.csv`
- 排除原因汇总：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_goal2_clean_task_audit\outputs\exclusion_recovery_audit\goal2_exclusion_reason_summary.csv`
- 恢复优先级汇总：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_goal2_clean_task_audit\outputs\exclusion_recovery_audit\goal2_recovery_priority_summary.csv`
- 每档抽查样本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_goal2_clean_task_audit\outputs\exclusion_recovery_audit\goal2_manual_review_sample_30_each_priority.csv`

## 2026-05-26 Goal2 人工审核图片整理

- 图片总入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_goal2_clean_task_audit\outputs\exclusion_recovery_audit\manual_review_images_by_priority\index.html`
- 图片目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_goal2_clean_task_audit\outputs\exclusion_recovery_audit\manual_review_images_by_priority`
- 图片索引：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_goal2_clean_task_audit\outputs\exclusion_recovery_audit\manual_review_images_by_priority\manual_review_images_index.csv`
- 缺图清单：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_goal2_clean_task_audit\outputs\exclusion_recovery_audit\manual_review_images_by_priority\manual_review_images_missing.csv`

## 2026-05-26 SILAB 横向偏移规则修正记录

- 规则说明日志：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\00_project_notes\daily_logs\2026-05-26.md`
- 项目状态入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\00_project_notes\PROJECT_STATUS_CN.md`
- 关键结论：横向偏移跳变可能来自 SILAB 道路/车道参考系切换，后续不能再作为“坐标异常/下马路/路边恢复”的硬排除依据，只能作为复核提示。

## 2026-05-26 道路设计高程与高度异常规则核查

- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage03_goal2_height_rule_design_audit_cn.md`
- 道路中心线总表：`F:\data_set_process\data_process\01_datasets\多模态数据\被试数据集合\道路信息\full_centerline_layout.csv`
- 关键设计文件：`F:\data_set_process\data_process\01_datasets\多模态数据\被试数据集合\道路信息\道路\curve1_Area2.cfg`
- 关键设计文件：`F:\data_set_process\data_process\01_datasets\多模态数据\被试数据集合\道路信息\道路\curve2_Area2.cfg`
- 关键结论：真实道路高程变化在 `curve1/curve2` 中是 6 到 7 m 量级；十几厘米到二十厘米的高度变化不能直接作为下马路/上斜坡/驶出道路的硬排除依据。

## 2026-05-26 v2.1 横向偏移参考系与道路高程修正后样本表

- 生成脚本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_goal2_v21_reference_height_recovery.py`
- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage02_record_episode_reconstruction_v2_1_user_summary_cn.md`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v2_1_reference_height_recovery`
- 全量 v2.1 表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v2_1_reference_height_recovery\tables\manifest_all_v2_1_reference_height_recovery.csv`
- v2.1 训练池/复核训练池：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v2_1_reference_height_recovery\tables\manifest_training_pool_v2_1.csv`
- 主训练候选：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v2_1_reference_height_recovery\tables\manifest_main_train_v2_1.csv`
- 恢复复核候选：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v2_1_reference_height_recovery\tables\manifest_review_recovered_v2_1.csv`
- 弱响应/对照候选：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v2_1_reference_height_recovery\tables\manifest_control_or_weak_v2_1.csv`
- 高度重点复核表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v2_1_reference_height_recovery\tables\manifest_height_review_v2_1.csv`
- 横向偏移参考系风险复核表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v2_1_reference_height_recovery\tables\manifest_lateral_reference_switch_review_v2_1.csv`
- 角色统计：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v2_1_reference_height_recovery\tables\v2_1_role_summary.csv`
- 关键结论：v2.1 不是最终干净训练集，而是避免误删的恢复候选表；横向偏移突变和小幅高度变化均降级为复核提示，不再直接作为硬排除依据。

## 2026-05-26 v2.2 epoch 边界精修审计

- 生成脚本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\scripts\build_record_episode_dataset_v2_2_epoch_refined.py`
- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage02_record_episode_reconstruction_v2_2_epoch_user_summary_cn.md`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v2_2_epoch_refined`
- 全量 v2.2 表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v2_2_epoch_refined\tables\record_level_episodes_all_v2_2_epoch_refined.csv`
- v2.2 边界训练池：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v2_2_epoch_refined\tables\training_pool_epoch_refined_v2_2.csv`
- 需要重划边界样本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v2_2_epoch_refined\tables\epoch_boundary_rework_needed_v2_2.csv`
- 边界状态统计：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v2_2_epoch_refined\tables\v2_2_epoch_status_summary.csv`
- 边界问题统计：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v2_2_epoch_refined\tables\v2_2_epoch_flag_summary.csv`
- 起止和锚点偏移统计：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v2_2_epoch_refined\tables\v2_2_shift_summary.csv`
- 复核图索引：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v2_2_epoch_refined\tables\epoch_boundary_review_figure_index_v2_2.csv`
- 复核图目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v2_2_epoch_refined\figures\epoch_boundary_review_v2_2`
- 关键结论：旧 epoch 边界确实不稳定。旧开始偏早、旧锚点偏早、旧结束偏早和旧结束偏晚都大量存在；后续训练应使用 v2.2 的模型锚点和输入/标签窗口，而不是直接沿用旧 `episode_start_s`、`episode_end_s`、`model_anchor_s_v1_8`。
## 2026-06-22 v222a 候选曲线缓存与轻量受限残差

### v222a candidate curve cache

- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v222a_candidate_curve_cache_20260622.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_candidate_curve_cache_20260622`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_candidate_curve_cache_20260622\reports\v222a_candidate_curve_cache_report_cn.md`
- 候选 manifest：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_candidate_curve_cache_20260622\candidate_manifest.csv`
- 样本 manifest：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_candidate_curve_cache_20260622\sample_manifest.csv`
- loose pool NPZ：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_candidate_curve_cache_20260622\candidate_predictions_loose_main_pool.npz`
- strict pool NPZ：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_candidate_curve_cache_20260622\candidate_predictions_strict_main_pool.npz`
- feature schema audit：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_candidate_curve_cache_20260622\tables\feature_schema_audit.csv`
- 泄漏守卫：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_candidate_curve_cache_20260622\tables\leakage_guard_result.csv`
- 候选曲线指标：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_candidate_curve_cache_20260622\tables\candidate_curve_metrics.csv`
- v219 交叉检查：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_candidate_curve_cache_20260622\tables\metric_crosscheck_vs_v219.csv`
- 缓存摘要：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_candidate_curve_cache_20260622\tables\pool_cache_summary.csv`
- 压缩包：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_candidate_curve_cache_20260622\v222a_candidate_curve_cache_pack.zip`
- 验证摘要：两个 NPZ 分别为 `(1167,14,21)` 与 `(963,14,21)` 候选预测；feature schema 458 行 fail=0；v219 交叉检查最大差异 `0.0019267822736031`，在阈值 `0.002` 内；ZIP `bad_file=None`。

### v222a light fusion residual

- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v222a_light_fusion_residual_20260622.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_light_fusion_residual_20260622`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_light_fusion_residual_20260622\reports\v222a_light_fusion_residual_report_cn.md`
- validation 选择表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_light_fusion_residual_20260622\tables\v222a_validation_selection.csv`
- selected 指标：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_light_fusion_residual_20260622\tables\v222a_selected_metrics.csv`
- fixed baseline 对照：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_light_fusion_residual_20260622\tables\v222a_reference_baseline_metrics.csv`
- selected 逐样本指标：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_light_fusion_residual_20260622\tables\v222a_selected_per_sample_metrics.csv`
- convex blend 权重：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_light_fusion_residual_20260622\tables\v222a_convex_blend_weights.csv`
- 模型 manifest：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_light_fusion_residual_20260622\tables\v222a_model_manifest.csv`
- 泄漏守卫：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_light_fusion_residual_20260622\tables\v222a_leakage_guard_result.csv`
- selected 模型：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_light_fusion_residual_20260622\models\v222a_loose_main_pool_selected.pkl`
- selected 模型：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_light_fusion_residual_20260622\models\v222a_strict_main_pool_selected.pkl`
- selected loose NPZ：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_light_fusion_residual_20260622\v222a_selected_predictions_loose_main_pool.npz`
- selected strict NPZ：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_light_fusion_residual_20260622\v222a_selected_predictions_strict_main_pool.npz`
- 压缩包：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_light_fusion_residual_20260622\v222a_light_fusion_residual_pack.zip`
- 验证摘要：validation selection 只含 val 排序行，共 108 行；feature schema 458 行 fail=0；ZIP `bad_file=None`；禁用名检查未命中。
- 结果摘要：loose pool 低估率从固定 baseline 的 `0.163043` 降到 `0.108696`，但 RMSE 从 `0.544884` 变为 `0.555940`、tail 从 `0.629752` 变为 `0.657612`；strict pool RMSE 与 `peak_floor_090` 基本持平但 tail 变差。因此 v222a 本轮不是新的 headline，只作为 underestimation-control 诊断证据。

---
## 2026-06-22 v222a no-harm gate 诊断

- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v222a_noharm_gate_diagnostic_20260622.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_noharm_gate_diagnostic_20260622`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_noharm_gate_diagnostic_20260622\reports\v222a_noharm_gate_diagnostic_report_cn.md`
- gain/harm 分解：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_noharm_gate_diagnostic_20260622\tables\gain_harm_decomposition.csv`
- oracle 上限：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_noharm_gate_diagnostic_20260622\tables\oracle_safe_gate_report.csv`
- validation gate 选择表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_noharm_gate_diagnostic_20260622\tables\val_gate_tradeoff_table.csv`
- locked test 报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_noharm_gate_diagnostic_20260622\tables\test_locked_gate_report.csv`
- 逐样本 gate 决策：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_noharm_gate_diagnostic_20260622\tables\per_sample_gate_decisions.csv`
- gate manifest：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_noharm_gate_diagnostic_20260622\logs\selected_gate_manifest.json`
- 泄漏守卫：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_noharm_gate_diagnostic_20260622\tables\leakage_guard_result.csv`
- 压缩包：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_noharm_gate_diagnostic_20260622\v222a_noharm_gate_diagnostic_pack.zip`
- 结果摘要：validation gate 两池均过，但 locked test 两池均未通过完整 formal gate。loose pool test 保 under 改善但伤 RMSE/tail；strict pool test 守 RMSE/tail 但 under 变差。oracle safe gate test 明显更好，说明 residual 局部有价值但可部署 gate 学不稳。
- 验证摘要：`py_compile` 通过，ZIP `bad_file=None`，feature schema 458 行 fail=0，leakage guard 全 pass，禁用名检查未命中。

---
## 2026-06-22 v226 formal robustness / confidence-interval audit

- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v226_formal_robustness_ci_audit_20260622.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v226_formal_robustness_ci_audit_20260622`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v226_formal_robustness_ci_audit_20260622\reports\v226_formal_robustness_ci_audit_cn.md`
- formal lock 复核：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v226_formal_robustness_ci_audit_20260622\tables\formal_model_lock_recheck.csv`
- sample bootstrap CI：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v226_formal_robustness_ci_audit_20260622\tables\formal_metric_ci_sample_bootstrap.csv`
- subject-block bootstrap CI：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v226_formal_robustness_ci_audit_20260622\tables\formal_metric_ci_subject_block_bootstrap.csv`
- subject 级指标：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v226_formal_robustness_ci_audit_20260622\tables\formal_subject_level_metrics.csv`
- route-event 级指标：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v226_formal_robustness_ci_audit_20260622\tables\formal_route_event_level_metrics.csv`
- bucket CI 指标：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v226_formal_robustness_ci_audit_20260622\tables\formal_bucket_ci_metrics.csv`
- tail error 集中度：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v226_formal_robustness_ci_audit_20260622\tables\formal_tail_error_concentration.csv`
- 低估 profile：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v226_formal_robustness_ci_audit_20260622\tables\formal_underestimation_profile.csv`
- 极端峰值 profile：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v226_formal_robustness_ci_audit_20260622\tables\formal_extreme_peak_profile.csv`
- 样本影响审计：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v226_formal_robustness_ci_audit_20260622\tables\formal_sample_influence_audit.csv`
- readiness 决策：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v226_formal_robustness_ci_audit_20260622\tables\formal_readiness_decision.csv`
- figure 目录：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v226_formal_robustness_ci_audit_20260622\figures\ci_forest_by_pool`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v226_formal_robustness_ci_audit_20260622\figures\subject_level_metric_distribution`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v226_formal_robustness_ci_audit_20260622\figures\tail_error_concentration`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v226_formal_robustness_ci_audit_20260622\figures\underestimation_profile`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v226_formal_robustness_ci_audit_20260622\figures\extreme_peak_cases_summary`
- manifest：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v226_formal_robustness_ci_audit_20260622\logs\run_manifest.json`
- input hashes：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v226_formal_robustness_ci_audit_20260622\logs\input_file_hashes.json`
- bootstrap config：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v226_formal_robustness_ci_audit_20260622\logs\bootstrap_config.json`
- metric reproduction：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v226_formal_robustness_ci_audit_20260622\logs\metric_reproduction_check.json`
- leakage guard：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v226_formal_robustness_ci_audit_20260622\logs\leakage_guard_report.json`
- forbidden scan：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v226_formal_robustness_ci_audit_20260622\logs\forbidden_scan_report.json`
- table alignment：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v226_formal_robustness_ci_audit_20260622\logs\table_alignment_check.json`
- file inventory：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v226_formal_robustness_ci_audit_20260622\logs\file_inventory.json`
- 压缩包：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v226_formal_robustness_ci_audit_20260622\v226_formal_robustness_ci_audit_pack.zip`
- 结果摘要：formal headline 仍只保留 `loose_main_pool=avg_joint_focus` 和 `strict_main_pool=peak_floor_090`。locked test 指标复现为 loose RMSE/tail `0.544884/0.629752`，strict RMSE/tail `0.571770/0.658306`。sample CI 与 subject-block CI 已生成；readiness 决策为 accepted=True、needs_new_model=False、needs_gate_or_router=False。
- 验证摘要：`py_compile` 通过，脚本完整运行通过，ZIP `bad_file=None`，required files `[]`，figure count 为 `4/4/2/2/2`，metric reproduction / leakage guard / forbidden scan / table alignment 全部 pass。
---
## 2026-06-22 v227 paper / claim readiness pack（最新）

- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v227_paper_claim_readiness_pack_20260622.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v227_paper_claim_readiness_pack_20260622`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v227_paper_claim_readiness_pack_20260622\reports\v227_paper_claim_readiness_cn.md`
- GPTPro 下一轮 ASCII prompt：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v227_paper_claim_readiness_pack_20260622\reports\v227_next_gptpro_prompt_ascii.md`
- 核心表：`paper_main_result_table.csv`、`paper_claim_support_matrix.csv`、`paper_limitation_table.csv`、`formal_guardrail_summary.csv`、`formal_artifact_manifest.csv`、`figure_selection_index.csv`、`gptpro_bridge_status.csv`
- 核心日志：`run_manifest.json`、`input_file_hashes.json`、`source_artifact_checks.json`、`no_model_change_guard.json`、`file_inventory.json`、`zip_integrity_check.json`
- GPTPro 阻塞归档：`F:\data_set_process\data_process\gptpro_reviews\20260622_v227_result_gptpro_response_blocked.md`、`F:\data_set_process\data_process\gptpro_reviews\20260622_v227_result_gptpro_decision_blocked.md`、`F:\data_set_process\data_process\gptpro_reviews\20260622_v227_result_gptpro_action_items_blocked.md`
- 压缩包：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v227_paper_claim_readiness_pack_20260622\v227_paper_claim_readiness_pack.zip`
- 验证摘要：ZIP `bad_file=None`，required files `[]`，`no_model_change_guard.pass=True`，`source_artifact_checks.pass=True`，主结果 formal model 仍为 `loose_main_pool=avg_joint_focus`、`strict_main_pool=peak_floor_090`。
- 用途：在 GPTPro 通道阻塞时，把 v225+v226 既有证据整理成写作/claim/readiness 材料；不是新模型实验。

---
## 2026-06-23 goal-level GPTPro 通道阻塞归档（最新）

- response：`F:\data_set_process\data_process\gptpro_reviews\20260623_goal_blocked_gptpro_channel_response.md`
- decision：`F:\data_set_process\data_process\gptpro_reviews\20260623_goal_blocked_gptpro_channel_decision.md`
- action items：`F:\data_set_process\data_process\gptpro_reviews\20260623_goal_blocked_gptpro_channel_action_items.md`
- 关联 prompt：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v227_paper_claim_readiness_pack_20260622\reports\v227_next_gptpro_prompt_ascii.md`
- 说明：这是 goal-level blocked audit，不是 GPTPro 回复，也不是新实验授权。

---
## 2026-06-23 v228 final paper artifact freeze（最新）

- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v228_final_paper_artifact_freeze_20260623.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v228_final_paper_artifact_freeze_20260623`
- 压缩包：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v228_final_paper_artifact_freeze_20260623\v228_final_paper_artifact_freeze_pack.zip`
- 核心表：
  - `tables/final_formal_model_lock.csv`
  - `tables/final_main_result_table.csv`
  - `tables/final_ci_table.csv`
  - `tables/final_claim_lock_table.csv`
  - `tables/final_limitations_table.csv`
  - `tables/final_figure_selection_table.csv`
  - `tables/final_artifact_manifest.csv`
  - `tables/final_guardrail_summary.csv`
- 核心报告：
  - `reports/v228_final_paper_artifact_freeze_cn.md`
  - `reports/manuscript_results_section_draft_cn.md`
  - `reports/manuscript_claim_boundary_notes_cn.md`
- 核心日志：
  - `logs/run_manifest.json`
  - `logs/input_file_hashes.json`
  - `logs/consistency_check.json`
  - `logs/forbidden_scan_report.json`
  - `logs/guardrail_check.json`
  - `logs/file_inventory.json`
- 图件目录：
  - `figures/selected_main_figures`
  - `figures/selected_appendix_figures`
- GPTPro 归档：
  - `F:\data_set_process\data_process\gptpro_reviews\20260623_v228_local_gptpro_response.md`
  - `F:\data_set_process\data_process\gptpro_reviews\20260623_v228_local_gptpro_decision.md`
  - `F:\data_set_process\data_process\gptpro_reviews\20260623_v228_local_gptpro_action_items.md`
- 验证摘要：ZIP `testzip=None`；required files missing `[]`；formal lock exact；main metric diffs `0`；CI row count `144/144`；forbidden hits `0`；guardrail/consistency 均 pass。

---
## 2026-06-26 v243 v241 guarded fine-tune（最新）

- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v243_v241_guarded_finetune_20260626.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v243_v241_guarded_finetune_20260626`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v243_v241_guarded_finetune_20260626\reports\v243_v241_guarded_finetune_cn.md`
- 压缩包：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v243_v241_guarded_finetune_20260626\v243_v241_guarded_finetune_pack.zip`
- 核心模型/预测：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v243_v241_guarded_finetune_20260626\models\v243_best_guarded_finetune_diagnostic.pt`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v243_v241_guarded_finetune_20260626\v243_v241_guarded_finetune_predictions.npz`
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v243_v241_guarded_finetune_20260626\models\v243_scalers_and_selection.pkl`
- 核心表：
  - `tables\v243_model_selection_validation_guarded.csv`
  - `tables\v243_metrics_by_delay_and_bucket.csv`
  - `tables\v243_compare_vs_v236_v239_v241_original_remaining.csv`
  - `tables\v243_candidate_test_robustness_summary.csv`
  - `tables\v243_per_sample_delta_vs_v241.csv`
  - `tables\v243_per_sample_delta_summary_vs_v241.csv`
  - `tables\v243_worst_regressions_vs_v241.csv`
  - `tables\v243_top_improvements_vs_v241.csv`
  - `tables\v243_training_history.csv`
  - `tables\v243_training_weight_plan.csv`
  - `tables\v243_next_decision.csv`
  - `tables\v243_split_integrity_check.csv`
- 核心图：
  - `figures\v243_guarded_tail_compare_all.png`
  - `figures\v243_guarded_tail_compare_normal_predictable.png`
  - `figures\v243_guarded_tail_compare_observe_later_like.png`
  - `figures\v243_guarded_tail_compare_strong_steer.png`
- 核心日志：
  - `logs\guardrail_check.json`
  - `logs\leakage_check.json`
  - `logs\run_manifest.json`
  - `logs\input_file_hashes.csv`
  - `logs\file_inventory.json`
  - `logs\train_stdout.log`
  - `logs\train_stderr.log`
- 结果摘要：
  - validation-selected：`v243_metric_hard36_guard08`，score `0.865386`，best_epoch `34`，`accepted_as_next_candidate=True`。
  - hard36 validation vs v241：all 0-800 mean tail delta `-0.007909`，observe 0-800 mean tail delta `-0.004415`，strong 400/1000 mean tail delta `-0.010060`。
  - hard36 test vs v241：all mean tail delta `-0.002128`，normal `-0.006139`，observe `+0.009219`，strong `+0.003896`，说明 hard bucket test 有退化。
  - test 稳定性最均衡：`v243_metric_hard24_guard04`，all `-0.003832`，normal `-0.003955`，observe `-0.006484`，strong `-0.003601`。
- 验证摘要：
  - `python -m py_compile` 通过。
  - 完整训练运行完成。
  - `guardrail_check.pass=True`。
  - `leakage_check.pass=True`。
  - 同一 `event_uid` 跨 split 数为 `0`。
  - ZIP `testzip=None`，条目数 `29`。
- 用途：v243 作为 v241 backbone 的 guarded fine-tune 候选包；下一步应做 locked audit 比较 hard36 与 hard24，不能直接视为 formal replacement。

---
## 2026-06-29 v244 hard36 vs hard24 locked audit（最新）

- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v244_locked_audit_compare_v243_hard36_vs_hard24_20260629.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v244_locked_audit_compare_v243_hard36_vs_hard24_20260629`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v244_locked_audit_compare_v243_hard36_vs_hard24_20260629\reports\v244_locked_audit_compare_v243_hard36_vs_hard24_cn.md`
- 压缩包：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v244_locked_audit_compare_v243_hard36_vs_hard24_20260629\v244_locked_audit_compare_v243_hard36_vs_hard24_pack.zip`
- 核心表：
  - `tables\v244_prediction_availability_audit.csv`
  - `tables\v244_validation_vs_test_candidate_compare.csv`
  - `tables\v244_per_delay_hard24_hard36_compare.csv`
  - `tables\v244_bucket_decision_matrix.csv`
  - `tables\v244_hard36_per_sample_risk_summary.csv`
  - `tables\v244_hard36_worst_regressions_vs_v241.csv`
  - `tables\v244_missing_hard24_granular_audit.csv`
  - `tables\v244_next_decision.csv`
- 核心图：
  - `figures\v244_candidate_test_mean_tail_delta.png`
  - `figures\v244_per_delay_tail_delta_hard24_vs_hard36.png`
  - `figures\v244_validation_vs_test_tradeoff.png`
- 核心日志：
  - `logs\guardrail_check.json`
  - `logs\run_manifest.json`
  - `logs\input_file_hashes.csv`
  - `logs\file_inventory.json`
- 结果摘要：
  - validation-selected candidate：`v243_metric_hard36_guard08`。
  - locked test more stable candidate：`v243_metric_hard24_guard04`。
  - hard36 test vs v241：all `-0.002128`，normal `-0.006139`，observe `+0.009219`，strong `+0.003896`。
  - hard24 test vs v241：all `-0.003832`，normal `-0.003955`，observe `-0.006484`，strong `-0.003601`。
  - hard36 observe/strong worse delay count `11/12`；hard24 observe/strong worse delay count `2/12`。
  - hard24 缺少完整 prediction/checkpoint/per-sample delta，所以不能直接 formal replacement。
- 验证摘要：
  - `python -m py_compile` 通过。
  - 完整脚本运行完成。
  - `guardrail_check.pass=True`。
  - ZIP `testzip=None`，条目数 `16`。
- 用途：锁定比较 v243 hard36 与 hard24 的 validation-vs-test tradeoff；给出“v243 不能直接替代 v241，若继续只应补齐 hard24 granular artifact”的审计结论。

---
## 2026-06-30 v245 差样本锚点后移效果审查（最新）

- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v245_bad_sample_anchor_shift_effect_audit_20260630.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v245_bad_sample_anchor_shift_effect_audit_20260630`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v245_bad_sample_anchor_shift_effect_audit_20260630\reports\v245_bad_sample_anchor_shift_effect_audit_cn.md`
- 压缩包：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v245_bad_sample_anchor_shift_effect_audit_20260630\v245_bad_sample_anchor_shift_effect_audit_pack.zip`
- 核心表：
  - `tables\v245_sample_metrics.csv`
  - `tables\v245_anchor_shift_pairs.csv`
  - `tables\v245_anchor_shift_summary_by_group.csv`
  - `tables\v245_anchor_shift_summary_by_base_delay.csv`
  - `tables\v245_anchor_shift_best_later_by_sample.csv`
  - `tables\v245_anchor_shift_best_later_summary.csv`
  - `tables\v245_bad_sample_thresholds.csv`
- 核心图：
  - `figures\v245_anchor_shift_effect_bad_top10_v241.png`
  - `figures\v245_anchor_shift_effect_by_base_delay_v241.png`
  - `figures\v245_anchor_shift_case_examples_v241.png`
- 核心日志：
  - `logs\guardrail_check.json`
  - `logs\input_file_hashes.csv`
  - `logs\run_manifest.json`
- 结果摘要：
  - test bad_top10：v241 tail RMSE q90=`0.727`，共 `111` 个样本。
  - v241 bad_top10 固定 `+400ms`：mean delta `-0.210`，改善率 `83.1%`。
  - v241 bad_top10 固定 `+600ms`：mean delta `-0.288`，改善率 `88.7%`。
  - early bad_top10 oracle 最佳后移：mean delta `-0.428`，改善率 `95.8%`。
  - overlap point n=`11`，true absolute alignment RMSE 约 `1e-7`，说明改善来自后移观察增加信息，而不是少预测。
- 验证摘要：
  - `python -m py_compile` 通过。
  - 完整脚本运行完成。
  - `guardrail_check.pass=True`。
  - ZIP `testzip=None`，条目数 `14`。
- 用途：证明差样本后移锚点有明确改善效果；为下一步 v246 “风险样本延后观察/重锚定”任务构造提供证据。

---

## 2026-06-30 v246 oracle 最佳锚点遍历与 input-only selector 审查（最新）

- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v246_oracle_best_anchor_and_selector_audit_20260630.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v246_oracle_best_anchor_and_selector_audit_20260630`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v246_oracle_best_anchor_and_selector_audit_20260630\reports\v246_oracle_best_anchor_and_selector_audit_cn.md`
- 压缩包：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v246_oracle_best_anchor_and_selector_audit_20260630\v246_oracle_best_anchor_and_selector_audit_pack.zip`
- 核心表：
  - `tables\v246_sample_tail_errors.csv`
  - `tables\v246_base_input_features.csv`
  - `tables\v246_anchor_candidate_table.csv`
  - `tables\v246_oracle_best_anchor_by_base_sample.csv`
  - `tables\v246_oracle_best_anchor_summary.csv`
  - `tables\v246_selector_candidate_error_fit_metrics.csv`
  - `tables\v246_selector_predictions_by_candidate.csv`
  - `tables\v246_selector_selected_anchor_by_base_sample.csv`
  - `tables\v246_policy_selected_anchor_by_base_sample.csv`
  - `tables\v246_selector_policy_summary.csv`
  - `tables\v246_anchor_shift_distribution.csv`
- 核心图：
  - `figures\v246_test_oracle_vs_selector_error.png`
  - `figures\v246_test_bad_top10_shift_distribution.png`
- 核心日志：
  - `logs\guardrail_check.json`
  - `logs\input_file_hashes.csv`
  - `logs\run_manifest.json`
- 结果摘要：
  - test bad_top10 oracle：RMSE `1.008 -> 0.656`，mean delta `-0.352`，改善率 `84.7%`。
  - early bad_top10 oracle：RMSE `1.021 -> 0.591`，mean delta `-0.431`，改善率 `95.8%`，最常见 oracle shift `+600ms`。
  - RF selector bad_top10：RMSE `1.008 -> 0.908`，mean delta `-0.100`，改善率 `29.7%`。
  - `policy_wait_to_latest_anchor` bad_top10：RMSE `1.008 -> 0.685`，mean delta `-0.322`；该策略与 Ridge selector 数值一致，说明 Ridge 主要学到“等到最晚锚点”。
- 验证摘要：
  - `python -m py_compile` 通过。
  - 完整脚本运行完成。
  - `guardrail_check.pass=True`。
  - ZIP `testzip=None`，条目数 `18`。
- 用途：把 v245 的“后移有效”推进到逐样本 oracle 上限与可部署 selector/固定等待策略对照；为 v247 带等待代价的重锚定任务构造提供依据。

---
## 2026-06-30 v247 50ms 多分辨率 best anchor discovery（最新）

- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v247_multi_resolution_best_anchor_discovery_20260630.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v247_multi_resolution_best_anchor_discovery_20260630`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v247_multi_resolution_best_anchor_discovery_20260630\reports\v247_multi_resolution_best_anchor_discovery_cn.md`
- 压缩包：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v247_multi_resolution_best_anchor_discovery_20260630_pack.zip`
- 预测数组：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v247_multi_resolution_best_anchor_discovery_20260630\v247_fine_grid_v241_predictions.npz`
- 核心表：
  - `tables\v247_fine_grid_sampling_audit.csv`
  - `tables\v247_coarse_replay_alignment.csv`
  - `tables\v247_coarse_replay_alignment_by_row.csv`
  - `tables\v247_fine_anchor_candidate_table.csv`
  - `tables\v247_bad_thresholds_by_split.csv`
  - `tables\v247_best_anchor_by_event.csv`
  - `tables\v247_best_anchor_distribution.csv`
  - `tables\v247_score_weight_sweep_summary.csv`
  - `tables\v247_selector_training_table.csv`
  - `tables\v247_selector_predictions_by_candidate.csv`
  - `tables\v247_selector_selected_anchor_by_event.csv`
  - `tables\v247_selector_policy_summary.csv`
  - `tables\v247_selector_fit_diagnostics.csv`
  - `tables\v247_signal_anchor_diagnostics.csv`
- 核心图：
  - `figures\v247_best_anchor_distribution_by_group.png`
  - `figures\v247_selector_vs_oracle_error.png`
  - `figures\v247_selected_delay_distribution.png`
  - `figures\v247_error_delay_score_curves_examples.png`
  - `figures\v247_signal_anchor_alignment.png`
- 日志与校验：
  - `logs\guardrail_check.json`
  - `logs\input_file_hashes.csv`
  - `logs\run_manifest.json`
  - `logs\file_inventory.csv`
  - `logs\v247_instability_feature_scales.json`
  - `logs\v247_selector_feature_encoding.csv`
- 结果摘要：
  - 50ms fine-grid 支持成立：`1167` 个事件、`24507` 个候选锚点、`dropped=0`。
  - coarse replay 与旧 v241 预测对齐：mean RMSE `0.000000`，max `0.000001`。
  - primary score `delay_l05_unstable_m05`：test/all 当前 `0.475` -> oracle best `0.253`；test/bad_top10 当前 `1.198` -> oracle best `0.616`。
  - 当前 RF selector test/bad_top10 为 `0.947`，弱于固定 `policy_wait_to_latest_anchor` 的 `0.695`。
- 用途：v247 是 best-anchor label 和 selector 可学习性的任务构造审计；证明细粒度 best anchor 有上限收益，但当前 selector 尚不能作为可部署方案。

---

## 2026-07-02 v257 同驾驶员生理状态记忆检索

- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v257_subject_personalized_physio_memory_20260702.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v257_subject_personalized_physio_memory_20260702`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v257_subject_personalized_physio_memory_20260702\reports\v257_subject_personalized_physio_memory_cn.md`
- 压缩包：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v257_subject_personalized_physio_memory_20260702_pack.zip`
- 核心表：
  - `tables\v257_memory_strategy_metrics.csv`
  - `tables\v257_candidate_coverage.csv`
  - `tables\v257_validation_model_selection.csv`
  - `tables\v257_per_sample_metrics.csv`
- 核心图：
  - `figures\v257_subject_memory_test_tail_rmse.png`
- 日志与校验：
  - `logs\guardrail_check.json`
  - `logs\input_file_hashes.csv`
  - `logs\file_inventory.csv`
- 结果摘要：
  - validation 选择 `same_subject_vehicle_k3`，不是生理增强策略。
  - test bad_top10：`v250_existing` tail `0.8383`；`same_subject_vehicle_k3` tail `1.3054`，delta `+0.4671`。
- 用途：验证同驾驶员历史记忆/个体化检索是否能让生理状态起作用；结果不支持作为主线。

---

## 2026-07-02 v258 生理增强 anchor selector

- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v258_physio_augmented_anchor_selector_20260702.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v258_physio_augmented_anchor_selector_20260702`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v258_physio_augmented_anchor_selector_20260702\reports\v258_physio_augmented_anchor_selector_cn.md`
- 压缩包：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v258_physio_augmented_anchor_selector_20260702_pack.zip`
- 核心表：
  - `tables\v258_anchor_selector_summary.csv`
  - `tables\v258_selected_anchor_by_event.csv`
  - `tables\v258_model_feature_summary.csv`
  - `tables\v258_augmented_selector_training_table.csv`
- 核心图：
  - `figures\v258_anchor_selector_test_badtop10.png`
- 日志与校验：
  - `logs\guardrail_check.json`
  - `logs\input_file_hashes.csv`
  - `logs\file_inventory.csv`
- 结果摘要：
  - test bad_top10：keep 0ms `1.1977`；wait-latest `0.6950`；oracle `0.6125`。
  - vehicle selector `0.9300`；vehicle+physio selector `0.9342`；badweighted vehicle+physio `0.9593`。
- 用途：验证生理是否能帮助“等不等/等多久”的可部署 anchor selector；结果不支持。

---

## 2026-07-02 v259 生理-车辆 cross-attention 直接预测

- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v259_physio_cross_attention_prediction_20260702.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v259_physio_cross_attention_prediction_20260702`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v259_physio_cross_attention_prediction_20260702\reports\v259_physio_cross_attention_prediction_cn.md`
- 压缩包：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v259_physio_cross_attention_prediction_20260702_pack.zip`
- 预测数组：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v259_physio_cross_attention_prediction_20260702\tensors\v259_predictions.npz`
- 核心表：
  - `tables\v259_prediction_metrics_by_bucket.csv`
  - `tables\v259_per_sample_prediction_metrics.csv`
  - `tables\v259_training_log.csv`
  - `tables\v259_validation_model_selection.csv`
  - `tables\v259_training_weight_audit.csv`
- 核心图：
  - `figures\v259_test_bucket_tail_rmse.png`
- 日志与校验：
  - `logs\guardrail_check.json`
  - `logs\input_file_hashes.csv`
  - `logs\file_inventory.csv`
- 结果摘要：
  - subject-disjoint bad_top10：v250 `0.8783`；v259 vehicle-only `0.9267`；vehicle+physio cross-attention `1.0889`；badweighted `1.0351`。
  - subject-aware bad_top10：v250 `0.8383`；v259 vehicle-only `1.0038`；vehicle+physio cross-attention `1.1288`。
- 用途：验证“更深 raw 生理时序融合”是否能直接提升单轨迹行为预测；结果不支持。

---
## 最新指针：2026-07-02 v288 ECG source-signal route audit

- 总结：v288 回到 cleaned 200Hz ECG 源信号，重新提取 R 峰/RR、短窗形态、质量和因果同步偏移特征，共 `518` 个 ECG source 特征、`27` 个 feature set，并复用 v278 vehicle top40 route gate。结果 `route_viable_now=false`：deployable top1 在 test bad_top10 上仍比 latest 差 `+0.1556`，在 bad_top10_vehicle_ambiguous 上差 `+0.1510`；test-best top1 仍差 `+0.0903`，best corr `0.0620`。ECG 源信号有效率和窗口覆盖正常，但不能稳定转成可部署候选选择。
- 入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v288_ecg_source_signal_route_audit_20260702.py`
- 产物：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v288_ecg_source_signal_route_audit_20260702`
- 报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v288_ecg_source_signal_route_audit_20260702\reports\v288_ecg_source_signal_route_audit_cn.md`
- ZIP：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v288_ecg_source_signal_route_audit_20260702_pack.zip`
- 核心表：
  - `tables\v288_ecg_source_features.csv`
  - `tables\v288_ecg_source_features_with_targets.csv`
  - `tables\v288_train_only_feature_screen.csv`
  - `tables\v288_feature_screen_summary.csv`
  - `tables\v288_ecg_quality_by_recording.csv`
  - `tables\v288_feature_set_audit.csv`
  - `tables\v288_train_scaler_audit.csv`
  - `tables\v288_route_gate_per_event.csv`
  - `tables\v288_route_group_summary.csv`
  - `tables\v288_val_chosen_generalization.csv`
  - `tables\v288_route_gate_decision.csv`
- 核心图：
  - `figures\v288_badtop10_val_test_delta.png`
  - `figures\v288_ecg_offset_group_summary.png`
  - `figures\v288_ecg_feature_screen_summary.png`
- 校验：`logs\guardrail_check.json`，`logs\input_hashes.csv`，`logs\file_inventory.csv`
# 最新产物索引：2026-07-02 v297 subject style stability audit 已加入。该实验给出当前路线判断：驾驶风格存在弱辅助信号，但不足以作为主线；事件级标签/实验条件标签优先级最高。

---

## 2026-07-02 v297 subject style stability audit（最新）

- 目的：审计同一被试多次独立事件之间是否存在稳定驾驶风格/响应倾向，以及这种倾向是否足以改善差样本预测。
- 脚本：`05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_v297_subject_style_stability_audit_20260702.py`
- 输出目录：`05_rebuild_from_raw_20260511/03_baselines/v297_subject_style_stability_audit_20260702`
- 中文报告：`05_rebuild_from_raw_20260511/03_baselines/v297_subject_style_stability_audit_20260702/reports/v297_subject_style_stability_audit_cn.md`
- 打包文件：`05_rebuild_from_raw_20260511/03_baselines/v297_subject_style_stability_audit_20260702_pack.zip`
- 关键表：
  - `tables/v297_event_response_descriptors.csv`
  - `tables/v297_subject_recording_eta.csv`
  - `tables/v297_subject_descriptor_summary.csv`
  - `tables/v297_pair_distance_summary.csv`
  - `tables/v297_pair_distance_sample.csv`
  - `tables/v297_rolling_history_predictability.csv`
  - `tables/v297_binary_history_auc.csv`
  - `tables/v297_oracle_label_candidate_counts.csv`
  - `tables/v297_style_route_decision.csv`
- 关键图：
  - `figures/v297_subject_eta_by_descriptor.png`
  - `figures/v297_rolling_history_improvement.png`
  - `figures/v297_same_vs_different_subject_distance.png`
- 关键结论：
  - `style_route_supported_now=False`
  - `weak_style_signal_exists=True`
  - `event_label_route_priority=True`
  - 驾驶风格适合做风险/不确定性辅助，不适合作为主轨迹预测器。

## 2026-07-02 v295 wait1 direct residual physiology

- 目的：在 v249 delay1000 wait1 基线之上，直接学习 residual，检验 post0_1 生理和 subject/context 是否能修正差样本轨迹。
- 脚本：`05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_v295_wait1_direct_residual_physio_20260702.py`
- 输出目录：`05_rebuild_from_raw_20260511/03_baselines/v295_wait1_direct_residual_physio_20260702`
- 中文报告：`05_rebuild_from_raw_20260511/03_baselines/v295_wait1_direct_residual_physio_20260702/reports/v295_wait1_direct_residual_physio_cn.md`
- 打包文件：`05_rebuild_from_raw_20260511/03_baselines/v295_wait1_direct_residual_physio_20260702_pack.zip`
- 关键图：
  - `figures/v295_chosen_selector_test_delta.png`
  - `figures/v295_test_bad_top6_curves.png`
- 关键结论：
  - `route_viable_now=False`
  - `goal_achieved_now=False`
  - 可部署生理 residual 对 bad_top10 只有极弱改善，全样本变差。
  - 非生理 residual ablation 反而更稳定，说明当前生理拼接/残差修正没有形成核心突破。

## 2026-07-02 v296 rawseq physio embedding residual（中断半成品）

- 状态：运行中被用户中断。
- 已产生部分中间表，但没有完整 guardrail、报告和 ZIP。
- 处理原则：不作为有效实验结果引用；后续如要继续，应重新设计并完整跑完。

---
# 最新产物索引：2026-07-03 v302 roll cause input audit

- 产物目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v302_roll_cause_input_audit_20260703`
- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v302_roll_cause_input_audit_20260703.py`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v302_roll_cause_input_audit_20260703\reports\v302_roll_cause_input_audit_cn.md`
- 打包文件：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v302_roll_cause_input_audit_20260703\v302_roll_cause_input_audit_20260703.zip`
- 关键表：
  - `tables/v302_roll_cause_raw_feature_audit.csv`：当前 v236 输入中侧倾诱因相关列数量。
  - `tables/v302_roll_cause_signal_coverage.csv`：steering/speed/ay/yaw/roll/roll_rate/roll_acc/brake/curvature 等信号覆盖。
  - `tables/v302_roll_cause_summary_features.csv`：逐事件显式 roll-cause summary。
  - `tables/v302_feature_set_audit.csv`：各输入集合特征数和缺失率。
  - `tables/v302_multiclass_predictability_by_input.csv`：不同输入集合的事件类型识别完整结果。
  - `tables/v302_multiclass_val_chosen_test_summary.csv`：validation 选分类器后的 test 对比。
  - `tables/v302_bad_sample_binary_by_input.csv`：不同输入集合的差样本识别完整结果。
  - `tables/v302_badtop10_val_chosen_test_summary.csv`：validation 选分类器后的 bad_top10 test 对比。
  - `tables/v302_residual_regression_selection.csv`：残差回归模型和 shrink 的 validation 选择结果。
  - `tables/v302_residual_regression_summary.csv`：v300 残差修正分组结果。
  - `tables/v302_residual_regression_event_deltas.csv`：逐事件残差修正变化。
- 关键图：
  - `figures/v302_event_type_macro_f1_by_input.png`
  - `figures/v302_badtop10_auc_by_input.png`
  - `figures/v302_residual_delta_by_input.png`
- 日志与校验：
  - `logs/guardrail_check.json`：`pass=True`，`uses_future_event_labels_as_features=false`，`roll_cause_features_already_in_v236=true`。
  - `logs/input_hashes.csv`
  - `logs/file_inventory.csv`
- 用途：
  - 证明侧倾诱因信号已经在当前输入中存在，并且显式聚合能改善事件类型识别。
  - 作为后续 roll-cause 辅助监督/多任务/混合专家分支的输入候选。
  - 不能被解释为 bad_top10 已经获得本质改善。

---
# 最新产物索引：2026-07-03 v301 event type multiclass label audit

- 产物目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v301_event_type_multiclass_label_audit_20260703`
- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v301_event_type_multiclass_label_audit_20260703.py`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v301_event_type_multiclass_label_audit_20260703\reports\v301_event_type_multiclass_label_audit_cn.md`
- 打包文件：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v301_event_type_multiclass_label_audit_20260703\v301_event_type_multiclass_label_audit_20260703.zip`
- 关键表：
  - `tables/v301_event_type_labels.csv`：每个事件的自动事件类型草稿、多标签标志、未来行为统计。
  - `tables/v301_manual_review_pack.csv`：建议人工优先复核的样本。
  - `tables/v301_event_type_counts_and_error.csv`：事件类型分布与 v300 误差分层。
  - `tables/v301_label_predictability_summary.csv`：锚点前输入预测事件类型的分类器效果。
  - `tables/v301_label_classifier_predictions.csv`：分类器逐事件预测。
  - `tables/v301_label_residual_correction_summary.csv`：标签残差修正理论收益。
  - `tables/v301_label_residual_event_deltas.csv`：逐事件残差修正变化。
  - `tables/v301_event_type_thresholds.csv`：自动标签阈值。
- 关键图：
  - `figures/v301_event_type_distribution.png`：自动事件类型分布。
  - `figures/v301_event_type_test_rmse.png`：各事件类型 test RMSE。
  - `figures/v301_event_type_classifier_confusion.png`：锚点前事件类型分类器混淆矩阵。
  - `figures/v301_label_residual_correction_delta.png`：标签残差修正对 RMSE 的影响。
- 日志与校验：
  - `logs/guardrail_check.json`：`pass=True`，`labels_deployable_as_direct_input_now=false`，`manual_review_required_before_model_input=true`。
  - `logs/input_hashes.csv`
  - `logs/file_inventory.csv`
- 用途：
  - 用于事件类型人工复核、分层评估、辅助监督候选设计。
  - 不能作为“未来行为标签直接输入模型”的正式建模依据。

---

# 最新产物索引：2026-07-04 v307 coarse scene-label conditioned curve model

- 产物目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v307_coarse_scene_label_conditioned_curve_model_20260704`
- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v307_coarse_scene_label_conditioned_curve_model_20260704.py`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v307_coarse_scene_label_conditioned_curve_model_20260704\reports\v307_coarse_scene_label_conditioned_curve_model_cn.md`
- 打包文件：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v307_coarse_scene_label_conditioned_curve_model_20260704\v307_coarse_scene_label_conditioned_curve_model_20260704.zip`
- 关键表：
  - `tables/v307_model_selection_validation.csv`：validation-only 候选选择表。
  - `tables/v307_delay0_group_summary.csv`：delay0 分组指标，含 v300 与 v307 对比。
  - `tables/v307_metrics_by_delay_and_bucket.csv`：全 delay / bucket 指标。
  - `tables/v307_per_sample_metrics_original_remaining.csv`：逐样本指标。
  - `tables/v307_coarse_scene_aux_metrics.csv`：粗场景辅助头分类指标。
  - `tables/v307_coarse_scene_class_mapping.csv`：粗场景类别索引和 class weight。
- 关键模型/预测：
  - `models/v307_coarse_scene_init_aux003_film005_h64.pt`
  - `models/v307_coarse_scene_init_aux005_film010_h64.pt`
  - `models/v307_coarse_scene_init_aux006_film010_hard110_h64.pt`
  - `models/v307_scalers_and_selection.pkl`
  - `v307_coarse_scene_label_conditioned_predictions.npz`
- 日志与校验：
  - `logs/guardrail_check.json`：`pass=True`，`uses_coarse_scene_labels_as_features=true`，`candidate_selection_uses_test=false`，`zip_testzip=True`。
  - `logs/input_hashes.csv`
  - `logs/file_inventory.csv`
- 用途：
  - 检验“下坡过弯 / 平路过弯 / 连续变道 / 紧急变道失稳”粗场景标签作为条件输入是否有效。
  - 当前选中模型 test/all `0.496138`、test/within_bad_top10 `0.777797`、test/within_bad_top20 `0.639121`，均优于 v300，并略优于 v304 细标签版本。
  - 不能被解释为最终部署模型；直道内连续/紧急子类仍部分来自 v305/v301 seed，需要人工或实验条件确认。

---

# 最新产物索引：2026-07-04 v306 coarse predefined scene label table

- 产物目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v306_coarse_predefined_scene_label_table_20260704`
- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v306_coarse_predefined_scene_label_table_20260704.py`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v306_coarse_predefined_scene_label_table_20260704\reports\v306_coarse_predefined_scene_label_table_cn.md`
- 打包文件：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v306_coarse_predefined_scene_label_table_20260704\v306_coarse_predefined_scene_label_table_20260704.zip`
- 关键表：
  - `tables/v306_coarse_scene_event_labels.csv`：当前 1167 个事件的粗场景标签 seed 表。
  - `tables/v306_coarse_scene_counts_total.csv`：粗场景总分布。
  - `tables/v306_coarse_scene_counts_by_split.csv`：按 split 的粗场景分布。
  - `tables/v306_scene_type_to_coarse_scene_crosstab.csv`：当前 `scene_type` 到粗场景标签的交叉表。
  - `tables/v306_v305_formal_to_coarse_scene_crosstab.csv`：v305 formal 标签到粗场景标签的交叉表。
  - `tables/v306_coarse_scene_manual_review_seed_pack.csv`：人工审核排序包。
- 关键图：
  - `figures/v306_coarse_scene_label_distribution.png`
- 日志与校验：
  - `logs/guardrail_check.json`：`pass=True`，`coarse_scene_class_n=5`，`curve_scene_labels_from_current_scene_type=true`，`uses_future_behavior_seed_for_some_noncurve_subtypes=true`，`zip_testzip=True`。
  - `logs/input_hashes.csv`
  - `logs/file_inventory.csv`
- 用途：
  - 将用户确认的粗场景体系固定为条件模型输入边界。
  - 过弯两类来自当前 manifest `scene_type`；直道内连续/紧急子类作为 seed 等待人工或实验条件确认。
  - 为 v307/v308 条件模型和人工复核提供统一标签表。

---

# 最新产物索引：2026-07-04 v305 formal predefined event label table

- 产物目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v305_formal_predefined_event_label_table_20260704`
- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v305_formal_predefined_event_label_table_20260704.py`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v305_formal_predefined_event_label_table_20260704\reports\v305_formal_predefined_event_label_table_cn.md`
- 打包文件：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v305_formal_predefined_event_label_table_20260704\v305_formal_predefined_event_label_table_20260704.zip`
- 关键表：
  - `tables/v305_formal_event_labels.csv`：正式事件标签 seed 表。
  - `tables/v305_manual_review_seed_pack.csv`：人工审核排序包。
  - `tables/v305_formal_primary_counts_total.csv`：正式主事件类型总分布。
  - `tables/v305_formal_primary_counts_by_split.csv`：按 split 的主事件类型分布。
  - `tables/v305_v301_to_formal_primary_crosstab.csv`：v301 自动主标签到 formal 主标签的映射交叉表。
- 关键图：
  - `figures/v305_formal_primary_type_distribution.png`
- 日志与校验：
  - `logs/guardrail_check.json`：`pass=True`，`task_allows_predefined_event_label_input=true`，`formal_primary_type_can_be_model_input_after_confirmation=true`，`diagnostic_tags_as_direct_input_allowed=false`。
  - `logs/input_hashes.csv`
  - `logs/file_inventory.csv`
- 用途：
  - 将“事件可提前打标签”固定为正式建模输入边界。
  - 为人工审核、实验条件标签接入、后续 v304/v305 条件模型重训提供统一标签表。
  - 不能被解释为人工审核已经完成；当前仍是 v301 自动标签 seed。

---

# 最新产物索引：2026-07-03 v304 fixed event-label conditioned curve model

- 产物目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v304_fixed_event_label_conditioned_curve_model_20260703`
- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v304_fixed_event_label_conditioned_curve_model_20260703.py`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v304_fixed_event_label_conditioned_curve_model_20260703\reports\v304_fixed_event_label_conditioned_curve_model_cn.md`
- 打包文件：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v304_fixed_event_label_conditioned_curve_model_20260703\v304_fixed_event_label_conditioned_curve_model_20260703.zip`
- 关键表：
  - `tables/v304_model_selection_validation.csv`：validation-only 候选模型选择与 no-harm gate。
  - `tables/v304_delay0_group_summary.csv`：v300 与选中 v304 在 delay0 分组上的 RMSE 对比。
  - `tables/v304_event_aux_metrics.csv`：事件辅助头分类指标。
  - `tables/v304_metrics_by_delay_and_bucket.csv`：不同 delay 与样本桶的指标。
  - `tables/v304_per_sample_metrics_original_remaining.csv`：逐样本原始剩余集指标。
  - `tables/v304_input_audit.csv`：输入、标签源和特征数量审计。
  - `tables/v304_roll_cause_signal_coverage.csv`：roll-cause 信号覆盖审计。
  - `tables/v304_event_class_mapping.csv`：事件类别映射与类别权重。
- 关键图：
  - `figures/v304_training_history.png`
  - `figures/v304_test_delay0_group_rmse.png`
  - `figures/v304_event_aux_macro_f1.png`
- 关键预测文件：
  - `v304_fixed_event_label_conditioned_predictions.npz`
- 日志与校验：
  - `logs/guardrail_check.json`：`pass=True`，`uses_fixed_event_labels_as_features=true`，`fixed_event_label_source=v301_future_behavior_auto_draft`，`fixed_event_label_deployable_without_external_or_manual_label=false`，`candidate_selection_uses_test=false`。
  - `logs/input_hashes.csv`
  - `logs/file_inventory.csv`
- 用途：
  - 检验“事件类型在训练/预测前已知”对轨迹预测的上限价值。
  - 为人工标签体系、实验条件标签接入、事件路由 mixture-of-experts 提供结构基线。
  - 不能被解释为当前无条件部署模型已经获得同等收益。

---

# 最新产物索引：2026-07-03 v303 roll-cause auxiliary multitask curve model

- 产物目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v303_roll_aux_multitask_curve_model_20260703`
- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v303_roll_aux_multitask_curve_model_20260703.py`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v303_roll_aux_multitask_curve_model_20260703\reports\v303_roll_aux_multitask_curve_model_cn.md`
- 打包文件：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v303_roll_aux_multitask_curve_model_20260703\v303_roll_aux_multitask_curve_model_20260703.zip`
- 关键表：
  - `tables/v303_model_selection_validation.csv`：validation-only 候选模型选择与 no-harm gate。
  - `tables/v303_delay0_group_summary.csv`：v300 与选中 v303 在 delay0 分组上的 RMSE 对比。
  - `tables/v303_event_aux_metrics.csv`：事件辅助头 accuracy、balanced accuracy、macro-F1、weighted-F1。
  - `tables/v303_metrics_by_delay_and_bucket.csv`：不同 delay 与样本桶的指标。
  - `tables/v303_per_sample_metrics_original_remaining.csv`：逐样本原始剩余集指标。
  - `tables/v303_training_history_*.csv`：各候选训练历史。
- 关键图：
  - `figures/v303_training_history.png`
  - `figures/v303_test_delay0_group_rmse.png`
  - `figures/v303_event_aux_macro_f1.png`
- 关键预测文件：
  - `v303_roll_aux_multitask_predictions.npz`
- 日志与校验：
  - `logs/guardrail_check.json`：`pass=True`，`model_structure_changed=true`，`uses_future_event_labels_as_features=false`，`uses_event_labels_as_auxiliary_targets=true`，`candidate_selection_uses_test=false`。
  - `logs/input_hashes.csv`
  - `logs/file_inventory.csv`
- 用途：
  - 作为 v300 之后第一个 roll-cause 辅助监督结构化小基线。
  - 证明 roll-cause 分支 + 辅助事件监督可以在不伤害全样本的前提下，对 bad_top10 / bad_top20 给出小幅改善。
  - 后续可作为 mixture-of-experts、多模态不确定性输出或 bad-focused 专家模型的初始化和对照基线。

---
# 最新产物索引：2026-07-04 第317版二阶段候选门控校正实验
- 脚本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v317_two_stage_candidate_gate_20260704.py`
- 产物目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v317_two_stage_candidate_gate_20260704`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v317_two_stage_candidate_gate_20260704\reports\v317_two_stage_candidate_gate_cn.md`
- 压缩包：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v317_two_stage_candidate_gate_20260704\v317_two_stage_candidate_gate_20260704.zip`
- 主要表格：
  - `tables/v317_gate_input_features.csv`：只含锚点前车辆信号、第316版预测摘要和粗场景标签的门控输入特征。
  - `tables/v317_candidate_library.csv`：20条候选曲线定义。
  - `tables/v317_train_residual_prototypes.csv`：只从训练集残差聚类得到的残差原型。
  - `tables/v317_validation_per_sample_metrics.csv`：验证集逐样本指标。
  - `tables/v317_validation_group_summary.csv`：验证集分组摘要。
  - `tables/v317_validation_gate_check.csv`：验证门槛检查。
  - `tables/v317_validation_candidate_usage.csv`：候选最优和门控选择次数。
- 主要图：
  - `figures/v317_validation_group_rmse.png`
  - `figures/v317_validation_candidate_usage.png`
- 核心结果：守卫通过但验证失败，`goal_validation_passed=false`，因此 `test_reported=false`。固定方案为 `随机森林-候选单选`，验证全部样本误差从第316版 `0.531658` 退化到 `0.586667`；候选最优上限为 `0.375611`，说明候选库本身有潜力，但门控选择机制失败。
- 失败分流：下一步优先做保守门控和原预测优先约束，而不是继续过滤样本或直接报告测试集。
---

# 最新产物索引：2026-07-04 本地高级模型第317版修正方案咨询
- 提问稿：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\gptpro_reviews\20260704_phase317_prompt.md`
- 回答稿：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\gptpro_reviews\20260704_phase317_response.md`
- 决策记录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\gptpro_reviews\20260704_phase317_decision.md`
- 行动项：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\gptpro_reviews\20260704_phase317_action_items.md`
- 页面截图：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\gptpro_reviews\20260704_phase317_after_reply_20260704_221634.png`
- 核心结论：第317版建议冻结或复用第316版基础预测，增加轻量二阶段候选校正器。候选库先覆盖原预测、幅值缩放、时间平移、少量残差原型；门控输入只能使用锚点前车辆信号、第316版预测摘要和可部署的预先事件标签；验证必须同时检查整体无伤、普通样本无伤、强方向盘改善和困难样本改善。
- 边界：这是外部建议归档，不是训练结果；测试集仍不得参与第317版参数选择。
---
# 最新产物索引：2026-07-05 本地高级模型第318版修正方案咨询

- 提问文件：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\gptpro_reviews\20260705_phase318_prompt.md`
- 回复文件：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\gptpro_reviews\20260705_phase318_response.md`
- 原始备份：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\gptpro_reviews\20260705_phase318_response_raw_with_prompt.md`
- 决策记录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\gptpro_reviews\20260705_phase318_decision.md`
- 行动项：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\gptpro_reviews\20260705_phase318_action_items.md`
- 页面截图：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\gptpro_reviews\20260705_phase318_after_reply_20260705_073659.png`
- 页面转储：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\gptpro_reviews\20260705_phase318_after_reply_20260705_073659.uia.txt`
- 核心结论：第318版应做保守两段式候选门控，默认保持第316版原预测；只有高置信可校正样本才进入候选选择，并用小幅残差融合降低普通样本大退化风险。该咨询是下一轮实现依据，不是训练结果；测试集仍不得参与第318版模型、阈值、候选或融合幅度选择。
---

# 最新产物索引：2026-07-05 v321 第320版困难样本可视化图册

- 脚本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v321_hard_sample_visual_gallery_20260705.py`
- 产物目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v321_hard_sample_visual_gallery_20260705`
- 图册首页：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v321_hard_sample_visual_gallery_20260705\index.html`
- 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v321_hard_sample_visual_gallery_20260705\reports\v321_hard_sample_visual_gallery_report_cn.md`
- 关键表：
  - `tables/v321_hard_sample_gallery_manifest.csv`：46个测试困难样本的图册清单、分组、收益、候选上限、图片路径。
  - `tables/v321_hard_sample_group_summary.csv`：按修正变坏、门控未抓住、未修正仍困难、修正变好等分组的摘要。
- 关键图：
  - `figures/困难样本分组总览.png`
  - `figures/困难样本第316对第320散点.png`
  - `figures/*.png`：每个困难样本一张完整图，包含方向盘角、方向盘变化、横向加速度、横摆角速度、侧倾，并展示0到2秒预测范围和2到6秒真实后续。
- 日志与校验：
  - `logs/guardrail_check.json`：`pass=True`，`only_visualization=True`，`does_not_retrain_model=True`，`reconstruction_max_abs_rmse_diff=1.665e-16`。
- 用途：
  - 给用户直接查看第320版困难样本形态。
  - 支撑下一步判断：第320版困难组失败主要来自门控没抓住或选错，而不是候选库没有上限空间。
---
