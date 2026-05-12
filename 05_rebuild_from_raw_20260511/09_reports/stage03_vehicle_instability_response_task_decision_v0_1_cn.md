# 阶段 3：失稳样本响应任务定义决策 v0.1

## 输入

- 原始样本清单：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_highconf_v0_1/tables/samples_master.csv`
- 标签窗口覆盖审计表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_label_window_coverage_audit_v0_1/tables/label_window_event_policy_table.csv`

## 决策原则

1. 不把 2 秒标签继续称为完整响应；它只能作为即时响应任务。
2. 2 秒后存在后续峰值或明显方向盘变化且 3 秒标签稳定的事件，转为 3 秒响应覆盖候选。
3. 3 秒仍未稳定的事件，不直接进入完整响应核心训练；优先回到阶段 2 复核或拆成启动响应/持续控制。
4. 所有这些决策只基于车辆标签窗口和样本规则，不涉及生理、脑电、连续风格或驾驶员 ID。

## 事件任务类别计数

```text
                  response_task_class                   response_task_track    response_task_class_cn  n_events     rate
             long_or_unsettled_review      D_long_event_or_unsettled_review 3秒仍未稳定或长事件复杂，需回到样本规则复核/拆分       588 0.649007
switch_to_3s_continuing_response_core           B_response3s_core_candidate    2秒后仍有明显变化，优先转为3秒响应覆盖候选       193 0.213024
                 instant2s_core_clean            A_instant2s_core_and_3s_ok   2秒即时响应核心样本，2秒和3秒标签都相对稳定        60 0.066225
  instant2s_ok_but_long_event_context A_instant2s_only_with_long_event_flag    2秒即时响应可用，但完整响应/长事件仍需复核        24 0.026490
      manual_2s_tail_or_anchor_review     C_manual_2s_anchor_or_tail_review        2秒尾段或锚点需复核，暂不进核心训练        24 0.026490
          switch_to_3s_late_peak_core           B_response3s_core_candidate     2秒漏掉后续峰值，优先转为3秒响应覆盖候选        17 0.018764
```

## 任务轨道计数

```text
                  response_task_track  n_events     rate
     D_long_event_or_unsettled_review       588 0.649007
          B_response3s_core_candidate       210 0.231788
           A_instant2s_core_and_3s_ok        60 0.066225
A_instant2s_only_with_long_event_flag        24 0.026490
    C_manual_2s_anchor_or_tail_review        24 0.026490
```

## 样本窗口角色计数

```text
             window_config_id                    task_sample_role  n_samples
    pre1_label2_event_trigger           long_event_review_holdout        588
    pre1_label2_event_trigger          not_primary_for_next_stage        210
    pre1_label2_event_trigger       early1s_control_for_instant2s         84
    pre1_label2_event_trigger manual_window_anchor_review_holdout         24
         pre2_label2_old_main           long_event_review_holdout        588
         pre2_label2_old_main          not_primary_for_next_stage        210
         pre2_label2_old_main            instant2s_core_candidate         84
         pre2_label2_old_main manual_window_anchor_review_holdout         24
pre3_label3_response_coverage           long_event_review_holdout        588
pre3_label3_response_coverage    response3s_strict_core_candidate        270
pre3_label3_response_coverage manual_window_anchor_review_holdout         24
pre3_label3_response_coverage         response3s_review_candidate         24
```

## 核心数字

- 事件总数：906
- 2 秒即时响应核心候选：84
- 3 秒响应覆盖候选：294
- 3 秒严格核心候选：270
- 长事件/持续控制复核：588
- 手动窗口/锚点复核：636
- 下一轮车辆-only 基线优先候选窗口样本：462

## 输出

- 事件级决策表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_response_task_decision_v0_1/tables/event_response_task_decision_table.csv`
- 样本级任务 manifest：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_response_task_decision_v0_1/tables/sample_response_task_manifest.csv`
- 任务类别计数：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_response_task_decision_v0_1/tables/response_task_decision_counts.csv`
- split 汇总：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_response_task_decision_v0_1/tables/response_task_split_summary.csv`
- subject 汇总：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_response_task_decision_v0_1/tables/response_task_subject_summary.csv`
- 图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_response_task_decision_v0_1/figures/response_task_decision_counts.png`
- 图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_response_task_decision_v0_1/figures/response_task_sample_roles_by_window.png`

## 下一步建议

先基于这个覆盖层重跑两个车辆-only 对照：A 轨道的 2 秒即时响应核心候选，以及 B 轨道的 3 秒响应覆盖核心候选。D 轨道长事件暂不进入最终主线训练，先做复核或拆分。
