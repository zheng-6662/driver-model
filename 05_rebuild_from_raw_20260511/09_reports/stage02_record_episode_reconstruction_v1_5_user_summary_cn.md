# 完整记录级 episode 样本集 v1.5：弯道高度下降单独判断

生成时间：2026-05-21 13:26:06

## 这次为什么改

用户继续复核 v1.4 后指出：v1.4 中保留的“高度大幅下降”样本实际上都来自弯道路段。弯道本身可能有道路高程变化和曲率引起的车身动态，因此不能把这些样本当作上下马路极限样本直接加入主训练。

所以 v1.5 把这 22 个高度大幅下降样本从主训练候选中拿出来，单独归入“弯道高度下降，单独判断”。

## v1.5 规则

- v1.4 中 `train_z_drop_extreme_keep` 且属于 `curve1/curve2` 或弯道上下文的样本，改为 `review_curve_z_drop_separate`。
- 这些样本不进入当前主训练候选，但不删除，后续可作为弯道专门任务或弯道复核池。
- v1.4 原本的目标极限事件和保守/弱操作极限事件继续作为主训练候选。

## 数量变化

- v1.5 主训练候选：820
- 弯道高度下降单独复核：22
- 全部弯道上下文样本：430

## v1.5 分类表

| v1_5_decision | v1_5_decision_cn | count |
| --- | --- | --- |
| discard_prior_review | 继承 v1.4：此前已舍弃或不适合作为当前主训练候选 | 751 |
| train_target_extreme | 继承 v1.4：目标极限事件，保留为当前主训练候选 | 472 |
| train_conservative_extreme | 继承 v1.4：保守/弱操作极限样本，保留为当前主训练候选 | 348 |
| defer_prior_review | 继承 v1.4：仍需复核或拆分，不进入当前主训练候选 | 170 |
| review_curve_z_drop_separate | 用户复核后调整：高度大幅下降样本属于弯道上下文，需单独判断，不进入当前主训练候选 | 22 |
| control_normal_or_curve | 继承 v1.4：正常弯道或普通操控，仅保留为对照样本 | 3 |

## 弯道高度下降单独复核样本

| episode_uid | subject | road_module_names | episode_duration_s | z_drop_from_start_v1_4 | v1_4_decision | review_panel_v1_5_path |
| --- | --- | --- | --- | --- | --- | --- |
| rec_v1_zxy_2025_09_28_16_35_30_0003 | zxy | curve1|middle_section | 59.2050 | 7.6646 | train_z_drop_extreme_keep | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_5_curve_separated\figures\review_panels_v1_5\01_弯道高度下降_单独判断\1732_rec_v1_zxy_2025_09_28_16_35_30_0003.png |
| rec_v1_hzh_2025_09_26_21_03_19_0004 | hzh | curve1 | 16.9940 | 7.0257 | train_z_drop_extreme_keep | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_5_curve_separated\figures\review_panels_v1_5\01_弯道高度下降_单独判断\0493_rec_v1_hzh_2025_09_26_21_03_19_0004.png |
| rec_v1_gf_2025_09_26_10_40_59_0003 | gf | curve1 | 19.0400 | 7.0009 | train_z_drop_extreme_keep | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_5_curve_separated\figures\review_panels_v1_5\01_弯道高度下降_单独判断\0292_rec_v1_gf_2025_09_26_10_40_59_0003.png |
| rec_v1_zxy_2025_09_28_16_25_51_0003 | zxy | curve1|middle_section | 37.8200 | 6.9973 | train_z_drop_extreme_keep | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_5_curve_separated\figures\review_panels_v1_5\01_弯道高度下降_单独判断\1708_rec_v1_zxy_2025_09_28_16_25_51_0003.png |
| rec_v1_byx_2025_09_28_17_35_43_0009 | byx | curve1 | 53.7750 | 6.9965 | train_z_drop_extreme_keep | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_5_curve_separated\figures\review_panels_v1_5\01_弯道高度下降_单独判断\0077_rec_v1_byx_2025_09_28_17_35_43_0009.png |
| rec_v1_byx_2025_09_28_17_05_51_0005 | byx | curve1 | 25.5100 | 6.9963 | train_z_drop_extreme_keep | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_5_curve_separated\figures\review_panels_v1_5\01_弯道高度下降_单独判断\0005_rec_v1_byx_2025_09_28_17_05_51_0005.png |
| rec_v1_zdq_2025_09_26_15_52_46_0003 | zdq | curve1 | 42.8750 | 6.9952 | train_z_drop_extreme_keep | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_5_curve_separated\figures\review_panels_v1_5\01_弯道高度下降_单独判断\1394_rec_v1_zdq_2025_09_26_15_52_46_0003.png |
| rec_v1_hzh_2025_09_27_19_44_05_0004 | hzh | curve1 | 16.8400 | 6.9924 | train_z_drop_extreme_keep | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_5_curve_separated\figures\review_panels_v1_5\01_弯道高度下降_单独判断\0607_rec_v1_hzh_2025_09_27_19_44_05_0004.png |
| rec_v1_hzh_2025_09_27_19_33_25_0004 | hzh | curve1 | 14.0100 | 6.4810 | train_z_drop_extreme_keep | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_5_curve_separated\figures\review_panels_v1_5\01_弯道高度下降_单独判断\0579_rec_v1_hzh_2025_09_27_19_33_25_0004.png |
| rec_v1_txj_2025_09_27_09_17_11_0005 | txj | curve1 | 29.1600 | 6.3570 | train_z_drop_extreme_keep | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_5_curve_separated\figures\review_panels_v1_5\01_弯道高度下降_单独判断\1027_rec_v1_txj_2025_09_27_09_17_11_0005.png |
| rec_v1_yyl_2025_09_28_09_14_23_0007 | yyl | curve1 | 18.3150 | 6.1376 | train_z_drop_extreme_keep | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_5_curve_separated\figures\review_panels_v1_5\01_弯道高度下降_单独判断\1151_rec_v1_yyl_2025_09_28_09_14_23_0007.png |
| rec_v1_lx_2025_09_26_09_17_22_0002 | lx | curve1 | 41.6150 | 6.0637 | train_z_drop_extreme_keep | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_5_curve_separated\figures\review_panels_v1_5\01_弯道高度下降_单独判断\0749_rec_v1_lx_2025_09_26_09_17_22_0002.png |
| rec_v1_jy_2025_09_26_17_29_44_0003 | jy | curve1 | 52.7830 | 6.0633 | train_z_drop_extreme_keep | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_5_curve_separated\figures\review_panels_v1_5\01_弯道高度下降_单独判断\0649_rec_v1_jy_2025_09_26_17_29_44_0003.png |
| rec_v1_zx_2025_09_27_17_45_11_0003 | zx | curve1 | 19.1350 | 5.1797 | train_z_drop_extreme_keep | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_5_curve_separated\figures\review_panels_v1_5\01_弯道高度下降_单独判断\1555_rec_v1_zx_2025_09_27_17_45_11_0003.png |
| rec_v1_yyl_2025_09_28_09_49_11_0004 | yyl | curve1 | 12.4760 | 4.4098 | train_z_drop_extreme_keep | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_5_curve_separated\figures\review_panels_v1_5\01_弯道高度下降_单独判断\1222_rec_v1_yyl_2025_09_28_09_49_11_0004.png |
| rec_v1_zx_2025_09_27_16_46_13_0004 | zx | curve1 | 10.7450 | 3.8995 | train_z_drop_extreme_keep | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_5_curve_separated\figures\review_panels_v1_5\01_弯道高度下降_单独判断\1481_rec_v1_zx_2025_09_27_16_46_13_0004.png |
| rec_v1_jy_2025_09_26_17_51_46_0007 | jy | curve2|middle_section | 17.3600 | 3.6218 | train_z_drop_extreme_keep | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_5_curve_separated\figures\review_panels_v1_5\01_弯道高度下降_单独判断\0690_rec_v1_jy_2025_09_26_17_51_46_0007.png |
| rec_v1_lx_2025_09_26_08_58_43_0003 | lx | curve1 | 8.3300 | 3.4421 | train_z_drop_extreme_keep | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_5_curve_separated\figures\review_panels_v1_5\01_弯道高度下降_单独判断\0730_rec_v1_lx_2025_09_26_08_58_43_0003.png |
| rec_v1_xst_2025_09_26_11_34_18_0004 | xst | curve1 | 79.6250 | 3.3011 | train_z_drop_extreme_keep | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_5_curve_separated\figures\review_panels_v1_5\01_弯道高度下降_单独判断\1129_rec_v1_xst_2025_09_26_11_34_18_0004.png |
| rec_v1_rjy_2025_09_28_20_02_20_0003 | rjy | curve1 | 13.3000 | 3.1070 | train_z_drop_extreme_keep | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_5_curve_separated\figures\review_panels_v1_5\01_弯道高度下降_单独判断\0897_rec_v1_rjy_2025_09_28_20_02_20_0003.png |
| rec_v1_zxy_2025_09_28_16_35_30_0014 | zxy | curve2|middle_section | 23.4500 | 2.2704 | train_z_drop_extreme_keep | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_5_curve_separated\figures\review_panels_v1_5\01_弯道高度下降_单独判断\1743_rec_v1_zxy_2025_09_28_16_35_30_0014.png |
| rec_v1_tyy_2025_09_28_14_23_43_0008 | tyy | curve2|middle_section | 10.5600 | 2.1351 | train_z_drop_extreme_keep | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_5_curve_separated\figures\review_panels_v1_5\01_弯道高度下降_单独判断\1062_rec_v1_tyy_2025_09_28_14_23_43_0008.png |

## 输出位置

- v1.5 全量表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_5_curve_separated\tables\record_level_episodes_all_v1_5.csv`
- v1.5 主训练候选：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_5_curve_separated\tables\train_candidate_target_episodes_v1_5.csv`
- 弯道高度下降单独复核表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_5_curve_separated\tables\review_curve_z_drop_separate_episodes_v1_5.csv`
- 全部弯道上下文表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_5_curve_separated\tables\all_curve_context_episodes_v1_5.csv`
- 复核图目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_5_curve_separated\figures\review_panels_v1_5`

## 当前建议

v1.5 更符合现在的判断：当前主训练集不再混入弯道高程下降片段。下一步如果要训练，可以先用 v1.5 主训练候选跑车辆-only；弯道样本另起一个“弯道专门判断/弯道专门模型”分支，不要和上下马路、低附着、避让事件混在一起。

本轮没有训练模型。
