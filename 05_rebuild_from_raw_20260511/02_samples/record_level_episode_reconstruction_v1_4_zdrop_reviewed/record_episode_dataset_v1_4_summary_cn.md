# 完整记录级 episode 样本集 v1.4：保留高度大幅下降极限样本

生成时间：2026-05-21 13:09:11

## 这次为什么改

用户复核 v1.3 后认为：高度 `z` 文件夹里的多数样本确实是开下马路或路边恢复，整体判断方向是对的；但其中有一类 `z` 明显大幅向下掉的片段，应先保留，因为它们代表明显的高度突变/路外极限工况。其它没有明显大幅下降的上下马路/路边恢复片段，可以先抛弃，不进入当前训练候选。

## v1.4 规则

- 只在 v1.3 已经标为“疑似路边恢复或上下马路”的样本里重新筛。
- 计算 `z_drop_from_start = episode 开始后 0.5 秒内 z 中位数 - episode 内最低 z`。
- 若 `z_drop_from_start >= 2.0 m`，保留为 `train_z_drop_extreme_keep`。
- 其它路边恢复/上下马路样本标为 `discard_roadedge_without_large_zdrop`，不进入当前训练候选。
- v1.3 已经保留的目标极限事件和保守/弱操作极限事件继续保留。

## 数量变化

- v1.4 主训练候选总数：842
- 其中新增保留的高度大幅下降极限样本：22
- 被抛弃的上下马路/路边恢复但无明显大幅下降样本：371

## v1.4 分类表

| v1_4_decision | v1_4_decision_cn | count |
| --- | --- | --- |
| train_target_extreme | 继承 v1.3：目标极限事件，保留为训练候选 | 472 |
| discard_prior_review | 继承 v1.3：此前已舍弃/暂缓，不进入当前训练候选 | 380 |
| discard_roadedge_without_large_zdrop | 用户复核后抛弃：属于疑似上下马路/路边恢复，但没有明显大幅向下 z_drop，不进入当前训练候选 | 371 |
| train_conservative_extreme | 继承 v1.3：保守/弱操作极限样本，保留为训练候选 | 348 |
| defer_prior_review | 继承 v1.3：仍需要复核或拆分，暂不进入当前训练候选 | 170 |
| train_z_drop_extreme_keep | 用户复核后保留：高度 z 相对 episode 开始大幅下降，作为高度大幅下降极限样本 | 22 |
| control_normal_or_curve | 继承 v1.3：正常弯道或普通操控，仅保留为对照样本 | 3 |

## 保留的高度大幅下降样本

| episode_uid | subject | road_module_names | episode_duration_s | z_drop_from_start_v1_4 | z_start_median_v1_4 | z_min_after_start_v1_4 | v1_3_decision | review_panel_v1_4_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rec_v1_zxy_2025_09_28_16_35_30_0003 | zxy | curve1|middle_section | 59.2050 | 7.6646 | -0.1434 | -7.8080 | defer_roadedge_or_offroad_long | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_4_zdrop_reviewed\figures\review_panels_v1_4\01_保留_高度大幅下降极限样本\1732_rec_v1_zxy_2025_09_28_16_35_30_0003.png |
| rec_v1_hzh_2025_09_26_21_03_19_0004 | hzh | curve1 | 16.9940 | 7.0257 | -0.1503 | -7.1760 | defer_roadedge_or_offroad | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_4_zdrop_reviewed\figures\review_panels_v1_4\01_保留_高度大幅下降极限样本\0493_rec_v1_hzh_2025_09_26_21_03_19_0004.png |
| rec_v1_gf_2025_09_26_10_40_59_0003 | gf | curve1 | 19.0400 | 7.0009 | -0.1433 | -7.1442 | defer_roadedge_or_offroad | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_4_zdrop_reviewed\figures\review_panels_v1_4\01_保留_高度大幅下降极限样本\0292_rec_v1_gf_2025_09_26_10_40_59_0003.png |
| rec_v1_zxy_2025_09_28_16_25_51_0003 | zxy | curve1|middle_section | 37.8200 | 6.9973 | -0.1485 | -7.1458 | defer_roadedge_or_offroad_long | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_4_zdrop_reviewed\figures\review_panels_v1_4\01_保留_高度大幅下降极限样本\1708_rec_v1_zxy_2025_09_28_16_25_51_0003.png |
| rec_v1_byx_2025_09_28_17_35_43_0009 | byx | curve1 | 53.7750 | 6.9965 | -0.1441 | -7.1406 | defer_roadedge_or_offroad_long | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_4_zdrop_reviewed\figures\review_panels_v1_4\01_保留_高度大幅下降极限样本\0077_rec_v1_byx_2025_09_28_17_35_43_0009.png |
| rec_v1_byx_2025_09_28_17_05_51_0005 | byx | curve1 | 25.5100 | 6.9963 | -0.1499 | -7.1462 | defer_roadedge_or_offroad_long | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_4_zdrop_reviewed\figures\review_panels_v1_4\01_保留_高度大幅下降极限样本\0005_rec_v1_byx_2025_09_28_17_05_51_0005.png |
| rec_v1_zdq_2025_09_26_15_52_46_0003 | zdq | curve1 | 42.8750 | 6.9952 | -0.1448 | -7.1400 | defer_roadedge_or_offroad_long | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_4_zdrop_reviewed\figures\review_panels_v1_4\01_保留_高度大幅下降极限样本\1394_rec_v1_zdq_2025_09_26_15_52_46_0003.png |
| rec_v1_hzh_2025_09_27_19_44_05_0004 | hzh | curve1 | 16.8400 | 6.9924 | -0.1446 | -7.1371 | defer_roadedge_or_offroad | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_4_zdrop_reviewed\figures\review_panels_v1_4\01_保留_高度大幅下降极限样本\0607_rec_v1_hzh_2025_09_27_19_44_05_0004.png |
| rec_v1_hzh_2025_09_27_19_33_25_0004 | hzh | curve1 | 14.0100 | 6.4810 | -0.1489 | -6.6299 | defer_roadedge_or_offroad | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_4_zdrop_reviewed\figures\review_panels_v1_4\01_保留_高度大幅下降极限样本\0579_rec_v1_hzh_2025_09_27_19_33_25_0004.png |
| rec_v1_txj_2025_09_27_09_17_11_0005 | txj | curve1 | 29.1600 | 6.3570 | -0.1480 | -6.5050 | defer_roadedge_or_offroad_long | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_4_zdrop_reviewed\figures\review_panels_v1_4\01_保留_高度大幅下降极限样本\1027_rec_v1_txj_2025_09_27_09_17_11_0005.png |
| rec_v1_yyl_2025_09_28_09_14_23_0007 | yyl | curve1 | 18.3150 | 6.1376 | -0.1432 | -6.2808 | defer_roadedge_or_offroad | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_4_zdrop_reviewed\figures\review_panels_v1_4\01_保留_高度大幅下降极限样本\1151_rec_v1_yyl_2025_09_28_09_14_23_0007.png |
| rec_v1_lx_2025_09_26_09_17_22_0002 | lx | curve1 | 41.6150 | 6.0637 | -0.1487 | -6.2123 | defer_roadedge_or_offroad_long | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_4_zdrop_reviewed\figures\review_panels_v1_4\01_保留_高度大幅下降极限样本\0749_rec_v1_lx_2025_09_26_09_17_22_0002.png |
| rec_v1_jy_2025_09_26_17_29_44_0003 | jy | curve1 | 52.7830 | 6.0633 | -0.1450 | -6.2084 | defer_roadedge_or_offroad_long | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_4_zdrop_reviewed\figures\review_panels_v1_4\01_保留_高度大幅下降极限样本\0649_rec_v1_jy_2025_09_26_17_29_44_0003.png |
| rec_v1_zx_2025_09_27_17_45_11_0003 | zx | curve1 | 19.1350 | 5.1797 | -0.1438 | -5.3234 | defer_roadedge_or_offroad | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_4_zdrop_reviewed\figures\review_panels_v1_4\01_保留_高度大幅下降极限样本\1555_rec_v1_zx_2025_09_27_17_45_11_0003.png |
| rec_v1_yyl_2025_09_28_09_49_11_0004 | yyl | curve1 | 12.4760 | 4.4098 | -0.1443 | -4.5540 | defer_roadedge_or_offroad | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_4_zdrop_reviewed\figures\review_panels_v1_4\01_保留_高度大幅下降极限样本\1222_rec_v1_yyl_2025_09_28_09_49_11_0004.png |
| rec_v1_zx_2025_09_27_16_46_13_0004 | zx | curve1 | 10.7450 | 3.8995 | -0.1432 | -4.0426 | defer_roadedge_or_offroad | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_4_zdrop_reviewed\figures\review_panels_v1_4\01_保留_高度大幅下降极限样本\1481_rec_v1_zx_2025_09_27_16_46_13_0004.png |
| rec_v1_jy_2025_09_26_17_51_46_0007 | jy | curve2|middle_section | 17.3600 | 3.6218 | -7.3783 | -11.0002 | defer_roadedge_or_offroad | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_4_zdrop_reviewed\figures\review_panels_v1_4\01_保留_高度大幅下降极限样本\0690_rec_v1_jy_2025_09_26_17_51_46_0007.png |
| rec_v1_lx_2025_09_26_08_58_43_0003 | lx | curve1 | 8.3300 | 3.4421 | -0.1452 | -3.5872 | defer_roadedge_or_offroad | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_4_zdrop_reviewed\figures\review_panels_v1_4\01_保留_高度大幅下降极限样本\0730_rec_v1_lx_2025_09_26_08_58_43_0003.png |
| rec_v1_xst_2025_09_26_11_34_18_0004 | xst | curve1 | 79.6250 | 3.3011 | -0.1444 | -3.4455 | defer_roadedge_or_offroad_long | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_4_zdrop_reviewed\figures\review_panels_v1_4\01_保留_高度大幅下降极限样本\1129_rec_v1_xst_2025_09_26_11_34_18_0004.png |
| rec_v1_rjy_2025_09_28_20_02_20_0003 | rjy | curve1 | 13.3000 | 3.1070 | -0.1489 | -3.2559 | defer_roadedge_or_offroad | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_4_zdrop_reviewed\figures\review_panels_v1_4\01_保留_高度大幅下降极限样本\0897_rec_v1_rjy_2025_09_28_20_02_20_0003.png |
| rec_v1_zxy_2025_09_28_16_35_30_0014 | zxy | curve2|middle_section | 23.4500 | 2.2704 | -7.1442 | -9.4146 | defer_roadedge_or_offroad_long | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_4_zdrop_reviewed\figures\review_panels_v1_4\01_保留_高度大幅下降极限样本\1743_rec_v1_zxy_2025_09_28_16_35_30_0014.png |
| rec_v1_tyy_2025_09_28_14_23_43_0008 | tyy | curve2|middle_section | 10.5600 | 2.1351 | -7.2854 | -9.4205 | defer_roadedge_or_offroad | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_4_zdrop_reviewed\figures\review_panels_v1_4\01_保留_高度大幅下降极限样本\1062_rec_v1_tyy_2025_09_28_14_23_43_0008.png |

## 输出位置

- v1.4 全量表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_4_zdrop_reviewed\tables\record_level_episodes_all_v1_4.csv`
- v1.4 主训练候选：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_4_zdrop_reviewed\tables\train_candidate_target_episodes_v1_4.csv`
- 高度大幅下降保留样本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_4_zdrop_reviewed\tables\train_z_drop_extreme_keep_episodes_v1_4.csv`
- 上下马路但无明显大幅下降抛弃样本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_4_zdrop_reviewed\tables\discard_roadedge_without_large_zdrop_episodes_v1_4.csv`
- 复核图目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_4_zdrop_reviewed\figures\review_panels_v1_4`

## 当前建议

v1.4 比 v1.3 更贴合你的人工复核意见。下一步建议先看：

1. `01_保留_高度大幅下降极限样本`：确认这 22 个是否确实应该保留；
2. `02_抛弃_上下马路但无明显大幅下降`：抽查是否还有漏掉的可用样本；
3. 如果这两个文件夹大体符合直觉，再用 v1.4 主训练候选重跑车辆-only。

本轮没有训练模型。
