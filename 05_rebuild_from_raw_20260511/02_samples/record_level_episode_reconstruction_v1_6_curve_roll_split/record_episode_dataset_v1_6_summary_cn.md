# 完整记录级 episode 样本集 v1.6：弯道侧倾单独筛选

生成时间：2026-05-21 13:50:24

## 这次为什么改

用户指出：弯道不能只看方向盘，因为正常过弯本来就需要打方向。弯道应该重点看车辆侧倾/横滚；但如果驾驶员入弯过快开到两侧斜坡上，导致高度突然变大、突然变小，或者和正常下坡趋势不一致，这类样本也不是目标弯道侧倾样本，应从弯道候选中排除。

因此 v1.6 将弯道从主训练集中完全拆出来，并在弯道内部按侧倾和高度异常重新分层。

## v1.6 规则

- 主训练集：只保留非弯道的 v1.5 主训练候选。
- 弯道高度异常排除：`z_drop >= 2.0m`，或 `z_rise >= 0.8m`，或 `z_residual_range >= 1.5m`。
- 弯道侧倾候选：弯道上下文中，`peak_abs_roll >= 0.10rad` 或 `peak_abs_roll_rate >= 0.80rad/s`，且没有触发高度异常。
- 其它弯道样本：作为普通弯道/弱侧倾复核或对照，不进入主训练。

## 数量变化

- v1.6 非弯道主训练候选：687
- 全部弯道上下文样本：430
- 弯道侧倾候选且高度正常：95
- 弯道高度/坡度异常，疑似斜坡或路边，排除：162
- 弯道普通或弱侧倾复核/对照：173

## v1.6 分类表

| v1_6_decision | v1_6_decision_cn | count |
| --- | --- | --- |
| train_noncurve_target_extreme | 非弯道主训练候选：继承 v1.5，作为当前主训练集 | 687 |
| discard_noncurve_prior_review | 非弯道已舍弃或不适合作为当前候选：继承 v1.5 | 630 |
| review_curve_normal_or_weak_roll | 弯道普通或弱侧倾样本：不进入主训练，保留为弯道复核/对照 | 173 |
| discard_curve_slope_or_z_abnormal | 弯道高度/坡度异常：疑似开上斜坡或道路边缘，不进入主训练和弯道侧倾候选 | 162 |
| review_curve_roll_candidate_clean | 弯道侧倾候选：侧倾/横滚明显，且未触发高度异常，单独进入弯道候选池 | 95 |
| defer_noncurve_prior_review | 非弯道仍需复核或拆分：继承 v1.5 | 19 |

## 弯道侧倾候选样本

| episode_uid | subject | road_module_names | episode_duration_s | peak_abs_roll | peak_abs_roll_rate | z_drop_from_start_v1_4 | z_rise_from_start_v1_4 | z_residual_range_v1_3 | review_panel_v1_6_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rec_v1_byx_2025_09_28_17_25_18_0022 | byx | nan | 8.4200 | 1.1492 | 3.6168 | 0.7201 | 0.3202 | 1.0404 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_6_curve_roll_split\figures\review_panels_v1_6\02_弯道侧倾候选_高度正常\0067_rec_v1_byx_2025_09_28_17_25_18_0022.png |
| rec_v1_zx_2025_09_27_18_07_01_0029 | zx | nan | 13.7170 | 0.8692 | 1.8290 | 0.5924 | 0.4899 | 1.1135 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_6_curve_roll_split\figures\review_panels_v1_6\02_弯道侧倾候选_高度正常\1621_rec_v1_zx_2025_09_27_18_07_01_0029.png |
| rec_v1_hzh_2025_09_27_19_33_25_0027 | hzh | nan | 15.1960 | 0.8429 | 2.7147 | 0.5693 | 0.4135 | 0.9790 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_6_curve_roll_split\figures\review_panels_v1_6\02_弯道侧倾候选_高度正常\0602_rec_v1_hzh_2025_09_27_19_33_25_0027.png |
| rec_v1_hzh_2025_09_27_19_44_05_0023 | hzh | nan | 15.7000 | 0.8311 | 3.1634 | 0.6258 | 0.3239 | 0.9755 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_6_curve_roll_split\figures\review_panels_v1_6\02_弯道侧倾候选_高度正常\0626_rec_v1_hzh_2025_09_27_19_44_05_0023.png |
| rec_v1_zx_2025_09_27_16_46_13_0024 | zx | nan | 17.7200 | 0.8226 | 1.8112 | 0.3840 | 0.3484 | 0.7322 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_6_curve_roll_split\figures\review_panels_v1_6\02_弯道侧倾候选_高度正常\1501_rec_v1_zx_2025_09_27_16_46_13_0024.png |
| rec_v1_txj_2025_09_27_09_06_19_0003 | txj | curve1 | 15.5000 | 0.8202 | 3.7463 | 0.6439 | 0.2940 | 0.9178 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_6_curve_roll_split\figures\review_panels_v1_6\02_弯道侧倾候选_高度正常\1002_rec_v1_txj_2025_09_27_09_06_19_0003.png |
| rec_v1_zx_2025_09_27_17_29_08_0017 | zx | curve1 | 26.7750 | 0.7797 | 1.8284 | 0.1808 | 0.5212 | 0.6985 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_6_curve_roll_split\figures\review_panels_v1_6\02_弯道侧倾候选_高度正常\1549_rec_v1_zx_2025_09_27_17_29_08_0017.png |
| rec_v1_gf_2025_09_26_10_40_59_0016 | gf | nan | 23.2750 | 0.7662 | 1.9668 | 0.5082 | 0.6724 | 1.1733 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_6_curve_roll_split\figures\review_panels_v1_6\02_弯道侧倾候选_高度正常\0305_rec_v1_gf_2025_09_26_10_40_59_0016.png |
| rec_v1_byx_2025_09_28_17_35_43_0027 | byx | nan | 17.3050 | 0.7243 | 3.5001 | 0.2994 | 0.6540 | 0.9539 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_6_curve_roll_split\figures\review_panels_v1_6\02_弯道侧倾候选_高度正常\0095_rec_v1_byx_2025_09_28_17_35_43_0027.png |
| rec_v1_lxy_2025_09_28_17_55_52_0021 | lxy | differentmu_road | 32.4440 | 0.7138 | 1.6432 | 0.5928 | 0.2189 | 0.8165 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_6_curve_roll_split\figures\review_panels_v1_6\02_弯道侧倾候选_高度正常\0785_rec_v1_lxy_2025_09_28_17_55_52_0021.png |
| rec_v1_byx_2025_09_28_17_05_51_0003 | byx | curve1|middle_section | 29.5700 | 0.7090 | 2.8082 | 0.4545 | 0.4968 | 0.9365 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_6_curve_roll_split\figures\review_panels_v1_6\02_弯道侧倾候选_高度正常\0003_rec_v1_byx_2025_09_28_17_05_51_0003.png |
| rec_v1_byx_2025_09_28_17_46_00_0024 | byx | nan | 12.7800 | 0.6942 | 2.5285 | 0.6058 | 0.3500 | 0.8415 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_6_curve_roll_split\figures\review_panels_v1_6\02_弯道侧倾候选_高度正常\0120_rec_v1_byx_2025_09_28_17_46_00_0024.png |
| rec_v1_jy_2025_09_26_17_17_11_0017 | jy | differentmu_road | 16.5650 | 0.6854 | 1.8346 | 0.4547 | 0.2824 | 0.7329 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_6_curve_roll_split\figures\review_panels_v1_6\02_弯道侧倾候选_高度正常\0644_rec_v1_jy_2025_09_26_17_17_11_0017.png |
| rec_v1_yyl_2025_09_28_09_29_01_0021 | yyl | nan | 16.0990 | 0.6348 | 1.8509 | 0.6291 | 0.2084 | 0.8420 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_6_curve_roll_split\figures\review_panels_v1_6\02_弯道侧倾候选_高度正常\1190_rec_v1_yyl_2025_09_28_09_29_01_0021.png |
| rec_v1_zx_2025_09_27_17_45_11_0022 | zx | nan | 18.8430 | 0.5928 | 1.3970 | 0.6024 | 0.2806 | 0.8216 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_6_curve_roll_split\figures\review_panels_v1_6\02_弯道侧倾候选_高度正常\1574_rec_v1_zx_2025_09_27_17_45_11_0022.png |
| rec_v1_zx_2025_09_27_16_32_00_0018 | zx | nan | 16.3310 | 0.5764 | 2.1567 | 0.5047 | 0.5127 | 0.8959 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_6_curve_roll_split\figures\review_panels_v1_6\02_弯道侧倾候选_高度正常\1472_rec_v1_zx_2025_09_27_16_32_00_0018.png |
| rec_v1_zx_2025_09_27_18_17_48_0012 | zx | curve2 | 27.0760 | 0.5620 | 1.9250 | 0.4970 | 0.2864 | 0.8004 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_6_curve_roll_split\figures\review_panels_v1_6\02_弯道侧倾候选_高度正常\1634_rec_v1_zx_2025_09_27_18_17_48_0012.png |
| rec_v1_yyl_2025_09_28_09_14_23_0024 | yyl | nan | 30.8400 | 0.5240 | 1.5823 | 0.5933 | 0.2594 | 0.6846 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_6_curve_roll_split\figures\review_panels_v1_6\02_弯道侧倾候选_高度正常\1168_rec_v1_yyl_2025_09_28_09_14_23_0024.png |
| rec_v1_zx_2025_09_27_18_07_01_0010 | zx | curve1 | 19.8250 | 0.4774 | 1.7326 | 0.4894 | 0.2708 | 0.6499 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_6_curve_roll_split\figures\review_panels_v1_6\02_弯道侧倾候选_高度正常\1602_rec_v1_zx_2025_09_27_18_07_01_0010.png |
| rec_v1_txj_2025_09_27_09_06_19_0013 | txj | curve2 | 17.2000 | 0.4728 | 2.6955 | 0.4830 | 0.1427 | 0.6473 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_6_curve_roll_split\figures\review_panels_v1_6\02_弯道侧倾候选_高度正常\1012_rec_v1_txj_2025_09_27_09_06_19_0013.png |
| rec_v1_yyl_2025_09_28_09_14_23_0002 | yyl | curve1|middle_section | 13.4050 | 0.4240 | 2.0463 | 0.4496 | 0.1389 | 0.5338 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_6_curve_roll_split\figures\review_panels_v1_6\02_弯道侧倾候选_高度正常\1146_rec_v1_yyl_2025_09_28_09_14_23_0002.png |
| rec_v1_gzj_2025_09_27_11_41_47_0025 | gzj | nan | 16.6350 | 0.3872 | 2.4077 | 0.5268 | 0.2065 | 0.6820 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_6_curve_roll_split\figures\review_panels_v1_6\02_弯道侧倾候选_高度正常\0356_rec_v1_gzj_2025_09_27_11_41_47_0025.png |
| rec_v1_tyy_2025_09_28_14_44_09_0019 | tyy | curve1 | 79.7640 | 0.3749 | 1.3962 | 0.4353 | 0.1174 | 0.5118 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_6_curve_roll_split\figures\review_panels_v1_6\02_弯道侧倾候选_高度正常\1107_rec_v1_tyy_2025_09_28_14_44_09_0019.png |
| rec_v1_yyl_2025_09_28_09_14_23_0005 | yyl | curve1|middle_section | 25.8100 | 0.3742 | 1.8866 | 0.4639 | 0.1254 | 0.6042 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_6_curve_roll_split\figures\review_panels_v1_6\02_弯道侧倾候选_高度正常\1149_rec_v1_yyl_2025_09_28_09_14_23_0005.png |
| rec_v1_txj_2025_09_27_08_40_46_0013 | txj | curve2 | 22.2300 | 0.3629 | 2.2102 | 0.4448 | 0.1331 | 0.6536 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_6_curve_roll_split\figures\review_panels_v1_6\02_弯道侧倾候选_高度正常\0953_rec_v1_txj_2025_09_27_08_40_46_0013.png |
| rec_v1_zx_2025_09_27_18_17_48_0017 | zx | differentmu_road | 33.4710 | 0.2998 | 1.9290 | 0.4726 | 0.3728 | 0.9856 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_6_curve_roll_split\figures\review_panels_v1_6\02_弯道侧倾候选_高度正常\1639_rec_v1_zx_2025_09_27_18_17_48_0017.png |
| rec_v1_rjy_2025_09_28_19_51_44_0019 | rjy | nan | 14.6950 | 0.2943 | 1.2650 | 0.0135 | 0.2980 | 0.3206 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_6_curve_roll_split\figures\review_panels_v1_6\02_弯道侧倾候选_高度正常\0893_rec_v1_rjy_2025_09_28_19_51_44_0019.png |
| rec_v1_xst_2025_09_26_11_34_18_0016 | xst | differentmu_road | 21.9550 | 0.2904 | 1.4812 | 0.4909 | 0.2137 | 0.7010 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_6_curve_roll_split\figures\review_panels_v1_6\02_弯道侧倾候选_高度正常\1141_rec_v1_xst_2025_09_26_11_34_18_0016.png |
| rec_v1_zx_2025_09_27_18_00_08_0010 | zx | curve1 | 26.5600 | 0.2857 | 1.2602 | 0.4570 | 0.2455 | 0.6545 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_6_curve_roll_split\figures\review_panels_v1_6\02_弯道侧倾候选_高度正常\1591_rec_v1_zx_2025_09_27_18_00_08_0010.png |
| rec_v1_zx_2025_09_27_16_32_00_0005 | zx | curve1|middle_section | 29.7510 | 0.2743 | 1.9200 | 0.4610 | 0.1494 | 0.7453 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_6_curve_roll_split\figures\review_panels_v1_6\02_弯道侧倾候选_高度正常\1459_rec_v1_zx_2025_09_27_16_32_00_0005.png |
| rec_v1_txj_2025_09_27_08_40_46_0003 | txj | curve1 | 8.2800 | 0.2670 | 1.8138 | 0.4755 | 0.0836 | 0.5132 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_6_curve_roll_split\figures\review_panels_v1_6\02_弯道侧倾候选_高度正常\0943_rec_v1_txj_2025_09_27_08_40_46_0003.png |
| rec_v1_lxy_2025_09_28_18_19_35_0021 | lxy | nan | 13.8960 | 0.2651 | 1.0692 | 0.0143 | 0.1600 | 0.1904 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_6_curve_roll_split\figures\review_panels_v1_6\02_弯道侧倾候选_高度正常\0840_rec_v1_lxy_2025_09_28_18_19_35_0021.png |
| rec_v1_zdq_2025_09_26_16_03_48_0015 | zdq | differentmu_road | 21.0100 | 0.2172 | 1.1831 | 0.0223 | 0.2294 | 0.2625 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_6_curve_roll_split\figures\review_panels_v1_6\02_弯道侧倾候选_高度正常\1424_rec_v1_zdq_2025_09_26_16_03_48_0015.png |
| rec_v1_gzj_2025_09_27_12_17_12_0029 | gzj | nan | 11.4360 | 0.1998 | 1.5464 | 0.0271 | 0.2077 | 0.2290 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_6_curve_roll_split\figures\review_panels_v1_6\02_弯道侧倾候选_高度正常\0432_rec_v1_gzj_2025_09_27_12_17_12_0029.png |
| rec_v1_zx_2025_09_27_16_46_13_0006 | zx | nan | 23.9850 | 0.1897 | 0.8147 | 0.1296 | 0.2315 | 0.3556 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_6_curve_roll_split\figures\review_panels_v1_6\02_弯道侧倾候选_高度正常\1483_rec_v1_zx_2025_09_27_16_46_13_0006.png |
| rec_v1_txj_2025_09_27_09_17_11_0031 | txj | nan | 16.6150 | 0.1883 | 0.9006 | 0.0299 | 0.1472 | 0.1929 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_6_curve_roll_split\figures\review_panels_v1_6\02_弯道侧倾候选_高度正常\1053_rec_v1_txj_2025_09_27_09_17_11_0031.png |
| rec_v1_hzh_2025_09_26_21_03_19_0018 | hzh | differentmu_road | 21.1850 | 0.1877 | 1.1476 | 0.0189 | 0.2374 | 0.2472 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_6_curve_roll_split\figures\review_panels_v1_6\02_弯道侧倾候选_高度正常\0507_rec_v1_hzh_2025_09_26_21_03_19_0018.png |
| rec_v1_yyl_2025_09_28_09_39_01_0025 | yyl | nan | 10.6600 | 0.1866 | 0.9347 | 0.0208 | 0.1898 | 0.2114 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_6_curve_roll_split\figures\review_panels_v1_6\02_弯道侧倾候选_高度正常\1217_rec_v1_yyl_2025_09_28_09_39_01_0025.png |
| rec_v1_gf_2025_09_26_10_18_49_0013 | gf | differentmu_road | 17.4150 | 0.1833 | 1.2054 | 0.0181 | 0.2289 | 0.2417 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_6_curve_roll_split\figures\review_panels_v1_6\02_弯道侧倾候选_高度正常\0265_rec_v1_gf_2025_09_26_10_18_49_0013.png |
| rec_v1_zx_2025_09_27_18_07_01_0021 | zx | curve2 | 19.2800 | 0.1774 | 0.8483 | 0.4676 | 0.1489 | 0.5985 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_6_curve_roll_split\figures\review_panels_v1_6\02_弯道侧倾候选_高度正常\1613_rec_v1_zx_2025_09_27_18_07_01_0021.png |
| rec_v1_zx_2025_09_27_16_46_13_0017 | zx | curve2 | 6.6150 | 0.1688 | 1.5855 | 0.0022 | 0.4864 | 0.4005 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_6_curve_roll_split\figures\review_panels_v1_6\02_弯道侧倾候选_高度正常\1494_rec_v1_zx_2025_09_27_16_46_13_0017.png |
| rec_v1_zxy_2025_09_28_16_25_51_0023 | zxy | differentmu_road | 27.2850 | 0.1680 | 0.8638 | 0.0250 | 0.0965 | 0.1221 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_6_curve_roll_split\figures\review_panels_v1_6\02_弯道侧倾候选_高度正常\1728_rec_v1_zxy_2025_09_28_16_25_51_0023.png |
| rec_v1_zx_2025_09_27_16_46_13_0015 | zx | curve2 | 11.4400 | 0.1640 | 0.9983 | 0.0184 | 0.0441 | 0.0628 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_6_curve_roll_split\figures\review_panels_v1_6\02_弯道侧倾候选_高度正常\1492_rec_v1_zx_2025_09_27_16_46_13_0015.png |
| rec_v1_byx_2025_09_28_17_15_52_0022 | byx | nan | 13.4550 | 0.1622 | 0.9836 | 0.0160 | 0.1081 | 0.1218 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_6_curve_roll_split\figures\review_panels_v1_6\02_弯道侧倾候选_高度正常\0044_rec_v1_byx_2025_09_28_17_15_52_0022.png |
| rec_v1_yzy_2025_09_27_14_13_03_0022 | yzy | differentmu_road | 13.8450 | 0.1614 | 0.8654 | 0.0308 | 0.0767 | 0.1078 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_6_curve_roll_split\figures\review_panels_v1_6\02_弯道侧倾候选_高度正常\1263_rec_v1_yzy_2025_09_27_14_13_03_0022.png |
| rec_v1_hzh_2025_09_27_19_22_27_0022 | hzh | nan | 15.6400 | 0.1600 | 0.7654 | 0.0825 | 0.0248 | 0.1117 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_6_curve_roll_split\figures\review_panels_v1_6\02_弯道侧倾候选_高度正常\0574_rec_v1_hzh_2025_09_27_19_22_27_0022.png |
| rec_v1_lxy_2025_09_28_17_55_52_0019 | lxy | nan | 13.5350 | 0.1536 | 0.7871 | 0.0090 | 0.0856 | 0.0950 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_6_curve_roll_split\figures\review_panels_v1_6\02_弯道侧倾候选_高度正常\0783_rec_v1_lxy_2025_09_28_17_55_52_0019.png |
| rec_v1_byx_2025_09_28_17_05_51_0021 | byx | nan | 19.0250 | 0.1512 | 0.8337 | 0.0134 | 0.0846 | 0.0995 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_6_curve_roll_split\figures\review_panels_v1_6\02_弯道侧倾候选_高度正常\0021_rec_v1_byx_2025_09_28_17_05_51_0021.png |
| rec_v1_gf_2025_09_26_10_30_12_0014 | gf | nan | 16.7150 | 0.1486 | 0.7828 | 0.0180 | 0.0764 | 0.0950 |  |
| rec_v1_byx_2025_09_28_17_25_18_0018 | byx | differentmu_road | 11.8850 | 0.1445 | 0.7944 | 0.0036 | 0.0861 | 0.0917 |  |
| rec_v1_txj_2025_09_27_09_06_19_0012 | txj | curve2|middle_section | 7.6090 | 0.1403 | 0.6805 | 1.7943 | 0.1078 | 0.4460 |  |
| rec_v1_yzy_2025_09_27_14_37_08_0020 | yzy | differentmu_road | 10.6750 | 0.1396 | 0.8703 | 0.0245 | 0.0782 | 0.1025 |  |
| rec_v1_zdq_2025_09_26_15_27_09_0013 | zdq | differentmu_road | 25.1650 | 0.1377 | 0.8699 | 0.0097 | 0.0701 | 0.0797 |  |
| rec_v1_hzh_2025_09_26_20_50_27_0026 | hzh | nan | 11.1900 | 0.1370 | 1.1869 | 0.0195 | 0.1864 | 0.2352 |  |
| rec_v1_yyl_2025_09_28_09_14_23_0009 | yyl | nan | 8.4750 | 0.1357 | 0.7233 | 0.0455 | 0.1725 | 0.3009 |  |
| rec_v1_gzj_2025_09_27_12_28_14_0022 | gzj | nan | 14.4080 | 0.1334 | 0.8747 | 0.0219 | 0.0876 | 0.1082 |  |
| rec_v1_lxy_2025_09_28_18_19_35_0010 | lxy | curve2 | 6.6750 | 0.1332 | 0.8184 | 0.0193 | 0.0888 | 0.1070 |  |
| rec_v1_txj_2025_09_27_09_06_19_0021 | txj | nan | 12.5600 | 0.1319 | 0.7741 | 0.0162 | 0.0832 | 0.0984 |  |
| rec_v1_gf_2025_09_26_10_40_59_0011 | gf | nan | 14.0950 | 0.1317 | 0.8529 | 0.0030 | 0.0842 | 0.0872 |  |
| rec_v1_zx_2025_09_27_18_17_48_0019 | zx | nan | 18.4450 | 0.1288 | 0.7864 | 0.0070 | 0.0953 | 0.1027 |  |
| rec_v1_hzh_2025_09_27_19_33_25_0003 | hzh | curve1|middle_section | 10.6600 | 0.1220 | 0.7856 | 0.0211 | 0.0842 | 0.1199 |  |
| rec_v1_gzj_2025_09_27_11_41_47_0003 | gzj | curve1|middle_section | 10.6300 | 0.1218 | 0.6711 | 0.0196 | 0.0738 | 0.0946 |  |
| rec_v1_rjy_2025_09_28_19_33_26_0020 | rjy | nan | 14.6880 | 0.1202 | 0.7660 | 0.0148 | 0.0896 | 0.1175 |  |
| rec_v1_jy_2025_09_26_18_01_40_0023 | jy | differentmu_road | 34.1650 | 0.1201 | 0.6443 | 0.0218 | 0.0755 | 0.0975 |  |
| rec_v1_zxy_2025_09_28_16_01_55_0015 | zxy | curve1 | 18.2360 | 0.1181 | 1.0954 | 0.0101 | 0.1829 | 0.1927 |  |
| rec_v1_hzh_2025_09_27_19_22_27_0018 | hzh | differentmu_road | 18.8120 | 0.1178 | 0.7694 | 0.0120 | 0.0826 | 0.0960 |  |
| rec_v1_hzh_2025_09_27_19_44_05_0019 | hzh | differentmu_road | 7.7200 | 0.1167 | 0.8388 | 0.0085 | 0.0805 | 0.0866 |  |
| rec_v1_rjy_2025_09_28_20_15_42_0001 | rjy | curve1 | 12.3840 | 0.1166 | 0.8682 | 0.0074 | 0.0660 | 0.0749 |  |
| rec_v1_txj_2025_09_27_08_53_44_0011 | txj | curve2 | 15.9880 | 0.1165 | 1.0434 | 0.0138 | 0.0665 | 0.0832 |  |
| rec_v1_zx_2025_09_27_18_07_01_0026 | zx | differentmu_road | 9.4770 | 0.1127 | 0.7317 | 0.0202 | 0.0971 | 0.1224 |  |
| rec_v1_tyy_2025_09_28_14_44_09_0018 | tyy | nan | 12.1610 | 0.1115 | 0.6951 | 0.0888 | 0.0103 | 0.0928 |  |
| rec_v1_zdq_2025_09_26_15_37_30_0011 | zdq | nan | 15.6950 | 0.1114 | 0.8356 | 0.0209 | 0.0735 | 0.0938 |  |
| rec_v1_lxy_2025_09_28_18_19_35_0027 | lxy | nan | 15.5550 | 0.1112 | 0.6722 | 0.0178 | 0.0133 | 0.0266 |  |
| rec_v1_lxy_2025_09_28_18_06_16_0028 | lxy | nan | 8.1250 | 0.1093 | 0.2666 | 0.0048 | 0.0086 | 0.0120 |  |
| rec_v1_zdq_2025_09_26_16_03_48_0018 | zdq | nan | 11.8600 | 0.1086 | 0.2802 | 0.0030 | 0.0125 | 0.0127 |  |
| rec_v1_rjy_2025_09_28_20_15_42_0021 | rjy | nan | 18.5260 | 0.1077 | 0.3451 | 0.0010 | 0.0128 | 0.0130 |  |
| rec_v1_byx_2025_09_28_17_15_52_0012 | byx | curve2 | 12.3500 | 0.1065 | 0.9525 | 0.0047 | 0.0101 | 0.0149 |  |
| rec_v1_zx_2025_09_27_16_32_00_0022 | zx | nan | 11.9050 | 0.1048 | 0.5958 | 0.0059 | 0.0073 | 0.0133 |  |
| rec_v1_yyl_2025_09_28_09_14_23_0008 | yyl | curve1 | 33.2300 | 0.1037 | 0.5721 | 0.0091 | 0.4756 | 0.3890 |  |
| rec_v1_tyy_2025_09_28_14_23_43_0002 | tyy | curve1|middle_section | 10.8850 | 0.1033 | 0.7170 | 0.0136 | 0.1057 | 0.1195 |  |
| rec_v1_jy_2025_09_26_17_17_11_0014 | jy | nan | 15.6350 | 0.1029 | 0.2292 | 0.0023 | 0.0073 | 0.0096 |  |
| rec_v1_txj_2025_09_27_08_40_46_0022 | txj | nan | 7.6200 | 0.1025 | 0.4215 | 0.0052 | 0.0075 | 0.0125 |  |
| rec_v1_zdq_2025_09_26_15_14_51_0004 | zdq | curve1|middle_section | 3.0150 | 0.1022 | 0.5789 | 0.0881 | 0.0126 | 0.0802 |  |
| rec_v1_rjy_2025_09_28_20_02_20_0016 | rjy | curve2 | 4.7150 | 0.1014 | 0.5852 | 0.0061 | 0.0086 | 0.0134 |  |
| rec_v1_txj_2025_09_27_09_06_19_0018 | txj | nan | 8.2800 | 0.1012 | 0.2382 | 0.0018 | 0.0086 | 0.0120 |  |
| rec_v1_txj_2025_09_27_08_53_44_0020 | txj | differentmu_road | 15.0900 | 0.1011 | 0.5731 | 0.0028 | 0.0097 | 0.0125 |  |
| rec_v1_zxy_2025_09_28_16_01_55_0017 | zxy | curve1 | 3.1110 | 0.1006 | 0.5931 | 0.0001 | 0.1238 | 0.0490 |  |
| rec_v1_zx_2025_09_27_17_14_07_0022 | zx | nan | 13.0400 | 0.1005 | 0.6516 | 0.0038 | 0.0092 | 0.0129 |  |
| rec_v1_rjy_2025_09_28_19_44_42_0000 | rjy | curve1|middle_section | 9.6300 | 0.1002 | 0.8200 | 0.0152 | 0.0858 | 0.1018 |  |
| rec_v1_byx_2025_09_28_17_05_51_0018 | byx | differentmu_road | 19.5000 | 0.0983 | 0.8129 | 0.0192 | 0.0773 | 0.0945 |  |
| rec_v1_cwh_2025_09_26_19_35_47_0019 | cwh | nan | 11.4000 | 0.0902 | 0.8088 | 0.0169 | 0.0740 | 0.0944 |  |
| rec_v1_yyl_2025_09_28_09_49_11_0022 | yyl | nan | 13.1540 | 0.0843 | 0.8335 | 0.0079 | 0.0769 | 0.0857 |  |
| rec_v1_hzh_2025_09_26_21_03_19_0003 | hzh | curve1|middle_section | 10.3100 | 0.0806 | 0.8775 | 0.0111 | 0.0813 | 0.0931 |  |
| rec_v1_byx_2025_09_28_17_46_00_0003 | byx | curve1 | 7.8250 | 0.0771 | 0.8641 | 0.0128 | 0.0834 | 0.0975 |  |
| rec_v1_tyy_2025_09_28_14_40_01_0003 | tyy | curve1|middle_section | 7.5700 | 0.0755 | 0.8454 | 0.0193 | 0.0806 | 0.1047 |  |

## 输出位置

- v1.6 全量表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_6_curve_roll_split\tables\record_level_episodes_all_v1_6.csv`
- 非弯道主训练候选：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_6_curve_roll_split\tables\train_candidate_noncurve_episodes_v1_6.csv`
- 弯道侧倾候选：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_6_curve_roll_split\tables\curve_roll_candidate_clean_episodes_v1_6.csv`
- 弯道高度异常排除：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_6_curve_roll_split\tables\discard_curve_slope_or_z_abnormal_episodes_v1_6.csv`
- 弯道普通或弱侧倾复核：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_6_curve_roll_split\tables\curve_normal_or_weak_roll_review_episodes_v1_6.csv`
- 复核图目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_6_curve_roll_split\figures\review_panels_v1_6`

## 当前建议

后续不要再把弯道和其它极限工况混在一个训练池里。可以先用非弯道主训练候选跑车辆-only；弯道路线则单独使用“弯道侧倾候选且高度正常”的样本做专门分析。

本轮没有训练模型。
