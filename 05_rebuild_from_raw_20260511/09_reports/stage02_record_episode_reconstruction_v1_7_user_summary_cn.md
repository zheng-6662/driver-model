# 完整记录级 episode 样本集 v1.7：弯道侧倾单独筛选

生成时间：2026-05-22 10:05:53

## 这次为什么改

用户进一步指出：我之前把“高度持续下降/残差较大”过度当成异常了。对于下坡弯道，正常样本应表现为高度连续、平滑下降，允许有少量波动；真正异常的是高度突变、台阶式变化、反常抬升、非平滑大残差，或强动态片段的高度轨迹不像正常下坡。

因此 v1.7 将弯道从主训练集中完全拆出来，并在弯道内部按侧倾和高度异常重新分层。

## v1.7 规则

- 主训练集：只保留非弯道的 v1.5 主训练候选。
- 平滑下坡弯道：`z_drop >= 1.0m`，`z_rise <= 0.3m`，`z_monotonic_fraction >= 0.82`，同时残差速度和残差范围不过大。
- 平滑下坡弯道侧倾候选：满足平滑下坡，并且 `peak_abs_roll >= 0.10rad` 或 `peak_abs_roll_rate >= 0.80rad/s`。
- 弯道高度轨迹异常：高度明显反常抬升、残差速度过大、残差范围过大、单调性过低，或强动态但高度轨迹不像正常下坡。
- 其它弯道样本：保留为高度形态不明或弱侧倾复核，不进入当前主训练。

## 数量变化

- v1.7 非弯道主训练候选：687
- 全部弯道上下文样本：430
- 平滑下坡弯道侧倾候选：54
- 平滑下坡弯道普通/弱侧倾：54
- 弯道高度轨迹异常，排除：202
- 弯道高度形态不明，复核：120

## v1.7 分类表

| v1_7_decision | v1_7_decision_cn | count |
| --- | --- | --- |
| train_noncurve_target_extreme | 非弯道主训练候选：继承 v1.5，作为当前主训练集 | 687 |
| discard_noncurve_prior_review | 非弯道已舍弃或不适合作为当前候选：继承 v1.5 | 630 |
| discard_curve_z_profile_abnormal | 弯道高度轨迹异常：高度突变、非平滑、非连续下降或强动态但高度轨迹不像正常下坡 | 202 |
| review_curve_unclear_or_weak | 弯道高度形态或侧倾较弱：保留为复核/对照 | 104 |
| review_curve_smooth_downhill_roll_candidate | 弯道有效样本：高度连续平滑下降，允许小波动，且侧倾/横滚明显 | 54 |
| review_curve_smooth_downhill_normal_or_weak | 弯道有效对照：高度连续平滑下降，但侧倾/横滚不强 | 54 |
| defer_noncurve_prior_review | 非弯道仍需复核或拆分：继承 v1.5 | 19 |
| review_curve_unclear_profile_roll_candidate | 弯道侧倾候选但高度形态不够明确：需要人工复核 | 16 |

## 平滑下坡弯道侧倾候选样本

| episode_uid | subject | road_module_names | episode_duration_s | peak_abs_roll | peak_abs_roll_rate | z_drop_from_start_v1_4 | z_rise_from_start_v1_4 | z_residual_range_v1_3 | z_residual_rate_peak_v1_3 | z_monotonic_fraction_v1_3 | review_panel_v1_7_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rec_v1_zx_2025_09_27_17_25_16_0006 | zx | curve1 | 15.4220 | 1.3907 | 1.1964 | 3.1751 | 0.0002 | 2.6204 | 1.7470 | 0.8554 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_7_curve_zprofile_revised\figures\review_panels_v1_7\01_平滑下坡弯道_侧倾候选\1531_rec_v1_zx_2025_09_27_17_25_16_0006.png |
| rec_v1_byx_2025_09_28_17_15_52_0006 | byx | curve1 | 16.9250 | 0.3130 | 1.3972 | 6.9595 | 0.0001 | 1.6932 | 1.5652 | 0.9117 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_7_curve_zprofile_revised\figures\review_panels_v1_7\01_平滑下坡弯道_侧倾候选\0028_rec_v1_byx_2025_09_28_17_15_52_0006.png |
| rec_v1_jy_2025_09_26_17_40_51_0009 | jy | curve2|middle_section | 19.2700 | 0.3123 | 1.6156 | 5.9041 | 0.0558 | 0.8380 | 1.6703 | 0.9284 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_7_curve_zprofile_revised\figures\review_panels_v1_7\01_平滑下坡弯道_侧倾候选\0672_rec_v1_jy_2025_09_26_17_40_51_0009.png |
| rec_v1_lxy_2025_09_28_18_06_16_0006 | lxy | curve1 | 17.4600 | 0.3105 | 2.4149 | 6.7856 | 0.0068 | 1.7475 | 1.7779 | 0.9099 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_7_curve_zprofile_revised\figures\review_panels_v1_7\01_平滑下坡弯道_侧倾候选\0794_rec_v1_lxy_2025_09_28_18_06_16_0006.png |
| rec_v1_gzj_2025_09_27_11_41_47_0013 | gzj | curve2|middle_section | 20.4250 | 0.3025 | 0.8905 | 4.1285 | 0.0002 | 0.9214 | 0.8242 | 0.8735 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_7_curve_zprofile_revised\figures\review_panels_v1_7\01_平滑下坡弯道_侧倾候选\0344_rec_v1_gzj_2025_09_27_11_41_47_0013.png |
| rec_v1_jy_2025_09_26_17_40_51_0005 | jy | curve1 | 24.3900 | 0.2850 | 0.8667 | 6.9949 | 0.0004 | 1.7021 | 1.0064 | 0.9086 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_7_curve_zprofile_revised\figures\review_panels_v1_7\01_平滑下坡弯道_侧倾候选\0668_rec_v1_jy_2025_09_26_17_40_51_0005.png |
| rec_v1_zdq_2025_09_26_15_52_46_0003 | zdq | curve1 | 42.8750 | 0.2672 | 1.3663 | 6.9952 | 0.0010 | 2.4898 | 1.4387 | 0.8569 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_7_curve_zprofile_revised\figures\review_panels_v1_7\01_平滑下坡弯道_侧倾候选\1394_rec_v1_zdq_2025_09_26_15_52_46_0003.png |
| rec_v1_zx_2025_09_27_16_46_13_0004 | zx | curve1 | 10.7450 | 0.2632 | 1.0114 | 3.8995 | 0.0006 | 1.8320 | 1.4795 | 0.8786 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_7_curve_zprofile_revised\figures\review_panels_v1_7\01_平滑下坡弯道_侧倾候选\1481_rec_v1_zx_2025_09_27_16_46_13_0004.png |
| rec_v1_zxy_2025_09_28_16_01_55_0005 | zxy | curve1 | 25.0300 | 0.2612 | 1.1908 | 5.3789 | 0.0902 | 0.8725 | 2.2392 | 0.9031 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_7_curve_zprofile_revised\figures\review_panels_v1_7\01_平滑下坡弯道_侧倾候选\1652_rec_v1_zxy_2025_09_28_16_01_55_0005.png |
| rec_v1_gzj_2025_09_27_11_53_25_0006 | gzj | curve1 | 27.6100 | 0.2552 | 1.1726 | 7.0215 | 0.0195 | 2.4348 | 1.2621 | 0.8244 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_7_curve_zprofile_revised\figures\review_panels_v1_7\01_平滑下坡弯道_侧倾候选\0364_rec_v1_gzj_2025_09_27_11_53_25_0006.png |
| rec_v1_hzh_2025_09_26_20_50_27_0005 | hzh | curve1 | 19.6050 | 0.2464 | 1.4323 | 6.9948 | 0.0028 | 1.8787 | 1.5860 | 0.8521 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_7_curve_zprofile_revised\figures\review_panels_v1_7\01_平滑下坡弯道_侧倾候选\0462_rec_v1_hzh_2025_09_26_20_50_27_0005.png |
| rec_v1_txj_2025_09_27_08_53_44_0009 | txj | curve2|middle_section | 15.2050 | 0.2191 | 0.9736 | 3.3761 | 0.1032 | 0.4160 | 0.7632 | 0.9635 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_7_curve_zprofile_revised\figures\review_panels_v1_7\01_平滑下坡弯道_侧倾候选\0980_rec_v1_txj_2025_09_27_08_53_44_0009.png |
| rec_v1_hzh_2025_09_27_19_22_27_0003 | hzh | curve1 | 19.6370 | 0.2080 | 1.4144 | 6.9998 | 0.0006 | 2.2656 | 1.8127 | 0.8392 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_7_curve_zprofile_revised\figures\review_panels_v1_7\01_平滑下坡弯道_侧倾候选\0555_rec_v1_hzh_2025_09_27_19_22_27_0003.png |
| rec_v1_gf_2025_09_26_10_30_12_0004 | gf | curve1 | 19.9700 | 0.2068 | 1.2111 | 6.9942 | 0.0033 | 2.1749 | 2.0256 | 0.8584 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_7_curve_zprofile_revised\figures\review_panels_v1_7\01_平滑下坡弯道_侧倾候选\0274_rec_v1_gf_2025_09_26_10_30_12_0004.png |
| rec_v1_gzj_2025_09_27_11_41_47_0005 | gzj | curve1 | 24.8700 | 0.1953 | 1.1589 | 7.0173 | 0.0028 | 1.3935 | 1.1852 | 0.9024 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_7_curve_zprofile_revised\figures\review_panels_v1_7\01_平滑下坡弯道_侧倾候选\0336_rec_v1_gzj_2025_09_27_11_41_47_0005.png |
| rec_v1_jy_2025_09_26_18_01_40_0007 | jy | curve1 | 21.7800 | 0.1908 | 1.1139 | 6.9931 | 0.0039 | 2.0337 | 1.5582 | 0.9291 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_7_curve_zprofile_revised\figures\review_panels_v1_7\01_平滑下坡弯道_侧倾候选\0709_rec_v1_jy_2025_09_26_18_01_40_0007.png |
| rec_v1_hzh_2025_09_27_19_22_27_0009 | hzh | curve2|middle_section | 15.8970 | 0.1821 | 1.4602 | 3.6172 | 0.0706 | 0.6652 | 1.8091 | 0.9088 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_7_curve_zprofile_revised\figures\review_panels_v1_7\01_平滑下坡弯道_侧倾候选\0561_rec_v1_hzh_2025_09_27_19_22_27_0009.png |
| rec_v1_txj_2025_09_27_08_40_46_0012 | txj | curve2|middle_section | 17.6200 | 0.1722 | 0.9733 | 4.4647 | 0.1068 | 0.6874 | 0.4739 | 0.9753 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_7_curve_zprofile_revised\figures\review_panels_v1_7\01_平滑下坡弯道_侧倾候选\0952_rec_v1_txj_2025_09_27_08_40_46_0012.png |
| rec_v1_gzj_2025_09_27_12_04_23_0004 | gzj | curve1 | 15.8050 | 0.1692 | 0.7079 | 7.0011 | 0.0022 | 1.7718 | 1.6896 | 0.9309 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_7_curve_zprofile_revised\figures\review_panels_v1_7\01_平滑下坡弯道_侧倾候选\0391_rec_v1_gzj_2025_09_27_12_04_23_0004.png |
| rec_v1_hzh_2025_09_26_21_17_02_0009 | hzh | curve1 | 23.6400 | 0.1674 | 1.2022 | 6.9963 | 0.0027 | 2.0457 | 1.2134 | 0.8858 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_7_curve_zprofile_revised\figures\review_panels_v1_7\01_平滑下坡弯道_侧倾候选\0522_rec_v1_hzh_2025_09_26_21_17_02_0009.png |
| rec_v1_byx_2025_09_28_17_05_51_0010 | byx | curve2|middle_section | 22.4950 | 0.1620 | 0.9369 | 5.8408 | 0.0639 | 0.7054 | 0.9801 | 0.9631 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_7_curve_zprofile_revised\figures\review_panels_v1_7\01_平滑下坡弯道_侧倾候选\0010_rec_v1_byx_2025_09_28_17_05_51_0010.png |
| rec_v1_zdq_2025_09_26_15_37_30_0002 | zdq | curve1 | 21.1400 | 0.1507 | 0.7961 | 6.9941 | 0.0001 | 2.0086 | 1.6151 | 0.9267 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_7_curve_zprofile_revised\figures\review_panels_v1_7\01_平滑下坡弯道_侧倾候选\1380_rec_v1_zdq_2025_09_26_15_37_30_0002.png |
| rec_v1_gzj_2025_09_27_11_53_25_0017 | gzj | curve2|middle_section | 8.1750 | 0.1419 | 0.6326 | 2.0316 | 0.0979 | 0.5754 | 0.3983 | 0.9627 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_7_curve_zprofile_revised\figures\review_panels_v1_7\01_平滑下坡弯道_侧倾候选\0375_rec_v1_gzj_2025_09_27_11_53_25_0017.png |
| rec_v1_txj_2025_09_27_09_06_19_0012 | txj | curve2|middle_section | 7.6090 | 0.1403 | 0.6805 | 1.7943 | 0.1078 | 0.4460 | 0.8145 | 0.9770 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_7_curve_zprofile_revised\figures\review_panels_v1_7\01_平滑下坡弯道_侧倾候选\1011_rec_v1_txj_2025_09_27_09_06_19_0012.png |
| rec_v1_tyy_2025_09_28_14_57_17_0013 | tyy | curve2|middle_section | 17.2100 | 0.1387 | 0.5222 | 2.6725 | 0.0736 | 0.4258 | 0.5824 | 0.9840 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_7_curve_zprofile_revised\figures\review_panels_v1_7\01_平滑下坡弯道_侧倾候选\1121_rec_v1_tyy_2025_09_28_14_57_17_0013.png |
| rec_v1_zdq_2025_09_26_15_52_46_0007 | zdq | curve2|middle_section | 16.6350 | 0.1374 | 0.5973 | 3.9575 | 0.0028 | 0.5445 | 0.4144 | 0.9763 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_7_curve_zprofile_revised\figures\review_panels_v1_7\01_平滑下坡弯道_侧倾候选\1398_rec_v1_zdq_2025_09_26_15_52_46_0007.png |
| rec_v1_jy_2025_09_26_17_51_46_0003 | jy | curve1 | 19.7450 | 0.1368 | 1.0808 | 6.9983 | 0.0001 | 2.0790 | 1.8659 | 0.9121 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_7_curve_zprofile_revised\figures\review_panels_v1_7\01_平滑下坡弯道_侧倾候选\0686_rec_v1_jy_2025_09_26_17_51_46_0003.png |
| rec_v1_gzj_2025_09_27_12_17_12_0016 | gzj | curve2|middle_section | 19.0110 | 0.1365 | 1.1639 | 3.2307 | 0.1025 | 0.6356 | 1.6028 | 0.9408 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_7_curve_zprofile_revised\figures\review_panels_v1_7\01_平滑下坡弯道_侧倾候选\0419_rec_v1_gzj_2025_09_27_12_17_12_0016.png |
| rec_v1_hzh_2025_09_26_21_17_02_0019 | hzh | curve2|middle_section | 20.1650 | 0.1364 | 1.0321 | 4.9578 | 0.0493 | 0.4756 | 0.7507 | 0.9556 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_7_curve_zprofile_revised\figures\review_panels_v1_7\01_平滑下坡弯道_侧倾候选\0532_rec_v1_hzh_2025_09_26_21_17_02_0019.png |
| rec_v1_tyy_2025_09_28_14_23_43_0003 | tyy | curve1 | 14.8350 | 0.1330 | 0.8696 | 5.0486 | 0.0002 | 1.6876 | 0.9359 | 0.9825 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_7_curve_zprofile_revised\figures\review_panels_v1_7\01_平滑下坡弯道_侧倾候选\1057_rec_v1_tyy_2025_09_28_14_23_43_0003.png |
| rec_v1_hzh_2025_09_27_19_33_25_0012 | hzh | curve2|middle_section | 15.2400 | 0.1300 | 0.9387 | 3.6469 | 0.0831 | 0.4403 | 1.2338 | 0.9580 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_7_curve_zprofile_revised\figures\review_panels_v1_7\01_平滑下坡弯道_侧倾候选\0587_rec_v1_hzh_2025_09_27_19_33_25_0012.png |
| rec_v1_zx_2025_09_27_17_14_07_0005 | zx | curve1 | 18.7650 | 0.1281 | 0.2072 | 6.9973 | 0.0021 | 1.9809 | 1.5053 | 0.9146 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_7_curve_zprofile_revised\figures\review_panels_v1_7\01_平滑下坡弯道_侧倾候选\1507_rec_v1_zx_2025_09_27_17_14_07_0005.png |
| rec_v1_zx_2025_09_27_18_00_08_0003 | zx | nan | 16.0600 | 0.1268 | 0.8257 | 4.1256 | 0.0957 | 0.5545 | 0.9373 | 0.9651 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_7_curve_zprofile_revised\figures\review_panels_v1_7\01_平滑下坡弯道_侧倾候选\1584_rec_v1_zx_2025_09_27_18_00_08_0003.png |
| rec_v1_zx_2025_09_27_16_32_00_0010 | zx | curve2|middle_section | 16.0650 | 0.1248 | 0.5071 | 3.5762 | 0.0852 | 0.6037 | 0.7230 | 0.9720 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_7_curve_zprofile_revised\figures\review_panels_v1_7\01_平滑下坡弯道_侧倾候选\1464_rec_v1_zx_2025_09_27_16_32_00_0010.png |
| rec_v1_txj_2025_09_27_09_17_11_0018 | txj | curve2|middle_section | 10.2750 | 0.1204 | 0.5930 | 2.6415 | 0.1031 | 0.3422 | 0.7522 | 1.0000 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_7_curve_zprofile_revised\figures\review_panels_v1_7\01_平滑下坡弯道_侧倾候选\1040_rec_v1_txj_2025_09_27_09_17_11_0018.png |
| rec_v1_hzh_2025_09_26_20_50_27_0017 | hzh | curve2|middle_section | 26.2450 | 0.1185 | 0.9443 | 4.4023 | 0.0013 | 1.5402 | 0.8465 | 0.8848 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_7_curve_zprofile_revised\figures\review_panels_v1_7\01_平滑下坡弯道_侧倾候选\0474_rec_v1_hzh_2025_09_26_20_50_27_0017.png |
| rec_v1_zx_2025_09_27_17_29_08_0004 | zx | nan | 15.4100 | 0.1163 | 1.1691 | 3.7879 | 0.0809 | 0.3981 | 0.8629 | 0.9789 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_7_curve_zprofile_revised\figures\review_panels_v1_7\01_平滑下坡弯道_侧倾候选\1536_rec_v1_zx_2025_09_27_17_29_08_0004.png |
| rec_v1_zxy_2025_09_28_16_12_11_0014 | zxy | curve2|middle_section | 15.0810 | 0.1147 | 0.4031 | 3.8240 | 0.0653 | 0.4844 | 1.9112 | 1.0000 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_7_curve_zprofile_revised\figures\review_panels_v1_7\01_平滑下坡弯道_侧倾候选\1691_rec_v1_zxy_2025_09_28_16_12_11_0014.png |
| rec_v1_zx_2025_09_27_16_46_13_0013 | zx | curve2|middle_section | 19.1200 | 0.1139 | 0.4904 | 4.5598 | 0.0756 | 0.5148 | 0.2969 | 0.9979 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_7_curve_zprofile_revised\figures\review_panels_v1_7\01_平滑下坡弯道_侧倾候选\1490_rec_v1_zx_2025_09_27_16_46_13_0013.png |
| rec_v1_zx_2025_09_27_18_07_01_0020 | zx | curve2|middle_section | 14.8280 | 0.1121 | 0.6441 | 3.2106 | 0.0898 | 0.4201 | 0.5351 | 0.9875 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_7_curve_zprofile_revised\figures\review_panels_v1_7\01_平滑下坡弯道_侧倾候选\1612_rec_v1_zx_2025_09_27_18_07_01_0020.png |
| rec_v1_byx_2025_09_28_17_35_43_0016 | byx | curve2|middle_section | 20.7750 | 0.1110 | 0.3977 | 5.2375 | 0.1089 | 0.7300 | 0.4383 | 0.9827 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_7_curve_zprofile_revised\figures\review_panels_v1_7\01_平滑下坡弯道_侧倾候选\0084_rec_v1_byx_2025_09_28_17_35_43_0016.png |
| rec_v1_gzj_2025_09_27_12_28_14_0004 | gzj | curve1 | 21.2400 | 0.1071 | 0.2581 | 6.9834 | 0.0144 | 1.6348 | 1.0697 | 0.9729 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_7_curve_zprofile_revised\figures\review_panels_v1_7\01_平滑下坡弯道_侧倾候选\0438_rec_v1_gzj_2025_09_27_12_28_14_0004.png |
| rec_v1_txj_2025_09_27_08_53_44_0005 | txj | curve1 | 14.0700 | 0.1058 | 0.1474 | 3.9787 | 0.0161 | 1.3017 | 0.8355 | 0.8969 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_7_curve_zprofile_revised\figures\review_panels_v1_7\01_平滑下坡弯道_侧倾候选\0976_rec_v1_txj_2025_09_27_08_53_44_0005.png |
| rec_v1_yyl_2025_09_28_09_39_01_0005 | yyl | curve1 | 17.2430 | 0.1056 | 0.3185 | 6.9971 | 0.0001 | 1.9304 | 1.5098 | 0.9493 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_7_curve_zprofile_revised\figures\review_panels_v1_7\01_平滑下坡弯道_侧倾候选\1197_rec_v1_yyl_2025_09_28_09_39_01_0005.png |
| rec_v1_gf_2025_09_26_10_18_49_0003 | gf | curve1 | 18.6920 | 0.1047 | 0.2990 | 6.9985 | 0.0010 | 1.9362 | 1.4617 | 0.9318 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_7_curve_zprofile_revised\figures\review_panels_v1_7\01_平滑下坡弯道_侧倾候选\0255_rec_v1_gf_2025_09_26_10_18_49_0003.png |
| rec_v1_gf_2025_09_26_10_30_12_0008 | gf | curve2|middle_section | 17.9200 | 0.1047 | 0.4365 | 5.1971 | 0.0917 | 0.6803 | 0.4243 | 1.0000 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_7_curve_zprofile_revised\figures\review_panels_v1_7\01_平滑下坡弯道_侧倾候选\0278_rec_v1_gf_2025_09_26_10_30_12_0008.png |
| rec_v1_gzj_2025_09_27_12_28_14_0010 | gzj | curve2|middle_section | 19.9920 | 0.1043 | 0.5299 | 5.3079 | 0.0008 | 0.4788 | 0.3876 | 0.9797 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_7_curve_zprofile_revised\figures\review_panels_v1_7\01_平滑下坡弯道_侧倾候选\0444_rec_v1_gzj_2025_09_27_12_28_14_0010.png |
| rec_v1_yzy_2025_09_27_14_13_03_0010 | yzy | curve2|middle_section | 17.3300 | 0.1041 | 0.3604 | 4.1001 | 0.0810 | 0.6579 | 0.3530 | 0.9824 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_7_curve_zprofile_revised\figures\review_panels_v1_7\01_平滑下坡弯道_侧倾候选\1251_rec_v1_yzy_2025_09_27_14_13_03_0010.png |
| rec_v1_rjy_2025_09_28_20_02_20_0011 | rjy | curve2|middle_section | 19.0400 | 0.1026 | 0.4327 | 4.1551 | 0.0003 | 0.5224 | 0.4608 | 0.9840 |  |
| rec_v1_gzj_2025_09_27_12_04_23_0015 | gzj | curve2|middle_section | 16.8550 | 0.1023 | 0.3211 | 3.3460 | 0.0728 | 0.8763 | 0.3164 | 0.9576 |  |
| rec_v1_zt_2025_09_28_11_20_08_0011 | zt | curve2|middle_section | 14.6700 | 0.1011 | 0.3414 | 3.9003 | 0.0969 | 0.3716 | 0.3728 | 1.0000 |  |
| rec_v1_yyl_2025_09_28_09_29_01_0009 | yyl | curve2|middle_section | 15.3190 | 0.1005 | 0.7368 | 3.6402 | 0.0848 | 0.5554 | 1.6029 | 0.9850 |  |
| rec_v1_zdq_2025_09_26_15_27_09_0007 | zdq | curve2|middle_section | 28.8150 | 0.1000 | 0.4648 | 5.9542 | 0.0021 | 0.6954 | 0.3260 | 0.9826 |  |
| rec_v1_yzy_2025_09_27_14_13_03_0004 | yzy | curve1 | 22.1200 | 0.0986 | 0.8360 | 6.9963 | 0.0001 | 2.0395 | 1.4801 | 0.9591 |  |

## 输出位置

- v1.7 全量表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_7_curve_zprofile_revised\tables\record_level_episodes_all_v1_7.csv`
- 非弯道主训练候选：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_7_curve_zprofile_revised\tables\train_candidate_noncurve_episodes_v1_7.csv`
- 平滑下坡弯道侧倾候选：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_7_curve_zprofile_revised\tables\curve_smooth_downhill_roll_candidate_episodes_v1_7.csv`
- 平滑下坡弯道普通/弱侧倾：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_7_curve_zprofile_revised\tables\curve_smooth_downhill_normal_or_weak_episodes_v1_7.csv`
- 弯道高度轨迹异常排除：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_7_curve_zprofile_revised\tables\discard_curve_z_profile_abnormal_episodes_v1_7.csv`
- 弯道高度形态不明复核：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_7_curve_zprofile_revised\tables\curve_unclear_profile_review_episodes_v1_7.csv`
- 复核图目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_7_curve_zprofile_revised\figures\review_panels_v1_7`

## 当前建议

后续不要再把弯道和其它极限工况混在一个训练池里。可以先用非弯道主训练候选跑车辆-only；弯道路线则单独使用“弯道侧倾候选且高度正常”的样本做专门分析。

本轮没有训练模型。
