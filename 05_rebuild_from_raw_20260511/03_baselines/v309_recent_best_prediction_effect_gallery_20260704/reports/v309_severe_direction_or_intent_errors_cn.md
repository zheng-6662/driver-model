# v309 严重方向/意图错误事件复核表

这张表不是按普通 RMSE 排名，而是按人工可解释的严重性筛选：方向盘峰值方向相反、真实近似无大动作却预测大动作、真实极端动作被大幅低估、或 v307 相比 v300 明显退化。

- 输入总事件数：232
- 严重方向/意图候选数：37
- 用户截图命中的事件数：5
- CSV 明细：`05_rebuild_from_raw_20260511\03_baselines\v309_recent_best_prediction_effect_gallery_20260704\tables\v309_severe_direction_or_intent_errors.csv`

## 用户截图对应事件

| 图中编号 | event_uid | 粗标签 | 原始场景/route | v307_rmse | v300_rmse | delta | true_peak | v307_peak | 严重问题 |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| #014 | `gzj_Entity_Recording_2025_09_27_12_28_14_v108_054` | 平路过弯 | 普通弯道事件 / zero_cross | 0.394 | 0.381 | +0.014 | +0.589 | -0.836 | 峰值方向相反: true_peak=0.589, v307_peak=-0.836 |
| #017 | `zx_Entity_Recording_2025_09_27_16_46_13_v108_035` | 紧急变道/猛打方向失稳 | 直道事件 / extreme_peak | 3.362 | 3.313 | +0.049 | +4.319 | +0.507 | 极端动作幅值严重低估: true_peak=4.319, v307_peak=0.507；大动作场景整体误差高: v307_rmse=3.362 |
| #019 | `zx_Entity_Recording_2025_09_27_17_45_11_v108_023` | 下坡过弯 | 下坡弯道事件 / strong_event | 2.035 | 2.019 | +0.016 | +3.121 | +1.968 | 大动作场景整体误差高: v307_rmse=2.035 |
| #020 | `zx_Entity_Recording_2025_09_27_17_14_07_v108_016` | 连续变道/连续左右修正 | 直道事件 / vehicle_strong | 1.591 | 1.106 | +0.485 | +0.089 | +2.981 | 真实近似无大动作但预测大动作: true_peak=0.089, v307_peak=2.981；v307 比 v300 更差: delta=+0.485 |
| #023 | `gzj_Entity_Recording_2025_09_27_11_41_47_v108_048` | 平路过弯 | 普通弯道事件 / vehicle_strong | 1.394 | 1.199 | +0.195 | -0.544 | +1.884 | 峰值方向相反: true_peak=-0.544, v307_peak=1.884；v307 比 v300 更差: delta=+0.195 |

## 全体 test delay0 严重候选 Top 20

| 严重序号 | 图中编号 | event_uid | 粗标签 | v307_rmse | v300_rmse | delta | true_peak | v307_peak | 问题标签 |
|---:|---|---|---|---:|---:|---:|---:|---:|---|
| 1 | #018 | `jy_Entity_Recording_2025_09_26_17_29_44_v108_006` | 下坡过弯 | 2.900 | 2.507 | +0.394 | -5.131 | -0.747 | missed_extreme_amplitude;large_event_high_rmse;regression_vs_v300 |
| 2 | #017 | `zx_Entity_Recording_2025_09_27_16_46_13_v108_035` | 紧急变道/猛打方向失稳 | 3.362 | 3.313 | +0.049 | +4.319 | +0.507 | missed_extreme_amplitude;large_event_high_rmse;shown_in_user_screenshot |
| 3 | #020 | `zx_Entity_Recording_2025_09_27_17_14_07_v108_016` | 连续变道/连续左右修正 | 1.591 | 1.106 | +0.485 | +0.089 | +2.981 | false_large_maneuver;regression_vs_v300;shown_in_user_screenshot |
| 4 | #023 | `gzj_Entity_Recording_2025_09_27_11_41_47_v108_048` | 平路过弯 | 1.394 | 1.199 | +0.195 | -0.544 | +1.884 | opposite_peak_direction;regression_vs_v300;shown_in_user_screenshot |
| 5 | #039 | `zdq_Entity_Recording_2025_09_26_15_14_51_v108_012` | 连续变道/连续左右修正 | 0.628 | 0.323 | +0.304 | -0.201 | -1.371 | false_large_maneuver;regression_vs_v300 |
| 6 | #021 | `zx_Entity_Recording_2025_09_27_17_25_16_v108_001` | 连续变道/连续左右修正 | 1.471 | 1.547 | -0.076 | -1.889 | +0.416 | opposite_peak_direction |
| 7 | #014 | `gzj_Entity_Recording_2025_09_27_12_28_14_v108_054` | 平路过弯 | 0.394 | 0.381 | +0.014 | +0.589 | -0.836 | opposite_peak_direction;shown_in_user_screenshot |
| 8 | #022 | `zx_Entity_Recording_2025_09_27_16_32_00_v108_040` | 平路过弯 | 1.470 | 1.425 | +0.045 | -0.033 | +2.636 | false_large_maneuver |
| 9 | #030 | `txj_Entity_Recording_2025_09_27_09_17_11_v108_019` | 连续变道/连续左右修正 | 0.718 | 1.056 | -0.337 | -0.882 | +0.589 | opposite_peak_direction |
| 10 | - | `hzh_Entity_Recording_2025_09_27_19_44_05_v108_020` | 下坡过弯 | 0.356 | 0.399 | -0.043 | +0.612 | -0.915 | opposite_peak_direction |
| 11 | - | `hzh_Entity_Recording_2025_09_27_19_33_25_v108_026` | 连续变道/连续左右修正 | 0.714 | 0.703 | +0.010 | -0.668 | +0.470 | opposite_peak_direction |
| 12 | - | `gzj_Entity_Recording_2025_09_27_12_28_14_v108_052` | 平路过弯 | 0.508 | 0.484 | +0.024 | +0.474 | -0.711 | opposite_peak_direction |
| 13 | - | `gzj_Entity_Recording_2025_09_27_11_53_25_v108_042` | 平路过弯 | 0.323 | 0.229 | +0.094 | +0.532 | -0.445 | opposite_peak_direction |
| 14 | #044 | `txj_Entity_Recording_2025_09_27_09_17_11_v108_037` | 平路过弯 | 1.109 | 1.062 | +0.047 | -3.344 | -0.985 | missed_extreme_amplitude |
| 15 | #036 | `txj_Entity_Recording_2025_09_27_08_40_46_v108_033` | 紧急变道/猛打方向失稳 | 0.985 | 1.221 | -0.236 | -3.144 | -1.164 | missed_extreme_amplitude |
| 16 | - | `byx_Entity_Recording_2025_09_28_17_25_18_v108_037` | 连续变道/连续左右修正 | 0.796 | 0.837 | -0.041 | -2.182 | -0.767 | missed_extreme_amplitude |
| 17 | - | `zx_Entity_Recording_2025_09_27_16_46_13_v108_053` | 平路过弯 | 0.770 | 0.964 | -0.194 | -2.227 | -0.968 | missed_extreme_amplitude |
| 18 | - | `lxy_Entity_Recording_2025_09_28_18_19_35_v108_027` | 其他/不确定 | 0.673 | 0.728 | -0.055 | -2.183 | -0.897 | missed_extreme_amplitude |
| 19 | - | `rjy_Entity_Recording_2025_09_28_20_02_20_v108_002` | 其他/不确定 | 0.596 | 0.648 | -0.052 | -2.087 | -0.856 | missed_extreme_amplitude |
| 20 | #019 | `zx_Entity_Recording_2025_09_27_17_45_11_v108_023` | 下坡过弯 | 2.035 | 2.019 | +0.016 | +3.121 | +1.968 | large_event_high_rmse;shown_in_user_screenshot |

## 判读说明

- `opposite_peak_direction`：真实和预测的主方向相反，是最接近你说的“完全偏离方向”的错误。
- `false_large_maneuver`：真实 0-2s 内没有明显方向盘动作，但模型预测了大幅转向，属于“凭空预测一个动作”。
- `missed_extreme_amplitude`：真实是极端猛打/失稳动作，但模型只给了很小幅值，属于漏掉危险动作。
- `large_event_high_rmse`：大动作场景的整体曲线形态/相位误差很高，需要人工看图判断是否可接受。
- `regression_vs_v300`：v307 比 v300 更差，优先作为回归错误检查。
