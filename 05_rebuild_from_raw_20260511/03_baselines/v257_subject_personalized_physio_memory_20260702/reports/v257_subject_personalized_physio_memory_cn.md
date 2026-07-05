# v257 同驾驶员生理状态记忆检索实验

## 本轮问题

- v254b-v256 已经说明当前生理数据不能直接提供 subject-disjoint 跨驾驶员增量。
- 本轮检验一个更合理但更窄的路线：同一驾驶员有历史事件时，生理状态是否能帮助从该驾驶员历史未来原型中检索出更接近的未来。
- 这是 subject-aware 个体化诊断，不是 subject-disjoint 正式泛化结果。

## 方法

- 候选池：同一 subject、同一 delay、训练 split 的历史事件。
- 同一 recording 内只允许 observation_s 更早的训练事件，避免在线时序泄漏。
- 特征距离：车辆输入、v254b 200Hz 生理统计、v256 raw 生理序列 PCA。
- 预测：用候选训练未来曲线的加权平均作为个体化记忆预测。
- 策略选择：只看 validation，score = bad_top10 tail + 3 * all-tail harm。

## 候选覆盖

| query_split   | strategy                               |   query_rows |   fallback_rate |   candidate_n_mean |   candidate_n_p10 |
|:--------------|:---------------------------------------|-------------:|----------------:|-------------------:|------------------:|
| test          | same_subject_physio_stats_k1           |         1398 |      0.00429185 |            47.2575 |                21 |
| test          | same_subject_raw_seq_k1                |         1398 |      0.00429185 |            47.2575 |                21 |
| test          | same_subject_vehicle_k1                |         1398 |      0.00429185 |            47.2575 |                21 |
| test          | same_subject_vehicle_k3                |         1398 |      0.00429185 |            47.2575 |                21 |
| test          | same_subject_vehicle_physio25_k3       |         1398 |      0.00429185 |            47.2575 |                21 |
| test          | same_subject_vehicle_physio50_raw25_k5 |         1398 |      0.00429185 |            47.2575 |                21 |
| test          | same_subject_vehicle_physio_raw15_k3   |         1398 |      0.00429185 |            47.2575 |                21 |
| test          | same_subject_vehicle_raw25_k3          |         1398 |      0.00429185 |            47.2575 |                21 |
| val           | same_subject_physio_stats_k1           |         1392 |      0          |            47.306  |                22 |
| val           | same_subject_raw_seq_k1                |         1392 |      0          |            47.306  |                22 |
| val           | same_subject_vehicle_k1                |         1392 |      0          |            47.306  |                22 |
| val           | same_subject_vehicle_k3                |         1392 |      0          |            47.306  |                22 |
| val           | same_subject_vehicle_physio25_k3       |         1392 |      0          |            47.306  |                22 |
| val           | same_subject_vehicle_physio50_raw25_k5 |         1392 |      0          |            47.306  |                22 |
| val           | same_subject_vehicle_physio_raw15_k3   |         1392 |      0          |            47.306  |                22 |
| val           | same_subject_vehicle_raw25_k3          |         1392 |      0          |            47.306  |                22 |

## Validation 选型

| strategy                               |   val_all_tail_rmse |   val_bad_top10_tail_rmse |   val_all_harm_vs_v250 |   selection_score | chosen_by_validation   |
|:---------------------------------------|--------------------:|--------------------------:|-----------------------:|------------------:|:-----------------------|
| same_subject_vehicle_k3                |            0.770115 |                   1.50139 |               0.446249 |           2.84014 | True                   |
| same_subject_vehicle_physio25_k3       |            0.803026 |                   1.52851 |               0.479161 |           2.96599 | False                  |
| same_subject_vehicle_k1                |            0.866544 |                   1.51974 |               0.542678 |           3.14777 | False                  |
| same_subject_vehicle_physio_raw15_k3   |            0.915806 |                   1.59023 |               0.59194  |           3.36605 | False                  |
| same_subject_vehicle_physio50_raw25_k5 |            0.951131 |                   1.69497 |               0.627265 |           3.57677 | False                  |
| same_subject_vehicle_raw25_k3          |            0.966834 |                   1.70061 |               0.642969 |           3.62951 | False                  |
| same_subject_physio_stats_k1           |            1.50291  |                   2.18346 |               1.17904  |           5.72059 | False                  |
| same_subject_raw_seq_k1                |            1.53459  |                   2.15138 |               1.21073  |           5.78357 | False                  |

- validation 选择策略：`same_subject_vehicle_k3`

## Test 结果

| bucket             | strategy                             |    n |   sample_rmse_mean |   tail_rmse_mean |   delta_tail_rmse_vs_v250 |
|:-------------------|:-------------------------------------|-----:|-------------------:|-----------------:|--------------------------:|
| all                | v250_existing                        | 1398 |           0.256234 |         0.280155 |                  0        |
| bad_top10_v250     | v250_existing                        |  140 |           0.727077 |         0.838343 |                  0        |
| strong_steer       | v250_existing                        |  756 |           0.317844 |         0.348837 |                  0        |
| observe_later_like | v250_existing                        |  174 |           0.336307 |         0.370924 |                  0        |
| all                | same_subject_vehicle_k1              | 1398 |           0.80133  |         0.905763 |                  0.625608 |
| bad_top10_v250     | same_subject_vehicle_k1              |  140 |           1.19666  |         1.36868  |                  0.530337 |
| strong_steer       | same_subject_vehicle_k1              |  756 |           0.950054 |         1.07233  |                  0.723491 |
| observe_later_like | same_subject_vehicle_k1              |  174 |           0.892043 |         1.01489  |                  0.643963 |
| all                | same_subject_vehicle_k3              | 1398 |           0.709107 |         0.802881 |                  0.522726 |
| bad_top10_v250     | same_subject_vehicle_k3              |  140 |           1.13272  |         1.30542  |                  0.467073 |
| strong_steer       | same_subject_vehicle_k3              |  756 |           0.864305 |         0.978597 |                  0.62976  |
| observe_later_like | same_subject_vehicle_k3              |  174 |           0.880075 |         1.01181  |                  0.640883 |
| all                | same_subject_vehicle_physio_raw15_k3 | 1398 |           0.785978 |         0.889934 |                  0.609779 |
| bad_top10_v250     | same_subject_vehicle_physio_raw15_k3 |  140 |           1.12255  |         1.2675   |                  0.429153 |
| strong_steer       | same_subject_vehicle_physio_raw15_k3 |  756 |           0.944602 |         1.0686   |                  0.719758 |
| observe_later_like | same_subject_vehicle_physio_raw15_k3 |  174 |           1.00014  |         1.14493  |                  0.774003 |

## 判读

- bad_top10：v250 tail=0.8383，same_subject_vehicle_k3 tail=1.3054，delta=+0.4671。
- all：same_subject_vehicle_k3 相对 v250 tail delta=+0.5227。
- 如果 chosen 在 bad_top10 上没有大幅优于 v250，说明即使转成个体化记忆范式，当前生理也没有达到 goal 要求。
- 如果 chosen 只小幅改善或明显伤害 all，则不能作为主线，只能作为个体化补充诊断。

## 关键图

- `figures\v257_subject_memory_test_tail_rmse.png`