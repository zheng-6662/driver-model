# v256 raw 200Hz 生理 CNN 融合预测基线

## 本轮问题

- v254b 的 200Hz 手工统计没有带来跨驾驶员轨迹行为增量。
- v256 改为直接输入锚点前 20s raw 生理序列，用 1D CNN 学时序状态，再与车辆 MLP 融合。

## 输入

- 生理通道：HR_bpm, EMG_RMS, EDA_Tonic, EDA_Phasic, RESP_filt200, ECG_filt200。
- 生理窗口：observation_s 前 20s，下采样到 20Hz，共 400 步。
- 每个样本用自身 observation_s-60s 到 observation_s-20s 做 baseline z-score，不使用锚点后数据。

## 对齐覆盖

| status            |    n |      rate |
|:------------------|-----:|----------:|
| missing_recording |  564 | 0.0805484 |
| ok                | 6438 | 0.919452  |

## Test 指标

| protocol         | bucket             | model_name              |    n |   sample_rmse_mean |   tail_rmse_mean |   delta_tail_rmse_vs_v256_vehicle |
|:-----------------|:-------------------|:------------------------|-----:|-------------------:|-----------------:|----------------------------------:|
| subject_disjoint | all                | v250_existing           | 1104 |           0.291083 |         0.323335 |                        -0.102825  |
| subject_disjoint | bad_top10_v250     | v250_existing           |  111 |           0.747735 |         0.878316 |                         0.0372457 |
| subject_disjoint | strong_steer       | v250_existing           |  480 |           0.379568 |         0.422244 |                        -0.132548  |
| subject_disjoint | observe_later_like | v250_existing           |  162 |           0.464497 |         0.520208 |                        -0.123621  |
| subject_disjoint | all                | v256_vehicle_only       | 1104 |           0.388052 |         0.42616  |                         0         |
| subject_disjoint | bad_top10_v250     | v256_vehicle_only       |  111 |           0.746406 |         0.84107  |                         0         |
| subject_disjoint | strong_steer       | v256_vehicle_only       |  480 |           0.506378 |         0.554792 |                         0         |
| subject_disjoint | observe_later_like | v256_vehicle_only       |  162 |           0.586842 |         0.643829 |                         0         |
| subject_disjoint | all                | v256_physio_cnn         | 1104 |           0.879556 |         1.02152  |                         0.595365  |
| subject_disjoint | bad_top10_v250     | v256_physio_cnn         |  111 |           1.43129  |         1.71189  |                         0.870824  |
| subject_disjoint | strong_steer       | v256_physio_cnn         |  480 |           1.2825   |         1.49624  |                         0.941449  |
| subject_disjoint | observe_later_like | v256_physio_cnn         |  162 |           1.451    |         1.69584  |                         1.05201   |
| subject_disjoint | all                | v256_vehicle_physio_cnn | 1104 |           0.418048 |         0.467117 |                         0.0409568 |
| subject_disjoint | bad_top10_v250     | v256_vehicle_physio_cnn |  111 |           0.800545 |         0.913802 |                         0.0727312 |
| subject_disjoint | strong_steer       | v256_vehicle_physio_cnn |  480 |           0.540406 |         0.599955 |                         0.0451632 |
| subject_disjoint | observe_later_like | v256_vehicle_physio_cnn |  162 |           0.598736 |         0.658021 |                         0.014192  |
| subject_aware    | all                | v250_existing           | 1398 |           0.256234 |         0.280155 |                        -0.241086  |
| subject_aware    | bad_top10_v250     | v250_existing           |  140 |           0.727077 |         0.838343 |                        -0.0888845 |
| subject_aware    | strong_steer       | v250_existing           |  756 |           0.317844 |         0.348837 |                        -0.296678  |
| subject_aware    | observe_later_like | v250_existing           |  174 |           0.336307 |         0.370924 |                        -0.275215  |
| subject_aware    | all                | v256_vehicle_only       | 1398 |           0.465219 |         0.521241 |                         0         |
| subject_aware    | bad_top10_v250     | v256_vehicle_only       |  140 |           0.814252 |         0.927228 |                         0         |
| subject_aware    | strong_steer       | v256_vehicle_only       |  756 |           0.57684  |         0.645516 |                         0         |
| subject_aware    | observe_later_like | v256_vehicle_only       |  174 |           0.570458 |         0.646139 |                         0         |
| subject_aware    | all                | v256_physio_cnn         | 1398 |           0.964258 |         1.1154   |                         0.594164  |
| subject_aware    | bad_top10_v250     | v256_physio_cnn         |  140 |           1.50934  |         1.74275  |                         0.815523  |
| subject_aware    | strong_steer       | v256_physio_cnn         |  756 |           1.31943  |         1.5282   |                         0.882683  |
| subject_aware    | observe_later_like | v256_physio_cnn         |  174 |           1.1618   |         1.35117  |                         0.705029  |
| subject_aware    | all                | v256_vehicle_physio_cnn | 1398 |           0.496032 |         0.55742  |                         0.0361796 |
| subject_aware    | bad_top10_v250     | v256_vehicle_physio_cnn |  140 |           0.794907 |         0.911412 |                        -0.0158155 |
| subject_aware    | strong_steer       | v256_vehicle_physio_cnn |  756 |           0.613055 |         0.686505 |                         0.0409898 |
| subject_aware    | observe_later_like | v256_vehicle_physio_cnn |  174 |           0.599234 |         0.683946 |                         0.0378076 |

## 判读

- subject_disjoint bad_top10：vehicle tail=0.8411，vehicle+physio tail=0.9138，delta=+0.0727。
- subject_aware bad_top10：vehicle tail=0.9272，vehicle+physio tail=0.9114，delta=-0.0158。
- 如果 fusion 仍不优于同架构 vehicle-only，说明问题不只是 v254b 手工统计太浅；当前生理在这个任务构造下没有稳定可用增量。
- 如果只在 subject-aware 改善，后续应转向个体化校准范式，而不是宣称跨驾驶员通用生理行为预测。

## 训练日志摘要

| protocol         | model_name              |   epoch |   val_rmse |
|:-----------------|:------------------------|--------:|-----------:|
| subject_aware    | v256_physio_cnn         |      11 |   1.39566  |
| subject_disjoint | v256_physio_cnn         |      13 |   1.34062  |
| subject_disjoint | v256_vehicle_physio_cnn |      20 |   0.726728 |
| subject_disjoint | v256_vehicle_only       |      20 |   0.701792 |
| subject_aware    | v256_vehicle_physio_cnn |      22 |   0.675406 |
| subject_aware    | v256_vehicle_only       |      34 |   0.630116 |

## 关键图

- `figures\v256_test_bucket_tail_rmse.png`