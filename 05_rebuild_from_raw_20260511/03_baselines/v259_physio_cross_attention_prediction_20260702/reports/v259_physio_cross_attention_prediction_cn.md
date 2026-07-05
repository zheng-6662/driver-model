# v259 生理-车辆 cross-attention 直接预测实验

## 本轮问题

- 前序 v254b-v258 说明：手工生理统计、raw-CNN 简单融合、候选重排序、同驾驶员记忆和 anchor selector 都没有形成稳定生理增量。
- v259 检查一个更强但仍干净的路线：raw 生理序列不再简单拼接，而是作为时序 token 被每个未来预测点 cross-attention 查询。
- 本轮不做删样本、不做 hard gate/router、不做 residual 修正；输出仍是单条可部署轨迹。

## 输入与模型

- 车辆输入：v250 minimal_lateral7 的 history/road/phase，history 形状为 31x7。
- 生理输入：v256 raw cache，通道为 HR_bpm, EMG_RMS, EDA_Tonic, EDA_Phasic, RESP_filt200, ECG_filt200，20s x 20Hz = 400 步。
- 生理覆盖率：0.9195。缺失生理时序保留为 0，并显式输入 physio_ok。
- 模型：vehicle_attn 是车辆时序 attention baseline；vehicle_physio_crossattn 在同一查询上额外 attend 生理 token。
- 损失：tail 点权重 2.0；badweighted 版本仅在训练 split 对 v250 bad_top10 样本乘 4.0 权重。

## Validation 选型

| protocol         | model_name                                |   val_bad_top10_tail_rmse |   val_all_tail_rmse |   val_all_harm_vs_v250 |   selection_score |   validation_rank | chosen_by_validation   |
|:-----------------|:------------------------------------------|--------------------------:|--------------------:|-----------------------:|------------------:|------------------:|:-----------------------|
| subject_aware    | v259_vehicle_attn                         |                   1.33876 |            0.537446 |               0.21358  |           1.76592 |                 1 | True                   |
| subject_aware    | v259_vehicle_physio_crossattn_badweighted |                   1.45862 |            0.664992 |               0.341126 |           2.14087 |                 2 | False                  |
| subject_aware    | v259_vehicle_physio_crossattn             |                   1.49024 |            0.671105 |               0.347239 |           2.18472 |                 3 | False                  |
| subject_disjoint | v259_vehicle_attn                         |                   1.71447 |            0.670655 |               0.174485 |           2.06344 |                 1 | True                   |
| subject_disjoint | v259_vehicle_physio_crossattn             |                   1.78245 |            0.794312 |               0.298143 |           2.37874 |                 2 | False                  |
| subject_disjoint | v259_vehicle_physio_crossattn_badweighted |                   1.80848 |            0.791602 |               0.295432 |           2.39934 |                 3 | False                  |

## Test 关键结果

| protocol         | bucket             | model_name                                |    n |   sample_rmse_mean |   tail_rmse_mean |   delta_tail_rmse_vs_v250 |   delta_tail_rmse_vs_v259_vehicle |
|:-----------------|:-------------------|:------------------------------------------|-----:|-------------------:|-----------------:|--------------------------:|----------------------------------:|
| subject_disjoint | all                | v250_existing                             | 1104 |           0.291083 |         0.323335 |                  0        |                        -0.133348  |
| subject_disjoint | bad_top10_v250     | v250_existing                             |  111 |           0.747735 |         0.878316 |                  0        |                        -0.048381  |
| subject_disjoint | strong_steer       | v250_existing                             |  480 |           0.379568 |         0.422244 |                  0        |                        -0.160607  |
| subject_disjoint | observe_later_like | v250_existing                             |  162 |           0.464497 |         0.520208 |                  0        |                        -0.18391   |
| subject_disjoint | all                | v259_vehicle_attn                         | 1104 |           0.417154 |         0.456683 |                  0.133348 |                         0         |
| subject_disjoint | bad_top10_v250     | v259_vehicle_attn                         |  111 |           0.816996 |         0.926697 |                  0.048381 |                         0         |
| subject_disjoint | strong_steer       | v259_vehicle_attn                         |  480 |           0.534502 |         0.58285  |                  0.160607 |                         0         |
| subject_disjoint | observe_later_like | v259_vehicle_attn                         |  162 |           0.645329 |         0.704118 |                  0.18391  |                         0         |
| subject_disjoint | all                | v259_vehicle_physio_crossattn             | 1104 |           0.507598 |         0.562247 |                  0.238911 |                         0.105564  |
| subject_disjoint | bad_top10_v250     | v259_vehicle_physio_crossattn             |  111 |           0.949477 |         1.0889   |                  0.210587 |                         0.162206  |
| subject_disjoint | strong_steer       | v259_vehicle_physio_crossattn             |  480 |           0.670022 |         0.744398 |                  0.322154 |                         0.161548  |
| subject_disjoint | observe_later_like | v259_vehicle_physio_crossattn             |  162 |           0.83717  |         0.940831 |                  0.420623 |                         0.236712  |
| subject_disjoint | all                | v259_vehicle_physio_crossattn_badweighted | 1104 |           0.491404 |         0.544107 |                  0.220772 |                         0.0874237 |
| subject_disjoint | bad_top10_v250     | v259_vehicle_physio_crossattn_badweighted |  111 |           0.904828 |         1.03515  |                  0.15683  |                         0.108449  |
| subject_disjoint | strong_steer       | v259_vehicle_physio_crossattn_badweighted |  480 |           0.627568 |         0.693509 |                  0.271265 |                         0.110659  |
| subject_disjoint | observe_later_like | v259_vehicle_physio_crossattn_badweighted |  162 |           0.754475 |         0.844395 |                  0.324187 |                         0.140277  |
| subject_aware    | all                | v250_existing                             | 1398 |           0.256234 |         0.280155 |                  0        |                        -0.269736  |
| subject_aware    | bad_top10_v250     | v250_existing                             |  140 |           0.727077 |         0.838343 |                  0        |                        -0.165475  |
| subject_aware    | strong_steer       | v250_existing                             |  756 |           0.317844 |         0.348837 |                  0        |                        -0.352949  |
| subject_aware    | observe_later_like | v250_existing                             |  174 |           0.336307 |         0.370924 |                  0        |                        -0.355788  |
| subject_aware    | all                | v259_vehicle_attn                         | 1398 |           0.491675 |         0.54989  |                  0.269736 |                         0         |
| subject_aware    | bad_top10_v250     | v259_vehicle_attn                         |  140 |           0.877693 |         1.00382  |                  0.165475 |                         0         |
| subject_aware    | strong_steer       | v259_vehicle_attn                         |  756 |           0.626819 |         0.701787 |                  0.352949 |                         0         |
| subject_aware    | observe_later_like | v259_vehicle_attn                         |  174 |           0.636182 |         0.726712 |                  0.355788 |                         0         |
| subject_aware    | all                | v259_vehicle_physio_crossattn             | 1398 |           0.602135 |         0.66772  |                  0.387565 |                         0.117829  |
| subject_aware    | bad_top10_v250     | v259_vehicle_physio_crossattn             |  140 |           1.00015  |         1.12878  |                  0.290437 |                         0.124962  |
| subject_aware    | strong_steer       | v259_vehicle_physio_crossattn             |  756 |           0.768878 |         0.856088 |                  0.507251 |                         0.154301  |
| subject_aware    | observe_later_like | v259_vehicle_physio_crossattn             |  174 |           0.743319 |         0.840797 |                  0.469873 |                         0.114084  |
| subject_aware    | all                | v259_vehicle_physio_crossattn_badweighted | 1398 |           0.605392 |         0.674484 |                  0.394329 |                         0.124593  |
| subject_aware    | bad_top10_v250     | v259_vehicle_physio_crossattn_badweighted |  140 |           0.994161 |         1.1312   |                  0.292856 |                         0.127381  |
| subject_aware    | strong_steer       | v259_vehicle_physio_crossattn_badweighted |  756 |           0.771967 |         0.863169 |                  0.514332 |                         0.161382  |
| subject_aware    | observe_later_like | v259_vehicle_physio_crossattn_badweighted |  174 |           0.778161 |         0.881528 |                  0.510604 |                         0.154816  |

## v256 raw-CNN 参照

| protocol         | bucket         | model_name              |    n |   sample_rmse_mean |   tail_rmse_mean |   delta_tail_rmse_vs_v256_vehicle |
|:-----------------|:---------------|:------------------------|-----:|-------------------:|-----------------:|----------------------------------:|
| subject_disjoint | all            | v256_vehicle_only       | 1104 |           0.388052 |         0.42616  |                         0         |
| subject_disjoint | bad_top10_v250 | v256_vehicle_only       |  111 |           0.746406 |         0.84107  |                         0         |
| subject_disjoint | all            | v256_vehicle_physio_cnn | 1104 |           0.418048 |         0.467117 |                         0.0409568 |
| subject_disjoint | bad_top10_v250 | v256_vehicle_physio_cnn |  111 |           0.800545 |         0.913802 |                         0.0727312 |
| subject_aware    | all            | v256_vehicle_only       | 1398 |           0.465219 |         0.521241 |                         0         |
| subject_aware    | bad_top10_v250 | v256_vehicle_only       |  140 |           0.814252 |         0.927228 |                         0         |
| subject_aware    | all            | v256_vehicle_physio_cnn | 1398 |           0.496032 |         0.55742  |                         0.0361796 |
| subject_aware    | bad_top10_v250 | v256_vehicle_physio_cnn |  140 |           0.794907 |         0.911412 |                        -0.0158155 |

## 判读

- subject_disjoint bad_top10：v250 tail=0.8783；v259 vehicle=0.9267；physio=1.0889；physio_badweighted=1.0351。
- subject_aware bad_top10：v250 tail=0.8383；v259 vehicle=1.0038；physio=1.1288；physio_badweighted=1.1312。
- 如果 vehicle+physio 明显低于 vehicle-only，说明 raw 生理 cross-attention 真的补充了车辆锚点前信息。
- 如果 vehicle+physio 没有低于 vehicle-only 或 v250，说明瓶颈不是融合结构太弱，而是当前生理片段对未来方向盘曲线的可判别信息仍不足。

## 训练日志摘要

| protocol         | model_name                                |   epoch |   val_rmse_weighted |       lr |
|:-----------------|:------------------------------------------|--------:|--------------------:|---------:|
| subject_aware    | v259_vehicle_physio_crossattn             |      28 |            0.8276   | 0.000175 |
| subject_disjoint | v259_vehicle_physio_crossattn             |      31 |            0.953434 | 0.000175 |
| subject_aware    | v259_vehicle_physio_crossattn_badweighted |      33 |            0.834272 | 8.75e-05 |
| subject_disjoint | v259_vehicle_physio_crossattn_badweighted |      40 |            0.946336 | 0.000175 |
| subject_disjoint | v259_vehicle_attn                         |      59 |            0.836548 | 8.75e-05 |
| subject_aware    | v259_vehicle_attn                         |      70 |            0.706621 | 0.000175 |

## 关键图

- `figures\v259_test_bucket_tail_rmse.png`