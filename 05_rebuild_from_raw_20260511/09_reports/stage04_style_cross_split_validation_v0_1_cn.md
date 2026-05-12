# 阶段 4：连续风格跨 split 复核 v0.1

## 输入与协议

- 样本：B 轨道 `response3s_strict_core_candidate`，窗口 `pre3_label3_response_coverage`，共 270 个样本。
- split：`session_level_split`、`subject_level_split`。
- 每个 split 重新计算 train-only 风格标准化。
- 主参照：`rbf_kernel_ridge_context_no_subject`。
- 融合方式：RBF 残差 Ridge。
- 禁用：生理、脑电、EMG、RESP、服务器、服务器凭据。

## split 与特征摘要

```text
     split_strategy                                          model_name  n_features  usable_train_standardized_features                                            source  selected_alpha split  n_samples  n_subjects                                                           subjects  n_sessions
session_level_split         rbf_plus_style_last60_guard3_residual_ridge        94.0                                93.0      raw last60 style; standardized on train only         10000.0   NaN        NaN         NaN                                                                NaN         NaN
session_level_split           rbf_plus_style_all_windows_residual_ridge       376.0                               371.0 raw all style windows; standardized on train only          1000.0   NaN        NaN         NaN                                                                NaN         NaN
session_level_split                   rbf_plus_driver_id_residual_ridge        18.0                                18.0                     train-subject one-hot control           100.0   NaN        NaN         NaN                                                                NaN         NaN
session_level_split                 rbf_plus_road_module_residual_ridge         7.0                                 7.0                 train-road-module one-hot control         10000.0   NaN        NaN         NaN                                                                NaN         NaN
session_level_split rbf_plus_style_last60_with_driver_id_residual_ridge       112.0                               111.0  raw last60 style + train-subject one-hot control         10000.0   NaN        NaN         NaN                                                                NaN         NaN
session_level_split                                                 NaN         NaN                                 NaN                                               NaN             NaN train      188.0        18.0 byx,cwh,gf,gzj,hzh,jy,lx,lxy,rjy,txj,tyy,xst,yyl,yzy,zdq,zt,zx,zxy        54.0
session_level_split                                                 NaN         NaN                                 NaN                                               NaN             NaN   val       42.0         7.0                                          cwh,jy,rjy,txj,yzy,zx,zxy        10.0
session_level_split                                                 NaN         NaN                                 NaN                                               NaN             NaN  test       40.0         8.0                                      byx,gf,gzj,hzh,tyy,yyl,zx,zxy        12.0
subject_level_split         rbf_plus_style_last60_guard3_residual_ridge        94.0                                93.0      raw last60 style; standardized on train only         10000.0   NaN        NaN         NaN                                                                NaN         NaN
subject_level_split           rbf_plus_style_all_windows_residual_ridge       376.0                               370.0 raw all style windows; standardized on train only          1000.0   NaN        NaN         NaN                                                                NaN         NaN
subject_level_split                   rbf_plus_driver_id_residual_ridge        13.0                                13.0                     train-subject one-hot control         10000.0   NaN        NaN         NaN                                                                NaN         NaN
subject_level_split                 rbf_plus_road_module_residual_ridge         7.0                                 7.0                 train-road-module one-hot control         10000.0   NaN        NaN         NaN                                                                NaN         NaN
subject_level_split rbf_plus_style_last60_with_driver_id_residual_ridge       107.0                               106.0  raw last60 style + train-subject one-hot control         10000.0   NaN        NaN         NaN                                                                NaN         NaN
subject_level_split                                                 NaN         NaN                                 NaN                                               NaN             NaN train      159.0        13.0                   cwh,gf,jy,lx,lxy,rjy,txj,tyy,xst,yyl,yzy,zdq,zxy        49.0
subject_level_split                                                 NaN         NaN                                 NaN                                               NaN             NaN   val       43.0         2.0                                                              zt,zx        11.0
subject_level_split                                                 NaN         NaN                                 NaN                                               NaN             NaN  test       68.0         3.0                                                        byx,gzj,hzh        16.0
```

## test 指标

```text
     split_strategy                                          model_name  n_samples  rmse_steer  wrong_side_rate  large_response_recall  peak_amp_ratio_pred_over_gt_mean  severe_amp_under_rate  tail_abs_error_mean  reversal_count_exact_match_rate  difficult_top20_rmse
session_level_split                 rbf_kernel_ridge_context_no_subject         40    0.533667         0.225000               0.750000                          1.109729               0.125000             0.181751                              0.0              0.678907
session_level_split                   rbf_plus_driver_id_residual_ridge         40    0.533661         0.225000               0.750000                          1.108888               0.125000             0.181435                              0.0              0.679859
session_level_split           rbf_plus_style_all_windows_residual_ridge         40    0.564143         0.175000               0.750000                          1.490872               0.125000             0.185513                              0.0              0.702179
session_level_split         rbf_plus_style_last60_guard3_residual_ridge         40    0.534559         0.225000               0.750000                          1.135936               0.125000             0.181904                              0.0              0.680891
session_level_split rbf_plus_style_last60_with_driver_id_residual_ridge         40    0.534558         0.225000               0.750000                          1.135930               0.125000             0.181897                              0.0              0.680895
subject_level_split                 rbf_kernel_ridge_context_no_subject         68    0.484847         0.147059               0.666667                          1.197963               0.117647             0.190184                              0.0              0.658887
subject_level_split                   rbf_plus_driver_id_residual_ridge         68    0.484992         0.147059               0.666667                          1.198666               0.117647             0.190218                              0.0              0.659192
subject_level_split           rbf_plus_style_all_windows_residual_ridge         68    0.482109         0.117647               0.666667                          1.243714               0.117647             0.191462                              0.0              0.655899
subject_level_split         rbf_plus_style_last60_guard3_residual_ridge         68    0.483510         0.147059               0.666667                          1.219608               0.117647             0.189769                              0.0              0.659204
subject_level_split rbf_plus_style_last60_with_driver_id_residual_ridge         68    0.483511         0.147059               0.666667                          1.219610               0.117647             0.189768                              0.0              0.659211
```

## gate

```text
                                 gate_item           status                                                                                          evidence                          decision_cn
session_level_split_style_last60_beats_rbf             fail                                  style60 RMSE=0.534559; RBF RMSE=0.533667; driverID RMSE=0.533661     必须同时看物理指标、驾驶员 ID 对照和跨 split 稳定性。
  session_level_split_physical_improvement             fail wrong_side 0.225000->0.225000; large_recall 0.750000->0.750000; difficult_rmse 0.678907->0.680891             若物理错误没有改善，不能升级为连续风格有效证据。
subject_level_split_style_last60_beats_rbf pass_exploratory                                  style60 RMSE=0.483510; RBF RMSE=0.484847; driverID RMSE=0.484992     必须同时看物理指标、驾驶员 ID 对照和跨 split 稳定性。
  subject_level_split_physical_improvement             fail wrong_side 0.147059->0.147059; large_recall 0.666667->0.666667; difficult_rmse 0.658887->0.659204             若物理错误没有改善，不能升级为连续风格有效证据。
         style_effectiveness_claim_allowed          blocked                           session_pass=False; subject_pass=True; no stable two-split improvement.         不能宣称连续风格有效；当前只完成否定/降级证据的一部分。
                        physio_eeg_allowed          blocked                                                                        连续风格没有形成强于车辆-only 的稳定公平参照。 生理/EEG 继续阻塞，除非先完成车辆-only 结构化或风格路线收口。
```

## 结论

当前连续风格路线在 session-level 与 subject-level 两类切分下均没有稳定超过 RBF，也没有稳定改善关键物理指标。因此阶段 4 不能支持“连续风格有效”的结论，生理/EEG 继续阻塞。建议后续先回到车辆-only 结构化轨迹模型。
