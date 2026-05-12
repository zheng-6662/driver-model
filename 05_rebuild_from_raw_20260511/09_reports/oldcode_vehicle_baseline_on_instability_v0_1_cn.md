# 旧车辆代码在全原始失稳高置信样本上的诊断测试 v0.1

生成时间：2026-05-12

## 这次测试做了什么

这次没有直接继续训练风格/生理模型，而是把 908 个高置信车辆失稳样本转换成旧阶段 3 车辆基线代码可读的窗口格式，然后复用旧车辆基线评价逻辑进行诊断。

## 输入

- 处理后车辆窗口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_processed_datasets/vehicle_instability_allraw_highconf_v0_1`
- 样本清单：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_processed_datasets/vehicle_instability_allraw_highconf_v0_1/tables/selected_samples_vehicle_instability_highconf_v0_1.csv`
- 旧代码逻辑：`03_baselines/scripts/evaluate_stage3_vehicle_baselines.py`

## 重要边界

1. 这不是正式阶段 3 结论，只是旧代码在新失稳样本上的第一轮诊断。
2. 锚点来自非转向车辆动力学 onset，不是道路弯道 onset。
3. `ridge_vehicle_summary` 沿用旧代码，会包含被试 one-hot；因此同时输出 `ridge_vehicle_no_subject` 作为去掉被试 one-hot 的对照。
4. 不使用生理、脑电、连续风格，不改原始 CSV。
5. 当前结果不能用于证明风格或生理有效。

## pre2 + session-level test 关键表

              model_name  n_samples  rmse_steer  peak_direction_accuracy  wrong_side_rate  large_response_recall  peak_amp_mae  peak_amp_ratio_pred_over_gt_mean  severe_amp_under_rate  peak_time_mae_s  tail_abs_error_mean  reversal_count_exact_match_rate  difficult_top20_rmse
   ridge_vehicle_summary        139    0.675055                 0.726619         0.273381                   0.08      0.682363                          1.155460               0.654676         0.446978             0.709058                         0.086331              1.298914
ridge_vehicle_no_subject        139    0.675174                 0.719424         0.280576                   0.08      0.686683                          1.105356               0.669065         0.460216             0.707152                         0.244604              1.305240
train_mean_by_event_type        139    0.677212                 0.568345         0.431655                   0.00      0.847065                          0.247128               0.913669         0.616151             0.719166                         0.086331              1.318363
            hold_current        139    0.683514                 0.517986         0.482014                   0.00      0.936887                          0.000000               1.000000         1.515432             0.725312                         0.064748              1.322311
           zero_response        139    0.683514                 0.517986         0.482014                   0.00      0.936887                          0.000000               1.000000         1.515432             0.725312                         0.064748              1.322311
          train_mean_all        139    0.685789                 0.482014         0.517986                   0.00      0.861593                          0.221422               0.942446         0.971978             0.724539                         0.258993              1.326782
     history_trend_250ms        139    1.455130                 0.244604         0.755396                   0.84      0.829240                          2.297632               0.115108         0.450216             2.045349                         0.064748              2.057304

## 各窗口/切分测试集最优行

             window_config_id      split_strategy               model_name  rmse_steer  peak_direction_accuracy  wrong_side_rate  severe_amp_under_rate  difficult_top20_rmse
    pre1_label2_event_trigger  random_event_split ridge_vehicle_no_subject    0.657165                 0.757353         0.242647               0.411765              1.047017
    pre1_label2_event_trigger session_level_split    ridge_vehicle_summary    0.655042                 0.769784         0.230216               0.589928              1.279100
    pre1_label2_event_trigger subject_level_split ridge_vehicle_no_subject    0.627600                 0.853556         0.146444               0.288703              1.120464
         pre2_label2_old_main  random_event_split    ridge_vehicle_summary    0.661532                 0.757353         0.242647               0.485294              1.108244
         pre2_label2_old_main session_level_split    ridge_vehicle_summary    0.675055                 0.726619         0.273381               0.654676              1.298914
         pre2_label2_old_main subject_level_split ridge_vehicle_no_subject    0.693907                 0.836820         0.163180               0.665272              1.271500
pre3_label3_response_coverage  random_event_split    ridge_vehicle_summary    0.756350                 0.742647         0.257353               0.580882              1.208743
pre3_label3_response_coverage session_level_split train_mean_by_event_type    0.740436                 0.546763         0.453237               0.935252              1.382109
pre3_label3_response_coverage subject_level_split ridge_vehicle_no_subject    0.804404                 0.778243         0.221757               0.794979              1.484213

## 模型拟合信息

             window_config_id      split_strategy               model_name status  selected_alpha  val_rmse_for_alpha  train_rmse_selected_alpha  feature_count removed_subject_onehot
    pre1_label2_event_trigger  random_event_split    ridge_vehicle_summary     ok          1000.0            0.748984                   0.692699             84                    NaN
    pre1_label2_event_trigger  random_event_split ridge_vehicle_no_subject     ok            10.0            0.736029                   0.637157             66                   True
    pre1_label2_event_trigger session_level_split    ridge_vehicle_summary     ok          1000.0            0.848708                   0.679077             84                    NaN
    pre1_label2_event_trigger session_level_split ridge_vehicle_no_subject     ok          1000.0            0.845615                   0.685402             66                   True
    pre1_label2_event_trigger subject_level_split    ridge_vehicle_summary     ok          1000.0            0.756153                   0.715736             84                    NaN
    pre1_label2_event_trigger subject_level_split ridge_vehicle_no_subject     ok             1.0            0.753428                   0.637693             66                   True
         pre2_label2_old_main  random_event_split    ridge_vehicle_summary     ok           100.0            0.735879                   0.641239             84                    NaN
         pre2_label2_old_main  random_event_split ridge_vehicle_no_subject     ok           100.0            0.726379                   0.656653             66                   True
         pre2_label2_old_main session_level_split    ridge_vehicle_summary     ok          1000.0            0.851333                   0.684315             84                    NaN
         pre2_label2_old_main session_level_split ridge_vehicle_no_subject     ok          1000.0            0.847662                   0.690154             66                   True
         pre2_label2_old_main subject_level_split    ridge_vehicle_summary     ok          1000.0            0.755000                   0.720481             84                    NaN
         pre2_label2_old_main subject_level_split ridge_vehicle_no_subject     ok          1000.0            0.755191                   0.723954             66                   True
pre3_label3_response_coverage  random_event_split    ridge_vehicle_summary     ok           100.0            0.813151                   0.732393             84                    NaN
pre3_label3_response_coverage  random_event_split ridge_vehicle_no_subject     ok           100.0            0.805588                   0.748353             66                   True
pre3_label3_response_coverage session_level_split    ridge_vehicle_summary     ok          1000.0            0.977179                   0.766639             84                    NaN
pre3_label3_response_coverage session_level_split ridge_vehicle_no_subject     ok          1000.0            0.973479                   0.774358             66                   True
pre3_label3_response_coverage subject_level_split    ridge_vehicle_summary     ok          1000.0            0.861457                   0.797740             84                    NaN
pre3_label3_response_coverage subject_level_split ridge_vehicle_no_subject     ok          1000.0            0.861643                   0.802380             66                   True

## 快速判断

这批失稳样本可以被旧车辆代码读取和评估。后续如果要真正比较旧深度模型，需要用本次输出的旧 manifest 做一个独立的旧模型 smoke/full run，并把结果和这里的无学习/车辆 ridge 诊断结果放在同一张表里。

## 旧深度模型 smoke 补充

已用旧 `run_event_conditioned_trajectory_baseline.py` 的 `vehicle_direct` 入口做了一个本地 CPU smoke run，只用于验证旧深度模型闭环能否读取新 manifest。

- run 目录：`F:/data_set_process/data_process/tmp/event_conditioned_runs/OLD_SMOKE_INSTABILITY_HIGHCONF_V0_1_20260512_165950`
- 子集：train=96, val=32, test=32
- 训练轮数：2
- 丢弃样本：0
- best val steer RMSE：0.976725
- smoke test steer RMSE：0.400123

该 smoke run 不是正式性能结论，不能和全量基线直接比较。它只证明旧深度模型入口已经能接上这批失稳样本。
