# 旧 `vehicle_direct` 全量车辆-only 对照：全原始失稳高置信样本 clean v0.1

生成时间：2026-05-12

## 这次跑了什么

按用户要求，使用旧流程深度模型入口 `run_event_conditioned_trajectory_baseline.py`，在全原始车辆 CSV 重筛得到的高置信车辆失稳样本上跑全量 `vehicle_direct` 车辆-only 对照。

- 训练入口：`F:/data_set_process/data_process/02_code/final_code/model/training/run_event_conditioned_trajectory_baseline.py`
- clean manifest：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_processed_datasets/vehicle_instability_allraw_highconf_v0_1/tables/oldcode_manifest_session_level_split_clean_vehicle_v0_1.csv`
- run 目录：`F:/data_set_process/data_process/tmp/event_conditioned_runs/OLD_FULL_INSTABILITY_HIGHCONF_VEHICLE_DIRECT_CLEAN_V0_1_20260512_181413`
- 输入模态：车辆历史 + 旧入口上下文字段；未使用生理、脑电、连续风格、驾驶员风格向量或教师状态。
- split：session-level split，train/val/test = 611/156/139。
- 样本定义：非方向盘车辆动力学 onset，即 `ay/roll_rate` 触发的失稳锚点；方向盘只作为事件后的响应标签。
- 服务器：未使用服务器，未读取服务器指令与密码文件。

注意：此前直接让旧深度入口读取原始车辆 CSV 的 run 已判定无效，因为旧代码会把原始 CSV 中的交替缺失点直接填 0，导致方向盘标签出现不真实的高频跳变。本报告只使用 clean manifest 结果。

## 关键结果

旧脚本按 `legacy_rmse` 选择的 active checkpoint 是 epoch 5；同一个 run 中另有 structure-aware checkpoint epoch 9，下面一起列出，便于看旧选择规则的影响。

    checkpoint_tag  checkpoint_epoch split  n_samples  sample_rmse_steer  sample_peak_direction_accuracy  sample_wrong_side_rate  sample_large_response_recall  sample_severe_amp_under_rate  sample_peak_time_mae_s  sample_tail_abs_error_mean  sample_reversal_count_exact_match_rate  selection_selection_score  selection_overall_primary_steer_rmse  selection_rmse_tail_abs_steer  selection_peak_time_abs_err_s
active_legacy_best                 5  test        139           0.637366                        0.870504                0.129496                      0.142857                      0.683453                0.553489                    0.530855                                0.086331                   1.120517                              0.553780                       0.528381                       0.507407
    structure_best                 9  test        139           0.647720                        0.856115                0.143885                      0.200000                      0.561151                0.545252                    0.496645                                0.057554                   1.106623                              0.594399                       0.475424                       0.777778

最主要的 active checkpoint 测试集结果：

- test RMSE：0.637366
- 主峰错侧率：0.129496
- 严重幅值不足率：0.683453
- structure-aware checkpoint 的 test RMSE：0.647720，但它不是本次旧脚本 `legacy_rmse` 选择的 active checkpoint。

## 固定图和坏样本图

- 固定预测图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/oldcode_vehicle_direct_full_clean_on_instability_v0_1/figures/oldcode_vehicle_direct_full_fixed_predictions_test.png`
- 坏样本图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/oldcode_vehicle_direct_full_clean_on_instability_v0_1/figures/oldcode_vehicle_direct_full_bad_samples_test.png`

固定图使用此前旧 ridge 诊断固定下来的 pre2 + session-level test 样本，避免只挑好看的样本。坏样本图按 active checkpoint 在 test 集上的逐样本 RMSE 排序取前 12 个。

## 分被试结果

    checkpoint_tag subject  sample_rmse  wrong_side  severe_amp_under  large_response_recalled  peak_time_abs_error_s  tail_abs_error  reversal_count_exact  n_large_response  n_samples
active_legacy_best     byx     0.410764    0.076923          0.615385                 0.000000               0.532692        0.297627              0.076923                 1         13
active_legacy_best     yyl     0.441062    0.083333          0.583333                 0.083333               0.467917        0.359929              0.000000                 2         12
active_legacy_best      gf     0.475106    0.166667          0.583333                 0.083333               0.629167        0.514063              0.083333                 5         12
active_legacy_best      zx     0.494667    0.181818          0.727273                 0.000000               0.554545        0.429081              0.000000                 1         11
active_legacy_best     hzh     0.571288    0.150943          0.698113                 0.056604               0.527453        0.559432              0.113208                16         53
active_legacy_best     tyy     0.599915    0.200000          0.500000                 0.000000               0.548500        0.502070              0.300000                 2         10
active_legacy_best     zxy     0.611399    0.181818          0.909091                 0.000000               0.842727        0.820255              0.090909                 2         11
active_legacy_best     gzj     0.647085    0.000000          0.764706                 0.000000               0.472647        0.648146              0.000000                 6         17
    structure_best     byx     0.412029    0.076923          0.384615                 0.000000               0.332308        0.231479              0.000000                 1         13
    structure_best     yyl     0.450905    0.250000          0.416667                 0.000000               0.427083        0.335668              0.000000                 2         12
    structure_best      zx     0.505464    0.363636          0.545455                 0.000000               0.531364        0.312503              0.000000                 1         11
    structure_best      gf     0.508134    0.083333          0.583333                 0.166667               0.647917        0.439230              0.000000                 5         12
    structure_best     hzh     0.574169    0.056604          0.603774                 0.075472               0.481792        0.542224              0.113208                16         53
    structure_best     zxy     0.621999    0.363636          0.818182                 0.000000               1.135000        0.815235              0.090909                 2         11
    structure_best     gzj     0.668440    0.058824          0.588235                 0.058824               0.468235        0.591407              0.000000                 6         17
    structure_best     tyy     0.678961    0.300000          0.400000                 0.000000               0.674500        0.552873              0.100000                 2         10

## 分响应类型结果

    checkpoint_tag eval_morphology_label  sample_rmse  wrong_side  severe_amp_under  large_response_recalled  peak_time_abs_error_s  tail_abs_error  reversal_count_exact  n_large_response  n_samples
active_legacy_best    reverse_correction     0.488970    0.148936          0.702128                 0.021277               0.598191        0.512451              0.085106                 8         47
active_legacy_best           single_lobe     0.571193    0.210526          0.473684                 0.157895               0.577632        0.706097              0.421053                 7         19
active_legacy_best      multi_correction     0.574579    0.095890          0.726027                 0.013699               0.518425        0.497093              0.000000                20         73
    structure_best    reverse_correction     0.496761    0.106383          0.553191                 0.021277               0.504468        0.452901              0.063830                 8         47
    structure_best      multi_correction     0.591744    0.123288          0.616438                 0.041096               0.548562        0.466176              0.013699                20         73
    structure_best           single_lobe     0.595050    0.315789          0.368421                 0.157895               0.633421        0.721918              0.210526                 7         19

## 如何解释

这次结果说明：旧 `vehicle_direct` 深度入口可以在 906 个可用高置信失稳事件上完整训练和评估，且在 session-level test 的整体 RMSE 上明显低于旧 ridge/no-learning 诊断。但是它仍然有较高的严重幅值不足和错侧问题，特别是坏样本图需要继续检查是否集中在大幅响应、反向修正或多段修正。

这不是“新流程强车辆基线”的最终结论。它只是旧代码在新失稳样本上的历史对照，后续仍应把同一批高置信失稳事件整理成新流程正式 manifest，再建立无泄漏、无驾驶员 ID、物理指标齐全的强车辆基线。

## 不能下的结论

- 不能据此证明连续风格有效。
- 不能据此证明生理或脑电有效。
- 不能把旧 `vehicle_direct` 的 RMSE 当作最终上限。
- 不能忽略本次锚点来自车辆动力学 onset，而不是失稳发生前预警。
