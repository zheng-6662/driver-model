# conditioned v2 快反应退化与 boundary 恶化专题归因报告（20260408）

## 数据与口径
- 主表：`F:\data_set_process\data_process\reports\attribution_master_table.csv`，共 749 条 sample-level 记录。
- 事件表：`F:\data_set_process\data_process\reports\attribution_event_table.csv`，共 4494 条 event-level 记录。
- `Q1_fast` 直接使用字段 `latency_proxy_bucket`；本次样本数为 188，非 `Q1_fast` 为 561。
- event-level 中将 `unconditional_baseline` 记为 baseline，将 `event_conditioned_baseline` 记为 conditioned。

## 结论摘要
1. `Q1_fast` 上的 `delta_rmse_tail_abs_steer` 均值为 0.0155，明显差于非 `Q1_fast` 的 -0.0345，确认快反应桶确实存在 tail 退化。
2. 但 `Q1_fast` 的 `delta_boundary_shift_abs_err` 均值只有 0.0316，低于非 `Q1_fast` 的 0.0842；`delta_peak_time_abs_err_s` 也没有转成明显更差（-0.0197 vs -0.0262）。因此，`Q1_fast` 的主要问题并不是更强的 boundary 或 peak timing 恶化。
3. 在 `Q1_fast` 且 `delta_rmse_tail_abs_steer > 0` 的样本里，若只看本任务要求的结构性 delta 指标，相关性最高的是 `delta_peak_time_abs_err_s`（Pearson r=0.267）；`delta_boundary_shift_abs_err` 的相关性接近 0，说明快反应退化并非 boundary 驱动。
4. 若扩展到 conditioned 结构指标，最强信号来自 `shape_corr_conditioned`（|r|=0.621）和 `peak_abs_amp_err_conditioned`（|r|=0.599），对应的是 shape / amplitude 失配，而不是 boundary 漂移。
5. `boundary_shift` 恶化主要集中在 morphology，而不是单一 subject：`single_lobe × cwh` 的均值最高（0.2202），其次是 `single_lobe × gf`（0.2034）；同时三位被试在 `single_lobe` 与 `reverse_correction` 上都为正。
6. event-level 上，`Q1_fast` 没有出现 conditioned 相对 baseline 的额外时间对齐惩罚：`first_major_turn_onset` 的 conditioned-baseline 均值差为 -0.0078，`main_peak` 为 -0.0137；其中 `main_peak` 在 `Q1_fast` 仍是轻微改善而非恶化。因此，快反应退化更像尾段 shape / amplitude 的问题，而不是 conditioned 带来的事件时间对齐系统性变差。

## 1. Q1_fast vs 非 Q1_fast 关键指标对比
| group       | metric                       | count | mean      | median    | std      | q75      | q90      | worsen_rate |
| ----------- | ---------------------------- | ----- | --------- | --------- | -------- | -------- | -------- | ----------- |
| Q1_fast     | delta_rmse_tail_abs_steer    | 188   | 0.015495  | -0.006617 | 0.242334 | 0.079874 | 0.231184 | 0.473404    |
| Q1_fast     | delta_boundary_shift_abs_err | 188   | 0.031559  | 0.014135  | 0.21982  | 0.11909  | 0.289126 | 0.579787    |
| Q1_fast     | delta_peak_time_abs_err_s    | 188   | -0.019681 | 0.0       | 0.421845 | 0.1      | 0.35     | 0.351064    |
| Q1_fast     | delta_turning_count_abs_err  | 188   | -0.409574 | 0.0       | 1.846318 | 1.0      | 2.0      | 0.265957    |
| non_Q1_fast | delta_rmse_tail_abs_steer    | 561   | -0.034544 | -0.022298 | 0.243727 | 0.069747 | 0.195222 | 0.427807    |
| non_Q1_fast | delta_boundary_shift_abs_err | 561   | 0.08416   | 0.064679  | 0.362631 | 0.27842  | 0.500179 | 0.614973    |
| non_Q1_fast | delta_peak_time_abs_err_s    | 561   | -0.026203 | 0.0       | 0.557401 | 0.15     | 0.45     | 0.418895    |
| non_Q1_fast | delta_turning_count_abs_err  | 561   | -0.178253 | 0.0       | 1.648687 | 1.0      | 2.0      | 0.26738     |

解读：`Q1_fast` 的 tail 指标是唯一明显转正的恶化项；而 `delta_boundary_shift_abs_err` 在非 `Q1_fast` 更大，`delta_turning_count_abs_err` 在 `Q1_fast` 反而更负，说明 turning count 不是主要问题。

## 2. Q1_fast 中 tail 恶化样本的结构指标排序
### 2.1 只看结构性 delta 指标
| metric                       | n  | pearson_corr | abs_pearson_corr | mean_metric | median_metric |
| ---------------------------- | -- | ------------ | ---------------- | ----------- | ------------- |
| delta_peak_time_abs_err_s    | 89 | 0.267209     | 0.267209         | 0.001124    | 0.0           |
| delta_turning_count_abs_err  | 89 | 0.240463     | 0.240463         | -0.404494   | 0.0           |
| delta_boundary_shift_abs_err | 89 | 0.027096     | 0.027096         | 0.003429    | 0.013178      |
| delta_tail_trend_corr        | 89 | 0.025734     | 0.025734         | 0.00807     | -0.010886     |

### 2.2 扩展到 conditioned 结构指标
| metric                            | n  | pearson_corr | abs_pearson_corr | mean_metric | median_metric |
| --------------------------------- | -- | ------------ | ---------------- | ----------- | ------------- |
| shape_corr_conditioned            | 89 | -0.620839    | 0.620839         | 0.509891    | 0.73373       |
| peak_abs_amp_err_conditioned      | 89 | 0.598904     | 0.598904         | 0.389932    | 0.254502      |
| turning_count_abs_err_conditioned | 89 | 0.425526     | 0.425526         | 1.539326    | 1.0           |
| peak_time_abs_err_s_conditioned   | 89 | 0.263731     | 0.263731         | 0.530899    | 0.2           |
| trend_corr_conditioned            | 89 | -0.240731    | 0.240731         | 0.293346    | 0.350957      |
| extrema_count_abs_err_conditioned | 89 | 0.228166     | 0.228166         | 2.359551    | 2.0           |

### 2.3 Q1_fast 内部：tail 恶化 vs 未恶化 的条件均值差
| metric                             | worsened_mean | improved_mean | mean_diff | effect_size_smd |
| ---------------------------------- | ------------- | ------------- | --------- | --------------- |
| peak_abs_amp_err_conditioned       | 0.389932      | 0.288685      | 0.101247  | 0.25766         |
| delta_boundary_shift_abs_err       | 0.003429      | 0.056847      | -0.053418 | -0.24417        |
| boundary_shift_abs_err_conditioned | 0.526052      | 0.709732      | -0.18368  | -0.237663       |
| extrema_count_abs_err_conditioned  | 2.359551      | 1.969697      | 0.389854  | 0.209939        |
| delta_tail_trend_corr              | 0.00807       | -0.073704     | 0.081774  | 0.189474        |
| shape_corr_conditioned             | 0.509891      | 0.58908       | -0.079189 | -0.161412       |
| range_abs_err_conditioned          | 0.599499      | 0.508468      | 0.091031  | 0.147222        |
| peak_time_abs_err_s_conditioned    | 0.530899      | 0.60101       | -0.070111 | -0.120438       |

解读：若限定在任务要求的 delta 指标里，`delta_peak_time_abs_err_s` 与 `delta_turning_count_abs_err` 的相关性高于 `delta_boundary_shift_abs_err`，但绝对值都不大；真正更强的伴随信号是 `shape_corr_conditioned` 下降和 `peak_abs_amp_err_conditioned` 升高。

## 3. `eval_morphology_label × subj` 的 boundary_shift 恶化交叉表
### 3.1 conditioned minus baseline 的均值
| eval_morphology_label | cwh       | gf       | tyy      |
| --------------------- | --------- | -------- | -------- |
| multi_correction      | -0.014979 | 0.031174 | 0.010546 |
| reverse_correction    | 0.150382  | 0.089949 | 0.074211 |
| single_lobe           | 0.220195  | 0.203372 | 0.132329 |

### 3.2 恶化率 `P(delta_boundary_shift_abs_err > 0)`
| eval_morphology_label | cwh      | gf       | tyy      |
| --------------------- | -------- | -------- | -------- |
| multi_correction      | 0.5      | 0.566474 | 0.539773 |
| reverse_correction    | 0.717391 | 0.629032 | 0.623656 |
| single_lobe           | 0.783784 | 0.65625  | 0.642857 |

解读：恶化不是只锁定在单一 subject。更强的共性是 morphology：`single_lobe` 在三位被试上都最差，`reverse_correction` 次之，`multi_correction` 最轻。被试差异体现在幅度上，`cwh` 在 `reverse_correction` / `single_lobe` 上最重，`gf` 在 `single_lobe` 上也很突出。

## 4. `gf / cwh / tyy` 的 conditioned vs baseline `boundary_shift` 分布对比
| subj | baseline_mean | conditioned_mean | delta_mean | baseline_median | conditioned_median | delta_median | baseline_q90 | conditioned_q90 | delta_q90 | delta_worsen_rate |
| ---- | ------------- | ---------------- | ---------- | --------------- | ------------------ | ------------ | ------------ | --------------- | --------- | ----------------- |
| cwh  | 0.511609      | 0.636482         | 0.124873   | 0.333218        | 0.464213           | 0.144178     | 1.273696     | 1.487123        | 0.481692  | 0.678363          |
| gf   | 0.596756      | 0.662216         | 0.06546    | 0.363697        | 0.465864           | 0.041908     | 1.355306     | 1.433535        | 0.447594  | 0.59176           |
| tyy  | 0.766727      | 0.812758         | 0.046031   | 0.420271        | 0.518927           | 0.026049     | 1.830009     | 1.87706         | 0.462731  | 0.578778          |

解读：三位被试的 `boundary_shift_abs_err_conditioned` 分布都整体右移。按 `delta_boundary_shift_abs_err` 均值看，`cwh` 最差，其次 `gf`，再到 `tyy`；这与背景里观察到的 subject heterogeneity 一致，但仍不是单一被试独占，因为三人都呈正向恶化。

## 5. event-level 时间对齐：`Q1_fast` vs 非 `Q1_fast`
### 5.1 `time_abs_err_s` 分布
| event_name             | model_name  | bucket_group | count | mean     | median | q75     | q90    |
| ---------------------- | ----------- | ------------ | ----- | -------- | ------ | ------- | ------ |
| first_major_turn_onset | baseline    | Q1_fast      | 188   | 0.14367  | 0.085  | 0.135   | 0.3175 |
| first_major_turn_onset | baseline    | non_Q1_fast  | 561   | 0.182451 | 0.1    | 0.165   | 0.5    |
| first_major_turn_onset | conditioned | Q1_fast      | 188   | 0.135904 | 0.085  | 0.12625 | 0.264  |
| first_major_turn_onset | conditioned | non_Q1_fast  | 561   | 0.159456 | 0.09   | 0.15    | 0.375  |
| main_peak              | baseline    | Q1_fast      | 188   | 0.554521 | 0.4175 | 0.98875 | 1.2715 |
| main_peak              | baseline    | non_Q1_fast  | 561   | 0.511417 | 0.355  | 0.81    | 1.25   |
| main_peak              | conditioned | Q1_fast      | 188   | 0.540798 | 0.3975 | 0.79375 | 1.3495 |
| main_peak              | conditioned | non_Q1_fast  | 561   | 0.515419 | 0.355  | 0.81    | 1.265  |

### 5.2 同一样本上 `conditioned - baseline` 的时间误差差值
| event_name             | bucket_group | count | mean      | median | q75     | q90    | worsen_rate |
| ---------------------- | ------------ | ----- | --------- | ------ | ------- | ------ | ----------- |
| first_major_turn_onset | Q1_fast      | 188   | -0.007766 | 0.0    | 0.0     | 0.035  | 0.191489    |
| first_major_turn_onset | non_Q1_fast  | 561   | -0.022995 | 0.0    | 0.0     | 0.05   | 0.181818    |
| main_peak              | Q1_fast      | 188   | -0.013723 | -0.01  | 0.12625 | 0.4865 | 0.473404    |
| main_peak              | non_Q1_fast  | 561   | 0.004002  | 0.015  | 0.16    | 0.445  | 0.547237    |

解读：`first_major_turn_onset` 上 conditioned 对两类桶都是改善，且非 `Q1_fast` 的改善更强；`main_peak` 上 `Q1_fast` 仍是轻微改善，非 `Q1_fast` 则接近持平略差。因此 event-level 的时间对齐结果不支持“Q1_fast 因时间对齐更差而退化”的解释。

## 6. 归因收口与下一步建议
- 归因事实 1：`Q1_fast` 的 tail 退化存在，但并不伴随更重的 `boundary_shift` 恶化，也不伴随 event-level 时间对齐系统性变差。
- 归因事实 2：在 `Q1_fast` 内部，更强的伴随信号是 `shape_corr_conditioned` 下降和 `peak_abs_amp_err_conditioned` 上升，说明 tail 的 shape / amplitude 失真比 boundary 漂移更像主因。
- 归因事实 3：`boundary_shift` 恶化更像 morphology 主导的共性问题，重点是 `single_lobe` 与 `reverse_correction`；subject 主要影响恶化幅度，而不是决定是否发生。
- 推荐下一步：继续做只读切片，优先检查 `Q1_fast` 中高 `peak_abs_amp_err_conditioned` / 低 `shape_corr_conditioned` 的具体样本，确认它们是否集中在某些尾段幅值模式或反向修正强度区间。
- 推荐下一步：对 `single_lobe` 与 `reverse_correction` 分别追加 trajectory 可视化抽样，验证 `boundary_shift` 恶化是由边界提前/滞后，还是由边界附近幅值不足引起。
