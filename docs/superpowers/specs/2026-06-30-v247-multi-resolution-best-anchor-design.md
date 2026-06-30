# v247 Multi-Resolution Best Anchor Discovery Design

## 背景

当前 v245/v246 已经证明：差样本里确实存在一批“当前锚点太早、可见信息不足”的事件。把锚点后移以后，同一段后续轨迹的预测误差明显下降。但这不等价于“后移锚点就是目标”。真正目标应该是：对每个事件找到最适合开始预测的锚点，并训练一个在线选择器在不看未来真值的情况下识别这个锚点。

v246 的候选锚点只覆盖 `0/200/400/600/800/1000ms`，因此它只能回答“六个粗候选里哪个最好”。如果真实最佳锚点在 `50ms`、`150ms` 或 `350ms`，粗网格会产生标签偏差。v247 需要把任务升级为 multi-resolution best anchor discovery。

## 目标

v247 的目标是构造并验证一个新的任务定义：

1. 离线发现每个事件的 best anchor label。
2. 候选锚点不局限于六个粗延迟，而是使用细网格和粗到细结构。
3. best anchor 不只由预测误差决定，还要加入等待代价和局部不稳定惩罚，避免退化成永远选择最晚锚点。
4. 训练一个 input-only selector 去学习离线 best anchor label。
5. 评价 selector 是否能在不使用未来真值、测试后验误差或人工响应标签的情况下接近离线 best anchor。

## 非目标

- 不把 oracle best anchor 当成可部署策略。
- 不使用 test 后验误差决定锚点。
- 不把所有事件统一后移到最晚锚点。
- 不删除差样本。
- 不回到 v222a gate、删除样本或轻量 residual 路线。
- v247 第一版不直接训练新的大轨迹预测模型；重点是任务构造、标签质量和 selector 可学习性。
- v247 第一版不直接做连续锚点回归，例如直接预测 `47.3ms`。连续回归需要等 50ms 细网格证明有效后再考虑。

## 核心定义

每个事件 `event_uid` 下构造多个候选锚点 `a`。对每个候选锚点计算：

```text
score(a) =
    prediction_error(a)
  + lambda_wait * waiting_cost(a)
  + mu_unstable * instability_penalty(a)
```

最终离线 best anchor 是：

```text
best_anchor(event) = argmin_a score(a)
```

其中：

- `prediction_error(a)`：从候选锚点 `a` 开始预测后续目标窗口的真实误差。
- `waiting_cost(a)`：等待越久惩罚越高，避免全部选择晚锚点。
- `instability_penalty(a)`：候选锚点附近驾驶/车辆信号越不稳定，惩罚越高。

## 候选锚点设计

v247 使用三层锚点体系。

### Layer 1: Coarse Grid

粗网格沿用当前数据已有锚点：

```text
0, 200, 400, 600, 800, 1000ms
```

用途：

- 与 v245/v246 可直接对齐。
- 快速判断最佳锚点的大致阶段。
- 作为 coarse anchor 分类标签。

### Layer 2: Fine Grid

细网格作为 v247 主候选集合：

```text
0, 50, 100, 150, ..., 950, 1000ms
```

用途：

- 支持真实最佳锚点出现在 `50ms`、`150ms`、`350ms` 等粗网格中间位置。
- 输出 `fine_best_delay_ms`。
- 进一步映射到 `nearest_coarse_delay_ms` 和 `residual_offset_ms`。

如果数据采样或原始数组无法直接构造 50ms 锚点，v247 必须先审查原始序列的可用频率和插值风险，再决定使用 50ms、100ms 或可用采样点上的最近锚点。不能在没有审查采样支持的情况下假装存在 50ms 锚点。

### Layer 3: Signal-Derived Anchors

信号锚点只做诊断，不作为 v247 第一版主标签：

```text
steer_onset_time
steer_peak_slope_time
steer_stabilization_time
yaw_response_onset_time
ay_response_onset_time
```

用途：

- 检查 fine best anchor 是否靠近真实行为阶段。
- 帮助解释“为什么这个时间点适合预测”。
- 为后续 v248 或扩展模型提供候选锚点生成依据。

## Scoring 方案

v247 至少输出三套 label，用于判断不同定义是否稳定。

### Label A: Error Only

```text
score_error_only(a) = prediction_error(a)
```

用途：

- 理论误差上限。
- 检查纯误差是否过度偏向晚锚点。

### Label B: Error Plus Delay Cost

```text
score_delay(a) =
    prediction_error(a)
  + lambda_wait * delay_ms / 1000
```

建议第一版使用：

```text
lambda_wait in {0.03, 0.05, 0.10}
```

用途：

- 约束“越晚越好”的退化。
- 检查最佳锚点能否从 1000ms 拉回 300-800ms。

### Label C: Error Plus Delay Plus Instability

```text
score_delay_instability(a) =
    prediction_error(a)
  + lambda_wait * delay_ms / 1000
  + mu_unstable * instability_penalty(a)
```

第一版建议：

```text
lambda_wait in {0.03, 0.05, 0.10}
mu_unstable in {0.03, 0.05}
```

`instability_penalty` 先用可解释特征构造：

```text
0.40 * normalized_abs_steer_slope_last_0p5s
+ 0.25 * normalized_abs_steer_second_diff_last_0p5s
+ 0.20 * normalized_abs_yaw_change_last_0p5s
+ 0.15 * normalized_abs_lat_offset_change_last_0p5s
```

如果某些原始字段不可用，脚本必须在报告中明确降级到实际可用字段，不能静默替换。

## Selector 设计

v247 selector 采用 input-only 约束。它不能使用：

- 未来真实轨迹。
- candidate 的真实预测误差作为输入。
- 人工响应类型标签。
- `event_uid` 或 `recording` 作为模型输入。
- test 后验误差。

### Selector 任务形式

第一版使用 candidate scoring：

```text
input: visible_features(event, candidate_anchor)
target: offline_score(candidate_anchor)
```

对同一事件的所有候选锚点预测 score，选择预测 score 最低的锚点。

同时输出 coarse + residual 解释：

```text
selected_coarse_delay_ms = nearest coarse delay
selected_residual_offset_ms = selected_fine_delay_ms - selected_coarse_delay_ms
```

### 输入特征

输入特征只来自候选锚点可见范围：

- 历史 steering 统计量。
- steering 斜率和二阶变化。
- 横向偏移统计量和变化率。
- yaw / ay / speed / brake / accel 统计量。
- 道路曲率和道路横向信息。
- phase 特征。
- scene/pool 类型。
- candidate delay 和 coarse/fine/residual 编码。

所有特征必须按 train split 拟合标准化或编码器，再应用到 val/test。

### 基线模型

v247 第一版使用简单模型：

- `policy_keep_current_anchor`
- `policy_wait_to_latest_anchor`
- `selector_ridge_score`
- `selector_random_forest_score`
- 可选 `selector_mlp_score`

如果 Ridge 或 RF 结果与固定等待策略完全一致，报告必须明确指出它只是学到了等待偏置，而不是逐事件最佳锚点规律。

## 评价指标

评价不能只看 exact match。v247 必须报告：

```text
exact_50ms_match_rate
within_50ms_rate
within_100ms_rate
within_200ms_rate
selected_score_gap
selected_prediction_error
selected_error_delta_vs_current
gain_capture_rate
mean_selected_delay_ms
selected_delay_distribution
```

必须按以下 group 分层：

```text
all
normal
bad_top10
very_bad_top5
early_bad_top10
observe_later_like
strong_steer
reverse
base_delay_0
base_delay_200
base_delay_400
```

关键判断标准：

- normal 不能明显受伤。
- bad_top10 和 early_bad_top10 应明显改善。
- selector 平均等待时间不能无约束地塌缩到 1000ms。
- selector 需要在 `within_100ms_rate` 或 `within_200ms_rate` 上明显优于固定策略。

## 输出产物

v247 输出目录建议：

```text
05_rebuild_from_raw_20260511/03_baselines/v247_multi_resolution_best_anchor_discovery_20260630
```

核心表：

```text
tables/v247_fine_anchor_candidate_table.csv
tables/v247_best_anchor_by_event.csv
tables/v247_best_anchor_distribution.csv
tables/v247_signal_anchor_diagnostics.csv
tables/v247_selector_training_table.csv
tables/v247_selector_predictions_by_candidate.csv
tables/v247_selector_selected_anchor_by_event.csv
tables/v247_selector_policy_summary.csv
tables/v247_score_weight_sweep_summary.csv
```

核心图：

```text
figures/v247_best_anchor_distribution_by_group.png
figures/v247_error_delay_score_curves_examples.png
figures/v247_selector_vs_oracle_error.png
figures/v247_selected_delay_distribution.png
figures/v247_signal_anchor_alignment.png
```

核心报告：

```text
reports/v247_multi_resolution_best_anchor_discovery_cn.md
```

核心日志：

```text
logs/guardrail_check.json
logs/input_file_hashes.csv
logs/run_manifest.json
```

## Guardrails

`guardrail_check.json` 至少包含：

```text
pass
stage
no_trajectory_model_training
input_only_selector
oracle_best_anchor_upper_bound_only
no_test_based_retuning
no_event_uid_or_recording_as_features
fine_grid_sampling_checked
score_weights_declared_before_test_summary
zip_testzip
```

## 实施顺序

1. 审查原始 rolling arrays 是否支持 50ms fine grid。
2. 如果支持，构造 `fine_anchor_candidate_table`；如果不支持，报告可用最小锚点间隔并降级。
3. 计算每个候选锚点的预测误差、等待代价、局部不稳定惩罚。
4. 生成三套 offline best anchor label。
5. 分析 best anchor 分布，重点看是否从“全部最晚”回到合理时间段。
6. 构造 input-only selector 训练表。
7. 训练 Ridge/RF/可选 MLP selector。
8. 与 keep-current、wait-latest、oracle labels 对照。
9. 输出中文报告、图表、guardrail 和 ZIP。
10. 同步 `PROJECT_STATUS_CN.md`、`TASK_QUEUE_CN.md`、`ARTIFACT_INDEX_CN.md` 和当天 daily log。

## 风险和判定

如果 v247 出现以下结果，路线成立：

- error-only 偏晚，但 delay/instability score 能把最佳锚点拉回更合理分布。
- bad_top10 和 early_bad_top10 的 fine best anchor 明显不同于 normal。
- selector 在 `within_100ms_rate`、`within_200ms_rate` 和 selected error 上优于固定等待策略。
- 平均等待时间没有塌缩到 1000ms。

如果出现以下结果，需要停止扩展：

- 合理 score 下仍几乎全部选择 1000ms。
- selector 与 wait-latest 完全一致。
- normal 明显受伤。
- fine grid 标签高度不稳定，相邻 50ms 的 best anchor 随噪声跳动。

这种情况下应回到任务定义本身，重新检查 prediction horizon、事件起点和 label window，而不是继续加复杂 selector。
