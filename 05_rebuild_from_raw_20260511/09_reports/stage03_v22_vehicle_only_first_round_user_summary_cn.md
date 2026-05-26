# v2.2 车辆-only 第一轮训练结果说明

生成时间：2026-05-26  
任务范围：只训练车辆-only，不加入连续驾驶风格、生理、脑电。

## 1. 本轮做了什么

本轮使用 `v2.2 epoch refined` 样本池，在服务器 GPU 上训练了第一批车辆-only 模型。

输入设定：
- 输入窗口：锚点前 3 秒车辆状态，20 Hz。
- 预测窗口：锚点后 2 秒。
- 输入特征：方向盘角、方向盘角速度、车速、制动、油门、纵向加速度、横向加速度、横摆角速度、横滚角速度、横滚角、横向偏移、附着系数、道路曲率。
- 输出目标：方向盘相对轨迹、横摆角速度、横向加速度。
- 没有使用未来峰值、未来响应强度、生理数据、脑电或连续驾驶风格。

服务器执行情况：
- 服务器 GPU：NVIDIA GeForce RTX 5090。
- 服务器上原来的 PyTorch 版本不能支持 5090 实际训练，已升级到 `torch 2.11.0+cu128` 后训练通过。
- 所有训练任务已结束，服务器当前没有残留 screen 训练进程，GPU 显存占用为 0。

## 2. 样本情况

v2.2 原始训练池共有 1721 行。构建固定窗口训练数组后，可用样本为 1696 行，主要是少量窗口或文件问题被剔除。

按被试划分：

| split | 样本数 |
|---|---:|
| train | 1100 |
| val | 328 |
| test | 268 |

按训练角色：

| 样本角色 | 可用数量 | 说明 |
|---|---:|---|
| 主训练候选 | 945 | 当前核心训练样本 |
| 待复核恢复候选 | 447 | 人工讨论后认为不能直接丢弃的样本 |
| 弱响应/对照候选 | 304 | 可作为普通/弱响应对照 |

本轮模型分两类训练：
- `core`：只使用主训练候选；
- `core_review`：使用主训练候选 + 待复核恢复候选。

注意：当前表中 `episode_type` 仍保留了一些历史分类名，例如 `excluded_slope_or_offroad`。这不等价于本轮仍按旧规则排除或纳入；本轮真正控制训练的是 `v2_1_role` 和 v2.2 边界字段。后续需要把图标题和样本表里的历史分类字段统一整理，避免人工看图时误读。

## 3. 第一轮结果

| 版本 | 模型 | 训练集 | test n | 方向盘 RMSE | 尾段 RMSE | 横摆 RMSE | 横向加速度 RMSE | 错侧率 | 严重幅值不足率 | 大响应召回 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M0 | 保持锚点值 | core | 113 | 0.6069 | 0.7088 | 0.1022 | 2.4113 | 0.8496 | 0.7080 | 0.0000 |
| M1 | GRU | core | 113 | 0.4200 | 0.4733 | 0.0739 | 1.5925 | 0.2743 | 0.1858 | 0.7625 |
| M2 | TCN | core | 113 | 0.4684 | 0.5107 | 0.0808 | 1.8135 | 0.1858 | 0.3274 | 0.5875 |
| M3 | Transformer | core | 113 | 0.4329 | 0.4898 | 0.0747 | 1.6129 | 0.1858 | 0.1681 | 0.8625 |
| M4A | GRU | core + review | 203 | 0.4267 | 0.4872 | 0.0652 | 1.4327 | 0.2463 | 0.1330 | 0.8205 |
| M4B | Transformer | core + review | 203 | 0.4287 | 0.4873 | 0.0681 | 1.4865 | 0.2266 | 0.2020 | 0.7009 |

## 4. 现在可以怎么理解

第一，学习模型明显强于“保持锚点值”的无学习基线。  
M0 的方向盘 RMSE、错侧率和严重幅值不足率都很差，说明当前任务不是单纯保持当前方向盘就能解决。

第二，`core` 训练下，GRU 的整体方向盘 RMSE 最低。  
M1 的方向盘 RMSE 为 0.4200，是本轮 core 样本上整体误差最低的版本。

第三，Transformer 在部分物理指标上更好。  
M3 的错侧率、严重幅值不足率和大响应召回优于 M1，但整体 RMSE 略高。这说明不同结构可能在“平均误差”和“物理合理性”之间有取舍。

第四，加入待复核恢复候选后，样本量变大，姿态类输出和幅值不足有所改善。  
M4A 的 test 样本从 113 增加到 203，横摆 RMSE、横向加速度 RMSE、严重幅值不足率都比 M1 更好，但方向盘 RMSE 略高。这个结果说明待复核样本不是完全无用，但也不能只看整体 RMSE，需要继续看预测图和分组。

第五，目前不能直接进入连续风格和生理结论。  
这轮只是确认 v2.2 样本池下车辆-only 任务能跑通，并给出第一批基线。下一步需要先看图、查错侧/幅值不足样本，再决定是否修正样本标签、模型输出或训练任务。

## 5. 结果文件在哪里

总结果表：
`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_v22_vehicle_only_baseline/v22_first_round_result_index.csv`

每个模型的结果目录：

| 版本 | 目录 |
|---|---|
| M1 GRU core | `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_v22_vehicle_only_baseline/runs/V22_M1_core_gru_seed2026/` |
| M3 Transformer core | `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_v22_vehicle_only_baseline/runs/V22_M3_core_transformer_seed2026/` |
| M4A GRU core+review | `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_v22_vehicle_only_baseline/runs/V22_M4A_core_review_gru_seed2026/` |
| M4B Transformer core+review | `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_v22_vehicle_only_baseline/runs/V22_M4B_core_review_transformer_seed2026/` |

每个目录里重点看：
- `experiment_summary.md`：该版本指标摘要；
- `metrics_summary.csv`：指标表；
- `per_sample_metrics.csv`：逐样本误差；
- `figure_index.html`：预测图索引；
- `figures/`：随机样本、最差样本、大响应、错侧、严重幅值不足等预测图。

## 6. 建议下一步

优先人工看三组图：

1. `V22_M1_core_gru_seed2026`  
   目的：看当前最低方向盘 RMSE 的预测形态是否真的合理。

2. `V22_M3_core_transformer_seed2026`  
   目的：看大响应召回和幅值不足较好的版本，是否比 GRU 更符合极限工况物理意义。

3. `V22_M4A_core_review_gru_seed2026`  
   目的：看加入待复核样本后，是否真的改善了大响应和姿态预测，还是只是增加了样本复杂度。

如果图上仍然存在明显锚点偏早/偏晚、方向反了、真实大幅但预测很小的问题，下一步应先做样本/锚点二次修正，而不是马上加入生理数据。
