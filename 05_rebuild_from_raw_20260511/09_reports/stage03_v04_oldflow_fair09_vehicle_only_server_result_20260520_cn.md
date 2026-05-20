# v0.4 新样本集接入旧流程车辆-only模型结果

生成时间：2026-05-20 11:25

## 这次跑的是什么

这次按照用户要求，不使用 `05_rebuild` 里新写的轻量车辆-only模型，而是对齐之前截图表格里的旧流程模型口径，只看车辆-only。

模型口径：

- 对应旧流程 `FAIR09 / E1`：车辆数据 + 粗细双头 + 无显式事件注入。
- 不加连续驾驶风格。
- 不加生理数据。
- 不加脑电。
- 不加教师蒸馏。
- 只把 v0.4 新筛选出来的样本接入旧流程模型。

## 训练参数

- seed：2026
- device：cuda
- epochs：40
- min_epochs：40
- patience：99
- batch size：64
- learning rate：0.001
- weight decay：0
- grad clip：1.0
- selection mode：legacy_rmse
- conditioning mode：vehicle_direct_coarse_fine
- 模型结构：旧流程 Transformer 编码器 + 粗细双头轨迹解码器
- d_model：128
- nhead：2
- encoder layers：2
- decoder layers：2
- ffn_dim：256
- dropout：0.1
- event_embed_dim：96
- event_bin_size：20
- teacher_forcing_ratio：0
- event_loss_weight：0

## 样本与数据读取

- v0.4 合并样本：1422 行。
- 旧流程 loader 实际保留：1410 个样本。
- 旧流程 loader 丢弃：12 个样本。
- 训练 / 验证 / 测试：814 / 273 / 323。
- 样本来源：
  - 主训练候选：1126
  - 待复核样本：183
  - 次级候选：101

服务器上的车辆数据不是本地 `原始车辆数据/被试/xxx_vehicle.csv` 结构，而是：

`/root/autodl-tmp/data_process/01_datasets/多模态数据/被试数据集合/被试/vehicle/*_vehicle_aligned_cleaned.csv`

因此脚本做了兼容：本地优先读取原始车辆 CSV；服务器找不到原始 CSV 时，自动映射到对应的已对齐车辆 CSV。这个过程只使用车辆数据，不引入风格、生理或脑电。

## 结果

| 指标 | 结果 |
|---|---:|
| test steer RMSE | 0.5408 |
| primary RMSE | 0.3481 |
| tail RMSE | 0.4781 |
| selection | 0.9841 |
| 最佳验证轮次 | epoch 25 |

测试集逐样本图表里还有一些物理相关指标：

| 指标 | 均值 | 中位数 |
|---|---:|---:|
| 逐样本 2s RMSE | 0.3747 | 0.2628 |
| 主响应段 RMSE | 0.3820 | 0.2407 |
| 尾段 RMSE | 0.4395 | 0.3072 |
| 方向一致率 | 0.7183 | 1.0000 |
| 尾段方向一致率 | 0.4644 | 0.0000 |
| 主峰幅值误差 | 0.2889 | 0.1573 |
| 峰值时间误差 / s | 0.6074 | 0.4000 |
| 幅值范围误差 | 0.5259 | 0.3078 |

## 怎么理解

这次结果不能直接和旧 FAIR01-16 表里的 E1 数值做强结论比较，因为样本清单已经换成了 v0.4 极限/近极限工况样本，任务难度不一样。

但从结果本身看：

- 旧流程车辆-only粗细双头可以正常训练和出图，说明 v0.4 样本能够接入旧模型。
- test RMSE=0.5408，高于旧表中 E1 的 0.4729，说明新样本集对旧模型更难。
- primary RMSE=0.3481，与旧 E1 的 0.3399 接近，说明主响应阶段没有完全崩。
- tail RMSE=0.4781，明显高于旧 E1 的 0.4040，说明新样本的后段回正、持续同侧或漂移更难预测。
- selection=0.9841，高于旧 E1 的 0.8627，说明旧流程综合选择指标下，新样本明显更难。

当前判断：

新 v0.4 样本不是“让模型指标更好看”的样本集，而是更偏向极限/近极限工况，旧流程车辆-only模型在这些样本上仍然暴露出后段和形态预测困难。这个结果更适合作为后续判断“是否需要响应类型、动作阶段、风格和生理信息”的诊断基线。

## 主要产物

- 本地运行目录：`F:/data_set_process/data_process/tmp/event_conditioned_runs/V04_OLD_FLOW_FAIR09_vehicle_only_coarse_fine_seed2026_20260520_111848`
- 本地指标文件：`F:/data_set_process/data_process/tmp/event_conditioned_runs/V04_OLD_FLOW_FAIR09_vehicle_only_coarse_fine_seed2026_20260520_111848/metrics.json`
- 本地训练日志：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/00_project_notes/server_logs/stage03_v04_oldflow_fair09_train_latest.log`
- 本地运行记录：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_v04_oldflow_fair09_vehicle_only/tables/oldflow_fair09_vehicle_only_run_record.csv`
- 本地预测总览图：`F:/data_set_process/data_process/tmp/event_conditioned_runs/V04_OLD_FLOW_FAIR09_vehicle_only_coarse_fine_seed2026_20260520_111848/prediction_figures/test/overview.png`
- 本地逐样本预测图目录：`F:/data_set_process/data_process/tmp/event_conditioned_runs/V04_OLD_FLOW_FAIR09_vehicle_only_coarse_fine_seed2026_20260520_111848/prediction_figures/test`

## 服务器记录

- 连接格式：`ssh -p 55060 root@connect.westc.seetacloud.com`
- 未在日志、报告或代码中记录密码。
- 远程项目路径：`/root/autodl-tmp/data_process`
- screen 名称：`oldfair09`
- 远程训练日志：`/root/autodl-tmp/data_process/05_rebuild_from_raw_20260511/00_project_notes/server_logs/stage03_v04_oldflow_fair09_train_latest.log`
- 远程运行目录：`/root/autodl-tmp/data_process/tmp/event_conditioned_runs/V04_OLD_FLOW_FAIR09_vehicle_only_coarse_fine_seed2026_20260520_111848`
- 当前状态：训练已完成，screen 已退出，GPU 显存回到 0 MiB。

## 下一步建议

先不要立刻加风格或生理。建议先看这 12 张测试预测图，尤其是：

- 是否仍然出现方向错侧；
- 是否仍然把大幅响应预测成小幅；
- 尾段为什么差；
- `manual_review` 样本是否明显拉乱；
- 主训练候选里是否仍有锚点偏晚或正常驾驶混入。

如果预测图显示主要问题仍是后段形态和响应阶段混合，就应该优先做“响应阶段/类型分层”或重新定义标签窗口，而不是直接把生理信号塞进去。
