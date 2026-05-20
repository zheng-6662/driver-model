# v0.5 服务器处理后车辆数据重筛 + 被试划分旧流程车辆-only

生成时间：2026-05-20 12:07:55

## 这次做了什么

本轮不再直接使用本地原始车辆 CSV 的 v0.4 筛选结果，而是在服务器现有的已对齐/清洗车辆数据上重新按之前的规则筛选。

服务器车辆数据入口：

`/root/autodl-tmp/data_process/01_datasets/多模态数据/被试数据集合/被试/vehicle/*_vehicle_aligned_cleaned.csv`

筛选规则保持和 v0.4 一致：

- 锚点后车辆动态仍明显变化：保留；
- 锚点后车辆有变化但驾驶员操作弱：保留；
- 车和驾驶员都有弱变化：次级保留；
- 快速打方向但车辆变化弱：待复核；
- 锚点后车和人都没有明显变化：排除；
- 锚点偏晚、窗口不完整或坐标风险：复核或排除。

## 样本筛选结果

- 初始 episode：1574
- 筛选用途统计：{'primary_train': 1160, 'exclude': 186, 'review': 158, 'secondary_train': 70}
- 进入本次训练的样本范围：primary + secondary + manual_review
- 训练 manifest 行数：1388
- 样本来源：{'primary': 1160, 'manual_review': 158, 'secondary': 70}

## 被试划分

本轮采用被试分组划分：

- test 被试：['cwh', 'gf', 'tyy']
- val 被试：['byx', 'gzj', 'yyl']
- train 被试：其余被试

manifest split：{'train': 960, 'val': 263, 'test': 165}

旧流程 loader 检查：

```json
{
  "status": "ok",
  "manifest_rows": 1388,
  "old_loader_kept_rows": 1376,
  "old_loader_dropped_rows": 12,
  "split_counts_after_old_loader": {
    "train": 953,
    "val": 260,
    "test": 163
  },
  "subject_counts_after_old_loader": {
    "zx": 178,
    "hzh": 120,
    "byx": 107,
    "txj": 99,
    "zdq": 99,
    "yzy": 93,
    "zxy": 91,
    "rjy": 77,
    "yyl": 77,
    "gzj": 76,
    "gf": 76,
    "jy": 58,
    "cwh": 58,
    "lxy": 48,
    "lx": 38,
    "xst": 30,
    "tyy": 29,
    "zt": 22
  },
  "source_group_counts_after_old_loader": {
    "primary": 1158,
    "manual_review": 148,
    "secondary": 70
  }
}
```

## 模型口径

- 旧流程 `FAIR09 / E1` 车辆-only；
- 车辆数据 + 粗细双头；
- 不加连续驾驶风格；
- 不加生理；
- 不加脑电；
- 不加教师蒸馏；
- seed=2026，epochs=40，batch=64，lr=0.001；
- device=`cuda`。

## 当前结果

- test steer RMSE：0.338616
- primary RMSE：0.218387
- tail RMSE：0.310550
- selection：0.820553
- best epoch：26

## 预测图

- 预测总览图：`/root/autodl-tmp/data_process/tmp/event_conditioned_runs/V05_SERVER_ALIGNED_SUBJECT_FAIR09_vehicle_only_seed2026_20260520_120144/prediction_figures/test/overview.png`
- 预测图目录：`/root/autodl-tmp/data_process/tmp/event_conditioned_runs/V05_SERVER_ALIGNED_SUBJECT_FAIR09_vehicle_only_seed2026_20260520_120144/prediction_figures/test`
- 逐样本指标：`/root/autodl-tmp/data_process/tmp/event_conditioned_runs/V05_SERVER_ALIGNED_SUBJECT_FAIR09_vehicle_only_seed2026_20260520_120144/prediction_figures/test/prediction_sample_metrics.csv`

## 产物位置

- v0.5 筛选总表：`/root/autodl-tmp/data_process/05_rebuild_from_raw_20260511/02_samples/extreme_condition_episodes_server_aligned_v0_5/tables/extreme_condition_episodes_refiltered_v0_5.csv`
- v0.5 主训练表：`/root/autodl-tmp/data_process/05_rebuild_from_raw_20260511/02_samples/extreme_condition_episodes_server_aligned_v0_5/tables/primary_train_episodes_v0_5.csv`
- v0.5 次级训练表：`/root/autodl-tmp/data_process/05_rebuild_from_raw_20260511/02_samples/extreme_condition_episodes_server_aligned_v0_5/tables/secondary_train_episodes_v0_5.csv`
- v0.5 待复核表：`/root/autodl-tmp/data_process/05_rebuild_from_raw_20260511/02_samples/extreme_condition_episodes_server_aligned_v0_5/tables/manual_review_episodes_v0_5.csv`
- 旧流程 manifest：`/root/autodl-tmp/data_process/05_rebuild_from_raw_20260511/03_processed_datasets/stage03_v05_server_aligned_subject_oldflow_fair09/tables/oldflow_fair09_vehicle_only_server_aligned_v05_subject_split_manifest.csv`
- manifest 检查：`/root/autodl-tmp/data_process/05_rebuild_from_raw_20260511/03_processed_datasets/stage03_v05_server_aligned_subject_oldflow_fair09/tables/oldflow_fair09_vehicle_only_server_aligned_v05_subject_split_manifest_check.json`
- 运行记录：`/root/autodl-tmp/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_v05_server_aligned_subject_oldflow_fair09/tables/server_aligned_v05_subject_oldflow_fair09_run_record.csv`
- 运行目录：`/root/autodl-tmp/data_process/tmp/event_conditioned_runs/V05_SERVER_ALIGNED_SUBJECT_FAIR09_vehicle_only_seed2026_20260520_120144`

## 怎么理解

这次和上一次最大的区别是：数据入口换成服务器上的处理后车辆 CSV，切分换成被试分组。这个结果更严格，但也更难；如果指标明显变差，不一定是筛选规则错，也可能是跨被试泛化难度上升。

## 本地拉回后的查看位置

- 本地运行目录：`F:\data_set_process\data_process\tmp\event_conditioned_runs\V05_SERVER_ALIGNED_SUBJECT_FAIR09_vehicle_only_seed2026_20260520_120144`
- 本地预测总览图：`F:\data_set_process\data_process\tmp\event_conditioned_runs\V05_SERVER_ALIGNED_SUBJECT_FAIR09_vehicle_only_seed2026_20260520_120144\prediction_figures\test\overview.png`
- 本地逐样本预测图目录：`F:\data_set_process\data_process\tmp\event_conditioned_runs\V05_SERVER_ALIGNED_SUBJECT_FAIR09_vehicle_only_seed2026_20260520_120144\prediction_figures\test`
- 本地分被试样本级指标：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_server_aligned_subject_oldflow_fair09\tables\test_subject_sample_metrics_v0_5.csv`
- 本地分道路类型样本级指标：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_server_aligned_subject_oldflow_fair09\tables\test_road_type_sample_metrics_v0_5.csv`
- 本地分机制标签样本级指标：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_server_aligned_subject_oldflow_fair09\tables\test_mechanism_sample_metrics_v0_5.csv`

## 分被试补充检查

这里使用的是预测图生成后的逐样本指标均值，和上面的全局 RMSE 口径不完全一样，但可以帮助判断哪个测试被试更难：

| 测试被试 | 样本数 | 样本均值 RMSE | 主阶段均值 RMSE | 尾段均值 RMSE | 主方向一致率 |
|---|---:|---:|---:|---:|---:|
| cwh | 58 | 0.2130 | 0.2239 | 0.2448 | 0.7931 |
| gf | 76 | 0.1971 | 0.1951 | 0.2280 | 0.6447 |
| tyy | 29 | 0.4391 | 0.3836 | 0.5851 | 0.7241 |

当前看法：`tyy` 仍然是明显更难的测试被试，尾段误差尤其高。这个现象和之前困难样本集中在 `tyy` 的观察方向一致，所以后续不能只看整体 RMSE，还要继续看分被试预测图和样本类型分布。
