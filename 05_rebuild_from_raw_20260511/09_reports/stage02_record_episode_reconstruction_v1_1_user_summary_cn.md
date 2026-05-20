# 完整记录级 episode 复核后样本集 v1.1

生成时间：2026-05-20 22:40:18

## 人工复核决策

用户查看 v1.0 复核图后给出的判断是：大部分自动筛出的极限/近极限 episode 可以继续保留；“需要复核”和“边界复核”类基本可以舍去。因此 v1.1 不重新检测 episode，只把 v1.0 候选库整理成更清晰的训练/对照/舍弃三类。

## 数量

- v1.0 episode 总数：1766
- v1.1 主训练候选：1383
- v1.1 对照样本：3
- v1.1 舍弃/暂缓：380

## 主训练候选保留规则

保留：

- 核心极限样本；
- 保守/弱操作极限样本；
- 次级训练样本。

不进入主训练：

- 需要复核；
- 边界复核样本；
- 正常弯道或普通操控。

其中正常弯道或普通操控不是删除，而是单独作为对照样本保存。

## 上下文覆盖

| 范围 | 数量 | 低附着 | 弯道 | 横滚/姿态 | 横向动态 |
|---|---:|---:|---:|---:|---:|
| 主训练候选 | 1383 | 1142 | 269 | 1339 | 1348 |
| 对照样本 | 3 | 0 | 3 | 0 | 0 |
| 舍弃/暂缓 | 380 | 283 | 36 | 93 | 293 |

## 输出位置

- 全量带复核决策表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_1_reviewed\tables\record_level_episodes_all_reviewed_v1_1.csv`
- 主训练候选表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_1_reviewed\tables\train_candidate_extreme_episodes_v1_1.csv`
- 对照样本表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_1_reviewed\tables\control_normal_or_curve_episodes_v1_1.csv`
- 舍弃/暂缓表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_1_reviewed\tables\discarded_review_episodes_v1_1.csv`
- 分组统计表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_1_reviewed\tables\record_episode_review_decision_summary_v1_1.csv`
- 复核图索引：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_1_reviewed\tables\record_episode_figure_index_v1_1.csv`

## 下一步

v1.1 已经可以作为下一轮车辆-only 数据集构建入口。但正式训练前建议先从主训练候选里再抽查一小批核心极限和保守/弱操作样本，确认“需要复核类整体舍弃”没有误删大量有效样本。
