# 完整记录级 episode 重建 v1.0 用户说明

生成时间：2026-05-20 22:00:45

## 这一步做了什么

本流程不训练模型，而是从完整一次实验车辆 CSV 中自动切出多个驾驶 episode。每个 episode 同时记录车辆状态、驾驶员操作、道路/场景上下文和锚点质量。

## 当前运行结果

- 扫描车辆记录数：91
- 成功处理记录数：91
- 检测到 episode 总数：1766

### episode 分组

| episode_group_cn    |   count |
|:--------------------|--------:|
| 核心极限样本        |     973 |
| 保守/弱操作极限样本 |     406 |
| 需要复核            |     335 |
| 边界复核样本        |      45 |
| 次级训练样本        |       4 |
| 正常弯道或普通操控  |       3 |

### 上下文覆盖

| context                    |   count |
|:---------------------------|--------:|
| is_low_mu_context          |    1425 |
| is_curve_context           |     308 |
| is_roll_context            |    1432 |
| is_lateral_dynamic_context |    1641 |

## 输出位置

- 总表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_0\tables\record_level_episodes_all_v1_0.csv`
- 分组统计：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_0\tables\record_episode_group_summary_v1_0.csv`
- 复核图目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_0\figures\review_panels`
- 3D 静态轨迹目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_0\figures\trajectory_3d_static`

## 解释边界

- 道路信息只作为工况上下文，不直接当作最终事件锚点。
- 当前是自动初筛，后续需要人工查看复核图后再定最终训练样本。
- 当前系统已经允许一条完整实验记录产生多个 episode，不再默认一条记录只有一个事件。