# v0.4 极限工况样本重新筛选说明

## 这次筛选改了什么

这次不是继续比较 809 个样本版本，而是回到 v0.3 的 1574 个初始 episode，按新的人工判断重新筛：

- 方向盘转动速度要作为驾驶员紧急操作证据；
- 要检查当前锚点是否偏晚，如果锚点后车辆和驾驶员都已经稳定，则不作为训练样本；
- 如果锚点后车辆状态有明显变化，即使驾驶员没有明显操作，也可以保留，因为这可能代表保守驾驶员、制动为主或车辆扰动主导；
- 如果只有方向盘快打但车辆变化弱，先放入人工复核，不直接作为核心极限样本。

## 总体数量

- 初始 episode 数：1574
- 主训练候选：1128
- 次级训练候选：101
- 主+次级候选合计：1229
- 待人工复核：193
- 暂排除：152

## 分类数量

| v04_label_cn                     |   count |
|:---------------------------------|--------:|
| 核心保留：车辆变化+驾驶员操作    |     721 |
| 核心保留：车辆变化但驾驶员操作弱 |     407 |
| 复核：坐标连续性风险             |     156 |
| 排除：锚点后车和人都弱           |     148 |
| 次级保留：快打方向且有弱车辆变化 |      59 |
| 次级保留：车和人都有弱变化       |      42 |
| 复核：语义不清                   |      17 |
| 复核：窗口不完整                 |      14 |
| 复核：快打方向但车辆变化弱       |       6 |
| 排除：锚点偏晚或事件已稳定       |       4 |

## 按场景/上下文统计

| condition_context_cn   |   exclude |   primary_train |   review |   secondary_train |
|:-----------------------|----------:|----------------:|---------:|------------------:|
| 低附着                 |       133 |             633 |      178 |                92 |
| 弯道/曲率              |         0 |             104 |        5 |                 3 |
| 普通驾驶对照           |        19 |              62 |        1 |                 3 |
| 横向动态               |         0 |               8 |        0 |                 0 |
| 横滚/姿态              |         0 |             321 |        9 |                 3 |

## 怎么理解

- 主训练候选不是只看方向盘，也不是只看车身，而是锚点后车辆动态仍然发生或增强的样本。
- 驾驶员操作弱但车辆变化明显的样本被保留，这符合用户最新判断。
- 锚点后车和人都弱的样本被排除，因为它们更像锚点偏晚、事件结束或直线轻微维持。
- 快打方向但车辆变化弱的样本先复核，因为其中一部分可能是直线维持方向盘，一部分可能是真正紧急操作但车辆响应不强。

## 输出位置

- 总表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\extreme_condition_episodes_v0_4\tables\extreme_condition_episodes_refiltered_v0_4.csv`
- 主训练候选：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\extreme_condition_episodes_v0_4\tables\primary_train_episodes_v0_4.csv`
- 次级训练候选：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\extreme_condition_episodes_v0_4\tables\secondary_train_episodes_v0_4.csv`
- 主+次级训练候选：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\extreme_condition_episodes_v0_4\tables\train_candidate_episodes_v0_4.csv`
- 待复核：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\extreme_condition_episodes_v0_4\tables\manual_review_episodes_v0_4.csv`
- 排除：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\extreme_condition_episodes_v0_4\tables\excluded_episodes_v0_4.csv`
- 复核图索引：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\extreme_condition_episodes_v0_4\tables\v04_review_figure_index.csv`
- 复核图目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\extreme_condition_episodes_v0_4\figures\review_panels`

本轮共生成复核图 257 张。