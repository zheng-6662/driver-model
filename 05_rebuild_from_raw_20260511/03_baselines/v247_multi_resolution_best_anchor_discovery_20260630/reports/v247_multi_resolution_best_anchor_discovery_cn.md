# v247 multi-resolution best anchor discovery 报告

## 结论摘要

- 50ms fine grid 采样审计：生成 `24507` 行，事件数 `1167`，完整事件比例 `1.000`，delay 值数量 `21`；fine_grid_supported=`True`。
- 锁定 v241 推理：device=`cuda`，point 数 `514647`，耗时 `15.8` 秒；没有训练新轨迹模型。
- v241 coarse replay 对齐：coarse replay mean RMSE=0.000000, max=0.000001。这一步检查 fine-grid 里 0/200/.../1000ms 是否能复现旧 v241 预测。
- primary score=`delay_l05_unstable_m05` 下，test/all 当前 0ms 平均误差 `0.475`，oracle best 平均误差 `0.253`，平均 best delay `596.2ms`。
- test/bad_top10 当前 0ms 平均误差 `1.198`，oracle best 平均误差 `0.616`，平均 best delay `789.5ms`。
- RF selector 在 test/all 的平均选中误差 `0.399`，相对当前 0ms delta `-0.076`，within100ms `0.299`，平均选中 delay `469.8ms`。
- RF selector 在 test/bad_top10 的平均选中误差 `0.947`，相对当前 0ms delta `-0.250`，gain capture `0.271`。
- 固定 wait-latest 在 test/bad_top10 的平均选中误差 `0.695`，平均 delay `1000.0ms`；这个基线用于判断 selector 是否只是学到“永远等到最后”。
- 信号代理锚点诊断：best 与 min-instability proxy 平均距离 290.6ms；best 与 peak-steer-change proxy 平均距离 504.9ms。

## 怎么理解这一步

v247 做的不是把所有样本强行后移，而是给每个事件同时生成 0ms 到 1000ms、间隔 50ms 的候选观察点。每个候选点都重新从原始车辆 CSV 中取历史窗口和未来监督窗口，然后用同一个 v241 模型预测。离线 best anchor 是对这些候选点打分后的最优点，score 同时考虑预测误差、等待代价和局部不稳定性。

如果 error-only 几乎总是选 1000ms，而加入等待代价/不稳定性后 best delay 回到中间区间，说明任务定义比单纯“后移锚点”更合理。如果 selector 明显优于 keep-0ms 且不只是等到 latest，才值得进入下一步更强模型训练。

## 关键产物

- `tables/v247_fine_anchor_candidate_table.csv`：每个事件 21 个 fine anchor 的误差和 score。
- `tables/v247_best_anchor_by_event.csv`：不同 score 定义下每个事件的离线 best anchor。
- `tables/v247_selector_training_table.csv`：selector 使用的 input-only 特征表。
- `tables/v247_selector_selected_anchor_by_event.csv`：selector/policy 选中的锚点。
- `tables/v247_selector_policy_summary.csv`：selector 与 current/latest/oracle 的分组对比。
- `figures/v247_best_anchor_distribution_by_group.png`：best anchor 分布。
- `figures/v247_error_delay_score_curves_examples.png`：典型差样本的 error/score-delay 曲线。

## 风险和下一步

- oracle best anchor 使用未来真实误差，只能作为离线标签和上限，不能部署。
- fine grid 的监督点是相对每个 candidate anchor 的 0.1s 网格；50ms candidate 的 tail 点会落在 1.05/1.15/... 这类原始相对时刻，这是本版使用 raw nearest 采样的直接结果。
- 下一步是否训练更强的 anchor-aware 轨迹模型，取决于 `selector_random_forest_score` 是否在 test/bad_top10 上超过 wait-latest，并且 normal 组没有明显变差。

## selector score 拟合诊断

| selector_name                | split   |     n |   target_score_rmse |   target_score_mae |
|:-----------------------------|:--------|------:|--------------------:|-------------------:|
| selector_random_forest_score | test    |  3864 |           0.367776  |          0.265683  |
| selector_random_forest_score | train   | 14154 |           0.042212  |          0.0314665 |
| selector_random_forest_score | val     |  6489 |           0.720702  |          0.485875  |
| selector_ridge_score         | test    |  3864 |           0.36867   |          0.26687   |
| selector_ridge_score         | train   | 14154 |           0.0659423 |          0.0486696 |
| selector_ridge_score         | val     |  6489 |           0.71888   |          0.485571  |