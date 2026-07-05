# 给 GPTPro 的中文复盘请求：v229 两个月经验与失败分类

请先阅读这个本地复盘包，然后只给一个 bounded 下一步建议，不要直接要求训练更大模型。

## 本地输出包

- v229 报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v229_two_month_lessons_failure_taxonomy_20260623\reports\v229_two_month_lessons_failure_taxonomy_cn.md`
- 失败桶统计：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v229_two_month_lessons_failure_taxonomy_20260623\tables\v229_failure_taxonomy_by_pool_event.csv`
- 高尾失败案例：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v229_two_month_lessons_failure_taxonomy_20260623\tables\v229_top_tail_failure_cases.csv`
- selector/candidate 诊断：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v229_two_month_lessons_failure_taxonomy_20260623\tables\v229_selector_candidate_diagnosis.csv`
- 下一步决策矩阵：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v229_two_month_lessons_failure_taxonomy_20260623\tables\v229_next_action_decision_matrix.csv`

## 当前正式锁定结果

- loose_main_pool = avg_joint_focus
  - test n = 184
  - RMSE = 0.544884
  - tail RMSE = 0.629752
  - direction_acc = 0.967391
  - under_rate = 0.163043
  - top20 tail-SSE share = 0.659320
- strict_main_pool = peak_floor_090
  - test n = 174
  - RMSE = 0.571770
  - tail RMSE = 0.658306
  - direction_acc = 0.948276
  - under_rate = 0.137931
  - top20 tail-SSE share = 0.672493

## 我希望你重点判断的问题

1. 这两个月的证据是否支持：当前应进入论文写作/结果整理，而不是继续模型搜索？
2. 如果还允许继续推进，是否应先做失败样本 taxonomy 和人工复核，而不是 v222b/v223、新 gate/router、新 tau/threshold？
3. 当前经典问题是否应表述为：方向和普通响应可预测，但强反应幅值、极端峰值、尾段、反转/多次修正仍是主要限制？
4. v225 diagnostic-only 结果显示 oracle/candidate 上限存在，但 learned selector 不稳。这个结论是否足以继续禁止同空间 current-window selector 扩大化？
5. 如果你认为必须继续实验，请只给一个窄范围任务，必须包含：
   - 允许输入与禁止输入；
   - validation-only 选择规则；
   - test reporting-only 规则；
   - 明确 stop condition；
   - 输出文件和验收命令。

## 本地边界

- 不允许 test-based retuning。
- 不允许把 oracle、true label、fallback 或 diagnostic-only 行写入 formal headline。
- 不允许把 W3_B4_original_soft 写入 formal leaderboard、formal gate、formal oracle、usage table 或 selected config。
- 不允许在没有明确 stop condition 的情况下启动 v222b/v223、大 gate/router、新 tau/threshold 或新模型训练。
- 请用中文回答，并优先给路线判断，不要只列模型名。
