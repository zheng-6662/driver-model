# 给 GPTPro 的提问词：生理数据如何真正弥补车辆锚点前信息不足

你现在作为外部方法审查者，请只从“如何改进模型预测行为”角度给建议，不要转成失败机制论文写作。

## 项目目标

我在做驾驶行为轨迹预测。当前最大问题是：锚点前车辆历史序列非常相似，但锚点后的方向盘/横向行为可能差异很大。用户希望充分利用生理数据和驾驶状态，弥补锚点前车辆信息不足，并让预测最差样本有本质性改善。

正式评价口径是 subject-disjoint：测试驾驶员不出现在训练驾驶员里。也有 subject-aware 诊断口径：同一驾驶员不同事件可分到 train/test，用来判断个体化潜力。

约束：
- 不使用 query 的锚点后信息。
- 不删除差样本。
- 不回到简单 gate / 轻量 residual / v222a 那类路线。
- 不做“失败机制分析论文”，目标是提升预测模型。
- 已经尝试过候选锚点/后移锚点，当前判断锚点已不是主要瓶颈。

## 已完成实验与关键结果

### v254a：1Hz/10Hz 生理深层信号审计

用 1Hz/10Hz 表格窗口特征做行为分类/回归诊断。

关键结果：
- 10Hz coverage 约 0.919，没有 post-observation 泄漏。
- subject-disjoint future_cluster4 test macro-F1：
  - vehicle_only = 0.7317
  - physio10hz = 0.2944
  - vehicle+physio10hz = 0.5020
- high_future_abs_q75 test macro-F1：
  - vehicle_only = 0.7112
  - physio10hz = 0.4897
  - vehicle+physio10hz = 0.6239
- 生理特征强烈编码 subject/recording，弱编码未来行为。future label eta² 顶部一般只有约 0.015 到 0.020。
- HRV_RMSSD 基本不可用，RESP_BPM/RESP_Amplitude 近似常量或 recording 级别常量。

结论：简单 1Hz/10Hz 表格拼接不能提供跨驾驶员行为增量。

### v254b：直接从清洗后 200Hz 连续生理层构造事件相关表征

方法：
- 用清洗后 200Hz 生理数据。
- 每个样本使用 observation_s 前 baseline (-60s,-20s) 做因果归一化。
- 事件窗口：pre20_pre10, pre10_pre5, pre5_pre2, pre2_0。
- 通道包括 HR_bpm, EMG_RMS, EMG_filt200, EDA_filt200, EDA_Tonic, EDA_Phasic, RESP_filt200, ECG_filt200, RESP_BPM, RESP_Amplitude。
- 构造 raw stats、z stats、burst_rate、recent arousal/motor/resp index。

覆盖：
- test ok_rate 0.8967；train ok_rate 0.8887；val ok_rate 1.0。
- uses_post_observation_rate = 0。
- 200Hz 窗口行数正常：pre2_0 约 400 rows。

subject-disjoint test 分类：
- future_cluster4:
  - vehicle_only macro-F1 0.7154
  - physio200_norm 0.2241
  - vehicle_plus_physio200_curated 0.6852
- high_future_abs_q75:
  - vehicle_only 0.7408
  - physio200_norm 0.5440
  - vehicle_plus_physio200_curated 0.6169
- high_future_range_q75:
  - vehicle_only 0.6553
  - physio200_norm 0.6252
  - vehicle_plus_physio200_curated 0.5971
- strong_steer_existing:
  - vehicle_only 0.6772
  - physio200_norm 0.5687
  - vehicle_plus_physio200_curated 0.6132
- bad_top10_v250_diagnostic:
  - vehicle_only 0.4958
  - physio200_norm 0.5006
  - physio200_curated 0.5095
  - vehicle_plus_physio200_curated 0.5170

subject-aware test 有一点现象：
- bad_top10_v250_diagnostic:
  - vehicle_only 0.4578
  - physio200_norm 0.5438
  - vehicle_plus_physio200_norm 0.6095
  - vehicle_plus_physio200_curated 0.5832

但对真正未来行为类别和回归没有改善。

结论：200Hz 手工事件表征也没有直接跨驾驶员增量，但可能对“哪些样本会差”有个体化诊断信号。

### v253b：车辆相似候选池 tie-break

方法：
- 先用 vehicle-only 找同 delay 的车辆相似候选池 top60。
- 在候选池里用 style/physio 最近邻选择未来原型。

subject-disjoint test bad_top10_v250:
- vehicle_rank1 selected_future_rmse_mean = 0.9934
- simple physio nearest = 1.1076，反而更差
- oracle_best_future_in_vehicle_pool = 0.3678

结论：候选池内确实有巨大上限，但简单生理距离不能选中。

### v255：学习式生理条件化候选重排序器

方法：
- 训练 pair ranker：query + 候选构成 pair。
- 候选池仍是车辆 top60。
- 特征包括 vehicle_rank/dist、候选训练未来原型摘要、query-candidate 200Hz 生理距离、recent 生理 index 差异。
- 训练 HistGradientBoostingRegressor 预测 pair 的 future RMSE，val 上调 no-harm 阈值。
- 有三个策略：learned_vehicle_context、learned_physio_state、badweighted_physio。

结果：
- val 上所有 learned ranker 一旦允许重排，bad_top10 都变差；no-harm 策略最终全部选择 threshold=1e9，即退回 vehicle_rank1。
- subject-disjoint test bad_top10：
  - vehicle_rank1 = 0.9934
  - learned_vehicle_context_guarded = 0.9934
  - learned_physio_state_guarded = 0.9934
  - learned_physio_badweighted_guarded = 0.9934
  - oracle = 0.3678
- subject-aware test bad_top10：
  - vehicle_rank1 = 0.9838
  - learned_physio_state_guarded = 0.9838
  - oracle = 0.4403

结论：候选池上限很强，但当前生理状态表示仍不能可靠选择候选。

## 我现在需要你回答的问题

1. 基于这些结果，你认为“继续从现有 ECG/EDA/EMG/RESP 生理数据挖跨驾驶员行为预测增量”还有没有现实可行性？如果有，最可能的突破口是什么？
2. 当前失败更像是：
   - 生理数据本身对未来驾驶行为信息不足；
   - 预处理/对齐/归一化破坏了信息；
   - 表征方式太浅，需要端到端时序模型；
   - subject-disjoint 口径下个体差异太强，必须改成个体化/校准范式；
   - 还是候选池/标签构造有问题？
3. 下一步请给一个具体、可执行、优先级最高的实验方案。请明确：
   - 输入是什么；
   - 模型结构是什么；
   - 训练目标是什么；
   - 如何避免 leakage；
   - 用什么验证指标判断是否继续；
   - 如果不成功，应该如何收口。
4. 请不要只说“用 Transformer/TCN/attention”。如果建议端到端时序，请说明为什么它比 v254b 的 200Hz 手工统计更可能有效，以及最小可行实现。
5. 如果你认为生理数据主要只能做 subject-aware 个体化，而不能做 subject-disjoint 泛化，请直接说明，并建议如何把论文/模型目标改成合理但仍然是“预测方法提升”的路线。

请用中文回答，直接给判断和下一步行动，不要泛泛而谈。
