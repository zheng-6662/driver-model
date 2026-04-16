# 项目实验登记表

目标不是复述完整实验报告，而是让人一眼看懂：

- 这个 run / 分析改了什么
- 它是不是可比
- 它得出了什么一句话结论
- 它值不值得继续

## 实验表

| 日期 | 实验 / 分析 | 命名拆解 | 白话解释 | 可比性 | 变更 | 关键结果 | 判定 | 详情 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-04-09 | matched `baseline-conditioned` vs `structured_v2` 公平双跑 | `baseline-conditioned` = 带条件输入的基础对照模型；`structured_v2` = 第二版结构化条件注入模型。 | 这是一次尽量公平的正面对打。结果是 `structured_v2` 某些局部更强，但关键边界指标更差，所以还不能直接接班。 | matched pair | `conditioning_mode` 对照，其他主条件对齐 | `structured_v2` 在 tail / peak / turning_count 上仍有收益，但 `boundary_shift_abs_err` 从 `0.535222` 恶化到 `0.967678` | keep, not replace | [历史总档](../project_progress_master.md) |
| 2026-04-09 | `structured_v2_noresid` | `noresid` = 把 residual 支路关掉。 | 这是在问：“是不是那条残差支路把边界指标搞坏了？”答案是只改善一点点，但整体更差，所以这条修法不成立。 | 单变量 | `event_residual_scale: 1.0 -> 0.0` | `boundary_shift_abs_err` 仅回落到 `0.900528`，但整体 `steer RMSE`、`tail RMSE`、`selection_score` 变差 | no-go | [历史总档](../project_progress_master.md) |
| 2026-04-09 | `structured_v2_TF0` | `TF0` = `teacher_forcing_ratio = 0.0`，训练时不再强喂真实后续。 | 这是在问：“如果训练时不再靠老师带着走，模型自己滚动预测后会不会更真实？”结果说明 teacher forcing 的影响非常大，但还缺一个完全公平的 baseline 对照。 | 单变量 | `teacher_forcing_ratio: 1.0 -> 0.0` | 多项指标优于原始 `structured_v2`，但仍未干净超过 baseline；teacher forcing 明显影响结果模式 | needs control | [历史总档](../project_progress_master.md) |
| 2026-04-14 | `pca_latent` 256 样本 smoke | `pca_latent` = 先把 teacher state 压缩成 PCA 低维表示；`smoke` = 很小规模的通路验证。 | 这是第一轮“先确认整条新主线能不能跑起来”。结果是能跑，而且能完整导出状态相关结果。 | 集成验证 | 新 teacher-state 接口 + response-state-aware 最小闭环联调 | 真实数据上可完整跑通，2 个 epoch 正常收敛并成功导出 state metadata | proceed | [历史总档](../project_progress_master.md) |
| 2026-04-14 | `pca_latent` 1000 样本短跑 | `1000 样本短跑` = 比 smoke 更大，但还不是正式长跑；重点是看稳定性。 | 这轮说明新 teacher-state 方向不是“碰巧跑通一次”，而是真的能在更大样本下稳定工作；当前难点转到了 reversal 标签太稀。 | 集成验证 | 在更大样本下复核稳定性 | train / test 收敛稳定，teacher-state 主线不再只是偶然跑通；新暴露问题转向 strong reversal 标签极稀 | proceed with label analysis | [历史总档](../project_progress_master.md) |
| 2026-04-15 | `tail/peak/reversal` 固定评估闭环 smoke | `tail/peak/reversal` = 新增尾段、峰值、反打结构指标；`smoke` = 用小跑验证导出与诊断是否成立。 | 这是在问：“新加的评估到底能不能把 tail flattening 和强反打 hard case 量化出来？”答案是能，而且结论和人工看图一致。 | 集成验证 | 在 maintained v5.8 评估段接入结构化指标后做 smoke 导出验证 | `tail_flatness_rate=0.961`，`strong_pos tail_amp_ratio≈0.065`，说明尾段发平和强反打尾段收缩被成功量化 | proceed with minimal loss tuning | [今日日志](daily/2026-04-15.md) |
| 2026-04-15 | `W_REVSEQ 0.05` smoke + `head/onset` eval | `W_REVSEQ 0.05` = 单独打开 reversal sequence loss；`head/onset eval` = 给早段启动问题补结构化评估。 | 这是在问：“只动一个 reversal loss 权重，能不能在不伤结构的前提下把晚峰和反打尾段一起拉起来？”答案是不能；同时它也确认了前段启动不足确实需要固定指标跟踪。 | 单变量 | smoke 中临时将 `W_REVSEQ: 0.00 -> 0.05`，并新增 `test_metrics_head.json` | `late_peak_recall: 0.145 -> 0.545`，但 `tail_amp_ratio: 0.156 -> 0.088`、`strong_pos tail_amp_ratio: 0.065 -> 0.035`，说明增益集中在晚峰，尾段结构反而恶化 | no-go | [今日日志](daily/2026-04-15.md) |

## 填写规则

- 一行只写一个 run、一个对照组，或一个明确的分析动作。
- `可比性` 只写三类：`单变量`、`matched pair`、`集成验证`。
- `命名拆解` 用来解释英文缩写、后缀和版本名，不要求很学术，但要让第一次看到的人也能猜到大意。
- `白话解释` 用来说明“这个实验到底在问什么，结论应该怎么理解”。
- `变更` 只写最重要的那一个改动；如果是集成验证，就直接说明是联调验证。
- `关键结果` 最多写 1 到 2 个数字，或者一句判定，不复制长段分析。
- 如果已经有独立长报告，这里只放一句话摘要和链接，不再重复。

## 追加模板

| 日期 | 实验 / 分析 | 命名拆解 | 白话解释 | 可比性 | 变更 | 关键结果 | 判定 | 详情 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YYYY-MM-DD | 名称 | 这里解释缩写和后缀 | 这里写“这个实验到底想验证什么” | 单变量 / matched pair / 集成验证 | 这里写最重要的改动 | 这里写一句结果 | proceed / no-go / needs control / archive | 链接 |
