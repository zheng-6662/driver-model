# 给 GPT Pro 的直接提示词

你现在拿到的是一个已经完成 `2026-04-22 effectiveness` 闭环的驾驶员转向响应预测项目 handoff 包。
请不要从泛泛的“可以再试更多优化器 / 更大模型 / 更多数据增强”开始，而是基于已经完成的证据，判断下一步最值得投入的方向。

## 你的角色

请把自己当成一个愿意快速推进模型的高级研究搭档，而不是保守 reviewer。
你可以给出更激进的建议，但必须：

- 建立在当前证据链上
- 不违反当前 protocol-safe 公平口径
- 给出最小可执行版本

## 你必须先接受的当前事实

1. 当前正式 keeper 仍然是 `baseline_fixed_input` full `best_by_structured`。
2. Run A 仍然只是 response-structure anchor，不是当前 fit/tail winner。
3. `H15` 的整体与绝对 tail 指标明显更好，但它在 `strong_pos` 上硬塌陷，因此不能直接晋级。
4. `OPT_A_20` 没有把 `2.0s` 主线救回来，因此问题不太像“只差优化器”。
5. `OPT_C_BEST`、`CAP_192_BEST`、`WINNER_CONFIRM` 都没有替换当前 keeper。
6. `H10` 明确显示 `1.0s` ceiling 更高，但它只属于诊断证据，不允许直接升为默认主线。
7. bridge / gate / loss 作为主线已经关闭，不要建议回头重开同一轮主线。

## 你应该优先读哪些文件

1. `PROGRESS_AND_RESULTS_SUMMARY_CN.md`
2. `context/current-state.md`
3. `context/daily_2026-04-22.md`
4. `context/decision_log.md`
5. `context/experiment_registry.md`
6. `evidence/effectiveness_summary.md`
7. `evidence/effectiveness_comparison_table.csv`
8. `code/future_steer_event_rollpeak_transformer_v5_8_amp_tuned_fixed.py`
9. `code/recalc_v58_checkpoint_with_current_metrics.py`
10. 必要时再看 `configs/` 和 `evidence/` 下的单个 run config / summary JSON

## 我希望你输出什么

请按下面顺序作答，不要只给高层空话。

### 1. 你对当前卡点的最尖锐诊断

用 `3` 到 `6` 条结论说清楚：

- 当前模型真正卡在什么地方
- 为什么 `H15` 有真实收益但会在 `strong_pos` 上塌
- 为什么 `OPT_A_20` 的失败说明这不是简单优化问题
- 你判断下一刀更该打在 loss / target / decoder / architecture 的哪一层

### 2. 你推荐的 3 个下一步实验

对每个实验都必须回答：

- 实验名称
- 核心假设
- 为什么它比继续扫优化器 / 轻量容量更值
- 需要改哪些代码位置
- 具体改法是什么
- 预计会拉动哪些指标
- 最大风险是什么
- 应该先 smoke 还是直接 full

### 3. 只选 1 条最推荐主线

如果现在只允许我再烧 `1` 到 `2` 个 full GPU run，请只选一条主线，并解释：

- 为什么它的 expected value 最高
- 为什么它比另外两条更适合现在立即推进
- 你会怎么安排 run 顺序

### 4. 给出代码级最小落地方案

不要只讲概念。
请尽量具体到：

- 该新增或修改哪些函数
- 大致在训练脚本的哪一类位置改
- loss / target / objective 的公式或伪代码
- 如果你建议的是结构改动，请说明最小可验证版本

### 5. 如果你认为应该直接切新架构，也请明确说

如果你判断继续补 `H15` 已经不够，应该直接切一个新的 architecture slice，也可以提出，但必须说明：

- 为什么这个切法比继续小修小补更值
- 为什么它符合当前证据链
- 它的最小可验证版本是什么

## 重要约束

- 不要建议回到 random split / smoke 结果上做正式结论
- 不要只盯 overall RMSE
- 不要建议破坏 protocol-safe 闭环
- 不要重新把 bridge / gate / loss 拉回主线
- 不要把 `H10` 直接当成默认生产主线
- 不要只给“多试几个组合”的低决策价值建议

## 我的真实需求

我现在不是要一个保守综述，我是要一个高决策价值的下一步推进方案。
如果你认为应该更激进，请直接说，但请把方案约束到当前证据和工程现实里。
