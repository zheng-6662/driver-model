# GPT Pro Prompt

你现在拿到的是一个已经推进到 `2026-04-17` 的驾驶员短时反应预测项目 handoff 包。请不要回到泛泛的“可以再调调权重/再加层数/再做数据增强”，而要直接基于最新主线和最新 no-go 证据，给出高决策价值的下一步。

## 你的角色

请把自己当成愿意快速推进模型的高级研究搭档，而不是保守 reviewer。你可以激进，但必须激进在当前证据链允许的边界内。

## 项目目标

目标不是只降总体 RMSE，而是在 `protocol-safe`、`subject-level fixed split` 下，让模型更可信地预测极限工况下驾驶员未来 `2s` 的真实反应轨迹，尤其希望：

- 保住当前 strongest mainline 的 overall/head/tail/late-peak 收益
- 提升 `strong_pos / reversal` 这些 hard bucket
- 不要为了补某一项而明显破坏其它 guardrail

## 当前必须接受的前提

1. 当前 fair baseline 是 `2026-04-16 103752`
2. 当前 strongest allowed mainline 是 `2026-04-16 220918`
3. `2026-04-17 003953` hard-late fine 是 `no-go`
4. `2026-04-17 011755` raw `rev_logit` late rev gate 是 `no-go`
5. `2026-04-17 100716` dedicated strong-pos gate 也是 `no-go`
6. `100716` 的关键新增证据是：gate source 已明显变准，`strong_pos_gate_prob` 对 `rev_gt_strong` 的 AUC 约 `0.7044`，而 `220918` 旧 `rev_prob` 约 `0.4725`
7. 因此，当前更像是 `activation / routing` 问题，而不是 gate source 仍然不准
8. 不要回去重扫 `W_STEER_RATE` / `W_REVSEQ` / `W_STEER_REV`
9. 不要重开 `teacher-state` 或 random split 争论

## 你要先读哪些文件

请按这个顺序读：

1. `MODEL_STATE_SUMMARY.md`
2. `evidence/20260417_gate_source_activation_diagnosis.md`
3. `evidence/20260417_hardlate003953_from_daily_log.md`
4. `context/daily_2026-04-16.md`
5. `context/experiment_registry.md`
6. `code/future_steer_event_rollpeak_transformer_v5_8_amp_tuned_fixed.py`
7. `code/recalc_v58_checkpoint_with_current_metrics.py`
8. `protocol/protocol_config.json`
9. `evidence/20260416_fair_baseline103752_summary.json`
10. `evidence/20260416_mainline220918_summary.json`
11. `evidence/20260417_revgate011755_summary.json`
12. `evidence/20260417_strongposgate100716_summary.json`
13. 必要时再看对应 `run_config.json`、`cases.csv`、`test_state_dump.csv`

## 我希望你输出什么

请直接输出以下内容，按顺序来，不要只给高层空话：

### 1. 你对当前主问题的最尖锐诊断

- 用 3 到 6 条结论说清楚：
  - 当前模型真正卡在哪里
  - 为什么 `220918` 能成立
  - 为什么 `011755` 和 `100716` 都没守住 `220918`
  - 你认为“下一刀”应该打在 `activation`、`loss routing`、`target design`、`decoder slicing`，还是更结构化的 late residual 建模上

### 2. 你推荐的 3 个下一步实验

对每个实验都必须给出：

- 实验名称
- 核心假设
- 为什么比继续扫旧 scalar 更值
- 需要改哪些代码位置
- 具体改动是什么
- 预期会拉动哪些指标
- 最大风险是什么
- 应该先 smoke 还是直接 full-regime

至少满足下面这 3 类中的 2 类：

- 保留 dedicated gate source，但改 activation
- 不改前向 activation，而把 gate 变成 training-only / routing-only 信号
- 更激进地切一个更结构化的新 slice，直接绕开当前乘法式 gate 方案

### 3. 你最推荐的一条主线

请只选一条作为主推方案，并解释：

- 为什么它的 expected value 最高
- 为什么它比另外两条更适合现在直接推进
- 如果只允许再烧 1 到 2 轮 full GPU，你会怎么排顺序

### 4. 请给出“直接可执行”的代码级建议

不要只写概念，请尽量具体到：

- 建议新增/修改哪些函数
- 大致在训练脚本的哪一类位置改
- loss / target / objective 的公式或伪代码
- 如果你建议的是结构改动，也请说清楚最小落地版本怎么做

### 5. 如果你认为应该停止 gate 路线，也请直说

如果你判断现在继续修 gate activation 的 EV 已经不高，应该直接切一个更强的新建模切片，也可以明确提出，但要满足：

- 仍然保持 protocol-safe 公平口径
- 说明为什么这个架构跳转比继续修 gate 更值
- 给出最小可验证版本，而不是抽象口号

## 重要约束

- 不要建议回到 random split / smoke 结果去下正式结论
- 不要建议只盯总体 RMSE
- 不要建议删除现有主线代码路径
- 不要建议继续做 raw-source 或旧 scalar 的温和扫参
- 可以大胆，但请基于 `220918 -> 011755 -> 100716` 这条证据链大胆

## 我的真实需求

我现在的目标是：**尽快决定下一步最高 EV 的实验，而不是再花很多轮在低价值扫参上。**

所以如果你判断某条路虽然更激进，但比继续温和修 gate 更值得，请明确说出来，并给出最小可执行版本。
